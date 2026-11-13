import { Debug } from '../../../core/debug.js';
import { basisTranscode } from '../../handlers/basis.js';
import { MESHLET_DATA_U32S } from '../../../scene/meshlet/constants.js';
import { MeshletDirectorGroup } from '../../../scene/meshlet/meshlet-director-group.js';
import { restoreMeshletRecords } from '../../parsers/glb/extensions/mag-meshlets.js';
import { ComponentSystem } from '../system.js';
import { MeshletComponent } from './component.js';

/**
 * @import { AppBase } from '../../app-base.js'
 * @import { ByteCache } from '../../../platform/net/byte-cache.js'
 * @import { CameraComponent } from '../camera/component.js'
 * @import { MeshletDirector } from '../../../scene/meshlet/meshlet-director.js'
 */

const _properties = ['asset', 'resource', 'baseUrl'];

// draw indices a fresh world's views start with; demand grows them from there
const INITIAL_DRAW_INDICES = 2 * 1024 * 1024;


// how long a split scene must go without a rebuild before its worlds' records are released
const RECORD_RELEASE_DELAY_MS = 5000;

/**
 * Allows an Entity to render streamed meshlet assets through the GPU-driven meshlet pipeline.
 * WebGPU only - on other devices meshlet components are inert.
 *
 * The system owns the meshlet pipeline - a `MeshletDirectorGroup` of one or more meshlet worlds -
 * and its pipeline-wide settings:
 *
 * ```javascript
 * app.systems.meshlet.poolBytes = 256 * 1024 * 1024; // geometry page pool budget
 * app.systems.meshlet.dagPixelThreshold = 1;         // LOD screen-space error target
 * app.systems.meshlet.occlusion = true;              // two-phase HZB occlusion culling
 * app.systems.meshlet.shadows = true;                // meshlets cast shadows
 * ```
 *
 * Adding or removing meshlet components (or changing their assets) rebuilds the meshlet world
 * at the start of a following frame - once the changes go quiet for {@link rebuildQuietMs}, or
 * after {@link rebuildMaxWaitMs} at the latest, so a scene whose assets arrive over a few
 * seconds gets a handful of rebuilds rather than one per asset - and the rebuild retains
 * streaming state for resources that persist across it, so unchanged assets do not re-download. Components sharing
 * one asset are deduplicated into a single set of streamed pages. Enabling/disabling a
 * component (or its entity) and entity transform changes are cheap per-frame updates with no
 * rebuild at all.
 *
 * A world keeps every meshlet record of its content resident in one buffer, so content past
 * {@link maxWorldMeshlets} is split across several worlds. They render as one pipeline - one HZB
 * and one draw pass per phase per camera - while each keeps its own page pool, streaming and LOD
 * budget; {@link poolBytes} and {@link texturePoolBytes} are divided between them.
 *
 * @category Graphics
 */
class MeshletComponentSystem extends ComponentSystem {
    /**
     * Fired each time the meshlet content becomes {@link MeshletComponentSystem#playable}: every
     * component's asset has loaded, the world has rebuilt, and the streamed geometry and
     * texture tails are resident down to the playable floor. Streaming slows to its budgeted,
     * smooth pace from here.
     *
     * @event
     * @example
     * app.systems.meshlet.on('playable', () => {
     *     hideLoadingScreen();
     * });
     */
    static EVENT_PLAYABLE = 'playable';

    /**
     * The meshlet pipeline, created on first use (see {@link group}).
     *
     * @type {MeshletDirectorGroup|null}
     * @private
     */
    _group = null;

    /** @private */
    _initialized = false;

    /**
     * The meshlet pipeline - one director per world - or null on non-WebGPU devices. Created on
     * first use: the first component added, or the first access to a pipeline setting, so an
     * application without meshlets pays nothing for it.
     *
     * @type {MeshletDirectorGroup|null}
     * @ignore
     */
    get group() {
        return this._initialized ? this._group : this._initialize();
    }

    /**
     * The first world's meshlet director, or null on non-WebGPU devices. A scene within
     * {@link maxWorldMeshlets} has only this one; see {@link directors} for all of them.
     *
     * @type {MeshletDirector|null}
     * @ignore
     */
    get director() {
        return this.group?.directors[0] ?? null;
    }

    /**
     * Every world's meshlet director, empty on non-WebGPU devices.
     *
     * @type {MeshletDirector[]}
     * @ignore
     */
    get directors() {
        return this.group?.directors ?? [];
    }

    /** @private */
    _dirty = false;

    /** @type {number} - ms timestamp of the first unserved dirty mark, 0 when clean. @private */
    _dirtySince = 0;

    /** @type {number} - ms timestamp of the latest dirty mark. @private */
    _dirtyLast = 0;

    /** @type {boolean} - {@link playable} as of the last frame, to fire on transitions. @private */
    _wasPlayable = true;

    /**
     * Components with {@link MeshletComponent#whenPlayable} waiters, polled each frame.
     *
     * @type {Set<MeshletComponent>}
     * @private
     */
    _playableWatch = new Set();

    /** @type {number} - world rebuilds performed so far (a load-pattern statistic). */
    rebuildCount = 0;

    /** @private */
    _rebuildQuietMs = 250;

    /** @private */
    _rebuildMaxWaitMs = 2000;

    /**
     * Hold a pending rebuild while an enabled component's asset is still loading, once the world
     * shows something. Every rebuild re-finalizes the whole world - seconds of main thread on a
     * world of millions of instances - so building again as each asset of a burst lands costs
     * far more than waiting for the burst to finish. The first assets still build at once, so
     * there is something to show. Set false to rebuild on the quiet window and
     * {@link rebuildMaxWaitMs} regardless, showing regions as they arrive. Defaults to true.
     *
     * @type {boolean}
     */
    deferRebuildWhileLoading = true;

    /**
     * Components contributing to the current world, in placement order.
     *
     * @type {MeshletComponent[]}
     * @private
     */
    _activeComponents = [];

    /**
     * Components whose world transform may differ from what the world holds. Filled by the
     * entities' dirty notifications and by rebuilds, drained by the per-frame sync - a static
     * scene of any size costs nothing per frame.
     *
     * @type {MeshletComponent[]}
     * @ignore
     */
    _transformQueue = [];

    /**
     * Components whose hidden or shadow-casting state may differ from what the world holds.
     *
     * @type {MeshletComponent[]}
     * @ignore
     */
    _stateQueue = [];

    /** @private */
    _poolBytes = 256 * 1024 * 1024;

    /** @private */
    _texturePoolBytes = 96 * 1024 * 1024;

    /** @private */
    _maxFineTextureSize = 2048;

    /** @private */
    _maxDrawIndices = 0;

    /** @private */
    _maxWorldMeshlets = 12 * 1024 * 1024;

    /**
     * Per world, what it was last built with (resource groups and their components, budgets,
     * settings), so a rebuild can leave unchanged worlds alone.
     *
     * @type {object[]}
     * @private
     */
    _partitionStates = [];

    /**
     * Where each resource whose records were released reloads them from.
     *
     * @type {Map<object, { url: string, credentials: RequestCredentials|null }>}
     * @private
     */
    _recordSources = new Map();

    /**
     * Record restores in flight, by resource.
     *
     * @type {Map<object, Promise<void>>}
     * @private
     */
    _restoring = new Map();

    /** @type {number} - performance.now() of the last world rebuild. @private */
    _lastRebuild = 0;

    /** @type {number} - {@link _lastRebuild} as of the last record release. @private */
    _recordsReleasedAt = -1;

    /**
     * Create a new MeshletComponentSystem.
     *
     * @param {AppBase} app - The Application.
     * @ignore
     */
    constructor(app) {
        super(app);

        this.id = 'meshlet';
        this.ComponentType = MeshletComponent;

        this.on('beforeremove', this.onBeforeRemove, this);
    }

    /**
     * Creates the meshlet pipeline (WebGPU only) and starts the per-frame world sync.
     *
     * @returns {MeshletDirectorGroup|null} The pipeline, or null on non-WebGPU devices.
     * @private
     */
    _initialize() {
        this._initialized = true;
        const app = this.app;
        if (app.graphicsDevice.isWebGPU) {
            const group = new MeshletDirectorGroup(app.graphicsDevice);
            this._applyWorldSettings(group.directors[0].world, this._poolBytes, this._texturePoolBytes);
            group.bindRenderer(app.renderer);
            // streamed textures are KTX2; the scene layer cannot import the Basis handler, so
            // the transcoder is injected here (basisInitialize() decides the worker options)
            group.transcode = basisTranscode;
            this._group = group;

            // rebuild the world and sync entity transforms after the app update, before rendering
            app.on('framerender', this._onFrameRender, this);
        } else {
            Debug.warnOnce('MeshletComponentSystem: meshlet rendering requires WebGPU; meshlet components will not render.');
        }
        return this._group;
    }

    /**
     * Sets the geometry page pool byte budget. Under pool pressure the LOD cut coarsens until
     * its working set fits, so a small budget trades quality for memory instead of failing.
     * Content split across several worlds (see {@link maxWorldMeshlets}) divides it between
     * them by meshlet count. Takes effect on the next world rebuild. Defaults to 256 MiB.
     *
     * @type {number}
     */
    set poolBytes(value) {
        if (this._poolBytes !== value) {
            this._poolBytes = value;
            if (this._group) this._markDirty();
        }
    }

    get poolBytes() {
        return this._poolBytes;
    }

    /**
     * Sets the fine-texture slot-pool byte budget. Under pressure the texture mip bias rises,
     * pushing more textures onto their always-resident tails, so a small budget trades
     * sharpness for memory instead of failing. Divided between worlds by the meshlet count of
     * their textured content. Takes effect on the next world rebuild. Defaults to 96 MiB.
     *
     * @type {number}
     */
    set texturePoolBytes(value) {
        if (this._texturePoolBytes !== value) {
            this._texturePoolBytes = value;
            if (this._group) this._markDirty();
        }
    }

    get texturePoolBytes() {
        return this._texturePoolBytes;
    }

    /**
     * The most meshlets one world holds. A world keeps every meshlet record of its content
     * resident in one buffer (with its LOD-cut tables alongside, ~190 bytes a meshlet), which
     * must fit what the browser allocates and WebGPU binds; content past this is split across
     * further worlds, each with its own share of the budgets and its own cull passes. Clamped
     * to the device's storage binding limit. Takes effect on the next world rebuild. Defaults to
     * 12M (~1.5 GB of records).
     *
     * @type {number}
     */
    set maxWorldMeshlets(value) {
        value = Math.max(1, Math.floor(value));
        if (this._maxWorldMeshlets !== value) {
            this._maxWorldMeshlets = value;
            if (this._group) this._markDirty();
        }
    }

    get maxWorldMeshlets() {
        return this._maxWorldMeshlets;
    }

    /**
     * Largest texture size streamed at full resolution. A family's fine slots are sized to its
     * largest texture, so one very large texture would make every slot that large; textures
     * above this size stream down to it. Takes effect on the next world rebuild. Defaults to
     * 2048.
     *
     * @type {number}
     */
    set maxFineTextureSize(value) {
        if (this._maxFineTextureSize !== value) {
            this._maxFineTextureSize = value;
            if (this._group) this._markDirty();
        }
    }

    get maxFineTextureSize() {
        return this._maxFineTextureSize;
    }

    /**
     * Sets an optional hard ceiling on the GPU-written draw index buffer, in indices (0 =
     * unbounded, the default). The buffer starts small and grows to the observed per-frame
     * demand, which the LOD cut bounds by screen area; set a ceiling to hard-cap that memory -
     * the cull shader clamps safely on overflow. Takes effect on the next world rebuild.
     *
     * @type {number}
     */
    set maxDrawIndices(value) {
        if (this._maxDrawIndices !== value) {
            this._maxDrawIndices = value;
            if (this._group) this._markDirty();
        }
    }

    get maxDrawIndices() {
        return this._maxDrawIndices;
    }

    /**
     * Cap on sidecar requests (geometry shards, texture containers) in flight across every
     * streamed asset; the rest wait in priority order. Defaults to 16.
     *
     * @type {number}
     */
    set maxConcurrentFetches(value) {
        if (this.group) this.group.maxConcurrentFetches = value;
    }

    get maxConcurrentFetches() {
        return this._group?.maxConcurrentFetches ?? 16;
    }

    /**
     * Persistent cache for streamed geometry pages and texture regions, or null for none. See
     * `MeshletDirector#byteCache`. The same cache can be shared with
     * {@link ResourceLoader#byteCache}.
     *
     * @type {ByteCache|null}
     */
    set byteCache(value) {
        if (this.group) this.group.byteCache = value;
    }

    get byteCache() {
        return this._group?.byteCache ?? null;
    }

    /**
     * Sets the screen-space error threshold in pixels for the DAG LOD cut, scaled per draw bucket
     * by {@link bucketThresholdScales}. Defaults to 2.
     *
     * @type {number}
     */
    set dagPixelThreshold(value) {
        if (this.group) this.group.dagPixelThreshold = value;
    }

    get dagPixelThreshold() {
        return this._group?.dagPixelThreshold ?? 2;
    }

    /**
     * The fraction of {@link rootMinPixels} an instance already drawn may shrink to before it is
     * dropped, so instances do not pop at the cut while the camera moves. Defaults to 0.75.
     *
     * @type {number}
     */
    set rootHysteresis(value) {
        if (this.group) this.group.rootHysteresis = value;
    }

    get rootHysteresis() {
        return this._group?.rootHysteresis ?? 0.75;
    }

    /**
     * Per draw bucket - opaque, opaque two-sided, alpha-masked - a multiplier on
     * {@link dagPixelThreshold}. Solid geometry can take a much coarser cut than foliage, whose
     * simplified levels thin out rather than coarsen. Defaults to [4, 1, 1]: solid geometry at
     * four times {@link dagPixelThreshold}, foliage at it.
     *
     * @type {number[]}
     * @example
     * // every bucket at the base threshold
     * app.systems.meshlet.bucketThresholdScales = [1, 1, 1];
     */
    set bucketThresholdScales(value) {
        if (this.group) this.group.bucketThresholdScales = value;
    }

    get bucketThresholdScales() {
        return this._group?.bucketThresholdScales ?? [4, 1, 1];
    }

    /**
     * Per draw bucket - opaque, opaque two-sided, alpha-masked - a multiplier on
     * {@link rootMinPixels}. Defaults to [1, 1, 1].
     *
     * @type {number[]}
     */
    set bucketRootMinScales(value) {
        if (this.group) this.group.bucketRootMinScales = value;
    }

    get bucketRootMinScales() {
        return this._group?.bucketRootMinScales ?? [1, 1, 1];
    }

    /**
     * Sets whether two-phase HZB occlusion culling is enabled. Requires the camera to render
     * into a render target with a depth texture. Falls back to single-phase culling
     * without one. Defaults to true.
     *
     * @type {boolean}
     */
    set occlusion(value) {
        if (this.group) this.group.occlusionEnabled = value;
    }

    get occlusion() {
        return this._group?.occlusionEnabled ?? false;
    }

    /**
     * Sets the camera meshlets are culled and drawn for. When null (the default), the first
     * camera of the layer composition is used.
     *
     * @type {CameraComponent|null}
     */
    set camera(value) {
        if (this.group) this.group.cameraComponent = value;
    }

    get camera() {
        return this._group?.cameraComponent ?? null;
    }

    /**
     * Sets the cameras meshlets are culled and drawn for - one independent LOD cut and
     * occlusion state per camera, streaming against the union of their demand. Empty (the
     * default) means the first camera of the layer composition. Capped at the director's
     * maxViews (2 by default).
     *
     * @type {CameraComponent[]}
     */
    set cameras(value) {
        if (this.group) this.group.cameras = value ?? [];
    }

    get cameras() {
        return this._group?.cameras ?? [];
    }

    /**
     * Whether the pipeline runs shadow passes - directional cascades and local (spot / omni)
     * shadow faces. Off by default: every shadow face costs one extra cull pass per frame. Which
     * instances cast is per component ({@link MeshletComponent#castShadows}).
     *
     * @type {boolean}
     */
    set shadows(value) {
        if (this.group) this.group.shadowsEnabled = !!value;
    }

    get shadows() {
        return this._group?.shadowsEnabled ?? false;
    }

    /**
     * Multiplier on {@link dagPixelThreshold} for shadow views. Shadow-map texels are coarser
     * than screen pixels, so casters can use a coarser LOD cut than the camera. Defaults to 4.
     *
     * @type {number}
     */
    set shadowThresholdScale(value) {
        if (this.group) this.group.shadowThresholdScale = value;
    }

    get shadowThresholdScale() {
        return this._group?.shadowThresholdScale ?? 4;
    }

    /**
     * Projected size on screen, in pixels, below which an instance is not drawn. Every drawn
     * instance costs at least its coarsest level, so wide views of worlds with very many small
     * placements are bound by that floor; raising this trades sub-pixel scatter for speed. An
     * instance already drawn stays until it is half this size, to avoid popping. 0 draws every
     * visible instance. Defaults to 3.
     *
     * @type {number}
     */
    set rootMinPixels(value) {
        if (this.group) this.group.rootMinPixels = value;
    }

    get rootMinPixels() {
        return this._group?.rootMinPixels ?? 3;
    }

    /**
     * Sets whether the screen-space thresholds - {@link dagPixelThreshold}, {@link rootMinPixels}
     * and {@link shadowRootMinPixels} - are measured in CSS pixels rather than render target
     * pixels, for cameras rendering to the canvas. In render target pixels, a display with a
     * pixel ratio of 2 refines the cut about four times further, mostly into detail too small to
     * see. The thresholds never go finer than one render target pixel. Defaults to true.
     *
     * @type {boolean}
     */
    set cssPixels(value) {
        if (this.group) this.group.cssPixels = value;
    }

    get cssPixels() {
        return this._group?.cssPixels ?? true;
    }

    /**
     * Projected size on screen, in pixels, below which an instance casts no meshlet shadow.
     * Far-reaching shadow cascades cover most of the world, and every caster they admit costs at
     * least its coarsest level; skipping casters this small keeps them from taking the index
     * budget the camera needs. 0 lets every visible instance cast. Defaults to 2.
     *
     * @type {number}
     */
    set shadowRootMinPixels(value) {
        if (this.group) this.group.shadowRootMinPixels = value;
    }

    get shadowRootMinPixels() {
        return this._group?.shadowRootMinPixels ?? 2;
    }

    initializeComponentData(component, data, properties) {
        if (!this._initialized) this._initialize();

        for (let i = 0; i < _properties.length; i++) {
            if (data.hasOwnProperty(_properties[i])) {
                component[_properties[i]] = data[_properties[i]];
            }
        }

        super.initializeComponentData(component, data);
    }

    cloneComponent(entity, clone) {
        const meshletComponent = entity.meshlet;
        return this.addComponent(clone, {
            enabled: meshletComponent.enabled,
            asset: meshletComponent.asset,
            resource: meshletComponent.resource,
            baseUrl: meshletComponent.baseUrl
        });
    }

    onBeforeRemove(entity, component) {
        component.onBeforeRemove();
    }

    /**
     * How long component changes must go quiet before the world rebuilds, in milliseconds.
     * Assets loading in a burst then cost a few rebuilds instead of one each (every rebuild
     * re-finalizes the whole world and re-streams what it cannot carry over). Defaults to 250.
     *
     * @type {number}
     */
    set rebuildQuietMs(value) {
        this._rebuildQuietMs = Math.max(0, value);
    }

    get rebuildQuietMs() {
        return this._rebuildQuietMs;
    }

    /**
     * Longest a pending rebuild waits for the changes to go quiet, in milliseconds, so a steady
     * trickle of assets still shows up. Defaults to 2000.
     *
     * @type {number}
     */
    set rebuildMaxWaitMs(value) {
        this._rebuildMaxWaitMs = Math.max(0, value);
    }

    get rebuildMaxWaitMs() {
        return this._rebuildMaxWaitMs;
    }

    /** @ignore */
    _markDirty() {
        const now = performance.now();
        if (!this._dirty) {
            this._dirty = true;
            this._dirtySince = now;
        }
        this._dirtyLast = now;
    }

    /**
     * Whether the meshlet content can be shown: no component is waiting for its asset to load,
     * no world rebuild is pending, and the streamed geometry and texture tails are resident down
     * to the playable floor - the coarsest LOD free of sloppy simplification (see
     * `MeshletDirector#playable`). True when there is no meshlet content, or no WebGPU.
     *
     * @type {boolean}
     */
    get playable() {
        return this._isPlayable(true);
    }

    /**
     * @param {boolean} scanAssets - Check every component for a loading asset.
     * @returns {boolean} Whether the content is playable.
     * @private
     */
    _isPlayable(scanAssets) {
        const group = this._group;
        if (!group) return true;
        if (this._dirty) return false;
        // a world with nothing in it is never finalized, and has nothing to stream
        if (!group.playable) return false;
        return !scanAssets || !this._hasPendingAssets();
    }

    /**
     * Resolves once the meshlet content is {@link playable} - at once if it already is. Call
     * after the scene's components are created, so a rebuild for them is already pending.
     *
     * @returns {Promise<void>} Resolves when playable.
     * @example
     * await app.systems.meshlet.whenPlayable();
     * hideLoadingScreen();
     */
    whenPlayable() {
        if (this.playable) return Promise.resolve();
        return new Promise((resolve) => {
            this.once(MeshletComponentSystem.EVENT_PLAYABLE, () => resolve());
        });
    }

    /**
     * Load as fast as possible until {@link playable}, then stream at the smooth budgeted pace.
     * Frames may stutter while fast-loading, so cover it with a loading screen; set false to
     * always stream smoothly. Defaults to true. See `MeshletDirector#fastLoad`.
     *
     * @type {boolean}
     */
    set fastLoad(value) {
        if (this.group) this.group.fastLoad = !!value;
    }

    get fastLoad() {
        return this._group?.fastLoad ?? true;
    }

    /**
     * @param {MeshletComponent} component - The component.
     * @returns {import('./component.js').MeshletStreamStatus} Its load state.
     * @ignore
     */
    _componentStatus(component) {
        if (!this._group) return { playable: true, failed: false, progress: 1, totalBytes: 0, loadedBytes: 0 };
        const director = this._group.directors[component._partition];
        const resource = component._effectiveResource;
        // not placed yet: the asset is loading, or the rebuild that adds it has not run
        if (!resource || !director || component._placementIndex < 0 || component._placedResource !== resource) {
            return { playable: false, failed: false, progress: 0, totalBytes: 0, loadedBytes: 0 };
        }
        return director.resourceStatus(resource) ?? { playable: false, failed: false, progress: 0, totalBytes: 0, loadedBytes: 0 };
    }

    /**
     * @param {MeshletComponent} component - A component with playable waiters.
     * @ignore
     */
    _watchPlayable(component) {
        this._playableWatch.add(component);
    }

    /**
     * @param {MeshletComponent} component - The component.
     * @ignore
     */
    _unwatchPlayable(component) {
        this._playableWatch.delete(component);
    }

    /**
     * Reports progress to, and settles, the components being waited on.
     *
     * @private
     */
    _updatePlayableWatch() {
        for (const component of this._playableWatch) {
            const status = this._componentStatus(component);
            if (status.playable) {
                this._playableWatch.delete(component);
                component.fire(MeshletComponent.EVENT_PROGRESS, 1, status.totalBytes);
                component.fire(MeshletComponent.EVENT_PLAYABLE, status);
                component._settlePlayable(status);
            } else if (status.progress !== component._reportedProgress) {
                component._reportedProgress = status.progress;
                component.fire(MeshletComponent.EVENT_PROGRESS, status.progress, status.totalBytes);
            }
        }
    }

    /**
     * Whether an enabled component's asset is still loading (its resource cannot join the world
     * yet).
     *
     * @returns {boolean} True while any such asset is loading.
     * @private
     */
    _hasPendingAssets() {
        for (const guid in this.store) {
            const component = this.store[guid].entity.meshlet;
            if (component && component.enabled && component.entity.enabled && component._assetReference.asset?.loading) {
                return true;
            }
        }
        return false;
    }

    /**
     * Rebuilds the world now if changes are pending, regardless of the quiet window.
     */
    flushRebuild() {
        if (this._group && this._dirty) this._rebuild();
    }

    /** @private */
    _onFrameRender() {
        if (!this._group) return;
        if (this._dirty) {
            const now = performance.now();
            const quiet = now - this._dirtyLast >= this._rebuildQuietMs;
            const overdue = now - this._dirtySince >= this._rebuildMaxWaitMs;
            // a world with nothing built yet has nothing to show either way: build at once
            const built = this._group.directors.some(d => d.world.finalized);
            const waiting = built && this.deferRebuildWhileLoading && this._hasPendingAssets();
            if (!built || (!waiting && (quiet || overdue))) {
                this._rebuild();
            }
        }
        this._syncTransforms();
        this._releaseSettledRecords();

        // the director's state is from last frame's update: a transition fires one frame late.
        // Once playable, a new asset marks the system dirty when assigned, so the per-component
        // asset scan only runs while waiting.
        const playable = this._isPlayable(!this._wasPlayable);
        if (playable !== this._wasPlayable) {
            this._wasPlayable = playable;
            if (playable) this.fire(MeshletComponentSystem.EVENT_PLAYABLE);
        }
        if (this._playableWatch.size) this._updatePlayableWatch();
    }

    /** @private */
    _rebuild() {
        // group components by (resource, baseUrl): each unique resource is added once, with a
        // merged instance list - N components share one set of pages, records and textures.
        // Disabled components are included (hidden via the per-instance hide bit) so a later
        // enable is a buffer write, not a rebuild.
        /** @type {Array<{ resource: object, baseUrl: string, fetchOptions: object, components: MeshletComponent[] }>} */
        const groups = [];
        // resource -> baseUrl -> group
        /** @type {Map<object, Map<string, object>>} */
        const groupIndex = new Map();
        for (const guid in this.store) {
            const component = this.store[guid].entity.meshlet;
            if (!component) continue;
            // a component no world places is reset to -1 after the build, so the sync skips it
            if (!component._effectiveResource) continue;
            if (!component._effectiveBaseUrl) {
                Debug.warnOnce('MeshletComponent: no stream base URL - assign an asset with an http(s) file URL (not a blob: or data: URL), or set baseUrl with the resource.');
                continue;
            }
            const resource = component._effectiveResource;
            const baseUrl = component._effectiveBaseUrl;
            let byUrl = groupIndex.get(resource);
            if (!byUrl) {
                byUrl = new Map();
                groupIndex.set(resource, byUrl);
            }
            let group = byUrl.get(baseUrl);
            if (!group) {
                group = { resource, baseUrl, fetchOptions: component._effectiveFetchOptions, components: [] };
                byUrl.set(baseUrl, group);
                groups.push(group);
            }
            group.components.push(component);
        }

        // Pack the groups into worlds in order, starting a new world when the next resource
        // would take the current one past the meshlet limit. A resource larger than the limit
        // gets a world of its own.
        const limit = this._worldMeshletLimit();
        const partitions = [{ groups: [], meshlets: 0, texturedMeshlets: 0 }];
        for (const group of groups) {
            const meshlets = group.resource.totalMeshlets;
            let partition = partitions[partitions.length - 1];
            if (partition.groups.length && partition.meshlets + meshlets > limit) {
                partition = { groups: [], meshlets: 0, texturedMeshlets: 0 };
                partitions.push(partition);
            }
            partition.groups.push(group);
            partition.meshlets += meshlets;
            if (group.resource.textureManifest) partition.texturedMeshlets += meshlets;
        }
        let totalMeshlets = 0;
        let totalTextured = 0;
        for (const p of partitions) {
            totalMeshlets += p.meshlets;
            totalTextured += p.texturedMeshlets;
        }

        // Budget shares and what each world keeps, decided before anything is built: a world
        // that must rebuild needs its resources' records, and records released after an earlier
        // build are fetched back first - the rebuild stays pending (dirty) until they arrive.
        const previous = this._partitionStates;
        const plans = partitions.map((partition, pi) => {
            const poolBytes = partitions.length > 1 && this._poolBytes > 0 ?
                Math.floor(this._poolBytes * partition.meshlets / Math.max(totalMeshlets, 1)) :
                this._poolBytes;
            const texturePoolBytes = partitions.length > 1 ?
                Math.floor(this._texturePoolBytes * partition.texturedMeshlets / Math.max(totalTextured, 1)) :
                this._texturePoolBytes;
            const state = { groups: partition.groups.map(g => ({ ...g, components: g.components.slice() })), poolBytes, texturePoolBytes, settings: this._worldSettingsKey() };
            const director = this._group.directors[pi];
            return { partition, state, keep: !!director && this._canKeepPartition(previous[pi], state, director) };
        });
        let restoring = false;
        for (const plan of plans) {
            if (plan.keep) continue;
            for (const group of plan.partition.groups) {
                if (!group.resource.recordsResident) {
                    this._restoreRecords(group.resource);
                    restoring = true;
                }
            }
        }
        if (restoring) {
            this._dirty = true;
            return;
        }

        this._dirty = false;
        this._dirtySince = 0;
        this._dirtyLast = 0;
        this.rebuildCount++;

        const components = [];
        const placed = new Set();
        const states = [];
        let rebuilt = 0;
        this._group.resize(partitions.length);
        for (let pi = 0; pi < partitions.length; pi++) {
            // budgets are divided by meshlet count: the fixed per-meshlet tables dominate a
            // world's cost, and its page count follows its meshlets
            const { partition, state, keep } = plans[pi];
            const { poolBytes, texturePoolBytes } = state;
            const director = this._group.directors[pi];
            if (keep) {
                // unchanged content and budget: the world, its streaming and its placements stay
                states.push(previous[pi]);
                for (const g of partition.groups) {
                    for (const component of g.components) {
                        placed.add(component);
                        components.push(component);
                    }
                }
                continue;
            }
            rebuilt++;
            director.rebuild((world) => {
                this._applyWorldSettings(world, poolBytes, texturePoolBytes);
                this._buildPartition(world, partition.groups, pi, components);
            });
            for (const g of partition.groups) {
                for (const component of g.components) placed.add(component);
            }
            states.push(state);
        }
        this._lastRebuild = performance.now();
        this._partitionStates = states;
        // components no longer in any world (removed asset, no base URL) stop syncing
        for (const component of this._activeComponents) {
            if (!placed.has(component)) {
                component._placementIndex = -1;
                component._placedResource = null;
            }
        }
        this._group.assignInstanceOffsets();
        if (partitions.length > 1) {
            Debug.log(`MeshletComponentSystem: ${totalMeshlets} meshlets in ${partitions.length} worlds (limit ${limit} each), ${rebuilt} rebuilt.`);
        }
        this._activeComponents = components;
    }

    /**
     * Whether a world can keep its current build: the same resource groups with the same
     * components in the same order, the same world settings, and a built world. Its budget
     * share is not compared: shares follow the scene total, which every added resource moves,
     * and rebuilding every world (re-fetching released records) for a slightly different page
     * pool would cost far more than the pool difference. A kept world keeps the share it was
     * built with; changing the budget setting itself rebuilds them all.
     *
     * @param {object|undefined} previous - The partition's state at its last build.
     * @param {object} next - The state it would be built with now.
     * @param {MeshletDirector} director - The partition's director.
     * @returns {boolean} True when the world needs no rebuild.
     * @private
     */
    _canKeepPartition(previous, next, director) {
        if (!previous || !director.world.finalized) return false;
        if (previous.settings !== next.settings) return false;
        if (previous.groups.length !== next.groups.length) return false;
        for (let i = 0; i < next.groups.length; i++) {
            const a = previous.groups[i], b = next.groups[i];
            if (a.resource !== b.resource || a.baseUrl !== b.baseUrl || a.components.length !== b.components.length) return false;
            for (let c = 0; c < b.components.length; c++) {
                if (a.components[c] !== b.components[c]) return false;
            }
        }
        return true;
    }

    /**
     * @returns {string} The world settings a partition is built with, other than its budgets.
     * @private
     */
    _worldSettingsKey() {
        return `${this._poolBytes}|${this._texturePoolBytes}|${this._maxFineTextureSize}|${this._maxDrawIndices}`;
    }

    /**
     * Once a split scene has settled - no asset loading, no rebuild pending, none for
     * RECORD_RELEASE_DELAY_MS - drops the resources' own copies of their meshlet records (nearly
     * all of each GLB) for every world but the last: each world has uploaded its own. The last
     * world takes newly added resources, so it keeps them for the rebuilds that brings. Releasing
     * earlier, while assets still arrive, only has them fetched straight back when a world's
     * budget share moves.
     *
     * @private
     */
    _releaseSettledRecords() {
        const states = this._partitionStates;
        if (states.length < 2 || this._dirty || this._restoring.size) return;
        if (performance.now() - this._lastRebuild < RECORD_RELEASE_DELAY_MS) return;
        if (this._recordsReleasedAt === this._lastRebuild || this._hasPendingAssets()) return;
        this._recordsReleasedAt = this._lastRebuild;
        for (let pi = 0; pi < states.length - 1; pi++) {
            for (const group of states[pi].groups) this._releaseRecords(group);
        }
    }

    /**
     * Releases a resource group's meshlet records when they can be fetched back: the resource
     * came from an asset with a file URL. A direct resource has nowhere to reload from.
     *
     * @param {{ resource: object, components: MeshletComponent[] }} group - The group.
     * @private
     */
    _releaseRecords(group) {
        const component = group.components[0];
        const url = component._assetReference.asset?.getFileUrl();
        if (!url || !group.resource.recordsResident) return;
        this._recordSources.set(group.resource, { url, credentials: component._effectiveFetchOptions?.credentials ?? null });
        group.resource.releaseRecords();
    }

    /**
     * Starts fetching a released resource's records back, once.
     *
     * @param {object} resource - The MeshletResource.
     * @private
     */
    _restoreRecords(resource) {
        if (this._restoring.has(resource)) return;
        const source = this._recordSources.get(resource);
        if (!source) {
            Debug.error('MeshletComponentSystem: a resource\'s meshlet records were released with no source to restore them from.');
            return;
        }
        const promise = restoreMeshletRecords(source.url, resource, source.credentials).then(() => {
            this._restoring.delete(resource);
            this._markDirty();
        }).catch((err) => {
            // left released: its world keeps its last build and the rebuild stays pending
            Debug.error(`MeshletComponentSystem: restoring meshlet records from ${source.url} failed`, err);
        });
        this._restoring.set(resource, promise);
    }

    /**
     * The meshlet limit per world: {@link maxWorldMeshlets}, clamped so a world's meshlet records
     * fit one storage binding.
     *
     * @returns {number} Meshlets per world.
     * @private
     */
    _worldMeshletLimit() {
        // WebGPU devices expose their adapter limits; others have none to clamp to
        const device = /** @type {{ limits?: { maxBufferSize?: number, maxStorageBufferBindingSize?: number } }} */ (/** @type {unknown} */ (this.app.graphicsDevice));
        const limits = device.limits ?? {};
        const bytes = Math.min(
            limits.maxBufferSize ?? Number.MAX_SAFE_INTEGER,
            limits.maxStorageBufferBindingSize ?? Number.MAX_SAFE_INTEGER
        );
        return Math.max(Math.min(this._maxWorldMeshlets, Math.floor(bytes / (MESHLET_DATA_U32S * 4))), 1);
    }

    /**
     * Applies the system's world settings to a world about to be built.
     *
     * @param {import('../../../scene/meshlet/meshlet-world.js').MeshletWorld} world - The world.
     * @param {number} poolBytes - The world's share of the geometry budget.
     * @param {number} texturePoolBytes - The world's share of the texture budget.
     * @private
     */
    _applyWorldSettings(world, poolBytes, texturePoolBytes) {
        world.poolBytes = poolBytes;
        world.texturePoolBytes = texturePoolBytes;
        world.maxFineTextureSize = this._maxFineTextureSize;
        world.initialIndices = INITIAL_DRAW_INDICES;
        world.maxIndices = this._maxDrawIndices;
    }

    /**
     * Adds one world's resource groups, placing their components in it.
     *
     * @param {import('../../../scene/meshlet/meshlet-world.js').MeshletWorld} world - The world.
     * @param {Array<{ resource: object, baseUrl: string, fetchOptions: object, components: MeshletComponent[] }>} groups - The world's groups.
     * @param {number} partition - The world's index in the group.
     * @param {MeshletComponent[]} components - Receives the placed components.
     * @private
     */
    _buildPartition(world, groups, partition, components) {
        for (let gi = 0; gi < groups.length; gi++) {
            const { resource, baseUrl, fetchOptions, components: members } = groups[gi];
            const per = resource.instances.length;

            // one placement per group; each member owns a sub-range of its instances
            let merged = null;
            if (members.length > 1) {
                merged = [];
                for (let mi = 0; mi < members.length; mi++) merged.push(...resource.instances);
            }
            // The build bakes each member's current transform and the default state (shown,
            // casting), so the first sync after it only rewrites what actually differs - not
            // every row of a world of millions of instances.
            const transforms = [];
            for (let mi = 0; mi < members.length; mi++) {
                const component = members[mi];
                component._partition = partition;
                component._placementIndex = gi;
                component._placedResource = resource;
                component._subBase = mi * per;
                component._subCount = per;
                const wt = component.entity.getWorldTransform();
                component._lastTransform.copy(wt);
                component._transformDirty = false;
                transforms.push(component._lastTransform);
                component._hiddenApplied = false;
                component._castShadowsApplied = true;
                component.onWorldDirty();
                component._queueState();
                components.push(component);
            }
            world.addStreamedResource(resource, transforms, baseUrl, merged, fetchOptions);
        }
    }

    /** @private */
    _syncTransforms() {
        const directors = this._group.directors;
        if (!directors.some(d => d.world.finalized)) return;

        const transforms = this._transformQueue;
        for (let i = 0; i < transforms.length; i++) {
            const component = transforms[i];
            component._transformQueued = false;
            if (component._placementIndex < 0) continue;
            const world = directors[component._partition]?.world;
            if (!world?.finalized) continue;
            const wt = component.entity.getWorldTransform();
            // a transform set to its current value still dirties the node: skip the upload
            if (component._transformDirty || !component._lastTransform.equals(wt)) {
                component._transformDirty = false;
                component._lastTransform.copy(wt);
                world.setPlacementTransform(component._placementIndex, wt, component._subBase, component._subCount);
            }
        }
        transforms.length = 0;

        const states = this._stateQueue;
        for (let i = 0; i < states.length; i++) {
            const component = states[i];
            component._stateQueued = false;
            if (component._placementIndex < 0) continue;
            const world = directors[component._partition]?.world;
            if (!world?.finalized) continue;
            const hidden = !(component.enabled && component.entity.enabled);
            if (component._hiddenApplied !== hidden) {
                component._hiddenApplied = hidden;
                world.setPlacementHidden(component._placementIndex, hidden, component._subBase, component._subCount);
            }
            const casts = component.castShadows !== false;
            if (component._castShadowsApplied !== casts) {
                component._castShadowsApplied = casts;
                world.setPlacementCastShadows(component._placementIndex, casts, component._subBase, component._subCount);
            }
        }
        states.length = 0;
    }

    destroy() {
        super.destroy();
        this.app.off('framerender', this._onFrameRender, this);
        this._group?.destroy();
        this._group = null;
    }
}

export { MeshletComponentSystem };
