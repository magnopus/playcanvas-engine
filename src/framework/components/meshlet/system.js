import { Debug } from '../../../core/debug.js';
import { basisTranscode } from '../../handlers/basis.js';
import { MeshletDirectorGroup } from '../../../scene/meshlet/meshlet-director-group.js';
import { magMeshlets } from '../../parsers/glb/extensions/mag-meshlets.js';
import { registerGlbResourceExtension, unregisterGlbResourceExtension } from '../../parsers/glb-resource-extension.js';
import { ComponentSystem } from '../system.js';
import { MeshletComponent } from './component.js';
import { MESHLET_DEFAULTS } from './meshlet-defaults.js';
import { MeshletPlayableTracker } from './meshlet-playable-tracker.js';
import { MeshletCutTableStore } from './meshlet-cut-table-store.js';
import { MeshletRecordStore } from './meshlet-record-store.js';
import { applyWorldSettings, buildPartition, groupComponents, planWorlds, worldMeshletLimit } from './meshlet-world-planner.js';

/**
 * @import { AppBase } from '../../app-base.js'
 * @import { ResourceHandler } from '../../handlers/handler.js'
 * @import { ByteCache } from '../../../platform/net/byte-cache.js'
 * @import { CameraComponent } from '../camera/component.js'
 * @import { MeshletDirector } from '../../../scene/meshlet/meshlet-director.js'
 * @import { MeshletPartitionState, MeshletWorldSettings } from './meshlet-world-planner.js'
 */

const _properties = ['asset', 'resource', 'baseUrl', 'castShadows'];

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
     * Container handler this system registered its glTF extension with.
     *
     * @type {ResourceHandler|null}
     * @private
     */
    _containerHandler = null;

    /** @private */
    _dirty = false;

    /**
     * Timestamp in milliseconds of the first unserved dirty mark, 0 when clean.
     *
     * @type {number}
     * @private
     */
    _dirtySince = 0;

    /**
     * Timestamp in milliseconds of the latest dirty mark.
     *
     * @type {number}
     * @private
     */
    _dirtyLast = 0;

    /**
     * World rebuilds performed so far.
     *
     * @type {number}
     * @ignore
     */
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
    _suspended = false;

    /**
     * The world budgets and limits, applied on the next rebuild.
     *
     * @type {MeshletWorldSettings}
     * @private
     */
    _worldSettings = {
        poolBytes: MESHLET_DEFAULTS.poolBytes,
        texturePoolBytes: MESHLET_DEFAULTS.texturePoolBytes,
        maxFineTextureSize: MESHLET_DEFAULTS.maxFineTextureSize,
        maxDrawIndices: MESHLET_DEFAULTS.maxDrawIndices,
        maxWorldMeshlets: MESHLET_DEFAULTS.maxWorldMeshlets
    };

    /**
     * Per world, what it was last built with, so a rebuild can leave unchanged worlds alone.
     *
     * @type {MeshletPartitionState[]}
     * @private
     */
    _partitionStates = [];

    /**
     * Releases settled resources' meshlet records and fetches them back on demand.
     *
     * @type {MeshletRecordStore}
     * @ignore
     */
    _records;

    /**
     * Saves and loads the worlds' cut tables through the byte cache.
     *
     * @type {MeshletCutTableStore}
     * @private
     */
    _cutTables;

    /**
     * Tracks when the content and the components waited on become playable.
     *
     * @type {MeshletPlayableTracker}
     * @ignore
     */
    _playable;

    /**
     * performance.now() of the last world rebuild.
     *
     * @type {number}
     * @private
     */
    _lastRebuild = 0;

    /**
     * {@link _lastRebuild} as of the last record release.
     *
     * @type {number}
     * @private
     */
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

        // a settled restore re-plans the rebuild that waited for it
        this._records = new MeshletRecordStore(() => this._markDirty());
        // the system stays dirty while it waits for a load, so settling needs no nudge: marking it
        // dirty again would restart the quiet window and add rebuilds
        this._cutTables = new MeshletCutTableStore(() => {});
        this._playable = new MeshletPlayableTracker(this);

        // Resource handlers are created before component systems, so register glTF meshlet support
        // here to keep its resource implementation tree-shakeable when this system is omitted.
        const containerHandler = app.loader.getHandler('container');
        if (containerHandler && registerGlbResourceExtension(containerHandler, magMeshlets)) {
            this._containerHandler = containerHandler;
        }

        this.on('beforeremove', this.onBeforeRemove, this);
    }

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
            const settings = this._worldSettings;
            applyWorldSettings(group.directors[0].world, settings, settings.poolBytes, settings.texturePoolBytes);
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
     * Sets a world setting, rebuilding the worlds when it changes.
     *
     * @param {'poolBytes'|'texturePoolBytes'|'maxFineTextureSize'|'maxDrawIndices'|'maxWorldMeshlets'} name - The setting.
     * @param {number} value - Its new value.
     * @private
     */
    _setWorldSetting(name, value) {
        if (this._worldSettings[name] !== value) {
            this._worldSettings[name] = value;
            if (this._group) this._markDirty();
        }
    }

    /**
     * Sets whether the system skips building and syncing meshlet worlds. Components still load
     * their assets, and every resource keeps its meshlet records resident (nothing is released,
     * no pool is allocated, nothing streams or draws), so the content can be read on the CPU
     * alone - an offline bake, for one, can then take the whole memory budget. Defaults to false.
     * Set it before the first world is built: clearing it builds the worlds as usual.
     *
     * @type {boolean}
     */
    set suspended(value) {
        value = !!value;
        if (this._suspended === value) return;
        this._suspended = value;
        if (!value && this._group) this._markDirty();
    }

    /**
     * Gets whether the system skips building and syncing meshlet worlds.
     *
     * @type {boolean}
     */
    get suspended() {
        return this._suspended;
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
        this._setWorldSetting('poolBytes', value);
    }

    /**
     * Gets the geometry page pool byte budget.
     *
     * @type {number}
     */
    get poolBytes() {
        return this._worldSettings.poolBytes;
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
        this._setWorldSetting('texturePoolBytes', value);
    }

    /**
     * Gets the fine-texture slot-pool byte budget.
     *
     * @type {number}
     */
    get texturePoolBytes() {
        return this._worldSettings.texturePoolBytes;
    }

    /**
     * Sets the most meshlets one world holds. A world keeps every meshlet record of its content
     * resident in one buffer (with its LOD-cut tables alongside, ~190 bytes a meshlet), which
     * must fit what the browser allocates and WebGPU binds; content past this is split across
     * further worlds, each with its own share of the budgets and its own cull passes. Clamped
     * to the device's storage binding limit. Takes effect on the next world rebuild. Defaults to
     * 12M (~1.5 GB of records).
     *
     * @type {number}
     */
    set maxWorldMeshlets(value) {
        this._setWorldSetting('maxWorldMeshlets', Math.max(1, Math.floor(value)));
    }

    /**
     * Gets the most meshlets one world holds.
     *
     * @type {number}
     */
    get maxWorldMeshlets() {
        return this._worldSettings.maxWorldMeshlets;
    }

    /**
     * Sets the largest texture size streamed at full resolution. A family's fine slots are sized
     * to its largest texture, so one very large texture would make every slot that large;
     * textures above this size stream down to it. Takes effect on the next world rebuild.
     * Defaults to 2048.
     *
     * @type {number}
     */
    set maxFineTextureSize(value) {
        this._setWorldSetting('maxFineTextureSize', value);
    }

    /**
     * Gets the largest texture size streamed at full resolution.
     *
     * @type {number}
     */
    get maxFineTextureSize() {
        return this._worldSettings.maxFineTextureSize;
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
        this._setWorldSetting('maxDrawIndices', value);
    }

    /**
     * Gets the ceiling on the draw index buffer, in indices (0 = unbounded).
     *
     * @type {number}
     */
    get maxDrawIndices() {
        return this._worldSettings.maxDrawIndices;
    }

    /**
     * Sets the cap on sidecar requests (geometry shards, texture containers) in flight across
     * every streamed asset; the rest wait in priority order. Defaults to 16.
     *
     * @type {number}
     */
    set maxConcurrentFetches(value) {
        if (this.group) this.group.maxConcurrentFetches = value;
    }

    /**
     * Gets the cap on sidecar requests in flight.
     *
     * @type {number}
     */
    get maxConcurrentFetches() {
        return this._group?.maxConcurrentFetches ?? MESHLET_DEFAULTS.maxConcurrentFetches;
    }

    /**
     * Sets the persistent cache for streamed geometry pages and texture regions, or null for
     * none. The same cache can be shared with {@link ResourceLoader#byteCache}.
     *
     * @type {ByteCache|null}
     */
    set byteCache(value) {
        if (this.group) this.group.byteCache = value;
    }

    /**
     * Gets the persistent cache for streamed geometry pages and texture regions.
     *
     * @type {ByteCache|null}
     */
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

    /**
     * Gets the screen-space error threshold in pixels for the DAG LOD cut.
     *
     * @type {number}
     */
    get dagPixelThreshold() {
        return this._group?.dagPixelThreshold ?? MESHLET_DEFAULTS.dagPixelThreshold;
    }

    /**
     * Sets the fraction of {@link rootMinPixels} an instance already drawn may shrink to before
     * it is dropped, so instances do not pop at the cut while the camera moves. Defaults to 0.75.
     *
     * @type {number}
     */
    set rootHysteresis(value) {
        if (this.group) this.group.rootHysteresis = value;
    }

    /**
     * Gets the fraction of {@link rootMinPixels} an instance already drawn may shrink to.
     *
     * @type {number}
     */
    get rootHysteresis() {
        return this._group?.rootHysteresis ?? MESHLET_DEFAULTS.rootHysteresis;
    }

    /**
     * Sets, per draw bucket - opaque, opaque two-sided, alpha-masked - a multiplier on
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

    /**
     * Gets the per draw bucket multipliers on {@link dagPixelThreshold}.
     *
     * @type {number[]}
     */
    get bucketThresholdScales() {
        return this._group?.bucketThresholdScales ?? MESHLET_DEFAULTS.bucketThresholdScales.slice();
    }

    /**
     * Sets, per draw bucket - opaque, opaque two-sided, alpha-masked - a multiplier on
     * {@link rootMinPixels}. Defaults to [1, 1, 1].
     *
     * @type {number[]}
     */
    set bucketRootMinScales(value) {
        if (this.group) this.group.bucketRootMinScales = value;
    }

    /**
     * Gets the per draw bucket multipliers on {@link rootMinPixels}.
     *
     * @type {number[]}
     */
    get bucketRootMinScales() {
        return this._group?.bucketRootMinScales ?? MESHLET_DEFAULTS.bucketRootMinScales.slice();
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

    /**
     * Gets whether two-phase HZB occlusion culling is enabled.
     *
     * @type {boolean}
     */
    get occlusion() {
        return this._group?.occlusionEnabled ?? MESHLET_DEFAULTS.occlusion;
    }

    /**
     * Sets whether root admission skips instances wholly hidden behind last frame's depth, so
     * they spend none of the index and record budget. Occlusion culling alone only saves draw
     * work: without this a dense scene seen from inside spends its budget on roots behind the
     * walls. Only applies while {@link occlusion} is on. An instance that stops being hidden
     * appears a frame later. Defaults to true.
     *
     * @type {boolean}
     */
    set occlusionAdmission(value) {
        if (this.group) this.group.occlusionAdmission = value;
    }

    /**
     * Gets whether root admission skips instances hidden behind last frame's depth.
     *
     * @type {boolean}
     */
    get occlusionAdmission() {
        return this._group?.occlusionAdmission ?? MESHLET_DEFAULTS.occlusionAdmission;
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

    /**
     * Gets the camera meshlets are culled and drawn for.
     *
     * @type {CameraComponent|null}
     */
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

    /**
     * Gets the cameras meshlets are culled and drawn for.
     *
     * @type {CameraComponent[]}
     */
    get cameras() {
        return this._group?.cameras ?? [];
    }

    /**
     * Sets whether the pipeline runs shadow passes - directional cascades and local (spot / omni)
     * shadow faces. Off by default: every shadow face costs one extra cull pass per frame. Which
     * instances cast is per component ({@link MeshletComponent#castShadows}).
     *
     * @type {boolean}
     */
    set shadows(value) {
        if (this.group) this.group.shadowsEnabled = !!value;
    }

    /**
     * Gets whether the pipeline runs shadow passes.
     *
     * @type {boolean}
     */
    get shadows() {
        return this._group?.shadowsEnabled ?? MESHLET_DEFAULTS.shadows;
    }

    /**
     * Sets the multiplier on {@link dagPixelThreshold} for shadow views. Shadow-map texels are
     * coarser than screen pixels, so casters can use a coarser LOD cut than the camera. Defaults
     * to 4.
     *
     * @type {number}
     */
    set shadowThresholdScale(value) {
        if (this.group) this.group.shadowThresholdScale = value;
    }

    /**
     * Gets the multiplier on {@link dagPixelThreshold} for shadow views.
     *
     * @type {number}
     */
    get shadowThresholdScale() {
        return this._group?.shadowThresholdScale ?? MESHLET_DEFAULTS.shadowThresholdScale;
    }

    /**
     * Sets the projected size on screen, in pixels, below which an instance is not drawn. Every
     * drawn instance costs at least its coarsest level, so wide views of worlds with very many
     * small placements are bound by that floor; raising this trades sub-pixel scatter for speed.
     * An instance already drawn stays until it shrinks to {@link rootHysteresis} of this size, to
     * avoid popping. 0 draws every visible instance. Defaults to 3.
     *
     * @type {number}
     */
    set rootMinPixels(value) {
        if (this.group) this.group.rootMinPixels = value;
    }

    /**
     * Gets the projected size on screen, in pixels, below which an instance is not drawn.
     *
     * @type {number}
     */
    get rootMinPixels() {
        return this._group?.rootMinPixels ?? MESHLET_DEFAULTS.rootMinPixels;
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

    /**
     * Gets whether the screen-space thresholds are measured in CSS pixels.
     *
     * @type {boolean}
     */
    get cssPixels() {
        return this._group?.cssPixels ?? MESHLET_DEFAULTS.cssPixels;
    }

    /**
     * Sets the projected size on screen, in pixels, below which an instance casts no meshlet
     * shadow. Far-reaching shadow cascades cover most of the world, and every caster they admit
     * costs at least its coarsest level; skipping casters this small keeps them from taking the
     * index budget the camera needs. 0 lets every visible instance cast. Defaults to 2.
     *
     * @type {number}
     */
    set shadowRootMinPixels(value) {
        if (this.group) this.group.shadowRootMinPixels = value;
    }

    /**
     * Gets the projected size on screen, in pixels, below which an instance casts no meshlet
     * shadow.
     *
     * @type {number}
     */
    get shadowRootMinPixels() {
        return this._group?.shadowRootMinPixels ?? MESHLET_DEFAULTS.shadowRootMinPixels;
    }

    /**
     * Sets whether to load as fast as possible until {@link playable}, then stream at the smooth
     * budgeted pace. Frames may stutter while fast-loading, so cover it with a loading screen;
     * set false to always stream smoothly. Defaults to true.
     *
     * @type {boolean}
     */
    set fastLoad(value) {
        if (this.group) this.group.fastLoad = !!value;
    }

    /**
     * Gets whether to load as fast as possible until {@link playable}.
     *
     * @type {boolean}
     */
    get fastLoad() {
        return this._group?.fastLoad ?? MESHLET_DEFAULTS.fastLoad;
    }

    /**
     * Sets how long component changes must go quiet before the world rebuilds, in milliseconds.
     * Assets loading in a burst then cost a few rebuilds instead of one each (every rebuild
     * re-finalizes the whole world and re-streams what it cannot carry over). Defaults to 250.
     *
     * @type {number}
     */
    set rebuildQuietMs(value) {
        this._rebuildQuietMs = Math.max(0, value);
    }

    /**
     * Gets how long component changes must go quiet before the world rebuilds, in milliseconds.
     *
     * @type {number}
     */
    get rebuildQuietMs() {
        return this._rebuildQuietMs;
    }

    /**
     * Sets the longest a pending rebuild waits for the changes to go quiet, in milliseconds, so a
     * steady trickle of assets still shows up. Defaults to 2000.
     *
     * @type {number}
     */
    set rebuildMaxWaitMs(value) {
        this._rebuildMaxWaitMs = Math.max(0, value);
    }

    /**
     * Gets the longest a pending rebuild waits for the changes to go quiet, in milliseconds.
     *
     * @type {number}
     */
    get rebuildMaxWaitMs() {
        return this._rebuildMaxWaitMs;
    }

    /**
     * Whether the meshlet content can be shown: no component is waiting for its asset to load,
     * no world rebuild is pending, and the streamed geometry and texture tails are resident down
     * to the playable floor - the coarsest LOD free of sloppy simplification. True when there is
     * no meshlet content, or no WebGPU.
     *
     * @type {boolean}
     */
    get playable() {
        return this._playable.isPlayable(true);
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
     * Rebuilds the world now if changes are pending, regardless of the quiet window.
     */
    flushRebuild() {
        if (this._group && this._dirty && !this._suspended) this._rebuild();
    }

    /**
     * Runs `callback` with every placed resource's CPU meshlet records in memory. A scene
     * releases its records once it settles, and a reader of them - a ray tracer building a scene
     * from the meshlets - would find them null. The released records are fetched back first,
     * and dropped again afterwards, with no world rebuild in between: only the records this call
     * restored are released, so a build that needed them in the meantime is not disturbed.
     *
     * Holds the whole scene's records at once while it runs (128 bytes per meshlet).
     *
     * @template T
     * @param {() => (T|Promise<T>)} callback - Reads the records.
     * @returns {Promise<T>} What the callback returns.
     */
    withMeshletRecords(callback) {
        return this._records.withRecords(callback);
    }

    initializeComponentData(component, data) {
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
            baseUrl: meshletComponent.baseUrl,
            castShadows: meshletComponent.castShadows
        });
    }

    onBeforeRemove(entity, component) {
        component.onBeforeRemove();
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
     * Whether an enabled component's asset is still loading (its resource cannot join the world
     * yet).
     *
     * @returns {boolean} True while any such asset is loading.
     * @ignore
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
     * Every component of the system.
     *
     * @yields {MeshletComponent} A component.
     * @private
     */
    *_components() {
        for (const guid in this.store) {
            const component = this.store[guid].entity.meshlet;
            if (component) yield component;
        }
    }

    /** @private */
    _onFrameRender() {
        if (!this._group || this._suspended) return;
        // a rebuild waiting on record restores re-plans once they settle
        if (this._dirty && !this._records.restoring && !this._cutTables.loading) {
            const now = performance.now();
            const quiet = now - this._dirtyLast >= this._rebuildQuietMs;
            const overdue = now - this._dirtySince >= this._rebuildMaxWaitMs;
            // a world with nothing built yet has nothing to show either way: build at once
            const built = this._anyWorldBuilt();
            const waiting = built && this.deferRebuildWhileLoading && this._hasPendingAssets();
            if (!built || (!waiting && (quiet || overdue))) {
                this._rebuild();
            }
        }
        this._syncTransforms();
        this._releaseSettledRecords();

        if (this._playable.updatePlayable()) this.fire(MeshletComponentSystem.EVENT_PLAYABLE);
        this._playable.updateWatch();
    }

    /** @private */
    _rebuild() {
        const settings = this._worldSettings;
        const limit = worldMeshletLimit(this.app.graphicsDevice, settings.maxWorldMeshlets);
        const previous = this._partitionStates;
        const { plans, totalMeshlets } = planWorlds(groupComponents(this._components()), settings, limit, previous, this._group.directors);

        // a world that must rebuild needs its resources' records: records released after an
        // earlier build are fetched back first, and the rebuild stays pending until they arrive
        let restoring = false;
        for (const plan of plans) {
            if (plan.keep) continue;
            for (const group of plan.groups) {
                if (!group.resource.recordsResident) {
                    this._records.restore(group.resource);
                    restoring = true;
                }
            }
        }
        // and the cut tables saved by an earlier build, which spare it deriving them again
        const cache = this.byteCache;
        let loadingTables = false;
        for (const plan of plans) {
            if (plan.keep) continue;
            for (const group of plan.groups) {
                if (this._cutTables.request(group, cache)) loadingTables = true;
            }
        }
        if (restoring || loadingTables) {
            this._dirty = true;
            return;
        }

        this._dirty = false;
        this._dirtySince = 0;
        this._dirtyLast = 0;
        this.rebuildCount++;

        const components = [];
        const placed = new Set();
        const placedResources = new Set();
        const states = [];
        let rebuilt = 0;
        const rebuiltGroups = [];
        this._group.resize(plans.length);
        for (let pi = 0; pi < plans.length; pi++) {
            const { groups, state, keep } = plans[pi];
            if (keep) {
                // unchanged content and budget: the world, its streaming and its placements stay
                states.push(previous[pi]);
                for (const g of groups) {
                    for (const component of g.components) {
                        placed.add(component);
                        components.push(component);
                    }
                }
                continue;
            }
            rebuilt++;
            rebuiltGroups.push(...groups);
            this._group.directors[pi].rebuild((world) => {
                applyWorldSettings(world, settings, state.poolBytes, state.texturePoolBytes);
                buildPartition(world, groups, pi, components);
            });
            for (const g of groups) {
                placedResources.add(g.resource);
                for (const component of g.components) placed.add(component);
            }
            states.push(state);
        }
        this._lastRebuild = performance.now();
        this._partitionStates = states;
        this._records.retain(placedResources);
        this._cutTables.finish(rebuiltGroups, cache);
        // components no longer in any world (removed asset, no base URL) stop syncing
        for (const component of this._activeComponents) {
            if (!placed.has(component)) {
                component._placementIndex = -1;
                component._placedResource = null;
            }
        }
        this._group.assignInstanceOffsets();
        if (plans.length > 1) {
            Debug.log(`MeshletComponentSystem: ${totalMeshlets} meshlets in ${plans.length} worlds (limit ${limit} each), ${rebuilt} rebuilt.`);
        }
        this._activeComponents = components;
    }

    /**
     * Once the scene has settled - no asset loading, no rebuild pending, none for
     * RECORD_RELEASE_DELAY_MS - drops the resources' own copies of their meshlet records (nearly
     * all of each GLB) for every world: each world has uploaded its own. A later rebuild of a
     * world fetches its records back first. Releasing earlier, while assets still arrive, only
     * has them fetched straight back.
     *
     * @private
     */
    _releaseSettledRecords() {
        const states = this._partitionStates;
        if (!states.length || this._dirty || this._records.restoring) return;
        if (performance.now() - this._lastRebuild < RECORD_RELEASE_DELAY_MS) return;
        if (this._recordsReleasedAt === this._lastRebuild || this._hasPendingAssets()) return;
        this._recordsReleasedAt = this._lastRebuild;
        for (let pi = 0; pi < states.length; pi++) {
            for (const group of states[pi].groups) this._records.release(group);
        }
    }

    /**
     * @returns {boolean} True when any partition's world has finished building.
     * @private
     */
    _anyWorldBuilt() {
        const directors = this._group.directors;
        for (let i = 0; i < directors.length; i++) {
            if (directors[i].world.finalized) return true;
        }
        return false;
    }

    /** @private */
    _syncTransforms() {
        const directors = this._group.directors;
        if (!this._anyWorldBuilt()) return;

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
            const hidden = component._removed || !(component.enabled && component.entity.enabled);
            if (component._hiddenApplied !== hidden) {
                component._hiddenApplied = hidden;
                world.setPlacementHidden(component._placementIndex, hidden, component._subBase, component._subCount);
            }
            if (component._removed) {
                component._placementIndex = -1;
                continue;
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
        this._playable.destroy();
        this._records.destroy();
        this._cutTables.destroy();
        this._transformQueue.length = 0;
        this._stateQueue.length = 0;
        this._activeComponents = [];
        this._partitionStates = [];
        this._group?.destroy();
        this._group = null;
        if (this._containerHandler) {
            unregisterGlbResourceExtension(this._containerHandler, magMeshlets.name);
            this._containerHandler = null;
        }
    }
}

export { MeshletComponentSystem };
