import { Debug } from '../../../core/debug.js';
import { basisTranscode } from '../../handlers/basis.js';
import { MeshletDirector } from '../../../scene/meshlet/meshlet-director.js';
import { ComponentSystem } from '../system.js';
import { MeshletComponent } from './component.js';

/**
 * @import { AppBase } from '../../app-base.js'
 * @import { ByteCache } from '../../../platform/net/byte-cache.js'
 * @import { CameraComponent } from '../camera/component.js'
 */

const _properties = ['asset', 'resource', 'baseUrl'];

/**
 * Allows an Entity to render streamed meshlet assets through the GPU-driven meshlet pipeline.
 * WebGPU only - on other devices meshlet components are inert.
 *
 * The system owns the shared `MeshletDirector` and its pipeline-wide settings:
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
 * seconds gets a handful of rebuilds rather than one per asset - and the rebuild RETAINS
 * streaming state for resources that persist across it, so unchanged assets do not re-download. Components sharing
 * one asset are deduplicated into a single set of streamed pages. Enabling/disabling a
 * component (or its entity) and entity transform changes are cheap per-frame updates with no
 * rebuild at all.
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
     * The shared meshlet director, or null on non-WebGPU devices.
     *
     * @type {MeshletDirector|null}
     */
    director = null;

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
    _maxDrawIndices = 0;

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

        if (app.graphicsDevice.isWebGPU) {
            this.director = new MeshletDirector(app.graphicsDevice);
            this.director.world.poolBytes = this._poolBytes;
            this.director.world.initialIndices = 2 * 1024 * 1024;
            this.director.world.maxIndices = this._maxDrawIndices;
            this.director.bindRenderer(app.renderer);
            // streamed textures are KTX2; the scene layer cannot import the Basis handler, so
            // the transcoder is injected here (basisInitialize() decides the worker options)
            this.director.transcode = basisTranscode;
        } else {
            Debug.warnOnce('MeshletComponentSystem: meshlet rendering requires WebGPU; meshlet components will not render.');
        }

        this.on('beforeremove', this.onBeforeRemove, this);

        // rebuild the world and sync entity transforms after the app update, before rendering
        this.app.on('framerender', this._onFrameRender, this);
    }

    /**
     * Sets the geometry page pool byte budget. Under pool pressure the LOD cut coarsens until
     * its working set fits, so a small budget trades quality for memory instead of failing.
     * Takes effect on the next world rebuild. Defaults to 256 MiB.
     *
     * @type {number}
     */
    set poolBytes(value) {
        this._poolBytes = value;
        if (this.director && this.director.world.poolBytes !== value) {
            this.director.world.poolBytes = value;
            this._markDirty();
        }
    }

    get poolBytes() {
        return this._poolBytes;
    }

    /**
     * Sets the fine-texture slot-pool byte budget. Under pressure the texture mip bias rises,
     * pushing more textures onto their always-resident tails, so a small budget trades
     * sharpness for memory instead of failing. Takes effect on the next world rebuild.
     * Defaults to 96 MiB.
     *
     * @type {number}
     */
    set texturePoolBytes(value) {
        if (this.director && this.director.world.texturePoolBytes !== value) {
            this.director.world.texturePoolBytes = value;
            this._markDirty();
        }
    }

    get texturePoolBytes() {
        return this.director?.world.texturePoolBytes ?? 0;
    }

    /**
     * Maximum shader samples for streamed texture anisotropy: 1, 2, 4 or 8. Defaults to
     * 1 (isotropic). Rounded down to a supported level and clamped; non-finite values
     * select 1. Takes effect immediately without reloading assets or rebuilding pools.
     * Higher values improve grazing-angle detail at additional texture-sampling cost.
     *
     * @type {number}
     * @example
     * app.systems.meshlet.textureAnisotropy = 4;
     */
    set textureAnisotropy(value) {
        if (this.director) this.director.world.textureAnisotropy = value;
    }

    get textureAnisotropy() {
        return this.director?.world.textureAnisotropy ?? 1;
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
        if (this.director && this.director.world.maxFineTextureSize !== value) {
            this.director.world.maxFineTextureSize = value;
            this._markDirty();
        }
    }

    get maxFineTextureSize() {
        return this.director?.world.maxFineTextureSize ?? 2048;
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
        this._maxDrawIndices = value;
        if (this.director && this.director.world.maxIndices !== value) {
            this.director.world.maxIndices = value;
            this._markDirty();
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
        if (this.director) this.director.maxConcurrentFetches = value;
    }

    get maxConcurrentFetches() {
        return this.director?.maxConcurrentFetches ?? 16;
    }

    /**
     * Persistent cache for streamed geometry pages and texture regions, or null for none. See
     * {@link MeshletDirector#byteCache}. The same cache can be shared with
     * {@link ResourceLoader#byteCache}.
     *
     * @type {ByteCache|null}
     */
    set byteCache(value) {
        if (this.director) this.director.byteCache = value;
    }

    get byteCache() {
        return this.director?.byteCache ?? null;
    }

    /**
     * Sets the screen-space error threshold in pixels for the DAG LOD cut. Defaults to 1.
     *
     * @type {number}
     */
    set dagPixelThreshold(value) {
        if (this.director) this.director.dagPixelThreshold = value;
    }

    get dagPixelThreshold() {
        return this.director?.dagPixelThreshold ?? 1;
    }

    /**
     * Sets whether two-phase HZB occlusion culling is enabled. Requires the camera to render
     * into a render target with a depth texture. Falls back to single-phase culling
     * without one. Defaults to true.
     *
     * @type {boolean}
     */
    set occlusion(value) {
        if (this.director) this.director.occlusionEnabled = value;
    }

    get occlusion() {
        return this.director?.occlusionEnabled ?? false;
    }

    /**
     * Sets the camera meshlets are culled and drawn for. When null (the default), the first
     * camera of the layer composition is used.
     *
     * @type {CameraComponent|null}
     */
    set camera(value) {
        if (this.director) this.director.cameraComponent = value;
    }

    get camera() {
        return this.director?.cameraComponent ?? null;
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
        if (this.director) this.director.cameras = value ?? [];
    }

    get cameras() {
        return this.director?.cameras ?? [];
    }

    /**
     * Whether the pipeline runs shadow passes - directional cascades and local (spot / omni)
     * shadow faces. Off by default: every shadow face costs one extra cull pass per frame. Which
     * instances cast is per component ({@link MeshletComponent#castShadows}).
     *
     * @type {boolean}
     */
    set shadows(value) {
        if (this.director) this.director.shadowsEnabled = !!value;
    }

    get shadows() {
        return this.director?.shadowsEnabled ?? false;
    }

    /**
     * Multiplier on {@link dagPixelThreshold} for shadow views. Shadow-map texels are coarser
     * than screen pixels, so casters can use a coarser LOD cut than the camera. Defaults to 4.
     *
     * @type {number}
     */
    set shadowThresholdScale(value) {
        if (this.director) this.director.shadowThresholdScale = value;
    }

    get shadowThresholdScale() {
        return this.director?.shadowThresholdScale ?? 4;
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
        if (this.director) this.director.rootMinPixels = value;
    }

    get rootMinPixels() {
        return this.director?.rootMinPixels ?? 3;
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
        if (this.director) this.director.shadowRootMinPixels = value;
    }

    get shadowRootMinPixels() {
        return this.director?.shadowRootMinPixels ?? 2;
    }

    initializeComponentData(component, data, properties) {
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
     * {@link MeshletDirector#playable}). True when there is no meshlet content, or no WebGPU.
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
        const director = this.director;
        if (!director) return true;
        if (this._dirty) return false;
        // a world with nothing in it is never finalized, and has nothing to stream
        if (director.world.finalized && !director.playable) return false;
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
     * always stream smoothly. Defaults to true. See {@link MeshletDirector#fastLoad}.
     *
     * @type {boolean}
     */
    set fastLoad(value) {
        if (this.director) this.director.fastLoad = !!value;
    }

    get fastLoad() {
        return this.director?.fastLoad ?? true;
    }

    /**
     * @param {MeshletComponent} component - The component.
     * @returns {import('./component.js').MeshletStreamStatus} Its load state.
     * @ignore
     */
    _componentStatus(component) {
        const director = this.director;
        if (!director) return { playable: true, failed: false, progress: 1, totalBytes: 0, loadedBytes: 0 };
        const resource = component._effectiveResource;
        // not placed yet: the asset is loading, or the rebuild that adds it has not run
        if (!resource || component._placementIndex < 0 || component._placedResource !== resource) {
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
        if (this.director && this._dirty) this._rebuild();
    }

    /** @private */
    _onFrameRender() {
        if (!this.director) return;
        if (this._dirty) {
            const now = performance.now();
            const quiet = now - this._dirtyLast >= this._rebuildQuietMs;
            const overdue = now - this._dirtySince >= this._rebuildMaxWaitMs;
            // a world with nothing built yet has nothing to show either way: build at once
            if (quiet || overdue || !this.director.world.finalized) {
                this._rebuild();
            }
        }
        this._syncTransforms();

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
        this._dirty = false;
        this._dirtySince = 0;
        this._dirtyLast = 0;
        this.rebuildCount++;

        // group components by (resource, baseUrl): each unique resource is added ONCE, with a
        // merged instance list - N components share one set of pages, records and textures.
        // Disabled components are included (hidden via the per-instance hide bit) so a later
        // enable is a buffer write, not a rebuild.
        const groups = [];
        for (const guid in this.store) {
            const component = this.store[guid].entity.meshlet;
            if (!component) continue;
            // left at -1 unless this rebuild places it, so the sync skips unplaced components
            component._placementIndex = -1;
            component._placedResource = null;
            if (!component._effectiveResource) continue;
            if (!component._effectiveBaseUrl) {
                Debug.warnOnce('MeshletComponent: no stream base URL - assign an asset with an http(s) file URL (not a blob: or data: URL), or set baseUrl with the resource.');
                continue;
            }
            const resource = component._effectiveResource;
            const baseUrl = component._effectiveBaseUrl;
            let group = groups.find(g => g.resource === resource && g.baseUrl === baseUrl);
            if (!group) {
                group = { resource, baseUrl, fetchOptions: component._effectiveFetchOptions, components: [] };
                groups.push(group);
            }
            group.components.push(component);
        }

        const components = [];
        this.director.rebuild((world) => {
            for (let gi = 0; gi < groups.length; gi++) {
                const { resource, baseUrl, fetchOptions, components: members } = groups[gi];
                const per = resource.instances.length;

                // one placement per group; each member owns a sub-range of its instances
                let merged = null;
                if (members.length > 1) {
                    merged = [];
                    for (let mi = 0; mi < members.length; mi++) merged.push(...resource.instances);
                }
                for (let mi = 0; mi < members.length; mi++) {
                    const component = members[mi];
                    component._placementIndex = gi;
                    component._placedResource = resource;
                    component._subBase = mi * per;
                    component._subCount = per;
                    component._transformDirty = true;
                    component._hiddenApplied = null;
                    component._castShadowsApplied = null;
                    component.onWorldDirty();
                    component._queueState();
                    components.push(component);
                }
                world.addStreamedResource(resource, null, baseUrl, merged, fetchOptions);
            }
        });
        this._activeComponents = components;
    }

    /** @private */
    _syncTransforms() {
        const world = this.director.world;
        if (!world.finalized) return;

        const transforms = this._transformQueue;
        for (let i = 0; i < transforms.length; i++) {
            const component = transforms[i];
            component._transformQueued = false;
            if (component._placementIndex < 0) continue;
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
        this.director?.destroy();
        this.director = null;
    }
}

export { MeshletComponentSystem };
