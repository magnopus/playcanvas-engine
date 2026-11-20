import { Debug } from '../../core/debug.js';
import { Frustum } from '../../core/shape/frustum.js';
import { Mat4 } from '../../core/math/mat4.js';
import { math } from '../../core/math/math.js';
import { Vec3 } from '../../core/math/vec3.js';
import { LAYERID_WORLD } from '../constants.js';
import { MESHLET_BUCKET_COUNT } from './constants.js';
import { MeshletBudgetManager } from './meshlet-budget-manager.js';
import { MeshletRebuildCarry } from './meshlet-retention.js';
import { MeshletShadowRenderer } from './meshlet-shadow-renderer.js';
import { MeshletView, sumBuckets } from './meshlet-view.js';
import { MeshletWorld } from './meshlet-world.js';
import { MeshletResidency } from './streaming/meshlet-residency.js';

/**
 * @import { CameraComponent } from '../../framework/components/camera/component.js'
 * @import { Color } from '../../core/math/color.js'
 * @import { ForwardRenderer } from '../renderer/forward-renderer.js'
 * @import { Layer } from '../layer.js'
 * @import { LayerComposition } from '../composition/layer-composition.js'
 * @import { MeshInstance } from '../mesh-instance.js'
 * @import { MeshletCullShaders } from './meshlet-cull-shaders.js'
 * @import { MeshletDirectorGroup } from './meshlet-director-group.js'
 * @import { MeshletFetchScheduler } from './streaming/meshlet-fetch-scheduler.js'
 * @import { MeshletTranscodeFn } from './textures/meshlet-ktx2.js'
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { RenderTarget } from '../../platform/graphics/render-target.js'
 * @import { StorageBuffer } from '../../platform/graphics/storage-buffer.js'
 */

/**
 * What {@link MeshletDirector#onBudgetExceeded} receives.
 *
 * @typedef {object} MeshletBudgetReport
 * @property {number} budgetBytes - The configured geometry budget.
 * @property {number} suggestedBudgetBytes - An estimate of the budget the visible set needs; not
 * a coverage guarantee.
 * @property {number} indexDemandRatio - Index demand over the view's share.
 * @property {number} indexCeiling - The world's index ceiling.
 * @property {number} pagesMissing - Pages the cut wanted that were not resident.
 * @property {number} poolPages - Pages the page pool holds.
 * @property {number} totalPages - Pages of the world.
 * @property {number} rootInstancesDeferred - Visible instances deferred for capacity (0 when
 * reporting a sustained overrun).
 * @property {number} rootInstancesUnready - Visible instances whose root pages were not resident.
 * @property {number} rootShortfallPages - Root pages the pool cannot pin.
 * @ignore
 */

const _viewProj = new Mat4();
const _frustum = new Frustum();

const BYTES_PER_MB = 1024 * 1024;

// Every culled view (camera or shadow face) takes MESHLET_ROOT_BANDS + 5 indirect dispatch slots
// a frame - one per root admission band - so the device default of 256 runs out at a dozen views.
const MIN_INDIRECT_DISPATCH_SLOTS = 1024;

// World settings a rebuild carries into the fresh world.
const WORLD_SETTINGS = [
    'poolBytes', 'texturePoolBytes', 'maxFineTextureSize', 'maxInstallBytesPerFrame', 'initialIndices',
    'initialRecords', 'indexBudgetFraction', 'rootBudgetFraction'
];

/**
 * Defaults of the director settings a {@link MeshletDirectorGroup} applies to every director;
 * each is documented on the director property of the same name.
 *
 * @ignore
 */
const MESHLET_DIRECTOR_SETTINGS = Object.freeze({
    dagPixelThreshold: 2,
    occlusionEnabled: true,
    occlusionAdmission: true,
    shadowsEnabled: false,
    shadowThresholdScale: 4,
    rootMinPixels: 3,
    rootHysteresis: 0.75,
    bucketThresholdScales: Object.freeze([4, 1, 1]),
    bucketRootMinScales: Object.freeze([1, 1, 1]),
    shadowRootMinPixels: 2,
    shadowMatchCameraLod: false,
    cssPixels: true,
    fastLoad: true
});

/**
 * Drives one meshlet world of a {@link MeshletDirectorGroup}: its streaming residency and
 * budget, a {@link MeshletView} per camera with its per-frame culling compute (encoded ahead of
 * the frame graph), and its shadow views. The group registers with the renderer and emits the
 * draw passes covering every world's views.
 *
 * Wire-up: add resources to {@link world} and call {@link finalize}, or {@link rebuild};
 * assign the camera to {@link cameraComponent}. For occlusion the camera must render into a
 * render target with a depth texture, or through a CameraFrame.
 *
 * @ignore
 */
class MeshletDirector {
    /** @type {GraphicsDevice} */
    device;

    /** @type {MeshletDirectorGroup} - the group this director is one world of. */
    group;

    /** @type {MeshletWorld} */
    world;

    /** @type {Map<CameraComponent, MeshletView>} - one per active camera. */
    views = new Map();

    /**
     * The cameras this director culls and draws for. Empty means the first camera of the
     * layer composition. Capped at {@link maxViews}.
     *
     * @type {CameraComponent[]}
     */
    cameras = [];

    /** @type {number} - per-view GPU state is real memory; cap and warn beyond it. */
    maxViews = 2;

    /** @type {MeshletResidency|null} */
    residency = null;

    /** @type {MeshletBudgetManager} */
    budget = new MeshletBudgetManager();

    /** @type {ForwardRenderer|null} - set by the group. */
    renderer = null;

    /**
     * The scheduler every sidecar fetch goes through - the group's, shared by every world, so
     * the concurrency cap and request priorities span them all.
     *
     * @type {MeshletFetchScheduler}
     */
    fetchScheduler;

    /**
     * The cull compute shaders - the group's, compiled once and shared by every view's culler
     * (the shadow phase alone adds one culler per face).
     *
     * @type {MeshletCullShaders|null}
     */
    cullShaders = null;

    /** @type {boolean} - two-phase HZB occlusion culling, see {@link occlusionEnabled}. */
    _occlusionEnabled = MESHLET_DIRECTOR_SETTINGS.occlusionEnabled;

    /** @type {Float32Array} */
    _frustumPlanes = new Float32Array(24);

    /** @type {number} - screen-space error threshold in pixels for the DAG LOD cut. */
    dagPixelThreshold = MESHLET_DIRECTOR_SETTINGS.dagPixelThreshold;

    /**
     * Render target pixels per CSS pixel of the last camera view updated (at least 1): the factor
     * the screen-space thresholds are scaled by when {@link cssPixels} is on, else 1.
     *
     * @type {number}
     * @ignore
     */
    pixelScale = 1;

    /**
     * What the last rebuild hands from the previous world's views to the new ones, taken as
     * the views are created.
     *
     * @type {MeshletRebuildCarry|null}
     * @private
     */
    _carry = null;

    /**
     * Buffers a rebuild copied from, destroyed one frame later (after their commands submit).
     *
     * @type {StorageBuffer[]}
     * @private
     */
    _retired = [];

    /**
     * Cast shadows from the meshlet geometry: directional cascades and, unless
     * `shadowRenderer.localLights` is switched off, spot and omni faces. Off by default - it adds
     * one single-phase cull chain per shadow face, so it is opt-in per scene.
     *
     * @type {boolean}
     */
    shadowsEnabled = MESHLET_DIRECTOR_SETTINGS.shadowsEnabled;

    /**
     * Multiplier on {@link dagPixelThreshold} for shadow views. Shadow-map texels are coarser
     * than screen pixels and a blurred depth comparison hides geometric detail, so casters can
     * run a far coarser cut than the camera. Also the release valve when shadow views enlarge
     * the streaming working set enough to coarsen the camera's own cut.
     *
     * @type {number}
     */
    shadowThresholdScale = MESHLET_DIRECTOR_SETTINGS.shadowThresholdScale;

    /**
     * Projected size on screen, in pixels, below which an instance is not drawn at all. Every
     * drawn instance costs at least its coarsest level, so a wide view of a world with very many
     * small placements is bound by that floor however coarse the cut gets, and seen from afar
     * most such placements are under a pixel. An instance already drawn stays until it shrinks
     * to {@link rootHysteresis} of this size, so the cut does not pop while the camera moves. 0
     * draws every visible instance.
     *
     * @type {number}
     */
    rootMinPixels = MESHLET_DIRECTOR_SETTINGS.rootMinPixels;

    /**
     * The fraction of {@link rootMinPixels} (scaled for its bucket) an instance already drawn may
     * shrink to before it is dropped. Lower holds instances longer - at 0.5 a 3 pixel cut keeps
     * drawing them down to 1.5 pixels.
     *
     * @type {number}
     */
    rootHysteresis = MESHLET_DIRECTOR_SETTINGS.rootHysteresis;

    /**
     * Per draw bucket - opaque, opaque two-sided, alpha-masked - a multiplier on
     * {@link dagPixelThreshold}, for the camera and shadow cuts alike. Solid geometry usually
     * takes a much coarser cut than foliage before it shows, while foliage's simplified levels
     * thin out rather than coarsen. The default puts opaque geometry at four times the base
     * threshold - with the default threshold of 2, 8 pixels for solid geometry and 2 for foliage,
     * which keeps a wide view of a large world near one triangle per pixel.
     *
     * @type {number[]}
     */
    bucketThresholdScales = MESHLET_DIRECTOR_SETTINGS.bucketThresholdScales.slice();

    /**
     * Per draw bucket, a multiplier on {@link rootMinPixels} (and {@link shadowRootMinPixels}).
     *
     * @type {number[]}
     */
    bucketRootMinScales = MESHLET_DIRECTOR_SETTINGS.bucketRootMinScales.slice();

    /**
     * Measure the screen-space thresholds - {@link dagPixelThreshold}, {@link rootMinPixels} and
     * {@link shadowRootMinPixels} - in CSS pixels instead of render target pixels, for cameras
     * rendering to the canvas. On a high-density display a render target pixel is a fraction of
     * a CSS pixel, and a cut measured in render target pixels refines with the density squared:
     * at a pixel ratio of 2 it draws about four times the geometry, mostly detail too small to
     * see. The thresholds never go finer than one render target pixel, so a render scale below
     * the pixel ratio still coarsens the cut.
     *
     * @type {boolean}
     */
    cssPixels = MESHLET_DIRECTOR_SETTINGS.cssPixels;

    /**
     * Shadow casters also refine at least as far as the camera's own cut does. The shadow cut is
     * otherwise measured in shadow-map texels, so a far cascade's casters can be far coarser
     * than the surfaces the camera draws. The shadow pass already pushes each simplified
     * cluster away from the light by its error, so a coarse caster cannot shadow the surface it
     * stands for; matching the camera's cut is only a further refinement, at a large cost in
     * wide views (every caster of a far cascade refined to camera detail).
     *
     * @type {boolean}
     */
    shadowMatchCameraLod = MESHLET_DIRECTOR_SETTINGS.shadowMatchCameraLod;

    /**
     * Projected size on screen, in pixels, below which an instance casts no meshlet shadow. A
     * cascade reaching far from the camera covers most of the world, and every instance it
     * admits costs at least its root - so without this cut the far cascades' root floor alone
     * outgrows the camera's whole index budget. A caster this small on screen throws a shadow
     * no wider than it.
     *
     * @type {number}
     */
    shadowRootMinPixels = MESHLET_DIRECTOR_SETTINGS.shadowRootMinPixels;

    /**
     * Index budget of a shadow face, per texel of its part of the shadow map (three indices to a
     * triangle), or 0 for no cap. Shadow faces otherwise share whatever index budget the cameras
     * leave, which on a large memory budget lets them refine far below the shadow map's
     * resolution - detail it cannot show, at full raster and cull cost. Over the cap the shadow
     * LOD controller coarsens them as for any other budget.
     *
     * @type {number}
     */
    shadowIndicesPerTexel = 8;

    /** @private */
    _transcode = null;

    /** @type {number} - the budget value the shortfall warning was last printed for. @private */
    _budgetWarnedAt = -1;

    /**
     * The KTX2 transcoder used for streamed textures - a `basisTranscode`-compatible function.
     * Scene code cannot import the framework's Basis handler, so the transcoder is injected:
     * {@link MeshletComponentSystem} sets it on the group, which hands it to every director.
     *
     * @type {MeshletTranscodeFn|null}
     */
    set transcode(value) {
        this._transcode = value;
        this.world.transcode = value;
    }

    get transcode() {
        return this._transcode;
    }

    /**
     * Whether the streamed content can be shown: every resource's root pages and its playable
     * floor - the coarsest LOD level free of sloppy (vertex-clustered) simplification - are
     * resident, as are the low-resolution tails of the streamed textures. Bakes without sloppy
     * flags instead need the current view's LOD cut to be fully resident. Finer detail keeps
     * streaming afterwards. False while a rebuild's content is still loading.
     *
     * @type {boolean}
     */
    get playable() {
        return this._playable;
    }

    /**
     * Load state of one streamed resource in the current world: its root pages, its playable
     * floor and its texture tails (see {@link playable}, which is the same test across every
     * resource).
     *
     * @param {object} resource - The MeshletResource.
     * @returns {{ playable: boolean, failed: boolean, progress: number, totalBytes: number, loadedBytes: number }|null}
     * Whether the resource is playable, whether its stream was given up on (it then counts as
     * playable, so nothing waits on it forever), its progress 0..1 and the bytes that progress
     * weighs; null before the world is finalized. Only meaningful for a resource placed in this
     * world (a resource it does not hold reads as resident).
     */
    resourceStatus(resource) {
        const world = this.world;
        if (!world.finalized) return null;
        // a resource with no stream was uploaded fully resident at finalize
        const geometry = this.residency?.streamStatus(resource) ?? { playable: true, failed: false, totalBytes: 0, loadedBytes: 0 };
        const tails = (resource.textureManifest && world.textures) ?
            world.textures.tailStatus(resource.textureManifest) :
            { pending: 0, totalBytes: 0, loadedBytes: 0 };
        const playable = geometry.playable && (geometry.failed || tails.pending === 0);
        const totalBytes = geometry.totalBytes + tails.totalBytes;
        const loadedBytes = geometry.loadedBytes + tails.loadedBytes;
        return {
            playable,
            failed: geometry.failed,
            progress: playable ? 1 : (totalBytes > 0 ? Math.min(loadedBytes / totalBytes, 0.999) : 0),
            totalBytes,
            loadedBytes
        };
    }

    /**
     * The shared scheduler loads fast while any of the group's worlds is fast-loading.
     *
     * @private
     */
    _applyFetchConcurrency() {
        this.group._applyFetchConcurrency();
    }

    /**
     * Re-evaluates {@link playable} and the fast-load phase. Runs each frame after the residency
     * update.
     *
     * @private
     */
    _updatePlayable() {
        const residency = this.residency;
        const playable = (!this.world.streamed || !!residency?.playable) && (this.world.textures?.tailsLoaded ?? true);
        if (residency) {
            const fast = this.fastLoad && !playable;
            if (residency.fastLoad !== fast) {
                residency.fastLoad = fast;
                this._applyFetchConcurrency();
            }
        }
        if (playable !== this._playable) {
            this._playable = playable;
            if (playable) this.onPlayable?.();
        }
    }

    /**
     * Creates the streaming residency for the finalized world and starts loading its roots and
     * playable floor.
     *
     * @param {object|null} carriedSlots - Slot state carried across a rebuild, or null.
     * @private
     */
    _startStreaming(carriedSlots) {
        this.residency = new MeshletResidency(this.device, this.world, carriedSlots);
        // fast from the first request: the floor fetches below are issued right away
        this.residency.fastLoad = this.fastLoad;
        this._applyFetchConcurrency();
        this.rootsLoaded = this.residency.loadRoots();
        this.residency.loadFloor();
        // the request readback carries the texture-mip feedback marks after the page marks
        if (this.world.textures) {
            this.residency.onTexelRateMarks = marks => this.world.textures?.processMarks(marks);
        }
    }

    /**
     * Fraction of the world's initial index budget a shadow cascade starts with. The budget is
     * tuned for the camera; a cascade's cut is coarser and its demand far lower, so matching it
     * would reserve memory growth would never ask for. Growth still reacts within a frame or
     * two, and a briefly clamped shadow is far less visible than clamped camera geometry.
     *
     * @type {number}
     */
    shadowInitialIndexScale = 0.25;

    /**
     * Shadow faces to assume when reserving the geometry budget. The real count is not known
     * until the light set is resolved, well after the world is built, so the budget assumes a
     * typical CSM. Raise it when the scene has shadow-casting local lights - a spot is one more
     * face, an omni six.
     *
     * @type {number}
     */
    shadowBudgetCascades = 3;

    /**
     * Called when visible coarse instances are deferred, or sustained demand exceeds the
     * budget at maximum LOD pressure, with budget, index demand and missing-page statistics.
     * The configured budget stays unchanged. Complete admitted instances continue rendering;
     * displaying the entire visible set requires cheaper coarse geometry or more capacity.
     *
     * @type {((info: MeshletBudgetReport) => void)|null}
     */
    onBudgetExceeded = null;

    /**
     * The scene camera's position and projection scale from its last cull, which shadow views
     * use to rank root admission (see MeshletCuller#priorityOrigin).
     *
     * @type {Vec3}
     * @ignore
     */
    priorityOrigin = new Vec3();

    /** @type {number} @ignore */
    priorityProjScale = 0;

    /** @type {number} - consecutive infeasible frames before {@link onBudgetExceeded} fires. */
    budgetOverrunFrames = 120;

    /** @type {Layer|null} - the outline renderer's layer, if registered. */
    _outlineLayer = null;

    /** @type {Float32Array} - selection outline colour applied to every view's instances. */
    _outlineColor = new Float32Array([1, 1, 1]);

    /** @type {Float32Array} - hover outline colour. */
    _hoverColor = new Float32Array([1, 1, 1]);

    /** @type {boolean} - whether the geometry is currently attached to the outline layer. */
    _outlineAttached = false;

    /** @type {MeshletShadowRenderer|null} */
    shadowRenderer = null;

    /** @type {Layer|null} - the layer whose lights shade the meshlet draws, resolved each update. */
    _lightLayer = null;

    /** @type {Promise<void>|null} - resolves when demand streaming is initialized. */
    rootsLoaded = null;

    /**
     * Load as fast as possible until the world is {@link playable}: arrived pages upload
     * without the per-frame budget and up to the group's `fastLoadConcurrentFetches` requests go
     * out at once. Frames may stutter meanwhile, so hide the load behind a loading screen. Once
     * playable, streaming returns to its smooth, budgeted pace; a rebuild that adds content
     * fast-loads again until that content is playable. Defaults to true.
     *
     * @type {boolean}
     */
    fastLoad = MESHLET_DIRECTOR_SETTINGS.fastLoad;

    /**
     * Called each time the world becomes {@link playable}.
     *
     * @type {(() => void)|null}
     */
    onPlayable = null;

    /** @type {boolean} @private */
    _playable = false;

    /**
     * Ceiling on how much a rebuild may pre-grow a view's carried index capacity (see
     * MeshletRebuildCarry#scaleCapacities).
     *
     * @type {number}
     */
    maxCarryScale = 2;

    /**
     * @param {MeshletDirectorGroup} group - The group this director is one world of; its fetch
     * scheduler and cull shaders are shared.
     */
    constructor(group) {
        const device = group.device;
        this.device = device;
        this.group = group;
        this.renderer = group.renderer;
        this.fetchScheduler = group.fetchScheduler;
        this.cullShaders = group.cullShaders;
        device.maxIndirectDispatchCount = Math.max(device.maxIndirectDispatchCount, MIN_INDIRECT_DISPATCH_SLOTS);
        this.world = this._createWorld(null);
        this.shadowRenderer = new MeshletShadowRenderer(device, this);
    }

    destroy() {
        this._destroyViews();
        this._carry?.destroy();
        this._carry = null;
        this._retired.forEach(b => b.destroy());
        this._retired.length = 0;
        this.residency?.destroy();
        this.world.destroy();
        this.shadowRenderer?.destroy();
        this.shadowRenderer = null;
        this.cullShaders = null;
    }

    _destroyViews() {
        this.views.forEach(v => this._destroyView(v));
        this.views.clear();
    }

    /**
     * @param {MeshletView} view - The view to destroy, after taking it off the outline layer.
     * @private
     */
    _destroyView(view) {
        this._outlineLayer?.removeMeshInstances(view.meshInstances.slice(0, MESHLET_BUCKET_COUNT));
        view.destroy();
    }

    /**
     * Creates an empty world with the settings of the one it replaces.
     *
     * @param {MeshletWorld|null} prev - The world being replaced, or null.
     * @returns {MeshletWorld} The world.
     * @private
     */
    _createWorld(prev) {
        const world = new MeshletWorld(this.device);
        if (prev) {
            for (const name of WORLD_SETTINGS) world[name] = prev[name];
        }
        world.transcode = this._transcode;
        world.fetchScheduler = this.fetchScheduler;
        return world;
    }

    /**
     * Finalizes the world and builds the culler. Call after all resources are added. For a
     * streamed world this enables demand streaming: root pages load for the active view, and an
     * instance becomes drawable once its complete coarse fallback is resident.
     *
     * @returns {MeshletWorld} The finalized world.
     */
    finalize() {
        this._applyBudgetViews();
        this.world.finalize();
        this._playable = false;
        if (this.world.streamed) {
            this._startStreaming(null);
        }
        return this.world;
    }

    /**
     * Tears down the current world and rebuilds it - the component system's add/remove path.
     * Streaming state is retained where possible (see MeshletRebuildCarry#adoptStreamingState):
     * the page pool and its resident pages for an unchanged leading run of resources, the
     * texture system for an unchanged textured prefix, and the views' visibility bits and grown
     * capacities. The colour mode carries too. An empty build leaves the director idle.
     *
     * @param {function(MeshletWorld): void} build - Adds resources to the fresh world.
     * @returns {MeshletWorld} The new world (finalized unless empty).
     */
    rebuild(build) {
        const prevWorld = this.world;
        const prevResidency = this.residency;
        const prevViews = this.views;
        const prevPairs = prevWorld.totalPairs || 1;
        const colorMode = prevWorld.colorMode;
        this.views = new Map();
        this._carry?.destroy();
        const carry = this._carry = new MeshletRebuildCarry();
        carry.captureCapacities(prevViews, this.shadowRenderer);
        // shadow views hold buffers sized to the old world: dropped, and rebuilt next frame
        this.shadowRenderer?.reset();
        this.residency = null;
        this.rootsLoaded = null;
        this._playable = false;
        this._applyFetchConcurrency();

        this.world = this._createWorld(prevWorld);
        this._outlineAttached = false;
        this.budget.reset();
        build(this.world);
        if (this.world.pendingCount === 0) {
            prevViews.forEach(v => this._destroyView(v));
            prevResidency?.destroy();
            prevWorld.destroy();
            return this.world;
        }

        const carriedSlots = (prevWorld.finalized && prevWorld.streamed && prevResidency) ?
            carry.adoptStreamingState(this.world, prevWorld, prevResidency, prevViews) : null;

        prevViews.forEach(v => this._destroyView(v));
        prevResidency?.destroy();
        prevWorld.destroy();

        this._applyBudgetViews();
        this.world.finalize();
        if (colorMode) this.world.setColorMode(colorMode);
        carry.scaleCapacities(this.world.totalPairs / prevPairs, this.maxCarryScale);
        if (this.world.streamed) {
            this._startStreaming(carriedSlots);
        }
        return this.world;
    }

    /**
     * Single-camera convenience accessor over {@link cameras}: assigns the one camera this
     * director culls and draws for (null restores the default - the composition's first
     * camera).
     *
     * @type {CameraComponent|null}
     */
    set cameraComponent(camera) {
        this.cameras = camera ? [camera] : [];
    }

    get cameraComponent() {
        return this.cameras[0] ?? null;
    }

    /**
     * Per-frame tick, called by the group before the frame graph is built. Runs the streaming
     * residency update and encodes the phase-1 culling chain of every view.
     *
     * @param {LayerComposition} [comp] - The layer composition, used to pick a camera when
     * {@link cameraComponent} is not set (the first composition camera).
     */
    update(comp) {
        if (!this.world.finalized) return;

        // the layer whose lights shade the meshlet draws (splitLights + clusters + light hash)
        this._lightLayer = comp?.getLayerById?.(LAYERID_WORLD) ?? null;

        if (this._retired.length) {
            this._retired.forEach(b => b.destroy());
            this._retired.length = 0;
        }

        // shared per-frame work: streaming, textures, budget - once, not per view
        this.residency?.frameUpdate();
        this.world.textures?.flush();
        this._updatePlayable();
        // index starvation feeds the same controller as page pressure: both say the cut is
        // too fine for the memory it has, and the answer to both is to coarsen it
        const budget = this.budget;
        const shadowViews = this.shadowRenderer?.views ?? null;
        const indexRatio = budget.distributeIndexBudget(this.world.indexCeiling, this.views, shadowViews, this.shadowIndicesPerTexel);
        budget.update(this.residency, indexRatio);
        budget.updateShadowPressure(shadowViews);
        this._checkBudgetFeasible(indexRatio);

        // resolve this frame's camera set
        let cameras = this.cameras.length ? this.cameras : (comp?.cameras?.[0] ? [comp.cameras[0]] : []);
        if (cameras.length > this.maxViews) {
            Debug.warnOnce(`MeshletDirector: ${cameras.length} cameras exceed maxViews (${this.maxViews}); extra cameras are ignored.`);
            cameras = cameras.slice(0, this.maxViews);
        }

        // drop views whose camera left the set
        this.views.forEach((view, cam) => {
            if (!cameras.includes(cam)) {
                this._destroyView(view);
                this.views.delete(cam);
                this._invalidateCameraFrame(cam);
            }
        });

        for (const cameraComponent of cameras) {
            let view = this.views.get(cameraComponent);
            if (!view) {
                const visCarry = this._carry?.takeVisBits(cameraComponent) ?? null;
                view = new MeshletView(this.device, this.world, cameraComponent, {
                    visCarry,
                    capCarry: this._carry?.takeCaps(cameraComponent) ?? null,
                    cullShaders: this.cullShaders
                });
                this.views.set(cameraComponent, view);
                // a CameraFrame builds its pass chain once and caches it, so it has to be told
                // to rebuild now that there are meshlet passes to splice in
                this._invalidateCameraFrame(cameraComponent);
                if (this._outlineLayer) this._registerOutlineView(view);
                // the copy is encoded; its source must outlive this frame's submission
                if (visCarry) this._retired.push(visCarry.buffer);
            }
            this._updateView(view);
        }
    }

    /**
     * Per-view frame tick: materials, index demand, projection and HZB setup and the phase-1
     * cull chain encode.
     *
     * @param {MeshletView} view - The view.
     * @private
     */
    _updateView(view) {
        const cameraComponent = view.cameraComponent;
        view.syncMaterials();
        view.culler.pressureScale = this.budget.pressureScale;
        // grow first, then read demand: buffers must not change once the frame is encoding
        view.applyPendingGrowth();
        view.monitorIndexDemand();

        // With a CameraFrame the camera's own renderTarget is the output (often the backbuffer);
        // the scene renders into the frame pass's internal target - which holds the depth the
        // HZB needs and, with a render scale, the resolution the cut is measured against.
        const rt = cameraComponent.framePasses?.[0]?.rt ?? cameraComponent.renderTarget;
        const projScale = this._setupProjection(view, rt);
        const useOcclusion = this._setupOcclusion(view, rt);
        view.culler.beginFrame(this._frustumPlanes, cameraComponent.entity.getPosition(), projScale, _viewProj.data,
            useOcclusion ? view.hzb : null);
        this._reportDeferredRoots(view);
    }

    /**
     * Sets up a view's frustum, projection scale and screen-space thresholds for this frame.
     *
     * @param {MeshletView} view - The view.
     * @param {RenderTarget|null} rt - The target the scene renders into.
     * @returns {number} The projection scale: pixels per unit of size at unit distance.
     * @private
     */
    _setupProjection(view, rt) {
        const cameraComponent = view.cameraComponent;
        const camera = cameraComponent.camera;
        _viewProj.mul2(camera.projectionMatrix, camera.viewMatrix);
        _frustum.setFromMat4(_viewProj);
        // same packed layout the cull shaders read: normal xyz, distance, per plane
        this._frustumPlanes.set(_frustum.planeData);

        const viewportHeight = rt?.height ?? this.device.height;
        const viewportWidth = rt?.width ?? this.device.width;
        // the projection scale wants the vertical field of view; a camera set up with a
        // horizontal one (the wide-screen convention) reports that in `fov`, which would
        // under-rate every cluster's screen size - and with it the LOD cut and the texture
        // demand - by up to the aspect ratio
        let halfFov = 0.5 * cameraComponent.fov * math.DEG_TO_RAD;
        if (cameraComponent.horizontalFov) {
            halfFov = Math.atan(Math.tan(halfFov) * viewportHeight / Math.max(viewportWidth, 1));
        }
        const projScale = viewportHeight / (2 * Math.tan(halfFov));
        // see cssPixels: render target pixels per CSS pixel, for a camera drawing to the canvas
        const cssHeight = this.cssPixels && !cameraComponent.renderTarget ? this.device.canvas?.clientHeight : 0;
        this.pixelScale = cssHeight > 0 ? Math.max(viewportHeight / cssHeight, 1) : 1;
        const culler = view.culler;
        culler.dagPixelThreshold = this.dagPixelThreshold * this.pixelScale;
        culler.rootMinPixels = this.rootMinPixels * this.pixelScale;
        culler.rootHysteresis = this.rootHysteresis;
        culler.bucketThresholdScales = this.bucketThresholdScales;
        culler.bucketRootMinScales = this.bucketRootMinScales;
        // shadow views rank their casters' root admission by this camera's view of them
        this.priorityOrigin.copy(cameraComponent.entity.getPosition());
        this.priorityProjScale = projScale;
        return projScale;
    }

    /**
     * Decides whether a view culls two-phase this frame and points it at the camera's HZB.
     *
     * Two-phase occlusion needs a scene depth the HZB can sample. A render target with a depth
     * texture supplies hardware depth. A CameraFrame keeps its scene depth as a renderbuffer, but
     * renders the linear scene depth as an extra attachment when asked (wantsSceneDepth ->
     * sceneTextureDepth), and the HZB reads that instead. Either way the pyramid is built
     * mid-frame, after phase 1 and the scene's opaque layers, so meshlet and scene geometry both
     * occlude.
     *
     * @param {MeshletView} view - The view.
     * @param {RenderTarget|null} rt - The target the scene renders into.
     * @returns {boolean} True when the view culls two-phase.
     * @private
     */
    _setupOcclusion(view, rt) {
        const cameraComponent = view.cameraComponent;
        const framePass = cameraComponent.framePasses?.[0];
        let depthTexture = rt?.depthBuffer ?? null;
        let hzbLinear = false;
        if (!depthTexture && framePass?.sceneDepthTexture) {
            depthTexture = framePass.sceneDepthTexture;
            hzbLinear = true;
        }
        const useOcclusion = this._occlusionEnabled && !!depthTexture;
        Debug.call(() => {
            // a CameraFrame marked for rebuild (the view was just created, or occlusion just
            // flipped) adds its scene depth on that rebuild - only a settled frame is a fallback
            const rebuildPending = cameraComponent.framePasses?.some(pass => pass.layersDirty);
            if (this._occlusionEnabled && !depthTexture && !rebuildPending) {
                Debug.warnOnce('MeshletDirector: occlusion requires the camera to render into a render target with a depth texture, or a CameraFrame able to render the scene depth (no multi-sampling, no depth prepass); falling back to single-phase culling.');
            }
        });
        if (useOcclusion) {
            // every world's view of this camera culls against the group's one pyramid
            view.hzb = this.group.hzbFor(cameraComponent);
            // the frame's scene textures are replaced when it rebuilds, which can happen between
            // this update and the pass executing - so the pass resolves its source as it runs
            view._hzbSource ??= () => {
                const passes = cameraComponent.framePasses;
                const target = passes?.[0]?.rt ?? cameraComponent.renderTarget;
                return target?.depthBuffer ?? passes?.[0]?.sceneDepthTexture ?? null;
            };
            view.hzb.resize(depthTexture, rt.width, rt.height, hzbLinear, view._hzbSource);
            view.hzb.setProjection(cameraComponent.camera.projectionMatrix);
        }
        // a CameraFrame caches its pass chain, and the HZB and phase-2 passes are only spliced
        // in (or dropped) when it rebuilds - so tell it when the occlusion state flips
        if (framePass && view.useOcclusion !== useOcclusion) {
            this._invalidateCameraFrame(cameraComponent);
        }
        view.culler.twoPhase = useOcclusion;
        view.culler.occlusionAdmission = this.occlusionAdmission;
        view.culler.viewDir = cameraComponent.entity.forward;
        view.useOcclusion = useOcclusion;
        return useOcclusion;
    }

    /**
     * Reports, once per episode, visible instances root admission deferred for capacity. Its
     * counters arrive a few frames late by readback.
     *
     * @param {MeshletView} view - The view.
     * @private
     */
    _reportDeferredRoots(view) {
        const roots = view.lastRoots;
        const deferred = roots.rejectedInstances;
        if (!deferred) {
            view._rootBudgetWarned = false;
            return;
        }
        if (view._rootBudgetWarned) return;
        view._rootBudgetWarned = true;
        Debug.warnOnce(`MeshletDirector: ${deferred} visible instances are deferred because their complete coarse geometry exceeds the current working-set capacity. The smallest on screen are dropped first; no partial instances are emitted. This bake needs cheaper coarse levels to display the entire visible set within this budget.`);
        const rejected = sumBuckets(roots.rejected);
        const admitted = sumBuckets(roots.admitted);
        this.onBudgetExceeded?.(this._budgetReport(this.world.poolBytes + rejected * 4,
            (admitted + rejected) / Math.max(view.indexTotal(), 1), roots));
    }

    /**
     * Builds an {@link onBudgetExceeded} report.
     *
     * @param {number} suggestedBudgetBytes - The estimated budget the visible set needs.
     * @param {number} indexDemandRatio - Index demand over the share.
     * @param {{ rejectedInstances: number, unready: number }|null} [roots] - The view's root
     * admission counters, when reporting deferred instances.
     * @returns {MeshletBudgetReport} The report.
     * @private
     */
    _budgetReport(suggestedBudgetBytes, indexDemandRatio, roots = null) {
        const world = this.world;
        const residency = this.residency;
        return {
            budgetBytes: world.poolBytes,
            suggestedBudgetBytes,
            indexDemandRatio,
            indexCeiling: world.indexCeiling,
            pagesMissing: residency?.lastMissingWanted ?? 0,
            poolPages: world.poolSlots,
            totalPages: world.totalPages,
            rootInstancesDeferred: roots?.rejectedInstances ?? 0,
            rootInstancesUnready: roots?.unready ?? 0,
            rootShortfallPages: residency?.rootShortfallPages ?? 0
        };
    }

    /**
     * Reports a budget the scene cannot meet (see MeshletBudgetManager#checkFeasible) to the
     * application, without raising it: silently increasing the ceiling would defeat a bounded
     * streaming budget. Cheaper coarse geometry or a smaller working set is the answer.
     *
     * @param {number} indexRatio - Worst per-view demand/share ratio.
     * @private
     */
    _checkBudgetFeasible(indexRatio) {
        const world = this.world;
        if (!(world.poolBytes > 0)) return;
        const pagesShort = this.residency?.lastMissingWanted ?? 0;
        if (!this.budget.checkFeasible(indexRatio, pagesShort, this.budgetOverrunFrames)) return;

        // a configuration problem the application must act on, so it ships in release builds
        if (this._budgetWarnedAt !== world.poolBytes) {
            this._budgetWarnedAt = world.poolBytes;
            console.warn(`MeshletDirector: the geometry budget (${(world.poolBytes / BYTES_PER_MB).toFixed(0)} MB) cannot render this scene even at the coarsest LOD - ${pagesShort} page(s) short, index demand ${indexRatio.toFixed(2)}x the share. The configured budget remains unchanged; cheaper coarse geometry or a smaller active working set is required.`);
        }
        this.onBudgetExceeded?.(this._budgetReport(world.poolBytes + pagesShort * world.pageSizeBytes, indexRatio));
    }

    /**
     * Tells the world how many views its geometry budget has to cover, so the per-view working
     * set can be reserved before anything is allocated. Shadow cascades are counted separately -
     * they share the claim plane and work items, so they cost far less than a camera view.
     *
     * @private
     */
    _applyBudgetViews() {
        this.world.budgetCameraViews = Math.max(Math.min(this.cameras.length || 1, this.maxViews), 1);
        this.world.budgetShadowViews = this.shadowsEnabled ? this.shadowBudgetCascades : 0;
    }

    /**
     * Hands over the index capacity a rebuild carried for one shadow view, if any.
     *
     * @param {number} lightId - The light's id.
     * @param {number} face - The shadow face index.
     * @returns {number[]|null} The carried capacities, or null.
     * @ignore
     */
    takeShadowCapCarry(lightId, face) {
        return this._carry?.takeShadowCaps(lightId, face) ?? null;
    }

    /**
     * Pre-cull half of the shadow phase: reconcile the per-(light, cascade) shadow views and
     * register their casters. Called by the forward renderer right after
     * `culler.updateLightVisibility`, which fills `cameraDirShadowLights`, and necessarily
     * before `cullComposition`, which reads `layer.shadowCasters`.
     *
     * @param {LayerComposition} comp - The layer composition.
     */
    updateShadowLights(comp) {
        if (!this.shadowsEnabled || !this.world.finalized) {
            if (this.shadowRenderer?.entries.size) this.shadowRenderer.reset();
            return;
        }
        this.shadowRenderer.syncLights(comp);
    }

    /**
     * Post-cull half: dispatch each cascade's cull chain. Called after `cullComposition`, which
     * is where the cascade shadow cameras are fitted - before that their frusta are stale. The
     * compute encoded here still runs ahead of every frame-graph pass, including the shadow
     * pass that consumes it.
     */
    updateShadows() {
        if (!this.shadowsEnabled || !this.world.finalized) return;
        this.shadowRenderer.cull();
    }

    /** @type {MeshInstance[]} - all views' mesh instances (shader-update sweep). */
    get allMeshInstances() {
        const mis = [];
        this.views.forEach(v => mis.push(...v.meshInstances));
        this.shadowRenderer?.entries.forEach((entry) => {
            entry.views.forEach(v => mis.push(...v.meshInstances));
        });
        return mis;
    }

    /**
     * Registers the meshlet geometry with an {@link OutlineRenderer}'s layer, so its camera
     * draws the selected instances into the outline texture.
     *
     * One indirect draw covers the whole world, so the outline renderer's usual contract - add an
     * entity's mesh instances to a layer - does not apply: the whole meshlet geometry is
     * registered once and selection is a per-instance flag (see
     * {@link MeshletWorld#setInstancesOutlined}).
     *
     * The layer must not be one the scene camera renders, or the main camera will draw the
     * entire meshlet world a second time. `OutlineRenderer` takes such a layer as its first
     * constructor argument.
     *
     * Only the phase-1 instances are registered: with occlusion on, phase 2's indirect arguments
     * are written part-way through the main camera's pass, and an outline camera runs before it.
     *
     * @param {Layer|null} layer - The outline renderer's layer, or null to unregister.
     * @param {Color|null} [color] - Selection outline colour. Defaults to white.
     * @param {Color|null} [hoverColor] - Hover outline colour. Defaults to white.
     */
    setOutlineLayer(layer, color = null, hoverColor = null) {
        const previous = this._outlineLayer;
        if (previous && previous !== layer) {
            this.views.forEach(v => previous.removeMeshInstances(v.meshInstances.slice(0, MESHLET_BUCKET_COUNT)));
        }
        this._outlineLayer = layer ?? null;
        this._outlineAttached = !!layer && this.world.outlinedCount > 0;
        if (!layer) return;

        this._outlineColor = color ?
            new Float32Array([color.r, color.g, color.b]) :
            new Float32Array([1, 1, 1]);
        this._hoverColor = hoverColor ?
            new Float32Array([hoverColor.r, hoverColor.g, hoverColor.b]) :
            new Float32Array([1, 1, 1]);
        this.views.forEach(view => this._registerOutlineView(view));
    }

    /**
     * Adds one view's phase-1 instances to the outline layer and gives them the outline colour.
     * Called both from {@link setOutlineLayer} and when a view is created afterwards - a view
     * only exists from the director's first update, so registering before then would otherwise
     * silently attach nothing (and leave the colour uniform holding whatever the last outlined
     * object set).
     *
     * @param {MeshletView} view - The view to register.
     * @private
     */
    _registerOutlineView(view) {
        const mis = view.meshInstances.slice(0, MESHLET_BUCKET_COUNT);
        mis.forEach((mi) => {
            mi.setParameter('pcOutlineColor', this._outlineColor);
            mi.setParameter('pcOutlineColorHover', this._hoverColor);
        });
        if (this._outlineAttached) {
            this._outlineLayer.addMeshInstances(mis, true);
        }
    }

    /**
     * Selects instances for outlining, by the same global instance index a pick returns.
     *
     * Use this rather than {@link MeshletWorld#setInstancesOutlined} directly: it also adds and
     * removes the geometry from the outline layer, so an empty selection - the usual state in an
     * editor - costs nothing at all. While registered, the outline camera runs the meshlet
     * vertex shader over every visible index - page pool reads and dequantisation included -
     * however few instances are selected.
     *
     * @param {number[]|Set<number>} instanceIndices - Global instance indices.
     * @param {boolean} [outlined] - True to add, false to remove. Defaults to true.
     */
    setOutlined(instanceIndices, outlined = true) {
        if (!this.world.finalized) return;
        this.world.setInstancesOutlined(instanceIndices, outlined);
        this._syncOutlineMembership();
    }

    /**
     * Marks instances as hovered - outlined in the hover colour rather than the selection one.
     * Kept separate from {@link setOutlined} so hovering never disturbs a selection, and a
     * selected instance can be hovered without changing colour underneath.
     *
     * @param {number[]|Set<number>} instanceIndices - Global instance indices.
     * @param {boolean} [hovered] - True to add, false to remove. Defaults to true.
     */
    setHovered(instanceIndices, hovered = true) {
        if (!this.world.finalized) return;
        this.world.setInstancesOutlined(instanceIndices, hovered, true);
        this._syncOutlineMembership();
    }

    /**
     * Clears outline state, and with it the outline pass's cost.
     *
     * @param {boolean} [hover] - Clear the hover state rather than the selection.
     */
    clearOutlines(hover = false) {
        if (!this.world.finalized) return;
        this.world.clearOutlines(hover);
        this._syncOutlineMembership();
    }

    /**
     * Adds the meshlet geometry to the outline layer only while something is selected.
     *
     * @private
     */
    _syncOutlineMembership() {
        const layer = this._outlineLayer;
        if (!layer) return;
        const wanted = this.world.outlinedCount > 0;
        if (wanted === this._outlineAttached) return;
        this._outlineAttached = wanted;
        this.views.forEach((view) => {
            const mis = view.meshInstances.slice(0, MESHLET_BUCKET_COUNT);
            if (wanted) layer.addMeshInstances(mis, true); else layer.removeMeshInstances(mis);
        });
    }

    /**
     * True when this is the layer whose lights shade the meshlet draws - the one place meshlet
     * geometry belongs in a per-layer walk.
     *
     * @param {Layer} layer - The layer to test.
     * @returns {boolean} True for the meshlet light layer.
     */
    isLightLayer(layer) {
        return !!layer && layer === this._lightLayer;
    }

    /**
     * Contributes the meshlet geometry to a pick render.
     *
     * Meshlet mesh instances live outside the layer system, so the picker - which walks each
     * layer's culled instances - never sees them. As with the gsplat renderer, the picker is
     * handed the instances directly, plus the id mapping it resolves a picked pixel through. The
     * id is per instance (see {@link MeshletWorld#pickRecords}); one indirect draw covers the
     * whole world, so it travels in objectData and the fragment shader emits it
     * (PICK_CUSTOM_ID in the chunk set) instead of the picker's per-mesh-instance uniform.
     *
     * @param {CameraComponent} cameraComponent - The camera being picked through.
     * @param {Map<number, object>} mapping - Picker id -> object map to add this world's ids to.
     * @returns {MeshInstance[]} Instances to draw into the pick buffer, empty when this camera
     * has no meshlet view.
     */
    preparePicking(cameraComponent, mapping) {
        const view = this.views.get(cameraComponent);
        if (!view || !this.world.finalized) return [];

        const records = this.world.pickRecords;
        for (let i = 0; i < records.length; i++) {
            mapping.set(records[i].pickId, records[i]);
        }

        // Phase 1 draws the previously-visible set and phase 2 the newly-disoccluded remainder,
        // so with occlusion on both are needed to cover what is actually on screen.
        const count = view.useOcclusion ? view.meshInstances.length : MESHLET_BUCKET_COUNT;
        return view.meshInstances.slice(0, count);
    }

    /**
     * Marks a camera's frame passes for rebuild, so a CameraFrame picks up (or drops) the
     * meshlet passes. Its chain is built once and cached; without this a camera that gained a
     * meshlet view after the CameraFrame was created would never draw meshlets at all.
     *
     * @param {CameraComponent} cameraComponent - The camera.
     * @private
     */
    _invalidateCameraFrame(cameraComponent) {
        const framePasses = cameraComponent?.camera?.framePasses;
        if (framePasses) {
            for (const pass of framePasses) {
                if ('layersDirty' in pass) pass.layersDirty = true;
            }
        }
    }

    /**
     * Sets whether two-phase HZB occlusion culling is enabled: phase 1 draws last frame's visible
     * set, an HZB is built from the depth so far, and phase 2 draws only what the HZB proves
     * newly visible. Needs a scene depth the HZB can sample - a depth texture on the camera's
     * render target, or a CameraFrame rendering the scene depth (which it does on request, see
     * {@link wantsSceneDepth}). Falls back to single-phase culling when neither is available.
     * Defaults to true.
     *
     * @type {boolean}
     */
    set occlusionEnabled(value) {
        value = !!value;
        if (this._occlusionEnabled === value) return;
        this._occlusionEnabled = value;
        // a CameraFrame decides whether to render the scene depth from wantsSceneDepth, and caches
        // its chain - so every driven camera's frame has to rebuild
        this.views.forEach((view, cameraComponent) => this._invalidateCameraFrame(cameraComponent));
    }

    get occlusionEnabled() {
        return this._occlusionEnabled;
    }

    /**
     * Whether root admission skips instances wholly hidden behind last frame's depth, so they
     * spend none of the view's index and record budget (see
     * {@link meshletRootClassifyWGSL}). Occlusion culling alone only skips drawing them: their
     * roots are admitted, and their cut counted as demand, before any cluster is tested. Only
     * applies while occlusion culling is running. An instance that stops being hidden is
     * admitted a frame later. Defaults to true.
     *
     * @type {boolean}
     */
    occlusionAdmission = MESHLET_DIRECTOR_SETTINGS.occlusionAdmission;

    /**
     * Whether a camera's CameraFrame should render the scene depth for the meshlet HZB: occlusion
     * is enabled and the director has a view for the camera. FramePassCameraFrame asks when it
     * decides its scene textures; the director marks the frame for rebuild whenever the answer
     * changes (the view appearing or leaving, the flag flipping).
     *
     * @param {CameraComponent} cameraComponent - The camera.
     * @returns {boolean} True when the frame should render the scene depth.
     */
    wantsSceneDepth(cameraComponent) {
        return this._occlusionEnabled && this.views.has(cameraComponent);
    }
}

export { MeshletDirector, MESHLET_DIRECTOR_SETTINGS };
