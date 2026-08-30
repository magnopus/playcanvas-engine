import { Debug } from '../../core/debug.js';
import { Frustum } from '../../core/shape/frustum.js';
import { Mat4 } from '../../core/math/mat4.js';
import { math } from '../../core/math/math.js';
import { LAYERID_WORLD } from '../constants.js';
import { MESHLET_BUCKET_COUNT } from './constants.js';
import { MeshletBudgetManager } from './meshlet-budget-manager.js';
import { MeshletCullShaders } from './meshlet-cull-shaders.js';
import { MeshletShadowRenderer } from './meshlet-shadow-renderer.js';
import { MeshletHzb } from './meshlet-hzb.js';
import { MeshletView } from './meshlet-view.js';
import { MeshletWorld } from './meshlet-world.js';
import { MeshletResidency } from './streaming/meshlet-residency.js';

/**
 * @import { CameraComponent } from '../../framework/components/camera/component.js'
 * @import { Color } from '../../core/math/color.js'
 * @import { ForwardRenderer } from '../renderer/forward-renderer.js'
 * @import { FrameGraph } from '../frame-graph.js'
 * @import { FramePass } from '../../platform/graphics/frame-pass.js'
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { Layer } from '../layer.js'
 * @import { LayerComposition } from '../composition/layer-composition.js'
 * @import { MeshInstance } from '../mesh-instance.js'
 * @import { MeshletTranscodeFn } from './textures/meshlet-ktx2.js'
 * @import { RenderTarget } from '../../platform/graphics/render-target.js'
 * @import { StorageBuffer } from '../../platform/graphics/storage-buffer.js'
 */

const _viewProj = new Mat4();
const _frustum = new Frustum();

// Margin over the pair-count ratio when pre-sizing carried index capacities for a scene that
// grew: enough to lead the demand readback by a step change, not a scene-size ratio.
const CARRY_GROWTH_MARGIN = 1.35;

// Every view keeps at least this fraction of an equal share of the index budget, so a view
// that is momentarily empty can grow back without waiting for a redistribution.
const SHARE_FLOOR_FRACTION = 0.25;

const BYTES_PER_MB = 1024 * 1024;

/**
 * Orchestrates the GPU-driven meshlet pipeline: per-frame culling compute (encoded ahead of the
 * frame graph), the meshlet draw passes inserted after the camera's main scene pass, and - when
 * two-phase occlusion is enabled - the HZB build and the phase-2 re-cull between them.
 *
 * Wire-up: create, add resources to {@link world}, call {@link finalize}, assign a camera to
 * {@link cameraComponent} and register on the renderer via {@link bindRenderer}. For occlusion
 * the camera must render into a render target with a depth texture.
 *
 * @ignore
 */
class MeshletDirector {
    /** @type {GraphicsDevice} */
    device;

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

    /** @type {ForwardRenderer|null} */
    renderer = null;

    /** @type {boolean} - two-phase HZB occlusion culling. */
    occlusionEnabled = false;

    /** @type {Float32Array} */
    _frustumPlanes = new Float32Array(24);

    /** @type {number} */
    _dagPixelThreshold = 1;

    /**
     * Rebuild retention: previous views' visibility-bit buffers keyed by camera, GPU-copied
     * into the recreated views' fresh buffers (see MeshletView), then retired.
     *
     * @type {{ words: number, byCamera: Map<CameraComponent, StorageBuffer> }|null}
     */
    _visCarry = null;

    /**
     * Rebuild retention: previous views' per-bucket demand-grown index capacities keyed by
     * camera. A fresh view would restart at the initial allocation and clamp its draws for the
     * frames the demand readback is in flight - and with occlusion on, those missing draws
     * punch holes in the HZB that phase 2 reads as disocclusion, bursting page demand.
     *
     * @type {Map<CameraComponent, number[]>|null}
     */
    _capCarry = null;

    /**
     * Buffers a rebuild copied from, destroyed one frame later (after their commands submit).
     *
     * @type {StorageBuffer[]}
     */
    _retired = [];

    /**
     * The cull compute shaders, compiled once and shared by every view's culler. Held on the
     * director rather than per view because the shadow phase adds one culler per cascade.
     *
     * @type {MeshletCullShaders|null}
     */
    cullShaders = null;

    /**
     * Cast shadows from the meshlet geometry: directional cascades and, unless
     * `shadowRenderer.localLights` is switched off, spot and omni faces. Off by default - it adds
     * one single-phase cull chain per shadow face, so it is opt-in per scene.
     *
     * @type {boolean}
     */
    shadowsEnabled = false;

    /**
     * Multiplier on {@link dagPixelThreshold} for shadow views. Shadow-map texels are coarser
     * than screen pixels and a blurred depth comparison hides geometric detail, so casters can
     * run a far coarser cut than the camera. Also the release valve when shadow views enlarge
     * the streaming working set enough to coarsen the camera's own cut.
     *
     * @type {number}
     */
    shadowThresholdScale = 4;

    /** @private */
    _transcode = null;

    /**
     * The KTX2 transcoder used for streamed textures - a `basisTranscode`-compatible function.
     * Scene code cannot import the framework's Basis handler, so the transcoder is injected:
     * {@link MeshletComponentSystem} sets it to the engine's `basisTranscode`; a director
     * created by hand must set it before loading a textured asset, or its textures reject.
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
     * Split the cull chains into separate submissions around GPU-written indirect dispatch args
     * (see {@link MeshletCuller#forceSubmitBoundaries}). Propagated to every view - camera and
     * shadow - so enabling it does not leave the shadow cascades broken on the platforms it
     * exists for.
     *
     * @type {boolean}
     */
    forceSubmitBoundaries = false;

    /**
     * Fraction of the world's initial index budget a shadow cascade starts with. The budget is
     * tuned for the camera; a cascade's cut is coarser and its demand far lower, so matching it
     * reserves hundreds of MB per cascade that growth would never ask for. Growth still reacts
     * within a frame or two, and a briefly clamped shadow is far less visible than clamped
     * camera geometry.
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
     * Called when the geometry budget is found to be unachievable - the cut is already at the
     * DAG roots and the scene still does not fit. Receives a summary (budget, index demand
     * ratio, pages missing, and a `suggestedBudgetBytes` that would cover the shortfall). The
     * pipeline keeps rendering, with gaps, and raises the index ceiling; the application decides
     * whether to raise the budget or shed content.
     *
     * @type {((info: object) => void)|null}
     */
    onBudgetExceeded = null;

    /** @type {number} - consecutive infeasible frames before {@link onBudgetExceeded} fires. */
    budgetOverrunFrames = 120;

    /** @type {number} - multiplier applied to the index ceiling on each overrun step. */
    budgetOverrunStep = 1.5;

    _infeasibleFrames = 0;

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

    /** @type {Promise<void>|null} - resolves when a streamed world's root pages are resident. */
    rootsLoaded = null;

    /**
     * Ceiling on how much a rebuild may pre-grow a view's carried index capacity. Pre-sizing
     * exists only to avoid a few clamped frames while the demand readback catches up, so it
     * needs to cover a step change, not a scene-size ratio.
     *
     * @type {number}
     */
    maxCarryScale = 2;

    /**
     * Rebuild retention for shadow views: demand-grown index capacities keyed `lightId:face`.
     * Same reasoning as {@link _capCarry} - a rebuild that resets shadow capacity clamps caster
     * draws while the demand readback is in flight, which shows as shadows popping in.
     *
     * @type {Map<string, number[]>|null}
     * @private
     */
    _shadowCapCarry = null;

    /**
     * @param {GraphicsDevice} device - The graphics device.
     */
    constructor(device) {
        this.device = device;
        this.world = new MeshletWorld(device);
        this.cullShaders = new MeshletCullShaders(device);
        this.shadowRenderer = new MeshletShadowRenderer(device, this);
    }

    destroy() {
        if (this.renderer && this.renderer.meshletDirector === this) {
            this.renderer.meshletDirector = null;
        }
        this._destroyViews();
        this._dropVisCarry();
        this.residency?.destroy();
        this.world.destroy();
        this.shadowRenderer?.destroy();
        this.shadowRenderer = null;
        this.cullShaders?.destroy();
        this.cullShaders = null;
    }

    _destroyViews() {
        this.views.forEach(v => v.destroy());
        this.views.clear();
    }

    _dropVisCarry() {
        this._visCarry?.byCamera.forEach(b => b.destroy());
        this._visCarry = null;
    }

    /**
     * Registers this director with the forward renderer, which ticks it each frame and inserts
     * its passes into the frame graph.
     *
     * @param {ForwardRenderer} renderer - The renderer (app.renderer).
     */
    bindRenderer(renderer) {
        this.renderer = renderer;
        renderer.meshletDirector = this;
    }

    /**
     * Finalizes the world and builds the culler. Call after all resources are added. For a
     * streamed world this also starts the eager root-page load - await {@link rootsLoaded}
     * (or just start rendering: nothing draws until the coarse fallback set is resident).
     *
     * @returns {MeshletWorld} The finalized world.
     */
    finalize() {
        this._applyBudgetViews();
        this.world.finalize();
        if (this.world.streamed) {
            this.residency = new MeshletResidency(this.device, this.world);
            this.rootsLoaded = this.residency.loadRoots();
            // the request readback carries the texture-mip feedback marks after the page marks
            if (this.world.textures) {
                this.residency.onTexelRateMarks = marks => this.world.textures?.processMarks(marks);
            }
        }
        return this.world;
    }

    /**
     * Tears down the current world and rebuilds it - the component system's add/remove path.
     * Streaming state is RETAINED where possible: when the leading resources of the new build
     * match the previous one (same resource, base URL and order) and the pool geometry is
     * unchanged, the GPU page pool, its resident pages, the slot/LRU state and pinned roots
     * carry over - only genuinely new resources re-stream. The streamed texture system carries
     * whenever the texture-bearing resource set is unchanged. An empty build leaves the
     * director idle.
     *
     * @param {function(MeshletWorld): void} build - Adds resources to the fresh world.
     * @returns {MeshletWorld} The new world (finalized unless empty).
     */
    rebuild(build) {
        const prevWorld = this.world;
        const prevResidency = this.residency;
        const prevViews = this.views;
        this.views = new Map();
        this._dropVisCarry();
        this._capCarry = new Map();
        const prevPairs = prevWorld.totalPairs || 1;
        prevViews.forEach((v, cam) => {
            this._capCarry.set(cam, v.indexCapacity.slice());
        });
        // shadow views hold buffers sized to the old world (claim bits are one bit per
        // instance-meshlet pair), so they are dropped and rebuilt next frame - carrying only
        // their grown index capacities, as the camera views do
        this._shadowCapCarry = new Map();
        this.shadowRenderer?.entries.forEach((entry, light) => {
            entry.views.forEach((v) => {
                this._shadowCapCarry.set(`${light.id}:${v.face}`, v.indexCapacity.slice());
            });
        });
        this.shadowRenderer?.reset();
        const { poolBytes, texturePoolBytes, initialIndices, initialRecords, indexBudgetFraction } = prevWorld;
        this.residency = null;
        this.rootsLoaded = null;

        this.world = new MeshletWorld(this.device);
        this.world.poolBytes = poolBytes;
        this.world.texturePoolBytes = texturePoolBytes;
        this.world.maxInstallBytesPerFrame = prevWorld.maxInstallBytesPerFrame;
        this.world.initialIndices = initialIndices;
        this.world.indexBudgetFraction = indexBudgetFraction;
        this.world.initialRecords = initialRecords;
        this.world.transcode = this._transcode;
        this.budget._lastDropped = 0;
        this.budget._lastEvicted = 0;
        this.budget._lastFetched = 0;
        build(this.world);
        if (this.world.pendingCount === 0) {
            prevViews.forEach(v => v.destroy());
            prevResidency?.destroy();
            prevWorld.destroy();
            return this.world;
        }

        // retention: compare the new build against the previous streamed world
        const carriedSlots = (prevWorld.finalized && prevWorld.streamed && prevResidency) ?
            this._adoptStreamingState(prevWorld, prevResidency, prevViews) : null;

        prevViews.forEach(v => v.destroy());
        prevResidency?.destroy();
        prevWorld.destroy();

        this._applyBudgetViews();
        this.world.finalize();
        // pre-size the recreated views' index buffers for grown scenes: index demand scales
        // roughly with the pair count, and starting under-sized clamps draws for the frames
        // the demand readback is in flight (with occlusion on, the missing draws punch HZB
        // holes phase 2 reads as disocclusion - a page-demand burst). The margin applies
        // ONLY when the scene grew - a flat multiplier would compound across same-size
        // rebuilds (budget sliders) into unbounded buffers. Shrunk scenes scale down
        // proportionally (the view floors at the world's initial target).
        // Bounded, because index demand does NOT scale with the scene. It is screen-bound: the
        // DAG cut coarsens with distance and the frustum limits what is in view, so multiplying
        // the instance count by 100 moved measured demand by ~3.5x. Scaling the carry by the
        // raw pair ratio turned one slider drag into a 476 MB index buffer for a cut that
        // wanted 17 MB - and successive drags compound it.
        const pairRatio = this.world.totalPairs / prevPairs;
        const pairScale = pairRatio > 1 ? Math.min(pairRatio * CARRY_GROWTH_MARGIN, this.maxCarryScale) : pairRatio;
        const scaleCarry = (/** @type {number[]} */ c) => {
            for (let b = 0; b < c.length; b++) c[b] = Math.ceil(c[b] * pairScale);
        };
        this._capCarry?.forEach(scaleCarry);
        this._shadowCapCarry?.forEach(scaleCarry);
        if (this.world.streamed) {
            this.residency = new MeshletResidency(this.device, this.world, carriedSlots);
            this.rootsLoaded = this.residency.loadRoots();
            if (this.world.textures) {
                this.residency.onTexelRateMarks = marks => this.world.textures?.processMarks(marks);
            }
        }
        return this.world;
    }

    /**
     * Decides what the new (pending, not yet finalized) world inherits from the previous
     * streamed one, and hands it over through the world's adopt* fields.
     *
     * Leading resources that are identical in both builds (same resource, base URL and order)
     * keep their page space; the stricter pair prefix - instance lists unchanged too - also
     * keeps its persistent visibility bits. The page pool buffer itself is reusable only when
     * its slot count and page size are unchanged. The texture system carries whenever the
     * texture-bearing resource sequence and the texture pool budget are unchanged.
     *
     * @param {MeshletWorld} prevWorld - The finalized, streamed world being replaced.
     * @param {MeshletResidency} prevResidency - Its residency manager.
     * @param {Map<CameraComponent, MeshletView>} prevViews - Its views (visibility-bit sources).
     * @returns {object|null} The residency slot state to carry into the new
     * {@link MeshletResidency}, or null when the pool is not reusable.
     * @private
     */
    _adoptStreamingState(prevWorld, prevResidency, prevViews) {
        const world = this.world;
        const pending = world._pending;
        const prevEntries = prevWorld.buildEntries;
        const { poolBytes, texturePoolBytes } = world;

        const sameInstances = (a, b) => {
            if (a === b) return true;
            if (!a || !b || a.length !== b.length) return false;
            for (let i = 0; i < a.length; i++) {
                if (a[i] !== b[i]) return false;
            }
            return true;
        };
        let prefix = 0;
        let adoptedPages = 0;
        let carriedPairs = 0;
        let pairPrefixIntact = true;
        while (prefix < pending.length && prefix < prevEntries.length) {
            const n = pending[prefix];
            const o = prevEntries[prefix];
            if (n.resource !== o.resource || (n.baseUrl ?? null) !== o.baseUrl ||
                !o.streamed || !!n.shardBuffers) break;
            if (pairPrefixIntact && sameInstances(n.instances ?? n.resource.instances, o.instances)) {
                carriedPairs += prevWorld.placements[prefix].pairCount;
            } else {
                pairPrefixIntact = false;
            }
            adoptedPages += o.pageCount;
            prefix++;
        }

        // the pool buffer is reusable only when its slot count is unchanged
        let newTotalPages = 0;
        for (const { resource } of pending) newTotalPages += resource.manifest.pageCount;
        // predict the new pool's slot count the way finalize will: the budget covers more
        // than the pool, so the previous world's RESOLVED page-pool size is the comparable
        // figure when the budget itself is unchanged
        const newPagePoolBytes = poolBytes === prevWorld.poolBytes ? prevWorld.pagePoolBytes : poolBytes;
        let newPoolSlots = newPagePoolBytes > 0 ?
            Math.max(Math.floor(newPagePoolBytes / prevWorld.pageSizeBytes), 1) : newTotalPages;
        newPoolSlots = Math.min(newPoolSlots, newTotalPages);

        let carriedSlots = null;
        if (prefix > 0 && newPoolSlots === prevWorld.poolSlots &&
            pending[0].resource.manifest.pageSizeBytes === prevWorld.pageSizeBytes) {
            world.adoptPagePool = prevWorld.pagePool;
            prevWorld.pagePool = null;
            world.adoptResidency = prevWorld.residency;
            world.adoptedPages = adoptedPages;
            if (carriedPairs > 0) {
                // each view's visibility bits are GPU-copied into its recreated counterpart
                // (matched by camera) when the view is next built; the source buffers are
                // retired after consumption. One bit per pair, 32 to a word.
                this._visCarry = { words: Math.floor(carriedPairs / 32), byCamera: new Map() };
                prevViews.forEach((v, cam) => {
                    if (v.visBitsBuffer) {
                        this._visCarry.byCamera.set(cam, v.visBitsBuffer);
                        v.visBitsBuffer = null;
                    }
                });
            }
            carriedSlots = {
                slotPage: prevResidency.slotPage,
                slotPinned: prevResidency.slotPinned,
                slotLastUsed: prevResidency.slotLastUsed,
                frame: prevResidency.frame
            };
        }

        // the texture budget compares against what the live system was built with
        // (prevWorld.texturePoolBytes may already hold the caller's new value)
        if (prevWorld.textures) {
            const prevTex = prevEntries.filter(e => e.resource.textureManifest);
            const newTex = pending.filter(e => e.resource.textureManifest);
            const same = prevTex.length === newTex.length && prevTex.every((e, i) => e.resource === newTex[i].resource && e.baseUrl === (newTex[i].baseUrl ?? null));
            if (same && prevWorld.textures.poolBytes === texturePoolBytes) {
                world.adoptTextures = prevWorld.textures;
                prevWorld.textures = null;
            }
        }
        return carriedSlots;
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

    /** @type {number} - screen-space error threshold in pixels for the DAG LOD cut. */
    set dagPixelThreshold(value) {
        this._dagPixelThreshold = value;
        this.views.forEach((v) => {
            v.culler.dagPixelThreshold = value;
        });
    }

    get dagPixelThreshold() {
        return this._dagPixelThreshold;
    }

    /**
     * Per-frame tick, called by the forward renderer before the frame graph is built. Runs the
     * streaming residency update and encodes the phase-1 culling chain.
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
        // index starvation feeds the same controller as page pressure: both say the cut is
        // too fine for the memory it has, and the answer to both is to coarsen it
        const indexRatio = this._distributeIndexBudget();
        this.budget.update(this.residency, indexRatio);
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
                view.destroy();
                this.views.delete(cam);
                this._invalidateCameraFrame(cam);
            }
        });

        for (const cameraComponent of cameras) {
            let view = this.views.get(cameraComponent);
            if (!view) {
                const carry = this._visCarry?.byCamera.get(cameraComponent);
                view = new MeshletView(this.device, this.world, this.renderer, cameraComponent, {
                    visCarry: carry ? { buffer: carry, words: this._visCarry.words } : null,
                    capCarry: this._capCarry?.get(cameraComponent) ?? null,
                    cullShaders: this.cullShaders
                });
                view.culler.dagPixelThreshold = this._dagPixelThreshold;
                this.views.set(cameraComponent, view);
                // a CameraFrame builds its pass chain once and caches it, so it has to be told
                // to rebuild now that there are meshlet passes to splice in
                this._invalidateCameraFrame(cameraComponent);
                if (this._outlineLayer) this._registerOutlineView(view);
                if (carry) {
                    // copy is encoded; the source buffer must outlive this frame's submission
                    this._visCarry.byCamera.delete(cameraComponent);
                    this._retired.push(carry);
                }
            }
            this._updateView(view);
        }
    }

    /**
     * Per-view frame tick: materials, index demand, frustum/HZB setup and the phase-1 cull
     * chain encode.
     *
     * @param {MeshletView} view - The view.
     * @private
     */
    _updateView(view) {
        const cameraComponent = view.cameraComponent;
        view._passesAdded = false;
        view.syncMaterials();
        view.culler.pressureScale = this.budget.pressureScale;
        view.culler.forceSubmitBoundaries = this.forceSubmitBoundaries;
        // grow first, then read demand: buffers must not change once the frame is encoding
        view.applyPendingGrowth();
        view.monitorIndexDemand();

        const camera = cameraComponent.camera;
        _viewProj.mul2(camera.projectionMatrix, camera.viewMatrix);
        _frustum.setFromMat4(_viewProj);
        const planes = this._frustumPlanes;
        for (let p = 0; p < 6; p++) {
            const plane = _frustum.planes[p];
            planes[p * 4 + 0] = plane.normal.x;
            planes[p * 4 + 1] = plane.normal.y;
            planes[p * 4 + 2] = plane.normal.z;
            planes[p * 4 + 3] = plane.distance;
        }

        // With a CameraFrame the camera's own renderTarget is the OUTPUT (often the backbuffer);
        // the scene is rendered into the frame pass's internal target, which is also where the
        // depth the HZB needs lives - and, with a render scale, the resolution the cut should
        // be measured against.
        const rt = cameraComponent.framePasses?.[0]?.rt ?? cameraComponent.renderTarget;
        const viewportHeight = rt?.height ?? this.device.height;
        const projScale = viewportHeight / (2 * Math.tan(0.5 * cameraComponent.fov * math.DEG_TO_RAD));

        // two-phase occlusion needs a sampleable depth texture on the camera's render target
        const depthTexture = rt?.depthBuffer ?? null;
        const useOcclusion = this.occlusionEnabled && !!depthTexture;
        Debug.call(() => {
            if (this.occlusionEnabled && !depthTexture) {
                Debug.warnOnce('MeshletDirector: occlusion requires the camera to render into a render target with a depth texture; falling back to single-phase culling.');
            }
        });
        if (useOcclusion) {
            view.hzb ??= new MeshletHzb(this.device);
            view.hzb.resize(depthTexture, rt.width, rt.height);
        }
        view.culler.twoPhase = useOcclusion;
        view.useOcclusion = useOcclusion;

        view.culler.beginFrame(planes, cameraComponent.entity.getPosition(), projScale, _viewProj.data,
            useOcclusion ? view.hzb : null);
    }

    /**
     * Shares the world's index budget across every live view in proportion to what each is
     * asking for, and returns the worst per-view pressure.
     *
     * An equal split is the wrong default: views want wildly different amounts. On the Bistro a
     * shadow cascade wanted 1.2x an equal share while the camera used 0.16x of its own - and
     * because index pressure feeds one global LOD controller, that one starved cascade drove
     * the whole scene to maximum coarseness.
     *
     * @returns {number} Highest demand/share ratio across views.
     * @private
     */
    _distributeIndexBudget() {
        const ceiling = this.world.indexCeiling;
        if (!(ceiling > 0) || !Number.isFinite(ceiling)) return 0;

        const all = [];
        this.views.forEach(v => all.push(v));
        this.shadowRenderer?.entries.forEach((entry) => {
            entry.views.forEach(v => all.push(v));
        });
        if (!all.length) return 0;

        let total = 0;
        for (const v of all) total += v.indexDemand();

        // Every view keeps a floor, so one that is momentarily empty (a cascade the camera has
        // turned away from) can still grow back without waiting for a redistribution.
        const floor = Math.floor((ceiling / all.length) * SHARE_FLOOR_FRACTION);
        let worst = 0;
        for (const v of all) {
            v.indexShare = total > 0 ?
                Math.max(Math.floor(ceiling * (v.indexDemand() / total)), floor) :
                Math.floor(ceiling / all.length);
            worst = Math.max(worst, v.indexPressure());
        }
        return worst;
    }

    /**
     * Detects a budget the scene cannot meet, and says so.
     *
     * LOD pressure has a floor: once the cut is at the DAG roots there is nothing coarser to
     * fall back to. If the scene still does not fit at that point, holding the budget produces
     * clamped draws and missing pages - gaps and flicker - forever. So when pressure is pinned
     * at maximum and the scene is still short, the budget is treated as unachievable: the index
     * ceiling is raised in steps (never past the device limit) and {@link onBudgetExceeded}
     * fires so the application can react - raise the budget, drop content, or tell the user.
     *
     * @param {number} indexRatio - Worst per-view demand/share ratio.
     * @private
     */
    _checkBudgetFeasible(indexRatio) {
        const world = this.world;
        if (!(world.poolBytes > 0)) return;

        const res = this.residency;
        const pagesShort = res ? (res.lastMissingWanted ?? 0) : 0;
        const pinned = this.budget.pressureScale >= this.budget.maxScale;
        const short = indexRatio > 1 || pagesShort > 0;
        this._infeasibleFrames = (pinned && short) ? this._infeasibleFrames + 1 : 0;
        if (this._infeasibleFrames < this.budgetOverrunFrames) return;
        this._infeasibleFrames = 0;

        const before = world.indexCeiling;
        if (indexRatio > 1 && before < world.deviceIndexCeiling) {
            world.indexOverrun = Math.min(
                Math.ceil((world.indexBudgetTotal + world.indexOverrun) * this.budgetOverrunStep) - world.indexBudgetTotal,
                world.deviceIndexCeiling - world.indexBudgetTotal
            );
        }

        // what the budget would have to be for the pages the cut is actually asking for; the
        // index side is already covered by the overrun above
        const suggested = world.poolBytes + pagesShort * world.pageSizeBytes;
        const info = {
            budgetBytes: world.poolBytes,
            suggestedBudgetBytes: suggested,
            indexDemandRatio: indexRatio,
            indexCeiling: world.indexCeiling,
            indexOverrun: world.indexOverrun,
            pagesMissing: pagesShort,
            poolPages: res ? res.slotPage.length : 0,
            totalPages: world.totalPages
        };
        Debug.warnOnce(`MeshletDirector: the geometry budget (${(world.poolBytes / BYTES_PER_MB).toFixed(0)} MB) cannot render this scene even at the coarsest LOD - ${pagesShort} page(s) short, index demand ${indexRatio.toFixed(2)}x the share. Raising the index ceiling; expect gaps until the budget is increased (about ${(suggested / BYTES_PER_MB).toFixed(0)} MB would cover it).`);
        this.onBudgetExceeded?.(info);
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
     * Consumes the retained index capacity for one shadow view after a rebuild, if any.
     *
     * @param {number} lightId - The light's id.
     * @param {number} face - The shadow face index.
     * @returns {number[]|null} The carried capacities, or null.
     * @ignore
     */
    takeShadowCapCarry(lightId, face) {
        return this._shadowCapCarry?.get(`${lightId}:${face}`) ?? null;
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
     * The outline renderer's usual contract - add an entity's mesh instances to a layer - cannot
     * work here: one indirect draw covers the whole world, so there is no per-object mesh
     * instance to add and no per-object uniform to colour. Instead the whole meshlet geometry is
     * registered once and selection is a per-instance flag
     * ({@link MeshletWorld#setInstancesOutlined}); the outline variant of the vertex shader
     * collapses unselected instances to a degenerate triangle.
     *
     * The layer MUST NOT be one the scene camera renders, or the main camera will draw the
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
            this.views.forEach(v => previous.removeMeshInstances(v.meshInstances));
        }
        this._outlineLayer = layer ?? null;
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
        if (this.world.outlinedCount > 0) {
            this._outlineLayer.addMeshInstances(mis, true);
        }
    }

    /**
     * Selects instances for outlining, by the same global instance index a pick returns.
     *
     * Use this rather than {@link MeshletWorld#setInstancesOutlined} directly: it also adds and
     * removes the geometry from the outline layer, so an empty selection - the usual state in an
     * editor - costs nothing at all. Leaving it registered is NOT free: the outline camera runs
     * the meshlet vertex shader over every visible index, and that shader pulls from the page
     * pool and dequantises, so on a dense scene it is several milliseconds of work that draws no
     * pixels.
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
     * layer's culled instances - never sees them. This is the same seam the gsplat renderer has,
     * and the same answer: hand the picker the instances directly, plus the id mapping it will
     * resolve a picked pixel through.
     *
     * The id is per INSTANCE - one primitive at one transform, which is what a glTF submesh
     * becomes here, so a table's top and legs pick apart as they would unbaked. It cannot be the
     * picker's per-mesh-instance uniform: one indirect draw covers the whole world. Instead it
     * travels in objectData and the fragment shader emits it (PICK_CUSTOM_ID in the chunk set).
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
     * The CameraFrame variant of {@link buildCameraPasses}. A camera with frame passes never
     * reaches the render-action path - the renderer schedules its `framePasses` and returns - so
     * the meshlet passes have to be handed to {@link FramePassCameraFrame} instead, which
     * splices them around its own scene pass.
     *
     * Returns the passes rather than adding them to a frame graph, because the CameraFrame owns
     * the ordering: `before` runs ahead of the scene's opaque layers (phase 1 also takes over
     * the clear), `middle` runs between the opaque and transparent halves.
     *
     * @param {CameraComponent} cameraComponent - The camera being set up.
     * @param {RenderTarget} renderTarget - The CameraFrame's scene render target.
     * @param {boolean} clears - True when this camera's first pass should clear.
     * @returns {{ before: FramePass[], middle: FramePass[] }|null} Passes to splice, or null
     * when the director is not driving this camera.
     */
    buildCameraFramePasses(cameraComponent, renderTarget, clears) {
        const view = this.views.get(cameraComponent);
        if (!view) {
            return null;
        }

        const camera = cameraComponent.camera;
        const mis = view.meshInstances;
        view.drawPass1.setup(cameraComponent, renderTarget, this._lightLayer, {
            clearColor: clears && camera.clearColorBuffer,
            clearDepth: clears && camera.clearDepthBuffer,
            clearStencil: clears && camera.clearStencilBuffer
        });
        view.drawPass1.meshInstances = mis.slice(0, MESHLET_BUCKET_COUNT);

        const middle = [];
        if (view.useOcclusion) {
            view.drawPass2.setup(cameraComponent, renderTarget, this._lightLayer);
            view.drawPass2.meshInstances = mis.slice(MESHLET_BUCKET_COUNT, MESHLET_BUCKET_COUNT * 2);
            middle.push(view.hzb.mip0Pass, view.hzbMipsPass, view.cullPhase2Pass, view.drawPass2);
        }

        // the render-action path is not going to run for this camera, so nothing else will mark
        // the view's passes as emitted
        view._passesAdded = true;
        return { before: [view.drawPass1], middle };
    }

    /**
     * Called by the forward renderer while building the frame graph, in place of the camera
     * block's single main render pass. Emits the meshlet-aware pass order:
     *
     *   draw P1 (owns the camera clear) -> opaque scene passes (load) ->
     *   [HZB -> cull P2 -> draw P2] -> transparent scene passes (load)
     *
     * Meshlet depth is laid down first so scene opaque early-z-culls against it (and scene
     * geometry composites correctly with meshlets), the HZB sees meshlet + scene occluders,
     * and phase-2 disocclusion draws land before transparents and UI.
     *
     * @param {FrameGraph} frameGraph - The frame graph being built.
     * @param {LayerComposition} layerComposition - The layer composition.
     * @param {RenderTarget|null} renderTarget - The block's render target.
     * @param {number} startIndex - First render action index of the block.
     * @param {number} endIndex - Last render action index of the block.
     * @param {CameraComponent} cameraComponent - The camera of the block.
     * @returns {boolean} True when the director emitted the block's passes; false leaves the
     * renderer's default path (director inactive, other camera, or non-full-viewport clears).
     */
    buildCameraPasses(frameGraph, layerComposition, renderTarget, startIndex, endIndex, cameraComponent) {
        const view = this.views.get(cameraComponent);
        if (!view || view._passesAdded) {
            return false;
        }
        if (!cameraComponent.camera.fullSizeClearRect) {
            Debug.warnOnce('MeshletDirector: camera does not clear the full viewport; meshlet draws fall back to post-scene ordering (incorrect compositing with scene geometry).');
            return false;
        }
        view._passesAdded = true;

        const renderActions = layerComposition._renderActions;
        const renderer = this.renderer;
        const mis = view.meshInstances;

        // meshlet P1 draws first and owns the camera clear; the scene passes load on top
        const firstAction = renderActions[startIndex];
        view.drawPass1.setup(cameraComponent, renderTarget, this._lightLayer, {
            clearColor: firstAction.clearColor,
            clearDepth: firstAction.clearDepth,
            clearStencil: firstAction.clearStencil
        });
        view.drawPass1.meshInstances = mis.slice(0, MESHLET_BUCKET_COUNT);
        frameGraph.addRenderPass(view.drawPass1);

        // split the block at its first transparent render action
        let splitIndex = endIndex + 1;
        for (let j = startIndex; j <= endIndex; j++) {
            if (renderActions[j].transparent) {
                splitIndex = j;
                break;
            }
        }

        // opaque scene passes (depth-tested against meshlet depth)
        if (splitIndex > startIndex) {
            renderer.addMainRenderPass(frameGraph, layerComposition, renderTarget, startIndex, splitIndex - 1, true);
        }

        // two-phase occlusion: HZB from the combined meshlet + scene opaque depth, then the
        // newly-disoccluded remainder - all before transparents
        if (view.useOcclusion) {
            frameGraph.addRenderPass(view.hzb.mip0Pass);
            frameGraph.addRenderPass(view.hzbMipsPass);
            frameGraph.addRenderPass(view.cullPhase2Pass);
            view.drawPass2.setup(cameraComponent, renderTarget, this._lightLayer);
            view.drawPass2.meshInstances = mis.slice(MESHLET_BUCKET_COUNT, MESHLET_BUCKET_COUNT * 2);
            frameGraph.addRenderPass(view.drawPass2);
        }

        // transparent scene passes (and anything after them - immediate, UI)
        if (splitIndex <= endIndex) {
            renderer.addMainRenderPass(frameGraph, layerComposition, renderTarget, splitIndex, endIndex, true);
        }
        return true;
    }
}

export { MeshletDirector };
