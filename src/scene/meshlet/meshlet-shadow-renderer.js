import { Debug } from '../../core/debug.js';
import { BUFFERUSAGE_COPY_DST } from '../../platform/graphics/constants.js';
import { StorageBuffer } from '../../platform/graphics/storage-buffer.js';
import { BoundingBox } from '../../core/shape/bounding-box.js';
import { Frustum } from '../../core/shape/frustum.js';
import { Mat4 } from '../../core/math/mat4.js';
import { math } from '../../core/math/math.js';
import { Vec3 } from '../../core/math/vec3.js';
import { LAYERID_WORLD, LIGHTTYPE_DIRECTIONAL } from '../constants.js';
import { CULL_FLAG_NO_TEXEL_RATE, WORK_ITEM_U32S } from './constants.js';
import { MeshletShadowView } from './meshlet-shadow-view.js';

/**
 * @import { Camera } from '../camera.js'
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { Layer } from '../layer.js'
 * @import { Light } from '../light.js'
 * @import { MeshletDirector } from './meshlet-director.js'
 * @import { StorageBuffer } from '../../platform/graphics/storage-buffer.js'
 */

// a light shining straight along the horizon has no vertical component to divide by; the
// horizontal caster padding is capped at height / MIN_LIGHT_VERTICAL world units
const MIN_LIGHT_VERTICAL = 1e-3;

// bytes per u32 of a word-addressed buffer
const BYTES_PER_WORD = 4;

const _viewProj = new Mat4();
const _view = new Mat4();
const _frustum = new Frustum();
const _planes = new Float32Array(24);
const _viewDir = new Vec3();
const _center = new Vec3();
const _tmp = new Vec3();
const _casterBox = new BoundingBox();

/**
 * @typedef {object} ShadowLightEntry
 * @property {Light} light - The light.
 * @property {Camera|null} camera - The scene camera the light's shadow render data belongs to
 * (directional shadow data is per camera; null for local lights).
 * @property {Layer} layer - The layer the casters are registered on.
 * @property {MeshletShadowView[]} views - One view per shadow face (cascade, or cube face).
 */

/**
 * Casts meshlet shadows: directional cascades and, when {@link localLights} is on, spot and
 * omni faces. Each (light, face) gets its own single-phase {@link MeshletShadowView}, whose
 * mesh instances are registered as layer shadow casters so the engine's own
 * {@link ShadowRenderer#submitCasters} draws them - it is already indirect-draw aware, so
 * nothing about the shadow pass itself needs changing.
 *
 * Lifecycle is split across the frame, mirroring the gsplat shadow renderer:
 * - {@link syncLights} runs pre-cull, once `culler.cameraDirShadowLights` is populated and before
 *   `cullComposition` reads `layer.shadowCasters`. It reconciles the view pool and registers or
 *   unregisters casters.
 * - {@link cull} runs post-cull, once each cascade's shadow camera has been fitted (which happens
 *   inside `cullComposition`), and dispatches one cull chain per cascade.
 *
 * @ignore
 */
class MeshletShadowRenderer {
    /** @type {GraphicsDevice} */
    device;

    /** @type {MeshletDirector} */
    director;

    /** @type {Map<Light, ShadowLightEntry>} */
    entries = new Map();

    /**
     * Cap on total shadow views (cascades summed across lights). Each carries its own records and
     * index buffers, so an unbounded set is a memory hazard on a big world.
     *
     * @type {number}
     */
    maxShadowViews = 8;

    /** @type {Set<Light>} - scratch, rebuilt every syncLights to diff against entries. */
    _desired = new Set();

    /**
     * One claim-bits plane shared by every shadow view. Claim bits are one bit per
     * instance-meshlet pair - ~18.6 MB on the jungle - and a per-cascade copy of that is the
     * single largest cost of adding shadows. Sharing is safe because every cascade's cull chain
     * is encoded back to back inside {@link cull}, each preceded by its own in-encoder clear, so
     * no two cascades' claim state is ever live at the same point in the command stream.
     *
     * Deliberately NOT shared with the camera views: their phase-2 cull is encoded later, during
     * the frame graph, and still reads the claim state phase 1 left behind.
     *
     * @type {StorageBuffer|null}
     * @private
     */
    _claimBits = null;

    /** @type {boolean} - set false to give every cascade its own claim plane (debugging). */
    shareClaimBits = true;

    /**
     * Include spot and omni lights. An omni costs SIX views - one cull chain per cube face - so
     * a scene with many shadowed point lights is expensive until the single-pass multi-view cull
     * lands. Set false to keep meshlet shadows to directional lights only.
     *
     * @type {boolean}
     */
    localLights = true;

    /**
     * Work items, shared by every shadow view for the same reason as {@link _claimBits}: the
     * cascades' chains are encoded back to back and each begins by clearing the counters that
     * index them. Sized from the world's work-item capacity, which on a scattered scene is tens
     * of MB - per cascade, that adds up faster than anything else here.
     *
     * @type {StorageBuffer|null}
     * @private
     */
    _workItems = null;

    /**
     * @param {GraphicsDevice} device - The graphics device.
     * @param {MeshletDirector} director - The owning director.
     */
    constructor(device, director) {
        this.device = device;
        this.director = director;
    }

    destroy() {
        this.entries.forEach(entry => this._destroyEntry(entry));
        this.entries.clear();
        this._claimBits?.destroy();
        this._claimBits = null;
        this._workItems?.destroy();
        this._workItems = null;
    }

    /** @type {MeshletShadowView[]} - every live shadow view, flattened. */
    get views() {
        const out = [];
        this.entries.forEach(entry => out.push(...entry.views));
        return out;
    }

    /**
     * Drops every view, unregistering the casters. Used when the world is rebuilt (the views hold
     * buffers sized to the old world) or when shadow casting is switched off.
     */
    reset() {
        this.destroy();
    }

    /**
     * Pre-cull reconciliation: one entry per shadow-casting directional light in the composition,
     * with `light.numCascades` views. Must run after `culler.updateLightVisibility` (which fills
     * `cameraDirShadowLights`) and before `cullComposition` (which reads `layer.shadowCasters`).
     *
     * @param {object} comp - The layer composition.
     */
    syncLights(comp) {
        const { director } = this;
        const layer = comp?.getLayerById?.(LAYERID_WORLD) ?? null;
        if (!layer || !director.world?.finalized) {
            return;
        }

        // The engine's own selection, so meshlet casters appear for exactly the lights that get
        // a shadow pass - keyed by scene camera, because directional shadow render data is per
        // (light, camera) and a cascade's shadow camera only exists for the camera it was fitted
        // for. Only the first camera's lights are served; a second camera would need its own set
        // of views (see the multi-camera note in the README).
        const desired = this._desired;
        desired.clear();
        let sceneCamera = null;
        const byCamera = director.renderer?.culler?.cameraDirShadowLights;
        byCamera?.forEach((lightList, camera) => {
            if (sceneCamera && camera !== sceneCamera) return;
            sceneCamera = camera;
            for (let i = 0; i < lightList.length; i++) {
                const light = lightList[i];
                if (light.enabled && light.castShadows) desired.add(light);
            }
        });

        // Local lights, if enabled. Their shadow render data is camera-independent
        // (getRenderData(null, face)), so unlike directional they need no per-camera keying -
        // one set of views serves every camera.
        if (this.localLights) {
            const locals = director.renderer?.localLights;
            for (let i = 0; i < (locals?.length ?? 0); i++) {
                const light = locals[i];
                if (light.enabled && light.castShadows && light.visibleThisFrame) desired.add(light);
            }
        }

        // retire entries whose light went away, stopped casting, or changed face count
        this.entries.forEach((entry, light) => {
            const camera = light._type === LIGHTTYPE_DIRECTIONAL ? sceneCamera : null;
            if (!desired.has(light) || entry.camera !== camera ||
                entry.views.length !== light.numShadowFaces) {
                this._destroyEntry(entry);
                this.entries.delete(light);
            }
        });

        let viewCount = 0;
        this.entries.forEach(entry => (viewCount += entry.views.length));

        desired.forEach((light) => {
            if (this.entries.has(light)) return;
            const faces = light.numShadowFaces;
            if (viewCount + faces > this.maxShadowViews) {
                Debug.warnOnce(`MeshletShadowRenderer: ${light._node?.name ?? 'light'} needs ${faces} shadow view(s) but only ${this.maxShadowViews - viewCount} remain (maxShadowViews ${this.maxShadowViews}); it will not cast meshlet shadows. An omni light costs six - one per cube face.`);
                return;
            }
            viewCount += faces;
            const camera = light._type === LIGHTTYPE_DIRECTIONAL ? sceneCamera : null;
            this.entries.set(light, this._createEntry(light, camera, layer));
        });

        this.entries.forEach((entry) => {
            entry.views.forEach(view => view.setCasterBounds(this._casterBounds(entry, view)));
        });
    }

    /**
     * The AABB reported for one cascade's casters. The engine derives each cascade's shadow
     * DEPTH RANGE from the union of its visible casters' bounds, so handing every cascade the
     * whole world is not the harmless conservative choice it looks like: on an 8 km landscape it
     * stretches a 19 m cascade's depth range to ~9.5 km, at which point the depth comparison
     * cannot resolve anything and NOTHING is ever in shadow - silently, again.
     *
     * So each cascade reports the volume it actually covers: the cascade's own frustum-slice
     * sphere, grown horizontally by how far a caster at the top of the world can reach into it
     * along the light, extended vertically over the world's height, and clipped to the world.
     * The cascade split distances come from the light's last fit, so this trails the camera by
     * one frame - immaterial next to the horizontal padding, and it avoids duplicating the
     * engine's split formula. Before the first fit there are no splits and the world bounds are
     * the only answer available.
     *
     * @param {ShadowLightEntry} entry - The light entry.
     * @param {MeshletShadowView} view - The cascade view.
     * @returns {BoundingBox} The caster bounds for this cascade.
     * @private
     */
    _casterBounds(entry, view) {
        const world = this.director.world.worldBounds;
        const { light, camera } = entry;

        // A local light only lights what is inside its attenuation sphere, so that box - clipped
        // to the world - is the caster volume, and it is far tighter than anything a directional
        // cascade can claim. Nothing outside it can cast into the shadow map at all.
        if (light._type !== LIGHTTYPE_DIRECTIONAL) {
            const range = light.attenuationEnd;
            const pos = light._node?.getPosition();
            if (!(range > 0) || !pos) return world;
            const min = world.getMin();
            const max = world.getMax();
            const x0 = Math.max(pos.x - range, min.x);
            const x1 = Math.min(pos.x + range, max.x);
            const y0 = Math.max(pos.y - range, min.y);
            const y1 = Math.min(pos.y + range, max.y);
            const z0 = Math.max(pos.z - range, min.z);
            const z1 = Math.min(pos.z + range, max.z);
            if (x1 <= x0 || y1 <= y0 || z1 <= z0) return world;   // light is outside the world
            _casterBox.center.set((x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5);
            _casterBox.halfExtents.set((x1 - x0) * 0.5, (y1 - y0) * 0.5, (z1 - z0) * 0.5);
            return _casterBox;
        }

        const dists = light._shadowCascadeDistances;
        const cascade = view.face;
        const far = dists?.[cascade] ?? 0;
        const near = cascade === 0 ? camera._nearClip : (dists?.[cascade - 1] ?? 0);
        const node = view.shadowCamera?._node;
        if (!(far > near) || !node) {
            return world;
        }

        // the cascade's frustum slice, as a world-space sphere (the same construction the engine
        // uses to fit the cascade itself)
        const pts = camera.getFrustumCorners(near, far);
        const camWorld = camera.node.getWorldTransform();
        _center.set(0, 0, 0);
        for (let i = 0; i < 8; i++) {
            camWorld.transformPoint(pts[i], pts[i]);
            _center.add(pts[i]);
        }
        _center.mulScalar(1 / 8);
        let radius = 0;
        for (let i = 0; i < 8; i++) {
            radius = Math.max(radius, _tmp.sub2(pts[i], _center).length());
        }

        // A caster h above the cascade shadows into it from h / tan(elevation) away
        // horizontally. Padding symmetrically rather than only up-sun costs a little depth range
        // and removes any dependence on getting the sign of the light direction right.
        const min = world.getMin();
        const max = world.getMax();
        const height = max.y - min.y;
        const fwd = node.forward;
        const horiz = Math.hypot(fwd.x, fwd.z);
        const vert = Math.max(Math.abs(fwd.y), MIN_LIGHT_VERTICAL);
        const pad = radius + height * (horiz / vert);

        const x0 = Math.max(_center.x - pad, min.x);
        const x1 = Math.min(_center.x + pad, max.x);
        const z0 = Math.max(_center.z - pad, min.z);
        const z1 = Math.min(_center.z + pad, max.z);
        if (x1 <= x0 || z1 <= z0) {
            return world; // the cascade is outside the world; nothing casts, bounds are moot
        }
        _casterBox.center.set((x0 + x1) * 0.5, (min.y + max.y) * 0.5, (z0 + z1) * 0.5);
        _casterBox.halfExtents.set((x1 - x0) * 0.5, height * 0.5, (z1 - z0) * 0.5);
        return _casterBox;
    }

    /**
     * Post-cull dispatch: one cull chain per cascade. Must run after `cullComposition`, which is
     * where each cascade's shadow camera is positioned and its depth range fitted.
     */
    cull() {
        const { director } = this;
        this.entries.forEach((entry) => {
            for (let c = 0; c < entry.views.length; c++) {
                this._cullView(entry, entry.views[c], director);
            }
        });
    }

    /**
     * @param {ShadowLightEntry} entry - The light entry.
     * @param {MeshletShadowView} view - The cascade view.
     * @param {MeshletDirector} director - The director (thresholds, pressure, world).
     * @private
     */
    _cullView(entry, view, director) {
        const shadowCam = view.shadowCamera;
        if (!shadowCam) return;

        const node = shadowCam._node;

        // Build the frustum from the FINAL shadow camera state rather than reading
        // shadowCam.frustum: the directional cull's last updateFrustum() happens in its pass 1,
        // before pass 2 translates the camera along its forward axis and rewrites farClip to
        // tighten the depth range. Culling against the stale frustum would clip casters that the
        // shadow pass then expects to draw.
        _view.copy(node.getWorldTransform()).invert();
        _viewProj.mul2(shadowCam.projectionMatrix, _view);
        _frustum.setFromMat4(_viewProj);
        for (let i = 0; i < 6; i++) {
            const plane = _frustum.planes[i];
            _planes[i * 4 + 0] = plane.normal.x;
            _planes[i * 4 + 1] = plane.normal.y;
            _planes[i * 4 + 2] = plane.normal.z;
            _planes[i * 4 + 3] = plane.distance;
        }

        const projection = this._projection(entry.light, view, shadowCam);
        if (!projection) {
            Debug.warnOnce('MeshletShadowRenderer: shadow camera has no usable projection scale; meshlet shadows for this face will be empty.');
            return;
        }

        const culler = view.culler;
        culler.twoPhase = false;
        culler.orthoScale = projection.orthoScale;
        // no texture-mip feedback from a light: the marks are a perspective texel rate
        // atomicMax'd into the WORLD-SHARED requests buffer, so a light-space rate would poison
        // the camera's texture streaming. Page-residency marks stay on, so off-screen casters
        // stream in.
        culler.cullFlags = CULL_FLAG_NO_TEXEL_RATE;
        culler.viewDir = _viewDir.copy(node.forward);
        culler.dagPixelThreshold = director.dagPixelThreshold * director.shadowThresholdScale;
        culler.pressureScale = director.budget.pressureScale;
        culler.forceSubmitBoundaries = director.forceSubmitBoundaries;

        view.applyPendingGrowth();
        view.syncMaterials();
        view.monitorIndexDemand();

        culler.beginFrame(_planes, node.getPosition(), projection.projScale, _viewProj.data, null);
    }

    /**
     * The LOD projection for one shadow face.
     *
     * A directional cascade is orthographic, so its projected error is distance-independent and
     * the cut is driven by a texel rate: shadow-map texels per world unit. Spot and omni faces
     * are ordinary perspective frusta, so they use the same `projScale / distance` form the
     * camera does - the cull shader's default branch - with projScale in shadow-map pixels.
     *
     * The face's pixel height comes from the render target and its viewport rect, which covers
     * both a cascade's slot in the directional map and a local light's tile in the clustered
     * shadow atlas. Before the first shadow render there is no target yet, so the light's
     * nominal resolution stands in.
     *
     * Cross-check for the ortho case: with a 2x2 cascade grid (viewport w = 0.5) and
     * orthoHeight = radius this is `0.25 * shadowResolution / radius`, exactly the engine's own
     * texel-snapping constant in shadow-renderer-directional.js.
     *
     * @param {Light} light - The light.
     * @param {MeshletShadowView} view - The face's view.
     * @param {Camera} shadowCam - The prepared shadow camera.
     * @returns {{ orthoScale: number, projScale: number }|null} The projection, or null when it
     * cannot be derived.
     * @private
     */
    _projection(light, view, shadowCam) {
        const renderData = view.renderData;
        const rt = shadowCam.renderTarget;
        const viewportH = renderData?.shadowViewport?.w ?? 1;
        const pixels = (rt?.height ?? light._shadowResolution) * viewportH;
        if (!(pixels > 0)) return null;

        if (light._type === LIGHTTYPE_DIRECTIONAL) {
            const orthoHeight = shadowCam.orthoHeight;
            if (!(orthoHeight > 0)) return null;
            return { orthoScale: pixels / (2 * orthoHeight), projScale: 1 };
        }

        // perspective: half the face's pixel height over the tangent of the half angle
        const halfFov = 0.5 * shadowCam.fov * math.DEG_TO_RAD;
        const t = Math.tan(halfFov);
        if (!(t > 0)) return null;
        return { orthoScale: 0, projScale: pixels / (2 * t) };
    }

    /**
     * @param {Light} light - The light.
     * @param {Camera|null} camera - The scene camera (directional lights), or null.
     * @param {Layer} layer - The layer to register casters on.
     * @returns {ShadowLightEntry} The created entry.
     * @private
     */
    _createEntry(light, camera, layer) {
        const { director } = this;
        const faces = light.numShadowFaces;
        // sized exactly as a view sizes its own (one bit per instance-meshlet pair, 32 to a
        // word; one work item per WORK_ITEM_U32S words)
        if (this.shareClaimBits && !this._claimBits) {
            const words = Math.max(Math.ceil(director.world.totalPairs / 32), 4);
            this._claimBits = new StorageBuffer(this.device, words * BYTES_PER_WORD, BUFFERUSAGE_COPY_DST);
        }
        if (!this._workItems) {
            this._workItems = new StorageBuffer(this.device,
                Math.max(director.world.workItemCapacity * WORK_ITEM_U32S * BYTES_PER_WORD, 16), BUFFERUSAGE_COPY_DST);
        }
        const views = [];
        for (let c = 0; c < faces; c++) {
            const view = new MeshletShadowView(
                this.device, director.world, director.renderer, light, c, camera,
                {
                    cullShaders: director.cullShaders,
                    capCarry: director.takeShadowCapCarry(light.id, c),
                    sharedClaimBits: this._claimBits,
                    sharedWorkItems: this._workItems,
                    initialIndexScale: director.shadowInitialIndexScale
                }
            );
            layer.addShadowCasters(view.meshInstances);
            views.push(view);
        }
        return { light, camera, layer, views };
    }

    /**
     * @param {ShadowLightEntry} entry - The entry to tear down.
     * @private
     */
    _destroyEntry(entry) {
        entry.views.forEach((view) => {
            entry.layer.removeShadowCasters(view.meshInstances);
            view.destroy();
        });
        entry.views.length = 0;
    }
}

export { MeshletShadowRenderer };
