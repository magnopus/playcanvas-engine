import { Debug } from '../../core/debug.js';
import { SHADOWUPDATE_NONE, SHADOWUPDATE_THISFRAME, LAYERID_WORLD, LIGHTTYPE_DIRECTIONAL } from '../constants.js';
import { BUFFERUSAGE_COPY_DST } from '../../platform/graphics/constants.js';
import { StorageBuffer } from '../../platform/graphics/storage-buffer.js';
import { BoundingBox } from '../../core/shape/bounding-box.js';
import { Frustum } from '../../core/shape/frustum.js';
import { Mat4 } from '../../core/math/mat4.js';
import { math } from '../../core/math/math.js';
import { Vec3 } from '../../core/math/vec3.js';
import { CULL_FLAG_NO_TEXEL_RATE, CULL_FLAG_SHADOW_VIEW, CULL_PARAMS, WORK_ITEM_U32S, claimWordCount } from './constants.js';
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

// caster volume scratch (see _setCasterVolume)
const _sliceCorners = Array.from({ length: 8 }, () => new Vec3());
const _lightDir = new Vec3();
const _axisU = new Vec3();
const _axisV = new Vec3();
const _polyA = [];
const _polyB = [];
const _hullPoints = [];
const _hullOrder = [];
const _hull = [];
// receiver region planes (see _setCasterVolume): up to 6 box faces + 12 silhouette edges, as
// (nx, ny, nz, d) keeping dot(n, p) <= d
const _regionPlanes = new Float32Array(18 * 4);
// slice planes in the same form, for testing world corners against the slice
const _slicePlanes = new Float32Array(6 * 4);
const _faceNormal = new Vec3();
const _facing = [0, 0, 0, 0, 0, 0];
const _edgeA = new Vec3();
const _edgeB = new Vec3();

// the six faces of a frustum slice, as corner indices of Camera#getFrustumCorners (near 0-3,
// far 4-7, each in the same winding)
const SLICE_FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];

// the world box: its corners as (x, y, z) picks of min (0) / max (1), its faces as (axis, side)
// with the corner indices of each face, and its edges as corner index pairs with the two faces
// they join
const BOX_CORNERS = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const BOX_EDGES = [
    [0, 1, 2, 4], [3, 2, 3, 4], [4, 5, 2, 5], [7, 6, 3, 5], // along x: faces -y/+y, -z/+z
    [0, 3, 0, 4], [1, 2, 1, 4], [4, 7, 0, 5], [5, 6, 1, 5], // along y: faces -x/+x, -z/+z
    [0, 4, 0, 2], [1, 5, 1, 2], [3, 7, 0, 3], [2, 6, 1, 3] //  along z: faces -x/+x, -y/+y
];

// PCF filter reach plus a little, in shadow-map texels: receivers this close outside the slice's
// outline still sample casters inside it
const CASTER_VOLUME_TEXELS = 4;

// matches the drift a reused cascade tolerates before it refreshes (CASCADE_REUSE_DRIFT in
// shadow-renderer-directional.js), as a fraction of the slice's bounding-sphere radius
const CASCADE_REUSE_DRIFT = 0.1;

/**
 * Clips a convex polygon (flat xyz triplets) to the half-space `dot(n, p) <= d`.
 *
 * @param {number[]} src - Input polygon.
 * @param {number[]} dst - Output polygon, overwritten.
 * @param {Float32Array} planes - Packed planes (nx, ny, nz, d).
 * @param {number} o - Offset of the plane in `planes`.
 */
const clipPolygon = (src, dst, planes, o) => {
    dst.length = 0;
    const nx = planes[o], ny = planes[o + 1], nz = planes[o + 2], d = planes[o + 3];
    const n = src.length / 3;
    for (let i = 0; i < n; i++) {
        const a = i * 3;
        const b = ((i + 1) % n) * 3;
        const da = nx * src[a] + ny * src[a + 1] + nz * src[a + 2] - d;
        const db = nx * src[b] + ny * src[b + 1] + nz * src[b + 2] - d;
        if (da <= 0) dst.push(src[a], src[a + 1], src[a + 2]);
        if ((da <= 0) !== (db <= 0)) {
            const t = da / (da - db);
            dst.push(src[a] + (src[b] - src[a]) * t, src[a + 1] + (src[b + 1] - src[a + 1]) * t, src[a + 2] + (src[b + 2] - src[a + 2]) * t);
        }
    }
};

/**
 * Projects a point onto the light-space plane (axes {@link _axisU}, {@link _axisV}), appends it
 * to the hull points and returns its depth along {@link _lightDir}.
 *
 * @param {number} x - X.
 * @param {number} y - Y.
 * @param {number} z - Z.
 * @returns {number} The point's depth along the light.
 */
const addHullPoint = (x, y, z) => {
    _hullPoints.push(
        _axisU.x * x + _axisU.y * y + _axisU.z * z,
        _axisV.x * x + _axisV.y * y + _axisV.z * z
    );
    return _lightDir.x * x + _lightDir.y * y + _lightDir.z * z;
};

/**
 * Writes the planes of the region a directional light can carry a caster's shadow into: the
 * box swept to infinity along the light. Its faces are the box faces that do not face away
 * from the light, plus a plane through each silhouette edge (between a lit and an unlit face),
 * parallel to the light.
 *
 * @param {Vec3} min - Box minimum.
 * @param {Vec3} max - Box maximum.
 * @param {Vec3} dir - Normalized light travel direction.
 * @param {Float32Array} out - Receives packed planes (nx, ny, nz, d), `dot(n, p) <= d` inside.
 * @returns {number} The number of planes written.
 */
const sweptBoxPlanes = (min, max, dir, out) => {
    // face f = axis * 2 + side, outward normal +-axis; its facing is the sign of dot(n, dir)
    const facing = _facing;
    facing[0] = -dir.x;
    facing[1] = dir.x;
    facing[2] = -dir.y;
    facing[3] = dir.y;
    facing[4] = -dir.z;
    facing[5] = dir.z;
    let count = 0;
    for (let f = 0; f < 6; f++) {
        if (facing[f] > 0) continue; // faces away from the light: the sweep passes through it
        const axis = f >> 1;
        const sign = (f & 1) ? 1 : -1;
        const o = count++ * 4;
        out[o] = axis === 0 ? sign : 0;
        out[o + 1] = axis === 1 ? sign : 0;
        out[o + 2] = axis === 2 ? sign : 0;
        const bound = (f & 1) ? max : min;
        out[o + 3] = sign * (axis === 0 ? bound.x : axis === 1 ? bound.y : bound.z);
    }
    for (const [ia, ib, fa, fb] of BOX_EDGES) {
        if ((facing[fa] < 0) === (facing[fb] < 0) || facing[fa] === 0 || facing[fb] === 0) continue;
        const ca = BOX_CORNERS[ia];
        const cb = BOX_CORNERS[ib];
        _edgeA.set(ca[0] ? max.x : min.x, ca[1] ? max.y : min.y, ca[2] ? max.z : min.z);
        _edgeB.set(cb[0] ? max.x : min.x, cb[1] ? max.y : min.y, cb[2] ? max.z : min.z);
        _faceNormal.sub2(_edgeB, _edgeA).cross(_faceNormal, dir).normalize();
        // orient outwards: away from the box centre
        const cx = (min.x + max.x) * 0.5 - _edgeA.x;
        const cy = (min.y + max.y) * 0.5 - _edgeA.y;
        const cz = (min.z + max.z) * 0.5 - _edgeA.z;
        if (_faceNormal.x * cx + _faceNormal.y * cy + _faceNormal.z * cz > 0) _faceNormal.mulScalar(-1);
        const o = count++ * 4;
        out[o] = _faceNormal.x;
        out[o + 1] = _faceNormal.y;
        out[o + 2] = _faceNormal.z;
        out[o + 3] = _faceNormal.dot(_edgeA);
    }
    return count;
};

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
     * instance-meshlet pair - tens of MB on a scattered scene - and a per-cascade copy is the
     * single largest cost of adding shadows. Sharing is safe because every cascade's cull chain
     * is encoded back to back inside {@link cull}, each preceded by its own in-encoder clear, so
     * no two cascades' claim state is ever live at the same point in the command stream.
     *
     * Deliberately not shared with the camera views: their phase-2 cull is encoded later, during
     * the frame graph, and still reads the claim state phase 1 left behind.
     *
     * @type {StorageBuffer|null}
     * @private
     */
    _claimBits = null;

    /** @type {boolean} - set false to give every cascade its own claim plane (debugging). */
    shareClaimBits = true;

    /**
     * Include spot and omni lights. An omni costs six views - one cull chain per cube face - so
     * a scene with many shadowed point lights is expensive. Set false to keep meshlet shadows to directional lights only.
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
     * Cull each directional cascade's casters against its caster volume: the part of the
     * camera's view slice the cascade shades (clipped to the world), extruded towards the light.
     * A caster outside it cannot shadow anything that cascade is sampled for. Without it a
     * cascade culls against its whole light-space box, which encloses the slice's bounding
     * sphere - including ground behind and beside the camera, and the sky above the world.
     *
     * @type {boolean}
     */
    tightCasterVolume = true;

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

    /**
     * Every live shadow view, flattened; rebuilt only when an entry is created or destroyed.
     *
     * @type {MeshletShadowView[]}
     * @private
     */
    _views = [];

    /** @private */
    _viewsDirty = false;

    /** @private */
    _viewsEntryCount = 0;

    /**
     * Every live shadow view, flattened. The array is reused: do not hold it across frames.
     *
     * @type {MeshletShadowView[]}
     */
    get views() {
        // the size check also catches entries changed without going through _createEntry
        if (this._viewsDirty || this._viewsEntryCount !== this.entries.size) {
            this._viewsDirty = false;
            this._viewsEntryCount = this.entries.size;
            const out = this._views;
            out.length = 0;
            for (const entry of this.entries.values()) {
                for (let i = 0; i < entry.views.length; i++) out.push(entry.views[i]);
            }
        }
        return this._views;
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
        // for. Only one camera's lights are served (chosen below); a second camera would need its
        // own set of views.
        const desired = this._desired;
        desired.clear();
        const byCamera = director.renderer?.culler?.cameraDirShadowLights;
        // Serve the camera the meshlets are culled and drawn for. The map's first entry is
        // whichever camera the composition renders first, and in an application with several
        // (a first-person rig, an overlay, a picker) that is not always the scene camera: the
        // views then fit the wrong camera's cascades and the meshlets cast nothing where the
        // player looks. Fall back to the first entry only when the director's camera has no
        // shadowed directional light of its own.
        let sceneCamera = null;
        const preferred = director.cameraComponent?.camera ?? null;
        if (preferred && byCamera?.has(preferred)) {
            sceneCamera = preferred;
        } else {
            byCamera?.forEach((lightList, camera) => {
                if (!sceneCamera) sceneCamera = camera;
            });
        }
        if (sceneCamera) {
            const lightList = byCamera.get(sceneCamera);
            for (let i = 0; i < lightList.length; i++) {
                const light = lightList[i];
                if (light.enabled && light.castShadows) desired.add(light);
            }
        }

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
            // A light with a static shadow map renders it once and then only on request. Its
            // first request can come before any page is resident, so meshlets would never cast
            // into it: ask for a re-render whenever the streamed content changed since the last one.
            const world = director.world;
            if (entry.contentWorld !== world || entry.contentVersion !== world.contentVersion) {
                entry.contentWorld = world;
                entry.contentVersion = world.contentVersion;
                if (entry.light.shadowUpdateMode === SHADOWUPDATE_NONE) {
                    entry.light.shadowUpdateMode = SHADOWUPDATE_THISFRAME;
                }
            }
        });
    }

    /**
     * The AABB reported for one cascade's casters. The engine derives each cascade's shadow
     * depth range from the union of its visible casters' bounds, so reporting the whole world
     * is not a safe conservative choice: on a large landscape it stretches a near cascade's
     * depth range by orders of magnitude, beyond what the depth comparison can resolve, and
     * nothing is in shadow.
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
            // A cached shadow map (SHADOWUPDATE_NONE) is not re-rendered this frame, so its cull
            // chains would be pure cost: root admission alone scales with the casters inside each
            // cascade, which over a large world is milliseconds per far cascade. The mode is read
            // before the renderer consumes SHADOWUPDATE_THISFRAME, so a requested update still culls.
            const { light } = entry;
            if (light.shadowUpdateMode === SHADOWUPDATE_NONE) return;
            const overrides = light.shadowUpdateOverrides;
            const scheduled = light._type === LIGHTTYPE_DIRECTIONAL && light.cascadeUpdateIntervals;
            for (let c = 0; c < entry.views.length; c++) {
                // a cascade update schedule records its per-camera decision on the render data
                const skip = scheduled ? !light.getRenderData(entry.camera, c).cascadeRefresh : overrides?.[c] === SHADOWUPDATE_NONE;
                if (skip) continue;
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

        // Build the frustum from the final shadow camera state rather than reading
        // shadowCam.frustum: the directional cull's last updateFrustum() happens in its pass 1,
        // before pass 2 translates the camera along its forward axis and rewrites farClip to
        // tighten the depth range. Culling against the stale frustum would clip casters that the
        // shadow pass then expects to draw.
        _view.copy(node.getWorldTransform()).invert();
        _viewProj.mul2(shadowCam.projectionMatrix, _view);
        _frustum.setFromMat4(_viewProj);
        // same packed layout the cull shaders read: normal xyz, distance, per plane
        _planes.set(_frustum.planeData);

        const projection = this._projection(entry.light, view, shadowCam);
        if (!projection) {
            Debug.warnOnce('MeshletShadowRenderer: shadow camera has no usable projection scale; meshlet shadows for this face will be empty.');
            return;
        }

        const culler = view.culler;
        culler.twoPhase = false;
        culler.orthoScale = projection.orthoScale;
        // no texture-mip feedback from a light: the marks are a perspective texel rate
        // atomicMax'd into the world-shared requests buffer, so a light-space rate would poison
        // the camera's texture streaming. Page-residency marks stay on, so off-screen casters
        // stream in.
        culler.cullFlags = CULL_FLAG_NO_TEXEL_RATE | CULL_FLAG_SHADOW_VIEW;
        culler.viewDir = _viewDir.copy(node.forward);
        culler.dagPixelThreshold = director.dagPixelThreshold * director.shadowThresholdScale;
        culler.pressureScale = director.budget.pressureScale * view.pressureScale;
        // casters tiny on screen cast nothing, or the far cascades' root floor swamps the budget
        culler.rootMinPixels = director.shadowRootMinPixels * director.pixelScale;
        culler.rootHysteresis = director.rootHysteresis;
        culler.bucketThresholdScales = director.bucketThresholdScales;
        culler.bucketRootMinScales = director.bucketRootMinScales;
        culler.forceSubmitBoundaries = director.forceSubmitBoundaries;
        // rank casters by what they cover on screen, not in the light's (orthographic) view;
        // before the scene camera has culled, the face ranks by its own projection
        if (director.priorityProjScale > 0) {
            culler.priorityOrigin = director.priorityOrigin;
            culler.priorityProjScale = director.priorityProjScale;
        }
        // optionally, casters refine at least as far as the camera's own cut (backed off by the
        // face's pressure when its budget is tight)
        culler.cameraLodThreshold = director.shadowMatchCameraLod ?
            director.dagPixelThreshold * director.pixelScale * director.budget.pressureScale * view.pressureScale : 0;
        // the faces share one claim buffer, so its persistent admission bits would mix
        culler.hysteresis = !this._claimBits;

        this._setCasterVolume(entry, view, projection.orthoScale, culler);

        view.applyPendingGrowth();
        view.syncMaterials();
        view.monitorIndexDemand();

        culler.beginFrame(_planes, node.getPosition(), projection.projScale, _viewProj.data, null);
    }

    /**
     * Fills the culler's caster planes with a directional cascade's caster volume, or clears them
     * (local lights, no camera, or a volume too complex to express - the light-space box alone
     * then culls). The volume is the convex hull of the cascade's view slice, clipped to the
     * world bounds swept along the light (the receivers any caster can reach), projected along
     * the light: one plane per hull edge, parallel to the light,
     * plus one rejecting casters beyond the slice's farthest point along the light. It is grown
     * by the shadow filter's reach, the receivers' normal offset and, for a reused cascade, the
     * drift it tolerates before it refreshes.
     *
     * @param {ShadowLightEntry} entry - The light entry.
     * @param {MeshletShadowView} view - The cascade view.
     * @param {number} orthoScale - Shadow-map texels per world unit.
     * @param {import('./meshlet-culler.js').MeshletCuller} culler - The face's culler.
     * @private
     */
    _setCasterVolume(entry, view, orthoScale, culler) {
        culler.casterPlaneCount = 0;
        const { light, camera } = entry;
        if (!this.tightCasterVolume || light._type !== LIGHTTYPE_DIRECTIONAL || !camera || !(orthoScale > 0)) return;

        const cascade = view.face;
        const dists = light._shadowCascadeDistances;
        const far = dists?.[cascade] ?? 0;
        // a pixel in the blend band before this cascade's start may already sample it
        const near = cascade === 0 ? camera._nearClip : (dists?.[cascade - 1] ?? 0) * (1 - light.cascadeBlend);
        if (!(far > near)) return;

        const pts = camera.getFrustumCorners(near, far);
        const camWorld = camera.node.getWorldTransform();
        _center.set(0, 0, 0);
        for (let i = 0; i < 8; i++) {
            camWorld.transformPoint(pts[i], _sliceCorners[i]);
            _center.add(_sliceCorners[i]);
        }
        _center.mulScalar(1 / 8);
        let radius = 0;
        for (let i = 0; i < 8; i++) radius = Math.max(radius, _tmp.sub2(_sliceCorners[i], _center).length());

        // light-space axes: the light's travel direction and two perpendiculars
        _lightDir.copy(view.shadowCamera._node.forward).normalize();
        _axisU.cross(Math.abs(_lightDir.y) < 0.99 ? Vec3.UP : Vec3.RIGHT, _lightDir).normalize();
        _axisV.cross(_lightDir, _axisU);

        // The receivers a meshlet caster can shadow lie in the world box swept along the light -
        // not in the box itself: the ground a meshlet's shadow falls on is often other geometry
        // (a floor, terrain) outside the meshlet world's bounds. The volume is the slice's
        // intersection with that region, whose vertices are the slice faces clipped to the
        // region plus the box corners inside the slice (a box wholly inside the slice clips
        // every slice face away).
        const world = this.director.world.worldBounds;
        const wmin = world.getMin();
        const wmax = world.getMax();
        const regionCount = sweptBoxPlanes(wmin, wmax, _lightDir, _regionPlanes);
        _hullPoints.length = 0;
        let maxDepth = -Infinity;
        for (let f = 0; f < 6; f++) {
            const face = SLICE_FACES[f];
            let src = _polyA;
            let dst = _polyB;
            src.length = 0;
            for (const index of face) {
                const p = _sliceCorners[index];
                src.push(p.x, p.y, p.z);
            }

            // the face's outward plane, for the corner test below
            const p0 = _sliceCorners[face[0]];
            _faceNormal.cross(_edgeA.sub2(_sliceCorners[face[1]], p0), _edgeB.sub2(_sliceCorners[face[2]], p0)).normalize();
            if (_faceNormal.dot(_tmp.sub2(_center, p0)) > 0) _faceNormal.mulScalar(-1);
            _slicePlanes[f * 4] = _faceNormal.x;
            _slicePlanes[f * 4 + 1] = _faceNormal.y;
            _slicePlanes[f * 4 + 2] = _faceNormal.z;
            _slicePlanes[f * 4 + 3] = _faceNormal.dot(p0);

            for (let r = 0; r < regionCount && src.length; r++) {
                clipPolygon(src, dst, _regionPlanes, r * 4);
                [src, dst] = [dst, src];
            }
            for (let i = 0; i < src.length; i += 3) maxDepth = Math.max(maxDepth, addHullPoint(src[i], src[i + 1], src[i + 2]));
        }
        const eps = 1e-4 * (radius + 1);
        for (const [cx, cy, cz] of BOX_CORNERS) {
            const x = cx ? wmax.x : wmin.x;
            const y = cy ? wmax.y : wmin.y;
            const z = cz ? wmax.z : wmin.z;
            let inside = true;
            for (let f = 0; f < 6 && inside; f++) {
                const o = f * 4;
                inside = _slicePlanes[o] * x + _slicePlanes[o + 1] * y + _slicePlanes[o + 2] * z <= _slicePlanes[o + 3] + eps;
            }
            if (inside) maxDepth = Math.max(maxDepth, addHullPoint(x, y, z));
        }

        const planes = culler.casterPlanes;
        if (_hullPoints.length < 6) {
            // the slice sees nothing any meshlet can shadow, so nothing casts
            planes.fill(0, 0, 4);
            planes[3] = -1e30;
            culler.casterPlaneCount = 1;
            return;
        }

        // convex hull of the projected points (Andrew's monotone chain, counter-clockwise)
        const count = _hullPoints.length / 2;
        _hullOrder.length = count;
        for (let i = 0; i < count; i++) _hullOrder[i] = i;
        _hullOrder.sort((a, b) => (_hullPoints[a * 2] - _hullPoints[b * 2]) || (_hullPoints[a * 2 + 1] - _hullPoints[b * 2 + 1]));
        const cross = (o, a, b) => (_hullPoints[a * 2] - _hullPoints[o * 2]) * (_hullPoints[b * 2 + 1] - _hullPoints[o * 2 + 1]) -
            (_hullPoints[a * 2 + 1] - _hullPoints[o * 2 + 1]) * (_hullPoints[b * 2] - _hullPoints[o * 2]);
        _hull.length = 0;
        for (let pass = 0; pass < 2; pass++) {
            const start = _hull.length;
            for (let k = 0; k < count; k++) {
                const i = _hullOrder[pass === 0 ? k : count - 1 - k];
                while (_hull.length >= start + 2 && cross(_hull[_hull.length - 2], _hull[_hull.length - 1], i) <= 0) _hull.pop();
                _hull.push(i);
            }
            _hull.pop();
        }
        const edges = _hull.length;
        if (edges < 3 || edges + 1 > CULL_PARAMS.MAX_CASTER_PLANES) return;

        let margin = CASTER_VOLUME_TEXELS / orthoScale + Math.max(light._normalOffsetBias, 0);
        if ((light.cascadeUpdateIntervals?.[cascade] ?? 1) > 1) margin += CASCADE_REUSE_DRIFT * radius;

        for (let e = 0; e < edges; e++) {
            const a = _hull[e];
            const b = _hull[(e + 1) % edges];
            const ax = _hullPoints[a * 2], ay = _hullPoints[a * 2 + 1];
            // inward (left) normal of a counter-clockwise edge
            let nx = -(_hullPoints[b * 2 + 1] - ay);
            let ny = _hullPoints[b * 2] - ax;
            const len = Math.hypot(nx, ny);
            if (!(len > 0)) return;
            nx /= len;
            ny /= len;
            const o = e * 4;
            planes[o] = nx * _axisU.x + ny * _axisV.x;
            planes[o + 1] = nx * _axisU.y + ny * _axisV.y;
            planes[o + 2] = nx * _axisU.z + ny * _axisV.z;
            planes[o + 3] = -(nx * ax + ny * ay) + margin;
        }
        // casters beyond every receiver along the light shadow nothing in the slice
        const o = edges * 4;
        planes[o] = -_lightDir.x;
        planes[o + 1] = -_lightDir.y;
        planes[o + 2] = -_lightDir.z;
        planes[o + 3] = maxDepth + margin;
        culler.casterPlaneCount = edges + 1;
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
            const words = claimWordCount(director.world.totalPairs, director.world.instanceCount);
            this._claimBits = new StorageBuffer(this.device, words * BYTES_PER_WORD, BUFFERUSAGE_COPY_DST);
        }
        if (!this._workItems) {
            this._workItems = new StorageBuffer(this.device,
                Math.max(director.world.workItemCapacity * WORK_ITEM_U32S * BYTES_PER_WORD, 16), BUFFERUSAGE_COPY_DST);
        }
        this._viewsDirty = true;
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
        this._viewsDirty = true;
        entry.views.forEach((view) => {
            entry.layer.removeShadowCasters(view.meshInstances);
            view.destroy();
        });
        entry.views.length = 0;
    }
}

export { MeshletShadowRenderer };
