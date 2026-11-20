import { Debug } from '../../core/debug.js';
import { now } from '../../core/time.js';
import { SHADOWUPDATE_NONE, SHADOWUPDATE_THISFRAME, LAYERID_WORLD, LIGHTTYPE_DIRECTIONAL } from '../constants.js';
import { BUFFERUSAGE_COPY_DST } from '../../platform/graphics/constants.js';
import { StorageBuffer } from '../../platform/graphics/storage-buffer.js';
import { BoundingBox } from '../../core/shape/bounding-box.js';
import { Frustum } from '../../core/shape/frustum.js';
import { Mat4 } from '../../core/math/mat4.js';
import { math } from '../../core/math/math.js';
import { Vec3 } from '../../core/math/vec3.js';
import { CULL_FLAG_NO_TEXEL_RATE, CULL_FLAG_SHADOW_VIEW, CULL_PARAMS, ROOT_LIST_U32S, claimWordCount } from './constants.js';
import { casterVolumePlanes, sliceSphere } from './meshlet-caster-volume.js';
import { MeshletShadowView } from './meshlet-shadow-view.js';
import { CASCADE_REUSE_DRIFT } from '../renderer/shadow-renderer-directional.js';

/**
 * @import { Camera } from '../camera.js'
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { Layer } from '../layer.js'
 * @import { Light } from '../light.js'
 * @import { MeshletCuller } from './meshlet-culler.js'
 * @import { MeshletDirector } from './meshlet-director.js'
 */

// a light shining straight along the horizon has no vertical component to divide by; the
// horizontal caster padding is capped at height / MIN_LIGHT_VERTICAL world units
const MIN_LIGHT_VERTICAL = 1e-3;

// bytes per u32 of a word-addressed buffer
const BYTES_PER_WORD = 4;

// PCF filter reach plus a little, in shadow-map texels: receivers this close outside the slice's
// outline still sample casters inside it
const CASTER_VOLUME_TEXELS = 4;

const _viewProj = new Mat4();
const _view = new Mat4();
const _frustum = new Frustum();
const _planes = new Float32Array(24);
const _viewDir = new Vec3();
const _center = new Vec3();
const _casterBox = new BoundingBox();
const _sliceCorners = Array.from({ length: 8 }, () => new Vec3());

/**
 * @typedef {object} ShadowLightEntry
 * @property {Light} light - The light.
 * @property {Camera|null} camera - The scene camera the light's shadow render data belongs to
 * (directional shadow data is per camera; null for local lights).
 * @property {Layer} layer - The layer the casters are registered on.
 * @property {MeshletShadowView[]} views - One view per shadow face (cascade, or cube face).
 * @property {number} [contentChangedAt] - When the streamed content last changed, in ms.
 * @property {number} [contentDirtyAt] - When it first changed after the last re-render request, in ms.
 * @property {object} [contentWorld] - The world the content was last compared against.
 * @property {number} [contentVersion] - The world content version last compared against.
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

    /**
     * How long streamed content must be unchanged, in milliseconds, before a cached shadow map
     * (SHADOWUPDATE_NONE) is re-rendered for it.
     *
     * @type {number}
     */
    staticSettleMs = 400;

    /**
     * The longest a cached shadow map waits for a re-render, in milliseconds, while the streamed
     * content keeps changing.
     *
     * @type {number}
     */
    staticMaxWaitMs = 3000;

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

    /**
     * Include spot and omni lights. An omni costs six views - one cull chain per cube face - so
     * a scene with many shadowed point lights is expensive. Set false to keep meshlet shadows to directional lights only.
     *
     * @type {boolean}
     */
    localLights = true;

    /**
     * Root list, shared by every shadow view for the same reason as {@link _claimBits}: the
     * cascades' chains are encoded back to back and each rebuilds it from scratch.
     *
     * @type {StorageBuffer|null}
     * @private
     */
    _rootList = null;

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
        this._rootList?.destroy();
        this._rootList = null;
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
        // a shadow pass. Directional shadow render data is per (light, camera), so only one
        // camera's lights are served: the camera the meshlets are drawn for, which need not be
        // the first the composition renders (an overlay, a picker), else the first.
        const desired = this._desired;
        desired.clear();
        const byCamera = director.renderer?.culler?.cameraDirShadowLights;
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
            const camera = light._type === LIGHTTYPE_DIRECTIONAL && !light._shadowBounds ? sceneCamera : null;
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
            const camera = light._type === LIGHTTYPE_DIRECTIONAL && !light._shadowBounds ? sceneCamera : null;
            this.entries.set(light, this._createEntry(light, camera, layer));
        });

        this.entries.forEach((entry) => {
            entry.views.forEach(view => view.setCasterBounds(this._casterBounds(entry, view)));
            // A static shadow map can render before its casters' pages are resident, so request
            // a re-render once the streamed content has changed and then settled (or waited too
            // long): re-rendering per page install would cost a shadow pass each frame and flicker.
            const world = director.world;
            const time = now();
            if (entry.contentWorld !== world || entry.contentVersion !== world.contentVersion) {
                entry.contentWorld = world;
                entry.contentVersion = world.contentVersion;
                entry.contentChangedAt = time;
                entry.contentDirtyAt ??= time;
            }
            if (entry.contentDirtyAt !== undefined && entry.light.shadowUpdateMode !== SHADOWUPDATE_NONE) {
                entry.contentDirtyAt = undefined;
            } else if (entry.contentDirtyAt !== undefined &&
                (time - entry.contentChangedAt >= this.staticSettleMs || time - entry.contentDirtyAt >= this.staticMaxWaitMs)) {
                const wanted = this._staticNeedsContent(entry);
                // undefined: the last render's counters are still on their way back
                if (wanted !== undefined) {
                    if (wanted) entry.light.shadowUpdateMode = SHADOWUPDATE_THISFRAME;
                    entry.contentDirtyAt = undefined;
                }
            }
        });
    }

    /**
     * Whether a cached shadow map would come out different now that the streamed content has
     * changed: only when its last render was incomplete in a way new pages or more capacity can
     * fix. The content version moves with every page the camera streams in or evicts, and a
     * shadow cut that waited on none of them renders the same casters again - yet the
     * re-render costs a whole shadow pass and cuts at whatever LOD pressure the face has at that
     * moment, which is what reads as the shadows flashing.
     *
     * @param {ShadowLightEntry} entry - The light entry.
     * @returns {boolean|undefined} True to re-render, false to keep the map, undefined while the
     * last render's counters have not been read back yet.
     * @private
     */
    _staticNeedsContent(entry) {
        for (let i = 0; i < entry.views.length; i++) {
            const view = entry.views[i];
            if (view._countersReadBusy) return undefined;
            if (!view.lastDemand) return true;
            const roots = view.lastRoots;
            // nothing admitted: the render ran before any caster could be (the first request
            // can come before the world, its cells or the camera's priority are ready)
            if (roots.candidates === 0) return true;
            // casters short of their pages: roots not resident, or groups held coarse
            if (roots.unready > 0 || roots.waiting > 0) return true;
            // casters dropped for capacity, which the next render can still grow into
            if (roots.rejectedInstances > 0 && view.canGrowIndices()) return true;
        }
        return false;
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

        const node = view.shadowCamera?._node;
        let radius = 0;
        if (light._shadowBounds) {
            // shadow bounds: one camera-independent map covering a fixed box, which stands in for
            // the cascade's slice
            if (!node) return world;
            _center.copy(light._shadowBounds.center);
            radius = light._shadowBounds.halfExtents.length();
        } else {
            const dists = light._shadowCascadeDistances;
            const cascade = view.face;
            const far = dists?.[cascade] ?? 0;
            const near = cascade === 0 ? camera._nearClip : (dists?.[cascade - 1] ?? 0);
            if (!(far > near) || !node) {
                return world;
            }

            radius = sliceSphere(camera, near, far, _sliceCorners, _center);
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
            // chains would be pure cost. The mode is read before the renderer consumes
            // SHADOWUPDATE_THISFRAME, so a requested update still culls.
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
        culler.hysteresis = false;

        this._setCasterVolume(entry, view, projection.orthoScale, culler);

        view.applyPendingGrowth();
        view.syncMaterials();
        view.monitorIndexDemand();

        culler.beginFrame(_planes, node.getPosition(), projection.projScale, _viewProj.data, null);
    }

    /**
     * Fills the culler's caster planes with a directional cascade's caster volume (see
     * {@link casterVolumePlanes}), or clears them for local lights, without a camera, or when the
     * volume is too complex to express - the light-space box alone then culls. The volume is
     * grown by the shadow filter's reach, the receivers' normal offset and, for a reused cascade,
     * the drift it tolerates before it refreshes.
     *
     * The receivers a meshlet caster can shadow lie in the world box swept along the light, not
     * in the box itself: the ground a meshlet's shadow falls on is often other geometry (a floor,
     * terrain) outside the meshlet world's bounds.
     *
     * @param {ShadowLightEntry} entry - The light entry.
     * @param {MeshletShadowView} view - The cascade view.
     * @param {number} orthoScale - Shadow-map texels per world unit.
     * @param {MeshletCuller} culler - The face's culler.
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

        const radius = sliceSphere(camera, near, far, _sliceCorners, _center);
        let margin = CASTER_VOLUME_TEXELS / orthoScale + Math.max(light._normalOffsetBias, 0);
        if ((light.cascadeUpdateIntervals?.[cascade] ?? 1) > 1) margin += CASCADE_REUSE_DRIFT * radius;

        const world = this.director.world.worldBounds;
        culler.casterPlaneCount = casterVolumePlanes(_sliceCorners, _center, radius, view.shadowCamera._node.forward,
            world.getMin(), world.getMax(), margin, culler.casterPlanes, CULL_PARAMS.MAX_CASTER_PLANES);
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
     * nominal resolution stands in. For a cascade this is the texel rate the engine snaps the
     * cascade to in shadow-renderer-directional.js.
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
        // sized exactly as a view sizes its own (see claimWordCount and MeshletWorld#rootListItems)
        if (!this._claimBits) {
            const words = claimWordCount(director.world.totalPairs, director.world.instanceCount);
            this._claimBits = new StorageBuffer(this.device, words * BYTES_PER_WORD, BUFFERUSAGE_COPY_DST);
        }
        if (!this._rootList) {
            this._rootList = new StorageBuffer(this.device,
                director.world.rootListItems * ROOT_LIST_U32S * BYTES_PER_WORD, BUFFERUSAGE_COPY_DST);
        }
        this._viewsDirty = true;
        const views = [];
        for (let c = 0; c < faces; c++) {
            const view = new MeshletShadowView(
                this.device, director.world, light, c, camera,
                {
                    cullShaders: director.cullShaders,
                    capCarry: director.takeShadowCapCarry(light.id, c),
                    sharedClaimBits: this._claimBits,
                    sharedRootList: this._rootList,
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
