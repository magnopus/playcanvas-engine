import { MESHLET_BUCKET_COUNT } from './constants.js';
import { MeshletView } from './meshlet-view.js';

/**
 * @import { BoundingBox } from '../../core/shape/bounding-box.js'
 * @import { Camera } from '../camera.js'
 * @import { ForwardRenderer } from '../renderer/forward-renderer.js'
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { Light } from '../light.js'
 * @import { MeshletCullShaders } from './meshlet-cull-shaders.js'
 * @import { MeshletWorld } from './meshlet-world.js'
 * @import { StorageBuffer } from '../../platform/graphics/storage-buffer.js'
 */

/**
 * One meshlet view per (light, shadow face): a single-phase cull producing an index buffer that
 * the engine's own shadow pass draws through the standard caster path. A face is a directional
 * cascade, a spot light's single frustum, or one of an omni's six cube faces.
 *
 * Each face is a separate view rather than a filter on the camera view because its LOD cut is
 * genuinely different - the cut is driven by that face's shadow-map texel rate, and a far
 * cascade or a distant cube face wants a far coarser cut than a near one. It also runs
 * single-phase: there is no shadow-map HZB to test against, so no visibility bits and no phase 2.
 *
 * The instances are registered as layer shadow casters ({@link MeshletShadowRenderer}) and
 * routed to their own face by {@link MeshInstance#isVisibleFunc}.
 *
 * @ignore
 */
class MeshletShadowView extends MeshletView {
    /** @type {Light} */
    light;

    /** @type {number} - shadow face index within the light (cascade, or cube face). */
    face;

    /**
     * The scene camera this light's shadow data belongs to, or null. Directional shadow render
     * data is per camera; local lights share one set across cameras.
     *
     * @type {Camera|null}
     */
    sceneCamera;

    /**
     * @param {GraphicsDevice} device - The graphics device.
     * @param {MeshletWorld} world - The finalized world.
     * @param {ForwardRenderer|null} renderer - The forward renderer.
     * @param {Light} light - The directional light.
     * @param {number} face - Shadow face index within the light.
     * @param {Camera|null} sceneCamera - The camera whose shadow render data this view culls
     * for; null for local lights, whose render data is camera-independent.
     * @param {object} [options] - Options.
     * @param {MeshletCullShaders|null} [options.cullShaders] - Shared compiled cull shaders.
     * @param {number[]|null} [options.capCarry] - Previous per-bucket demand-grown index
     * capacities (rebuild retention).
     * @param {StorageBuffer|null} [options.sharedClaimBits] - Claim bits shared with the sibling faces.
     * @param {StorageBuffer|null} [options.sharedWorkItems] - Work items shared with the sibling faces.
     * @param {number} [options.initialIndexScale] - Fraction of the world's initial index budget.
     */
    constructor(device, world, renderer, light, face, sceneCamera, options = {}) {
        super(device, world, renderer, null, {
            capCarry: options.capCarry ?? null,
            cullShaders: options.cullShaders ?? null,
            sharedClaimBits: options.sharedClaimBits ?? null,
            sharedWorkItems: options.sharedWorkItems ?? null,
            initialIndexScale: options.initialIndexScale ?? 1,
            singlePhase: true,
            castShadow: true
        });

        this.light = light;
        this.face = face;
        this.sceneCamera = sceneCamera;

        // Route each caster to exactly one face. The shadow camera object is resolved lazily
        // rather than captured: getRenderData is idempotent but the render data does not exist
        // until the light has been prepared for this camera.
        this.meshInstances.forEach((mi) => {
            mi.isVisibleFunc = (/** @type {Camera} */ camera) => camera === this.shadowCamera;
        });

        this.syncMaterials();
    }

    /**
     * This face's shadow camera, or null before the light has render data. Resolved on demand -
     * the engine recreates render data when the camera set changes.
     *
     * @type {Camera|null}
     */
    get shadowCamera() {
        return this.renderData?.shadowCamera ?? null;
    }

    /**
     * This face's light render data (shadow camera, viewport, visible casters) - a
     * LightRenderData, which the engine does not export, so it is typed loosely here.
     *
     * @type {object|null}
     */
    get renderData() {
        return this.light.getRenderData(this.sceneCamera, this.face) ?? null;
    }

    /**
     * Sets the caster bounds used by the engine to fit each cascade's depth range.
     *
     * @param {BoundingBox} aabb - World-space bounds of all meshlet geometry.
     */
    setCasterBounds(aabb) {
        this.meshInstances.forEach(mi => mi.setCustomAabb(aabb));
    }

    /**
     * Always the lit materials, whatever debug colour mode the world is in - the debug materials
     * are unlit and have no shadow variant, so using them would drop every meshlet shadow.
     *
     * @override
     */
    syncMaterials() {
        const materials = this.world.litMaterials;
        if (materials === this._appliedMaterials) return;
        this._appliedMaterials = materials;
        this.meshInstances.forEach((mi, k) => {
            mi.material = materials[k % MESHLET_BUCKET_COUNT];
        });
    }
}

export { MeshletShadowView };
