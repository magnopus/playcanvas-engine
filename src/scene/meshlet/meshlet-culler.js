import { Compute } from '../../platform/graphics/compute.js';
import {
    CULL_PARAMS, CULL_PARAMS_VEC4S, MESHLET_BUCKET_COUNT, MESHLET_INSTANCE_CULL_WORKGROUP
} from './constants.js';
import { MeshletCullShaders } from './meshlet-cull-shaders.js';

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { MeshletHzb } from './meshlet-hzb.js'
 * @import { MeshletWorld } from './meshlet-world.js'
 * @import { MeshletView } from './meshlet-view.js'
 * @import { Vec3 } from '../../core/math/vec3.js'
 */


/**
 * Owns and dispatches the meshlet culling compute chains. Single-phase mode runs one chain per
 * frame (no occlusion). Two-phase mode runs the phase-1 chain up front (drawing last frame's
 * visible set), then after the phase-1 draws have built the HZB, the phase-2 chain re-culls and
 * emits only the newly visible remainder, updating the persistent visibility bits.
 *
 * @ignore
 */
/**
 * Binds the per-bucket index capacities. Both the cull pass (to clamp reservations and to place
 * each bucket's range) and finalizeArgs (to compute each draw's firstIndex) read them; unused
 * ones are simply not referenced by that shader.
 *
 * @param {Compute} compute - The compute instance.
 * @param {MeshletView} view - The view holding the capacities.
 */
const bindCapacities = (compute, view) => {
    for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
        compute.setParameter(`indexCapacity${b}`, view.indexCapacity[b]);
    }
};

class MeshletCuller {
    /** @type {GraphicsDevice} */
    device;

    /** @type {MeshletWorld} */
    world;

    /** @type {Float32Array} - the per-view parameter rows (CULL_PARAMS.*), uploaded each frame. */
    cullParams = new Float32Array(CULL_PARAMS_VEC4S * 4);

    dagPixelThreshold = 1;

    /** @type {number} - memory-pressure multiplier from the budget manager. */
    pressureScale = 1;

    /**
     * Light-space texel rate for an orthographic view, or 0 for a perspective one. Selects the
     * distance-independent LOD projection in the cull shader - see projectError there. Shadow
     * cascades MUST set this: the directional shadow camera is pushed ~1e6 units back, which
     * collapses the perspective projection to zero error and culls everything.
     *
     * @type {number}
     */
    orthoScale = 0;

    /** @type {number} - CULL_FLAG_* bits (CULL_FLAG_NO_TEXEL_RATE for orthographic views). */
    cullFlags = 0;

    /** @type {Vec3|null} - view direction, replaces apex - camPos in the ortho cone cull. */
    viewDir = null;

    /** @type {boolean} - two-phase occlusion (requires an HZB). */
    twoPhase = false;


    /**
     * Split the culling chains into separate submissions around GPU-written indirect dispatch
     * args. On some drivers (observed on Windows/D3D12, see the same workaround in
     * gsplat-hybrid-renderer.js) args written by a compute pass are intermittently not visible
     * to dispatchWorkgroupsIndirect recorded in the same command buffer. Off by default; enable
     * on affected platforms.
     *
     * @type {boolean}
     */
    forceSubmitBoundaries = false;

    /**
     * @param {GraphicsDevice} device - The graphics device.
     * @param {MeshletWorld} world - The finalized world.
     * @param {MeshletView} view - The view this culler produces a frame for.
     * @param {MeshletCullShaders|null} [cullShaders] - Shared compiled cull shaders. When
     * omitted a private set is compiled and owned by this culler.
     */
    constructor(device, world, view, cullShaders = null) {
        this.device = device;
        this.world = world;
        this.view = view;

        this.shaders = cullShaders ?? new MeshletCullShaders(device);
        this._ownsShaders = !cullShaders;
        this.dummyHzb = this.shaders.dummyHzb;

        const makeCompute = name => new Compute(device, this.shaders[name], `Meshlet${name}`);

        this.instanceCull = makeCompute('instanceCull');
        this.dispatchArgs = makeCompute('dispatchArgs');
        this.meshletCull = makeCompute('meshletCull');
        this.finalizeArgs = makeCompute('finalizeArgs');
        this.indexWrite = makeCompute('indexWrite');
        this.resetPhase2 = makeCompute('resetPhase2');

        // the phase-2 chain uses its OWN compute instances (sharing the shaders): uniform writes
        // land on the queue before the whole frame's command buffer, so a single instance
        // dispatched twice per frame would give both dispatches the later uniform values
        this.meshletCullPhase2 = new Compute(device, this.meshletCull.shader, 'MeshletMeshletCullP2');
        this.finalizeArgsPhase2 = new Compute(device, this.finalizeArgs.shader, 'MeshletFinalizeArgsP2');
        this.indexWritePhase2 = new Compute(device, this.indexWrite.shader, 'MeshletIndexWriteP2');

        // static parameters - the same bindings for a stage's phase-1 and phase-2 instances
        const instanceCull = this.instanceCull;
        instanceCull.setParameter('instanceCount', world.instanceCount);
        instanceCull.setParameter('workItemCapacity', world.workItemCapacity);
        instanceCull.setParameter('objectData', world.objectDataBuffer);
        instanceCull.setParameter('cullParams', view.cullParamsBuffer);
        instanceCull.setParameter('counters', view.countersBuffer);
        instanceCull.setParameter('workItems', view.workItemsBuffer);

        this.dispatchArgs.setParameter('workItemCapacity', world.workItemCapacity);
        this.dispatchArgs.setParameter('counters', view.countersBuffer);

        for (const meshletCull of [this.meshletCull, this.meshletCullPhase2]) {
            meshletCull.setParameter('workItemCapacity', world.workItemCapacity);
            meshletCull.setParameter('recordCapacity', view.recordCapacity);
            bindCapacities(meshletCull, view);
            meshletCull.setParameter('objectData', world.objectDataBuffer);
            meshletCull.setParameter('meshletData', world.meshletDataBuffer);
            meshletCull.setParameter('cullParams', view.cullParamsBuffer);
            meshletCull.setParameter('workItems', view.workItemsBuffer);
            meshletCull.setParameter('counters', view.countersBuffer);
            meshletCull.setParameter('records', view.recordsBuffer);
            meshletCull.setParameter('residency', world.residencyBuffer);
            meshletCull.setParameter('requests', world.requestsBuffer);
            meshletCull.setParameter('claimBits', view.claimBitsBuffer);
            meshletCull.setParameter('visBits', view.visBitsBuffer);
            meshletCull.setParameter('hzbTexture', this.dummyHzb);
        }
        this.meshletCullPhase2.setParameter('phase', 2);

        for (const finalizeArgs of [this.finalizeArgs, this.finalizeArgsPhase2]) {
            finalizeArgs.setParameter('recordCapacity', view.recordCapacity);
            bindCapacities(finalizeArgs, view);
            finalizeArgs.setParameter('counters', view.countersBuffer);
        }

        for (const indexWrite of [this.indexWrite, this.indexWritePhase2]) {
            indexWrite.setParameter('recordCapacity', view.recordCapacity);
            indexWrite.setParameter('pageSizeWords', world.pageSizeBytes / 4);
            indexWrite.setParameter('objectData', world.objectDataBuffer);
            indexWrite.setParameter('meshletData', world.meshletDataBuffer);
            indexWrite.setParameter('pagePool', world.pagePool);
            indexWrite.setParameter('residency', world.residencyBuffer);
            indexWrite.setParameter('counters', view.countersBuffer);
            indexWrite.setParameter('records', view.recordsBuffer);
            indexWrite.setParameter('drawIndices', view.indexBuffer);
        }

        this.resetPhase2.setParameter('counters', view.countersBuffer);
    }

    /**
     * Re-applies the index capacities and the index buffer binding after
     * {@link MeshletView#growIndexBuffer}. Call before this frame's chains are encoded.
     *
     * @param {MeshletView} view - The view.
     */
    bindIndexState(view) {
        for (const compute of [this.meshletCull, this.meshletCullPhase2, this.finalizeArgs, this.finalizeArgsPhase2]) {
            bindCapacities(compute, view);
        }
        for (const compute of [this.indexWrite, this.indexWritePhase2]) {
            compute.setParameter('drawIndices', view.indexBuffer);
        }
    }

    /**
     * Re-applies the record capacity and buffer binding after
     * {@link MeshletView#growRecordsBuffer}.
     *
     * @param {MeshletView} view - The view.
     */
    bindRecordState(view) {
        for (const compute of [this.meshletCull, this.meshletCullPhase2, this.indexWrite, this.indexWritePhase2]) {
            compute.setParameter('records', view.recordsBuffer);
            compute.setParameter('recordCapacity', view.recordCapacity);
        }
        for (const compute of [this.finalizeArgs, this.finalizeArgsPhase2]) {
            compute.setParameter('recordCapacity', view.recordCapacity);
        }
    }

    destroy() {
        // the shaders and the placeholder HZB are shared across cullers unless this one
        // compiled its own set
        if (this._ownsShaders) {
            this.shaders.destroy();
        }
        this.shaders = null;
        this.dummyHzb = null;
    }

    /**
     * Sets up frame parameters and dispatches the phase-1 (or single-phase) chain. Encoded ahead
     * of the frame graph, so it executes before any of the frame's render passes.
     *
     * @param {Float32Array} frustumPlanes - 24 floats, world frustum planes (normal, distance).
     * @param {import('../../core/math/vec3.js').Vec3} cameraPos - Camera world position.
     * @param {number} projScale - viewportHeight / (2 * tan(fovY / 2)).
     * @param {Float32Array|number[]} viewProj - 16 floats, column-major view-projection matrix.
     * @param {MeshletHzb|null} hzb - The HZB (two-phase mode), or null.
     */
    beginFrame(frustumPlanes, cameraPos, projScale, viewProj, hzb) {
        const { device, world, view } = this;
        const twoPhase = this.twoPhase && !!hzb?.texture;

        // frame params, one vec4 row each (CULL_PARAMS.*)
        const params = this.cullParams;
        const row = index => index * 4;
        params.set(frustumPlanes, row(CULL_PARAMS.PLANES));
        params[row(CULL_PARAMS.CAMERA) + 0] = cameraPos.x;
        params[row(CULL_PARAMS.CAMERA) + 1] = cameraPos.y;
        params[row(CULL_PARAMS.CAMERA) + 2] = cameraPos.z;
        params[row(CULL_PARAMS.CAMERA) + 3] = projScale;
        params[row(CULL_PARAMS.LOD) + 0] = this.dagPixelThreshold * this.pressureScale;
        params[row(CULL_PARAMS.LOD) + 1] = hzb?.width ?? 1;
        params[row(CULL_PARAMS.LOD) + 2] = hzb?.height ?? 1;
        params[row(CULL_PARAMS.LOD) + 3] = hzb?.mipCount ?? 1;
        params.set(viewProj, row(CULL_PARAMS.VIEW_PROJ));
        // base of the texture-mip feedback marks inside the requests buffer (= page count)
        params[row(CULL_PARAMS.STREAMING) + 0] = world.totalPages;
        params[row(CULL_PARAMS.STREAMING) + 1] = this.orthoScale;
        params[row(CULL_PARAMS.STREAMING) + 2] = this.cullFlags;
        params[row(CULL_PARAMS.VIEW_DIR) + 0] = this.viewDir?.x ?? 0;
        params[row(CULL_PARAMS.VIEW_DIR) + 1] = this.viewDir?.y ?? 0;
        params[row(CULL_PARAMS.VIEW_DIR) + 2] = this.viewDir?.z ?? 1;

        // queue.writeBuffer executes before this frame's command buffer, so these clears land
        // ahead of every dispatch below. The requests buffer is NOT cleared here - the residency
        // manager clears it in-encoder after its readback copy, or the copy would see zeros.
        // The counters buffer clears in-encoder for the same reason: the director's index-demand
        // readback copy is encoded earlier this frame and must see last frame's values.
        view.cullParamsBuffer.write(0, params);
        const encoder = device.getCommandEncoder();
        encoder.clearBuffer(view.countersBuffer.impl.buffer, 0, view.countersBuffer.byteSize);
        // claim bits clear on the GPU: the CPU mirror is one bit per instance-meshlet pair,
        // which reaches tens of MB on scattered scenes - far too much to upload every frame
        encoder.clearBuffer(view.claimBitsBuffer.impl.buffer, 0, view.claimBitsBuffer.byteSize);

        // per-frame indirect slots
        // one draw slot per bucket per phase: [phase 1 buckets..., phase 2 buckets...]
        this._drawSlots = [];
        const slotCount = (view.singlePhase ? 1 : 2) * MESHLET_BUCKET_COUNT;
        for (let i = 0; i < slotCount; i++) {
            this._drawSlots.push(device.getIndirectDrawSlot());
        }
        this._dispatchSlotCull = device.getIndirectDispatchSlot();
        this._dispatchSlotWrite = device.getIndirectDispatchSlot();

        this._hzbTexture = twoPhase ? hzb.texture : null;
        this.meshletCull.setParameter('hzbTexture', this._hzbTexture ?? this.dummyHzb);

        this.dispatchArgs.setParameter('dispatchSlot', this._dispatchSlotCull);
        this.dispatchArgs.setParameter('indirectDispatch', device.indirectDispatchBuffer);

        this.finalizeArgs.setParameter('indirectDraw', device.indirectDrawBuffer);
        this.finalizeArgs.setParameter('indirectDispatch', device.indirectDispatchBuffer);
        this.finalizeArgs.setParameter('dispatchSlot', this._dispatchSlotWrite);

        // phase-1 (or single-phase) chain
        this.meshletCull.setParameter('phase', twoPhase ? 1 : 0);
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            this.finalizeArgs.setParameter(`drawSlot${b}`, this._drawSlots[b]);
        }

        this.instanceCull.setupDispatch(Math.ceil(world.instanceCount / MESHLET_INSTANCE_CULL_WORKGROUP));
        this.dispatchArgs.setupDispatch(1);
        this.meshletCull.setupIndirectDispatch(this._dispatchSlotCull);
        this.finalizeArgs.setupDispatch(1);
        this.indexWrite.setupIndirectDispatch(this._dispatchSlotWrite);

        if (this.forceSubmitBoundaries) {
            device.computeDispatch([this.instanceCull, this.dispatchArgs], 'MeshletCullPhase1a');
            device.submit();
            device.computeDispatch([this.meshletCull, this.finalizeArgs], 'MeshletCullPhase1b');
            device.submit();
            device.computeDispatch([this.indexWrite], 'MeshletCullPhase1c');
        } else {
            device.computeDispatch([
                this.instanceCull, this.dispatchArgs, this.meshletCull, this.finalizeArgs, this.indexWrite
            ], 'MeshletCullPhase1');
        }

        for (let i = 0; i < view.meshInstances.length; i++) {
            view.meshInstances[i].setIndirect(null, this._drawSlots[i]);
        }
    }

    /**
     * Dispatches the phase-2 chain: reset counters, re-cull with the HZB test (emitting only
     * newly visible clusters, updating visibility bits), rebuild args and indices. Must be
     * encoded after the phase-1 draws and the HZB build.
     */
    dispatchPhase2() {
        const { device } = this;

        this.meshletCullPhase2.setParameter('hzbTexture', this._hzbTexture ?? this.dummyHzb);
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            this.finalizeArgsPhase2.setParameter(`drawSlot${b}`, this._drawSlots[MESHLET_BUCKET_COUNT + b]);
        }
        this.finalizeArgsPhase2.setParameter('indirectDraw', device.indirectDrawBuffer);
        this.finalizeArgsPhase2.setParameter('indirectDispatch', device.indirectDispatchBuffer);
        this.finalizeArgsPhase2.setParameter('dispatchSlot', this._dispatchSlotWrite);

        this.resetPhase2.setupDispatch(1);
        this.meshletCullPhase2.setupIndirectDispatch(this._dispatchSlotCull);
        this.finalizeArgsPhase2.setupDispatch(1);
        this.indexWritePhase2.setupIndirectDispatch(this._dispatchSlotWrite);

        if (this.forceSubmitBoundaries) {
            device.computeDispatch([this.resetPhase2, this.meshletCullPhase2, this.finalizeArgsPhase2], 'MeshletCullPhase2a');
            device.submit();
            device.computeDispatch([this.indexWritePhase2], 'MeshletCullPhase2b');
        } else {
            device.computeDispatch([
                this.resetPhase2, this.meshletCullPhase2, this.finalizeArgsPhase2, this.indexWritePhase2
            ], 'MeshletCullPhase2');
        }
    }
}

export { MeshletCuller };
