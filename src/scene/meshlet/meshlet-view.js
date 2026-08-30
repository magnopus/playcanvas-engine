import {
    BUFFERUSAGE_COPY_DST, BUFFERUSAGE_COPY_SRC, BUFFER_STATIC, INDEXFORMAT_UINT32, PRIMITIVE_TRIANGLES
} from '../../platform/graphics/constants.js';
import { IndexBuffer } from '../../platform/graphics/index-buffer.js';
import { StorageBuffer } from '../../platform/graphics/storage-buffer.js';
import { WebgpuReadbackPool } from '../../platform/graphics/webgpu/webgpu-readback-pool.js';
import { Debug } from '../../core/debug.js';
import {
    CULL_PARAMS_VEC4S, MESHLET_BUCKET_COUNT, MESHLET_COUNTER, MESHLET_COUNTER_U32S, RECORD_U32S, WORK_ITEM_U32S
} from './constants.js';
import { GraphNode } from '../graph-node.js';
import { Mesh } from '../mesh.js';
import { MeshInstance } from '../mesh-instance.js';
import { FramePassMeshletCompute } from './frame-pass-meshlet-compute.js';
import { MeshletCuller } from './meshlet-culler.js';
import { RenderPassMeshletDraw } from './render-pass-meshlet-draw.js';

/**
 * @import { CameraComponent } from '../../framework/components/camera/component.js'
 * @import { ForwardRenderer } from '../renderer/forward-renderer.js'
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { MeshletWorld } from './meshlet-world.js'
 * @import { MeshletCullShaders } from './meshlet-cull-shaders.js'
 * @import { MeshletHzb } from './meshlet-hzb.js'
 */

// the draw index buffer is never smaller than one triangle, so an empty view still binds a
// valid buffer
const MIN_INDICES = 3;

// bytes per vec4 row of the cull parameter block, and per u32 of a word-addressed buffer
const BYTES_PER_VEC4 = 16;
const BYTES_PER_WORD = 4;

/**
 * The per-camera half of the meshlet pipeline: everything that depends on one camera's cull
 * result. Owns the frame working buffers (cull params, counters, work items, records, claim
 * bits), the persistent occlusion visibility bits, the GPU-written draw index buffer with its
 * mesh instances, the culling compute chains, the HZB and the draw passes. Everything else -
 * page pool, meshlet/object/material data, streaming residency, textures, the shared feedback
 * requests buffer and the budget - lives on the world and is shared by all views; request
 * marks from concurrent views union naturally (max semantics).
 *
 * @ignore
 */
class MeshletView {
    /** @type {CameraComponent|null} */
    cameraComponent;

    /** @type {MeshletWorld} */
    world;

    /** @type {MeshletHzb|null} */
    hzb = null;

    /** @type {boolean} - whether this view culls two-phase this frame. */
    useOcclusion = false;

    /** @type {boolean} - one cull pass, no occlusion test, no phase 2 (shadow views). */
    singlePhase = false;

    /** @type {boolean} - false when visBitsBuffer is the world's shared placeholder. */
    _ownsVisBits = true;

    /** @type {boolean} - false when claimBitsBuffer is borrowed from a sibling-view pool. */
    _ownsClaimBits = true;

    /** @type {boolean} - false when workItemsBuffer is borrowed from a sibling-view pool. */
    _ownsWorkItems = true;

    _countersData = new Uint32Array(MESHLET_COUNTER_U32S);

    _countersReadBusy = false;

    /** @type {{ indices: number[], records: number }|null} */
    _pendingDemand = null;

    /**
     * Last unclamped demand read back from the counters - what the cull WANTED, before any
     * capacity clamp. Kept after {@link applyPendingGrowth} consumes _pendingDemand, as the
     * pipeline's numeric oracle: a view reporting zero emitted nothing.
     *
     * @type {{ indices: number[], records: number }|null}
     */
    lastDemand = null;

    /**
     * Occupancy at which a buffer grows. Demand is read back a frame or two late, so growth
     * must lead it: waiting for an actual overflow means a clamped frame, and a clamped frame
     * is not just missing triangles - with occlusion on, the holes it leaves in the HZB are
     * recorded as disocclusion.
     *
     * @type {number}
     */
    growAt = 0.7;

    /** @type {number} - demand multiplier when growing; over-allocating beats clamping. */
    growTo = 1.6;

    /**
     * Occupancy below which the index buffer is a candidate for shrinking. The gap to
     * {@link growAt} is the hysteresis band: a shrink retargets to demand * growTo, landing at
     * ~0.62 occupancy, comfortably under the 0.7 that triggers growth again.
     *
     * @type {number}
     */
    shrinkAt = 0.35;

    /**
     * Consecutive frames demand must stay under {@link shrinkAt} before shrinking. Growth is
     * urgent - a clamped frame is a visible glitch - but shrinking never is, so it waits long
     * enough that a camera turning away from the dense part of a scene does not cost a
     * reallocation it will immediately undo.
     *
     * @type {number}
     */
    shrinkFrames = 180;

    _lowFrames = 0;

    /**
     * This view's slice of the world's shared index budget, in indices. Set by the director each
     * frame in proportion to what every view is asking for; 0 means "not yet allocated", and the
     * world's whole ceiling applies.
     *
     * @type {number}
     */
    indexShare = 0;

    /** @type {Array<object>|null} */
    _appliedMaterials = null;


    /**
     * @param {GraphicsDevice} device - The graphics device.
     * @param {MeshletWorld} world - The finalized world.
     * @param {ForwardRenderer|null} renderer - The forward renderer (for the draw passes;
     * unused by a single-phase view, which has none).
     * @param {CameraComponent|null} cameraComponent - The camera this view culls and draws
     * for, or null for a view not driven by a scene camera (shadow cascades).
     * @param {object} [options] - View options.
     * @param {{ buffer: object, words: number }|null} [options.visCarry] - Previous visibility
     * bits to GPU-copy into this view's fresh visBits (rebuild retention). Not owned.
     * @param {number[]|null} [options.capCarry] - Previous per-bucket demand-grown index
     * capacities (rebuild retention) - starting back at the initial
     * allocation would clamp draws while the demand readback is in flight, and with occlusion
     * on the missing draws punch HZB holes that phase 2 reads as disocclusion (a page-demand
     * burst).
     * @param {MeshletCullShaders|null} [options.cullShaders] - Shared compiled cull shaders.
     * @param {boolean} [options.singlePhase] - Cull in one pass with no occlusion test. Skips
     * the visibility bits, the HZB and the phase-2 half of the pipeline, and halves the mesh
     * instance count. Shadow views use this.
     * @param {boolean} [options.castShadow] - Register this view's mesh instances as shadow
     * casters rather than as directly-drawn camera geometry.
     * @param {StorageBuffer|null} [options.sharedClaimBits] - Claim bits to borrow instead of
     * allocating. Only safe between views whose cull chains are encoded strictly back to back
     * (the shadow cascades) - never with a two-phase view, whose phase 2 is encoded later in
     * the frame and still depends on the claim state phase 1 left behind.
     * @param {StorageBuffer|null} [options.sharedWorkItems] - Work items to borrow, under
     * exactly the same back-to-back restriction: a two-phase view's phase 2 re-dispatches over
     * the work items phase 1 produced, so it must own them.
     * @param {number} [options.initialIndexScale] - Fraction of the world's initial index
     * budget this view starts with. Shadow views take less: the budget is tuned for the camera,
     * and a cascade's cut is coarser, so pre-sizing them for the camera reserves hundreds of MB
     * that demand-driven growth would never ask for.
     */
    constructor(device, world, renderer, cameraComponent, options = {}) {
        const {
            visCarry = null, capCarry = null, cullShaders = null,
            singlePhase = false, castShadow = false, sharedClaimBits = null,
            sharedWorkItems = null, initialIndexScale = 1
        } = options;
        this.device = device;
        this.world = world;
        this.cameraComponent = cameraComponent;
        this.singlePhase = singlePhase;

        // one bit per instance-meshlet pair, 32 to a word; a few words minimum so tiny scenes
        // still get a real buffer
        const pairWords = Math.max(Math.ceil(world.totalPairs / 32), 4);
        this.cullParamsBuffer = new StorageBuffer(device, CULL_PARAMS_VEC4S * BYTES_PER_VEC4, BUFFERUSAGE_COPY_DST);
        this.countersBuffer = new StorageBuffer(device, MESHLET_COUNTER_U32S * BYTES_PER_WORD, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
        // demand readbacks go through a pooled staging buffer, the same way the residency's
        // request-marks readback does, rather than allocating a staging buffer per read
        this.readbackPool = new WebgpuReadbackPool(device);
        this.workItemsBuffer = sharedWorkItems ??
            new StorageBuffer(device, Math.max(world.workItemCapacity * WORK_ITEM_U32S * BYTES_PER_WORD, 16), BUFFERUSAGE_COPY_DST);
        this._ownsWorkItems = !sharedWorkItems;
        this.recordsBuffer = null; // allocated below, once recordCapacity is known
        this.claimBitsBuffer = sharedClaimBits ?? new StorageBuffer(device, pairWords * BYTES_PER_WORD, BUFFERUSAGE_COPY_DST);
        this._ownsClaimBits = !sharedClaimBits;
        // Records are demand-grown like the index buffer. Sizing them for the worst case (every
        // meshlet of every instance drawn at once) is hopeless on scattered scenes - a jungle
        // with 41k plant instances reaches 149M pairs, a 2.3 GB buffer - while a real frame's
        // LOD cut emits a few hundred thousand. counters[MESHLET_COUNTER.RECORDS] counts unclamped
        // demand, so the buffer starts small and grows to what frames actually ask for.
        this.recordCapacity = Math.min(world.recordCapacity, world.initialRecords || world.recordCapacity);
        // persistent visibility bits for two-phase occlusion (never cleared - phase 2 maintains
        // them); carried across a rebuild for the unchanged-placement prefix, or the first
        // phase-2 pass re-discovers the whole frustum and its page-demand burst evicts the hot
        // working set. A single-phase view never reads or writes them - only the phase 1/2
        // branches of the cull shader touch the binding - so it borrows the world's placeholder
        // rather than allocating one bit per instance-meshlet pair (tens of MB on the jungle).
        if (singlePhase) {
            this.visBitsBuffer = world.dummyBits;
            this._ownsVisBits = false;
        } else {
            this.visBitsBuffer = new StorageBuffer(device, pairWords * BYTES_PER_WORD, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
            this.visBitsBuffer.write(0, new Uint32Array(pairWords));
            this._ownsVisBits = true;
            if (visCarry && visCarry.words > 0) {
                const words = Math.min(visCarry.words, pairWords, Math.floor(visCarry.buffer.byteSize / BYTES_PER_WORD));
                device.getCommandEncoder().copyBufferToBuffer(visCarry.buffer.impl.buffer, 0, this.visBitsBuffer.impl.buffer, 0, words * BYTES_PER_WORD);
            }
        }

        // this view's slice of the demand-grown draw index space (carried across a rebuild,
        // ceilinged at the worst case - every meshlet drawn - so carries can never compound
        // past what the scene could possibly want)
        this.indexCapacity = [];
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            this.indexCapacity.push(Math.min(
                Math.max(Math.floor(world.indexCapacity[b] * initialIndexScale), capCarry?.[b] ?? 0),
                world.indexWorst[b]));
        }
        this._clampToCeiling(this.indexCapacity);
        this._allocIndexBuffer();

        const mesh = new Mesh(device);
        mesh.indexBuffer[0] = this.indexBuffer;
        mesh.primitive[0] = { type: PRIMITIVE_TRIANGLES, base: 0, count: 0, indexed: true };
        this._mesh = mesh;

        // One mesh instance per bucket per phase - phase 1 first, then phase 2 (a single-phase
        // view has no phase 2, so one set). Materials are shared across views; the per-view
        // records buffer rides as an instance parameter.
        this._node = new GraphNode('MeshletView');
        this.meshInstances = [];
        const instanceCount = (singlePhase ? 1 : 2) * MESHLET_BUCKET_COUNT;
        for (let k = 0; k < instanceCount; k++) {
            const mi = new MeshInstance(mesh, world.bucketMaterials[k % MESHLET_BUCKET_COUNT]);
            mi.node = this._node;
            // A caster must keep cull = true: ShadowRenderer short-circuits on !mi.cull and
            // never calls isVisibleFunc, which would route one caster into every cascade.
            mi.cull = castShadow;
            mi.castShadow = castShadow;
            this.meshInstances.push(mi);
        }
        this._appliedMaterials = world.bucketMaterials;

        this.recordsBuffer = new StorageBuffer(device, Math.max(this.recordCapacity * RECORD_U32S * BYTES_PER_WORD, 16), BUFFERUSAGE_COPY_DST);
        this.meshInstances.forEach(mi => mi.setParameter('records', this.recordsBuffer));

        this.culler = new MeshletCuller(device, world, this, cullShaders);

        // a single-phase view is drawn by whoever owns its instances (the shadow renderer's
        // caster submission); it has no HZB, no phase-2 cull and no draw passes of its own
        if (!singlePhase) {
            this.drawPass1 = new RenderPassMeshletDraw(device, renderer, 'MeshletDrawPhase1');
            this.drawPass2 = new RenderPassMeshletDraw(device, renderer, 'MeshletDrawPhase2');
            this.hzbMipsPass = new FramePassMeshletCompute(device, 'MeshletHzbMips', () => this.hzb.buildMips());
            this.cullPhase2Pass = new FramePassMeshletCompute(device, 'MeshletCullPhase2', () => this.culler.dispatchPhase2());
        }
    }

    /**
     * Reallocates this view's record buffer to the observed demand. The contents are
     * GPU-written every frame, so nothing copies.
     *
     * @param {number} wanted - Peak record demand observed (unclamped, counters[MESHLET_COUNTER.RECORDS]).
     * @returns {boolean} True when the buffer was replaced.
     */
    growRecordsBuffer(wanted) {
        const target = Math.min(Math.ceil(wanted * this.growTo), this.world.recordCapacity);
        if (target <= this.recordCapacity) return false;
        this.recordCapacity = target;
        const old = this.recordsBuffer;
        this.recordsBuffer = new StorageBuffer(this.device, Math.max(target * RECORD_U32S * BYTES_PER_WORD, 16), BUFFERUSAGE_COPY_DST);
        this.meshInstances.forEach(mi => mi.setParameter('records', this.recordsBuffer));
        old.destroy();
        return true;
    }

    destroy() {
        this.culler?.destroy();
        this.culler = null;
        this.hzb?.destroy();
        this.hzb = null;
        this.readbackPool?.destroy();
        this.cullParamsBuffer?.destroy();
        this.countersBuffer?.destroy();
        if (this._ownsWorkItems) {
            this.workItemsBuffer?.destroy();
        }
        this.workItemsBuffer = null;
        this.recordsBuffer?.destroy();
        if (this._ownsClaimBits) {
            this.claimBitsBuffer?.destroy();
        }
        this.claimBitsBuffer = null;
        if (this._ownsVisBits) {
            this.visBitsBuffer?.destroy();
        }
        this.visBitsBuffer = null;
        // the shared mesh owns the index buffer - the last mesh instance destroy releases both
        this.meshInstances.forEach(mi => mi.destroy());
        this.meshInstances = [];
        this.indexBuffer = null;
        this._mesh = null;
    }

    /**
     * Applies the world's current bucket materials (lit or debug colour mode) to this view's
     * mesh instances. Cheap identity check - called every frame.
     */
    syncMaterials() {
        const materials = this.world.bucketMaterials;
        if (materials === this._appliedMaterials) return;
        this._appliedMaterials = materials;
        this.meshInstances.forEach((mi, k) => {
            mi.material = materials[k % MESHLET_BUCKET_COUNT];
        });
    }

    /**
     * Scales the per-bucket capacities down proportionally when their total exceeds what the
     * device (or the world's own {@link MeshletWorld#maxIndices}) allows. Growth is driven by
     * observed demand and ceilinged at the worst case, but on a scattered scene the worst case
     * is billions of indices - so without this the allocation eventually throws in the middle
     * of a frame's encoding, and every pass after it fails validation.
     *
     * @param {number[]} capacity - Per-bucket capacities, modified in place.
     * @returns {boolean} True when a clamp was applied.
     * @private
     */
    _clampToCeiling(capacity) {
        const ceiling = this.indexShare > 0 ? this.indexShare : this.world.indexCeiling;
        const total = capacity.reduce((a, b) => a + b, 0);
        if (!(total > ceiling)) return false;
        const s = ceiling / total;
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) capacity[b] = Math.floor(capacity[b] * s);
        Debug.warnOnce(`MeshletView: index demand (${total}) exceeds the ${ceiling}-index ceiling; the cut will be clamped. Lower dagPixelThreshold pressure or set world.maxIndices.`);
        return true;
    }

    /**
     * Allocates this view's index buffer at the current capacity. No initial data: every index
     * is written by the index-write pass before any draw reads it (finalizeArgs bounds each
     * draw to the committed end), so uploading a zeroed copy of a buffer this size is pure
     * cost - and IndexBuffer keeps no CPU shadow unless one is asked for.
     *
     * @private
     */
    _allocIndexBuffer() {
        const total = Math.max(this.indexTotal(), MIN_INDICES);
        this.indexBuffer = new IndexBuffer(this.device, INDEXFORMAT_UINT32, total, BUFFER_STATIC,
            undefined, { storage: true });
    }

    /**
     * This view's index demand as a fraction of what its budget can ever hold. Above 1 the cull
     * clamps, and the clusters that survive are decided by an atomicAdd race - so they differ
     * every frame, which is seen as wild flicker. The caller feeds this to the budget manager,
     * which coarsens the DAG cut before it gets there.
     *
     * @returns {number} Demand / ceiling, or 0 when unbudgeted.
     */
    indexPressure() {
        const d = this.lastDemand;
        if (!d) return 0;
        const ceiling = this.indexShare > 0 ? this.indexShare : this.world.indexCeiling;
        if (!(ceiling > 0) || !Number.isFinite(ceiling)) return 0;
        return d.indices.reduce((a, b) => a + b, 0) / ceiling;
    }

    /** @returns {number} This view's total unclamped index demand, or 0 before the first readback. */
    indexDemand() {
        return this.lastDemand ? this.lastDemand.indices.reduce((a, b) => a + b, 0) : 0;
    }

    /** @returns {number} Total index capacity across the buckets. */
    indexTotal() {
        let n = 0;
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) n += this.indexCapacity[b];
        return n;
    }

    /**
     * Reallocates this view's draw index buffer to the observed demand (see the world's
     * initialIndices/maxIndices). Contents are GPU-written every frame, so nothing copies.
     *
     * @param {number[]} wanted - Peak index demand per bucket (unclamped).
     * @returns {boolean} True when the buffer was replaced.
     */
    growIndexBuffer(wanted) {
        const world = this.world;
        const next = [];
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            next.push(Math.max(this.indexCapacity[b], wanted[b]));
        }
        if (world.maxIndices > 0) {
            const total = next.reduce((a, b) => a + b, 0);
            if (total > world.maxIndices) {
                const s = world.maxIndices / total;
                for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) next[b] = Math.floor(next[b] * s);
            }
        }
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            next[b] = Math.min(next[b], world.indexWorst[b]);
        }
        this._clampToCeiling(next);
        let grew = false;
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            if (next[b] > this.indexCapacity[b]) grew = true;
        }
        if (!grew) {
            return false;
        }
        const previous = this.indexCapacity.slice();
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            this.indexCapacity[b] = Math.max(next[b], this.indexCapacity[b]);
        }
        // the ceiling applies to the total, and taking the per-bucket max above can push it
        // back over
        this._clampToCeiling(this.indexCapacity);

        const old = this.indexBuffer;
        try {
            this._allocIndexBuffer();
        } catch (e) {
            // out of memory despite the ceiling. Keep the buffer we have - a clamped cut is a
            // visual glitch, whereas throwing here aborts the frame mid-encode and every pass
            // after it fails validation
            Debug.error(`MeshletView: index buffer allocation failed at ${this.indexTotal()} indices; keeping the previous allocation.`, e);
            this.indexCapacity = previous;
            this.indexBuffer = old;
            return false;
        }
        this._mesh.indexBuffer[0] = this.indexBuffer;
        // previous frames' submissions already hold their own reference; this frame onwards
        // encodes against the new buffer only
        old.destroy();
        return true;
    }

    /**
     * Releases index capacity a view has stopped using. Without this the buffer only ever
     * ratchets to the session's high-water mark: a camera that once looked at the dense part of
     * a scene, or a rebuild that carried a grown capacity forward, holds that allocation for
     * good - on the jungle, 357 MB against a cut wanting 43 MB.
     *
     * @param {number[]} demand - Per-bucket demand this frame (unclamped).
     * @param {number[]} target - Per-bucket demand * growTo, the size a fresh grow would pick.
     * @returns {boolean} True when the buffer was replaced.
     */
    shrinkIndexBuffer(demand, target) {
        const world = this.world;
        const total = this.indexTotal();
        const wanted = demand.reduce((a, b) => a + b, 0);
        if (total <= 0 || wanted > total * this.shrinkAt) {
            this._lowFrames = 0;
            return false;
        }
        if (++this._lowFrames < this.shrinkFrames) return false;
        this._lowFrames = 0;

        // never drop below what the world hands a fresh view - that is the size chosen to avoid
        // clamped frames on the way back up
        const next = [];
        let changed = false;
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            next.push(Math.min(Math.max(target[b], world.indexCapacity[b]), this.indexCapacity[b]));
            if (next[b] < this.indexCapacity[b]) changed = true;
        }
        if (!changed) return false;

        const previous = this.indexCapacity;
        this.indexCapacity = next;
        const old = this.indexBuffer;
        try {
            this._allocIndexBuffer();
        } catch (e) {
            this.indexCapacity = previous;
            return false;
        }
        this._mesh.indexBuffer[0] = this.indexBuffer;
        old.destroy();
        return true;
    }

    /**
     * Reads back this view's unclamped index-demand counters (end-of-frame copy) and grows the
     * index buffer when a frame wanted more than the current allocation.
     */
    monitorIndexDemand() {
        if (this._countersReadBusy) return;
        this._countersReadBusy = true;
        this.readbackPool.read(this.countersBuffer, 0, MESHLET_COUNTER_U32S * BYTES_PER_WORD, this._countersData).then((data) => {
            this._countersReadBusy = false;
            if (!this.culler) return; // destroyed while in flight
            // Only RECORD the demand here. This callback resolves at an arbitrary point in
            // the frame, and swapping a buffer mid-frame is a correctness bug: the cull chain
            // may already be encoded against the old buffer while the draw reads the new,
            // never-written one - which rasterises garbage indices as stretched degenerate
            // triangles. {@link applyPendingGrowth} does the swap before anything is encoded.
            const indices = [];
            for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) indices.push(data[MESHLET_COUNTER.DEMAND_BASE + b]);
            this._pendingDemand = { indices, records: data[MESHLET_COUNTER.RECORDS] };
            this.lastDemand = this._pendingDemand;
        }).catch(() => {
            this._countersReadBusy = false;
        });
    }

    /**
     * Applies any growth the demand readback asked for. Must be called at the top of the
     * frame, before this view's culling is encoded, so the whole frame sees one set of
     * buffers.
     */
    applyPendingGrowth() {
        const d = this._pendingDemand;
        if (!d || !this.culler) return;
        this._pendingDemand = null;

        // grow at `growAt` occupancy, before demand overflows: clamped draws are not only a
        // visual glitch - with occlusion on, the missing draws punch HZB holes that
        // phase 2 records as disocclusion, poisoning the visibility bits with a false
        // working set the streamer then fetches
        let wantsGrowth = false;
        const target = [];
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            if (d.indices[b] > this.indexCapacity[b] * this.growAt) wantsGrowth = true;
            target.push(Math.ceil(d.indices[b] * this.growTo));
        }
        if (wantsGrowth) {
            this._lowFrames = 0;
            if (this.growIndexBuffer(target)) {
                this.culler.bindIndexState(this);
            }
        } else if (this.shrinkIndexBuffer(d.indices, target)) {
            this.culler.bindIndexState(this);
        }
        // records overflow the same way - dropped records are missing draws, which the
        // HZB then reads as disocclusion
        if (d.records > this.recordCapacity * this.growAt && this.growRecordsBuffer(d.records)) {
            this.culler.bindRecordState(this);
        }
    }
}

export { MeshletView };
