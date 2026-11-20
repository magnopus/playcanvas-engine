import {
    CULL_PARAMS, CUT_BUDGET, MESHLET_COUNTER, MESHLET_CUT_INVALID, MESHLET_CUT_ROOT_FLAG, MESHLET_LIST_WORKGROUP,
    MESHLET_NO_PARENT
} from '../constants.js';
import {
    meshletBucketWGSL, meshletCullParamsWGSL, meshletCullVolumeWGSL, meshletDispatchArgsWGSL, meshletDispatchGroupWGSL, meshletRequestPagesWGSL
} from './meshlet-cull-common-wgsl.js';
import { meshletStructsWGSL, meshletDataWGSL, meshletObjectDataWGSL } from './meshlet-page-wgsl.js';

/**
 * Shared declarations and the per-task body of the cut stage: `cutTask(task)` decides one
 * replacement group (an (instance, group row) pair). The entry points below decide which tasks run.
 *
 * @param {string} entry - Extra declarations and the `main` entry point.
 * @returns {string} The shader source.
 */
const cutShader = entry => /* wgsl */ `
    ${meshletStructsWGSL}
    uniform totalPairs: u32;
    // bit base of the off-frustum refinement flags, after the pair and admission bits
    uniform freeBase: u32;
    uniform recordCapacity: u32;
    ${meshletBucketWGSL}
    ${meshletDataWGSL}
    ${meshletObjectDataWGSL}
    var<storage, read> cutGroups: array<u32>;
    ${meshletCullParamsWGSL}
    var<storage, read> residency: array<u32>;
    var<storage, read_write> requests: array<atomic<u32>>;
    var<storage, read_write> claimBits: array<atomic<u32>>;
    var<storage, read_write> counters: array<atomic<u32>>;

    fn bitSet(bit: u32) -> bool {
        return (atomicLoad(&claimBits[bit >> 5u]) & (1u << (bit & 31u))) != 0u;
    }
    fn markBit(bit: u32) {
        atomicOr(&claimBits[bit >> 5u], 1u << (bit & 31u));
    }
    ${meshletCullVolumeWGSL}
    ${meshletRequestPagesWGSL}
    ${meshletDispatchGroupWGSL}

    // Decides one replacement group: true when its bit was marked - refined, or flagged as
    // refined off-frustum - so the tasks that wait on it may now run.
    fn cutTask(task: vec2u) -> bool {
        let instance = task.x;
        let admissionBit = uniform.totalPairs + instance;
        if (!bitSet(admissionBit)) { return false; }
        let offset = task.y;
        let object = objectData[instance];
        let firstMeshlet = meshletPrimitiveData(instance).firstMeshlet;
        var unchargedIndices = 0u;
        var unchargedRecords = 0u;
        let first = cutGroups[offset];
        let parentCount = cutGroups[offset + 1u];
        let pageStart = cutGroups[offset + 2u];
        let pageCount = cutGroups[offset + 3u];
        // Every coarse member must belong to the active cut. A DAG replacement may depend on
        // several different ancestor groups, all of which must have refined. Members under an
        // off-frustum parent were never charged (see below).
        for (var c = 0u; c < parentCount; c++) {
            let parent = cutGroups[offset + 8u + c * 3u];
            if (!bitSet(object.firstPairBit + parent)) { return false; }
            if (bitSet(uniform.freeBase + object.firstPairBit + parent)) {
                unchargedIndices += cutGroups[offset + 9u + c * 3u];
                unchargedRecords += cutGroups[offset + 10u + c * 3u];
            }
        }
        let group = meshletData[first];
        let bucket = meshletBucket(group.flags);
        // this bucket's threshold, and its ratio to the base one for the camera LOD test
        let bucketLod = uniform.cullParams[${CULL_PARAMS.BUCKET_LOD}u][bucket];
        let center = (meshletInstanceMatrix(instance) * vec4f(group.groupSphere.xyz, 1.0)).xyz;
        let radius = group.groupSphere.w * object.maxScale;
        let ortho = uniform.cullParams[${CULL_PARAMS.STREAMING}u].y;
        let distanceToGroup = max(distance(center, uniform.cullParams[${CULL_PARAMS.CAMERA}u].xyz) - radius, 1e-5);
        let scale = select(uniform.cullParams[${CULL_PARAMS.CAMERA}u].w / distanceToGroup, ortho, ortho > 0.0);
        let worldError = group.clusterError * object.maxScale;
        var refine = worldError * scale > bucketLod;
        // Shadow views also refine as far as the viewing camera does. A far cascade's texel
        // spans metres, and a caster simplified to a few of them stands proud of the surface
        // the camera draws - which then shadows itself in broad blotches no bias covers.
        let cameraLod = uniform.cullParams[${CULL_PARAMS.VIEW_DIR}u].w * bucketLod / max(uniform.cullParams[${CULL_PARAMS.LOD}u].x, 1e-6);
        if (!refine && cameraLod > 0.0) {
            let priority = uniform.cullParams[${CULL_PARAMS.ROOT_PRIORITY}u];
            let cameraDistance = max(distance(center, priority.xyz) - radius, 1e-5);
            refine = worldError * priority.w / cameraDistance > cameraLod;
        }
        if (!refine) { return false; }
        // An off-frustum group still refines by distance, otherwise an on-screen replacement
        // depending on it (a DAG group straddling the screen edge) stays coarse up close. Its
        // finer members are never drawn - compaction skips children of flagged groups - so it
        // needs no pages, charges nothing and refunds its charged coarse members, which leave
        // the cut. The margin keeps the flag clear of any group whose children's bounds could
        // still reach the frustum.
        if (!meshletInCullVolume(center, radius * 1.01)) {
            let refundIndices = cutGroups[offset + 5u] - unchargedIndices;
            let refundRecords = cutGroups[offset + 7u] - unchargedRecords;
            atomicSub(&counters[${MESHLET_COUNTER.DEMAND_BASE}u + bucket], refundIndices);
            atomicSub(&counters[${CUT_BUDGET.INDICES_BASE}u + bucket], refundIndices);
            atomicSub(&counters[${CUT_BUDGET.RECORDS}u], refundRecords);
            markBit(object.firstPairBit + first - firstMeshlet);
            markBit(uniform.freeBase + object.firstPairBit + first - firstMeshlet);
            return true;
        }
        if (!meshletRequestPages(pageStart, pageCount)) {
            atomicAdd(&counters[${MESHLET_COUNTER.CUT_WAITING}u], 1u);
            return false;
        }
        let fineIndices = cutGroups[offset + 4u];
        let coarseIndices = cutGroups[offset + 5u] - unchargedIndices;
        let fineRecords = cutGroups[offset + 6u];
        let coarseRecords = cutGroups[offset + 7u] - unchargedRecords;
        // Keep non-negative charges: if reclustering increases coarse record count, retaining
        // the extra reservation is conservative and avoids capacity being spent twice.
        let indexNeed = max(fineIndices, coarseIndices) - coarseIndices;
        let recordNeed = max(fineRecords, coarseRecords) - coarseRecords;
        let capacity = meshletBucketCapacity(bucket);
        // Capacity belongs to the complete cut before visibility culling. Counting only
        // drawn meshlets lets occlusion shrink these buffers and randomly reject otherwise
        // affordable replacements on the next frame. Include accepted and rejected growth.
        atomicAdd(&counters[${MESHLET_COUNTER.DEMAND_BASE}u + bucket], indexNeed);
        let recordBefore = atomicAdd(&counters[${CUT_BUDGET.RECORDS}u], recordNeed);
        let indexBefore = atomicAdd(&counters[${CUT_BUDGET.INDICES_BASE}u + bucket], indexNeed);
        if (recordBefore + recordNeed > uniform.recordCapacity || indexBefore + indexNeed > capacity) {
            atomicSub(&counters[${CUT_BUDGET.RECORDS}u], recordNeed);
            atomicSub(&counters[${CUT_BUDGET.INDICES_BASE}u + bucket], indexNeed);
            atomicAdd(&counters[${CUT_BUDGET.REJECTED_RECORDS}u], recordNeed);
            return false;
        }
        markBit(object.firstPairBit + first - firstMeshlet);
        return true;
    }

    ${entry}
`;

/**
 * Appends tasks to the cut queue: the `count` task rows listed in cutGroups from `listStart`, all
 * of `instance`. Past the queue's capacity they are dropped (their groups stay coarse this frame)
 * but still counted, so the CPU grows the queue to the demand. Needs `cutGroups`, `counters`,
 * `cutQueue` and `uniform.queueCapacity`.
 *
 * @ignore
 */
export const meshletCutEnqueueWGSL = /* wgsl */ `
    fn cutEnqueue(instance: u32, listStart: u32) {
        let count = cutGroups[listStart];
        if (count == 0u) { return; }
        let base = atomicAdd(&counters[${MESHLET_COUNTER.CUT_QUEUE_END}u], count);
        for (var k = 0u; k < count; k++) {
            if (base + k < uniform.queueCapacity) {
                cutQueue[base + k] = vec2u(instance, cutGroups[listStart + 1u + k]);
            }
        }
    }
`;

/**
 * The cut stage as a queue traversal: one thread per task queued for this stage, rather than per
 * task of every visible instance. The root admission queues the tasks that depend on the roots
 * alone; a stage queues, for each group it marks, the tasks that group owns (see MeshletCutData),
 * which run at the next level. A task enqueued ahead of its level (a root-owned task below the
 * coarsest level) is carried forward unchanged until its level's stage, so every task still runs
 * after all of its parents - one level coarser - have been decided.
 *
 * @ignore
 */
export const meshletCutQueueWGSL = cutShader(/* wgsl */ `
    // this stage's DAG level
    uniform stageLevel: u32;
    uniform queueCapacity: u32;
    var<storage, read_write> cutQueue: array<vec2u>;

    ${meshletCutEnqueueWGSL}

    @compute @workgroup_size(${MESHLET_LIST_WORKGROUP})
    fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) lane: vec3u) {
        let begin = atomicLoad(&counters[${MESHLET_COUNTER.CUT_STAGE_BEGIN}u]);
        let index = begin + meshletGroupIndex(group) * ${MESHLET_LIST_WORKGROUP}u + lane.x;
        if (index >= atomicLoad(&counters[${MESHLET_COUNTER.CUT_STAGE_END}u])) { return; }
        let task = cutQueue[index];
        // root lists ride the queue for the emit pass only
        if ((task.x & ${MESHLET_CUT_ROOT_FLAG}u) != 0u) { return; }
        let offset = task.y;
        if (meshletData[cutGroups[offset]].lodLevel < uniform.stageLevel) {
            let slot = atomicAdd(&counters[${MESHLET_COUNTER.CUT_QUEUE_END}u], 1u);
            if (slot < uniform.queueCapacity) { cutQueue[slot] = task; }
            // the emit pass visits the task where it is decided, not where it waited
            cutQueue[index] = vec2u(${MESHLET_CUT_INVALID}u, 0u);
            return;
        }
        if (cutTask(task)) {
            // the owned list follows the page list
            cutEnqueue(task.x, cutGroups[offset + 2u] + cutGroups[offset + 3u]);
        }
    }
`);

/**
 * One thread, ahead of each queued cut stage: the stage's range of the queue is everything
 * appended since the previous stage's range began (clamped to the capacity), and its indirect
 * dispatch runs one thread per entry.
 *
 * @ignore
 */
export const meshletCutQueueArgsWGSL = /* wgsl */ `
    uniform dispatchSlot: u32;
    uniform queueCapacity: u32;

    var<storage, read_write> counters: array<u32>;
    var<storage, read_write> indirectDispatch: array<u32>;

    ${meshletDispatchArgsWGSL}

    @compute @workgroup_size(1)
    fn main() {
        let begin = counters[${MESHLET_COUNTER.CUT_STAGE_END}u];
        let end = min(counters[${MESHLET_COUNTER.CUT_QUEUE_END}u], uniform.queueCapacity);
        counters[${MESHLET_COUNTER.CUT_STAGE_BEGIN}u] = begin;
        counters[${MESHLET_COUNTER.CUT_STAGE_END}u] = max(begin, end);
        meshletWriteDispatch(uniform.dispatchSlot, (max(begin, end) - begin + ${MESHLET_LIST_WORKGROUP - 1}u) / ${MESHLET_LIST_WORKGROUP}u);
    }
`;

/**
 * Emits the selected cut straight from the cut queue, one thread per queue entry, after the last
 * stage. A meshlet is drawn when the group it replaces (its parent) refined on-frustum and the
 * group it belongs to (its birth group) did not. Every group that refined was decided from the
 * queue, and every admitted instance queued its roots there, so the queue lists each candidate
 * once: a refined group's children, and the roots, so this costs the size of the cut.
 *
 * @ignore
 */
export const meshletCutEmitWGSL = /* wgsl */ `
    ${meshletStructsWGSL}
    uniform totalPairs: u32;
    // bit base of the off-frustum refinement flags, after the pair and admission bits
    uniform freeBase: u32;
    uniform recordCapacity: u32;
    uniform queueCapacity: u32;
    ${meshletDataWGSL}
    ${meshletObjectDataWGSL}
    var<storage, read> cutGroups: array<u32>;
    var<storage, read> cutQueue: array<vec2u>;
    var<storage, read> claimBits: array<u32>;
    var<storage, read> selectionTopology: array<vec2u>;
    var<storage, read_write> counters: array<atomic<u32>>;
    var<storage, read_write> selectedMeshlets: array<vec2u>;

    fn bitSet(bit: u32) -> bool {
        return (claimBits[bit >> 5u] & (1u << (bit & 31u))) != 0u;
    }
    ${meshletDispatchGroupWGSL}

    // drawn unless its own group refined, and only when it has triangles: a cull-to-empty bake
    // (gltf-tools cullToEmpty, alpha-masked primitives) caps the DAG with triangle-less roots
    fn emits(firstMeshlet: u32, local: u32, pairBase: u32) -> bool {
        let meshlet = firstMeshlet + local;
        let birth = selectionTopology[meshlet].y;
        if (birth != ${MESHLET_NO_PARENT}u && bitSet(pairBase + birth)) { return false; }
        return meshletData[meshlet].triangleCount != 0u;
    }

    @compute @workgroup_size(${MESHLET_LIST_WORKGROUP})
    fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) lane: vec3u) {
        let index = meshletGroupIndex(group) * ${MESHLET_LIST_WORKGROUP}u + lane.x;
        if (index >= min(atomicLoad(&counters[${MESHLET_COUNTER.CUT_QUEUE_END}u]), uniform.queueCapacity)) { return; }
        let entry = cutQueue[index];
        if (entry.x == ${MESHLET_CUT_INVALID}u) { return; }

        // the list of candidates: a root list, or a refined group's children
        var instance = entry.x & ${~MESHLET_CUT_ROOT_FLAG >>> 0}u;
        var list = entry.y;
        let pairBase = objectData[instance].firstPairBit;
        let firstMeshlet = meshletPrimitiveData(instance).firstMeshlet;
        if ((entry.x & ${MESHLET_CUT_ROOT_FLAG}u) == 0u) {
            let offset = entry.y;
            let groupBit = pairBase + cutGroups[offset] - firstMeshlet;
            // undecided, or refined off-frustum only to unblock its dependants
            if (!bitSet(groupBit) || bitSet(uniform.freeBase + groupBit)) { return; }
            // the children follow the owned task list, which follows the page list
            let owned = cutGroups[offset + 2u] + cutGroups[offset + 3u];
            list = owned + 1u + cutGroups[owned];
        }

        // count, then write: the first 64 decisions are kept in a mask rather than re-evaluated
        let count = cutGroups[list];
        var emitted = 0u;
        var mask = vec2u(0u);
        for (var k = 0u; k < count; k++) {
            if (emits(firstMeshlet, cutGroups[list + 1u + k], pairBase)) {
                emitted++;
                if (k < 64u) { mask[k >> 5u] |= 1u << (k & 31u); }
            }
        }
        if (emitted == 0u) { return; }
        var slot = atomicAdd(&counters[${MESHLET_COUNTER.SELECTED}u], emitted);
        for (var k = 0u; k < count; k++) {
            let local = cutGroups[list + 1u + k];
            var drawn: bool;
            if (k < 64u) {
                drawn = (mask[k >> 5u] & (1u << (k & 31u))) != 0u;
            } else {
                drawn = emits(firstMeshlet, local, pairBase);
            }
            if (drawn) {
                if (slot < uniform.recordCapacity) {
                    selectedMeshlets[slot] = vec2u(instance, local);
                }
                slot++;
            }
        }
    }
`;
