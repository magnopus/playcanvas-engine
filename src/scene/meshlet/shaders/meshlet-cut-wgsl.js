import {
    CULL_PARAMS, INDIRECT_DISPATCH_U32S, MESHLET_COUNTER, MESHLET_DISPATCH_WIDTH, MESHLET_FLAG_ALPHA_MASKED,
    MESHLET_FLAG_TWO_SIDED, PAGE_NOT_RESIDENT, PAGE_REQUEST, MESHLET_LEVEL_BIAS
} from '../constants.js';
import { meshletCullVolumeWGSL } from './meshlet-cull-wgsl.js';
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
    // word base of the per-instance finest-cut-level words, after every claim bit
    uniform levelBase: u32;
    uniform recordCapacity: u32;
    uniform indexCapacity0: u32;
    uniform indexCapacity1: u32;
    uniform indexCapacity2: u32;
    ${meshletDataWGSL}
    ${meshletObjectDataWGSL}
    var<storage, read> cutGroups: array<u32>;
    var<storage, read> cullParams: array<vec4f>;
    var<storage, read> residency: array<u32>;
    var<storage, read_write> requests: array<atomic<u32>>;
    var<storage, read_write> claimBits: array<atomic<u32>>;
    // used indices per bucket, used records, then rejected record demand
    var<storage, read_write> cutBudget: array<atomic<u32>>;
    var<storage, read_write> counters: array<atomic<u32>>;

    fn bitSet(bit: u32) -> bool {
        return (atomicLoad(&claimBits[bit >> 5u]) & (1u << (bit & 31u))) != 0u;
    }
    fn markBit(bit: u32) {
        atomicOr(&claimBits[bit >> 5u], 1u << (bit & 31u));
    }
    ${meshletCullVolumeWGSL}
    fn inFrustum(center: vec3f, radius: f32) -> bool {
        return meshletInCullVolume(center, radius);
    }

    // Decides one replacement group: true when its bit was marked - refined, or flagged as
    // refined off-frustum - so the tasks that wait on it may now run.
    fn cutTask(task: vec2u) -> bool {
        let instance = task.x;
        let admissionBit = uniform.totalPairs + instance;
        if (!bitSet(admissionBit)) { return false; }
        let offset = task.y;
        let object = objectData[instance];
        var unchargedIndices = 0u;
        var unchargedRecords = 0u;
        let first = cutGroups[offset];
        let parentCount = cutGroups[offset + 1u];
        let pageStart = cutGroups[offset + 2u];
        let pageCount = cutGroups[offset + 3u];
        {
            // Every coarse member must belong to the active cut. A DAG replacement may
            // depend on several different ancestor groups, all of which must have refined.
            // Members under an off-frustum parent were never charged (see below).
            for (var c = 0u; c < parentCount; c++) {
                let parent = cutGroups[offset + 8u + c * 3u];
                if (!bitSet(object.firstPairBit + parent)) { return false; }
                if (bitSet(uniform.freeBase + object.firstPairBit + parent)) {
                    unchargedIndices += cutGroups[offset + 9u + c * 3u];
                    unchargedRecords += cutGroups[offset + 10u + c * 3u];
                }
            }
            let group = meshletData[first];
            let groupBucket = select(select(0u, 1u, (group.flags & ${MESHLET_FLAG_TWO_SIDED}u) != 0u), 2u, (group.flags & ${MESHLET_FLAG_ALPHA_MASKED}u) != 0u);
            // this bucket's threshold, and its ratio to the base one for the camera LOD test
            let bucketLod = cullParams[${CULL_PARAMS.BUCKET_LOD}u][groupBucket];
            let center = (object.worldMatrix * vec4f(group.groupSphere.xyz, 1.0)).xyz;
            let radius = group.groupSphere.w * object.maxScale;
            let ortho = cullParams[${CULL_PARAMS.STREAMING}u].y;
            let distanceToGroup = max(distance(center, cullParams[${CULL_PARAMS.CAMERA}u].xyz) - radius, 1e-5);
            let scale = select(cullParams[${CULL_PARAMS.CAMERA}u].w / distanceToGroup, ortho, ortho > 0.0);
            let worldError = group.clusterError * object.maxScale;
            var refine = worldError * scale > bucketLod;
            // Shadow views also refine as far as the viewing camera does. A far cascade's texel
            // spans metres, and a caster simplified to a few of them stands proud of the surface
            // the camera draws - which then shadows itself in broad blotches no bias covers.
            let cameraLod = cullParams[${CULL_PARAMS.VIEW_DIR}u].w * bucketLod / max(cullParams[${CULL_PARAMS.LOD}u].x, 1e-6);
            if (!refine && cameraLod > 0.0) {
                let priority = cullParams[${CULL_PARAMS.ROOT_PRIORITY}u];
                let cameraDistance = max(distance(center, priority.xyz) - radius, 1e-5);
                refine = worldError * priority.w / cameraDistance > cameraLod;
            }
            if (!refine) { return false; }
            // An off-frustum group still refines by distance, otherwise an on-screen
            // replacement depending on it (a DAG group straddling the screen edge) stays
            // coarse up close. Its finer members are never drawn - compaction skips children
            // of flagged groups - so it needs no pages, charges nothing and refunds its
            // charged coarse members, which leave the cut. The margin keeps the flag clear
            // of any group whose children's bounds could still reach the frustum.
            if (!inFrustum(center, radius * 1.01)) {
                let refundIndices = cutGroups[offset + 5u] - unchargedIndices;
                let refundRecords = cutGroups[offset + 7u] - unchargedRecords;
                atomicSub(&counters[${MESHLET_COUNTER.DEMAND_BASE}u + groupBucket], refundIndices);
                atomicSub(&cutBudget[groupBucket], refundIndices);
                atomicSub(&cutBudget[3u], refundRecords);
                markBit(object.firstPairBit + first - object.firstMeshlet);
                markBit(uniform.freeBase + object.firstPairBit + first - object.firstMeshlet);
                return true;
            }
        }
        var ready = true;
        for (var c = 0u; c < pageCount; c++) {
            let page = cutGroups[pageStart + c];
            let resident = residency[page] != ${PAGE_NOT_RESIDENT}u;
            let mark = select(${PAGE_REQUEST.MISSING}u, ${PAGE_REQUEST.USED}u, resident);
            if (atomicLoad(&requests[page]) < mark) { atomicMax(&requests[page], mark); }
            ready = ready && resident;
        }
        if (!ready) { return false; }
        let fineIndices = cutGroups[offset + 4u];
        let coarseIndices = cutGroups[offset + 5u] - unchargedIndices;
        let fineRecords = cutGroups[offset + 6u];
        let coarseRecords = cutGroups[offset + 7u] - unchargedRecords;
        // Keep non-negative charges: if reclustering increases coarse record count, retaining
        // the extra reservation is conservative and avoids capacity being spent twice.
        let indexNeed = max(fineIndices, coarseIndices) - coarseIndices;
        let recordNeed = max(fineRecords, coarseRecords) - coarseRecords;
        let flags = meshletData[first].flags;
        let bucket = select(select(0u, 1u, (flags & ${MESHLET_FLAG_TWO_SIDED}u) != 0u), 2u, (flags & ${MESHLET_FLAG_ALPHA_MASKED}u) != 0u);
        let capacity = select(select(uniform.indexCapacity0, uniform.indexCapacity1, bucket == 1u), uniform.indexCapacity2, bucket == 2u);
        // Capacity belongs to the complete cut before visibility culling. Counting only
        // drawn meshlets lets occlusion shrink these buffers and randomly reject otherwise
        // affordable replacements on the next frame. Include accepted and rejected growth.
        atomicAdd(&counters[${MESHLET_COUNTER.DEMAND_BASE}u + bucket], indexNeed);
        let recordBefore = atomicAdd(&cutBudget[3u], recordNeed);
        let indexBefore = atomicAdd(&cutBudget[bucket], indexNeed);
        if (recordBefore + recordNeed > uniform.recordCapacity || indexBefore + indexNeed > capacity) {
            atomicSub(&cutBudget[3u], recordNeed);
            atomicSub(&cutBudget[bucket], indexNeed);
            atomicAdd(&cutBudget[4u], recordNeed);
            return false;
        }
        markBit(object.firstPairBit + first - object.firstMeshlet);
        // this group's children (one level finer) are now selectable: compaction starts the
        // instance no finer than the finest such level (see instanceCullWGSL)
        let childLevel = meshletData[first].lodLevel - 1u;
        atomicMax(&claimBits[uniform.levelBase + instance], ${MESHLET_LEVEL_BIAS}u - childLevel);
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

    @compute @workgroup_size(64)
    fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) lane: vec3u) {
        let begin = atomicLoad(&counters[${MESHLET_COUNTER.CUT_STAGE_BEGIN}u]);
        let index = begin + (group.y * ${MESHLET_DISPATCH_WIDTH}u + group.x) * 64u + lane.x;
        if (index >= atomicLoad(&counters[${MESHLET_COUNTER.CUT_STAGE_END}u])) { return; }
        let task = cutQueue[index];
        let offset = task.y;
        if (meshletData[cutGroups[offset]].lodLevel < uniform.stageLevel) {
            let slot = atomicAdd(&counters[${MESHLET_COUNTER.CUT_QUEUE_END}u], 1u);
            if (slot < uniform.queueCapacity) { cutQueue[slot] = task; }
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

    @compute @workgroup_size(1)
    fn main() {
        let begin = counters[${MESHLET_COUNTER.CUT_STAGE_END}u];
        let end = min(counters[${MESHLET_COUNTER.CUT_QUEUE_END}u], uniform.queueCapacity);
        counters[${MESHLET_COUNTER.CUT_STAGE_BEGIN}u] = begin;
        counters[${MESHLET_COUNTER.CUT_STAGE_END}u] = max(begin, end);
        let groups = (max(begin, end) - begin + 63u) / 64u;
        let base = uniform.dispatchSlot * ${INDIRECT_DISPATCH_U32S}u;
        indirectDispatch[base + 0u] = min(groups, ${MESHLET_DISPATCH_WIDTH}u);
        indirectDispatch[base + 1u] = (groups + ${MESHLET_DISPATCH_WIDTH - 1}u) / ${MESHLET_DISPATCH_WIDTH}u;
        indirectDispatch[base + 2u] = 1u;
    }
`;
