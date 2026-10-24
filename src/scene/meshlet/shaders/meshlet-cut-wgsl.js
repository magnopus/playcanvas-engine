import {
    CULL_PARAMS, MESHLET_COUNTER, MESHLET_CUT_CHUNK, MESHLET_DISPATCH_WIDTH, MESHLET_FLAG_ALPHA_MASKED,
    MESHLET_FLAG_TWO_SIDED, MESHLET_STAGE_CHUNKS_BASE, PAGE_NOT_RESIDENT, PAGE_REQUEST, MESHLET_LEVEL_BIAS
} from '../constants.js';
import { meshletCullVolumeWGSL } from './meshlet-cull-wgsl.js';
import { meshletStructsWGSL, meshletDataWGSL, meshletObjectDataWGSL } from './meshlet-page-wgsl.js';

/**
 * Shared declarations and the per-task body of the cut stage: `cutTask(taskIndex)` selects one
 * replacement group (an index into cutTasks). The entry points below decide which tasks run.
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
    var<storage, read> cutTasks: array<vec2u>;
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

    fn cutTask(taskIndex: u32) {
        let task = cutTasks[taskIndex];
        let instance = task.x;
        let admissionBit = uniform.totalPairs + instance;
        if (!bitSet(admissionBit)) { return; }
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
                if (!bitSet(object.firstPairBit + parent)) { return; }
                if (bitSet(uniform.freeBase + object.firstPairBit + parent)) {
                    unchargedIndices += cutGroups[offset + 9u + c * 3u];
                    unchargedRecords += cutGroups[offset + 10u + c * 3u];
                }
            }
            let group = meshletData[first];
            let center = (object.worldMatrix * vec4f(group.groupSphere.xyz, 1.0)).xyz;
            let radius = group.groupSphere.w * object.maxScale;
            let ortho = cullParams[${CULL_PARAMS.STREAMING}u].y;
            let distanceToGroup = max(distance(center, cullParams[${CULL_PARAMS.CAMERA}u].xyz) - radius, 1e-5);
            let scale = select(cullParams[${CULL_PARAMS.CAMERA}u].w / distanceToGroup, ortho, ortho > 0.0);
            let worldError = group.clusterError * object.maxScale;
            var refine = worldError * scale > cullParams[${CULL_PARAMS.LOD}u].x;
            // Shadow views also refine as far as the viewing camera does. A far cascade's texel
            // spans metres, and a caster simplified to a few of them stands proud of the surface
            // the camera draws - which then shadows itself in broad blotches no bias covers.
            let cameraLod = cullParams[${CULL_PARAMS.VIEW_DIR}u].w;
            if (!refine && cameraLod > 0.0) {
                let priority = cullParams[${CULL_PARAMS.ROOT_PRIORITY}u];
                let cameraDistance = max(distance(center, priority.xyz) - radius, 1e-5);
                refine = worldError * priority.w / cameraDistance > cameraLod;
            }
            if (!refine) { return; }
            // An off-frustum group still refines by distance, otherwise an on-screen
            // replacement depending on it (a DAG group straddling the screen edge) stays
            // coarse up close. Its finer members are never drawn - compaction skips children
            // of flagged groups - so it needs no pages, charges nothing and refunds its
            // charged coarse members, which leave the cut. The margin keeps the flag clear
            // of any group whose children's bounds could still reach the frustum.
            if (!inFrustum(center, radius * 1.01)) {
                let refundIndices = cutGroups[offset + 5u] - unchargedIndices;
                let refundRecords = cutGroups[offset + 7u] - unchargedRecords;
                let flags = group.flags;
                let bucket = select(select(0u, 1u, (flags & ${MESHLET_FLAG_TWO_SIDED}u) != 0u), 2u, (flags & ${MESHLET_FLAG_ALPHA_MASKED}u) != 0u);
                atomicSub(&counters[${MESHLET_COUNTER.DEMAND_BASE}u + bucket], refundIndices);
                atomicSub(&cutBudget[bucket], refundIndices);
                atomicSub(&cutBudget[3u], refundRecords);
                markBit(object.firstPairBit + first - object.firstMeshlet);
                markBit(uniform.freeBase + object.firstPairBit + first - object.firstMeshlet);
                return;
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
        if (!ready) { return; }
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
            return;
        }
        markBit(object.firstPairBit + first - object.firstMeshlet);
        // this group's children (one level finer) are now selectable: compaction starts the
        // instance no finer than the finest such level (see instanceCullWGSL)
        let childLevel = meshletData[first].lodLevel - 1u;
        atomicMax(&claimBits[uniform.levelBase + instance], ${MESHLET_LEVEL_BIAS}u - childLevel);
    }

    ${entry}
`;

/**
 * Select complete replacements from coarse to fine, reserving capacity before refinement. The
 * roots are admitted beforehand (meshletRootAdmitWGSL); a stage runs one DAG level's groups,
 * one thread per task of the level.
 *
 * @ignore
 */
export const meshletCutWGSL = cutShader(/* wgsl */ `
    uniform taskStart: u32;
    uniform taskCount: u32;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
        let index = gid.x + gid.y * (65535u * 64u);
        if (index >= uniform.taskCount) { return; }
        cutTask(uniform.taskStart + index);
    }
`);

/**
 * The cut stage over the visible cells only: one workgroup per chunk of a visible cell's tasks
 * at this level (see cellExpandWGSL), a lane per task. An instance outside the visible cells
 * was never admitted, so its tasks would only early-out; skipping them is what keeps a shadow
 * cascade from paying for the whole world.
 *
 * @ignore
 */
export const meshletCutChunksWGSL = cutShader(/* wgsl */ `
    uniform stage: u32;
    // word base in cutGroups of this stage's chunk -> first task row
    uniform chunkTaskBase: u32;
    // word base in counters of this stage's visible-chunk list
    uniform listBase: u32;

    @compute @workgroup_size(${MESHLET_CUT_CHUNK})
    fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) lane: vec3u) {
        let slot = group.y * ${MESHLET_DISPATCH_WIDTH}u + group.x;
        if (slot >= atomicLoad(&counters[${MESHLET_STAGE_CHUNKS_BASE}u + uniform.stage])) { return; }
        let chunk = atomicLoad(&counters[uniform.listBase + slot]);
        let task = cutGroups[uniform.chunkTaskBase + chunk] + lane.x;
        if (task < cutGroups[uniform.chunkTaskBase + chunk + 1u]) {
            cutTask(task);
        }
    }
`);
