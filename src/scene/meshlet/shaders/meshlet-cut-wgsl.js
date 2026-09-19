import {
    CULL_PARAMS, CULL_PARAMS_VEC4S, CULL_FLAG_SHADOW_VIEW, MESHLET_COUNTER, MESHLET_FLAG_ALPHA_MASKED,
    MESHLET_FLAG_TWO_SIDED, OBJECT_FLAG_HIDDEN, OBJECT_FLAG_NO_SHADOW,
    PAGE_NOT_RESIDENT, PAGE_REQUEST
} from '../constants.js';
import { meshletStructsWGSL, meshletDataWGSL, meshletObjectDataWGSL } from './meshlet-page-wgsl.js';

/** Select complete replacements from coarse to fine, reserving capacity before refinement. @ignore */
export const meshletCutWGSL = /* wgsl */ `
    ${meshletStructsWGSL}
    uniform taskStart: u32;
    uniform taskCount: u32;
    uniform rootStage: u32;
    uniform totalPairs: u32;
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
    fn inFrustum(center: vec3f, radius: f32) -> bool {
        for (var p = 0u; p < ${CULL_PARAMS.PLANE_COUNT}u; p++) {
            let plane = cullParams[${CULL_PARAMS.PLANES}u + p];
            if (dot(plane.xyz, center) + plane.w < -radius) { return false; }
        }
        return true;
    }

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
        let index = gid.x + gid.y * (65535u * 64u);
        if (index >= uniform.taskCount) { return; }
        let task = cutTasks[uniform.taskStart + index];
        let instance = task.x;
        let admissionBit = uniform.totalPairs + instance;
        if (uniform.rootStage == 0u && !bitSet(admissionBit)) { return; }
        let offset = task.y;
        let object = objectData[instance];
        let first = cutGroups[offset];
        let parentCount = cutGroups[offset + 1u];
        let pageStart = cutGroups[offset + 2u];
        let pageCount = cutGroups[offset + 3u];
        if (uniform.rootStage != 0u) {
            if (arrayLength(&cullParams) > ${CULL_PARAMS_VEC4S}u &&
                cullParams[${CULL_PARAMS_VEC4S}u + instance / 4u][instance % 4u] == 0.0) { return; }
            let shadow = (u32(cullParams[${CULL_PARAMS.STREAMING}u].z) & ${CULL_FLAG_SHADOW_VIEW}u) != 0u;
            if ((object.flags & ${OBJECT_FLAG_HIDDEN}u) != 0u ||
                (shadow && (object.flags & ${OBJECT_FLAG_NO_SHADOW}u) != 0u)) { return; }
            let center = (object.worldMatrix * vec4f(object.sphere.xyz, 1.0)).xyz;
            if (!inFrustum(center, object.sphere.w * object.maxScale)) { return; }
        } else {
            // Every coarse member must belong to the active cut. A DAG replacement may
            // depend on several different ancestor groups, all of which must have refined.
            for (var c = 0u; c < parentCount; c++) {
                let parent = cutGroups[offset + 8u + c];
                if (!bitSet(object.firstPairBit + parent)) { return; }
            }
            let group = meshletData[first];
            let center = (object.worldMatrix * vec4f(group.groupSphere.xyz, 1.0)).xyz;
            let radius = group.groupSphere.w * object.maxScale;
            if (!inFrustum(center, radius)) { return; }
            let ortho = cullParams[${CULL_PARAMS.STREAMING}u].y;
            let distanceToGroup = max(distance(center, cullParams[${CULL_PARAMS.CAMERA}u].xyz) - radius, 1e-5);
            let scale = select(cullParams[${CULL_PARAMS.CAMERA}u].w / distanceToGroup, ortho, ortho > 0.0);
            if (group.clusterError * object.maxScale * scale <= cullParams[${CULL_PARAMS.LOD}u].x) { return; }
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
        if (uniform.rootStage != 0u) {
            // Reserve complete roots before refinement. Admission is bounded by the current
            // buffers, never by the sum of all placements' potential coarse draws.
            let flags = meshletData[first].flags;
            let bucket = select(select(0u, 1u, (flags & ${MESHLET_FLAG_TWO_SIDED}u) != 0u), 2u, (flags & ${MESHLET_FLAG_ALPHA_MASKED}u) != 0u);
            let capacity = select(select(uniform.indexCapacity0, uniform.indexCapacity1, bucket == 1u), uniform.indexCapacity2, bucket == 2u);
            let indices = cutGroups[offset + 4u];
            let records = cutGroups[offset + 6u];
            let indexBefore = atomicAdd(&cutBudget[bucket], indices);
            let recordBefore = atomicAdd(&cutBudget[3u], records);
            if (indexBefore + indices > capacity || recordBefore + records > uniform.recordCapacity) {
                atomicSub(&cutBudget[bucket], indices);
                atomicSub(&cutBudget[3u], records);
                return;
            }
            atomicAdd(&counters[${MESHLET_COUNTER.DEMAND_BASE}u + bucket], indices);
            markBit(admissionBit);
            return;
        }
        let fineIndices = cutGroups[offset + 4u];
        let coarseIndices = cutGroups[offset + 5u];
        let fineRecords = cutGroups[offset + 6u];
        let coarseRecords = cutGroups[offset + 7u];
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
    }
`;
