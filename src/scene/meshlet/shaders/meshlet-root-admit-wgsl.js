import {
    CULL_FLAG_SHADOW_VIEW, CULL_PARAMS, MESHLET_COUNTER, MESHLET_DISPATCH_WIDTH, OBJECT_FLAG_HIDDEN,
    OBJECT_FLAG_NO_SHADOW, PAGE_NOT_RESIDENT, PAGE_REQUEST
} from '../constants.js';
import { meshletStructsWGSL, meshletObjectDataWGSL } from './meshlet-page-wgsl.js';

// band index in the top bits of a root list entry, instance id below
const BAND_SHIFT = 27;
const INSTANCE_MASK = (1 << BAND_SHIFT) - 1;

/**
 * Root classification, one thread per instance: drops hidden (and, in a shadow view,
 * non-casting) instances and those outside the frustum, ranks the rest by projected size - the
 * LOD metric - into priority bands, and appends them to the root list for admission. An instance
 * admitted last frame is promoted one band (hysteresis), so admission does not flicker at the
 * budget's edge.
 *
 * @ignore
 */
export const meshletRootClassifyWGSL = /* wgsl */ `
    ${meshletStructsWGSL}
    uniform instanceCount: u32;
    // word base of last frame's persistent admission bits (see claimPersistBase)
    uniform prevBase: u32;
    // 1 when the previous region holds a real previous frame
    uniform hasPrevious: u32;
    ${meshletObjectDataWGSL}
    var<storage, read> cullParams: array<vec4f>;
    var<storage, read> claimBits: array<u32>;
    var<storage, read_write> counters: array<atomic<u32>>;
    var<storage, read_write> rootList: array<u32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
        let instance = gid.x + gid.y * (65535u * 64u);
        if (instance >= uniform.instanceCount) { return; }
        let object = objectData[instance];
        let shadow = (u32(cullParams[${CULL_PARAMS.STREAMING}u].z) & ${CULL_FLAG_SHADOW_VIEW}u) != 0u;
        if ((object.flags & ${OBJECT_FLAG_HIDDEN}u) != 0u ||
            (shadow && (object.flags & ${OBJECT_FLAG_NO_SHADOW}u) != 0u)) { return; }
        let center = (object.worldMatrix * vec4f(object.sphere.xyz, 1.0)).xyz;
        let radius = object.sphere.w * object.maxScale;
        for (var p = 0u; p < ${CULL_PARAMS.PLANE_COUNT}u; p++) {
            let plane = cullParams[${CULL_PARAMS.PLANES}u + p];
            if (dot(plane.xyz, center) + plane.w < -radius) { return; }
        }

        // projected size from the priority origin (the viewing camera, also for shadow views)
        let priority = cullParams[${CULL_PARAMS.ROOT_PRIORITY}u];
        let bands = cullParams[${CULL_PARAMS.ROOT_BANDS}u];
        let gap = max(distance(center, priority.xyz) - radius, 1e-4);
        let pixels = radius * priority.w / gap;
        if (pixels < bands.w) {
            atomicAdd(&counters[${MESHLET_COUNTER.ROOT_SUBPIXEL}u], 1u);
            return;
        }
        let bandCount = u32(bands.z);
        var band = u32(clamp(floor(log2(bands.x / max(pixels, 1e-6)) * bands.y), 0.0, f32(bandCount - 1u)));
        if (uniform.hasPrevious != 0u &&
            (claimBits[uniform.prevBase + (instance >> 5u)] & (1u << (instance & 31u))) != 0u) {
            band = max(band, 1u) - 1u;
        }
        let slot = atomicAdd(&counters[${MESHLET_COUNTER.ROOT_CANDIDATES}u], 1u);
        if (slot < arrayLength(&rootList)) {
            rootList[slot] = instance | (band << ${BAND_SHIFT}u);
        }
    }
`;

/**
 * Root admission for one priority band, one thread per root list entry: entries of other bands
 * return at once, the rest check their root pages and reserve their root indices and records
 * against the view's capacity. Bands dispatch in order, so larger instances claim capacity
 * first. Accepted and rejected root cost both count as demand (the capacity grows to the full
 * pre-cull cut); rejected and unready instances are counted for the budget report.
 *
 * @ignore
 */
export const meshletRootAdmitWGSL = /* wgsl */ `
    uniform rootBand: u32;
    uniform totalPairs: u32;
    // word base of this frame's persistent admission bits (see claimPersistBase)
    uniform nextBase: u32;
    uniform recordCapacity: u32;
    uniform indexCapacity0: u32;
    uniform indexCapacity1: u32;
    uniform indexCapacity2: u32;
    var<storage, read> cutGroups: array<u32>;
    var<storage, read> cutTasks: array<vec2u>;
    var<storage, read> residency: array<u32>;
    var<storage, read_write> requests: array<atomic<u32>>;
    var<storage, read_write> claimBits: array<atomic<u32>>;
    // used indices per bucket, used records, then rejected record demand
    var<storage, read_write> cutBudget: array<atomic<u32>>;
    var<storage, read_write> counters: array<atomic<u32>>;
    var<storage, read> rootList: array<u32>;

    @compute @workgroup_size(64)
    fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) lane: vec3u) {
        let index = (group.y * ${MESHLET_DISPATCH_WIDTH}u + group.x) * 64u + lane.x;
        if (index >= min(atomicLoad(&counters[${MESHLET_COUNTER.ROOT_CANDIDATES}u]), arrayLength(&rootList))) { return; }
        let entry = rootList[index];
        if ((entry >> ${BAND_SHIFT}u) != uniform.rootBand) { return; }
        let instance = entry & ${INSTANCE_MASK}u;
        // root tasks come first in the task list, one per instance in instance order
        let offset = cutTasks[instance].y;
        let pageStart = cutGroups[offset + 2u];
        let pageCount = cutGroups[offset + 3u];
        var ready = true;
        for (var c = 0u; c < pageCount; c++) {
            let page = cutGroups[pageStart + c];
            let resident = residency[page] != ${PAGE_NOT_RESIDENT}u;
            let mark = select(${PAGE_REQUEST.MISSING}u, ${PAGE_REQUEST.USED}u, resident);
            if (atomicLoad(&requests[page]) < mark) { atomicMax(&requests[page], mark); }
            ready = ready && resident;
        }
        if (!ready) {
            atomicAdd(&counters[${MESHLET_COUNTER.ROOT_UNREADY_INSTANCES}u], 1u);
            return;
        }
        // root header: indices at +4, bucket at +5, records at +6 (see MeshletCutData)
        let indices = cutGroups[offset + 4u];
        let bucket = cutGroups[offset + 5u];
        let records = cutGroups[offset + 6u];
        let capacity = select(select(uniform.indexCapacity0, uniform.indexCapacity1, bucket == 1u), uniform.indexCapacity2, bucket == 2u);
        atomicAdd(&counters[${MESHLET_COUNTER.DEMAND_BASE}u + bucket], indices);
        let indexBefore = atomicAdd(&cutBudget[bucket], indices);
        let recordBefore = atomicAdd(&cutBudget[3u], records);
        if (indexBefore + indices > capacity || recordBefore + records > uniform.recordCapacity) {
            atomicSub(&cutBudget[bucket], indices);
            atomicSub(&cutBudget[3u], records);
            atomicAdd(&cutBudget[4u], records);
            atomicAdd(&counters[${MESHLET_COUNTER.ROOT_REJECTED_BASE}u + bucket], indices);
            atomicAdd(&counters[${MESHLET_COUNTER.ROOT_REJECTED_INSTANCES}u], 1u);
            return;
        }
        atomicAdd(&counters[${MESHLET_COUNTER.ROOT_ADMITTED_BASE}u + bucket], indices);
        let admission = uniform.totalPairs + instance;
        atomicOr(&claimBits[admission >> 5u], 1u << (admission & 31u));
        atomicOr(&claimBits[uniform.nextBase + (instance >> 5u)], 1u << (instance & 31u));
    }
`;
