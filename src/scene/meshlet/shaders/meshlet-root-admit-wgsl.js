import {
    CULL_FLAG_SHADOW_VIEW, CULL_PARAMS, CUT_BUDGET, MESHLET_CELL_SIZE, MESHLET_COUNTER, MESHLET_CUT_ROOT_CHUNK,
    MESHLET_CUT_ROOT_FLAG, MESHLET_LIST_WORKGROUP, MESHLET_ROOT_BANDS, OBJECT_BUCKET_SHIFT, OBJECT_FLAG_HIDDEN,
    OBJECT_FLAG_NO_SHADOW
} from '../constants.js';
import {
    meshletBucketWGSL, meshletCullParamsWGSL, meshletCullVolumeWGSL, meshletDispatchArgsWGSL, meshletDispatchGroupWGSL, meshletHzbVisibleWGSL,
    meshletRequestPagesWGSL, meshletStridedMainWGSL, meshletVisibleCellWGSL
} from './meshlet-cull-common-wgsl.js';
import { meshletStructsWGSL, meshletObjectDataWGSL } from './meshlet-page-wgsl.js';

// band index in the top bits of a root list entry, instance id below
const BAND_SHIFT = 27;
const INSTANCE_MASK = (1 << BAND_SHIFT) - 1;

/**
 * Root classification, one workgroup per visible cell and one thread per instance of it: drops
 * hidden (and, in a shadow view, non-casting) instances and those outside the frustum, ranks the
 * rest by projected size - the LOD metric - into priority bands, and appends them to the root
 * list for admission. When last frame's HZB is valid, instances wholly behind it are dropped
 * too: admission reserves index and record capacity for every root it takes, before any
 * cluster is tested for occlusion, so a hidden instance would otherwise spend the budget a
 * visible one needs to refine. An instance that stops being hidden is admitted a frame later.
 * An instance admitted last frame is promoted one band (hysteresis), so admission does not
 * flicker at the budget's edge.
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
    // word base of the band-sorted copy, which follows this classify-order list
    uniform sortedBase: u32;
    ${meshletObjectDataWGSL}
    ${meshletCullParamsWGSL}
    var<storage, read> claimBits: array<u32>;
    var<storage, read_write> counters: array<atomic<u32>>;
    var<storage, read_write> rootList: array<u32>;
    // per-band entry counts (see ROOT_BANDS_WORDS)
    var<storage, read_write> rootBands: array<atomic<u32>>;
    // instance ids in cell order
    var<storage, read> cellInstances: array<u32>;
    // last frame's pyramid (a placeholder when it is not valid - see PREV_CAMERA.w)
    var hzbTexture: texture_2d<f32>;

    ${meshletHzbVisibleWGSL}

    ${meshletDispatchGroupWGSL}
    ${meshletVisibleCellWGSL}
    ${meshletCullVolumeWGSL}

    // one virtual workgroup per visible cell; the list holds at most one entry per cell
    fn meshletItemGroups() -> u32 {
        return min(atomicLoad(&counters[${MESHLET_COUNTER.VISIBLE_CELLS}u]),
                   (uniform.instanceCount + ${MESHLET_CELL_SIZE - 1}u) / ${MESHLET_CELL_SIZE}u);
    }

    ${meshletStridedMainWGSL(MESHLET_CELL_SIZE)}

    fn meshletItem(group: vec3u, lane: vec3u) {
        let cell = meshletVisibleCell(group);
        if (cell == 0xFFFFFFFFu) { return; }
        let position = cell * ${MESHLET_CELL_SIZE}u + lane.x;
        if (position >= uniform.instanceCount) { return; }
        let instance = cellInstances[position];
        let object = objectData[instance];
        let shadow = (u32(uniform.cullParams[${CULL_PARAMS.STREAMING}u].z) & ${CULL_FLAG_SHADOW_VIEW}u) != 0u;
        if ((object.flags & ${OBJECT_FLAG_HIDDEN}u) != 0u ||
            (shadow && (object.flags & ${OBJECT_FLAG_NO_SHADOW}u) != 0u)) { return; }
        let priority = uniform.cullParams[${CULL_PARAMS.ROOT_PRIORITY}u];
        let bands = uniform.cullParams[${CULL_PARAMS.ROOT_BANDS}u];
        // the instance's draw bucket sets its size cut (see OBJECT_BUCKET_SHIFT)
        let bucketRoot = uniform.cullParams[${CULL_PARAMS.BUCKET_ROOT}u];
        let rootMin = bucketRoot[min((object.flags >> ${OBJECT_BUCKET_SHIFT}u) & 3u, 2u)];

        let sphere = meshletPrimitiveData(instance).sphere;
        let center = (meshletInstanceMatrix(instance) * vec4f(sphere.xyz, 1.0)).xyz;
        let radius = sphere.w * object.maxScale;
        if (!meshletInCullVolume(center, radius)) { return; }

        // projected size from the priority origin (the viewing camera, also for shadow views)
        let gap = max(distance(center, priority.xyz) - radius, 1e-4);
        let pixels = radius * priority.w / gap;
        let incumbent = uniform.hasPrevious != 0u &&
            (claimBits[uniform.prevBase + (instance >> 5u)] & (1u << (instance & 31u))) != 0u;
        // an instance admitted last frame stays until it shrinks to the hysteresis fraction of
        // the size cut, so the cut does not pop instances in and out while the camera moves
        if (pixels < select(rootMin, rootMin * bucketRoot.w, incumbent)) {
            atomicAdd(&counters[${MESHLET_COUNTER.ROOT_SUBPIXEL}u], 1u);
            return;
        }
        // wholly behind what last frame drew, as seen from last frame's camera
        let prevCamera = uniform.cullParams[${CULL_PARAMS.PREV_CAMERA}u];
        if (prevCamera.w != 0.0) {
            let prevViewProj = mat4x4f(
                uniform.cullParams[${CULL_PARAMS.PREV_VIEW_PROJ}u], uniform.cullParams[${CULL_PARAMS.PREV_VIEW_PROJ + 1}u],
                uniform.cullParams[${CULL_PARAMS.PREV_VIEW_PROJ + 2}u], uniform.cullParams[${CULL_PARAMS.PREV_VIEW_PROJ + 3}u]);
            if (!hzbVisibleFrom(prevViewProj, prevCamera.xyz, uniform.cullParams[${CULL_PARAMS.PREV_VIEW_DIR}u].xyz, center, radius)) {
                atomicAdd(&counters[${MESHLET_COUNTER.ROOT_OCCLUDED}u], 1u);
                return;
            }
        }
        let bandCount = u32(bands.z);
        var band = u32(clamp(floor(log2(bands.x / max(pixels, 1e-6)) * bands.y), 0.0, f32(bandCount - 1u)));
        if (incumbent) {
            band = max(band, 1u) - 1u;
        }
        let slot = atomicAdd(&counters[${MESHLET_COUNTER.ROOT_CANDIDATES}u], 1u);
        if (slot < uniform.sortedBase) {
            rootList[slot] = instance | (band << ${BAND_SHIFT}u);
            atomicAdd(&rootBands[band], 1u);
        }
    }
`;

/**
 * One thread: turns the per-band counts into each band's start in the band-sorted root list
 * (an exclusive prefix sum) and writes each band's admission dispatch to its own indirect slot,
 * so a band pass only visits its own entries.
 *
 * @ignore
 */
export const meshletRootBandArgsWGSL = /* wgsl */ `
    uniform firstSlot: u32;

    var<storage, read_write> rootBands: array<u32>;
    var<storage, read_write> indirectDispatch: array<u32>;

    ${meshletDispatchArgsWGSL}

    @compute @workgroup_size(1)
    fn main() {
        var start = 0u;
        for (var band = 0u; band < ${MESHLET_ROOT_BANDS}u; band++) {
            let count = rootBands[band];
            rootBands[${2 * MESHLET_ROOT_BANDS}u + band] = start;
            start += count;
            meshletWriteDispatch(uniform.firstSlot + band, (count + ${MESHLET_LIST_WORKGROUP - 1}u) / ${MESHLET_LIST_WORKGROUP}u);
        }
    }
`;

/**
 * One thread per root list entry: copies it into its band's range of the band-sorted list,
 * which follows the classify order list at word `sortedBase`.
 *
 * @ignore
 */
export const meshletRootScatterWGSL = /* wgsl */ `
    uniform sortedBase: u32;

    var<storage, read_write> counters: array<atomic<u32>>;
    var<storage, read_write> rootList: array<u32>;
    var<storage, read_write> rootBands: array<atomic<u32>>;

    ${meshletDispatchGroupWGSL}

    fn meshletItemGroups() -> u32 {
        let count = min(atomicLoad(&counters[${MESHLET_COUNTER.ROOT_CANDIDATES}u]), uniform.sortedBase);
        return (count + ${MESHLET_LIST_WORKGROUP - 1}u) / ${MESHLET_LIST_WORKGROUP}u;
    }

    ${meshletStridedMainWGSL(MESHLET_LIST_WORKGROUP)}

    fn meshletItem(group: vec3u, lane: vec3u) {
        let index = meshletGroupIndex(group) * ${MESHLET_LIST_WORKGROUP}u + lane.x;
        if (index >= min(atomicLoad(&counters[${MESHLET_COUNTER.ROOT_CANDIDATES}u]), uniform.sortedBase)) { return; }
        let entry = rootList[index];
        let band = entry >> ${BAND_SHIFT}u;
        let slot = atomicLoad(&rootBands[${2 * MESHLET_ROOT_BANDS}u + band]) +
            atomicAdd(&rootBands[${MESHLET_ROOT_BANDS}u + band], 1u);
        rootList[uniform.sortedBase + slot] = entry;
    }
`;

/**
 * Root admission for one priority band, one thread per entry of that band's range of the
 * band-sorted root list: each checks its root pages and reserve their root indices and records
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
    ${meshletBucketWGSL}
    // word base of the band-sorted root list
    uniform sortedBase: u32;
    uniform queueCapacity: u32;
    // the share of the index, record and queue capacity roots may take; the rest is kept for
    // the cut to refine what was admitted (see MeshletWorld#rootBudgetFraction)
    uniform rootShare: f32;
    var<storage, read_write> cutQueue: array<vec2u>;
    var<storage, read> cutGroups: array<u32>;
    var<storage, read> residency: array<u32>;
    var<storage, read_write> requests: array<atomic<u32>>;
    var<storage, read_write> claimBits: array<atomic<u32>>;
    var<storage, read_write> counters: array<atomic<u32>>;
    var<storage, read> rootList: array<u32>;
    var<storage, read> rootBands: array<u32>;

    ${meshletRequestPagesWGSL}
    ${meshletDispatchGroupWGSL}

    fn meshletItemGroups() -> u32 {
        return (rootBands[uniform.rootBand] + ${MESHLET_LIST_WORKGROUP - 1}u) / ${MESHLET_LIST_WORKGROUP}u;
    }

    ${meshletStridedMainWGSL(MESHLET_LIST_WORKGROUP)}

    fn meshletItem(group: vec3u, lane: vec3u) {
        let index = meshletGroupIndex(group) * ${MESHLET_LIST_WORKGROUP}u + lane.x;
        if (index >= rootBands[uniform.rootBand]) { return; }
        let entry = rootList[uniform.sortedBase + rootBands[${2 * MESHLET_ROOT_BANDS}u + uniform.rootBand] + index];
        let instance = entry & ${INSTANCE_MASK}u;
        let offset = cutGroups[instance];
        let pageStart = cutGroups[offset + 2u];
        let pageCount = cutGroups[offset + 3u];
        if (!meshletRequestPages(pageStart, pageCount)) {
            atomicAdd(&counters[${MESHLET_COUNTER.ROOT_UNREADY_INSTANCES}u], 1u);
            return;
        }
        // root header: indices at +4, bucket at +5, records at +6 (see MeshletCutData)
        let indices = cutGroups[offset + 4u];
        let bucket = cutGroups[offset + 5u];
        let records = cutGroups[offset + 6u];
        let capacity = meshletBucketCapacity(bucket);
        // Roots stop at their share of each capacity. Bands admit largest on screen first, so
        // what does not fit is the far tail - and the rest stays free for the cut to refine the
        // near instances, rather than every instance in view sitting at its coarsest level.
        let indexLimit = u32(f32(capacity) * uniform.rootShare);
        let recordLimit = u32(f32(uniform.recordCapacity) * uniform.rootShare);
        atomicAdd(&counters[${MESHLET_COUNTER.DEMAND_BASE}u + bucket], indices);
        let indexBefore = atomicAdd(&counters[${CUT_BUDGET.INDICES_BASE}u + bucket], indices);
        let recordBefore = atomicAdd(&counters[${CUT_BUDGET.RECORDS}u], records);
        // the root-owned task list follows the root page list, then the root lists for the
        // emit pass (see meshletCutEmitWGSL)
        let ownedStart = pageStart + pageCount;
        let ownedCount = cutGroups[ownedStart];
        let chunkStart = ownedStart + 1u + ownedCount;
        let chunkCount = cutGroups[chunkStart];
        let queueNeed = ownedCount + chunkCount;
        // An admitted instance must get its queue entries - dropping them afterwards would leave
        // it charged for roots it never draws - so they are charged here with the rest. A charge
        // is a tally, undone on rejection like the others; the entries themselves are appended
        // only once admitted, so their total never passes the limit. Reserving the ranges
        // directly would need a compare-exchange loop, serialising every instance in view on one
        // counter.
        let queueBefore = atomicAdd(&counters[${CUT_BUDGET.QUEUE}u], queueNeed);
        let queueFits = queueBefore + queueNeed <= u32(f32(uniform.queueCapacity) * uniform.rootShare);
        if (!queueFits) {
            atomicAdd(&counters[${MESHLET_COUNTER.CUT_QUEUE_REJECTED}u], queueNeed);
        }
        let fits = queueFits && indexBefore + indices <= indexLimit && recordBefore + records <= recordLimit;
        if (!fits) {
            atomicSub(&counters[${CUT_BUDGET.QUEUE}u], queueNeed);
            atomicSub(&counters[${CUT_BUDGET.INDICES_BASE}u + bucket], indices);
            atomicSub(&counters[${CUT_BUDGET.RECORDS}u], records);
            atomicAdd(&counters[${CUT_BUDGET.REJECTED_RECORDS}u], records);
            atomicAdd(&counters[${MESHLET_COUNTER.ROOT_REJECTED_BASE}u + bucket], indices);
            atomicAdd(&counters[${MESHLET_COUNTER.ROOT_REJECTED_INSTANCES}u], 1u);
            return;
        }
        atomicAdd(&counters[${MESHLET_COUNTER.ROOT_ADMITTED_BASE}u + bucket], indices);
        let admission = uniform.totalPairs + instance;
        atomicOr(&claimBits[admission >> 5u], 1u << (admission & 31u));
        atomicOr(&claimBits[uniform.nextBase + (instance >> 5u)], 1u << (instance & 31u));
        // the tasks that depend on the roots alone, then the root lists - ahead of every stage's
        // tasks, so a full queue drops refinements (which stay coarse) and never roots
        let queueBase = atomicAdd(&counters[${MESHLET_COUNTER.CUT_QUEUE_END}u], queueNeed);
        for (var k = 0u; k < ownedCount; k++) {
            cutQueue[queueBase + k] = vec2u(instance, cutGroups[ownedStart + 1u + k]);
        }
        for (var k = 0u; k < chunkCount; k++) {
            cutQueue[queueBase + ownedCount + k] = vec2u(instance | ${MESHLET_CUT_ROOT_FLAG}u, chunkStart + 1u + k * ${MESHLET_CUT_ROOT_CHUNK + 1}u);
        }
    }
`;
