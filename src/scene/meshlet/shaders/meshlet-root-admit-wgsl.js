import {
    CULL_FLAG_SHADOW_VIEW, CULL_PARAMS, INDIRECT_DISPATCH_U32S, INDIRECT_DRAW_U32S, MESHLET_CELL_SIZE, MESHLET_COUNTER, MESHLET_DISPATCH_WIDTH,
    MESHLET_ROOT_BANDS, OBJECT_FLAG_HIDDEN, OBJECT_FLAG_IMPOSTOR, OBJECT_FLAG_IMPOSTOR_PRIMARY,
    OBJECT_BUCKET_SHIFT, OBJECT_FLAG_NO_SHADOW, OBJECT_IMPOSTOR_LAYER_SHIFT, PAGE_NOT_RESIDENT, PAGE_REQUEST
} from '../constants.js';
import { meshletCullVolumeWGSL, meshletVisibleCellWGSL } from './meshlet-cull-wgsl.js';
import { meshletCutEnqueueWGSL } from './meshlet-cut-wgsl.js';
import { meshletStructsWGSL, meshletObjectDataWGSL } from './meshlet-page-wgsl.js';

// band index in the top bits of a root list entry, instance id below
const BAND_SHIFT = 27;
const INSTANCE_MASK = (1 << BAND_SHIFT) - 1;

/**
 * Root classification, one workgroup per visible cell and one thread per instance of it: drops
 * hidden (and, in a shadow view, non-casting) instances and those outside the frustum, ranks the
 * rest by projected size - the LOD metric - into priority bands, and appends them to the root
 * list for admission. An instance admitted last frame is promoted one band (hysteresis), so
 * admission does not flicker at the budget's edge.
 *
 * Instances of an impostor prototype are decided first, from the whole prototype's sphere: while
 * it projects under the impostor switch size none of them reach the root list, and the primary
 * one appends itself to the impostor list instead (one billboard per tree, drawn by the impostor
 * material). The verdict is stateless and uses the prototype sphere through the shared instance
 * matrix, so every primitive of a tree switches on the same frame.
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
    // per-band entry counts (see ROOT_BANDS_WORDS)
    var<storage, read_write> rootBands: array<atomic<u32>>;
    // instance ids in cell order
    var<storage, read> cellInstances: array<u32>;
    // two per atlas layer: prototype sphere (centre xyz, radius), then the framed box (half width, half height)
    var<storage, read> impostorProtos: array<vec4f>;
    // the billboards to draw: [instance id, fade-in as f32 bits] each
    var<storage, read_write> impostorList: array<u32>;

    ${meshletVisibleCellWGSL}
    ${meshletCullVolumeWGSL}

    @compute @workgroup_size(${MESHLET_CELL_SIZE})
    fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) lane: vec3u) {
        let cell = meshletVisibleCell(group);
        if (cell == 0xFFFFFFFFu) { return; }
        let position = cell * ${MESHLET_CELL_SIZE}u + lane.x;
        if (position >= uniform.instanceCount) { return; }
        let instance = cellInstances[position];
        let object = objectData[instance];
        let shadow = (u32(cullParams[${CULL_PARAMS.STREAMING}u].z) & ${CULL_FLAG_SHADOW_VIEW}u) != 0u;
        if ((object.flags & ${OBJECT_FLAG_HIDDEN}u) != 0u ||
            (shadow && (object.flags & ${OBJECT_FLAG_NO_SHADOW}u) != 0u)) { return; }
        let priority = cullParams[${CULL_PARAMS.ROOT_PRIORITY}u];
        let bands = cullParams[${CULL_PARAMS.ROOT_BANDS}u];
        // the instance's draw bucket sets its size cut (see OBJECT_BUCKET_SHIFT)
        let bucketRoot = cullParams[${CULL_PARAMS.BUCKET_ROOT}u];
        let rootMin = bucketRoot[min((object.flags >> ${OBJECT_BUCKET_SHIFT}u) & 3u, 2u)];

        let impostorParams = cullParams[${CULL_PARAMS.IMPOSTOR}u];
        let impostorSwitch = impostorParams.x;
        if (impostorSwitch > 0.0 && (object.flags & ${OBJECT_FLAG_IMPOSTOR}u) != 0u) {
            let proto = impostorProtos[(object.flags >> ${OBJECT_IMPOSTOR_LAYER_SHIFT}u) * 2u];
            let protoCenter = (object.worldMatrix * vec4f(proto.xyz, 1.0)).xyz;
            let protoRadius = proto.w * object.maxScale;
            let protoGap = max(distance(protoCenter, priority.xyz) - protoRadius, 1e-4);
            let protoPixels = protoRadius * priority.w / protoGap;
            // the billboard fades in over a band above the switch size (dithered, over the meshlets,
            // which are still drawn); from the switch size down it is opaque and the meshlets drop
            let fadeTop = impostorSwitch * (1.0 + impostorParams.y);
            if (protoPixels < fadeTop) {
                // billboards vanish at the same size as any other instance
                if ((object.flags & ${OBJECT_FLAG_IMPOSTOR_PRIMARY}u) != 0u && protoPixels >= rootMin &&
                    meshletInCullVolume(protoCenter, protoRadius)) {
                    let billboard = atomicAdd(&counters[${MESHLET_COUNTER.IMPOSTORS}u], 1u);
                    if (billboard * 2u + 1u < arrayLength(&impostorList)) {
                        impostorList[billboard * 2u] = instance;
                        impostorList[billboard * 2u + 1u] = bitcast<u32>(clamp((fadeTop - protoPixels) / max(fadeTop - impostorSwitch, 1e-3), 0.0, 1.0));
                    }
                }
                if (protoPixels < impostorSwitch) {
                    return;
                }
            }
        }

        let center = (object.worldMatrix * vec4f(object.sphere.xyz, 1.0)).xyz;
        let radius = object.sphere.w * object.maxScale;
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
        let bandCount = u32(bands.z);
        var band = u32(clamp(floor(log2(bands.x / max(pixels, 1e-6)) * bands.y), 0.0, f32(bandCount - 1u)));
        if (incumbent) {
            band = max(band, 1u) - 1u;
        }
        let slot = atomicAdd(&counters[${MESHLET_COUNTER.ROOT_CANDIDATES}u], 1u);
        if (slot < arrayLength(&rootList)) {
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

    @compute @workgroup_size(1)
    fn main() {
        var start = 0u;
        for (var band = 0u; band < ${MESHLET_ROOT_BANDS}u; band++) {
            let count = rootBands[band];
            rootBands[${2 * MESHLET_ROOT_BANDS}u + band] = start;
            start += count;
            let groups = (count + 63u) / 64u;
            let base = (uniform.firstSlot + band) * ${INDIRECT_DISPATCH_U32S}u;
            indirectDispatch[base + 0u] = min(groups, ${MESHLET_DISPATCH_WIDTH}u);
            indirectDispatch[base + 1u] = (groups + ${MESHLET_DISPATCH_WIDTH - 1}u) / ${MESHLET_DISPATCH_WIDTH}u;
            indirectDispatch[base + 2u] = 1u;
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

    @compute @workgroup_size(64)
    fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) lane: vec3u) {
        let index = (group.y * ${MESHLET_DISPATCH_WIDTH}u + group.x) * 64u + lane.x;
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
    uniform indexCapacity0: u32;
    uniform indexCapacity1: u32;
    uniform indexCapacity2: u32;
    // word base of the band-sorted root list
    uniform sortedBase: u32;
    uniform queueCapacity: u32;
    var<storage, read_write> cutQueue: array<vec2u>;
    var<storage, read> cutGroups: array<u32>;
    var<storage, read> residency: array<u32>;
    var<storage, read_write> requests: array<atomic<u32>>;
    var<storage, read_write> claimBits: array<atomic<u32>>;
    // used indices per bucket, used records, then rejected record demand
    var<storage, read_write> cutBudget: array<atomic<u32>>;
    var<storage, read_write> counters: array<atomic<u32>>;
    var<storage, read> rootList: array<u32>;
    var<storage, read> rootBands: array<u32>;

    ${meshletCutEnqueueWGSL}

    @compute @workgroup_size(64)
    fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) lane: vec3u) {
        let index = (group.y * ${MESHLET_DISPATCH_WIDTH}u + group.x) * 64u + lane.x;
        if (index >= rootBands[uniform.rootBand]) { return; }
        let entry = rootList[uniform.sortedBase + rootBands[${2 * MESHLET_ROOT_BANDS}u + uniform.rootBand] + index];
        let instance = entry & ${INSTANCE_MASK}u;
        let offset = cutGroups[instance];
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
        // the root-owned task list follows the root page list
        cutEnqueue(instance, pageStart + pageCount);
    }
`;

/**
 * One thread: writes the impostor draw's indirect arguments - six indices, one instance per
 * billboard the classify appended (clamped to the list's capacity).
 *
 * @ignore
 */
export const meshletImpostorArgsWGSL = /* wgsl */ `
    uniform drawSlot: u32;
    uniform capacity: u32;

    var<storage, read_write> counters: array<u32>;
    var<storage, read_write> indirectDraw: array<u32>;

    @compute @workgroup_size(1)
    fn main() {
        let base = uniform.drawSlot * ${INDIRECT_DRAW_U32S}u;
        indirectDraw[base + 0u] = 6u;
        indirectDraw[base + 1u] = min(counters[${MESHLET_COUNTER.IMPOSTORS}u], uniform.capacity);
        indirectDraw[base + 2u] = 0u;
        indirectDraw[base + 3u] = 0u;
        indirectDraw[base + 4u] = 0u;
    }
`;
