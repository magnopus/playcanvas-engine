/**
 * WGSL compute shaders for the GPU-driven meshlet pipeline. All shaders use the engine's
 * simplified auto-reflected syntax (loose `uniform` declarations and unattributed
 * `var<storage>` resources).
 *
 * The world's buffers are read through the typed views in meshlet-page-wgsl.js
 * (`objectData[instance].flags`, `meshletData[meshlet].page`, ...); their layouts are the
 * OBJECT_DATA / MESHLET_DATA enums in constants.js. The remaining raw buffers:
 * - counters: MESHLET_COUNTER.* — records, then per-bucket cursors, committed
 *   ends and unclamped demand (the demand block survives the phase-2 reset; the CPU grows the
 *   index buffer from it). Bucket order is opaque, opaque-two-sided, masked (MESHLET_BUCKET_*).
 * - cullParams: a uniform array<vec4f>, rows CULL_PARAMS.* (planes, camera, LOD, view-projection,
 *   streaming, view direction). orthoScale > 0 selects the orthographic LOD projection (shadow
 *   cascades); CULL_FLAG_NO_TEXEL_RATE suppresses the texture-mip feedback.
 *   All are float-encoded and read back through u32() where integral, matching the mipCount
 *   convention.
 * - residency: page -> pool slot (PAGE_NOT_RESIDENT when absent); requests: page marks then
 *   texel-rate marks; claimBits / visBits: one bit per (instance, meshlet) pair.
 * - drawn index format: (recordIndex << 8) | meshletLocalVertexIndex
 *
 * @ignore
 */

import {
    CULL_FLAG_NO_TEXEL_RATE, CULL_PARAMS, CUT_BUDGET, INDIRECT_DRAW_U32S, MESHLET_BUCKET_OPAQUE,
    MESHLET_CELL_CULL_WORKGROUP, MESHLET_COUNTER, MESHLET_CULL_SLICE, MESHLET_INDEX_WRITE_WORKGROUP,
    MESHLET_INDEX_WRITE_RECORDS, MESHLET_VISIBLE_CELLS_BASE, OBJECT_FLAG_HAS_TANGENTS, OBJECT_FLAG_HAS_COLORS,
    PAGE_REQUEST, TEXEL_RATE_PER_MIP
} from '../constants.js';
import {
    meshletBucketWGSL, meshletCullParamsWGSL, meshletCullVolumeWGSL, meshletDispatchArgsWGSL, meshletDispatchGroupWGSL,
    meshletHzbVisibleWGSL, meshletStridedMainWGSL
} from './meshlet-cull-common-wgsl.js';
import {
    meshletDataWGSL, meshletObjectDataWGSL, meshletPageLayoutWGSL, meshletRecordsWGSL, meshletStructsWGSL
} from './meshlet-page-wgsl.js';

/**
 * Cell culling: one thread per spatial cell (see MeshletCells). Appends the cells whose
 * bounding sphere touches the frustum to the view's visible-cell list, which the per-instance
 * and per-task passes then dispatch over - one workgroup per visible cell.
 */
export const cellCullWGSL = /* wgsl */ `
    ${meshletCullVolumeWGSL}
    ${meshletDispatchGroupWGSL}
    uniform cellCount : u32;

    var<storage, read> cellBounds : array<vec4f>;
    // per cell, the radius of its largest member
    var<storage, read> cellMaxRadius : array<f32>;
    ${meshletCullParamsWGSL}
    var<storage, read_write> counters : array<atomic<u32>>;

    @compute @workgroup_size(${MESHLET_CELL_CULL_WORKGROUP})
    fn main(@builtin(workgroup_id) group : vec3u, @builtin(local_invocation_id) lane : vec3u) {
        let cell = meshletGroupIndex(group) * ${MESHLET_CELL_CULL_WORKGROUP}u + lane.x;
        if (cell >= uniform.cellCount) {
            return;
        }
        let sphere = cellBounds[cell];
        if (!meshletInCullVolume(sphere.xyz, sphere.w)) {
            return;
        }
        // Root classification drops an instance smaller on screen than its bucket's size cut (a
        // fraction of it for one admitted last frame). Every member lies inside the cell sphere and
        // is no larger than its largest, so if that one at the cell's nearest point is under the
        // smallest such cut, classification would drop them all.
        let bands = uniform.cullParams[${CULL_PARAMS.ROOT_BANDS}u];
        if (bands.w > 0.0) {
            let priority = uniform.cullParams[${CULL_PARAMS.ROOT_PRIORITY}u];
            let gap = max(distance(sphere.xyz, priority.xyz) - sphere.w, 1e-4);
            if (cellMaxRadius[cell] * priority.w / gap < bands.w * uniform.cullParams[${CULL_PARAMS.BUCKET_ROOT}u].w) {
                return;
            }
        }
        let slot = atomicAdd(&counters[${MESHLET_COUNTER.VISIBLE_CELLS}u], 1u);
        atomicStore(&counters[${MESHLET_VISIBLE_CELLS_BASE}u + slot], cell);
    }
`;

/**
 * Writes indirect dispatch args from a counter and the number of items per workgroup.
 * Uses a 2D grid to stay within the per-dimension limit. One thread.
 */
export const dispatchArgsWGSL = /* wgsl */ `
    uniform dispatchSlot : u32;
    uniform itemCapacity : u32;
    uniform counterIndex : u32;
    uniform groupSize : u32;

    var<storage, read_write> counters : array<atomic<u32>>;
    var<storage, read_write> indirectDispatch : array<u32>;

    ${meshletDispatchArgsWGSL}

    @compute @workgroup_size(1)
    fn main() {
        let items = min(atomicLoad(&counters[uniform.counterIndex]), uniform.itemCapacity);
        meshletWriteDispatch(uniform.dispatchSlot, (items + uniform.groupSize - 1u) / uniform.groupSize);
    }
`;

/**
 * Meshlet culling: one thread per meshlet in the compacted resident cut. Tests the frustum
 * sphere, cone backface and occlusion. Survivors reserve an index range in their bucket and
 * append a record.
 */
export const meshletCullWGSL = /* wgsl */ `
    ${meshletCullVolumeWGSL}
    ${meshletStructsWGSL}

    uniform recordCapacity : u32;
    uniform phase : u32; // 0 = single-phase (no occlusion), 1 = draw prev-visible, 2 = newly-visible vs HZB
    ${meshletBucketWGSL}
    ${meshletDispatchGroupWGSL}

    // at most eight storage buffers, the WebGPU guaranteed per-stage minimum (the shader tests check it)
    ${meshletObjectDataWGSL}
    ${meshletDataWGSL}
    var<storage, read> selectedMeshlets : array<vec2u>;
    ${meshletRecordsWGSL('read_write')}

    ${meshletCullParamsWGSL}
    var<storage, read_write> counters : array<atomic<u32>>;
    var<storage, read_write> requests : array<atomic<u32>>;
    var<storage, read_write> visBits : array<atomic<u32>>;
    var hzbTexture : texture_2d<f32>;

    fn viewProj() -> mat4x4f {
        return mat4x4f(uniform.cullParams[${CULL_PARAMS.VIEW_PROJ}u], uniform.cullParams[${CULL_PARAMS.VIEW_PROJ + 1}u], uniform.cullParams[${CULL_PARAMS.VIEW_PROJ + 2}u], uniform.cullParams[${CULL_PARAMS.VIEW_PROJ + 3}u]);
    }

    ${meshletHzbVisibleWGSL}

    // conservative occlusion test of a world-space sphere against the max-depth HZB, from this
    // frame's camera. Returns true when potentially visible.
    fn hzbVisible(center : vec3f, radius : f32, camPos : vec3f) -> bool {
        return hzbVisibleFrom(viewProj(), camPos, uniform.cullParams[${CULL_PARAMS.VIEW_DIR}u].xyz, center, radius);
    }

    fn meshletItemGroups() -> u32 {
        let count = min(atomicLoad(&counters[${MESHLET_COUNTER.SELECTED}u]), uniform.recordCapacity);
        return (count + ${MESHLET_CULL_SLICE - 1}u) / ${MESHLET_CULL_SLICE}u;
    }

    ${meshletStridedMainWGSL(MESHLET_CULL_SLICE)}

    fn meshletItem(workgroupId : vec3u, localId : vec3u) {
        let selectedIndex = meshletGroupIndex(workgroupId) * ${MESHLET_CULL_SLICE}u + localId.x;
        if (selectedIndex >= min(atomicLoad(&counters[${MESHLET_COUNTER.SELECTED}u]), uniform.recordCapacity)) { return; }
        let selected = selectedMeshlets[selectedIndex];
        let instance = selected.x;
        let localIndex = selected.y;
        let primitive = meshletPrimitiveData(instance);
        let meshlet = primitive.firstMeshlet + localIndex;

        let worldMatrix = meshletInstanceMatrix(instance);
        let maxScale = objectData[instance].maxScale;
        let camPos = uniform.cullParams[${CULL_PARAMS.CAMERA}u].xyz;
        let projScale = uniform.cullParams[${CULL_PARAMS.CAMERA}u].w;
        let orthoScale = uniform.cullParams[${CULL_PARAMS.STREAMING}u].y;
        let cullFlags = u32(uniform.cullParams[${CULL_PARAMS.STREAMING}u].z);

        // frustum: world-space meshlet bounding sphere
        let localSphere = meshletData[meshlet].sphere;
        let radius = localSphere.w * maxScale;
        let center = (worldMatrix * vec4f(localSphere.xyz, 1.0)).xyz;
        if (!meshletInCullVolume(center, radius)) {
            return;
        }

        let bucket = meshletBucket(meshletData[meshlet].flags);

        // cone backface culling - only sound when the bucket actually culls back faces, and
        // only for instances whose transform keeps angles (rotation, uniform scale, mirroring):
        // non-uniform scale or shear bends the normals, so the baked cone no longer bounds them
        if (bucket == ${MESHLET_BUCKET_OPAQUE}u) {
            let coneCutoff = meshletData[meshlet].coneCutoff;
            let m0 = worldMatrix[0].xyz;
            let m1 = worldMatrix[1].xyz;
            let m2 = worldMatrix[2].xyz;
            let l0 = dot(m0, m0);
            let l1 = dot(m1, m1);
            let l2 = dot(m2, m2);
            let lMax = max(l0, max(l1, l2));
            let conformal = lMax <= min(l0, min(l1, l2)) * 1.002 &&
                            abs(dot(m0, m1)) + abs(dot(m1, m2)) + abs(dot(m2, m0)) <= lMax * 1e-3;
            if (coneCutoff < 1.0 && conformal) {
                let apex = (worldMatrix * vec4f(meshletData[meshlet].coneApex, 1.0)).xyz;
                let axis = normalize((worldMatrix * vec4f(meshletConeAxis(meshlet), 0.0)).xyz);
                // under ortho every ray is parallel, and apex - camPos would be a catastrophic
                // ~1e6 subtraction anyway - use the view direction directly
                let view = select(apex - camPos, uniform.cullParams[${CULL_PARAMS.VIEW_DIR}u].xyz, orthoScale > 0.0);
                if (dot(normalize(view), axis) >= coneCutoff + 1e-3 && dot(view, view) > 1e-10) {
                    return;
                }
            }
        }

        // two-phase occlusion via the persistent visibility bit of this (instance, meshlet)
        // pair. Phase 1 draws what the bit says was visible; phase 2 tests survivors against the
        // HZB built from phase 1's depth, draws only the newly visible remainder and updates the
        // bit. Phase 0 is the single-phase path with no occlusion.
        //
        // The bit means "visible the last time this pair was tested", not "visible last frame":
        // a pair that leaves the frustum or the LOD cut keeps its bit untouched, so it draws in
        // phase 1 the moment it comes back (no HZB round trip) at the cost of one possibly
        // occluded draw until phase 2 clears it again.
        let pairBit = objectData[instance].firstPairBit + localIndex;
        let pairMask = 1u << (pairBit & 31u);
        if (uniform.phase == 1u) {
            if ((atomicLoad(&visBits[pairBit >> 5u]) & pairMask) == 0u) {
                return;
            }
        } else if (uniform.phase == 2u) {
            let visibleNow = hzbVisible(center, radius, camPos);
            let wasVisible = (atomicLoad(&visBits[pairBit >> 5u]) & pairMask) != 0u;
            if (visibleNow && !wasVisible) {
                atomicOr(&visBits[pairBit >> 5u], pairMask);
            } else if (!visibleNow && wasVisible) {
                atomicAnd(&visBits[pairBit >> 5u], ~pairMask);
            }
            if (!visibleNow || wasVisible) {
                return;
            }
        }

        // texture-mip feedback: the texel rate (see TEXEL_RATE_PER_MIP) this drawn cluster's
        // material needs, atomicMax'd per material after the page marks. Off for orthographic
        // (shadow) views, whose light-space rate would poison the camera's texture streaming.
        let uvExtentWord = select(meshletData[meshlet].uvExtent, 0u, (cullFlags & ${CULL_FLAG_NO_TEXEL_RATE}u) != 0u);
        if (uvExtentWord != 0u) {
            let uvExtent = unpack2x16float(uvExtentWord);
            let uvSpan = max(max(uvExtent.x, uvExtent.y), 1e-4);
            // measured from the cluster's near edge, not its centre: a large cluster the camera
            // stands at (a roof, a floor) is far larger on screen than its centre distance says,
            // and rating it by the centre would starve its texture right where it is looked at closest
            let dist = max(distance(center, camPos) - radius, 0.05);
            let screenPx = 2.0 * radius * projScale / dist;
            // +1 so a mark of 0 means "not seen this frame"; 4095 caps the mark at 12 bits
            let texelRate = clamp(u32(${TEXEL_RATE_PER_MIP}.0 * log2(max(screenPx / uvSpan, 1.0))) + 1u, 1u, 4095u);
            // a handful of material words take a mark from every drawn cluster: read first so
            // only a higher rate pays for the contended atomic
            let mark = u32(uniform.cullParams[${CULL_PARAMS.STREAMING}u].x) + primitive.material;
            if (atomicLoad(&requests[mark]) < texelRate) { atomicMax(&requests[mark], texelRate); }
        }

        // keeps the drawn page off the eviction list
        let page = meshletData[meshlet].page;
        if (atomicLoad(&requests[page]) < ${PAGE_REQUEST.USED}u) { atomicMax(&requests[page], ${PAGE_REQUEST.USED}u); }

        let triangleCount = meshletData[meshlet].triangleCount;
        if (triangleCount == 0u) {
            return;
        }

        // Reserve only the drawn range here. The selection pass already reported capacity
        // demand for the complete cut, independently of either phase's occlusion results.
        let indexNeed = triangleCount * 3u;
        let capacity = meshletBucketCapacity(bucket);
        let recordIndex = atomicAdd(&counters[${MESHLET_COUNTER.RECORDS}u], 1u);
        if (recordIndex >= uniform.recordCapacity) {
            return;
        }
        let cursor = atomicAdd(&counters[${MESHLET_COUNTER.CURSOR_BASE}u + bucket], indexNeed);
        if (cursor + indexNeed > capacity) {
            records[recordIndex] = MeshletRecord(instance, meshlet, 0xffffffffu, bucket);
            return;
        }
        atomicMax(&counters[${MESHLET_COUNTER.COMMITTED_BASE}u + bucket], cursor + indexNeed);
        let baseIndexOffset = meshletBucketBase(bucket) + cursor;

        records[recordIndex] = MeshletRecord(instance, meshlet, baseIndexOffset, bucket);
    }
`;

/**
 * Finalizes the frame's GPU-written arguments: one DrawIndexedIndirectArgs per bucket and the
 * index-write pass's indirect dispatch args. One thread.
 */
export const finalizeArgsWGSL = /* wgsl */ `
    uniform drawSlot0 : u32;
    uniform drawSlot1 : u32;
    uniform drawSlot2 : u32;
    uniform dispatchSlot : u32;
    uniform recordCapacity : u32;
    uniform indexCapacity0 : u32;
    uniform indexCapacity1 : u32;

    var<storage, read_write> counters : array<atomic<u32>>;
    var<storage, read_write> indirectDraw : array<u32>;
    var<storage, read_write> indirectDispatch : array<u32>;

    ${meshletDispatchArgsWGSL}

    fn writeDraw(slot : u32, indexCount : u32, firstIndex : u32) {
        let base = slot * ${INDIRECT_DRAW_U32S}u;
        indirectDraw[base + 0u] = indexCount;
        indirectDraw[base + 1u] = 1u;
        indirectDraw[base + 2u] = firstIndex;
        indirectDraw[base + 3u] = 0u;
        indirectDraw[base + 4u] = 0u;
    }

    @compute @workgroup_size(1)
    fn main() {
        // committed ends are contiguous: once a reservation overflows, every later reservation
        // starts beyond it and overflows too, so [0, committedEnd) is fully written
        writeDraw(uniform.drawSlot0, atomicLoad(&counters[${MESHLET_COUNTER.COMMITTED_BASE}u]), 0u);
        writeDraw(uniform.drawSlot1, atomicLoad(&counters[${MESHLET_COUNTER.COMMITTED_BASE + 1}u]), uniform.indexCapacity0);
        writeDraw(uniform.drawSlot2, atomicLoad(&counters[${MESHLET_COUNTER.COMMITTED_BASE + 2}u]),
                  uniform.indexCapacity0 + uniform.indexCapacity1);

        // record demand is the whole cut's, charged and rejected, not just what this phase drew
        atomicMax(&counters[${MESHLET_COUNTER.RECORD_DEMAND}u],
                  atomicLoad(&counters[${CUT_BUDGET.RECORDS}u]) + atomicLoad(&counters[${CUT_BUDGET.REJECTED_RECORDS}u]));
        let recordCount = min(atomicLoad(&counters[${MESHLET_COUNTER.RECORDS}u]), uniform.recordCapacity);
        atomicAdd(&counters[${MESHLET_COUNTER.RENDERED}u], recordCount);
        // index write: MESHLET_INDEX_WRITE_RECORDS records per workgroup
        meshletWriteDispatch(uniform.dispatchSlot, (recordCount + ${MESHLET_INDEX_WRITE_RECORDS - 1}u) / ${MESHLET_INDEX_WRITE_RECORDS}u);
    }
`;

/**
 * Index generation: each workgroup expands MESHLET_INDEX_WRITE_RECORDS surviving records, a
 * slice of threads per record, from the meshlet's u8 corner stream in the page pool into the
 * global index buffer as (recordIndex << 8) | localVertex.
 */
export const indexWriteWGSL = /* wgsl */ `
    ${meshletStructsWGSL}

    uniform recordCapacity : u32;
    uniform pageSizeWords : u32;

    ${meshletDataWGSL}
    ${meshletObjectDataWGSL}
    ${meshletRecordsWGSL('read')}

    var<storage, read> pagePool : array<u32>;
    var<storage, read> residency : array<u32>;
    var<storage, read_write> counters : array<atomic<u32>>;
    var<storage, read_write> drawIndices : array<u32>;

    ${meshletPageLayoutWGSL}
    ${meshletDispatchGroupWGSL}

    fn meshletItemGroups() -> u32 {
        let recordCount = min(atomicLoad(&counters[${MESHLET_COUNTER.RECORDS}u]), uniform.recordCapacity);
        return (recordCount + ${MESHLET_INDEX_WRITE_RECORDS - 1}u) / ${MESHLET_INDEX_WRITE_RECORDS}u;
    }

    ${meshletStridedMainWGSL(MESHLET_INDEX_WRITE_WORKGROUP)}

    fn meshletItem(workgroupId : vec3u, localId : vec3u) {
        let lanes = ${MESHLET_INDEX_WRITE_WORKGROUP / MESHLET_INDEX_WRITE_RECORDS}u;
        let lane = localId.x % lanes;
        let recordIndex = meshletGroupIndex(workgroupId) * ${MESHLET_INDEX_WRITE_RECORDS}u + localId.x / lanes;
        let recordCount = min(atomicLoad(&counters[${MESHLET_COUNTER.RECORDS}u]), uniform.recordCapacity);
        if (recordIndex >= recordCount) {
            return;
        }

        let instance = records[recordIndex].instance;
        let meshlet = records[recordIndex].meshlet;
        let baseIndexOffset = records[recordIndex].baseIndexOffset;
        if (baseIndexOffset == 0xffffffffu) {
            if (lane == 0u) {
                atomicSub(&counters[${MESHLET_COUNTER.RENDERED}u], 1u);
            }
            return;
        }
        let triangleOffset = meshletData[meshlet].triangleOffset;
        let cornerCount = meshletData[meshlet].triangleCount * 3u;
        let page = meshletData[meshlet].page;

        // per-instance page attribute layout (resources may differ)
        let uvFloatsPerVertex = meshletPrimitiveData(instance).uvFloatsPerVertex;
        let hasTangents = (objectData[instance].flags & ${OBJECT_FLAG_HAS_TANGENTS}u) != 0u;
        let hasColors = (objectData[instance].flags & ${OBJECT_FLAG_HAS_COLORS}u) != 0u;

        // the u8 triangle corner stream follows the page's meshlet-vertex table (shared layout
        // in meshlet-page-wgsl.js)
        let pageBase = residency[page] * uniform.pageSizeWords;
        let pageLayout = meshletPageLayout(pageBase, hasTangents, uvFloatsPerVertex, hasColors);
        let triangleStreamByte = (pageLayout.meshletVertexBase + pageLayout.meshletVertexCount) * 4u + triangleOffset;
        let worldMatrix = meshletInstanceMatrix(instance);
        let mirrored = dot(cross(worldMatrix[0].xyz, worldMatrix[1].xyz), worldMatrix[2].xyz) < 0.0;

        for (var corner = lane; corner < cornerCount; corner += lanes) {
            let triangleCorner = corner % 3u;
            let sourceCorner = select(corner, corner - triangleCorner + (3u - triangleCorner) % 3u, mirrored);
            let byteIndex = triangleStreamByte + sourceCorner;
            let localVert = (pagePool[byteIndex >> 2u] >> ((byteIndex & 3u) * 8u)) & 0xFFu;
            drawIndices[baseIndexOffset + corner] = (recordIndex << 8u) | localVert;
        }
    }
`;

/**
 * Resets the record/index counters between the phase-1 and phase-2 cull dispatches, keeping the
 * work item count. Runs in-encoder (queue.writeBuffer would land before the whole frame). One
 * thread.
 */
export const resetPhase2WGSL = /* wgsl */ `
    var<storage, read_write> counters : array<atomic<u32>>;

    @compute @workgroup_size(1)
    fn main() {
        // the record count, the per-bucket cursors and the per-bucket committed ends. The
        // unclamped demand counters deliberately survive: the CPU reads them back to size the
        // index buffer and must see the frame's whole demand, not just phase 2's.
        atomicStore(&counters[${MESHLET_COUNTER.RECORDS}u], 0u);
        for (var i = ${MESHLET_COUNTER.CURSOR_BASE}u; i < ${MESHLET_COUNTER.DEMAND_BASE}u; i++) {
            atomicStore(&counters[i], 0u);
        }
    }
`;
