/**
 * WGSL compute shaders for the GPU-driven meshlet pipeline. All shaders use the engine's
 * simplified auto-reflected syntax (loose `uniform` declarations and unattributed
 * `var<storage>` resources).
 *
 * The world's buffers are read through the typed views in meshlet-page-wgsl.js
 * (`objectData[instance].sphere`, `meshletData[meshlet].page`, ...); their layouts are the
 * OBJECT_DATA / MESHLET_DATA enums in constants.js. The remaining raw buffers:
 * - counters: MESHLET_COUNTER.* — work items, records, then per-bucket cursors, committed
 *   ends and unclamped demand (the demand block survives the phase-2 reset; the CPU grows the
 *   index buffer from it). Bucket order is opaque, opaque-two-sided, masked (MESHLET_BUCKET_*).
 * - cullParams: array<vec4f>, rows CULL_PARAMS.* (planes, camera, LOD, view-projection,
 *   streaming, view direction). orthoScale > 0 selects the orthographic LOD projection (shadow
 *   cascades); CULL_FLAG_NO_TEXEL_RATE suppresses the texture-mip feedback. Both are
 *   float-encoded and read back through u32() where integral, matching the mipCount convention.
 * - residency: page -> pool slot (PAGE_NOT_RESIDENT when absent); requests: page marks then
 *   texel-rate marks; claimBits / visBits: one bit per (instance, meshlet) pair.
 * - drawn index format: (recordIndex << 8) | meshletLocalVertexIndex
 *
 * @ignore
 */

import {
    CULL_FLAG_NO_TEXEL_RATE, CULL_PARAMS, INDIRECT_DISPATCH_U32S, INDIRECT_DRAW_U32S, MESHLET_BUCKET_MASKED,
    MESHLET_BUCKET_OPAQUE, MESHLET_BUCKET_OPAQUE_TWO_SIDED, MESHLET_COUNTER, MESHLET_CULL_SLICE,
    MESHLET_DISPATCH_WIDTH, MESHLET_FLAG_ALPHA_MASKED, MESHLET_FLAG_TWO_SIDED, MESHLET_INDEX_WRITE_WORKGROUP,
    MESHLET_INSTANCE_CULL_WORKGROUP, MESHLET_NO_PARENT, OBJECT_FLAG_HAS_TANGENTS, OBJECT_FLAG_HIDDEN,
    PAGE_NOT_RESIDENT, PAGE_REQUEST, TEXEL_RATE_PER_MIP
} from '../constants.js';
import {
    meshletDataWGSL, meshletObjectDataWGSL, meshletPageLayoutWGSL, meshletRecordsWGSL, meshletStructsWGSL,
    meshletWorkItemsWGSL
} from './meshlet-page-wgsl.js';

/**
 * Instance culling: one thread per instance. World-sphere frustum test, then fan the instance's
 * meshlets out into fixed-size slice work items with a single atomicAdd.
 */
export const instanceCullWGSL = /* wgsl */ `
    ${meshletStructsWGSL}

    uniform instanceCount : u32;
    uniform workItemCapacity : u32;

    ${meshletObjectDataWGSL}
    ${meshletWorkItemsWGSL('read_write')}

    var<storage, read> cullParams : array<vec4f>;
    var<storage, read_write> counters : array<atomic<u32>>;

    @compute @workgroup_size(${MESHLET_INSTANCE_CULL_WORKGROUP})
    fn main(@builtin(global_invocation_id) gid : vec3u) {
        let instance = gid.x;
        if (instance >= uniform.instanceCount) {
            return;
        }
        if ((objectData[instance].flags & ${OBJECT_FLAG_HIDDEN}u) != 0u) {
            return;
        }

        let worldMatrix = objectData[instance].worldMatrix;
        let localSphere = objectData[instance].sphere;
        let radius = localSphere.w * objectData[instance].maxScale;
        let center = (worldMatrix * vec4f(localSphere.xyz, 1.0)).xyz;

        for (var planeIndex = 0u; planeIndex < ${CULL_PARAMS.PLANE_COUNT}u; planeIndex++) {
            let plane = cullParams[${CULL_PARAMS.PLANES}u + planeIndex];
            if (dot(plane.xyz, center) + plane.w < -radius) {
                return;
            }
        }

        let meshletCount = objectData[instance].meshletCount;
        let sliceCount = (meshletCount + ${MESHLET_CULL_SLICE - 1}u) / ${MESHLET_CULL_SLICE}u;
        let firstItem = atomicAdd(&counters[${MESHLET_COUNTER.WORK_ITEMS}u], sliceCount);
        for (var slice = 0u; slice < sliceCount; slice++) {
            let item = firstItem + slice;
            if (item < uniform.workItemCapacity) {
                workItems[item] = MeshletWorkItem(instance, slice * ${MESHLET_CULL_SLICE}u);
            }
        }
    }
`;

/**
 * Writes the meshlet-cull pass's indirect dispatch args (2D grid to dodge the 65535 per-dimension
 * limit) from the work item counter. One thread.
 */
export const dispatchArgsWGSL = /* wgsl */ `
    uniform dispatchSlot : u32;
    uniform workItemCapacity : u32;

    var<storage, read_write> counters : array<atomic<u32>>;
    var<storage, read_write> indirectDispatch : array<u32>;

    @compute @workgroup_size(1)
    fn main() {
        let count = min(atomicLoad(&counters[${MESHLET_COUNTER.WORK_ITEMS}u]), uniform.workItemCapacity);
        let base = uniform.dispatchSlot * ${INDIRECT_DISPATCH_U32S}u;
        indirectDispatch[base + 0u] = min(count, ${MESHLET_DISPATCH_WIDTH}u);
        indirectDispatch[base + 1u] = (count + ${MESHLET_DISPATCH_WIDTH - 1}u) / ${MESHLET_DISPATCH_WIDTH}u;
        indirectDispatch[base + 2u] = 1u;
    }
`;

/**
 * Meshlet culling: one workgroup per work item, one thread per meshlet in the slice. Frustum
 * sphere, DAG LOD cut (crack-free via group-shared bounds), cone backface. Survivors reserve an
 * index range in their bucket and append a record.
 */
export const meshletCullWGSL = /* wgsl */ `
    ${meshletStructsWGSL}

    uniform workItemCapacity : u32;
    uniform recordCapacity : u32;
    uniform indexCapacity0 : u32;
    uniform indexCapacity1 : u32;
    uniform indexCapacity2 : u32;
    uniform phase : u32; // 0 = single-phase (no occlusion), 1 = draw prev-visible, 2 = newly-visible vs HZB

    // the buckets are contiguous ranges of one index buffer, in bucket order
    fn bucketCapacity(bucket : u32) -> u32 {
        return select(select(uniform.indexCapacity0, uniform.indexCapacity1, bucket == 1u),
                      uniform.indexCapacity2, bucket == 2u);
    }

    fn bucketBase(bucket : u32) -> u32 {
        return select(select(0u, uniform.indexCapacity0, bucket == 1u),
                      uniform.indexCapacity0 + uniform.indexCapacity1, bucket == 2u);
    }

    // Ten storage buffers in total (objectData, meshletData, workItems and records come from
    // the typed views): exactly WebGPU's default maxStorageBuffersPerShaderStage. An eleventh
    // makes pipeline creation fail on adapters at the default limit, and the symptom is a blank
    // meshlet layer with no error - fold new data into an existing buffer instead (the
    // texel-rate marks live in requests for this reason). meshlet-cull-shaders.test.mjs pins
    // the count.
    ${meshletObjectDataWGSL}
    ${meshletDataWGSL}
    ${meshletWorkItemsWGSL('read')}
    ${meshletRecordsWGSL('read_write')}

    var<storage, read> cullParams : array<vec4f>;
    var<storage, read_write> counters : array<atomic<u32>>;
    var<storage, read> residency : array<u32>;
    var<storage, read_write> requests : array<atomic<u32>>;
    var<storage, read_write> claimBits : array<atomic<u32>>;
    var<storage, read_write> visBits : array<atomic<u32>>;
    var hzbTexture : texture_2d<f32>;

    // Projected screen-space error in pixels of an error value over a bounding sphere.
    //
    // Under an orthographic projection there is no foreshortening, so the projected size is
    // distance-independent and the whole cut degenerates to a plain error threshold scaled by
    // the light-space texel rate. This branch is mandatory for shadow cascades, not an
    // optimisation: the directional shadow camera sits ~1e6 units back, which drives the
    // perspective form to ~0 for every cluster and silently culls the entire scene.
    //
    // BOTH call sites (parentTooCoarse, clusterFits) must pass the same orthoScale. That is
    // what keeps the cut crack-free - siblings share group bounds and error, so as long as
    // they evaluate the same monotonic function they agree on the cut.
    fn projectError(error : f32, center : vec3f, radius : f32, camPos : vec3f, projScale : f32, orthoScale : f32) -> f32 {
        if (orthoScale > 0.0) {
            return error * orthoScale;
        }
        let dist = max(distance(camPos, center) - radius, 1e-5);
        return error * projScale / dist;
    }

    fn viewProj() -> mat4x4f {
        return mat4x4f(cullParams[${CULL_PARAMS.VIEW_PROJ}u], cullParams[${CULL_PARAMS.VIEW_PROJ + 1}u], cullParams[${CULL_PARAMS.VIEW_PROJ + 2}u], cullParams[${CULL_PARAMS.VIEW_PROJ + 3}u]);
    }

    // conservative occlusion test of a world-space sphere against the max-depth HZB.
    // Returns true when potentially visible.
    fn hzbVisible(center : vec3f, radius : f32, camPos : vec3f) -> bool {
        let toCenter = center - camPos;
        let dist = length(toCenter);
        if (dist <= radius * 1.05) {
            return true; // camera inside or at the sphere
        }

        let viewProjMatrix = viewProj();

        // nearest point of the sphere toward the camera - the sphere's minimal depth
        let nearestPoint = center - (toCenter / dist) * radius;
        let nearestClip = viewProjMatrix * vec4f(nearestPoint, 1.0);
        if (nearestClip.w <= 1e-5) {
            return true; // crosses the near plane
        }
        let sphereDepth = nearestClip.z / nearestClip.w;

        // screen rect from camera-facing billboard corners
        let right = normalize(vec3f(viewProjMatrix[0].x, viewProjMatrix[1].x, viewProjMatrix[2].x));
        let up = normalize(vec3f(viewProjMatrix[0].y, viewProjMatrix[1].y, viewProjMatrix[2].y));
        var ndcMin = vec2f(1e30);
        var ndcMax = vec2f(-1e30);
        for (var corner = 0u; corner < 4u; corner++) {
            let signX = select(-1.0, 1.0, (corner & 1u) != 0u);
            let signY = select(-1.0, 1.0, (corner & 2u) != 0u);
            let clip = viewProjMatrix * vec4f(center + right * (radius * signX) + up * (radius * signY), 1.0);
            if (clip.w <= 1e-5) {
                return true;
            }
            let ndc = clip.xy / clip.w;
            ndcMin = min(ndcMin, ndc);
            ndcMax = max(ndcMax, ndc);
        }

        // ndc -> uv (y down)
        let uvMin = clamp(vec2f(ndcMin.x, -ndcMax.y) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));
        let uvMax = clamp(vec2f(ndcMax.x, -ndcMin.y) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));

        let hzbSize = cullParams[${CULL_PARAMS.LOD}u].yz;
        let rectPx = (uvMax - uvMin) * hzbSize;
        let mipCount = u32(cullParams[${CULL_PARAMS.LOD}u].w);
        // pick the mip where the rect spans at most ~2 texels, test a 2x2 footprint
        let mip = min(u32(max(ceil(log2(max(max(rectPx.x, rectPx.y), 1.0))) - 1.0, 0.0)), mipCount - 1u);
        let mipSize = vec2f(f32(max(u32(hzbSize.x) >> mip, 1u)), f32(max(u32(hzbSize.y) >> mip, 1u)));
        let texelMin = vec2u(clamp(uvMin * mipSize, vec2f(0.0), mipSize - 1.0));
        let texelMax = vec2u(clamp(uvMax * mipSize, vec2f(0.0), mipSize - 1.0));
        // the four corners of the 2x2 footprint; the HZB is a max-depth pyramid, so the
        // farthest of them is the conservative occluder depth
        let depth00 = textureLoad(hzbTexture, vec2i(i32(texelMin.x), i32(texelMin.y)), i32(mip)).x;
        let depth10 = textureLoad(hzbTexture, vec2i(i32(texelMax.x), i32(texelMin.y)), i32(mip)).x;
        let depth01 = textureLoad(hzbTexture, vec2i(i32(texelMin.x), i32(texelMax.y)), i32(mip)).x;
        let depth11 = textureLoad(hzbTexture, vec2i(i32(texelMax.x), i32(texelMax.y)), i32(mip)).x;
        let occluderDepth = max(max(depth00, depth10), max(depth01, depth11));
        return sphereDepth <= occluderDepth + 1e-5;
    }

    @compute @workgroup_size(${MESHLET_CULL_SLICE})
    fn main(@builtin(workgroup_id) workgroupId : vec3u, @builtin(local_invocation_id) localId : vec3u) {
        let itemIndex = workgroupId.y * ${MESHLET_DISPATCH_WIDTH}u + workgroupId.x;
        let workItemCount = min(atomicLoad(&counters[${MESHLET_COUNTER.WORK_ITEMS}u]), uniform.workItemCapacity);
        if (itemIndex >= workItemCount) {
            return;
        }

        let instance = workItems[itemIndex].instance;
        let localIndex = workItems[itemIndex].sliceStart + localId.x;
        if (localIndex >= objectData[instance].meshletCount) {
            return;
        }

        let firstMeshlet = objectData[instance].firstMeshlet;
        let meshlet = firstMeshlet + localIndex;
        if (meshletData[meshlet].triangleCount == 0u) {
            return; // cull-to-empty synthetic root
        }

        let worldMatrix = objectData[instance].worldMatrix;
        let maxScale = objectData[instance].maxScale;
        let camPos = cullParams[${CULL_PARAMS.CAMERA}u].xyz;
        let projScale = cullParams[${CULL_PARAMS.CAMERA}u].w;
        let threshold = cullParams[${CULL_PARAMS.LOD}u].x;
        let orthoScale = cullParams[${CULL_PARAMS.STREAMING}u].y;
        let cullFlags = u32(cullParams[${CULL_PARAMS.STREAMING}u].z);

        // frustum: world-space meshlet bounding sphere
        let localSphere = meshletData[meshlet].sphere;
        let radius = localSphere.w * maxScale;
        let center = (worldMatrix * vec4f(localSphere.xyz, 1.0)).xyz;
        for (var planeIndex = 0u; planeIndex < ${CULL_PARAMS.PLANE_COUNT}u; planeIndex++) {
            let plane = cullParams[${CULL_PARAMS.PLANES}u + planeIndex];
            if (dot(plane.xyz, center) + plane.w < -radius) {
                return;
            }
        }

        // DAG LOD cut: draw when the parent is too coarse and this cluster fits the target.
        // Sibling groups share parent bounds/error and sharedSiblingsBounds, so all siblings
        // agree on the decision - the cut is crack-free.
        let parentSphere = meshletData[meshlet].parentSphere;
        let parentCenter = (worldMatrix * vec4f(parentSphere.xyz, 1.0)).xyz;
        let parentTooCoarse = projectError(meshletData[meshlet].parentError * maxScale, parentCenter, parentSphere.w * maxScale, camPos, projScale, orthoScale) > threshold;
        if (!parentTooCoarse) {
            return;
        }

        let groupSphere = meshletData[meshlet].groupSphere;
        let groupCenter = (worldMatrix * vec4f(groupSphere.xyz, 1.0)).xyz;
        let clusterFits = projectError(meshletData[meshlet].clusterError * maxScale, groupCenter, groupSphere.w * maxScale, camPos, projScale, orthoScale) <= threshold;
        if (!clusterFits) {
            return;
        }

        let flags = meshletData[meshlet].flags;
        let alphaMasked = (flags & ${MESHLET_FLAG_ALPHA_MASKED}u) != 0u;
        let twoSided = (flags & ${MESHLET_FLAG_TWO_SIDED}u) != 0u;
        // bucket order: opaque, opaque two-sided, masked (MESHLET_BUCKET_* in constants.js)
        let bucket = select(select(${MESHLET_BUCKET_OPAQUE}u, ${MESHLET_BUCKET_OPAQUE_TWO_SIDED}u, twoSided), ${MESHLET_BUCKET_MASKED}u, alphaMasked);

        // cone backface culling - only sound when the bucket actually culls back faces
        if (bucket == ${MESHLET_BUCKET_OPAQUE}u) {
            let coneCutoff = meshletData[meshlet].coneCutoff;
            if (coneCutoff < 1.0) {
                let apex = (worldMatrix * vec4f(meshletData[meshlet].coneApex, 1.0)).xyz;
                let axis = normalize((worldMatrix * vec4f(meshletConeAxis(meshlet), 0.0)).xyz);
                // under ortho every ray is parallel, and apex - camPos would be a catastrophic
                // ~1e6 subtraction anyway - use the view direction directly
                let view = select(apex - camPos, cullParams[${CULL_PARAMS.VIEW_DIR}u].xyz, orthoScale > 0.0);
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
        // The bit means "visible the last time this pair was TESTED", not "visible last frame":
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

        // texture-mip feedback: this cluster is in the drawn set, so record the texel rate its
        // material would need - texelRate = TEXEL_RATE_PER_MIP * log2(screen pixels / baked uv extent), i.e.
        // sixteenths of a mip level below the texture's top, atomicMax'd per material into the
        // texelRateMarks region of the requests buffer (after the page marks). The CPU converts
        // it to a desired source mip per texture. uvExtent 0 = no UVs; those
        // materials stay tail-only.
        // Suppressed for orthographic (shadow) views: screenPx below is a perspective texel
        // rate, meaningless here, and the marks are atomicMax'd into the WORLD-SHARED requests
        // buffer - a light-space rate would poison the camera's texture streaming. Page
        // residency marks below stay on, so off-screen casters still stream their geometry.
        let uvExtentWord = select(meshletData[meshlet].uvExtent, 0u, (cullFlags & ${CULL_FLAG_NO_TEXEL_RATE}u) != 0u);
        if (uvExtentWord != 0u) {
            let uvExtent = unpack2x16float(uvExtentWord);
            let uvSpan = max(max(uvExtent.x, uvExtent.y), 1e-4);
            let dist = max(distance(center, camPos), 1e-5);
            let screenPx = 2.0 * radius * projScale / dist;
            // +1 so a mark of 0 means "not seen this frame"; 4095 caps the mark at 12 bits
            let texelRate = clamp(u32(${TEXEL_RATE_PER_MIP}.0 * log2(max(screenPx / uvSpan, 1.0))) + 1u, 1u, 4095u);
            let texelRateMarkBase = u32(cullParams[${CULL_PARAMS.STREAMING}u].x);
            atomicMax(&requests[texelRateMarkBase + objectData[instance].material], texelRate);
        }

        // residency resolve: when the desired cluster's page is not resident, mark it wanted and
        // walk the DAG parent chain to the nearest resident ancestor - the surface renders
        // momentarily coarser, never holed. Root pages are pinned, so the walk terminates.
        var chosen = meshlet;
        var page = meshletData[meshlet].page;
        var walkDepth = 0u;
        loop {
            if (residency[page] != ${PAGE_NOT_RESIDENT}u) {
                // mark the resident page as used this frame, so the CPU LRU sees hot pages.
                // A page is either resident (USED) or missing (MISSING), never both.
                atomicStore(&requests[page], ${PAGE_REQUEST.USED}u);
                break;
            }
            atomicStore(&requests[page], ${PAGE_REQUEST.MISSING}u);
            let parent = meshletData[chosen].parent;
            if (parent == ${MESHLET_NO_PARENT}u || walkDepth >= 32u) {
                return; // no resident ancestor (roots not yet loaded)
            }
            chosen = firstMeshlet + parent;
            page = meshletData[chosen].page;
            walkDepth++;
        }
        if (chosen != meshlet) {
            // several missing siblings resolve to the same ancestor - claim it so one thread emits
            let chosenPairBit = objectData[instance].firstPairBit + (chosen - firstMeshlet);
            let mask = 1u << (chosenPairBit & 31u);
            if ((atomicOr(&claimBits[chosenPairBit >> 5u], mask) & mask) != 0u) {
                return;
            }
        }

        let chosenTriangleCount = meshletData[chosen].triangleCount;
        if (chosenTriangleCount == 0u) {
            return;
        }

        // reserve an index range in the bucket, then append the record. The demand block
        // tracks unclamped demand (it survives phase-2 resets) - the CPU reads it back and
        // grows the index buffer when the cut wants more than the current allocation.
        let indexNeed = chosenTriangleCount * 3u;
        let capacity = bucketCapacity(bucket);
        let cursor = atomicAdd(&counters[${MESHLET_COUNTER.CURSOR_BASE}u + bucket], indexNeed);
        atomicMax(&counters[${MESHLET_COUNTER.DEMAND_BASE}u + bucket], cursor + indexNeed);
        if (cursor + indexNeed > capacity) {
            return;
        }
        atomicMax(&counters[${MESHLET_COUNTER.COMMITTED_BASE}u + bucket], cursor + indexNeed);
        let baseIndexOffset = bucketBase(bucket) + cursor;

        let recordIndex = atomicAdd(&counters[${MESHLET_COUNTER.RECORDS}u], 1u);
        if (recordIndex >= uniform.recordCapacity) {
            return;
        }
        records[recordIndex] = MeshletRecord(instance, chosen, baseIndexOffset, bucket);
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

        let recordCount = min(atomicLoad(&counters[${MESHLET_COUNTER.RECORDS}u]), uniform.recordCapacity);
        let base = uniform.dispatchSlot * ${INDIRECT_DISPATCH_U32S}u;
        indirectDispatch[base + 0u] = min(recordCount, ${MESHLET_DISPATCH_WIDTH}u);
        indirectDispatch[base + 1u] = (recordCount + ${MESHLET_DISPATCH_WIDTH - 1}u) / ${MESHLET_DISPATCH_WIDTH}u;
        indirectDispatch[base + 2u] = 1u;
    }
`;

/**
 * Index generation: one workgroup per surviving record, threads expand the meshlet's u8 corner
 * stream from the page pool into the global index buffer as (recordIndex << 8) | localVertex.
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

    @compute @workgroup_size(${MESHLET_INDEX_WRITE_WORKGROUP})
    fn main(@builtin(workgroup_id) workgroupId : vec3u, @builtin(local_invocation_id) localId : vec3u) {
        let recordIndex = workgroupId.y * ${MESHLET_DISPATCH_WIDTH}u + workgroupId.x;
        let recordCount = min(atomicLoad(&counters[${MESHLET_COUNTER.RECORDS}u]), uniform.recordCapacity);
        if (recordIndex >= recordCount) {
            return;
        }

        let instance = records[recordIndex].instance;
        let meshlet = records[recordIndex].meshlet;
        let baseIndexOffset = records[recordIndex].baseIndexOffset;
        let triangleOffset = meshletData[meshlet].triangleOffset;
        let cornerCount = meshletData[meshlet].triangleCount * 3u;
        let page = meshletData[meshlet].page;

        // per-instance page attribute layout (resources may differ)
        let uvFloatsPerVertex = objectData[instance].uvFloatsPerVertex;
        let hasTangents = (objectData[instance].flags & ${OBJECT_FLAG_HAS_TANGENTS}u) != 0u;

        // the u8 triangle corner stream follows the page's meshlet-vertex table (shared layout
        // in meshlet-page-wgsl.js)
        let pageBase = residency[page] * uniform.pageSizeWords;
        let pageLayout = meshletPageLayout(pageBase, hasTangents, uvFloatsPerVertex);
        let triangleStreamByte = (pageLayout.meshletVertexBase + pageLayout.meshletVertexCount) * 4u + triangleOffset;

        for (var corner = localId.x; corner < cornerCount; corner += ${MESHLET_INDEX_WRITE_WORKGROUP}u) {
            let byteIndex = triangleStreamByte + corner;
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
