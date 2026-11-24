import {
    CULL_PARAMS, CULL_PARAMS_VEC4S, INDIRECT_DISPATCH_U32S, MESHLET_BUCKET_COUNT, MESHLET_BUCKET_MASKED, MESHLET_BUCKET_OPAQUE,
    MESHLET_BUCKET_OPAQUE_TWO_SIDED, MESHLET_COUNTER, MESHLET_DISPATCH_WIDTH, MESHLET_FLAG_ALPHA_MASKED,
    MESHLET_FLAG_TWO_SIDED, MESHLET_VISIBLE_CELLS_BASE, PAGE_NOT_RESIDENT, PAGE_REQUEST
} from '../constants.js';

/**
 * WGSL snippets shared by the cull, cut and root admission compute shaders. Each lists the
 * declarations the including shader must provide.
 *
 * @ignore
 */

/**
 * The view's cull parameters (rows CULL_PARAMS.*), declared as a uniform array: as a storage
 * buffer they would take one of the eight storage bindings WebGPU guarantees per stage.
 *
 * @ignore
 */
export const meshletCullParamsWGSL = /* wgsl */ `
    uniform cullParams: array<vec4f, ${CULL_PARAMS_VEC4S}>;
`;

/**
 * The entry point of a pass dispatched with a fixed number of workgroups (MESHLET_STRIDED_GROUPS)
 * while its work is sized on the GPU: each workgroup loops over the virtual workgroups
 * `meshletItemGroups()` counts, a stride of the dispatch size apart, running
 * `meshletItem(group, lane)` for each as the indirect dispatch it replaces ran `main` - the
 * group is the virtual one, in the same two-dimensional layout (see meshletGroupIndex). The
 * including shader provides both functions; neither may use a workgroup barrier.
 *
 * @param {number} workgroupSize - The workgroup size.
 * @returns {string} The entry point.
 * @ignore
 */
export const meshletStridedMainWGSL = workgroupSize => /* wgsl */ `
    var<workgroup> meshletGroupCount: u32;

    @compute @workgroup_size(${workgroupSize})
    fn main(@builtin(workgroup_id) workgroup: vec3u, @builtin(num_workgroups) workgroups: vec3u,
            @builtin(local_invocation_id) lane: vec3u, @builtin(local_invocation_index) laneIndex: u32) {
        // one lane reads the counters, rather than every thread of every workgroup the same words
        if (laneIndex == 0u) { meshletGroupCount = meshletItemGroups(); }
        let groups = workgroupUniformLoad(&meshletGroupCount);
        for (var g = workgroup.x; g < groups; g += workgroups.x) {
            meshletItem(meshletGroupOf(g), lane);
        }
    }
`;

/**
 * `meshletGroupIndex(group)`: the linear index of a workgroup in a two-dimensional dispatch grid
 * MESHLET_DISPATCH_WIDTH wide (see {@link meshletDispatchArgsWGSL}), and `meshletGroupOf(index)`,
 * its inverse.
 *
 * @ignore
 */
export const meshletDispatchGroupWGSL = /* wgsl */ `
    fn meshletGroupIndex(group: vec3u) -> u32 {
        return group.y * ${MESHLET_DISPATCH_WIDTH}u + group.x;
    }
    fn meshletGroupOf(index: u32) -> vec3u {
        return vec3u(index % ${MESHLET_DISPATCH_WIDTH}u, index / ${MESHLET_DISPATCH_WIDTH}u, 0u);
    }
`;

/**
 * `meshletWriteDispatch(slot, groups)`: writes indirect dispatch args for `groups` workgroups
 * into `indirectDispatch : array<u32>`, wrapping into a second dimension past
 * MESHLET_DISPATCH_WIDTH to stay within the per-dimension workgroup limit.
 *
 * @ignore
 */
export const meshletDispatchArgsWGSL = /* wgsl */ `
    fn meshletWriteDispatch(slot: u32, groups: u32) {
        let base = slot * ${INDIRECT_DISPATCH_U32S}u;
        indirectDispatch[base + 0u] = min(groups, ${MESHLET_DISPATCH_WIDTH}u);
        indirectDispatch[base + 1u] = (groups + ${MESHLET_DISPATCH_WIDTH - 1}u) / ${MESHLET_DISPATCH_WIDTH}u;
        indirectDispatch[base + 2u] = 1u;
    }
`;

/**
 * The per-bucket index capacities (`uniform indexCapacity0..2`) and the bucket helpers:
 * `meshletBucket(flags)` maps meshlet flags to a draw bucket (MESHLET_BUCKET_*),
 * `meshletBucketCapacity(bucket)` and `meshletBucketBase(bucket)` give its range of the index
 * buffer, where the buckets are contiguous in bucket order.
 *
 * @ignore
 */
export const meshletBucketWGSL = /* wgsl */ `
    ${Array.from({ length: MESHLET_BUCKET_COUNT }, (_, b) => `uniform indexCapacity${b} : u32;`).join('\n    ')}

    fn meshletBucket(flags: u32) -> u32 {
        let alphaMasked = (flags & ${MESHLET_FLAG_ALPHA_MASKED}u) != 0u;
        let twoSided = (flags & ${MESHLET_FLAG_TWO_SIDED}u) != 0u;
        return select(select(${MESHLET_BUCKET_OPAQUE}u, ${MESHLET_BUCKET_OPAQUE_TWO_SIDED}u, twoSided), ${MESHLET_BUCKET_MASKED}u, alphaMasked);
    }

    fn meshletBucketCapacity(bucket: u32) -> u32 {
        return select(select(uniform.indexCapacity0, uniform.indexCapacity1, bucket == 1u),
                      uniform.indexCapacity2, bucket == 2u);
    }

    fn meshletBucketBase(bucket: u32) -> u32 {
        return select(select(0u, uniform.indexCapacity0, bucket == 1u),
                      uniform.indexCapacity0 + uniform.indexCapacity1, bucket == 2u);
    }
`;

/**
 * `meshletRequestPages(start, count)`: marks each page of a page list in `cutGroups` as used when
 * resident or missing when not (see PAGE_REQUEST), and returns true when all are resident. Every
 * page is marked, so a group waiting on several pages requests them all at once. Needs
 * `cutGroups`, `residency` and `requests`.
 *
 * @ignore
 */
export const meshletRequestPagesWGSL = /* wgsl */ `
    fn meshletRequestPages(start: u32, count: u32) -> bool {
        var ready = true;
        for (var c = 0u; c < count; c++) {
            let page = cutGroups[start + c];
            let resident = residency[page] != ${PAGE_NOT_RESIDENT}u;
            let mark = select(${PAGE_REQUEST.MISSING}u, ${PAGE_REQUEST.USED}u, resident);
            // most pages are already marked this frame: read first so only a raise pays for the atomic
            if (atomicLoad(&requests[page]) < mark) { atomicMax(&requests[page], mark); }
            ready = ready && resident;
        }
        return ready;
    }
`;

/**
 * The cull volume test every pass shares: the view's frustum planes, then any extra planes the
 * view adds (a shadow face's caster volume, see MeshletShadowRenderer). Reads `cullParams`.
 *
 * @ignore
 */
export const meshletCullVolumeWGSL = /* wgsl */ `
    fn meshletInCullVolume(center: vec3f, radius: f32) -> bool {
        for (var p = 0u; p < ${CULL_PARAMS.PLANE_COUNT}u; p++) {
            let plane = uniform.cullParams[${CULL_PARAMS.PLANES}u + p];
            if (dot(plane.xyz, center) + plane.w < -radius) { return false; }
        }
        let extra = u32(uniform.cullParams[${CULL_PARAMS.STREAMING}u].w);
        for (var p = 0u; p < extra; p++) {
            let plane = uniform.cullParams[${CULL_PARAMS.CASTER_PLANES}u + p];
            if (dot(plane.xyz, center) + plane.w < -radius) { return false; }
        }
        return true;
    }
`;

/**
 * `meshletVisibleCell(group)`: the visible cell a workgroup of a cell-driven pass serves, or
 * 0xFFFFFFFF past the end of the list (the 2D dispatch grid rounds up). The list sits after the
 * counters in `counters`. Requires {@link meshletDispatchGroupWGSL}.
 *
 * @ignore
 */
export const meshletVisibleCellWGSL = /* wgsl */ `
    fn meshletVisibleCell(group: vec3u) -> u32 {
        let slot = meshletGroupIndex(group);
        if (slot >= atomicLoad(&counters[${MESHLET_COUNTER.VISIBLE_CELLS}u])) {
            return 0xFFFFFFFFu;
        }
        return atomicLoad(&counters[${MESHLET_VISIBLE_CELLS_BASE}u + slot]);
    }
`;

/**
 * `hzbVisibleFrom(viewProj, camPos, viewDir, center, radius)`: the conservative occlusion test of
 * a world-space sphere against the max-depth HZB, for any view of the pyramid: the current
 * frame's camera (the cluster cull), or the camera the pyramid was built from (root admission,
 * which reads last frame's). Returns true when the sphere is potentially visible. Expects the
 * including shader to declare `hzbTexture` and the `cullParams` array (which carries the
 * pyramid's size and mip count). The pyramid holds linear view depth, so no projection
 * conventions enter the comparison.
 *
 * @ignore
 */
export const meshletHzbVisibleWGSL = /* wgsl */ `
// the texels a uv rect covers at a mip, as (min, max). Texel i covers mip-0 texels
// [i * 2^mip, (i + 1) * 2^mip), the last one also the remainder an odd size floors away, so map
// through mip 0 rather than the floored mip size.
fn hzbTexelRange(uvMin : vec2f, uvMax : vec2f, hzbSize : vec2f, mip : u32) -> vec4u {
    let mipSize = vec2f(f32(max(u32(hzbSize.x) >> mip, 1u)), f32(max(u32(hzbSize.y) >> mip, 1u)));
    let mipScale = hzbSize / f32(1u << mip);
    return vec4u(vec2u(clamp(uvMin * mipScale, vec2f(0.0), mipSize - 1.0)),
                 vec2u(clamp(uvMax * mipScale, vec2f(0.0), mipSize - 1.0)));
}

fn hzbVisibleFrom(viewProjMatrix : mat4x4f, camPos : vec3f, viewDir : vec3f, center : vec3f, radius : f32) -> bool {
    let toCenter = center - camPos;
    if (dot(toCenter, toCenter) <= radius * radius * 1.1025) {
        return true; // camera inside or at the sphere
    }

    // nearest view depth; the point nearest the camera is deeper than this off the view axis
    let sphereDepth = dot(toCenter, viewDir) - radius;
    if (sphereDepth <= 0.0) {
        return true;
    }

    // view axes and projection scales from the view-projection rows: row i's xyz is
    // P[i][i] * axis - P[2][i] * viewDir, the second term only for off-centre (XR) frusta
    let rowX = vec3f(viewProjMatrix[0].x, viewProjMatrix[1].x, viewProjMatrix[2].x);
    let rowY = vec3f(viewProjMatrix[0].y, viewProjMatrix[1].y, viewProjMatrix[2].y);
    let ndcOffset = vec2f(dot(rowX, viewDir), dot(rowY, viewDir));
    let axisX = rowX - viewDir * ndcOffset.x;
    let axisY = rowY - viewDir * ndcOffset.y;
    let ndcScale = vec2f(length(axisX), length(axisY));

    var ndcMin : vec2f;
    var ndcMax : vec2f;
    let perspective = dot(vec3f(viewProjMatrix[0].w, viewProjMatrix[1].w, viewProjMatrix[2].w), viewDir) > 0.5;
    if (perspective) {
        // exact bounds of the projected sphere (Mara and McGuire 2013, "2D Polyhedral Bounds of a
        // Clipped, Perspective-Projected 3D Sphere"); denominators are positive as c.z > radius
        let c = vec3f(dot(toCenter, axisX / ndcScale.x), dot(toCenter, axisY / ndcScale.y), dot(toCenter, viewDir));
        let czr2 = c.z * c.z - radius * radius;
        let vx = sqrt(c.x * c.x + czr2);
        let vy = sqrt(c.y * c.y + czr2);
        let tanMin = vec2f((vx * c.x - radius * c.z) / (vx * c.z + radius * c.x),
                           (vy * c.y - radius * c.z) / (vy * c.z + radius * c.y));
        let tanMax = vec2f((vx * c.x + radius * c.z) / (vx * c.z - radius * c.x),
                           (vy * c.y + radius * c.z) / (vy * c.z - radius * c.y));
        ndcMin = tanMin * ndcScale + ndcOffset;
        ndcMax = tanMax * ndcScale + ndcOffset;
    } else {
        let ndcCenter = (viewProjMatrix * vec4f(center, 1.0)).xy;
        ndcMin = ndcCenter - ndcScale * radius;
        ndcMax = ndcCenter + ndcScale * radius;
    }

    // ndc -> uv (y down)
    let uvMin = clamp(vec2f(ndcMin.x, -ndcMax.y) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));
    let uvMax = clamp(vec2f(ndcMax.x, -ndcMin.y) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));

    let hzbSize = uniform.cullParams[${CULL_PARAMS.LOD}u].yz;
    let rectPx = (uvMax - uvMin) * hzbSize;
    let mipCount = u32(uniform.cullParams[${CULL_PARAMS.LOD}u].w);
    // the mip where the rect is at most one texel wide straddles at most 2x2 texels, so the four
    // corner samples cover it. One mip finer often still fits, with tighter depths.
    var mip = min(u32(max(ceil(log2(max(max(rectPx.x, rectPx.y), 1.0))), 0.0)), mipCount - 1u);
    var texels = hzbTexelRange(uvMin, uvMax, hzbSize, mip);
    if (mip > 0u) {
        let fine = hzbTexelRange(uvMin, uvMax, hzbSize, mip - 1u);
        if (all(fine.zw - fine.xy <= vec2u(1u))) {
            mip -= 1u;
            texels = fine;
        }
    }
    // max-depth pyramid: the farthest corner is the conservative occluder depth
    let depth00 = textureLoad(hzbTexture, vec2i(i32(texels.x), i32(texels.y)), i32(mip)).x;
    let depth10 = textureLoad(hzbTexture, vec2i(i32(texels.z), i32(texels.y)), i32(mip)).x;
    let depth01 = textureLoad(hzbTexture, vec2i(i32(texels.x), i32(texels.w)), i32(mip)).x;
    let depth11 = textureLoad(hzbTexture, vec2i(i32(texels.z), i32(texels.w)), i32(mip)).x;
    let occluderDepth = max(max(depth00, depth10), max(depth01, depth11));
    // float rounding only; source precision is already in the pyramid
    return sphereDepth <= occluderDepth * (1.0 + 1e-5) + 1e-4;
}
`;
