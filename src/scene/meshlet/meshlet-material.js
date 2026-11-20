import { CULLFACE_NONE } from '../../platform/graphics/constants.js';
import { ShaderMaterial } from '../materials/shader-material.js';
import { MESHLET_BUCKET_OPAQUE } from './constants.js';
import {
    meshletDataWGSL, meshletDecodeDrawIndexWGSL, meshletDecodeOct16WGSL, meshletFetchVertexWGSL, meshletNormalMatrixWGSL,
    meshletObjectDataWGSL, meshletPageLayoutWGSL, meshletRecordsWGSL, meshletStructsWGSL
} from './shaders/meshlet-page-wgsl.js';

/**
 * The meshlet debug material (MESHLET_COLOR_MODE.MESHLET): each meshlet in a hashed colour,
 * with the same page-pool vertex pull as the lit path (shared snippets in
 * shaders/meshlet-page-wgsl.js) and a flat lambert fragment. Normal shading uses the
 * engine-lit StandardMaterial (meshlet-lit-material.js); the world swaps between the two via
 * setColorMode.
 *
 * @ignore
 */

const vertexWGSL = /* wgsl */ `
    ${meshletStructsWGSL}

    uniform matrix_viewProjection : mat4x4f;
    uniform pageSizeWords : u32;

    ${meshletDataWGSL}
    ${meshletObjectDataWGSL}
    ${meshletRecordsWGSL('read')}
    var<storage, read> pagePool : array<u32>;
    var<storage, read> residency : array<u32>;

    varying vColor : vec3f;
    varying vNormal : vec3f;

    ${meshletDecodeDrawIndexWGSL}
    ${meshletDecodeOct16WGSL}
    ${meshletPageLayoutWGSL}
    ${meshletFetchVertexWGSL}
    ${meshletNormalMatrixWGSL}

    // debug colour per meshlet (integer hash)
    fn meshletColor(id : u32) -> vec3f {
        var h = id * 747796405u + 2891336453u;
        h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
        h = (h >> 22u) ^ h;
        let hue = f32(h & 65535u) / 65536.0;
        let rgb = clamp(abs(fract(vec3f(hue) + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
        return mix(vec3f(0.02), vec3f(0.8), rgb);
    }

    @vertex
    fn vertexMain(input : VertexInput) -> VertexOutput {
        var output : VertexOutput;

        let fetched = meshletFetchVertex(input.vertexIndex, uniform.pageSizeWords);
        let primitive = fetched.primitive;
        let localPos = meshletPagePosition(fetched.pageLayout, fetched.vertex, primitive.gridOrigin, primitive.gridStep);

        let worldMatrix = meshletInstanceMatrix(fetched.instance);
        output.position = uniform.matrix_viewProjection * worldMatrix * vec4f(localPos, 1.0);
        let normal = meshletDecodeOct16(pagePool[fetched.pageLayout.normalBase + fetched.vertex]);
        output.vNormal = normalize(meshletNormalMatrix(worldMatrix) * normal);

        output.vColor = meshletColor(fetched.meshlet);

        return output;
    }
`;

// the processor gives the fragment stage the vertex stage's storage declarations too, so the
// struct block is needed here as well
const fragmentWGSL = /* wgsl */ `
    ${meshletStructsWGSL}

    varying vColor : vec3f;
    varying vNormal : vec3f;

    @fragment
    fn fragmentMain(input : FragmentInput) -> FragmentOutput {
        var output : FragmentOutput;
        let light = normalize(vec3f(0.5, 1.0, 0.4));
        let n = normalize(input.vNormal);
        let ndl = max(dot(n, light), 0.0) * 0.55 + abs(dot(n, light)) * 0.2 + 0.3;
        output.color = vec4f(input.vColor * ndl, 1.0);
        return output;
    }
`;

/**
 * Creates a meshlet debug material.
 *
 * @param {number} bucket - MESHLET_BUCKET_OPAQUE / _OPAQUE_TWO_SIDED / _MASKED. The debug
 * shader has no alpha test, so the two two-sided buckets differ only in name.
 * @returns {ShaderMaterial} The material.
 * @ignore
 */
function createMeshletMaterial(bucket) {
    const doubleSided = bucket !== MESHLET_BUCKET_OPAQUE;
    const material = new ShaderMaterial({
        uniqueName: ['MeshletForward', 'MeshletForwardTwoSided', 'MeshletForwardMasked'][bucket],
        vertexWGSL: vertexWGSL,
        fragmentWGSL: fragmentWGSL
    });
    if (doubleSided) {
        material.cull = CULLFACE_NONE;
    }
    return material;
}

export { createMeshletMaterial };
