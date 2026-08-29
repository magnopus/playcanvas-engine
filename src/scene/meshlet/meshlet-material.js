import { CULLFACE_NONE } from '../../platform/graphics/constants.js';
import { ShaderMaterial } from '../materials/shader-material.js';
import { MESHLET_COLOR_MODE, OBJECT_FLAG_HAS_TANGENTS } from './constants.js';
import {
    meshletDataWGSL, meshletDecodeDrawIndexWGSL, meshletDecodeOct16WGSL, meshletMaterialTableWGSL,
    meshletObjectDataWGSL, meshletPageLayoutWGSL, meshletRecordsWGSL, meshletStructsWGSL
} from './shaders/meshlet-page-wgsl.js';

/**
 * The meshlet DEBUG material (colour modes: material baseColor / LOD tint / per-meshlet hash):
 * a ShaderMaterial with the same page-pool vertex pull as the lit path (shared snippets in
 * shaders/meshlet-page-wgsl.js) but a flat lambert fragment. Normal shading uses the
 * engine-lit StandardMaterial (meshlet-lit-material.js); the world swaps between the two via
 * setColorMode.
 *
 * @ignore
 */

const vertexWGSL = /* wgsl */ `
    ${meshletStructsWGSL}

    uniform matrix_viewProjection : mat4x4f;
    uniform pageSizeWords : u32;
    uniform colorMode : u32;

    ${meshletDataWGSL}
    ${meshletObjectDataWGSL}
    ${meshletRecordsWGSL('read')}
    ${meshletMaterialTableWGSL}
    var<storage, read> pagePool : array<u32>;
    var<storage, read> residency : array<u32>;

    varying vColor : vec3f;
    varying vNormal : vec3f;

    ${meshletDecodeDrawIndexWGSL}
    ${meshletDecodeOct16WGSL}
    ${meshletPageLayoutWGSL}

    // debug tint per LOD tier
    fn lodTint(level : u32) -> vec3f {
        let h = f32(level % 8u) / 8.0;
        return clamp(abs(fract(vec3f(h) + vec3f(0.0, 0.33, 0.67)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0)) * 0.8 + 0.2;
    }

    // debug colour per meshlet (integer hash)
    fn meshletColor(id : u32) -> vec3f {
        var h = id * 747796405u + 2891336453u;
        h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
        h = (h >> 22u) ^ h;
        return vec3f(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.0 * 0.75 + 0.25;
    }

    @vertex
    fn vertexMain(input : VertexInput) -> VertexOutput {
        var output : VertexOutput;

        let drawIndex = meshletDecodeDrawIndex(input.vertexIndex);
        let record = drawIndex.x;
        let localVert = drawIndex.y;

        let instance = records[record].instance;
        let meshlet = records[record].meshlet;

        // per-instance page attribute layout (resources may differ)
        let uvFloatsPerVertex = objectData[instance].uvFloatsPerVertex;
        let hasTangents = (objectData[instance].flags & ${OBJECT_FLAG_HAS_TANGENTS}u) != 0u;

        let pageLayout = meshletPageLayout(residency[meshletData[meshlet].page] * uniform.pageSizeWords, hasTangents, uvFloatsPerVertex);
        let vertex = pagePool[pageLayout.meshletVertexBase + meshletData[meshlet].verticesOffset + localVert];

        // per-instance position grid (resources may be baked on different grids)
        let localPos = meshletPagePosition(pageLayout, vertex, meshletObjectGridOrigin(instance), objectData[instance].gridStep);

        let worldMatrix = objectData[instance].worldMatrix;
        output.position = uniform.matrix_viewProjection * worldMatrix * vec4f(localPos, 1.0);
        output.vNormal = normalize((worldMatrix * vec4f(meshletDecodeOct16(pagePool[pageLayout.normalBase + vertex]), 0.0)).xyz);

        if (uniform.colorMode == ${MESHLET_COLOR_MODE.LOD_TIER}u) {
            output.vColor = lodTint(meshletData[meshlet].lodLevel);
        } else if (uniform.colorMode == ${MESHLET_COLOR_MODE.MESHLET}u) {
            output.vColor = meshletColor(meshlet);
        } else {
            output.vColor = materialTable[objectData[instance].material].baseColor.rgb;
        }

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
    const doubleSided = bucket !== 0;
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
