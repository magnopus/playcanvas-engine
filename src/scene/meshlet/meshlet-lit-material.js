import { CULLFACE_NONE } from '../../platform/graphics/constants.js';
import { StandardMaterial } from '../materials/standard-material.js';
import { MESHLET_BUCKET_MASKED, MESHLET_BUCKET_OPAQUE } from './constants.js';

/**
 * The engine-lit meshlet forward material: a StandardMaterial whose vertex stage is replaced by
 * page-pool vertex pulling and whose fragment frontend reads the 32-word material record - and,
 * with streamed textures, samples the resident tail arrays (see
 * shaders/meshlet-lit-chunks-wgsl.js, built per world). Inherits the full lit backend -
 * clustered lights, received shadows, IBL/ambient, fog - plus the engine's env-atlas binding
 * and shader-variant plumbing.
 *
 * The caller binds the world's storage buffers (meshletData, objectData, records, pagePool,
 * residency, materialTable, texResidency when textured), pageSizeWords, and the family tail
 * textures as material parameters.
 *
 * One material per draw bucket - they differ only in the two pieces of pipeline state that
 * cannot vary within an indirect draw: backface culling, and whether the alpha-test discard is
 * compiled in. Keeping the discard out of the opaque buckets is what preserves early-Z, and
 * that is worth a bucket of its own: on the sample assets it is ~30% of the meshlet draw time.
 *
 * @param {number} bucket - MESHLET_BUCKET_OPAQUE / _OPAQUE_TWO_SIDED / _MASKED.
 * @param {object} chunks - The chunk override map from buildMeshletLitChunks (shared per world).
 * @returns {StandardMaterial} The material.
 * @ignore
 */
function createMeshletLitMaterial(bucket, chunks) {
    const twoSided = bucket !== MESHLET_BUCKET_OPAQUE;
    const masked = bucket === MESHLET_BUCKET_MASKED;
    const material = new StandardMaterial();
    material.name = ['MeshletLit', 'MeshletLitTwoSided', 'MeshletLitMasked'][bucket];

    // metalness workflow; the actual values come from the material record in the overridden
    // frontend chunks - these properties only drive the lit-option defines
    material.useMetalness = true;
    material.diffuse.set(1, 1, 1);
    material.metalness = 1;
    material.gloss = 1;

    if (masked) {
        // any value > 0 compiles LIT_ALPHA_TEST; the overridden alphaTestPS chunk compares
        // against the per-record cutoff, not this uniform
        material.alphaTest = 0.5;
    }
    if (twoSided) {
        material.cull = CULLFACE_NONE;
        material.twoSidedLighting = true;
    }

    // the pick id comes from objectData, not from the per-mesh-instance uniform
    material.setDefine('PICK_CUSTOM_ID', true);

    material.getShaderChunks('wgsl').add(chunks);
    material.shaderChunksVersion = '2.8';
    material.update();
    return material;
}

export { createMeshletLitMaterial };
