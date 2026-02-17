export default /* wgsl */`
#ifdef STD_METALNESS_CONSTANT
uniform material_metalness: f32;
#endif

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
uniform metalnessStereoVideoType: i32; // 0: None, 1: SideBySide, 2: TopBottom
#endif
// end magnopus patched

fn getMetalness() {
    var metalness: f32 = 1.0;

    #ifdef STD_METALNESS_CONSTANT
        metalness = metalness * uniform.material_metalness;
    #endif

    #ifdef STD_METALNESS_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
            var stereoUV: vec2f = getStereoVideoUV({STD_METALNESS_TEXTURE_UV}, uniform.metalnessStereoVideoType);

            // Identical to the unpatched version except for the texture coordinates
            metalness = metalness * textureSampleBias({STD_METALNESS_TEXTURE_NAME}, {STD_METALNESS_TEXTURE_NAME}Sampler, stereoUV, uniform.textureBias).{STD_METALNESS_TEXTURE_CHANNEL};
        #else
        // end magnopus patched
        metalness = metalness * textureSampleBias({STD_METALNESS_TEXTURE_NAME}, {STD_METALNESS_TEXTURE_NAME}Sampler, {STD_METALNESS_TEXTURE_UV}, uniform.textureBias).{STD_METALNESS_TEXTURE_CHANNEL};
        // magnopus patched
        #endif
        // end magnopus patched
    #endif

    #ifdef STD_METALNESS_VERTEX
    metalness = metalness * saturate(vVertexColor.{STD_METALNESS_VERTEX_CHANNEL});
    #endif

    dMetalness = metalness;
}
`;
