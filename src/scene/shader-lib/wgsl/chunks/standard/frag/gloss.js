export default /* wgsl */`
#ifdef STD_GLOSS_CONSTANT
    uniform material_gloss: f32;
#endif

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
uniform glossStereoVideoType: i32; // 0: None, 1: SideBySide, 2: TopBottom
#endif
// end magnopus patched

fn getGlossiness() {
    dGlossiness = 1.0;

    #ifdef STD_GLOSS_CONSTANT
    dGlossiness = dGlossiness * uniform.material_gloss;
    #endif

    #ifdef STD_GLOSS_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
            var stereoUV: vec2f = getStereoVideoUV({STD_GLOSS_TEXTURE_UV}, uniform.glossStereoVideoType);

            // Identical to the unpatched version except for the texture coordinates
            dGlossiness = dGlossiness * textureSampleBias({STD_GLOSS_TEXTURE_NAME}, {STD_GLOSS_TEXTURE_NAME}Sampler, stereoUV, uniform.textureBias).{STD_GLOSS_TEXTURE_CHANNEL};
        #else
        // end magnopus patched
    dGlossiness = dGlossiness * textureSampleBias({STD_GLOSS_TEXTURE_NAME}, {STD_GLOSS_TEXTURE_NAME}Sampler, {STD_GLOSS_TEXTURE_UV}, uniform.textureBias).{STD_GLOSS_TEXTURE_CHANNEL};
        // magnopus patched
        #endif
        // end magnopus patched
    #endif

    #ifdef STD_GLOSS_VERTEX
    dGlossiness = dGlossiness * saturate(vVertexColor.{STD_GLOSS_VERTEX_CHANNEL});
    #endif

    #ifdef STD_GLOSS_INVERT
    dGlossiness = 1.0 - dGlossiness;
    #endif

    dGlossiness = dGlossiness + 0.0000001;
}
`;
