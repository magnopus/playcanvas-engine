export default /* glsl */`
#ifdef STD_GLOSS_CONSTANT
uniform float material_gloss;
#endif

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
uniform int glossStereoVideoType; // 0: None, 1: SideBySide, 2: TopBottom
#endif
// end magnopus patched

void getGlossiness() {
    dGlossiness = 1.0;

    #ifdef STD_GLOSS_CONSTANT
    dGlossiness *= material_gloss;
    #endif

    #ifdef STD_GLOSS_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
            vec2 stereoUV = getStereoVideoUV({STD_GLOSS_TEXTURE_UV}, glossStereoVideoType);

            // Identical to the unpatched version except for the texture coordinates
            dGlossiness *= texture2DBias({STD_GLOSS_TEXTURE_NAME}, stereoUV, textureBias).{STD_GLOSS_TEXTURE_CHANNEL};
        #else
        // end magnopus patched
    dGlossiness *= texture2DBias({STD_GLOSS_TEXTURE_NAME}, {STD_GLOSS_TEXTURE_UV}, textureBias).{STD_GLOSS_TEXTURE_CHANNEL};
        // magnopus patched
        #endif
        // end magnopus patched
    #endif

    #ifdef STD_GLOSS_VERTEX
    dGlossiness *= saturate(vVertexColor.{STD_GLOSS_VERTEX_CHANNEL});
    #endif

    #ifdef STD_GLOSS_INVERT
    dGlossiness = 1.0 - dGlossiness;
    #endif

    dGlossiness += 0.0000001;
}
`;
