export default /* glsl */`
#ifdef STD_METALNESS_CONSTANT
uniform float material_metalness;
#endif

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
uniform int metalnessStereoVideoType; // 0: None, 1: SideBySide, 2: TopBottom
#endif
// end magnopus patched

void getMetalness() {
    float metalness = 1.0;

    #ifdef STD_METALNESS_CONSTANT
    metalness *= material_metalness;
    #endif

    #ifdef STD_METALNESS_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
        vec2 stereoUV = getStereoVideoUV({STD_METALNESS_TEXTURE_UV}, metalnessStereoVideoType);

        metalness *= texture2DBias({STD_METALNESS_TEXTURE_NAME}, stereoUV, textureBias).{STD_METALNESS_TEXTURE_CHANNEL};
        #else
        // end magnopus patched
    metalness *= texture2DBias({STD_METALNESS_TEXTURE_NAME}, {STD_METALNESS_TEXTURE_UV}, textureBias).{STD_METALNESS_TEXTURE_CHANNEL};
        // magnopus patched
        #endif
        // end magnopus patched
    #endif

    #ifdef STD_METALNESS_VERTEX
    metalness *= saturate(vVertexColor.{STD_METALNESS_VERTEX_CHANNEL});
    #endif

    dMetalness = metalness;
}
`;
