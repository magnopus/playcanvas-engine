export default /* glsl */`

#if defined(STD_AO_TEXTURE) || defined(STD_AO_VERTEX)
    uniform float material_aoIntensity;
#endif

#ifdef STD_AODETAIL_TEXTURE
    #include "detailModesPS"
#endif

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
uniform int aoStereoVideoType; // 0: None, 1: SideBySide, 2: TopBottom
#endif
// end magnopus patched

void getAO() {
    dAo = 1.0;

    #ifdef STD_AO_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
            vec2 stereoUV = getStereoVideoUV({STD_AO_TEXTURE_UV}, aoStereoVideoType);

            float aoBase = texture2DBias({STD_AO_TEXTURE_NAME}, stereoUV, textureBias).{STD_AO_TEXTURE_CHANNEL};
        #else
        // end magnopus patched
        float aoBase = texture2DBias({STD_AO_TEXTURE_NAME}, {STD_AO_TEXTURE_UV}, textureBias).{STD_AO_TEXTURE_CHANNEL};
        // magnopus patched
        #endif

        #ifdef STD_AODETAIL_TEXTURE
            float aoDetail = texture2DBias({STD_AODETAIL_TEXTURE_NAME}, {STD_AODETAIL_TEXTURE_UV}, textureBias).{STD_AODETAIL_TEXTURE_CHANNEL};
            aoBase = detailMode_{STD_AODETAIL_DETAILMODE}(vec3(aoBase), vec3(aoDetail)).r;
        #endif

        dAo *= aoBase;
    #endif

    #ifdef STD_AO_VERTEX
        dAo *= saturate(vVertexColor.{STD_AO_VERTEX_CHANNEL});
    #endif

    #if defined(STD_AO_TEXTURE) || defined(STD_AO_VERTEX)
        dAo = mix(1.0, dAo, material_aoIntensity);
    #endif
}
`;
