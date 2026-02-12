export default /* wgsl */`

#if defined(STD_AO_TEXTURE) || defined(STD_AO_VERTEX)
    uniform material_aoIntensity: f32;
#endif

#ifdef STD_AODETAIL_TEXTURE
    #include "detailModesPS"
#endif

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
uniform aoStereoVideoType: i32; // 0: None, 1: SideBySide, 2: TopBottom
#endif
// end magnopus patched

fn getAO() {
    dAo = 1.0;

    #ifdef STD_AO_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
            var stereoUV: vec2f = getStereoVideoUV({STD_AO_TEXTURE_UV}, uniform.aoStereoVideoType);

            // Identical to the unpatched version except for the texture coordinates
            var aoBase: f32 = textureSampleBias({STD_AO_TEXTURE_NAME}, {STD_AO_TEXTURE_NAME}Sampler, stereoUV, uniform.textureBias).{STD_AO_TEXTURE_CHANNEL};
        #else
        // end magnopus patched
        var aoBase: f32 = textureSampleBias({STD_AO_TEXTURE_NAME}, {STD_AO_TEXTURE_NAME}Sampler, {STD_AO_TEXTURE_UV}, uniform.textureBias).{STD_AO_TEXTURE_CHANNEL};
        // magnopus patched
        #endif
        // end magnopus patched

        #ifdef STD_AODETAIL_TEXTURE
            var aoDetail: f32 = textureSampleBias({STD_AODETAIL_TEXTURE_NAME}, {STD_AODETAIL_TEXTURE_NAME}Sampler, {STD_AODETAIL_TEXTURE_UV}, uniform.textureBias).{STD_AODETAIL_TEXTURE_CHANNEL};
            aoBase = detailMode_{STD_AODETAIL_DETAILMODE}(vec3f(aoBase), vec3f(aoDetail)).r;
        #endif

        dAo = dAo * aoBase;
    #endif

    #ifdef STD_AO_VERTEX
        dAo = dAo * saturate(vVertexColor.{STD_AO_VERTEX_CHANNEL});
    #endif

    #if defined(STD_AO_TEXTURE) || defined(STD_AO_VERTEX)
        dAo = mix(1.0, dAo, uniform.material_aoIntensity);
    #endif
}
`;
