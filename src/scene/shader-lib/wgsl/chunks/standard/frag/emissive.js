export default /* wgsl */`
uniform material_emissive: vec3f;
uniform material_emissiveIntensity: f32;

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
uniform emissiveStereoVideoType: i32; // 0: None, 1: SideBySide, 2: TopBottom
#endif
// end magnopus patched

fn getEmission() {
    dEmission = uniform.material_emissive * uniform.material_emissiveIntensity;

    #ifdef STD_EMISSIVE_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
            var stereoUV: vec2f = getStereoVideoUV({STD_EMISSIVE_TEXTURE_UV}, uniform.emissiveStereoVideoType);

            // Identical to the unpatched version except for the texture coordinates
            dEmission *= {STD_EMISSIVE_TEXTURE_DECODE}(textureSampleBias({STD_EMISSIVE_TEXTURE_NAME}, {STD_EMISSIVE_TEXTURE_NAME}Sampler, stereoUV, uniform.textureBias)).{STD_EMISSIVE_TEXTURE_CHANNEL};
        #else
        // end magnopus patched
    dEmission *= {STD_EMISSIVE_TEXTURE_DECODE}(textureSampleBias({STD_EMISSIVE_TEXTURE_NAME}, {STD_EMISSIVE_TEXTURE_NAME}Sampler, {STD_EMISSIVE_TEXTURE_UV}, uniform.textureBias)).{STD_EMISSIVE_TEXTURE_CHANNEL};
        // magnopus patched
        #endif
        // end magnopus patched
    #endif

    #ifdef STD_EMISSIVE_VERTEX
    dEmission = dEmission * gammaCorrectInputVec3(saturate3(vVertexColor.{STD_EMISSIVE_VERTEX_CHANNEL}));
    #endif
}
`;
