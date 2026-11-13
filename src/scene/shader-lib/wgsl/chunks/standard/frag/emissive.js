export default /* wgsl */`
#ifndef MESH_COLOR
    uniform material_emissive: vec3f;
#endif
uniform material_emissiveIntensity: f32;

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
uniform emissiveStereoVideoType: i32; // 0: None, 1: SideBySide, 2: TopBottom
uniform emissiveIsStereoFlipped: u32; // 0: Not Flipped, 1: Flipped
#endif
// end magnopus patched

fn getEmission() {
    #ifdef MESH_COLOR
        dEmission = uniform.mesh_color.rgb * uniform.material_emissiveIntensity;
    #else
        dEmission = uniform.material_emissive * uniform.material_emissiveIntensity;
    #endif

    #ifdef STD_EMISSIVE_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
            var stereoUV: vec2f = getStereoVideoUV({STD_EMISSIVE_TEXTURE_UV}, uniform.emissiveStereoVideoType, uniform.emissiveIsStereoFlipped);

            // Identical to the unpatched version except for the texture coordinates
            dEmission *= {STD_EMISSIVE_TEXTURE_DECODE}(textureSampleBias({STD_EMISSIVE_TEXTURE_NAME}, {STD_EMISSIVE_TEXTURE_NAME}Sampler, stereoUV, {STD_TEXTURE_BIAS})).{STD_EMISSIVE_TEXTURE_CHANNEL};
        #else
        // end magnopus patched
    dEmission *= {STD_EMISSIVE_TEXTURE_DECODE}(textureSampleBias({STD_EMISSIVE_TEXTURE_NAME}, {STD_EMISSIVE_TEXTURE_NAME}Sampler, {STD_EMISSIVE_TEXTURE_UV}, {STD_TEXTURE_BIAS})).{STD_EMISSIVE_TEXTURE_CHANNEL};
        // magnopus patched
        #endif
        // end magnopus patched
    #endif

    #ifdef STD_EMISSIVE_VERTEX
    dEmission = dEmission * saturate(vVertexColor.{STD_EMISSIVE_VERTEX_CHANNEL});
    #endif
}
`;
