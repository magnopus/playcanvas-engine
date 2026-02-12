export default /* glsl */ `
uniform vec3 material_emissive;
uniform float material_emissiveIntensity;

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
uniform int emissiveStereoVideoType; // 0: None, 1: SideBySide, 2: TopBottom
#endif
// end magnopus patched

void getEmission() {
    dEmission = material_emissive * material_emissiveIntensity;

    #ifdef STD_EMISSIVE_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
            vec2 stereoUV = getStereoVideoUV({STD_EMISSIVE_TEXTURE_UV}, emissiveStereoVideoType);

            // Identical to the unpatched version except for the texture coordinates
            dEmission *= {STD_EMISSIVE_TEXTURE_DECODE}(texture2DBias({STD_EMISSIVE_TEXTURE_NAME}, stereoUV, textureBias)).{STD_EMISSIVE_TEXTURE_CHANNEL};
        #else
        // end magnopus patched
    dEmission *= {STD_EMISSIVE_TEXTURE_DECODE}(texture2DBias({STD_EMISSIVE_TEXTURE_NAME}, {STD_EMISSIVE_TEXTURE_UV}, textureBias)).{STD_EMISSIVE_TEXTURE_CHANNEL};
        // magnopus patched
        #endif
        // end magnopus patched
    #endif

    #ifdef STD_EMISSIVE_VERTEX
    dEmission *= saturate(vVertexColor.{STD_EMISSIVE_VERTEX_CHANNEL});
    #endif
}
`;
