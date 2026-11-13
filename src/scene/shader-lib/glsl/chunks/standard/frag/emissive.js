export default /* glsl */`
#ifndef MESH_COLOR
    uniform vec3 material_emissive;
#endif
uniform float material_emissiveIntensity;

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
uniform int emissiveStereoVideoType; // 0: None, 1: SideBySide, 2: TopBottom
uniform uint emissiveIsStereoFlipped; // 0: Not Flipped, 1: Flipped
#endif
// end magnopus patched

void getEmission() {
    #ifdef MESH_COLOR
        dEmission = mesh_color.rgb * material_emissiveIntensity;
    #else
        dEmission = material_emissive * material_emissiveIntensity;
    #endif

    #ifdef STD_EMISSIVE_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
            vec2 stereoUV = getStereoVideoUV({STD_EMISSIVE_TEXTURE_UV}, emissiveStereoVideoType, emissiveIsStereoFlipped);

            // Identical to the unpatched version except for the texture coordinates
            dEmission *= {STD_EMISSIVE_TEXTURE_DECODE}(texture2DBias({STD_EMISSIVE_TEXTURE_NAME}, stereoUV, {STD_TEXTURE_BIAS})).{STD_EMISSIVE_TEXTURE_CHANNEL};
        #else
        // end magnopus patched
    dEmission *= {STD_EMISSIVE_TEXTURE_DECODE}(texture2DBias({STD_EMISSIVE_TEXTURE_NAME}, {STD_EMISSIVE_TEXTURE_UV}, {STD_TEXTURE_BIAS})).{STD_EMISSIVE_TEXTURE_CHANNEL};
        // magnopus patched
        #endif
        // end magnopus patched
    #endif

    #ifdef STD_EMISSIVE_VERTEX
    dEmission *= saturate(vVertexColor.{STD_EMISSIVE_VERTEX_CHANNEL});
    #endif
}
`;
