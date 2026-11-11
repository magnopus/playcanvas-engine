export default /* glsl */`
#ifndef MESH_COLOR
    uniform float material_opacity;
#endif
uniform float material_alphaDitherScale;

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
uniform int opacityStereoVideoType; // 0: None, 1: SideBySide, 2: TopBottom
uniform uint opacityIsStereoFlipped; // 0: Not Flipped, 1: Flipped
#endif
// end magnopus patched

void getOpacity() {
    #ifdef MESH_COLOR
        dAlpha = mesh_color.a;
    #else
        dAlpha = material_opacity;
    #endif

    #ifdef STD_OPACITY_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
            vec2 stereoUV = getStereoVideoUV({STD_OPACITY_TEXTURE_UV}, opacityStereoVideoType, opacityIsStereoFlipped);

            // Identical to the unpatched version except for the texture coordinates
            dAlpha *= texture2DBias({STD_OPACITY_TEXTURE_NAME}, stereoUV, {STD_TEXTURE_BIAS}).{STD_OPACITY_TEXTURE_CHANNEL};
        #else
        // end magnopus patched
    dAlpha *= texture2DBias({STD_OPACITY_TEXTURE_NAME}, {STD_OPACITY_TEXTURE_UV}, {STD_TEXTURE_BIAS}).{STD_OPACITY_TEXTURE_CHANNEL};
        // magnopus patched
        #endif
        // end magnopus patched
    #endif

    #ifdef STD_OPACITY_VERTEX
    dAlpha *= clamp(vVertexColor.{STD_OPACITY_VERTEX_CHANNEL}, 0.0, 1.0);
    #endif
}
`;
