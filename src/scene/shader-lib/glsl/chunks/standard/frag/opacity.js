export default /* glsl */`
uniform float material_opacity;

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
uniform int opacityStereoVideoType; // 0: None, 1: SideBySide, 2: TopBottom
#endif
// end magnopus patched

void getOpacity() {
    dAlpha = material_opacity;

    #ifdef STD_OPACITY_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
            vec2 stereoUV = getStereoVideoUV({STD_OPACITY_TEXTURE_UV}, opacityStereoVideoType);

            // Identical to the unpatched version except for the texture coordinates
            dAlpha *= texture2DBias({STD_OPACITY_TEXTURE_NAME}, stereoUV, textureBias).{STD_OPACITY_TEXTURE_CHANNEL};
        #else
        // end magnopus patched
    dAlpha *= texture2DBias({STD_OPACITY_TEXTURE_NAME}, {STD_OPACITY_TEXTURE_UV}, textureBias).{STD_OPACITY_TEXTURE_CHANNEL};
        // magnopus patched
        #endif
        // end magnopus patched
    #endif

    #ifdef STD_OPACITY_VERTEX
    dAlpha *= clamp(vVertexColor.{STD_OPACITY_VERTEX_CHANNEL}, 0.0, 1.0);
    #endif
}
`;
