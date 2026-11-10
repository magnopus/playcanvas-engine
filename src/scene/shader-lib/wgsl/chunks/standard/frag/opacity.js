export default /* wgsl */`
#ifndef MESH_COLOR
    uniform material_opacity: f32;
#endif
uniform material_alphaDitherScale: f32;

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
uniform opacityStereoVideoType: i32; // 0: None, 1: SideBySide, 2: TopBottom
uniform opacityIsStereoFlipped: u32; // 0: Not Flipped, 1: Flipped
#endif
// end magnopus patched

fn getOpacity() {
    #ifdef MESH_COLOR
        dAlpha = uniform.mesh_color.a;
    #else
        dAlpha = uniform.material_opacity;
    #endif

    #ifdef STD_OPACITY_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
            var stereoUV: vec2f = getStereoVideoUV({STD_OPACITY_TEXTURE_UV}, uniform.opacityStereoVideoType, uniform.opacityIsStereoFlipped);

            // Identical to the unpatched version except for the texture coordinates
            dAlpha = dAlpha * textureSampleBias({STD_OPACITY_TEXTURE_NAME}, {STD_OPACITY_TEXTURE_NAME}Sampler, stereoUV, {STD_TEXTURE_BIAS}).{STD_OPACITY_TEXTURE_CHANNEL};
        #else
        // end magnopus patched
    dAlpha = dAlpha * textureSampleBias({STD_OPACITY_TEXTURE_NAME}, {STD_OPACITY_TEXTURE_NAME}Sampler, {STD_OPACITY_TEXTURE_UV}, {STD_TEXTURE_BIAS}).{STD_OPACITY_TEXTURE_CHANNEL};
        // magnopus patched
        #endif
        // end magnopus patched
    #endif

    #ifdef STD_OPACITY_VERTEX
    dAlpha = dAlpha * clamp(vVertexColor.{STD_OPACITY_VERTEX_CHANNEL}, 0.0, 1.0);
    #endif
}
`;
