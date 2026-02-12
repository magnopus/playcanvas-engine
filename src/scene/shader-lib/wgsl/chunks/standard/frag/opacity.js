export default /* wgsl */`
uniform material_opacity: f32;

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
uniform opacityStereoVideoType: i32; // 0: None, 1: SideBySide, 2: TopBottom
#endif
// end magnopus patched

fn getOpacity() {
    dAlpha = uniform.material_opacity;

    #ifdef STD_OPACITY_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
            var stereoUV: vec2f = getStereoVideoUV({STD_OPACITY_TEXTURE_UV}, uniform.opacityStereoVideoType);

            // Identical to the unpatched version except for the texture coordinates
            dAlpha = dAlpha * textureSampleBias({STD_OPACITY_TEXTURE_NAME}, {STD_OPACITY_TEXTURE_NAME}Sampler, stereoUV, uniform.textureBias).{STD_OPACITY_TEXTURE_CHANNEL};
        #else
        // end magnopus patched
    dAlpha = dAlpha * textureSampleBias({STD_OPACITY_TEXTURE_NAME}, {STD_OPACITY_TEXTURE_NAME}Sampler, {STD_OPACITY_TEXTURE_UV}, uniform.textureBias).{STD_OPACITY_TEXTURE_CHANNEL};
        // magnopus patched
        #endif
        // end magnopus patched
    #endif

    #ifdef STD_OPACITY_VERTEX
    dAlpha = dAlpha * clamp(vVertexColor.{STD_OPACITY_VERTEX_CHANNEL}, 0.0, 1.0);
    #endif
}
`;
