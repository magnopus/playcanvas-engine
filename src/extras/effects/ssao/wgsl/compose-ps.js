export default /* wgsl */`
    var ssaoTexture: texture_2d<f32>;
    var ssaoTextureSampler: sampler;

    // Global variable for debug
    var<private> dSsao: f32;

    fn applySsao(color: vec3f, uv: vec2f) -> vec3f {

        // the occlusion is applied before the composition - by the lit shaders as the scene renders
        // in the lighting mode, to the scene before it is resolved in the combine mode - and it is
        // sampled here for the debug view only
        #if DEBUG_COMPOSE == ssao
            dSsao = textureSampleLevel(ssaoTexture, ssaoTextureSampler, uv, 0.0).r;
        #endif
        return color;
    }

    // the 'ssao' debug view, displaying the occlusion
    fn debugSsao() -> vec3f {
        return vec3f(dSsao);
    }
`;
