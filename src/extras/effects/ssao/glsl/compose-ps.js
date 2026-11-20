export default /* glsl */`
    uniform sampler2D ssaoTexture;

    // Global variable for debug
    float dSsao;

    vec3 applySsao(vec3 color, vec2 uv) {

        // the occlusion is applied before the composition - by the lit shaders as the scene renders
        // in the lighting mode, to the scene before it is resolved in the combine mode - and it is
        // sampled here for the debug view only
        #if DEBUG_COMPOSE == ssao
            dSsao = texture2DLod(ssaoTexture, uv, 0.0).r;
        #endif
        return color;
    }

    // the 'ssao' debug view, displaying the occlusion
    vec3 debugSsao() {
        return vec3(dSsao);
    }
`;
