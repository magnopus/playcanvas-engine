export default /* glsl */`
uniform vec3 view_position;

// magnopus patched
#ifdef LOD_PASS
uniform int lod_level;
#endif
// end magnopus patched

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
// Set by the engine but not declared anywhere else
uniform float view_index;

// stereoVideoType - 0: None, 1: SideBySide, 2: TopBottom
vec2 getStereoVideoUV(vec2 uv, int stereoVideoType) {
    vec2 stereoUV = uv;

    if (stereoVideoType > 0) {
        float isLeftEye = float(view_index == 0.0);

        vec2 offset = vec2(0.0, 0.0);
        vec2 scale = vec2(1.0, 1.0);

        // SideBySide
        if (stereoVideoType == 1) {
            scale.x = 0.5;
            offset.x = (1.0 - isLeftEye) * 0.5;
        }
        // TopBottom
        else if (stereoVideoType == 2) {
            scale.y = 0.5;
            offset.y = (1.0 - isLeftEye) * 0.5;
        }

        stereoUV = stereoUV * scale + offset;
    }

    return stereoUV;
}
#endif
// end magnopus patched

uniform vec3 light_globalAmbient;

float square(float x) {
    return x*x;
}

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

vec3 saturate(vec3 x) {
    return clamp(x, vec3(0.0), vec3(1.0));
}
`;
