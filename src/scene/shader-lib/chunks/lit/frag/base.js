export default /* glsl */`
uniform vec3 view_position;
// magnopus patched
#ifdef LOD_PASS
uniform int lod_level;
#endif
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
