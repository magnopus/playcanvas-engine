export default /* glsl */`
vec4 encodePickOutput(uint id) {
    const vec4 inv = vec4(1.0 / 255.0);
    const uvec4 shifts = uvec4(16, 8, 0, 24);
    uvec4 col = (uvec4(id) >> shifts) & uvec4(0xff);
    return vec4(col) * inv;
}

#ifndef PICK_CUSTOM_ID
    uniform uint meshInstanceId;

    vec4 getPickOutput() {
        return encodePickOutput(meshInstanceId);
    }
#endif

#ifdef DEPTH_PICK_PASS
    #include "floatAsUintPS"

    vec4 getPickDepth() {
        // emit forward-z depth (0=near, 1=far) regardless of hardware convention so that
        // the picker decoder can always treat the encoded value as standard NDC depth
        #ifdef REVERSE_Z
            return float2uint(1.0 - gl_FragCoord.z);
        #else
            return float2uint(gl_FragCoord.z);
        #endif
    }
#endif
`;
