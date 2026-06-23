// main shader of the lit fragment shader
export default /* glsl */`

#include "varyingsPS"
#include "litUserDeclarationPS"
#include "frontendDeclPS"

// magnopus patched
// Defined here (not in basePS) so the stereo UV helper is available to the
// standard material front-end (getOpacity/getAlbedo/...) in ALL passes — pick,
// prepass and shadow include frontendCodePS but not basePS, so a forward-only
// definition left alpha-cutout passes calling an undefined function.
#ifdef MAG_STEREO_TEXTURE
// Set by the engine but not declared anywhere else
uniform float view_index;

// stereoVideoType - 0: None, 1: SideBySide, 2: TopBottom
vec2 getStereoVideoUV(vec2 uv, int stereoVideoType, int isStereoFlipped) {
    vec2 stereoUV = uv;

    if (stereoVideoType > 0) {
        float isLeftEye = (view_index == float(isStereoFlipped)) ? 1.0 : 0.0;

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

#if defined(PICK_PASS) || defined(PREPASS_PASS)

    #include "frontendCodePS"
    #include "litUserCodePS"
    #include "litOtherMainPS"

#elif defined(SHADOW_PASS)

    #include "frontendCodePS"
    #include "litUserCodePS"
    #include "litShadowMainPS"

#else // FORWARD_PASS

    #include "litForwardDeclarationPS"
    #include "litForwardPreCodePS"
    #include "frontendCodePS"
    #include "litForwardPostCodePS"
    #include "litForwardBackendPS"
    #include "litUserCodePS"
    #include "litForwardMainPS"

#endif

`;
