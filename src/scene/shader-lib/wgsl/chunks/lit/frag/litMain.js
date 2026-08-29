// main shader of the lit fragment shader
export default /* wgsl */`

#include "varyingsPS"
#include "litEngineDeclarationPS"
#include "litUserDeclarationPS"
#include "frontendDeclPS"
#include "outlineDeclarationPS"

#if defined(PICK_PASS) || defined(PREPASS_PASS)

    #include "frontendCodePS"
    #include "litEngineCodePS"
    #include "litUserCodePS"
    #include "litOtherMainPS"

#elif defined(SHADOW_PASS)

    #include "frontendCodePS"
    #include "litEngineCodePS"
    #include "litUserCodePS"
    #include "litShadowMainPS"

#else // FORWARD_PASS

    #include "litForwardDeclarationPS"
    #include "litForwardPreCodePS"
    #include "frontendCodePS"
    #include "litForwardPostCodePS"
    #include "litForwardBackendPS"
    #include "litEngineCodePS"
    #include "litUserCodePS"
    #include "litForwardMainPS"

#endif

`;
