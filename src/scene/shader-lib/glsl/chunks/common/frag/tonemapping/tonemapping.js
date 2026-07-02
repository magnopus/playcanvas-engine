export default /* glsl */`

#ifndef TONEMAP_NO_EXPOSURE_UNIFORM
    #if TONEMAP != NONE
        uniform float exposure;
        float getExposure() { return exposure; }
    #else
        float getExposure() { return 1.0; }
    #endif
#endif

#if (TONEMAP == NONE)
    #include "tonemappingNonePS"
#elif TONEMAP == FILMIC
    #include "tonemappingFilmicPS"
#elif TONEMAP == LINEAR
    #include "tonemappingLinearPS"
#elif TONEMAP == HEJL
    #include "tonemappingHejlPS"
#elif TONEMAP == ACES
    #include "tonemappingAcesPS"
#elif TONEMAP == ACES2
    #include "tonemappingAces2PS"
    // magnopus patched - to remove exposure for cinematic camera preview
#elif TONEMAP == ACES2_NOEXPOSURE
    #include "tonemappingAces2NoExposurePS"
#elif TONEMAP == NEUTRAL
    #include "tonemappingNeutralPS"
#endif

`;
