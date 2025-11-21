export default /* glsl */`
uniform vec3 material_diffuse;

// magnopus patched
#ifdef MAG_STEREO_TEXTURE
    uniform int diffuseStereoVideoType; // 0: None, 1: SideBySide, 2: TopBottom
#endif
// end magnopus patched

#ifdef STD_DIFFUSEDETAIL_TEXTURE
    #include "detailModesPS"
#endif

void getAlbedo() {
    dAlbedo = material_diffuse.rgb;

    #ifdef STD_DIFFUSE_TEXTURE
        // magnopus patched
        #ifdef MAG_STEREO_TEXTURE
            vec2 stereoUV = getStereoVideoUV({STD_DIFFUSE_TEXTURE_UV}, diffuseStereoVideoType);

            // Identical to the unpatched version except for the texture coordinates
            vec3 albedoTexture = {STD_DIFFUSE_TEXTURE_DECODE}(texture2DBias({STD_DIFFUSE_TEXTURE_NAME}, stereoUV, textureBias)).{STD_DIFFUSE_TEXTURE_CHANNEL};
        #else
        // end magnopus patched

        vec3 albedoTexture = {STD_DIFFUSE_TEXTURE_DECODE}(texture2DBias({STD_DIFFUSE_TEXTURE_NAME}, {STD_DIFFUSE_TEXTURE_UV}, textureBias)).{STD_DIFFUSE_TEXTURE_CHANNEL};

        // magnopus patched
        #endif
        // end magnopus patched

        #ifdef STD_DIFFUSEDETAIL_TEXTURE
            vec3 albedoDetail = {STD_DIFFUSEDETAIL_TEXTURE_DECODE}(texture2DBias({STD_DIFFUSEDETAIL_TEXTURE_NAME}, {STD_DIFFUSEDETAIL_TEXTURE_UV}, textureBias)).{STD_DIFFUSEDETAIL_TEXTURE_CHANNEL};
            albedoTexture = detailMode_{STD_DIFFUSEDETAIL_DETAILMODE}(albedoTexture, albedoDetail);
        #endif

        dAlbedo *= albedoTexture;
    #endif

    #ifdef STD_DIFFUSE_VERTEX
        dAlbedo *= gammaCorrectInput(saturate(vVertexColor.{STD_DIFFUSE_VERTEX_CHANNEL}));
    #endif
}
`;
