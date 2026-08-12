export default /* wgsl */`
var sourceTexture: texture_2d<f32>;
var sourceTextureSampler: sampler;
uniform sourceInvResolution: vec2f;
varying uv0: vec2f;

#ifdef PREMULTIPLY
    var premultiplyTexture: texture_2d<f32>;
    var premultiplyTextureSampler: sampler;
#endif

#ifdef PREFILTER
    // x: threshold, y: knee
    uniform bloomThresholdKnee: vec2f;
#endif

@fragment
fn fragmentMain(input: FragmentInput) -> FragmentOutput {
    var output: FragmentOutput;

    let e: half3 = half3(textureSample(sourceTexture, sourceTextureSampler, input.uv0).rgb);

    #ifdef BOXFILTER
        var value: half3 = e;

        #ifdef PREMULTIPLY
            let premultiply: half = half(textureSample(premultiplyTexture, premultiplyTextureSampler, input.uv0).{PREMULTIPLY_SRC_CHANNEL});
            value *= premultiply;
        #endif
    #else

        let x: f32 = uniform.sourceInvResolution.x;
        let y: f32 = uniform.sourceInvResolution.y;

        let a: half3 = half3(textureSample(sourceTexture, sourceTextureSampler, vec2f(input.uv0.x - 2.0 * x, input.uv0.y + 2.0 * y)).rgb);
        let b: half3 = half3(textureSample(sourceTexture, sourceTextureSampler, vec2f(input.uv0.x,           input.uv0.y + 2.0 * y)).rgb);
        let c: half3 = half3(textureSample(sourceTexture, sourceTextureSampler, vec2f(input.uv0.x + 2.0 * x, input.uv0.y + 2.0 * y)).rgb);

        let d: half3 = half3(textureSample(sourceTexture, sourceTextureSampler, vec2f(input.uv0.x - 2.0 * x, input.uv0.y)).rgb);
        let f: half3 = half3(textureSample(sourceTexture, sourceTextureSampler, vec2f(input.uv0.x + 2.0 * x, input.uv0.y)).rgb);

        let g: half3 = half3(textureSample(sourceTexture, sourceTextureSampler, vec2f(input.uv0.x - 2.0 * x, input.uv0.y - 2.0 * y)).rgb);
        let h: half3 = half3(textureSample(sourceTexture, sourceTextureSampler, vec2f(input.uv0.x,           input.uv0.y - 2.0 * y)).rgb);
        let i: half3 = half3(textureSample(sourceTexture, sourceTextureSampler, vec2f(input.uv0.x + 2.0 * x, input.uv0.y - 2.0 * y)).rgb);

        let j: half3 = half3(textureSample(sourceTexture, sourceTextureSampler, vec2f(input.uv0.x - x, input.uv0.y + y)).rgb);
        let k: half3 = half3(textureSample(sourceTexture, sourceTextureSampler, vec2f(input.uv0.x + x, input.uv0.y + y)).rgb);
        let l: half3 = half3(textureSample(sourceTexture, sourceTextureSampler, vec2f(input.uv0.x - x, input.uv0.y - y)).rgb);
        let m: half3 = half3(textureSample(sourceTexture, sourceTextureSampler, vec2f(input.uv0.x + x, input.uv0.y - y)).rgb);

        var value: half3 = e * half(0.125);
        value += (a + c + g + i) * half(0.03125);
        value += (b + d + f + h) * half(0.0625);
        value += (j + k + l + m) * half(0.125);
    #endif

    #ifdef REMOVE_INVALID
        value = max(value, half3(0.0));
    #endif

    #ifdef PREFILTER
        // Soft-knee high-pass (Unity URP style): scales the contribution by how
        // far the pixel's luminance sits above the threshold, with a quadratic
        // knee region for a smooth transition. threshold = 0 is an identity.
        // Computed in f32 — half precision is too coarse around the knee.
        let fullValue: vec3f = vec3f(value);
        let luma: f32 = max(fullValue.r, max(fullValue.g, fullValue.b));
        let knee: f32 = uniform.bloomThresholdKnee.y;
        var soft: f32 = clamp(luma - uniform.bloomThresholdKnee.x + knee, 0.0, 2.0 * knee);
        soft = soft * soft / max(4.0 * knee, 1e-4);
        let contribution: f32 = max(soft, luma - uniform.bloomThresholdKnee.x) / max(luma, 1e-4);
        value *= half3(vec3f(clamp(contribution, 0.0, 1.0)));
    #endif

    output.color = vec4f(vec3f(value), 1.0);
    return output;
}
`;
