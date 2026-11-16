export default /* wgsl */`
#include "screenDepthPS"

varying uv0: vec2f;

var sourceTexture: texture_2d<f32>;
var sourceTextureSampler: sampler;
uniform sourceInvResolution: vec2f;
uniform filterSize: i32;

fn random(w: vec2f) -> f32 {
    const m: vec3f = vec3f(0.06711056, 0.00583715, 52.9829189);
    return fract(m.z * fract(dot(w, m.xy)));
}

// A sample blends when it lies on the centre pixel's surface: its depth is compared with the
// depth extrapolated along the surface's slope, relative to the depth. Absolute units blended a
// column with the wall a metre behind it at any distance (a halo around every silhouette), and a
// flat test rejects a sloped or curved surface's own neighbours, leaving the sampling noise.
// Relative to the plane, a sample about 5% of the distance off it no longer blends.
fn bilateralWeight(expected: f32, sampleDepth: f32) -> f32 {
    let diff: f32 = (sampleDepth - expected) / max(expected, 1e-4) * 20.0;
    return max(0.0, 1.0 - diff * diff);
}

fn tap(sum_ptr: ptr<function, f32>, totalWeight_ptr: ptr<function, f32>, expected: f32, position: vec2f) {
    let color: f32 = textureSample(sourceTexture, sourceTextureSampler, position).r;
    let textureDepth: f32 = -getLinearScreenDepth(position);
    let bilateral: f32 = bilateralWeight(expected, textureDepth);
    *sum_ptr = *sum_ptr + color * bilateral;
    *totalWeight_ptr = *totalWeight_ptr + bilateral;
}

@fragment
fn fragmentMain(input: FragmentInput) -> FragmentOutput {
    var output: FragmentOutput;

    #ifdef HORIZONTAL
        let texelStep: vec2f = vec2f(uniform.sourceInvResolution.x, 0.0);
    #else
        let texelStep: vec2f = vec2f(0.0, uniform.sourceInvResolution.y);
    #endif

    // handle the center pixel separately because it doesn't participate in bilateral filtering
    let depth: f32 = -getLinearScreenDepth(input.uv0);
    var totalWeight: f32 = 1.0;
    let color: f32 = textureSample(sourceTexture, sourceTextureSampler, input.uv0).r;
    var sum: f32 = color;

    // the surface's depth slope along the blur axis: the flatter of the two one-sided differences,
    // so a silhouette next to the pixel does not tilt the plane
    let before: f32 = depth - (-getLinearScreenDepth(input.uv0 - texelStep));
    let after: f32 = (-getLinearScreenDepth(input.uv0 + texelStep)) - depth;
    let slope: f32 = select(after, before, abs(before) < abs(after));

    for (var i: i32 = -uniform.filterSize; i <= uniform.filterSize; i = i + 1) {
        if (i == 0) { continue; }
        tap(&sum, &totalWeight, depth + f32(i) * slope, input.uv0 + f32(i) * texelStep);
    }

    let ao: f32 = sum / totalWeight;

    // simple dithering helps a lot (assumes 8 bits target)
    // this is most useful with high quality/large blurs
    // ao += ((random(gl_FragCoord.xy) - 0.5) / 255.0);

    output.color = vec4f(ao, ao, ao, 1.0);
    return output;
}
`;
