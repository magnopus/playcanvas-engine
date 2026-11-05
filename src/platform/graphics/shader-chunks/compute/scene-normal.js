/**
 * WGSL compute shader chunk giving access to the scene normal map of a camera, attached to the
 * compute instance with {@link Compute#setSceneNormalMap}.
 *
 * Usage in compute shaders: #include "sceneNormalCS"
 *
 * Provides:
 * - sceneNormalSize() -> vec2u: the dimensions of the normal map.
 * - sceneNormalViewport() -> vec4u: the viewport the camera rendered the normal map with, in texels.
 * - sceneNormalCoverage(texel: vec2i) -> f32: 1 where a surface was rendered at the texel, 0 where
 * nothing was.
 * - sceneNormalWorld(texel: vec2i) -> vec3f: the world space unit normal of the surface at the
 * texel, or a zero vector where nothing was rendered.
 *
 * @ignore
 */
export default /* wgsl */`
var computeSceneNormalMap: texture_2d<uff>;

// the viewport the normal map was rendered with, in texels: x and y of its first texel, width and height
uniform computeSceneNormalViewport: vec4u;

fn sceneNormalSize() -> vec2u {
    return textureDimensions(computeSceneNormalMap, 0);
}

fn sceneNormalViewport() -> vec4u {
    return uniform.computeSceneNormalViewport;
}

fn sceneNormalCoverage(texel: vec2i) -> f32 {
    return textureLoad(computeSceneNormalMap, texel, 0).a;
}

fn sceneNormalWorld(texel: vec2i) -> vec3f {
    // the normal is stored premultiplied by the coverage, which is 1 for the opaque geometry and 0
    // where nothing was rendered
    let value = textureLoad(computeSceneNormalMap, texel, 0);
    return select(vec3f(0.0), normalize(value.rgb), value.a > 0.0);
}
`;
