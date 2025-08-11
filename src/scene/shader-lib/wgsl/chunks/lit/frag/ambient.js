export default /* wgsl */ `
#if LIT_AMBIENT_SOURCE == AMBIENTSH
	uniform ambientSH: array<vec3f, 9>;
#endif

#include "envAtlasPS"
#if LIT_AMBIENT_SOURCE == ENVALATLAS
	#include "envAtlasPS"
	#ifndef ENV_ATLAS
		#define ENV_ATLAS
		var texture_envAtlas: texture_2d_array<f32>;
		var texture_envAtlasSampler: sampler;
	#endif
#endif
#ifndef ENV_ATLAS_PROBE
#define ENV_ATLAS_PROBE
    uniform probeCount: u32;
    uniform probePosition: array<vec3f, 4>;
    uniform probeBlendRadius: array<f32, 4>;
    uniform probeArrayIndex: array<f32, 4>;

    struct Probe {
        position: vec3f,
        blendRadius: f32,
        arrayIndex: f32, // stored as float to match uniform array packing
    };

    fn getProbe(i: u32) -> Probe {
        let pos = uniform.probePosition[i];
        let radius = uniform.probeBlendRadius[i].element;
        let idx = uniform.probeArrayIndex[i].element;
        return Probe(pos, radius, idx);
    }
#endif
fn sampleReflectionProbes(worldPos: vec3f, worldNormal: vec3f) -> vec3f {
    var totalWeight = 0.0;
    var accumulated = vec3f(0.0);
    let radius = 5.0;
    var probePosition: array<vec3f,4>;
    probePosition[0] = vec3f(-2.5, 1.0, 0.0);
    probePosition[1] = vec3f(2.5, 1.0, 0.0);
    for (var i: u32 = 0u; i < 2; i = i + 1u) {
        let probe = getProbe(i);
        let dist = distance(worldPos, probePosition[i]);
        if (dist < radius) {
            let weight = max(0.0, 1.0 - (dist / radius));
            totalWeight += weight;
            // Sample the probe's atlas layer
            let dir = normalize(cubeMapRotate(worldNormal) * vec3f(-1.0, 1.0, 1.0));
            let uv = mapUv(toSphericalUv(dir), vec4f(128.0, 256.0 + 128.0, 64.0, 32.0) / atlasSize);

            let raw = textureSampleLevel(texture_envAtlas, texture_envAtlasSampler, uv, i, 0);
            let linear = {ambientDecode}(raw);
            accumulated  += linear * weight;
        }
    }

    if (totalWeight > 0.0) {
        return accumulated / totalWeight;
    }
    return vec3f(0.0);
}

fn addAmbient(worldNormal: vec3f) {
	#ifdef LIT_AMBIENT_SOURCE == AMBIENTSH
		let n: vec3f = cubeMapRotate(worldNormal);
		let color: vec3f =
			uniform.ambientSH[0] +
			uniform.ambientSH[1] * n.x +
			uniform.ambientSH[2] * n.y +
			uniform.ambientSH[3] * n.z +
			uniform.ambientSH[4] * n.x * n.z +
			uniform.ambientSH[5] * n.z * n.y +
			uniform.ambientSH[6] * n.y * n.x +
			uniform.ambientSH[7] * (3.0 * n.z * n.z - 1.0) +
			uniform.ambientSH[8] * (n.x * n.x - n.y * n.y);
		dDiffuseLight += processEnvironment(max(color, vec3f(0.0)));
	#endif
	#if LIT_AMBIENT_SOURCE == ENVALATLAS
        let blendedReflection = sampleReflectionProbes(vPositionW, worldNormal);
        dDiffuseLight += processEnvironment(blendedReflection);
	#endif
	#if LIT_AMBIENT_SOURCE == CONSTANT
		dDiffuseLight += uniform.light_globalAmbient;
	#endif
}
`;
