// @ts-nocheck
import { Color } from '../../core/math/color.js';
import { Texture } from '../../platform/graphics/texture.js';
import { RenderTarget } from '../../platform/graphics/render-target.js';
import { ADDRESS_CLAMP_TO_EDGE, FILTER_NEAREST, PIXELFORMAT_111110F } from '../../platform/graphics/constants.js';
import { RenderPassShaderQuad } from '../../scene/graphics/render-pass-shader-quad.js';
import { shaderChunks } from '../../scene/shader-lib/chunks/chunks.js';

const MAX_LIGHTS = 8;

const tentFilterFs = /* glsl */ `
// 9-tap bilinear upsampler (tent filter)
mediump vec4 UpsampleTent(sampler2D tex, vec2 uv, vec2 texelSize, vec4 sampleScale)
{
    vec4 d = texelSize.xyxy * vec4(1.0, 1.0, -1.0, 0.0) * sampleScale;

    mediump vec4 s;
    s =  texture2D(tex, uv - d.xy);
    s += texture2D(tex, uv - d.wy) * 2.0;
    s += texture2D(tex, uv - d.zy);

    s += texture2D(tex, uv + d.zw) * 2.0;
    s += texture2D(tex, uv       ) * 4.0;
    s += texture2D(tex, uv + d.xw) * 2.0;

    s += texture2D(tex, uv + d.zy);
    s += texture2D(tex, uv + d.wy) * 2.0;
    s += texture2D(tex, uv + d.xy);

    return s * (1.0 / 16.0);
}
`;

const fsUpscale = /* glsl */ `
  uniform sampler2D uVolumetricsBuffer;
  uniform sampler2D uColorBuffer;
  uniform vec2 uSize;


  in vec2 uv0;
  ${tentFilterFs}

  void main() {
    vec4 volumetrics =  UpsampleTent(uVolumetricsBuffer, uv0, 1.0/uSize, vec4(1.0));

    vec4 colorSample = texture2D(uColorBuffer, uv0);

    gl_FragColor = vec4(volumetrics.rgb + colorSample.rgb, 1.0);
  }
`;

const fs = /* glsl */  `
#define MAX_LIGHTS 8
precision highp float;
uniform mat4 matrix_projection;
uniform mat4 matrix_viewProjection;
uniform mat4 matrix_inverseViewProjection;
uniform vec3 view_position;
uniform vec4 uLightProps[MAX_LIGHTS];
uniform mat4 matrix_lightmodel[MAX_LIGHTS];
uniform vec3 uLightColor[MAX_LIGHTS];
uniform int uLightCount;
uniform float uTime;
uniform sampler2DShadow uShadowMap[MAX_LIGHTS];
uniform mat4 uLightViewProjMatrix[MAX_LIGHTS];
uniform sampler2D uSceneDepthMap;
in vec2 uv0;
float random(vec2 co) {
	return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}
const mat4 LIGHT_DIRECTION_CORRECTION = mat4(
	1.0, 0.0, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.0, -1.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 1.0
);
vec3 calculateVolumetricLight(
	vec3 rayStart,
	vec3 rayDir,
	float stepSize,
	int numSteps,
	mat4 lightModelMatrix,
	mat4 lightViewProjMatrix,
	sampler2DShadow shadowMap,
	vec3 lightColor,
	float lightIntensity,
	float innerConeAngle,
	float outerConeAngle,
	float lightRange
) {
	vec3 lightPos = (lightModelMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
	vec3 lightDir = normalize((lightModelMatrix * LIGHT_DIRECTION_CORRECTION * vec4(0.0, 0.0, -1.0, 0.0)).xyz);
	float cosOuterCone = cos(outerConeAngle);
	float cosInnerCone = cos(innerConeAngle);
	vec3 finalColor = vec3(0.0);
	float randomOffset = random(uv0) * stepSize;
	for (int i = 0; i < numSteps; i++) {
		float t = stepSize * float(i) + randomOffset;
		vec3 samplePos = rayStart + rayDir * t;
		vec3 lightToSample = samplePos - lightPos;
		float distToLight = length(lightToSample);
		if (distToLight > lightRange) continue;
		float cosAngle = dot(normalize(lightToSample), -lightDir);
		if (cosAngle > cosOuterCone) {
			vec4 samplePosLightSpace = lightViewProjMatrix * vec4(samplePos, 1.0);
			samplePosLightSpace.xyz /= samplePosLightSpace.w;
			vec3 shadowCoord = samplePosLightSpace.xyz;
			//shadowCoord.z = shadowCoord.z * 0.5 + 0.5;
			if (shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 &&
				shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0 &&
				shadowCoord.z >= 0.0 && shadowCoord.z <= 1.0) {
				float bias = 0.01;
			float shadowMapDepth = texture(shadowMap, vec3(shadowCoord.xy, shadowCoord.z));
				float near = 0.005;
				float far = 5.0;
				float linearDepth = (2.0 * near * far) / (far + near - (shadowMapDepth * 2.0 - 1.0) * (far - near));
				float coord = (2.0 * near * far) / (far + near - (shadowCoord.z * 2.0 - 1.0) * (far - near));
				float depthLightSpace = shadowCoord.z;
				float depthShadowMap = shadowMapDepth;
				float shadow = (depthLightSpace - bias) <= depthShadowMap ? 1.0 : 0.0;
				float attenuation = 1.0 / (distToLight * distToLight);
				float invAttenuation = smoothstep(0.25, 0.8, distToLight);
				float spotEffect = smoothstep(cosOuterCone, cosInnerCone, cosAngle);
				float absorption = exp(-distToLight * 0.1);
				float noise = 1.0;
				finalColor += lightColor * spotEffect * attenuation * absorption * shadow;
			}
		}
	}
	return finalColor;
}
void main() {
	float depth = texture2D(uSceneDepthMap, uv0).r;
	vec4 clipSpace = vec4(uv0 * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
	vec4 viewSpace = matrix_inverseViewProjection * clipSpace;
	vec3 worldPos = viewSpace.xyz / viewSpace.w;
	vec3 rayStart = view_position;
	vec3 rayDir = normalize(worldPos - rayStart);
	float rayLength = length(worldPos - rayStart);
	float stepSize = 0.1;
	float maxDistance = 50.0;
	int numOfSteps = int(min(maxDistance / stepSize, rayLength / stepSize));
	vec3 totalVolumetricLighting = vec3(0.0);
	for (int v = 0; v < MAX_LIGHTS; v++) {
		if (v > uLightCount) continue;
		int i = v;
		float innerConeAngle = uLightProps[i].x;
		float outerConeAngle = uLightProps[i].y;
		float lightIntensity = uLightProps[i].z;
		float lightRange = uLightProps[i].w;
		 switch (int(v)){
			case 0:
				totalVolumetricLighting += calculateVolumetricLight(
					rayStart,
					rayDir,
					stepSize,
					numOfSteps,
					matrix_lightmodel[i],
					uLightViewProjMatrix[i],
					uShadowMap[0],
					uLightColor[i],
					lightIntensity,
					innerConeAngle,
					outerConeAngle,
					lightRange
				);
				break;
			case 1:
				totalVolumetricLighting += calculateVolumetricLight(
					rayStart,
					rayDir,
					stepSize,
					numOfSteps,
					matrix_lightmodel[i],
					uLightViewProjMatrix[i],
					uShadowMap[1],
					uLightColor[i],
					lightIntensity,
					innerConeAngle,
					outerConeAngle,
					lightRange
				);
				break;
			case 2:
				totalVolumetricLighting += calculateVolumetricLight(
					rayStart,
					rayDir,
					stepSize,
					numOfSteps,
					matrix_lightmodel[i],
					uLightViewProjMatrix[i],
					uShadowMap[2],
					uLightColor[i],
					lightIntensity,
					innerConeAngle,
					outerConeAngle,
					lightRange
				);
				break;
			case 3:
				totalVolumetricLighting += calculateVolumetricLight(
					rayStart,
					rayDir,
					stepSize,
					numOfSteps,
					matrix_lightmodel[i],
					uLightViewProjMatrix[i],
					uShadowMap[3],
					uLightColor[i],
					lightIntensity,
					innerConeAngle,
					outerConeAngle,
					lightRange
				);
				break;
		 }
	}
	gl_FragColor = vec4(clamp(totalVolumetricLighting, 0.0, 1.0), 1.0);
}

`;

/**
 * Volumetric light pass
 *
 * @category Graphics
 * @ignore
 */
class RenderPassVolumetricLight extends RenderPassShaderQuad {
    textureFormat;

    /**
     * The intensity.
     *
     * @type {number}
     */
    intensity = 1;

    /**
     * The number of samples to take.
     *
     * @type {number}
     */
    sampleCount = 10;

    /**
     * The texture containing the occlusion information in the red channel.
     *
     * @type {Texture}
     * @readonly
     */
    volumetricsTexture;

    /** @type {number} */
    _scale = 0.1;

    /** @type {AppBase} */
    app;

    constructor(app, device, sourceTexture, textureFormat) {
        super(device);
        this.app = app;
        this.textureFormat = textureFormat;
        this.sourceTexture = sourceTexture;

        // main Volumetric render pass
        this.shader = this.createQuadShader('VolumeShader', fs);

        const rt = this.createRenderTarget('VolumeFinalTexture');
        this.volumetricsTexture = rt.colorBuffer;

        this.init(rt, {
            resizeSource: this.sourceTexture,
            scaleX: 0.5,
            scaleY: 0.5
        });

        // clear the color to avoid load op
        const clearColor = new Color(0, 0, 0, 0);
        this.setClearColor(clearColor);

        this.volumetricsTextureId = device.scope.resolve('volumetricsTexture');
    }

    destroy() {

        this.renderTarget?.destroyTextureBuffers();
        this.renderTarget?.destroy();
        this.renderTarget = null;

        if (this.afterPasses.length > 0) {
            const blurRt = this.afterPasses[0].renderTarget;
            blurRt?.destroyTextureBuffers();
            blurRt?.destroy();
        }

        this.afterPasses.forEach(pass => pass.destroy());
        this.afterPasses.length = 0;

        super.destroy();
    }

    /**
     * The scale multiplier for the render target size.
     *
     * @type {number}
     */
    set scale(value) {
        this._scale = value;
        this.options.scaleX = value;
        this.options.scaleY = value;
    }

    get scale() {
        return this._scale;
    }

    createRenderTarget(name) {
        return new RenderTarget({
            depth: false,
            colorBuffer: new Texture(this.device, {
                name: name,
                width: 1,
                height: 1,
                format: this.textureFormat,
                mipmaps: false,
                minFilter: FILTER_NEAREST,
                magFilter: FILTER_NEAREST,
                addressU: ADDRESS_CLAMP_TO_EDGE,
                addressV: ADDRESS_CLAMP_TO_EDGE
            })
        });
    }

    execute() {
        if (!this.renderTarget?.colorBuffer) return;
        const { device, sourceTexture, sampleCount, scale } = this;
        const { width, height } = this.renderTarget.colorBuffer;
        const scope = device.scope;


        const lights = this.app.root.findComponents('light');
        let volumetricLights = lights.filter(
            light => light.enabled && light.light.visibleThisFrame && light.type === 'spot'
        );
        volumetricLights = volumetricLights.slice(0, MAX_LIGHTS);
        scope.resolve('uLightCount').setValue(volumetricLights.length);
        const values = {
            uLightProps: [],
            uLightColour: [],
            matrix_lightmodel: [],
            uShadowMap: [],
            uLightViewProjMatrix: []
        };
        for (let i = 0; i < volumetricLights.length; i++) {
            const light = volumetricLights[i];
            const { r, g, b } = light.color;
            values.uLightProps = [
                ...values.uLightProps,
                (light.innerConeAngle * Math.PI) / 180,
                (light.outerConeAngle * Math.PI) / 180,
                light.intensity,
                light.range ?? 20
            ];
            values.uLightColour = [...values.uLightColour, r, g, b];
            values.uShadowMap.push(light.light._shadowMap.texture);
            values.uLightViewProjMatrix = [...values.uLightViewProjMatrix, ...light.light._renderData[0].shadowMatrix.data];
            values.matrix_lightmodel = [...values.matrix_lightmodel, ...Array.from(light.entity.getWorldTransform().data)];
        }
        if (volumetricLights.length) {
            scope.resolve('uLightProps[0]').setValue(values.uLightProps);
            scope.resolve('uLightColor[0]').setValue(values.uLightColour);
            scope.resolve('matrix_lightmodel[0]').setValue(values.matrix_lightmodel);
            scope.resolve('uShadowMap[0]').setValue(values.uShadowMap);
            scope.resolve('uLightViewProjMatrix[0]').setValue(values.uLightViewProjMatrix);
            super.execute();
        }

    }

    after() {
        this.volumetricsTextureId.setValue(this.volumetricsTexture);
    }
}

export { RenderPassVolumetricLight };
