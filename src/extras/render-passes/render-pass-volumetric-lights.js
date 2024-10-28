// @ts-nocheck
import { Color } from '../../core/math/color.js';
import { Texture } from '../../platform/graphics/texture.js';
import { RenderTarget } from '../../platform/graphics/render-target.js';
import { ADDRESS_CLAMP_TO_EDGE, FILTER_NEAREST } from '../../platform/graphics/constants.js';
import { RenderPassShaderQuad } from '../../scene/graphics/render-pass-shader-quad.js';

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

// Existing uniforms remain the same
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

// Improved noise function that maintains temporal stability
float gold_noise(vec2 coordinate, float seed) {
    float phi = 1.61803398874989484820459;
    float alpha = fract(dot(coordinate, vec2(phi, phi * phi)) * seed);
    return fract(sin(alpha) * 43758.5453123);
}

// Light scattering phase function
float henyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
}

const mat4 LIGHT_DIRECTION_CORRECTION = mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0
);

struct LightSample {
    float shadow;
    float distToLight;
    float cosAngle;
    vec3 lightToSample;
};

// Separate shadow sampling function to handle the WebGL2 sampler array limitation
float getShadowSample(int lightIndex, vec3 shadowCoord, float bias) {
    // Using early returns to potentially skip unnecessary checks
    if (lightIndex >= MAX_LIGHTS) return 0.0;
    if (any(lessThan(shadowCoord, vec3(0.0))) || any(greaterThan(shadowCoord, vec3(1.0)))) return 0.0;
    
    switch (lightIndex) {
        case 0: return texture(uShadowMap[0], vec3(shadowCoord.xy, shadowCoord.z - bias));
        case 1: return texture(uShadowMap[1], vec3(shadowCoord.xy, shadowCoord.z - bias));
        case 2: return texture(uShadowMap[2], vec3(shadowCoord.xy, shadowCoord.z - bias));
        case 3: return texture(uShadowMap[3], vec3(shadowCoord.xy, shadowCoord.z - bias));
        default: return 0.0;
    }
}

vec3 calculateVolumetricLight(
    vec3 rayStart,
    vec3 rayDir,
    float stepSize,
    int numSteps,
    mat4 lightModelMatrix,
    mat4 lightViewProjMatrix,
    int lightIndex,
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
    
    // Initialize accumulation variables
    vec3 finalColor = vec3(0.0);
    float transmittance = 1.0;
    
    // Temporal jittering for noise reduction
    float timeOffset = fract(uTime * 0.1);
    float randomOffset = gold_noise(uv0, timeOffset) * stepSize;
    
    // Calculate base distance for adaptive sampling
    float baseDistance = length(lightPos - rayStart);
    float maxDistance = min(lightRange * 1.2, length(rayDir) * float(numSteps) * stepSize);
    
    for (int i = 0; i < numSteps && transmittance > 0.01; i++) {
        // Adaptive step size based on distance from light
        float currentDist = float(i) * stepSize;
        float adaptiveStep = stepSize * (1.0 + currentDist * 0.1);
        float t = currentDist + randomOffset;
        
        // Early termination
        if (t > maxDistance) break;
        
        vec3 samplePos = rayStart + rayDir * t;
        vec3 lightToSample = samplePos - lightPos;
        float distToLight = length(lightToSample);
        
        if (distToLight > lightRange) continue;
        
        float cosAngle = dot(normalize(lightToSample), -lightDir);
        if (cosAngle > cosOuterCone) {
            // Transform to light space
            vec4 samplePosLightSpace = lightViewProjMatrix * vec4(samplePos, 1.0);
            vec3 shadowCoord = samplePosLightSpace.xyz / samplePosLightSpace.w;
            
            // Calculate bias based on angle
            float bias = 0.01 * (1.0 - cosAngle);
            
            // Get shadow value using the separate function
            float shadow = getShadowSample(lightIndex, shadowCoord, bias);
            
            if (shadow > 0.0) {
                // Calculate scattering
                float phase = henyeyGreenstein(dot(rayDir, normalize(lightToSample)), -0.2);
                
                // Distance attenuation with smoother falloff
                float attenuation = 1.0 / (1.0 + distToLight * distToLight * 0.02);
                
                // Improved spot light falloff
                float spotEffect = smoothstep(cosOuterCone, cosInnerCone, cosAngle);
                
                // Variable density using 3D noise
                float density = 0.1 * (1.0 + 0.5 * gold_noise(samplePos.xy * 0.1, timeOffset));
                
                // Calculate extinction
                float extinction = exp(-distToLight * density);
                
                // Accumulate light contribution
                vec3 contribution = lightColor * spotEffect * attenuation * extinction * shadow * phase;
                finalColor += contribution * transmittance * adaptiveStep;
                
                // Update transmittance
                transmittance *= exp(-density * adaptiveStep);
            }
        }
    }
    
    return finalColor * lightIntensity;
}

void main() {
    // Get world position from depth
    float depth = texture(uSceneDepthMap, uv0).r;
    vec4 clipSpace = vec4(uv0 * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 viewSpace = matrix_inverseViewProjection * clipSpace;
    vec3 worldPos = viewSpace.xyz / viewSpace.w;
    
    // Setup ray
    vec3 rayStart = view_position;
    vec3 rayDir = normalize(worldPos - rayStart);
    float rayLength = length(worldPos - rayStart);
    
    // Adaptive step size based on view distance
    float baseStepSize = 0.1;
    float adaptiveStepSize = baseStepSize * (1.0 + length(worldPos - rayStart) * 0.01);
    float maxDistance = 50.0;
    int numOfSteps = int(min(maxDistance / adaptiveStepSize, rayLength / adaptiveStepSize));
    
    vec3 totalVolumetricLighting = vec3(0.0);
    
    // Process each light
    for (int v = 0; v < MAX_LIGHTS; v++) {
        if (v >= uLightCount) break;
        
        float innerConeAngle = uLightProps[v].x;
        float outerConeAngle = uLightProps[v].y;
        float lightIntensity = uLightProps[v].z;
        float lightRange = uLightProps[v].w;
        
        totalVolumetricLighting += calculateVolumetricLight(
            rayStart,
            rayDir,
            adaptiveStepSize,
            numOfSteps,
            matrix_lightmodel[v],
            uLightViewProjMatrix[v],
            v,
            uLightColor[v],
            lightIntensity,
            innerConeAngle,
            outerConeAngle,
            lightRange
        );
    }
    
    // Tone mapping and exposure adjustment
    vec3 finalColor = totalVolumetricLighting / (vec3(1.0) + totalVolumetricLighting);
    
    gl_FragColor = vec4(finalColor, 1.0);
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
            scope.resolve('uTime').setValue(performance.now());
            super.execute();
        }

    }

    after() {
        this.volumetricsTextureId.setValue(this.volumetricsTexture);
    }
}

export { RenderPassVolumetricLight };
