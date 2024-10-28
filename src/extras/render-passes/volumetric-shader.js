export const MAX_LIGHTS = 8;

export const voluemtricLightShader = /* glsl */ `
/**
 * Advanced Volumetric Light Shader
 * 
 * This shader creates volumetric lighting effects (also known as "god rays" or "light shafts")
 * by simulating how light scatters through participating media (fog, dust, etc.)
 * 
 * The algorithm works by:
 * 1. Ray marching from the camera through the scene
 * 2. Sampling points along each ray within light volumes
 * 3. Calculating light contribution at each sample point
 * 4. Accumulating the results with proper attenuation and shadowing
 */

#define MAX_LIGHTS ${MAX_LIGHTS}
precision highp float;

// Matrix uniforms for coordinate transformations
uniform mat4 matrix_projection;
uniform mat4 matrix_viewProjection;
uniform mat4 matrix_inverseViewProjection;
uniform vec3 view_position;

// Light properties for up to MAX_LIGHTS lights
uniform vec4 uLightProps[MAX_LIGHTS];           // x: innerConeAngle, y: outerConeAngle, z: intensity, w: range
uniform mat4 matrix_lightmodel[MAX_LIGHTS];     // World space transformation for each light
uniform vec3 uLightColor[MAX_LIGHTS];           // RGB color of each light
uniform int uLightCount;                        // Number of active lights

uniform float uTime;                            // Time for temporal variation
uniform sampler2DShadow uShadowMap;             // Shadow map for each light
uniform mat4 uLightViewProjMatrix[MAX_LIGHTS];  // Light space matrices for shadow mapping
uniform sampler2D uSceneDepthMap;               // Scene depth for ray length determination
uniform sampler2D uBlueNoiseTexture;            // Blue noise for dithering
in vec2 uv0;                                    // Screen space UV coordinates

/**
 * Rotates the sampling grid to break up artifacts
 * This helps prevent visible banding patterns in the volumetric effect
 */
vec2 getRotatedGrid(vec2 pos) {
    float rotation = 0.785398; // 45 degrees in radians
    float c = cos(rotation);
    float s = sin(rotation);
    mat2 rot = mat2(c, -s, s, c);
    return rot * pos;
}

/**
 * Samples blue noise texture with temporal variation
 * Blue noise provides better distribution than random noise
 * and helps break up banding artifacts
 */
float getBlueNoise(vec2 screenPos) {
    // Rotate sampling grid to avoid alignment with screen space
    vec2 rotatedPos = getRotatedGrid(screenPos);
    vec2 pixelPos = floor(rotatedPos);
    
    // Create temporal variation to prevent static noise patterns
    float frameOffset = floor(uTime * 60.0);
    vec2 noiseUV = mod((pixelPos + vec2(frameOffset * 13.0, frameOffset * 7.0)), 64.0) / 64.0;
    
    // Sample noise texture twice and blend for better distribution
    float noise1 = texture(uBlueNoiseTexture, noiseUV).r;
    float noise2 = texture(uBlueNoiseTexture, noiseUV + vec2(0.31, 0.57)).r;
    return mix(noise1, noise2, 0.5);
}

/**
 * Creates a complex noise pattern for ray marching
 * Combines multiple noise frequencies for natural-looking variation
 */
float getRayMarchingOffset(vec2 screenPos, float depth, float index) {
    // Sample noise at different frequencies and positions
    float noise1 = getBlueNoise(screenPos);                              // Base noise
    float noise2 = getBlueNoise(screenPos * 1.7 + vec2(index * 0.13));  // Higher frequency detail
    float noise3 = getBlueNoise((screenPos + depth * 0.1) * 0.77);      // Depth-dependent variation
    
    // Combine noise samples with different weights
    return (noise1 * 0.5 + noise2 * 0.3 + noise3 * 0.2);
}

/**
 * Finds the closest point on a ray to a given position
 * Used for determining sampling density near light sources
 */
vec3 closestPointOnRay(vec3 rayOrigin, vec3 rayDir, vec3 point) {
    vec3 v = point - rayOrigin;
    float t = dot(v, rayDir);
    return rayOrigin + rayDir * max(t, 0.0);
}

/**
 * Calculates ray-sphere intersection
 * Used to determine where rays enter and exit light volumes
 * Returns false if no intersection, true and intersection points if ray hits sphere
 */
bool rayIntersectsSphere(vec3 rayOrigin, vec3 rayDir, vec3 sphereCenter, float radius, out float t0, out float t1) {
    vec3 oc = rayOrigin - sphereCenter;
    float a = dot(rayDir, rayDir);
    float b = 2.0 * dot(oc, rayDir);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0 * a * c;
    
    if (discriminant < 0.0) {
        return false;
    }
    
    float sqrtd = sqrt(discriminant);
    t0 = (-b - sqrtd) / (2.0 * a);
    t1 = (-b + sqrtd) / (2.0 * a);
    
    return true;
}

// Matrix to correct light direction for spot lights
const mat4 LIGHT_DIRECTION_CORRECTION = mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0
);

/**
 * Samples shadow map for a specific light
 * Read the shadows from the shadow atlas
 */
float getShadowSample(int lightIndex, vec3 shadowCoord, float bias) {
    if (lightIndex >= MAX_LIGHTS) return 0.0;
    return texture(uShadowMap, vec3(shadowCoord.xy, shadowCoord.z - bias));
}

/**
 * Main function for calculating volumetric lighting for a single light
 * Performs ray marching through the light volume and accumulates light contribution
 */
vec3 calculateVolumetricLight(
    vec3 rayStart,          // Camera position
    vec3 rayDir,           // Ray direction
    float rayLength,       // Maximum ray distance (from scene depth)
    mat4 lightModelMatrix,  // Light's transform matrix
    mat4 lightViewProjMatrix, // Light's view-projection matrix for shadows
    int lightIndex,        // Index of current light
    vec3 lightColor,       // Light's color
    float lightIntensity,  // Light's intensity
    float innerConeAngle,  // Spotlight inner cone angle
    float outerConeAngle,  // Spotlight outer cone angle
    float lightRange       // Light's maximum range
) {
    // Transform light position and direction to world space
    vec3 lightPos = (lightModelMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec3 lightDir = normalize((lightModelMatrix * LIGHT_DIRECTION_CORRECTION * vec4(0.0, 0.0, -1.0, 0.0)).xyz);
    
    // Calculate spotlight cone angles
    float cosOuterCone = cos(outerConeAngle);
    float cosInnerCone = cos(innerConeAngle);
    
    // Check if ray intersects light volume
    float t0, t1;
    if (!rayIntersectsSphere(rayStart, rayDir, lightPos, lightRange * 1.1, t0, t1)) {
        return vec3(0.0); // Skip if ray doesn't intersect light volume
    }
    
    // Clamp intersection points to ray length
    t0 = max(0.0, t0);
    t1 = min(t1, rayLength);
    
    if (t1 <= t0) return vec3(0.0);
    
    // Calculate sampling parameters based on distance to light
    vec3 closestPoint = closestPointOnRay(rayStart, rayDir, lightPos);
    float minDistToLight = length(closestPoint - lightPos);
    float baseStepSize = lightRange * 0.02;
    
    // Adapt number of samples based on distance to light
    int numSteps = int(mix(32.0, 64.0, 1.0 - smoothstep(0.0, lightRange * 0.5, minDistToLight)));
    float stepSize = (t1 - t0) / float(numSteps);
    
    // Initialize ray marching with noise offset
    float initialOffset = getRayMarchingOffset(gl_FragCoord.xy, minDistToLight, 0.0);
    float t = t0 + stepSize * initialOffset;
    
    vec3 finalColor = vec3(0.0);
    float transmittance = 1.0;
    
    // Ray marching loop
    for (int i = 0; i < numSteps && transmittance > 0.01; i++) {
        if (t > t1) break;
        
        // Calculate sample position with noise offset
        float sampleOffset = getRayMarchingOffset(gl_FragCoord.xy, float(i), t);
        vec3 samplePos = rayStart + rayDir * (t + sampleOffset * stepSize * 0.5);
        
        vec3 lightToSample = samplePos - lightPos;
        float distToLight = length(lightToSample);
        
        // Check if sample is within light range
        if (distToLight < lightRange) {
            float cosAngle = dot(normalize(lightToSample), -lightDir);
            
            // Check if sample is within spotlight cone
            if (cosAngle > cosOuterCone) {
                // Transform sample position to light space for shadow mapping
                vec4 samplePosLightSpace = lightViewProjMatrix * vec4(samplePos, 1.0);
                vec3 shadowCoord = samplePosLightSpace.xyz / samplePosLightSpace.w;
                
                float bias = 0.01 * (1.0 - cosAngle);
                float shadow = getShadowSample(lightIndex, shadowCoord, bias);
                
                if (shadow > 0.0) {
                    // Calculate light contribution with distance attenuation
                    float distanceFactor = 1.0 - smoothstep(0.0, lightRange, distToLight);
                    float attenuation = 1.0 / (1.0 + distToLight * distToLight * 0.02);
                    
                    // Add noise to attenuation
                    float attenuationNoise = getRayMarchingOffset(samplePos.xy, distToLight, float(i));
                    attenuation *= distanceFactor * (0.95 + attenuationNoise * 0.1);
                    
                    // Calculate spotlight effect
                    float spotEffect = smoothstep(cosOuterCone, cosInnerCone, cosAngle);
                    
                    // Calculate density with noise variation
                    vec2 rotatedPos = getRotatedGrid(samplePos.xy);
                    float densityNoise = getBlueNoise(floor(rotatedPos * 10.0));
                    float density = 0.1 * (0.9 + densityNoise * 0.2);
                    
                    // Calculate extinction (light absorption)
                    float extinction = exp(-distToLight * density);
                    
                    // Accumulate light contribution
                    vec3 contribution = lightColor * spotEffect * attenuation * extinction * shadow;
                    finalColor += contribution * transmittance * stepSize;
                    
                    // Update transmittance (accumulated absorption)
                    transmittance *= exp(-density * stepSize * (0.97 + densityNoise * 0.06));
                }
            }
        }
        
        // Vary step size slightly for more natural results
        float stepNoise = getRayMarchingOffset(gl_FragCoord.xy, float(i), t) * 0.1 + 0.95;
        t += stepSize * stepNoise;
    }
    
    return finalColor * lightIntensity;
}

/**
 * Main shader entry point
 * Processes all active lights and combines their volumetric contributions
 */
void main() {
    // Convert screen space to world space
    float depth = texture(uSceneDepthMap, uv0).r;
    vec4 clipSpace = vec4(uv0 * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 viewSpace = matrix_inverseViewProjection * clipSpace;
    vec3 worldPos = viewSpace.xyz / viewSpace.w;
    
    // Calculate ray parameters
    vec3 rayStart = view_position;
    vec3 rayDir = normalize(worldPos - rayStart);
    float rayLength = length(worldPos - rayStart);
    
    // Initialize final color
    vec3 totalVolumetricLighting = vec3(0.0);
    
    // Process each active light
    for (int v = 0; v < MAX_LIGHTS; v++) {
        if (v >= uLightCount) break;
        
        // Get light properties
        float innerConeAngle = uLightProps[v].x;
        float outerConeAngle = uLightProps[v].y;
        float lightIntensity = uLightProps[v].z;
        float lightRange = uLightProps[v].w;
        
        // Calculate this light's contribution
        totalVolumetricLighting += calculateVolumetricLight(
            rayStart,
            rayDir,
            rayLength,
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
    
    // Apply final noise and tone mapping
    float finalNoise = getBlueNoise(floor(gl_FragCoord.xy));
    totalVolumetricLighting *= (1.0 + (finalNoise - 0.5) * 0.02);
    
    // Tone mapping (convert HDR to LDR)
    vec3 finalColor = totalVolumetricLighting / (vec3(1.0) + totalVolumetricLighting);
    
    gl_FragColor = vec4(finalColor, 1.0);
}
`;