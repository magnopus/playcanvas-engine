export const MAX_LIGHTS = 8;

export const voluemtricLightShader = /* glsl */ `
/**
 * Advanced Volumetric Light Shader
 * 
 * This shader creates volumetric lighting effects (god rays) by ray marching through
 * a volume and accumulating light contributions. It supports multiple lights and 
 * includes features like:
 * - Physical light attenuation
 * - Shadow mapping
 * - Blue noise dithering for smooth results
 * - Spotlights with cone angles
 * - Distance-based optimization
 * 
 * The basic principle is:
 * 1. For each pixel, cast a ray from the camera to the scene
 * 2. March along this ray, sampling points within light volumes
 * 3. At each sample point, calculate light contribution considering:
 *    - Distance from light
 *    - Shadows
 *    - Light cone angles (for spotlights)
 *    - Scattering and extinction
 * 4. Accumulate these samples with proper depth and transparency
 */

#define MAX_LIGHTS 8
precision highp float;

// Matrix uniforms for coordinate transformations
uniform mat4 matrix_projection;
uniform mat4 matrix_viewProjection;
uniform mat4 matrix_inverseViewProjection;
uniform vec3 view_position;
uniform vec2 scatteringCoeff[MAX_LIGHTS];

// Light properties
uniform vec4 uLightProps[MAX_LIGHTS];        // x: innerConeAngle, y: outerConeAngle, z: intensity, w: range
uniform mat4 matrix_lightmodel[MAX_LIGHTS];   // World space transformation for each light
uniform vec3 uLightColor[MAX_LIGHTS];         // RGB color of each light
uniform int uLightCount;                      // Number of active lights

// Shadow mapping
uniform sampler2DShadow uShadowMap;
uniform mat4 uLightViewProjMatrix[MAX_LIGHTS];

// Scene information
uniform highp sampler2D uSceneDepthMap;            // Scene depth buffer
uniform sampler2D uBlueNoiseTexture;         // Blue noise for dithering
uniform float uTime;                         // Current time for temporal variation

// Input from vertex shader
in vec2 uv0;                                // Screen-space UV coordinates

/**
 * Converts view space depth to linear depth
 * Required for proper depth comparisons during ray marching
 */

float linearizeDepth(float z, vec4 cameraParams) {
    if (cameraParams.w == 0.0)
        return (cameraParams.z * cameraParams.y) / (cameraParams.y + z * (cameraParams.z - cameraParams.y));
    else
        return cameraParams.z + z * (cameraParams.y - cameraParams.z);
}



/**
 * Converts view space depth to linear depth
 * Required for proper depth comparisons during ray marching
 */
float linearizeDepth2(float depth, mat4 projMatrix) {
    float near = projMatrix[3][2] / (projMatrix[2][2] - 1.0);
    float far = projMatrix[3][2] / (projMatrix[2][2] + 1.0);
    return (2.0 * near * far) / (far + near - depth * (far - near));
}
/**
 * Blue noise sampling function
 * Uses temporal offset to prevent static noise patterns
 * Returns: value between 0 and 1
 */
float getBlueNoise(vec2 screenPos) {
    vec2 pixelPos = floor(screenPos);
    float frameOffset = floor(uTime * 60.0);
    // Create temporally-varying UV coordinates for noise sampling
    vec2 noiseUV = mod((pixelPos + vec2(frameOffset * 13.0, frameOffset * 7.0)), 64.0) / 64.0;
    return texture(uBlueNoiseTexture, noiseUV).r;
}

/**
 * Creates a complex noise pattern for ray marching
 * Combines multiple noise frequencies for natural-looking variation
 * Parameters:
 * screenPos: Screen position for noise sampling
 * depth: Current sample depth for variation
 * index: Current sample index for temporal variation
 */
float getRayMarchingOffset(vec2 screenPos, float depth, float index) {
    float noise1 = getBlueNoise(screenPos);
    float noise2 = getBlueNoise(screenPos * 1.7 + vec2(index * 0.13));
    float noise3 = getBlueNoise((screenPos + depth * 0.1) * 0.77);
    
    return (noise1 * 0.5 + noise2 * 0.3 + noise3 * 0.2);
}

/**
 * Finds closest point on a ray to a given position
 * Used for determining optimal sampling density near lights
 */
vec3 closestPointOnRay(vec3 rayOrigin, vec3 rayDir, vec3 point) {
    vec3 v = point - rayOrigin;
    float t = dot(v, rayDir);
    return rayOrigin + rayDir * max(t, 0.0);
}

/**
 * Ray-sphere intersection test
 * Used to determine where rays enter and exit light volumes
 * Returns intersection points t0 (entry) and t1 (exit)
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

// Matrix to correct light direction for spotlights
const mat4 LIGHT_DIRECTION_CORRECTION = mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0
);

/**
 * Shadow map sampling
 * Returns: Shadow factor (0 = fully shadowed, 1 = fully lit)
 */
float getShadowSample(vec3 shadowCoord, float bias) {
    if (any(lessThan(shadowCoord, vec3(0.0))) || any(greaterThan(shadowCoord, vec3(1.0)))) return 0.0;
    return texture(uShadowMap, vec3(shadowCoord.xy, shadowCoord.z - bias));
}

/**
 * Main volumetric lighting calculation for a single light
 * Parameters:
 * - rayStart: Camera position
 * - rayDir: Ray direction
 * - rayLength: Maximum ray distance (from scene depth)
 * - lightModelMatrix: Light's transform matrix
 * - lightViewProjMatrix: Light's view-projection matrix for shadows
 * - lightIndex: Current light index
 * - lightColor: Light's color
 * - lightIntensity: Light's intensity
 * - innerConeAngle: Spotlight inner cone angle
 * - outerConeAngle: Spotlight outer cone angle
 * - lightRange: Light's maximum range
 */
vec3 calculateVolumetricLight(
    vec3 rayStart,
    vec3 rayDir,
    float rayLength,
    mat4 lightModelMatrix,
    mat4 lightViewProjMatrix,
    int lightIndex,
    vec3 lightColor,
    float lightIntensity,
    float innerConeAngle,
    float outerConeAngle,
    float lightRange,
    vec2 scatteringProps
) {
    // Transform light position and direction to world space
    vec3 lightPos = (lightModelMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec3 lightDir = normalize((lightModelMatrix * LIGHT_DIRECTION_CORRECTION * vec4(0.0, 0.0, -1.0, 0.0)).xyz);
    float cosOuterCone = cos(outerConeAngle);
    float cosInnerCone = cos(innerConeAngle);
    
    // Check if ray intersects light volume
    float t0, t1;
    if (!rayIntersectsSphere(rayStart, rayDir, lightPos, lightRange * 1.1, t0, t1)) {
        return vec3(0.0);
    }
    
    // Clamp intersection points to ray length
    t0 = max(0.0, t0);
    t1 = min(t1, rayLength);
    
    if (t1 <= t0) return vec3(0.0);
    
    // Calculate sampling parameters based on distance to light
    vec3 closestPoint = closestPointOnRay(rayStart, rayDir, lightPos);
    float minDistToLight = length(closestPoint - lightPos);
    float baseStepSize = lightRange * 0.02;
    
    // Adapt number of samples based on distance
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
        
        // Calculate current sample position
        vec3 samplePos = rayStart + rayDir * t;
        
        // Project sample position to screen space for depth testing
        vec4 sampleProjected = matrix_viewProjection * vec4(samplePos, 1.0);
        vec3 screenPos = sampleProjected.xyz / sampleProjected.w;
        
        // Convert to UV coordinates
        vec2 sampleUV = screenPos.xy * 0.5 + 0.5;
        
        // Sample scene depth and compare with current sample depth
        float sceneDepth = texture(uSceneDepthMap, uv0).r;
        float sampleLinearDepth = linearizeDepth2(screenPos.z, matrix_projection);
        float sceneLinearDepth = linearizeDepth2(sceneDepth, matrix_projection);
        
        // Skip this sample if it's behind scene geometry
        if (sampleLinearDepth > sceneLinearDepth + 0.001) {
            t += stepSize;
            continue;
        }
        
        float sampleOffset = getRayMarchingOffset(gl_FragCoord.xy, float(i), t);
        
        // Calculate sample position with distance-based noise reduction
        float distToLight = length(samplePos - lightPos);
        float normalizedDist = smoothstep(0.0, lightRange * 0.3, distToLight);
        
        // Apply less position noise near the light
        samplePos += rayDir * (sampleOffset * stepSize * 0.5 * normalizedDist);
        
        vec3 lightToSample = samplePos - lightPos;
        distToLight = length(lightToSample);
        
        if (distToLight < lightRange) {
            float cosAngle = dot(normalize(lightToSample), -lightDir);
            
            // Check if sample is within spotlight cone
            if (cosAngle > cosOuterCone) {
                // Transform to light space for shadow mapping
                vec4 samplePosLightSpace = lightViewProjMatrix * vec4(samplePos, 1.0);
                vec3 shadowCoord = samplePosLightSpace.xyz / samplePosLightSpace.w;
                
                float bias = 0.01 * (1.0 - cosAngle);
                float shadow = getShadowSample(shadowCoord, bias);
                
                if (shadow > 0.0) {
                    // Physical light calculations
                    float falloff = 1.0 / (distToLight * distToLight); // Inverse square law
                    float spotEffect = smoothstep(cosOuterCone, cosInnerCone, cosAngle);
                    
                    // Distance-based noise for density
                    float densityNoise = getBlueNoise(floor(samplePos.xy * 10.0));
                    float noiseInfluence = mix(0.02, 0.1, normalizedDist);
                    float density = scatteringProps.x * (0.98 + densityNoise * noiseInfluence);
                    
                    // Beer-Lambert law for extinction
                    float extinction = exp(-scatteringProps.y * distToLight);
                    
                    // Accumulate light contribution
                    vec3 contribution = lightColor * lightIntensity * falloff * spotEffect * shadow * extinction;
                    finalColor += contribution * transmittance * density * stepSize;
                    
                    // Update transmittance with distance-based noise
                    float transNoiseInfluence = mix(0.02, 0.06, normalizedDist);
                    transmittance *= exp(-scatteringProps.y * stepSize * (0.98 + densityNoise * transNoiseInfluence));
                }
            }
        }
        
        // Variable step size with distance-based noise
        float stepNoiseInfluence = mix(0.05, 0.1, normalizedDist);
        float stepNoise = getRayMarchingOffset(gl_FragCoord.xy, float(i), t) * stepNoiseInfluence + 0.95;
        t += stepSize * stepNoise;
    }
    
    return finalColor;
}

/**
 * Main shader entry point
 * Processes all active lights and combines their volumetric contributions
 */
void main() {
    // Convert screen coordinates to world space
    float depth = texture(uSceneDepthMap, uv0).r;
    vec4 clipSpace = vec4(uv0 * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 viewSpace = matrix_inverseViewProjection * clipSpace;
    vec3 worldPos = viewSpace.xyz / viewSpace.w;
    
    // Calculate ray parameters
    vec3 rayStart = view_position;
    vec3 rayDir = normalize(worldPos - rayStart);
    float rayLength = length(worldPos - rayStart);
    
    // Process each active light
    vec3 totalVolumetricLighting = vec3(0.0);
    
    for (int v = 0; v < MAX_LIGHTS; v++) {
        if (v >= uLightCount) break;
        
        float innerConeAngle = uLightProps[v].x;
        float outerConeAngle = uLightProps[v].y;
        float lightIntensity = uLightProps[v].z;
        float lightRange = uLightProps[v].w;
        
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
            lightRange,
            scatteringCoeff[v]
        );
    }
    
    // Tone mapping (HDR -> LDR conversion)
    vec3 finalColor = totalVolumetricLighting / (vec3(1.0) + totalVolumetricLighting);
            // Convert to UV coordinates
       // vec2 sampleUV = screenPos.xy * 0.5 + 0.5;
                float sceneDepth = texture(uSceneDepthMap, uv0).r;
        //float sampleLinearDepth = linearizeDepth(screenPos.z, matrix_projection);
        float sceneLinearDepth = linearizeDepth(sceneDepth, vec4(1.0 / 10.0,  10.0,  0.1, 0.0));
    gl_FragColor = vec4(finalColor,1.0);
}
`;
