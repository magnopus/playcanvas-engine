export default /* glsl */`
    #include "screenDepthPS"

    varying vec2 uv0;

    uniform sampler2D sourceTexture;
    uniform vec2 sourceInvResolution;
    uniform int filterSize;

    float random(const highp vec2 w) {
        const vec3 m = vec3(0.06711056, 0.00583715, 52.9829189);
        return fract(m.z * fract(dot(w, m.xy)));
    }

    // A sample blends when it lies on the centre pixel's surface: its depth is compared with the
    // depth extrapolated along the surface's slope, relative to the depth. Absolute units blended a
    // column with the wall a metre behind it at any distance (a halo around every silhouette), and
    // a flat test rejects a sloped or curved surface's own neighbours, leaving the sampling noise.
    // Relative to the plane, a sample about 5% of the distance off it no longer blends.
    float bilateralWeight(in float expected, in float sampleDepth) {
        float diff = (sampleDepth - expected) / max(expected, 1e-4) * 20.0;
        return max(0.0, 1.0 - diff * diff);
    }

    void tap(inout float sum, inout float totalWeight, float expected, vec2 position) {
        float color = texture2D(sourceTexture, position).r;
        float textureDepth = -getLinearScreenDepth(position);
        float bilateral = bilateralWeight(expected, textureDepth);
        sum += color * bilateral;
        totalWeight += bilateral;
    }

    void main() {

        #ifdef HORIZONTAL
            vec2 texelStep = vec2(sourceInvResolution.x, 0.0);
        #else
            vec2 texelStep = vec2(0.0, sourceInvResolution.y);
        #endif

        // handle the center pixel separately because it doesn't participate in bilateral filtering
        float depth = -getLinearScreenDepth(uv0);
        float totalWeight = 1.0;
        float color = texture2D(sourceTexture, uv0).r;
        float sum = color;

        // the surface's depth slope along the blur axis: the flatter of the two one-sided
        // differences, so a silhouette next to the pixel does not tilt the plane
        float before = depth - (-getLinearScreenDepth(uv0 - texelStep));
        float after = (-getLinearScreenDepth(uv0 + texelStep)) - depth;
        float slope = abs(before) < abs(after) ? before : after;

        for (int i = -filterSize; i <= filterSize; i++) {
            if (i == 0) continue;
            tap(sum, totalWeight, depth + float(i) * slope, uv0 + float(i) * texelStep);
        }

        float ao = sum / totalWeight;

        // simple dithering helps a lot (assumes 8 bits target)
        // this is most useful with high quality/large blurs
        // ao += ((random(gl_FragCoord.xy) - 0.5) / 255.0);

        gl_FragColor.r = ao;
    }
`;
