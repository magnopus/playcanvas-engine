import {
    BLENDEQUATION_ADD, BLENDMODE_DST_COLOR, BLENDMODE_ONE, BLENDMODE_ZERO, SEMANTIC_POSITION
} from '../../../../platform/graphics/constants.js';
import { BlendState } from '../../../../platform/graphics/blend-state.js';
import { RenderPassShaderQuad } from '../../../../scene/graphics/render-pass-shader-quad.js';
import { ShaderUtils } from '../../../../scene/shader-lib/shader-utils.js';

/**
 * @import { GraphicsDevice } from '../../../../platform/graphics/graphics-device.js'
 * @import { RenderPassSsao } from './render-pass-ssao.js'
 */

const glslFS = /* glsl */`
    varying vec2 uv0;
    uniform sampler2D uSsaoCombineTexture;

    void main() {
        float ao = texture2D(uSsaoCombineTexture, uv0).r;
        gl_FragColor = vec4(ao, ao, ao, 1.0);
    }
`;

const wgslFS = /* wgsl */`
    varying uv0: vec2f;
    var uSsaoCombineTexture: texture_2d<f32>;
    var uSsaoCombineTextureSampler: sampler;

    @fragment
    fn fragmentMain(input: FragmentInput) -> FragmentOutput {
        var output: FragmentOutput;
        let ao: f32 = textureSampleLevel(uSsaoCombineTexture, uSsaoCombineTextureSampler, input.uv0, 0.0).r;
        output.color = vec4f(ao, ao, ao, 1.0);
        return output;
    }
`;

/**
 * Multiplies the scene render target by the ambient occlusion, for SSAO in the combine mode. It
 * runs after the scene and before it is resolved, so the temporal anti-aliasing accumulates the
 * occlusion with the rest of the image: the occlusion of a jittered frame lands on that frame, and
 * a sampling pattern randomized per frame averages out. Applied in the composition instead, after
 * the resolve, the occlusion of a jittered frame is drawn on the unjittered image, and nothing
 * averages its noise.
 *
 * @ignore
 */
class RenderPassSsaoCombine extends RenderPassShaderQuad {
    /**
     * @param {GraphicsDevice} device - The graphics device.
     * @param {RenderPassSsao} ssaoPass - The pass generating the occlusion.
     */
    constructor(device, ssaoPass) {
        super(device);
        this.ssaoPass = ssaoPass;

        this.shader = ShaderUtils.createShader(device, {
            uniqueName: 'SsaoCombineShader',
            attributes: { aPosition: SEMANTIC_POSITION },
            vertexChunk: 'quadVS',
            fragmentGLSL: glslFS,
            fragmentWGSL: wgslFS
        });

        // scene.rgb * ao, scene alpha is preserved
        this.blendState = new BlendState(true, BLENDEQUATION_ADD, BLENDMODE_DST_COLOR, BLENDMODE_ZERO,
            BLENDEQUATION_ADD, BLENDMODE_ZERO, BLENDMODE_ONE);

        this.ssaoTextureId = device.scope.resolve('uSsaoCombineTexture');
    }

    execute() {
        this.ssaoTextureId.setValue(this.ssaoPass.ssaoTexture);
        super.execute();
    }
}

export { RenderPassSsaoCombine };
