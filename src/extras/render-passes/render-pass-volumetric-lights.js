// @ts-nocheck
import { Color } from '../../core/math/color.js';
import { Texture } from '../../platform/graphics/texture.js';
import { RenderTarget } from '../../platform/graphics/render-target.js';
import { ADDRESS_CLAMP_TO_EDGE, FILTER_LINEAR, FILTER_NEAREST } from '../../platform/graphics/constants.js';
import { RenderPassShaderQuad } from '../../scene/graphics/render-pass-shader-quad.js';
import { getBlueNoiseTexture } from '../../scene/graphics/noise-textures.js';
import { voluemtricLightShader, MAX_LIGHTS } from './volumetric-shader.js';
import { RenderPassUpsample } from './render-pass-upsample.js';
import { RenderPassDepthAwareBlur } from './render-pass-depth-aware-blur.js';


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

    constructor(app, device, sourceTexture, textureFormat, sceneDepth) {
        super(device);
        this.app = app;
        this.textureFormat = textureFormat;
        this.sourceTexture = sourceTexture;
        this.sceneDepth = sceneDepth;

        // main Volumetric render pass
        this.shader = this.createQuadShader('VolumeShader', voluemtricLightShader);

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

        const blurRT = this.createRenderTarget('SsaoTempTexture');
        const blurRT2 = this.createRenderTarget('SsaoTempTexture2');
        const blurRT3 = this.createRenderTarget('SsaoTempTexture3');

        const upscalePass = new RenderPassUpsample(device, rt.colorBuffer);
        upscalePass.init(blurRT, {
            resizeSource: this.sourceTexture
        });

        const blurPassHorizontal = new RenderPassDepthAwareBlur(device, blurRT.colorBuffer, true);
        blurPassHorizontal.init(blurRT2, {
            resizeSource: this.sourceTexture
        });

        const blurPassVertical = new RenderPassDepthAwareBlur(device, blurRT2.colorBuffer, false);
        blurPassVertical.init(blurRT3, {
            resizeSource: this.sourceTexture
        });
        
        this.afterPasses.push(upscalePass);
        this.afterPasses.push(blurPassHorizontal);
        this.afterPasses.push(blurPassVertical);
        this.volumetricsTexture = blurRT3.colorBuffer;
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
                minFilter: FILTER_LINEAR,
                magFilter: FILTER_LINEAR,
                addressU: ADDRESS_CLAMP_TO_EDGE,
                addressV: ADDRESS_CLAMP_TO_EDGE
            })
        });
    }

    execute() {
        if (!this.renderTarget?.colorBuffer) return;
        const { device } = this;
        const scope = device.scope;


        const lights = this.app.root.findComponents('light');
        let volumetricLights = lights.filter(
            light => light.enabled && light.volumetric && light.light.visibleThisFrame && light.type === 'spot'
        );
        volumetricLights = volumetricLights.slice(0, MAX_LIGHTS);
        // nothing to do
        if (volumetricLights.length === 0) return;
        scope.resolve('uLightCount').setValue(volumetricLights.length);
        const values = {
            uLightProps: [],
            scatteringCoeff: [],
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
            values.scatteringCoeff = [
                ...values.scatteringCoeff,
                light.scattering,
                light.extinction
            ];
            values.uLightColour = [...values.uLightColour, r, g, b];
            values.uShadowMap = light.light._shadowMap?.texture;
            const shadowMat = light.light._renderData[0]?.shadowMatrix?.data;
            values.uLightViewProjMatrix.push(...[...shadowMat]);
            values.matrix_lightmodel = [...values.matrix_lightmodel, ...Array.from(light.entity.getWorldTransform().data)];
        }
        if (volumetricLights.length) {
            scope.resolve('uLightProps[0]').setValue(values.uLightProps);
            scope.resolve('uLightColor[0]').setValue(values.uLightColour);
            scope.resolve('matrix_lightmodel[0]').setValue(values.matrix_lightmodel);
            scope.resolve('uShadowMap').setValue(values.uShadowMap);
        //    scope.resolve('uHasShadowMap[0]').setValue(values.uHasShadowMap);
            scope.resolve('uLightViewProjMatrix[0]').setValue(values.uLightViewProjMatrix);
            scope.resolve('uTime').setValue(1);
            scope.resolve('uSceneDepthMap').setValue(this.sceneDepth);
            scope.resolve('scatteringCoeff[0]').setValue(values.scatteringCoeff);
            scope.resolve('uBlueNoiseTexture').setValue(getBlueNoiseTexture(device));
            super.execute();

            
        }
    }

    after() {
        this.volumetricsTextureId.setValue(this.volumetricsTexture);
    }
}

export { RenderPassVolumetricLight };
