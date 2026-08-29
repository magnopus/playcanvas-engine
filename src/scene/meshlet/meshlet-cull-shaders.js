import { PIXELFORMAT_R32F, SHADERLANGUAGE_WGSL } from '../../platform/graphics/constants.js';
import { Shader } from '../../platform/graphics/shader.js';
import { Texture } from '../../platform/graphics/texture.js';
import {
    instanceCullWGSL, dispatchArgsWGSL, meshletCullWGSL, finalizeArgsWGSL, indexWriteWGSL, resetPhase2WGSL
} from './shaders/meshlet-cull-wgsl.js';

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 */

/**
 * The compiled cull compute shaders, plus the placeholder HZB texture bound when no HZB is
 * active. These are stateless - all per-view state rides on the {@link Compute} instances - so
 * one set is shared by every {@link MeshletCuller}. That matters once shadows land: a four
 * cascade CSM adds four more cullers per light, and compiling six compute shaders each would be
 * twenty-four redundant pipeline compiles.
 *
 * @ignore
 */
class MeshletCullShaders {
    /**
     * @param {GraphicsDevice} device - The graphics device.
     */
    constructor(device) {
        const make = (/** @type {string} */ name, /** @type {string} */ code) => new Shader(device, {
            name: `Meshlet${name}`,
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: code
        });

        this.instanceCull = make('InstanceCull', instanceCullWGSL);
        this.dispatchArgs = make('DispatchArgs', dispatchArgsWGSL);
        this.meshletCull = make('MeshletCull', meshletCullWGSL);
        this.finalizeArgs = make('FinalizeArgs', finalizeArgsWGSL);
        this.indexWrite = make('IndexWrite', indexWriteWGSL);
        this.resetPhase2 = make('ResetPhase2', resetPhase2WGSL);

        // bound when no HZB is active (the shader always declares the binding)
        this.dummyHzb = new Texture(device, {
            name: 'MeshletHzbDummy', width: 1, height: 1, format: PIXELFORMAT_R32F, mipmaps: false
        });
    }

    destroy() {
        this.instanceCull?.destroy();
        this.dispatchArgs?.destroy();
        this.meshletCull?.destroy();
        this.finalizeArgs?.destroy();
        this.indexWrite?.destroy();
        this.resetPhase2?.destroy();
        this.dummyHzb?.destroy();
    }
}

export { MeshletCullShaders };
