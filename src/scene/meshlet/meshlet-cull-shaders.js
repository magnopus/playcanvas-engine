import { meshletCutWGSL } from './shaders/meshlet-cut-wgsl.js';
import { meshletRootAdmitWGSL, meshletRootClassifyWGSL } from './shaders/meshlet-root-admit-wgsl.js';
import { PIXELFORMAT_R32F, SHADERLANGUAGE_WGSL } from '../../platform/graphics/constants.js';
import { Shader } from '../../platform/graphics/shader.js';
import { Texture } from '../../platform/graphics/texture.js';
import {
    instanceCullWGSL, compactMeshletsWGSL, dispatchArgsWGSL, meshletCullWGSL, finalizeArgsWGSL, indexWriteWGSL, resetPhase2WGSL
} from './shaders/meshlet-cull-wgsl.js';

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 */

/**
 * The compiled cull compute shaders, plus the placeholder HZB texture bound when no HZB is
 * active. These are stateless - all per-view state rides on the {@link Compute} instances - so
 * one set is shared by every {@link MeshletCuller}, including the cullers for shadow cascades.
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

        this.cut = make('Cut', meshletCutWGSL);
        this.rootClassify = make('RootClassify', meshletRootClassifyWGSL);
        this.rootAdmit = make('RootAdmit', meshletRootAdmitWGSL);
        this.instanceCull = make('InstanceCull', instanceCullWGSL);
        this.compactMeshlets = make('CompactMeshlets', compactMeshletsWGSL);
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
        this.cut?.destroy();
        this.rootClassify?.destroy();
        this.rootAdmit?.destroy();
        this.instanceCull?.destroy();
        this.compactMeshlets?.destroy();
        this.dispatchArgs?.destroy();
        this.meshletCull?.destroy();
        this.finalizeArgs?.destroy();
        this.indexWrite?.destroy();
        this.resetPhase2?.destroy();
        this.dummyHzb?.destroy();
    }
}

export { MeshletCullShaders };
