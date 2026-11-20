import { meshletCutEmitWGSL, meshletCutQueueArgsWGSL, meshletCutQueueWGSL } from './shaders/meshlet-cut-wgsl.js';
import {
    meshletRootAdmitWGSL, meshletRootBandArgsWGSL, meshletRootClassifyWGSL, meshletRootScatterWGSL
} from './shaders/meshlet-root-admit-wgsl.js';
import { PIXELFORMAT_R32F, SHADERLANGUAGE_WGSL } from '../../platform/graphics/constants.js';
import { Shader } from '../../platform/graphics/shader.js';
import { Texture } from '../../platform/graphics/texture.js';
import {
    cellCullWGSL, dispatchArgsWGSL, meshletCullWGSL, finalizeArgsWGSL, indexWriteWGSL, resetPhase2WGSL
} from './shaders/meshlet-cull-wgsl.js';

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 */

// the compiled shaders by property name; MeshletCuller creates its computes from these names
const SHADERS = {
    cellCull: cellCullWGSL,
    cutQueue: meshletCutQueueWGSL,
    cutQueueArgs: meshletCutQueueArgsWGSL,
    cutEmit: meshletCutEmitWGSL,
    rootClassify: meshletRootClassifyWGSL,
    rootBandArgs: meshletRootBandArgsWGSL,
    rootScatter: meshletRootScatterWGSL,
    rootAdmit: meshletRootAdmitWGSL,
    dispatchArgs: dispatchArgsWGSL,
    meshletCull: meshletCullWGSL,
    finalizeArgs: finalizeArgsWGSL,
    indexWrite: indexWriteWGSL,
    resetPhase2: resetPhase2WGSL
};

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
        for (const [name, code] of Object.entries(SHADERS)) {
            this[name] = new Shader(device, {
                name: `Meshlet${name[0].toUpperCase()}${name.slice(1)}`,
                shaderLanguage: SHADERLANGUAGE_WGSL,
                cshader: code
            });
        }

        // bound when no HZB is active (the shader always declares the binding)
        this.dummyHzb = new Texture(device, {
            name: 'MeshletHzbDummy', width: 1, height: 1, format: PIXELFORMAT_R32F, mipmaps: false
        });
    }

    destroy() {
        for (const name of Object.keys(SHADERS)) {
            this[name]?.destroy();
            this[name] = null;
        }
        this.dummyHzb?.destroy();
        this.dummyHzb = null;
    }
}

export { MeshletCullShaders };
