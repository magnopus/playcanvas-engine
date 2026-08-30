import {
    ADDRESS_CLAMP_TO_EDGE, FILTER_NEAREST, PIXELFORMAT_R32F, SEMANTIC_POSITION, SHADERLANGUAGE_WGSL
} from '../../platform/graphics/constants.js';
import { Compute } from '../../platform/graphics/compute.js';
import { RenderTarget } from '../../platform/graphics/render-target.js';
import { Shader } from '../../platform/graphics/shader.js';
import { Texture } from '../../platform/graphics/texture.js';
import { RenderPassShaderQuad } from '../graphics/render-pass-shader-quad.js';
import { ShaderUtils } from '../shader-lib/shader-utils.js';

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 */

// mip 0: max of the 2x2 full-res depth texels under each half-res HZB texel
const mip0FragmentWGSL = /* wgsl */ `
    var uDepthMap : texture_depth_2d;

    @fragment
    fn fragmentMain(input : FragmentInput) -> FragmentOutput {
        var output : FragmentOutput;
        let dst = vec2i(input.position.xy);
        let size = vec2i(textureDimensions(uDepthMap));
        let src = dst * 2;
        let d00 = textureLoad(uDepthMap, min(src, size - 1), 0);
        let d10 = textureLoad(uDepthMap, min(src + vec2i(1, 0), size - 1), 0);
        let d01 = textureLoad(uDepthMap, min(src + vec2i(0, 1), size - 1), 0);
        let d11 = textureLoad(uDepthMap, min(src + vec2i(1, 1), size - 1), 0);
        output.color = vec4f(max(max(d00, d10), max(d01, d11)), 0.0, 0.0, 1.0);
        return output;
    }
`;

// threads per side of a mip-reduce workgroup: one 8x8 tile of destination texels per group
const REDUCE_WORKGROUP = 8;

// mip i: max-reduce of the 2x2 texels of mip i-1
const mipReduceWGSL = /* wgsl */ `
    var srcMip : texture_2d<f32>;
    var dstMip : texture_storage_2d<r32float, write>;

    @compute @workgroup_size(${REDUCE_WORKGROUP}, ${REDUCE_WORKGROUP})
    fn main(@builtin(global_invocation_id) gid : vec3u) {
        let dstSize = vec2i(textureDimensions(dstMip));
        let dst = vec2i(gid.xy);
        if (dst.x >= dstSize.x || dst.y >= dstSize.y) {
            return;
        }
        let srcSize = vec2i(textureDimensions(srcMip));
        let src = dst * 2;
        let d00 = textureLoad(srcMip, min(src, srcSize - 1), 0).x;
        let d10 = textureLoad(srcMip, min(src + vec2i(1, 0), srcSize - 1), 0).x;
        let d01 = textureLoad(srcMip, min(src + vec2i(0, 1), srcSize - 1), 0).x;
        let d11 = textureLoad(srcMip, min(src + vec2i(1, 1), srcSize - 1), 0).x;
        textureStore(dstMip, dst, vec4f(max(max(d00, d10), max(d01, d11)), 0.0, 0.0, 1.0));
    }
`;

/**
 * A max-depth hierarchical Z pyramid built from a depth texture: a half-resolution R32F mip
 * chain where each texel holds the maximum (farthest) depth of the screen region it covers. The
 * phase-2 meshlet cull tests cluster bounding spheres against it. Mip 0 is a fragment quad pass
 * (depth textures cannot be storage-written); the remaining mips reduce in compute through
 * per-mip texture views.
 *
 * @ignore
 */
class MeshletHzb {
    /** @type {GraphicsDevice} */
    device;

    /** @type {Texture|null} */
    texture = null;

    width = 0;

    height = 0;

    mipCount = 0;

    /** @type {RenderPassShaderQuad|null} - mip 0 build (added to the frame graph). */
    mip0Pass = null;

    /** @type {Compute[]} - mip reduce chain, dispatched by the owner between passes. */
    _reduces = [];

    /** @type {RenderTarget|null} */
    _mip0Target = null;

    /** @type {Shader|null} */
    _reduceShader = null;

    /**
     * @param {GraphicsDevice} device - The graphics device.
     */
    constructor(device) {
        this.device = device;

        this._reduceShader = new Shader(device, {
            name: 'MeshletHzbReduce',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: mipReduceWGSL
        });
    }

    destroy() {
        this._mip0Target?.destroy();
        this.texture?.destroy();
        this.mip0Pass = null;
        this._reduces = [];
    }

    /**
     * (Re)creates the pyramid for a depth source of the given size.
     *
     * @param {Texture} depthTexture - The scene depth texture to reduce from.
     * @param {number} width - Depth texture width.
     * @param {number} height - Depth texture height.
     */
    resize(depthTexture, width, height) {
        const w = Math.max(width >> 1, 1);
        const h = Math.max(height >> 1, 1);
        if (this.texture && this.width === w && this.height === h) {
            this._depthTexture = depthTexture;
            return;
        }
        const device = this.device;

        this._mip0Target?.destroy();
        this.texture?.destroy();

        this.width = w;
        this.height = h;
        this.mipCount = Math.floor(Math.log2(Math.max(w, h))) + 1;

        this.texture = new Texture(device, {
            name: 'MeshletHzb',
            width: w,
            height: h,
            format: PIXELFORMAT_R32F,
            mipmaps: true,
            storage: true,
            minFilter: FILTER_NEAREST,
            magFilter: FILTER_NEAREST,
            addressU: ADDRESS_CLAMP_TO_EDGE,
            addressV: ADDRESS_CLAMP_TO_EDGE
        });

        this._mip0Target = new RenderTarget({
            name: 'MeshletHzbMip0',
            colorBuffer: this.texture,
            depth: false,
            mipLevel: 0
        });

        // mip 0 quad pass
        this._depthTexture = depthTexture;
        const pass = new RenderPassShaderQuad(device);
        pass.name = 'MeshletHzbMip0';
        pass.shader = ShaderUtils.createShader(device, {
            uniqueName: 'MeshletHzbMip0',
            attributes: { aPosition: SEMANTIC_POSITION },
            vertexChunk: 'quadVS',
            fragmentWGSL: mip0FragmentWGSL
        });
        const self = this;
        const superExecute = pass.execute.bind(pass);
        pass.execute = function () {
            device.scope.resolve('uDepthMap').setValue(self._depthTexture);
            superExecute();
        };
        pass.init(this._mip0Target, {});
        // the quad writes every texel, so never load the previous contents
        pass.colorOps.clear = true;
        pass.colorOps.store = true;
        this.mip0Pass = pass;

        // mip reduce chain through per-mip views
        this._reduces = [];
        for (let mip = 1; mip < this.mipCount; mip++) {
            const compute = new Compute(device, this._reduceShader, `MeshletHzbReduce${mip}`);
            compute.setParameter('srcMip', this.texture.getView(mip - 1));
            compute.setParameter('dstMip', this.texture.getView(mip));
            const mw = Math.max(w >> mip, 1);
            const mh = Math.max(h >> mip, 1);
            compute.setupDispatch(Math.ceil(mw / REDUCE_WORKGROUP), Math.ceil(mh / REDUCE_WORKGROUP));
            this._reduces.push(compute);
        }
    }

    /**
     * Dispatches the mip reduce chain (mip 0 must have rendered already).
     */
    buildMips() {
        this.device.computeDispatch(this._reduces, 'MeshletHzbMips');
    }
}

export { MeshletHzb };
