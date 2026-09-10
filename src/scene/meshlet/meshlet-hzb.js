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
const mip0DepthWGSL = /* wgsl */ `
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

// mip 0 from the scene depth a CameraFrame renders as a colour attachment: a coverage weighted
// average of reciprocal view depths, cleared to the reciprocal of the far clip. Decoded back to
// linear depth here, so the pyramid and its max-reduce work in view units; a texel nothing wrote
// (zero) reads as infinitely far.
const mip0LinearWGSL = /* wgsl */ `
    var uDepthMap : texture_2d<f32>;

    fn linearDepth(reciprocal : f32) -> f32 {
        return select(1e30, 1.0 / reciprocal, reciprocal > 0.0);
    }

    @fragment
    fn fragmentMain(input : FragmentInput) -> FragmentOutput {
        var output : FragmentOutput;
        let dst = vec2i(input.position.xy);
        let size = vec2i(textureDimensions(uDepthMap));
        let src = dst * 2;
        let d00 = linearDepth(textureLoad(uDepthMap, min(src, size - 1), 0).x);
        let d10 = linearDepth(textureLoad(uDepthMap, min(src + vec2i(1, 0), size - 1), 0).x);
        let d01 = linearDepth(textureLoad(uDepthMap, min(src + vec2i(0, 1), size - 1), 0).x);
        let d11 = linearDepth(textureLoad(uDepthMap, min(src + vec2i(1, 1), size - 1), 0).x);
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
 * A max-depth hierarchical Z pyramid built from a scene depth source: a half-resolution R32F mip
 * chain where each texel holds the maximum (farthest) depth of the screen region it covers. The
 * phase-2 meshlet cull tests cluster bounding spheres against it. Mip 0 is a fragment quad pass
 * (depth textures cannot be storage-written); the remaining mips reduce in compute through
 * per-mip texture views.
 *
 * The source is either a depth texture, giving a pyramid of NDC depth, or the linear scene depth
 * a CameraFrame renders as a colour attachment, giving a pyramid in view units - see
 * {@link linear}. The cull compares in whichever units the pyramid holds.
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

    /**
     * True when the pyramid holds linear view depth decoded from a scene depth attachment, false
     * when it holds NDC depth from a depth texture.
     *
     * @type {boolean}
     */
    linear = false;

    /**
     * Mip 0 build (added to the frame graph). One pass for the lifetime of the pyramid: a
     * CameraFrame caches the pass objects it splices in, so a resize re-targets this pass rather
     * than replacing it.
     *
     * @type {RenderPassShaderQuad|null}
     */
    mip0Pass = null;

    /** @type {Compute[]} - mip reduce chain, dispatched by the owner between passes. */
    _reduces = [];

    /** @type {RenderTarget|null} */
    _mip0Target = null;

    /** @type {Shader|null} */
    _reduceShader = null;

    /** @type {Shader|null} */
    _mip0DepthShader = null;

    /** @type {Shader|null} */
    _mip0LinearShader = null;

    /** @type {Texture|null} - the source the mip 0 pass reads, bound at execute. */
    _depthTexture = null;

    /**
     * Resolves the source at execute time, when set. A CameraFrame replaces its scene textures
     * when it rebuilds, which happens while the frame graph is being built - after the director
     * pointed this pyramid at the frame's texture and before the pass runs on it - so the pass
     * asks again as it executes rather than trusting the reference it was given.
     *
     * @type {(() => Texture|null)|null}
     */
    _source = null;

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
        this._mip0Target = null;
        this.texture?.destroy();
        this.texture = null;
        this.mip0Pass = null;
        this._reduces = [];
    }

    /**
     * The mip 0 shader for a source kind, compiled on first use.
     *
     * @param {boolean} linear - True for the linear scene depth source.
     * @returns {Shader} The shader.
     * @private
     */
    _mip0Shader(linear) {
        if (linear) {
            this._mip0LinearShader ??= ShaderUtils.createShader(this.device, {
                uniqueName: 'MeshletHzbMip0Linear',
                attributes: { aPosition: SEMANTIC_POSITION },
                vertexChunk: 'quadVS',
                fragmentWGSL: mip0LinearWGSL
            });
            return this._mip0LinearShader;
        }
        this._mip0DepthShader ??= ShaderUtils.createShader(this.device, {
            uniqueName: 'MeshletHzbMip0',
            attributes: { aPosition: SEMANTIC_POSITION },
            vertexChunk: 'quadVS',
            fragmentWGSL: mip0DepthWGSL
        });
        return this._mip0DepthShader;
    }

    /**
     * (Re)creates the pyramid for a depth source of the given size and kind.
     *
     * @param {Texture} depthTexture - The scene depth to reduce from: a depth texture, or the
     * linear scene depth attachment when `linear` is set.
     * @param {number} width - Source width.
     * @param {number} height - Source height.
     * @param {boolean} [linear] - True when the source is the reciprocal linear depth a
     * CameraFrame renders (see the class notes). Defaults to false.
     * @param {(() => Texture|null)|null} [source] - Resolves the source texture as the mip 0
     * pass executes, for sources that can be replaced between now and then (see {@link _source}).
     * Null reads `depthTexture` as given.
     */
    resize(depthTexture, width, height, linear = false, source = null) {
        const w = Math.max(width >> 1, 1);
        const h = Math.max(height >> 1, 1);
        this._depthTexture = depthTexture;
        this._source = source;
        if (this.texture && this.width === w && this.height === h && this.linear === linear) {
            return;
        }
        const device = this.device;

        this._mip0Target?.destroy();
        this.texture?.destroy();

        this.width = w;
        this.height = h;
        this.linear = linear;
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

        // mip 0 quad pass - created once, re-targeted on every resize
        if (!this.mip0Pass) {
            const pass = new RenderPassShaderQuad(device);
            pass.name = 'MeshletHzbMip0';
            const self = this;
            const superExecute = pass.execute.bind(pass);
            pass.execute = function () {
                // a frame rebuilt this frame has no depth yet to reduce - skip rather than bind a
                // destroyed texture; the pyramid then keeps last frame's contents
                const texture = self._source ? self._source() : self._depthTexture;
                if (!texture) return;
                device.scope.resolve('uDepthMap').setValue(texture);
                superExecute();
            };
            this.mip0Pass = pass;
        }
        const pass = this.mip0Pass;
        pass.shader = this._mip0Shader(linear);
        pass.init(this._mip0Target);
        // the quad writes every texel, so never load the previous contents
        pass.colorOps.clear = true;
        pass.colorOps.store = true;

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
