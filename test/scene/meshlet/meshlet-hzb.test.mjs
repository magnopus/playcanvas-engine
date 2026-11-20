import { expect } from 'chai';
import { stub } from 'sinon';

import { Mat4 } from '../../../src/core/math/mat4.js';
import { PIXELFORMAT_R32F } from '../../../src/platform/graphics/constants.js';
import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
import { Camera } from '../../../src/scene/camera.js';
import { MeshletHzb } from '../../../src/scene/meshlet/meshlet-hzb.js';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

describe('MeshletHzb', function () {
    it('preserves explicit half-resolution dimensions during frame updates', function () {
        jsdomSetup();
        const device = new NullGraphicsDevice(document.createElement('canvas'));
        device.supportsCompute = true;
        device.createComputeImpl = () => ({ destroy() {} });
        const depth = new Texture(device, { width: 1281, height: 851, format: PIXELFORMAT_R32F });
        const hzb = new MeshletHzb(device);
        stub(hzb, '_mip0Shader').returns(null);
        try {
            for (const [width, height] of [[1281, 851], [640, 480], [1, 1]]) {
                hzb.resize(depth, width, height, true);
                hzb.mip0Pass.frameUpdate();
                expect(hzb.texture.width).to.equal(Math.max(width >> 1, 1));
                expect(hzb.texture.height).to.equal(Math.max(height >> 1, 1));
                expect(hzb.width).to.equal(hzb.texture.width);
                expect(hzb.height).to.equal(hzb.texture.height);
            }
        } finally {
            hzb.destroy();
            depth.destroy();
            device.destroy();
            jsdomTeardown();
        }
    });

    describe('#setProjection', function () {
        // the float32 depth stored for view depth v
        const storedDepth = (projection, device, v) => {
            const m = Camera.applyShaderProjectionTransform(projection, new Mat4(), false, device.isWebGPU, device.isReverseZ).data;
            const z = -v;
            return Math.fround((m[10] * z + m[14]) / (m[11] * z + m[15]));
        };

        // mip 0's decode, without its conservative step
        const decode = (hzb, d) => {
            const [x, y, z, w] = hzb.depthDecode;
            return (x - d * y) / (z - d * w);
        };

        // the precision float32 depth allows: relative for reverse-z perspective, worse with
        // distance for forward z, and fixed world units for orthographic
        const near = 0.1;
        const far = 10000;
        const cases = [
            { name: 'forward perspective', reverseZ: false, ortho: false, depths: [0.5, 3, 40, 200], tolerance: v => 1e-4 * v },
            { name: 'reverse perspective', reverseZ: true, ortho: false, depths: [0.5, 3, 40, 200, 4000, 9900], tolerance: v => 1e-5 * v },
            { name: 'forward orthographic', reverseZ: false, ortho: true, depths: [0.5, 3, 40, 200, 4000, 9900], tolerance: () => (far - near) * 2 ** -22 },
            { name: 'reverse orthographic', reverseZ: true, ortho: true, depths: [0.5, 3, 40, 200, 4000, 9900], tolerance: () => (far - near) * 2 ** -22 }
        ];

        for (const { name, reverseZ, ortho, depths, tolerance } of cases) {
            it(`decodes ${name} depth into view depth`, function () {
                jsdomSetup();
                const device = new NullGraphicsDevice(document.createElement('canvas'));
                device.supportsCompute = true;
                device.isWebGPU = true;
                device.isReverseZ = reverseZ;
                const hzb = new MeshletHzb(device);
                try {
                    const projection = new Mat4();
                    if (ortho) {
                        projection.setOrtho(-10, 10, -10, 10, near, far);
                    } else {
                        projection.setPerspective(60, 16 / 9, near, far);
                    }
                    hzb.setProjection(projection);
                    for (const v of depths) {
                        const decoded = decode(hzb, storedDepth(projection, device, v));
                        expect(Math.abs(decoded - v)).to.be.below(tolerance(v), `view depth ${v}`);
                    }
                } finally {
                    hzb.destroy();
                    device.destroy();
                    jsdomTeardown();
                }
            });
        }
    });
});
