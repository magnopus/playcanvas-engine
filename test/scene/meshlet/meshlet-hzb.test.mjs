import { expect } from 'chai';
import { stub } from 'sinon';

import { PIXELFORMAT_R32F } from '../../../src/platform/graphics/constants.js';
import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
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
});
