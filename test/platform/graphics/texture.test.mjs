import { expect } from 'chai';

import {
    PIXELFORMAT_RGBA8, PIXELFORMAT_SRGBA8, PIXELFORMAT_DXT1, PIXELFORMAT_DXT1_SRGB, PIXELFORMAT_RGBA16F
} from '../../../src/platform/graphics/constants.js';
import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

describe('Texture', function () {

    /** @type {NullGraphicsDevice} */
    let device;

    beforeEach(function () {
        jsdomSetup();
        const canvas = document.createElement('canvas');
        device = new NullGraphicsDevice(canvas);
    });

    afterEach(function () {
        device.destroy();
        device = null;
        jsdomTeardown();
    });

    describe('#constructor: srgb option', function () {

        it('creates the sRGB variant of the format when srgb is true', function () {
            const texture = new Texture(device, { format: PIXELFORMAT_RGBA8, srgb: true });
            expect(texture.format).to.equal(PIXELFORMAT_SRGBA8);
            expect(texture.srgb).to.be.true;
            texture.destroy();
        });

        it('creates the sRGB variant of a compressed format when srgb is true', function () {
            const texture = new Texture(device, { format: PIXELFORMAT_DXT1, srgb: true });
            expect(texture.format).to.equal(PIXELFORMAT_DXT1_SRGB);
            texture.destroy();
        });

        it('keeps the requested format when srgb is not set', function () {
            const texture = new Texture(device, { format: PIXELFORMAT_RGBA8 });
            expect(texture.format).to.equal(PIXELFORMAT_RGBA8);
            expect(texture.srgb).to.be.false;
            texture.destroy();
        });

        it('ignores srgb for a format with no sRGB variant', function () {
            const texture = new Texture(device, { format: PIXELFORMAT_RGBA16F, srgb: true });
            expect(texture.format).to.equal(PIXELFORMAT_RGBA16F);
            texture.destroy();
        });
    });
});

describe('Texture#gpuSize', function () {

    /** @type {NullGraphicsDevice} */
    let device;

    beforeEach(function () {
        jsdomSetup();
        device = new NullGraphicsDevice(document.createElement('canvas'));
    });

    afterEach(function () {
        device.destroy();
        device = null;
        jsdomTeardown();
    });

    it('counts every layer of a 2D array texture', function () {
        const single = new Texture(device, { width: 64, height: 64, format: PIXELFORMAT_RGBA8, mipmaps: false });
        const array = new Texture(device, { width: 64, height: 64, format: PIXELFORMAT_RGBA8, mipmaps: false, arrayLength: 4 });
        expect(single.gpuSize).to.equal(64 * 64 * 4);
        expect(array.gpuSize).to.equal(single.gpuSize * 4);
        single.destroy();
        array.destroy();
    });
});
