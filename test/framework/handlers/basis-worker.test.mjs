import { runInNewContext } from 'node:vm';

import { expect } from 'chai';

import { BasisWorker } from '../../../src/framework/handlers/basis-worker.js';
import { PIXELFORMAT_DXT1, PIXELFORMAT_DXT5, PIXELFORMAT_RGBA8 } from '../../../src/platform/graphics/constants.js';
import { transcodeKtx2 } from '../../../src/scene/meshlet/textures/meshlet-ktx2.js';

describe('Basis texture array transcodes', function () {
    const runWorker = async (formats, isTextureArray) => {
        const results = [];
        const selected = [];
        const self = {
            postMessage: message => results.push(message),
            BASIS: () => Promise.resolve({
                initializeBasis() {},
                KTX2File: class {
                    constructor(data) {
                        this.flags = data[0];
                    }

                    getWidth() {
                        return 128;
                    }

                    getHeight() {
                        return 128;
                    }

                    getLevels() {
                        return 1;
                    }

                    getHasAlpha() {
                        return this.flags & 1;
                    }

                    isUASTC() {
                        return this.flags & 2;
                    }

                    startTranscoding() {
                        return true;
                    }

                    getImageTranscodedSizeInBytes(mip, layer, face, format) {
                        selected.push(format);
                        return format === 13 ? 65536 : format === 2 ? 8192 : 16384;
                    }

                    transcodeImage() {
                        return true;
                    }

                    close() {}

                    delete() {}
                }
            })
        };
        runInNewContext(`(${BasisWorker.toString()})()`, { self });
        self.onmessage({ data: { type: 'init', config: { rgbPriority: ['dxt', 'etc2'], rgbaPriority: ['dxt', 'etc2'] } } });
        await Promise.resolve();
        for (const flags of [0, 1, 2, 3]) {
            self.onmessage({ data: {
                type: 'transcode',
                url: `mip-${flags}`,
                data: new Uint8Array([flags]).buffer,
                options: { isKTX2: true, isTextureArray, deviceDetails: { formats } }
            } });
        }
        for (const result of results) expect(result.err).to.equal(undefined);
        return { results, selected };
    };

    it('uses the same alpha-capable blocks for RGB/RGBA and ETC1S/UASTC payloads', async function () {
        const { results, selected } = await runWorker({ dxt: true, etc2: true }, true);
        expect(selected).to.deep.equal([3, 3, 3, 3]);
        expect(results.map(result => result.data.format)).to.deep.equal(Array(4).fill(PIXELFORMAT_DXT5));
        expect(results.map(result => result.data.levels[0].byteLength)).to.deep.equal(Array(4).fill(16384));
    });

    it('uses RGBA8 consistently without GPU compression support', async function () {
        const { results } = await runWorker({}, true);
        expect(results.map(result => result.data.format)).to.deep.equal(Array(4).fill(PIXELFORMAT_RGBA8));
        expect(results.map(result => result.data.levels[0].byteLength)).to.deep.equal(Array(4).fill(65536));
    });

    it('preserves ordinary texture alpha-based format selection', async function () {
        const { results } = await runWorker({ dxt: true }, false);
        expect(results.map(result => result.data.format)).to.deep.equal([PIXELFORMAT_DXT1, PIXELFORMAT_DXT5, PIXELFORMAT_DXT1, PIXELFORMAT_DXT5]);
    });

    it('requests the array contract for every streamed KTX2 job', async function () {
        let options;
        await transcodeKtx2((device, url, data, callback, requestOptions) => {
            options = requestOptions;
            callback(null, { format: PIXELFORMAT_DXT5, levels: [] });
            return true;
        }, {}, new ArrayBuffer(0));
        expect(options).to.deep.equal({ isKTX2: true, isTextureArray: true });
    });
});
