import { expect } from 'chai';

import { WebgpuGraphicsDevice } from '../../../../src/platform/graphics/webgpu/webgpu-graphics-device.js';
import { WebgpuReadbackPool } from '../../../../src/platform/graphics/webgpu/webgpu-readback-pool.js';

// A fake WebGPU device: storage buffers hold their bytes in an ArrayBuffer, copyBufferToBuffer
// copies between them synchronously, mapBufferAsync resolves immediately.
const makeDevice = () => {
    const log = { created: 0, destroyed: 0, submits: 0, copies: [] };
    const makeGpuBuffer = (size) => {
        const bytes = new Uint8Array(size);
        return {
            size,
            bytes,
            getMappedRange: (offset, length) => bytes.buffer.slice(offset, offset + length),
            unmap: () => {}
        };
    };
    const device = {
        contextLost: false,
        createBufferImpl: () => ({
            buffer: null,
            allocate(dev, size) {
                log.created++;
                this.buffer = makeGpuBuffer(size);
            },
            destroy() {
                log.destroyed++;
                this.buffer = null;
            }
        }),
        getCommandEncoder: () => ({
            copyBufferToBuffer: (src, srcOffset, dst, dstOffset, size) => {
                log.copies.push(size);
                dst.bytes.set(src.bytes.subarray(srcOffset, srcOffset + size), dstOffset);
            }
        }),
        mapBufferAsync: () => Promise.resolve(true),
        submit: () => log.submits++
    };
    const storage = values => ({ impl: { buffer: { bytes: Uint8Array.from(values) } } });
    return { device, log, storage, makeGpuBuffer };
};

describe('WebgpuReadbackPool', function () {

    let savedMapMode;

    before(function () {
        savedMapMode = globalThis.GPUMapMode;
        globalThis.GPUMapMode = { READ: 1 };
    });

    after(function () {
        globalThis.GPUMapMode = savedMapMode;
    });

    it('copies the requested range out through a staging buffer', async function () {
        const { device, log, storage } = makeDevice();
        const pool = new WebgpuReadbackPool(device);
        const data = await pool.read(storage([1, 2, 3, 4, 5, 6, 7, 8]), 4, 4);
        expect(Array.from(data)).to.deep.equal([5, 6, 7, 8]);
        expect(log.copies).to.deep.equal([4]);
        expect(log.submits, 'no submit unless immediate').to.equal(0);
        pool.destroy();
    });

    it('fills a caller-provided typed array', async function () {
        const { device, storage } = makeDevice();
        const pool = new WebgpuReadbackPool(device);
        const target = new Uint32Array(2);
        const data = await pool.read(storage([1, 0, 0, 0, 2, 0, 0, 0]), 0, 8, target);
        expect(data).to.equal(target);
        expect(Array.from(target)).to.deep.equal([1, 2]);
        pool.destroy();
    });

    it('reuses a returned staging buffer of the same size class and keeps classes apart', async function () {
        const { device, log, storage } = makeDevice();
        const pool = new WebgpuReadbackPool(device);
        const small = storage(new Array(512).fill(9));
        await pool.read(small, 0, 300);
        await pool.read(small, 0, 400);
        expect(log.created, 'two reads in the 512-byte class share one buffer').to.equal(1);
        await pool.read(storage(new Array(4096).fill(1)), 0, 4096);
        expect(log.created, 'a 4096-byte read needs its own class').to.equal(2);
        expect(pool._classOf(300)).to.equal(9);
        expect(pool._classOf(4096)).to.equal(12);
        expect(pool._classOf(1), 'minimum class is 256 bytes').to.equal(8);
        pool.destroy();
        expect(log.destroyed).to.equal(2);
    });

    it('destroys a staging buffer returned after the pool was destroyed', async function () {
        const { device, log, storage } = makeDevice();
        const pool = new WebgpuReadbackPool(device);
        const pending = pool.read(storage([1, 2, 3, 4]), 0, 4);
        pool.destroy();
        expect(log.destroyed, 'nothing was free to destroy yet').to.equal(0);
        await pending;
        expect(log.destroyed, 'the in-flight buffer is not pooled').to.equal(1);
        expect(pool._free.size).to.equal(0);
    });

    it('submits immediately when asked', async function () {
        const { device, log, storage } = makeDevice();
        const pool = new WebgpuReadbackPool(device);
        await pool.read(storage([1, 2, 3, 4]), 0, 4, null, true);
        expect(log.submits).to.equal(1);
        pool.destroy();
    });

    it('rejects when the staging buffer cannot be mapped and drops that buffer', async function () {
        const { device, log, storage } = makeDevice();
        device.mapBufferAsync = () => Promise.resolve(false);
        const pool = new WebgpuReadbackPool(device);
        let error = null;
        try {
            await pool.read(storage([1, 2, 3, 4]), 0, 4);
        } catch (e) {
            error = e;
        }
        expect(error).to.be.an.instanceof(Error);
        expect(log.destroyed).to.equal(1);
        expect(pool._free.size, 'an unmappable buffer is not returned to the pool').to.equal(0);
    });
});

describe('WebgpuGraphicsDevice#mapBufferAsync', function () {

    const mapBufferAsync = (device, buffer) => WebgpuGraphicsDevice.prototype.mapBufferAsync.call(device, buffer, 1);

    // the misuse path asserts through console.error; capture it for the whole block
    let originalError;
    let asserts;

    beforeEach(function () {
        originalError = console.error;
        asserts = [];
        console.error = (...args) => asserts.push(args.join(' '));
    });

    afterEach(function () {
        console.error = originalError;
    });

    it('resolves true when the mapping succeeds', async function () {
        const buffer = { mapAsync: () => Promise.resolve() };
        expect(await mapBufferAsync({ contextLost: false }, buffer)).to.equal(true);
        expect(asserts).to.deep.equal([]);
    });

    it('resolves false without touching the buffer when the context is lost', async function () {
        let calls = 0;
        const buffer = { mapAsync: () => {
            calls++;
            return Promise.resolve();
        } };
        expect(await mapBufferAsync({ contextLost: true }, buffer)).to.equal(false);
        expect(calls).to.equal(0);
    });

    it('resolves false on the abort a lost device or destroyed buffer produces', async function () {
        const abort = new Error('aborted');
        abort.name = 'AbortError';
        const buffer = { mapAsync: () => Promise.reject(abort) };
        expect(await mapBufferAsync({ contextLost: false }, buffer)).to.equal(false);
        expect(asserts, 'an abort is expected, not asserted').to.deep.equal([]);
    });

    it('resolves false but asserts on any other rejection', async function () {
        const buffer = { mapAsync: () => Promise.reject(new Error('misuse')) };
        expect(await mapBufferAsync({ contextLost: false }, buffer)).to.equal(false);
        expect(asserts.join('\n')).to.match(/mapAsync failed/);
    });
});
