import { expect } from 'chai';

import { BUFFER_STATIC, INDEXFORMAT_UINT16, INDEXFORMAT_UINT32 } from '../../../src/platform/graphics/constants.js';
import { IndexBuffer } from '../../../src/platform/graphics/index-buffer.js';
import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import { WebglBuffer } from '../../../src/platform/graphics/webgl/webgl-buffer.js';
import { WebgpuBuffer } from '../../../src/platform/graphics/webgpu/webgpu-buffer.js';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

describe('IndexBuffer', function () {

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

    describe('CPU copy', function () {

        it('allocates no CPU copy when constructed without data, and still creates the GPU buffer', function () {
            let unlocks = 0;
            const original = device.createIndexBufferImpl;
            device.createIndexBufferImpl = (ib, options) => {
                const impl = original.call(device, ib, options);
                impl.unlock = () => unlocks++;
                return impl;
            };
            const ib = new IndexBuffer(device, INDEXFORMAT_UINT32, 100, BUFFER_STATIC);
            expect(ib.storage).to.equal(null);
            expect(ib.numBytes).to.equal(400);
            expect(unlocks, 'unlock() allocates the GPU buffer').to.equal(1);
        });

        it('allocates the CPU copy on the first lock() and returns the same buffer afterwards', function () {
            const ib = new IndexBuffer(device, INDEXFORMAT_UINT16, 10, BUFFER_STATIC);
            const storage = ib.lock();
            expect(storage).to.be.an.instanceof(ArrayBuffer);
            expect(storage.byteLength).to.equal(20);
            expect(ib.lock()).to.equal(storage);
            expect(ib.storage).to.equal(storage);
        });

        it('keeps initial data by reference as the CPU copy', function () {
            const indices = new Uint16Array([0, 1, 2]);
            const ib = new IndexBuffer(device, INDEXFORMAT_UINT16, 3, BUFFER_STATIC, indices);
            expect(ib.storage).to.equal(indices);
            expect(ib.lock()).to.equal(indices);
        });

        it('does not upload on restoreContext when there is no CPU copy to restore from', function () {
            const ib = new IndexBuffer(device, INDEXFORMAT_UINT32, 4, BUFFER_STATIC);
            let unlocks = 0;
            ib.impl.unlock = () => unlocks++;
            ib.restoreContext();
            expect(unlocks).to.equal(0);
            ib.lock();
            ib.restoreContext();
            expect(unlocks).to.equal(1);
        });
    });

    describe('VRAM accounting', function () {

        it('tracks numBytes on creation and releases the same amount on destroy without a CPU copy', function () {
            const before = device._vram.ib;
            const ib = new IndexBuffer(device, INDEXFORMAT_UINT32, 25, BUFFER_STATIC);
            expect(device._vram.ib - before).to.equal(100);
            // the null impl never reports itself initialized; stand in for a real backend
            ib.impl.initialized = true;
            ib.impl.destroy = () => {};
            ib.destroy();
            expect(device._vram.ib).to.equal(before);
        });
    });
});

describe('WebglBuffer.unlock', function () {

    const TARGET = 0x8893;

    const makeGl = () => {
        const calls = [];
        return {
            calls,
            gl: {
                STATIC_DRAW: 1,
                DYNAMIC_DRAW: 2,
                STREAM_DRAW: 3,
                DYNAMIC_COPY: 4,
                createBuffer: () => ({ id: calls.length }),
                bindBuffer: () => {},
                bufferData: (target, sizeOrData, usage) => calls.push(['bufferData', sizeOrData, usage]),
                bufferSubData: (target, offset, data) => calls.push(['bufferSubData', data])
            }
        };
    };

    it('allocates from byteSize without uploading when there is no CPU copy', function () {
        const { gl, calls } = makeGl();
        const buffer = new WebglBuffer();
        buffer.unlock({ gl }, BUFFER_STATIC, TARGET, null, 64);
        expect(calls).to.deep.equal([['bufferData', 64, gl.STATIC_DRAW]]);
        expect(buffer.initialized).to.equal(true);

        // a later unlock with still no CPU copy is a no-op
        buffer.unlock({ gl }, BUFFER_STATIC, TARGET, null, 64);
        expect(calls).to.have.lengthOf(1);
    });

    it('uploads the CPU copy on creation and on later unlocks', function () {
        const { gl, calls } = makeGl();
        const buffer = new WebglBuffer();
        const data = new Uint16Array([1, 2, 3]);
        buffer.unlock({ gl }, BUFFER_STATIC, TARGET, data, 6);
        buffer.unlock({ gl }, BUFFER_STATIC, TARGET, data, 6);
        expect(calls).to.deep.equal([['bufferData', data, gl.STATIC_DRAW], ['bufferSubData', data]]);
    });
});

describe('WebgpuBuffer.unlock', function () {

    let savedUsage;

    before(function () {
        // the WebGPU enum global does not exist under node
        savedUsage = globalThis.GPUBufferUsage;
        globalThis.GPUBufferUsage = { COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128 };
    });

    after(function () {
        globalThis.GPUBufferUsage = savedUsage;
    });

    const makeDevice = () => {
        const created = [];
        const writes = [];
        return {
            created,
            writes,
            device: {
                wgpu: {
                    createBuffer: (desc) => {
                        created.push(desc.size);
                        return { size: desc.size, label: '' };
                    },
                    queue: { writeBuffer: (buffer, offset, data) => writes.push(data.byteLength) }
                }
            }
        };
    };

    it('allocates the rounded byteSize and skips the staging upload when there is no CPU copy', function () {
        const { device, created, writes } = makeDevice();
        const buffer = new WebgpuBuffer(GPUBufferUsage.INDEX);
        buffer.unlock(device, null, 10);
        expect(created).to.deep.equal([12]);
        expect(writes).to.deep.equal([]);
        expect(buffer.usageFlags & GPUBufferUsage.COPY_DST).to.not.equal(0);

        buffer.unlock(device, null, 10);
        expect(created, 'no reallocation').to.have.lengthOf(1);
    });

    it('sizes from the CPU copy and uploads it when one is given', function () {
        const { device, created, writes } = makeDevice();
        const buffer = new WebgpuBuffer(GPUBufferUsage.INDEX);
        buffer.unlock(device, new Uint16Array([1, 2, 3]), 6);
        expect(created).to.deep.equal([8]);
        expect(writes).to.deep.equal([8]);
    });
});
