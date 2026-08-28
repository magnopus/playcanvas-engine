import { Debug } from '../../../core/debug.js';
import { BUFFERUSAGE_COPY_DST, BUFFERUSAGE_READ } from '../constants.js';

/**
 * @import { WebgpuGraphicsDevice } from './webgpu-graphics-device.js'
 * @import { StorageBuffer } from '../storage-buffer.js'
 */

/**
 * A pool of reusable MAP_READ staging buffers for per-frame GPU to CPU readbacks.
 *
 * The device's readStorageBuffer creates and destroys a staging buffer per call, which is fine
 * for one-off reads but unsuitable for readbacks issued every frame (streaming page requests,
 * counter monitors). This pool keeps returned staging buffers in power-of-two size classes and
 * reuses them; a read borrows a buffer, records the copy on the device's command encoder, maps
 * it when the commands complete, copies the data out and returns the buffer to the pool.
 *
 * Owned by its consumer (not the device); call destroy() when done.
 *
 * @ignore
 */
class WebgpuReadbackPool {
    /** @type {WebgpuGraphicsDevice} */
    device;

    /**
     * Free staging buffers by power-of-two size class exponent.
     *
     * @type {Map<number, Array<object>>}
     */
    _free = new Map();

    /** @type {number} */
    _inFlight = 0;

    /** @type {boolean} */
    _destroyed = false;

    /**
     * @param {WebgpuGraphicsDevice} device - The graphics device.
     */
    constructor(device) {
        this.device = device;
    }

    destroy() {
        this._destroyed = true;
        for (const list of this._free.values()) {
            for (const stagingBuffer of list) {
                stagingBuffer.destroy(this.device);
            }
        }
        this._free.clear();
    }

    _classOf(size) {
        return Math.ceil(Math.log2(Math.max(size, 256)));
    }

    _borrow(size) {
        const cls = this._classOf(size);
        const list = this._free.get(cls);
        if (list?.length) {
            return { stagingBuffer: list.pop(), cls };
        }
        const stagingBuffer = this.device.createBufferImpl(BUFFERUSAGE_READ | BUFFERUSAGE_COPY_DST);
        stagingBuffer.allocate(this.device, 2 ** cls);
        return { stagingBuffer, cls };
    }

    _return(stagingBuffer, cls) {
        if (this._destroyed) {
            stagingBuffer.destroy(this.device);
            return;
        }
        let list = this._free.get(cls);
        if (!list) {
            list = [];
            this._free.set(cls, list);
        }
        list.push(stagingBuffer);
    }

    /**
     * Read a range of a storage buffer through a pooled staging buffer.
     *
     * @param {StorageBuffer} storageBuffer - The storage buffer to read.
     * @param {number} offset - Byte offset of the range to read.
     * @param {number} size - Byte size of the range to read.
     * @param {ArrayBufferView|null} [data] - Optional typed array to copy the data into; when
     * omitted a new Uint8Array is allocated.
     * @param {boolean} [immediate] - When true, submits the command buffer immediately instead of
     * mapping on the next event cycle.
     * @returns {Promise<ArrayBufferView>} The data read.
     */
    read(storageBuffer, offset, size, data = null, immediate = false) {
        const device = this.device;
        Debug.assert(!this._destroyed, 'WebgpuReadbackPool used after destroy');

        const { stagingBuffer, cls } = this._borrow(size);

        // The copy is recorded on the current encoder NOW, so the bytes read are the storage
        // buffer's contents at this point in the frame's command stream - a consumer that calls
        // at the top of its update, before encoding this frame's work, reads last frame's result.
        const commandEncoder = device.getCommandEncoder();
        commandEncoder.copyBufferToBuffer(storageBuffer.impl.buffer, offset, stagingBuffer.buffer, 0, size);
        this._inFlight++;

        return new Promise((resolve, reject) => {
            const read = () => {
                device.mapBufferAsync(stagingBuffer.buffer, GPUMapMode.READ).then((mapped) => {
                    this._inFlight--;
                    if (!mapped) {
                        stagingBuffer.destroy(device);
                        reject(new Error('Failed to map a pooled staging buffer for reading, most likely because the device was lost.'));
                        return;
                    }
                    data ??= new Uint8Array(size);
                    const copySrc = stagingBuffer.buffer.getMappedRange(0, size);
                    const srcType = data.constructor;
                    data.set(new srcType(copySrc));
                    stagingBuffer.buffer.unmap();
                    this._return(stagingBuffer, cls);
                    resolve(data);
                });
            };

            if (immediate) {
                device.submit();
                read();
            } else {
                setTimeout(() => {
                    read();
                });
            }
        });
    }
}

export { WebgpuReadbackPool };
