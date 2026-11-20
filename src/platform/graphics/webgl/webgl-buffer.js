import { BUFFER_DYNAMIC, BUFFER_GPUDYNAMIC, BUFFER_STATIC, BUFFER_STREAM } from '../constants.js';

/**
 * A WebGL implementation of the Buffer.
 *
 * @ignore
 */
class WebglBuffer {
    bufferId = null;

    /** @type {Uint8Array|null} */
    uploadView = null;

    destroy(device) {
        this.uploadView = null;
        if (this.bufferId) {
            device.gl.deleteBuffer(this.bufferId);
            this.bufferId = null;
        }
    }

    get initialized() {
        return !!this.bufferId;
    }

    loseContext() {
        this.bufferId = null;
    }

    /**
     * @param {object} device - Graphics device.
     * @param {number} usage - BUFFER_* usage hint.
     * @param {number} target - GL buffer target.
     * @param {ArrayBuffer|ArrayBufferView|null} storage - The CPU copy to upload, or null to
     * allocate an empty buffer of byteSize (contents written elsewhere).
     * @param {number} [byteOffset] - Byte offset in both the storage and GPU buffer. Defaults to 0.
     * @param {number} [byteLength] - Number of bytes to upload. Defaults to the remaining storage.
     * @param {number} [byteSize] - Size to allocate when there is no storage to size it from.
     */
    unlock(device, usage, target, storage, byteOffset = 0, byteLength = storage ? storage.byteLength - byteOffset : 0, byteSize) {
        const gl = device.gl;

        // Do not retain old CPU storage after setData replaces its backing buffer.
        if (this.uploadView && this.uploadView.buffer !== (storage?.buffer ?? storage)) {
            this.uploadView = null;
        }

        if (!this.bufferId) {
            let glUsage;
            switch (usage) {
                case BUFFER_STATIC:
                    glUsage = gl.STATIC_DRAW;
                    break;
                case BUFFER_DYNAMIC:
                    glUsage = gl.DYNAMIC_DRAW;
                    break;
                case BUFFER_STREAM:
                    glUsage = gl.STREAM_DRAW;
                    break;
                case BUFFER_GPUDYNAMIC:
                    glUsage = gl.DYNAMIC_COPY;
                    break;
            }

            this.bufferId = gl.createBuffer();
            gl.bindBuffer(target, this.bufferId);
            // the size overload allocates without uploading
            gl.bufferData(target, storage ?? byteSize, glUsage);
        } else if (storage) {
            gl.bindBuffer(target, this.bufferId);
            if (byteOffset === 0 && byteLength === storage.byteLength) {
                gl.bufferSubData(target, 0, storage);
            } else {
                // Byte units work uniformly for ArrayBuffers and all typed array storage types.
                // Cache the view to avoid allocating one on each partial upload.
                const buffer = storage.buffer ?? storage;
                const offset = storage.byteOffset ?? 0;
                let view = this.uploadView;
                if (!view || view.buffer !== buffer || view.byteOffset !== offset || view.byteLength !== storage.byteLength) {
                    view = this.uploadView = new Uint8Array(buffer, offset, storage.byteLength);
                }
                gl.bufferSubData(target, byteOffset, view, byteOffset, byteLength);
            }
        }
    }
}

export { WebglBuffer };
