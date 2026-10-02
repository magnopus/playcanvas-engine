import { BUFFER_DYNAMIC, BUFFER_GPUDYNAMIC, BUFFER_STATIC, BUFFER_STREAM } from '../constants.js';

/**
 * A WebGL implementation of the Buffer.
 *
 * @ignore
 */
class WebglBuffer {
    bufferId = null;

    destroy(device) {
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
     * @param {number} [byteSize] - Size to allocate when there is no storage to size it from.
     */
    unlock(device, usage, target, storage, byteSize) {
        const gl = device.gl;

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
            gl.bufferSubData(target, 0, storage);
        }
    }
}

export { WebglBuffer };
