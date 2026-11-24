import { BUFFERUSAGE_UNIFORM } from '../constants.js';
import { WebgpuBuffer } from './webgpu-buffer.js';

/**
 * A WebGPU implementation of the UniformBuffer.
 *
 * @ignore
 */
class WebgpuUniformBuffer extends WebgpuBuffer {
    /**
     * The contents last uploaded, to skip uploading them again unchanged.
     *
     * @type {Int32Array|null}
     * @private
     */
    _uploaded = null;

    constructor(uniformBuffer) {
        super(BUFFERUSAGE_UNIFORM);
    }

    unlock(uniformBuffer) {
        const device = uniformBuffer.device;
        const storage = uniformBuffer.storageInt32;

        // A persistent uniform buffer keeps its contents on the GPU, and most are set to the same
        // values every frame (a compute dispatched each frame with unchanged parameters). Each
        // upload is a queue write the GPU process handles separately, so an unchanged one is
        // skipped. The comparison covers the whole storage, the range the upload writes.
        const uploaded = this._uploaded;
        if (this.buffer && uploaded && uploaded.length === storage.length) {
            let i = 0;
            const n = storage.length;
            while (i < n && uploaded[i] === storage[i]) i++;
            if (i === n) return;
            uploaded.set(storage);
        } else {
            this._uploaded = storage.slice();
        }
        super.unlock(device, storage.buffer);
    }

    destroy(device) {
        super.destroy(device);
        this._uploaded = null;
    }
}

export { WebgpuUniformBuffer };
