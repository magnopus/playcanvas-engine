import { FramePass } from '../../platform/graphics/frame-pass.js';

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 */

/**
 * A compute-only frame pass: runs a callback from execute(), used to encode meshlet compute
 * work (phase-2 cull, HZB mip reduction) at a specific point in the frame graph, between the
 * render passes whose output it consumes and the ones that consume its output.
 *
 * @ignore
 */
class FramePassMeshletCompute extends FramePass {
    /** @type {Function|null} */
    callback = null;

    /**
     * @param {GraphicsDevice} device - The graphics device.
     * @param {string} name - Pass name for debugging.
     * @param {Function} callback - Called from execute().
     */
    constructor(device, name, callback) {
        super(device);
        this.name = name;
        this.callback = callback;
    }

    execute() {
        this.callback?.();
    }
}

export { FramePassMeshletCompute };
