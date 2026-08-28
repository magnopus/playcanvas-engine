/**
 * Promise wrapper over a Basis-style transcoder for standalone KTX2 payloads (the per-mip
 * files inside a MAG_texture_streaming container). Two rules:
 *
 * - The transcode queue dedups jobs BY URL and would hand every caller the first job's pixels;
 *   each call gets a unique synthetic key.
 * - The job's buffer is transferred to the worker, which detaches it; callers keep slices of a
 *   shared region buffer, so the buffer passed in must already be a private copy (the texture
 *   source's sliceMip returns copies).
 *
 * The transcoder itself is injected rather than imported: it lives in the framework layer
 * (`basisTranscode`), which scene code must not depend on. {@link MeshletComponentSystem}
 * wires the engine's transcoder into the director; a manually created director has to set
 * `director.transcode` before loading textured meshlet assets.
 *
 * @ignore
 */

let _jobId = 0;

/**
 * A `basisTranscode`-compatible function: `(device, url, data, callback, options) => boolean`,
 * where the callback receives `(err, result)` and the return value is false when the
 * transcoder is not initialized.
 *
 * @typedef {(device: import('../../../platform/graphics/graphics-device.js').GraphicsDevice,
 *   url: string, data: ArrayBuffer, callback: (err: string|null, result: object|null) => void,
 *   options: { isKTX2: boolean }) => boolean} MeshletTranscodeFn
 */

/**
 * Transcodes one single-level KTX2 payload to the device's compressed target format.
 *
 * @param {MeshletTranscodeFn|null} transcode - The injected transcoder.
 * @param {import('../../../platform/graphics/graphics-device.js').GraphicsDevice} device - The device.
 * @param {ArrayBuffer} bytes - The KTX2 file bytes (a private copy - it is transferred away).
 * @returns {Promise<{ format: number, width: number, height: number, levels: ArrayBuffer[] }>}
 * The transcoded level data and its engine pixel format (linear variant - callers building
 * sRGB textures apply the sRGB flag at texture creation).
 */
function transcodeKtx2(transcode, device, bytes) {
    return new Promise((resolve, reject) => {
        if (!transcode) {
            reject(new Error('MeshletKtx2: no transcoder - the meshlet component system wires basisTranscode; a manually created MeshletDirector must set director.transcode before loading textured assets.'));
            return;
        }
        const key = `meshlet-ktx2-${_jobId++}`;
        const found = transcode(device, key, bytes, (err, result) => {
            // the worker can report failure with err set OR with a silent null result
            if (err || !result) {
                reject(new Error(err || 'MeshletKtx2: transcode returned no data'));
            } else {
                resolve(result);
            }
        }, { isKTX2: true });
        if (!found) {
            reject(new Error('MeshletKtx2: Basis module not initialized - call basisInitialize() before loading textured meshlet assets.'));
        }
    });
}

export { transcodeKtx2 };
