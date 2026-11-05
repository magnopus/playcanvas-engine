/**
 * @import { GraphicsDevice } from './graphics-device.js'
 * @import { Texture } from './texture.js'
 */

/**
 * A handle to the scene normal map of a camera, obtained from
 * {@link CameraComponent#sceneNormalMapHandle}. The handle is a single object for the lifetime of
 * the camera, and always identifies the normal map the camera rendered most recently, even as the
 * camera renders it to a different texture.
 *
 * The normal map stores the signed world space surface normal in RGB (RGBA16F, or RGBA32F where
 * the device cannot render the former), and 0 in alpha where nothing was rendered.
 *
 * Pass the handle to {@link Compute#setSceneNormalMap} to make the normal map available to a
 * compute shader.
 *
 * @hideconstructor
 * @category Graphics
 */
class SceneNormalMapHandle {
    /**
     * The name of the global uniform the scene normal map is published to, for the shaders sampling
     * it.
     *
     * @type {string}
     * @ignore
     */
    static uniformName = 'uSceneNormalMap';

    /**
     * Sets the global uniform the scene normal map is published to. The uniform is shared by all
     * cameras, so it holds whichever normal map was published last.
     *
     * @param {GraphicsDevice} device - The graphics device.
     * @param {Texture|null} texture - The normal map texture, or null to clear the uniform.
     * @ignore
     */
    static setUniform(device, texture) {
        device.scope.resolve(SceneNormalMapHandle.uniformName).setValue(texture);
    }

    /**
     * The texture the normal map was rendered to most recently, or null when the camera has not
     * rendered one. The texture is an attachment of the scene render target, so it is overwritten
     * each frame.
     *
     * @type {Texture|null}
     * @ignore
     */
    texture = null;

    /**
     * The render version the normal map was rendered in, so a consumer can tell a normal map
     * rendered this frame from one left over from an earlier frame.
     *
     * @type {number}
     * @ignore
     */
    renderVersion = -1;

    /**
     * The flipY of the render target the camera rendered the normal map with, which the texture
     * matches the layout of.
     *
     * @type {boolean}
     * @ignore
     */
    flipY = false;

    /**
     * The viewport the camera rendered the normal map with, in texels of the normal map as a
     * compute shader addresses them: the x and y of its first texel, and its width and height.
     * Outside of it, the normal map holds what the camera did not render, such as the views of
     * other cameras rendering to the same target.
     *
     * @type {Uint32Array}
     * @ignore
     */
    viewport = new Uint32Array(4);

    /**
     * True once the camera the handle belongs to has been destroyed.
     *
     * @type {boolean}
     * @ignore
     */
    destroyed = false;

    /**
     * Gets the width of the normal map the camera rendered most recently, in texels, or 0 when the
     * camera has not rendered one. A compute shader processing each texel of the normal map is
     * dispatched over this size, see {@link Compute#setupDispatchThreads}. It changes with the
     * normal map, so read it right before the dispatch: once the camera renders in the frame, it is
     * the width of the normal map of this frame, and before that, of the previous frame.
     *
     * @type {number}
     */
    get width() {
        return this.texture?.width ?? 0;
    }

    /**
     * Gets the height of the normal map the camera rendered most recently, in texels, or 0 when the
     * camera has not rendered one. See {@link SceneNormalMapHandle#width}.
     *
     * @type {number}
     */
    get height() {
        return this.texture?.height ?? 0;
    }
}

export { SceneNormalMapHandle };
