import { SHADER_FORWARD } from '../constants.js';
import { RenderPass } from '../../platform/graphics/render-pass.js';

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { ForwardRenderer } from '../renderer/forward-renderer.js'
 * @import { CameraComponent } from '../../framework/components/camera/component.js'
 * @import { Layer } from '../layer.js'
 * @import { MeshInstance } from '../mesh-instance.js'
 * @import { RenderTarget } from '../../platform/graphics/render-target.js'
 */

/**
 * Renders one phase's meshlet mesh instances into the camera's render target. Phase 1 runs
 * before the scene's opaque passes and owns the camera clear (color/depth/stencil); phase 2
 * loads everything. Driven entirely by the GPU-written indirect draw args each instance was
 * pointed at.
 *
 * Lighting rides on a borrowed layer (the camera's World layer): its splitLights feed the
 * directional/local light uniforms and its light hash keys the shader variants. For clustered
 * lighting the pass exposes a one-entry layerRenderSteps array - WorldClustersAllocator assigns
 * real clusters to any pass carrying that property, hash-deduped with the main pass, so meshlet
 * draws share the World layer's cluster data with no allocator changes.
 *
 * @ignore
 */
class RenderPassMeshletDraw extends RenderPass {
    /** @type {ForwardRenderer} */
    renderer;

    /** @type {CameraComponent|null} */
    cameraComponent = null;

    /** @type {MeshInstance[]} */
    meshInstances = [];

    /**
     * Duck-typed {@link LayerRenderStep} list for WorldClustersAllocator._assignClustersForPass:
     * one entry pointing at the borrowed light layer; the allocator writes lightClusters.
     *
     * @type {Array<{ layer: Layer|null, lightClusters: object|null }>}
     */
    layerRenderSteps = [{ layer: null, lightClusters: null }];

    /**
     * @param {GraphicsDevice} device - The graphics device.
     * @param {ForwardRenderer} renderer - The forward renderer.
     * @param {string} name - Pass name for debugging.
     */
    constructor(device, renderer, name) {
        super(device);
        this.renderer = renderer;
        this.name = name;
    }

    /**
     * Points the pass at this frame's target, camera and light layer. Named setup, not
     * frameUpdate - the frame graph calls frameUpdate() on every pass with no arguments each
     * frame.
     *
     * @param {CameraComponent} cameraComponent - The camera.
     * @param {RenderTarget|null} renderTarget - The target.
     * @param {Layer|null} lightLayer - The layer whose lights shade the meshlet draws
     * (typically the camera's World layer). Null renders unlit-by-scene (no lights).
     * @param {{ clearColor: boolean, clearDepth: boolean, clearStencil: boolean }|null} [clears] - When
     * set, this pass owns the camera clear: the flagged targets clear with the camera's clear
     * values (phase 1). Null loads everything (phase 2).
     */
    setup(cameraComponent, renderTarget, lightLayer = null, clears = null) {
        this.cameraComponent = cameraComponent;
        this.layerRenderSteps[0].layer = lightLayer;

        // null is the backbuffer. A camera whose renderTarget was assigned undefined (rather than
        // never assigned) hands undefined down through the render action, and comparing that
        // against the not-yet-initialised pass would skip init() - the pass would then execute
        // without a render pass having been started.
        renderTarget ??= null;

        if (this.renderTarget !== renderTarget) {
            this.init(renderTarget, {});
            this.colorOps.store = true;
            this.depthStencilOps.storeDepth = true;
        }
        const camera = cameraComponent.camera;
        this.setClearColor(clears?.clearColor ? camera.clearColor : undefined);
        this.setClearDepth(clears?.clearDepth ? camera.clearDepth : undefined);
        this.setClearStencil(clears?.clearStencil ? camera.clearStencil : undefined);
    }

    /**
     * Claims this camera's directional shadow passes as before-passes, the same way
     * RenderPassForward does - this pass is added to the frame graph ahead of the scene
     * passes, so without the claim the meshlet draws would shade with last frame's shadow
     * maps. The claim registry (culler.dirLightShadows) makes the later scene passes skip
     * re-adding them.
     */
    frameUpdate() {
        super.frameUpdate();
        const camera = this.cameraComponent?.camera;
        const renderer = this.renderer;
        if (!camera) return;
        const shadowDirLights = renderer.culler.cameraDirShadowLights.get(camera);
        if (shadowDirLights) {
            for (let l = 0; l < shadowDirLights.length; l++) {
                const light = shadowDirLights[l];
                if (renderer.culler.dirLightShadows.get(light) !== camera) {
                    renderer.culler.dirLightShadows.set(light, camera);
                    const shadowPass = renderer._shadowRendererDirectional.getLightRenderPass(light, camera);
                    if (shadowPass) {
                        this.beforePasses.push(shadowPass);
                    }
                }
            }
        }
    }

    after() {
        super.after();
        // dynamically claimed shadow passes are re-added each frame
        this.beforePasses.length = 0;
    }

    execute() {
        const camera = this.cameraComponent?.camera;
        if (!camera || !this.meshInstances.length) {
            return;
        }
        const step = this.layerRenderSteps[0];
        this.renderer.renderForwardLayer(camera, this.renderTarget, null, false, SHADER_FORWARD, {
            meshInstances: this.meshInstances,
            lightLayer: step.layer,
            lightClusters: step.lightClusters ?? undefined
        });
    }
}

export { RenderPassMeshletDraw };
