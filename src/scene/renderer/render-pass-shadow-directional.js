import { DebugHelper } from '../../core/debug.js';
import { RenderPass } from '../../platform/graphics/render-pass.js';
import { SHADOWUPDATE_NONE, SHADOWUPDATE_THISFRAME } from '../constants.js';

/**
 * A render pass used to render directional shadows.
 *
 * @ignore
 */
class RenderPassShadowDirectional extends RenderPass {
    constructor(device, shadowRenderer, light, camera, allCascadesRendering) {
        super(device);
        DebugHelper.setName(this, `RenderPassShadowDir-${light._node.name}`);

        this.shadowRenderer = shadowRenderer;
        this.light = light;
        this.camera = camera;
        this.allCascadesRendering = allCascadesRendering;
    }

    execute() {

        const { light, camera, shadowRenderer, allCascadesRendering } = this;
        const faceCount = light.numShadowFaces;
        const shadowUpdateOverrides = light.shadowUpdateOverrides;
        // a cascade update schedule decides per camera and leaves the overrides alone
        const scheduled = !!light.cascadeUpdateIntervals;

        // render all faces
        for (let face = 0; face < faceCount; face++) {

            if (shadowRenderer.isCascadeRendering(light, camera, face, shadowUpdateOverrides)) {
                shadowRenderer.renderFace(light, camera, face, !allCascadesRendering);
            }

            if (!scheduled && shadowUpdateOverrides?.[face] === SHADOWUPDATE_THISFRAME) {
                shadowUpdateOverrides[face] = SHADOWUPDATE_NONE;
            }
        }
    }

    after() {
        // apply VSM blur if needed
        this.shadowRenderer.renderVsm(this.light, this.camera);
    }
}

export { RenderPassShadowDirectional };
