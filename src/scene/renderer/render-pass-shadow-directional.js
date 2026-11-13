import { DebugHelper } from '../../core/debug.js';
import { RenderPass } from '../../platform/graphics/render-pass.js';
import { SHADOWUPDATE_NONE, SHADOWUPDATE_THISFRAME } from '../constants.js';

/**
 * A render pass used to render directional shadows.
 *
 * @ignore
 */
class RenderPassShadowDirectional extends RenderPass {
    // the shadow map does not depend on the XR view, so it is rendered once for all of them
    perView = false;

    constructor(device, shadowRenderer, light, camera, cascadeMask) {
        super(device);
        DebugHelper.setName(this, `RenderPassShadowDir-${light._node.name}`);

        this.shadowRenderer = shadowRenderer;
        this.light = light;
        this.camera = camera;
        this.cascadeMask = cascadeMask;
        // A cascade update schedule decides during cull, after this pass is created, so a
        // scheduled light always clears per cascade: a whole-target clear would wipe the cascades
        // it skips.
        this.allCascadesRendering = cascadeMask === (1 << light.numShadowFaces) - 1 && !light.cascadeUpdateIntervals;

        // the cascades rendered by the last execute, which the VSM blur is limited to
        this.renderedMask = 0;
    }

    frameUpdate() {
        super.frameUpdate();
        if (this.enabled && this.executeEnabled) {
            this.shadowRenderer.renderer.culler.requestDirectionalShadowCull(this.light, this.camera, this.cascadeMask);
        }
    }

    execute() {

        const { light, camera, shadowRenderer, allCascadesRendering, cascadeMask } = this;
        const faceCount = light.numShadowFaces;
        const shadowUpdateOverrides = light.shadowUpdateOverrides;
        // a cascade update schedule decides per camera and leaves the overrides alone
        const scheduled = !!light.cascadeUpdateIntervals;

        // the shadow map is shared by the cameras, which are all culled before any of them renders,
        // so the split distances this camera's cascades are fitted to are set as the map renders.
        // A light with shadow bounds has no camera and no split distances
        if (camera) {
            light._shadowCascadeDistances.set(light.getRenderData(camera, 0).shadowCascadeDistances);
        }

        // Use the scheduled cascades even if an earlier camera consumed their one-shot overrides.
        let renderedMask = 0;
        for (let face = 0; face < faceCount; face++) {
            if (shadowRenderer.isCascadeRendering(light, camera, face, cascadeMask)) {
                shadowRenderer.renderFace(light, camera, face, !allCascadesRendering);

                renderedMask |= 1 << face;

                if (!scheduled && shadowUpdateOverrides?.[face] === SHADOWUPDATE_THISFRAME) {
                    shadowUpdateOverrides[face] = SHADOWUPDATE_NONE;
                }
            }
        }

        this.renderedMask = renderedMask;
        if (renderedMask === (1 << faceCount) - 1) {
            light._shadowCascadesInvalidated = false;
        }
    }

    after() {
        // apply VSM blur if needed
        if (this.executeEnabled && this.renderedMask) {
            this.shadowRenderer.renderVsm(this.light, this.camera, this.renderedMask);
        }
    }
}

export { RenderPassShadowDirectional };
