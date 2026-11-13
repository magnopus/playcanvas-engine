import { Debug } from '../../core/debug.js';
import { math } from '../../core/math/math.js';
import { Vec3 } from '../../core/math/vec3.js';
import { Mat4 } from '../../core/math/mat4.js';
import { BoundingBox } from '../../core/shape/bounding-box.js';
import {
    LIGHTTYPE_DIRECTIONAL, SHADOWUPDATE_NONE
} from '../constants.js';
import { ShadowMap } from './shadow-map.js';
import { RenderPassShadowDirectional } from './render-pass-shadow-directional.js';

/**
 * @import { Camera } from '../camera.js'
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { Light } from '../light.js'
 * @import { Renderer } from './renderer.js'
 * @import { ShadowRenderer } from './shadow-renderer.js'
 */

// Per-cascade scratch state used to make the directional shadow camera tightening cascade-stable.
// Pass 1 of `cull()` collects each cascade's visible-caster AABB into _cascadeAabbs and its
// frustum-slice bounding-sphere radius into _cascadeRadii (with a validity flag). For PCSS shadow
// types pass 2 then tightens every cascade against the *union* of all valid cascade AABBs, so the
// `depthRange` plumbed to the shader (cascade 0's farClip-nearClip) doesn't twitch as casters
// move between cascade-frustum bins.
const _unionSceneAabb = new BoundingBox();
const _cascadeAabbs = [new BoundingBox(), new BoundingBox(), new BoundingBox(), new BoundingBox()];
const _cascadeAabbValid = [false, false, false, false];
const _cascadeRadii = [0, 0, 0, 0];
const _cascadeActive = [false, false, false, false];
const center = new Vec3();
const _lightDir = new Vec3();

// Cascade reuse (Light#cascadeUpdateIntervals): a skipped cascade is refreshed early once its
// frustum slice's bounding sphere has moved, or changed size, by more than this fraction of the
// sphere it was fitted to. The sphere is a loose bound on the slice, so a small drift still leaves
// the visible part of the slice inside the cached map.
const CASCADE_REUSE_DRIFT = 0.1;

// ... or once the light direction has turned further than this (cosine of ~0.25 degrees)
const CASCADE_REUSE_LIGHT_DOT = 0.99999;
const shadowCamView = new Mat4();

const aabbPoints = [
    new Vec3(), new Vec3(), new Vec3(), new Vec3(),
    new Vec3(), new Vec3(), new Vec3(), new Vec3()
];

// evaluate depth range the aabb takes in the space of the camera
const _depthRange = { min: 0, max: 0 };
function getDepthRange(cameraViewMatrix, aabbMin, aabbMax) {
    aabbPoints[0].x = aabbPoints[1].x = aabbPoints[2].x = aabbPoints[3].x = aabbMin.x;
    aabbPoints[1].y = aabbPoints[3].y = aabbPoints[7].y = aabbPoints[5].y = aabbMin.y;
    aabbPoints[2].z = aabbPoints[3].z = aabbPoints[6].z = aabbPoints[7].z = aabbMin.z;
    aabbPoints[4].x = aabbPoints[5].x = aabbPoints[6].x = aabbPoints[7].x = aabbMax.x;
    aabbPoints[0].y = aabbPoints[2].y = aabbPoints[4].y = aabbPoints[6].y = aabbMax.y;
    aabbPoints[0].z = aabbPoints[1].z = aabbPoints[4].z = aabbPoints[5].z = aabbMax.z;

    let minz = 9999999999;
    let maxz = -9999999999;

    for (let i = 0; i < 8; ++i) {
        cameraViewMatrix.transformPoint(aabbPoints[i], aabbPoints[i]);
        const z = aabbPoints[i].z;
        if (z < minz) minz = z;
        if (z > maxz) maxz = z;
    }

    _depthRange.min = minz;
    _depthRange.max = maxz;
    return _depthRange;
}

class ShadowRendererDirectional {
    /** @type {Renderer} */
    renderer;

    /** @type {ShadowRenderer} */
    shadowRenderer;

    /** @type {GraphicsDevice} */
    device;

    constructor(renderer, shadowRenderer) {
        this.renderer = renderer;
        this.shadowRenderer = shadowRenderer;
        this.device = renderer.device;
    }

    // Minimal prerequisite for directional shadow-pass creation: ensure the shadow map exists
    // and the light is marked visible. This is caster- and camera-independent, so it can run
    // early in the frame (before mesh culling), unlike cull() which sets up the per-cascade
    // shadow cameras and culls casters.
    prepareShadowMap(light) {

        // force light visibility if function was manually called
        light.visibleThisFrame = true;

        if (!light._shadowMap) {
            light._shadowMap = ShadowMap.create(this.device, light);
        }
    }

    // cull directional shadow map. prepareShadowMap(light) must have been called first.
    cull(light, comp, camera, casters = null) {

        Debug.assert(light._shadowMap, 'ShadowRendererDirectional.cull requires prepareShadowMap() to have been called for the light first.');

        // generate splits for the cascades
        const nearDist = camera._nearClip;
        this.generateSplitDistances(light, nearDist, Math.min(camera._farClip, light.shadowDistance));

        // A cascade update schedule decides per cascade, per frame and per camera, recording the
        // decision on the camera's render data (cameras cull before any of them render, so a
        // light-wide flag would be overwritten by the next camera's cull).
        const intervals = light.cascadeUpdateIntervals;
        const shadowUpdateOverrides = light.shadowUpdateOverrides;
        _lightDir.copy(light._node.forward);
        // the camera's schedule clock: cascade c on interval n refreshes when (frame + c) % n == 0,
        // so cascades sharing an interval take turns instead of landing on the same frame
        const scheduleFrame = intervals ? light.getRenderData(camera, 0).scheduleFrame++ : 0;
        const mapCamera = light._cascadeMapCamera;
        let numActiveCascades = 0;

        // PASS 1: per-cascade culling + per-cascade visible-caster AABB building. Defers the depth-range
        // tightening (which moves the shadow camera and sets farClip) until pass 2 so that PCSS
        // can use a stable cross-cascade union AABB rather than each cascade's own jumpy
        // visible-caster bounds.
        for (let cascade = 0; cascade < light.numCascades; cascade++) {

            _cascadeActive[cascade] = false;
            _cascadeAabbValid[cascade] = false;

            const lightRenderData = light.getRenderData(camera, cascade);

            // get camera's frustum corners for the cascade, convert them to world space and find their center
            const frustumNearDist = cascade === 0 ? nearDist : light._shadowCascadeDistances[cascade - 1];
            const frustumFarDist = light._shadowCascadeDistances[cascade];
            const frustumPoints = camera.getFrustumCorners(frustumNearDist, frustumFarDist);
            center.set(0, 0, 0);
            const cameraWorldMat = camera.node.getWorldTransform();
            for (let i = 0; i < 8; i++) {
                cameraWorldMat.transformPoint(frustumPoints[i], frustumPoints[i]);
                center.add(frustumPoints[i]);
            }
            center.mulScalar(1 / 8);

            // radius of the world space bounding sphere for the frustum slice
            let radius = 0;
            for (let i = 0; i < 8; i++) {
                const dist = frustumPoints[i].sub(center).length();
                if (dist > radius) {
                    radius = dist;
                }
            }

            if (intervals) {
                const interval = Math.max(1, intervals[cascade] | 0);
                // the map region must hold this camera's fit to be reused: another camera
                // rendering the same light last overwrote it
                const due = mapCamera[cascade] !== camera ||
                    this._cascadeDue(lightRenderData, interval, (scheduleFrame + cascade) % interval === 0, radius);
                lightRenderData.cascadeRefresh = due;
                if (!due) {
                    // keep the cached map: its shadow camera, and so its matrix, stay as fitted
                    continue;
                }
                mapCamera[cascade] = camera;
                lightRenderData.fitValid = true;
                lightRenderData.fitCenter.copy(center);
                lightRenderData.fitRadius = radius;
                lightRenderData.fitLightDir.copy(_lightDir);
            } else if (shadowUpdateOverrides?.[cascade] === SHADOWUPDATE_NONE) {
                // manually controlled, and the cascade does not render this frame
                continue;
            }

            const shadowCam = lightRenderData.shadowCamera;

            // assign render target
            // Note: this is done during rendering for all shadow maps, but do it here for the case shadow rendering for the directional light
            // is disabled - we need shadow map to be assigned for rendering to work even in this case. This needs further refactoring - as when
            // shadow rendering is set to SHADOWUPDATE_NONE, we should not even execute shadow map culling
            shadowCam.renderTarget = light._shadowMap.renderTargets[0];

            // viewport
            lightRenderData.shadowViewport.copy(light.cascades[cascade]);
            lightRenderData.shadowScissor.copy(light.cascades[cascade]);

            const shadowCamNode = shadowCam._node;
            const lightNode = light._node;

            shadowCamNode.setPosition(lightNode.getPosition());

            // Camera looks down the negative Z, and directional light points down the negative Y
            shadowCamNode.setRotation(lightNode.getRotation());
            shadowCamNode.rotateLocal(-90, 0, 0);

            // axis of light coordinate system
            const right = shadowCamNode.right;
            const up = shadowCamNode.up;
            const lightDir = shadowCamNode.forward;

            // transform the sphere's center into the center of the shadow map, pixel aligned.
            // this makes the shadow map stable and avoids shimmering on the edges when the camera moves
            const sizeRatio = 0.25 * light._shadowResolution / radius;
            const x = Math.ceil(center.dot(up) * sizeRatio) / sizeRatio;
            const y = Math.ceil(center.dot(right) * sizeRatio) / sizeRatio;

            const scaledUp = up.mulScalar(x);
            const scaledRight = right.mulScalar(y);
            const dot = center.dot(lightDir);
            const scaledDir = lightDir.mulScalar(dot);
            center.add2(scaledUp, scaledRight).add(scaledDir);

            // look at the center from far away to include all casters during culling
            shadowCamNode.setPosition(center);
            shadowCamNode.translateLocal(0, 0, 1000000);
            shadowCam.nearClip = 0.01;
            shadowCam.farClip = 2000000;
            shadowCam.orthoHeight = radius;

            // cull shadow casters
            shadowCam.updateFrustum();
            this.shadowRenderer.cullShadowCasters(comp, light, lightRenderData.visibleCasters, shadowCam, casters);

            const cascadeFlag = 1 << cascade;
            const visibleCasters = lightRenderData.visibleCasters;
            const origNumVisibleCasters = visibleCasters.length;

            let numVisibleCasters = 0;
            const cascadeAabb = _cascadeAabbs[cascade];

            // exclude all mesh instances that are hidden for this cascade.
            // find out AABB of visible shadow casters
            for (let i = 0; i < origNumVisibleCasters; i++) {
                const meshInstance = visibleCasters[i];
                if (meshInstance.shadowCascadeMask & cascadeFlag) {
                    visibleCasters[numVisibleCasters++] = meshInstance;
                    if (numVisibleCasters === 1) {
                        cascadeAabb.copy(meshInstance.aabb);
                    } else {
                        cascadeAabb.add(meshInstance.aabb);
                    }
                }
            }

            // remove empty tail
            if (origNumVisibleCasters !== numVisibleCasters) {
                visibleCasters.length = numVisibleCasters;
            }

            _cascadeAabbValid[cascade] = numVisibleCasters > 0;
            if (intervals) {
                lightRenderData.fitCasterAabbValid = numVisibleCasters > 0;
                if (numVisibleCasters) lightRenderData.fitCasterAabb.copy(cascadeAabb);
            }
            _cascadeRadii[cascade] = radius;
            _cascadeActive[cascade] = true;
            numActiveCascades++;
        }

        // For PCSS, build a stable union AABB across all valid cascades. The shader reads
        // cascade-0's near/far as `depthRange`, so if cascade 0 swings 10x as a single mesh
        // straddles its cull boundary, the whole scene's PCSS softness jumps. Tightening every
        // cascade against the union eliminates the binning-driven jumps. Non-PCSS shadow types
        // skip this — they don't use depthRange in the shader and benefit from per-cascade
        // tightening for depth precision.
        // A cascade the schedule skipped contributes the casters of its last refresh, so the
        // union does not change size each time the far cascades skip a frame.
        let useUnion = false;
        if (light._isPcss && numActiveCascades) {
            for (let cascade = 0; cascade < light.numCascades; cascade++) {
                let aabb = null;
                if (_cascadeActive[cascade]) {
                    if (_cascadeAabbValid[cascade]) aabb = _cascadeAabbs[cascade];
                } else if (intervals) {
                    const data = light.getRenderData(camera, cascade);
                    if (data.fitValid && data.fitCasterAabbValid) aabb = data.fitCasterAabb;
                }
                if (!aabb) continue;
                if (!useUnion) {
                    _unionSceneAabb.copy(aabb);
                    useUnion = true;
                } else {
                    _unionSceneAabb.add(aabb);
                }
            }
        }

        // PASS 2: depth-range tightening per cascade. PCSS uses the cross-cascade union AABB
        // (cascade-stable; tightens every cascade including ones with no own casters, so that
        // cascade 0's near/far — which the shader reads via cameraParams — is always sensible).
        // Non-PCSS uses the per-cascade AABB and skips cascades with no casters (nothing to render).
        for (let cascade = 0; cascade < light.numCascades; cascade++) {
            if (!_cascadeActive[cascade]) continue;
            let aabbSource;
            if (useUnion) {
                aabbSource = _unionSceneAabb;
            } else if (_cascadeAabbValid[cascade]) {
                aabbSource = _cascadeAabbs[cascade];
            } else {
                continue;
            }

            const lightRenderData = light.getRenderData(camera, cascade);
            const shadowCam = lightRenderData.shadowCamera;
            const shadowCamNode = shadowCam._node;

            // calculate depth range of the chosen AABB from the point of view of the shadow camera
            shadowCamView.copy(shadowCamNode.getWorldTransform()).invert();
            const depthRange = getDepthRange(shadowCamView, aabbSource.getMin(), aabbSource.getMax());

            // adjust shadow camera's near and far plane to the depth range. Made slightly larger
            // to avoid clipping on near / far plane.
            shadowCamNode.translateLocal(0, 0, depthRange.max + 0.1);
            shadowCam.farClip = depthRange.max - depthRange.min + 0.2;

            lightRenderData.projectionCompensation = _cascadeRadii[cascade];
        }
    }

    /**
     * Whether a cascade on an update schedule must refresh this frame: it has no fit yet, it is
     * its turn, the view has drifted out of what it was fitted to, or the light has turned.
     *
     * @param {object} lightRenderData - The cascade's render data.
     * @param {number} interval - Frames between refreshes.
     * @param {boolean} scheduled - This frame is the cascade's turn.
     * @param {number} radius - This frame's frustum-slice bounding sphere radius (its center is
     * in `center`).
     * @returns {boolean} True when the cascade refreshes.
     * @private
     */
    _cascadeDue(lightRenderData, interval, scheduled, radius) {
        if (interval <= 1 || scheduled || !lightRenderData.fitValid) return true;
        const fitRadius = lightRenderData.fitRadius;
        const drift = CASCADE_REUSE_DRIFT * fitRadius;
        return center.distance(lightRenderData.fitCenter) > drift ||
            Math.abs(radius - fitRadius) > drift ||
            _lightDir.dot(lightRenderData.fitLightDir) < CASCADE_REUSE_LIGHT_DOT;
    }

    // function to generate frustum split distances
    generateSplitDistances(light, nearDist, farDist) {

        light._shadowCascadeDistances.fill(farDist);
        for (let i = 1; i < light.numCascades; i++) {

            //  lerp between linear and logarithmic distance, called practical split distance
            const fraction = i / light.numCascades;
            const linearDist = nearDist + (farDist - nearDist) * fraction;
            const logDist = nearDist * (farDist / nearDist) ** fraction;
            const dist = math.lerp(linearDist, logDist, light.cascadeDistribution);
            light._shadowCascadeDistances[i - 1] = dist;
        }
    }

    /**
     * Create a render pass for directional light shadow rendering for a specified camera.
     *
     * @param {Light} light - The directional light.
     * @param {Camera} camera - The camera.
     * @returns {RenderPassShadowDirectional|null} - The render pass if the shadow rendering is
     * required, or null otherwise.
     */
    getLightRenderPass(light, camera) {

        Debug.assert(light && light._type === LIGHTTYPE_DIRECTIONAL);

        let renderPass = null;
        if (this.shadowRenderer.needsShadowRendering(light)) {

            // shadow cascades have more faces rendered within a singe render pass
            const faceCount = light.numShadowFaces;
            const shadowUpdateOverrides = light.shadowUpdateOverrides;

            // prepare render targets / cameras for rendering. A cascade update schedule decides
            // during cull, which can run after this pass is created, so a scheduled light always
            // clears per cascade: a whole-target clear would wipe the cascades it skips.
            let allCascadesRendering = !light.cascadeUpdateIntervals;
            let shadowCamera;
            for (let face = 0; face < faceCount; face++) {

                if (shadowUpdateOverrides?.[face] === SHADOWUPDATE_NONE) {
                    allCascadesRendering = false;
                }

                shadowCamera = this.shadowRenderer.prepareFace(light, camera, face);
            }

            renderPass = new RenderPassShadowDirectional(this.device, this.shadowRenderer, light, camera, allCascadesRendering);

            // setup render pass using any of the cameras, they all have the same pass related properties
            this.shadowRenderer.setupRenderPass(renderPass, shadowCamera, allCascadesRendering);
        }

        return renderPass;
    }
}

export { ShadowRendererDirectional };
