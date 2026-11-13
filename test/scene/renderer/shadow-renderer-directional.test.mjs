import { expect } from 'chai';

import { Entity } from '../../../src/framework/entity.js';
import { SHADOW_PCSS_32F } from '../../../src/scene/constants.js';
import { createApp } from '../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

/**
 * @import { Application } from '../../../src/framework/application.js'
 * @import { Camera } from '../../../src/scene/camera.js'
 * @import { Light } from '../../../src/scene/light.js'
 * @import { MeshInstance } from '../../../src/scene/mesh-instance.js'
 */

describe('ShadowRendererDirectional', function () {
    /** @type {Application} */
    let app;

    beforeEach(function () {
        jsdomSetup();
        app = createApp();
    });

    afterEach(function () {
        app?.destroy();
        app = null;
        jsdomTeardown();
    });

    /**
     * @param {object} [options] - Extra light component data.
     * @returns {Light} A cascaded directional light.
     */
    const createLight = (options = {}) => {
        const entity = new Entity();
        entity.addComponent('light', {
            type: 'directional',
            castShadows: true,
            numCascades: 4,
            shadowDistance: 100,
            ...options
        });
        entity.setEulerAngles(45, 30, 0);
        app.root.addChild(entity);
        return entity.light.light;
    };

    /**
     * @returns {Camera} A camera at the origin.
     */
    const createCamera = () => {
        const entity = new Entity();
        entity.addComponent('camera');
        app.root.addChild(entity);
        return entity.camera.camera;
    };

    /**
     * @param {number} [z] - Distance along the camera's view (negative z).
     * @param {number} [scale] - Uniform scale.
     * @returns {MeshInstance} A shadow-casting box.
     */
    const createCaster = (z = 0, scale = 1) => {
        const entity = new Entity();
        entity.addComponent('render', { type: 'box' });
        entity.setPosition(0, 0, z);
        entity.setLocalScale(scale, scale, scale);
        app.root.addChild(entity);
        return entity.render.meshInstances[0];
    };

    const cull = (light, camera, casters) => {
        const renderer = app.renderer._shadowRendererDirectional;
        renderer.prepareShadowMap(light);
        renderer.cull(light, app.scene.layers, camera, casters);
    };

    const refreshes = (light, camera) => {
        return [0, 1, 2, 3].map(c => light.getRenderData(camera, c).cascadeRefresh);
    };

    it('refreshes cascades on their interval, taking turns, without touching the overrides', function () {
        const light = createLight();
        light.cascadeUpdateIntervals = [1, 1, 2, 4];
        const camera = createCamera();
        const casters = [createCaster()];

        const frames = [];
        for (let f = 0; f < 5; f++) {
            cull(light, camera, casters);
            frames.push(refreshes(light, camera));
        }
        expect(frames[0], 'first frame fits everything').to.deep.equal([true, true, true, true]);
        // cascade c on interval n refreshes when (frame + c) % n == 0
        expect(frames[1]).to.deep.equal([true, true, false, true]);
        expect(frames[2]).to.deep.equal([true, true, true, false]);
        expect(frames[3]).to.deep.equal([true, true, false, false]);
        expect(frames[4]).to.deep.equal([true, true, true, false]);
        expect(light.shadowUpdateOverrides).to.equal(null);
    });

    it('keeps each camera\'s decision, and refreshes a cascade another camera rendered last', function () {
        const light = createLight();
        light.cascadeUpdateIntervals = [1, 4, 4, 4];
        const cameraA = createCamera();
        const cameraB = createCamera();
        const casters = [createCaster()];

        // a single camera settles into skipping
        cull(light, cameraA, casters);
        cull(light, cameraA, casters);
        expect(refreshes(light, cameraA)).to.deep.equal([true, false, false, true]);

        // a second camera takes over the map regions it renders...
        cull(light, cameraB, casters);
        expect(refreshes(light, cameraB)).to.deep.equal([true, true, true, true]);
        // ...without clobbering what the first camera decided for its own pass this frame
        expect(refreshes(light, cameraA)).to.deep.equal([true, false, false, true]);

        // next frame the first camera cannot reuse regions the second overwrote
        cull(light, cameraA, casters);
        expect(refreshes(light, cameraA)).to.deep.equal([true, true, true, true]);

        const shadowRenderer = app.renderer.shadowRenderer;
        expect(shadowRenderer.isCascadeRendering(light, cameraA, 1, 0b1111)).to.equal(true);
        // a cascade outside of the pass's cascade mask never renders
        expect(shadowRenderer.isCascadeRendering(light, cameraA, 1, 0b0001)).to.equal(false);
    });

    it('clears per cascade when scheduled, so a cascade skipped this frame keeps its map', function () {
        const light = createLight();
        light.cascadeUpdateIntervals = [1, 2, 4, 4];
        const camera = createCamera();
        const renderer = app.renderer._shadowRendererDirectional;
        // the pass is built before this frame's cull, when every cascade still reads as refreshing
        cull(light, camera, [createCaster()]);
        expect(refreshes(light, camera)).to.deep.equal([true, true, true, true]);
        const pass = renderer.getLightRenderPass(light, camera);
        expect(pass.allCascadesRendering).to.equal(false);
        expect(pass.depthStencilOps.clearDepth).to.equal(false);
    });

    it('keeps the PCSS depth range stable while far cascades are skipped', function () {
        const light = createLight({ shadowType: SHADOW_PCSS_32F });
        // the null device cannot render float depth, so the light fell back from PCSS
        light._isPcss = true;
        const camera = createCamera();
        // a near caster, and a tall one only the last cascade renders
        const far = createCaster(-90, 30);
        far.shadowCascadeMask = 1 << 3;
        const casters = [createCaster(-2), far];

        cull(light, camera, casters);
        const farClip = light.getRenderData(camera, 0).shadowCamera.farClip;

        light.cascadeUpdateIntervals = [1, 1, 1, 4];
        for (let f = 0; f < 4; f++) {
            cull(light, camera, casters);
            expect(light.getRenderData(camera, 0).shadowCamera.farClip, `frame ${f}`).to.be.closeTo(farClip, 1e-3);
        }
    });
});
