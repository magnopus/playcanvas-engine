import { expect } from 'chai';

import { Color } from '../../../src/core/math/color.js';
import { Camera } from '../../../src/scene/camera.js';
import { SHADER_FORWARD } from '../../../src/scene/constants.js';
import { RenderPassMeshletDraw } from '../../../src/scene/meshlet/render-pass-meshlet-draw.js';
import { createApp } from '../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

describe('RenderPassMeshletDraw', function () {

    let app;
    let renderer;
    let cameraComponent;

    beforeEach(function () {
        jsdomSetup();
        app = createApp();
        renderer = app.renderer;
        const camera = new Camera(app.graphicsDevice);
        camera.clearColor = new Color(0.1, 0.2, 0.3, 1);
        camera.clearDepth = 0.5;
        camera.clearStencil = 3;
        cameraComponent = { camera, shaderParams: {} };
    });

    afterEach(function () {
        app.destroy();
        app = null;
        jsdomTeardown();
    });

    it('treats an undefined render target as the backbuffer and initialises the pass once', function () {
        const pass = new RenderPassMeshletDraw(app.graphicsDevice, renderer, 'MeshletDraw');
        expect(pass.renderTarget, 'not yet initialised').to.equal(undefined);
        pass.setup(cameraComponent, undefined, null, null);
        expect(pass.renderTarget, 'undefined normalised to the backbuffer').to.equal(null);
        expect(pass.colorOps.store).to.equal(true);
        expect(pass.depthStencilOps.storeDepth).to.equal(true);

        const colorOps = pass.colorOps;
        pass.setup(cameraComponent, null, null, null);
        expect(pass.colorOps, 'same target: no re-init').to.equal(colorOps);
    });

    it('owns the camera clear only for the flagged attachments', function () {
        const pass = new RenderPassMeshletDraw(app.graphicsDevice, renderer, 'MeshletDraw');
        pass.setup(cameraComponent, null, null, { clearColor: true, clearDepth: false, clearStencil: true });
        expect(pass.colorOps.clear).to.equal(true);
        expect(pass.colorOps.clearValue.equals(cameraComponent.camera.clearColor)).to.equal(true);
        expect(pass.depthStencilOps.clearDepth).to.equal(false);
        expect(pass.depthStencilOps.clearStencil).to.equal(true);
        expect(pass.depthStencilOps.clearStencilValue).to.equal(3);

        pass.setup(cameraComponent, null, null, null);
        expect(pass.colorOps.clear, 'phase 2 loads everything').to.equal(false);
        expect(pass.depthStencilOps.clearDepth).to.equal(false);
        expect(pass.depthStencilOps.clearStencil).to.equal(false);
    });

    it('renders its mesh instances through renderForwardLayer with the light list of the borrowed layer', function () {
        const pass = new RenderPassMeshletDraw(app.graphicsDevice, renderer, 'MeshletDraw');
        const calls = [];
        renderer.renderForwardLayer = (...args) => calls.push(args);
        const lightList = {};
        const clustered = [];
        const lightLayer = {
            getLightList: (c) => {
                clustered.push(c);
                return lightList;
            }
        };
        pass.setup(cameraComponent, null, lightLayer, null);
        expect(pass.layerRenderSteps[0].layer).to.equal(lightLayer);

        pass.execute();
        expect(calls, 'nothing to draw: no call').to.have.lengthOf(0);

        pass.meshInstances = [{}, {}];
        pass.layerRenderSteps[0].lightClusters = null;
        pass.execute();
        expect(calls).to.have.lengthOf(1);
        const [camera, renderTarget, layer, transparent, shaderPass, options] = calls[0];
        expect(camera).to.equal(cameraComponent.camera);
        expect(renderTarget).to.equal(null);
        expect(layer, 'no layer: the instances come from options').to.equal(null);
        expect(transparent).to.equal(false);
        expect(shaderPass).to.equal(SHADER_FORWARD);
        expect(options.meshInstances).to.equal(pass.meshInstances);
        expect(options.lightList).to.equal(lightList);
        expect(clustered).to.deep.equal([app.scene.clusteredLightingEnabled]);
        expect(options.lightClusters, 'unassigned clusters fall through to the allocator default').to.equal(undefined);
    });

    it('uses the camera debug shader pass in either draw phase and restores forward shading', function () {
        const calls = [];
        renderer.renderForwardLayer = (...args) => calls.push(args[4]);
        const passes = ['Phase1', 'Phase2'].map((name) => {
            const pass = new RenderPassMeshletDraw(app.graphicsDevice, renderer, name);
            pass.setup(cameraComponent, null);
            pass.meshInstances = [{}];
            return pass;
        });
        cameraComponent.camera.shaderPassInfo = { index: 7 };
        passes.forEach(pass => pass.execute());
        cameraComponent.camera.shaderPassInfo = null;
        passes.forEach(pass => pass.execute());
        expect(calls).to.deep.equal([7, 7, SHADER_FORWARD, SHADER_FORWARD]);
    });

    it('claims the camera directional shadow passes once and releases them after the frame', function () {
        const pass = new RenderPassMeshletDraw(app.graphicsDevice, renderer, 'MeshletDraw');
        const camera = cameraComponent.camera;
        const light = {};
        const shadowPass = { name: 'DirShadow' };
        renderer.culler.cameraDirShadowLights.set(camera, [light]);
        renderer._shadowRendererDirectional.getLightRenderPass = (l, c) => ((l === light && c === camera) ? shadowPass : null);

        pass.setup(cameraComponent, null, null, null);
        pass.frameUpdate();
        expect(pass.beforePasses).to.deep.equal([shadowPass]);
        expect(renderer.culler.dirLightShadows.get(light), 'claim registered for the scene passes').to.equal(camera);

        const second = new RenderPassMeshletDraw(app.graphicsDevice, renderer, 'MeshletDraw2');
        second.setup(cameraComponent, null, null, null);
        second.frameUpdate();
        expect(second.beforePasses, 'already claimed this frame').to.deep.equal([]);

        pass.after();
        expect(pass.beforePasses).to.deep.equal([]);
    });
});
