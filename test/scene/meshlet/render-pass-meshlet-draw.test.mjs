import { expect } from 'chai';

import { Color } from '../../../src/core/math/color.js';
import { Camera } from '../../../src/scene/camera.js';
import { SHADER_FORWARD } from '../../../src/scene/constants.js';
import { GraphNode } from '../../../src/scene/graph-node.js';
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
        cameraComponent = { camera };
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

    it('renders its mesh instances through renderForwardLayer with the borrowed light layer', function () {
        const pass = new RenderPassMeshletDraw(app.graphicsDevice, renderer, 'MeshletDraw');
        const calls = [];
        renderer.renderForwardLayer = (...args) => calls.push(args);
        const lightLayer = { splitLights: [[], [], []] };
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
        expect(options.lightLayer).to.equal(lightLayer);
        expect(options.lightClusters, 'unassigned clusters fall through to the allocator default').to.equal(undefined);
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

describe('ForwardRenderer.renderForwardLayer with a borrowed light layer', function () {

    let app;

    beforeEach(function () {
        jsdomSetup();
        app = createApp();
    });

    afterEach(function () {
        app.destroy();
        app = null;
        jsdomTeardown();
    });

    it('takes the split lights and the layer for shader variants from options.lightLayer', function () {
        const renderer = app.renderer;
        const calls = [];
        renderer.renderForward = (...args) => calls.push(args);
        const camera = new Camera(app.graphicsDevice);
        camera.node = new GraphNode();
        // view uniforms the scene would normally have set before any layer renders
        const scope = app.graphicsDevice.scope;
        scope.resolve('cubeMapRotationMatrix').setValue(new Float32Array(9));
        scope.resolve('skyboxIntensity').setValue(1);
        scope.resolve('shadowAtlasParams').setValue(new Float32Array(2));
        const splitLights = [[{}], [], []];
        const lightLayer = { splitLights };
        const meshInstances = [];

        renderer.renderForwardLayer(camera, null, null, false, SHADER_FORWARD, { meshInstances, lightLayer });
        expect(calls).to.have.lengthOf(1);
        const [, , visible, lights, , , layer] = calls[0];
        expect(visible).to.equal(meshInstances);
        expect(lights).to.equal(splitLights);
        expect(layer).to.equal(lightLayer);

        const explicit = [[], [], []];
        renderer.renderForwardLayer(camera, null, null, false, SHADER_FORWARD, { meshInstances, lightLayer, splitLights: explicit });
        expect(calls[1][3], 'explicit split lights win').to.equal(explicit);

        renderer.renderForwardLayer(camera, null, null, false, SHADER_FORWARD, { meshInstances });
        expect(calls[2][3], 'no layer, no lights').to.deep.equal([[], [], []]);
        expect(calls[2][6]).to.equal(null);
    });
});
