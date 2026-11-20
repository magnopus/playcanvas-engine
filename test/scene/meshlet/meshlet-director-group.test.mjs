import { expect } from 'chai';

import { dressComputeDevice, makeMeshletResource } from './meshlet-test-utils.mjs';
import { RenderTarget } from '../../../src/platform/graphics/render-target.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
import { WebgpuReadbackPool } from '../../../src/platform/graphics/webgpu/webgpu-readback-pool.js';
import { Camera } from '../../../src/scene/camera.js';
import { LAYERID_WORLD } from '../../../src/scene/constants.js';
import { GraphNode } from '../../../src/scene/graph-node.js';
import { MESHLET_BUCKET_COUNT } from '../../../src/scene/meshlet/constants.js';
import { MeshletDirectorGroup } from '../../../src/scene/meshlet/meshlet-director-group.js';
import { MESHLET_DIRECTOR_SETTINGS } from '../../../src/scene/meshlet/meshlet-director.js';
import { createApp } from '../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

const BASE_URL = 'http://meshlets.test/';

describe('MeshletDirectorGroup', function () {

    let app;
    let device;
    let group;
    let renderer;
    let layer;
    let comp;
    let cameraComponent;
    let savedFetch;
    let savedRead;
    let savedWarn;

    const makeCameraComponent = (renderTarget = null) => {
        const node = new GraphNode();
        node.setPosition(0, 5, 20);
        const camera = new Camera(device);
        camera.node = node;
        return { camera, entity: node, fov: 45, renderTarget, framePasses: undefined };
    };

    beforeEach(function () {
        jsdomSetup();
        app = createApp();
        device = app.graphicsDevice;
        dressComputeDevice(device);
        // no network and no GPU readbacks: root loads and demand reads stay pending
        savedFetch = globalThis.fetch;
        globalThis.fetch = () => new Promise(() => {});
        savedRead = WebgpuReadbackPool.prototype.read;
        WebgpuReadbackPool.prototype.read = () => new Promise(() => {});
        savedWarn = console.warn;
        console.warn = () => {};

        renderer = {
            meshletDirector: null,
            calls: [],
            addMainRenderPass(...args) {
                this.calls.push(['main', args[3], args[4], args[5]]);
            }
        };
        group = new MeshletDirectorGroup(device);
        group.bindRenderer(renderer);
        layer = { id: LAYERID_WORLD };
        cameraComponent = makeCameraComponent();
        comp = { cameras: [cameraComponent], getLayerById: id => (id === LAYERID_WORLD ? layer : null) };
    });

    afterEach(function () {
        group.destroy();
        globalThis.fetch = savedFetch;
        WebgpuReadbackPool.prototype.read = savedRead;
        console.warn = savedWarn;
        app.destroy();
        app = null;
        device = null;
        jsdomTeardown();
    });

    // finalizes `worlds` worlds of one streamed resource each and runs a frame
    const finalizeStreamed = (worlds = 1) => {
        group.resize(worlds);
        for (const director of group.directors) {
            director.world.addStreamedResource(makeMeshletResource(device), null, BASE_URL);
            director.finalize();
        }
        group.assignInstanceOffsets();
        group.update(comp);
    };

    it('registers with the renderer and hands every director its settings, including later ones', function () {
        expect(renderer.meshletDirector).to.equal(group);
        expect(group.directors).to.have.lengthOf(1);
        const first = group.directors[0];
        for (const name of Object.keys(MESHLET_DIRECTOR_SETTINGS)) {
            expect(first[name], name).to.deep.equal(MESHLET_DIRECTOR_SETTINGS[name]);
        }
        group.dagPixelThreshold = 5;
        group.bucketThresholdScales = [2, 2, 2];
        group.occlusionEnabled = false;
        group.resize(2);
        for (const director of group.directors) {
            expect(director.group).to.equal(group);
            expect(director.renderer).to.equal(renderer);
            expect(director.fetchScheduler).to.equal(group.fetchScheduler);
            expect(director.cullShaders).to.equal(group.cullShaders);
            expect(director.dagPixelThreshold).to.equal(5);
            expect(director.bucketThresholdScales).to.deep.equal([2, 2, 2]);
            expect(director.occlusionEnabled).to.equal(false);
        }
    });

    it('emits the meshlet-aware camera block: phase 1 with the clear, opaque, then transparent', function () {
        finalizeStreamed(2);
        const views = group.directors.map(d => d.views.get(cameraComponent));
        const state = group._cameras.get(cameraComponent);
        const frameGraph = {
            passes: [],
            addPass(pass) {
                this.passes.push(pass);
            }
        };
        const renderActions = [
            { clearColor: true, clearDepth: true, clearStencil: false, transparent: false },
            { transparent: false },
            { transparent: true },
            { transparent: true }
        ];
        const emitted = group.buildCameraPasses(frameGraph, { _renderActions: renderActions }, null, 0, 3, cameraComponent);
        expect(emitted).to.equal(true);
        expect(frameGraph.passes, 'one phase-1 draw for every world').to.deep.equal([state.drawPass1]);
        expect(renderer.calls, 'opaque 0..1 then transparent 2..3, both stripped of clears').to.deep.equal([['main', 0, 1, true], ['main', 2, 3, true]]);
        expect(state.drawPass1.colorOps.clear).to.equal(true);
        expect(state.drawPass1.depthStencilOps.clearDepth).to.equal(true);
        expect(state.drawPass1.depthStencilOps.clearStencil).to.equal(false);
        expect(state.drawPass1.meshInstances).to.deep.equal(views.flatMap(v => v.meshInstances.slice(0, MESHLET_BUCKET_COUNT)));
        expect(state.drawPass1.layerRenderSteps[0].layer).to.equal(layer);

        expect(group.buildCameraPasses(frameGraph, { _renderActions: renderActions }, null, 0, 3, cameraComponent), 'once per frame').to.equal(false);
        expect(group.buildCameraPasses(frameGraph, { _renderActions: renderActions }, null, 0, 3, makeCameraComponent()), 'not our camera').to.equal(false);
    });

    it('hands a CameraFrame its before and middle passes', function () {
        finalizeStreamed();
        const state = group._cameras.get(cameraComponent);
        const camera = cameraComponent.camera;
        camera.clearColorBuffer = true;
        camera.clearDepthBuffer = false;
        camera.clearStencilBuffer = false;
        const colorBuffer = new Texture(device, { width: 64, height: 64 });
        const rt = new RenderTarget({ colorBuffer, depth: false });
        const passes = group.buildCameraFramePasses(cameraComponent, rt, true);
        expect(passes.before).to.deep.equal([state.drawPass1]);
        expect(passes.middle, 'single-phase: nothing between opaque and transparent').to.deep.equal([]);
        expect(state.drawPass1.renderTarget).to.equal(rt);
        expect(state.drawPass1.colorOps.clear).to.equal(true);
        expect(state.drawPass1.depthStencilOps.clearDepth).to.equal(false);
        expect(state.passesAdded).to.equal(true);
        expect(group.buildCameraFramePasses(makeCameraComponent(), rt, true)).to.equal(null);

        state.useOcclusion = true;
        const twoPhase = group.buildCameraFramePasses(cameraComponent, rt, false);
        expect(twoPhase.middle).to.deep.equal([state.hzb.mip0Pass, state.hzbMipsPass, state.cullPhase2Pass, state.drawPass2]);
        expect(state.drawPass1.colorOps.clear, 'no clear when the CameraFrame says so').to.equal(false);
        rt.destroy();
        colorBuffer.destroy();
    });

    it('drops a camera\'s shared state once no world has a view of it', function () {
        finalizeStreamed();
        expect(group._cameras.has(cameraComponent)).to.equal(true);
        group.cameras = [makeCameraComponent()];
        group.update(comp);
        expect(group._cameras.has(cameraComponent)).to.equal(false);
    });
});
