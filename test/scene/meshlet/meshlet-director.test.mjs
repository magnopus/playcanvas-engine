import { expect } from 'chai';

import { Color } from '../../../src/core/math/color.js';
import { RenderTarget } from '../../../src/platform/graphics/render-target.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
import { WebgpuReadbackPool } from '../../../src/platform/graphics/webgpu/webgpu-readback-pool.js';
import { Camera } from '../../../src/scene/camera.js';
import { LAYERID_WORLD } from '../../../src/scene/constants.js';
import { GraphNode } from '../../../src/scene/graph-node.js';
import {
    MESHLET_BUCKET_COUNT, MESHLET_DATA, MESHLET_DATA_U32S, PAGE_TABLE, PAGE_TABLE_FIELDS
} from '../../../src/scene/meshlet/constants.js';
import { MeshletDirector } from '../../../src/scene/meshlet/meshlet-director.js';
import { MeshletPrimitive, MeshletResource } from '../../../src/scene/meshlet/meshlet-resource.js';
import { createApp } from '../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

const PAGE_BYTES = 256;
const MESHLETS = 3;
const BASE_URL = 'http://meshlets.test/';

// a streamable resource of one primitive, MESHLETS meshlets over two pages, `instances`
// placements along x (see meshlet-world.test.mjs for the fully resident variant)
const makeResource = (device, instances) => {
    const prim = new MeshletPrimitive();
    prim.meshletData = new Uint32Array(MESHLETS * MESHLET_DATA_U32S);
    for (let m = 0; m < MESHLETS; m++) {
        prim.meshletData[m * MESHLET_DATA_U32S + MESHLET_DATA.TRIANGLE_COUNT] = 10;
        prim.meshletData[m * MESHLET_DATA_U32S + MESHLET_DATA.PAGE] = m % 2;
    }
    prim.meshletDataF32 = new Float32Array(prim.meshletData.buffer);
    prim.meshletCount = MESHLETS;
    prim.uvChannelMask = 1;
    prim.vertexCount = 12;
    prim.lods = [];
    prim.aabbCenter = [1, 2, 3];
    prim.aabbHalfExtents = [1, 1, 1];
    prim.materialIndex = 0;
    prim.baseColorFactor = [0.5, 0.25, 1, 1];
    prim.meshIndex = 0;
    prim.primIndex = 0;

    const pageTable = new Uint32Array(2 * PAGE_TABLE_FIELDS);
    pageTable[1 * PAGE_TABLE_FIELDS + PAGE_TABLE.OFFSET_LO] = PAGE_BYTES;
    const manifest = {
        blobs: [{ uri: 'pages_roots.dat', byteLength: 2 * PAGE_BYTES }],
        pageTable,
        pageCount: 2,
        rootPages: [0],
        attributeLayout: { uvComponents: 1, tangents: false },
        pageSizeBytes: PAGE_BYTES,
        pageAlignment: 256,
        positionGrid: { origin: [0.5, 0, 0], step: 0.01, bits: 16 }
    };
    const placements = Array.from({ length: instances }, (_, i) => {
        const matrix = new Float32Array(16);
        matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1;
        matrix[12] = i * 10;
        return { primIndex: 0, matrix };
    });
    return new MeshletResource(device, [prim], manifest, placements);
};

describe('MeshletDirector', function () {

    let app;
    let device;
    let director;
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
        // the null device dressed up as compute-capable WebGPU with recording stubs
        device.isWebGPU = true;
        device.supportsCompute = true;
        device.createComputeImpl = () => ({ destroy() {} });
        device.createBufferImpl = () => ({
            allocate() {}, write() {}, destroy() {}, loseContext() {}, buffer: {}
        });
        device.getCommandEncoder = () => ({ copyBufferToBuffer() {}, clearBuffer() {} });
        device.computeDispatch = () => {};
        device.getIndirectDrawSlot = () => 0;
        device.getIndirectDispatchSlot = () => 0;
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
        director = new MeshletDirector(device);
        director.bindRenderer(renderer);
        layer = {
            id: LAYERID_WORLD,
            added: [],
            removed: [],
            addMeshInstances(mis) {
                this.added.push(...mis);
            },
            removeMeshInstances(mis) {
                this.removed.push(...mis);
            }
        };
        cameraComponent = makeCameraComponent();
        comp = { cameras: [cameraComponent], getLayerById: id => (id === LAYERID_WORLD ? layer : null) };
    });

    afterEach(function () {
        director.destroy();
        globalThis.fetch = savedFetch;
        WebgpuReadbackPool.prototype.read = savedRead;
        console.warn = savedWarn;
        app.destroy();
        app = null;
        device = null;
        jsdomTeardown();
    });

    const finalizeStreamed = (instances = 2) => {
        const resource = makeResource(device, instances);
        director.world.addStreamedResource(resource, null, BASE_URL);
        director.finalize();
        return resource;
    };

    it('finalizes the world with the view counts the budget must cover and starts streaming', function () {
        director.shadowsEnabled = true;
        finalizeStreamed();
        expect(director.world.finalized).to.equal(true);
        expect(director.world.budgetCameraViews).to.equal(1);
        expect(director.world.budgetShadowViews).to.equal(director.shadowBudgetCascades);
        expect(director.residency, 'streamed world: residency created').to.not.equal(null);
        expect(director.rootsLoaded).to.be.a('promise');
        expect(renderer.meshletDirector).to.equal(director);
    });

    it('creates one view for the composition camera and encodes its frame', function () {
        finalizeStreamed();
        director.dagPixelThreshold = 2.5;
        director.update(comp);
        expect(director.views.size).to.equal(1);
        const view = director.views.get(cameraComponent);
        expect(view.culler.dagPixelThreshold).to.equal(2.5);
        expect(view.culler.pressureScale).to.equal(director.budget.pressureScale);
        expect(view.useOcclusion, 'backbuffer camera: no depth to test against').to.equal(false);
        expect(view._countersReadBusy, 'demand readback issued').to.equal(true);
        expect(director.isLightLayer(layer)).to.equal(true);
        expect(director.isLightLayer({})).to.equal(false);

        director.update(comp);
        expect(director.views.get(cameraComponent), 'same camera keeps its view').to.equal(view);
    });

    it('caps the camera set and drops views whose camera left', function () {
        finalizeStreamed();
        const second = makeCameraComponent();
        director.cameras = [cameraComponent, second];
        director.maxViews = 1;
        director.update(comp);
        expect([...director.views.keys()]).to.deep.equal([cameraComponent]);

        const first = director.views.get(cameraComponent);
        let destroyed = false;
        const superDestroy = first.destroy.bind(first);
        first.destroy = () => {
            destroyed = true;
            superDestroy();
        };
        director.cameras = [second];
        director.update(comp);
        expect(destroyed).to.equal(true);
        expect([...director.views.keys()]).to.deep.equal([second]);
    });

    it('runs two-phase only when the camera target carries a depth texture', function () {
        finalizeStreamed();
        director.occlusionEnabled = true;
        director.update(comp);
        expect(director.views.get(cameraComponent).useOcclusion).to.equal(false);

        const colorBuffer = new Texture(device, { width: 640, height: 400 });
        const depthBuffer = new Texture(device, { width: 640, height: 400 });
        const rt = new RenderTarget({ colorBuffer, depthBuffer });
        const withDepth = makeCameraComponent(rt);
        director.cameras = [withDepth];
        director.update(comp);
        const view = director.views.get(withDepth);
        expect(view.useOcclusion).to.equal(true);
        expect(view.culler.twoPhase).to.equal(true);
        expect([view.hzb.width, view.hzb.height], 'half-resolution pyramid of the target').to.deep.equal([320, 200]);
        rt.destroy();
        colorBuffer.destroy();
        depthBuffer.destroy();
    });

    it('shares the index ceiling across views in proportion to demand, with a floor', function () {
        finalizeStreamed();
        director.cameras = [cameraComponent, makeCameraComponent()];
        director.update(comp);
        const [a, b] = [...director.views.values()];
        director.world.deviceIndexCeiling = 0;
        director.world.indexBudgetTotal = 1000;
        expect(director.world.indexCeiling).to.equal(1000);

        a.lastDemand = { indices: [900, 0, 0], records: 1 };
        b.lastDemand = { indices: [100, 0, 0], records: 1 };
        expect(director._distributeIndexBudget()).to.be.closeTo(0.9 / 0.9, 1e-9);
        expect(a.indexShare).to.equal(900);
        expect(b.indexShare, 'floor: a quarter of an equal share').to.equal(125);

        a.lastDemand = { indices: [3000, 0, 0], records: 1 };
        b.lastDemand = null;
        expect(director._distributeIndexBudget(), 'worst view: 3000 over its 1000 share').to.equal(3);
        expect(b.indexShare, 'an empty view keeps the floor').to.equal(125);
    });

    it('raises the index ceiling and reports when the coarsest cut still does not fit', function () {
        finalizeStreamed();
        const world = director.world;
        world.poolBytes = 64 * 1024 * 1024;
        world.deviceIndexCeiling = 100000;
        world.indexBudgetTotal = 1000;
        director.budget.pressureScale = director.budget.maxScale;
        director.budgetOverrunFrames = 3;
        const reports = [];
        director.onBudgetExceeded = info => reports.push(info);

        director._checkBudgetFeasible(1.5);
        director._checkBudgetFeasible(0.5);
        director._checkBudgetFeasible(1.5);
        director._checkBudgetFeasible(1.5);
        expect(reports, 'the streak restarts on a frame that fits').to.have.lengthOf(0);
        director._checkBudgetFeasible(1.5);
        expect(reports).to.have.lengthOf(1);
        expect(world.indexOverrun, 'one step of budgetOverrunStep').to.equal(500);
        expect(world.indexCeiling).to.equal(1500);
        expect(reports[0].indexDemandRatio).to.equal(1.5);
        expect(reports[0].suggestedBudgetBytes).to.equal(world.poolBytes);
    });

    it('emits the meshlet-aware camera block: phase 1 with the clear, opaque, then transparent', function () {
        finalizeStreamed();
        director.update(comp);
        const view = director.views.get(cameraComponent);
        const frameGraph = {
            passes: [],
            addRenderPass(pass) {
                this.passes.push(pass);
            }
        };
        const renderActions = [
            { clearColor: true, clearDepth: true, clearStencil: false, transparent: false },
            { transparent: false },
            { transparent: true },
            { transparent: true }
        ];
        const emitted = director.buildCameraPasses(frameGraph, { _renderActions: renderActions }, null, 0, 3, cameraComponent);
        expect(emitted).to.equal(true);
        expect(frameGraph.passes).to.deep.equal([view.drawPass1]);
        expect(renderer.calls, 'opaque 0..1 then transparent 2..3, both stripped of clears').to.deep.equal([['main', 0, 1, true], ['main', 2, 3, true]]);
        expect(view.drawPass1.colorOps.clear).to.equal(true);
        expect(view.drawPass1.depthStencilOps.clearDepth).to.equal(true);
        expect(view.drawPass1.depthStencilOps.clearStencil).to.equal(false);
        expect(view.drawPass1.meshInstances).to.deep.equal(view.meshInstances.slice(0, MESHLET_BUCKET_COUNT));
        expect(view.drawPass1.layerRenderSteps[0].layer).to.equal(layer);

        expect(director.buildCameraPasses(frameGraph, { _renderActions: renderActions }, null, 0, 3, cameraComponent), 'once per frame').to.equal(false);
        expect(director.buildCameraPasses(frameGraph, { _renderActions: renderActions }, null, 0, 3, makeCameraComponent()), 'not our camera').to.equal(false);
    });

    it('hands a CameraFrame its before and middle passes', function () {
        finalizeStreamed();
        director.update(comp);
        const view = director.views.get(cameraComponent);
        const camera = cameraComponent.camera;
        camera.clearColorBuffer = true;
        camera.clearDepthBuffer = false;
        camera.clearStencilBuffer = false;
        const colorBuffer = new Texture(device, { width: 64, height: 64 });
        const rt = new RenderTarget({ colorBuffer, depth: false });
        const passes = director.buildCameraFramePasses(cameraComponent, rt, true);
        expect(passes.before).to.deep.equal([view.drawPass1]);
        expect(passes.middle, 'single-phase: nothing between opaque and transparent').to.deep.equal([]);
        expect(view.drawPass1.renderTarget).to.equal(rt);
        expect(view.drawPass1.colorOps.clear).to.equal(true);
        expect(view.drawPass1.depthStencilOps.clearDepth).to.equal(false);
        expect(view._passesAdded).to.equal(true);
        expect(director.buildCameraFramePasses(makeCameraComponent(), rt, true)).to.equal(null);

        view.useOcclusion = true;
        view.hzb = { mip0Pass: { name: 'mip0' } };
        const twoPhase = director.buildCameraFramePasses(cameraComponent, rt, false);
        expect(twoPhase.middle).to.deep.equal([view.hzb.mip0Pass, view.hzbMipsPass, view.cullPhase2Pass, view.drawPass2]);
        expect(view.drawPass1.colorOps.clear, 'no clear when the CameraFrame says so').to.equal(false);
        view.hzb = null;
        rt.destroy();
        colorBuffer.destroy();
    });

    it('retains the page pool, residency and grown capacities across an identical rebuild', function () {
        const resource = finalizeStreamed(20);
        director.update(comp);
        const prevWorld = director.world;
        const prevResidency = director.residency;
        const prevPool = prevWorld.pagePool;
        const prevView = director.views.get(cameraComponent);
        prevView.indexCapacity = [900, 60, 30];
        prevResidency.frame = 77;

        director.rebuild(world => world.addStreamedResource(resource, null, BASE_URL));
        const world = director.world;
        expect(world).to.not.equal(prevWorld);
        expect(world.finalized).to.equal(true);
        expect(world.pagePool, 'the pool buffer carried over').to.equal(prevPool);
        expect(director.residency.slotPage, 'slot state carried into the new residency').to.equal(prevResidency.slotPage);
        expect(director.residency.frame).to.equal(77);
        expect(director._visCarry.words, '60 pairs: one word of visibility bits').to.equal(1);
        expect(director._capCarry.get(cameraComponent), 'same size: capacities carried as they were').to.deep.equal([900, 60, 30]);
        expect(director.views.size, 'views are recreated on the next update').to.equal(0);

        director.update(comp);
        const view = director.views.get(cameraComponent);
        expect(view).to.not.equal(prevView);
        expect(view.indexCapacity[0]).to.equal(Math.min(Math.max(world.indexCapacity[0], 900), world.indexWorst[0]));
        expect(director._visCarry.byCamera.size, 'the carried bits were consumed').to.equal(0);
        expect(director._retired, 'and their source buffer waits one frame').to.have.lengthOf(1);
        director.update(comp);
        expect(director._retired).to.have.lengthOf(0);
    });

    it('scales carried capacities for a grown scene and drops the bits when the instances changed', function () {
        const resource = finalizeStreamed(20);
        director.update(comp);
        director.views.get(cameraComponent).indexCapacity = [900, 60, 30];

        const bigger = makeResource(device, 40);
        director.rebuild(world => world.addStreamedResource(bigger, null, BASE_URL));
        expect(director._capCarry.get(cameraComponent), 'pair ratio 2 x margin, capped at maxCarryScale').to.deep.equal([1800, 120, 60]);
        expect(director._visCarry, 'a different resource: nothing to carry').to.equal(null);

        director.update(comp);
        director.views.get(cameraComponent).indexCapacity = [900, 60, 30];
        director.rebuild(world => world.addStreamedResource(resource, null, BASE_URL, resource.instances.slice(0, 10)));
        expect(director._capCarry.get(cameraComponent), '120 pairs down to 30: scaled to a quarter').to.deep.equal([225, 15, 8]);
    });

    it('goes idle on an empty rebuild', function () {
        finalizeStreamed();
        director.update(comp);
        director.rebuild(() => {});
        expect(director.world.finalized).to.equal(false);
        expect(director.residency).to.equal(null);
        expect(director.views.size).to.equal(0);
        director.update(comp);
        expect(director.views.size, 'nothing to drive').to.equal(0);
    });

    it('attaches the phase-1 instances to the outline layer only while something is outlined', function () {
        finalizeStreamed();
        director.setOutlineLayer(layer, new Color(1, 0, 0), new Color(0, 1, 0));
        director.update(comp);
        const view = director.views.get(cameraComponent);
        const phase1 = view.meshInstances.slice(0, MESHLET_BUCKET_COUNT);
        expect(Array.from(phase1[0].getParameter('pcOutlineColor').data)).to.deep.equal([1, 0, 0]);
        expect(Array.from(phase1[0].getParameter('pcOutlineColorHover').data)).to.deep.equal([0, 1, 0]);
        expect(layer.added, 'nothing selected: not attached').to.have.lengthOf(0);

        director.setOutlined([0]);
        expect(layer.added).to.deep.equal(phase1);
        director.setHovered([1]);
        expect(layer.added, 'already attached').to.have.lengthOf(MESHLET_BUCKET_COUNT);
        director.clearOutlines();
        expect(layer.removed, 'hover still keeps it attached').to.have.lengthOf(0);
        director.clearOutlines(true);
        expect(layer.removed).to.deep.equal(phase1);
    });

    it('hands the picker the pick records and the instances that cover the screen', function () {
        finalizeStreamed(2);
        director.update(comp);
        const view = director.views.get(cameraComponent);
        const mapping = new Map();
        let instances = director.preparePicking(cameraComponent, mapping);
        expect(mapping.size).to.equal(2);
        expect(instances, 'single-phase: phase 1 only').to.have.lengthOf(MESHLET_BUCKET_COUNT);
        view.useOcclusion = true;
        instances = director.preparePicking(cameraComponent, mapping);
        expect(instances, 'two-phase: both sets').to.have.lengthOf(2 * MESHLET_BUCKET_COUNT);
        expect(director.preparePicking(makeCameraComponent(), mapping)).to.deep.equal([]);
    });

    it('tears the shadow views down when shadows are switched off', function () {
        finalizeStreamed();
        director.shadowRenderer.entries.set({}, { views: [], layer });
        director.updateShadowLights(comp);
        expect(director.shadowRenderer.entries.size).to.equal(0);
    });
});
