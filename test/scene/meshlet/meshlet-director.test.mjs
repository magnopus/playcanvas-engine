import { expect } from 'chai';

import { dressComputeDevice, makeMeshletResource } from './meshlet-test-utils.mjs';
import { Color } from '../../../src/core/math/color.js';
import { RenderTarget } from '../../../src/platform/graphics/render-target.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
import { WebgpuReadbackPool } from '../../../src/platform/graphics/webgpu/webgpu-readback-pool.js';
import { Camera } from '../../../src/scene/camera.js';
import { LAYERID_WORLD } from '../../../src/scene/constants.js';
import { GraphNode } from '../../../src/scene/graph-node.js';
import { MESHLET_BUCKET_COUNT, MESHLET_COLOR_MODE } from '../../../src/scene/meshlet/constants.js';
import { MeshletDirectorGroup } from '../../../src/scene/meshlet/meshlet-director-group.js';
import { createApp } from '../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

const BASE_URL = 'http://meshlets.test/';

const makeResource = (device, instances) => makeMeshletResource(device, { instances });

describe('MeshletDirector', function () {

    let app;
    let device;
    let group;
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
        // a lone director is a group of one
        group = new MeshletDirectorGroup(device);
        group.bindRenderer(renderer);
        director = group.directors[0];
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
        group.destroy();
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
        expect(renderer.meshletDirector, 'the group drives the renderer').to.equal(group);
        expect(director.renderer).to.equal(renderer);
    });

    it('creates one view for the composition camera and encodes its frame', function () {
        finalizeStreamed();
        director.dagPixelThreshold = 2.5;
        director.update(comp);
        expect(director.views.size).to.equal(1);
        const view = director.views.get(cameraComponent);
        expect(view.culler.dagPixelThreshold).to.equal(2.5);
        expect(view.culler.pressureScale).to.equal(director.budget.pressureScale);
        expect(view.culler.rootMinPixels, 'sub-pixel instances are not drawn by default').to.equal(3);
        expect(view.useOcclusion, 'backbuffer camera: no depth to test against').to.equal(false);
        expect(view._countersReadBusy, 'demand readback issued').to.equal(true);
        expect(director.isLightLayer(layer)).to.equal(true);
        expect(director.isLightLayer({})).to.equal(false);

        director.update(comp);
        expect(director.views.get(cameraComponent), 'same camera keeps its view').to.equal(view);
    });

    it('measures the screen-space thresholds in CSS pixels on a high-density canvas', function () {
        finalizeStreamed();
        director.dagPixelThreshold = 1.5;
        Object.defineProperty(device.canvas, 'clientHeight', { value: device.height / 2, configurable: true });
        director.update(comp);
        const view = director.views.get(cameraComponent);
        expect(director.pixelScale).to.equal(2);
        expect(view.culler.dagPixelThreshold).to.equal(3);
        expect(view.culler.rootMinPixels).to.equal(6);

        director.cssPixels = false;
        director.update(comp);
        expect(director.pixelScale).to.equal(1);
        expect(view.culler.dagPixelThreshold).to.equal(1.5);
        expect(view.culler.rootMinPixels).to.equal(3);

        // never finer than one render target pixel: a canvas rendered below its CSS size
        director.cssPixels = true;
        Object.defineProperty(device.canvas, 'clientHeight', { value: device.height * 2, configurable: true });
        director.update(comp);
        expect(director.pixelScale).to.equal(1);
        delete device.canvas.clientHeight;
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
        expect(view.hzb, 'the group\'s pyramid for the camera').to.equal(group.hzbFor(withDepth));
        expect([view.hzb.width, view.hzb.height], 'half-resolution pyramid of the target').to.deep.equal([320, 200]);
        rt.destroy();
        colorBuffer.destroy();
        depthBuffer.destroy();
    });

    const distribute = () => director.budget.distributeIndexBudget(director.world.indexCeiling, director.views,
        director.shadowRenderer.views, director.shadowIndicesPerTexel);

    it('budgets camera views before shadow views, which coarsen under their own pressure', function () {
        finalizeStreamed();
        director.update(comp);
        const [cam] = [...director.views.values()];
        cam.lastRoots.admitted = [0, 0, 0];
        director.world.deviceIndexCeiling = 0;
        director.world.indexBudgetTotal = 1000;
        const shadow = {
            demand: 2000,
            indexShare: 0,
            indexDemand() {
                return this.demand;
            },
            indexPressure() {
                return this.demand / this.indexShare;
            }
        };
        director.shadowRenderer.entries.set({}, { views: [shadow] });

        // floor 125 each; the camera is budgeted for its whole demand at half occupancy first
        cam.lastDemand = { indices: [300, 0, 0], records: 1 };
        expect(distribute(), 'camera pressure only').to.be.closeTo(0.5, 1e-9);
        expect(cam.indexShare).to.equal(600);
        expect(shadow.indexShare, 'the shadow view splits what the camera left').to.equal(400);
        expect(director.budget.shadowIndexRatio).to.be.closeTo(2000 / 400, 1e-9);

        // a coarsened camera whose demand fell under the floor still gets headroom to recover
        cam.lastDemand = { indices: [100, 0, 0], records: 1 };
        expect(distribute()).to.be.closeTo(0.5, 1e-9);
        expect(cam.indexShare).to.equal(200);

        cam.lastDemand = { indices: [3000, 0, 0], records: 1 };
        distribute();
        expect(cam.indexShare, 'a camera wanting everything takes it').to.equal(875);
        expect(shadow.indexShare, 'the shadow view keeps only its floor').to.equal(125);

        // the shadow controller steps in proportion to how far outside the hold band the ratio is,
        // rate-limited per frame
        const budget = director.budget;
        const up = ((budget.indexStarvedAt + 0.1) / budget.indexStarvedAt) ** 0.1;
        let scale = director.budget.stepShadowPressure(1, budget.indexStarvedAt + 0.1);
        expect(scale).to.be.closeTo(up, 1e-9);
        expect(budget.pressureScale, 'the camera controller is untouched').to.equal(1);
        scale = director.budget.stepShadowPressure(scale, (budget.indexStarvedAt + budget.indexEasedAt) / 2);
        expect(scale, 'held between the thresholds').to.be.closeTo(up, 1e-9);
        scale = director.budget.stepShadowPressure(scale, 100);
        expect(scale, 'a large overshoot still only raises by the rate limit').to.be.closeTo(up * 1.012, 1e-9);
        scale = director.budget.stepShadowPressure(scale, 0);
        expect(scale, 'and decays by the rate limit').to.be.closeTo(up * 1.012 * 0.992, 1e-9);
        director.shadowRenderer.entries.clear();
    });

    it('shares the index ceiling across views in proportion to demand, with a floor', function () {
        finalizeStreamed();
        director.cameras = [cameraComponent, makeCameraComponent()];
        director.update(comp);
        const [a, b] = [...director.views.values()];
        director.world.deviceIndexCeiling = 0;
        director.world.indexBudgetTotal = 1000;
        expect(director.world.indexCeiling).to.equal(1000);

        // pressure measures the refinement above each view's admitted roots (the coarsest the cut can go)
        a.lastRoots.admitted = [100, 0, 0];
        const rootsA = 100;
        a.lastDemand = { indices: [900, 0, 0], records: 1 };
        b.lastDemand = { indices: [100, 0, 0], records: 1 };
        expect(distribute()).to.be.closeTo((900 - rootsA) / (875 - rootsA), 1e-9);
        expect(a.indexShare).to.equal(875);
        expect(b.indexShare, 'a floor for the active working set').to.equal(125);
        expect(a.indexShare + b.indexShare).to.equal(1000);

        a.lastDemand = { indices: [3000, 0, 0], records: 1 };
        b.lastDemand = null;
        expect(distribute(), 'demand over the share is starved').to.be.closeTo((3000 - rootsA) / (875 - rootsA), 1e-9);
        expect(b.indexShare, 'an empty view keeps capacity for renewed demand').to.equal(125);

        director.world.indexBudgetTotal = 300;
        distribute();
        expect(a.indexShare + b.indexShare).to.be.at.most(300);

        // a root floor filling most of the share, with nothing refined above it, is not starvation:
        // coarsening cannot shrink it (this pinned the controller at maxScale on wide views)
        director.world.indexBudgetTotal = 1000;
        a.lastRoots.admitted = [800, 0, 0];
        a.lastDemand = { indices: [800, 0, 0], records: 1 };
        distribute();
        expect(a.indexPressure()).to.equal(0);
    });

    it('reports an infeasible working set without silently exceeding the budget', function () {
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
        expect(world.indexCeiling, 'the configured budget remains a limit').to.equal(1000);
        expect(reports[0].indexDemandRatio).to.equal(1.5);
        expect(reports[0].suggestedBudgetBytes).to.equal(world.poolBytes);
        expect(reports[0].poolPages).to.equal(world.poolSlots);
        expect(reports[0].rootInstancesDeferred, 'an overrun, not a deferral').to.equal(0);
    });

    it('reports instances deferred for capacity once per episode, in the same report shape', function () {
        finalizeStreamed();
        director.update(comp);
        const view = director.views.get(cameraComponent);
        const reports = [];
        director.onBudgetExceeded = info => reports.push(info);
        view.lastRoots.rejectedInstances = 3;
        view.lastRoots.unready = 1;
        view.lastRoots.admitted = [90, 0, 0];
        view.lastRoots.rejected = [10, 0, 0];
        director.update(comp);
        director.update(comp);
        expect(reports, 'once while the deferral lasts').to.have.lengthOf(1);
        expect(reports[0]).to.include({ rootInstancesDeferred: 3, rootInstancesUnready: 1, budgetBytes: director.world.poolBytes });
        expect(reports[0].suggestedBudgetBytes).to.equal(director.world.poolBytes + 10 * 4);
        expect(reports[0].indexDemandRatio).to.equal(100 / view.indexTotal());
        expect(Object.keys(reports[0])).to.have.members(['budgetBytes', 'suggestedBudgetBytes', 'indexDemandRatio', 'indexCeiling',
            'pagesMissing', 'poolPages', 'totalPages', 'rootInstancesDeferred', 'rootInstancesUnready', 'rootShortfallPages']);
        view.lastRoots.rejectedInstances = 0;
        director.update(comp);
        view.lastRoots.rejectedInstances = 3;
        director.update(comp);
        expect(reports, 'a new episode reports again').to.have.lengthOf(2);
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
        expect(director.residency.slotPage, 'slot state carried into the new residency').to.deep.equal(prevResidency.slotPage);
        expect(director.residency.frame).to.equal(77);
        const carry = director._carry;
        expect(carry.visWords, '60 pairs: one word of visibility bits').to.equal(1);
        expect(carry.visBits.has(cameraComponent)).to.equal(true);
        expect(carry.caps.get(cameraComponent), 'same size: capacities carried as they were').to.deep.equal([900, 60, 30]);
        expect(director.views.size, 'views are recreated on the next update').to.equal(0);

        director.update(comp);
        const view = director.views.get(cameraComponent);
        expect(view).to.not.equal(prevView);
        expect(view.indexCapacity[0]).to.equal(Math.min(Math.max(world.indexCapacity[0], 900), world.indexWorst[0]));
        expect(carry.visBits.size, 'the carried bits were consumed').to.equal(0);
        expect(carry.caps.size, 'and the capacities').to.equal(0);
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
        expect(director._carry.caps.get(cameraComponent), 'pair ratio 2 x margin, capped at maxCarryScale').to.deep.equal([1800, 120, 60]);
        expect(director._carry.visBits.size, 'a different resource: nothing to carry').to.equal(0);

        director.update(comp);
        director.views.get(cameraComponent).indexCapacity = [900, 60, 30];
        director.rebuild(world => world.addStreamedResource(resource, null, BASE_URL, resource.instances.slice(0, 10)));
        expect(director._carry.caps.get(cameraComponent), '120 pairs down to 30: scaled to a quarter').to.deep.equal([225, 15, 8]);
    });

    it('carries the colour mode across a rebuild', function () {
        const resource = finalizeStreamed();
        director.world.setColorMode(MESHLET_COLOR_MODE.MESHLET);
        const world = director.rebuild(w => w.addStreamedResource(resource, null, BASE_URL));
        expect(world.colorMode).to.equal(MESHLET_COLOR_MODE.MESHLET);
        expect(world.bucketMaterials, 'the debug materials').to.not.equal(world.litMaterials);
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
