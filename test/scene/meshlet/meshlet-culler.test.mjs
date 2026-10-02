import { expect } from 'chai';

import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
import {
    CULL_FLAG_NO_TEXEL_RATE, CULL_PARAMS, MESHLET_BUCKET_COUNT, MESHLET_COUNTER, MESHLET_COUNTER_U32S, MESHLET_INSTANCE_CULL_WORKGROUP,
    MESHLET_ROOT_BANDS, MESHLET_ROOT_BAND_MAX_PIXELS, MESHLET_ROOT_BAND_OCTAVES, claimPersistBase
} from '../../../src/scene/meshlet/constants.js';
import { FramePassMeshletCompute } from '../../../src/scene/meshlet/frame-pass-meshlet-compute.js';
import { MeshletCullShaders } from '../../../src/scene/meshlet/meshlet-cull-shaders.js';
import { MeshletCuller } from '../../../src/scene/meshlet/meshlet-culler.js';
import { MeshletHzb } from '../../../src/scene/meshlet/meshlet-hzb.js';
import { createApp } from '../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

// The culler runs on the null device dressed up as compute-capable: Compute instances keep their
// parameters and dispatch sizes but never execute, and the device methods the culler needs for
// encoding are stubbed to record what was asked of them.
const inertCompute = (device) => {
    device.supportsCompute = true;
    device.createComputeImpl = () => ({ destroy() {} });
};

const makeDevice = () => {
    const device = new NullGraphicsDevice(document.createElement('canvas'));
    inertCompute(device);
    const log = { clears: [], dispatches: [], submits: 0, drawSlot: 0, dispatchSlot: 0 };
    device.createBufferImpl = () => ({ allocate() {}, write() {}, destroy() {}, loseContext() {}, buffer: null });
    device.getCommandEncoder = () => ({ clearBuffer: (buffer, offset, size) => log.clears.push(size) });
    device.computeDispatch = (computes, name) => log.dispatches.push({ name, computes: computes.map(c => c.name) });
    device.submit = () => log.submits++;
    device.getIndirectDrawSlot = () => log.drawSlot++;
    device.getIndirectDispatchSlot = () => log.dispatchSlot++;
    return { device, log };
};

// a storage buffer stand-in that records its uploads
const storage = (byteSize = 16) => ({
    impl: { buffer: {} },
    byteSize,
    writes: [],
    write(offset, data, dataOffset = 0, size = data.length - dataOffset) {
        this.writes.push({ offset, data: data.slice(), dataOffset, size });
    }
});

const makeWorld = () => ({
    cut: { levels: [{ start: 0, count: 130, root: true }, { start: 130, count: 50, root: false }], groups: storage(), tasks: storage(), selectionTopology: storage() },
    totalPairs: 1000,
    instanceCount: 130,
    workItemCapacity: 80,
    pageSizeBytes: 65536,
    totalPages: 12,
    objectDataBuffer: storage(),
    meshletDataBuffer: storage(),
    residencyBuffer: storage(),
    requestsBuffer: storage(),
    pagePool: storage()
});

const makeView = (singlePhase = false) => ({
    singlePhase,
    recordCapacity: 100,
    indexCapacity: [10, 20, 30],
    cullParamsBuffer: storage(256),
    countersBuffer: storage(MESHLET_COUNTER_U32S * 4),
    cutBudgetBuffer: storage(20),
    cutBudgetInitial: new Uint32Array(5),
    claimBitsBuffer: storage(64),
    workItemsBuffer: storage(),
    recordsBuffer: storage(),
    selectedMeshletsBuffer: storage(),
    visBitsBuffer: storage(),
    indexBuffer: storage(),
    meshInstances: Array.from({ length: (singlePhase ? 1 : 2) * MESHLET_BUCKET_COUNT }, () => ({
        slot: -1,
        setIndirect(buffer, slot) {
            this.slot = slot;
        }
    }))
});

const planes = Float32Array.from({ length: 24 }, (_, i) => i + 1);
const viewProj = Float32Array.from({ length: 16 }, (_, i) => 100 + i);
const camera = { x: 1, y: 2, z: 3 };

describe('MeshletCuller', function () {

    let device;
    let log;
    let shaders;

    beforeEach(function () {
        jsdomSetup();
        ({ device, log } = makeDevice());
        shaders = new MeshletCullShaders(device);
    });

    afterEach(function () {
        shaders.destroy();
        device.destroy();
        device = null;
        jsdomTeardown();
    });

    it('binds identical world and view buffers to the phase-1 and phase-2 instances of each stage', function () {
        const view = makeView();
        const world = makeWorld();
        const culler = new MeshletCuller(device, world, view, shaders);
        expect(culler.compactMeshlets.getParameter('selectionTopology')).to.equal(world.cut.selectionTopology);
        for (const name of ['objectData', 'meshletData', 'cullParams', 'selectedMeshlets', 'counters', 'records', 'residency', 'requests', 'visBits']) {
            expect(culler.meshletCull.getParameter(name), name).to.equal(culler.meshletCullPhase2.getParameter(name));
        }
        expect(culler.meshletCullPhase2.getParameter('phase')).to.equal(2);
        expect(culler.compactMeshlets.getParameter('workItems')).to.equal(view.workItemsBuffer);
        expect(culler.compactMeshlets.getParameter('claimBits')).to.equal(view.claimBitsBuffer);
        expect(culler.compactMeshlets.getParameter('selectedMeshlets')).to.equal(view.selectedMeshletsBuffer);
        expect(culler.indexWrite.getParameter('drawIndices')).to.equal(view.indexBuffer);
        expect(culler.indexWritePhase2.getParameter('drawIndices')).to.equal(view.indexBuffer);
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            expect(culler.finalizeArgsPhase2.getParameter(`indexCapacity${b}`)).to.equal(view.indexCapacity[b]);
        }
        culler.destroy();
    });

    it('builds one root admission pass per priority band, bound to the root list, before the cut stages', function () {
        const view = makeView();
        const world = makeWorld();
        const culler = new MeshletCuller(device, world, view, shaders);
        expect(culler.cutStages, 'the root level is admitted, not cut').to.have.lengthOf(1);
        expect(culler.cutStages[0].getParameter('taskStart')).to.equal(130);
        expect(culler.rootAdmit).to.have.lengthOf(MESHLET_ROOT_BANDS);
        culler.rootAdmit.forEach((compute, band) => {
            expect(compute.getParameter('rootBand')).to.equal(band);
            expect(compute.getParameter('rootList')).to.equal(view.workItemsBuffer);
            expect(compute.getParameter('cutBudget')).to.equal(view.cutBudgetBuffer);
            for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
                expect(compute.getParameter(`indexCapacity${b}`)).to.equal(view.indexCapacity[b]);
            }
        });
        expect(culler.rootClassify.getParameter('rootList')).to.equal(view.workItemsBuffer);
        expect(culler.rootArgs.getParameter('counterIndex')).to.equal(MESHLET_COUNTER.ROOT_CANDIDATES);

        culler.beginFrame(planes, camera, 640, viewProj, null);
        expect(log.dispatches[0].name).to.equal('MeshletRootAdmit');
        expect(log.dispatches[0].computes).to.deep.equal(['MeshletrootClassify', 'MeshletdispatchArgs',
            ...Array.from({ length: MESHLET_ROOT_BANDS }, (_, band) => `MeshletRootAdmit${band}`)]);
        for (const compute of culler.rootAdmit) {
            expect(compute.indirectSlotIndex).to.equal(culler.rootArgs.getParameter('dispatchSlot'));
        }
        culler.destroy();
    });

    it('rejects a work-item buffer too small to hold the root list', function () {
        const world = makeWorld();
        world.workItemCapacity = 10;
        expect(() => new MeshletCuller(device, world, makeView(), shaders)).to.throw(/root list/);
    });

    it('writes the root priority rows, from the view camera or a given priority origin', function () {
        const view = makeView();
        const culler = new MeshletCuller(device, makeWorld(), view, shaders);
        culler.beginFrame(planes, camera, 640, viewProj, null);
        const row = index => index * 4;
        let params = view.cullParamsBuffer.writes[0].data;
        expect(Array.from(params.subarray(row(CULL_PARAMS.ROOT_PRIORITY), row(CULL_PARAMS.ROOT_PRIORITY) + 4))).to.deep.equal([1, 2, 3, 640]);
        expect(params[row(CULL_PARAMS.ROOT_BANDS)]).to.equal(MESHLET_ROOT_BAND_MAX_PIXELS);
        expect(params[row(CULL_PARAMS.ROOT_BANDS) + 1]).to.be.closeTo(MESHLET_ROOT_BANDS / MESHLET_ROOT_BAND_OCTAVES, 1e-6);
        expect(params[row(CULL_PARAMS.ROOT_BANDS) + 2]).to.equal(MESHLET_ROOT_BANDS);
        expect(params[row(CULL_PARAMS.ROOT_BANDS) + 3]).to.equal(0);

        // a shadow view ranks by the scene camera, not its own orthographic one
        culler.priorityOrigin = { x: 7, y: 8, z: 9 };
        culler.priorityProjScale = 900;
        culler.rootMinPixels = 0.5;
        culler.beginFrame(planes, camera, 640, viewProj, null);
        params = view.cullParamsBuffer.writes[1].data;
        expect(Array.from(params.subarray(row(CULL_PARAMS.ROOT_PRIORITY), row(CULL_PARAMS.ROOT_PRIORITY) + 4))).to.deep.equal([7, 8, 9, 900]);
        expect(params[row(CULL_PARAMS.ROOT_BANDS) + 3]).to.equal(0.5);
        culler.destroy();
    });

    it('keeps last frame\'s admissions for hysteresis, swapping the persistent regions each frame', function () {
        const view = makeView();
        const world = makeWorld();
        const culler = new MeshletCuller(device, world, view, shaders);
        const base = claimPersistBase(world.totalPairs, world.instanceCount);
        const words = Math.ceil(world.instanceCount / 32);

        culler.beginFrame(planes, camera, 640, viewProj, null);
        expect(culler.rootClassify.getParameter('hasPrevious'), 'no previous frame yet').to.equal(0);
        const next0 = culler.rootAdmit[0].getParameter('nextBase');
        expect(next0).to.equal(base + words);

        culler.beginFrame(planes, camera, 640, viewProj, null);
        expect(culler.rootClassify.getParameter('hasPrevious')).to.equal(1);
        expect(culler.rootClassify.getParameter('prevBase'), 'reads what the last frame wrote').to.equal(next0);
        expect(culler.rootAdmit[0].getParameter('nextBase')).to.equal(base);

        culler.hysteresis = false;
        culler.beginFrame(planes, camera, 640, viewProj, null);
        expect(culler.rootClassify.getParameter('hasPrevious')).to.equal(0);
        culler.destroy();
    });

    it('packs the frame parameters into their CULL_PARAMS rows and uploads them', function () {
        const view = makeView();
        const culler = new MeshletCuller(device, makeWorld(), view, shaders);
        culler.dagPixelThreshold = 2;
        culler.pressureScale = 1.5;
        culler.orthoScale = 7;
        culler.cullFlags = CULL_FLAG_NO_TEXEL_RATE;
        culler.viewDir = { x: 0, y: 1, z: 0 };
        culler.beginFrame(planes, camera, 640, viewProj, null);

        const params = view.cullParamsBuffer.writes[0].data;
        const row = index => index * 4;
        expect(Array.from(params.subarray(row(CULL_PARAMS.PLANES), row(CULL_PARAMS.PLANES) + 24))).to.deep.equal(Array.from(planes));
        expect(Array.from(params.subarray(row(CULL_PARAMS.CAMERA), row(CULL_PARAMS.CAMERA) + 4))).to.deep.equal([1, 2, 3, 640]);
        expect(Array.from(params.subarray(row(CULL_PARAMS.LOD), row(CULL_PARAMS.LOD) + 4)), 'threshold x pressure, dummy HZB').to.deep.equal([3, 1, 1, 1]);
        expect(Array.from(params.subarray(row(CULL_PARAMS.VIEW_PROJ), row(CULL_PARAMS.VIEW_PROJ) + 16))).to.deep.equal(Array.from(viewProj));
        expect(Array.from(params.subarray(row(CULL_PARAMS.STREAMING), row(CULL_PARAMS.STREAMING) + 3))).to.deep.equal([12, 7, CULL_FLAG_NO_TEXEL_RATE]);
        expect(Array.from(params.subarray(row(CULL_PARAMS.VIEW_DIR), row(CULL_PARAMS.VIEW_DIR) + 3))).to.deep.equal([0, 1, 0]);
        culler.destroy();
    });

    it('clears the counters and claim bits in-encoder and dispatches the phase-1 chain in order', function () {
        const view = makeView();
        const culler = new MeshletCuller(device, makeWorld(), view, shaders);
        culler.beginFrame(planes, camera, 640, viewProj, null);

        const base = claimPersistBase(1000, 130);
        expect(log.clears, 'counters, transient claim bits, this frame\'s admissions; never the requests buffer')
        .to.deep.equal([MESHLET_COUNTER_U32S * 4, base * 4, Math.ceil(130 / 32) * 4]);
        expect(log.dispatches).to.have.lengthOf(4);
        expect(log.dispatches[0].name).to.equal('MeshletRootAdmit');
        expect(log.dispatches[1].name).to.equal('MeshletResidentCut');
        expect(log.dispatches[1].computes).to.deep.equal(['Meshletcut']);
        expect(log.dispatches[2].name).to.equal('MeshletCompact');
        expect(log.dispatches[2].computes).to.deep.equal(['MeshletinstanceCull', 'MeshletdispatchArgs', 'MeshletcompactMeshlets', 'MeshletdispatchArgs']);
        expect(log.dispatches[3].name).to.equal('MeshletCullPhase1');
        expect(log.dispatches[3].computes).to.deep.equal(['MeshletmeshletCull', 'MeshletfinalizeArgs', 'MeshletindexWrite']);
        expect(culler.instanceCull.countX).to.equal(Math.ceil(130 / MESHLET_INSTANCE_CULL_WORKGROUP));
        expect(culler.meshletCull.getParameter('phase'), 'no HZB: single phase').to.equal(0);
        expect(culler.meshletCull.getParameter('hzbTexture')).to.equal(shaders.dummyHzb);
        // one draw slot per bucket per phase, phase-1 slots handed to the view's mesh instances
        expect(log.drawSlot).to.equal(2 * MESHLET_BUCKET_COUNT);
        expect(view.meshInstances.map(mi => mi.slot)).to.deep.equal([0, 1, 2, 3, 4, 5]);
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            expect(culler.finalizeArgs.getParameter(`drawSlot${b}`)).to.equal(b);
        }
        expect(culler.compactMeshlets.indirectSlotIndex).to.equal(culler.dispatchArgs.getParameter('dispatchSlot'));
        expect(culler.meshletCull.indirectSlotIndex).to.equal(culler.selectedArgs.getParameter('dispatchSlot'));
        expect(culler.indexWrite.indirectSlotIndex).to.equal(culler.finalizeArgs.getParameter('dispatchSlot'));
        culler.destroy();
    });

    it('runs two-phase against the HZB texture and gives phase 2 its own draw slots', function () {
        const view = makeView();
        const culler = new MeshletCuller(device, makeWorld(), view, shaders);
        culler.twoPhase = true;
        const hzb = { texture: new Texture(device, { width: 4, height: 4 }), width: 320, height: 200, mipCount: 9 };
        culler.beginFrame(planes, camera, 640, viewProj, hzb);
        expect(culler.meshletCull.getParameter('phase')).to.equal(1);
        expect(culler.meshletCull.getParameter('hzbTexture')).to.equal(hzb.texture);
        expect(Array.from(view.cullParamsBuffer.writes[0].data.subarray(CULL_PARAMS.LOD * 4 + 1, CULL_PARAMS.LOD * 4 + 4))).to.deep.equal([320, 200, 9]);

        culler.dispatchPhase2();
        expect(log.dispatches[4].name).to.equal('MeshletCullPhase2');
        expect(log.dispatches[4].computes[0], 'counters reset first').to.equal('MeshletresetPhase2');
        expect(culler.meshletCullPhase2.getParameter('hzbTexture')).to.equal(hzb.texture);
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            expect(culler.finalizeArgsPhase2.getParameter(`drawSlot${b}`)).to.equal(MESHLET_BUCKET_COUNT + b);
        }
        hzb.texture.destroy();
        culler.destroy();
    });

    it('allocates only one set of draw slots for a single-phase view', function () {
        const view = makeView(true);
        const culler = new MeshletCuller(device, makeWorld(), view, shaders);
        culler.beginFrame(planes, camera, 640, viewProj, null);
        expect(log.drawSlot).to.equal(MESHLET_BUCKET_COUNT);
        culler.destroy();
    });

    it('splits the chains into submissions around the indirect arguments when asked', function () {
        const culler = new MeshletCuller(device, makeWorld(), makeView(), shaders);
        culler.forceSubmitBoundaries = true;
        culler.beginFrame(planes, camera, 640, viewProj, null);
        expect(log.dispatches.map(d => d.name)).to.deep.equal(['MeshletRootClassify', 'MeshletRootAdmit', 'MeshletResidentCut', 'MeshletCompacta', 'MeshletCompactb', 'MeshletCullPhase1b', 'MeshletCullPhase1c']);
        expect(log.submits).to.equal(4);
        culler.destroy();
    });

    it('re-binds grown index and record state on both phases', function () {
        const view = makeView();
        const culler = new MeshletCuller(device, makeWorld(), view, shaders);
        view.indexCapacity = [11, 22, 33];
        view.indexBuffer = storage();
        culler.bindIndexState(view);
        expect(culler.meshletCullPhase2.getParameter('indexCapacity2')).to.equal(33);
        expect(culler.indexWritePhase2.getParameter('drawIndices')).to.equal(view.indexBuffer);
        view.recordCapacity = 500;
        view.recordsBuffer = storage();
        view.selectedMeshletsBuffer = storage();
        culler.bindRecordState(view);
        for (const compute of [culler.compactMeshlets, culler.meshletCull, culler.meshletCullPhase2]) {
            expect(compute.getParameter('selectedMeshlets')).to.equal(view.selectedMeshletsBuffer);
            expect(compute.getParameter('recordCapacity')).to.equal(500);
        }
        expect(culler.selectedArgs.getParameter('workItemCapacity')).to.equal(500);
        expect(culler.meshletCullPhase2.getParameter('records')).to.equal(view.recordsBuffer);
        expect(culler.finalizeArgsPhase2.getParameter('recordCapacity')).to.equal(500);
        culler.destroy();
    });
});

describe('MeshletHzb', function () {

    let app;
    let device;

    // the mip-0 quad shader goes through the program library, so this needs an application; the
    // null device is flagged as WebGPU so the WGSL source is the one that gets picked
    beforeEach(function () {
        jsdomSetup();
        app = createApp();
        device = app.graphicsDevice;
        device.isWebGPU = true;
        inertCompute(device);
    });

    afterEach(function () {
        app.destroy();
        app = null;
        device = null;
        jsdomTeardown();
    });

    it('builds a half-resolution pyramid and only rebuilds when the size changes', function () {
        const hzb = new MeshletHzb(device);
        const depth = new Texture(device, { width: 1280, height: 800 });
        hzb.resize(depth, 1280, 800);
        expect([hzb.width, hzb.height]).to.deep.equal([640, 400]);
        expect(hzb.mipCount, 'floor(log2(640)) + 1').to.equal(10);
        expect(hzb.mip0Pass.name).to.equal('MeshletHzbMip0');
        expect(hzb.mip0Pass.colorOps.clear, 'the quad writes every texel').to.equal(true);
        const texture = hzb.texture;
        hzb.resize(depth, 1280, 800);
        expect(hzb.texture, 'same size: no reallocation').to.equal(texture);
        hzb.resize(depth, 640, 400);
        expect(hzb.texture).to.not.equal(texture);
        expect(hzb.mipCount).to.equal(9);
        hzb.destroy();
        depth.destroy();
    });
});

describe('FramePassMeshletCompute', function () {

    it('runs its callback from execute', function () {
        let runs = 0;
        const pass = new FramePassMeshletCompute({}, 'MeshletTest', () => runs++);
        expect(pass.name).to.equal('MeshletTest');
        pass.execute();
        pass.execute();
        expect(runs).to.equal(2);
    });
});
