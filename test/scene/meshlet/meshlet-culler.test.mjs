import { expect } from 'chai';

import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
import {
    CULL_FLAG_NO_TEXEL_RATE, CULL_PARAMS, MESHLET_BUCKET_COUNT, MESHLET_INSTANCE_CULL_WORKGROUP
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
    write(offset, data) {
        this.writes.push({ offset, data: data.slice() });
    }
});

const makeWorld = () => ({
    instanceCount: 130,
    workItemCapacity: 40,
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
    countersBuffer: storage(48),
    claimBitsBuffer: storage(64),
    workItemsBuffer: storage(),
    recordsBuffer: storage(),
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
        const culler = new MeshletCuller(device, makeWorld(), view, shaders);
        for (const name of ['objectData', 'meshletData', 'cullParams', 'workItems', 'counters', 'records', 'residency', 'requests', 'claimBits', 'visBits']) {
            expect(culler.meshletCull.getParameter(name), name).to.equal(culler.meshletCullPhase2.getParameter(name));
        }
        expect(culler.meshletCullPhase2.getParameter('phase')).to.equal(2);
        expect(culler.indexWrite.getParameter('drawIndices')).to.equal(view.indexBuffer);
        expect(culler.indexWritePhase2.getParameter('drawIndices')).to.equal(view.indexBuffer);
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            expect(culler.finalizeArgsPhase2.getParameter(`indexCapacity${b}`)).to.equal(view.indexCapacity[b]);
        }
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

        expect(log.clears, 'counters then claim bits, never the requests buffer').to.deep.equal([48, 64]);
        expect(log.dispatches).to.have.lengthOf(1);
        expect(log.dispatches[0].name).to.equal('MeshletCullPhase1');
        expect(log.dispatches[0].computes).to.deep.equal(['MeshletinstanceCull', 'MeshletdispatchArgs', 'MeshletmeshletCull', 'MeshletfinalizeArgs', 'MeshletindexWrite']);
        expect(culler.instanceCull.countX).to.equal(Math.ceil(130 / MESHLET_INSTANCE_CULL_WORKGROUP));
        expect(culler.meshletCull.getParameter('phase'), 'no HZB: single phase').to.equal(0);
        expect(culler.meshletCull.getParameter('hzbTexture')).to.equal(shaders.dummyHzb);
        // one draw slot per bucket per phase, phase-1 slots handed to the view's mesh instances
        expect(log.drawSlot).to.equal(2 * MESHLET_BUCKET_COUNT);
        expect(view.meshInstances.map(mi => mi.slot)).to.deep.equal([0, 1, 2, 3, 4, 5]);
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            expect(culler.finalizeArgs.getParameter(`drawSlot${b}`)).to.equal(b);
        }
        expect(culler.meshletCull.indirectSlotIndex).to.equal(culler.dispatchArgs.getParameter('dispatchSlot'));
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
        expect(log.dispatches[1].name).to.equal('MeshletCullPhase2');
        expect(log.dispatches[1].computes[0], 'counters reset first').to.equal('MeshletresetPhase2');
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
        expect(log.dispatches.map(d => d.name)).to.deep.equal(['MeshletCullPhase1a', 'MeshletCullPhase1b', 'MeshletCullPhase1c']);
        expect(log.submits).to.equal(2);
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
        culler.bindRecordState(view);
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
