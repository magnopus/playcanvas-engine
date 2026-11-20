import { expect } from 'chai';

import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
import {
    CULL_FLAG_NO_TEXEL_RATE, CULL_PARAMS, MESHLET_BUCKET_COUNT, MESHLET_CELL_CULL_WORKGROUP, MESHLET_COUNTER,
    MESHLET_COUNTER_CLEAR_U32S, MESHLET_COUNTER_U32S, MESHLET_DISPATCH_WIDTH, MESHLET_LIST_WORKGROUP, MESHLET_ROOT_BANDS, MESHLET_ROOT_BAND_MAX_PIXELS, MESHLET_ROOT_BAND_OCTAVES, claimPersistBase
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
    device.getIndirectDispatchSlot = (count = 1) => {
        const slot = log.dispatchSlot;
        log.dispatchSlot += count;
        return slot;
    };
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
    cut: {
        groups: storage(),
        selectionTopology: storage(),
        stageLevels: [1]
    },
    cells: { cellCount: 3, boundsBuffer: storage(), radiiBuffer: storage(), instancesBuffer: storage() },
    totalPairs: 1000,
    instanceCount: 130,
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
    rootListCapacity: 130,
    indexCapacity: [10, 20, 30],
    countersBuffer: storage(MESHLET_COUNTER_U32S * 4),
    claimBitsBuffer: storage(64),
    rootListBuffer: storage(),
    cutQueueBuffer: storage(),
    cutQueueCapacity: 1000,
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

// the cull parameters as each beginFrame left them (the culler reuses one array)
const recordCullParams = (culler) => {
    const frames = [];
    const beginFrame = culler.beginFrame.bind(culler);
    culler.beginFrame = (...args) => {
        beginFrame(...args);
        frames.push(culler.cullParams.slice());
    };
    return frames;
};

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

    it('sets every uniform, storage buffer and texture each compute shader declares', function () {
        // a name the shader declares but the culler never sets (or sets under another name, such
        // as a uniform array's `name[0]`) reaches the GPU unset
        const view = makeView();
        const culler = new MeshletCuller(device, makeWorld(), view, shaders);
        culler.twoPhase = true;
        culler.beginFrame(planes, camera, 640, viewProj, { texture: new Texture(device, { width: 4, height: 4 }), width: 4, height: 4, mipCount: 3, valid: true });
        culler.dispatchPhase2();
        for (const compute of culler._computes()) {
            const code = compute.shader.definition.cshader;
            const names = [
                ...Array.from(code.matchAll(/^\s*uniform (\w+)\s*:\s*(array<)?/gm), ([, name, array]) => (array ? `${name}[0]` : name)),
                ...Array.from(code.matchAll(/var<storage[^>]*>\s*(\w+)/g), ([, name]) => name),
                ...Array.from(code.matchAll(/^\s*var (\w+)\s*:\s*texture/gm), ([, name]) => name)
            ];
            for (const name of names) {
                expect(compute.getParameter(name), `${compute.name}: ${name}`).to.not.equal(undefined);
            }
        }
        culler.destroy();
    });

    it('binds identical world and view buffers to the phase-1 and phase-2 instances of each stage', function () {
        const view = makeView();
        const world = makeWorld();
        const culler = new MeshletCuller(device, world, view, shaders);
        expect(culler.cutEmit.getParameter('selectionTopology')).to.equal(world.cut.selectionTopology);
        for (const name of ['objectData', 'meshletData', 'cullParams[0]', 'selectedMeshlets', 'counters', 'records', 'requests', 'visBits']) {
            expect(culler.meshletCull.getParameter(name), name).to.equal(culler.meshletCullPhase2.getParameter(name));
        }
        expect(culler.meshletCullPhase2.getParameter('phase')).to.equal(2);
        expect(culler.cutEmit.getParameter('claimBits')).to.equal(view.claimBitsBuffer);
        expect(culler.cutEmit.getParameter('selectedMeshlets')).to.equal(view.selectedMeshletsBuffer);
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
        expect(culler.queueStages, 'the root level is admitted, not cut').to.have.lengthOf(1);
        expect(culler.queueStages[0].compute.getParameter('stageLevel')).to.equal(1);
        expect(culler.rootAdmit).to.have.lengthOf(MESHLET_ROOT_BANDS);
        culler.rootAdmit.forEach((compute, band) => {
            expect(compute.getParameter('rootBand')).to.equal(band);
            expect(compute.getParameter('rootList')).to.equal(view.rootListBuffer);
            for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
                expect(compute.getParameter(`indexCapacity${b}`)).to.equal(view.indexCapacity[b]);
            }
        });
        expect(culler.rootClassify.getParameter('rootList')).to.equal(view.rootListBuffer);
        // classify fills the list's first half, the scatter sorts it into the second
        expect(culler.rootClassify.getParameter('sortedBase')).to.equal(world.instanceCount);
        expect(culler.rootScatter.getParameter('sortedBase')).to.equal(world.instanceCount);
        expect(culler.rootArgs.getParameter('counterIndex')).to.equal(MESHLET_COUNTER.ROOT_CANDIDATES);
        expect(culler.rootArgs.getParameter('groupSize')).to.equal(MESHLET_LIST_WORKGROUP);
        expect(culler.cutEmitArgs.getParameter('groupSize')).to.equal(MESHLET_LIST_WORKGROUP);

        device.gpuProfiler = { enabled: true, passDetail: true, destroy() {} };
        culler.beginFrame(planes, camera, 640, viewProj, null);
        expect(log.dispatches[0].name).to.equal('MeshletCellCull');
        expect(log.dispatches[0].computes).to.deep.equal(['MeshletcellCull', 'MeshletdispatchArgs']);
        // the per-instance passes run one workgroup per visible cell, the cut stage from its queue args
        const cellSlot = culler.cellArgs.getParameter('dispatchSlot');
        expect(culler.rootClassify.indirectSlotIndex).to.equal(cellSlot);
        expect(culler.queueStages[0].compute.indirectSlotIndex).to.equal(culler.queueStages[0].args.getParameter('dispatchSlot'));
        expect(log.dispatches[1].name).to.equal('MeshletRootClassify');
        expect(log.dispatches[1].computes).to.deep.equal(['MeshletrootClassify', 'MeshletdispatchArgs',
            'MeshletrootBandArgs', 'MeshletrootScatter']);
        expect(log.dispatches[2].name).to.equal('MeshletRootAdmit');
        expect(log.dispatches[2].computes).to.deep.equal(Array.from({ length: MESHLET_ROOT_BANDS }, (_, band) => `MeshletRootAdmit${band}`));
        // the scatter walks every candidate; each band pass only its own band's range
        expect(culler.rootScatter.indirectSlotIndex).to.equal(culler.rootArgs.getParameter('dispatchSlot'));
        const firstBandSlot = culler.rootBandArgs.getParameter('firstSlot');
        culler.rootAdmit.forEach((compute, band) => {
            expect(compute.indirectSlotIndex).to.equal(firstBandSlot + band);
            expect(compute.getParameter('sortedBase')).to.equal(130);
        });
        expect(culler.rootScatter.getParameter('sortedBase')).to.equal(130);
        culler.destroy();
    });

    it('rejects a root list buffer too small for the instances', function () {
        const view = makeView();
        view.rootListCapacity = 10;
        expect(() => new MeshletCuller(device, makeWorld(), view, shaders)).to.throw(/root list/);
    });

    it('writes the root priority rows, from the view camera or a given priority origin', function () {
        const view = makeView();
        const culler = new MeshletCuller(device, makeWorld(), view, shaders);
        const frames = recordCullParams(culler);
        culler.beginFrame(planes, camera, 640, viewProj, null);
        const row = index => index * 4;
        let params = frames[0];
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
        params = frames[1];
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
        const frames = recordCullParams(culler);
        culler.dagPixelThreshold = 2;
        culler.pressureScale = 1.5;
        culler.orthoScale = 7;
        culler.cullFlags = CULL_FLAG_NO_TEXEL_RATE;
        culler.viewDir = { x: 0, y: 1, z: 0 };
        culler.rootMinPixels = 3;
        culler.bucketThresholdScales = [4, 1, 2];
        culler.bucketRootMinScales = [2, 1, 1];
        culler.rootHysteresis = 0.75;
        culler.beginFrame(planes, camera, 640, viewProj, null);

        const params = frames[0];
        const row = index => index * 4;
        expect(Array.from(params.subarray(row(CULL_PARAMS.PLANES), row(CULL_PARAMS.PLANES) + 24))).to.deep.equal(Array.from(planes));
        expect(Array.from(params.subarray(row(CULL_PARAMS.CAMERA), row(CULL_PARAMS.CAMERA) + 4))).to.deep.equal([1, 2, 3, 640]);
        expect(Array.from(params.subarray(row(CULL_PARAMS.LOD), row(CULL_PARAMS.LOD) + 4)), 'threshold x pressure, dummy HZB').to.deep.equal([3, 1, 1, 1]);
        expect(Array.from(params.subarray(row(CULL_PARAMS.VIEW_PROJ), row(CULL_PARAMS.VIEW_PROJ) + 16))).to.deep.equal(Array.from(viewProj));
        expect(Array.from(params.subarray(row(CULL_PARAMS.STREAMING), row(CULL_PARAMS.STREAMING) + 3))).to.deep.equal([12, 7, CULL_FLAG_NO_TEXEL_RATE]);
        expect(Array.from(params.subarray(row(CULL_PARAMS.VIEW_DIR), row(CULL_PARAMS.VIEW_DIR) + 3))).to.deep.equal([0, 1, 0]);
        // per draw bucket: the scaled thresholds and their smallest, the scaled size cuts and the hysteresis
        expect(Array.from(params.subarray(row(CULL_PARAMS.BUCKET_LOD), row(CULL_PARAMS.BUCKET_LOD) + 4))).to.deep.equal([12, 3, 6, 3]);
        expect(Array.from(params.subarray(row(CULL_PARAMS.BUCKET_ROOT), row(CULL_PARAMS.BUCKET_ROOT) + 4))).to.deep.equal([6, 3, 3, 0.75]);
        expect(params[row(CULL_PARAMS.ROOT_BANDS) + 3], 'the cell cull uses the smallest size cut').to.equal(3);
        culler.destroy();
    });

    it('arms root admission\'s occlusion test only against the pyramid and camera of the frame before', function () {
        const view = makeView();
        const culler = new MeshletCuller(device, makeWorld(), view, shaders);
        const frames = recordCullParams(culler);
        const row = index => index * 4;
        const hzb = { texture: {}, width: 256, height: 128, mipCount: 8, valid: true };
        culler.twoPhase = true;
        culler.viewDir = { x: 0, y: 0, z: -1 };
        const armed = (frame) => {
            const params = frames[frame];
            return params[row(CULL_PARAMS.PREV_CAMERA) + 3];
        };

        culler.beginFrame(planes, camera, 640, viewProj, hzb);
        expect(armed(0), 'no frame before this one').to.equal(0);

        culler.viewDir = { x: 1, y: 0, z: 0 };
        const next = new Float32Array(viewProj).fill(2);
        culler.beginFrame(planes, { x: 4, y: 5, z: 6 }, 640, next, hzb);
        expect(armed(1), 'the same pyramid one frame on').to.equal(1);
        const params = frames[1];
        expect(Array.from(params.subarray(row(CULL_PARAMS.PREV_VIEW_PROJ), row(CULL_PARAMS.PREV_VIEW_PROJ) + 16))).to.deep.equal(Array.from(viewProj));
        expect(Array.from(params.subarray(row(CULL_PARAMS.PREV_CAMERA), row(CULL_PARAMS.PREV_CAMERA) + 3)), 'the camera that built it').to.deep.equal([camera.x, camera.y, camera.z]);
        expect(Array.from(params.subarray(row(CULL_PARAMS.PREV_VIEW_DIR), row(CULL_PARAMS.PREV_VIEW_DIR) + 3))).to.deep.equal([0, 0, -1]);

        // the frame after sees the second frame's camera
        culler.beginFrame(planes, camera, 640, viewProj, hzb);
        expect(Array.from(frames[2].subarray(row(CULL_PARAMS.PREV_CAMERA), row(CULL_PARAMS.PREV_CAMERA) + 3))).to.deep.equal([4, 5, 6]);

        // a reallocated or resized pyramid holds nothing to test against
        culler.beginFrame(planes, camera, 640, viewProj, { ...hzb, texture: {} });
        expect(armed(3), 'a new texture').to.equal(0);
        culler.beginFrame(planes, camera, 640, viewProj, { ...hzb, texture: hzb.texture, width: 512 });
        expect(armed(4), 'the texture changed back, and the size').to.equal(0);

        culler.beginFrame(planes, camera, 640, viewProj, { ...hzb, texture: hzb.texture, width: 512 });
        expect(armed(5), 'a resized pyramid, once a frame has built it').to.equal(1);
        // recreated, but its mip 0 not yet built: zeros, which would hide everything
        culler.beginFrame(planes, camera, 640, viewProj, { ...hzb, texture: hzb.texture, width: 512, valid: false });
        expect(armed(6), 'not built yet').to.equal(0);
        culler.occlusionAdmission = false;
        culler.beginFrame(planes, camera, 640, viewProj, { ...hzb, texture: hzb.texture, width: 512 });
        expect(armed(7), 'switched off').to.equal(0);
        culler.destroy();
    });

    it('runs single-phase while the HZB holds no built pyramid', function () {
        const culler = new MeshletCuller(device, makeWorld(), makeView(), shaders);
        culler.twoPhase = true;
        const hzb = { texture: new Texture(device, { width: 4, height: 4 }), width: 320, height: 200, mipCount: 9, valid: false };
        culler.beginFrame(planes, camera, 640, viewProj, hzb);
        expect(culler.meshletCull.getParameter('phase')).to.equal(0);
        expect(culler.meshletCull.getParameter('hzbTexture')).to.equal(shaders.dummyHzb);
        hzb.valid = true;
        culler.beginFrame(planes, camera, 640, viewProj, hzb);
        expect(culler.meshletCull.getParameter('phase')).to.equal(1);
        expect(culler.meshletCull.getParameter('hzbTexture')).to.equal(hzb.texture);
        hzb.texture.destroy();
        culler.destroy();
    });

    it('dispatches the cell cull over a grid no wider than the dispatch width', function () {
        const world = makeWorld();
        world.cells.cellCount = (MESHLET_DISPATCH_WIDTH + 5) * MESHLET_CELL_CULL_WORKGROUP;
        const culler = new MeshletCuller(device, world, makeView(), shaders);
        expect([culler.cellCull.countX, culler.cellCull.countY]).to.deep.equal([MESHLET_DISPATCH_WIDTH, 2]);
        culler.destroy();
    });

    it('dispatches the whole phase-1 chain in one compute pass when the GPU profiler is off', function () {
        const culler = new MeshletCuller(device, makeWorld(), makeView(), shaders);
        culler.passSuffix = 'Shadow0';
        culler.beginFrame(planes, camera, 640, viewProj, null);
        expect(log.dispatches.map(d => d.name)).to.deep.equal(['MeshletCullShadow0']);
        expect(log.dispatches[0].computes).to.deep.equal([
            'MeshletcellCull', 'MeshletdispatchArgs',
            'MeshletrootClassify', 'MeshletdispatchArgs', 'MeshletrootBandArgs', 'MeshletrootScatter',
            ...Array.from({ length: MESHLET_ROOT_BANDS }, (_, band) => `MeshletRootAdmit${band}`),
            'MeshletcutQueueArgs', 'MeshletCutQueue0',
            'MeshletdispatchArgs', 'MeshletcutEmit', 'MeshletdispatchArgs',
            'MeshletmeshletCull', 'MeshletfinalizeArgs',
            'MeshletindexWrite'
        ]);
        culler.destroy();
    });

    it('keeps one compute pass while the profiler only measures the frame total', function () {
        const culler = new MeshletCuller(device, makeWorld(), makeView(), shaders);
        device.gpuProfiler = { enabled: true, passDetail: false, destroy() {} };
        culler.beginFrame(planes, camera, 640, viewProj, null);
        expect(log.dispatches.map(d => d.name)).to.deep.equal(['MeshletCull']);
        culler.destroy();
    });

    it('destroys every compute it owns, keeping the shared shaders', function () {
        const culler = new MeshletCuller(device, makeWorld(), makeView(), shaders);
        const computes = [culler.cellCull, culler.rootClassify, ...culler.rootAdmit, culler.queueStages[0].args,
            culler.queueStages[0].compute, culler.meshletCull, culler.meshletCullPhase2, culler.indexWritePhase2, culler.resetPhase2];
        culler.destroy();
        for (const compute of computes) expect(compute.impl).to.equal(null);
        expect(shaders.cellCull, 'shared shaders survive').to.not.equal(null);
    });

    it('clears the counters and claim bits in-encoder and dispatches the phase-1 chain in order', function () {
        const view = makeView();
        const culler = new MeshletCuller(device, makeWorld(), view, shaders);
        device.gpuProfiler = { enabled: true, passDetail: true, destroy() {} };
        culler.beginFrame(planes, camera, 640, viewProj, null);

        const base = claimPersistBase(1000, 130);
        expect(log.clears, 'counters, transient claim bits, this frame\'s admissions; never the requests buffer')
        .to.deep.equal([MESHLET_COUNTER_CLEAR_U32S * 4, 3 * MESHLET_ROOT_BANDS * 4, base * 4, Math.ceil(130 / 32) * 4]);
        // each stage profiles as its own pass
        expect(log.dispatches.map(d => d.name)).to.deep.equal([
            'MeshletCellCull', 'MeshletRootClassify', 'MeshletRootAdmit', 'MeshletResidentCut',
            'MeshletCutEmit', 'MeshletCullPhase1', 'MeshletIndexWrite'
        ]);
        // the queued cut: per stage, an args pass sizing it from the queue, then the stage
        expect(log.dispatches[0].computes).to.deep.equal(['MeshletcellCull', 'MeshletdispatchArgs']);
        expect(log.dispatches[3].computes).to.deep.equal(['MeshletcutQueueArgs', 'MeshletCutQueue0']);
        // the selected cut straight from the queue, one thread per entry
        expect(log.dispatches[4].computes).to.deep.equal(['MeshletdispatchArgs', 'MeshletcutEmit', 'MeshletdispatchArgs']);
        expect(log.dispatches[5].computes).to.deep.equal(['MeshletmeshletCull', 'MeshletfinalizeArgs']);
        expect(log.dispatches[6].computes).to.deep.equal(['MeshletindexWrite']);
        expect(culler.cellCull.countX, 'one thread per cell').to.equal(1);
        expect(culler.meshletCull.getParameter('phase'), 'no HZB: single phase').to.equal(0);
        expect(culler.meshletCull.getParameter('hzbTexture')).to.equal(shaders.dummyHzb);
        // one draw slot per bucket per phase, phase-1 slots handed to the view's mesh instances
        expect(log.drawSlot).to.equal(2 * MESHLET_BUCKET_COUNT);
        expect(view.meshInstances.map(mi => mi.slot)).to.deep.equal([0, 1, 2, 3, 4, 5]);
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            expect(culler.finalizeArgs.getParameter(`drawSlot${b}`)).to.equal(b);
        }
        expect(culler.cutEmit.indirectSlotIndex).to.equal(culler.cutEmitArgs.getParameter('dispatchSlot'));
        expect(culler.meshletCull.indirectSlotIndex).to.equal(culler.selectedArgs.getParameter('dispatchSlot'));
        expect(culler.indexWrite.indirectSlotIndex).to.equal(culler.finalizeArgs.getParameter('dispatchSlot'));
        culler.destroy();
    });

    it('runs two-phase against the HZB texture and gives phase 2 its own draw slots', function () {
        const view = makeView();
        const culler = new MeshletCuller(device, makeWorld(), view, shaders);
        const frames = recordCullParams(culler);
        culler.twoPhase = true;
        const hzb = { texture: new Texture(device, { width: 4, height: 4 }), width: 320, height: 200, mipCount: 9, valid: true };
        culler.beginFrame(planes, camera, 640, viewProj, hzb);
        expect(culler.meshletCull.getParameter('phase')).to.equal(1);
        expect(culler.meshletCull.getParameter('hzbTexture')).to.equal(hzb.texture);
        expect(Array.from(frames[0].subarray(CULL_PARAMS.LOD * 4 + 1, CULL_PARAMS.LOD * 4 + 4))).to.deep.equal([320, 200, 9]);

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
        for (const compute of [culler.cutEmit, culler.meshletCull, culler.meshletCullPhase2]) {
            expect(compute.getParameter('selectedMeshlets')).to.equal(view.selectedMeshletsBuffer);
            expect(compute.getParameter('recordCapacity')).to.equal(500);
        }
        expect(culler.selectedArgs.getParameter('itemCapacity')).to.equal(500);
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
        expect(hzb.valid, 'nothing built yet').to.equal(false);
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
