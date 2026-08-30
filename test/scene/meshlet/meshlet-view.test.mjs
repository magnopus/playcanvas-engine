import { expect } from 'chai';

import { Vec3 } from '../../../src/core/math/vec3.js';
import { BoundingBox } from '../../../src/core/shape/bounding-box.js';
import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import { ShaderMaterial } from '../../../src/scene/materials/shader-material.js';
import {
    CULL_PARAMS_VEC4S, MESHLET_BUCKET_COUNT, MESHLET_COUNTER, MESHLET_COUNTER_U32S, RECORD_U32S, WORK_ITEM_U32S
} from '../../../src/scene/meshlet/constants.js';
import { MeshletCullShaders } from '../../../src/scene/meshlet/meshlet-cull-shaders.js';
import { MeshletShadowView } from '../../../src/scene/meshlet/meshlet-shadow-view.js';
import { MeshletView } from '../../../src/scene/meshlet/meshlet-view.js';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

// The view is exercised on the null device dressed up as compute-capable (see
// meshlet-culler.test.mjs); storage buffers record their writes and the command encoder
// records buffer copies.
const makeDevice = () => {
    const device = new NullGraphicsDevice(document.createElement('canvas'));
    device.supportsCompute = true;
    device.createComputeImpl = () => ({ destroy() {} });
    const log = { copies: [], destroyed: [] };
    device.createBufferImpl = () => ({
        allocate() {}, write() {}, destroy() {}, loseContext() {}, buffer: {}
    });
    device.getCommandEncoder = () => ({
        copyBufferToBuffer: (src, srcOffset, dst, dstOffset, size) => log.copies.push({ src, dst, size }),
        clearBuffer() {}
    });
    device.computeDispatch = () => {};
    device.getIndirectDrawSlot = () => 0;
    device.getIndirectDispatchSlot = () => 0;
    return { device, log };
};

const PAIRS = 1000;
const materials = () => Array.from({ length: MESHLET_BUCKET_COUNT }, () => new ShaderMaterial());

const makeWorld = (device, overrides = {}) => {
    const lit = materials();
    return {
        totalPairs: PAIRS,
        workItemCapacity: 50,
        recordCapacity: 4000,
        initialRecords: 100,
        instanceCount: 7,
        pageSizeBytes: 65536,
        totalPages: 3,
        indexCapacity: [300, 60, 30],
        indexWorst: [3000, 600, 300],
        indexCeiling: 100000,
        maxIndices: 0,
        dummyBits: { destroy() {} },
        objectDataBuffer: {},
        meshletDataBuffer: {},
        residencyBuffer: {},
        requestsBuffer: {},
        pagePool: {},
        litMaterials: lit,
        bucketMaterials: lit,
        ...overrides
    };
};

const counters = (indices, records) => {
    const data = new Uint32Array(MESHLET_COUNTER_U32S);
    data[MESHLET_COUNTER.RECORDS] = records;
    indices.forEach((n, b) => {
        data[MESHLET_COUNTER.DEMAND_BASE + b] = n;
    });
    return data;
};

describe('MeshletView', function () {

    let device;
    let log;
    let shaders;
    let world;

    beforeEach(function () {
        jsdomSetup();
        ({ device, log } = makeDevice());
        shaders = new MeshletCullShaders(device);
        world = makeWorld(device);
    });

    afterEach(function () {
        shaders.destroy();
        device.destroy();
        device = null;
        jsdomTeardown();
    });

    const makeView = (options = {}, w = world) => new MeshletView(device, w, null, null, { cullShaders: shaders, ...options });

    it('sizes its frame buffers from the layout constants and the world', function () {
        const view = makeView();
        const pairWords = Math.ceil(PAIRS / 32);
        expect(view.cullParamsBuffer.byteSize).to.equal(CULL_PARAMS_VEC4S * 16);
        expect(view.countersBuffer.byteSize).to.equal(MESHLET_COUNTER_U32S * 4);
        expect(view.workItemsBuffer.byteSize).to.equal(50 * WORK_ITEM_U32S * 4);
        expect(view.recordCapacity, 'starts at the initial record count').to.equal(100);
        expect(view.recordsBuffer.byteSize).to.equal(100 * RECORD_U32S * 4);
        expect(view.claimBitsBuffer.byteSize).to.equal(pairWords * 4);
        expect(view.visBitsBuffer.byteSize).to.equal(pairWords * 4);
        expect(view.indexCapacity).to.deep.equal([300, 60, 30]);
        expect(view.indexBuffer.numIndices).to.equal(390);
        expect(view.meshInstances, 'three buckets, two phases').to.have.lengthOf(2 * MESHLET_BUCKET_COUNT);
        expect(view.meshInstances.map(mi => mi.material)).to.deep.equal([...world.litMaterials, ...world.litMaterials]);
        expect(view.meshInstances.every(mi => mi.getParameter('records').data === view.recordsBuffer)).to.equal(true);
        expect(view.meshInstances.every(mi => !mi.castShadow && !mi.cull)).to.equal(true);
        expect(view.drawPass1.name).to.equal('MeshletDrawPhase1');
        expect(view.cullPhase2Pass.name).to.equal('MeshletCullPhase2');
        view.destroy();
    });

    it('starts a single-phase view on the shared placeholders with one set of instances', function () {
        const shared = { claim: { byteSize: 8, destroy() {} }, work: { byteSize: 8, destroy() {} } };
        const view = makeView({
            singlePhase: true, castShadow: true, sharedClaimBits: shared.claim, sharedWorkItems: shared.work, initialIndexScale: 0.5
        });
        expect(view.visBitsBuffer).to.equal(world.dummyBits);
        expect(view.claimBitsBuffer).to.equal(shared.claim);
        expect(view.workItemsBuffer).to.equal(shared.work);
        expect(view.indexCapacity).to.deep.equal([150, 30, 15]);
        expect(view.meshInstances).to.have.lengthOf(MESHLET_BUCKET_COUNT);
        expect(view.meshInstances.every(mi => mi.castShadow && mi.cull), 'casters keep cull on').to.equal(true);
        expect(view.drawPass1).to.equal(undefined);
        let destroyed = 0;
        shared.claim.destroy = shared.work.destroy = () => destroyed++;
        view.destroy();
        expect(destroyed, 'borrowed buffers survive the view').to.equal(0);
    });

    it('carries capacities and visibility bits across a rebuild, capped at the worst case', function () {
        const carry = { buffer: { byteSize: 40, impl: { buffer: 'old-bits' } }, words: 20 };
        const view = makeView({ capCarry: [5000, 10, 100], visCarry: carry });
        expect(view.indexCapacity, 'max(initial, carry) then min(worst)').to.deep.equal([3000, 60, 100]);
        expect(log.copies).to.have.lengthOf(1);
        expect(log.copies[0].src).to.equal('old-bits');
        expect(log.copies[0].size, 'min(carry words, own words, carry bytes / 4) * 4').to.equal(10 * 4);
        view.destroy();
    });

    it('scales the capacities down to the ceiling, preferring the view share when set', function () {
        const view = makeView({}, makeWorld(device, { indexCeiling: 195 }));
        expect(view.indexCapacity, 'half of every bucket').to.deep.equal([150, 30, 15]);
        view.indexShare = 39;
        const capacity = [100, 50, 45];
        expect(view._clampToCeiling(capacity)).to.equal(true);
        expect(capacity).to.deep.equal([20, 10, 9]);
        expect(view._clampToCeiling([1, 1, 1])).to.equal(false);
        view.destroy();
    });

    it('grows the index buffer to the demand, honouring maxIndices and the worst case', function () {
        const view = makeView();
        const first = view.indexBuffer;
        expect(view.growIndexBuffer([100, 10, 10]), 'nothing above the current capacity').to.equal(false);
        expect(view.indexBuffer).to.equal(first);
        expect(view.growIndexBuffer([600, 10, 10])).to.equal(true);
        expect(view.indexCapacity).to.deep.equal([600, 60, 30]);
        expect(view.indexBuffer).to.not.equal(first);
        expect(view._mesh.indexBuffer[0]).to.equal(view.indexBuffer);
        expect(view.growIndexBuffer([9999, 9999, 9999]), 'worst case caps every bucket').to.equal(true);
        expect(view.indexCapacity).to.deep.equal([3000, 600, 300]);

        world.maxIndices = 1950;
        const capped = makeView();
        capped.growIndexBuffer([3000, 600, 300]);
        expect(capped.indexCapacity, 'scaled to maxIndices').to.deep.equal([1500, 300, 150]);
        capped.destroy();
        view.destroy();
    });

    it('shrinks only after a sustained low, never below the initial allocation', function () {
        const view = makeView();
        view.growIndexBuffer([3000, 600, 300]);
        view.shrinkFrames = 3;
        const demand = [100, 10, 5];
        const target = [160, 16, 8];
        expect(view.shrinkIndexBuffer(demand, target)).to.equal(false);
        expect(view.shrinkIndexBuffer(demand, target)).to.equal(false);
        expect(view.shrinkIndexBuffer([3000, 0, 0], target), 'demand back up resets the streak').to.equal(false);
        expect(view.shrinkIndexBuffer(demand, target)).to.equal(false);
        expect(view.shrinkIndexBuffer(demand, target)).to.equal(false);
        expect(view.shrinkIndexBuffer(demand, target)).to.equal(true);
        expect(view.indexCapacity, 'max(target, world initial)').to.deep.equal([300, 60, 30]);
        view.destroy();
    });

    it('records demand from the readback and applies growth only at the top of a frame', async function () {
        const view = makeView();
        const bound = { index: 0, record: 0 };
        view.culler.bindIndexState = () => bound.index++;
        view.culler.bindRecordState = () => bound.record++;
        let resolveRead;
        view.readbackPool.read = (buffer, offset, size, data) => {
            expect(buffer).to.equal(view.countersBuffer);
            expect(size).to.equal(MESHLET_COUNTER_U32S * 4);
            return new Promise((resolve) => {
                resolveRead = () => resolve(counters([250, 10, 10], 90));
            });
        };
        view.monitorIndexDemand();
        view.monitorIndexDemand();
        expect(view._countersReadBusy, 'one readback in flight at a time').to.equal(true);
        const before = view.indexBuffer;
        resolveRead();
        await Promise.resolve();
        expect(view.lastDemand).to.deep.equal({ indices: [250, 10, 10], records: 90 });
        expect(view.indexBuffer, 'no swap from the callback').to.equal(before);
        expect(view.indexDemand()).to.equal(270);
        expect(view.indexPressure()).to.be.closeTo(270 / 100000, 1e-9);

        view.applyPendingGrowth();
        expect(view.indexCapacity, 'bucket 0 above growAt: grows to demand * growTo').to.deep.equal([400, 60, 30]);
        expect(bound.index).to.equal(1);
        expect(view.recordCapacity, 'records above growAt too').to.equal(144);
        expect(bound.record).to.equal(1);
        view.applyPendingGrowth();
        expect(bound.index, 'demand consumed').to.equal(1);
        view.destroy();
    });

    it('swaps the bucket materials only when the world changes them', function () {
        const view = makeView();
        const debug = materials();
        world.bucketMaterials = debug;
        view.syncMaterials();
        expect(view.meshInstances.map(mi => mi.material)).to.deep.equal([...debug, ...debug]);
        view.destroy();
    });
});

describe('MeshletShadowView', function () {

    let device;
    let shaders;
    let world;

    beforeEach(function () {
        jsdomSetup();
        ({ device } = makeDevice());
        shaders = new MeshletCullShaders(device);
        world = makeWorld(device);
    });

    afterEach(function () {
        shaders.destroy();
        device.destroy();
        device = null;
        jsdomTeardown();
    });

    it('is a single-phase caster view routed to its own shadow camera and always lit', function () {
        const shadowCamera = {};
        const sceneCamera = {};
        const renderData = { shadowCamera };
        const light = { getRenderData: (camera, face) => ((camera === sceneCamera && face === 1) ? renderData : null) };
        world.bucketMaterials = materials(); // debug colour mode
        const view = new MeshletShadowView(device, world, null, light, 1, sceneCamera, { cullShaders: shaders });
        expect(view.singlePhase).to.equal(true);
        expect(view.meshInstances).to.have.lengthOf(MESHLET_BUCKET_COUNT);
        expect(view.meshInstances.map(mi => mi.material), 'lit, never the debug set').to.deep.equal(world.litMaterials);
        expect(view.shadowCamera).to.equal(shadowCamera);
        expect(view.meshInstances[0].isVisibleFunc(shadowCamera)).to.equal(true);
        expect(view.meshInstances[0].isVisibleFunc({})).to.equal(false);

        const aabb = new BoundingBox(new Vec3(1, 2, 3), new Vec3(4, 5, 6));
        view.setCasterBounds(aabb);
        expect(view.meshInstances.every(mi => mi._customAabb.center.equals(aabb.center) && mi._customAabb.halfExtents.equals(aabb.halfExtents))).to.equal(true);
        view.destroy();
    });
});
