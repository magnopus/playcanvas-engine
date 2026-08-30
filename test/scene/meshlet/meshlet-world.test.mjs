import { expect } from 'chai';

import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import {
    MATERIAL_FLAG_ALPHA_MASK, MATERIAL_FLAG_DOUBLE_SIDED, MATERIAL_RECORD, MATERIAL_RECORD_U32S, MATERIAL_SLOT_ABSENT,
    MESHLET_BUCKET_MASKED, MESHLET_BUCKET_OPAQUE, MESHLET_BUCKET_OPAQUE_TWO_SIDED, MESHLET_COLOR_MODE, MESHLET_DATA,
    MESHLET_DATA_U32S, MESHLET_FLAG_ALPHA_MASKED, MESHLET_FLAG_TWO_SIDED, OBJECT_DATA, OBJECT_DATA_U32S,
    OBJECT_FLAG_HIDDEN, OBJECT_FLAG_OUTLINED, PAGE_TABLE, PAGE_TABLE_FIELDS
} from '../../../src/scene/meshlet/constants.js';
import { MeshletPrimitive, MeshletResource } from '../../../src/scene/meshlet/meshlet-resource.js';
import { MeshletWorld } from '../../../src/scene/meshlet/meshlet-world.js';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

const PAGE_BYTES = 256;
const MESHLETS = 3;
const TRIANGLES = 10;

// a resident resource of one primitive, MESHLETS meshlets of TRIANGLES triangles spread over two
// pages, with an optional baked material record whose flags select the draw bucket
const makeResource = (device, { materialFlags = null, instances = 2, scale = 1 } = {}) => {
    const prim = new MeshletPrimitive();
    prim.meshletData = new Uint32Array(MESHLETS * MESHLET_DATA_U32S);
    for (let m = 0; m < MESHLETS; m++) {
        prim.meshletData[m * MESHLET_DATA_U32S + MESHLET_DATA.TRIANGLE_COUNT] = TRIANGLES;
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
        matrix[0] = scale; matrix[5] = scale; matrix[10] = scale; matrix[15] = 1;
        matrix[12] = i * 10;   // translate each instance along x
        return { primIndex: 0, matrix };
    });
    const resource = new MeshletResource(device, [prim], manifest, placements);
    if (materialFlags !== null) {
        resource.materialTable = new Uint32Array(MATERIAL_RECORD_U32S);
        resource.materialTable[MATERIAL_RECORD.FLAGS] = materialFlags;
        resource.materialCount = 1;
    }
    return { resource, shards: [new ArrayBuffer(2 * PAGE_BYTES)] };
};

describe('MeshletWorld', function () {

    /** @type {NullGraphicsDevice} */
    let device;
    let savedWarn;

    beforeEach(function () {
        jsdomSetup();
        device = new NullGraphicsDevice(document.createElement('canvas'));
        // the null device has no storage buffers; record what the world uploads
        device.createBufferImpl = () => ({
            writes: [],
            allocate(dev, size) {
                this.size = size;
            },
            write(dev, offset, data, dataOffset = 0, size) {
                this.writes.push({ offset, data: data.slice(dataOffset, size === undefined ? undefined : dataOffset + size) });
            },
            read() {
                return Promise.resolve();
            },
            clear() {},
            destroy() {},
            loseContext() {},
            buffer: null
        });
        savedWarn = console.warn;
        console.warn = () => {};
    });

    afterEach(function () {
        console.warn = savedWarn;
        device.destroy();
        device = null;
        jsdomTeardown();
    });

    const build = (world, ...entries) => {
        for (const { resource, shards } of entries) world.addResource(resource, null, shards);
        world.finalize();
        return world;
    };
    const uploaded = buffer => buffer.impl.writes.find(w => w.offset === 0).data;

    describe('finalize', function () {

        it('writes one objectData row per placement by field name', function () {
            const world = build(new MeshletWorld(device), makeResource(device, { instances: 2, scale: 2 }));
            expect(world.instanceCount).to.equal(2);
            expect(world.totalPairs).to.equal(2 * MESHLETS);

            const u32 = world.objectDataCpu;
            const f32 = world.objectDataCpuF;
            const row = 1 * OBJECT_DATA_U32S;
            expect(f32[row + OBJECT_DATA.MATRIX + 12], 'translation of the second instance').to.equal(10);
            expect(Array.from(f32.subarray(row + OBJECT_DATA.SPHERE, row + OBJECT_DATA.SPHERE + 3))).to.deep.equal([1, 2, 3]);
            expect(f32[row + OBJECT_DATA.SPHERE + 3]).to.be.closeTo(Math.sqrt(3), 1e-6);
            expect(u32[row + OBJECT_DATA.FIRST_MESHLET]).to.equal(0);
            expect(u32[row + OBJECT_DATA.MESHLET_COUNT]).to.equal(MESHLETS);
            expect(u32[row + OBJECT_DATA.MATERIAL]).to.equal(0);
            expect(u32[row + OBJECT_DATA.FLAGS]).to.equal(0);
            expect(f32[row + OBJECT_DATA.MAX_SCALE]).to.equal(2);
            expect(u32[row + OBJECT_DATA.FIRST_PAIR_BIT], 'second instance starts after the first one\'s pairs').to.equal(MESHLETS);
            expect(f32[row + OBJECT_DATA.GRID_ORIGIN]).to.equal(0.5);
            expect(f32[row + OBJECT_DATA.GRID_STEP]).to.be.closeTo(0.01, 1e-9);
            expect(u32[row + OBJECT_DATA.UV_FLOATS_PER_VERTEX]).to.equal(2);
            expect(u32[row + OBJECT_DATA.PICK_ID]).to.equal(world.pickIdBase + 1);
            expect(uploaded(world.objectDataBuffer)).to.have.lengthOf(2 * OBJECT_DATA_U32S);
            world.destroy();
        });

        it('rebases page indices of later resources and totals the capacities', function () {
            const world = build(new MeshletWorld(device), makeResource(device, { instances: 1 }), makeResource(device, { instances: 1 }));
            const meshletData = uploaded(world.meshletDataBuffer);
            const secondResource = MESHLETS * MESHLET_DATA_U32S;
            expect(meshletData[0 * MESHLET_DATA_U32S + MESHLET_DATA.PAGE]).to.equal(0);
            expect(meshletData[secondResource + 1 * MESHLET_DATA_U32S + MESHLET_DATA.PAGE], 'second resource pages start at 2').to.equal(2 + 1);
            expect(world.totalPages).to.equal(4);
            expect(world.workItemCapacity).to.equal(2);
            expect(world.recordCapacity).to.equal(2 * MESHLETS);
            expect(world.indexWorst, 'no table, flags 0: everything opaque').to.deep.equal([2 * MESHLETS * TRIANGLES * 3, 0, 0]);
            expect(world.indexCapacity).to.deep.equal(world.indexWorst);
            // resident: page p sits in pool slot p
            expect(Array.from(uploaded(world.residencyBuffer))).to.deep.equal([0, 1, 2, 3]);
            world.destroy();
        });

        it('synthesizes a material record from the primitive when the bake has no table', function () {
            const world = build(new MeshletWorld(device), makeResource(device, { instances: 1 }));
            const table = uploaded(world.materialTableBuffer);
            const f32 = new Float32Array(table.buffer);
            expect(Array.from(f32.subarray(MATERIAL_RECORD.BASE_COLOR, MATERIAL_RECORD.BASE_COLOR + 4))).to.deep.equal([0.5, 0.25, 1, 1]);
            expect(f32[MATERIAL_RECORD.ROUGHNESS]).to.equal(1);
            expect(f32[MATERIAL_RECORD.ALPHA_CUTOFF]).to.equal(0.5);
            expect(table[MATERIAL_RECORD.SLOT_WORDS]).to.equal(MATERIAL_SLOT_ABSENT);
            expect(table[MATERIAL_RECORD.SLOT_WORDS + 3]).to.equal(MATERIAL_SLOT_ABSENT);
            world.destroy();
        });

        it('derives the draw bucket from the baked material flags and restamps the meshlet flags', function () {
            const masked = build(new MeshletWorld(device), makeResource(device, { instances: 1, materialFlags: MATERIAL_FLAG_ALPHA_MASK | MATERIAL_FLAG_DOUBLE_SIDED }));
            expect(masked.indexWorst).to.deep.equal([0, 0, MESHLETS * TRIANGLES * 3]);
            const flags = uploaded(masked.meshletDataBuffer)[MESHLET_DATA.FLAGS];
            expect(flags & MESHLET_FLAG_ALPHA_MASKED).to.not.equal(0);
            expect(flags & MESHLET_FLAG_TWO_SIDED).to.not.equal(0);
            masked.destroy();

            const twoSided = build(new MeshletWorld(device), makeResource(device, { instances: 1, materialFlags: MATERIAL_FLAG_DOUBLE_SIDED }));
            expect(twoSided.indexWorst[MESHLET_BUCKET_OPAQUE_TWO_SIDED]).to.equal(MESHLETS * TRIANGLES * 3);
            expect(twoSided.indexWorst[MESHLET_BUCKET_MASKED]).to.equal(0);
            expect(twoSided.indexWorst[MESHLET_BUCKET_OPAQUE]).to.equal(0);
            twoSided.destroy();
        });

        it('splits a geometry budget between the page pool and the index buffers after the fixed costs', function () {
            const world = new MeshletWorld(device);
            world.poolBytes = 1024 * 1024;
            build(world, makeResource(device, { instances: 4 }));
            const b = world.budgetBreakdown;
            expect(b.budget).to.equal(1024 * 1024);
            expect(b.fixed).to.be.above(0);
            expect(b.pagePool + b.indices + b.fixed + b.perView).to.equal(b.budget);
            expect(b.indices).to.equal(Math.floor((b.budget - b.fixed - b.perView) * world.indexBudgetFraction));
            expect(world.indexBudgetTotal).to.equal(Math.floor(b.indices / 4));
            expect(world.pagePoolBytes).to.equal(b.pagePool);
            world.destroy();
        });

        it('falls back to a minimum page pool when the budget cannot even hold the fixed costs', function () {
            const world = new MeshletWorld(device);
            world.poolBytes = 64;
            build(world, makeResource(device, { instances: 4 }));
            expect(world.pagePoolBytes, 'min(pages, 64) pages').to.be.above(0);
            expect(world.budgetBreakdown.pagePool + world.budgetBreakdown.indices).to.equal(2 * PAGE_BYTES);
            world.destroy();
        });
    });

    describe('after finalize', function () {

        it('rewrites only the transform-derived words of a placement', function () {
            const world = build(new MeshletWorld(device), makeResource(device, { instances: 2 }));
            world.objectDataBuffer.impl.writes.length = 0;
            const t = new Float32Array(16);
            t[0] = 3; t[5] = 3; t[10] = 3; t[15] = 1; t[12] = 100;
            world.setPlacementTransform(0, { data: t, mul2: () => {} }, 1, 1);
            const row = 1 * OBJECT_DATA_U32S;
            // the test transform object only carries data; the world multiplies inst.matrix into it
            expect(world.objectDataCpuF[row + OBJECT_DATA.MAX_SCALE]).to.equal(3);
            expect(world.objectDataCpu[row + OBJECT_DATA.MESHLET_COUNT], 'untouched').to.equal(MESHLETS);
            const write = world.objectDataBuffer.impl.writes[0];
            expect(write.offset, 'only the second row is uploaded').to.equal(OBJECT_DATA_U32S * 4);
            expect(write.data).to.have.lengthOf(OBJECT_DATA_U32S);
            world.destroy();
        });

        it('hides a sub-range of a placement through the flags word without a rebuild', function () {
            const world = build(new MeshletWorld(device), makeResource(device, { instances: 3 }));
            world.setPlacementHidden(0, true, 1, 2);
            const flag = i => world.objectDataCpu[i * OBJECT_DATA_U32S + OBJECT_DATA.FLAGS] & OBJECT_FLAG_HIDDEN;
            expect([flag(0), flag(1), flag(2)]).to.deep.equal([0, OBJECT_FLAG_HIDDEN, OBJECT_FLAG_HIDDEN]);
            world.setPlacementHidden(0, false);
            expect([flag(0), flag(1), flag(2)]).to.deep.equal([0, 0, 0]);
            world.destroy();
        });

        it('flags outlined instances and uploads one contiguous range', function () {
            const world = build(new MeshletWorld(device), makeResource(device, { instances: 4 }));
            world.objectDataBuffer.impl.writes.length = 0;
            world.setInstancesOutlined([1, 3]);
            const flag = i => world.objectDataCpu[i * OBJECT_DATA_U32S + OBJECT_DATA.FLAGS] & OBJECT_FLAG_OUTLINED;
            expect([flag(0), flag(1), flag(2), flag(3)]).to.deep.equal([0, OBJECT_FLAG_OUTLINED, 0, OBJECT_FLAG_OUTLINED]);
            expect(world.outlinedCount).to.equal(2);
            const write = world.objectDataBuffer.impl.writes[0];
            expect(write.offset).to.equal(1 * OBJECT_DATA_U32S * 4);
            expect(write.data, 'rows 1..3').to.have.lengthOf(3 * OBJECT_DATA_U32S);
            world.clearOutlines();
            expect(world.outlinedCount).to.equal(0);
            expect(flag(1) | flag(3)).to.equal(0);
            world.destroy();
        });

        it('resolves pick ids to per-instance records', function () {
            const world = build(new MeshletWorld(device), makeResource(device, { instances: 2 }), makeResource(device, { instances: 1 }));
            const records = world.pickRecords;
            expect(records).to.have.lengthOf(3);
            expect(records.map(r => r.pickId)).to.deep.equal([world.pickIdBase, world.pickIdBase + 1, world.pickIdBase + 2]);
            expect(records[2].placement).to.equal(world.placements[1]);
            expect(records[2].instanceIndex).to.equal(2);
            expect(world.pickRecords, 'cached').to.equal(records);
            world.destroy();
        });

        it('swaps to the debug materials for colour modes and back to lit', function () {
            const world = build(new MeshletWorld(device), makeResource(device, { instances: 1 }));
            const lit = world.bucketMaterials;
            expect(lit).to.equal(world.litMaterials);
            world.setColorMode(MESHLET_COLOR_MODE.MESHLET);
            expect(world.bucketMaterials).to.not.equal(lit);
            expect(world.bucketMaterials).to.have.lengthOf(3);
            world.setColorMode(MESHLET_COLOR_MODE.LIT);
            expect(world.bucketMaterials).to.equal(lit);
            world.destroy();
        });

        it('caps the index ceiling by the budget and by maxIndices', function () {
            const world = build(new MeshletWorld(device), makeResource(device, { instances: 1 }));
            world.deviceIndexCeiling = 1000;
            world.indexBudgetTotal = 0;
            world.maxIndices = 0;
            expect(world.indexCeiling, 'device only').to.equal(1000);
            world.indexBudgetTotal = 600;
            expect(world.indexCeiling, 'budget pool').to.equal(600);
            world.indexOverrun = 100;
            expect(world.indexCeiling, 'budget plus overrun').to.equal(700);
            world.maxIndices = 500;
            expect(world.indexCeiling, 'maxIndices lowers it').to.equal(500);
            world.destroy();
        });
    });
});
