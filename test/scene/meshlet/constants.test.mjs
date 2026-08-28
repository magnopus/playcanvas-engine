import { expect } from 'chai';

import {
    CULL_PARAMS, CULL_PARAMS_VEC4S, INDIRECT_DISPATCH_U32S, INDIRECT_DRAW_U32S, MATERIAL_RECORD, MATERIAL_RECORD_U32S, MATERIAL_SLOT,
    MATERIAL_TEXTURE_SLOTS, MESHLET_BUCKET_COUNT, MESHLET_BUCKET_MASKED, MESHLET_BUCKET_OPAQUE, MESHLET_COLOR_MODE,
    MESHLET_MAX_UV_CHANNELS,
    MESHLET_COUNTER, MESHLET_DATA,
    MESHLET_BUCKET_OPAQUE_TWO_SIDED, MESHLET_COUNTER_U32S, MESHLET_CULL_SLICE, MESHLET_DATA_U32S,
    MESHLET_DISPATCH_WIDTH, MESHLET_FLAG_ALPHA_MASKED, MESHLET_FLAG_DAG_ROOT, MESHLET_FLAG_TWO_SIDED,
    MESHLET_INDEX_WRITE_WORKGROUP, MESHLET_INSTANCE_CULL_WORKGROUP,
    OBJECT_DATA, OBJECT_DATA_U32S, OBJECT_FLAG_HAS_TANGENTS, OBJECT_FLAG_HIDDEN, OBJECT_FLAG_HOVERED, OBJECT_FLAG_OUTLINED,
    PAGE_HEADER, PAGE_HEADER_BYTES, PAGE_TABLE, PAGE_TABLE_FIELDS, RECORD_U32S, TEX_RESIDENCY_U32S, TEXEL_RATE_PER_MIP,
    WORK_ITEM_U32S
} from '../../../src/scene/meshlet/constants.js';
import { MeshletPrimitive, MeshletResource } from '../../../src/scene/meshlet/meshlet-resource.js';

// These constants are the runtime half of a contract with gltf-tools' MAG_meshlets_gpu /
// MAG_meshlets_stream v2 format and with the WGSL cull shaders. The relations below are the
// ones the shaders and the world's buffer sizing depend on; a change here must be deliberate.
describe('meshlet constants', function () {

    it('keeps the GPU record sizes at the baked layout', function () {
        expect(MESHLET_DATA_U32S).to.equal(32);
        expect(MATERIAL_RECORD_U32S).to.equal(32);
        expect(PAGE_TABLE_FIELDS).to.equal(8);
        expect(PAGE_HEADER_BYTES).to.equal(48);
    });

    it('sizes the counter buffer for three blocks of one word per bucket, padded to 4', function () {
        // [0] workItems, [1] records, then cursors / committed ends / unclamped demand per bucket
        const needed = 2 + 3 * MESHLET_BUCKET_COUNT;
        expect(MESHLET_COUNTER_U32S).to.be.at.least(needed);
        expect(MESHLET_COUNTER_U32S % 4).to.equal(0);
        expect(MESHLET_COUNTER_U32S - needed).to.be.below(4);
    });

    it('numbers the draw buckets contiguously in index-buffer order', function () {
        expect([MESHLET_BUCKET_OPAQUE, MESHLET_BUCKET_OPAQUE_TWO_SIDED, MESHLET_BUCKET_MASKED]).to.deep.equal([0, 1, 2]);
        expect(MESHLET_BUCKET_COUNT).to.equal(3);
    });

    it('keeps the meshlet and object flag bits distinct', function () {
        const meshletFlags = [MESHLET_FLAG_ALPHA_MASKED, MESHLET_FLAG_DAG_ROOT, MESHLET_FLAG_TWO_SIDED];
        const objectFlags = [OBJECT_FLAG_HIDDEN, OBJECT_FLAG_HAS_TANGENTS, OBJECT_FLAG_OUTLINED, OBJECT_FLAG_HOVERED];
        for (const set of [meshletFlags, objectFlags]) {
            const union = set.reduce((a, b) => a | b, 0);
            expect(set.reduce((a, b) => a + b, 0), 'flags overlap').to.equal(union);
            set.forEach(f => expect(f & (f - 1), 'flag is a single bit').to.equal(0));
        }
    });

    it('keeps the page table and header offsets within their records', function () {
        Object.values(PAGE_TABLE).forEach(v => expect(v).to.be.below(PAGE_TABLE_FIELDS));
        Object.values(PAGE_HEADER).forEach(v => expect(v * 4).to.be.below(PAGE_HEADER_BYTES));
    });

    it('keeps every record field inside its record', function () {
        Object.values(MESHLET_DATA).forEach(v => expect(v).to.be.below(MESHLET_DATA_U32S));
        Object.values(OBJECT_DATA).forEach(v => expect(v).to.be.below(OBJECT_DATA_U32S));
        expect(OBJECT_DATA.PICK_ID).to.equal(OBJECT_DATA_U32S - 1);
        expect(MESHLET_DATA.SPHERE + 4).to.be.at.most(MESHLET_DATA.CONE_APEX);
        expect(MESHLET_DATA.GROUP_SPHERE + 4).to.be.at.most(MESHLET_DATA.PARENT);
    });

    it('lays the material record out with the slot words followed by their transforms', function () {
        Object.values(MATERIAL_RECORD).forEach(v => expect(v).to.be.below(MATERIAL_RECORD_U32S));
        expect(MATERIAL_RECORD.SLOT_WORDS + MATERIAL_TEXTURE_SLOTS).to.equal(MATERIAL_RECORD.SLOT_TRANSFORMS);
        expect(MATERIAL_RECORD.SLOT_TRANSFORMS + 2 * MATERIAL_TEXTURE_SLOTS).to.be.at.most(MATERIAL_RECORD_U32S);
        expect(Object.values(MATERIAL_SLOT).sort()).to.deep.equal([0, 1, 2, 3]);
        expect(TEX_RESIDENCY_U32S).to.equal(2);
        expect(MESHLET_MAX_UV_CHANNELS, 'the slot word has a 2-bit texCoord field').to.equal(4);
        expect(new Set(Object.values(MESHLET_COLOR_MODE)).size).to.equal(3);
    });

    it('lays the counter blocks out contiguously, one word per bucket', function () {
        expect(MESHLET_COUNTER.COMMITTED_BASE).to.equal(MESHLET_COUNTER.CURSOR_BASE + MESHLET_BUCKET_COUNT);
        expect(MESHLET_COUNTER.DEMAND_BASE).to.equal(MESHLET_COUNTER.COMMITTED_BASE + MESHLET_BUCKET_COUNT);
        expect(MESHLET_COUNTER.DEMAND_BASE + MESHLET_BUCKET_COUNT).to.be.at.most(MESHLET_COUNTER_U32S);
    });

    it('matches the WebGPU indirect argument layouts and the record strides', function () {
        expect(INDIRECT_DRAW_U32S, 'DrawIndexedIndirect args').to.equal(5);
        expect(INDIRECT_DISPATCH_U32S, 'DispatchWorkgroupsIndirect args').to.equal(3);
        expect(RECORD_U32S).to.equal(4);
        expect(WORK_ITEM_U32S).to.equal(2);
    });

    it('keeps the texture feedback scale and the material slot count the shaders assume', function () {
        expect(TEXEL_RATE_PER_MIP).to.equal(16);
        expect(MATERIAL_TEXTURE_SLOTS).to.equal(4);
    });

    it('uses a power-of-two dispatch grid and a workgroup-sized cull slice', function () {
        expect(MESHLET_DISPATCH_WIDTH & (MESHLET_DISPATCH_WIDTH - 1)).to.equal(0);
        expect(MESHLET_CULL_SLICE & (MESHLET_CULL_SLICE - 1)).to.equal(0);
        expect(MESHLET_CULL_SLICE).to.be.at.most(256);
        expect(MESHLET_INSTANCE_CULL_WORKGROUP).to.be.at.most(256);
        expect(MESHLET_INDEX_WRITE_WORKGROUP).to.be.at.most(256);
    });

    it('lays the cull parameter rows out back to back inside the parameter block', function () {
        expect(CULL_PARAMS.PLANES).to.equal(0);
        expect(CULL_PARAMS.PLANES + CULL_PARAMS.PLANE_COUNT).to.equal(CULL_PARAMS.CAMERA);
        expect(CULL_PARAMS.CAMERA + 1).to.equal(CULL_PARAMS.LOD);
        expect(CULL_PARAMS.LOD + 1).to.equal(CULL_PARAMS.VIEW_PROJ);
        expect(CULL_PARAMS.VIEW_PROJ + 4, 'a 4x4 matrix').to.equal(CULL_PARAMS.STREAMING);
        expect(CULL_PARAMS.STREAMING + 1).to.equal(CULL_PARAMS.VIEW_DIR);
        expect(CULL_PARAMS.VIEW_DIR).to.be.below(CULL_PARAMS_VEC4S);
    });
});

describe('MeshletResource', function () {

    const makePrimitive = (meshletCount) => {
        const prim = new MeshletPrimitive();
        prim.meshletData = new Uint32Array(meshletCount * MESHLET_DATA_U32S);
        prim.meshletDataF32 = new Float32Array(prim.meshletData.buffer);
        prim.meshletCount = meshletCount;
        prim.lods = [];
        prim.aabbCenter = [0, 0, 0];
        prim.aabbHalfExtents = [1, 1, 1];
        return prim;
    };

    const makeManifest = pageCount => ({
        blobs: [{ uri: 'pages_roots.dat', byteLength: 0 }],
        pageTable: new Uint32Array(pageCount * PAGE_TABLE_FIELDS),
        pageCount,
        rootPages: [0],
        attributeLayout: {},
        pageSizeBytes: 65536,
        pageAlignment: 256,
        positionGrid: { origin: [0, 0, 0], step: 0.01, bits: 16 }
    });

    it('totals the meshlets across primitives and keeps the placements it is given', function () {
        const instances = [{ primIndex: 0, matrix: new Float32Array(16) }, { primIndex: 1, matrix: new Float32Array(16) }];
        const resource = new MeshletResource(null, [makePrimitive(5), makePrimitive(7)], makeManifest(3), instances);
        expect(resource.totalMeshlets).to.equal(12);
        expect(resource.instances).to.equal(instances);
        expect(resource.materialTable).to.equal(null);
        expect(resource.textureManifest).to.equal(null);
    });

    it('defaults to no placements and releases its data on destroy', function () {
        const resource = new MeshletResource(null, [makePrimitive(1)], makeManifest(1));
        expect(resource.instances).to.deep.equal([]);
        resource.destroy();
        expect(resource.primitives).to.deep.equal([]);
        expect(resource.manifest).to.equal(null);
    });
});
