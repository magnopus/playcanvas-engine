import { expect } from 'chai';

import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import {
    INDIRECT_DRAW_U32S, MATERIAL_RECORD, MATERIAL_RECORD_U32S, MESHLET_BUCKET_COUNT, MESHLET_BUCKET_MASKED,
    MESHLET_BUCKET_OPAQUE_TWO_SIDED, MESHLET_COUNTER, MESHLET_CULL_SLICE, MESHLET_DATA, MESHLET_DATA_U32S,
    MESHLET_DISPATCH_WIDTH, OBJECT_DATA, OBJECT_DATA_U32S, PAGE_HEADER, RECORD_U32S, TEX_RESIDENCY_U32S,
    WORK_ITEM_U32S
} from '../../../src/scene/meshlet/constants.js';
import { MeshletCullShaders } from '../../../src/scene/meshlet/meshlet-cull-shaders.js';
import {
    dispatchArgsWGSL, finalizeArgsWGSL, indexWriteWGSL, instanceCullWGSL, meshletCullWGSL, resetPhase2WGSL
} from '../../../src/scene/meshlet/shaders/meshlet-cull-wgsl.js';
import * as pageWGSL from '../../../src/scene/meshlet/shaders/meshlet-page-wgsl.js';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

// The cull shaders cannot run headless, but their text is a contract with the JS that binds
// and reads the buffers they write. These tests pin the interpolated layout so a change to
// constants.js that the shaders silently disagree with fails here rather than as garbage
// geometry.
describe('meshlet cull shaders', function () {

    const shaders = { instanceCullWGSL, dispatchArgsWGSL, meshletCullWGSL, finalizeArgsWGSL, indexWriteWGSL, resetPhase2WGSL };
    const count = (text, re) => (text.match(re) ?? []).length;

    it('interpolate every constant', function () {
        for (const [name, text] of Object.entries(shaders)) {
            expect(text, name).to.not.match(/\$\{|undefined|NaN/);
        }
    });

    it('bind at most ten storage buffers in the meshlet cull, the WebGPU default per-stage limit', function () {
        // adding a storage binding here silently disables the pipeline on adapters at the
        // default limit; two views of one buffer count twice
        expect(count(meshletCullWGSL, /var<storage/g)).to.equal(10);
    });

    it('size the cull workgroup to the work-item slice and round the fan-out up', function () {
        expect(meshletCullWGSL).to.include(`@workgroup_size(${MESHLET_CULL_SLICE})`);
        expect(instanceCullWGSL).to.include(`(meshletCount + ${MESHLET_CULL_SLICE - 1}u) / ${MESHLET_CULL_SLICE}u`);
        expect(instanceCullWGSL).to.include(`workItems[item] = MeshletWorkItem(instance, slice * ${MESHLET_CULL_SLICE}u);`);
    });

    it('dispatch the cull and index-write passes over a 2D grid to dodge the per-dimension limit', function () {
        for (const text of [dispatchArgsWGSL, finalizeArgsWGSL]) {
            expect(text).to.include(`min(count, ${MESHLET_DISPATCH_WIDTH}u)`.replace('count', text === finalizeArgsWGSL ? 'recordCount' : 'count'));
        }
        expect(meshletCullWGSL).to.include(`workgroupId.y * ${MESHLET_DISPATCH_WIDTH}u + workgroupId.x`);
        expect(indexWriteWGSL).to.include(`workgroupId.y * ${MESHLET_DISPATCH_WIDTH}u + workgroupId.x`);
    });

    it('reserve, commit and record demand in the three counter blocks, one word per bucket', function () {
        expect(meshletCullWGSL).to.include(`atomicAdd(&counters[${MESHLET_COUNTER.CURSOR_BASE}u + bucket], indexNeed)`);
        expect(meshletCullWGSL).to.include(`atomicMax(&counters[${MESHLET_COUNTER.DEMAND_BASE}u + bucket], cursor + indexNeed)`);
        expect(meshletCullWGSL).to.include(`atomicMax(&counters[${MESHLET_COUNTER.COMMITTED_BASE}u + bucket], cursor + indexNeed)`);
        // demand is recorded BEFORE the capacity check so an overflowing frame still reports it
        expect(meshletCullWGSL.indexOf(`counters[${MESHLET_COUNTER.DEMAND_BASE}u + bucket]`))
        .to.be.below(meshletCullWGSL.indexOf('if (cursor + indexNeed > capacity)'));
        expect(count(meshletCullWGSL, /uniform indexCapacity\d : u32;/g)).to.equal(MESHLET_BUCKET_COUNT);
    });

    it('derive the draw bucket from the meshlet flags in bucket order', function () {
        expect(meshletCullWGSL).to.include(`select(select(0u, ${MESHLET_BUCKET_OPAQUE_TWO_SIDED}u, twoSided), ${MESHLET_BUCKET_MASKED}u, alphaMasked)`);
    });

    it('write one indirect draw per bucket from the committed ends', function () {
        expect(count(finalizeArgsWGSL, /writeDraw\(uniform\.drawSlot\d/g)).to.equal(MESHLET_BUCKET_COUNT);
        expect(finalizeArgsWGSL).to.include(`let base = slot * ${INDIRECT_DRAW_U32S}u;`);
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            expect(finalizeArgsWGSL).to.include(`atomicLoad(&counters[${MESHLET_COUNTER.COMMITTED_BASE + b}u])`);
        }
    });

    it('reset everything but the demand block between the two cull phases', function () {
        expect(resetPhase2WGSL).to.include(`atomicStore(&counters[${MESHLET_COUNTER.RECORDS}u], 0u)`);
        expect(resetPhase2WGSL).to.include(`for (var i = ${MESHLET_COUNTER.CURSOR_BASE}u; i < ${MESHLET_COUNTER.DEMAND_BASE}u; i++)`);
    });

    it('share one page layout and vertex decode with the materials', function () {
        for (const [name, text] of Object.entries(pageWGSL)) {
            const wgsl = typeof text === 'function' ? text('read') : text;
            if (typeof wgsl === 'string') expect(wgsl, name).to.not.match(/\$\{|undefined|NaN/);
        }
        expect(pageWGSL.meshletPageLayoutWGSL).to.include('struct MeshletPageLayout');
        expect(indexWriteWGSL).to.include('let pageLayout = meshletPageLayout(pageBase, hasTangents, uvFloatsPerVertex);');
        expect(indexWriteWGSL).to.include('records[recordIndex].baseIndexOffset');
        expect(pageWGSL.meshletDecodeDrawIndexWGSL).to.include('vec2u(index >> 8u, index & 0xFFu)');
    });

    // WGSL std layout rules: scalars align 4, vec2 8, vec3/vec4/mat 16; an array's alignment
    // is its element's, its size the padded element size times the count; a struct is padded
    // to its largest alignment. The typed views must land every field exactly where the CPU
    // enums say the flat buffers put it.
    const wgslLayout = (fields) => {
        const scalar = { u32: [4, 4], f32: [4, 4], vec2u: [8, 8], vec3f: [16, 12], vec4f: [16, 16], mat4x4f: [16, 64] };
        const typeOf = (type) => {
            const array = /^array<(\w+), (\d+)>$/.exec(type);
            if (!array) return scalar[type];
            const [align, size] = scalar[array[1]];
            return [align, Math.ceil(size / align) * align * Number(array[2])];
        };
        const offsets = {};
        let offset = 0;
        let structAlign = 1;
        for (const [name, type] of fields) {
            const [align, size] = typeOf(type);
            offset = Math.ceil(offset / align) * align;
            offsets[name] = offset;
            offset += size;
            structAlign = Math.max(structAlign, align);
        }
        return { offsets, size: Math.ceil(offset / structAlign) * structAlign };
    };
    const enumName = (field, aliases = {}) => aliases[field] ?? field.replace(/([A-Z])/g, '_$1').toUpperCase();
    const expectLayout = (fields, enums, wordsPerRecord, aliases) => {
        const { offsets, size } = wgslLayout(fields);
        expect(size, 'record stride').to.equal(wordsPerRecord * 4);
        for (const [field] of fields) {
            const key = enumName(field, aliases);
            if (!(key in enums)) continue;   // reserved / padding fields have no enum
            expect(offsets[field] / 4, `${field} -> ${key}`).to.equal(enums[key]);
        }
    };

    it('lay the typed struct views out exactly as the CPU-side enums describe the flat buffers', function () {
        expectLayout(pageWGSL.MESHLET_OBJECT_STRUCT, OBJECT_DATA, OBJECT_DATA_U32S, { worldMatrix: 'MATRIX' });
        expectLayout(pageWGSL.MESHLET_DATA_STRUCT, MESHLET_DATA, MESHLET_DATA_U32S);
        expectLayout(pageWGSL.MESHLET_MATERIAL_STRUCT, MATERIAL_RECORD, MATERIAL_RECORD_U32S);
        expect(wgslLayout(pageWGSL.MESHLET_RECORD_STRUCT).size).to.equal(RECORD_U32S * 4);
        expect(wgslLayout(pageWGSL.MESHLET_WORK_ITEM_STRUCT).size).to.equal(WORK_ITEM_U32S * 4);
        expect(wgslLayout(pageWGSL.MESHLET_TEX_RESIDENCY_STRUCT).size).to.equal(TEX_RESIDENCY_U32S * 4);
        // every enum field is covered by a struct field (nothing addressable only from the CPU)
        const covered = new Set(pageWGSL.MESHLET_OBJECT_STRUCT.map(([f]) => enumName(f, { worldMatrix: 'MATRIX' })));
        Object.keys(OBJECT_DATA).forEach(k => expect(covered.has(k), `OBJECT_DATA.${k} has a struct field`).to.equal(true));
    });

    it('encode drawn indices as record << 8 | local vertex and read the page header by name', function () {
        expect(indexWriteWGSL).to.include('(recordIndex << 8u) | localVert');
        expect(meshletCullWGSL).to.include('records[recordIndex] = MeshletRecord(instance, chosen, baseIndexOffset, bucket);');
        expect(indexWriteWGSL).to.include(`pagePool[pageBase + ${PAGE_HEADER.MV_COUNT}u]`);
        expect(indexWriteWGSL).to.include(`pagePool[pageBase + ${PAGE_HEADER.VERTEX_COUNT}u]`);
    });

    describe('MeshletCullShaders', function () {

        let device;

        beforeEach(function () {
            jsdomSetup();
            device = new NullGraphicsDevice(document.createElement('canvas'));
        });

        afterEach(function () {
            device.destroy();
            device = null;
            jsdomTeardown();
        });

        it('compiles the six compute shaders once and owns the placeholder HZB', function () {
            const set = new MeshletCullShaders(device);
            for (const name of ['instanceCull', 'dispatchArgs', 'meshletCull', 'finalizeArgs', 'indexWrite', 'resetPhase2']) {
                expect(set[name], name).to.exist;
                expect(set[name].definition.cshader, `${name} is a compute shader`).to.be.a('string');
            }
            expect(set.dummyHzb.width).to.equal(1);
            set.destroy();
        });
    });
});
