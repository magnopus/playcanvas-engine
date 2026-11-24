import { expect } from 'chai';

import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import {
    CULL_PARAMS_VEC4S, CUT_BUDGET, MATERIAL_RECORD, MATERIAL_RECORD_U32S, MESHLET_BUCKET_COUNT,
    MESHLET_CELL_CULL_WORKGROUP, MESHLET_CELL_SIZE, MESHLET_COUNTER, MESHLET_COUNTER_CLEAR_U32S, MESHLET_CULL_SLICE,
    MESHLET_DATA, MESHLET_DATA_U32S, MESHLET_DISPATCH_WIDTH, MESHLET_INDEX_WRITE_WORKGROUP, MESHLET_LIST_WORKGROUP,
    MESHLET_VISIBLE_CELLS_BASE, OBJECT_DATA, OBJECT_DATA_U32S, PRIMITIVE_DATA, RECORD_U32S, TEX_RESIDENCY_U32S
} from '../../../src/scene/meshlet/constants.js';
import { MeshletCullShaders } from '../../../src/scene/meshlet/meshlet-cull-shaders.js';
import * as commonWGSL from '../../../src/scene/meshlet/shaders/meshlet-cull-common-wgsl.js';
import * as cullWGSL from '../../../src/scene/meshlet/shaders/meshlet-cull-wgsl.js';
import * as cutWGSL from '../../../src/scene/meshlet/shaders/meshlet-cut-wgsl.js';
import * as pageWGSL from '../../../src/scene/meshlet/shaders/meshlet-page-wgsl.js';
import * as rootWGSL from '../../../src/scene/meshlet/shaders/meshlet-root-admit-wgsl.js';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

// The cull shaders cannot run headless, but their text is a contract with the JS that binds and
// reads the buffers they write. These tests check the interpolated layout, so a change to
// constants.js that the shaders silently disagree with fails here rather than as garbage geometry.
describe('meshlet cull shaders', function () {

    // every compute entry point, by export name
    const compute = Object.fromEntries([cullWGSL, cutWGSL, rootWGSL]
    .flatMap(module => Object.entries(module))
    .filter(([, text]) => typeof text === 'string' && text.includes('@compute')));

    // the shared snippets, with the entry point template instantiated
    const commonSnippets = Object.fromEntries(Object.entries(commonWGSL)
    .map(([name, snippet]) => [name, typeof snippet === 'function' ? snippet(64) : snippet]));

    const count = (text, re) => (text.match(re) ?? []).length;

    // the u32 indices a shader reads or writes in a buffer: `buffer[N]`, or `buffer[N + bucket]`
    // reported as the bucket block's last word
    const indices = (text, buffer) => Array.from(text.matchAll(new RegExp(`\\b${buffer}\\[(\\d+)u( \\+ bucket)?\\]`, 'g')))
    .map(([, index, bucket]) => Number(index) + (bucket ? MESHLET_BUCKET_COUNT - 1 : 0));

    it('interpolate every constant', function () {
        expect(Object.keys(compute)).to.include.members(['meshletCullWGSL', 'meshletCutQueueWGSL', 'meshletRootAdmitWGSL']);
        for (const [name, text] of Object.entries({ ...compute, ...commonSnippets })) {
            expect(text, name).to.not.match(/\$\{|undefined|NaN/);
        }
    });

    it('bind at most eight storage buffers per stage, the WebGPU guaranteed minimum', function () {
        // a storage binding past the adapter's limit disables the pipeline outright; two views of
        // one buffer count twice
        for (const [name, text] of Object.entries(compute)) {
            expect(count(text, /var<storage/g), name).to.be.at.most(8);
        }
    });

    it('size each workgroup by the constant its dispatch is computed from', function () {
        const size = text => Number(/@workgroup_size\((\d+)/.exec(text)[1]);
        expect(size(cullWGSL.cellCullWGSL)).to.equal(MESHLET_CELL_CULL_WORKGROUP);
        expect(size(cullWGSL.meshletCullWGSL)).to.equal(MESHLET_CULL_SLICE);
        expect(size(cullWGSL.indexWriteWGSL)).to.equal(MESHLET_INDEX_WRITE_WORKGROUP);
        expect(size(rootWGSL.meshletRootClassifyWGSL), 'one lane per instance of a cell').to.equal(MESHLET_CELL_SIZE);
        for (const name of ['meshletCutQueueWGSL', 'meshletCutEmitWGSL', 'meshletRootScatterWGSL', 'meshletRootAdmitWGSL']) {
            expect(size(compute[name]), name).to.equal(MESHLET_LIST_WORKGROUP);
        }
        // the GPU-sized list dispatches round up by the same workgroup
        const listGroups = `+ ${MESHLET_LIST_WORKGROUP - 1}u) / ${MESHLET_LIST_WORKGROUP}u`;
        expect(cutWGSL.meshletCutQueueArgsWGSL).to.include(listGroups);
        expect(rootWGSL.meshletRootBandArgsWGSL).to.include(listGroups);
    });

    it('lay out every 2D dispatch grid through the shared helpers', function () {
        // the grid's width lives only in the two shared snippets, so a writer and its reader agree
        const width = new RegExp(`\\b${MESHLET_DISPATCH_WIDTH}u`);
        expect(commonWGSL.meshletDispatchGroupWGSL).to.match(width);
        expect(commonWGSL.meshletDispatchArgsWGSL).to.match(width);
        for (const [name, text] of Object.entries(compute)) {
            const own = text.split(commonWGSL.meshletDispatchGroupWGSL).join('').split(commonWGSL.meshletDispatchArgsWGSL).join('');
            expect(own, name).to.not.match(width);
            if (/indirectDispatch\[/.test(own)) {
                expect(own, `${name} writes its args through meshletWriteDispatch`).to.not.match(/indirectDispatch\[base/);
            }
        }
    });

    it('stay within the parameter rows the culler uploads', function () {
        for (const [name, text] of Object.entries({ ...compute, ...commonSnippets })) {
            for (const row of indices(text, 'cullParams')) {
                expect(row, name).to.be.below(CULL_PARAMS_VEC4S);
            }
        }
    });

    it('charge the cut budget in the counter words its enum names', function () {
        const used = new Set();
        for (const text of Object.values(compute)) {
            for (const index of indices(text, 'counters')) used.add(index);
        }
        // a per-bucket index reads as its last bucket's word
        expect(used).to.include.members([CUT_BUDGET.INDICES_BASE + MESHLET_BUCKET_COUNT - 1, CUT_BUDGET.RECORDS, CUT_BUDGET.REJECTED_RECORDS, CUT_BUDGET.QUEUE]);
        for (const text of Object.values(compute)) expect(text).to.not.include('cutBudget');
    });

    it('keep the per-frame counters inside the cleared block and the cell list after it', function () {
        for (const [name, text] of Object.entries({ ...compute, ...commonSnippets })) {
            for (const index of indices(text, 'counters')) {
                expect(index, name).to.be.below(MESHLET_COUNTER_CLEAR_U32S);
            }
        }
        expect(MESHLET_VISIBLE_CELLS_BASE).to.be.at.least(MESHLET_COUNTER_CLEAR_U32S);
        expect(Math.max(...Object.values(MESHLET_COUNTER))).to.be.below(MESHLET_COUNTER_CLEAR_U32S);
    });

    it('reset only the drawn records and index reservations between the two phases', function () {
        // demand, rendered totals and record demand accumulate across both phases
        const text = cullWGSL.resetPhase2WGSL;
        const cleared = new Set(indices(text, 'counters'));
        const [, from, to] = /for \(var i = (\d+)u; i < (\d+)u; i\+\+\)/.exec(text);
        for (let i = Number(from); i < Number(to); i++) cleared.add(i);
        const expected = [MESHLET_COUNTER.RECORDS];
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) expected.push(MESHLET_COUNTER.CURSOR_BASE + b, MESHLET_COUNTER.COMMITTED_BASE + b);
        expect([...cleared].sort((a, b) => a - b)).to.deep.equal(expected.sort((a, b) => a - b));
    });

    it('declare one index capacity per bucket wherever indices are reserved', function () {
        for (const name of ['meshletCullWGSL', 'meshletCutQueueWGSL', 'meshletRootAdmitWGSL']) {
            expect(count(compute[name], /uniform indexCapacity\d : u32;/g), name).to.equal(MESHLET_BUCKET_COUNT);
        }
        // finalize writes one draw per bucket, with the whole cut's record demand
        expect(count(cullWGSL.finalizeArgsWGSL, /writeDraw\(uniform\.drawSlot\d/g)).to.equal(MESHLET_BUCKET_COUNT);
        expect(indices(cullWGSL.finalizeArgsWGSL, 'counters')).to.include.members([CUT_BUDGET.RECORDS, CUT_BUDGET.REJECTED_RECORDS]);
    });

    it('charge and record demand before checking capacity, and claim only what fits', function () {
        const order = (text, ...anchors) => {
            const at = anchors.map(a => text.indexOf(a));
            at.forEach((i, k) => expect(i, anchors[k]).to.be.at.least(0));
            expect(at, anchors.join(' < ')).to.deep.equal([...at].sort((a, b) => a - b));
        };
        // an overflowing frame still reports its whole demand, so the CPU grows the buffers to it
        order(cutWGSL.meshletCutQueueWGSL, `counters[${MESHLET_COUNTER.DEMAND_BASE}u + bucket], indexNeed`, '> uniform.recordCapacity');
        order(rootWGSL.meshletRootAdmitWGSL, `counters[${MESHLET_COUNTER.DEMAND_BASE}u + bucket], indices`, 'if (!fits)', 'atomicOr(&claimBits');
        order(rootWGSL.meshletRootAdmitWGSL, 'if (!fits)', `counters[${MESHLET_COUNTER.CUT_QUEUE_END}u], queueNeed`);
        // the cull reserves a record before the indices it then draws
        order(cullWGSL.meshletCullWGSL, 'recordIndex >= uniform.recordCapacity', `counters[${MESHLET_COUNTER.CURSOR_BASE}u + bucket]`);
    });

    it('encode drawn indices the way the vertex stages decode them', function () {
        expect(cullWGSL.indexWriteWGSL).to.include('(recordIndex << 8u) | localVert');
        expect(pageWGSL.meshletDecodeDrawIndexWGSL).to.include('vec2u(index >> 8u, index & 0xFFu)');
        // a mirrored instance reverses each triangle's winding in place
        const sourceCorners = Array.from({ length: 6 }, (_, corner) => corner - corner % 3 + (3 - corner % 3) % 3);
        expect(sourceCorners).to.deep.equal([0, 2, 1, 3, 5, 4]);
        expect(cullWGSL.indexWriteWGSL).to.include('select(corner, corner - triangleCorner + (3u - triangleCorner) % 3u, mirrored)');
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
    const enumName = field => field.replace(/([A-Z])/g, '_$1').toUpperCase();
    const expectLayout = (fields, enums, wordsPerRecord) => {
        const { offsets, size } = wgslLayout(fields);
        expect(size, 'record stride').to.equal(wordsPerRecord * 4);
        for (const [field] of fields) {
            const key = enumName(field);
            if (!(key in enums)) continue;   // reserved / padding fields have no enum
            expect(offsets[field] / 4, `${field} -> ${key}`).to.equal(enums[key]);
        }
    };

    it('lay the typed struct views out exactly as the CPU-side enums describe the flat buffers', function () {
        expectLayout(pageWGSL.MESHLET_OBJECT_STRUCT, OBJECT_DATA, OBJECT_DATA_U32S);
        expectLayout(pageWGSL.MESHLET_PRIMITIVE_STRUCT, PRIMITIVE_DATA, OBJECT_DATA_U32S);
        expectLayout(pageWGSL.MESHLET_DATA_STRUCT, MESHLET_DATA, MESHLET_DATA_U32S);
        expectLayout(pageWGSL.MESHLET_MATERIAL_STRUCT, MATERIAL_RECORD, MATERIAL_RECORD_U32S);
        expect(wgslLayout(pageWGSL.MESHLET_RECORD_STRUCT).size).to.equal(RECORD_U32S * 4);
        expect(wgslLayout(pageWGSL.MESHLET_TEX_RESIDENCY_STRUCT).size).to.equal(TEX_RESIDENCY_U32S * 4);
        // every enum field is covered by a struct field (nothing addressable only from the CPU)
        for (const [fields, enums] of [[pageWGSL.MESHLET_OBJECT_STRUCT, OBJECT_DATA], [pageWGSL.MESHLET_PRIMITIVE_STRUCT, PRIMITIVE_DATA]]) {
            const covered = new Set(fields.map(([f]) => enumName(f)));
            Object.keys(enums).forEach(k => expect(covered.has(k), `${k} has a struct field`).to.equal(true));
        }
    });

    it('read each primitive field from the instance-struct word its enum names', function () {
        // meshletPrimitiveData reinterprets a primitive row through the instance struct: map each
        // assignment's source back to a word of the row
        const accessor = pageWGSL.meshletObjectDataWGSL;
        const component = { x: 0, y: 1, z: 2, w: 3, xyz: 0 };
        const wordsRow = Number(/let words = bitcast<vec4u>\(row\.matrixRows\[(\d)\]\);/.exec(accessor)[1]);
        const sourceWord = (source) => {
            let m = /^row\.matrixRows\[(\d)\](?:\.(\w+))?$/.exec(source);
            if (m) return Number(m[1]) * 4 + (m[2] ? component[m[2]] : 0);
            m = /^words\.(\w)$/.exec(source);
            if (m) return wordsRow * 4 + component[m[1]];
            if (source === 'row.primitive') return OBJECT_DATA.PRIMITIVE;
            throw new Error(`unrecognised source ${source}`);
        };
        const assigned = Object.fromEntries(Array.from(accessor.matchAll(/primitive\.(\w+) = ([^;]+);/g), ([, field, source]) => [enumName(field), sourceWord(source)]));
        expect(Object.keys(assigned)).to.have.members(Object.keys(PRIMITIVE_DATA));
        for (const [key, word] of Object.entries(PRIMITIVE_DATA)) {
            expect(assigned[key], key).to.equal(word);
        }
    });

    it('unpack the texture residency word as its field table documents', function () {
        // the documented packing, low bits first, drives the accessors' shifts and masks
        const [, , comment] = pageWGSL.MESHLET_TEX_RESIDENCY_STRUCT[0];
        const fields = Array.from(comment.matchAll(/(\w+):(\d+)/g), ([, name, bits]) => [name, Number(bits)]);
        expect(fields.reduce((sum, [, bits]) => sum + bits, 0)).to.equal(32);
        const accessors = { fineSlotLayer: 'meshletTexSlotLayer', family: 'meshletTexFamily', tailStart: 'meshletTexTailStart', sizeBias: 'meshletTexSizeBias' };
        let shift = 0;
        for (const [name, bits] of fields) {
            const fn = new RegExp(`fn ${accessors[name]}\\(r : MeshletTexResidency\\) -> u32 \\{ return ([^;]+); \\}`).exec(pageWGSL.meshletStructsWGSL)[1];
            const m = /^\(?r\.slotFamilyBias(?: >> (\d+)u\))?(?: & 0x([0-9A-F]+)u)?$/.exec(fn);
            expect(m, `${accessors[name]}: ${fn}`).to.not.equal(null);
            expect(Number(m[1] ?? 0), `${name} shift`).to.equal(shift);
            const mask = m[2] ? parseInt(m[2], 16) : 2 ** (32 - shift) - 1;
            expect(mask, `${name} mask`).to.equal(2 ** bits - 1);
            shift += bits;
        }
    });

    describe('MeshletCullShaders', function () {

        let device;

        beforeEach(function () {
            jsdomSetup();
            device = new NullGraphicsDevice(document.createElement('canvas'));
            device.supportsCompute = true;
        });

        afterEach(function () {
            device.destroy();
            device = null;
            jsdomTeardown();
        });

        it('compiles every compute shader once and releases them with the placeholder HZB', function () {
            const set = new MeshletCullShaders(device);
            const names = ['cellCull', 'cutQueue', 'cutQueueArgs', 'cutEmit', 'rootClassify', 'rootBandArgs', 'rootScatter',
                'rootAdmit', 'dispatchArgs', 'meshletCull', 'finalizeArgs', 'indexWrite', 'resetPhase2'];
            for (const name of names) {
                expect(set[name], name).to.exist;
                expect(set[name].definition.cshader, `${name} is a compute shader`).to.be.a('string');
            }
            expect(set.dummyHzb.width).to.equal(1);
            set.destroy();
            for (const name of [...names, 'dummyHzb']) {
                expect(set[name], name).to.equal(null);
            }
        });
    });
});
