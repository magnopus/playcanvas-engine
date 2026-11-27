import { expect } from 'chai';

import { createCutFixture } from './meshlet-cut-fixture.mjs';
import { MESHLET_BUCKET_OPAQUE, MESHLET_DATA as M, MESHLET_DATA_U32S } from '../../../src/scene/meshlet/constants.js';
import { MeshletCutData } from '../../../src/scene/meshlet/meshlet-cut-data.js';
import {
    U32Builder, buildCutTable, deserializeCutTables, fingerprintRecords, relocateCutTable, serializeCutTables
} from '../../../src/scene/meshlet/meshlet-cut-table.js';

const withPageBase = (meshlets, pageBase) => {
    const copy = meshlets.slice();
    for (let m = 0; m < copy.length / MESHLET_DATA_U32S; m++) copy[m * MESHLET_DATA_U32S + M.PAGE] += pageBase;
    return copy;
};

const makeDevice = () => ({
    buffers: new Set(),
    _vram: { sb: 0 },
    createBufferImpl: () => ({
        allocate() {},
        destroy() {},
        write(device, offset, data) {
            this.words = data.slice();
        }
    })
});

describe('meshlet cut tables', function () {

    describe('#relocateCutTable', function () {
        it('lays a primitive-local table into a world as a table built from rebased records would be', function () {
            const { meshlets } = createCutFixture();
            const local = buildCutTable(meshlets);

            const viaTable = new U32Builder();
            for (let i = 0; i < 7; i++) viaTable.push1(0); // words of earlier primitives
            const rootA = relocateCutTable(local, viaTable, 100, 50, MESHLET_BUCKET_OPAQUE);

            // the same primitive with its pages already rebased, placed at the same spot
            const rebased = buildCutTable(withPageBase(meshlets, 50));
            const direct = new U32Builder();
            for (let i = 0; i < 7; i++) direct.push1(0);
            const rootB = relocateCutTable(rebased, direct, 100, 0, MESHLET_BUCKET_OPAQUE);

            expect(rootA).to.equal(rootB);
            expect(Array.from(viaTable.view())).to.deep.equal(Array.from(direct.view()));
        });

        it('keeps meshlet-local indices and shifts offsets, pages and the first meshlet', function () {
            const { meshlets } = createCutFixture();
            const local = buildCutTable(meshlets);
            const out = new U32Builder();
            out.push1(0);
            const root = relocateCutTable(local, out, 1000, 500, MESHLET_BUCKET_OPAQUE);
            const words = out.view();
            expect(words[root], 'root header first meshlet').to.equal(1000);
            expect(words[root + 2], 'root page list offset').to.equal(root + 8);
            expect(words[root + 5], 'root bucket').to.equal(MESHLET_BUCKET_OPAQUE);
            expect(words[1], 'first group representative').to.equal(1000 + 4);
            expect(words[1 + 3], 'its page count').to.equal(local.words[3]);
            expect(words[root + 8]).to.be.at.least(500);
        });
    });

    describe('MeshletCutData with a kept table', function () {
        it('writes the same tables as when it builds them', function () {
            const { meshlets, instances } = createCutFixture();
            const plain = new MeshletCutData(makeDevice(), meshlets.slice(), instances);

            const local = buildCutTable(meshlets);
            const records = meshlets.slice();
            const source = {
                count: 10,
                get: () => records.slice(),
                done: () => {},
                cutTable: () => ({ table: local, pageBase: 0 })
            };
            const kept = new MeshletCutData(makeDevice(), source, instances);
            try {
                expect(Array.from(kept.groups.impl.words)).to.deep.equal(Array.from(plain.groups.impl.words));
                expect(Array.from(kept.selectionTopology.impl.words)).to.deep.equal(Array.from(plain.selectionTopology.impl.words));
                expect(kept.floorPages).to.deep.equal(plain.floorPages);
                expect(kept.stageLevels).to.deep.equal(plain.stageLevels);
            } finally {
                plain.destroy();
                kept.destroy();
            }
        });
    });

    describe('serialization', function () {
        it('round-trips a resource\'s tables, with a null slot for a primitive without one', function () {
            const { meshlets } = createCutFixture();
            const table = buildCutTable(meshlets);
            const blob = serializeCutTables([null, table, null]);
            const back = deserializeCutTables(blob.buffer, 3);
            expect(back[0]).to.equal(null);
            expect(back[2]).to.equal(null);
            expect(back[1].count).to.equal(table.count);
            expect(back[1].rootOffset).to.equal(table.rootOffset);
            expect(back[1].fingerprint).to.equal(table.fingerprint);
            expect(Array.from(back[1].words)).to.deep.equal(Array.from(table.words));
            expect(Array.from(back[1].birth)).to.deep.equal(Array.from(table.birth));
            expect(Array.from(back[1].floor)).to.deep.equal(Array.from(table.floor));
            expect(Array.from(back[1].levels)).to.deep.equal(Array.from(table.levels));
        });

        it('rejects a blob of another primitive count, a truncated blob and foreign bytes', function () {
            const { meshlets } = createCutFixture();
            const blob = serializeCutTables([buildCutTable(meshlets)]);
            expect(deserializeCutTables(blob.buffer, 2)).to.equal(null);
            expect(deserializeCutTables(blob.buffer.slice(0, blob.byteLength - 8), 1)).to.equal(null);
            expect(deserializeCutTables(new Uint32Array(16).buffer, 1)).to.equal(null);
        });
    });

    describe('#fingerprintRecords', function () {
        it('is stable for the same records and changes when the structure does', function () {
            const { meshlets } = createCutFixture();
            expect(fingerprintRecords(meshlets)).to.equal(fingerprintRecords(meshlets.slice()));
            const edited = meshlets.slice();
            edited[3 * MESHLET_DATA_U32S + M.PARENT] = 8;
            expect(fingerprintRecords(edited)).to.not.equal(fingerprintRecords(meshlets));
            expect(fingerprintRecords(meshlets.slice(0, 9 * MESHLET_DATA_U32S))).to.not.equal(fingerprintRecords(meshlets));
        });
    });
});
