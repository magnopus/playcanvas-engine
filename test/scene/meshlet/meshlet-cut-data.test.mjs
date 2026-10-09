import { expect } from 'chai';

import { createCutFixture } from './meshlet-cut-fixture.mjs';
import { MESHLET_DATA as M, MESHLET_DATA_U32S, MESHLET_NO_PARENT } from '../../../src/scene/meshlet/constants.js';
import { buildMeshletGroups, MeshletCutData } from '../../../src/scene/meshlet/meshlet-cut-data.js';

describe('meshlet replacement groups', function () {
    it('checks each shared page and parent representative once per replacement', function () {
        const { meshlets, objects } = createCutFixture();
        // Every meshlet shares a page, and the two coarse members share one parent.
        for (let m = 0; m < 10; m++) meshlets[m * MESHLET_DATA_U32S + M.PAGE] = 0;
        meshlets[5 * MESHLET_DATA_U32S + M.PARENT] = 8;
        const device = {
            buffers: new Set(),
            _vram: { sb: 0 },
            createBufferImpl: () => ({
                allocate() {},
                destroy() {},
                write(device, offset, data) {
                    this.words = data.slice();
                }
            })
        };
        const cut = new MeshletCutData(device, meshlets, objects);
        try {
            const words = cut.groups.impl.words;
            expect(words[1], 'unique ancestor count').to.equal(1);
            expect(words[3], 'unique finer page count').to.equal(1);
            const roots = cut.tasks.impl.words[1];
            expect(words[roots + 3], 'unique root page count').to.equal(1);
            expect(cut.rootIndices).to.deep.equal([6, 0, 0]);
            expect(cut.selectionTopology.byteSize).to.equal(10 * 8);
            const topology = cut.selectionTopology.impl.words;
            expect(Array.from(topology)).to.deep.equal([
                4, MESHLET_NO_PARENT, 4, MESHLET_NO_PARENT,
                6, MESHLET_NO_PARENT, 6, MESHLET_NO_PARENT,
                8, 4, 8, 4, 8, 6, 9, 6,
                MESHLET_NO_PARENT, 8, MESHLET_NO_PARENT, 9
            ]);
        } finally {
            cut.destroy();
        }
    });

    it('tables where each LOD level starts and the lowest root level, for compaction to skip finer levels', function () {
        const device = {
            buffers: new Set(),
            _vram: { sb: 0 },
            createBufferImpl: () => ({ allocate() {}, destroy() {}, write() {} })
        };
        // fixture: level 0 = meshlets 0-3, level 1 = 4-7, level 2 = 8-9 (the roots)
        let { meshlets, objects } = createCutFixture();
        let cut = new MeshletCutData(device, meshlets, objects);
        try {
            expect(cut.levelStarts.byteSize).to.equal(10 * 4);
            expect(Array.from(cut.levelStartsCpu.subarray(0, 3)), '[lowest root level, level 1 start, level 2 start]').to.deep.equal([2, 4, 8]);
        } finally {
            cut.destroy();
        }
        // a terminal group leaves a root at level 1: compaction must never start above it
        ({ meshlets, objects } = createCutFixture());
        meshlets[5 * MESHLET_DATA_U32S + M.PARENT] = MESHLET_NO_PARENT;
        cut = new MeshletCutData(device, meshlets, objects);
        try {
            expect(cut.levelStartsCpu[0]).to.equal(1);
        } finally {
            cut.destroy();
        }
    });

    it('stores the coarse cost each parent contributes to a replacement', function () {
        const { meshlets, objects } = createCutFixture();
        const device = {
            buffers: new Set(),
            _vram: { sb: 0 },
            createBufferImpl: () => ({
                allocate() {},
                destroy() {},
                write(device, offset, data) {
                    this.words = data.slice();
                }
            })
        };
        const cut = new MeshletCutData(device, meshlets, objects);
        try {
            // Group 4 replaces coarse members 4 (under root 8) and 5 (under root 9) with 0 and 1.
            const words = Array.from(cut.groups.impl.words.slice(0, 16));
            expect(words).to.deep.equal([
                4, 2, 14, 2, 12, 6, 2, 2,
                8, 3, 1, 9, 3, 1,
                0, 1
            ]);
        } finally {
            cut.destroy();
        }
    });

    it('recovers all coarse members and cross-linked fine dependencies', function () {
        const { groups, roots } = buildMeshletGroups(createCutFixture().meshlets);
        expect(roots).to.deep.equal([8, 9]);
        expect(groups).to.deep.equal([
            { start: 4, count: 2, level: 1, children: [0, 1] },
            { start: 6, count: 2, level: 1, children: [2, 3] },
            { start: 8, count: 1, level: 2, children: [4, 6] },
            { start: 9, count: 1, level: 2, children: [5, 7] }
        ]);
    });

    it('rejects a coarse boundary inconsistent with the stored group bounds', function () {
        const { meshlets } = createCutFixture();
        meshlets[5 * MESHLET_DATA_U32S + M.GROUP_SPHERE] = 1;
        expect(() => buildMeshletGroups(meshlets)).to.throw('inconsistent error or bounds');
    });

    it('rejects cycles and parents outside the primitive', function () {
        for (const parent of [0, 100]) {
            const { meshlets } = createCutFixture();
            meshlets[M.PARENT] = parent;
            expect(() => buildMeshletGroups(meshlets)).to.throw('invalid parent');
        }
    });

    it('promotes real coarse geometry beneath legacy empty roots into the pinned fallback', function () {
        const { meshlets, objects } = createCutFixture();
        const f = new Float32Array(meshlets.buffer);
        for (const m of [8, 9]) {
            meshlets[m * MESHLET_DATA_U32S + M.TRIANGLE_COUNT] = 0;
            f[m * MESHLET_DATA_U32S + M.CLUSTER_ERROR] = 1e30;
        }
        const device = {
            buffers: new Set(),
            _vram: { sb: 0 },
            createBufferImpl: () => ({
                allocate() {},
                write(device, offset, data) {
                    this.words = data.slice();
                },
                destroy() {}
            })
        };
        const cut = new MeshletCutData(device, meshlets, objects);
        try {
            for (const m of [4, 5, 6, 7]) expect(meshlets[m * MESHLET_DATA_U32S + M.PARENT]).to.equal(MESHLET_NO_PARENT);
            for (const m of [4, 5, 6, 7]) expect(cut.selectionTopology.impl.words[m * 2]).to.equal(MESHLET_NO_PARENT);
            expect(cut.rootIndices).to.deep.equal([12, 0, 0]);
            expect(cut.rootRecords).to.equal(4);
            expect(cut.rootPages).to.deep.equal([4, 5, 6, 7, 8, 9]);
        } finally {
            cut.destroy();
        }
    });
});
