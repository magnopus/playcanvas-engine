import { expect } from 'chai';

import { createCutFixture } from './meshlet-cut-fixture.mjs';
import { MESHLET_CELL_SIZE, MESHLET_CUT_CHUNK, OBJECT_DATA as O, OBJECT_DATA_U32S } from '../../../src/scene/meshlet/constants.js';
import { MeshletCells } from '../../../src/scene/meshlet/meshlet-cells.js';
import { MeshletCutData } from '../../../src/scene/meshlet/meshlet-cut-data.js';

const createDevice = () => ({
    buffers: new Set(),
    _vram: { sb: 0 },
    createBufferImpl: () => ({
        writes: [],
        allocate() {},
        destroy() {},
        write(device, offset, data, dataOffset = 0, size = data.length - dataOffset) {
            this.words = data.slice();
            this.writes.push({ offset, dataOffset, size });
        }
    })
});

// `count` placements of the fixture primitive at the given positions and bounding radii
const createObjects = (positions, radii) => {
    const { objects: one } = createCutFixture();
    const objects = new Uint32Array(positions.length * OBJECT_DATA_U32S);
    const f = new Float32Array(objects.buffer);
    positions.forEach((p, i) => {
        const row = i * OBJECT_DATA_U32S;
        objects.set(one, row);
        f[row + 12] = p[0];
        f[row + 13] = p[1];
        f[row + 14] = p[2];
        f[row + O.SPHERE + 3] = radii[i];
    });
    return { objects, f };
};

// the world sphere of instance i, as the cull shaders compute it
const worldSphere = (f, i) => {
    const row = i * OBJECT_DATA_U32S;
    return [f[row + 12], f[row + 13], f[row + 14], f[row + O.SPHERE + 3] * f[row + O.MAX_SCALE]];
};

const contains = (outer, inner) => Math.hypot(outer[0] - inner[0], outer[1] - inner[1], outer[2] - inner[2]) + inner[3] <= outer[3];

describe('MeshletCells', function () {

    it('groups neighbours into cells, keeps large instances apart and bounds every member', function () {
        // two clusters of 64 small instances 1 km apart, interleaved in id order, then 64 large
        const positions = [];
        const radii = [];
        for (let i = 0; i < 128; i++) {
            positions.push([(i % 2) * 1000 + (i % 8), 0, Math.floor(i / 8)]);
            radii.push(1);
        }
        for (let i = 0; i < 64; i++) {
            positions.push([i * 10, 0, 500]);
            radii.push(600);
        }
        const { objects, f } = createObjects(positions, radii);
        const cells = new MeshletCells(createDevice(), f, positions.length);
        try {
            expect(cells.cellCount).to.equal(3);
            expect(Array.from(cells.instanceOrder).sort((a, b) => a - b), 'a permutation of the instances')
            .to.deep.equal(positions.map((_, i) => i));
            for (let c = 0; c < cells.cellCount; c++) {
                const members = Array.from(cells.instanceOrder.subarray(c * MESHLET_CELL_SIZE, (c + 1) * MESHLET_CELL_SIZE));
                const sphere = Array.from(cells.spheres.subarray(c * 4, c * 4 + 4));
                members.forEach((i) => {
                    expect(cells.instanceCell[i]).to.equal(c);
                    expect(contains(sphere, worldSphere(f, i)), `cell ${c} bounds instance ${i}`).to.equal(true);
                });
                // a cell is one cluster or the large instances, never a mix
                const kinds = new Set(members.map(i => (i >= 128 ? 'large' : `cluster${i % 2}`)));
                expect(kinds.size, `cell ${c} holds ${[...kinds]}`).to.equal(1);
            }
            expect(objects.length).to.equal(positions.length * OBJECT_DATA_U32S);
        } finally {
            cells.destroy();
        }
    });

    it('refits only the cells of moved instances and uploads just their range', function () {
        const positions = Array.from({ length: 130 }, (_, i) => [i, 0, 0]);
        const { f } = createObjects(positions, positions.map(() => 1));
        const cells = new MeshletCells(createDevice(), f, positions.length);
        try {
            const moved = 70;
            const cell = cells.instanceCell[moved];
            f[moved * OBJECT_DATA_U32S + 12] = 5000;
            cells.boundsBuffer.impl.writes.length = 0;
            cells.refit(f, moved, 1);
            const sphere = Array.from(cells.spheres.subarray(cell * 4, cell * 4 + 4));
            expect(contains(sphere, worldSphere(f, moved))).to.equal(true);
            expect(cells.boundsBuffer.impl.writes).to.deep.equal([{ offset: cell * 16, dataOffset: cell * 4, size: 4 }]);
        } finally {
            cells.destroy();
        }
    });
});

describe('MeshletCutData cell chunks', function () {

    it('splits each stage into chunks that cover every cell\'s tasks, cell by cell', function () {
        const positions = Array.from({ length: 150 }, (_, i) => [(i * 37) % 200, 0, (i * 11) % 90]);
        const { objects, f } = createObjects(positions, positions.map(() => 1));
        const { meshlets } = createCutFixture();
        const device = createDevice();
        const cells = new MeshletCells(device, f, positions.length);
        const cut = new MeshletCutData(device, meshlets, objects, cells.instanceOrder);
        try {
            const words = cut.groups.impl.words;
            const tasks = cut.tasks.impl.words;
            const stages = cut.levels.filter(level => !level.root);
            expect(stages.length).to.be.above(0);
            expect(cut.cellCount).to.equal(cells.cellCount);
            let listEnd = 0;
            stages.forEach((level, s) => {
                const [cellChunks, chunkTasks, listOffset, chunkCount] = cut.stageTable.slice(s * 4, s * 4 + 4);
                expect(Array.from(words.subarray(cut.stageTableBase + s * 4, cut.stageTableBase + s * 4 + 4)), 'stage table on the GPU')
                .to.deep.equal([cellChunks, chunkTasks, listOffset, chunkCount]);
                expect(listOffset, 'stage lists are packed').to.equal(listEnd);
                listEnd += chunkCount;
                expect(words[cellChunks + cut.cellCount]).to.equal(chunkCount);
                // the chunks span the stage's tasks exactly, never wider than a chunk
                expect(words[chunkTasks]).to.equal(level.start);
                expect(words[chunkTasks + chunkCount]).to.equal(level.start + level.count);
                for (let c = 0; c < cut.cellCount; c++) {
                    for (let chunk = words[cellChunks + c]; chunk < words[cellChunks + c + 1]; chunk++) {
                        const start = words[chunkTasks + chunk];
                        const end = words[chunkTasks + chunk + 1];
                        expect(end - start).to.be.within(1, MESHLET_CUT_CHUNK);
                        for (let t = start; t < end; t++) {
                            expect(cells.instanceCell[tasks[t * 2]], `task ${t} of stage ${s} belongs to cell ${c}`).to.equal(c);
                        }
                    }
                }
            });
            expect(cut.totalChunks).to.equal(listEnd);
            // root tasks stay in instance order for the root admission
            for (let i = 0; i < positions.length; i++) expect(tasks[i * 2]).to.equal(i);
        } finally {
            cut.destroy();
            cells.destroy();
        }
    });
});
