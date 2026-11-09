import { expect } from 'chai';

import { createCutFixture } from './meshlet-cut-fixture.mjs';
import {
    MESHLET_CELL_SIZE, OBJECT_DATA as O, OBJECT_DATA_U32S,
    OBJECT_FLAG_IMPOSTOR_PRIMARY, OBJECT_IMPOSTOR_LAYER_SHIFT
} from '../../../src/scene/meshlet/constants.js';
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

    it('encloses a primary impostor instance\'s whole prototype sphere in its cell and maxRadius', function () {
        const { objects, f } = createObjects([[10, 20, 30]], [1.5]);
        objects[O.FLAGS] = OBJECT_FLAG_IMPOSTOR_PRIMARY | (1 << OBJECT_IMPOSTOR_LAYER_SHIFT);
        const protos = new Float32Array(16);
        protos.set([0, 4, 0, 12, 6, 12, 0, 0], 8);
        const cells = new MeshletCells(createDevice(), f, 1, protos);
        try {
            const sphere = Array.from(cells.spheres.subarray(0, 4));
            expect(contains(sphere, [10, 24, 30, 12])).to.equal(true);
            expect(cells.maxRadii[0]).to.be.at.least(12);
        } finally {
            cells.destroy();
        }
    });
});

describe('MeshletCutData queue ownership', function () {

    it('gives every task exactly one owner: its first parent, or the root when it has none', function () {
        const { meshlets, objects } = createCutFixture();
        const device = createDevice();
        const cut = new MeshletCutData(device, meshlets, objects);
        try {
            const words = cut.groups.impl.words;
            const tasks = cut.tasks.impl.words;
            const ownedList = offset => words.subarray(words[offset + 2] + words[offset + 3] + 1,
                words[offset + 2] + words[offset + 3] + 1 + words[words[offset + 2] + words[offset + 3]]);
            // primitive-local representative -> task row, from the non-root tasks of instance 0
            const rowOf = new Map();
            const groupTasks = [];
            for (let t = cut.levels[0].count; t < tasks.length / 2; t++) {
                if (tasks[t * 2] !== 0) continue;
                const offset = tasks[t * 2 + 1];
                rowOf.set(words[offset] - objects[O.FIRST_MESHLET], offset);
                groupTasks.push(offset);
            }
            expect(groupTasks.length).to.be.above(1);
            const owners = new Map();
            const record = (owner, list) => list.forEach((task) => {
                expect(owners.has(task), `task ${task} owned twice`).to.equal(false);
                owners.set(task, owner);
            });
            record('root', Array.from(ownedList(tasks[1])));
            groupTasks.forEach(offset => record(offset, Array.from(ownedList(offset))));
            groupTasks.forEach((offset) => {
                expect(owners.has(offset), `task ${offset} has an owner`).to.equal(true);
                const parentCount = words[offset + 1];
                const owner = owners.get(offset);
                if (parentCount === 0) {
                    expect(owner).to.equal('root');
                } else {
                    expect(owner, 'owned by its first parent').to.equal(rowOf.get(words[offset + 8]));
                }
            });
        } finally {
            cut.destroy();
        }
    });
});
