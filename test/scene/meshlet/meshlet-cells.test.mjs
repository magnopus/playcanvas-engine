import { expect } from 'chai';

import { createCutFixture } from './meshlet-cut-fixture.mjs';
import {
    MESHLET_CELL_SIZE, MESHLET_CUT_ROOT_CHUNK, MESHLET_DATA as M, MESHLET_DATA_U32S, MESHLET_NO_PARENT
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

// world spheres of instances at the given positions and bounding radii, and the callback the
// cells read them through
const createSpheres = (positions, radii) => {
    const spheres = new Float32Array(positions.length * 4);
    positions.forEach((p, i) => spheres.set([p[0], p[1], p[2], radii[i]], i * 4));
    const sphereOf = (i, out, offset) => out.set(spheres.subarray(i * 4, i * 4 + 4), offset);
    return { spheres, sphereOf };
};

const worldSphere = (spheres, i) => Array.from(spheres.subarray(i * 4, i * 4 + 4));

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
        const { spheres, sphereOf } = createSpheres(positions, radii);
        const cells = new MeshletCells(createDevice(), positions.length, sphereOf);
        try {
            expect(cells.cellCount).to.equal(3);
            expect(Array.from(cells.instanceOrder).sort((a, b) => a - b), 'a permutation of the instances')
            .to.deep.equal(positions.map((_, i) => i));
            for (let c = 0; c < cells.cellCount; c++) {
                const members = Array.from(cells.instanceOrder.subarray(c * MESHLET_CELL_SIZE, (c + 1) * MESHLET_CELL_SIZE));
                const sphere = Array.from(cells.spheres.subarray(c * 4, c * 4 + 4));
                members.forEach((i) => {
                    expect(cells.instanceCell[i]).to.equal(c);
                    expect(contains(sphere, worldSphere(spheres, i)), `cell ${c} bounds instance ${i}`).to.equal(true);
                });
                // a cell is one cluster or the large instances, never a mix
                const kinds = new Set(members.map(i => (i >= 128 ? 'large' : `cluster${i % 2}`)));
                expect(kinds.size, `cell ${c} holds ${[...kinds]}`).to.equal(1);
            }
        } finally {
            cells.destroy();
        }
    });

    it('refits only the cells of moved instances and uploads just their range', function () {
        const positions = Array.from({ length: 130 }, (_, i) => [i, 0, 0]);
        const { spheres, sphereOf } = createSpheres(positions, positions.map(() => 1));
        const cells = new MeshletCells(createDevice(), positions.length, sphereOf);
        try {
            const moved = 70;
            const cell = cells.instanceCell[moved];
            spheres[moved * 4] = 5000;
            cells.boundsBuffer.impl.writes.length = 0;
            cells.refit(moved, 1);
            const sphere = Array.from(cells.spheres.subarray(cell * 4, cell * 4 + 4));
            expect(contains(sphere, worldSphere(spheres, moved))).to.equal(true);
            expect(cells.boundsBuffer.impl.writes).to.deep.equal([{ offset: cell * 16, dataOffset: cell * 4, size: 4 }]);
        } finally {
            cells.destroy();
        }
    });
});

describe('MeshletCutData queue ownership', function () {

    it('gives every task exactly one owner: its first parent, or the root when it has none', function () {
        const { meshlets, instances } = createCutFixture();
        const device = createDevice();
        const cut = new MeshletCutData(device, meshlets, instances);
        try {
            const words = cut.groups.impl.words;
            const rootOffset = words[0];
            const ownedList = offset => words.subarray(words[offset + 2] + words[offset + 3] + 1,
                words[offset + 2] + words[offset + 3] + 1 + words[words[offset + 2] + words[offset + 3]]);
            // primitive-local representative -> task row, walking the rows before rootOffset
            const rowOf = new Map();
            const groupTasks = [];
            for (let offset = 1; offset < rootOffset;) {
                rowOf.set(words[offset] - instances.first(0), offset);
                groupTasks.push(offset);
                const ownedStart = words[offset + 2] + words[offset + 3];
                // the children list follows the owned task list
                const childStart = ownedStart + 1 + words[ownedStart];
                offset = childStart + 1 + words[childStart];
            }
            expect(groupTasks.length).to.be.above(1);
            const owners = new Map();
            const record = (owner, list) => list.forEach((task) => {
                expect(owners.has(task), `task ${task} owned twice`).to.equal(false);
                owners.set(task, owner);
            });
            record('root', Array.from(ownedList(rootOffset)));
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

    it('lists each group\'s children and the roots in chunks, for the emit pass', function () {
        const { meshlets, instances } = createCutFixture();
        const device = createDevice();
        const cut = new MeshletCutData(device, meshlets, instances);
        try {
            const words = cut.groups.impl.words;
            const rootOffset = words[0];
            const first = instances.first(0);
            const parentOf = m => meshlets[m * MESHLET_DATA_U32S + M.PARENT];
            const listAfterOwned = (offset) => {
                const ownedStart = words[offset + 2] + words[offset + 3];
                return ownedStart + 1 + words[ownedStart];
            };
            const seen = [];
            for (let offset = 1; offset < rootOffset;) {
                const childStart = listAfterOwned(offset);
                const children = Array.from(words.subarray(childStart + 1, childStart + 1 + words[childStart]));
                const representative = words[offset] - first;
                expect(children.length).to.be.above(0);
                children.forEach(child => expect(parentOf(child), `child ${child}`).to.equal(representative));
                seen.push(...children);
                offset = childStart + 1 + words[childStart];
            }
            // the roots: chunk count, then each chunk's count and members
            const chunkStart = listAfterOwned(rootOffset);
            const roots = [];
            for (let k = 0; k < words[chunkStart]; k++) {
                const chunk = chunkStart + 1 + k * (MESHLET_CUT_ROOT_CHUNK + 1);
                roots.push(...words.subarray(chunk + 1, chunk + 1 + words[chunk]));
            }
            roots.forEach(root => expect(parentOf(root)).to.equal(MESHLET_NO_PARENT));
            // every meshlet is a candidate exactly once: as one group's child, or as a root
            const all = [...seen, ...roots].sort((a, b) => a - b);
            expect(all).to.deep.equal(Array.from({ length: meshlets.length / MESHLET_DATA_U32S }, (_, m) => m));
        } finally {
            cut.destroy();
        }
    });
});
