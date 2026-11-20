import { StorageBuffer } from '../../platform/graphics/storage-buffer.js';
import { BUFFERUSAGE_COPY_DST } from '../../platform/graphics/constants.js';
import {
    MESHLET_CUT_ROOT_CHUNK, MESHLET_DATA as M, MESHLET_DATA_U32S, MESHLET_NO_PARENT, MESHLET_FLAG_SLOPPY
} from './constants.js';
import { bucketOfMeshletFlags } from './meshlet-material-table.js';

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 */

// fields every member of a coarse group shares with its representative
const GROUP_FIELDS = [M.CLUSTER_ERROR, M.GROUP_SPHERE, M.GROUP_SPHERE + 1, M.GROUP_SPHERE + 2, M.GROUP_SPHERE + 3];

/**
 * Recover complete replacement groups from the baker's level-major cluster order. Parent
 * representatives start contiguous coarse groups. Validate the shared error/bounds before
 * using this ordering, so an incompatible asset cannot silently produce partial fallbacks.
 *
 * @param {Uint32Array} data - Primitive meshlet records.
 * @returns {{ groups: { start: number, count: number, level: number, children: number[] }[], roots: number[] }} Groups and root clusters.
 * @ignore
 */
function buildMeshletGroups(data) {
    const count = data.length / MESHLET_DATA_U32S;
    const children = new Map();
    const roots = [];
    for (let m = 0; m < count; m++) {
        const parent = data[m * MESHLET_DATA_U32S + M.PARENT];
        if (parent === MESHLET_NO_PARENT) {
            roots.push(m);
        } else {
            if (parent >= count || data[parent * MESHLET_DATA_U32S + M.LOD_LEVEL] !== data[m * MESHLET_DATA_U32S + M.LOD_LEVEL] + 1) {
                throw new Error('Meshlet DAG has an invalid parent or non-consecutive levels.');
            }
            let list = children.get(parent);
            if (!list) children.set(parent, list = []);
            list.push(m);
        }
    }
    const starts = Array.from(children.keys()).sort((a, b) => a - b);
    const groups = [];
    let previousEnd = 0;
    for (let i = 0; i < starts.length; i++) {
        const start = starts[i];
        const row = start * MESHLET_DATA_U32S;
        const level = data[row + M.LOD_LEVEL];
        let end = start + 1;
        while (end < count && end !== starts[i + 1] && data[end * MESHLET_DATA_U32S + M.LOD_LEVEL] === level) end++;
        for (let m = previousEnd; m < start; m++) {
            if (data[m * MESHLET_DATA_U32S + M.LOD_LEVEL] !== 0 && data[m * MESHLET_DATA_U32S + M.TRIANGLE_COUNT] !== 0) throw new Error('Meshlet DAG is missing a coarse group representative.');
        }
        for (let m = start; m < end; m++) {
            const base = m * MESHLET_DATA_U32S;
            for (const field of GROUP_FIELDS) {
                if (data[base + field] !== data[row + field]) throw new Error('Meshlet DAG coarse group has inconsistent error or bounds.');
            }
        }
        const fine = children.get(start);
        for (const m of fine) {
            const base = m * MESHLET_DATA_U32S;
            if (data[base + M.PARENT_ERROR] !== data[row + M.CLUSTER_ERROR]) throw new Error('Meshlet DAG parent error does not match its replacement.');
            for (let k = 0; k < 4; k++) {
                if (data[base + M.PARENT_SPHERE + k] !== data[row + M.GROUP_SPHERE + k]) throw new Error('Meshlet DAG parent bounds do not match its replacement.');
            }
        }
        groups.push({ start, count: end - start, level, children: fine });
        previousEnd = end;
    }
    for (let m = previousEnd; m < count; m++) {
        if (data[m * MESHLET_DATA_U32S + M.LOD_LEVEL] !== 0 && data[m * MESHLET_DATA_U32S + M.TRIANGLE_COUNT] !== 0) throw new Error('Meshlet DAG has coarse clusters without a replacement group.');
    }
    return { groups, roots };
}

/**
 * The pages a primitive needs resident before its coarsest presentable level can draw: the
 * playable floor. A replacement group is tainted when its own simplification was sloppy
 * ({@link MESHLET_FLAG_SLOPPY}) or any cluster it replaces is tainted - sloppy geometry stays in
 * everything coarsened from it. The floor is every cluster that is a root or whose replacement
 * group (parent) is tainted: the untainted level each branch first reaches, plus every tainted
 * cluster above it, whose pages the cut must hold to refine down to it.
 *
 * @param {Uint32Array} source - The primitive's meshlet records.
 * @param {{ start: number, count: number, level: number, children: number[] }[]} groups - Its
 * replacement groups (see {@link buildMeshletGroups}).
 * @param {Map<number, number>} pages - Floor pages, extended in place: page -> coarsest LOD level
 * of a floor cluster on it.
 * @ignore
 */
function collectFloorPages(source, groups, pages) {
    const count = source.length / MESHLET_DATA_U32S;
    // per cluster: its replacement group is tainted (level 0 clusters belong to none)
    const tainted = new Uint8Array(count);
    const ordered = groups.slice().sort((a, b) => a.level - b.level);
    for (const group of ordered) {
        let taint = false;
        for (let m = group.start; m < group.start + group.count && !taint; m++) {
            taint = (source[m * MESHLET_DATA_U32S + M.FLAGS] & MESHLET_FLAG_SLOPPY) !== 0;
        }
        for (let i = 0; i < group.children.length && !taint; i++) {
            taint = tainted[group.children[i]] === 1;
        }
        if (taint) tainted.fill(1, group.start, group.start + group.count);
    }
    for (let m = 0; m < count; m++) {
        const row = m * MESHLET_DATA_U32S;
        const parent = source[row + M.PARENT];
        if (parent === MESHLET_NO_PARENT || tainted[parent]) {
            const page = source[row + M.PAGE];
            const level = source[row + M.LOD_LEVEL];
            if (!(pages.get(page) >= level)) pages.set(page, level);
        }
    }
}

/**
 * A Uint32Array that grows by doubling. The cut tables reach tens of millions of words on large
 * worlds; plain arrays of numbers take 8 bytes a word and copy themselves while growing, which
 * can exhaust the JS heap while a large world rebuilds.
 *
 * @ignore
 */
class U32Builder {
    data = new Uint32Array(1024);

    length = 0;

    /** @param {...number} values - Words to append. */
    push(...values) {
        this._reserve(values.length);
        for (let i = 0; i < values.length; i++) this.data[this.length++] = values[i];
    }

    /**
     * Appends one word, without the rest-array allocation of {@link push}: the build makes tens
     * of millions of these calls on a large world.
     *
     * @param {number} value - The word.
     */
    push1(value) {
        if (this.length === this.data.length) this._reserve(1);
        this.data[this.length++] = value;
    }

    /**
     * @param {number} count - Words about to be appended.
     * @private
     */
    _reserve(count) {
        if (this.length + count <= this.data.length) return;
        const grown = new Uint32Array(Math.max(this.data.length * 2, this.length + count));
        grown.set(this.data.subarray(0, this.length));
        this.data = grown;
    }

    /** @returns {Uint32Array} The words pushed so far. */
    view() {
        return this.data.subarray(0, this.length);
    }
}

/**
 * Supplies a world's meshlet records one primitive at a time, so the cut tables can be built
 * without the whole world's records in memory at once.
 *
 * @typedef {object} MeshletRecordSource
 * @property {number} count - Meshlets in the world.
 * @property {(first: number, count: number) => Uint32Array} get - The world records of the
 * primitive starting at meshlet `first`; the cut writes its derived words into them.
 * @property {(first: number, records: Uint32Array) => void} done - Called once per primitive with
 * the records as the cut left them, ready to upload.
 * @ignore
 */

/**
 * @param {Uint32Array|MeshletRecordSource} meshlets - Whole-world records (the tables' tests
 * build from one array), or a per-primitive source.
 * @returns {MeshletRecordSource} A per-primitive source.
 */
function recordSource(meshlets) {
    if (!(meshlets instanceof Uint32Array)) return meshlets;
    return {
        count: meshlets.length / MESHLET_DATA_U32S,
        get: (first, count) => meshlets.subarray(first * MESHLET_DATA_U32S, (first + count) * MESHLET_DATA_U32S),
        done: () => {}
    };
}

/** GPU tables shared by every view's resident, capacity-bounded cut. @ignore */
class MeshletCutData {
    /**
     * Per cut stage, its group LOD level (coarsest first).
     *
     * @type {number[]}
     */
    stageLevels;

    /**
     * Global pages of the playable floor (see {@link collectFloorPages}), roots included,
     * coarsest first - the order to load them in when the pool cannot hold them all.
     *
     * @type {number[]}
     */
    floorPages;

    /** @type {StorageBuffer} - per instance its root header offset, then the group rows. */
    groups;

    /** @type {StorageBuffer} - per meshlet its parent and birth group. */
    selectionTopology;

    /**
     * @param {GraphicsDevice} device - Device.
     * @param {Uint32Array|MeshletRecordSource} meshlets - World records, whole or one primitive at
     * a time; reserved words receive primitive-local birth representatives.
     * @param {{ count: number, first: (instance: number) => number, meshletCount: (instance: number) => number }} instances - The
     * world's instances: per instance, its primitive's first world-global meshlet and meshlet count.
     */
    constructor(device, meshlets, instances) {
        const records = recordSource(meshlets);
        const words = new U32Builder();
        const levelSet = new Set();
        // per primitive (by its first meshlet), its root header offset
        const primitives = new Map();
        const floorPages = new Map();
        // each meshlet's two cut links (parent and birth group), contiguous so the cut emit reads
        // them without fetching the 128-byte geometry records
        const topology = new Uint32Array(records.count * 2);
        const instanceCount = instances.count;
        // Leading words [0, instanceCount): each instance's root header offset in cutGroups.
        for (let i = 0; i < instanceCount; i++) words.push1(0);
        for (let instance = 0; instance < instanceCount; instance++) {
            const first = instances.first(instance);
            const count = instances.meshletCount(instance);
            let rootOffset = primitives.get(first);
            if (rootOffset === undefined) {
                const source = records.get(first, count);
                const { groups, roots } = buildMeshletGroups(source);
                collectFloorPages(source, groups, floorPages);
                for (let m = 0; m < count; m++) source[m * MESHLET_DATA_U32S + M.BIRTH_GROUP] = MESHLET_NO_PARENT;
                const costs = (members) => {
                    let indices = 0, records = 0;
                    for (const m of members) {
                        const triangles = source[m * MESHLET_DATA_U32S + M.TRIANGLE_COUNT];
                        indices += triangles * 3;
                        if (triangles) records++;
                    }
                    return [indices, records];
                };
                // Gather each group's dependencies and pages first, then lay the rows out: a
                // row ends with the tasks it owns (see below), which need their own offsets.
                const rows = groups.map((group) => {
                    levelSet.add(group.level);
                    const coarse = [];
                    const parents = new Map();
                    for (let m = group.start; m < group.start + group.count; m++) {
                        source[m * MESHLET_DATA_U32S + M.BIRTH_GROUP] = group.start;
                        coarse.push(m);
                        const parent = source[m * MESHLET_DATA_U32S + M.PARENT];
                        if (parent !== MESHLET_NO_PARENT) {
                            let members = parents.get(parent);
                            if (!members) parents.set(parent, members = []);
                            members.push(m);
                        }
                    }
                    const pages = new Set();
                    for (const child of group.children) pages.add(source[child * MESHLET_DATA_U32S + M.PAGE]);
                    /** @type {{ offset: number }[]} */
                    const owned = [];
                    return { group, coarse, parents, pages, offset: 0, owned };
                });
                // Ownership for the queued cut traversal: a task is enqueued by exactly one of the
                // groups it depends on - its first parent - when that one refines, and checks the
                // rest itself. Every parent sits one LOD level coarser, so all of them have been
                // decided by the time the task's own level runs. A task with no parent depends only
                // on the root admission, which enqueues it.
                const byStart = new Map(rows.map(row => [row.group.start, row]));
                /** @type {typeof rows} */
                const rootOwnedRows = [];
                for (const row of rows) {
                    const owner = row.parents.size ? byStart.get(row.parents.keys().next().value) : null;
                    if (row.parents.size && !owner) throw new Error('Meshlet DAG parent is not a replacement group.');
                    (owner ? owner.owned : rootOwnedRows).push(row);
                }
                let cursor = words.length;
                for (const row of rows) {
                    row.offset = cursor;
                    cursor += 8 + row.parents.size * 3 + row.pages.size + 1 + row.owned.length + 1 + row.group.children.length;
                }
                const rootOwned = rootOwnedRows.map(row => row.offset);
                for (const row of rows) {
                    const { group, coarse, parents, pages, offset } = row;
                    const owned = row.owned.map(child => child.offset);
                    const fineCost = costs(group.children), coarseCost = costs(coarse);
                    // Header: representative, dependency count, page-list offset/count,
                    // fine/coarse index and record costs. Lists contain unique parent
                    // representatives and pages, avoiding serial scans over every meshlet.
                    // Each parent also carries the coarse cost it contributed: an off-frustum
                    // parent refines without charging its children, so a replacement must not
                    // subtract those uncharged members from its own charge. After the pages:
                    // the count, then the task offsets, of the tasks this group owns. Last, the count
                    // and primitive-local indices of its children, which the emit pass draws when
                    // the group refines and their own group does not.
                    words.push(first + group.start, parents.size, offset + 8 + parents.size * 3, pages.size,
                        fineCost[0], coarseCost[0], fineCost[1], coarseCost[1]);
                    for (const [parent, members] of parents) {
                        const [indices, records] = costs(members);
                        words.push(parent, indices, records);
                    }
                    for (const page of pages) words.push1(page);
                    words.push1(owned.length);
                    for (const task of owned) words.push1(task);
                    words.push1(group.children.length);
                    for (const child of group.children) words.push1(child);
                    if (words.length !== offset + 8 + parents.size * 3 + pages.size + 1 + owned.length + 1 + group.children.length) {
                        throw new Error('MeshletCutData: task row layout mismatch.');
                    }
                }
                const rootCost = costs(roots);
                const pages = new Set();
                for (const root of roots) pages.add(source[root * MESHLET_DATA_U32S + M.PAGE]);
                const bucket = bucketOfMeshletFlags(source[M.FLAGS]);
                rootOffset = words.length;
                // root header: word 5 carries the draw bucket (there is no coarser cost to hold),
                // so the root admission needs no meshlet record. After the pages: the tasks that
                // depend on the roots alone, which the admission enqueues. Last, the roots in
                // lists of MESHLET_CUT_ROOT_CHUNK (the chunk count, then each list's count and
                // primitive-local indices), which the admission queues for the emit pass.
                words.push(first, 0, rootOffset + 8, pages.size, rootCost[0], bucket, rootCost[1], 0);
                for (const page of pages) words.push1(page);
                words.push1(rootOwned.length);
                for (const task of rootOwned) words.push1(task);
                words.push1(Math.ceil(roots.length / MESHLET_CUT_ROOT_CHUNK));
                for (let r = 0; r < roots.length; r += MESHLET_CUT_ROOT_CHUNK) {
                    const chunk = roots.slice(r, r + MESHLET_CUT_ROOT_CHUNK);
                    words.push1(chunk.length);
                    for (const root of chunk) words.push1(root);
                }
                for (let m = 0; m < count; m++) {
                    topology[(first + m) * 2] = source[m * MESHLET_DATA_U32S + M.PARENT];
                    topology[(first + m) * 2 + 1] = source[m * MESHLET_DATA_U32S + M.BIRTH_GROUP];
                }
                records.done(first, source);
                primitives.set(first, rootOffset);
            }
            words.data[instance] = rootOffset;
        }
        this.stageLevels = Array.from(levelSet).sort((a, b) => b - a);
        const create = (values) => {
            const buffer = new StorageBuffer(device, Math.max(values.byteLength, 16), BUFFERUSAGE_COPY_DST);
            if (values.length) buffer.write(0, values);
            return buffer;
        };
        this.groups = create(words.view());
        this.selectionTopology = new StorageBuffer(device, Math.max(topology.byteLength, 16), BUFFERUSAGE_COPY_DST);
        if (topology.length) this.selectionTopology.write(0, topology);
        this.floorPages = Array.from(floorPages.keys()).sort((a, b) => (floorPages.get(b) - floorPages.get(a)) || (a - b));
    }

    destroy() {
        this.groups.destroy();
        this.selectionTopology.destroy();
    }
}

export { buildMeshletGroups, collectFloorPages, MeshletCutData };
