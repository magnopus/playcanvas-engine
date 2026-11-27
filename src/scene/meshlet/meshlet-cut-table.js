import {
    MESHLET_CUT_ROOT_CHUNK, MESHLET_DATA as M, MESHLET_DATA_U32S, MESHLET_NO_PARENT, MESHLET_FLAG_SLOPPY
} from './constants.js';

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
 * One primitive's cut tables in primitive-local terms: what {@link MeshletCutData} derives from
 * the primitive's meshlet records alone. Nothing in it depends on where the primitive sits in a
 * world, so it can be built once, saved, and laid into any world with {@link relocateCutTable}.
 *
 * @typedef {object} CutTable
 * @property {number} count - Meshlets in the primitive.
 * @property {Uint32Array} words - The group rows then the root header, with row and task
 * offsets relative to the start of this array, pages primitive-local and meshlet indices
 * primitive-local.
 * @property {number} rootOffset - Where the root header starts in {@link words}.
 * @property {Uint32Array} levels - The distinct group LOD levels.
 * @property {Uint32Array} floor - Pairs of (primitive-local page, coarsest level of a floor
 * cluster on it): the playable floor.
 * @property {Uint32Array} birth - Per meshlet, the start of the group it is born in, or
 * MESHLET_NO_PARENT.
 * @property {number} fingerprint - Of the records it was built from, to reject a stale table.
 * @ignore
 */

/**
 * A cheap fingerprint of a primitive's records: its size and a sample of the fields the cut
 * tables derive from. A table saved for content that has since been rebaked under the same name
 * almost always differs in one of them.
 *
 * @param {Uint32Array} source - The primitive's meshlet records.
 * @returns {number} A 32-bit fingerprint.
 * @ignore
 */
function fingerprintRecords(source) {
    const count = source.length / MESHLET_DATA_U32S;
    let h = Math.imul(count ^ 0x811c9dc5, 16777619);
    const samples = Math.min(count, 512);
    for (let i = 0; i < samples; i++) {
        const row = (samples === count ? i : Math.floor(i * count / samples)) * MESHLET_DATA_U32S;
        h = Math.imul(h ^ source[row + M.PARENT], 16777619);
        h = Math.imul(h ^ source[row + M.LOD_LEVEL], 16777619);
        h = Math.imul(h ^ source[row + M.TRIANGLE_COUNT], 16777619);
        h = Math.imul(h ^ source[row + M.PAGE], 16777619);
        h = Math.imul(h ^ source[row + M.CLUSTER_ERROR], 16777619);
    }
    return h >>> 0;
}

/**
 * Builds a primitive's cut tables from its records (see {@link CutTable}). Pages are taken as
 * the records hold them: primitive-local for a table that will be saved, or already rebased for
 * one that is used at once.
 *
 * @param {Uint32Array} source - The primitive's meshlet records.
 * @returns {CutTable} The tables.
 * @ignore
 */
function buildCutTable(source) {
    const count = source.length / MESHLET_DATA_U32S;
    const words = new U32Builder();
    const levelSet = new Set();
    const floorPages = new Map();
    const birth = new Uint32Array(count).fill(MESHLET_NO_PARENT);
    const { groups, roots } = buildMeshletGroups(source);
    collectFloorPages(source, groups, floorPages);
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
            birth[m] = group.start;
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
    let cursor = 0;
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
        words.push(group.start, parents.size, offset + 8 + parents.size * 3, pages.size,
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
    const rootPages = new Set();
    for (const root of roots) rootPages.add(source[root * MESHLET_DATA_U32S + M.PAGE]);
    const rootOffset = words.length;
    // root header: word 5 carries the draw bucket (there is no coarser cost to hold),
    // so the root admission needs no meshlet record; it is a placeholder here, stamped by
    // relocateCutTable. After the pages: the tasks that depend on the roots alone, which the
    // admission enqueues. Last, the roots in lists of MESHLET_CUT_ROOT_CHUNK (the chunk
    // count, then each list's count and primitive-local indices), which the admission queues
    // for the emit pass.
    words.push(0, 0, rootOffset + 8, rootPages.size, rootCost[0], 0, rootCost[1], 0);
    for (const page of rootPages) words.push1(page);
    words.push1(rootOwned.length);
    for (const task of rootOwned) words.push1(task);
    words.push1(Math.ceil(roots.length / MESHLET_CUT_ROOT_CHUNK));
    for (let r = 0; r < roots.length; r += MESHLET_CUT_ROOT_CHUNK) {
        const chunk = roots.slice(r, r + MESHLET_CUT_ROOT_CHUNK);
        words.push1(chunk.length);
        for (const root of chunk) words.push1(root);
    }

    const floor = new Uint32Array(floorPages.size * 2);
    let f = 0;
    for (const [page, level] of floorPages) {
        floor[f++] = page;
        floor[f++] = level;
    }
    return {
        count,
        words: words.view().slice(),
        rootOffset,
        levels: Uint32Array.from(levelSet),
        floor,
        birth,
        fingerprint: fingerprintRecords(source)
    };
}

/**
 * Appends a table's words to a world's cut words, turning its relative offsets into absolute
 * ones and its primitive-local pages and meshlet indices into the world's.
 *
 * @param {CutTable} table - The table.
 * @param {U32Builder} out - The world's cut words.
 * @param {number} first - The world-global index of the primitive's first meshlet.
 * @param {number} pageBase - Added to the table's pages (zero when they are already rebased).
 * @param {number} bucket - The primitive's draw bucket, for the root header.
 * @returns {number} The root header's offset in `out`.
 * @ignore
 */
function relocateCutTable(table, out, first, pageBase, bucket) {
    const length = table.words.length;
    out._reserve(length);
    const base = out.length;
    out.data.set(table.words, base);
    out.length += length;
    const w = out.data;

    // the group rows, back to back
    for (let o = base; o < base + table.rootOffset;) {
        const parents = w[o + 1];
        const pageCount = w[o + 3];
        w[o] += first;
        w[o + 2] += base;
        const pagesAt = o + 8 + parents * 3;
        for (let i = 0; i < pageCount; i++) w[pagesAt + i] += pageBase;
        const ownedAt = pagesAt + pageCount;
        const ownedCount = w[ownedAt];
        for (let i = 1; i <= ownedCount; i++) w[ownedAt + i] += base;
        const childAt = ownedAt + 1 + ownedCount;
        o = childAt + 1 + w[childAt];
    }

    const root = base + table.rootOffset;
    w[root] = first;
    w[root + 2] += base;
    w[root + 5] = bucket;
    const pageCount = w[root + 3];
    for (let i = 0; i < pageCount; i++) w[root + 8 + i] += pageBase;
    const ownedAt = root + 8 + pageCount;
    const ownedCount = w[ownedAt];
    for (let i = 1; i <= ownedCount; i++) w[ownedAt + i] += base;
    return root;
}

const TABLE_HEADER_WORDS = 6;
const TABLES_MAGIC = 0x31545543; // 'CUT1'

/**
 * Packs a resource's cut tables, one slot per primitive (null where none was built), into one
 * binary blob for the byte cache.
 *
 * @param {Array<CutTable|null>} tables - Per primitive, its table or null.
 * @returns {Uint32Array} The blob.
 * @ignore
 */
function serializeCutTables(tables) {
    const directory = 2 + tables.length * 2;
    let total = 1 + directory;
    const sizes = tables.map((t) => {
        if (!t) return 0;
        const size = TABLE_HEADER_WORDS + t.levels.length + t.floor.length + t.birth.length + t.words.length;
        total += size;
        return size;
    });
    const out = new Uint32Array(total);
    out[0] = TABLES_MAGIC;
    out[1] = tables.length;
    let at = 1 + directory;
    tables.forEach((t, i) => {
        out[2 + i * 2] = t ? at : 0;
        out[3 + i * 2] = sizes[i];
        if (!t) return;
        out[at] = t.count;
        out[at + 1] = t.words.length;
        out[at + 2] = t.rootOffset;
        out[at + 3] = t.levels.length;
        out[at + 4] = t.floor.length / 2;
        out[at + 5] = t.fingerprint;
        let p = at + TABLE_HEADER_WORDS;
        out.set(t.levels, p); p += t.levels.length;
        out.set(t.floor, p); p += t.floor.length;
        out.set(t.birth, p); p += t.birth.length;
        out.set(t.words, p);
        at += sizes[i];
    });
    return out;
}

/**
 * Reads back what {@link serializeCutTables} wrote, as views into the blob.
 *
 * @param {ArrayBuffer} buffer - The blob.
 * @param {number} primitiveCount - Primitives the resource has.
 * @returns {Array<CutTable|null>|null} Per primitive its table or null, or null when the blob
 * is not a valid one for a resource of that many primitives.
 * @ignore
 */
function deserializeCutTables(buffer, primitiveCount) {
    if (buffer.byteLength < 8 || buffer.byteLength % 4) return null;
    const data = new Uint32Array(buffer);
    if (data[0] !== TABLES_MAGIC || data[1] !== primitiveCount) return null;
    const tables = [];
    for (let i = 0; i < primitiveCount; i++) {
        const at = data[2 + i * 2];
        const size = data[3 + i * 2];
        if (!at) {
            tables.push(null);
            continue;
        }
        const count = data[at];
        const wordsLength = data[at + 1];
        const levelCount = data[at + 3];
        const floorPairs = data[at + 4];
        if (at + size > data.length || size !== TABLE_HEADER_WORDS + levelCount + floorPairs * 2 + count + wordsLength) return null;
        let p = at + TABLE_HEADER_WORDS;
        const levels = data.subarray(p, p + levelCount); p += levelCount;
        const floor = data.subarray(p, p + floorPairs * 2); p += floorPairs * 2;
        const birth = data.subarray(p, p + count); p += count;
        const words = data.subarray(p, p + wordsLength);
        tables.push({ count, words, rootOffset: data[at + 2], levels, floor, birth, fingerprint: data[at + 5] });
    }
    return tables;
}

export {
    U32Builder, buildMeshletGroups, collectFloorPages, buildCutTable, relocateCutTable,
    fingerprintRecords, serializeCutTables, deserializeCutTables
};
