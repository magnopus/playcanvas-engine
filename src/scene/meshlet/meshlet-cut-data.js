import { StorageBuffer } from '../../platform/graphics/storage-buffer.js';
import { BUFFERUSAGE_COPY_DST } from '../../platform/graphics/constants.js';
import {
    MESHLET_DATA as M, MESHLET_DATA_U32S, MESHLET_NO_PARENT,
    MESHLET_FLAG_ALPHA_MASKED, MESHLET_FLAG_TWO_SIDED, OBJECT_DATA as O, OBJECT_DATA_U32S
} from './constants.js';

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
            for (const field of [M.CLUSTER_ERROR, M.GROUP_SPHERE, M.GROUP_SPHERE + 1, M.GROUP_SPHERE + 2, M.GROUP_SPHERE + 3]) {
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

/** GPU tables shared by every view's resident, capacity-bounded cut. @ignore */
class MeshletCutData {
    /**
     * @param {import('../../platform/graphics/graphics-device.js').GraphicsDevice} device - Device.
     * @param {Uint32Array} meshlets - World records; reserved words receive primitive-local birth representatives.
     * @param {Uint32Array} objects - World instance records.
     */
    constructor(device, meshlets, objects) {
        const words = [];
        const levels = [];
        const rootTasks = [];
        const primitives = new Map();
        const rootPages = new Set();
        this.rootIndices = [0, 0, 0];
        this.rootRecords = 0;
        this.instanceRoots = [];
        // Per primitive, in its meshlet index range: [first] = the lowest LOD level holding a root
        // (terminal groups leave roots below the top), [first + level] = where that level starts
        // (level-major order). Compaction starts each instance at the finer of its cut's finest
        // level and that root level, instead of scanning every LOD.
        const levelStarts = new Uint32Array(meshlets.length / MESHLET_DATA_U32S);
        for (let instance = 0; instance < objects.length / OBJECT_DATA_U32S; instance++) {
            const first = objects[instance * OBJECT_DATA_U32S + O.FIRST_MESHLET];
            const count = objects[instance * OBJECT_DATA_U32S + O.MESHLET_COUNT];
            let primitive = primitives.get(first);
            if (!primitive) {
                const source = meshlets.subarray(first * MESHLET_DATA_U32S, (first + count) * MESHLET_DATA_U32S);
                // Older foliage bakes cap real geometry with synthetic empty roots. Empty
                // geometry cannot cover a missing page up close: retain and pin the real
                // coarse level instead. Such bakes consequently need a larger resident set.
                for (let m = 0; m < count; m++) {
                    const row = m * MESHLET_DATA_U32S;
                    const parent = source[row + M.PARENT];
                    if (parent < count && source[parent * MESHLET_DATA_U32S + M.PARENT] === MESHLET_NO_PARENT &&
                        source[parent * MESHLET_DATA_U32S + M.TRIANGLE_COUNT] === 0) {
                        source[row + M.PARENT] = MESHLET_NO_PARENT;
                    }
                }
                const { groups, roots } = buildMeshletGroups(source);
                let minRootLevel = Infinity;
                for (const root of roots) minRootLevel = Math.min(minRootLevel, source[root * MESHLET_DATA_U32S + M.LOD_LEVEL]);
                for (let m = count - 1; m > 0; m--) {
                    const level = source[m * MESHLET_DATA_U32S + M.LOD_LEVEL];
                    if (level !== source[(m - 1) * MESHLET_DATA_U32S + M.LOD_LEVEL]) levelStarts[first + level] = m;
                }
                levelStarts[first] = Number.isFinite(minRootLevel) ? minRootLevel : 0;
                for (const root of roots) rootPages.add(source[root * MESHLET_DATA_U32S + M.PAGE]);
                for (let m = 0; m < count; m++) source[m * MESHLET_DATA_U32S + M.BIRTH_GROUP] = MESHLET_NO_PARENT;
                const tasks = [];
                const costs = (members) => {
                    let indices = 0, records = 0;
                    for (const m of members) {
                        const triangles = source[m * MESHLET_DATA_U32S + M.TRIANGLE_COUNT];
                        indices += triangles * 3;
                        if (triangles) records++;
                    }
                    return [indices, records];
                };
                for (const group of groups) {
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
                    const fineCost = costs(group.children), coarseCost = costs(coarse);
                    const offset = words.length;
                    // Header: representative, dependency count, page-list offset/count,
                    // fine/coarse index and record costs. Lists contain unique parent
                    // representatives and pages, avoiding serial scans over every meshlet.
                    // Each parent also carries the coarse cost it contributed: an off-frustum
                    // parent refines without charging its children, so a replacement must not
                    // subtract those uncharged members from its own charge.
                    words.push(first + group.start, parents.size, offset + 8 + parents.size * 3, pages.size,
                        fineCost[0], coarseCost[0], fineCost[1], coarseCost[1]);
                    for (const [parent, members] of parents) words.push(parent, ...costs(members));
                    for (const page of pages) words.push(page);
                    tasks.push({ level: group.level, offset });
                }
                const rootCost = costs(roots);
                const pages = new Set();
                for (const root of roots) pages.add(source[root * MESHLET_DATA_U32S + M.PAGE]);
                const rootOffset = words.length;
                words.push(first, 0, rootOffset + 8, pages.size, rootCost[0], 0, rootCost[1], 0);
                for (const page of pages) words.push(page);
                const flags = source[M.FLAGS];
                const bucket = flags & MESHLET_FLAG_ALPHA_MASKED ? 2 : (flags & MESHLET_FLAG_TWO_SIDED ? 1 : 0);
                primitive = { tasks, rootOffset, rootCost, bucket, admission: { pages: Array.from(pages), cost: rootCost, bucket } };
                primitives.set(first, primitive);
            }
            rootTasks.push(instance, primitive.rootOffset);
            this.instanceRoots.push(primitive.admission);
            this.rootIndices[primitive.bucket] += primitive.rootCost[0];
            this.rootRecords += primitive.rootCost[1];
            for (const task of primitive.tasks) {
                const list = levels[task.level] ?? (levels[task.level] = []);
                list.push(instance, task.offset);
            }
        }
        const tasks = rootTasks;
        this.levels = [{ start: 0, count: rootTasks.length / 2, root: true }];
        for (let level = levels.length - 1; level > 0; level--) {
            const list = levels[level];
            if (!list?.length) continue;
            this.levels.push({ start: tasks.length / 2, count: list.length / 2, root: false });
            for (const word of list) tasks.push(word);
        }
        const create = (values) => {
            const buffer = new StorageBuffer(device, Math.max(values.length * 4, 16), BUFFERUSAGE_COPY_DST);
            if (values.length) buffer.write(0, new Uint32Array(values));
            return buffer;
        };
        this.groups = create(words);
        this.tasks = create(tasks);
        // Compaction scans every LOD. Keep its two links contiguous instead of fetching
        // them from 128-byte geometry records for every instance of a primitive.
        const topology = new Uint32Array(meshlets.length / MESHLET_DATA_U32S * 2);
        for (let m = 0; m < topology.length / 2; m++) {
            topology[m * 2] = meshlets[m * MESHLET_DATA_U32S + M.PARENT];
            topology[m * 2 + 1] = meshlets[m * MESHLET_DATA_U32S + M.BIRTH_GROUP];
        }
        this.selectionTopology = new StorageBuffer(device, Math.max(topology.byteLength, 16), BUFFERUSAGE_COPY_DST);
        if (topology.length) this.selectionTopology.write(0, topology);
        this.levelStarts = new StorageBuffer(device, Math.max(levelStarts.byteLength, 16), BUFFERUSAGE_COPY_DST);
        if (levelStarts.length) this.levelStarts.write(0, levelStarts);
        this.levelStartsCpu = levelStarts;
        this.rootPages = Array.from(rootPages).sort((a, b) => a - b);
    }

    destroy() {
        this.groups.destroy();
        this.tasks.destroy();
        this.selectionTopology.destroy();
        this.levelStarts.destroy();
    }
}

export { buildMeshletGroups, MeshletCutData };
