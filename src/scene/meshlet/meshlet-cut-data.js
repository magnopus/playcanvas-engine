import { StorageBuffer } from '../../platform/graphics/storage-buffer.js';
import { BUFFERUSAGE_COPY_DST } from '../../platform/graphics/constants.js';
import { MESHLET_DATA as M, MESHLET_DATA_U32S } from './constants.js';
import { bucketOfMeshletFlags } from './meshlet-material-table.js';
import {
    U32Builder, buildCutTable, buildMeshletGroups, collectFloorPages, relocateCutTable
} from './meshlet-cut-table.js';

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { CutTable } from './meshlet-cut-table.js'
 */

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
 * @property {(first: number, source: Uint32Array) => ({ table: CutTable, pageBase: number }|null)} [cutTable] -
 * The primitive's cut tables in primitive-local terms, and the page base to add to their pages,
 * when the source keeps them (built or loaded earlier); null or absent builds them from the
 * records `get` returned.
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
                const kept = records.cutTable?.(first, source) ?? null;
                // records the world rebased are built from as they are, so no page base remains
                const table = kept?.table ?? buildCutTable(source);
                const pageBase = kept?.pageBase ?? 0;
                for (let m = 0; m < count; m++) source[m * MESHLET_DATA_U32S + M.BIRTH_GROUP] = table.birth[m];
                for (const level of table.levels) levelSet.add(level);
                for (let f = 0; f < table.floor.length; f += 2) {
                    const page = table.floor[f] + pageBase;
                    if (!(floorPages.get(page) >= table.floor[f + 1])) floorPages.set(page, table.floor[f + 1]);
                }
                rootOffset = relocateCutTable(table, words, first, pageBase, bucketOfMeshletFlags(source[M.FLAGS]));
                for (let m = 0; m < count; m++) {
                    topology[(first + m) * 2] = source[m * MESHLET_DATA_U32S + M.PARENT];
                    topology[(first + m) * 2 + 1] = table.birth[m];
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
