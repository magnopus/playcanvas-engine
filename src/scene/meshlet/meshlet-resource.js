import { Debug } from '../../core/debug.js';
import { MESHLET_DATA, MESHLET_DATA_U32S, MESHLET_NO_PARENT, PAGE_TABLE_FIELDS } from './constants.js';
import { buildCutTable } from './meshlet-cut-table.js';
import { MeshletInstanceList } from './meshlet-instance-list.js';

/**
 * @import { MeshletInstanceListBuilder } from './meshlet-instance-list.js'
 */

/**
 * @typedef {object} MeshletLightmap
 * @property {{ arrayId: number, layer: number, texCoord: number }} texture - Streamed lightmap reference.
 * @property {number[]} coordinateScaleBias - Atlas UV scale and offset.
 * @property {number[]} lightmapScale - EPIC irradiance decode scale.
 * @property {number[]} lightmapAdd - EPIC irradiance decode offset.
 */

/**
 * One primitive's parsed MAG_meshlets_gpu v2 payload.
 *
 * @ignore
 */
class MeshletPrimitive {
    /**
     * meshletData records, MESHLET_DATA_U32S words per meshlet; null while released (see
     * {@link MeshletResource#releaseRecords}).
     *
     * @type {Uint32Array|null}
     */
    meshletData;

    /** @type {Float32Array|null} - f32 view aliasing meshletData. */
    meshletDataF32;

    /** @type {number} */
    meshletCount;

    /** @type {number} */
    uvChannelMask;

    /** @type {number[]} - local-space AABB center. */
    aabbCenter;

    /** @type {number[]} - local-space AABB half extents. */
    aabbHalfExtents;

    /** @type {number} - index of the material in the container, or -1. */
    materialIndex;

    /** @type {number[]} - the material's baseColorFactor rgba. */
    baseColorFactor;

    /** @type {number} - index of the gltf mesh this primitive belongs to. */
    meshIndex;

    /** @type {number} - index of the primitive within its gltf mesh. */
    primIndex;

    // What a world build reads of the records besides the copy it uploads, cached so a resource
    // whose records were released can still be laid out (see MeshletWorld#_countTotals).

    /** @type {number[]|null} - primitive-local pages of the root clusters. @ignore */
    _rootPages = null;

    /** @type {number} - indices of every meshlet, or -1 before the first build. @ignore */
    _indexCount = -1;

    /** @type {number} - the first meshlet's flags. @ignore */
    _firstFlags = 0;

    /**
     * Cut tables for the next world build to use (see MeshletCutData), built by
     * {@link MeshletResource#prepare} or loaded from the byte cache, or null. Dropped once the
     * build is done.
     *
     * @type {import('./meshlet-cut-table.js').CutTable|null}
     * @ignore
     */
    _cutTable = null;

    /**
     * Cut tables the last world build had to compute, waiting to be saved to the byte cache.
     *
     * @type {import('./meshlet-cut-table.js').CutTable|null}
     * @ignore
     */
    _builtCutTable = null;
}

/**
 * The primitive-local pages of a primitive's root clusters (those without a parent): what the
 * residency pins for as long as the primitive is placed. Kept on the primitive, with
 * {@link primitiveStats}, so a resource whose records were released can still be laid out.
 *
 * @param {MeshletPrimitive} prim - The primitive.
 * @returns {number[]} Its root pages (may repeat).
 * @ignore
 */
const primitiveRootPages = (prim) => {
    if (!prim._rootPages) {
        const data = prim.meshletData;
        const pages = [];
        for (let m = 0; m < prim.meshletCount; m++) {
            const row = m * MESHLET_DATA_U32S;
            if (data[row + MESHLET_DATA.PARENT] === MESHLET_NO_PARENT) pages.push(data[row + MESHLET_DATA.PAGE]);
        }
        prim._rootPages = pages;
    }
    return prim._rootPages;
};

/**
 * Caches a primitive's index count and its first meshlet's flags (see {@link primitiveRootPages}).
 *
 * @param {MeshletPrimitive} prim - The primitive.
 * @ignore
 */
const primitiveStats = (prim) => {
    if (prim._indexCount >= 0) return;
    const src = prim.meshletData;
    let triangles = 0;
    for (let m = 0; m < prim.meshletCount; m++) triangles += src[m * MESHLET_DATA_U32S + MESHLET_DATA.TRIANGLE_COUNT];
    prim._indexCount = triangles * 3;
    prim._firstFlags = prim.meshletCount ? src[MESHLET_DATA.FLAGS] : 0;
};

/**
 * Parsed streamed-meshlet asset: the resident per-meshlet records of every primitive plus the
 * MAG_meshlets_stream v2 manifest (binary page table, shard blob URIs, position grid). Created
 * by the glb parser; consumed by the meshlet world which uploads the records, streams pages and
 * renders. This class holds CPU data only.
 *
 * @ignore
 */
class MeshletResource {
    /** @type {MeshletPrimitive[]} */
    primitives;

    /**
     * The stream manifest.
     *
     * @type {{
     *     blobs: Array<{ uri: string, byteLength: number }>,
     *     pageTable: Uint32Array,
     *     pageCount: number,
     *     rootPages: number[],
     *     attributeLayout: object,
     *     pageSizeBytes: number,
     *     pageAlignment: number,
     *     positionGrid: { origin: number[], step: number, bits: number },
     *     sloppyFlags?: boolean,
     *     bakeId?: string|null
     * }}
     */
    manifest;

    /** @type {number} - total meshlets across primitives. */
    totalMeshlets;

    /**
     * Baked material table (root MAG_meshlets_gpu block): MATERIAL_RECORD_U32S words per glTF
     * material, indexed by {@link MeshletPrimitive#materialIndex}. Null when the bake carries no table - the
     * world synthesizes per-primitive records from baseColorFactor instead.
     *
     * @type {Uint32Array|null}
     */
    materialTable = null;

    /** @type {number} - records in {@link materialTable}. */
    materialCount = 0;

    /**
     * Streamed-texture manifest (MAG_texture_streaming root extension): container-packed
     * texture arrays with per-(layer, mip) byte ranges. Null on texture-less bakes.
     *
     * @type {{ arrays: object[] }|null}
     */
    textureManifest = null;

    /** @type {MeshletInstanceList} */
    _instances;

    /**
     * Document-space placements: one per (gltf node or scatter entry, primitive) pair. Set
     * either a list or an array of `{ primIndex, matrix, lightmap }` placements, which is packed
     * into a list.
     *
     * @type {MeshletInstanceList}
     */
    set instances(value) {
        this._instances = MeshletInstanceList.from(value);
    }

    get instances() {
        return this._instances;
    }

    /**
     * @param {MeshletPrimitive[]} primitives - Parsed primitives.
     * @param {object} manifest - The stream manifest.
     * @param {MeshletInstanceList|MeshletInstanceListBuilder|Array<{ primIndex: number, matrix: Float32Array, lightmap?: MeshletLightmap|null }>} [instances] - Placements.
     */
    constructor(primitives, manifest, instances = null) {
        this.primitives = primitives;
        this.manifest = manifest;
        this.instances = instances;
        this.totalMeshlets = primitives.reduce((n, p) => n + p.meshletCount, 0);

        Debug.assert(manifest.pageTable.length === manifest.pageCount * PAGE_TABLE_FIELDS,
            'MeshletResource: page table length does not match pageCount');
        Debug.assert(primitives.every(p => p.meshletData.length === p.meshletCount * MESHLET_DATA_U32S),
            'MeshletResource: meshletData length does not match meshletCount');
    }

    destroy() {
        this.primitives = [];
        this.manifest = null;
    }

    /**
     * Whether every primitive's meshlet records are in memory. A world reads them only while it
     * builds (it uploads its own rebased copy), so they can be released between builds - see
     * {@link releaseRecords}.
     *
     * @type {boolean}
     */
    get recordsResident() {
        return this.primitives.every(p => !!p.meshletData);
    }

    /**
     * Drops every primitive's meshlet records - views into the parsed GLB, and almost all of its
     * bytes - so the file's buffer can be collected. A large scene otherwise holds a second copy
     * of every record it has uploaded. Restore them with {@link restoreRecords} before the
     * resource joins another world build.
     */
    releaseRecords() {
        for (const prim of this.primitives) {
            prim.meshletData = null;
            prim.meshletDataF32 = null;
        }
    }

    /**
     * Restores the records {@link releaseRecords} dropped, one array per primitive.
     *
     * @param {Uint32Array[]} records - Each primitive's records, in primitive order.
     */
    restoreRecords(records) {
        Debug.assert(records.length === this.primitives.length, 'MeshletResource: restored record count does not match the primitives');
        this.primitives.forEach((prim, i) => {
            Debug.assert(records[i].length === prim.meshletCount * MESHLET_DATA_U32S, 'MeshletResource: restored records do not match the primitive');
            prim.meshletData = records[i];
            prim.meshletDataF32 = new Float32Array(records[i].buffer, records[i].byteOffset, records[i].length);
        });
    }

    /**
     * Derives, from the records alone, what a world build would otherwise derive itself: each
     * primitive's root pages, index count and cut tables. Called as each resource arrives, so a
     * large scene does this while the rest of it downloads, rather than all at once inside the
     * build that waits for the last container. A primitive whose tables cannot be built (an
     * invalid DAG) is left to the build, which reports it.
     */
    prepare() {
        for (const prim of this.primitives) {
            if (!prim.meshletData) continue;
            primitiveRootPages(prim);
            primitiveStats(prim);
            if (!prim._cutTable) {
                try {
                    // built here rather than loaded, so the system's cut-table store saves it
                    prim._cutTable = prim._builtCutTable = buildCutTable(prim.meshletData);
                } catch (err) {
                    prim._cutTable = prim._builtCutTable = null;
                }
            }
        }
    }
}

export { MeshletResource, MeshletPrimitive, primitiveRootPages, primitiveStats };
