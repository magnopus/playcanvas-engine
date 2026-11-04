import { Debug } from '../../core/debug.js';
import { MESHLET_DATA_U32S, PAGE_TABLE_FIELDS } from './constants.js';

/**
 * @typedef {object} MeshletLightmap
 * @property {{ arrayId: number, layer: number, texCoord: number }} texture - Streamed lightmap reference.
 * @property {number[]} coordinateScaleBias - Atlas UV scale and offset.
 * @property {number[]} lightmapScale - EPIC irradiance decode scale.
 * @property {number[]} lightmapAdd - EPIC irradiance decode offset.
 */

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
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

    /** @type {number} */
    vertexCount;

    /** @type {Array<{ lodLevel: number, meshletStart: number, meshletCount: number }>} */
    lods;

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

    /** @type {string|null} - name of the gltf mesh this primitive belongs to. */
    meshName = null;

    /** @type {boolean} - true for the first meshlet primitive of its gltf mesh. */
    firstOfMesh = false;
}

/**
 * Parsed streamed-meshlet asset: the resident per-meshlet records of every primitive plus the
 * MAG_meshlets_stream v2 manifest (binary page table, shard blob URIs, position grid). Created
 * by the glb parser; consumed by the meshlet world which uploads the records, streams pages and
 * renders. This class holds CPU data only.
 *
 * @ignore
 */
class MeshletResource {
    /** @type {GraphicsDevice} */
    device;

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

    /**
     * Document-space placements: one entry per (gltf node, primitive) pair.
     *
     * @type {Array<{ primIndex: number, matrix: Float32Array, lightmap?: MeshletLightmap|null }>}
     */
    instances;

    /**
     * @param {GraphicsDevice} device - The graphics device.
     * @param {MeshletPrimitive[]} primitives - Parsed primitives.
     * @param {object} manifest - The stream manifest.
     * @param {Array<{ primIndex: number, matrix: Float32Array, lightmap?: MeshletLightmap|null }>} [instances] - Placements.
     */
    constructor(device, primitives, manifest, instances = []) {
        this.device = device;
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
}

export { MeshletResource, MeshletPrimitive };
