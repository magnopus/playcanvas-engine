import { Debug } from '../../core/debug.js';
import { PickerId } from '../picker-id.js';
import { Mat4 } from '../../core/math/mat4.js';
import { BoundingBox } from '../../core/shape/bounding-box.js';
import {
    BUFFERUSAGE_COPY_DST, BUFFERUSAGE_COPY_SRC
} from '../../platform/graphics/constants.js';
import { StorageBuffer } from '../../platform/graphics/storage-buffer.js';
import {
    MATERIAL_FLAG_ALPHA_MASK, MATERIAL_FLAG_DOUBLE_SIDED, MATERIAL_RECORD, MATERIAL_RECORD_U32S, MATERIAL_SLOT_ABSENT,
    MATERIAL_TEXTURE_SLOTS, MESHLET_BUCKET_COUNT, MESHLET_BUCKET_MASKED, MESHLET_BUCKET_OPAQUE,
    MESHLET_BUCKET_OPAQUE_TWO_SIDED, MESHLET_CULL_SLICE, MESHLET_DATA, MESHLET_DATA_U32S, MESHLET_FLAG_ALPHA_MASKED,
    MESHLET_FLAG_TWO_SIDED, MESHLET_MAX_UV_CHANNELS, OBJECT_DATA, OBJECT_DATA_U32S, OBJECT_FLAG_HAS_TANGENTS,
    OBJECT_FLAG_HIDDEN, OBJECT_FLAG_HOVERED, OBJECT_FLAG_OUTLINED, PAGE_NOT_RESIDENT, PAGE_TABLE, PAGE_TABLE_FIELDS,
    RECORD_U32S, TEXEL_RATE_PER_MIP, WORK_ITEM_U32S
} from './constants.js';
import { createMeshletLitMaterial } from './meshlet-lit-material.js';
import { createMeshletMaterial } from './meshlet-material.js';
import { buildMeshletLitChunks } from './shaders/meshlet-lit-chunks-wgsl.js';
import { MeshletTextures, MESHLET_TEX_FAMILIES } from './textures/meshlet-textures.js';

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { MeshletResource } from './meshlet-resource.js'
 */

const _tmpMat = new Mat4();
const _tmpBox = new BoundingBox();
const _tmpLocalBox = new BoundingBox();

// largest axis scale of a column-major matrix - what a local-space radius or error must be
// multiplied by to bound its world-space size
const maxAxisScale = m => Math.max(Math.hypot(m[0], m[1], m[2]), Math.hypot(m[4], m[5], m[6]), Math.hypot(m[8], m[9], m[10]));

// material parameter names for a texture family's tail / fine-pool array bindings
const _familyName = family => `${MESHLET_TEX_FAMILIES[family].charAt(0).toUpperCase()}${MESHLET_TEX_FAMILIES[family].slice(1)}`;
const _tailBindName = family => `meshletTail${_familyName(family)}`;
const _fineBindName = family => `meshletFine${_familyName(family)}`;

/**
 * The GPU residency and instance state of a set of streamed meshlet assets: page pool, resident
 * meshletData records (rebased into one global meshlet space), per-instance objectData, material
 * palette and the frame's record/index working buffers plus the draw mesh instances.
 *
 * Build is one-shot: add resources, then {@link finalize}. Incremental component add/remove goes
 * through {@link MeshletDirector#rebuild}. Per-placement transforms may change after finalize
 * via {@link setPlacementTransform}. Resources may differ in position grid and page attribute
 * layout (carried per instance in objectData); all must share one page size.
 *
 * @ignore
 */
class MeshletWorld {
    /** @type {GraphicsDevice} */
    device;

    /** @type {Array<{ resource: MeshletResource, transform: Mat4|null, shardBuffers: ArrayBuffer[] }>} */
    _pending = [];

    finalized = false;

    // GPU buffers
    pagePool = null;

    residencyBuffer = null;

    /** @type {StorageBuffer|null} - 16-byte placeholder bound as visBits by single-phase views. */
    dummyBits = null;

    /** @type {number} - first id of this world's reserved per-instance picker block. */
    pickIdBase = 0;

    /** @type {Array<object>|null} - lazily built; see {@link pickRecords}. */
    _pickRecords = null;

    /** @type {Set<number>} - instance indices flagged as selected. */
    _outlined = new Set();

    /** @type {Set<number>} - instance indices flagged as hovered. */
    _hovered = new Set();

    meshletDataBuffer = null;

    objectDataBuffer = null;

    materialTableBuffer = null;

    /** @type {import('./textures/meshlet-textures.js').MeshletTextures|null} */
    textures = null;

    // capacities (index capacities are the initial per-view values; each MeshletView owns
    // its own demand-grown copy)
    instanceCount = 0;

    workItemCapacity = 0;

    recordCapacity = 0;

    /** @type {number[]} - initial per-view index capacity, per draw bucket (MESHLET_BUCKET_*). */
    indexCapacity = [0, 0, 0];

    /** @type {number[]} - worst case per bucket (every meshlet of every instance drawn). */
    indexWorst = [0, 0, 0];

    /**
     * Initial per-view record allocation. recordCapacity above is the worst case (every
     * meshlet of every instance drawn at once), which scattered scenes blow far past; views
     * start here and grow to the demand frames actually show. 0 = allocate the worst case.
     *
     * @type {number}
     */
    initialRecords = 1 << 18;

    // format (all resources in a world share one page size; attribute layouts may differ,
    // carried per instance in objectData)
    pageSizeBytes = 0;

    /** @type {BoundingBox} */
    worldBounds = new BoundingBox();

    /**
     * @param {GraphicsDevice} device - The graphics device.
     */
    constructor(device) {
        this.device = device;
    }

    destroy() {
        this.pagePool?.destroy();
        this.residencyBuffer?.destroy();
        this.requestsBuffer?.destroy();
        this.dummyBits?.destroy();
        this.meshletDataBuffer?.destroy();
        this.objectDataBuffer?.destroy();
        this.materialTableBuffer?.destroy();
        // bucket materials are shared across the views' mesh instances - destroy each set once
        this._litMaterials?.forEach(m => m.destroy());
        this._litMaterials = null;
        this._debugMaterials?.forEach(m => m.destroy());
        this._debugMaterials = null;
        this.textures?.destroy();
        this.textures = null;
    }

    /**
     * Total geometry-side VRAM budget in bytes, or 0 for unbudgeted (size the page pool to hold
     * every page). This covers everything the meshlet pipeline allocates for geometry - the
     * page pool, the world tables (meshletData, objectData), and the per-view working set
     * (claim/visibility bit planes, work items, records, draw indices) - not just the page pool.
     *
     * Budgeting only the pool is misleading: in one example, a 32 MB pool sat inside 1.3 GB of
     * VRAM, because the buffers that actually dominate were demand-grown with no ceiling. The
     * fixed and per-view costs are computed at finalize and subtracted; what remains is split
     * between the page pool and the draw index buffers by {@link indexBudgetFraction}, and the
     * index share becomes {@link maxIndices}, so growth is bounded by the budget rather than by
     * the device.
     *
     * @type {number}
     */
    poolBytes = 0;

    /**
     * Share of the budget (after fixed and per-view costs) given to the demand-grown draw index
     * buffers; the rest goes to the page pool. Indices are the working set of what is ON SCREEN,
     * pages the working set of what is NEARBY, so the split trades detail-in-view against
     * streaming headroom.
     *
     * @type {number}
     */
    indexBudgetFraction = 0.35;

    /** @type {number} - camera views to budget the per-view working set for. */
    budgetCameraViews = 1;

    /** @type {number} - shadow cascade views to budget for (they share more, so cost less). */
    budgetShadowViews = 0;

    /** @type {number} - page-pool bytes resolved from the budget at finalize. */
    pagePoolBytes = 0;

    /**
     * Total draw indices the budget allows ACROSS all views. This is a pool, not a per-view
     * allowance: the director hands each view a share proportional to what it is asking for.
     * Splitting it equally starves whichever view happens to be the hungry one - on the Bistro
     * a shadow cascade wanted 1.2x an equal share while the camera used 0.16x of its own, and
     * the starved cascade then drove global LOD pressure to maximum for everything.
     *
     * @type {number}
     */
    indexBudgetTotal = 0;

    /**
     * Extra indices granted above {@link indexBudgetTotal} when the budget turns out to be
     * unachievable - see {@link MeshletDirector#budgetOverrunStep}. Capped by the device.
     *
     * @type {number}
     */
    indexOverrun = 0;

    /** @type {{ budget: number, fixed: number, perView: number, pagePool: number, indices: number }|null} */
    budgetBreakdown = null;

    /** @type {number} - fine-texture slot-pool byte budget (see MeshletTextures#poolBytes). */
    texturePoolBytes = 96 * 1024 * 1024;

    /**
     * KTX2 transcoder handed to the texture system (see {@link MeshletDirector#transcode}).
     * Scene code cannot import the framework's Basis transcoder, so it arrives by injection.
     *
     * @type {import('./textures/meshlet-ktx2.js').MeshletTranscodeFn|null}
     */
    transcode = null;

    /**
     * Upload budget per frame for freshly streamed pages (see
     * {@link MeshletResidency#maxInstallBytesPerFrame}). Streamed arrivals are lumpy, so they
     * install over several frames rather than spiking one. Raise to fill faster, lower for
     * smoother frames.
     *
     * @type {number}
     */
    maxInstallBytesPerFrame = 4 * 1024 * 1024;

    /**
     * Optional starting size of the GPU-written draw index buffer, in indices (0 = exact worst
     * case: the sum of every instance's meshlet corners). The worst case is unreachable for any
     * real camera - the DAG cut bounds the drawn set by screen area - so large worlds and heavy
     * instancing should start small: the director monitors the cull shader's demand counters
     * and grows the buffer when a frame wants more (a frame or two of clamped overflow while
     * the readback is in flight, then stable).
     *
     * @type {number}
     */
    initialIndices = 0;

    /**
     * Optional ceiling on the draw index buffer, in indices. 0 means "whatever the device can
     * hold" - see {@link indexCeiling}, which is what actually bounds growth. NOT unbounded:
     * the worst case on a scattered scene is billions of indices, and a request that large
     * throws mid-frame, taking the encoding down with it.
     *
     * @type {number}
     */
    maxIndices = 0;

    /**
     * Hard ceiling the device itself imposes, in indices - the smaller of the maximum buffer
     * size and the maximum storage binding (the index buffer is bound as both). Established at
     * finalize; {@link maxIndices} can only lower it.
     *
     * @type {number}
     */
    deviceIndexCeiling = 0;

    /** @type {number} - total (instance, meshlet) pairs, sizing the claim bitmask. */
    totalPairs = 0;

    /** @type {number} - total pages across resources. */
    totalPages = 0;

    /** @type {boolean} - true when any resource streams (pages fetched on demand). */
    streamed = false;

    /**
     * Per-resource streaming info: { resource, pageBase, baseUrl }.
     *
     * @type {Array<object>}
     */
    streamInfo = [];

    /**
     * Per added-resource placement info, in add order. `instances` is the effective placement
     * list (the add-time override or the resource's own); `instanceBase`/`instanceCount` are the
     * objectData row range for {@link setPlacementTransform}.
     *
     * @type {Array<{ resource: MeshletResource, instances: Array<object>, instanceBase: number,
     *     instanceCount: number }>}
     */
    placements = [];

    /**
     * Per added-resource identity, in add order - what the director's retention path compares
     * across rebuilds: `{ resource, baseUrl, streamed, pageCount }`.
     *
     * @type {Array<object>}
     */
    buildEntries = [];

    // Retention (set by MeshletDirector.rebuild before finalize): a compatible previous world's
    // page pool buffer is adopted instead of recreated, its residency map prefix is carried
    // (adoptedPages entries), its persistent visibility bits prefix is GPU-copied (placements
    // with unchanged instance lists - else phase 2 re-discovers the whole frustum and the page
    // demand burst evicts the hot set), and its texture system is reused wholesale when the
    // streamed texture set is unchanged.
    adoptPagePool = null;

    adoptResidency = null;

    adoptedPages = 0;

    adoptTextures = null;

    /** @type {import('../../platform/graphics/storage-buffer.js').StorageBuffer|null} */
    adoptVisBits = null;

    adoptVisBitsWords = 0;

    /** @type {number} - resources added but not yet finalized. */
    get pendingCount() {
        return this._pending.length;
    }

    /**
     * Adds a resource with all its shard data (fully resident).
     *
     * @param {MeshletResource} resource - The parsed asset.
     * @param {Mat4|null} transform - Root transform applied to the asset's placements.
     * @param {ArrayBuffer[]} shardBuffers - The asset's shard blobs, in manifest order.
     * @param {Array<{ primIndex: number, matrix: Float32Array }>|null} [instances] - Optional
     * placement list replacing the resource's own - many instances share one set of pages.
     */
    addResource(resource, transform, shardBuffers, instances = null) {
        Debug.assert(!this.finalized, 'MeshletWorld: addResource after finalize');
        this._pending.push({ resource, transform, shardBuffers, instances });
    }

    /**
     * Adds a resource whose pages stream on demand over HTTP Range requests.
     *
     * @param {MeshletResource} resource - The parsed asset.
     * @param {Mat4|null} transform - Root transform applied to the asset's placements.
     * @param {string} baseUrl - URL directory the manifest's blob URIs are relative to.
     * @param {Array<{ primIndex: number, matrix: Float32Array }>|null} [instances] - Optional
     * placement list replacing the resource's own - many instances share one set of pages.
     */
    addStreamedResource(resource, transform, baseUrl, instances = null) {
        Debug.assert(!this.finalized, 'MeshletWorld: addStreamedResource after finalize');
        this.streamed = true;
        this._pending.push({ resource, transform, baseUrl, instances });
    }

    /**
     * Splits {@link poolBytes} into a page-pool size and an index ceiling.
     *
     * The fixed costs (world tables) and the per-view working set are exactly computable from
     * the scene's counts, so they come off the top; what remains is divided between the page
     * pool and the draw index buffers. A shadow view costs far less than a camera one - it
     * borrows the claim plane and work items from its siblings and binds a placeholder for the
     * visibility bits it never touches - so the two are counted separately.
     *
     * @param {object} counts - Scene totals.
     * @returns {number} Bytes for the page pool (0 = unbudgeted, size to every page).
     * @private
     */
    _resolveBudget(counts) {
        const { totalPages, totalMeshlets, totalInstances, totalMaterialRows, pairs, workItems } = counts;
        if (!(this.poolBytes > 0)) {
            this.budgetBreakdown = null;
            this.maxIndices = 0;
            return 0;
        }

        const pairWords = Math.max(Math.ceil(pairs / 32), 4);
        const fixed = totalMeshlets * MESHLET_DATA_U32S * 4 +      // meshletData
            totalInstances * OBJECT_DATA_U32S * 4 +                // objectData
            totalPages * 4 +                                       // residency
            (totalPages + totalMaterialRows) * 4 +                 // requests + texel-rate marks
            totalMaterialRows * MATERIAL_RECORD_U32S * 4;          // material table
        const workItemBytes = Math.max(workItems * WORK_ITEM_U32S * 4, 16);
        const recordBytes = Math.min(this.initialRecords || pairs, pairs) * RECORD_U32S * 4;
        // camera view: own claim plane, own visibility bits, own work items, own records
        const perCamera = pairWords * 4 * 2 + workItemBytes + recordBytes;
        // shadow view: claim plane and work items are shared across cascades, visibility bits
        // are a 16-byte placeholder
        const perShadow = recordBytes;
        const shared = this.budgetShadowViews > 0 ? pairWords * 4 + workItemBytes : 0;

        const views = Math.max(this.budgetCameraViews, 1);
        const reserved = fixed + perCamera * views + perShadow * this.budgetShadowViews + shared;
        let available = this.poolBytes - reserved;

        // A budget that cannot even hold the fixed cost is not a budget. Keep going with the
        // smallest workable pool rather than allocating nothing - the roots must stay resident
        // or the scene cannot draw at all - and say so.
        const minPool = Math.min(totalPages, 64) * this.pageSizeBytes;
        if (available < minPool) {
            Debug.warnOnce(`MeshletWorld: geometry budget ${(this.poolBytes / 1048576).toFixed(0)} MB is below the ${(reserved / 1048576).toFixed(0)} MB this scene needs for its fixed and per-view buffers; falling back to a minimum page pool.`);
            available = minPool;
        }

        const indexBytes = Math.floor(available * this.indexBudgetFraction);
        const pagePool = available - indexBytes;
        // one pool shared across every view; the director distributes it by demand
        this.indexBudgetTotal = Math.max(Math.floor(indexBytes / 4), 3);
        this.indexOverrun = 0;
        this.maxIndices = 0;
        this.budgetBreakdown = { budget: this.poolBytes, fixed, perView: reserved - fixed, pagePool, indices: indexBytes };
        return pagePool;
    }

    /**
     * Builds all GPU state from the added resources.
     */
    finalize() {
        Debug.assert(!this.finalized);
        this.finalized = true;
        const device = this.device;
        const pending = this._pending;
        Debug.assert(pending.length, 'MeshletWorld: no resources added');

        // all resources must share one page size (pool slot size); attribute layouts may differ
        this.pageSizeBytes = pending[0].resource.manifest.pageSizeBytes;
        Debug.assert(pending.every(({ resource }) => resource.manifest.pageSizeBytes === this.pageSizeBytes),
            'MeshletWorld: all resources must share one page size');

        // totals. Material rows: a resource with a baked material table contributes its records
        // verbatim (plus one synthesized fallback row when any primitive has no material); a
        // resource without one gets one synthesized row per primitive (baseColorFactor only).
        let totalPages = 0;
        let totalMeshlets = 0;
        let totalInstances = 0;
        let totalMaterialRows = 0;
        let anyTextures = false;
        let uvChannels = 0;
        let anyTangents = false;
        for (const { resource, instances } of pending) {
            totalPages += resource.manifest.pageCount;
            totalMeshlets += resource.totalMeshlets;
            totalInstances += (instances ?? resource.instances).length;
            if (resource.materialTable) {
                const unassigned = resource.primitives.some(p => p.materialIndex < 0) ? 1 : 0;
                totalMaterialRows += resource.materialCount + unassigned;
            } else {
                totalMaterialRows += resource.primitives.length;
            }
            if (resource.textureManifest) anyTextures = true;
            uvChannels = Math.max(uvChannels, resource.manifest.attributeLayout.uvComponents ?? 0);
            if (resource.manifest.attributeLayout.tangents) anyTangents = true;
        }

        // streamed-texture state (resident coarse tails + demand-streamed fine pools); the
        // flat texture index space concatenates resources like every other address space.
        // Retention: when the director determined the streamed texture set is unchanged, the
        // previous world's system - tails, fine pools, residency - is adopted wholesale.
        const texturesAdopted = !!this.adoptTextures;
        this.textures = this.adoptTextures ?? (anyTextures ? new MeshletTextures(device, this.transcode) : null);
        this.adoptTextures = null;
        if (this.textures && !texturesAdopted) this.textures.poolBytes = this.texturePoolBytes;

        // page pool + residency map. Resident worlds: slot = rebased page index. Streamed
        // worlds: pool sized to the budget, everything starts non-resident (the residency
        // manager pins roots and streams the rest on demand).
        this.totalPages = totalPages;

        // Resolve the geometry budget before anything is allocated. The per-view working set is
        // sized from the (instance, meshlet) pair count, so count the pairs up front - the main
        // loop below recomputes them as it writes the tables.
        let budgetPairs = 0;
        let budgetWorkItems = 0;
        for (const { resource, instances } of pending) {
            for (const inst of (instances ?? resource.instances)) {
                const n = resource.primitives[inst.primIndex].meshletCount;
                budgetPairs += n;
                budgetWorkItems += Math.ceil(n / MESHLET_CULL_SLICE);
            }
        }
        this.pagePoolBytes = this._resolveBudget({
            totalPages,
            totalMeshlets,
            totalInstances,
            totalMaterialRows,
            pairs: budgetPairs,
            workItems: budgetWorkItems
        });

        let poolSlots = totalPages;
        if (this.streamed) {
            poolSlots = this.pagePoolBytes > 0 ?
                Math.max(Math.floor(this.pagePoolBytes / this.pageSizeBytes), 1) :
                totalPages;
            poolSlots = Math.min(poolSlots, totalPages);
        }
        this.poolSlots = poolSlots;
        // retention: adopt a compatible previous world's page pool (its resident pages stay
        // valid because the carried resources keep their page-space prefix); otherwise create
        if (this.adoptPagePool && this.adoptPagePool.byteSize === poolSlots * this.pageSizeBytes) {
            this.pagePool = this.adoptPagePool;
        } else {
            this.adoptPagePool?.destroy();
            this.adoptResidency = null;
            this.adoptedPages = 0;
            this.pagePool = new StorageBuffer(device, poolSlots * this.pageSizeBytes, BUFFERUSAGE_COPY_DST);
        }
        this.adoptPagePool = null;
        this.residency = new Uint32Array(totalPages);
        if (this.streamed) {
            this.residency.fill(PAGE_NOT_RESIDENT);
            // carried prefix: those pages keep their pool slots
            if (this.adoptResidency) {
                const carry = Math.min(this.adoptedPages, totalPages);
                for (let p = 0; p < carry; p++) this.residency[p] = this.adoptResidency[p];
            }
        } else {
            for (let p = 0; p < totalPages; p++) this.residency[p] = p;
        }
        this.adoptResidency = null;
        this.residencyBuffer = new StorageBuffer(device, this.residency.byteLength, BUFFERUSAGE_COPY_DST);
        this.residencyBuffer.write(0, this.residency);

        // meshletData (page indices rebased), per-prim global meshlet bases
        const meshletData = new Uint32Array(totalMeshlets * MESHLET_DATA_U32S);
        const objectData = new Uint32Array(totalInstances * OBJECT_DATA_U32S);
        const objectDataF = new Float32Array(objectData.buffer);
        const materialTable = new Uint32Array(Math.max(totalMaterialRows, 1) * MATERIAL_RECORD_U32S);
        const materialTableF = new Float32Array(materialTable.buffer);

        // synthesized fallback record: matte white/baseColor, no textures
        const writeSynthRecord = (row, baseColor) => {
            const recordBase = row * MATERIAL_RECORD_U32S;
            for (let c = 0; c < 4; c++) materialTableF[recordBase + MATERIAL_RECORD.BASE_COLOR + c] = baseColor?.[c] ?? 1;
            materialTableF[recordBase + MATERIAL_RECORD.EMISSIVE_STRENGTH] = 1;
            materialTableF[recordBase + MATERIAL_RECORD.METALLIC] = 0;
            materialTableF[recordBase + MATERIAL_RECORD.ROUGHNESS] = 1;
            materialTableF[recordBase + MATERIAL_RECORD.ALPHA_CUTOFF] = 0.5;
            for (let slot = 0; slot < MATERIAL_TEXTURE_SLOTS; slot++) {
                materialTable[recordBase + MATERIAL_RECORD.SLOT_WORDS + slot] = MATERIAL_SLOT_ABSENT;
            }
        };

        // Five address spaces, each concatenated in resource add order: pages (pool slot /
        // residency index), meshlets (global id), instances (objectData row = pick record),
        // material rows and flat texture indices. A resource's block starts where the previous
        // one ended, which is what lets MeshletDirector#rebuild carry a previous world's page
        // pool, residency and visibility bits: only an UNCHANGED PREFIX of the add order keeps
        // its addresses, so anything after the first changed resource is rebuilt cold.
        let pageBase = 0;
        let meshletBase = 0;
        let instanceBase = 0;
        let materialBase = 0;
        let runningTexBase = 0;
        this.worldBounds.center.set(0, 0, 0);
        this.worldBounds.halfExtents.set(0, 0, 0);
        let boundsInit = false;

        this.instanceCount = totalInstances;
        this.workItemCapacity = 0;
        this.recordCapacity = 0;
        this.indexCapacity = [0, 0, 0];
        this.indexWorst = [0, 0, 0];

        // One pick id PER INSTANCE, from one reserved block. An instance is one primitive at
        // one transform - which is exactly the granularity a glTF submesh has so, for example, picking a
        // table's top and its legs apart works the way it would with ordinary meshes. The ids
        // share the engine's picker space, so they never collide with mesh instance ids.
        this.pickIdBase = PickerId.reserve(totalInstances);
        this._pickRecords = null;

        for (const { resource, transform, shardBuffers, baseUrl, instances } of pending) {
            const manifest = resource.manifest;
            const grid = manifest.positionGrid;
            const uvFloats = (manifest.attributeLayout.uvComponents ?? 0) * 2;
            const hasTangents = manifest.attributeLayout.tangents ? 1 : 0;
            const instList = instances ?? resource.instances;
            this.placements.push({
                resource,
                instances: instList,
                instanceBase,
                instanceCount: instList.length,
                pairBase: this.totalPairs,
                pairCount: 0
            });
            this.buildEntries.push({
                resource,
                baseUrl: baseUrl ?? null,
                streamed: !shardBuffers,
                pageCount: manifest.pageCount,
                instances: instList
            });

            if (shardBuffers) {
                // fully resident: upload every page now, slot = rebased page index
                const table = manifest.pageTable;
                for (let p = 0; p < manifest.pageCount; p++) {
                    const entry = p * PAGE_TABLE_FIELDS;
                    const blob = table[entry + PAGE_TABLE.BLOB];
                    const offset = table[entry + PAGE_TABLE.OFFSET_HI] * 0x100000000 + table[entry + PAGE_TABLE.OFFSET_LO];
                    this.pagePool.write((pageBase + p) * this.pageSizeBytes,
                        new Uint32Array(shardBuffers[blob], offset, this.pageSizeBytes / 4));
                }
            } else {
                this.streamInfo.push({ resource, pageBase, baseUrl });
            }

            // material rows: baked table verbatim, or one synthesized row per primitive
            const baked = resource.materialTable;
            let resourceRows;
            let fallbackRow = -1;
            if (baked) {
                materialTable.set(baked, materialBase * MATERIAL_RECORD_U32S);
                resourceRows = resource.materialCount;
                if (resource.primitives.some(p => p.materialIndex < 0)) {
                    fallbackRow = resourceRows++;
                    writeSynthRecord(materialBase + fallbackRow, null);
                }
            } else {
                resourceRows = resource.primitives.length;
            }

            // streamed textures: register the manifest (unless the texture system was adopted -
            // then it already holds this resource's textures) and rebase the copied records'
            // slot texture indices into the world-global flat index space (like pageBase for
            // pages).
            if (resource.textureManifest && this.textures) {
                if (!baseUrl) {
                    Debug.warnOnce('MeshletWorld: resource has a texture manifest but no base URL (resident add) - its textures are skipped.');
                } else {
                    const textureBase = runningTexBase;
                    for (const arr of resource.textureManifest.arrays ?? []) {
                        runningTexBase += Array.isArray(arr.layers) ? arr.layers.length : 0;
                    }
                    if (!texturesAdopted) {
                        this.textures.addResource(resource.textureManifest, baseUrl);
                    }
                    if (baked && textureBase > 0) {
                        for (let row = 0; row < resource.materialCount; row++) {
                            const recordBase = (materialBase + row) * MATERIAL_RECORD_U32S;
                            for (let slot = 0; slot < MATERIAL_TEXTURE_SLOTS; slot++) {
                                const word = materialTable[recordBase + MATERIAL_RECORD.SLOT_WORDS + slot];
                                if ((word & 0xFFFF) !== MATERIAL_SLOT_ABSENT) {
                                    materialTable[recordBase + MATERIAL_RECORD.SLOT_WORDS + slot] = (word & 0xFFFF0000) | (((word & 0xFFFF) + textureBase) & 0xFFFF);
                                }
                            }
                        }
                    }
                }
            }

            // meshletData with page rebasing; per-prim bases + material rows, index budgets
            const primBases = [];
            const primMatRows = [];
            const primBucketCorners = [];
            for (let primIndex = 0; primIndex < resource.primitives.length; primIndex++) {
                const prim = resource.primitives[primIndex];
                primBases.push(meshletBase);
                const src = prim.meshletData;
                const dstBase = meshletBase * MESHLET_DATA_U32S;
                meshletData.set(src, dstBase);
                for (let m = 0; m < prim.meshletCount; m++) {
                    meshletData[dstBase + m * MESHLET_DATA_U32S + MESHLET_DATA.PAGE] += pageBase;
                }
                meshletBase += prim.meshletCount;

                if (baked) {
                    primMatRows.push(prim.materialIndex >= 0 ? prim.materialIndex : fallbackRow);
                } else {
                    primMatRows.push(primIndex);
                    writeSynthRecord(materialBase + primIndex, prim.baseColorFactor);
                }

                // This primitive's draw bucket comes from its MATERIAL record, which keeps
                // alpha masking and two-sidedness apart (word 11) - and a primitive has exactly
                // one material, so the bucket is a per-primitive property. Folding the two facts
                // into one bit would put opaque geometry in the alpha-tested bucket and hand it a
                // `discard` that defeats early-Z.
                //
                // Without a baked material table there is nothing better than the meshlet flags
                // themselves; a flag set that cannot tell the two apart reads as masked, which is
                // the safe way to be wrong.
                let bucket;
                if (baked) {
                    const matFlags = materialTable[(materialBase + primMatRows[primIndex]) * MATERIAL_RECORD_U32S + MATERIAL_RECORD.FLAGS];
                    bucket = (matFlags & MATERIAL_FLAG_ALPHA_MASK) ? MESHLET_BUCKET_MASKED :
                        ((matFlags & MATERIAL_FLAG_DOUBLE_SIDED) ? MESHLET_BUCKET_OPAQUE_TWO_SIDED :
                            MESHLET_BUCKET_OPAQUE);
                } else {
                    const meshletFlags = src[MESHLET_DATA.FLAGS];
                    bucket = (meshletFlags & MESHLET_FLAG_ALPHA_MASKED) ? MESHLET_BUCKET_MASKED :
                        ((meshletFlags & MESHLET_FLAG_TWO_SIDED) ? MESHLET_BUCKET_OPAQUE_TWO_SIDED :
                            MESHLET_BUCKET_OPAQUE);
                }
                const bucketBits = (bucket === MESHLET_BUCKET_MASKED ? MESHLET_FLAG_ALPHA_MASKED : 0) |
                    (bucket === MESHLET_BUCKET_OPAQUE ? 0 : MESHLET_FLAG_TWO_SIDED);

                // stamp the bucket onto every meshlet, and total the worst-case index count per
                // bucket once per PRIMITIVE - the per-instance loop below then just adds these,
                // instead of re-walking every meshlet for each of (on the jungle) 41k instances
                const primCorners = [0, 0, 0];
                for (let m = 0; m < prim.meshletCount; m++) {
                    const flagsWord = dstBase + m * MESHLET_DATA_U32S + MESHLET_DATA.FLAGS;
                    meshletData[flagsWord] = (meshletData[flagsWord] & ~(MESHLET_FLAG_ALPHA_MASKED | MESHLET_FLAG_TWO_SIDED)) | bucketBits;
                    primCorners[bucket] += src[m * MESHLET_DATA_U32S + MESHLET_DATA.TRIANGLE_COUNT] * 3;
                }
                primBucketCorners.push(primCorners);
            }

            // instances
            for (const inst of instList) {
                const prim = resource.primitives[inst.primIndex];
                _tmpMat.set(inst.matrix);
                if (transform) _tmpMat.mul2(transform, _tmpMat);
                const matrix = _tmpMat.data;

                // the objectData row (OBJECT_DATA.* words; the shaders read it as MeshletObjectData)
                const row = instanceBase * OBJECT_DATA_U32S;
                for (let k = 0; k < 16; k++) objectDataF[row + OBJECT_DATA.MATRIX + k] = matrix[k];

                const halfExtents = prim.aabbHalfExtents;
                for (let c = 0; c < 3; c++) objectDataF[row + OBJECT_DATA.SPHERE + c] = prim.aabbCenter[c];
                objectDataF[row + OBJECT_DATA.SPHERE + 3] = Math.hypot(halfExtents[0], halfExtents[1], halfExtents[2]);
                objectData[row + OBJECT_DATA.FIRST_MESHLET] = primBases[inst.primIndex];
                objectData[row + OBJECT_DATA.MESHLET_COUNT] = prim.meshletCount;
                objectData[row + OBJECT_DATA.MATERIAL] = materialBase + primMatRows[inst.primIndex];
                objectData[row + OBJECT_DATA.FLAGS] = hasTangents ? OBJECT_FLAG_HAS_TANGENTS : 0;
                objectDataF[row + OBJECT_DATA.MAX_SCALE] = maxAxisScale(matrix);
                objectData[row + OBJECT_DATA.FIRST_PAIR_BIT] = this.totalPairs;
                for (let c = 0; c < 3; c++) objectDataF[row + OBJECT_DATA.GRID_ORIGIN + c] = grid.origin[c];
                objectDataF[row + OBJECT_DATA.GRID_STEP] = grid.step;
                objectData[row + OBJECT_DATA.UV_FLOATS_PER_VERTEX] = uvFloats;
                objectData[row + OBJECT_DATA.PICK_ID] = this.pickIdBase + instanceBase;
                this.totalPairs += prim.meshletCount;
                instanceBase++;

                // capacities: worst case is every meshlet of the instance drawn
                this.workItemCapacity += Math.ceil(prim.meshletCount / MESHLET_CULL_SLICE);
                this.recordCapacity += prim.meshletCount;
                const primCorners = primBucketCorners[inst.primIndex];
                for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
                    this.indexCapacity[b] += primCorners[b];
                }

                // World bounds, from the baked AABB transformed by the instance matrix. Using
                // the bounding SPHERE radius on all three axes instead - which this did - turns
                // a flat 8 km landscape into an 8 km cube, and that inflated box then sets the
                // depth range of every shadow cascade fitted around it.
                _tmpLocalBox.center.set(prim.aabbCenter[0], prim.aabbCenter[1], prim.aabbCenter[2]);
                _tmpLocalBox.halfExtents.set(halfExtents[0], halfExtents[1], halfExtents[2]);
                _tmpBox.setFromTransformedAabb(_tmpLocalBox, _tmpMat);
                if (!boundsInit) {
                    this.worldBounds.copy(_tmpBox);
                    boundsInit = true;
                } else {
                    this.worldBounds.add(_tmpBox);
                }
            }

            const placement = this.placements[this.placements.length - 1];
            placement.pairCount = this.totalPairs - placement.pairBase;

            pageBase += manifest.pageCount;
            materialBase += resourceRows;
        }

        // start below the worst case - GPU-side per-bucket clamps make a smaller allocation
        // safe, and the director grows the buffer to the observed demand. The unscaled worst
        // case (every meshlet drawn) is kept as the views' absolute capacity ceiling
        this.indexWorst = this.indexCapacity.slice();
        // u32 indices, and the buffer is bound both as an index buffer and as storage (the
        // index-write pass writes it), so both limits apply
        const limits = device.limits ?? {};
        const limitBytes = Math.min(
            limits.maxBufferSize ?? Number.MAX_SAFE_INTEGER,
            limits.maxStorageBufferBindingSize ?? Number.MAX_SAFE_INTEGER
        );
        this.deviceIndexCeiling = Math.floor(limitBytes / 4);
        const totalIndexWanted = this.indexCapacity.reduce((a, b) => a + b, 0);
        let indexTarget = Math.min(totalIndexWanted, this.indexCeiling);
        if (this.initialIndices > 0) indexTarget = Math.min(indexTarget, this.initialIndices);
        if (indexTarget < totalIndexWanted) {
            const s = indexTarget / totalIndexWanted;
            this.indexCapacity = this.indexCapacity.map(c => Math.floor(c * s));
        }

        this.meshletDataBuffer = new StorageBuffer(device, meshletData.byteLength, BUFFERUSAGE_COPY_DST);
        this.meshletDataBuffer.write(0, meshletData);
        this.objectDataBuffer = new StorageBuffer(device, objectData.byteLength, BUFFERUSAGE_COPY_DST);
        this.objectDataBuffer.write(0, objectData);
        // CPU mirror kept for per-placement transform updates
        this.objectDataCpu = objectData;
        this.objectDataCpuF = objectDataF;
        this.materialTableBuffer = new StorageBuffer(device, materialTable.byteLength, BUFFERUSAGE_COPY_DST);
        this.materialTableBuffer.write(0, materialTable);

        // streaming request marks (u32 per page) followed by the texture-mip feedback marks
        // (u32 per material row, atomicMax'd texel rates) - shared by all views (marks union
        // via max semantics), cleared once per frame by the residency manager. Per-view frame
        // buffers (cull params, counters, work items, records, claim/vis bits, draw indices)
        // live on each MeshletView.
        this.materialRowCount = totalMaterialRows;
        this.requestsBuffer = new StorageBuffer(device, Math.max((totalPages + totalMaterialRows) * 4, 16), BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
        // Placeholder for the visBits binding of single-phase (shadow) views. The cull shader
        // always declares the binding but only its phase 1/2 branches touch it, and those views
        // run phase 0 - so they bind this 16-byte buffer instead of allocating one bit per
        // instance-meshlet pair, which is tens of MB on a scattered scene.
        this.dummyBits = new StorageBuffer(device, 16, BUFFERUSAGE_COPY_DST);
        this._claimClear = new Uint32Array(Math.max(Math.ceil(this.totalPairs / 32), 4));
        this._requestClear = new Uint32Array(Math.max(totalPages, 4));

        // streamed textures: tails + residency exist before the materials bind them (an
        // adopted system is already finalized and possibly fully loaded)
        if (!texturesAdopted) this.textures?.finalize();

        // binds the world's storage buffers to a bucket material (lit or debug). The per-view
        // records buffer is NOT bound here - each view's mesh instances carry it as an
        // instance-level parameter override
        const bindWorldParams = (material) => {
            material.setParameter('meshletData', this.meshletDataBuffer);
            material.setParameter('objectData', this.objectDataBuffer);
            material.setParameter('pagePool', this.pagePool);
            material.setParameter('residency', this.residencyBuffer);
            material.setParameter('materialTable', this.materialTableBuffer);
            material.setParameter('pageSizeWords', this.pageSizeBytes / 4);
        };
        const bindTextureParams = (material) => {
            if (!this.textures) return;
            material.setParameter('texResidency', this.textures.residencyBuffer);
            for (let f = 0; f < MESHLET_TEX_FAMILIES.length; f++) {
                const { tail, fine } = this.textures.familyTextures(f);
                material.setParameter(_tailBindName(f), tail);
                material.setParameter(_fineBindName(f), fine);
            }
        };

        // per-world chunk set: texture sampling, UV varyings and tangents compile in only when
        // the resources carry them
        const chunks = buildMeshletLitChunks({
            textures: !!this.textures,
            uvChannels: Math.min(uvChannels, MESHLET_MAX_UV_CHANNELS),
            tangents: anyTangents,
            familySizes: this.textures?.families.map(fam => fam.slotSize),
            familyLevels: this.textures?.families.map(fam => ({ slotLevels: fam.slotLevels, tailLevels: fam.levels }))
        });

        // engine-lit bucket materials (one per MESHLET_BUCKET_*) shared by both phases; the
        // unlit debug materials (colour modes) are created lazily on first setColorMode(>0)
        this._litMaterials = [];
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            this._litMaterials.push(createMeshletLitMaterial(b, chunks));
        }
        this._litMaterials.forEach(bindWorldParams);
        this._litMaterials.forEach(bindTextureParams);
        this._debugMaterials = null;
        this._bindWorldParams = bindWorldParams;

        // family arrays are created asynchronously (format known after the first transcode) -
        // swap the placeholder bindings for the real textures when each family lands
        if (this.textures) {
            this.textures.onFamilyTexturesReady = (family, tail, fine) => {
                this._litMaterials?.forEach((m) => {
                    m.setParameter(_tailBindName(family), tail);
                    if (fine) m.setParameter(_fineBindName(family), fine);
                });
            };

            // material row -> up to 4 flat texture indices, for turning the cull pass's
            // per-material feedback marks into per-texture fine-mip demand. The cull pass
            // measures texel rate in pre-transform UV space, but sampling applies each
            // slot's KHR_texture_transform scale to the derivatives - a tiled texture
            // (scale > 1) resolves that much coarser than the raw mark suggests, so each
            // slot carries a tiling bias of 16*log2(scale), subtracted from its mark, to keep demand honest.
            const f16 = (h) => {
                const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
                return e === 0 ? s * m * (2 ** -24) : e === 31 ? s * Infinity : s * (1 + m / 1024) * (2 ** (e - 15));
            };
            const rowTex = new Int32Array(totalMaterialRows * MATERIAL_TEXTURE_SLOTS).fill(-1);
            const rowTilingBias = new Int16Array(totalMaterialRows * MATERIAL_TEXTURE_SLOTS);
            for (let row = 0; row < totalMaterialRows; row++) {
                for (let s = 0; s < MATERIAL_TEXTURE_SLOTS; s++) {
                    const word = materialTable[row * MATERIAL_RECORD_U32S + MATERIAL_RECORD.SLOT_WORDS + s];
                    if ((word & 0xFFFF) === 0xFFFF) continue;
                    rowTex[row * MATERIAL_TEXTURE_SLOTS + s] = word & 0xFFFF;
                    const t0 = materialTable[row * MATERIAL_RECORD_U32S + MATERIAL_RECORD.SLOT_TRANSFORMS + s * 2];
                    const t1 = materialTable[row * MATERIAL_RECORD_U32S + MATERIAL_RECORD.SLOT_TRANSFORMS + s * 2 + 1];
                    if ((t0 | t1) !== 0) {
                        const scale = Math.max(Math.abs(f16(t1 & 0xFFFF)), Math.abs(f16(t1 >>> 16)));
                        if (scale > 0 && Number.isFinite(scale)) {
                            rowTilingBias[row * MATERIAL_TEXTURE_SLOTS + s] = Math.round(TEXEL_RATE_PER_MIP * Math.log2(scale));
                        }
                    }
                }
            }
            this.textures.setMaterialSlotMap(rowTex, rowTilingBias);
        }

        this._colorMode = 0;
        this._pending = [];
    }

    /**
     * The current bucket materials, indexed by MESHLET_BUCKET_* - lit, or the unlit debug
     * materials when a colour mode is active. Views compare this by identity each frame and
     * swap their mesh instances' materials when it changes.
     *
     * @type {Array<object>}
     */
    get bucketMaterials() {
        return this._colorMode > 0 ? this._debugMaterials : this._litMaterials;
    }

    /**
     * The most indices one view may allocate: the device's own limit, lowered by
     * {@link maxIndices} when the caller sets one.
     *
     * @type {number}
     */
    get indexCeiling() {
        const device = this.deviceIndexCeiling || Infinity;
        const budget = this.indexBudgetTotal > 0 ? this.indexBudgetTotal + this.indexOverrun : 0;
        const capped = this.maxIndices > 0 ? Math.min(this.maxIndices, budget || this.maxIndices) : budget;
        return capped > 0 ? Math.min(capped, device) : device;
    }

    /**
     * One record per instance, for resolving a picked id. An instance is one PRIMITIVE at one
     * transform, so this is submesh granularity - a table baked with its top and legs as
     * separate glTF primitives picks apart exactly as it would have unbaked.
     *
     * Built on the first pick and cached: an app that never picks pays nothing, and the records
     * are what the picker's id map needs distinct values of (it de-duplicates by identity, so
     * handing it one shared object per placement would collapse every instance into one hit).
     *
     * @type {Array<{ pickId: number, placement: object, resource: object, instanceIndex: number, primIndex: number, matrix: Float32Array }>}
     */
    get pickRecords() {
        if (this._pickRecords) return this._pickRecords;
        const records = [];
        for (const placement of this.placements) {
            const instances = placement.instances;
            for (let i = 0; i < instances.length; i++) {
                const inst = instances[i];
                records.push({
                    pickId: this.pickIdBase + placement.instanceBase + i,
                    placement,
                    resource: placement.resource,
                    instanceIndex: placement.instanceBase + i,
                    primIndex: inst.primIndex,
                    matrix: inst.matrix
                });
            }
        }
        this._pickRecords = records;
        return records;
    }

    /**
     * The lit bucket materials, ignoring any active debug colour mode. Shadow views use
     * these unconditionally - the debug materials are unlit {@link ShaderMaterial}s with no
     * shadow variant, so a colour mode would otherwise silently drop every meshlet shadow.
     *
     * @type {Array<object>}
     */
    get litMaterials() {
        return this._litMaterials;
    }

    /**
     * Sets the debug color mode on the bucket materials (0 = engine-lit material shading,
     * 1 = LOD tint, 2 = per-meshlet colour) - modes > 0 swap in the unlit debug materials
     * (applied to each view's mesh instances via {@link MeshletView#syncMaterials}).
     *
     * @param {number} mode - The color mode.
     */
    setColorMode(mode) {
        if (mode > 0 && !this._debugMaterials) {
            this._debugMaterials = [];
            for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
                this._debugMaterials.push(createMeshletMaterial(b));
            }
            this._debugMaterials.forEach(this._bindWorldParams);
        }
        this._colorMode = mode;
        if (mode > 0) this._debugMaterials.forEach(m => m.setParameter('colorMode', mode));
    }

    /**
     * Updates the root transform of an added resource (placement), rewriting its objectData rows.
     * Only the transform-derived fields change; culling uses the new matrix next frame. Note that
     * {@link worldBounds} keeps its finalize-time value.
     *
     * @param {number} index - The placement index (resource add order).
     * @param {Mat4|null} transform - New root transform applied to the resource's placements.
     */
    setPlacementTransform(index, transform, subBase = 0, subCount = -1) {
        Debug.assert(this.finalized && this.placements[index], 'MeshletWorld: invalid placement');
        const { instances, instanceBase, instanceCount } = this.placements[index];
        const start = Math.min(subBase, instanceCount);
        const count = subCount < 0 ? instanceCount - start : Math.min(subCount, instanceCount - start);
        if (count === 0) return;

        const objectDataF = this.objectDataCpuF;
        for (let i = start; i < start + count; i++) {
            const inst = instances[i];
            _tmpMat.set(inst.matrix);
            if (transform) _tmpMat.mul2(transform, _tmpMat);
            const matrix = _tmpMat.data;
            const row = (instanceBase + i) * OBJECT_DATA_U32S;
            for (let k = 0; k < 16; k++) objectDataF[row + OBJECT_DATA.MATRIX + k] = matrix[k];
            objectDataF[row + OBJECT_DATA.MAX_SCALE] = maxAxisScale(matrix);
        }
        this._uploadObjectRows(instanceBase + start, count);
    }

    /**
     * Hides or shows a range of a placement's instances without a rebuild - the instance cull
     * pass skips hidden rows (objectData word 23 bit 0), so hidden geometry stops drawing AND
     * stops marking pages/textures, letting the LRU drain its residency naturally.
     *
     * @param {number} index - The placement index (resource add order).
     * @param {boolean} hidden - True to hide the instances.
     * @param {number} [subBase] - First instance within the placement (default 0).
     * @param {number} [subCount] - Instance count (default: the rest of the placement).
     */
    setPlacementHidden(index, hidden, subBase = 0, subCount = -1) {
        Debug.assert(this.finalized && this.placements[index], 'MeshletWorld: invalid placement');
        const { instanceBase, instanceCount } = this.placements[index];
        const start = Math.min(subBase, instanceCount);
        const count = subCount < 0 ? instanceCount - start : Math.min(subCount, instanceCount - start);
        if (count === 0) return;

        const objectData = this.objectDataCpu;
        for (let i = start; i < start + count; i++) {
            const flagsWord = (instanceBase + i) * OBJECT_DATA_U32S + OBJECT_DATA.FLAGS;
            objectData[flagsWord] = hidden ? (objectData[flagsWord] | OBJECT_FLAG_HIDDEN) : (objectData[flagsWord] & ~OBJECT_FLAG_HIDDEN);
        }
        this._uploadObjectRows(instanceBase + start, count);
    }

    /**
     * Marks instances for the outline pass, by global instance index - the same index a pick
     * returns, so an editor can outline exactly what was clicked.
     *
     * Selection has to live in objectData: one indirect draw covers every instance, so there is
     * no mesh instance to add to an outline layer and no per-object uniform to set. The outline
     * variant of the vertex shader collapses unselected instances to a degenerate triangle, so
     * they cost a vertex fetch and no rasterisation.
     *
     * @param {number[]|Set<number>} instanceIndices - Global instance indices to outline.
     * @param {boolean} [outlined] - True to add, false to remove. Defaults to true.
     */
    setInstancesOutlined(instanceIndices, outlined = true, hover = false) {
        Debug.assert(this.finalized, 'MeshletWorld: setInstancesOutlined before finalize');
        const flag = hover ? OBJECT_FLAG_HOVERED : OBJECT_FLAG_OUTLINED;
        const set = hover ? this._hovered : this._outlined;
        const objectData = this.objectDataCpu;
        let lo = Infinity;
        let hi = -Infinity;
        for (const index of instanceIndices) {
            if (index < 0 || index >= this.instanceCount) continue;
            const flagsWord = index * OBJECT_DATA_U32S + OBJECT_DATA.FLAGS;
            objectData[flagsWord] = outlined ? (objectData[flagsWord] | flag) : (objectData[flagsWord] & ~flag);
            if (outlined) set.add(index); else set.delete(index);
            lo = Math.min(lo, index);
            hi = Math.max(hi, index);
        }
        if (lo > hi) return;
        // one contiguous upload covering the touched range - a scattered selection writes a few
        // extra rows rather than issuing a write per instance
        this._uploadObjectRows(lo, hi - lo + 1);
    }

    /**
     * Clears the outline flags on every instance.
     *
     * @param {boolean} [hover] - Clear the hover flag rather than the selection flag.
     */
    clearOutlines(hover = false) {
        Debug.assert(this.finalized, 'MeshletWorld: clearOutlines before finalize');
        const flag = hover ? OBJECT_FLAG_HOVERED : OBJECT_FLAG_OUTLINED;
        const set = hover ? this._hovered : this._outlined;
        if (set.size === 0) return;
        const objectData = this.objectDataCpu;
        for (const index of set) objectData[index * OBJECT_DATA_U32S + OBJECT_DATA.FLAGS] &= ~flag;
        set.clear();
        this.objectDataBuffer.write(0, objectData);
    }

    /**
     * Uploads a contiguous range of objectData rows from the CPU mirror.
     *
     * @param {number} first - First instance index.
     * @param {number} count - Row count.
     * @private
     */
    _uploadObjectRows(first, count) {
        const rowBytes = OBJECT_DATA_U32S * 4;
        this.objectDataBuffer.write(first * rowBytes,
            this.objectDataCpu.subarray(first * OBJECT_DATA_U32S, (first + count) * OBJECT_DATA_U32S));
    }

    /** @type {number} - instances flagged for the outline pass, selected or hovered. */
    get outlinedCount() {
        return this._outlined.size + this._hovered.size;
    }
}

export { MeshletWorld };
