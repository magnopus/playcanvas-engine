import { MeshletCells } from './meshlet-cells.js';
import { MeshletCutData } from './meshlet-cut-data.js';
import { MeshletInstanceList } from './meshlet-instance-list.js';
import { MeshletMaterialTable, bucketOfMeshletFlags, meshletFlagsOfBucket } from './meshlet-material-table.js';
import { resolveMeshletBudget, rootListItems } from './meshlet-world-budget.js';
import { meshletBakeStamp } from './streaming/meshlet-bake-stamp.js';
import { Debug } from '../../core/debug.js';
import { PickerId } from '../picker-id.js';
import { Mat4 } from '../../core/math/mat4.js';
import { BoundingBox } from '../../core/shape/bounding-box.js';
import {
    BUFFERUSAGE_COPY_DST, BUFFERUSAGE_COPY_SRC
} from '../../platform/graphics/constants.js';
import { StorageBuffer } from '../../platform/graphics/storage-buffer.js';
import { ShaderChunks } from '../shader-lib/shader-chunks.js';
import { OBJECT_BUCKET_SHIFT, OBJECT_FLAG_NO_SHADOW,
    MATERIAL_RECORD, MATERIAL_RECORD_U32S, MESHLET_BUCKET_COUNT, MESHLET_DATA, MESHLET_DATA_U32S,
    MESHLET_FLAG_ALPHA_MASKED, MESHLET_FLAG_TWO_SIDED, MESHLET_MAX_UV_CHANNELS, MESHLET_NO_PARENT,
    OBJECT_DATA, OBJECT_DATA_U32S, OBJECT_FLAG_HAS_TANGENTS, OBJECT_FLAG_HAS_COLORS,
    PRIMITIVE_DATA,
    OBJECT_FLAG_HIDDEN, OBJECT_FLAG_HOVERED, OBJECT_FLAG_OUTLINED,
    PAGE_NOT_RESIDENT, PAGE_TABLE, PAGE_TABLE_FIELDS,
    TEXEL_RATE_PER_MIP, MESHLET_COLOR_MODE } from './constants.js';
import { createMeshletLitMaterial } from './meshlet-lit-material.js';
import { createMeshletMaterial } from './meshlet-material.js';
import { buildMeshletLitChunks } from './shaders/meshlet-lit-chunks-wgsl.js';
import { MeshletTextures, MESHLET_TEX_FAMILIES } from './textures/meshlet-textures.js';

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { Material } from '../materials/material.js'
 * @import { MeshletFetchScheduler } from './streaming/meshlet-fetch-scheduler.js'
 * @import { MeshletPrimitive, MeshletResource } from './meshlet-resource.js'
 * @import { MeshletTranscodeFn } from './textures/meshlet-ktx2.js'
 */

/**
 * How a streamed resource's sidecars (geometry shards, texture containers) are fetched.
 *
 * @typedef {object} MeshletFetchOptions
 * @property {((uri: string) => string)|null} [resolveUrl] - Maps a manifest URI to the URL it
 * is fetched from, in place of appending it to the resource's base URL.
 * @property {RequestCredentials|null} [credentials] - Fetch credentials mode for the requests.
 * @property {string|null} [cacheVersion] - Versions the sidecars' persistent cache keys (see
 * `MeshletFetchScheduler#byteCache`); the world sets it to the resource's bake stamp.
 */

/** Page attribute streams this build can place (see meshletPageLayoutWGSL). */
const KNOWN_LAYOUT_KEYS = new Set(['positions', 'normals', 'tangents', 'uvComponents', 'colors']);

// meshlet records (128 bytes each) gathered per upload while building a world: 16 MB
const RECORD_UPLOAD_BATCH = 128 * 1024;

// objectData instance rows generated per upload (64 bytes each): 4 MB. Rows are generated from
// the placements on demand rather than mirrored, so this is all the CPU holds of them.
const OBJECT_ROW_BATCH = 64 * 1024;

// floats per instance of the lightmap table: atlas scale/bias, EPIC scale, EPIC add, then
// (texture index, UV channel, enabled, 0)
const LIGHTMAP_ROW_FLOATS = 16;

// longest the world bounds stay loose after instances move, in milliseconds
const WORLD_BOUNDS_REFIT_MS = 1000;

// outline and hover changes up to this many instances write just their flags words
const FLAG_WRITE_LIMIT = 256;

const BYTES_PER_MB = 1024 * 1024;

const _tmpMat = new Mat4();

/** @type {number[]} */
const _touched = [];

/** @type {{ placement: object, run: number, index: number }} - see MeshletWorld#_locate. */
const _located = { placement: null, run: 0, index: 0 };
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
 * Largest storage buffer the device can bind, in bytes: both the buffer size and the storage
 * binding limits apply, and exceeding either fails creation with a validation error instead of
 * allocating.
 *
 * @param {GraphicsDevice} device - The device.
 * @returns {number} Bytes.
 */
const maxStorageBytes = (device) => {
    const limits = device.limits ?? {};
    return Math.min(
        limits.maxBufferSize ?? Number.MAX_SAFE_INTEGER,
        limits.maxStorageBufferBindingSize ?? Number.MAX_SAFE_INTEGER
    );
};

/**
 * The primitive-local pages of a primitive's root clusters (those without a parent): what the
 * residency pins for as long as the primitive is placed. Kept on the primitive, with
 * {@link primitiveStats}, so a resource whose records were released can still be laid out.
 *
 * @param {MeshletPrimitive} prim - The primitive.
 * @returns {number[]} Its root pages (may repeat).
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
 * What a picked meshlet id resolves to: one instance, a primitive at one transform.
 *
 * @ignore
 */
class MeshletPickRecord {
    /** @type {number} - the picker id. */
    pickId;

    /** @type {object} - the world placement holding the instance. */
    placement;

    /** @type {number} - global instance index (see MeshletWorld#instanceOffset). */
    instanceIndex;

    /** @type {number} - the resource primitive. */
    primIndex;

    /** @type {number} - the instance's position in the placement's list. @private */
    _listIndex;

    /**
     * @param {number} pickId - The picker id.
     * @param {object} placement - The world placement holding the instance.
     * @param {number} instanceIndex - Global instance index (see MeshletWorld#instanceOffset).
     * @param {number} primIndex - The resource primitive.
     * @param {number} listIndex - The instance's position in the placement's list.
     */
    constructor(pickId, placement, instanceIndex, primIndex, listIndex) {
        this.pickId = pickId;
        this.placement = placement;
        this.instanceIndex = instanceIndex;
        this.primIndex = primIndex;
        this._listIndex = listIndex;
    }

    /** @type {MeshletResource} */
    get resource() {
        return this.placement.resource;
    }

    /** @type {Float32Array} - the instance's document-space matrix, before its root transform. */
    get matrix() {
        return this.placement.instances.get(this._listIndex).matrix;
    }
}

/**
 * Scene totals of a world's pending resources (see MeshletWorld#_countTotals).
 *
 * @typedef {object} MeshletBuildTotals
 * @property {number} totalPages - Pages across resources.
 * @property {number} totalMeshlets - Meshlets across resources.
 * @property {number} totalInstances - Instances across placements.
 * @property {number} totalPrimitiveRows - Primitive rows of objectData.
 * @property {number} totalMaterialRows - Material table rows.
 * @property {number} pairs - Instance-meshlet pairs.
 * @property {number[]} rootPages - Global pages of every placed primitive's roots, ascending.
 * @property {boolean} anyLightmaps - Some instance carries a lightmap.
 * @property {boolean} anyTextures - Some resource streams textures.
 * @property {number} uvChannels - Most UV channels of any resource.
 * @property {boolean} anyTangents - Some resource carries tangents.
 * @property {boolean} anyColors - Some resource carries vertex colours.
 * @ignore
 */

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
 * Five address spaces are each concatenated in resource add order: pages (pool slot / residency
 * index), meshlets (global id), instances (objectData row = pick record), material rows and flat
 * texture indices. A resource's block starts where the previous one ended, which is what lets a
 * rebuild carry a previous world's page pool, residency and visibility bits: only an unchanged
 * prefix of the add order keeps its addresses, so anything after the first changed resource is
 * rebuilt cold.
 *
 * @ignore
 */
class MeshletWorld {
    /** @type {GraphicsDevice} */
    device;

    /** @type {Array<{ resource: MeshletResource, transform: Mat4|Mat4[]|null, shardBuffers: ArrayBuffer[] }>} */
    _pending = [];

    finalized = false;

    /** @type {StorageBuffer|null} - the page pool, {@link poolSlots} pages. */
    pagePool = null;

    /** @type {number} - pages {@link pagePool} holds. */
    poolSlots = 0;

    /** @type {Uint32Array} - page -> pool slot (PAGE_NOT_RESIDENT when not resident). */
    residency = new Uint32Array(0);

    /** @type {StorageBuffer|null} - {@link residency} on the GPU. */
    residencyBuffer = null;

    /**
     * Per page a streaming request mark, then per material row a texture-mip feedback mark
     * (atomicMax'd texel rates). Shared by every view - marks union by max - and cleared once
     * a frame by the residency.
     *
     * @type {StorageBuffer|null}
     */
    requestsBuffer = null;

    /** @type {number} - material rows, and texel-rate marks after the page marks. */
    materialRowCount = 0;

    /**
     * Placeholder bound as visBits by single-phase (shadow) views. The cull shader always
     * declares the binding but only its phase 1/2 branches touch it, so those views bind this
     * 16-byte buffer instead of one bit per instance-meshlet pair.
     *
     * @type {StorageBuffer|null}
     */
    dummyBits = null;

    /** @type {number} - first id of this world's reserved per-instance picker block. */
    pickIdBase = 0;

    /**
     * Global index of this world's first instance. A director group splits a scene across
     * several worlds and numbers their instances consecutively, so pick records and outline
     * selections use one index space across them. 0 for a world on its own.
     *
     * @type {number}
     */
    instanceOffset = 0;

    /** @type {Array<object>|null} - lazily built; see {@link pickRecords}. */
    _pickRecords = null;

    /** @type {Set<number>} - instance indices flagged as selected. */
    _outlined = new Set();

    /** @type {Set<number>} - instance indices flagged as hovered. */
    _hovered = new Set();

    /** @type {StorageBuffer|null} */
    meshletDataBuffer = null;

    /** @type {MeshletCutData|null} */
    cut = null;

    /** @type {MeshletCells|null} - spatial instance cells the views cull before their instances. */
    cells = null;

    /** @type {StorageBuffer|null} - instance rows, then primitive rows (see OBJECT_DATA_U32S). */
    objectDataBuffer = null;

    /**
     * Per instance, its objectData flags word (OBJECT_FLAG_* and the draw bucket). The only
     * per-instance state the CPU keeps: the rest of a row is derived from the placements when
     * it is uploaded.
     *
     * @type {Uint32Array}
     */
    instanceFlags = new Uint32Array(0);

    /** @type {number} - index of the placement {@link _locate} found last. */
    _placementCursor = 0;

    /** @type {StorageBuffer|null} */
    materialTableBuffer = null;

    /** @type {StorageBuffer|null} */
    lightmapBuffer = null;

    /** @type {MeshletTextures|null} */
    textures = null;

    /** @type {Material[]|null} - engine-lit bucket materials, one per MESHLET_BUCKET_*. */
    _litMaterials = null;

    /** @type {Material[]|null} - unlit debug bucket materials, created on first use. */
    _debugMaterials = null;

    /** @type {number} - the MESHLET_COLOR_MODE in effect (see {@link setColorMode}). */
    _colorMode = 0;

    // capacities (index capacities are the initial per-view values; each MeshletView owns
    // its own demand-grown copy)
    instanceCount = 0;

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

    /**
     * Global pages of every placed primitive's root clusters, ascending: pinned by the
     * residency, and reserved in the page pool before anything else.
     *
     * @type {number[]}
     */
    rootPages = [];

    /** @type {BoundingBox} */
    _worldBounds = new BoundingBox();

    /**
     * True while {@link _worldBounds} may be looser than the instances: moved instances grow it
     * at once, and it is refitted exactly at most every WORLD_BOUNDS_REFIT_MS.
     *
     * @type {boolean}
     * @private
     */
    _worldBoundsDirty = false;

    /** @type {number} - performance.now() of the last exact refit. @private */
    _worldBoundsRefitAt = 0;

    /**
     * Current world-space bounds. Always encloses every instance; after instances move it can be
     * loose for up to WORLD_BOUNDS_REFIT_MS, so a component moving every frame does not cost a
     * pass over every instance per frame.
     *
     * @type {BoundingBox}
     */
    get worldBounds() {
        if (this._worldBoundsDirty && performance.now() - this._worldBoundsRefitAt >= WORLD_BOUNDS_REFIT_MS) {
            this._worldBoundsDirty = false;
            this._worldBoundsRefitAt = performance.now();
            this._worldBounds.center.set(0, 0, 0);
            this._worldBounds.halfExtents.set(0, 0, 0);
            this._growWorldBounds(0, this.instanceCount, false);
        }
        return this._worldBounds;
    }

    /**
     * Grows {@link _worldBounds} by the world boxes of a range of instances: each primitive's
     * baked AABB through its instance matrix, which keeps a flat instance flat where its
     * bounding sphere would not - and with it the depth range of the shadow cascades fitted
     * around the world.
     *
     * @param {number} first - First instance.
     * @param {number} count - Instance count.
     * @param {boolean} initialized - False when the bounds hold nothing yet.
     * @private
     */
    _growWorldBounds(first, count, initialized) {
        for (let instance = first; instance < first + count; instance++) {
            const prim = this._instanceMatrix(instance, _tmpMat);
            _tmpLocalBox.center.set(prim.aabbCenter[0], prim.aabbCenter[1], prim.aabbCenter[2]);
            _tmpLocalBox.halfExtents.set(prim.aabbHalfExtents[0], prim.aabbHalfExtents[1], prim.aabbHalfExtents[2]);
            _tmpBox.setFromTransformedAabb(_tmpLocalBox, _tmpMat);
            if (initialized) {
                this._worldBounds.add(_tmpBox);
            } else {
                this._worldBounds.copy(_tmpBox);
                initialized = true;
            }
        }
    }

    /**
     * Bumped whenever what the world would draw changes without a rebuild: a page installed or
     * evicted, an instance moved, hidden or its shadow casting toggled. Consumers that render on
     * demand (a light with a static shadow map) compare it to know a re-render is due.
     *
     * @type {number}
     */
    contentVersion = 0;

    /**
     * Grow-only scratch for generated objectData rows (see _writeObjectRows).
     *
     * @type {Uint32Array|null}
     * @private
     */
    _rowScratch = null;

    /**
     * Float view of {@link _rowScratch}.
     *
     * @type {Float32Array|null}
     * @private
     */
    _rowScratchF32 = null;

    /**
     * @param {GraphicsDevice} device - The graphics device.
     */
    constructor(device) {
        this.device = device;
    }

    destroy() {
        // carried from the previous world but never consumed by a build
        this.adoptPagePool?.destroy();
        this.adoptPagePool = null;
        this.adoptTextures?.destroy();
        this.adoptTextures = null;
        this.pagePool?.destroy();
        this.residencyBuffer?.destroy();
        this.requestsBuffer?.destroy();
        this.dummyBits?.destroy();
        this.meshletDataBuffer?.destroy();
        this.cut?.destroy();
        this.cells?.destroy();
        this.objectDataBuffer?.destroy();
        this.materialTableBuffer?.destroy();
        this.lightmapBuffer?.destroy();
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
     * (claim/visibility bit planes, work items, records, draw indices) - not just the page pool,
     * which the demand-grown buffers can outgrow many times over when they have no ceiling.
     *
     * The fixed and per-view costs are computed at finalize and subtracted; what remains is
     * split between the page pool and the draw index buffers by {@link indexBudgetFraction}, and
     * the index share becomes {@link indexBudgetTotal}, so growth is bounded by the budget rather
     * than by the device.
     *
     * @type {number}
     */
    poolBytes = 0;

    /**
     * Share of the budget (after fixed and per-view costs) given to the demand-grown draw index
     * buffers; the rest goes to the page pool. Indices are the working set of what is on screen,
     * pages the working set of what is nearby, so the split trades detail-in-view against
     * streaming headroom.
     *
     * @type {number}
     */
    indexBudgetFraction = 0.35;

    /**
     * Share of each view's index, record and cut-queue capacity root admission may give to
     * instances' coarsest levels under a budget; the rest is kept for the cut to refine what
     * was admitted. Admission takes instances largest on screen first, so a lower share drops
     * more of the far tail and keeps the near instances sharper; 1 lets the roots of everything
     * in view take the whole budget, leaving nothing to refine. Unbudgeted worlds ignore it.
     *
     * @type {number}
     */
    rootBudgetFraction = 0.5;

    /**
     * Records each camera view may grow to under a budget, resolved at finalize (0 =
     * unbudgeted: bounded by the worst case only). Shadow views get {@link shadowRecordCeiling}.
     *
     * @type {number}
     */
    recordCeiling = 0;

    /**
     * Cut queue entries each view may grow to under a budget, resolved at finalize (0 =
     * unbudgeted).
     *
     * @type {number}
     */
    cutQueueCeiling = 0;

    /** @type {number} - {@link recordCeiling} for a shadow view. */
    shadowRecordCeiling = 0;

    /** @type {number} - {@link cutQueueCeiling} for a shadow view. */
    shadowCutQueueCeiling = 0;

    /** @type {number} - entries a view allocates for its root list (see rootListItems). */
    get rootListItems() {
        return rootListItems(this.instanceCount);
    }

    /** @type {number} - camera views to budget the per-view working set for. */
    budgetCameraViews = 1;

    /** @type {number} - shadow cascade views to budget for (they share more, so cost less). */
    budgetShadowViews = 0;

    /** @type {number} - page-pool bytes resolved from the budget at finalize. */
    pagePoolBytes = 0;

    /**
     * Total draw indices the budget allows across all views. This is a pool, not a per-view
     * allowance: the director hands each view a share proportional to what it is asking for.
     * Splitting it equally starves whichever view happens to be the hungry one, and a starved
     * view drives LOD pressure up for everything.
     *
     * @type {number}
     */
    indexBudgetTotal = 0;

    /** @type {{ budget: number, fixed: number, perView: number, pagePool: number, indices: number, records: number, cutQueue: number, rootPages: number }|null} */
    budgetBreakdown = null;

    /** @type {number} - fine-texture slot-pool byte budget (see MeshletTextures#poolBytes). */
    texturePoolBytes = 96 * 1024 * 1024;

    /**
     * KTX2 transcoder handed to the texture system (see {@link MeshletDirector#transcode}).
     * Scene code cannot import the framework's Basis transcoder, so it arrives by injection.
     *
     * @type {MeshletTranscodeFn|null}
     */
    transcode = null;

    /**
     * The scheduler the world's sidecar fetches (geometry shards, texture containers) go
     * through; a director group shares one across its worlds. Null selects the page-wide default.
     *
     * @type {MeshletFetchScheduler|null}
     */
    fetchScheduler = null;

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
     * hold" - see {@link indexCeiling}, which is what actually bounds growth. It is not unbounded:
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
     * Per-resource streaming info: { resource, pageBase, baseUrl, fetchOptions }.
     *
     * @type {Array<object>}
     */
    streamInfo = [];

    /**
     * Per added-resource placement info, in add order. `instances` is the effective placement
     * list (the add-time override or the resource's own), repeated `repeat` times: once per
     * root transform in `roots`, which a merged placement (several components of one resource)
     * holds one of per member. `instanceBase`/`instanceCount` are the objectData row range,
     * `primitiveRow` the objectData row of the resource's first primitive, and pair bits of
     * instance `run * instances.count + i` start at `pairBase + run * runPairs + pairOffsets[i]`.
     *
     * @type {Array<{ resource: MeshletResource, instances: MeshletInstanceList, repeat: number,
     *     roots: Array<Mat4|null>, instanceBase: number, instanceCount: number, primitiveRow: number,
     *     pairBase: number, pairCount: number, runPairs: number, pairOffsets: Uint32Array }>}
     */
    placements = [];

    /**
     * Per added-resource identity, in add order - what a rebuild compares to decide what carries
     * (see MeshletRebuildCarry#adoptStreamingState): `{ resource, baseUrl, streamed, pageCount,
     * instances, repeat }`, `instances` being the placement list as added and `repeat` its run
     * count.
     *
     * @type {Array<object>}
     */
    buildEntries = [];

    // What a rebuild hands this world from the previous one (see
    // MeshletRebuildCarry#adoptStreamingState), consumed by finalize: the page pool buffer, the
    // residency map whose first adoptedPages entries stay valid, and the texture system.
    /** @type {StorageBuffer|null} */
    adoptPagePool = null;

    /** @type {Uint32Array|null} */
    adoptResidency = null;

    adoptedPages = 0;

    /** @type {MeshletTextures|null} */
    adoptTextures = null;

    /** @type {number} - textured resources (in add order) the adopted texture system already holds; the rest are appended to it. */
    adoptTexturesPrefix = 0;

    /** @type {boolean} - texture-state debug view on the lit materials (MESHLET_COLOR_MODE.TEXTURES). @private */
    _texDebug = false;

    /**
     * Largest source size streamed into the fine texture pool (see
     * {@link MeshletTextures#maxFineSize}). Applied to the texture system this world creates.
     *
     * @type {number}
     */
    maxFineTextureSize = 2048;

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
     * @param {MeshletInstanceList|Array<{ primIndex: number, matrix: Float32Array }>|null} [instances] - Optional
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
     * @param {Mat4|Mat4[]|null} transform - Root transform applied to the asset's placements, or
     * an array of them: the placement list is then repeated once per transform (a merged
     * placement, one run per component).
     * @param {string} baseUrl - URL directory the manifest's blob URIs are relative to.
     * @param {MeshletInstanceList|Array<{ primIndex: number, matrix: Float32Array }>|null} [instances] - Optional
     * placement list replacing the resource's own - many instances share one set of pages.
     * @param {MeshletFetchOptions|null} [fetchOptions] - How the resource's sidecars (geometry
     * shards, texture containers) are fetched. `resolveUrl` maps a manifest URI to the URL it is
     * fetched from in place of appending it to the base URL - the component system passes the
     * application's URL resolver, so a host storing a package's files behind rewritten or signed
     * URLs can serve them; `credentials` is the fetch credentials mode for hosts that need
     * cookies. Null appends the URI to the base URL and sends no credentials.
     */
    addStreamedResource(resource, transform, baseUrl, instances = null, fetchOptions = null) {
        Debug.assert(!this.finalized, 'MeshletWorld: addStreamedResource after finalize');
        this.streamed = true;
        this._pending.push({ resource, transform, baseUrl, instances, fetchOptions });
    }

    /**
     * Largest page pool, in pages, a single buffer can hold on this device (see
     * {@link maxStorageBytes}).
     *
     * @returns {number} Page count.
     */
    maxPoolSlots() {
        return Math.max(Math.floor(maxStorageBytes(this.device) / this.pageSizeBytes), 1);
    }

    /**
     * Builds all GPU state from the added resources.
     */
    finalize() {
        Debug.assert(!this.finalized);
        this.finalized = true;
        try {
            this._build();
        } catch (err) {
            // a half-built world must not read as usable: the director and the component
            // system gate on `finalized`, so leaving it set would crash every frame on the
            // missing buffers and materials instead of surfacing the build error once
            this.finalized = false;
            throw err;
        }
    }

    /**
     * The finalize body: builds every GPU table and the bucket materials from the pending
     * resources.
     *
     * @private
     */
    _build() {
        const pending = this._pending;
        Debug.assert(pending.length, 'MeshletWorld: no resources added');

        // all resources must share one page size (pool slot size); attribute layouts may differ
        this.pageSizeBytes = pending[0].resource.manifest.pageSizeBytes;
        Debug.assert(pending.every(({ resource }) => resource.manifest.pageSizeBytes === this.pageSizeBytes),
            'MeshletWorld: all resources must share one page size');

        const totals = this._countTotals(pending);
        this.totalPages = totals.totalPages;
        this.rootPages = totals.rootPages;
        this.instanceCount = totals.totalInstances;
        this.materialRowCount = totals.totalMaterialRows;

        // an adopted texture system is already finalized and possibly fully loaded; resources
        // after the prefix it holds are appended to it
        const texturesAdopted = !!this.adoptTextures;
        this.textures = this.adoptTextures ?? (totals.anyTextures ? new MeshletTextures(this.device, this.transcode, this.fetchScheduler) : null);
        if (this.textures && !texturesAdopted) {
            this.textures.poolBytes = this.texturePoolBytes;
            this.textures.maxFineSize = this.maxFineTextureSize;
        }

        // the budget is resolved before anything is allocated
        this._applyBudget(totals);
        this._createPagePool();

        const primitiveRows = new Uint32Array(totals.totalPrimitiveRows * OBJECT_DATA_U32S);
        const build = {
            materials: new MeshletMaterialTable(totals.totalMaterialRows),
            // meshletData is not assembled on the CPU: a world of millions of meshlets would hold
            // a GB-scale copy of records its resources already carry. Each primitive's records are
            // copied, rebased and stamped only as the cut tables consume them (see _createTables),
            // from what this keeps per global first meshlet.
            /** @type {Map<number, { prim: MeshletPrimitive, pageBase: number, bucketBits: number }>} */
            primRecords: new Map(),
            primitiveRows,
            primitiveRowsF: new Float32Array(primitiveRows.buffer),
            // per instance, its primitive row
            instancePrimitive: new Uint32Array(totals.totalInstances),
            lightmapData: totals.anyLightmaps ? new Float32Array(totals.totalInstances * LIGHTMAP_ROW_FLOATS) : null,
            /** @type {number[]} - (material row, texture index, tiling bias) per lightmapped instance */
            lightmapSlots: [],
            texturesAdopted,
            adoptedTexturedResources: texturesAdopted ? this.adoptTexturesPrefix : 0,
            texturedResources: 0,
            // the running bases of the address spaces (see the class description)
            pageBase: 0,
            meshletBase: 0,
            instanceBase: 0,
            primitiveRow: 0,
            materialBase: 0,
            textureBase: 0
        };
        this.adoptTextures = null;
        this.adoptTexturesPrefix = 0;

        this.instanceFlags = new Uint32Array(totals.totalInstances);
        this._placementCursor = 0;
        this.recordCapacity = 0;
        this.indexCapacity = [0, 0, 0];
        this.indexWorst = [0, 0, 0];

        // One pick id per instance (see pickRecords), from one block of the engine's picker
        // space, so they never collide with mesh instance ids.
        this.pickIdBase = PickerId.reserve(totals.totalInstances);
        this._pickRecords = null;

        for (const entry of pending) this._addPlacement(entry, build);

        this._worldBoundsDirty = false;
        this._worldBounds.center.set(0, 0, 0);
        this._worldBounds.halfExtents.set(0, 0, 0);
        this._growWorldBounds(0, totals.totalInstances, false);

        this._initIndexCapacities();
        this._createTables(build, totals);
        this._createMaterials(build, totals);
        this._colorMode = 0;
        this._pending = [];
    }

    /**
     * Totals the pending resources, and resolves each entry's placement list and run count.
     *
     * @param {Array<object>} pending - The pending resources.
     * @returns {MeshletBuildTotals} The totals.
     * @private
     */
    _countTotals(pending) {
        const totals = {
            totalPages: 0,
            totalMeshlets: 0,
            totalInstances: 0,
            totalPrimitiveRows: 0,
            totalMaterialRows: 0,
            pairs: 0,
            rootPages: [],
            anyLightmaps: false,
            anyTextures: false,
            uvChannels: 0,
            anyTangents: false,
            anyColors: false
        };
        const rootPages = new Set();
        const scanned = new Set();
        for (const entry of pending) {
            const { resource, instances, transform } = entry;
            const manifest = resource.manifest;
            const layout = manifest.attributeLayout;
            const list = entry.list = MeshletInstanceList.from(instances ?? resource.instances);
            const repeat = entry.repeat = Array.isArray(transform) ? transform.length : 1;
            const pageBase = totals.totalPages;
            totals.totalPages += manifest.pageCount;
            totals.totalMeshlets += resource.totalMeshlets;
            totals.totalPrimitiveRows += resource.primitives.length;
            totals.totalInstances += list.count * repeat;
            totals.anyLightmaps ||= !!list.lightmaps?.some(lightmap => !!lightmap);
            if (resource.materialTable) {
                // the baked records, plus a fallback row for primitives without a material
                const unassigned = resource.primitives.some(p => p.materialIndex < 0) ? 1 : 0;
                totals.totalMaterialRows += resource.materialCount + unassigned;
            } else {
                // a synthesized row per primitive
                totals.totalMaterialRows += resource.primitives.length;
            }
            totals.anyTextures ||= !!resource.textureManifest;
            totals.uvChannels = Math.max(totals.uvChannels, layout.uvComponents ?? 0);
            totals.anyTangents ||= !!layout.tangents;
            totals.anyColors ||= !!layout.colors;
            for (const key of Object.keys(layout)) {
                // a stream this build cannot place shifts every block after it; shipped in release
                // builds because an engine older than its assets otherwise renders garbage silently
                if (!KNOWN_LAYOUT_KEYS.has(key)) console.warn(`MeshletWorld: unknown page attribute '${key}' in a streamed asset's attributeLayout - its pages will decode wrongly; update the engine.`);
            }

            // pairs, and the root pages of the primitives some instance uses
            scanned.clear();
            for (let i = 0; i < list.count; i++) {
                const prim = resource.primitives[list.primIndex[i]];
                totals.pairs += prim.meshletCount * repeat;
                if (scanned.has(prim)) continue;
                scanned.add(prim);
                for (const page of primitiveRootPages(prim)) rootPages.add(pageBase + page);
            }
        }
        totals.rootPages = Array.from(rootPages).sort((a, b) => a - b);
        return totals;
    }

    /**
     * Resolves {@link poolBytes} into the page pool and the draw working set (see
     * resolveMeshletBudget).
     *
     * @param {MeshletBuildTotals} totals - The scene totals.
     * @private
     */
    _applyBudget(totals) {
        const budget = resolveMeshletBudget({
            poolBytes: this.poolBytes,
            indexBudgetFraction: this.indexBudgetFraction,
            cameraViews: this.budgetCameraViews,
            shadowViews: this.budgetShadowViews,
            pageSizeBytes: this.pageSizeBytes
        }, {
            totalPages: totals.totalPages,
            totalMeshlets: totals.totalMeshlets,
            totalInstances: totals.totalInstances,
            totalPrimitiveRows: totals.totalPrimitiveRows,
            totalMaterialRows: totals.totalMaterialRows,
            pairs: totals.pairs,
            lightmapBytes: totals.anyLightmaps ? totals.totalInstances * LIGHTMAP_ROW_FLOATS * 4 : 0,
            rootPages: totals.rootPages.length
        });
        this.maxIndices = 0;
        this.budgetBreakdown = budget?.breakdown ?? null;
        this.pagePoolBytes = budget?.pagePoolBytes ?? 0;
        this.recordCeiling = budget?.recordCeiling ?? 0;
        this.cutQueueCeiling = budget?.cutQueueCeiling ?? 0;
        this.shadowRecordCeiling = budget?.shadowRecordCeiling ?? 0;
        this.shadowCutQueueCeiling = budget?.shadowCutQueueCeiling ?? 0;
        if (budget) this.indexBudgetTotal = budget.indexBudgetTotal;
    }

    /**
     * Creates the page pool and the residency map. A resident world holds every page, slot =
     * rebased page index; a streamed one is sized to the budget and starts with nothing
     * resident (the residency pins the roots and streams the rest on demand).
     *
     * A pool carried from the previous world keeps its resident pages valid, because the carried
     * resources keep their page-space prefix. It is kept whole while large enough - the resolved
     * pool drifts down a little between builds once the budget caps it, and shrinking to match
     * would restart streaming cold for a few slots - unless it is more than twice what the scene
     * wants. One the scene outgrew is copied into the larger pool, so its pages stay resident at
     * their slots. Otherwise its slots would not fit, and streaming starts cold.
     *
     * @private
     */
    _createPagePool() {
        const device = this.device;
        const totalPages = this.totalPages;
        const pageSize = this.pageSizeBytes;
        let poolSlots = totalPages;
        if (this.streamed) {
            poolSlots = this.pagePoolBytes > 0 ? Math.max(Math.floor(this.pagePoolBytes / pageSize), 1) : totalPages;
            poolSlots = Math.min(poolSlots, totalPages);
        }
        const maxSlots = this.maxPoolSlots();
        if (poolSlots > maxSlots) {
            Debug.warnOnce(`MeshletWorld: page pool of ${poolSlots} pages (${(poolSlots * pageSize / BYTES_PER_MB).toFixed(0)} MB) exceeds the device buffer limit; clamped to ${maxSlots} pages.`);
            poolSlots = maxSlots;
        }

        const adopted = this.adoptPagePool;
        this.adoptPagePool = null;
        const poolByteSize = poolSlots * pageSize;
        if (adopted && adopted.byteSize >= poolByteSize && adopted.byteSize <= poolByteSize * 2) {
            this.pagePool = adopted;
            poolSlots = adopted.byteSize / pageSize;
        } else if (adopted && adopted.byteSize < poolByteSize) {
            this.pagePool = new StorageBuffer(device, poolByteSize, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
            // submitted at once rather than on the frame's encoder (StorageBuffer#copy): page
            // installs are queue writes, and must land after the copy, not be overwritten by it
            const wgpu = device.wgpu;
            const encoder = wgpu.createCommandEncoder();
            encoder.copyBufferToBuffer(adopted.impl.buffer, 0, this.pagePool.impl.buffer, 0, adopted.byteSize);
            wgpu.queue.submit([encoder.finish()]);
            adopted.destroy();
        } else {
            adopted?.destroy();
            this.adoptResidency = null;
            this.adoptedPages = 0;
            this.pagePool = new StorageBuffer(device, poolByteSize, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
        }
        this.poolSlots = poolSlots;

        this.residency = new Uint32Array(totalPages);
        if (this.streamed) {
            this.residency.fill(PAGE_NOT_RESIDENT);
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
    }

    /**
     * Lays out one pending resource: its pages (uploaded now, or registered for streaming), its
     * material rows and textures, its primitive rows and its instances.
     *
     * @param {object} entry - The pending resource.
     * @param {object} build - The build state (see _build).
     * @private
     */
    _addPlacement(entry, build) {
        const { resource, transform, shardBuffers, baseUrl, instances, list, repeat } = entry;
        const manifest = resource.manifest;
        // the bake stamp versions every sidecar's cache key, so a rebake over the same URLs can
        // never be served the previous bake's cached pages or texture regions
        const fetchOptions = { ...entry.fetchOptions, cacheVersion: meshletBakeStamp(resource) };
        // the world keeps its own copy of each root: rows are regenerated from them later, and
        // the caller's matrices (an entity's world transform) keep changing
        const ownRoot = (root) => {
            if (!root) return null;
            const own = new Mat4();
            own.data.set(root.data);
            return own;
        };
        const placement = {
            resource,
            instances: list,
            repeat,
            roots: Array.isArray(transform) ? transform.map(ownRoot) : [ownRoot(transform)],
            instanceBase: build.instanceBase,
            instanceCount: list.count * repeat,
            primitiveRow: build.primitiveRow,
            pairBase: this.totalPairs,
            pairCount: 0,
            runPairs: 0,
            pairOffsets: new Uint32Array(list.count)
        };
        this.placements.push(placement);
        this.buildEntries.push({
            resource,
            baseUrl: baseUrl ?? null,
            streamed: !shardBuffers,
            pageCount: manifest.pageCount,
            instances: instances ?? resource.instances,
            repeat
        });

        if (shardBuffers) {
            // fully resident: upload every page now, slot = rebased page index
            const table = manifest.pageTable;
            for (let p = 0; p < manifest.pageCount; p++) {
                const field = p * PAGE_TABLE_FIELDS;
                const blob = table[field + PAGE_TABLE.BLOB];
                const offset = table[field + PAGE_TABLE.OFFSET_HI] * 0x100000000 + table[field + PAGE_TABLE.OFFSET_LO];
                this.pagePool.write((build.pageBase + p) * this.pageSizeBytes,
                    new Uint32Array(shardBuffers[blob], offset, this.pageSizeBytes / 4));
            }
        } else {
            this.streamInfo.push({ resource, pageBase: build.pageBase, baseUrl, fetchOptions });
        }

        // material rows: the baked table verbatim, or one synthesized row per primitive
        const materials = build.materials;
        const materialBase = build.materialBase;
        const baked = resource.materialTable;
        let resourceRows = resource.primitives.length;
        let fallbackRow = -1;
        if (baked) {
            materials.setRecords(baked, materialBase);
            resourceRows = resource.materialCount;
            if (resource.primitives.some(p => p.materialIndex < 0)) {
                fallbackRow = resourceRows++;
                materials.writeFallback(materialBase + fallbackRow, null);
            }
        }

        const resourceTextureBase = build.textureBase;
        this._addTextures(resource, baseUrl, fetchOptions, build);

        const primMatRows = [];
        const primBucketCorners = [];
        const primFlags = [];
        const layout = manifest.attributeLayout;
        const objectFlags = (layout.tangents ? OBJECT_FLAG_HAS_TANGENTS : 0) | (layout.colors ? OBJECT_FLAG_HAS_COLORS : 0);
        for (let primIndex = 0; primIndex < resource.primitives.length; primIndex++) {
            const prim = resource.primitives[primIndex];
            primitiveStats(prim);
            const primFirst = build.meshletBase;
            build.meshletBase += prim.meshletCount;

            let materialRow = primIndex;
            if (baked) {
                materialRow = prim.materialIndex >= 0 ? prim.materialIndex : fallbackRow;
            } else {
                materials.writeFallback(materialBase + primIndex, prim.baseColorFactor);
            }
            primMatRows.push(materialRow);

            // A primitive has exactly one material, so its draw bucket is a per-primitive
            // property. Without a baked material record there is nothing better than the
            // meshlet flags themselves.
            const bucket = baked ? materials.bucket(materialBase + materialRow) : bucketOfMeshletFlags(prim._firstFlags);

            // the bucket is stamped onto every meshlet as its records are copied; the worst-case
            // index count per bucket is totalled once per primitive here, so the per-instance
            // loop below just adds these up
            build.primRecords.set(primFirst, { prim, pageBase: build.pageBase, bucketBits: meshletFlagsOfBucket(bucket) });
            const primCorners = [0, 0, 0];
            primCorners[bucket] = prim._indexCount;
            primBucketCorners.push(primCorners);
            primFlags.push(objectFlags | (bucket << OBJECT_BUCKET_SHIFT));

            this._writePrimitiveRow(build, build.primitiveRow + primIndex, prim, primFirst, materialBase + materialRow, manifest);
        }

        // One run of the placement list: where each instance's pair bits start within it, and
        // the run's worst-case capacities (every meshlet of every instance drawn).
        let runPairs = 0;
        const runCorners = [0, 0, 0];
        for (let i = 0; i < list.count; i++) {
            const p = list.primIndex[i];
            placement.pairOffsets[i] = runPairs;
            runPairs += resource.primitives[p].meshletCount;
            for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) runCorners[b] += primBucketCorners[p][b];
        }
        placement.runPairs = runPairs;
        placement.pairCount = runPairs * repeat;
        this.totalPairs += placement.pairCount;
        this.recordCapacity += placement.pairCount;
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) this.indexCapacity[b] += runCorners[b] * repeat;

        // instances, run by run: one run of the list per root transform
        for (let run = 0; run < repeat; run++) {
            for (let i = 0; i < list.count; i++) {
                const instance = build.instanceBase++;
                const p = list.primIndex[i];
                this.instanceFlags[instance] = primFlags[p];
                build.instancePrimitive[instance] = build.primitiveRow + p;
                const lightmap = build.lightmapData && baseUrl ? list.getLightmap(i) : null;
                if (lightmap) {
                    this._writeLightmap(build, instance, lightmap, resource.primitives[p], resource, resourceTextureBase, materialBase + primMatRows[p]);
                }
            }
        }

        build.primitiveRow += resource.primitives.length;
        build.pageBase += manifest.pageCount;
        build.materialBase += resourceRows;
    }

    /**
     * Registers a resource's streamed textures with the texture system (unless an adopted system
     * already holds them) and rebases its material rows' slot texture indices into the world's
     * flat texture space.
     *
     * @param {MeshletResource} resource - The resource.
     * @param {string|undefined} baseUrl - Its base URL; resident adds carry none.
     * @param {MeshletFetchOptions} fetchOptions - How its containers are fetched.
     * @param {object} build - The build state (see _build).
     * @private
     */
    _addTextures(resource, baseUrl, fetchOptions, build) {
        if (!resource.textureManifest || !this.textures) return;
        // position among the textured resources, in add order
        const textured = build.texturedResources++;
        if (!baseUrl) {
            Debug.warnOnce('MeshletWorld: resource has a texture manifest but no base URL (resident add) - its textures are skipped.');
            return;
        }
        const textureBase = build.textureBase;
        for (const array of resource.textureManifest.arrays ?? []) {
            if (array.containerVersion === 2) build.textureBase += Array.isArray(array.layers) ? array.layers.length : 0;
        }
        if (!build.texturesAdopted) {
            this.textures.addResource(resource.textureManifest, baseUrl, fetchOptions);
        } else if (textured >= build.adoptedTexturedResources) {
            this.textures.appendResource(resource.textureManifest, baseUrl, fetchOptions);
        }
        if (resource.materialTable && textureBase > 0) {
            build.materials.rebaseTextures(build.materialBase, resource.materialCount, textureBase);
        }
    }

    /**
     * Writes a primitive's objectData row, shared by every instance of the primitive.
     *
     * @param {object} build - The build state (see _build).
     * @param {number} rowIndex - The primitive row.
     * @param {MeshletPrimitive} prim - The primitive.
     * @param {number} firstMeshlet - Global id of its first meshlet.
     * @param {number} materialRow - Its material row.
     * @param {object} manifest - Its resource's stream manifest.
     * @private
     */
    _writePrimitiveRow(build, rowIndex, prim, firstMeshlet, materialRow, manifest) {
        const { primitiveRows: rows, primitiveRowsF: rowsF } = build;
        const grid = manifest.positionGrid;
        const row = rowIndex * OBJECT_DATA_U32S;
        const halfExtents = prim.aabbHalfExtents;
        for (let c = 0; c < 3; c++) {
            rowsF[row + PRIMITIVE_DATA.SPHERE + c] = prim.aabbCenter[c];
            rowsF[row + PRIMITIVE_DATA.GRID_ORIGIN + c] = grid.origin[c];
        }
        rowsF[row + PRIMITIVE_DATA.SPHERE + 3] = Math.hypot(halfExtents[0], halfExtents[1], halfExtents[2]);
        rowsF[row + PRIMITIVE_DATA.GRID_STEP] = grid.step;
        rows[row + PRIMITIVE_DATA.FIRST_MESHLET] = firstMeshlet;
        rows[row + PRIMITIVE_DATA.MESHLET_COUNT] = prim.meshletCount;
        rows[row + PRIMITIVE_DATA.MATERIAL] = materialRow;
        rows[row + PRIMITIVE_DATA.UV_FLOATS_PER_VERTEX] = (manifest.attributeLayout.uvComponents ?? 0) * 2;
        rows[row + PRIMITIVE_DATA.PICK_ID_BASE] = this.pickIdBase;
    }

    /**
     * Writes an instance's lightmap row and registers its texture for mip feedback. A reference
     * to a missing texture layer or UV channel, or malformed coefficients, leaves the instance
     * unlightmapped.
     *
     * @param {object} build - The build state (see _build).
     * @param {number} instance - The instance.
     * @param {object} lightmap - Its MeshletLightmap.
     * @param {MeshletPrimitive} prim - Its primitive.
     * @param {MeshletResource} resource - Its resource.
     * @param {number} resourceTextureBase - Flat index of the resource's first texture.
     * @param {number} materialRow - The primitive's material row.
     * @private
     */
    _writeLightmap(build, instance, lightmap, prim, resource, resourceTextureBase, materialRow) {
        const reference = lightmap.texture;
        let textureIndex = resourceTextureBase;
        let textureArray = null;
        for (const array of resource.textureManifest?.arrays ?? []) {
            if (array.containerVersion !== 2) continue;
            if (array.id === reference.arrayId) {
                textureArray = array;
                break;
            }
            textureIndex += array.layers.length;
        }
        const channel = reference.texCoord ?? 0;
        const coefficients = [lightmap.coordinateScaleBias, lightmap.lightmapScale, lightmap.lightmapAdd];
        if (!textureArray || !Number.isInteger(reference.layer) || reference.layer < 0 || reference.layer >= textureArray.layers.length ||
            !Number.isInteger(channel) || channel < 0 || channel >= MESHLET_MAX_UV_CHANNELS || !(prim.uvChannelMask & (1 << channel)) ||
            !coefficients.every(values => values?.length === 4 && values.every(Number.isFinite))) {
            Debug.warnOnce('MeshletWorld: invalid streamed EPIC lightmap reference, coefficients or UV channel; lightmap ignored.');
            return;
        }
        textureIndex += reference.layer;
        const data = build.lightmapData;
        const row = instance * LIGHTMAP_ROW_FLOATS;
        data.set(lightmap.coordinateScaleBias, row);
        data.set(lightmap.lightmapScale, row + 4);
        data.set(lightmap.lightmapAdd, row + 8);
        data.set([textureIndex, channel, 1, 0], row + 12);
        const scale = Math.max(Math.abs(lightmap.coordinateScaleBias[0]), Math.abs(lightmap.coordinateScaleBias[1]) * 0.5, 1e-8);
        build.lightmapSlots.push(materialRow, textureIndex, Math.round(TEXEL_RATE_PER_MIP * Math.log2(scale)));
    }

    /**
     * Sets the views' initial index capacities: below the worst case, which is kept as their
     * absolute ceiling. GPU-side per-bucket clamps make a smaller allocation safe, and the views
     * grow to the observed demand.
     *
     * @private
     */
    _initIndexCapacities() {
        this.indexWorst = this.indexCapacity.slice();
        // u32 indices, in a buffer bound both as an index buffer and as storage
        this.deviceIndexCeiling = Math.floor(maxStorageBytes(this.device) / 4);
        const totalIndexWanted = this.indexCapacity.reduce((a, b) => a + b, 0);
        let indexTarget = Math.min(totalIndexWanted, this.indexCeiling);
        if (this.initialIndices > 0) indexTarget = Math.min(indexTarget, this.initialIndices);
        if (indexTarget < totalIndexWanted) {
            const s = indexTarget / totalIndexWanted;
            this.indexCapacity = this.indexCapacity.map(c => Math.floor(c * s));
        }
    }

    /**
     * Creates the GPU tables: spatial cells, meshlet records with their cut tables, objectData,
     * the material table, lightmaps, the shared request marks, and the texture tables.
     *
     * @param {object} build - The build state (see _build).
     * @param {MeshletBuildTotals} totals - The scene totals.
     * @private
     */
    _createTables(build, totals) {
        const device = this.device;
        const { totalMeshlets, totalInstances, totalPages, totalMaterialRows } = totals;
        const { primRecords, primitiveRows, instancePrimitive, lightmapData, materials } = build;

        this.cells = new MeshletCells(device, totalInstances, (instance, out, offset) => this._instanceSphere(instance, out, offset));

        const recordBytes = MESHLET_DATA_U32S * 4;
        const meshletDataBuffer = this.meshletDataBuffer = new StorageBuffer(device, Math.max(totalMeshlets * recordBytes, 16), BUFFERUSAGE_COPY_DST);
        // One primitive's records at a time: a copy of the resource's (which stay untouched for
        // later rebuilds) with page indices rebased and the draw bucket stamped, handed to the
        // cut tables to complete, then uploaded to their place. Only primitives some instance
        // uses are visited. Consecutive primitives - mostly visited in meshlet order - gather in
        // a staging array and go up in one write.
        const staging = new Uint32Array(Math.min(totalMeshlets, RECORD_UPLOAD_BATCH) * MESHLET_DATA_U32S);
        let stagedFirst = 0;
        let stagedCount = 0;
        const flush = () => {
            if (stagedCount) meshletDataBuffer.write(stagedFirst * recordBytes, staging, 0, stagedCount * MESHLET_DATA_U32S);
            stagedCount = 0;
        };
        const recordsOf = {
            count: totalMeshlets,
            get: (first) => {
                const { prim, pageBase, bucketBits } = primRecords.get(first);
                Debug.assert(prim.meshletData, 'MeshletWorld: building with released meshlet records - restore them first');
                const records = prim.meshletData.slice();
                for (let m = 0; m < prim.meshletCount; m++) {
                    const row = m * MESHLET_DATA_U32S;
                    records[row + MESHLET_DATA.PAGE] += pageBase;
                    records[row + MESHLET_DATA.FLAGS] = (records[row + MESHLET_DATA.FLAGS] & ~(MESHLET_FLAG_ALPHA_MASKED | MESHLET_FLAG_TWO_SIDED)) | bucketBits;
                }
                return records;
            },
            done: (first, records) => {
                const count = records.length / MESHLET_DATA_U32S;
                if (count > RECORD_UPLOAD_BATCH) {
                    flush();
                    meshletDataBuffer.write(first * recordBytes, records);
                    return;
                }
                if (stagedCount && (first !== stagedFirst + stagedCount || stagedCount + count > RECORD_UPLOAD_BATCH)) flush();
                if (!stagedCount) stagedFirst = first;
                staging.set(records, stagedCount * MESHLET_DATA_U32S);
                stagedCount += count;
            }
        };
        const primitiveWord = (instance, word) => primitiveRows[instancePrimitive[instance] * OBJECT_DATA_U32S + word];
        this.cut = new MeshletCutData(device, recordsOf, {
            count: totalInstances,
            first: instance => primitiveWord(instance, PRIMITIVE_DATA.FIRST_MESHLET),
            meshletCount: instance => primitiveWord(instance, PRIMITIVE_DATA.MESHLET_COUNT)
        });
        flush();

        this.objectDataBuffer = new StorageBuffer(device, Math.max((totalInstances + totals.totalPrimitiveRows) * OBJECT_DATA_U32S * 4, 16), BUFFERUSAGE_COPY_DST);
        this._writeObjectRows(0, totalInstances);
        if (primitiveRows.length) this.objectDataBuffer.write(totalInstances * OBJECT_DATA_U32S * 4, primitiveRows);
        this.materialTableBuffer = new StorageBuffer(device, materials.data.byteLength, BUFFERUSAGE_COPY_DST);
        this.materialTableBuffer.write(0, materials.data);
        if (lightmapData) {
            this.lightmapBuffer = new StorageBuffer(device, lightmapData.byteLength, BUFFERUSAGE_COPY_DST);
            this.lightmapBuffer.write(0, lightmapData, 0, lightmapData.length);
        }
        this.requestsBuffer = new StorageBuffer(device, Math.max((totalPages + totalMaterialRows) * 4, 16), BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
        this.dummyBits = new StorageBuffer(device, 16, BUFFERUSAGE_COPY_DST);

        // the texture tails and residency exist before the materials bind them
        if (build.texturesAdopted) {
            this.textures.appendFinalize();
        } else {
            this.textures?.finalize();
        }
        this.textures?.setMaterialSlots(materials.data, totalMaterialRows, MATERIAL_RECORD_U32S, MATERIAL_RECORD.SLOT_WORDS);
    }

    /**
     * Creates the engine-lit bucket materials (one per MESHLET_BUCKET_*, shared by both phases
     * and every view) and wires the texture system's feedback maps. The unlit debug materials of
     * the colour modes are created on first use (see {@link setColorMode}).
     *
     * @param {object} build - The build state (see _build).
     * @param {MeshletBuildTotals} totals - The scene totals.
     * @private
     */
    _createMaterials(build, totals) {
        const textures = this.textures;
        // per-world chunk set: texture sampling, UV varyings and tangents compile in only when
        // the resources carry them
        const chunks = buildMeshletLitChunks({
            textures: !!textures,
            uvChannels: Math.min(totals.uvChannels, MESHLET_MAX_UV_CHANNELS),
            tangents: totals.anyTangents,
            colors: totals.anyColors,
            lightmaps: totals.anyLightmaps,
            forwardBackend: ShaderChunks.get(this.device, 'wgsl').get('litForwardBackendPS'),
            familySizes: textures?.families.map(fam => fam.slotSize),
            familyLevels: textures?.families.map(fam => ({ slotLevels: fam.slotLevels, tailLevels: fam.levels }))
        });

        this._litMaterials = [];
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            const material = createMeshletLitMaterial(b, chunks);
            if (this._texDebug) {
                material.setDefine('MESHLET_TEX_DEBUG', true);
                material.update();
            }
            this._bindWorldParams(material);
            this._bindTextureParams(material);
            this._litMaterials.push(material);
        }
        this._debugMaterials = null;

        if (textures) {
            // family arrays are created once the first transcode decides their format - swap the
            // placeholder bindings for the real textures as each family lands
            textures.onFamilyTexturesReady = (family, tail, fine) => {
                this._litMaterials?.forEach((m) => {
                    m.setParameter(_tailBindName(family), tail);
                    if (fine) m.setParameter(_fineBindName(family), fine);
                });
            };
            const { rowTex, rowTilingBias } = build.materials.slotTextureMaps();
            textures.setMaterialSlotMap(rowTex, rowTilingBias);
            textures.setLightmapSlotMap(new Int32Array(build.lightmapSlots));
        }
    }

    /**
     * Binds the world's storage buffers to a bucket material (lit or debug). The per-view
     * records buffer is not bound here - each view's mesh instances carry it as an
     * instance-level parameter.
     *
     * @param {Material} material - The material.
     * @private
     */
    _bindWorldParams(material) {
        material.setParameter('meshletData', this.meshletDataBuffer);
        material.setParameter('objectData', this.objectDataBuffer);
        material.setParameter('pagePool', this.pagePool);
        material.setParameter('residency', this.residencyBuffer);
        material.setParameter('materialTable', this.materialTableBuffer);
        if (this.lightmapBuffer) material.setParameter('meshletLightmaps', this.lightmapBuffer);
        material.setParameter('pageSizeWords', this.pageSizeBytes / 4);
    }

    /**
     * Binds the texture system's tables and family arrays (placeholders until they land) to a
     * lit material.
     *
     * @param {Material} material - The material.
     * @private
     */
    _bindTextureParams(material) {
        const textures = this.textures;
        if (!textures) return;
        material.setParameter('texResidency', textures.residencyBuffer);
        material.setParameter('meshletSlotDesc', textures.slotDescBuffer);
        for (let f = 0; f < MESHLET_TEX_FAMILIES.length; f++) {
            const { tail, fine } = textures.familyTextures(f);
            material.setParameter(_tailBindName(f), tail);
            material.setParameter(_fineBindName(f), fine);
        }
    }

    /**
     * The current bucket materials, indexed by MESHLET_BUCKET_* - lit, or the unlit debug
     * materials when a colour mode is active. Views compare this by identity each frame and
     * swap their mesh instances' materials when it changes.
     *
     * @type {Material[]}
     */
    get bucketMaterials() {
        return this._colorMode > 0 && this._colorMode !== MESHLET_COLOR_MODE.TEXTURES ? this._debugMaterials : this._litMaterials;
    }

    /**
     * The most indices one view may allocate: the device's own limit, lowered by
     * {@link maxIndices} when the caller sets one.
     *
     * @type {number}
     */
    get indexCeiling() {
        const device = this.deviceIndexCeiling || Infinity;
        const budget = this.indexBudgetTotal;
        const capped = this.maxIndices > 0 ? Math.min(this.maxIndices, budget || this.maxIndices) : budget;
        return capped > 0 ? Math.min(capped, device) : device;
    }

    /**
     * One record per instance, for resolving a picked id. An instance is one primitive at one
     * transform, so this is glTF submesh granularity - a table baked with its top and legs as
     * separate primitives picks apart exactly as it would have unbaked.
     *
     * Built on the first pick and cached: an app that never picks pays nothing, and the records
     * are what the picker's id map needs distinct values of (it de-duplicates by identity, so
     * handing it one shared object per placement would collapse every instance into one hit).
     *
     * @type {MeshletPickRecord[]}
     */
    get pickRecords() {
        if (this._pickRecords) return this._pickRecords;
        const records = [];
        for (const placement of this.placements) {
            const list = placement.instances;
            for (let i = 0; i < placement.instanceCount; i++) {
                const index = i % list.count;
                records.push(new MeshletPickRecord(
                    this.pickIdBase + placement.instanceBase + i,
                    placement,
                    this.instanceOffset + placement.instanceBase + i,
                    list.primIndex[index],
                    index
                ));
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
     * @type {Material[]}
     */
    get litMaterials() {
        return this._litMaterials;
    }

    /**
     * The debug color mode (MESHLET_COLOR_MODE) set by {@link setColorMode}.
     *
     * @type {number}
     */
    get colorMode() {
        return this._colorMode;
    }

    /**
     * Sets the debug color mode (MESHLET_COLOR_MODE): MESHLET swaps in the unlit per-meshlet
     * debug materials (applied to each view's mesh instances via
     * {@link MeshletView#syncMaterials}); TEXTURES keeps the lit ones, painting each pixel with
     * its texture-sample state. Call after {@link finalize}.
     *
     * @param {number} mode - The color mode.
     */
    setColorMode(mode) {
        const texDebug = mode === MESHLET_COLOR_MODE.TEXTURES;
        if (mode > 0 && !texDebug && !this._debugMaterials) {
            this._debugMaterials = [];
            for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
                const material = createMeshletMaterial(b);
                this._bindWorldParams(material);
                this._debugMaterials.push(material);
            }
        }
        this._colorMode = mode;
        if (this._texDebug !== texDebug) {
            this._texDebug = texDebug;
            this._litMaterials?.forEach((m) => {
                m.setDefine('MESHLET_TEX_DEBUG', texDebug);
                m.update();
            });
        }
    }

    /**
     * Updates the root transform of an added resource (placement), rewriting its objectData rows.
     * Only the transform-derived fields change; culling uses the new matrix next frame.
     * Grows {@link worldBounds} to the new placement; it is refitted exactly later.
     *
     * A placement holds one root transform per run of its instance list (see
     * {@link placements}), so the range is widened to the whole runs it touches.
     *
     * @param {number} index - The placement index (resource add order).
     * @param {Mat4|null} transform - New root transform applied to the resource's placements.
     * @param {number} [subBase] - First instance within the placement (default 0).
     * @param {number} [subCount] - Instance count (default: the rest of the placement).
     */
    setPlacementTransform(index, transform, subBase = 0, subCount = -1) {
        Debug.assert(this.finalized && this.placements[index], 'MeshletWorld: invalid placement');
        const placement = this.placements[index];
        const { instanceBase, instanceCount } = placement;
        const start = Math.min(subBase, instanceCount);
        const count = subCount < 0 ? instanceCount - start : Math.min(subCount, instanceCount - start);
        if (count === 0) return;

        const per = placement.instances.count;
        const firstRun = Math.floor(start / per);
        const lastRun = Math.floor((start + count - 1) / per);
        Debug.assert(start % per === 0 && count % per === 0, 'MeshletWorld: a placement transform applies to whole runs of its instance list');
        for (let run = firstRun; run <= lastRun; run++) {
            if (transform) {
                const root = placement.roots[run] ?? (placement.roots[run] = new Mat4());
                root.data.set(transform.data);
            } else {
                placement.roots[run] = null;
            }
        }
        const first = instanceBase + firstRun * per;
        const rows = (lastRun - firstRun + 1) * per;
        this._growWorldBounds(first, rows, true);
        this._worldBoundsDirty = true;
        this.cells?.refit(first, rows);
        this._uploadObjectRows(first, rows);
    }

    /**
     * Hides or shows a range of a placement's instances without a rebuild - the instance cull
     * pass skips hidden rows (OBJECT_FLAG_HIDDEN), so hidden geometry stops drawing and stops
     * marking pages/textures, letting the LRU drain its residency naturally.
     *
     * @param {number} index - The placement index (resource add order).
     * @param {boolean} hidden - True to hide the instances.
     * @param {number} [subBase] - First instance within the placement (default 0).
     * @param {number} [subCount] - Instance count (default: the rest of the placement).
     */
    setPlacementHidden(index, hidden, subBase = 0, subCount = -1) {
        this._setPlacementFlag(index, OBJECT_FLAG_HIDDEN, hidden, subBase, subCount);
    }

    /**
     * Sets whether a placement's instances cast shadows. A buffer write, not a rebuild: shadow
     * views skip flagged instances in their cull.
     *
     * @param {number} index - Placement index.
     * @param {boolean} casts - True to cast shadows.
     * @param {number} [subBase] - First instance of the placement to affect.
     * @param {number} [subCount] - Instance count (default: the rest of the placement).
     */
    setPlacementCastShadows(index, casts, subBase = 0, subCount = -1) {
        this._setPlacementFlag(index, OBJECT_FLAG_NO_SHADOW, !casts, subBase, subCount);
    }

    /**
     * Sets or clears a flag on a range of a placement's instances and uploads their rows.
     *
     * @param {number} index - Placement index.
     * @param {number} flag - The OBJECT_FLAG_* bit.
     * @param {boolean} set - True to set it.
     * @param {number} subBase - First instance of the placement to affect.
     * @param {number} subCount - Instance count (negative: the rest of the placement).
     * @private
     */
    _setPlacementFlag(index, flag, set, subBase, subCount) {
        Debug.assert(this.finalized && this.placements[index], 'MeshletWorld: invalid placement');
        const { instanceBase, instanceCount } = this.placements[index];
        const start = Math.min(subBase, instanceCount);
        const count = subCount < 0 ? instanceCount - start : Math.min(subCount, instanceCount - start);
        if (count === 0) return;

        const flags = this.instanceFlags;
        for (let i = instanceBase + start; i < instanceBase + start + count; i++) {
            flags[i] = set ? (flags[i] | flag) : (flags[i] & ~flag);
        }
        this._uploadObjectRows(instanceBase + start, count);
    }

    /**
     * Marks instances for the outline pass, by global instance index - the same index a pick
     * returns, so an editor can outline exactly what was clicked.
     *
     * Selection lives in objectData: one indirect draw covers every instance, so there is no
     * mesh instance to add to an outline layer and no per-object uniform to set. The outline
     * variant of the vertex shader collapses unselected instances to a degenerate triangle, so
     * they cost a vertex fetch and no rasterisation.
     *
     * @param {number[]|Set<number>} instanceIndices - Global instance indices to outline.
     * @param {boolean} [outlined] - True to add, false to remove. Defaults to true.
     * @param {boolean} [hover] - Set the hover flag rather than the selection flag.
     */
    setInstancesOutlined(instanceIndices, outlined = true, hover = false) {
        Debug.assert(this.finalized, 'MeshletWorld: setInstancesOutlined before finalize');
        const flag = hover ? OBJECT_FLAG_HOVERED : OBJECT_FLAG_OUTLINED;
        const set = hover ? this._hovered : this._outlined;
        const flags = this.instanceFlags;
        const touched = _touched;
        touched.length = 0;
        for (const globalIndex of instanceIndices) {
            const index = globalIndex - this.instanceOffset;
            if (index < 0 || index >= this.instanceCount) continue;
            flags[index] = outlined ? (flags[index] | flag) : (flags[index] & ~flag);
            if (outlined) set.add(index); else set.delete(index);
            touched.push(index);
        }
        this._writeFlags(touched);
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
        const flags = this.instanceFlags;
        const touched = _touched;
        touched.length = 0;
        for (const index of set) {
            flags[index] &= ~flag;
            touched.push(index);
        }
        set.clear();
        this._writeFlags(touched);
    }

    /**
     * Uploads only the flags word of the given instances' rows. Outline and hover flags change
     * nothing a shadow map depends on, so unlike {@link _uploadObjectRows} this leaves
     * {@link contentVersion} alone.
     *
     * @param {number[]} indices - Instance indices; reordered in place.
     * @private
     */
    _writeFlags(indices) {
        const count = indices.length;
        if (count === 0) return;
        indices.sort((a, b) => a - b);
        const lo = indices[0];
        const span = indices[count - 1] - lo + 1;
        // one write per instance costs a queue call each: past a few hundred, regenerating the
        // whole span in batched writes is cheaper
        if (count > FLAG_WRITE_LIMIT) {
            this._writeObjectRows(lo, span);
            return;
        }
        const flags = this.instanceFlags;
        const flagsOffset = OBJECT_DATA.FLAGS * 4;
        for (let i = 0; i < count; i++) {
            const index = indices[i];
            this.objectDataBuffer.write(index * OBJECT_DATA_U32S * 4 + flagsOffset, flags, index, 1);
        }
    }

    /**
     * Regenerates and uploads a contiguous range of instance rows.
     *
     * @param {number} first - First instance index.
     * @param {number} count - Row count.
     * @private
     */
    _uploadObjectRows(first, count) {
        this.contentVersion++;
        this._writeObjectRows(first, count);
    }

    /**
     * Generates a contiguous range of objectData instance rows from the placements and writes
     * them to the GPU, a batch at a time - the CPU keeps no copy of the rows.
     *
     * @param {number} first - First instance index.
     * @param {number} count - Row count.
     * @private
     */
    _writeObjectRows(first, count) {
        if (count <= 0) return;
        const batch = Math.min(count, OBJECT_ROW_BATCH);
        if (!this._rowScratch || this._rowScratch.length < batch * OBJECT_DATA_U32S) {
            this._rowScratch = new Uint32Array(batch * OBJECT_DATA_U32S);
            this._rowScratchF32 = new Float32Array(this._rowScratch.buffer);
        }
        const u32 = this._rowScratch;
        const f32 = this._rowScratchF32;
        const flags = this.instanceFlags;
        const primitiveBase = this.instanceCount;
        for (let start = first; start < first + count; start += batch) {
            const rows = Math.min(batch, first + count - start);
            for (let k = 0; k < rows; k++) {
                const instance = start + k;
                const { placement, run, index } = this._locate(instance);
                const list = placement.instances;
                list.getMatrix(index, _tmpMat);
                const root = placement.roots[run];
                if (root) _tmpMat.mul2(root, _tmpMat);
                const m = _tmpMat.data;
                const row = k * OBJECT_DATA_U32S;
                const r = row + OBJECT_DATA.MATRIX_ROWS;
                // rows of the column-major matrix
                f32[r] = m[0]; f32[r + 1] = m[4]; f32[r + 2] = m[8]; f32[r + 3] = m[12];
                f32[r + 4] = m[1]; f32[r + 5] = m[5]; f32[r + 6] = m[9]; f32[r + 7] = m[13];
                f32[r + 8] = m[2]; f32[r + 9] = m[6]; f32[r + 10] = m[10]; f32[r + 11] = m[14];
                u32[row + OBJECT_DATA.PRIMITIVE] = primitiveBase + placement.primitiveRow + list.primIndex[index];
                u32[row + OBJECT_DATA.FLAGS] = flags[instance];
                f32[row + OBJECT_DATA.MAX_SCALE] = maxAxisScale(m);
                u32[row + OBJECT_DATA.FIRST_PAIR_BIT] = placement.pairBase + run * placement.runPairs + placement.pairOffsets[index];
            }
            this.objectDataBuffer.write(start * OBJECT_DATA_U32S * 4, u32, 0, rows * OBJECT_DATA_U32S);
        }
    }

    /**
     * Finds the placement holding an instance, and the instance's run and position in the
     * placement's list. Consecutive instances hit the cached placement.
     *
     * @param {number} instance - The instance index.
     * @returns {{ placement: object, run: number, index: number }} Shared scratch - read before
     * the next call.
     * @private
     */
    _locate(instance) {
        const placements = this.placements;
        let placement = placements[this._placementCursor];
        if (!placement || instance < placement.instanceBase || instance >= placement.instanceBase + placement.instanceCount) {
            // the last placement starting at or before the instance: an empty placement shares
            // its start with the next one, which comes later and wins
            let lo = 0;
            let hi = placements.length - 1;
            while (lo < hi) {
                const mid = (lo + hi + 1) >> 1;
                if (placements[mid].instanceBase <= instance) lo = mid; else hi = mid - 1;
            }
            this._placementCursor = lo;
            placement = placements[lo];
        }
        const per = placement.instances.count;
        const local = instance - placement.instanceBase;
        const run = Math.floor(local / per);
        _located.placement = placement;
        _located.run = run;
        _located.index = local - run * per;
        return _located;
    }

    /**
     * Writes an instance's world matrix.
     *
     * @param {number} instance - The instance index.
     * @param {Mat4} out - Receives the matrix.
     * @returns {MeshletPrimitive} The instance's primitive.
     * @private
     */
    _instanceMatrix(instance, out) {
        const { placement, run, index } = this._locate(instance);
        const list = placement.instances;
        list.getMatrix(index, out);
        const root = placement.roots[run];
        if (root) out.mul2(root, out);
        return placement.resource.primitives[list.primIndex[index]];
    }

    /**
     * World-space bounding sphere of one instance, as the cull shaders compute it: the
     * primitive's local sphere through the instance matrix, its radius scaled by the largest
     * axis scale.
     *
     * @param {number} instance - The instance index.
     * @param {Float32Array} out - Destination (x, y, z, radius).
     * @param {number} offset - Destination offset.
     * @private
     */
    _instanceSphere(instance, out, offset) {
        const prim = this._instanceMatrix(instance, _tmpMat);
        const m = _tmpMat.data;
        const [x, y, z] = prim.aabbCenter;
        const h = prim.aabbHalfExtents;
        out[offset] = m[0] * x + m[4] * y + m[8] * z + m[12];
        out[offset + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
        out[offset + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
        out[offset + 3] = Math.fround(Math.hypot(h[0], h[1], h[2])) * Math.fround(maxAxisScale(m));
    }

    /** @type {number} - instances flagged for the outline pass, selected or hovered. */
    get outlinedCount() {
        return this._outlined.size + this._hovered.size;
    }
}

export { MeshletWorld };
