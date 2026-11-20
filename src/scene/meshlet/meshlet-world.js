import { MeshletCells } from './meshlet-cells.js';
import { MeshletCutData } from './meshlet-cut-data.js';
import { MeshletInstanceList } from './meshlet-instance-list.js';
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
    MATERIAL_FLAG_ALPHA_MASK, MATERIAL_FLAG_DOUBLE_SIDED, MATERIAL_RECORD, MATERIAL_RECORD_U32S, MATERIAL_SLOT_ABSENT,
    MATERIAL_TEXTURE_SLOTS, MESHLET_BUCKET_COUNT, MESHLET_BUCKET_MASKED, MESHLET_BUCKET_OPAQUE,
    MESHLET_BUCKET_OPAQUE_TWO_SIDED, MESHLET_DATA, MESHLET_DATA_U32S, MESHLET_FLAG_ALPHA_MASKED,
    MESHLET_FLAG_TWO_SIDED, MESHLET_MAX_UV_CHANNELS, MESHLET_NO_PARENT, OBJECT_DATA, OBJECT_DATA_U32S, OBJECT_FLAG_HAS_TANGENTS, OBJECT_FLAG_HAS_COLORS,
    PRIMITIVE_DATA,
    OBJECT_FLAG_HIDDEN, OBJECT_FLAG_HOVERED, OBJECT_FLAG_OUTLINED,
    PAGE_NOT_RESIDENT, PAGE_TABLE, PAGE_TABLE_FIELDS,
    RECORD_U32S, ROOT_LIST_U32S, TEXEL_RATE_PER_MIP,
    MESHLET_COLOR_MODE, claimWordCount } from './constants.js';
import { createMeshletLitMaterial } from './meshlet-lit-material.js';
import { createMeshletMaterial } from './meshlet-material.js';
import { buildMeshletLitChunks } from './shaders/meshlet-lit-chunks-wgsl.js';
import { MeshletTextures, MESHLET_TEX_FAMILIES } from './textures/meshlet-textures.js';

/** Page attribute streams this build can place (see meshletPageLayoutWGSL). */
const KNOWN_LAYOUT_KEYS = new Set(['positions', 'normals', 'tangents', 'uvComponents', 'colors']);

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { MeshletResource } from './meshlet-resource.js'
 */

/**
 * How a streamed resource's sidecars (geometry shards, texture containers) are fetched.
 *
 * @typedef {object} MeshletFetchOptions
 * @property {((uri: string) => string)|null} [resolveUrl] - Maps a manifest URI to the URL it
 * is fetched from, in place of appending it to the resource's base URL.
 * @property {RequestCredentials|null} [credentials] - Fetch credentials mode for the requests.
 * @property {string|null} [cacheVersion] - Versions the sidecars' persistent cache keys (see
 * {@link MeshletFetchScheduler#byteCache}); the world sets it to the resource's bake stamp.
 */

// meshlet records (128 bytes each) gathered per upload while building a world: 16 MB
const RECORD_UPLOAD_BATCH = 128 * 1024;

// Least the draw index budget is resolved to, whatever the geometry budget leaves for it: 4M
// indices. Root admission takes instances nearest first, so any real capacity draws the closest
// ones; a budget that leaves the indices nothing would admit no instance at all and draw an
// empty scene.
const MIN_INDEX_BUDGET_BYTES = 16 * 1024 * 1024;

// Bytes per record a view holds: the record itself and its selected-meshlet entry.
const RECORD_BYTES = RECORD_U32S * 4 + 8;

// Bytes per cut queue entry: an (instance, group row) pair.
const CUT_QUEUE_ENTRY_BYTES = 8;

// Least each view's records and cut queue are resolved to under a budget: enough for a few
// thousand instances' roots and their first refinements.
const MIN_VIEW_RECORDS = 16 * 1024;
const MIN_VIEW_CUT_QUEUE = 16 * 1024;

// Shares of the draw working set given to all views' records and cut queues, the rest going to
// the indices. A drawn meshlet costs hundreds of indices (four bytes each) against one 24-byte
// record and a cut task for its group, so these are small.
const RECORD_BUDGET_SHARE = 0.05;
const CUT_QUEUE_BUDGET_SHARE = 0.03;

// A shadow view's weight against a camera view's in that split. A cascade cuts far coarser
// than the camera it shadows, and an even split hands a scene with a dozen shadow views a
// camera share too small to refine anything near.
const SHADOW_VIEW_DRAW_WEIGHT = 0.25;

/**
 * Root list entries a view holds: one per instance. Admission ranks every instance in view, so
 * this scales with the scene, not the budget.
 *
 * @param {number} instanceCount - The world's instances.
 * @returns {number} Entries.
 * @ignore
 */
const rootListItems = instanceCount => Math.max(instanceCount, 4);

// objectData instance rows generated per upload (64 bytes each): 4 MB. Rows are generated from
// the placements on demand rather than mirrored, so this is all the CPU holds of them.
const OBJECT_ROW_BATCH = 64 * 1024;

// longest the world bounds stay loose after instances move, in milliseconds
const WORLD_BOUNDS_REFIT_MS = 1000;

// outline and hover changes up to this many instances write just their flags words
const FLAG_WRITE_LIMIT = 256;

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
 * What a picked meshlet id resolves to: one instance, a primitive at one transform.
 *
 * @ignore
 */
class MeshletPickRecord {
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

    /** @type {Array<{ resource: MeshletResource, transform: Mat4|Mat4[]|null, shardBuffers: ArrayBuffer[] }>} */
    _pending = [];

    finalized = false;

    // GPU buffers
    pagePool = null;

    residencyBuffer = null;

    /** @type {StorageBuffer|null} - 16-byte placeholder bound as visBits by single-phase views. */
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

    /** @type {Uint32Array} - the primitive rows of objectData, as uploaded after the instance rows. */
    primitiveRows = new Uint32Array(0);

    /** @type {number} - index of the placement {@link _placementOf} found last. */
    _placementCursor = 0;

    materialTableBuffer = null;

    /** @type {StorageBuffer|null} */
    lightmapBuffer = null;

    /** @type {import('./textures/meshlet-textures.js').MeshletTextures|null} */
    textures = null;

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
     * Grows {@link _worldBounds} by the world boxes of a range of instances.
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
     * (claim/visibility bit planes, work items, records, draw indices) - not just the page pool.
     *
     * Budgeting only the pool is misleading: the demand-grown buffers can dominate it many
     * times over when they have no ceiling. The
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
     * @type {import('./textures/meshlet-ktx2.js').MeshletTranscodeFn|null}
     */
    transcode = null;

    /**
     * The scheduler the world's sidecar fetches (geometry shards, texture containers) go
     * through; the director shares one across its worlds. Null selects the page-wide default.
     *
     * @type {import('./streaming/meshlet-fetch-scheduler.js').MeshletFetchScheduler|null}
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
     * Largest page pool, in pages, a single buffer can hold on this device. The pool is bound
     * as storage, so both the buffer and the storage binding limits apply; exceeding them fails
     * buffer creation with a validation error instead of allocating.
     *
     * @param {import('../../platform/graphics/graphics-device.js').GraphicsDevice} device - Device.
     * @returns {number} Page count.
     */
    maxPoolSlots(device) {
        const limits = device.limits ?? {};
        const limitBytes = Math.min(
            limits.maxBufferSize ?? Number.MAX_SAFE_INTEGER,
            limits.maxStorageBufferBindingSize ?? Number.MAX_SAFE_INTEGER
        );
        return Math.max(Math.floor(limitBytes / this.pageSizeBytes), 1);
    }

    /**
     * Splits {@link poolBytes} into the page pool and the draw working set: the shared index
     * pool and each view's record and cut-queue ceilings. Nothing a view allocates later grows
     * past these, so the budget bounds the pipeline's memory rather than its starting point -
     * what does not fit is not drawn, farthest first (see {@link rootBudgetFraction}).
     *
     * The world tables and each view's scene-sized state (claim and visibility bits, one per
     * instance-meshlet pair, and the root list, one entry per instance) are exactly computable
     * from the scene's counts and come off the top. A shadow view costs far less than a camera
     * one - it borrows the claim plane and root list from its siblings and binds a placeholder
     * for the visibility bits it never touches - so the two are counted separately.
     *
     * @param {object} counts - Scene totals.
     * @returns {number} Bytes for the page pool (0 = unbudgeted, size to every page).
     * @private
     */
    _resolveBudget(counts) {
        const { totalPages, totalMeshlets, totalInstances, totalPrimitiveRows = 0, totalMaterialRows, pairs, lightmapBytes = 0, rootPages = 0 } = counts;
        this.recordCeiling = 0;
        this.cutQueueCeiling = 0;
        this.shadowRecordCeiling = 0;
        this.shadowCutQueueCeiling = 0;
        if (!(this.poolBytes > 0)) {
            this.budgetBreakdown = null;
            this.maxIndices = 0;
            return 0;
        }

        const pairWords = Math.max(Math.ceil((pairs + totalInstances) / 32), 4);
        const claimWords = claimWordCount(pairs, totalInstances);
        // Group headers and membership lists, packed selection links, per-instance root offsets
        // and cell tables.
        const cutBytes = totalMeshlets * (56 + 8) + totalInstances * 44;
        const fixed = cutBytes + lightmapBytes + totalMeshlets * MESHLET_DATA_U32S * 4 +      // meshletData
            (totalInstances + totalPrimitiveRows) * OBJECT_DATA_U32S * 4 +  // objectData
            totalPages * 4 +                                       // residency
            (totalPages + totalMaterialRows) * 4 +                 // requests + texel-rate marks
            totalMaterialRows * MATERIAL_RECORD_U32S * 4;          // material table
        const rootListBytes = rootListItems(totalInstances) * ROOT_LIST_U32S * 4;
        // camera view: own claim plane, own visibility bits, own root list
        const perCamera = (claimWords + pairWords) * 4 + rootListBytes;
        // shadow views: one claim plane and root list shared by every cascade
        const shadows = this.budgetShadowViews;
        const shared = shadows > 0 ? claimWords * 4 + rootListBytes : 0;

        const views = Math.max(this.budgetCameraViews, 1);
        const reserved = fixed + perCamera * views + shared;
        let available = this.poolBytes - reserved;

        // The pool must hold every resource's pinned root pages plus a demand reserve, and the
        // draw working set enough for the nearest instances' roots - without either nothing
        // draws. A budget that cannot even hold the fixed cost is not a budget: keep going with
        // those minimums rather than allocating nothing, and say so.
        const viewCount = views + shadows;
        const minRecordBytes = MIN_VIEW_RECORDS * RECORD_BYTES;
        const minQueueBytes = MIN_VIEW_CUT_QUEUE * CUT_QUEUE_ENTRY_BYTES;
        const minDraw = MIN_INDEX_BUDGET_BYTES + viewCount * (minRecordBytes + minQueueBytes);
        const minPool = Math.min(totalPages, rootPages + Math.min(64, Math.max(totalPages - rootPages, 0))) * this.pageSizeBytes;
        const minimum = minPool + minDraw;
        if (available < minimum) {
            Debug.warnOnce(`MeshletWorld: geometry budget ${(this.poolBytes / 1048576).toFixed(0)} MB is below the ${((reserved + minimum) / 1048576).toFixed(0)} MB this scene needs for its fixed and per-view buffers, its ${rootPages} root pages and a minimal draw working set; falling back to those minimums, so only the nearest instances draw.`);
            available = minimum;
        }

        let drawBytes = Math.floor(available * this.indexBudgetFraction);
        let pagePool = available - drawBytes;
        if (pagePool < minPool) {
            // the roots come before the draw working set
            drawBytes -= minPool - pagePool;
            pagePool = minPool;
        }
        if (drawBytes < minDraw) {
            // and the minimal draw working set before the rest of the page pool
            const take = Math.min(minDraw - drawBytes, pagePool - minPool);
            pagePool -= take;
            drawBytes = Math.max(drawBytes + take, minDraw);
        }
        // The pool never allocates more slots than the scene has pages (many instances of a
        // small asset fit whole): hand the rest to the draw working set instead of leaving it
        // unused.
        const allPages = totalPages * this.pageSizeBytes;
        if (allPages > 0 && pagePool > allPages) {
            drawBytes += pagePool - allPages;
            pagePool = allPages;
        }

        // Each view's records and cut queue take a small share of the draw working set; the
        // indices, by far the largest per drawn meshlet, take the rest as one pool the director
        // shares between the views by demand.
        const weights = views + shadows * SHADOW_VIEW_DRAW_WEIGHT;
        const recordBytes = Math.max(Math.floor(drawBytes * RECORD_BUDGET_SHARE / weights), minRecordBytes);
        const queueBytes = Math.max(Math.floor(drawBytes * CUT_QUEUE_BUDGET_SHARE / weights), minQueueBytes);
        const shadowRecordBytes = Math.max(Math.floor(recordBytes * SHADOW_VIEW_DRAW_WEIGHT), minRecordBytes);
        const shadowQueueBytes = Math.max(Math.floor(queueBytes * SHADOW_VIEW_DRAW_WEIGHT), minQueueBytes);
        const records = recordBytes * views + shadowRecordBytes * shadows;
        const cutQueue = queueBytes * views + shadowQueueBytes * shadows;
        const indexBytes = Math.max(drawBytes - records - cutQueue, MIN_INDEX_BUDGET_BYTES);
        this.recordCeiling = Math.floor(recordBytes / RECORD_BYTES);
        this.cutQueueCeiling = Math.floor(queueBytes / CUT_QUEUE_ENTRY_BYTES);
        this.shadowRecordCeiling = Math.floor(shadowRecordBytes / RECORD_BYTES);
        this.shadowCutQueueCeiling = Math.floor(shadowQueueBytes / CUT_QUEUE_ENTRY_BYTES);
        this.indexBudgetTotal = Math.max(Math.floor(indexBytes / 4), 3);
        this.maxIndices = 0;
        this.budgetBreakdown = {
            budget: this.poolBytes,
            fixed,
            perView: reserved - fixed,
            pagePool,
            indices: indexBytes,
            records,
            cutQueue,
            rootPages
        };
        return pagePool;
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
        let totalPrimitiveRows = 0;
        let anyLightmaps = false;
        let totalMaterialRows = 0;
        let anyTextures = false;
        let uvChannels = 0;
        let anyTangents = false;
        let anyColors = false;
        for (const entry of pending) {
            const { resource, instances, transform } = entry;
            // the effective placement list, repeated once per root transform
            entry.list = MeshletInstanceList.from(instances ?? resource.instances);
            entry.repeat = Array.isArray(transform) ? transform.length : 1;
            totalPages += resource.manifest.pageCount;
            totalMeshlets += resource.totalMeshlets;
            totalPrimitiveRows += resource.primitives.length;
            totalInstances += entry.list.count * entry.repeat;
            anyLightmaps ||= !!entry.list.lightmaps?.some(lightmap => !!lightmap);
            if (resource.materialTable) {
                const unassigned = resource.primitives.some(p => p.materialIndex < 0) ? 1 : 0;
                totalMaterialRows += resource.materialCount + unassigned;
            } else {
                totalMaterialRows += resource.primitives.length;
            }
            if (resource.textureManifest) anyTextures = true;
            uvChannels = Math.max(uvChannels, resource.manifest.attributeLayout.uvComponents ?? 0);
            if (resource.manifest.attributeLayout.tangents) anyTangents = true;
            if (resource.manifest.attributeLayout.colors) anyColors = true;
            for (const key of Object.keys(resource.manifest.attributeLayout)) {
                // a stream this build cannot place would shift every block after it
                if (!KNOWN_LAYOUT_KEYS.has(key)) console.warn(`MeshletWorld: unknown page attribute '${key}' in a streamed asset's attributeLayout - its pages will decode wrongly; update the engine.`);
            }
        }

        // streamed-texture state (resident coarse tails + demand-streamed fine pools); the
        // flat texture index space concatenates resources like every other address space.
        // Retention: when the director determined the streamed texture set is unchanged, the
        // previous world's system - tails, fine pools, residency - is adopted wholesale.
        const texturesAdopted = !!this.adoptTextures;
        const adoptedTexResources = texturesAdopted ? this.adoptTexturesPrefix : 0;
        let texResourceIndex = 0;
        this.textures = this.adoptTextures ?? (anyTextures ? new MeshletTextures(device, this.transcode, this.fetchScheduler) : null);
        this.adoptTextures = null;
        this.adoptTexturesPrefix = 0;
        if (this.textures && !texturesAdopted) {
            this.textures.poolBytes = this.texturePoolBytes;
            this.textures.maxFineSize = this.maxFineTextureSize;
        }

        // page pool + residency map. Resident worlds: slot = rebased page index. Streamed
        // worlds: pool sized to the budget, everything starts non-resident (the residency
        // manager pins roots and streams the rest on demand).
        this.totalPages = totalPages;

        // Resolve the geometry budget before anything is allocated. The per-view working set is
        // sized from the (instance, meshlet) pair count, so count the pairs up front - the main
        // loop below recomputes them as it writes the tables.
        let budgetPairs = 0;
        // unique root pages per resource (the pages of meshlets without a parent): pinned for
        // the resource's lifetime, so the pool is sized to hold them before anything else
        const rootPagesOf = new Map();
        const rootScanned = new Set();
        for (const { resource, list, repeat } of pending) {
            let roots = rootPagesOf.get(resource);
            if (!roots) rootPagesOf.set(resource, roots = new Set());
            for (let i = 0; i < list.count; i++) {
                const prim = resource.primitives[list.primIndex[i]];
                const n = prim.meshletCount;
                budgetPairs += n * repeat;
                if (!rootScanned.has(prim)) {
                    rootScanned.add(prim);
                    if (!prim._rootPages) {
                        prim._rootPages = [];
                        for (let m = 0; m < n; m++) {
                            const row = m * MESHLET_DATA_U32S;
                            if (prim.meshletData[row + MESHLET_DATA.PARENT] === MESHLET_NO_PARENT) {
                                prim._rootPages.push(prim.meshletData[row + MESHLET_DATA.PAGE]);
                            }
                        }
                    }
                    for (const page of prim._rootPages) roots.add(page);
                }
            }
        }
        let rootPageCount = 0;
        rootPagesOf.forEach((roots) => {
            rootPageCount += roots.size;
        });
        this.pagePoolBytes = this._resolveBudget({
            totalPages,
            totalMeshlets,
            totalInstances,
            totalPrimitiveRows,
            totalMaterialRows,
            lightmapBytes: anyLightmaps ? totalInstances * 64 : 0,
            pairs: budgetPairs,
            rootPages: rootPageCount
        });

        let poolSlots = totalPages;
        if (this.streamed) {
            poolSlots = this.pagePoolBytes > 0 ?
                Math.max(Math.floor(this.pagePoolBytes / this.pageSizeBytes), 1) :
                totalPages;
            poolSlots = Math.min(poolSlots, totalPages);
        }
        const maxSlots = this.maxPoolSlots(device);
        if (poolSlots > maxSlots) {
            Debug.warnOnce(`MeshletWorld: page pool of ${poolSlots} pages (${(poolSlots * this.pageSizeBytes / 1048576).toFixed(0)} MB) exceeds the device buffer limit; clamped to ${maxSlots} pages.`);
            poolSlots = maxSlots;
        }
        this.poolSlots = poolSlots;
        // retention: adopt a compatible previous world's page pool (its resident pages stay
        // valid because the carried resources keep their page-space prefix); otherwise create
        const poolByteSize = poolSlots * this.pageSizeBytes;
        if (this.adoptPagePool && this.adoptPagePool.byteSize >= poolByteSize &&
            this.adoptPagePool.byteSize <= poolByteSize * 2) {
            // the carried pool is large enough: keep it whole, extra slots included. The resolved
            // pool drifts down a little from build to build once the budget caps it (the
            // per-view reservation tracks the scene), and shrinking to match would mean
            // starting streaming cold for a few slots' worth of budget. A pool more than twice
            // what the scene wants (most of it removed) is let go instead
            this.pagePool = this.adoptPagePool;
            poolSlots = this.adoptPagePool.byteSize / this.pageSizeBytes;
            this.poolSlots = poolSlots;
        } else if (this.adoptPagePool && this.adoptPagePool.byteSize < poolByteSize) {
            // the scene outgrew the carried pool (its slot count tracks the page count until the
            // budget caps it): allocate the larger pool and copy the old slots across on the
            // queue, ordered before any install that follows, so every resident page stays
            // resident at its slot instead of the whole world re-streaming from its roots
            this.pagePool = new StorageBuffer(device, poolByteSize, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
            const wgpu = device.wgpu;
            const encoder = wgpu.createCommandEncoder();
            encoder.copyBufferToBuffer(this.adoptPagePool.impl.buffer, 0, this.pagePool.impl.buffer, 0, this.adoptPagePool.byteSize);
            wgpu.queue.submit([encoder.finish()]);
            this.adoptPagePool.destroy();
        } else {
            // no pool to carry, or one the new pool cannot hold (a budget cut, or a pool more
            // than twice what the scene wants): its slots would not fit and their residency
            // would point past the new pool, so streaming starts cold from the roots
            this.adoptPagePool?.destroy();
            this.adoptResidency = null;
            this.adoptedPages = 0;
            this.pagePool = new StorageBuffer(device, poolByteSize, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
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

        // meshletData is not assembled on the CPU: a world of millions of meshlets would hold a
        // GB-scale copy of records its resources already carry. Each primitive's records are
        // copied, rebased and stamped only as the cut tables consume them, then uploaded into
        // place (see recordsOf below). This keeps, per global first meshlet, how to rebuild them.
        /** @type {Map<number, { prim: object, pageBase: number, bucketBits: number }>} */
        const primRecords = new Map();
        const instanceFlags = new Uint32Array(totalInstances);
        const primitiveRows = new Uint32Array(totalPrimitiveRows * OBJECT_DATA_U32S);
        const primitiveRowsF = new Float32Array(primitiveRows.buffer);
        // per instance, its primitive row - only while the cut tables are built
        const instancePrimitive = new Uint32Array(totalInstances);
        const lightmapData = anyLightmaps ? new Float32Array(totalInstances * 16) : null;
        const lightmapSlots = [];
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
        // pool, residency and visibility bits: only an unchanged prefix of the add order keeps
        // its addresses, so anything after the first changed resource is rebuilt cold.
        let pageBase = 0;
        let meshletBase = 0;
        let instanceBase = 0;
        let primitiveRow = 0;
        let materialBase = 0;
        let runningTexBase = 0;
        this._worldBoundsDirty = false;
        this.worldBounds.center.set(0, 0, 0);
        this.worldBounds.halfExtents.set(0, 0, 0);
        let boundsInit = false;

        this.instanceCount = totalInstances;
        this.instanceFlags = instanceFlags;
        this.primitiveRows = primitiveRows;
        this._placementCursor = 0;
        this.recordCapacity = 0;
        this.indexCapacity = [0, 0, 0];
        this.indexWorst = [0, 0, 0];

        // One pick id per instance, from one reserved block. An instance is one primitive at
        // one transform - which is exactly the granularity a glTF submesh has so, for example, picking a
        // table's top and its legs apart works the way it would with ordinary meshes. The ids
        // share the engine's picker space, so they never collide with mesh instance ids.
        this.pickIdBase = PickerId.reserve(totalInstances);
        this._pickRecords = null;

        for (const { resource, transform, shardBuffers, baseUrl, instances, list, repeat, fetchOptions: options } of pending) {
            // the bake stamp versions every sidecar's cache key, so a rebake over the same URLs
            // can never be served the previous bake's cached pages or texture regions
            const fetchOptions = { ...options, cacheVersion: meshletBakeStamp(resource) };
            const manifest = resource.manifest;
            const grid = manifest.positionGrid;
            const uvFloats = (manifest.attributeLayout.uvComponents ?? 0) * 2;
            const hasTangents = manifest.attributeLayout.tangents ? 1 : 0;
            const hasColors = manifest.attributeLayout.colors ? 1 : 0;
            const resourceTextureBase = runningTexBase;
            // the world keeps its own copy of each root: rows are regenerated from them later,
            // and the caller's matrices (an entity's world transform) keep changing
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
                instanceBase,
                instanceCount: list.count * repeat,
                primitiveRow,
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
                    const entry = p * PAGE_TABLE_FIELDS;
                    const blob = table[entry + PAGE_TABLE.BLOB];
                    const offset = table[entry + PAGE_TABLE.OFFSET_HI] * 0x100000000 + table[entry + PAGE_TABLE.OFFSET_LO];
                    this.pagePool.write((pageBase + p) * this.pageSizeBytes,
                        new Uint32Array(shardBuffers[blob], offset, this.pageSizeBytes / 4));
                }
            } else {
                this.streamInfo.push({ resource, pageBase, baseUrl, fetchOptions: fetchOptions ?? null });
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
                // position among the textured resources, in add order - the adopted system
                // already holds the first adoptedTexResources of them
                const texResource = texResourceIndex++;
                if (!baseUrl) {
                    Debug.warnOnce('MeshletWorld: resource has a texture manifest but no base URL (resident add) - its textures are skipped.');
                } else {
                    const textureBase = runningTexBase;
                    for (const arr of resource.textureManifest.arrays ?? []) {
                        if (arr.containerVersion === 2) runningTexBase += Array.isArray(arr.layers) ? arr.layers.length : 0;
                    }
                    // an adopted system already holds the carried prefix; resources after it
                    // join it in place (new tail layers, no re-download of what is resident)
                    if (!texturesAdopted) {
                        this.textures.addResource(resource.textureManifest, baseUrl, fetchOptions ?? null);
                    } else if (texResource >= adoptedTexResources) {
                        this.textures.appendResource(resource.textureManifest, baseUrl, fetchOptions ?? null);
                    }
                    if (baked && textureBase > 0) {
                        for (let row = 0; row < resource.materialCount; row++) {
                            const recordBase = (materialBase + row) * MATERIAL_RECORD_U32S;
                            for (let slot = 0; slot < MATERIAL_TEXTURE_SLOTS; slot++) {
                                const word = materialTable[recordBase + MATERIAL_RECORD.SLOT_WORDS + slot];
                                if ((word & 0xFFFF) === MATERIAL_SLOT_ABSENT) continue;
                                // texture indices are 16-bit, with the top value meaning absent:
                                // a world past that drops the slot rather than wrap onto another
                                // texture
                                const rebased = (word & 0xFFFF) + textureBase;
                                if (rebased >= MATERIAL_SLOT_ABSENT) {
                                    Debug.errorOnce(`MeshletWorld: more than ${MATERIAL_SLOT_ABSENT} streamed textures in one world; textures past the limit are dropped.`);
                                }
                                materialTable[recordBase + MATERIAL_RECORD.SLOT_WORDS + slot] = (word & 0xFFFF0000) | Math.min(rebased, MATERIAL_SLOT_ABSENT);
                            }
                        }
                    }
                }
            }

            // meshletData with page rebasing; per-prim bases + material rows, index budgets
            const primMatRows = [];
            const primBucketCorners = [];
            const primFlags = [];
            for (let primIndex = 0; primIndex < resource.primitives.length; primIndex++) {
                const prim = resource.primitives[primIndex];
                // what the build needs of the records besides the copy recordsOf makes, kept on
                // the primitive so a resource whose records were released can still be laid out
                if (prim._indexCount === undefined) {
                    const src = prim.meshletData;
                    let triangles = 0;
                    for (let m = 0; m < prim.meshletCount; m++) triangles += src[m * MESHLET_DATA_U32S + MESHLET_DATA.TRIANGLE_COUNT];
                    prim._indexCount = triangles * 3;
                    prim._firstFlags = prim.meshletCount ? src[MESHLET_DATA.FLAGS] : 0;
                }
                const primFirst = meshletBase;
                meshletBase += prim.meshletCount;

                if (baked) {
                    primMatRows.push(prim.materialIndex >= 0 ? prim.materialIndex : fallbackRow);
                } else {
                    primMatRows.push(primIndex);
                    writeSynthRecord(materialBase + primIndex, prim.baseColorFactor);
                }

                // This primitive's draw bucket comes from its material record, which keeps
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
                    const meshletFlags = prim._firstFlags;
                    bucket = (meshletFlags & MESHLET_FLAG_ALPHA_MASKED) ? MESHLET_BUCKET_MASKED :
                        ((meshletFlags & MESHLET_FLAG_TWO_SIDED) ? MESHLET_BUCKET_OPAQUE_TWO_SIDED :
                            MESHLET_BUCKET_OPAQUE);
                }
                const bucketBits = (bucket === MESHLET_BUCKET_MASKED ? MESHLET_FLAG_ALPHA_MASKED : 0) |
                    (bucket === MESHLET_BUCKET_OPAQUE ? 0 : MESHLET_FLAG_TWO_SIDED);

                // the bucket is stamped onto every meshlet as its records are copied (recordsOf);
                // total the worst-case index count per bucket once per primitive - the
                // per-instance loop below then just adds these, instead of re-walking every
                // meshlet for each instance
                primRecords.set(primFirst, { prim, pageBase, bucketBits });
                const primCorners = [0, 0, 0];
                primCorners[bucket] = prim._indexCount;
                primBucketCorners.push(primCorners);
                primFlags.push((hasTangents ? OBJECT_FLAG_HAS_TANGENTS : 0) | (hasColors ? OBJECT_FLAG_HAS_COLORS : 0) |
                    (bucket << OBJECT_BUCKET_SHIFT));

                // the primitive row, shared by every instance of the primitive
                const row = (primitiveRow + primIndex) * OBJECT_DATA_U32S;
                const halfExtents = prim.aabbHalfExtents;
                for (let c = 0; c < 3; c++) {
                    primitiveRowsF[row + PRIMITIVE_DATA.SPHERE + c] = prim.aabbCenter[c];
                    primitiveRowsF[row + PRIMITIVE_DATA.GRID_ORIGIN + c] = grid.origin[c];
                }
                primitiveRowsF[row + PRIMITIVE_DATA.SPHERE + 3] = Math.hypot(halfExtents[0], halfExtents[1], halfExtents[2]);
                primitiveRowsF[row + PRIMITIVE_DATA.GRID_STEP] = grid.step;
                primitiveRows[row + PRIMITIVE_DATA.FIRST_MESHLET] = primFirst;
                primitiveRows[row + PRIMITIVE_DATA.MESHLET_COUNT] = prim.meshletCount;
                primitiveRows[row + PRIMITIVE_DATA.MATERIAL] = materialBase + primMatRows[primIndex];
                primitiveRows[row + PRIMITIVE_DATA.UV_FLOATS_PER_VERTEX] = uvFloats;
                primitiveRows[row + PRIMITIVE_DATA.PICK_ID_BASE] = this.pickIdBase;
            }

            // One run of the placement list: where each instance's pair bits start within it, and
            // the run's worst-case capacities (every meshlet of every instance drawn).
            let runPairs = 0;
            const runCorners = [0, 0, 0];
            for (let i = 0; i < list.count; i++) {
                const p = list.primIndex[i];
                const meshletCount = resource.primitives[p].meshletCount;
                placement.pairOffsets[i] = runPairs;
                runPairs += meshletCount;
                for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) runCorners[b] += primBucketCorners[p][b];
            }
            placement.runPairs = runPairs;
            placement.pairCount = runPairs * repeat;
            this.totalPairs += placement.pairCount;
            this.recordCapacity += placement.pairCount;
            for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) this.indexCapacity[b] += runCorners[b] * repeat;

            // instances, run by run: one run of the list per root transform
            for (let run = 0; run < repeat; run++) {
                const root = placement.roots[run];
                for (let i = 0; i < list.count; i++) {
                    const instance = instanceBase++;
                    const p = list.primIndex[i];
                    const prim = resource.primitives[p];
                    instanceFlags[instance] = primFlags[p];
                    instancePrimitive[instance] = primitiveRow + p;
                    list.getMatrix(i, _tmpMat);
                    if (root) _tmpMat.mul2(root, _tmpMat);

                    const lightmap = list.getLightmap(i);
                    if (lightmapData && lightmap && baseUrl) {
                        const reference = lightmap.texture;
                        const arrays = resource.textureManifest?.arrays ?? [];
                        let textureIndex = resourceTextureBase;
                        let textureArray = null;
                        for (const array of arrays) {
                            if (array.containerVersion !== 2) continue;
                            if (array.id === reference.arrayId) {
                                textureArray = array;
                                break;
                            }
                            textureIndex += array.layers.length;
                        }
                        const channel = reference.texCoord ?? 0;
                        const coefficients = [lightmap.coordinateScaleBias, lightmap.lightmapScale, lightmap.lightmapAdd];
                        if (textureArray && Number.isInteger(reference.layer) && reference.layer >= 0 && reference.layer < textureArray.layers.length &&
                            Number.isInteger(channel) && channel >= 0 && channel < MESHLET_MAX_UV_CHANNELS && (prim.uvChannelMask & (1 << channel)) &&
                            coefficients.every(values => values?.length === 4 && values.every(Number.isFinite))) {
                            textureIndex += reference.layer;
                            const lightmapRow = instance * 16;
                            lightmapData.set(lightmap.coordinateScaleBias, lightmapRow);
                            lightmapData.set(lightmap.lightmapScale, lightmapRow + 4);
                            lightmapData.set(lightmap.lightmapAdd, lightmapRow + 8);
                            lightmapData.set([textureIndex, channel, 1, 0], lightmapRow + 12);
                            const scale = Math.max(Math.abs(lightmap.coordinateScaleBias[0]), Math.abs(lightmap.coordinateScaleBias[1]) * 0.5, 1e-8);
                            lightmapSlots.push(materialBase + primMatRows[p], textureIndex, Math.round(TEXEL_RATE_PER_MIP * Math.log2(scale)));
                        } else {
                            Debug.warnOnce('MeshletWorld: invalid streamed EPIC lightmap reference, coefficients or UV channel; lightmap ignored.');
                        }
                    }

                    // World bounds, from the baked AABB transformed by the instance matrix - not the
                    // bounding sphere radius on every axis, which turns a flat landscape into a cube
                    // and inflates the depth range of every shadow cascade fitted around it.
                    const halfExtents = prim.aabbHalfExtents;
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
            }

            primitiveRow += resource.primitives.length;
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

        this.cells = new MeshletCells(device, this.instanceCount, (instance, out, offset) => this._instanceSphere(instance, out, offset));
        const recordBytes = MESHLET_DATA_U32S * 4;
        this.meshletDataBuffer = new StorageBuffer(device, Math.max(totalMeshlets * recordBytes, 16), BUFFERUSAGE_COPY_DST);
        // One primitive's records at a time: a copy of the resource's (which stay untouched for
        // later rebuilds) with page indices rebased and the draw bucket stamped, handed to the
        // cut tables to complete, then uploaded to their place. Only primitives some instance
        // uses are visited; nothing reads the records of the rest.
        const meshletDataBuffer = this.meshletDataBuffer;
        // Uploads are batched: primitives are mostly visited in meshlet order, so consecutive
        // ones gather in a staging array and go up in one write. A world of many small
        // primitives would otherwise issue a write per primitive.
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
        this.objectDataBuffer = new StorageBuffer(device, Math.max((totalInstances + totalPrimitiveRows) * OBJECT_DATA_U32S * 4, 16), BUFFERUSAGE_COPY_DST);
        this._writeObjectRows(0, totalInstances);
        if (primitiveRows.length) this.objectDataBuffer.write(totalInstances * OBJECT_DATA_U32S * 4, primitiveRows);
        this.materialTableBuffer = new StorageBuffer(device, materialTable.byteLength, BUFFERUSAGE_COPY_DST);
        this.materialTableBuffer.write(0, materialTable);
        if (lightmapData) {
            this.lightmapBuffer = new StorageBuffer(device, lightmapData.byteLength, BUFFERUSAGE_COPY_DST);
            this.lightmapBuffer.write(0, lightmapData, 0, lightmapData.length);
        }

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

        // streamed textures: tails + residency exist before the materials bind them (an
        // adopted system is already finalized and possibly fully loaded)
        if (texturesAdopted) {
            this.textures.appendFinalize();
        } else {
            this.textures?.finalize();
        }
        this.textures?.setMaterialSlots(materialTable, totalMaterialRows, MATERIAL_RECORD_U32S, MATERIAL_RECORD.SLOT_WORDS);

        // binds the world's storage buffers to a bucket material (lit or debug). The per-view
        // records buffer is not bound here - each view's mesh instances carry it as an
        // instance-level parameter override
        const bindWorldParams = (material) => {
            material.setParameter('meshletData', this.meshletDataBuffer);
            material.setParameter('objectData', this.objectDataBuffer);
            material.setParameter('pagePool', this.pagePool);
            material.setParameter('residency', this.residencyBuffer);
            material.setParameter('materialTable', this.materialTableBuffer);
            if (this.lightmapBuffer) material.setParameter('meshletLightmaps', this.lightmapBuffer);
            material.setParameter('pageSizeWords', this.pageSizeBytes / 4);
        };
        const bindTextureParams = (material) => {
            if (!this.textures) return;
            material.setParameter('texResidency', this.textures.residencyBuffer);
            material.setParameter('meshletSlotDesc', this.textures.slotDescBuffer);
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
            colors: anyColors,
            lightmaps: anyLightmaps,
            forwardBackend: ShaderChunks.get(device, 'wgsl').get('litForwardBackendPS'),
            familySizes: this.textures?.families.map(fam => fam.slotSize),
            familyLevels: this.textures?.families.map(fam => ({ slotLevels: fam.slotLevels, tailLevels: fam.levels }))
        });

        // engine-lit bucket materials (one per MESHLET_BUCKET_*) shared by both phases; the
        // unlit debug materials (colour modes) are created lazily on first setColorMode(>0)
        this._litMaterials = [];
        for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
            const material = createMeshletLitMaterial(b, chunks);
            if (this._texDebug) {
                material.setDefine('MESHLET_TEX_DEBUG', true);
                material.update();
            }
            this._litMaterials.push(material);
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
            this.textures.setLightmapSlotMap(new Int32Array(lightmapSlots));
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
     * transform, so this is submesh granularity - a table baked with its top and legs as
     * separate glTF primitives picks apart exactly as it would have unbaked.
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
     * @type {Array<object>}
     */
    get litMaterials() {
        return this._litMaterials;
    }

    /**
     * Sets the debug color mode (MESHLET_COLOR_MODE): MESHLET swaps in the unlit per-meshlet
     * debug materials (applied to each view's mesh instances via
     * {@link MeshletView#syncMaterials}); TEXTURES keeps the lit ones, painting each pixel with
     * its texture-sample state.
     *
     * @param {number} mode - The color mode.
     */
    setColorMode(mode) {
        const texDebug = mode === MESHLET_COLOR_MODE.TEXTURES;
        if (mode > 0 && !texDebug && !this._debugMaterials) {
            this._debugMaterials = [];
            for (let b = 0; b < MESHLET_BUCKET_COUNT; b++) {
                this._debugMaterials.push(createMeshletMaterial(b));
            }
            this._debugMaterials.forEach(this._bindWorldParams);
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
     * Selection has to live in objectData: one indirect draw covers every instance, so there is
     * no mesh instance to add to an outline layer and no per-object uniform to set. The outline
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
     * @returns {import('./meshlet-resource.js').MeshletPrimitive} The instance's primitive.
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
