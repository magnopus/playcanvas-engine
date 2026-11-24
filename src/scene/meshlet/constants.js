/**
 * Shared constants for the GPU-driven meshlet subsystem. Data layouts here are the runtime
 * contract with the MAG_meshlets_gpu / MAG_meshlets_stream asset format (format version 2) produced by
 * gltf-tools — see its docs/MAG_meshlets_gpu.md and docs/MAG_meshlets_stream.md.
 *
 * @ignore
 */

/**
 * Words per meshlet in the resident meshletData records (the exact GPU-resident layout, upload
 * is a memcpy). Mixed u32/f32; f32 fields are IEEE-754 bit patterns in the u32 stream.
 *
 * Word layout:
 * - [0] u32 meshletVerticesOffset (page-local, in meshletVertices entries)
 * - [1] u32 triangleCount
 * - [2] u32 vertexCount
 * - [3] u32 triangleOffset (page-local, logical u8 corner index)
 * - [4-7] f32 bounding sphere (cx, cy, cz, r)
 * - [8-10] f32 cone apex, [11-13] f32 cone axis, [14] f32 cone cutoff
 * - [15] u32 uvChannelMask
 * - [16-19] f32 parent bounding sphere
 * - [20] f32 clusterError, [21] f32 parentError (~1e30 at DAG roots)
 * - [22] u32 lodLevel
 * - [23] u32 pageIndex (asset-global; rebased when assets merge into a world)
 * - [24-27] f32 sharedSiblingsBounds (cx, cy, cz, r)
 * - [28] u32 parentMeshletIndex (primitive-local; MESHLET_NO_PARENT at DAG roots)
 * - [29] u32 flags (MESHLET_FLAG_*)
 * - [30] uvExtent: f16x2 (uSpan | vSpan << 16) UV span of the meshlet's first packed UV
 *   channel - the texel-density source for texture-mip feedback. 0 = no UVs.
 * - [31] reserved
 *
 * @type {number}
 */
export const MESHLET_DATA_U32S = 32;

/**
 * Word offsets into a meshletData record (see {@link MESHLET_DATA_U32S} for the full layout).
 * Interpolated into the WGSL so the shaders and the JS never disagree about a field.
 *
 * @ignore
 */
export const MESHLET_DATA = {
    VERTICES_OFFSET: 0,
    TRIANGLE_COUNT: 1,
    VERTEX_COUNT: 2,
    TRIANGLE_OFFSET: 3,
    SPHERE: 4,          // 4 f32: cx, cy, cz, r
    CONE_APEX: 8,       // 3 f32
    CONE_AXIS: 11,      // 3 f32
    CONE_CUTOFF: 14,
    UV_CHANNEL_MASK: 15,
    PARENT_SPHERE: 16,  // 4 f32
    CLUSTER_ERROR: 20,
    PARENT_ERROR: 21,
    LOD_LEVEL: 22,
    PAGE: 23,
    GROUP_SPHERE: 24,   // 4 f32: sharedSiblingsBounds
    PARENT: 28,
    FLAGS: 29,
    UV_EXTENT: 30,
    BIRTH_GROUP: 31     // runtime only: local replacement representative, stamped over reserved
};

/**
 * Words per row of the objectData buffer. The buffer holds one row per instance, then one row
 * per primitive of every added resource. Instances of a primitive share its row, so an
 * instance row carries only what differs between instances: a scattered forest pays 64 bytes
 * per tree primitive instead of repeating the primitive's bounds, meshlet range, material and
 * page layout in every copy.
 *
 * The primitive rows share the buffer instead of having their own binding because several
 * cull passes are already at the device's storage-buffer limit per stage.
 *
 * Instance row (OBJECT_DATA):
 * - [0-11] f32 the first three rows of the affine world matrix (row major)
 * - [12] u32 objectData row of the instance's primitive (see PRIMITIVE_DATA)
 * - [13] u32 flags (OBJECT_FLAG_*), [14] f32 max scale of the matrix
 * - [15] u32 first pair bit (base of the instance's claim / visibility bit range)
 *
 * @type {number}
 */
export const OBJECT_DATA_U32S = 16;

/**
 * Word offsets into an objectData instance row.
 *
 * @ignore
 */
export const OBJECT_DATA = {
    MATRIX_ROWS: 0,
    PRIMITIVE: 12,
    FLAGS: 13,
    MAX_SCALE: 14,
    FIRST_PAIR_BIT: 15
};

/**
 * Word offsets into an objectData primitive row (OBJECT_DATA_U32S words, after the instance
 * rows):
 * - [0-3] f32 local bounding sphere (cx, cy, cz, r)
 * - [4-6] f32 position grid origin, [7] f32 grid step (per-resource quantisation)
 * - [8] u32 first meshlet id (world-global), [9] u32 meshlet count, [10] u32 material row
 * - [11] u32 UV floats per vertex (per-resource page attribute layout)
 * - [12] u32 the world's first pick id (an instance picks as this plus its index)
 * - [13-15] reserved
 *
 * @ignore
 */
export const PRIMITIVE_DATA = {
    SPHERE: 0,
    GRID_ORIGIN: 4,
    GRID_STEP: 7,
    FIRST_MESHLET: 8,
    MESHLET_COUNT: 9,
    MATERIAL: 10,
    UV_FLOATS_PER_VERTEX: 11,
    PICK_ID_BASE: 12
};

/**
 * u32 per instance in a view's root list: root classification appends each candidate to the
 * first half, and the band sort copies it into the second.
 *
 * @type {number}
 */
export const ROOT_LIST_U32S = 2;

/**
 * Rows (vec4 indices) of the per-view cull parameter buffer the culler fills each frame and
 * the cull shaders read. PLANES are the six world frustum planes (xyz normal, w distance);
 * CAMERA = (position.xyz, projScale); LOD = (dagPixelThreshold, hzbWidth, hzbHeight,
 * hzbMipCount); VIEW_PROJ = four matrix columns; VIEW_DIR = (direction.xyz, camera LOD
 * threshold or 0); CASTER_PLANES = up to MAX_CASTER_PLANES more planes, tested after the
 * frustum's (a shadow view's caster volume, see MeshletShadowRenderer).
 *
 * STREAMING is a grab bag of per-view scalars with no better home, not only streaming ones:
 * x = word offset of the texel-rate marks in the requests buffer (the world's page count),
 * y = orthoScale (shadow-map texels per world unit; 0 for a perspective view), z = CULL_FLAG_*
 * bits, w = how many CASTER_PLANES rows are in use.
 *
 * Scalars that the shader reads as integers are float-encoded and converted with u32().
 *
 * @ignore
 */
export const CULL_PARAMS = {
    PLANES: 0,
    PLANE_COUNT: 6,
    CAMERA: 6,
    LOD: 7,
    VIEW_PROJ: 8,
    STREAMING: 12,
    VIEW_DIR: 13,
    // root admission priority: origin xyz, projection scale (px per unit at distance 1)
    ROOT_PRIORITY: 14,
    // root admission bands: largest projected size (px), bands per octave, band count, and the
    // projected size below which an instance is not admitted (0 = never)
    ROOT_BANDS: 15,
    CASTER_PLANES: 16,
    MAX_CASTER_PLANES: 12,
    // per draw bucket (MESHLET_BUCKET_*), the LOD threshold in xyz - LOD.x scaled by the bucket's
    // scale - and their smallest in w (for tests over mixed buckets)
    BUCKET_LOD: 28,
    // per draw bucket, the root admission size cut (px) in xyz; w = the fraction of it an
    // instance admitted last frame may shrink to before it is dropped (hysteresis)
    BUCKET_ROOT: 29,
    // last frame's view-projection (4 rows), then its camera position (xyz; w = 1 when last
    // frame's HZB is valid to test against, 0 when not) and view direction: root admission tests
    // instances against the pyramid last frame built, so it needs the camera that built it
    PREV_VIEW_PROJ: 30,
    PREV_CAMERA: 34,
    PREV_VIEW_DIR: 35
};

/** vec4 rows in the cull parameter buffer. @type {number} */
export const CULL_PARAMS_VEC4S = 36;

/**
 * Priority bands of GPU root admission. Instances are ranked by projected size - the quantity
 * the LOD test uses - into this many log-spaced bands, and admitted band by band, largest on
 * screen first, so a budget that cannot take every visible root keeps the ones that matter.
 *
 * @ignore
 */
export const MESHLET_ROOT_BANDS = 16;

/**
 * Projected size (pixels) of the first root admission band; bands step down by octaves from it.
 *
 * @ignore
 */
export const MESHLET_ROOT_BAND_MAX_PIXELS = 4096;

/**
 * Octaves the root admission bands span below MESHLET_ROOT_BAND_MAX_PIXELS (to 0.5 px).
 *
 * @ignore
 */
export const MESHLET_ROOT_BAND_OCTAVES = 13;

/** Cull flag: suppress the texel-rate feedback marks (orthographic / shadow views). @type {number} */
export const CULL_FLAG_NO_TEXEL_RATE = 1 << 0;

/**
 * Cull flag: the view is a shadow face. Instances flagged OBJECT_FLAG_NO_SHADOW are skipped.
 *
 * @type {number}
 */
export const CULL_FLAG_SHADOW_VIEW = 1 << 2;

/**
 * Instances per spatial cell (see MeshletCells). The per-instance passes - root classify and
 * instance cull - run one workgroup per visible cell, one lane per instance, so this is also
 * their workgroup size.
 *
 * @type {number}
 */
export const MESHLET_CELL_SIZE = 64;

/** Threads per workgroup of the index-write pass. @type {number} */
export const MESHLET_INDEX_WRITE_WORKGROUP = 64;

/**
 * Records (meshlets) expanded per index-write workgroup, each by WORKGROUP / RECORDS threads.
 * One record per workgroup leaves the pass bound by each group's chain of dependent loads
 * (record, meshlet, object, page) rather than by the writes; several records per group keep
 * several chains in flight and cut the group count.
 *
 * @type {number}
 */
export const MESHLET_INDEX_WRITE_RECORDS = 8;

/** u32 per draw record: [instance, meshlet, base index offset, bucket]. @type {number} */
export const RECORD_U32S = 4;

/** u32 per DrawIndexedIndirect argument block, as WebGPU defines it. @type {number} */
export const INDIRECT_DRAW_U32S = 5;

/** u32 per DispatchWorkgroupsIndirect argument block, as WebGPU defines it. @type {number} */
export const INDIRECT_DISPATCH_U32S = 3;

/**
 * Alpha-tested material: the fragment shader must run the per-record cutoff discard. Together
 * with {@link MESHLET_FLAG_TWO_SIDED} this selects the draw bucket - the two are independent,
 * because only alpha masking needs the discard and only two-sidedness needs CULLFACE_NONE.
 *
 * The world re-derives both from the material record at load ({@link MATERIAL_FLAG_ALPHA_MASK} /
 * {@link MATERIAL_FLAG_DOUBLE_SIDED}) and restamps them here, so the material record is the
 * single authority for the bucket.
 *
 * @type {number}
 */
export const MESHLET_FLAG_ALPHA_MASKED = 1 << 0;

/** @type {number} */
export const MESHLET_FLAG_DAG_ROOT = 1 << 1;

/** @type {number} - render without backface culling. */
export const MESHLET_FLAG_TWO_SIDED = 1 << 2;

/**
 * The cluster's replacement group was simplified by vertex clustering (meshopt
 * simplifySloppy), which changes topology and reads as garbled up close. It and every coarser
 * cluster derived from it are kept out of the playable floor (see
 * {@link MeshletCutData#floorPages}). Only meaningful on bakes whose stream manifest sets
 * `sloppyFlags`; older bakes never set it.
 *
 * @type {number}
 */
export const MESHLET_FLAG_SLOPPY = 1 << 3;

/**
 * Draw buckets, in index-buffer order. One indirect draw per bucket per phase - they differ in
 * pipeline state, which cannot vary within a draw:
 * - 0 OPAQUE: backface culled, no alpha test (early-Z intact)
 * - 1 OPAQUE_TWO_SIDED: no backface culling, still no alpha test
 * - 2 MASKED: no backface culling, per-record alpha-test discard
 *
 * @type {number}
 */
export const MESHLET_BUCKET_COUNT = 3;

/** @type {number} */
export const MESHLET_BUCKET_OPAQUE = 0;

/** @type {number} */
export const MESHLET_BUCKET_OPAQUE_TWO_SIDED = 1;

/** @type {number} */
export const MESHLET_BUCKET_MASKED = 2;

/**
 * Counter buffer words: [0] unused, [1] recordCount, then three MESHLET_BUCKET_COUNT
 * blocks - index cursors, committed ends, unclamped index demand - then the frame's rendered
 * meshlet count across both phases and peak per-phase record demand, the root admission
 * counters, the visible cell count, the cut queue's cursor and current stage range, the
 * instances root admission skipped as occluded, then the cut budget (CUT_BUDGET), padded to a
 * multiple of 4 words.
 *
 * @type {number}
 */
export const MESHLET_COUNTER_U32S = 40;

/**
 * Word offsets into the counter buffer; the three per-bucket blocks are indexed as
 * `BASE + bucket`.
 *
 * @ignore
 */
export const MESHLET_COUNTER = {
    RECORDS: 1,
    CURSOR_BASE: 2,     // index cursor per bucket (reservation)
    COMMITTED_BASE: 5,  // highest reservation that fit per bucket - the draw's index count
    DEMAND_BASE: 8,     // unclamped demand per bucket - survives the phase-2 reset
    RENDERED: 11,
    RECORD_DEMAND: 12,
    SELECTED: 13,      // compact cut count, retained across both occlusion phases
    ROOT_CANDIDATES: 14,          // instances the root classify appended for admission
    ROOT_ADMITTED_BASE: 15,       // admitted root indices per bucket
    ROOT_REJECTED_BASE: 18,       // root indices per bucket that did not fit
    ROOT_REJECTED_INSTANCES: 21,  // instances whose roots did not fit
    ROOT_UNREADY_INSTANCES: 22,   // instances whose root pages were not resident
    ROOT_SUBPIXEL: 23,            // instances below the admission size threshold
    VISIBLE_CELLS: 24,            // cells the cell cull appended to the visible list
    CUT_QUEUE_END: 25,            // cut tasks appended to the queue (unclamped - the queue's demand)
    CUT_STAGE_BEGIN: 26,          // the current cut stage's range of the queue
    CUT_STAGE_END: 27,
    ROOT_OCCLUDED: 28,            // instances hidden behind last frame's depth, not admitted
    CUT_WAITING: 29,              // cut groups that would refine but wait for pages to stream in
    CUT_QUEUE_REJECTED: 30        // queue entries root admission could not reserve (queue demand past its capacity)
};

/**
 * Word offsets of the cut budget in the counter buffer: the capacity the root admission and the
 * cut stages have charged this frame. They live with the counters, which clear at the start of
 * the chain, rather than in a buffer of their own: the cut and root admission shaders are at the
 * eight storage bindings WebGPU guarantees per stage. The index words are per bucket,
 * `INDICES_BASE + bucket`; the phase-2 reset leaves them alone.
 *
 * @ignore
 */
export const CUT_BUDGET = {
    INDICES_BASE: 31,     // indices charged per bucket (MESHLET_BUCKET_COUNT words)
    RECORDS: 34,          // records charged
    REJECTED_RECORDS: 35, // records of roots and refinements that did not fit (still demand)
    QUEUE: 36             // cut queue entries admitted roots have claimed
};

/**
 * Leading counter-buffer words cleared every frame: the counters themselves. The visible-cell
 * list after them is written up to its count and needs no clear.
 *
 * @type {number}
 */
export const MESHLET_COUNTER_CLEAR_U32S = MESHLET_COUNTER_U32S;

/**
 * Word offset of the visible-cell list in the counter buffer.
 *
 * @type {number}
 */
export const MESHLET_VISIBLE_CELLS_BASE = MESHLET_COUNTER_CLEAR_U32S;

/** @type {number} - material record word 11, bit 0. */
export const MATERIAL_FLAG_DOUBLE_SIDED = 1 << 0;

/** @type {number} - material record word 11, bit 1. */
export const MATERIAL_FLAG_ALPHA_MASK = 1 << 1;

/**
 * Material record word 11, bit 2: KHR_materials_unlit. The forward shader outputs the base
 * colour (factor x texture x vertex colour) as-is and skips the lighting backend entirely; it
 * selects no draw bucket because nothing about the pipeline state differs.
 *
 * @type {number}
 */
export const MATERIAL_FLAG_UNLIT = 1 << 2;

/** @type {number} */
export const MESHLET_NO_PARENT = 0xFFFFFFFF;

/**
 * Roots per root-emit entry of the cut queue. Root admission queues an admitted instance's roots
 * in lists of this many, so the emit pass spreads a primitive with thousands of roots across
 * threads (see meshletCutEmitWGSL).
 *
 * @type {number}
 * @ignore
 */
export const MESHLET_CUT_ROOT_CHUNK = 64;

/**
 * Set in the instance word of a cut queue entry that lists roots to emit rather than a group to
 * decide (see MESHLET_CUT_ROOT_CHUNK). Instance ids stay below it.
 *
 * @type {number}
 * @ignore
 */
export const MESHLET_CUT_ROOT_FLAG = 0x80000000;

/**
 * The instance word of a cut queue entry left behind when a stage carries its task forward to a
 * later slot, so the emit pass visits each task once.
 *
 * @type {number}
 * @ignore
 */
export const MESHLET_CUT_INVALID = 0xFFFFFFFF;

/**
 * Fixed-point scale of the cull pass's texture feedback: a texel-rate mark is
 * `16 * log2(screen pixels / mesh-space UV extent)`, i.e. sixteenths of a mip level below a
 * texture's top. Integer so the marks can be atomicMax'd; 1/16 mip is finer than sampling
 * ever resolves. The shader and MeshletTextures.processMarks must agree on it.
 *
 * @type {number}
 */
export const TEXEL_RATE_PER_MIP = 16;

/**
 * Texture slots per material record: baseColor, normal, ORM, emissive - the four slot words
 * at material record [12-15] and the stride of the row -> texture maps.
 *
 * @type {number}
 */
export const MATERIAL_TEXTURE_SLOTS = 4;

/** @type {number} - {@link OBJECT_DATA}.FLAGS word, bit 0: instance hidden. */
export const OBJECT_FLAG_HIDDEN = 1 << 0;

/** @type {number} - {@link OBJECT_DATA}.FLAGS word, bit 1: pages carry tangents. */
export const OBJECT_FLAG_HAS_TANGENTS = 1 << 1;

/**
 * {@link OBJECT_DATA}.FLAGS word, bit 2: pages carry rgba8 vertex colours (COLOR_0), which multiply
 * the base colour as they do on the regular glTF path.
 *
 * @type {number}
 */
export const OBJECT_FLAG_HAS_COLORS = 1 << 2;

/**
 * {@link OBJECT_DATA}.FLAGS word, bit 3: the instance casts no shadow - shadow views skip it in the
 * cull, so casting is per instance rather than per pipeline.
 *
 * @type {number}
 */
export const OBJECT_FLAG_NO_SHADOW = 1 << 3;

/**
 * {@link OBJECT_DATA}.FLAGS word, bits 4-5: the instance's draw bucket (MESHLET_BUCKET_*), so the
 * root classification can apply the bucket's size cut without reading the cut tables.
 *
 * @type {number}
 */
export const OBJECT_BUCKET_SHIFT = 4;

/**
 * {@link OBJECT_DATA}.FLAGS word, bit 8: draw this instance in the outline pass. Selection is a
 * per-instance property because one indirect draw covers the whole world - there is no mesh
 * instance to add to an outline layer, and no per-object uniform to set.
 *
 * @type {number}
 */
export const OBJECT_FLAG_OUTLINED = 1 << 8;

/**
 * {@link OBJECT_DATA}.FLAGS word, bit 9: draw this instance in the outline pass in the hover colour.
 * Separate from {@link OBJECT_FLAG_OUTLINED} so a hovered instance reads differently from a
 * selected one, and a selected instance can be hovered without losing its colour.
 *
 * @type {number}
 */
export const OBJECT_FLAG_HOVERED = 1 << 9;

/**
 * Material record - 32 u32 words (128 B) per glTF material, baked by gltf-tools (root
 * MAG_meshlets_gpu block) and uploaded verbatim; synthesized engine-side for assets without the
 * baked table. Mixed u32/f32 via bitcast. Word offsets:
 * - [0-3] f32 baseColor RGBA
 * - [4-6] f32 emissive RGB, [7] f32 emissiveStrength
 * - [8] f32 metallic, [9] f32 roughness (stored as roughness, never gloss)
 * - [10] f32 alphaCutoff
 * - [11] u32 flags: bit0 doubleSided, bit1 alpha MASK, bit2 unlit, bits 8-11 slot-present (bit 8+s).
 *   These two are the authority for the draw bucket - unlike the meshlet flag, the bake keeps
 *   them apart (see MESHLET_FLAG_ALPHA_MASKED).
 * - [12-15] u32 slot words (s = 0 baseColor / 1 normal / 2 ORM / 3 emissive):
 *   texIndex:16 | arrayId:8 | texCoord:2 | reserved:6; texIndex 0xFFFF = absent
 * - [16-23] u32 per-slot KHR_texture_transform: packF16x2(offX, offY), packF16x2(sclX, sclY);
 *   both words 0 = identity
 * - [24-31] reserved
 *
 * @type {number}
 */
export const MATERIAL_RECORD_U32S = 32;

/**
 * Word offsets into a material record.
 *
 * @ignore
 */
export const MATERIAL_RECORD = {
    BASE_COLOR: 0,          // 4 f32 rgba
    EMISSIVE: 4,            // 3 f32 rgb
    EMISSIVE_STRENGTH: 7,
    METALLIC: 8,
    ROUGHNESS: 9,
    ALPHA_CUTOFF: 10,
    FLAGS: 11,
    SLOT_WORDS: 12,         // MATERIAL_TEXTURE_SLOTS words: texIndex:16 | arrayId:8 | texCoord:2
    SLOT_TRANSFORMS: 16     // 2 words per slot: packF16x2(offset), packF16x2(scale)
};

/**
 * UV channels threaded through meshlet shading (0-4). Per-instance lightmaps can select all
 * five channels. Baked PBR slots retain their two-bit texCoord field and address channels 0-3.
 *
 * @type {number}
 */
export const MESHLET_MAX_UV_CHANNELS = 5;

/**
 * Material texture slot indices, in slot-word order.
 *
 * @ignore
 */
export const MATERIAL_SLOT = {
    BASE_COLOR: 0,
    NORMAL: 1,
    ORM: 2,                 // r occlusion (unused), g roughness, b metalness
    EMISSIVE: 3
};

/**
 * u32 per texture in the texResidency buffer: word 0 = fineSlotLayer:16 (0xFFFF = tail only) |
 * family:2 | tailStart:6 | sizeBias:8, word 1 = minLod:16 (source-mip space, 0x7FFF = nothing
 * resident) | tailLayer:16. Written by MeshletTextures, read by the lit chunks.
 *
 * @type {number}
 */
export const TEX_RESIDENCY_U32S = 2;

/**
 * Debug colour modes of the meshlet world (see MeshletWorld#setColorMode): the engine-lit
 * material, or the flat debug material tinted per meshlet. Examples pass the values as URL
 * parameters, so they stay stable.
 *
 * @ignore
 */
export const MESHLET_COLOR_MODE = {
    LIT: 0,
    MESHLET: 2,
    /** Lit materials, painted with the state of each pixel's base-colour texture sample. */
    TEXTURES: 3
};

// ---------------------------------------------------------------------------
// Streamed page binary format
// ---------------------------------------------------------------------------

/** @type {number} */
export const PAGE_HEADER_BYTES = 48;

/** 'MPGS' little-endian. @type {number} */
export const PAGE_MAGIC = 0x4D504753;

/** @type {number} */
export const PAGE_FORMAT_VERSION = 2;

/**
 * Page size in bytes. Every resource in a world shares one page pool of page-sized slots, so
 * assets baked with any other size are rejected at parse time rather than mixed.
 *
 * @type {number}
 */
export const PAGE_SIZE_BYTES = 65536;

/** @type {number} */
export const PAGE_FLAG_QUANTIZED = 1 << 0;

/** @type {number} */
export const PAGE_FLAG_ROOT = 1 << 1;

/** Positions stored as absolute i32x3 grid coords instead of anchor-relative u16x4. @type {number} */
export const PAGE_FLAG_WIDE_POSITIONS = 1 << 2;

/** u32 fields per binary page table entry. @type {number} */
export const PAGE_TABLE_FIELDS = 8;

/** Page not resident sentinel in the residency map. @type {number} */
export const PAGE_NOT_RESIDENT = 0xFFFFFFFF;

/**
 * Per-page marks the cull shader writes into the requests buffer each frame and the residency
 * reads back: a page is either resident and used this frame (touch + protect from eviction) or
 * missing and wanted (fetch candidate) - never both, and 0 = not wanted.
 *
 * @ignore
 */
export const PAGE_REQUEST = {
    NONE: 0,
    MISSING: 1,
    USED: 2
};

/** texIndex value of a material slot word that carries no texture. @type {number} */
export const MATERIAL_SLOT_ABSENT = 0xFFFF;

/**
 * minLod value of a texResidency entry with nothing resident yet - the shader falls back to
 * the material factors until the texture's tail lands.
 *
 * @type {number}
 */
export const MESHLET_TEX_NO_MINLOD = 0x7FFF;

/**
 * Page header word offsets (u32 indices).
 *
 * @ignore
 */
export const PAGE_HEADER = {
    MAGIC: 0,
    VERSION: 1,
    VERTEX_COUNT: 2,
    TRIANGLE_INDEX_COUNT: 3,
    MESHLET_COUNT: 4,
    FLAGS: 5,
    MV_COUNT: 6,
    ANCHOR_X: 7,
    ANCHOR_Y: 8,
    ANCHOR_Z: 9
};

/**
 * Page table entry word offsets (u32 indices within one PAGE_TABLE_FIELDS-sized record).
 *
 * @ignore
 */
export const PAGE_TABLE = {
    BLOB: 0,
    OFFSET_LO: 1,
    OFFSET_HI: 2,
    BYTE_LENGTH: 3,
    VERTEX_COUNT: 4,
    TRIANGLE_INDEX_COUNT: 5,
    MESHLET_COUNT: 6,
    FLAGS: 7
};

/**
 * Lanes per workgroup of the meshlet cull, each processing one entry of the selected cut.
 *
 * @ignore
 */
export const MESHLET_CULL_SLICE = 64;

/**
 * Threads per workgroup of the passes that run one thread per entry of a GPU-built list: the cut
 * queue stages and emit, and the root scatter and admission.
 *
 * @ignore
 */
export const MESHLET_LIST_WORKGROUP = 64;

/**
 * Threads per workgroup of the cell cull, one per spatial cell.
 *
 * @ignore
 */
export const MESHLET_CELL_CULL_WORKGROUP = 64;

/**
 * Word offset of the persistent per-instance admission bits in a view's claim buffer: two
 * regions of one bit per instance, after the pair bits, the admission bits and the off-frustum
 * refinement bits, that swap roles each frame. The root admission marks this frame's
 * admissions in one and reads the previous frame's from the other (hysteresis); the per-frame
 * clear leaves the previous region intact.
 *
 * @param {number} totalPairs - Instance-meshlet pairs in the world.
 * @param {number} instanceCount - Instances in the world.
 * @returns {number} Word offset of the first region; the second follows ceil(instanceCount / 32) words later.
 * @ignore
 */
export const claimPersistBase = (totalPairs, instanceCount) => Math.max(Math.ceil((totalPairs * 2 + instanceCount) / 32), 4);

/**
 * Words in a view's claim buffer: the claim bits, then the two persistent admission bit regions.
 *
 * @param {number} totalPairs - Instance-meshlet pairs in the world.
 * @param {number} instanceCount - Instances in the world.
 * @returns {number} Word count.
 * @ignore
 */
export const claimWordCount = (totalPairs, instanceCount) => claimPersistBase(totalPairs, instanceCount) + 2 * Math.ceil(instanceCount / 32);

/**
 * Width of the two-dimensional indirect dispatch grid. Work-item counts above it wrap into the
 * second dimension so a dispatch never exceeds the per-dimension workgroup limit.
 *
 * @ignore
 */
export const MESHLET_DISPATCH_WIDTH = 4096;

/**
 * Workgroups dispatched for each pass whose work is sized on the GPU (the cut stages, root
 * admission, the cluster cull, the index write). They are dispatched directly, this many
 * workgroups at a time, each looping over a share of the work, rather than with indirect
 * arguments: browsers validate every indirect dispatch on the GPU process's main thread, and at
 * a few hundred dispatches a frame that validation alone saturates it. Large enough to fill a
 * large GPU; a workgroup with nothing to do reads one counter and exits.
 *
 * @ignore
 */
export const MESHLET_STRIDED_GROUPS = 1024;
