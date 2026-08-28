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
    UV_EXTENT: 30
};

/**
 * Words per instance in objectData - the per-placement record the world uploads and the cull /
 * material shaders read.
 *
 * Word layout:
 * - [0-15] f32 world matrix (column major)
 * - [16-19] f32 local bounding sphere (cx, cy, cz, r)
 * - [20] u32 first meshlet id (world-global), [21] u32 meshlet count
 * - [22] u32 material row, [23] u32 flags (OBJECT_FLAG_*), [24] f32 max scale of the matrix
 * - [25] u32 first pair bit (base of the instance's claim / visibility bit range)
 * - [26-28] f32 position grid origin, [29] f32 grid step (per-resource quantisation)
 * - [30] u32 UV floats per vertex (per-resource page attribute layout)
 * - [31] u32 pick id (the placement's, shared by its instances)
 *
 * @type {number}
 */
export const OBJECT_DATA_U32S = 32;

/**
 * Word offsets into an objectData record.
 *
 * @ignore
 */
export const OBJECT_DATA = {
    MATRIX: 0,
    SPHERE: 16,
    FIRST_MESHLET: 20,
    MESHLET_COUNT: 21,
    MATERIAL: 22,
    FLAGS: 23,
    MAX_SCALE: 24,
    FIRST_PAIR_BIT: 25,
    GRID_ORIGIN: 26,
    GRID_STEP: 29,
    UV_FLOATS_PER_VERTEX: 30,
    PICK_ID: 31
};

/** u32 per cull work item: [instance, slice start]. @type {number} */
export const WORK_ITEM_U32S = 2;

/**
 * Rows (vec4 indices) of the per-view cull parameter buffer the culler fills each frame and
 * the cull shaders read. PLANES are the six world frustum planes (xyz normal, w distance);
 * CAMERA = (position.xyz, projScale); LOD = (dagPixelThreshold, hzbWidth, hzbHeight,
 * hzbMipCount); VIEW_PROJ = four matrix columns; STREAMING = (texelRateMarkBase, orthoScale,
 * cullFlags, -); VIEW_DIR = (direction.xyz, -). Scalars that the shader reads as integers are
 * float-encoded and converted with u32().
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
    VIEW_DIR: 13
};

/** vec4 rows in the cull parameter buffer. @type {number} */
export const CULL_PARAMS_VEC4S = 16;

/** cullFlags bit 0: suppress the texel-rate feedback marks (orthographic / shadow views). @type {number} */
export const CULL_FLAG_NO_TEXEL_RATE = 1 << 0;

/** Threads per workgroup of the instance cull (one thread per instance). @type {number} */
export const MESHLET_INSTANCE_CULL_WORKGROUP = 64;

/** Threads per workgroup of the index-write pass (one workgroup per record). @type {number} */
export const MESHLET_INDEX_WRITE_WORKGROUP = 64;

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
 * Counter buffer words: [0] workItemCount, [1] recordCount, then three MESHLET_BUCKET_COUNT
 * blocks - index cursors, committed ends, unclamped index demand. Padded to a multiple of 4.
 *
 * @type {number}
 */
export const MESHLET_COUNTER_U32S = 12;

/**
 * Word offsets into the counter buffer; the three per-bucket blocks are indexed as
 * `BASE + bucket`.
 *
 * @ignore
 */
export const MESHLET_COUNTER = {
    WORK_ITEMS: 0,
    RECORDS: 1,
    CURSOR_BASE: 2,     // index cursor per bucket (reservation)
    COMMITTED_BASE: 5,  // highest reservation that fit per bucket - the draw's index count
    DEMAND_BASE: 8      // unclamped demand per bucket - survives the phase-2 reset
};

/** @type {number} - material record word 11, bit 0. */
export const MATERIAL_FLAG_DOUBLE_SIDED = 1 << 0;

/** @type {number} - material record word 11, bit 1. */
export const MATERIAL_FLAG_ALPHA_MASK = 1 << 1;

/** @type {number} */
export const MESHLET_NO_PARENT = 0xFFFFFFFF;

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

/** @type {number} - objectData flags word (23), bit 0: instance hidden. */
export const OBJECT_FLAG_HIDDEN = 1 << 0;

/** @type {number} - objectData flags word (23), bit 1: pages carry tangents. */
export const OBJECT_FLAG_HAS_TANGENTS = 1 << 1;

/**
 * objectData flags word (23), bit 8: draw this instance in the outline pass. Selection is a
 * per-instance property because one indirect draw covers the whole world - there is no mesh
 * instance to add to an outline layer, and no per-object uniform to set.
 *
 * @type {number}
 */
export const OBJECT_FLAG_OUTLINED = 1 << 8;

/**
 * objectData flags word (23), bit 9: draw this instance in the outline pass in the HOVER colour.
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
 * - [11] u32 flags: bit0 doubleSided, bit1 alpha MASK, bits 8-11 slot-present (bit 8+s).
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
 * UV channels a material slot can address: the slot word's texCoord field is 2 bits, so
 * channels 0-3. Pages may carry more (uvFloatsPerVertex is per resource); the shaders thread
 * through at most this many. Per-instance lightmaps would need per-instance texture state
 * (objectData is full) and are a separate part.
 *
 * @type {number}
 */
export const MESHLET_MAX_UV_CHANNELS = 4;

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
 * family:8 | sizeBias:8, word 1 = minLod:16 (source-mip space, 0x7FFF = nothing resident) |
 * tailLayer:16. Written by MeshletTextures, read by the lit chunks.
 *
 * @type {number}
 */
export const TEX_RESIDENCY_U32S = 2;

/**
 * Debug colour modes of the meshlet world (see MeshletWorld#setColorMode): the engine-lit
 * material, or the flat debug material tinted per LOD tier or per meshlet.
 *
 * @ignore
 */
export const MESHLET_COLOR_MODE = {
    LIT: 0,
    LOD_TIER: 1,
    MESHLET: 2
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
 * Meshlets per cull work item. One workgroup of the meshlet-cull compute shader processes one
 * slice of this many consecutive meshlets of one instance; the instance cull emits
 * ceil(meshletCount / MESHLET_CULL_SLICE) work items per surviving instance.
 *
 * @ignore
 */
export const MESHLET_CULL_SLICE = 64;

/**
 * Width of the two-dimensional indirect dispatch grid. Work-item counts above it wrap into the
 * second dimension so a dispatch never exceeds the per-dimension workgroup limit.
 *
 * @ignore
 */
export const MESHLET_DISPATCH_WIDTH = 4096;
