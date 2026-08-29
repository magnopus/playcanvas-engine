import {
    MATERIAL_TEXTURE_SLOTS, PAGE_FLAG_WIDE_POSITIONS, PAGE_HEADER, PAGE_HEADER_BYTES
} from '../constants.js';

/**
 * WGSL shared by every shader that reads meshlet data: the typed views of the world's
 * buffers, and the page layout + vertex decode. Declared once here so the cull, the debug
 * material and the engine-lit chunk set cannot drift apart.
 *
 * The buffers are flat u32 arrays on the CPU side (laid out per the OBJECT_DATA / MESHLET_DATA
 * / MATERIAL_RECORD enums in constants.js) and typed structs on the GPU side. The struct
 * field tables below are the single source of both the WGSL text and the layout test that
 * checks the struct offsets against the enums (std alignment: scalars 4, vec2 8, vec3/vec4/mat
 * 16, arrays at their element alignment) - fields that do not sit on a 16-byte boundary are
 * declared as `array<f32, 3>` rather than `vec3f` so nothing pads.
 *
 * @ignore
 */

/** @typedef {Array<[string, string, string?]>} StructFields - [name, wgsl type, comment] */

/** @type {StructFields} */
export const MESHLET_OBJECT_STRUCT = [
    ['worldMatrix', 'mat4x4f', 'column major'],
    ['sphere', 'vec4f', 'local bounding sphere: center, radius'],
    ['firstMeshlet', 'u32', 'world-global id of the primitive\'s first meshlet'],
    ['meshletCount', 'u32'],
    ['material', 'u32', 'material record row'],
    ['flags', 'u32', 'OBJECT_FLAG_*'],
    ['maxScale', 'f32', 'largest axis scale of worldMatrix - scales radii and errors'],
    ['firstPairBit', 'u32', 'base of this instance\'s (instance, meshlet) bit range'],
    ['gridOrigin', 'array<f32, 3>', 'position grid of the instance\'s resource'],
    ['gridStep', 'f32'],
    ['uvFloatsPerVertex', 'u32', 'page attribute layout of the instance\'s resource'],
    ['pickId', 'u32', 'the placement\'s picker id']
];

/** @type {StructFields} */
export const MESHLET_DATA_STRUCT = [
    ['verticesOffset', 'u32', 'page-local, into the meshlet-vertex table'],
    ['triangleCount', 'u32'],
    ['vertexCount', 'u32'],
    ['triangleOffset', 'u32', 'page-local byte offset into the corner stream'],
    ['sphere', 'vec4f', 'bounding sphere: center, radius'],
    ['coneApex', 'vec3f'],
    ['coneAxis', 'array<f32, 3>'],
    ['coneCutoff', 'f32', '>= 1 disables the cone test'],
    ['uvChannelMask', 'u32'],
    ['parentSphere', 'vec4f'],
    ['clusterError', 'f32'],
    ['parentError', 'f32', '~1e30 at DAG roots'],
    ['lodLevel', 'u32'],
    ['page', 'u32', 'world-global page index'],
    ['groupSphere', 'vec4f', 'shared sibling bounds'],
    ['parent', 'u32', 'primitive-local parent meshlet, MESHLET_NO_PARENT at roots'],
    ['flags', 'u32', 'MESHLET_FLAG_*'],
    ['uvExtent', 'u32', 'f16x2 UV span of the first packed channel; 0 = no UVs'],
    ['reserved', 'u32']
];

/** @type {StructFields} */
export const MESHLET_MATERIAL_STRUCT = [
    ['baseColor', 'vec4f'],
    ['emissive', 'vec3f'],
    ['emissiveStrength', 'f32'],
    ['metallic', 'f32'],
    ['roughness', 'f32'],
    ['alphaCutoff', 'f32'],
    ['flags', 'u32', 'MATERIAL_FLAG_* | slot-present bits 8+s'],
    ['slotWords', `array<u32, ${MATERIAL_TEXTURE_SLOTS}>`, 'texIndex:16 | arrayId:8 | texCoord:2 per slot; 0xFFFF = absent'],
    ['slotTransforms', `array<vec2u, ${MATERIAL_TEXTURE_SLOTS}>`, 'KHR_texture_transform per slot: packF16x2(offset), packF16x2(scale)'],
    ['reserved', 'array<u32, 8>']
];

/** @type {StructFields} */
export const MESHLET_RECORD_STRUCT = [
    ['instance', 'u32'],
    ['meshlet', 'u32', 'world-global meshlet id'],
    ['baseIndexOffset', 'u32', 'first draw index of the meshlet\'s triangles'],
    ['bucket', 'u32', 'MESHLET_BUCKET_*']
];

/** @type {StructFields} */
export const MESHLET_WORK_ITEM_STRUCT = [
    ['instance', 'u32'],
    ['sliceStart', 'u32', 'first meshlet of the slice, primitive-local']
];

/** @type {StructFields} */
export const MESHLET_TEX_RESIDENCY_STRUCT = [
    ['slotFamilyBias', 'u32', 'fineSlotLayer:16 (0xFFFF = tail only) | family:8 | sizeBias:8'],
    ['minLodTailLayer', 'u32', 'minLod:16 (source mips, 0x7FFF = nothing resident) | tailLayer:16']
];

/**
 * Renders a struct declaration from a field table.
 *
 * @param {string} name - Struct name.
 * @param {StructFields} fields - The field table.
 * @returns {string} WGSL.
 */
const structWGSL = (name, fields) => `
    struct ${name} {
${fields.map(([field, type, comment]) => `        ${field} : ${type},${comment ? `  // ${comment}` : ''}`).join('\n')}
    };`;

/**
 * Every struct a typed buffer view names, plus the accessors that only need a struct value.
 * The engine's WGSL processor inserts each stage's storage declarations - the vertex AND the
 * fragment stage's, merged - at that stage's first uniform or resource line, so a shader must
 * declare these structs before its first uniform: for the lit chunks that is the
 * litEngineDeclaration{VS,PS} slots, for a ShaderMaterial the top of both stages, for a
 * compute shader the top of the source.
 */
export const meshletStructsWGSL = /* wgsl */ `
    ${structWGSL('MeshletObjectData', MESHLET_OBJECT_STRUCT)}
    ${structWGSL('MeshletData', MESHLET_DATA_STRUCT)}
    ${structWGSL('MeshletMaterial', MESHLET_MATERIAL_STRUCT)}
    ${structWGSL('MeshletRecord', MESHLET_RECORD_STRUCT)}
    ${structWGSL('MeshletWorkItem', MESHLET_WORK_ITEM_STRUCT)}
    ${structWGSL('MeshletTexResidency', MESHLET_TEX_RESIDENCY_STRUCT)}

    fn meshletTexSlotLayer(r : MeshletTexResidency) -> u32 { return r.slotFamilyBias & 0xFFFFu; }
    fn meshletTexFamily(r : MeshletTexResidency) -> u32 { return (r.slotFamilyBias >> 16u) & 0xFFu; }
    fn meshletTexSizeBias(r : MeshletTexResidency) -> u32 { return (r.slotFamilyBias >> 24u) & 0xFFu; }
    fn meshletTexMinLod(r : MeshletTexResidency) -> u32 { return r.minLodTailLayer & 0xFFFFu; }
    fn meshletTexTailLayer(r : MeshletTexResidency) -> u32 { return r.minLodTailLayer >> 16u; }
`;

/** `objectData : array<MeshletObjectData>` (read). Requires meshletStructsWGSL. */
export const meshletObjectDataWGSL = /* wgsl */ `
    var<storage, read> objectData : array<MeshletObjectData>;

    fn meshletObjectGridOrigin(instance : u32) -> vec3f {
        let origin = objectData[instance].gridOrigin;
        return vec3f(origin[0], origin[1], origin[2]);
    }
`;

/** `meshletData : array<MeshletData>` (read). Requires meshletStructsWGSL. */
export const meshletDataWGSL = /* wgsl */ `
    var<storage, read> meshletData : array<MeshletData>;

    fn meshletConeAxis(meshlet : u32) -> vec3f {
        let axis = meshletData[meshlet].coneAxis;
        return vec3f(axis[0], axis[1], axis[2]);
    }
`;

/** `materialTable : array<MeshletMaterial>` (read). Requires meshletStructsWGSL. */
export const meshletMaterialTableWGSL = /* wgsl */ `
    var<storage, read> materialTable : array<MeshletMaterial>;
`;

/**
 * `records : array<MeshletRecord>` - one per drawn meshlet, written by the cull. The access
 * mode is a parameter because the cull writes them and the materials only read them.
 *
 * @param {string} access - 'read' or 'read_write'.
 * @returns {string} WGSL.
 */
export const meshletRecordsWGSL = access => /* wgsl */ `
    var<storage, ${access}> records : array<MeshletRecord>;
`;

/**
 * `workItems : array<MeshletWorkItem>` - one per slice of MESHLET_CULL_SLICE meshlets of a
 * surviving instance.
 *
 * @param {string} access - 'read' or 'read_write'.
 * @returns {string} WGSL.
 */
export const meshletWorkItemsWGSL = access => /* wgsl */ `
    var<storage, ${access}> workItems : array<MeshletWorkItem>;
`;

/** `texResidency : array<MeshletTexResidency>` (read). Requires meshletStructsWGSL. */
export const meshletTexResidencyWGSL = /* wgsl */ `
    var<storage, read> texResidency : array<MeshletTexResidency>;
`;

/**
 * `meshletDecodeDrawIndex(index)`: splits a drawn index into (record, local vertex). The
 * index-write pass encodes `record << 8 | localVertex`, so a meshlet holds at most 256 vertices.
 */
export const meshletDecodeDrawIndexWGSL = /* wgsl */ `
    fn meshletDecodeDrawIndex(index : u32) -> vec2u {
        return vec2u(index >> 8u, index & 0xFFu);
    }
`;

/**
 * `meshletDecodeOct16(word)`: octahedral normal, snorm16 x | snorm16 y << 16.
 */
export const meshletDecodeOct16WGSL = /* wgsl */ `
    fn meshletDecodeOct16(word : u32) -> vec3f {
        let sx = f32(i32(word << 16u) >> 16u) / 32767.0;
        let sy = f32(i32(word & 0xFFFF0000u) >> 16u) / 32767.0;
        var n = vec3f(sx, sy, 1.0 - abs(sx) - abs(sy));
        if (n.z < 0.0) {
            let px = (1.0 - abs(n.y)) * select(-1.0, 1.0, n.x >= 0.0);
            let py = (1.0 - abs(n.x)) * select(-1.0, 1.0, n.y >= 0.0);
            n = vec3f(px, py, n.z);
        }
        return normalize(n);
    }
`;

/**
 * `meshletDecodeTangent(word)`: oct16_sign tangent - snorm15 x | snorm15 y << 15 | bitangent
 * sign << 30. Returns xyz = tangent, w = bitangent sign.
 */
export const meshletDecodeTangentWGSL = /* wgsl */ `
    fn meshletDecodeTangent(word : u32) -> vec4f {
        let sx = f32((i32(word << 17u)) >> 17u) / 16383.0;
        let sy = f32((i32(word << 2u)) >> 17u) / 16383.0;
        var n = vec3f(sx, sy, 1.0 - abs(sx) - abs(sy));
        if (n.z < 0.0) {
            let px = (1.0 - abs(n.y)) * select(-1.0, 1.0, n.x >= 0.0);
            let py = (1.0 - abs(n.x)) * select(-1.0, 1.0, n.y >= 0.0);
            n = vec3f(px, py, n.z);
        }
        return vec4f(normalize(n), select(1.0, -1.0, (word & 0x40000000u) != 0u));
    }
`;

/**
 * The SoA layout of one resident page and the vertex decode over it. Requires
 * `pagePool : array<u32>`. A page is: header (PAGE_HEADER_BYTES) | positions (u16x4 anchor-
 * relative, or i32x3 when PAGE_FLAG_WIDE_POSITIONS) | oct16 normals | oct16_sign tangents when
 * the resource has them | UV floats | the meshlet-vertex remap table | the u8 triangle corner
 * stream. The attribute layout is per resource, so the two per-instance parameters come from
 * objectData.
 */
export const meshletPageLayoutWGSL = /* wgsl */ `
    struct MeshletPageLayout {
        positionBase : u32,
        normalBase : u32,
        tangentBase : u32,        // equals uvBase when the page carries no tangents
        uvBase : u32,
        meshletVertexBase : u32,  // the per-meshlet vertex remap table
        meshletVertexCount : u32,
        anchor : vec3i,           // grid anchor the compact u16 positions are relative to
        widePositions : bool
    };

    fn meshletPageLayout(pageBase : u32, hasTangents : bool, uvFloatsPerVertex : u32) -> MeshletPageLayout {
        var pageLayout : MeshletPageLayout;
        let vertexCount = pagePool[pageBase + ${PAGE_HEADER.VERTEX_COUNT}u];
        let flags = pagePool[pageBase + ${PAGE_HEADER.FLAGS}u];
        pageLayout.widePositions = (flags & ${PAGE_FLAG_WIDE_POSITIONS}u) != 0u;
        pageLayout.anchor = vec3i(i32(pagePool[pageBase + ${PAGE_HEADER.ANCHOR_X}u]), i32(pagePool[pageBase + ${PAGE_HEADER.ANCHOR_Y}u]), i32(pagePool[pageBase + ${PAGE_HEADER.ANCHOR_Z}u]));
        pageLayout.positionBase = pageBase + ${PAGE_HEADER_BYTES / 4}u;
        pageLayout.normalBase = pageLayout.positionBase + select(vertexCount * 2u, vertexCount * 3u, pageLayout.widePositions);
        pageLayout.tangentBase = pageLayout.normalBase + vertexCount;
        pageLayout.uvBase = pageLayout.tangentBase + select(0u, vertexCount, hasTangents);
        pageLayout.meshletVertexBase = pageLayout.uvBase + vertexCount * uvFloatsPerVertex;
        pageLayout.meshletVertexCount = pagePool[pageBase + ${PAGE_HEADER.MV_COUNT}u];
        return pageLayout;
    }

    // dequantised local-space position of one page vertex on the instance's grid
    fn meshletPagePosition(pageLayout : MeshletPageLayout, vertex : u32, gridOrigin : vec3f, gridStep : f32) -> vec3f {
        var coord : vec3i;
        if (pageLayout.widePositions) {
            let b = pageLayout.positionBase + vertex * 3u;
            coord = vec3i(i32(pagePool[b]), i32(pagePool[b + 1u]), i32(pagePool[b + 2u]));
        } else {
            let b = pageLayout.positionBase + vertex * 2u;
            let word0 = pagePool[b];
            let word1 = pagePool[b + 1u];
            coord = pageLayout.anchor + vec3i(i32(word0 & 0xFFFFu), i32(word0 >> 16u), i32(word1 & 0xFFFFu));
        }
        return vec3f(coord) * gridStep + gridOrigin;
    }
`;
