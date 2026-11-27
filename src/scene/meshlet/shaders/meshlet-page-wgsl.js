import {
    MATERIAL_TEXTURE_SLOTS, MESHLET_BUCKET_COUNT, MESHLET_DRAW_BUCKET_SHIFT, MESHLET_PAGE_SLOT_SHIFT, OBJECT_FLAG_HAS_COLORS, OBJECT_FLAG_HAS_TANGENTS, PAGE_FLAG_COMPACT,
    PAGE_FLAG_WIDE_POSITIONS, PAGE_HEADER, PAGE_HEADER_BYTES, PAGE_UV_BOX_WORDS
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
    ['matrixRows', 'array<vec4f, 3>', 'rows 0-2 of the affine world matrix'],
    ['primitive', 'u32', 'objectData row of the instance\'s primitive'],
    ['flags', 'u32', 'OBJECT_FLAG_*'],
    ['maxScale', 'f32', 'largest axis scale of the world matrix - scales radii and errors'],
    ['firstPairBit', 'u32', 'base of this instance\'s (instance, meshlet) bit range']
];

/**
 * A primitive row of objectData, as {@link meshletObjectDataWGSL}'s `meshletPrimitiveData`
 * returns it. The row is stored in the instance struct's words (both are OBJECT_DATA_U32S
 * long), so this table documents and tests the reinterpretation rather than a buffer view.
 *
 * @type {StructFields}
 */
export const MESHLET_PRIMITIVE_STRUCT = [
    ['sphere', 'vec4f', 'local bounding sphere: center, radius'],
    ['gridOrigin', 'vec3f', 'position grid of the primitive\'s resource'],
    ['gridStep', 'f32'],
    ['firstMeshlet', 'u32', 'world-global id of the primitive\'s first meshlet'],
    ['meshletCount', 'u32'],
    ['material', 'u32', 'material record row'],
    ['uvFloatsPerVertex', 'u32', 'page attribute layout of the primitive\'s resource'],
    ['pickIdBase', 'u32', 'the world\'s first pick id'],
    ['reserved', 'array<u32, 3>']
];

/** @type {StructFields} */
export const MESHLET_DATA_STRUCT = [
    ['verticesOffset', 'u32', 'page-local: into the meshlet-vertex table, or the first vertex on a compact page'],
    ['triangleCount', 'u32'],
    ['vertexCount', 'u32'],
    ['triangleOffset', 'u32', 'page-local byte offset into the corner stream'],
    ['sphere', 'vec4f', 'bounding sphere: center, radius'],
    ['coneApex', 'vec3f'],
    ['coneAxis', 'array<f32, 3>'],
    ['coneCutoff', 'f32', '>= 1 disables the cone test'],
    ['uvChannelMask', 'u32', 'mask:16 | compact-page UV box slot:16'],
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
    ['meshlet', 'u32', 'world-global meshlet id']
];

/** @type {StructFields} */
export const MESHLET_TEX_RESIDENCY_STRUCT = [
    ['slotFamilyBias', 'u32', 'fineSlotLayer:16 (0xFFFF = tail only) | family:2 | tailStart:6 | sizeBias:8'],
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
 * The engine's WGSL processor inserts each stage's storage declarations - the vertex and the
 * fragment stage's, merged - at that stage's first uniform or resource line, so a shader must
 * declare these structs before its first uniform: for the lit chunks that is the
 * litEngineDeclaration{VS,PS} slots, for a ShaderMaterial the top of both stages, for a
 * compute shader the top of the source.
 */
export const meshletStructsWGSL = /* wgsl */ `
    ${structWGSL('MeshletObjectData', MESHLET_OBJECT_STRUCT)}
    ${structWGSL('MeshletPrimitiveData', MESHLET_PRIMITIVE_STRUCT)}
    ${structWGSL('MeshletData', MESHLET_DATA_STRUCT)}
    ${structWGSL('MeshletMaterial', MESHLET_MATERIAL_STRUCT)}
    ${structWGSL('MeshletRecord', MESHLET_RECORD_STRUCT)}
    ${structWGSL('MeshletTexResidency', MESHLET_TEX_RESIDENCY_STRUCT)}

    fn meshletTexSlotLayer(r : MeshletTexResidency) -> u32 { return r.slotFamilyBias & 0xFFFFu; }
    fn meshletTexFamily(r : MeshletTexResidency) -> u32 { return (r.slotFamilyBias >> 16u) & 0x3u; }
    // the texture's own fine/tail boundary in source-mip space (its tail top can sit below the
    // family's tail size)
    fn meshletTexTailStart(r : MeshletTexResidency) -> u32 { return (r.slotFamilyBias >> 18u) & 0x3Fu; }
    fn meshletTexSizeBias(r : MeshletTexResidency) -> u32 { return (r.slotFamilyBias >> 24u) & 0xFFu; }
    fn meshletTexMinLod(r : MeshletTexResidency) -> u32 { return r.minLodTailLayer & 0xFFFFu; }
    fn meshletTexTailLayer(r : MeshletTexResidency) -> u32 { return r.minLodTailLayer >> 16u; }
`;

/**
 * `objectData : array<MeshletObjectData>` (read), with the accessors for what an instance row
 * does not hold itself: `meshletInstanceMatrix(instance)` expands the affine rows to a full
 * matrix, and `meshletPrimitiveData(instance)` reads the instance's primitive row (see
 * OBJECT_DATA_U32S). Requires meshletStructsWGSL.
 */
export const meshletObjectDataWGSL = /* wgsl */ `
    var<storage, read> objectData : array<MeshletObjectData>;

    fn meshletInstanceMatrix(instance : u32) -> mat4x4f {
        let rows = objectData[instance].matrixRows;
        return transpose(mat4x4f(rows[0], rows[1], rows[2], vec4f(0.0, 0.0, 0.0, 1.0)));
    }

    // the primitive row reinterpreted from the instance struct's words, per PRIMITIVE_DATA
    fn meshletPrimitiveData(instance : u32) -> MeshletPrimitiveData {
        let row = objectData[objectData[instance].primitive];
        let words = bitcast<vec4u>(row.matrixRows[2]);
        var primitive : MeshletPrimitiveData;
        primitive.sphere = row.matrixRows[0];
        primitive.gridOrigin = row.matrixRows[1].xyz;
        primitive.gridStep = row.matrixRows[1].w;
        primitive.firstMeshlet = words.x;
        primitive.meshletCount = words.y;
        primitive.material = words.z;
        primitive.uvFloatsPerVertex = words.w;
        primitive.pickIdBase = row.primitive;
        return primitive;
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

/** `texResidency : array<MeshletTexResidency>` (read). Requires meshletStructsWGSL. */
export const meshletTexResidencyWGSL = /* wgsl */ `
    var<storage, read> texResidency : array<MeshletTexResidency>;
`;

/**
 * `meshletDecodeOct16(word)`: octahedral normal, snorm16 x | snorm16 y << 16. Also declares
 * `meshletOctDecode(x, y)`, the octahedral unfold the tangent decode shares.
 */
export const meshletDecodeOct16WGSL = /* wgsl */ `
    fn meshletOctDecode(x : f32, y : f32) -> vec3f {
        var n = vec3f(x, y, 1.0 - abs(x) - abs(y));
        if (n.z < 0.0) {
            let px = (1.0 - abs(n.y)) * select(-1.0, 1.0, n.x >= 0.0);
            let py = (1.0 - abs(n.x)) * select(-1.0, 1.0, n.y >= 0.0);
            n = vec3f(px, py, n.z);
        }
        return normalize(n);
    }

    fn meshletDecodeOct16(word : u32) -> vec3f {
        return meshletOctDecode(f32(i32(word << 16u) >> 16u) / 32767.0, f32(i32(word & 0xFFFF0000u) >> 16u) / 32767.0);
    }
`;

/**
 * `meshletDecodeTangent(word)`: oct16_sign tangent - snorm15 x | snorm15 y << 15 | bitangent
 * sign << 30. Returns xyz = tangent, w = bitangent sign. Requires meshletDecodeOct16WGSL.
 */
export const meshletDecodeTangentWGSL = /* wgsl */ `
    fn meshletDecodeTangent(word : u32) -> vec4f {
        let tangent = meshletOctDecode(f32((i32(word << 17u)) >> 17u) / 16383.0, f32((i32(word << 2u)) >> 17u) / 16383.0);
        return vec4f(tangent, select(1.0, -1.0, (word & 0x40000000u) != 0u));
    }
`;

/**
 * `meshletNormalMatrix(modelMatrix)`: the cofactor of the upper 3x3 - the inverse transpose up to
 * scale, so normals stay perpendicular under non-uniform scale - negated when the determinant is,
 * so mirrored instances keep their normals facing out.
 */
export const meshletNormalMatrixWGSL = /* wgsl */ `
    fn meshletNormalMatrix(modelMatrix : mat4x4f) -> mat3x3f {
        let m0 = modelMatrix[0].xyz;
        let m1 = modelMatrix[1].xyz;
        let m2 = modelMatrix[2].xyz;
        let c0 = cross(m1, m2);
        let s = select(1.0, -1.0, dot(m0, c0) < 0.0);
        return mat3x3f(c0 * s, cross(m2, m0) * s, cross(m0, m1) * s);
    }
`;

/**
 * The SoA layout of one resident page and the vertex decode over it. Requires
 * `pagePool : array<u32>`. A page is: header (PAGE_HEADER_BYTES) | positions | oct16 normals |
 * oct16_sign tangents when the resource has them | UVs | rgba8 vertex colours when the resource
 * has them | a per-meshlet table | the u8 triangle corner stream. Two encodings, chosen per page:
 *
 * - v2: positions u16x4 anchor-relative (i32x3 when PAGE_FLAG_WIDE_POSITIONS), UVs f32x2 per
 *   channel, and the table is the meshlet-vertex remap (u32 page vertex per meshlet corner slot).
 * - v3 (PAGE_FLAG_COMPACT): each meshlet's vertices are its own, in order, so a meshlet's vertex
 *   i is page vertex verticesOffset + i; positions u16x3 packed (i32x3 when wide); UVs unorm16x2
 *   per channel, and the table holds each meshlet's UV box per channel (f32 origin, f32 extent).
 *
 * The attribute layout is per resource, so its parameters come from the instance's primitive
 * row in objectData.
 */
export const meshletPageLayoutWGSL = /* wgsl */ `
    struct MeshletPageLayout {
        positionBase : u32,
        normalBase : u32,
        tangentBase : u32,        // equals uvBase when the page carries no tangents
        uvBase : u32,
        colorBase : u32,          // equals the table base when the page carries no colours
        meshletVertexBase : u32,  // v2: the per-meshlet vertex remap table
        uvBoxBase : u32,          // compact: the per-meshlet UV box table
        triangleByteBase : u32,   // byte index of the u8 corner stream in pagePool
        uvWordsPerVertex : u32,
        anchor : vec3i,           // grid anchor the u16 positions are relative to
        widePositions : bool,
        compact : bool
    };

    fn meshletPageLayout(pageBase : u32, hasTangents : bool, uvFloatsPerVertex : u32, hasColors : bool) -> MeshletPageLayout {
        var pageLayout : MeshletPageLayout;
        let vertexCount = pagePool[pageBase + ${PAGE_HEADER.VERTEX_COUNT}u];
        let flags = pagePool[pageBase + ${PAGE_HEADER.FLAGS}u];
        pageLayout.widePositions = (flags & ${PAGE_FLAG_WIDE_POSITIONS}u) != 0u;
        pageLayout.compact = (flags & ${PAGE_FLAG_COMPACT}u) != 0u;
        pageLayout.anchor = vec3i(i32(pagePool[pageBase + ${PAGE_HEADER.ANCHOR_X}u]), i32(pagePool[pageBase + ${PAGE_HEADER.ANCHOR_Y}u]), i32(pagePool[pageBase + ${PAGE_HEADER.ANCHOR_Z}u]));
        pageLayout.positionBase = pageBase + ${PAGE_HEADER_BYTES / 4}u;
        // narrow positions: u16x4 (two words), or u16x3 packed and padded to a word on compact pages
        let narrowWords = select(vertexCount * 2u, (vertexCount * 3u + 1u) / 2u, pageLayout.compact);
        pageLayout.normalBase = pageLayout.positionBase + select(narrowWords, vertexCount * 3u, pageLayout.widePositions);
        pageLayout.tangentBase = pageLayout.normalBase + vertexCount;
        pageLayout.uvBase = pageLayout.tangentBase + select(0u, vertexCount, hasTangents);
        // one unorm16x2 word per channel on compact pages, two f32 otherwise
        pageLayout.uvWordsPerVertex = select(uvFloatsPerVertex, uvFloatsPerVertex / 2u, pageLayout.compact);
        pageLayout.colorBase = pageLayout.uvBase + vertexCount * pageLayout.uvWordsPerVertex;
        let tableBase = pageLayout.colorBase + select(0u, vertexCount, hasColors);
        pageLayout.meshletVertexBase = tableBase;
        pageLayout.uvBoxBase = tableBase;
        let meshletCount = pagePool[pageBase + ${PAGE_HEADER.MESHLET_COUNT}u];
        let tableWords = select(pagePool[pageBase + ${PAGE_HEADER.MV_COUNT}u], meshletCount * (uvFloatsPerVertex / 2u) * ${PAGE_UV_BOX_WORDS}u, pageLayout.compact);
        pageLayout.triangleByteBase = (tableBase + tableWords) * 4u;
        return pageLayout;
    }

    // one u16 of a packed u16 stream starting at word base
    fn meshletPageU16(base : u32, index : u32) -> i32 {
        return i32((pagePool[base + (index >> 1u)] >> ((index & 1u) * 16u)) & 0xFFFFu);
    }

    // dequantised local-space position of one page vertex on the instance's grid
    fn meshletPagePosition(pageLayout : MeshletPageLayout, vertex : u32, gridOrigin : vec3f, gridStep : f32) -> vec3f {
        var coord : vec3i;
        if (pageLayout.widePositions) {
            let b = pageLayout.positionBase + vertex * 3u;
            coord = vec3i(i32(pagePool[b]), i32(pagePool[b + 1u]), i32(pagePool[b + 2u]));
        } else if (pageLayout.compact) {
            let h = vertex * 3u;
            coord = pageLayout.anchor + vec3i(meshletPageU16(pageLayout.positionBase, h), meshletPageU16(pageLayout.positionBase, h + 1u), meshletPageU16(pageLayout.positionBase, h + 2u));
        } else {
            let b = pageLayout.positionBase + vertex * 2u;
            let word0 = pagePool[b];
            let word1 = pagePool[b + 1u];
            coord = pageLayout.anchor + vec3i(i32(word0 & 0xFFFFu), i32(word0 >> 16u), i32(word1 & 0xFFFFu));
        }
        return vec3f(coord) * gridStep + gridOrigin;
    }

    // UV channel of one page vertex; pageSlot is the meshlet's UV box on a compact page
    fn meshletPageUv(pageLayout : MeshletPageLayout, vertex : u32, pageSlot : u32, channel : u32) -> vec2f {
        if (pageLayout.compact) {
            let word = pagePool[pageLayout.uvBase + vertex * pageLayout.uvWordsPerVertex + channel];
            let boxWord = pageLayout.uvBoxBase + (pageSlot * pageLayout.uvWordsPerVertex + channel) * ${PAGE_UV_BOX_WORDS}u;
            let origin = vec2f(bitcast<f32>(pagePool[boxWord]), bitcast<f32>(pagePool[boxWord + 1u]));
            let extent = vec2f(bitcast<f32>(pagePool[boxWord + 2u]), bitcast<f32>(pagePool[boxWord + 3u]));
            return origin + vec2f(f32(word & 0xFFFFu), f32(word >> 16u)) / 65535.0 * extent;
        }
        let b = pageLayout.uvBase + vertex * pageLayout.uvWordsPerVertex + channel * 2u;
        return vec2f(bitcast<f32>(pagePool[b]), bitcast<f32>(pagePool[b + 1u]));
    }

    // COLOR_0 of one page vertex (rgba8 unorm, r in the low byte); only meaningful when the
    // instance's resource carries colours
    fn meshletPageColor(pageLayout : MeshletPageLayout, vertex : u32) -> vec4f {
        return unpack4x8unorm(pagePool[pageLayout.colorBase + vertex]);
    }
`;

/**
 * `meshletFetchVertex(instanceIndex, vertexIndex, pageSizeWords)`: the vertex pull every meshlet
 * vertex stage starts with. A bucket's draw is one instance per drawn meshlet with a fixed vertex
 * count (the world's largest meshlet's corners), starting at vertex `bucket <<
 * MESHLET_DRAW_BUCKET_SHIFT`: the instance index is the bucket's record slot, the low vertex
 * bits the corner. Finds the record's instance and meshlet, lays out the meshlet's resident page
 * for the instance's resource, reads the corner's local vertex from the page's u8 triangle stream
 * (winding reversed for a mirrored instance) and resolves the page vertex. Corners past the
 * meshlet's triangles all read its first corner, so their triangles are degenerate and never
 * rasterise. Requires meshletStructsWGSL, meshletPageLayoutWGSL and the `records`, `meshletData`,
 * `objectData`, `residency` and `pagePool` buffers.
 */
export const meshletFetchVertexWGSL = /* wgsl */ `
    struct MeshletVertex {
        instance : u32,
        meshlet : u32,
        primitive : MeshletPrimitiveData,
        hasTangents : bool,
        hasColors : bool,
        pageLayout : MeshletPageLayout,
        vertex : u32,             // page vertex index
        pageSlot : u32            // the meshlet's UV box on a compact page
    };

    fn meshletFetchVertex(instanceIndex : u32, vertexIndex : u32, pageSizeWords : u32) -> MeshletVertex {
        let bucket = vertexIndex >> ${MESHLET_DRAW_BUCKET_SHIFT}u;
        let record = records[instanceIndex * ${MESHLET_BUCKET_COUNT}u + bucket];
        var v : MeshletVertex;
        v.instance = record.instance;
        v.meshlet = record.meshlet;
        // per-primitive page attribute layout and position grid (resources may differ)
        v.primitive = meshletPrimitiveData(v.instance);
        let flags = objectData[v.instance].flags;
        v.hasTangents = (flags & ${OBJECT_FLAG_HAS_TANGENTS}u) != 0u;
        v.hasColors = (flags & ${OBJECT_FLAG_HAS_COLORS}u) != 0u;
        let pageBase = residency[meshletData[v.meshlet].page] * pageSizeWords;
        v.pageLayout = meshletPageLayout(pageBase, v.hasTangents, v.primitive.uvFloatsPerVertex, v.hasColors);

        // the corner's local vertex, from the u8 triangle stream ending the page
        var corner = vertexIndex & ${(1 << MESHLET_DRAW_BUCKET_SHIFT) - 1}u;
        if (corner >= meshletData[v.meshlet].triangleCount * 3u) {
            corner = 0u;
        } else {
            let rows = objectData[v.instance].matrixRows;
            // det of the upper 3x3 (the rows' determinant equals the matrix's)
            if (dot(cross(rows[0].xyz, rows[1].xyz), rows[2].xyz) < 0.0) {
                let triangleCorner = corner % 3u;
                corner = corner - triangleCorner + (3u - triangleCorner) % 3u;
            }
        }
        let byteIndex = v.pageLayout.triangleByteBase + meshletData[v.meshlet].triangleOffset + corner;
        let localVertex = (pagePool[byteIndex >> 2u] >> ((byteIndex & 3u) * 8u)) & 0xFFu;
        let meshletVertex = meshletData[v.meshlet].verticesOffset + localVertex;
        if (v.pageLayout.compact) {
            v.vertex = meshletVertex;
        } else {
            v.vertex = pagePool[v.pageLayout.meshletVertexBase + meshletVertex];
        }
        v.pageSlot = meshletData[v.meshlet].uvChannelMask >> ${MESHLET_PAGE_SLOT_SHIFT}u;
        return v;
    }
`;
