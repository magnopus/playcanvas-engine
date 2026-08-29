import {
    MATERIAL_SLOT, MATERIAL_SLOT_ABSENT, MESHLET_MAX_UV_CHANNELS, MESHLET_TEX_NO_MINLOD, OBJECT_FLAG_HAS_TANGENTS,
    OBJECT_FLAG_HOVERED, OBJECT_FLAG_OUTLINED
} from '../constants.js';
import {
    meshletDataWGSL, meshletDecodeDrawIndexWGSL, meshletDecodeOct16WGSL, meshletDecodeTangentWGSL,
    meshletMaterialTableWGSL, meshletObjectDataWGSL, meshletPageLayoutWGSL, meshletRecordsWGSL,
    meshletStructsWGSL, meshletTexResidencyWGSL
} from './meshlet-page-wgsl.js';

/**
 * WGSL chunk overrides that turn a StandardMaterial into the meshlet forward material: the
 * vertex stage pulls geometry from the page pool via the frame's record + index buffers
 * (typed buffer views, decode and page layout in meshlet-page-wgsl.js), the fragment frontend
 * reads the material record (`materialTable[row].roughness`) instead of material_* uniforms, and - when the
 * world carries streamed textures - samples the per-family resident TAIL arrays through the
 * texResidency indirection (see textures/meshlet-textures.js). Everything else - clustered
 * lights, shadows, IBL, fog, tonemapping - is the engine's own lit backend.
 *
 * The litEngine* slots (not litUser*) are the engine-internal hook set; litUser* stays free for
 * application overrides. See shader-chunks-wgsl.js.
 *
 * Chunk contracts honoured (see wgsl/chunks/lit/vert/litMain.js):
 * - transformCoreVS supplies vertex_position, getModelMatrix(), getLocalPosition(). Replacing
 *   the `attribute` declaration with a private var removes the vertex buffer entirely - the
 *   WGSL processor only emits VertexInput fields for literal attribute lines. It also declares
 *   vertex_normal, which normalCoreVS would normally own - see the note there.
 * - normalCoreVS supplies getLocalNormal(), getNormalMatrix().
 * - litEngineMainStartVS runs first in vertexMain: it decodes pcVertexIndex and fills the
 *   private vars the engine chunks then consume.
 * - UV derivatives are cached in litEngineMainStartPS (uniform control flow) because sampling
 *   happens inside data-dependent branches where derivative builtins are illegal.
 *
 * The chunk set is BUILT per world ({@link buildMeshletLitChunks}) - texture sampling, UV
 * varyings and tangent support are compiled in only when the world's resources carry them.
 *
 * @ignore
 */

/**
 * @param {object} options - What the world's resources carry.
 * @param {boolean} options.textures - True when streamed textures (tail arrays) exist.
 * @param {number} options.uvChannels - UV channels to thread through (0 to MESHLET_MAX_UV_CHANNELS).
 * @param {boolean} options.tangents - True when pages carry oct16+sign tangents.
 * @param {number[]} [options.familySizes] - Fine slot-pool base size per family (4 entries).
 * @param {Array<{ slotLevels: number, tailLevels: number }>} [options.familyLevels] - Fine
 * levels above the tail and populated tail levels, per family.
 * @returns {object} The chunk override map for StandardMaterial.getShaderChunks('wgsl').add().
 * @ignore
 */
function buildMeshletLitChunks({ textures = false, uvChannels = 0, tangents = false, familySizes = [], familyLevels = [] } = {}) {

    // one varying + cached derivatives per channel; a slot picks its channel from its slot word
    const uvs = Array.from({ length: Math.min(uvChannels, MESHLET_MAX_UV_CHANNELS) }, (_, n) => n);
    const uv0 = uvs.length > 0;
    const tan = tangents && textures;

    // ------------------------------------------------------------------ vertex stage

    const transformCoreVS = /* wgsl */ `
        var<private> vertex_position: vec4f;

        // declared here rather than in normalCoreVS because that chunk is NORMALS-gated
        // (litMain.js:51) and shadow passes force needsNormal = false (lit-shader.js:170),
        // while litEngineMainStartVS assigns vertex_normal unconditionally. transformCoreVS
        // is included on every pass, so the declaration always exists; when NORMALS is off
        // the assignment is a dead store.
        var<private> vertex_normal: vec3f;

        uniform matrix_viewProjection: mat4x4f;
        uniform pageSizeWords: u32;

        ${meshletDataWGSL}
        ${meshletObjectDataWGSL}
        ${meshletRecordsWGSL('read')}
        var<storage, read> pagePool : array<u32>;
        var<storage, read> residency : array<u32>;

        var<private> dMeshletModelMatrix: mat4x4f;

        fn getModelMatrix() -> mat4x4f {
            return dMeshletModelMatrix;
        }

        fn getLocalPosition(vertexPosition: vec3f) -> vec3f {
            return vertexPosition;
        }
    `;

    const normalCoreVS = /* wgsl */ `
        fn getLocalNormal(vertexNormal: vec3f) -> vec3f {
            return vertexNormal;
        }

        fn getNormalMatrix(modelMatrix: mat4x4f) -> mat3x3f {
            return mat3x3f(modelMatrix[0].xyz, modelMatrix[1].xyz, modelMatrix[2].xyz);
        }
    `;

    // the struct block precedes every uniform in the stage (see meshletStructsWGSL); the
    // processor merges both stages' storage declarations, so both stages need all the structs
    const litEngineDeclarationVS = /* wgsl */ `
        ${meshletStructsWGSL}
        varying @interpolate(flat) vMeshletMatRow: u32;
        var<private> dMeshletMatRow: u32;
        varying @interpolate(flat) vMeshletPickId: u32;
        var<private> dMeshletPickId: u32;
        #ifdef PCOUTLINE_PASS
        varying @interpolate(flat) vMeshletHovered: u32;
        var<private> dMeshletHovered: u32;
        #endif
        ${uvs.map(n => `varying vMeshletUv${n}: vec2f; var<private> dMeshletUv${n}: vec2f;`).join('\n        ')}
        ${tan ? `varying vMeshletTangentW: vec3f; var<private> dMeshletTangentW: vec3f;
        varying @interpolate(flat) vMeshletBtSign: f32; var<private> dMeshletBtSign: f32;` : ''}
    `;

    const litEngineCodeVS = /* wgsl */ `
        ${meshletDecodeDrawIndexWGSL}
        ${meshletDecodeOct16WGSL}
        ${tan ? meshletDecodeTangentWGSL : ''}
        ${meshletPageLayoutWGSL}
    `;

    const litEngineMainStartVS = /* wgsl */ `
        let meshletDrawIndex = meshletDecodeDrawIndex(pcVertexIndex);
        let meshletRecord = meshletDrawIndex.x;
        let meshletLocalVert = meshletDrawIndex.y;

        let meshletInstance = records[meshletRecord].instance;
        let meshletIndex = records[meshletRecord].meshlet;

        let meshletPage = meshletData[meshletIndex].page;
        let meshletVerticesOffset = meshletData[meshletIndex].verticesOffset;

        // per-instance page attribute layout and position grid (resources may differ)
        let meshletUvFloats = objectData[meshletInstance].uvFloatsPerVertex;
        let meshletHasTangents = (objectData[meshletInstance].flags & ${OBJECT_FLAG_HAS_TANGENTS}u) != 0u;
        let meshletGridOrigin = meshletObjectGridOrigin(meshletInstance);
        let meshletGridStep = objectData[meshletInstance].gridStep;

        let meshletLayout = meshletPageLayout(residency[meshletPage] * uniform.pageSizeWords, meshletHasTangents, meshletUvFloats);
        let meshletVert = pagePool[meshletLayout.meshletVertexBase + meshletVerticesOffset + meshletLocalVert];

        vertex_position = vec4f(meshletPagePosition(meshletLayout, meshletVert, meshletGridOrigin, meshletGridStep), 1.0);

        #ifdef PCOUTLINE_PASS
        // Selection is per instance (OBJECT_FLAG_OUTLINED / OBJECT_FLAG_HOVERED in the objectData
        // flags) because one indirect draw covers the whole world - there is no mesh instance
        // for the outline renderer to add to its layer. Instances in neither state collapse to a
        // single point, so the triangle is degenerate and never rasterises: they cost this vertex
        // fetch and nothing else.
        let meshletOutlineFlags = objectData[meshletInstance].flags;
        if ((meshletOutlineFlags & ${OBJECT_FLAG_OUTLINED | OBJECT_FLAG_HOVERED}u) == 0u) {
            vertex_position = vec4f(meshletGridOrigin, 1.0);
        }
        dMeshletHovered = select(0u, 1u, (meshletOutlineFlags & ${OBJECT_FLAG_HOVERED}u) != 0u);
        #endif
        vertex_normal = meshletDecodeOct16(pagePool[meshletLayout.normalBase + meshletVert]);
        dMeshletModelMatrix = objectData[meshletInstance].worldMatrix;
        dMeshletMatRow = objectData[meshletInstance].material;
        dMeshletPickId = objectData[meshletInstance].pickId;

        ${uvs.map(n => /* wgsl */ `
        // channel ${n}: two floats per channel per vertex, present when the page carries them
        dMeshletUv${n} = vec2f(0.0);
        if (meshletUvFloats >= ${2 * (n + 1)}u) {
            let meshletUv${n}Word = meshletLayout.uvBase + meshletVert * meshletUvFloats + ${2 * n}u;
            dMeshletUv${n} = vec2f(bitcast<f32>(pagePool[meshletUv${n}Word]), bitcast<f32>(pagePool[meshletUv${n}Word + 1u]));
        }`).join('')}
        ${tan ? /* wgsl */ `
        dMeshletTangentW = vec3f(1.0, 0.0, 0.0);
        dMeshletBtSign = 1.0;
        if (meshletHasTangents) {
            let meshletTangent = meshletDecodeTangent(pagePool[meshletLayout.tangentBase + meshletVert]);
            dMeshletTangentW = normalize((dMeshletModelMatrix * vec4f(meshletTangent.xyz, 0.0)).xyz);
            dMeshletBtSign = meshletTangent.w;
        }` : ''}
    `;

    const litEngineMainEndVS = /* wgsl */ `
        output.vMeshletMatRow = dMeshletMatRow;
        output.vMeshletPickId = dMeshletPickId;
        #ifdef PCOUTLINE_PASS
        output.vMeshletHovered = dMeshletHovered;
        #endif
        ${uvs.map(n => `output.vMeshletUv${n} = dMeshletUv${n};`).join('\n        ')}
        ${tan ? 'output.vMeshletTangentW = dMeshletTangentW; output.vMeshletBtSign = dMeshletBtSign;' : ''}
    `;

    // ------------------------------------------------------------------ fragment stage

    // Per-family sampling: family order is fixed (srgb, srgba, normal, linear); absent
    // families bind placeholders and are unreachable thanks to the minLod sentinel.
    // The virtual mip chain per texture is [fine slot levels | tail levels], with the
    // texture's sizeBias (its level offset inside the family arrays) applied: lod is
    // computed in SOURCE-mip space, clamped to the resident range, then dispatched to the
    // fine slot pool when the texture holds a slot and the level lies above the tail.
    const famNames = ['Srgb', 'Srgba', 'Normal', 'Linear'];
    // the WGSL processor's resource regex matches one declaration per line
    const texDecls = textures ? famNames.map(n => [
        `var meshletTail${n}: texture_2d_array<f32>;`,
        `var meshletTail${n}Sampler: sampler;`,
        `var meshletFine${n}: texture_2d_array<f32>;`,
        `var meshletFine${n}Sampler: sampler;`
    ].join('\n        ')).join('\n        ') : '';
    const texDispatch = textures ? famNames.map((n, f) => {
        const slotSize = (familySizes[f] ?? 4).toFixed(1);
        const slotLevels = (familyLevels[f]?.slotLevels ?? 0).toFixed(1);
        const tailMaxLod = ((familyLevels[f]?.tailLevels ?? 1) - 1).toFixed(1);
        return /* wgsl */ `
            if (family == ${f}u) {
                let srcSize = ${slotSize} / exp2(sizeBias);
                let lod = clamp(meshletTexLod(ddx, ddy, srcSize), minLod, 30.0);
                let comb = lod + sizeBias;
                if (slotLayer != 0xFFFFu && comb < ${slotLevels}) {
                    // clamp keeps trilinear off the unwritten level below the finest chain
                    return textureSampleLevel(meshletFine${n}, meshletFine${n}Sampler, uv, i32(slotLayer), min(comb, ${slotLevels} - 1.0));
                }
                let tailLod = clamp(comb - ${slotLevels}, 0.0, ${tailMaxLod});
                return textureSampleLevel(meshletTail${n}, meshletTail${n}Sampler, uv, tailLayer, tailLod);
            }`;
    }).join('') : '';

    const litEngineDeclarationPS = /* wgsl */ `
        ${meshletStructsWGSL}
        varying @interpolate(flat) vMeshletMatRow: u32;
        varying @interpolate(flat) vMeshletPickId: u32;
        ${meshletMaterialTableWGSL}

        // One indirect draw covers every instance in the world, so the picker's per-mesh-instance
        // id uniform is meaningless here - the id has to travel with the geometry. PICK_CUSTOM_ID
        // (set on the bucket materials) suppresses that uniform so this takes over.
        #ifdef PICK_PASS
        fn getPickOutput() -> vec4f {
            return encodePickOutput(vMeshletPickId);
        }
        #endif

        #ifdef PCOUTLINE_PASS
        varying @interpolate(flat) vMeshletHovered: u32;
        uniform pcOutlineColorHover: vec3f;
        #endif

        ${uvs.map(n => `varying vMeshletUv${n}: vec2f; var<private> dMeshletUv${n}Dx: vec2f; var<private> dMeshletUv${n}Dy: vec2f;`).join('\n        ')}
        ${tan ? 'varying vMeshletTangentW: vec3f; varying @interpolate(flat) vMeshletBtSign: f32;' : ''}

        ${textures ? /* wgsl */ `
        ${meshletTexResidencyWGSL}
        ${texDecls}

        fn meshletTexLod(ddx: vec2f, ddy: vec2f, size: f32) -> f32 {
            let dx = ddx * size;
            let dy = ddy * size;
            return 0.5 * log2(max(max(dot(dx, dx), dot(dy, dy)), 1e-12));
        }

        // samples the given material slot (MATERIAL_SLOT.*) from its texture's resident mips
        // (fine slot pool above the tail, resident tail below), or returns the fallback when
        // the slot is absent or nothing is resident yet. Derivatives are the cached
        // uniform-flow ones.
        fn meshletSampleSlot(slot: u32, fallback: vec4f) -> vec4f {
            let slotWord = materialTable[vMeshletMatRow].slotWords[slot];
            let texIndex = slotWord & 0xFFFFu;
            if (texIndex == ${MATERIAL_SLOT_ABSENT}u) { return fallback; }
            let resident = texResidency[texIndex];
            if (meshletTexMinLod(resident) == ${MESHLET_TEX_NO_MINLOD}u) { return fallback; }  // nothing resident yet
            let minLod = f32(meshletTexMinLod(resident));
            let family = meshletTexFamily(resident);
            let sizeBias = f32(meshletTexSizeBias(resident));
            let slotLayer = meshletTexSlotLayer(resident);
            let tailLayer = i32(meshletTexTailLayer(resident));

            var uv = ${uv0 ? 'vMeshletUv0' : 'vec2f(0.0)'};
            var ddx = ${uv0 ? 'dMeshletUv0Dx' : 'vec2f(0.0)'};
            var ddy = ${uv0 ? 'dMeshletUv0Dy' : 'vec2f(0.0)'};
            ${uvs.length > 1 ? /* wgsl */ `
            // slot word bits 24-25: the UV channel this slot samples (0-3)
            let slotUvChannel = (slotWord >> 24u) & 3u;` : ''}${uvs.slice(1).map(n => /* wgsl */ `
            if (slotUvChannel == ${n}u) {
                uv = vMeshletUv${n}; ddx = dMeshletUv${n}Dx; ddy = dMeshletUv${n}Dy;
            }`).join('')}

            // KHR_texture_transform (packed f16 pairs: x = offset, y = scale; both 0 = identity)
            let transform = materialTable[vMeshletMatRow].slotTransforms[slot];
            if ((transform.x | transform.y) != 0u) {
                let transformScale = unpack2x16float(transform.y);
                uv = uv * transformScale + unpack2x16float(transform.x);
                ddx = ddx * transformScale;
                ddy = ddy * transformScale;
            }
            ${texDispatch}
            return fallback;
        }` : ''}
    `;

    const litEngineMainStartPS = uv0 ? /* wgsl */ `
        ${uvs.map(n => `dMeshletUv${n}Dx = dpdx(vMeshletUv${n}); dMeshletUv${n}Dy = dpdy(vMeshletUv${n});`).join('\n        ')}
    ` : null;

    const sample = (slot, fallback) => (textures ? `meshletSampleSlot(${slot}u, ${fallback})` : fallback);
    // this fragment's material record
    const material = 'materialTable[vMeshletMatRow]';

    const diffusePS = /* wgsl */ `
        fn getAlbedo() {
            dAlbedo = ${material}.baseColor.rgb * ${sample(MATERIAL_SLOT.BASE_COLOR, 'vec4f(1.0)')}.rgb;
        }
    `;

    const opacityPS = /* wgsl */ `
        fn getOpacity() {
            dAlpha = ${material}.baseColor.a * ${sample(MATERIAL_SLOT.BASE_COLOR, 'vec4f(1.0)')}.a;
        }
    `;

    // ORM: r occlusion (unused for now), g roughness, b metalness
    const metalnessPS = /* wgsl */ `
        fn getMetalness() {
            dMetalness = ${material}.metallic * ${sample(MATERIAL_SLOT.ORM, 'vec4f(1.0)')}.b;
        }
    `;

    // the engine works in gloss; the record stores roughness. The epsilon keeps gloss off an
    // exact zero, which the specular maths does not like.
    const glossPS = /* wgsl */ `
        fn getGlossiness() {
            dGlossiness = 1.0 - ${material}.roughness * ${sample(MATERIAL_SLOT.ORM, 'vec4f(1.0)')}.g + 0.0000001;
        }
    `;

    const emissivePS = /* wgsl */ `
        fn getEmission() {
            dEmission = ${material}.emissive * ${material}.emissiveStrength * ${sample(MATERIAL_SLOT.EMISSIVE, 'vec4f(1.0)')}.rgb;
        }
    `;

    // per-record cutoff: one indirect draw spans materials with different cutoffs, so the
    // engine's per-material alpha_ref uniform cannot work here
    const alphaTestPS = /* wgsl */ `
        fn alphaTest(a: f32) {
            if (a < ${material}.alphaCutoff) {
                discard;
            }
        }
    `;

    // two-sided normal flip without LIT_TBN, plus (when tangents stream) normal mapping from
    // the page-pool tangent frame
    const normalMapPS = /* wgsl */ `
        fn getNormal() {
            var n = select(-dVertexNormalW, dVertexNormalW, pcFrontFacing);
            ${tan ? /* wgsl */ `
            let normalSlotWord = ${material}.slotWords[${MATERIAL_SLOT.NORMAL}u];
            if ((normalSlotWord & 0xFFFFu) != ${MATERIAL_SLOT_ABSENT}u) {
                let normalSample = ${sample(MATERIAL_SLOT.NORMAL, 'vec4f(0.5, 0.5, 1.0, 1.0)')}.xyz * 2.0 - 1.0;
                let T = normalize(vMeshletTangentW - n * dot(vMeshletTangentW, n));
                let B = cross(n, T) * vMeshletBtSign;
                // max(z, 0.05) softens a degenerate (flat or inverted) normal-map texel
                n = normalize(T * normalSample.x + B * normalSample.y + n * max(normalSample.z, 0.05));
            }` : ''}
            dNormalW = n;
        }
    `;

    // The stock chunk emits one uniform colour for the whole draw. Selection and hover are
    // per-instance states here, so pick between the two colours from the flag the vertex stage
    // forwarded.
    const outlineOutputPS = /* wgsl */ `
        #ifdef PCOUTLINE_PASS
        let meshletOutlineRgb = select(uniform.pcOutlineColor, uniform.pcOutlineColorHover, vMeshletHovered != 0u);
        output.color = vec4f(gammaCorrectOutput(meshletOutlineRgb), output.color.a);
        #endif
    `;

    const chunks = {
        outlineOutputPS,
        transformCoreVS,
        normalCoreVS,
        litEngineDeclarationVS,
        litEngineCodeVS,
        litEngineMainStartVS,
        litEngineMainEndVS,
        litEngineDeclarationPS,
        diffusePS,
        opacityPS,
        metalnessPS,
        glossPS,
        emissivePS,
        alphaTestPS,
        normalMapPS
    };
    if (litEngineMainStartPS) {
        chunks.litEngineMainStartPS = litEngineMainStartPS;
    }
    return chunks;
}

export { buildMeshletLitChunks };
