import { Debug } from '../../../core/debug.js';
import litForwardBackend from '../../shader-lib/wgsl/chunks/lit/frag/pass-forward/litForwardBackend.js';
import {
    MATERIAL_FLAG_UNLIT, MATERIAL_SLOT, MATERIAL_SLOT_ABSENT, MESHLET_MAX_UV_CHANNELS, MESHLET_TEX_NO_MINLOD,
    OBJECT_FLAG_HAS_TANGENTS, OBJECT_FLAG_HAS_COLORS, OBJECT_FLAG_HOVERED, OBJECT_FLAG_OUTLINED
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
 * @param {boolean} [options.colors] - True when pages carry rgba8 vertex colours (COLOR_0), which
 * multiply the base colour as on the regular glTF path.
 * @param {boolean} [options.lightmaps] - True when placements carry streamed EPIC lightmaps.
 * @param {string} [options.forwardBackend] - Active device backend, including application overrides.
 * @param {number[]} [options.familySizes] - Fine slot-pool base size per family (4 entries).
 * @param {Array<{ slotLevels: number, tailLevels: number }>} [options.familyLevels] - Fine
 * levels above the tail and populated tail levels, per family.
 * @returns {object} The chunk override map for StandardMaterial.getShaderChunks('wgsl').add().
 * @ignore
 */
function buildMeshletLitChunks({ textures = false, uvChannels = 0, tangents = false, colors = false, lightmaps = false, forwardBackend = litForwardBackend, familySizes = [], familyLevels = [] } = {}) {

    // one varying + cached derivatives per channel; a slot picks its channel from its slot word
    const uvs = Array.from({ length: Math.min(uvChannels, MESHLET_MAX_UV_CHANNELS) }, (_, n) => n);
    const uv0 = uvs.length > 0;
    const tan = tangents && textures;
    const col = colors;
    const lightmapped = lightmaps && textures && uv0;

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
        ${lightmapped ? 'varying @interpolate(flat) vMeshletInstance: u32;' : ''}
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
        ${col ? 'varying vMeshletColor: vec4f; var<private> dMeshletColor: vec4f;' : ''}
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
        let meshletHasColors = (objectData[meshletInstance].flags & ${OBJECT_FLAG_HAS_COLORS}u) != 0u;
        let meshletGridOrigin = meshletObjectGridOrigin(meshletInstance);
        let meshletGridStep = objectData[meshletInstance].gridStep;

        let meshletLayout = meshletPageLayout(residency[meshletPage] * uniform.pageSizeWords, meshletHasTangents, meshletUvFloats, meshletHasColors);
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
            let meshletMirrored = dot(cross(dMeshletModelMatrix[0].xyz, dMeshletModelMatrix[1].xyz), dMeshletModelMatrix[2].xyz) < 0.0;
            dMeshletBtSign = meshletTangent.w * select(1.0, -1.0, meshletMirrored);
        }` : ''}
        ${col ? /* wgsl */ `
        // COLOR_0 multiplies the base colour (getAlbedo), as on the regular glTF path; a
        // resource without colours reads white
        dMeshletColor = vec4f(1.0);
        if (meshletHasColors) {
            dMeshletColor = meshletPageColor(meshletLayout, meshletVert);
        }` : ''}
    `;

    const litEngineMainEndVS = /* wgsl */ `
        ${lightmapped ? 'output.vMeshletInstance = meshletInstance;' : ''}
        output.vMeshletMatRow = dMeshletMatRow;
        output.vMeshletPickId = dMeshletPickId;
        #ifdef PCOUTLINE_PASS
        output.vMeshletHovered = dMeshletHovered;
        #endif
        ${uvs.map(n => `output.vMeshletUv${n} = dMeshletUv${n};`).join('\n        ')}
        ${tan ? 'output.vMeshletTangentW = dMeshletTangentW; output.vMeshletBtSign = dMeshletBtSign;' : ''}
        ${col ? 'output.vMeshletColor = dMeshletColor;' : ''}
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
    const texFunctions = textures ? famNames.map((n, f) => {
        const slotLevels = (familyLevels[f]?.slotLevels ?? 0).toFixed(1);
        const tailMaxLod = ((familyLevels[f]?.tailLevels ?? 1) - 1).toFixed(1);
        return /* wgsl */ `
            fn meshletSample${n}(uv: vec2f, wanted: f32, minLod: f32, sizeBias: f32, tailStart: f32, slotLayer: u32, tailLayer: i32) -> vec4f {
                let lod = clamp(wanted, minLod, 30.0);
                let comb = lod + sizeBias;
                // debug view: magenta tint where the surface wants a finer mip than is resident
                let starved = select(0.0, 0.6, wanted < minLod - 0.5);
                let tailFloor = max(tailStart + sizeBias - ${slotLevels}, 0.0);
                if (slotLayer != 0xFFFFu && lod < tailStart) {
                    let fineTop = tailStart + sizeBias - 1.0;
                    let fineLod = clamp(comb, minLod + sizeBias, fineTop);
                    dMeshletTexDebug = mix(mix(vec3f(0.2, 1.0, 0.2), vec3f(0.0, 0.3, 0.0), comb / max(fineTop + 1.0, 1.0)), vec3f(1.0, 0.0, 1.0), starved);
                    let fineColor = textureSampleLevel(meshletFine${n}, meshletFine${n}Sampler, uv, i32(slotLayer), fineLod);
                    let tailWeight = clamp(lod - (tailStart - 1.0), 0.0, 1.0);
                    if (tailWeight > 0.0) {
                        let tailColor = textureSampleLevel(meshletTail${n}, meshletTail${n}Sampler, uv, tailLayer, tailFloor);
                        return mix(fineColor, tailColor, tailWeight);
                    }
                    return fineColor;
                }
                let tailLod = clamp(comb - ${slotLevels}, tailFloor, ${tailMaxLod});
                dMeshletTexDebug = mix(mix(vec3f(0.2, 0.4, 1.0), vec3f(0.0, 0.0, 0.25), tailLod / max(${tailMaxLod}, 1.0)), vec3f(1.0, 0.0, 1.0), starved);
                return textureSampleLevel(meshletTail${n}, meshletTail${n}Sampler, uv, tailLayer, tailLod);
            }`;
    }).join('') : '';

    const texDispatch = textures ? famNames.map((name, family) => /* wgsl */ `
            if (family == ${family}u) {
                let srcSize = ${(familySizes[family] ?? 4).toFixed(1)} / exp2(sizeBias);
                let isotropicLod = meshletTexLod(ddx, ddy, srcSize);
                if (uniform.meshletTextureAnisotropy <= 1.0) {
                    return meshletSample${name}(uv, isotropicLod, minLod, sizeBias, tailStart, slotLayer, tailLayer);
                }
                let footprint = meshletTexFootprint(ddx, ddy);
                let major = footprint.w * srcSize;
                let minor = footprint.z * srcSize;
                let filterWidth = max(max(minor, major / uniform.meshletTextureAnisotropy), exp2(minLod));
                let sampleCount = u32(clamp(ceil(major / filterWidth), 1.0, uniform.meshletTextureAnisotropy));
                if (sampleCount == 1u) {
                    return meshletSample${name}(uv, isotropicLod, minLod, sizeBias, tailStart, slotLayer, tailLayer);
                }
                let wanted = log2(filterWidth);
                var color = vec4f(0.0);
                for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) {
                    let offset = (f32(sampleIndex) + 0.5) / f32(sampleCount) - 0.5;
                    let sampleUv = uv + footprint.xy * footprint.w * offset;
                    color += meshletSample${name}(sampleUv, wanted, minLod, sizeBias, tailStart, slotLayer, tailLayer);
                }
                return color / f32(sampleCount);
            }`).join('') : '';

    const litEngineDeclarationPS = /* wgsl */ `
        ${meshletStructsWGSL}
        ${lightmapped ? `varying @interpolate(flat) vMeshletInstance: u32;
        var<storage, read> meshletLightmaps: array<vec4f>;` : ''}
        varying @interpolate(flat) vMeshletMatRow: u32;
        varying @interpolate(flat) vMeshletPickId: u32;
        ${meshletMaterialTableWGSL}

        // KHR_materials_unlit, per material record. The row is a flat varying, so the branches
        // on this are uniform across each triangle.
        fn meshletUnlit() -> bool {
            return (materialTable[vMeshletMatRow].flags & ${MATERIAL_FLAG_UNLIT}u) != 0u;
        }

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
        ${col ? 'varying vMeshletColor: vec4f;' : ''}

        ${textures ? /* wgsl */ `
        ${meshletTexResidencyWGSL}
        ${texDecls}

        // texture-state debug view (MESHLET_COLOR_MODE.TEXTURES): every slot sample records
        // what it sampled, getAlbedo keeps the base-colour slot's record, outputPS shows it
        uniform meshletTexDebug: u32;
        uniform meshletTextureAnisotropy: f32;
        var<private> dMeshletTexDebug: vec3f;
        var<private> dMeshletTexDebugBase: vec3f;

        ${texFunctions}

        fn meshletTexFootprint(ddx: vec2f, ddy: vec2f) -> vec4f {
            let covariance = vec3f(ddx.x * ddx.x + ddy.x * ddy.x,
                                   ddx.x * ddx.y + ddy.x * ddy.y,
                                   ddx.y * ddx.y + ddy.y * ddy.y);
            let trace = covariance.x + covariance.z;
            let difference = covariance.x - covariance.z;
            let discriminant = length(vec2f(difference, 2.0 * covariance.y));
            let majorSquared = max(0.5 * (trace + discriminant), 0.0);
            let determinant = ddx.x * ddy.y - ddx.y * ddy.x;
            let minorSquared = determinant * determinant / max(majorSquared, 1e-20);
            var axis = vec2f(1.0, 0.0);
            if (covariance.z > covariance.x) { axis = vec2f(0.0, 1.0); }
            let eigenvector = select(vec2f(majorSquared - covariance.z, covariance.y),
                                     vec2f(covariance.y, majorSquared - covariance.x), covariance.z > covariance.x);
            if (dot(eigenvector, eigenvector) > 1e-30) {
                axis = normalize(eigenvector);
            }
            return vec4f(axis, sqrt(max(minorSquared, 0.0)), sqrt(majorSquared));
        }

        fn meshletTexLod(ddx: vec2f, ddy: vec2f, size: f32) -> f32 {
            let dx = ddx * size;
            let dy = ddy * size;
            return 0.5 * log2(max(max(dot(dx, dx), dot(dy, dy)), 1e-12));
        }

        // samples the given material slot (MATERIAL_SLOT.*) from its texture's resident mips
        // (fine slot pool above the tail, resident tail below), or returns the fallback when
        // the slot is absent or nothing is resident yet. Derivatives are the cached
        // uniform-flow ones.
        fn meshletSampleTexture(texIndex: u32, uv: vec2f, ddx: vec2f, ddy: vec2f, fallback: vec4f) -> vec4f {
            if (texIndex == ${MATERIAL_SLOT_ABSENT}u) { dMeshletTexDebug = vec3f(0.15); return fallback; }
            let resident = texResidency[texIndex];
            if (meshletTexMinLod(resident) == ${MESHLET_TEX_NO_MINLOD}u) { dMeshletTexDebug = vec3f(1.0, 0.1, 0.1); return fallback; }  // nothing resident yet
            let minLod = f32(meshletTexMinLod(resident));
            let family = meshletTexFamily(resident);
            let sizeBias = f32(meshletTexSizeBias(resident));
            let tailStart = f32(meshletTexTailStart(resident));
            let slotLayer = meshletTexSlotLayer(resident);
            let tailLayer = i32(meshletTexTailLayer(resident));

            ${texDispatch}
            dMeshletTexDebug = vec3f(0.6, 0.6, 0.0);
            return fallback;
        }

        fn meshletSampleSlot(slot: u32, fallback: vec4f) -> vec4f {
            let slotWord = materialTable[vMeshletMatRow].slotWords[slot];
            let texIndex = slotWord & 0xFFFFu;

            var uv = ${uv0 ? 'vMeshletUv0' : 'vec2f(0.0)'};
            var ddx = ${uv0 ? 'dMeshletUv0Dx' : 'vec2f(0.0)'};
            var ddy = ${uv0 ? 'dMeshletUv0Dy' : 'vec2f(0.0)'};
            ${uvs.length > 1 ? /* wgsl */ `
            // slot word bits 24-25: the UV channel this slot samples (0-3)
            let slotUvChannel = (slotWord >> 24u) & 3u;` : ''}${uvs.slice(1, 4).map(n => /* wgsl */ `
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
            return meshletSampleTexture(texIndex, uv, ddx, ddy, fallback);
        }` : ''}
        ${lightmapped ? /* wgsl */ `
        fn meshletHasLightmap() -> bool {
            return meshletLightmaps[vMeshletInstance * 4u + 3u].z != 0.0;
        }

        fn meshletLightmapIrradiance() -> vec3f {
            let recordBase = vMeshletInstance * 4u;
            let transform = meshletLightmaps[recordBase];
            let decodeScale = meshletLightmaps[recordBase + 1u];
            let decodeAdd = meshletLightmaps[recordBase + 2u];
            let textureInfo = meshletLightmaps[recordBase + 3u];
            if (meshletTexMinLod(texResidency[u32(textureInfo.x)]) == ${MESHLET_TEX_NO_MINLOD}u) {
                return vec3f(0.0);
            }
            var uv = vMeshletUv0;
            var ddx = dMeshletUv0Dx;
            var ddy = dMeshletUv0Dy;
            ${uvs.slice(1).map(channel => `if (textureInfo.y == ${channel}.0) {
                uv = vMeshletUv${channel}; ddx = dMeshletUv${channel}Dx; ddy = dMeshletUv${channel}Dy;
            }`).join('\n')}
            let atlasScale = transform.xy * vec2f(1.0, 0.5);
            let atlasUv = (uv * transform.xy + transform.zw) * vec2f(1.0, 0.5);
            let atlasDx = ddx * atlasScale;
            let atlasDy = ddy * atlasScale;
            let upper = meshletSampleTexture(u32(textureInfo.x), atlasUv, atlasDx, atlasDy, vec4f(0.0));
            let lower = meshletSampleTexture(u32(textureInfo.x), atlasUv + vec2f(0.0, 0.5), atlasDx, atlasDy, vec4f(0.0));
            let logLuminance = (upper.a + lower.a / 255.0 - 0.5 / 255.0) * decodeScale.a + decodeAdd.a;
            let chromaticity = upper.rgb * upper.rgb * decodeScale.rgb + decodeAdd.rgb;
            return max(exp2(logLuminance) - 0.01858136, 0.0) * 0.6 * chromaticity;
        }` : ''}
    `;

    const litEngineMainStartPS = uv0 ? /* wgsl */ `
        ${uvs.map(n => `dMeshletUv${n}Dx = dpdx(vMeshletUv${n}); dMeshletUv${n}Dy = dpdy(vMeshletUv${n});`).join('\n        ')}
    ` : null;

    const sample = (slot, fallback) => (textures ? `meshletSampleSlot(${slot}u, ${fallback})` : fallback);
    // this fragment's material record
    const material = 'materialTable[vMeshletMatRow]';

    // base colour: factor x texture x vertex colour. A lit record shades it as albedo; an unlit
    // one emits it instead (emissivePS) and zeroes the albedo so the frontend contributes nothing
    // else - the forward backend then returns before any lighting runs (see below)
    const diffusePS = /* wgsl */ `
        fn meshletSurfaceRgb() -> vec3f {
            var rgb = ${material}.baseColor.rgb * ${sample(MATERIAL_SLOT.BASE_COLOR, 'vec4f(1.0)')}.rgb;
            ${col ? 'rgb = rgb * clamp(vMeshletColor.rgb, vec3f(0.0), vec3f(1.0));' : ''}
            return rgb;
        }

        fn getAlbedo() {
            dAlbedo = select(meshletSurfaceRgb(), vec3f(0.0), meshletUnlit());
            //${textures ? 'dMeshletTexDebugBase = dMeshletTexDebug;' : ''}
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

    // an unlit record carries its colour here; the branch also spares it the emissive sample
    const emissivePS = /* wgsl */ `
        fn getEmission() {
            if (meshletUnlit()) {
                dEmission = meshletSurfaceRgb();
            } else {
                dEmission = ${material}.emissive * ${material}.emissiveStrength * ${sample(MATERIAL_SLOT.EMISSIVE, 'vec4f(1.0)')}.rgb;
            }
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
            // a degenerate tangent frame (zero, or parallel to the normal - a coarse DAG level
            // can carry either) keeps the geometric normal: normalising it would give NaN and
            // shade the whole cluster black
            let tangentProjected = vMeshletTangentW - n * dot(vMeshletTangentW, n);
            if ((normalSlotWord & 0xFFFFu) != ${MATERIAL_SLOT_ABSENT}u && dot(tangentProjected, tangentProjected) > 1e-6) {
                let normalSample = ${sample(MATERIAL_SLOT.NORMAL, 'vec4f(0.5, 0.5, 1.0, 1.0)')}.xyz * 2.0 - 1.0;
                let T = normalize(tangentProjected);
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

    // texture-state debug view: the base-colour sample's state replaces the shaded colour
    const outputPS = textures ? /* wgsl */ `
        if (uniform.meshletTexDebug != 0u) {
            output.color = vec4f(gammaCorrectOutput(dMeshletTexDebugBase), output.color.a);
        }
    ` : null;

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
    if (outputPS) {
        chunks.outputPS = outputPS;
    }
    // Unlit early-out, hooked right after the backend declares its output: the surface colour
    // already sits in litArgs_emission, so only fog / tonemap / gamma and the output chunks
    // remain. Returning here skips ambient, IBL, the clustered light loop and reflections - on
    // a skydome that is every fragment on screen.
    const outputAnchor = /var output\s*:\s*FragmentOutput;/;
    Debug.assert(outputAnchor.test(forwardBackend), 'meshlet lit chunks: the forward backend declares no FragmentOutput, unlit records will be shaded');
    let backend = forwardBackend.replace(outputAnchor, `$&

    if (meshletUnlit()) {
        var meshletUnlitRgb = addFog(litArgs_emission);
        meshletUnlitRgb = toneMap(meshletUnlitRgb);
        meshletUnlitRgb = gammaCorrectOutput(meshletUnlitRgb);
        output.color = vec4f(meshletUnlitRgb, 1.0);
        #include "outputAlphaPS"
        #include "outputPS"
        #include "debugOutputPS"
        #include "outlineOutputPS"
        return output;
    }`);
    if (lightmapped) {
        backend = backend.replace('#ifdef LIT_LIGHTMAP', `
    if (meshletHasLightmap()) {
        dDiffuseLight = meshletLightmapIrradiance();
    }
    let meshletDiffuseBeforeProbes = dDiffuseLight;
    #ifdef LIT_LIGHTMAP`).replace('#ifdef AREA_LIGHTS', `
        #ifdef LIT_REFLECTIONS
            if (meshletHasLightmap()) {
                dDiffuseLight = meshletDiffuseBeforeProbes;
            }
        #endif
        #ifdef AREA_LIGHTS`);
    }
    chunks.litForwardBackendPS = backend;
    return chunks;
}

export { buildMeshletLitChunks };
