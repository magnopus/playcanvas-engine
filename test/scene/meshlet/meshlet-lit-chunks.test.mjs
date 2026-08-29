import { expect } from 'chai';

import { CULLFACE_BACK, CULLFACE_NONE } from '../../../src/platform/graphics/constants.js';
import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import {
    MATERIAL_SLOT, MESHLET_BUCKET_MASKED, MESHLET_BUCKET_OPAQUE, MESHLET_BUCKET_OPAQUE_TWO_SIDED,
    MESHLET_COLOR_MODE, MESHLET_MAX_UV_CHANNELS
} from '../../../src/scene/meshlet/constants.js';
import { createMeshletLitMaterial } from '../../../src/scene/meshlet/meshlet-lit-material.js';
import { createMeshletMaterial } from '../../../src/scene/meshlet/meshlet-material.js';
import { buildMeshletLitChunks } from '../../../src/scene/meshlet/shaders/meshlet-lit-chunks-wgsl.js';
import litMainPS from '../../../src/scene/shader-lib/wgsl/chunks/lit/frag/litMain.js';
import litForwardMainPS from '../../../src/scene/shader-lib/wgsl/chunks/lit/frag/pass-forward/litForwardMain.js';
import litOtherMainPS from '../../../src/scene/shader-lib/wgsl/chunks/lit/frag/pass-other/litOtherMain.js';
import litShadowMainPS from '../../../src/scene/shader-lib/wgsl/chunks/lit/frag/pass-shadow/litShadowMain.js';
import litMainVS from '../../../src/scene/shader-lib/wgsl/chunks/lit/vert/litMain.js';
import { shaderChunksWGSL } from '../../../src/scene/shader-lib/wgsl/collections/shader-chunks-wgsl.js';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

const ENGINE_SLOTS = [
    'litEngineDeclarationPS', 'litEngineDeclarationVS', 'litEngineCodePS', 'litEngineCodeVS',
    'litEngineMainStartPS', 'litEngineMainStartVS', 'litEngineMainEndPS', 'litEngineMainEndVS'
];

describe('litEngine* chunk hooks', function () {

    it('exist in the WGSL collection as empty chunks', function () {
        for (const slot of ENGINE_SLOTS) {
            expect(shaderChunksWGSL[slot], slot).to.equal('');
        }
    });

    it('are included before the matching litUser* slot at every site', function () {
        const sites = [
            [litMainVS, 'litEngineDeclarationVS', 'litUserDeclarationVS'],
            [litMainVS, 'litEngineCodeVS', 'litUserCodeVS'],
            [litMainVS, 'litEngineMainStartVS', 'litUserMainStartVS'],
            [litMainVS, 'litEngineMainEndVS', 'litUserMainEndVS'],
            [litMainPS, 'litEngineDeclarationPS', 'litUserDeclarationPS'],
            [litForwardMainPS, 'litEngineMainStartPS', 'litUserMainStartPS'],
            [litForwardMainPS, 'litEngineMainEndPS', 'litUserMainEndPS'],
            [litOtherMainPS, 'litEngineMainStartPS', 'litUserMainStartPS'],
            [litOtherMainPS, 'litEngineMainEndPS', 'litUserMainEndPS'],
            [litShadowMainPS, 'litEngineMainStartPS', 'litUserMainStartPS'],
            [litShadowMainPS, 'litEngineMainEndPS', 'litUserMainEndPS']
        ];
        for (const [source, engine, user] of sites) {
            const e = source.indexOf(`"${engine}"`);
            const u = source.indexOf(`"${user}"`);
            expect(e, engine).to.be.above(-1);
            expect(e, `${engine} before ${user}`).to.be.below(u);
        }
        // litEngineCodePS precedes litUserCodePS in each of the three pass branches
        expect((litMainPS.match(/"litEngineCodePS"/g) ?? []).length).to.equal(3);
    });
});

describe('buildMeshletLitChunks', function () {

    const textured = {
        textures: true,
        uvChannels: 2,
        tangents: true,
        familySizes: [1024, 512, 1024, 256],
        familyLevels: [{ slotLevels: 2, tailLevels: 7 }, { slotLevels: 1, tailLevels: 7 }, { slotLevels: 2, tailLevels: 7 }, { slotLevels: 0, tailLevels: 7 }]
    };

    it('interpolates every constant for the geometry-only and textured configurations', function () {
        for (const options of [{}, { textures: true, uvChannels: 1 }, textured]) {
            const chunks = buildMeshletLitChunks(options);
            for (const [name, text] of Object.entries(chunks)) {
                expect(text, `${JSON.stringify(options)} ${name}`).to.not.match(/\$\{|undefined|NaN/);
            }
        }
    });

    it('replaces the vertex buffer with page-pool vertex pulling', function () {
        const chunks = buildMeshletLitChunks();
        expect(chunks.transformCoreVS).to.include('var<private> vertex_position: vec4f;');
        expect(chunks.transformCoreVS).to.not.match(/^\s*attribute\s/m);
        expect(chunks.litEngineCodeVS).to.include('struct MeshletPageLayout');
        expect(chunks.litEngineMainStartVS).to.include('meshletDecodeDrawIndex(pcVertexIndex)');
        expect(chunks.litEngineMainStartVS).to.include('meshletPagePosition(meshletLayout, meshletVert, meshletGridOrigin, meshletGridStep)');
        expect(chunks.litEngineMainStartPS, 'no UV derivatives without UVs').to.equal(undefined);
    });

    it('reads material factors from the typed record by field name', function () {
        const chunks = buildMeshletLitChunks();
        expect(chunks.diffusePS).to.include('materialTable[vMeshletMatRow].baseColor.rgb');
        expect(chunks.opacityPS).to.include('materialTable[vMeshletMatRow].baseColor.a');
        expect(chunks.metalnessPS).to.include('materialTable[vMeshletMatRow].metallic');
        expect(chunks.glossPS).to.include('materialTable[vMeshletMatRow].roughness');
        expect(chunks.emissivePS).to.include('materialTable[vMeshletMatRow].emissive * materialTable[vMeshletMatRow].emissiveStrength');
        expect(chunks.alphaTestPS).to.include('materialTable[vMeshletMatRow].alphaCutoff');
        expect(chunks.litEngineDeclarationPS).to.include('struct MeshletMaterial');
        expect(chunks.litEngineDeclarationVS, 'both stages declare the structs').to.include('struct MeshletMaterial');
        expect(chunks.litEngineMainStartVS).to.include('objectData[meshletInstance].worldMatrix');
        expect(chunks.litEngineMainStartVS).to.include('meshletObjectGridOrigin(meshletInstance)');
        expect(chunks.diffusePS, 'no texture sampling without textures').to.not.include('meshletSampleSlot');
    });

    it('compiles texture sampling, UV derivatives and normal mapping in only when the world has them', function () {
        const chunks = buildMeshletLitChunks(textured);
        expect(chunks.diffusePS).to.include(`meshletSampleSlot(${MATERIAL_SLOT.BASE_COLOR}u`);
        expect(chunks.metalnessPS).to.include(`meshletSampleSlot(${MATERIAL_SLOT.ORM}u`);
        expect(chunks.normalMapPS).to.include(`meshletSampleSlot(${MATERIAL_SLOT.NORMAL}u`);
        expect(chunks.litEngineMainStartPS).to.include('dpdx(vMeshletUv0)');
        expect(chunks.litEngineMainStartPS).to.include('dpdx(vMeshletUv1)');
        expect(chunks.litEngineDeclarationPS).to.include('materialTable[vMeshletMatRow].slotWords[slot]');
        expect(chunks.litEngineDeclarationPS).to.include('materialTable[vMeshletMatRow].slotTransforms[slot]');
        expect(chunks.litEngineDeclarationPS).to.include('meshletTexMinLod(resident)');
        // the WGSL processor reads one resource declaration per line
        const declarations = chunks.litEngineDeclarationPS.split('\n').filter(l => /var meshlet(?:Tail|Fine)\w+: (?:texture_2d_array<f32>|sampler);/.test(l));
        expect(declarations, 'four families x (tail, sampler, fine, sampler)').to.have.lengthOf(16);
        expect(chunks.litEngineMainStartVS).to.include('meshletDecodeTangent(');
    });

    it('threads one varying per UV channel up to the slot word limit and selects per slot', function () {
        const chunks = buildMeshletLitChunks({ textures: true, uvChannels: MESHLET_MAX_UV_CHANNELS + 5 });
        for (let n = 0; n < MESHLET_MAX_UV_CHANNELS; n++) {
            expect(chunks.litEngineDeclarationVS).to.include(`varying vMeshletUv${n}: vec2f;`);
            expect(chunks.litEngineMainStartVS).to.include(`if (meshletUvFloats >= ${2 * (n + 1)}u)`);
            expect(chunks.litEngineMainEndVS).to.include(`output.vMeshletUv${n} = dMeshletUv${n};`);
            expect(chunks.litEngineMainStartPS).to.include(`dpdx(vMeshletUv${n})`);
        }
        expect(chunks.litEngineDeclarationVS, 'capped at the slot word field width').to.not.include(`vMeshletUv${MESHLET_MAX_UV_CHANNELS}`);
        expect(chunks.litEngineDeclarationPS).to.include('let slotUvChannel = (slotWord >> 24u) & 3u;');
        for (let n = 1; n < MESHLET_MAX_UV_CHANNELS; n++) {
            expect(chunks.litEngineDeclarationPS).to.include(`if (slotUvChannel == ${n}u)`);
        }
        const single = buildMeshletLitChunks({ textures: true, uvChannels: 1 });
        expect(single.litEngineDeclarationPS, 'no channel select with one channel').to.not.include('slotUvChannel');
    });

    it('routes the pick id through objectData and collapses unselected instances in the outline pass', function () {
        const chunks = buildMeshletLitChunks();
        expect(chunks.litEngineDeclarationPS).to.include('encodePickOutput(vMeshletPickId)');
        expect(chunks.litEngineMainStartVS).to.include('#ifdef PCOUTLINE_PASS');
        expect(chunks.litEngineMainStartVS).to.include('vertex_position = vec4f(meshletGridOrigin, 1.0);');
        expect(chunks.outlineOutputPS).to.include('pcOutlineColorHover');
    });
});

describe('meshlet materials', function () {

    let device;

    beforeEach(function () {
        jsdomSetup();
        device = new NullGraphicsDevice(document.createElement('canvas'));
    });

    afterEach(function () {
        device.destroy();
        device = null;
        jsdomTeardown();
    });

    it('creates one lit material per bucket differing only in culling and alpha test', function () {
        const chunks = buildMeshletLitChunks();
        const opaque = createMeshletLitMaterial(MESHLET_BUCKET_OPAQUE, chunks);
        const twoSided = createMeshletLitMaterial(MESHLET_BUCKET_OPAQUE_TWO_SIDED, chunks);
        const masked = createMeshletLitMaterial(MESHLET_BUCKET_MASKED, chunks);
        expect([opaque.name, twoSided.name, masked.name]).to.deep.equal(['MeshletLit', 'MeshletLitTwoSided', 'MeshletLitMasked']);
        expect(opaque.cull).to.equal(CULLFACE_BACK);
        expect(twoSided.cull).to.equal(CULLFACE_NONE);
        expect(masked.cull).to.equal(CULLFACE_NONE);
        expect(opaque.alphaTest, 'opaque buckets keep early-Z: no alpha test').to.equal(0);
        expect(twoSided.alphaTest).to.equal(0);
        expect(masked.alphaTest).to.be.above(0);
        expect(twoSided.twoSidedLighting).to.equal(true);
        expect(opaque.useMetalness).to.equal(true);
        [opaque, twoSided, masked].forEach(m => m.destroy());
    });

    it('creates the debug material with the colour-mode uniform and per-bucket culling', function () {
        const opaque = createMeshletMaterial(MESHLET_BUCKET_OPAQUE);
        const masked = createMeshletMaterial(MESHLET_BUCKET_MASKED);
        expect(opaque.cull).to.equal(CULLFACE_BACK);
        expect(masked.cull).to.equal(CULLFACE_NONE);
        const vertex = opaque.shaderDesc?.vertexWGSL ?? opaque._shaderDesc?.vertexWGSL ?? '';
        expect(vertex).to.include(`uniform.colorMode == ${MESHLET_COLOR_MODE.LOD_TIER}u`);
        expect(vertex).to.include(`uniform.colorMode == ${MESHLET_COLOR_MODE.MESHLET}u`);
        expect(vertex).to.include('struct MeshletPageLayout');
        [opaque, masked].forEach(m => m.destroy());
    });
});
