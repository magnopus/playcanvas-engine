import { expect } from 'chai';

import { createMeshlets, hasMeshletExtension } from '../../../src/framework/parsers/glb/extensions/mag-meshlets.js';
import { GlbParser } from '../../../src/framework/parsers/glb-parser.js';
import { MATERIAL_RECORD_U32S, MESHLET_DATA_U32S, PAGE_TABLE_FIELDS } from '../../../src/scene/meshlet/constants.js';
import { MeshletResource } from '../../../src/scene/meshlet/meshlet-resource.js';
import { createApp } from '../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

const U32 = 5125;
const F32 = 5126;

const PAGE_COUNT = 2;
const MESHLET_COUNT = 3;
const SCATTER = [[10, 0, 0], [0, 20, 0]];

// Builds a minimal streamed-meshlet document: one primitive whose inline geometry is only a
// 3-vertex placeholder, a v2 stream manifest, a baked one-material table, a texture manifest,
// one plain node placement and one EXT_mesh_gpu_instancing scatter of the same mesh.
const buildAsset = ({ version = 2, materialCount = 1, textures = true } = {}) => {
    const blocks = [];
    const views = [];
    let offset = 0;
    const push = (typed) => {
        const bytes = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
        views.push({ buffer: 0, byteOffset: offset, byteLength: bytes.byteLength });
        blocks.push(bytes);
        offset += bytes.byteLength;
        return views.length - 1;
    };

    const pageTable = new Uint32Array(PAGE_COUNT * PAGE_TABLE_FIELDS).map((_, i) => i);
    const meshletData = new Uint32Array(MESHLET_COUNT * MESHLET_DATA_U32S).map((_, i) => i * 7);
    const materialTable = new Uint32Array(MATERIAL_RECORD_U32S).map((_, i) => 100 + i);
    const translation = new Float32Array(SCATTER.flat());
    const rotation = new Float32Array(SCATTER.flatMap(() => [0, 0, 0, 1]));
    const scale = new Float32Array(SCATTER.flatMap(() => [1, 1, 1]));
    const position = new Float32Array(9);

    const accessors = [];
    const accessor = (typed, componentType, type, components) => {
        accessors.push({ bufferView: push(typed), componentType, type, count: typed.length / components });
        return accessors.length - 1;
    };
    const pagesAccessor = accessor(pageTable, U32, 'SCALAR', 1);
    const meshletAccessor = accessor(meshletData, U32, 'SCALAR', 1);
    const tableAccessor = accessor(materialTable, U32, 'SCALAR', 1);
    const tAccessor = accessor(translation, F32, 'VEC3', 3);
    const rAccessor = accessor(rotation, F32, 'VEC4', 4);
    const sAccessor = accessor(scale, F32, 'VEC3', 3);
    const positionAccessor = accessor(position, F32, 'VEC3', 3);

    const bytes = new Uint8Array(offset);
    blocks.forEach((b, i) => bytes.set(b, views[i].byteOffset));

    const gltf = {
        asset: { version: '2.0' },
        extensionsUsed: ['MAG_meshlets_gpu', 'MAG_meshlets_stream'],
        buffers: [{ byteLength: bytes.byteLength, uri: `data:application/octet-stream;base64,${Buffer.from(bytes).toString('base64')}` }],
        bufferViews: views,
        accessors,
        materials: [{ pbrMetallicRoughness: { baseColorFactor: [0.5, 0.25, 1, 1] } }],
        meshes: [{
            primitives: [{
                attributes: { POSITION: positionAccessor },
                material: 0,
                extensions: {
                    MAG_meshlets_gpu: {
                        version,
                        accessors: { meshletData: meshletAccessor },
                        meshletCount: MESHLET_COUNT,
                        vertexCount: 12,
                        uvChannelMask: 0,
                        lods: [],
                        aabbCenter: [0, 0, 0],
                        aabbHalfExtents: [1, 1, 1]
                    }
                }
            }]
        }],
        nodes: [
            { mesh: 0, translation: [1, 2, 3] },
            { mesh: 0, extensions: { EXT_mesh_gpu_instancing: { attributes: { TRANSLATION: tAccessor, ROTATION: rAccessor, SCALE: sAccessor } } } }
        ],
        scenes: [{ nodes: [0, 1] }],
        scene: 0,
        extensions: {
            MAG_meshlets_stream: {
                version,
                pagesAccessor,
                pageCount: PAGE_COUNT,
                blobs: [{ uri: 'asset_meshlets/pages_roots.dat', byteLength: 65536 }],
                rootPages: [0],
                attributeLayout: { normals: true },
                pageSizeBytes: 65536,
                pageAlignment: 256,
                positionGrid: { origin: [0, 0, 0], step: 0.01, bits: 16 }
            },
            MAG_meshlets_gpu: { version, materialCount, materialTable: tableAccessor }
        }
    };
    if (textures) {
        gltf.extensions.MAG_texture_streaming = { arrays: [{ name: 'srgb_0', width: 256, height: 256 }] };
    }

    const bufferViews = views.map(v => new Uint8Array(bytes.buffer, v.byteOffset, v.byteLength));
    return { gltf, bufferViews };
};

// Debug.error is how the parser reports a rejected asset; keep it out of the test output
const silenced = (fn) => {
    const original = console.error;
    const messages = [];
    console.error = (...args) => messages.push(args.join(' '));
    try {
        return { result: fn(), messages };
    } finally {
        console.error = original;
    }
};

const translationOf = matrix => [matrix[12], matrix[13], matrix[14]];

describe('MAG_meshlets extensions', function () {

    describe('createMeshlets', function () {

        it('parses a v2 document into one resource with its manifest, table and placements', function () {
            const { gltf, bufferViews } = buildAsset();
            const resource = createMeshlets(null, gltf, bufferViews);

            expect(resource).to.be.an.instanceof(MeshletResource);
            expect(resource.primitives).to.have.lengthOf(1);
            const prim = resource.primitives[0];
            expect(prim.meshletCount).to.equal(MESHLET_COUNT);
            expect(prim.meshletData).to.have.lengthOf(MESHLET_COUNT * MESHLET_DATA_U32S);
            expect(prim.meshletData[1]).to.equal(7);
            expect(prim.meshletDataF32.buffer).to.equal(prim.meshletData.buffer);
            expect(prim.materialIndex).to.equal(0);
            expect(prim.baseColorFactor).to.deep.equal([0.5, 0.25, 1, 1]);
            expect(prim.meshIndex).to.equal(0);
            expect(prim.primIndex).to.equal(0);
            expect(resource.totalMeshlets).to.equal(MESHLET_COUNT);

            const manifest = resource.manifest;
            expect(manifest.pageCount).to.equal(PAGE_COUNT);
            expect(manifest.pageTable).to.have.lengthOf(PAGE_COUNT * PAGE_TABLE_FIELDS);
            expect(manifest.pageTable[5]).to.equal(5);
            expect(manifest.rootPages).to.deep.equal([0]);
            expect(manifest.blobs[0].uri).to.equal('asset_meshlets/pages_roots.dat');
            expect(manifest.pageSizeBytes).to.equal(65536);
            expect(manifest.positionGrid.step).to.equal(0.01);

            expect(resource.materialTable).to.have.lengthOf(MATERIAL_RECORD_U32S);
            expect(resource.materialTable[0]).to.equal(100);
            expect(resource.materialCount).to.equal(1);
            expect(resource.textureManifest.arrays).to.have.lengthOf(1);
        });

        it('creates one placement per node and one per EXT_mesh_gpu_instancing scatter entry', function () {
            const { gltf, bufferViews } = buildAsset();
            const resource = createMeshlets(null, gltf, bufferViews);

            expect(resource.instances).to.have.lengthOf(1 + SCATTER.length);
            resource.instances.forEach(inst => expect(inst.primIndex).to.equal(0));
            expect(translationOf(resource.instances[0].matrix)).to.deep.equal([1, 2, 3]);
            expect(translationOf(resource.instances[1].matrix)).to.deep.equal(SCATTER[0]);
            expect(translationOf(resource.instances[2].matrix)).to.deep.equal(SCATTER[1]);
        });

        it('rejects an unsupported manifest version', function () {
            const { gltf, bufferViews } = buildAsset({ version: 7 });
            const { result, messages } = silenced(() => createMeshlets(null, gltf, bufferViews));
            expect(result).to.equal(null);
            expect(messages.join('\n')).to.match(/version 7 is not supported/);
        });

        it('ignores a mis-sized material table', function () {
            const { gltf, bufferViews } = buildAsset({ materialCount: 2 });
            const { result, messages } = silenced(() => createMeshlets(null, gltf, bufferViews));
            expect(result).to.be.an.instanceof(MeshletResource);
            expect(result.materialTable).to.equal(null);
            expect(result.materialCount).to.equal(0);
            expect(messages.join('\n')).to.match(/mis-sized/);
        });

        it('leaves textureManifest null on texture-less bakes', function () {
            const { gltf, bufferViews } = buildAsset({ textures: false });
            expect(createMeshlets(null, gltf, bufferViews).textureManifest).to.equal(null);
        });

        it('returns null for documents without the stream manifest or without meshlet primitives', function () {
            const plain = buildAsset();
            delete plain.gltf.extensions.MAG_meshlets_stream;
            expect(createMeshlets(null, plain.gltf, plain.bufferViews)).to.equal(null);

            const noPrims = buildAsset();
            delete noPrims.gltf.meshes[0].primitives[0].extensions;
            expect(createMeshlets(null, noPrims.gltf, noPrims.bufferViews)).to.equal(null);
        });
    });

    describe('hasMeshletExtension', function () {

        it('detects the per-primitive extension', function () {
            const { gltf } = buildAsset();
            expect(hasMeshletExtension(gltf.meshes[0].primitives[0])).to.equal(true);
            expect(hasMeshletExtension({ attributes: {} })).to.equal(false);
            expect(hasMeshletExtension(undefined)).to.equal(false);
        });
    });

    describe('GlbParser integration', function () {

        let app;

        beforeEach(function () {
            jsdomSetup();
            app = createApp();
        });

        afterEach(function () {
            app?.destroy();
            app = null;
            jsdomTeardown();
        });

        it('hands the placeholder primitive to the meshlet path instead of the mesh path', function () {
            const { gltf } = buildAsset();
            const data = new TextEncoder().encode(JSON.stringify(gltf));

            return new Promise((resolve, reject) => {
                GlbParser.parse('meshlets.gltf', '', '', data, app.graphicsDevice, app.assets, {}, (err, result) => {
                    if (err) {
                        reject(err);
                        return;
                    }
                    try {
                        expect(result.meshlets).to.be.an.instanceof(MeshletResource);
                        expect(result.meshlets.instances).to.have.lengthOf(1 + SCATTER.length);
                        // the primitive's inline geometry is a placeholder and must not become a mesh
                        expect(result.renders).to.have.lengthOf(1);
                        expect(result.renders[0].meshes).to.deep.equal([]);
                        resolve();
                    } catch (e) {
                        reject(e);
                    }
                });
            });
        });
    });
});
