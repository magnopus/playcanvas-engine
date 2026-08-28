import { Debug } from '../../../../core/debug.js';
import { Mat4 } from '../../../../core/math/mat4.js';
import { Quat } from '../../../../core/math/quat.js';
import { Vec3 } from '../../../../core/math/vec3.js';
import { MeshletResource, MeshletPrimitive } from '../../../../scene/meshlet/meshlet-resource.js';
import { GltfAccessor } from '../gltf-accessor.js';

// MAG_meshlets_gpu + MAG_meshlets_stream: streamed Nanite-style meshlet DAG assets
// produced by gltf-tools. The per-primitive extension carries a single resident meshletData
// accessor (32 u32 per meshlet, the exact GPU layout); the root extension carries the stream
// manifest (binary page table, shard blob URIs, uniform page size, position grid). See the
// gltf-tools specs MAG_meshlets_gpu.md / MAG_meshlets_stream.md.

const gpuExtensionName = 'MAG_meshlets_gpu';
const streamExtensionName = 'MAG_meshlets_stream';

/**
 * @import { GraphicsDevice } from '../../../../platform/graphics/graphics-device.js'
 */

/**
 * Whether a glTF primitive is a streamed meshlet primitive. Its inline geometry is only a
 * placeholder (a 3-vertex AABB proxy), so the regular mesh path must skip it.
 *
 * @param {object} primitive - The glTF primitive definition.
 * @returns {boolean} True when the primitive carries the MAG_meshlets_gpu extension.
 * @ignore
 */
const hasMeshletExtension = (primitive) => {
    return !!primitive?.extensions?.[gpuExtensionName];
};

/**
 * Computes the document-space transform of every glTF node, walking the default scene's
 * hierarchy (nodes outside it get their own subtree walked so nothing is left null).
 *
 * @param {object} gltf - The glTF document.
 * @returns {(Mat4|null)[]} One world matrix per node index.
 */
const computeNodeWorldTransforms = (gltf) => {
    const nodes = gltf.nodes ?? [];
    const world = new Array(nodes.length).fill(null);

    const local = (node) => {
        const m = new Mat4();
        if (node.matrix) {
            m.set(node.matrix);
        } else {
            const t = node.translation ?? [0, 0, 0];
            const r = node.rotation ?? [0, 0, 0, 1];
            const s = node.scale ?? [1, 1, 1];
            m.setTRS(new Vec3(t[0], t[1], t[2]), new Quat(r[0], r[1], r[2], r[3]), new Vec3(s[0], s[1], s[2]));
        }
        return m;
    };

    const visit = (index, parent) => {
        const node = nodes[index];
        const m = local(node);
        if (parent) m.mul2(parent, m);
        world[index] = m;
        node.children?.forEach(c => visit(c, m));
    };

    const sceneDef = gltf.scenes?.[gltf.scene ?? 0];
    if (sceneDef?.nodes) {
        sceneDef.nodes.forEach(n => visit(n, null));
    } else {
        nodes.forEach((n, i) => {
            if (!world[i]) visit(i, null);
        });
    }
    return world;
};

const _scatterMat = new Mat4();
const _scatterPos = new Vec3();
const _scatterRot = new Quat();
const _scatterScale = new Vec3();

/**
 * Reads a float accessor of `components`-wide elements as a flat Float32Array (the
 * EXT_mesh_gpu_instancing TRS attributes).
 *
 * @param {number|undefined} index - The accessor index, or undefined when the attribute is absent.
 * @param {object[]} accessors - The glTF accessors.
 * @param {Uint8Array[]} bufferViews - The decoded buffer views.
 * @param {number} components - Elements per entry (3 for translation / scale, 4 for rotation).
 * @returns {Float32Array|null} The flat data, or null when absent or too short.
 */
const readVecAccessor = (index, accessors, bufferViews, components) => {
    const accessor = accessors?.[index];
    if (!accessor) return null;
    const data = GltfAccessor.getData(accessor, bufferViews, true);
    if (!data || data.length < components) return null;
    return data instanceof Float32Array ? data : Float32Array.from(data);
};

/**
 * Reads a SCALAR u32 accessor as a Uint32Array view over its buffer view (no copy).
 *
 * @param {number|undefined} index - The accessor index.
 * @param {object[]} accessors - The glTF accessors.
 * @param {Uint8Array[]} bufferViews - The decoded buffer views.
 * @returns {Uint32Array|null} The data, or null when the accessor is absent.
 */
const readU32Accessor = (index, accessors, bufferViews) => {
    const accessor = accessors?.[index];
    if (!accessor) return null;
    const data = GltfAccessor.getData(accessor, bufferViews, true);
    if (!data) return null;
    return data instanceof Uint32Array ? data : new Uint32Array(data.buffer, data.byteOffset, data.length);
};

/**
 * Parses the document's streamed meshlet primitives into a single {@link MeshletResource}:
 * one entry per MAG_meshlets_gpu primitive, one instance per (node placement, primitive) -
 * expanding EXT_mesh_gpu_instancing scatters - plus the stream manifest, the optional baked
 * material table and the optional MAG_texture_streaming manifest.
 *
 * Documents with an unsupported extension version are rejected with a Debug error.
 *
 * @param {GraphicsDevice} device - The graphics device.
 * @param {object} gltf - The glTF document.
 * @param {Uint8Array[]} bufferViews - The decoded buffer views.
 * @returns {MeshletResource|null} The resource, or null when the document carries no streamed
 * meshlet primitives.
 * @ignore
 */
const createMeshlets = (device, gltf, bufferViews) => {
    const manifestDef = gltf.extensions?.[streamExtensionName];
    if (!manifestDef) {
        return null;
    }

    if (manifestDef.version !== 2) {
        Debug.error(`glTF ${streamExtensionName} version ${manifestDef.version} is not supported, expected 2. Re-bake the asset with current gltf-tools.`);
        return null;
    }

    const pageTable = readU32Accessor(manifestDef.pagesAccessor, gltf.accessors, bufferViews);
    if (!pageTable) {
        Debug.error(`glTF ${streamExtensionName} manifest has no readable page table accessor.`);
        return null;
    }

    const manifest = {
        blobs: manifestDef.blobs ?? [],
        pageTable,
        pageCount: manifestDef.pageCount ?? 0,
        rootPages: manifestDef.rootPages ?? [],
        attributeLayout: manifestDef.attributeLayout ?? {},
        pageSizeBytes: manifestDef.pageSizeBytes ?? 65536,
        pageAlignment: manifestDef.pageAlignment ?? 256,
        positionGrid: manifestDef.positionGrid ?? null
    };

    if (!manifest.positionGrid) {
        Debug.error(`glTF ${streamExtensionName} v2 manifest is missing positionGrid.`);
        return null;
    }

    const primitives = [];
    gltf.meshes?.forEach((gltfMesh, meshIndex) => {
        gltfMesh.primitives.forEach((primitive, primIndex) => {
            const ext = primitive.extensions?.[gpuExtensionName];
            if (!ext) return;

            if (ext.version !== 2 || ext.accessors?.meshletData === undefined) {
                Debug.error(`glTF ${gpuExtensionName} primitive (mesh ${meshIndex} prim ${primIndex}) has an unsupported version (${ext.version}) or no meshletData accessor, skipped.`);
                return;
            }

            const meshletData = readU32Accessor(ext.accessors.meshletData, gltf.accessors, bufferViews);
            if (!meshletData) {
                Debug.error(`glTF ${gpuExtensionName} meshletData accessor is not readable (mesh ${meshIndex} prim ${primIndex}), skipped.`);
                return;
            }

            const prim = new MeshletPrimitive();
            prim.meshletData = meshletData;
            prim.meshletDataF32 = new Float32Array(meshletData.buffer, meshletData.byteOffset, meshletData.length);
            prim.meshletCount = ext.meshletCount ?? 0;
            prim.uvChannelMask = ext.uvChannelMask ?? 0;
            prim.vertexCount = ext.vertexCount ?? 0;
            prim.lods = ext.lods ?? [];
            prim.aabbCenter = ext.aabbCenter ?? [0, 0, 0];
            prim.aabbHalfExtents = ext.aabbHalfExtents ?? [0, 0, 0];
            prim.materialIndex = primitive.material ?? -1;
            const materialDef = gltf.materials?.[primitive.material];
            prim.baseColorFactor = materialDef?.pbrMetallicRoughness?.baseColorFactor ?? [1, 1, 1, 1];
            prim.meshIndex = meshIndex;
            prim.primIndex = primIndex;
            primitives.push(prim);
        });
    });

    if (!primitives.length) {
        return null;
    }

    // one instance per (node placement, meshlet primitive)
    const nodeWorld = computeNodeWorldTransforms(gltf);
    const primsByMesh = new Map();
    primitives.forEach((p, i) => {
        let list = primsByMesh.get(p.meshIndex);
        if (!list) {
            list = []; primsByMesh.set(p.meshIndex, list);
        }
        list.push(i);
    });
    const instances = [];
    gltf.nodes?.forEach((node, nodeIndex) => {
        if (node.mesh === undefined || !nodeWorld[nodeIndex]) return;
        const list = primsByMesh.get(node.mesh);
        if (!list) return;

        // EXT_mesh_gpu_instancing: the node carries a scatter of TRS attributes, each one a
        // placement of its mesh in the node's space. Scattered foliage arrives this way, and
        // one meshlet instance per scatter entry is exactly the world's instance model - the
        // whole scatter shares one set of pages.
        const scatter = node.extensions?.EXT_mesh_gpu_instancing?.attributes;
        if (scatter) {
            const t = readVecAccessor(scatter.TRANSLATION, gltf.accessors, bufferViews, 3);
            const r = readVecAccessor(scatter.ROTATION, gltf.accessors, bufferViews, 4);
            const s = readVecAccessor(scatter.SCALE, gltf.accessors, bufferViews, 3);
            const count = Math.max(
                t ? t.length / 3 : 0, r ? r.length / 4 : 0, s ? s.length / 3 : 0
            );
            for (let i = 0; i < count; i++) {
                _scatterPos.set(t ? t[i * 3] : 0, t ? t[i * 3 + 1] : 0, t ? t[i * 3 + 2] : 0);
                _scatterRot.set(
                    r ? r[i * 4] : 0, r ? r[i * 4 + 1] : 0,
                    r ? r[i * 4 + 2] : 0, r ? r[i * 4 + 3] : 1
                );
                _scatterScale.set(s ? s[i * 3] : 1, s ? s[i * 3 + 1] : 1, s ? s[i * 3 + 2] : 1);
                _scatterMat.setTRS(_scatterPos, _scatterRot, _scatterScale);
                _scatterMat.mul2(nodeWorld[nodeIndex], _scatterMat);
                for (const primIndex of list) {
                    instances.push({ primIndex, matrix: new Float32Array(_scatterMat.data) });
                }
            }
            if (count > 0) return;
        }

        for (const primIndex of list) {
            instances.push({ primIndex, matrix: new Float32Array(nodeWorld[nodeIndex].data) });
        }
    });

    const resource = new MeshletResource(device, primitives, manifest, instances);

    // streamed-texture manifest (MAG_texture_streaming root extension) - consumed by the
    // meshlet world's texture system; absent on texture-less bakes
    const textureDef = gltf.extensions?.MAG_texture_streaming;
    if (textureDef && Array.isArray(textureDef.arrays)) {
        resource.textureManifest = { arrays: textureDef.arrays };
    }

    // baked material table (root MAG_meshlets_gpu block, optional):
    // MATERIAL_RECORD_U32S words per glTF material, uploaded verbatim by the world
    const rootGpuDef = gltf.extensions?.[gpuExtensionName];
    if (rootGpuDef?.materialTable !== undefined) {
        const table = readU32Accessor(rootGpuDef.materialTable, gltf.accessors, bufferViews);
        const count = rootGpuDef.materialCount ?? 0;
        if (table && table.length === count * 32) {
            resource.materialTable = table;
            resource.materialCount = count;
        } else {
            Debug.error(`glTF ${gpuExtensionName} root material table is unreadable or mis-sized (${table?.length} words for ${count} materials), ignored.`);
        }
    }

    return resource;
};

export { createMeshlets, hasMeshletExtension };
