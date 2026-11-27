import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import { ShaderMaterial } from '../../../src/scene/materials/shader-material.js';
import {
    MATERIAL_RECORD, MATERIAL_RECORD_U32S, MESHLET_BUCKET_COUNT, MESHLET_DATA, MESHLET_DATA_U32S, MESHLET_NO_PARENT,
    PAGE_TABLE, PAGE_TABLE_FIELDS
} from '../../../src/scene/meshlet/constants.js';
import { MeshletPrimitive, MeshletResource } from '../../../src/scene/meshlet/meshlet-resource.js';

/** Bytes per page of the {@link makeMeshletResource} resources. */
export const TEST_PAGE_BYTES = 256;

/** Meshlets of the {@link makeMeshletResource} primitive. */
export const TEST_MESHLETS = 3;

/** Triangles of each {@link makeMeshletResource} meshlet. */
export const TEST_TRIANGLES = 10;

/**
 * Dresses a device up as compute-capable WebGPU: compute instances keep their parameters but
 * never execute, storage buffers allocate nothing, and the buffer copies and clears asked of the
 * command encoder are recorded.
 *
 * @param {object} device - The device, usually a NullGraphicsDevice.
 * @returns {{ copies: Array<{ src: *, dst: *, size: number }>, clears: number[] }} The log of
 * what the command encoder was asked to do.
 */
export const dressComputeDevice = (device) => {
    const log = { copies: [], clears: [] };
    device.isWebGPU = true;
    device.supportsCompute = true;
    device.createComputeImpl = () => ({ destroy() {} });
    device.createBufferImpl = () => ({
        allocate() {},
        write() {},
        destroy() {},
        loseContext() {},
        buffer: {},
        clear: (dev, offset, size) => log.clears.push(size)
    });
    device.getCommandEncoder = () => ({
        copyBufferToBuffer: (src, srcOffset, dst, dstOffset, size) => log.copies.push({ src, dst, size }),
        clearBuffer: (buffer, offset, size) => log.clears.push(size)
    });
    device.computeDispatch = () => {};
    device.getIndirectDrawSlot = () => 0;
    device.getIndirectDispatchSlot = () => 0;
    return log;
};

/**
 * A null device dressed up as compute-capable WebGPU (see {@link dressComputeDevice}). Needs a
 * DOM (jsdomSetup) for its canvas.
 *
 * @returns {{ device: NullGraphicsDevice, log: { copies: Array<object>, clears: number[] } }} The
 * device and its command log.
 */
export const makeComputeDevice = () => {
    const device = new NullGraphicsDevice(document.createElement('canvas'));
    const log = dressComputeDevice(device);
    return { device, log };
};

/**
 * Bucket materials, one per MESHLET_BUCKET_*.
 *
 * @returns {ShaderMaterial[]} The materials.
 */
export const makeBucketMaterials = () => Array.from({ length: MESHLET_BUCKET_COUNT }, () => new ShaderMaterial());

/**
 * A stand-in for a finalized MeshletWorld with everything a MeshletView reads: 1000
 * instance-meshlet pairs over 7 instances in one cell, buffers as plain objects.
 *
 * @param {object} [overrides] - Fields to replace.
 * @returns {object} The world.
 */
export const makeViewWorld = (overrides = {}) => {
    const lit = makeBucketMaterials();
    return {
        finalized: true,
        totalPairs: 1000,
        rootListItems: 7,
        cut: { stageLevels: [] },
        cells: { cellCount: 1, boundsBuffer: {}, instancesBuffer: {} },
        recordCapacity: 4000,
        initialRecords: 100,
        instanceCount: 7,
        pageSizeBytes: 65536,
        totalPages: 3,
        indexCapacity: [300, 60, 30],
        indexWorst: [3000, 600, 300],
        indexCeiling: 100000,
        maxIndices: 0,
        dummyBits: { destroy() {} },
        objectDataBuffer: {},
        meshletDataBuffer: {},
        residencyBuffer: {},
        requestsBuffer: {},
        pagePool: {},
        litMaterials: lit,
        bucketMaterials: lit,
        ...overrides
    };
};

/**
 * A meshlet resource of one primitive: {@link TEST_MESHLETS} root meshlets of
 * {@link TEST_TRIANGLES} triangles over two pages, placed `instances` times 10 units apart
 * along x.
 *
 * @param {object} device - The graphics device.
 * @param {object} [options] - Options.
 * @param {number} [options.instances] - Placements. Defaults to 2.
 * @param {number} [options.scale] - Uniform scale of every placement. Defaults to 1.
 * @param {number|null} [options.materialFlags] - Flags of a baked material record, or null for
 * a bake without a material table.
 * @returns {MeshletResource} The resource.
 */
export const makeMeshletResource = (device, { instances = 2, scale = 1, materialFlags = null } = {}) => {
    const prim = new MeshletPrimitive();
    prim.meshletData = new Uint32Array(TEST_MESHLETS * MESHLET_DATA_U32S);
    for (let m = 0; m < TEST_MESHLETS; m++) {
        prim.meshletData[m * MESHLET_DATA_U32S + MESHLET_DATA.PARENT] = MESHLET_NO_PARENT;
        prim.meshletData[m * MESHLET_DATA_U32S + MESHLET_DATA.TRIANGLE_COUNT] = TEST_TRIANGLES;
        prim.meshletData[m * MESHLET_DATA_U32S + MESHLET_DATA.PAGE] = m % 2;
    }
    prim.meshletDataF32 = new Float32Array(prim.meshletData.buffer);
    prim.meshletCount = TEST_MESHLETS;
    prim.uvChannelMask = 1;
    prim.aabbCenter = [1, 2, 3];
    prim.aabbHalfExtents = [1, 1, 1];
    prim.materialIndex = 0;
    prim.baseColorFactor = [0.5, 0.25, 1, 1];
    prim.meshIndex = 0;
    prim.primIndex = 0;

    const pageTable = new Uint32Array(2 * PAGE_TABLE_FIELDS);
    pageTable[1 * PAGE_TABLE_FIELDS + PAGE_TABLE.OFFSET_LO] = TEST_PAGE_BYTES;
    const manifest = {
        blobs: [{ uri: 'pages_roots.dat', byteLength: 2 * TEST_PAGE_BYTES }],
        pageTable,
        pageCount: 2,
        rootPages: [0],
        attributeLayout: { uvComponents: 1, tangents: false },
        pageSizeBytes: TEST_PAGE_BYTES,
        pageAlignment: 256,
        positionGrid: { origin: [0.5, 0, 0], step: 0.01, bits: 16 }
    };
    const placements = Array.from({ length: instances }, (_, i) => {
        const matrix = new Float32Array(16);
        matrix[0] = scale; matrix[5] = scale; matrix[10] = scale; matrix[15] = 1;
        matrix[12] = i * 10;
        return { primIndex: 0, matrix };
    });
    const resource = new MeshletResource([prim], manifest, placements);
    if (materialFlags !== null) {
        resource.materialTable = new Uint32Array(MATERIAL_RECORD_U32S);
        resource.materialTable[MATERIAL_RECORD.FLAGS] = materialFlags;
        resource.materialCount = 1;
    }
    return resource;
};
