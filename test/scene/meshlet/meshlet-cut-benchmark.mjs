// Manual browser benchmark for the large root sets common in poorly simplifying foliage.
// Call with rendering paused. Timings are GPU timestamps, excluding setup and readback.
import { Compute } from '../../../src/platform/graphics/compute.js';
import { BUFFERUSAGE_COPY_DST, SHADERLANGUAGE_WGSL } from '../../../src/platform/graphics/constants.js';
import { Shader } from '../../../src/platform/graphics/shader.js';
import { StorageBuffer } from '../../../src/platform/graphics/storage-buffer.js';
import { WebgpuGpuProfiler } from '../../../src/platform/graphics/webgpu/webgpu-gpu-profiler.js';
import {
    CULL_PARAMS_VEC4S, MESHLET_COUNTER_U32S, MESHLET_DATA as M, MESHLET_DATA_U32S,
    MESHLET_NO_PARENT, OBJECT_DATA as O, OBJECT_DATA_U32S
} from '../../../src/scene/meshlet/constants.js';
import { MeshletCutData } from '../../../src/scene/meshlet/meshlet-cut-data.js';
import { meshletCutWGSL } from '../../../src/scene/meshlet/shaders/meshlet-cut-wgsl.js';

export async function benchmarkMeshletRoots(device, CutData = MeshletCutData, shaderSource = meshletCutWGSL, legacy = false) {
    if (!device.supportsTimestampQuery) throw new Error('GPU timestamp queries are required.');
    const roots = 65536, instances = 64, pageCount = roots / 64;
    const meshlets = new Uint32Array(roots * MESHLET_DATA_U32S);
    for (let m = 0; m < roots; m++) {
        meshlets[m * MESHLET_DATA_U32S + M.PARENT] = MESHLET_NO_PARENT;
        meshlets[m * MESHLET_DATA_U32S + M.PAGE] = Math.floor(m / 64);
        meshlets[m * MESHLET_DATA_U32S + M.TRIANGLE_COUNT] = 1;
    }
    const objects = new Uint32Array(instances * OBJECT_DATA_U32S);
    const floats = new Float32Array(objects.buffer);
    for (let i = 0; i < instances; i++) {
        const row = i * OBJECT_DATA_U32S;
        floats[row] = floats[row + 5] = floats[row + 10] = floats[row + 15] = 1;
        floats[row + O.SPHERE + 3] = floats[row + O.MAX_SCALE] = 1;
        objects[row + O.MESHLET_COUNT] = roots;
        objects[row + O.FIRST_PAIR_BIT] = i * roots;
    }
    const cut = new CutData(device, meshlets, objects);
    const buffers = [];
    const buffer = (data) => {
        const b = new StorageBuffer(device, Math.max(data.byteLength, 16), BUFFERUSAGE_COPY_DST);
        b.write(0, data);
        buffers.push(b);
        return b;
    };
    const shader = new Shader(device, { name: 'MeshletRootBenchmark', shaderLanguage: SHADERLANGUAGE_WGSL, cshader: shaderSource });
    const compute = new Compute(device, shader, 'MeshletRootBenchmark');
    const bindings = {
        cutGroups: cut.groups,
        cutTasks: cut.tasks,
        meshletData: buffer(meshlets),
        objectData: buffer(objects),
        cullParams: buffer(new Float32Array(CULL_PARAMS_VEC4S * 4)),
        residency: buffer(new Uint32Array(pageCount)),
        requests: buffer(new Uint32Array(pageCount)),
        claimBits: buffer(new Uint32Array(Math.ceil((roots * instances + instances) / 32))),
        cutBudget: buffer(new Uint32Array(5)),
        counters: buffer(new Uint32Array(MESHLET_COUNTER_U32S))
    };
    for (const [name, value] of Object.entries(bindings)) compute.setParameter(name, value);
    for (const [name, value] of Object.entries({
        taskStart: 0,
        taskCount: instances,
        rootStage: 1,
        totalPairs: roots * instances,
        recordCapacity: roots * instances,
        indexCapacity0: roots * instances * 3,
        indexCapacity1: 0,
        indexCapacity2: 0
    })) compute.setParameter(name, value);
    compute.setupDispatch(1);
    const savedProfiler = device.gpuProfiler;
    const restoreProfiler = () => {
        device.gpuProfiler = savedProfiler;
    };
    const profiler = new WebgpuGpuProfiler(device);
    profiler.enabled = true;
    device.gpuProfiler = profiler;
    const timings = [];
    try {
        await Array.from({ length: 6 }).reduce((previous, unused, sample) => previous.then(async () => {
            bindings.claimBits.clear();
            bindings.requests.clear();
            bindings.cutBudget.write(0, new Uint32Array(legacy ? [roots * instances * 3, 0, 0, roots * instances, 0] : 5));
            profiler.frameAllocations.length = 0;
            profiler.frameStart();
            device.computeDispatch([compute], 'MeshletRootBenchmark');
            profiler.frameEnd();
            device.submit();
            const result = await profiler.timestampQueriesSet.request(profiler.slotCount, sample);
            if (sample > 0) timings.push(result.timings[0]);
        }), Promise.resolve());
        return { roots, instances, pageCount, milliseconds: timings };
    } finally {
        restoreProfiler();
        profiler.destroy();
        compute.destroy(); shader.destroy(); cut.destroy();
        for (const b of buffers) b.destroy();
    }
}
