// Browser regression entry point. Run against a real WebGPU GraphicsDevice; checks execute
// the production shader and read its cut bits, not a CPU reimplementation of the selector.
import { createCutFixture, coverage } from './meshlet-cut-fixture.mjs';
import { Compute } from '../../../src/platform/graphics/compute.js';
import { BUFFERUSAGE_COPY_DST, BUFFERUSAGE_COPY_SRC, SHADERLANGUAGE_WGSL } from '../../../src/platform/graphics/constants.js';
import { Shader } from '../../../src/platform/graphics/shader.js';
import { StorageBuffer } from '../../../src/platform/graphics/storage-buffer.js';
import { CULL_PARAMS, CULL_PARAMS_VEC4S, MESHLET_COUNTER, MESHLET_COUNTER_U32S, MESHLET_DATA as M, MESHLET_DATA_U32S, MESHLET_NO_PARENT, OBJECT_DATA as O, OBJECT_DATA_U32S } from '../../../src/scene/meshlet/constants.js';
import { MeshletCutData } from '../../../src/scene/meshlet/meshlet-cut-data.js';
import { compactMeshletsWGSL, dispatchArgsWGSL, finalizeArgsWGSL, resetPhase2WGSL } from '../../../src/scene/meshlet/shaders/meshlet-cull-wgsl.js';
import { meshletCutWGSL } from '../../../src/scene/meshlet/shaders/meshlet-cut-wgsl.js';

export async function runMeshletCutGpuChecks(device) {
    const fixture = createCutFixture();
    const { meshlets } = fixture;
    // A second placement outside the frustum must not consume the visible one's budget.
    const objects = new Uint32Array(OBJECT_DATA_U32S * 2);
    objects.set(fixture.objects);
    objects.set(fixture.objects, OBJECT_DATA_U32S);
    objects[OBJECT_DATA_U32S + O.FIRST_PAIR_BIT] = 10;
    new Float32Array(objects.buffer)[OBJECT_DATA_U32S + 12] = 100;
    const cut = new MeshletCutData(device, meshlets, objects);
    const shader = new Shader(device, { name: 'MeshletCutRegression', shaderLanguage: SHADERLANGUAGE_WGSL, cshader: meshletCutWGSL });
    const buffers = [];
    const buffer = (data) => {
        const result = new StorageBuffer(device, Math.max(data.byteLength, 16), BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
        result.write(0, data);
        buffers.push(result);
        return result;
    };
    const params = new Float32Array(CULL_PARAMS_VEC4S * 4);
    params[CULL_PARAMS.CAMERA * 4 + 3] = 100;
    params[CULL_PARAMS.LOD * 4] = 1;
    params[CULL_PARAMS.PLANES * 4] = -1;
    params[CULL_PARAMS.PLANES * 4 + 3] = 10;
    const bindings = {
        cutGroups: cut.groups,
        cutTasks: cut.tasks,
        meshletData: buffer(meshlets),
        objectData: buffer(objects),
        cullParams: buffer(params),
        residency: buffer(new Uint32Array(10)),
        requests: buffer(new Uint32Array(10)),
        claimBits: buffer(new Uint32Array(4)),
        cutBudget: buffer(new Uint32Array(5)),
        counters: buffer(new Uint32Array(MESHLET_COUNTER_U32S))
    };
    const stages = cut.levels.map((level) => {
        const compute = new Compute(device, shader, 'CutRegression');
        for (const [name, value] of Object.entries(bindings)) compute.setParameter(name, value);
        compute.setParameter('taskStart', level.start);
        compute.setParameter('taskCount', level.count);
        compute.setParameter('rootStage', level.root ? 1 : 0);
        compute.setParameter('totalPairs', 20);
        compute.setParameter('freeBase', 22);
        compute.setupDispatch(1);
        return compute;
    });
    // No meshlets are drawn: model a completely occluded cut. Both phase finalizers must
    // preserve its capacity demand instead of reporting zero or charging it twice.
    const finalizeShader = new Shader(device, { name: 'CutDemandFinalize', shaderLanguage: SHADERLANGUAGE_WGSL, cshader: finalizeArgsWGSL });
    const resetShader = new Shader(device, { name: 'CutDemandReset', shaderLanguage: SHADERLANGUAGE_WGSL, cshader: resetPhase2WGSL });
    const finalize = new Compute(device, finalizeShader, 'CutDemandFinalize');
    finalize.setParameter('cutBudget', bindings.cutBudget);
    finalize.setParameter('counters', bindings.counters);
    finalize.setParameter('indirectDraw', buffer(new Uint32Array(15)));
    finalize.setParameter('indirectDispatch', buffer(new Uint32Array(4)));
    for (let b = 0; b < 3; b++) finalize.setParameter(`drawSlot${b}`, b);
    finalize.setParameter('dispatchSlot', 0);
    finalize.setParameter('indexCapacity0', 24);
    finalize.setParameter('indexCapacity1', 0);
    finalize.setParameter('recordCapacity', 10);
    finalize.setupDispatch(1);
    const reset = new Compute(device, resetShader, 'CutDemandReset');
    reset.setParameter('counters', bindings.counters);
    reset.setupDispatch(1);
    const compactShader = new Shader(device, { name: 'CutCompaction', shaderLanguage: SHADERLANGUAGE_WGSL, cshader: compactMeshletsWGSL });
    const compact = new Compute(device, compactShader, 'CutCompaction');
    for (const name of ['objectData', 'meshletData', 'claimBits', 'counters']) compact.setParameter(name, bindings[name]);
    compact.setParameter('selectionTopology', cut.selectionTopology);
    const selectedBuffer = buffer(new Uint32Array(20));
    compact.setParameter('selectedMeshlets', selectedBuffer);
    compact.setParameter('workItems', buffer(new Uint32Array([0, 0, 1, 0])));
    compact.setParameter('totalPairs', 20);
    compact.setParameter('freeBase', 22);
    compact.setParameter('workItemCapacity', 2);
    compact.setupDispatch(2);
    const selectedArgsShader = new Shader(device, { name: 'CutSelectedArgs', shaderLanguage: SHADERLANGUAGE_WGSL, cshader: dispatchArgsWGSL });
    const selectedArgs = new Compute(device, selectedArgsShader, 'CutSelectedArgs');
    const selectedDispatch = buffer(new Uint32Array(4));
    selectedArgs.setParameter('counters', bindings.counters);
    selectedArgs.setParameter('indirectDispatch', selectedDispatch);
    selectedArgs.setParameter('dispatchSlot', 0);
    selectedArgs.setParameter('counterIndex', MESHLET_COUNTER.SELECTED);
    selectedArgs.setParameter('groupSize', 64);
    selectedArgs.setupDispatch(1);
    const results = [];
    const scenarios = [
        { name: 'root indices cannot fit', missing: [], indices: 3, records: 10, admitted: false },
        { name: 'root records cannot fit', missing: [], indices: 24, records: 1, admitted: false },
        { name: 'root page still loading', missing: [9], indices: 24, records: 10, admitted: false },
        { name: 'all detail resident', missing: [], indices: 24, records: 10, expected: [0, 1, 2, 3] },
        { name: 'one fine page missing', missing: [0], indices: 24, records: 10, expected: [2, 3, 4, 5] },
        { name: 'shared ancestor unavailable', missing: [7], indices: 24, records: 10, expected: [4, 6, 9] },
        { name: 'root-only residency', missing: [0, 1, 2, 3, 4, 5, 6, 7], indices: 24, records: 10, expected: [8, 9] },
        { name: 'index capacity admits only roots', missing: [], indices: 6, records: 10, expected: [8, 9] },
        { name: 'record capacity admits only roots', missing: [], indices: 24, records: 2, expected: [8, 9] },
        { name: 'capacity admits one complete replacement', missing: [], indices: 9, records: 10 },
        { name: 'evicted fine page', missing: [2], indices: 24, records: 10, expected: [0, 1, 6, 7] },
        { name: 'fine page returns', missing: [], indices: 24, records: 10, expected: [0, 1, 2, 3] },
        // Root 9's group lies outside the frustum. Both middle replacements depend on it, so
        // it must still refine (without pages or charge) or the visible surface stays coarse.
        { name: 'off-screen shared ancestor', offscreen: true, missing: [5, 7], indices: 24, records: 10, expected: [0, 1, 2, 3] },
        { name: 'off-screen ancestor, visible page missing', offscreen: true, missing: [2], indices: 24, records: 10, expected: [0, 1, 6] }
    ];
    // Moves root 9's replacement group (its bounds and its children's parent bounds) off-screen.
    const offscreenMeshlets = meshlets.slice();
    const offscreenFloats = new Float32Array(offscreenMeshlets.buffer);
    for (const field of [9 * MESHLET_DATA_U32S + M.GROUP_SPHERE, 5 * MESHLET_DATA_U32S + M.PARENT_SPHERE, 7 * MESHLET_DATA_U32S + M.PARENT_SPHERE]) {
        offscreenFloats[field] = 50;
    }
    try {
        // Keep submissions sequential so page eviction/return cases reuse the same GPU state.
        await scenarios.reduce((previous, scenario) => previous.then(async () => {
            const residency = new Uint32Array(10);
            for (const page of scenario.missing) residency[page] = MESHLET_NO_PARENT;
            bindings.residency.write(0, residency);
            bindings.meshletData.write(0, scenario.offscreen ? offscreenMeshlets : meshlets);
            bindings.claimBits.clear(); bindings.requests.clear();
            const initialCounters = new Uint32Array(MESHLET_COUNTER_U32S);
            initialCounters[MESHLET_COUNTER.WORK_ITEMS] = 2;
            bindings.counters.write(0, initialCounters);
            bindings.cutBudget.write(0, new Uint32Array(5));
            for (const stage of stages) {
                stage.setParameter('indexCapacity0', scenario.indices);
                stage.setParameter('indexCapacity1', 0); stage.setParameter('indexCapacity2', 0);
                stage.setParameter('recordCapacity', scenario.records);
            }
            device.computeDispatch(stages, 'MeshletCutRegression');
            compact.setParameter('recordCapacity', scenario.records);
            selectedArgs.setParameter('workItemCapacity', scenario.records);
            device.computeDispatch([compact, selectedArgs], 'CutCompactionRegression');
            const compactCounters = new Uint32Array(MESHLET_COUNTER_U32S);
            const compactEntries = new Uint32Array(20);
            await bindings.counters.read(0, compactCounters.byteLength, compactCounters, true);
            await selectedBuffer.read(0, compactEntries.byteLength, compactEntries, true);
            const compactCount = compactCounters[MESHLET_COUNTER.SELECTED];
            if (compactCount > scenario.records) throw new Error(`${scenario.name}: compact list exceeds reserved capacity`);
            const dispatch = new Uint32Array(4);
            await selectedDispatch.read(0, 16, dispatch, true);
            if (dispatch[0] !== Math.ceil(compactCount / 64)) throw new Error(`${scenario.name}: incorrect selected dispatch count`);
            const compacted = [];
            for (let i = 0; i < compactCount; i++) {
                if (compactEntries[i * 2] !== 0) throw new Error(`${scenario.name}: compacted off-screen placement`);
                compacted.push(compactEntries[i * 2 + 1]);
            }
            compacted.sort((a, b) => a - b);
            const bits = new Uint32Array(4);
            await bindings.claimBits.read(0, 16, bits, true);
            const set = bit => (bits[bit >> 5] & (1 << (bit & 31))) !== 0;
            if (scenario.admitted === false) {
                if (compactCount !== 0) throw new Error(`${scenario.name}: compacted an unadmitted cut`);
                if (set(20)) throw new Error(`${scenario.name}: admitted incomplete roots`);
                results.push({ name: scenario.name, selected: [], indexCount: 0 });
                return;
            }
            if (!set(20)) throw new Error(`${scenario.name}: root cut was not admitted`);
            if (set(21)) throw new Error(`${scenario.name}: admitted off-screen instance`);
            const freeGroups = [8, 9].filter(g => set(22 + g));
            if (JSON.stringify(freeGroups) !== JSON.stringify(scenario.offscreen ? [9] : [])) throw new Error(`${scenario.name}: unexpected off-frustum refinements ${freeGroups}`);
            const selected = [];
            let covered = 0, indexCount = 0;
            for (let m = 0; m < 10; m++) {
                const row = m * MESHLET_DATA_U32S, parent = meshlets[row + M.PARENT], birth = meshlets[row + M.BIRTH_GROUP];
                if ((parent === MESHLET_NO_PARENT || set(parent)) && (birth === MESHLET_NO_PARENT || !set(birth))) {
                    if (covered & coverage[m]) throw new Error(`${scenario.name}: overlapping replacement`);
                    covered |= coverage[m];
                    // off-screen by construction: covered, but never compacted or resident
                    if (parent !== MESHLET_NO_PARENT && set(22 + parent)) continue;
                    selected.push(m);
                    indexCount += meshlets[row + M.TRIANGLE_COUNT] * 3;
                    if (scenario.missing.includes(m)) throw new Error(`${scenario.name}: selected unavailable geometry`);
                }
            }
            if (JSON.stringify(compacted) !== JSON.stringify(selected)) throw new Error(`${scenario.name}: compacted cut has missing or duplicate meshlets`);
            if (covered !== 255) throw new Error(`${scenario.name}: surface coverage has holes`);
            if (indexCount > scenario.indices || selected.length > scenario.records) throw new Error(`${scenario.name}: capacity overflow`);
            if (scenario.expected && JSON.stringify(selected) !== JSON.stringify(scenario.expected)) throw new Error(`${scenario.name}: unexpected cut ${selected}`);
            device.computeDispatch([finalize], 'CutDemandPhase1');
            const demand = new Uint32Array(MESHLET_COUNTER_U32S);
            await bindings.counters.read(0, demand.byteLength, demand, true);
            if (demand[MESHLET_COUNTER.DEMAND_BASE] < indexCount || demand[MESHLET_COUNTER.RECORD_DEMAND] < selected.length) {
                throw new Error(`${scenario.name}: occlusion discarded selected cut demand`);
            }
            if (scenario.name === 'all detail resident' && demand[MESHLET_COUNTER.DEMAND_BASE] !== 24) {
                throw new Error('Fully refined cut must report exactly 24 indices even when nothing is drawn');
            }
            if (scenario.offscreen && demand[MESHLET_COUNTER.DEMAND_BASE] !== indexCount) {
                throw new Error(`${scenario.name}: off-frustum refinement must charge nothing and refund its coarse members`);
            }
            device.computeDispatch([reset, finalize], 'CutDemandPhase2');
            const phase2 = new Uint32Array(MESHLET_COUNTER_U32S);
            await bindings.counters.read(0, phase2.byteLength, phase2, true);
            if (phase2[MESHLET_COUNTER.SELECTED] !== compactCount) throw new Error(`${scenario.name}: phase 2 lost the compact list`);
            if (phase2[MESHLET_COUNTER.DEMAND_BASE] !== demand[MESHLET_COUNTER.DEMAND_BASE] ||
                phase2[MESHLET_COUNTER.RECORD_DEMAND] !== demand[MESHLET_COUNTER.RECORD_DEMAND]) {
                throw new Error(`${scenario.name}: second occlusion phase changed cut demand`);
            }
            results.push({ name: scenario.name, selected, indexCount, indexDemand: demand[MESHLET_COUNTER.DEMAND_BASE] });
        }), Promise.resolve());
        return results;
    } finally {
        for (const b of buffers) b.destroy();
        cut.destroy(); shader.destroy();
        finalizeShader.destroy(); resetShader.destroy();
        compact.destroy(); compactShader.destroy(); selectedArgs.destroy(); selectedArgsShader.destroy();
    }
}
