// @config
//
// The Caldera points of interest and the simplified terrain, each baked to its own streamed meshlet
// asset and loaded into one meshlet world. Collision soups are ordinary glTF meshes, loaded on
// demand and drawn as translucent overlays. See assets/meshlets/README.md for building the assets.
//
// @flag WEBGL_DISABLED
// @flag WEBGPU_BARE_DISABLED
// @flag HIDDEN

import {
    AppBase,
    AppOptions,
    Asset,
    BLEND_NORMAL,
    basisInitialize,
    Picker,
    ByteCache,
    CameraComponentSystem,
    CameraFrame,
    Color,
    ContainerHandler,
    Entity,
    FILLMODE_FILL_WINDOW,
    Keyboard,
    LightComponentSystem,
    MeshletComponentSystem,
    OpfsByteStore,
    Mouse,
    RenderComponentSystem,
    RESOLUTION_AUTO,
    SHADERPASS_WORLDNORMAL,
    ScriptComponentSystem,
    StandardMaterial,
    TextureHandler,
    TouchDevice,
    Vec3,
    createGraphicsDevice
} from 'playcanvas';
import { CameraControls } from 'playcanvas/scripts/esm/camera-controls.mjs';

import { data, deviceType } from 'examples/context';

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('application-canvas'));
window.focus();

// KTX2 textures in the bake are Basis-compressed, so the Basis transcoder must be ready before load
basisInitialize({
    glueUrl: './assets/wasm/basis/basis.wasm.js',
    wasmUrl: './assets/wasm/basis/basis.wasm.wasm',
    fallbackUrl: './assets/wasm/basis/basis.js'
});

// ?base=<dir>/ loads a different bake folder (e.g. a single-region test bake)
const BASE = new URLSearchParams(location.search).get('base') ?? './assets/meshlets/caldera/';

/**
 * @typedef {{ name: string, label: string, url: string, baked: boolean, collision?: string,
 * clutter?: string, clutterAll?: string, min: number[], max: number[] }} Region
 */
/** @type {{ regions: Region[] }} */
const manifest = await fetch(`${BASE}caldera_scene.json`).then(r => r.json());
// ?regions=tile_p,docks loads only those; ?pos=x,y,z&look=x,y,z starts the camera there (glTF metres)
const params = new URLSearchParams(location.search);
const only = params.get('regions')?.split(',');
const regions = manifest.regions.filter(r => r.baked && (!only || only.includes(r.name)));

// reverse-z is on by default (?reverseZ=0 turns it off) and renders with a reversed depth buffer (near = 1, far = 0) for float precision across
// the island's view distance; WebGPU only
const device = await createGraphicsDevice(canvas, {
    deviceTypes: [deviceType],
    antialias: false,
    reverseZ: params.get('reverseZ') !== '0'
});
device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);

const createOptions = new AppOptions();
createOptions.graphicsDevice = device;
createOptions.mouse = new Mouse(document.body);
createOptions.touch = new TouchDevice(document.body);
createOptions.keyboard = new Keyboard(window);
createOptions.componentSystems = [
    CameraComponentSystem,
    LightComponentSystem,
    RenderComponentSystem,
    ScriptComponentSystem,
    MeshletComponentSystem
];
createOptions.resourceHandlers = [ContainerHandler, TextureHandler];

const app = new AppBase(canvas);
app.init(createOptions);

app.setCanvasFillMode(FILLMODE_FILL_WINDOW);
app.setCanvasResolution(RESOLUTION_AUTO);

// ?cache=1 keeps fetched bytes (containers, meshlet pages, texture tails) in an OPFS cache, so a
// reload reads them from disk instead of the network. The whole island fetches ~2.2 GB, past the
// default 2 GiB cap, which would evict on every load.
if (params.get('cache') === '1') {
    const byteCache = new ByteCache(new OpfsByteStore('caldera'), { maxBytes: 4 * 1024 * 1024 * 1024 });
    app.loader.byteCache = byteCache;
    app.systems.meshlet.byteCache = byteCache;
}

const resize = () => app.resizeCanvas();
window.addEventListener('resize', resize);
app.on('destroy', () => {
    window.removeEventListener('resize', resize);
});

/**
 * @param {string} name - Asset name.
 * @param {string} url - Asset URL.
 * @returns {Promise<Asset>} The loaded container asset.
 */
const loadContainer = (name, url) => new Promise((resolve, reject) => {
    const asset = new Asset(name, 'container', { url });
    asset.once('load', () => resolve(asset));
    asset.once('error', reject);
    app.assets.add(asset);
    app.assets.load(asset);
});

// every region shares one meshlet world, so the pool covers the sum of their resident sets
// ?pool=<MiB>: the whole island needs ~4096; the default suits a handful of regions
app.systems.meshlet.poolBytes = (Number(params.get('pool')) || 2048) * 1024 * 1024;
// ?worldMeshlets=<millions>: split the content across worlds of at most this many meshlets
if (params.get('worldMeshlets')) app.systems.meshlet.maxWorldMeshlets = Number(params.get('worldMeshlets')) * 1e6;
app.systems.meshlet.occlusion = true;
// ?cssPixels=0: measure the LOD threshold in render target pixels (finer on high-density displays)
app.systems.meshlet.cssPixels = params.get('cssPixels') !== '0';

const scene = new Entity('Caldera');
app.root.addChild(scene);

// Every region's component exists up front and loads its own asset, so the meshlet system can
// see which are still loading: the first region builds at once, the rest wait for the burst to
// finish and join in a single rebuild (?progressive=1 instead rebuilds as regions arrive)
app.systems.meshlet.deferRebuildWhileLoading = params.get('progressive') !== '1';
// ?clutter=1 also loads the terrain tiles' baked trees (pipeline.sh CLUTTER=trees), ?clutter=all
// their grass, plants and rocks too (CLUTTER=all), where present
const clutter = params.get('clutter');
for (const region of regions) {
    const extra = clutter === 'all' ? region.clutterAll ?? region.clutter : clutter === '1' ? region.clutter : undefined;
    const urls = extra ? [region.url, extra] : [region.url];
    for (const url of urls) {
        const asset = new Asset(region.label, 'container', { url: BASE + url });
        asset.once('error', err => console.error(`Caldera: failed to load ${url}`, err));
        app.assets.add(asset);
        const entity = new Entity(region.label);
        entity.addComponent('meshlet', { asset });
        scene.addChild(entity);
    }
}

const light = new Entity('Sun');
light.addComponent('light', {
    type: 'directional',
    color: new Color(1, 0.96, 0.9),
    intensity: 1.6
});
light.setEulerAngles(50, 35, 0);
app.root.addChild(light);

// ?shadows=1: the sun casts four cascaded shadows over the whole island, meshlets included.
// ?cascadeIntervals=1,1,2,2 (the default) refreshes the two far cascades on alternate frames,
// which keeps frame times even; 1,1,2,4 is cheaper on average but lumpier; 1,1,1,1 refreshes
// every cascade every frame
if (params.get('shadows') === '1') {
    app.systems.meshlet.shadows = true;
    light.light.castShadows = true;
    light.light.numCascades = 4;
    light.light.shadowResolution = 2048;
    light.light.shadowDistance = 1500;
    const intervals = (params.get('cascadeIntervals') ?? '1,1,2,2').split(',').map(Number);
    light.light.cascadeUpdateIntervals = intervals.some(v => v > 1) ? intervals : null;
    // ?shadowLod=<multiplier on the LOD error threshold for casters> (4) and
    // ?shadowRootMin=<screen px below which an instance casts no meshlet shadow> (2)
    const group = app.systems.meshlet.group;
    if (group && params.get('shadowLod')) group.shadowThresholdScale = Number(params.get('shadowLod'));
    if (group && params.get('shadowRootMin')) group.shadowRootMinPixels = Number(params.get('shadowRootMin'));
}

app.scene.ambientLight = new Color(0.32, 0.36, 0.42);

const camera = new Entity('Camera');
camera.addComponent('camera', {
    clearColor: new Color(0.55, 0.68, 0.82),
    nearClip: 0.5,
    farClip: 6000
});
camera.addComponent('script');
app.root.addChild(camera);
const cc = /** @type {CameraControls} */ (camera.script.create(CameraControls));
Object.assign(cc, {
    moveSpeed: 40,
    moveFastSpeed: 200,
    moveSlowSpeed: 8
});

// CameraFrame supplies the scene-depth attachment the two-phase occlusion reads
const cameraFrame = new CameraFrame(app, camera.camera);
cameraFrame.rendering.samples = 1;
cameraFrame.bloom.enabled = false;
cameraFrame.update();
app.on('destroy', () => cameraFrame.destroy());

// orbit pivot: the world point under the screen centre, from the meshlet depth. It is picked once
// when an orbit drag starts (left or middle button), not every frame: a pick costs a meshlet pick pass
// and a GPU readback. The pick is queued from pointerdown and run from postrender, because meshlet draw
// args are cleared at the end of each frame and a pick issued earlier finds nothing to draw.
const PICK_SCALE = 0.5;
const picker = new Picker(app, Math.round(device.width * PICK_SCALE), Math.round(device.height * PICK_SCALE), true);
const pivotTarget = new Vec3();
let pickRequested = false;
let pickInFlight = false;
window.addEventListener('pointerdown', (e) => {
    // left and middle start an orbit in CameraControls; right button starts fly, which keeps its own look
    if (e.button === 0 || e.button === 1) pickRequested = true;
});
app.on('destroy', () => picker.destroy());
app.on('postrender', () => {
    if (!pickRequested || pickInFlight) return;
    pickRequested = false;
    pickInFlight = true;
    picker.prepare(camera.camera, app.scene);
    picker.getWorldPointAsync(picker.width * 0.5, picker.height * 0.5).then((point) => {
        // setting focusPoint re-aims the camera, so it is set once per orbit start and only when hit
        if (point) {
            pivotTarget.copy(point);
            cc.focusPoint = pivotTarget;
        }
    }).finally(() => {
        pickInFlight = false;
    });
});

/**
 * @param {Region} region - Region to frame.
 */
const frameRegion = (region) => {
    const lo = new Vec3(region.min);
    const hi = new Vec3(region.max);
    const center = lo.clone().add(hi).mulScalar(0.5);
    const radius = hi.clone().sub(lo).length() * 0.5;
    const position = new Vec3(0.6, 0.45, 1).normalize().mulScalar(radius * 1.2).add(center);
    cc.reset(center, position);
};

// collision overlays: one translucent material per collision class
const collisionMaterials = {};
for (const [cls, color] of [['player_clip', [1, 0.45, 0.1]], ['weapon_clip', [0.2, 0.6, 1]], ['world', [0.3, 1, 0.4]]]) {
    const m = new StandardMaterial();
    m.diffuse.set(0, 0, 0);
    m.emissive.set(...color);
    m.opacity = 0.35;
    m.blendType = BLEND_NORMAL;
    m.depthWrite = false;
    m.useLighting = false;
    m.update();
    collisionMaterials[cls] = m;
}
/** @type {Map<string, Promise<Entity>>} */
const collisionEntities = new Map();

const updateCollision = () => {
    const show = data.get('data.collision');
    const filter = data.get('data.collisionClass');
    for (const region of regions) {
        if (!region.collision) continue;
        if (show && !collisionEntities.has(region.name)) {
            collisionEntities.set(region.name, loadContainer(`${region.label} collision`, BASE + region.collision).then((asset) => {
                const entity = asset.resource.instantiateRenderEntity();
                for (const cls of Object.keys(collisionMaterials)) {
                    entity.findByName(cls)?.findComponents('render').forEach((render) => {
                        render.meshInstances.forEach((mi) => {
                            mi.material = collisionMaterials[cls];
                        });
                    });
                }
                app.root.addChild(entity);
                return entity;
            }));
        }
        collisionEntities.get(region.name)?.then((entity) => {
            entity.enabled = show;
            for (const cls of Object.keys(collisionMaterials)) {
                const group = entity.findByName(cls);
                if (group) group.enabled = filter === 'all' || filter === cls;
            }
        });
    }
};

data.set('data', {
    region: regions[0]?.name ?? '',
    visualization: 'material',
    occlusion: true,
    threshold: 1,
    collision: false,
    collisionClass: 'all',
    stats: ''
});
data.on('data.region:set', (value) => {
    const region = regions.find(r => r.name === value);
    if (region) frameRegion(region);
});
data.on('data.visualization:set', (value) => {
    camera.camera.setShaderPass(value === 'normals' ? SHADERPASS_WORLDNORMAL : null);
});
data.on('data.occlusion:set', (value) => {
    app.systems.meshlet.occlusion = value;
});
data.on('data.threshold:set', (value) => {
    app.systems.meshlet.dagPixelThreshold = value;
});
data.on('data.collision:set', updateCollision);
data.on('data.collisionClass:set', updateCollision);

// open on the first POI rather than the whole-island terrain
const firstPoi = regions.find(r => r.name !== 'terrain') ?? regions[0];
if (firstPoi) {
    data.set('data.region', firstPoi.name);
    frameRegion(firstPoi);
}

/**
 * @param {string} key - Query parameter holding `x,y,z`.
 * @returns {Vec3|null} The parsed vector, or null when absent or malformed.
 */
const vecParam = (key) => {
    const v = params.get(key)?.split(',').map(Number);
    return v?.length === 3 && v.every(Number.isFinite) ? new Vec3(v) : null;
};
const startPos = vecParam('pos');
if (startPos) {
    cc.reset(vecParam('look') ?? startPos.clone().add(Vec3.FORWARD), startPos);
}

let statFrames = 0;
app.on('framerender', () => {
    // content past maxWorldMeshlets is split across several worlds: sum their stats
    const directors = app.systems.meshlet.directors.filter(d => d.world.finalized);
    if (!directors.length) return;
    const visualization = data.get('data.visualization');
    const colorMode = visualization === 'meshlet' ? 2 : visualization === 'lod' ? 1 : 0;
    let meshlets = 0, indices = 0, resident = 0, pages = 0, demand = false;
    for (const director of directors) {
        director.world.setColorMode(colorMode);
        const view = director.views.get(camera.camera);
        if (view?.lastDemand) {
            demand = true;
            meshlets += view.renderedMeshlets;
            indices += view.lastDemand.indices.reduce((a, b) => a + b, 0);
        }
        resident += director.residency?.residentPages ?? 0;
        pages += director.world.totalPages;
    }
    if (demand && ++statFrames % 30 === 0) {
        const worlds = directors.length > 1 ? ` (${directors.length} worlds)` : '';
        data.set('data.stats', `${meshlets} meshlets, cut ${(indices / 3e6).toFixed(2)}M tris, ` +
            `${resident}/${pages} pages${worlds}`);
    }
});

// ---------------------------------------------------------------------------------------------
// Benchmark (?benchmark=1): flies a fixed loop through the main regions at low altitude, twice,
// and reports frame-time statistics per pass and per leg. The first pass looks along the
// direction of travel, the second runs the same path looking at the mountain, so the two cover
// the streaming load of a moving view and the culling/shadow load of a fixed far target. A third
// pass cuts between the regions (teleports) and measures how long frames take to recover.
// Run it with `npm run bench:caldera` (see examples/utils/caldera/benchmark.mjs). Results land in
// `window.__benchmark` and on the console as a `BENCHMARK_RESULT` JSON line.
//
// ?benchSpeed=<m/s> (60), ?benchClearance=<m above the sampled surface> (12),
// ?benchPasses=forward,mountain,jump, ?benchSettle=<max seconds to wait for streaming before a pass> (45),
// ?benchDwell=<seconds spent at each stop of the jump pass> (8)
// ---------------------------------------------------------------------------------------------

// Path through the region centres, x/z in glTF metres and the surface height picked from above
// every ~80 m. WAYPOINTS index into PATH: the camera passes through each region centre.
const PATH = [
    [428, 1093, 21.1], [426, 1011, 53.8], [425, 928, 103], [423, 846, 128.6], [421, 764, 140.1], [419, 681, 139.6],
    [418, 599, 128.9], [416, 516, 98.7], [414, 434, 77.4], [494, 426, 87.2], [574, 417, 93.4], [654, 409, 120.8],
    [734, 400, 131.7], [813, 392, 110.9], [893, 383, 94.8], [973, 375, 42.3], [1053, 366, 18.3], [1061, 288, 16.1],
    [1068, 210, 13.9], [1076, 131, 10.4], [1083, 53, 9.4], [1091, -25, 15.9], [1098, -103, 19.9], [1106, -181, 11.8],
    [1113, -260, 11.3], [1121, -338, 9.4], [1128, -416, 7.1], [1072, -477, 17.4], [1017, -538, 31.4], [961, -599, 32.7],
    [906, -660, 42.4], [850, -721, 46], [795, -782, 56.9], [739, -842, 72.2], [684, -903, 64.7], [628, -964, 59],
    [573, -1025, 48], [517, -1086, 16.7], [462, -1147, 10.7], [406, -1208, 10.6], [326, -1208, 23], [245, -1209, 23.4],
    [165, -1209, 22.2], [84, -1209, 20.8], [4, -1210, 26.4], [-76, -1210, 28.7], [-157, -1210, 25.9], [-237, -1211, 21.8],
    [-318, -1211, 18.4], [-398, -1211, 29.3], [-479, -1212, 8.3], [-559, -1212, 10.4], [-536, -1138, 8.3], [-514, -1063, 18.5],
    [-491, -989, 18.3], [-468, -914, 27.1], [-445, -840, 36], [-423, -766, 53.9], [-400, -691, 79.4], [-377, -617, 84.3],
    [-355, -542, 83.5], [-332, -468, 62.8], [-409, -480, 90.2], [-485, -493, 86.9], [-562, -505, 74.9], [-638, -517, 58.4],
    [-715, -530, 48.2], [-791, -542, 24], [-868, -554, 25.6], [-944, -567, 44.6], [-1021, -579, 34.4], [-1042, -489, 46.1],
    [-1063, -400, 48.2], [-1084, -310, 38.2], [-1028, -248, 40.9], [-973, -187, 47.2], [-917, -125, 71.7], [-861, -63, 70.8],
    [-806, -2, 80.2], [-750, 60, 79.6], [-694, 122, 42.6], [-639, 183, 18.5], [-583, 245, 17.2], [-527, 307, 22.5],
    [-472, 368, 22.5], [-416, 430, 22.6], [-351, 481, 22.5], [-286, 532, 36.6], [-221, 583, 43.1], [-156, 634, 44.1],
    [-91, 685, 50.2], [-26, 736, 64.8], [38, 787, 75], [103, 838, 95.9], [168, 889, 127.1], [233, 940, 102.7],
    [298, 991, 91.9], [363, 1042, 59.9], [428, 1093, 21.1]
];
const WAYPOINTS = [[0, 'capital'], [8, 'agricultural_center'], [16, 'subpen'], [26, 'beachhead'], [39, 'docks'], [51, 'arsenal'], [61, 'phosphate_mine'], [70, 'ruins'], [73, 'village'], [85, 'airfield'], [98, 'capital']];

const BENCH_LOOK_AHEAD = 60;
// jump pass order: each cut lands on the far side of the island from the last
const JUMP_ORDER = ['capital', 'docks', 'airfield', 'beachhead', 'ruins', 'subpen', 'arsenal', 'village', 'agricultural_center', 'phosphate_mine'];
const HITCH_MS = [33.4, 50, 100];

/**
 * @param {number[]} values - Frame times in ms.
 * @returns {object} Summary statistics.
 */
const summarize = (values) => {
    const sorted = Float64Array.from(values).sort();
    const n = sorted.length;
    const at = q => (n ? sorted[Math.min(n - 1, Math.floor(q * n))] : 0);
    const sum = values.reduce((a, b) => a + b, 0);
    const round = v => Math.round(v * 100) / 100;
    return {
        frames: n,
        avgMs: round(n ? sum / n : 0),
        p50Ms: round(at(0.5)),
        p95Ms: round(at(0.95)),
        p99Ms: round(at(0.99)),
        maxMs: round(n ? sorted[n - 1] : 0),
        over33: values.filter(v => v > HITCH_MS[0]).length,
        over50: values.filter(v => v > HITCH_MS[1]).length,
        over100: values.filter(v => v > HITCH_MS[2]).length
    };
};

// the passes and the settle wait are inherently sequential, one frame at a time
/* eslint-disable no-await-in-loop */
const runBenchmark = async () => {
    const speed = Number(params.get('benchSpeed')) || 60;
    const clearance = Number(params.get('benchClearance') ?? 12);
    const settleSeconds = Number(params.get('benchSettle')) || 45;
    const passes = (params.get('benchPasses') ?? 'forward,mountain,jump').split(',');
    const caldera = manifest.regions.find(r => r.name === 'caldera');
    const mountain = new Vec3(
        (caldera.min[0] + caldera.max[0]) / 2,
        caldera.max[1],
        (caldera.min[2] + caldera.max[2]) / 2
    );

    // camera height: the highest surface within two samples either side, so a hill or roof just
    // ahead does not swallow the camera between samples
    const points = PATH.map(([x, z], i) => {
        let top = 0;
        for (let k = Math.max(0, i - 2); k <= Math.min(PATH.length - 1, i + 2); k++) top = Math.max(top, PATH[k][2]);
        return new Vec3(x, top + clearance, z);
    });
    const cumulative = [0];
    for (let i = 1; i < points.length; i++) cumulative.push(cumulative[i - 1] + points[i].distance(points[i - 1]));
    const length = cumulative[cumulative.length - 1];

    const pathIndex = (d) => {
        let i = 1;
        while (i < points.length - 1 && cumulative[i] < d) i++;
        return i;
    };
    const sampleAt = (d, out) => {
        d = Math.min(Math.max(d, 0), length);
        const i = pathIndex(d);
        const span = cumulative[i] - cumulative[i - 1];
        return out.lerp(points[i - 1], points[i], span > 0 ? (d - cumulative[i - 1]) / span : 0);
    };
    const legOf = (d) => {
        const i = pathIndex(d);
        let leg = 0;
        while (leg < WAYPOINTS.length - 2 && i > WAYPOINTS[leg + 1][0]) leg++;
        return leg;
    };

    const gpuProfiler = device.gpuProfiler;
    if (gpuProfiler) gpuProfiler.enabled = true;
    const directors = () => app.systems.meshlet.directors.filter(d => d.world.finalized);
    const streamCounters = () => directors().reduce((acc, d) => {
        acc.fetched += d.residency?.fetchedPages ?? 0;
        acc.evicted += d.residency?.evictedPages ?? 0;
        return acc;
    }, { fetched: 0, evicted: 0 });
    const nextFrame = () => new Promise((resolve) => {
        requestAnimationFrame(() => resolve(performance.now()));
    });
    const sleep = ms => new Promise((resolve) => {
        setTimeout(resolve, ms);
    });

    cc.enabled = false;
    const position = new Vec3();
    const target = new Vec3();
    const place = (d, mode) => {
        sampleAt(d, position);
        if (mode === 'mountain') target.copy(mountain);
        else sampleAt(d + BENCH_LOOK_AHEAD, target);
        camera.setPosition(position);
        camera.lookAt(target);
    };

    // wait for the world to exist and stream in around a view
    const settle = async (d, mode, minSeconds) => {
        const start = performance.now();
        for (;;) {
            place(d, mode);
            await nextFrame();
            const elapsed = (performance.now() - start) / 1000;
            // meshlet.playable also covers regions still loading: their join-in rebuild is a ~2 s stall
            const ready = app.systems.meshlet.playable && directors().length > 0 &&
                directors().every(dir => dir.residency?.playable && dir.residency.demandSettled);
            if (elapsed >= minSeconds && ready) return true;
            if (elapsed > settleSeconds) return false;
        }
    };

    // Jump pass: cut straight to each region in turn (far side of the island every time), look
    // along the route for a few seconds and record how the frames recover - streaming everything
    // in from nothing is what a teleport or a fast flight into new ground costs.
    const dwellSeconds = Number(params.get('benchDwell')) || 8;
    const runJumps = async () => {
        await settle(0, 'forward', 3);
        const order = JUMP_ORDER.filter(name => WAYPOINTS.some(w => w[1] === name));
        const all = [];
        const legs = {};
        const streamStart = streamCounters();
        const t0 = performance.now();
        for (const name of order) {
            const d = cumulative[WAYPOINTS.find(w => w[1] === name)[0]];
            const jump = { frames: [] };
            const timesOver20 = [];
            place(d, 'forward');
            const start = performance.now();
            let last = start;
            for (;;) {
                const now = await nextFrame();
                if (now - start > dwellSeconds * 1000) break;
                place(d, 'forward');
                const dt = now - last;
                last = now;
                jump.frames.push(dt);
                if (dt > 20) timesOver20.push(now - start);
            }
            all.push(...jump.frames);
            legs[`jump to ${name}`] = { ...summarize(jump.frames), lastSlowMs: Math.round(timesOver20.length ? timesOver20[timesOver20.length - 1] : 0) };
        }
        const streamEnd = streamCounters();
        const stats = summarize(all);
        console.log(`Benchmark jump: avg ${stats.avgMs} ms, p99 ${stats.p99Ms} ms, max ${stats.maxMs} ms, >50 ms: ${stats.over50}, >100 ms: ${stats.over100}`);
        return {
            settledBeforeStart: true,
            seconds: Math.round((performance.now() - t0) / 100) / 10,
            ...stats,
            gpuAvgMs: null,
            pagesFetched: streamEnd.fetched - streamStart.fetched,
            pagesEvicted: streamEnd.evicted - streamStart.evicted,
            legs
        };
    };

    const results = {
        speed,
        clearance,
        pathLengthM: Math.round(length),
        device: deviceType,
        pool: app.systems.meshlet.poolBytes / (1024 * 1024),
        userAgent: navigator.userAgent,
        passes: {}
    };
    window.__benchmark = { done: false, results };

    for (const mode of passes) {
        if (mode === 'jump') {
            results.passes.jump = await runJumps();
            continue;
        }
        const settled = await settle(0, mode, 3);
        const frames = [];
        const legFrames = WAYPOINTS.slice(1).map(() => []);
        const gpuSamples = [];
        const gpuPassSums = new Map();
        const streamStart = streamCounters();
        const t0 = performance.now();
        let last = t0;
        for (;;) {
            const now = await nextFrame();
            const d = ((now - t0) / 1000) * speed;
            if (d >= length) break;
            place(d, mode);
            const dt = now - last;
            last = now;
            frames.push(dt);
            legFrames[legOf(d)].push(dt);
            if (gpuProfiler?.enabled && gpuProfiler._frameTime) {
                gpuSamples.push(gpuProfiler._frameTime);
                gpuProfiler.passTimings.forEach((ms, name) => gpuPassSums.set(name, (gpuPassSums.get(name) ?? 0) + ms));
            }
        }
        const streamEnd = streamCounters();
        const passResult = {
            settledBeforeStart: settled,
            seconds: Math.round((performance.now() - t0) / 100) / 10,
            ...summarize(frames),
            gpuAvgMs: gpuSamples.length ? Math.round(gpuSamples.reduce((a, b) => a + b, 0) / gpuSamples.length * 100) / 100 : null,
            // average GPU ms per frame of the heaviest passes
            gpuPasses: Object.fromEntries([...gpuPassSums.entries()]
            .sort((a, b) => b[1] - a[1]).slice(0, 14)
            .map(([name, sum]) => [name, Math.round(sum / Math.max(gpuSamples.length, 1) * 100) / 100])),
            pagesFetched: streamEnd.fetched - streamStart.fetched,
            pagesEvicted: streamEnd.evicted - streamStart.evicted,
            legs: {}
        };
        legFrames.forEach((values, k) => {
            passResult.legs[`${WAYPOINTS[k][1]} -> ${WAYPOINTS[k + 1][1]}`] = summarize(values);
        });
        results.passes[mode] = passResult;
        console.log(`Benchmark ${mode}: avg ${passResult.avgMs} ms, p99 ${passResult.p99Ms} ms, max ${passResult.maxMs} ms, >50 ms: ${passResult.over50}, >100 ms: ${passResult.over100}`);
        await sleep(250);
    }

    // hand the camera back to the controls where the benchmark left it
    cc.reset(target, position);
    cc.enabled = true;

    window.__benchmark = { done: true, results };
    console.log(`BENCHMARK_RESULT ${JSON.stringify(results)}`);
};

/* eslint-enable no-await-in-loop */

// ?gpuStats=1: an on-screen table of GPU time per render pass, averaged over two seconds - for
// devices without remote debugging. Needs the 'timestamp-query' WebGPU feature.
if (params.get('gpuStats') === '1') {
    const panel = document.createElement('pre');
    Object.assign(panel.style, {
        position: 'absolute',
        left: '8px',
        bottom: '8px',
        margin: '0',
        padding: '6px 8px',
        zIndex: '10',
        font: '11px/1.3 monospace',
        color: '#fff',
        background: 'rgba(0, 0, 0, 0.65)',
        pointerEvents: 'none'
    });
    document.body.appendChild(panel);
    const profiler = device.gpuProfiler;
    const sums = new Map();
    let frames = 0;
    let gpuTotal = 0;
    let started = performance.now();
    if (!device.supportsTimestampQuery || !profiler) {
        panel.textContent = 'GPU timings unavailable: no timestamp-query support on this device';
    } else {
        profiler.enabled = true;
        app.on('frameend', () => {
            if (profiler._frameTime) {
                frames++;
                gpuTotal += profiler._frameTime;
                profiler.passTimings.forEach((ms, name) => sums.set(name, (sums.get(name) ?? 0) + ms));
            }
            const now = performance.now();
            if (now - started < 2000 || !frames) return;
            const avg = ms => (ms / frames).toFixed(2).padStart(6);
            let camera = 0, shadow = 0;
            sums.forEach((ms, name) => {
                if (/Shadow/.test(name)) shadow += ms;
                else if (/^Meshlet/.test(name)) camera += ms;
            });
            const rows = [...sums.entries()].sort((a, b) => b[1] - a[1]).slice(0, 18)
            .map(([name, ms]) => `${avg(ms)}  ${name}`);
            panel.textContent = [
                `GPU ${avg(gpuTotal)} ms/frame (${frames} frames)`,
                `meshlet camera ${avg(camera)}  shadows ${avg(shadow)}`,
                '',
                ...rows
            ].join('\n');
            sums.clear();
            frames = 0;
            gpuTotal = 0;
            started = now;
        });
    }
}

if (params.get('benchmark') === '1') {
    runBenchmark().catch((err) => {
        cc.enabled = true;
        window.__benchmark = { done: true, error: String(err?.stack ?? err) };
        console.error(`BENCHMARK_ERROR ${err?.stack ?? err}`);
    });
}

app.start();
