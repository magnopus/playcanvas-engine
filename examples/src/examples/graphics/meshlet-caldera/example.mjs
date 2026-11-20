// @config
//
// The Caldera map as streamed meshlets (WebGPU only): each region is its own streamed meshlet
// asset, and all of them share one meshlet budget. Collision soups are ordinary glTF meshes,
// loaded on demand and drawn as translucent overlays. Needs a local bake, see
// examples/assets/meshlets/README.md.
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
    CircleGeometry,
    Color,
    ContainerHandler,
    Entity,
    FILLMODE_FILL_WINDOW,
    Keyboard,
    Layer,
    LightComponentSystem,
    Mesh,
    MeshInstance,
    math,
    MESHLET_COLOR_MODE,
    MeshletComponentSystem,
    OpfsByteStore,
    Mouse,
    RenderComponentSystem,
    RESOLUTION_AUTO,
    SHADERPASS_WORLDNORMAL,
    ScriptComponentSystem,
    StandardMaterial,
    TONEMAP_ACES2,
    TextureHandler,
    TouchDevice,
    Vec3,
    createGraphicsDevice
} from 'playcanvas';
import { CameraControls } from 'playcanvas/scripts/esm/camera-controls.mjs';
import { ProceduralSky } from 'playcanvas/scripts/esm/sky/procedural-sky.mjs';
import { Water } from 'playcanvas/scripts/esm/water.mjs';

import { data, deviceType } from 'examples/context';

import { runBenchmark, showGpuStats } from './benchmark.mjs';

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('application-canvas'));
window.focus();

// streamed textures are KTX2 / Basis
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
const manifest = await fetch(`${BASE}caldera_scene.json`).then((r) => r.json());
// ?regions=tile_p,docks loads only those; ?pos=x,y,z&look=x,y,z starts the camera there (glTF metres)
const params = new URLSearchParams(location.search);
const only = params.get('regions')?.split(',');
const regions = manifest.regions.filter((r) => r.baked && (!only || only.includes(r.name)));

// reverse-z keeps depth precision across the island's view distance (?reverseZ=0 turns it off)
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
// reload reads them from disk instead of the network. The cap is raised above the default 2 GiB,
// which the whole island exceeds and would evict on every load.
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
const loadContainer = (name, url) =>
    new Promise((resolve, reject) => {
        const asset = new Asset(name, 'container', { url });
        asset.once('load', () => resolve(asset));
        asset.once('error', reject);
        app.assets.add(asset);
        app.assets.load(asset);
    });

// every region shares one meshlet world, so the pool covers the sum of their resident sets
// ?pool=<MiB>: the default suits a handful of regions, the whole island needs about twice that
app.systems.meshlet.poolBytes = (Number(params.get('pool')) || 2048) * 1024 * 1024;
// ?worldMeshlets=<millions>: split the content across worlds of at most this many meshlets
if (params.get('worldMeshlets')) app.systems.meshlet.maxWorldMeshlets = Number(params.get('worldMeshlets')) * 1e6;
app.systems.meshlet.occlusion = true;
// ?cssPixels=0: measure the LOD threshold in render target pixels (finer on high-density displays)
app.systems.meshlet.cssPixels = params.get('cssPixels') !== '0';

const scene = new Entity('Caldera');
app.root.addChild(scene);

// every region's component exists up front, so the meshlet system can see which are still
// loading and join them in a single rebuild (?progressive=1 rebuilds as regions arrive)
app.systems.meshlet.deferRebuildWhileLoading = params.get('progressive') !== '1';
// ?clutter=1 also loads the terrain tiles' baked trees (pipeline.sh CLUTTER=trees), ?clutter=all
// their grass, plants and rocks too (CLUTTER=all), where present
const clutter = params.get('clutter');
for (const region of regions) {
    const extra =
        clutter === 'all' ? (region.clutterAll ?? region.clutter) : clutter === '1' ? region.clutter : undefined;
    const urls = extra ? [region.url, extra] : [region.url];
    for (const url of urls) {
        const asset = new Asset(region.label, 'container', { url: BASE + url });
        asset.once('error', (err) => console.error(`Caldera: failed to load ${url}`, err));
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
    intensity: 2
});
light.setEulerAngles(50, 35, 0);
app.root.addChild(light);

// ?shadows=1: the sun casts four cascaded shadows over the island, meshlets included.
// ?cascadeIntervals=1,1,2,2 (the default) refreshes the two far cascades on alternate frames
if (params.get('shadows') === '1') {
    app.systems.meshlet.shadows = true;
    light.light.castShadows = true;
    light.light.numCascades = 4;
    light.light.shadowResolution = 2048;
    light.light.shadowDistance = 1500;
    const intervals = (params.get('cascadeIntervals') ?? '1,1,2,2').split(',').map(Number);
    light.light.cascadeUpdateIntervals = intervals.some((v) => v > 1) ? intervals : null;
    // ?shadowLod=<multiplier on the LOD error threshold for casters> (4) and
    // ?shadowRootMin=<screen px below which an instance casts no meshlet shadow> (2)
    const group = app.systems.meshlet.group;
    if (group && params.get('shadowLod')) group.shadowThresholdScale = Number(params.get('shadowLod'));
    if (group && params.get('shadowRootMin')) group.shadowRootMinPixels = Number(params.get('shadowRootMin'));
}

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

// a procedural sky lights the scene through its env-atlas; the sun disk sits where the light
// shines from (its +Y). The sky is not handed the light, as it would recolour and re-aim it.
const sky = new Entity('ProceduralSky');
sky.addComponent('script');
const skyScript = /** @type {ProceduralSky} */ (sky.script.create(ProceduralSky));
const toSun = light.up;
skyScript.elevation = Math.asin(toSun.y) * math.RAD_TO_DEG;
skyScript.azimuth = Math.atan2(toSun.x, toSun.z) * math.RAD_TO_DEG;
// the sky's share of the light, against the sun's intensity: brighter flattens the shading
skyScript.luminance = 0.3;
app.root.addChild(sky);
app.scene.ambientLight = new Color(0, 0, 0);
cameraFrame.rendering.toneMapping = TONEMAP_ACES2;
cameraFrame.update();

// The sea (?water=0 turns it off) at ?seaLevel=<m>, reflecting the sky's env-atlas. Planar
// reflection and refraction are off, as their cameras would render the meshlet scene again. So
// are the depth effects: they need a depth prepass, which has no meshlets in it.
if (params.get('water') !== '0') {
    const normalMap = await new Promise((resolve, reject) => {
        const asset = new Asset('water normal', 'texture', { url: './assets/textures/normal-map.png' });
        asset.once('load', () => resolve(asset.resource));
        asset.once('error', reject);
        app.assets.add(asset);
        app.assets.load(asset);
    });

    // the water blends over the opaque scene, so it renders after the world's transparent pass
    const worldLayer = app.scene.layers.getLayerByName('World');
    const waterLayer = new Layer({ name: 'Water' });
    app.scene.layers.insert(waterLayer, app.scene.layers.getTransparentIndex(worldLayer) + 1);
    camera.camera.layers = [...camera.camera.layers, waterLayer.id];

    // a far clip and water disc well past the island, so no grey band of sky shows between the
    // water's edge and the horizon
    camera.camera.farClip = 60000;
    const waterMesh = Mesh.fromGeometry(device, new CircleGeometry({ radius: 62000, sectors: 64, rings: 4 }));
    const water = new Entity('Water');
    water.addComponent('render', {
        meshInstances: [new MeshInstance(waterMesh, new StandardMaterial())],
        layers: [waterLayer.id],
        castShadows: false
    });
    const seaLevel = Number(params.get('seaLevel') ?? 3.5);
    water.setLocalPosition(0, seaLevel, 0);
    app.root.addChild(water);

    water.addComponent('script');
    water.script.create(Water, {
        properties: {
            cameraEntity: camera,
            lightEntity: light,
            normalMap,
            reflectionSource: 'sky',
            refraction: false,
            depthEffects: false,
            shallowColor: new Color(0.6, 0.9, 0.85),
            deepColor: new Color(0.03, 0.16, 0.24),
            depthFade: 3,
            // larger, softer ripples than the water example's, which shimmer at these distances
            rippleTiling: 0.02,
            bumpiness: 0.25,
            waves: false
        }
    });
}

// orbit pivot: the world point under the screen centre, picked once when an orbit drag starts.
// The pick runs from postrender, as meshlet draw args are cleared at the end of each frame.
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
    // the world layer only: the water's shader has no pick output
    picker.prepare(camera.camera, app.scene, [app.scene.layers.getLayerByName('World')]);
    picker
        .getWorldPointAsync(picker.width * 0.5, picker.height * 0.5)
        .then((point) => {
            // setting focusPoint re-aims the camera, so it is set once per orbit start and only when hit
            if (point) {
                pivotTarget.copy(point);
                cc.focusPoint = pivotTarget;
            }
        })
        .finally(() => {
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
    const position = new Vec3(0.6, 0.45, 1)
        .normalize()
        .mulScalar(radius * 1.2)
        .add(center);
    cc.reset(center, position);
};

// collision overlays: one translucent material per collision class
const collisionMaterials = {};
for (const [cls, color] of [
    ['player_clip', [1, 0.45, 0.1]],
    ['weapon_clip', [0.2, 0.6, 1]],
    ['world', [0.3, 1, 0.4]]
]) {
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
            collisionEntities.set(
                region.name,
                loadContainer(`${region.label} collision`, BASE + region.collision).then((asset) => {
                    const entity = asset.resource.instantiateRenderEntity();
                    for (const cls of Object.keys(collisionMaterials)) {
                        entity
                            .findByName(cls)
                            ?.findComponents('render')
                            .forEach((render) => {
                                render.meshInstances.forEach((mi) => {
                                    mi.material = collisionMaterials[cls];
                                });
                            });
                    }
                    app.root.addChild(entity);
                    return entity;
                })
            );
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
    threshold: app.systems.meshlet.dagPixelThreshold,
    collision: false,
    collisionClass: 'all',
    stats: ''
});
data.on('data.region:set', (value) => {
    const region = regions.find((r) => r.name === value);
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
const firstPoi = regions.find((r) => r.name !== 'terrain') ?? regions[0];
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
    const directors = app.systems.meshlet.directors.filter((d) => d.world.finalized);
    if (!directors.length) return;
    const colorMode =
        data.get('data.visualization') === 'meshlet' ? MESHLET_COLOR_MODE.MESHLET : MESHLET_COLOR_MODE.LIT;
    let meshlets = 0,
        indices = 0,
        resident = 0,
        pages = 0,
        demand = false;
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
        data.set(
            'data.stats',
            `${meshlets} meshlets, cut ${(indices / 3e6).toFixed(2)}M tris, ${resident}/${pages} pages${worlds}`
        );
    }
});

// ?gpuStats=1: an on-screen table of GPU time per render pass
if (params.get('gpuStats') === '1') {
    showGpuStats(app, device);
}

// ?benchmark=1: run by `npm run bench:caldera`, see benchmark.mjs
if (params.get('benchmark') === '1') {
    runBenchmark({ app, camera, controls: cc, params, manifest, deviceType });
}

app.start();
