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
 * min: number[], max: number[] }} Region
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

const scene = new Entity('Caldera');
app.root.addChild(scene);

// Every region's component exists up front and loads its own asset, so the meshlet system can
// see which are still loading: the first region builds at once, the rest wait for the burst to
// finish and join in a single rebuild (?progressive=1 instead rebuilds as regions arrive)
app.systems.meshlet.deferRebuildWhileLoading = params.get('progressive') !== '1';
for (const region of regions) {
    const asset = new Asset(region.label, 'container', { url: BASE + region.url });
    asset.once('error', err => console.error(`Caldera: failed to load ${region.url}`, err));
    app.assets.add(asset);
    const entity = new Entity(region.label);
    entity.addComponent('meshlet', { asset });
    scene.addChild(entity);
}

const light = new Entity('Sun');
light.addComponent('light', {
    type: 'directional',
    color: new Color(1, 0.96, 0.9),
    intensity: 1.6
});
light.setEulerAngles(50, 35, 0);
app.root.addChild(light);

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

app.start();
