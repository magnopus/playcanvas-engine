// @config
//
// The JungleRuins trees in isolation, for tuning their impostors: each prototype stands twice on a
// ground plane under the scene's own sky and sun - on the left as real meshlets, on the right as the
// billboard of its baked views - so the two can be compared at any angle and light. Build the
// assets with examples/utils/jungle: protos.py + the meshlet-impostor-bake example for the atlas,
// then `pipeline.sh treetest` for the scene (baked to <bake>/treetest/).
//
// @flag WEBGL_DISABLED
// @flag WEBGPU_BARE_DISABLED
// @flag HIDDEN
// @flag HIDDEN_IN_DEV

import {
    ADDRESS_CLAMP_TO_EDGE,
    AppBase,
    AppOptions,
    Asset,
    basisInitialize,
    CameraComponentSystem,
    CameraFrame,
    Color,
    ContainerHandler,
    Entity,
    EnvLighting,
    FILLMODE_FILL_WINDOW,
    FILTER_LINEAR,
    FILTER_LINEAR_MIPMAP_LINEAR,
    LightComponentSystem,
    MeshletComponentSystem,
    Quat,
    RenderComponentSystem,
    RESOLUTION_AUTO,
    ScriptComponentSystem,
    StandardMaterial,
    TONEMAP_ACES2,
    Texture,
    TextureHandler,
    Vec3,
    createGraphicsDevice
} from 'playcanvas';
import { CameraControls } from 'playcanvas/scripts/esm/camera-controls.mjs';

import { data, deviceType } from 'examples/context';

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('application-canvas'));
window.focus();

basisInitialize({
    glueUrl: './assets/wasm/basis/basis.wasm.js',
    wasmUrl: './assets/wasm/basis/basis.wasm.wasm',
    fallbackUrl: './assets/wasm/basis/basis.js'
});

const params = new URLSearchParams(location.search);
// ?base=<dir>/ is the bake folder; ?scale= the scale the test scene's instances were built with
const BASE = params.get('base') ?? './assets/meshlets/jungle/';
const INSTANCE_SCALE = Number(params.get('scale') ?? 170);

const device = await createGraphicsDevice(canvas, { deviceTypes: [deviceType], antialias: false });
device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);

const createOptions = new AppOptions();
createOptions.graphicsDevice = device;
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
const resize = () => app.resizeCanvas();
window.addEventListener('resize', resize);
app.on('destroy', () => window.removeEventListener('resize', resize));

// ---------------------------------------------------------------------------------------------
// impostor atlas

const manifest = await fetch(`${BASE}jungle_impostors.json`).then(r => r.json());
const atlas = async (/** @type {string} */ kind, /** @type {boolean} */ srgb) => {
    // one UASTC KTX2 per layer (bake_server.py), transcoded to the device's block format by the
    // engine's own transcoder
    const transcode = app.systems.meshlet.group.transcode;
    const layers = await Promise.all(manifest.prototypes.map(async (p) => {
        const bytes = await fetch(`${BASE}impostor_${kind}_${p.layer}.ktx2`).then(r => r.arrayBuffer());
        return new Promise((resolve, reject) => {
            transcode(device, `impostor-${kind}-${p.layer}`, bytes, (err, result) => {
                return err || !result ? reject(new Error(err ?? 'impostor transcode failed')) : resolve(result);
            }, { isKTX2: true, isTextureArray: true });
        });
    }));
    const { format, width, height } = layers[0];
    return new Texture(device, {
        name: `impostor-${kind}`,
        width,
        height,
        format,
        srgb,
        arrayLength: layers.length,
        mipmaps: true,
        numLevels: layers[0].levels.length,
        levels: layers[0].levels.map((_, mip) => layers.map(layer => new Uint8Array(layer.levels[mip]))),
        addressU: ADDRESS_CLAMP_TO_EDGE,
        addressV: ADDRESS_CLAMP_TO_EDGE,
        minFilter: FILTER_LINEAR_MIPMAP_LINEAR,
        magFilter: FILTER_LINEAR
    });
};
app.systems.meshlet.impostors = {
    albedo: await atlas('albedo', true),
    normal: await atlas('normal', false),
    grid: manifest.grid,
    prototypes: Object.fromEntries(manifest.prototypes.map(p => [p.name, {
        layer: p.layer, center: p.center, radius: p.radius, half: p.half
    }]))
};
app.systems.meshlet.poolBytes = 3072 * 1024 * 1024;
app.systems.meshlet.occlusion = false;
// the real trees are the reference: refine them as far as the LOD allows
app.systems.meshlet.dagPixelThreshold = Number(params.get('threshold') ?? 0.25);
app.systems.meshlet.rootMinPixels = 0;

// the test scene lays the prototypes out in name order, each pair as protos.py lays it out: real left of
// the billboard, `gap` either side of the pair's centre
const prototypes = [...manifest.prototypes].sort((a, b) => a.name.localeCompare(b.name));
const pairs = [];
let rowX = 0;
for (const p of prototypes) {
    const gap = p.radius * INSTANCE_SCALE * 1.25;
    pairs.push({ name: p.name, x: rowX, gap, height: p.half[1] * INSTANCE_SCALE, radius: p.radius * INSTANCE_SCALE });
    rowX += 4 * gap;
}

const treeAsset = new Asset('treetest', 'container', { url: `${BASE}${params.get('tree') ?? 'treetest'}/jungle_treetest.glb` });
treeAsset.once('error', err => console.error('failed to load the tree test scene', err));
app.assets.add(treeAsset);
const trees = new Entity('Trees');
trees.addComponent('meshlet', { asset: treeAsset });
app.root.addChild(trees);

// ---------------------------------------------------------------------------------------------
// sky and sun: the jungle's own (sky.py)

const sky = await fetch(`${BASE}jungle_sky.json`).then(r => r.json());
const loadHdr = (/** @type {string} */ file) => new Promise((resolve, reject) => {
    const asset = new Asset(file, 'texture', { url: BASE + file }, { mipmaps: false });
    asset.once('load', () => resolve(asset.resource));
    asset.once('error', reject);
    app.assets.add(asset);
    app.assets.load(asset);
});
const [skyTexture, envTexture] = await Promise.all([loadHdr('jungle_sky.hdr'), loadHdr('jungle_sky_env.hdr')]);
app.scene.skybox = EnvLighting.generateSkyboxCubemap(skyTexture);
const lighting = EnvLighting.generateLightingSource(envTexture);
app.scene.envAtlas = EnvLighting.generateAtlas(lighting);
lighting.destroy();
app.scene.ambientLight = new Color(0, 0, 0);

const light = new Entity('Sun');
light.addComponent('light', { type: 'directional', color: new Color(...sky.sunColor), intensity: sky.sunIntensity });
light.setRotation(new Quat().setFromDirections(Vec3.UP, new Vec3(sky.toSun)));
app.root.addChild(light);

const ground = new Entity('Ground');
ground.addComponent('render', { type: 'plane' });
ground.setLocalScale(rowX * 2 + 4000, 1, 4000);
ground.setPosition(rowX * 0.5, 0, 0);
const groundMaterial = new StandardMaterial();
groundMaterial.diffuse = new Color(0.1, 0.12, 0.05);
groundMaterial.gloss = 0.1;
groundMaterial.useMetalness = true;
groundMaterial.metalness = 0;
groundMaterial.update();
ground.render.material = groundMaterial;
app.root.addChild(ground);

// ---------------------------------------------------------------------------------------------
// camera

const camera = new Entity('Camera');
camera.addComponent('camera', { clearColor: new Color(0.55, 0.68, 0.82), nearClip: 0.1, farClip: 6000 });
camera.addComponent('script');
app.root.addChild(camera);
const cc = /** @type {CameraControls} */ (camera.script.create(CameraControls));
Object.assign(cc, { moveSpeed: 10, moveFastSpeed: 60, moveSlowSpeed: 2 });

const cameraFrame = new CameraFrame(app, camera.camera);
cameraFrame.rendering.samples = 1;
cameraFrame.bloom.enabled = false;
cameraFrame.rendering.toneMapping = TONEMAP_ACES2;
cameraFrame.update();
app.on('destroy', () => cameraFrame.destroy());

/** @param {number} index - Pair to look at. */
const focusPair = (index) => {
    const pair = pairs[Math.max(0, Math.min(pairs.length - 1, index | 0))];
    const focus = new Vec3(pair.x, pair.height * 0.6, 0);
    cc.reset(focus, new Vec3(pair.x, pair.height * 0.6, pair.radius * Number(params.get('dist') ?? 2.8)));
    data.set('data.stats', `${pair.name} - real left, impostor right`);
};

// ---------------------------------------------------------------------------------------------
// controls

data.set('data', {
    prototype: Number(params.get('proto') ?? 3),
    stats: '',
    sun: Number(params.get('sun') ?? 3.5),
    sky: Number(params.get('skyScale') ?? 1),
    exposure: Number(params.get('exposure') ?? 0.6),
    bent: Number(params.get('bent') ?? 0),
    occlusion: Number(params.get('ao') ?? 0.3),
    albedo: Number(params.get('albedo') ?? 0.7),
    debug: Number(params.get('debug') ?? 0),
    forceBillboards: params.get('force') !== '0',
    impostorPx: Number(params.get('impostorPx') ?? 128)
});

const applyLighting = () => {
    light.light.intensity = data.get('data.sun');
    app.scene.skyboxIntensity = data.get('data.sky');
    app.scene.exposure = data.get('data.exposure');
};
for (const key of ['sun', 'sky', 'exposure']) data.on(`data.${key}:set`, applyLighting);
applyLighting();

const applyImpostorPixels = () => {
    // forced on, every tagged tree is a billboard at any distance (rootMinPixels is 0 here)
    app.systems.meshlet.impostorPixels = data.get('data.forceBillboards') ? 1e6 : data.get('data.impostorPx');
};
data.on('data.forceBillboards:set', applyImpostorPixels);
data.on('data.impostorPx:set', applyImpostorPixels);
applyImpostorPixels();

data.on('data.prototype:set', focusPair);
focusPair(data.get('data.prototype'));

// the shading and debug values ride on the impostor material, which a world rebuild replaces
const applied = new WeakMap();
app.on('framerender', () => {
    const key = [data.get('data.bent'), data.get('data.occlusion'), data.get('data.albedo'), data.get('data.debug')].join();
    for (const director of app.systems.meshlet.directors) {
        const material = director.world.impostorMaterial;
        if (material && applied.get(material) !== key) {
            applied.set(material, key);
            material.setParameter('impostorShading', [data.get('data.bent'), data.get('data.occlusion'), data.get('data.albedo'), 0]);
            material.setParameter('impostorDebug', data.get('data.debug'));
        }
    }
});

app.start();
