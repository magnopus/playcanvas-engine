// @config
//
// The complete Zorah dataset, streamed through the GPU-driven meshlet pipeline.
// Fly with WASD and mouse look; Shift/Ctrl changes speed. See assets/meshlets/README.md
// for the local dataset setup.
//
// @flag WEBGL_DISABLED
// @flag WEBGPU_BARE_DISABLED

import {
    AppBase,
    AppOptions,
    Asset,
    AssetListLoader,
    CameraComponentSystem,
    CameraFrame,
    Color,
    ContainerHandler,
    Entity,
    EnvLighting,
    FILLMODE_FILL_WINDOW,
    Keyboard,
    LightComponentSystem,
    MeshletComponentSystem,
    Mouse,
    RenderComponentSystem,
    RESOLUTION_AUTO,
    ScriptComponentSystem,
    TONEMAP_ACES2,
    TextureHandler,
    TouchDevice,
    Vec3,
    basisInitialize,
    createGraphicsDevice
} from 'playcanvas';
import { CameraControls } from 'playcanvas/scripts/esm/camera-controls.mjs';

import { deviceType } from 'examples/context';

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('application-canvas'));
window.focus();

const manifestUrl = new URL('./assets/meshlets/zorah/scene.all.json', document.baseURI);
const response = await fetch(manifestUrl);
if (!response.ok) {
    throw new Error(`Cannot load Zorah (${response.status}); see assets/meshlets/README.md.`);
}
const scene = await response.json();
if (scene.version !== 1 || !scene.entities?.length) {
    throw new Error('Invalid Zorah scene manifest.');
}

basisInitialize({
    glueUrl: './assets/wasm/basis/basis.wasm.js',
    wasmUrl: './assets/wasm/basis/basis.wasm.wasm',
    fallbackUrl: './assets/wasm/basis/basis.js'
});

const device = await createGraphicsDevice(canvas, {
    deviceTypes: [deviceType],
    antialias: false,
    reverseZ: true
});
device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);

const options = new AppOptions();
options.graphicsDevice = device;
options.mouse = new Mouse(document.body);
options.keyboard = new Keyboard(window);
options.touch = new TouchDevice(document.body);
options.componentSystems = [
    CameraComponentSystem,
    LightComponentSystem,
    RenderComponentSystem,
    ScriptComponentSystem,
    MeshletComponentSystem
];
options.resourceHandlers = [ContainerHandler, TextureHandler];

const app = new AppBase(canvas);
app.init(options);
app.setCanvasFillMode(FILLMODE_FILL_WINDOW);
app.setCanvasResolution(RESOLUTION_AUTO);
const resize = () => app.resizeCanvas();
window.addEventListener('resize', resize);
app.on('destroy', () => window.removeEventListener('resize', resize));

app.systems.meshlet.poolBytes = 17179869184;
app.systems.meshlet.occlusion = true;

// Manifest model URLs are relative to assets/meshlets, not the manifest directory.
const assetBase = new URL('../', manifestUrl);
const skySource = new Asset(
    'Jungle sky',
    'texture',
    { url: './assets/meshlets/jungle/jungle_sky.hdr' },
    { mipmaps: false }
);
const environmentSource = new Asset(
    'Jungle environment',
    'texture',
    { url: './assets/meshlets/jungle/jungle_sky_env.hdr' },
    { mipmaps: false }
);
const assets = new Map();
for (const entry of scene.entities) {
    for (const model of entry.models) {
        const url = new URL(model.url, assetBase).href;
        if (!assets.has(url)) {
            assets.set(url, new Asset(model.name, 'container', { url }));
        }
    }
}
await new Promise((resolve, reject) => {
    new AssetListLoader([skySource, environmentSource, ...assets.values()], app.assets).load(error => error ? reject(error) : resolve());
});
const skybox = EnvLighting.generateSkyboxCubemap(skySource.resource);
const lighting = EnvLighting.generateLightingSource(environmentSource.resource);
const envAtlas = EnvLighting.generateAtlas(lighting);
app.scene.skybox = skybox;
app.scene.envAtlas = envAtlas;
lighting.destroy();
skySource.unload();
environmentSource.unload();
app.on('destroy', () => {
    skybox.destroy();
    envAtlas.destroy();
});

/**
 * @param {Entity} entity - The entity to place.
 * @param {object} transform - The manifest transform.
 * @param {[number, number, number]} transform.position - Local position.
 * @param {[number, number, number, number]} transform.rotation - Local quaternion.
 * @param {[number, number, number]} transform.scale - Local scale.
 */
function setTransform(entity, transform) {
    entity.setLocalPosition(...transform.position);
    entity.setLocalRotation(...transform.rotation);
    entity.setLocalScale(...transform.scale);
}

const entities = new Map();
for (const entry of scene.entities) {
    if (entities.has(entry.id)) {
        throw new Error(`Duplicate Zorah entity ${entry.id}.`);
    }
    const entity = new Entity(entry.name);
    setTransform(entity, entry);
    entities.set(entry.id, entity);
    for (const model of entry.models) {
        const asset = assets.get(new URL(model.url, assetBase).href);
        const child = new Entity(model.name);
        setTransform(child, model);
        child.enabled = model.visible;
        child.addComponent('meshlet', { asset, castShadows: model.shadowCaster });
        // Blended surfaces remain inline for the ordinary sorted forward renderer.
        if (asset.resource.renders.some(render => render.resource.meshes.length > 0)) {
            child.addChild(asset.resource.instantiateRenderEntity());
        }
        entity.addChild(child);
    }
}
for (const entry of scene.entities) {
    const parent = entry.parentId === null ? app.root : entities.get(entry.parentId);
    if (!parent) {
        throw new Error(`Missing Zorah parent ${entry.parentId}.`);
    }
    parent.addChild(entities.get(entry.id));
}

const sun = new Entity('Sun');
sun.addComponent('light', { enabled: false, type: 'directional', color: Color.WHITE, intensity: 1.5 });
sun.setEulerAngles(45, 30, 0);
app.root.addChild(sun);
app.scene.ambientLight = new Color(0.25, 0.27, 0.3);

const camera = new Entity('Camera');
camera.addComponent('camera', {
    clearColor: new Color(0.12, 0.14, 0.18),
    nearClip: 0.05,
    farClip: scene.farClip
});
camera.addComponent('script');
app.root.addChild(camera);
setTransform(camera, { ...scene.start, scale: [1, 1, 1] });
const controls = /** @type {CameraControls} */ (camera.script.create(CameraControls));
Object.assign(controls, { moveSpeed: 5, moveFastSpeed: 20, moveSlowSpeed: 1 });
const position = camera.getPosition().clone();
controls.reset(new Vec3().copy(camera.forward).add(position), position);

const frame = new CameraFrame(app, camera.camera);
frame.rendering.samples = 1;
frame.rendering.toneMapping = TONEMAP_ACES2;
frame.bloom.enabled = false;
frame.update();
app.on('destroy', () => frame.destroy());

app.start();
