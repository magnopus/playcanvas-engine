// @config
//
// Local Zorah streaming reproduction. Load a chunk as a container asset and stream its
// geometry pages on demand. Orbit, pan and zoom into surfaces to inspect residency transitions;
// toggle occlusion to exercise CameraFrame's scene-depth attachment and two-phase HZB path.
// See assets/meshlets/README.md for the local asset setup.
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
    FILLMODE_FILL_WINDOW,
    LightComponentSystem,
    MeshletComponentSystem,
    Mouse,
    RESOLUTION_AUTO,
    ScriptComponentSystem,
    ScriptHandler,
    TouchDevice,
    Vec3,
    createGraphicsDevice
} from 'playcanvas';

import { data, deviceType } from 'examples/context';

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('application-canvas'));
window.focus();

const assets = {
    model: new Asset('Zorah chunk 001', 'container', {
        url: './assets/meshlets/zorah/chunk_003.streamed.glb'
    }),
    orbit: new Asset('script', 'script', { url: './scripts/camera/orbit-camera.js' })
};

const device = await createGraphicsDevice(canvas, { deviceTypes: [deviceType], antialias: false });
device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);

const createOptions = new AppOptions();
createOptions.graphicsDevice = device;
createOptions.mouse = new Mouse(document.body);
createOptions.touch = new TouchDevice(document.body);
createOptions.componentSystems = [
    CameraComponentSystem,
    LightComponentSystem,
    ScriptComponentSystem,
    MeshletComponentSystem
];
createOptions.resourceHandlers = [ContainerHandler, ScriptHandler];

const app = new AppBase(canvas);
app.init(createOptions);

app.setCanvasFillMode(FILLMODE_FILL_WINDOW);
app.setCanvasResolution(RESOLUTION_AUTO);

const resize = () => app.resizeCanvas();
window.addEventListener('resize', resize);
app.on('destroy', () => {
    window.removeEventListener('resize', resize);
});

const assetListLoader = new AssetListLoader(Object.values(assets), app.assets);
await new Promise((resolve) => {
    assetListLoader.load(resolve);
});

// This chunk needs ~279 MiB of metadata, ~126 MiB of root pages and ~347 MiB of
// root draw indices alone. Reserve room for the cut tables, records and streamed detail too.
app.systems.meshlet.poolBytes = 2048 * 1024 * 1024;
const model = new Entity('Zorah chunk 001');
model.addComponent('meshlet', { asset: assets.model });
app.root.addChild(model);

const light = new Entity('Sun');
light.addComponent('light', {
    type: 'directional',
    color: Color.WHITE,
    intensity: 1.5
});
light.setEulerAngles(45, 30, 0);
app.root.addChild(light);

app.scene.ambientLight = new Color(0.25, 0.27, 0.3);

const camera = new Entity('Camera');
camera.addComponent('camera', {
    clearColor: new Color(0.12, 0.14, 0.18),
    nearClip: 0.01,
    farClip: 1000
});
camera.addComponent('script');
camera.script.create('orbitCamera', {
    attributes: {
        inertiaFactor: 0.2,
        distanceMin: 0.05,
        distanceMax: 0,
        frameOnStart: false
    }
});
camera.script.create('orbitCameraInputMouse');
camera.script.create('orbitCameraInputTouch');
app.root.addChild(camera);

// CameraFrame supplies the scene-depth colour attachment when occlusion is enabled.
// Single-sample rendering keeps that path available; no separate depth prepass is requested.
const cameraFrame = new CameraFrame(app, camera.camera);
cameraFrame.rendering.samples = 1;
cameraFrame.bloom.enabled = false;
cameraFrame.update();
app.on('destroy', () => cameraFrame.destroy());

// debug colour toggle: mode 2 tints every meshlet cluster its own colour, 0 restores the lit
// material. Re-applied every frame so it also survives world rebuilds.
data.set('data', { meshletColours: false, occlusion: false });
data.on('data.occlusion:set', (value) => {
    app.systems.meshlet.occlusion = value;
});

// The orbit script only discovers render components. Frame the meshlet world's transformed
// bounds once the component system has built it, so the camera fits the full chunk.
let framed = false;
app.on('framerender', () => {
    const world = app.systems.meshlet.director?.world;
    if (!world?.finalized) {
        return;
    }
    world.setColorMode(data.get('data.meshletColours') ? 2 : 0);
    if (!framed) {
        const bounds = world.worldBounds;
        const radius = bounds.halfExtents.length();
        const distance = radius / Math.sin(camera.camera.fov * Math.PI / 360);
        const position = new Vec3(0.5, 0.3, 1).normalize().mulScalar(distance).add(bounds.center);
        camera.camera.farClip = Math.max(distance * 4, 100);
        // @ts-ignore
        camera.script.orbitCamera.resetAndLookAtPoint(position, bounds.center);
        framed = true;
    }
});

app.start();
