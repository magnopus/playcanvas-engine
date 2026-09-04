// @config
//
// The smallest possible meshlet setup: load a baked GLB as a regular container asset, add a
// meshlet component with that asset, done. The engine streams geometry pages on demand over
// HTTP Range requests and culls / LOD-selects the clusters on the GPU every frame - there is
// nothing else to wire up. A directional light and an orbit camera complete the scene.
//
// @flag WEBGL_DISABLED

import {
    AppBase,
    AppOptions,
    Asset,
    AssetListLoader,
    CameraComponentSystem,
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
    bunny: new Asset('bunny', 'container', { url: './assets/meshlets/bunny-v2.glb' }),
    orbit: new Asset('script', 'script', { url: './scripts/camera/orbit-camera.js' })
};

const device = await createGraphicsDevice(canvas, { deviceTypes: [deviceType] });
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

app.start();

// the bunny: a container asset baked with the meshlet extension, one component call
const bunny = new Entity('Bunny');
bunny.addComponent('meshlet', { asset: assets.bunny });
app.root.addChild(bunny);

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
    farClip: 100
});
camera.addComponent('script');
camera.script.create('orbitCamera', {
    attributes: {
        inertiaFactor: 0.2,
        distanceMin: 0.05,
        distanceMax: 5,
        frameOnStart: false
    }
});
camera.script.create('orbitCameraInputMouse');
camera.script.create('orbitCameraInputTouch');
app.root.addChild(camera);

// pivot on the bunny's centre explicitly: the orbit script frames entities by walking their
// render components, and a meshlet component does not expose one
// @ts-ignore
camera.script.orbitCamera.resetAndLookAtPoint(new Vec3(0.32, 0.28, 0.32), new Vec3(0, 0.12, 0));

// debug colour toggle: mode 2 tints every meshlet cluster its own colour, 0 restores the lit
// material. Re-applied every frame so it also survives world rebuilds.
data.set('data', { meshletColours: false });
app.on('framerender', () => {
    app.systems.meshlet.director?.world.setColorMode(data.get('data.meshletColours') ? 2 : 0);
});
