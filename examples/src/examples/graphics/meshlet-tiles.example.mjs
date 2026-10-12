// @config
//
// Isolated source geometry from Zorah's marble floor. Select one of the four pieces,
// orbit and zoom to inspect the geometric relief before testing coarse replacements.
// Source geometry has no textures or normals; the importer generates normals.
// See assets/meshlets/README.md for extraction and placement metadata.
//
// @flag HIDDEN
// @flag NO_MINISTATS

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
    Mouse,
    RenderComponentSystem,
    RESOLUTION_AUTO,
    ScriptComponentSystem,
    ScriptHandler,
    StandardMaterial,
    TextureHandler,
    TouchDevice,
    Vec3,
    createGraphicsDevice
} from 'playcanvas';

import { deviceType } from 'examples/context';

const params = new URLSearchParams(location.search);
const tile = ['a', 'b', 'c', 'd'].includes(params.get('tile')) ? params.get('tile') : 'a';
const directory = './assets/meshlets/zorah-tiles/';
const response = await fetch(`${directory}extraction.json`);
if (!response.ok) throw new Error('Extract the Zorah tiles first; see assets/meshlets/README.md.');
const report = await response.json();
const entry = report.assets.find((asset) => asset.file === `marble-${tile}.glb`);

const toolbar = document.createElement('div');
toolbar.style.cssText =
    'position:fixed;z-index:10;top:12px;left:12px;right:12px;padding:14px;background:#17202ee8;color:#e7edf5;font:14px system-ui;border-radius:8px;display:flex;align-items:center;gap:14px;flex-wrap:wrap';
toolbar.innerHTML = `<strong>Zorah · marble floor</strong>
    <label>Piece <select id="tile">${['a', 'b', 'c', 'd'].map((letter) => `<option value="${letter}">${letter.toUpperCase()}1</option>`).join('')}</select></label>
    <button id="top">Top view</button><button id="angle">Angled view</button>
    <span>${entry.triangles.toLocaleString()} source triangles · ${entry.placements} placements in Zorah</span>
    <small id="status">Loading original geometry…</small>`;
document.body.appendChild(toolbar);
const select = /** @type {HTMLSelectElement} */ (toolbar.querySelector('#tile'));
select.value = tile;
select.addEventListener('change', () => {
    const url = new URL(location.href);
    url.searchParams.set('tile', select.value);
    location.assign(url.href);
});

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('application-canvas'));
const device = await createGraphicsDevice(canvas, { deviceTypes: [deviceType], antialias: true });
device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);
const options = new AppOptions();
options.graphicsDevice = device;
options.mouse = new Mouse(canvas);
options.touch = new TouchDevice(canvas);
options.componentSystems = [CameraComponentSystem, LightComponentSystem, RenderComponentSystem, ScriptComponentSystem];
options.resourceHandlers = [ContainerHandler, TextureHandler, ScriptHandler];
const app = new AppBase(canvas);
app.init(options);
app.setCanvasFillMode(FILLMODE_FILL_WINDOW);
app.setCanvasResolution(RESOLUTION_AUTO);
const resize = () => app.resizeCanvas();
window.addEventListener('resize', resize);

const asset = new Asset(entry.name, 'container', {
    url: `${directory}${entry.file}`
});
const orbit = new Asset('orbit camera', 'script', { url: './scripts/camera/orbit-camera.js' });
await new Promise((resolve, reject) => {
    new AssetListLoader([asset, orbit], app.assets).load((error) => (error ? reject(error) : resolve()));
});
const model = asset.resource.instantiateRenderEntity();
app.root.addChild(model);
const renders = model.findComponents('render');
const bounds = renders[0].meshInstances[0].aabb.clone();
const material = new StandardMaterial();
material.diffuse = new Color(0.65, 0.68, 0.72);
material.useMetalness = true;
material.metalness = 0;
material.gloss = 0.3;
material.update();
for (const render of renders) {
    for (const meshInstance of render.meshInstances) {
        bounds.add(meshInstance.aabb);
        meshInstance.material = material;
    }
}
app.scene.ambientLight = new Color(0.2, 0.2, 0.2);
const light = new Entity('Sun');
light.addComponent('light', { type: 'directional', intensity: 1.8 });
light.setEulerAngles(35, 25, 0);
app.root.addChild(light);
const camera = new Entity('Camera');
camera.addComponent('camera', { clearColor: new Color(0.06, 0.08, 0.11), nearClip: 0.0001, farClip: 100 });
camera.addComponent('script');
camera.script.create('orbitCamera', {
    attributes: { inertiaFactor: 0.1, distanceMin: 0.005, frameOnStart: false }
});
camera.script.create('orbitCameraInputMouse');
camera.script.create('orbitCameraInputTouch');
app.root.addChild(camera);
const frame = (top) => {
    const distance = bounds.halfExtents.length() * 2.7;
    const offset = top ? new Vec3(0, 1, 0.0001) : new Vec3(0.25, 0.85, 0.5).normalize();
    camera.script.orbitCamera.resetAndLookAtPoint(offset.mulScalar(distance).add(bounds.center), bounds.center);
};
toolbar.querySelector('#top').addEventListener('click', () => frame(true));
toolbar.querySelector('#angle').addEventListener('click', () => frame(false));
resize();
frame(false);
toolbar.querySelector('#status').textContent = 'Original geometry · neutral material · drag to orbit / scroll to zoom';
document.body.dataset.tileLoaded = tile;
app.start();
app.on('destroy', () => {
    window.removeEventListener('resize', resize);
    material.destroy();
    toolbar.remove();
});
