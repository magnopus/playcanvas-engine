// @config
//
// Load a local test asset as a container and stream its
// geometry pages on demand. Fly (WASD + mouse look, Shift/Ctrl for speed) or orbit and pan to inspect
// residency transitions; toggle occlusion to exercise CameraFrame's scene-depth attachment and
// two-phase HZB path.
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
    Keyboard,
    LightComponentSystem,
    MeshletComponentSystem,
    Mouse,
    RenderComponentSystem,
    RESOLUTION_AUTO,
    SHADERPASS_ALBEDO,
    SHADERPASS_METALNESS,
    SHADERPASS_ROUGHNESS,
    SHADERPASS_WORLDNORMAL,
    ScriptComponentSystem,
    ScriptHandler,
    TextureHandler,
    TouchDevice,
    Vec3,
    basisInitialize,
    createGraphicsDevice
} from 'playcanvas';
import { CameraControls } from 'playcanvas/scripts/esm/camera-controls.mjs';

import { data, deviceType } from 'examples/context';

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('application-canvas'));
window.focus();

const models = {
    bunny: { name: 'Bunny', url: './assets/meshlets/bunny-v2.glb' },
    zorah: { name: 'Zorah chunk 003', url: './assets/meshlets/zorah/chunk_003.streamed.glb' },
    magoffice: { name: 'MagOffice', url: './assets/meshlets/magoffice_streamed.glb' }
};
const requestedAsset = new URLSearchParams(location.search).get('asset') ?? '';
const assetName = Object.hasOwn(models, requestedAsset) ? requestedAsset : 'bunny';
const selectedModel = models[assetName];

// The office has streamed KTX2 textures and inline transparent meshes.
basisInitialize({
    glueUrl: './assets/wasm/basis/basis.wasm.js',
    wasmUrl: './assets/wasm/basis/basis.wasm.wasm',
    fallbackUrl: './assets/wasm/basis/basis.js'
});

const assets = {
    model: new Asset(selectedModel.name, 'container', { url: selectedModel.url })
};

const device = await createGraphicsDevice(canvas, { deviceTypes: [deviceType], antialias: false });
device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);

const createOptions = new AppOptions();
createOptions.graphicsDevice = device;
createOptions.mouse = new Mouse(document.body);
createOptions.keyboard = new Keyboard(window);
createOptions.touch = new TouchDevice(document.body);
createOptions.componentSystems = [
    CameraComponentSystem,
    LightComponentSystem,
    RenderComponentSystem,
    ScriptComponentSystem,
    MeshletComponentSystem
];
createOptions.resourceHandlers = [ContainerHandler, ScriptHandler, TextureHandler];

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
await new Promise((resolve, reject) => {
    assetListLoader.load((error) => (error ? reject(error) : resolve()));
});

// Zorah chunk 003 needs ~279 MiB of metadata, ~126 MiB of root pages and ~347 MiB of
// root draw indices alone. Reserve room for the cut tables, records and streamed detail too.
app.systems.meshlet.poolBytes = 2048 * 1024 * 1024;
const model = new Entity(selectedModel.name);
model.addComponent('meshlet', { asset: assets.model });
// The parser skips streamed placeholders on the regular path. Instantiate the remaining
// geometry too: alpha-blended windows stay inline so the forward renderer can sort them.
if (assets.model.resource.renders.some((asset) => asset.resource.meshes.length > 0)) {
    model.addChild(assets.model.resource.instantiateRenderEntity());
}
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
app.root.addChild(camera);
const cc = /** @type {CameraControls} */ (camera.script.create(CameraControls));

// CameraFrame supplies the scene-depth colour attachment when occlusion is enabled.
// Single-sample rendering keeps that path available; no separate depth prepass is requested.
const cameraFrame = new CameraFrame(app, camera.camera);
cameraFrame.rendering.samples = 1;
cameraFrame.bloom.enabled = false;
cameraFrame.update();
app.on('destroy', () => cameraFrame.destroy());

const shaderPasses = {
    normals: SHADERPASS_WORLDNORMAL,
    albedo: SHADERPASS_ALBEDO,
    metalness: SHADERPASS_METALNESS,
    roughness: SHADERPASS_ROUGHNESS
};
// Two-phase HZB occlusion builds its pyramid from the CameraFrame's single-sample scene depth.
app.systems.meshlet.occlusion = true;
data.set('data', { asset: assetName, visualization: 'material', occlusion: true, threshold: 1, stats: '' });
data.on('data.asset:set', (value) => {
    if (value === assetName || !Object.hasOwn(models, value)) return;
    // Reload releases the previous world's page and texture pools before loading another asset.
    const url = new URL(location.href);
    url.searchParams.set('asset', value);
    location.assign(url.href);
});
data.on('data.visualization:set', (value) => {
    camera.camera.setShaderPass(shaderPasses[value]);
});
data.on('data.occlusion:set', (value) => {
    app.systems.meshlet.occlusion = value;
});
data.on('data.threshold:set', (value) => {
    app.systems.meshlet.dagPixelThreshold = value;
});

// Frame the meshlet world's transformed bounds once the component system has built it, so the
// camera fits the full chunk, and scale the fly speeds and near plane to its size.
let framed = false;
let statFrames = 0;
app.on('framerender', () => {
    const director = app.systems.meshlet.director;
    const world = director?.world;
    if (!world?.finalized) {
        return;
    }
    // The view reads its counters back each frame to size its buffers; reuse them. The index
    // demand covers the whole LOD cut before occlusion, so it does not change when occlusion is
    // toggled - the drawn meshlet count does.
    const view = director.views.get(camera.camera);
    if (view?.lastDemand && ++statFrames % 30 === 0) {
        const tris = Math.round(view.lastDemand.indices.reduce((a, b) => a + b, 0) / 3);
        const res = director.residency;
        data.set('data.stats', `${view.renderedMeshlets} meshlets drawn, cut ${(tris / 1e6).toFixed(2)}M tris, ` +
            `${res ? res.residentPages : 0}/${world.totalPages} pages`);
    }
    const visualization = data.get('data.visualization');
    world.setColorMode(visualization === 'meshlet' ? 2 : visualization === 'lod' ? 1 : 0);
    if (!framed) {
        const bounds = world.worldBounds;
        const radius = bounds.halfExtents.length();
        const distance = radius / Math.sin((camera.camera.fov * Math.PI) / 360);
        const position = new Vec3(0.5, 0.3, 1).normalize().mulScalar(distance).add(bounds.center);
        camera.camera.farClip = Math.max(distance * 4, 100);
        // A fixed 0.01 near plane leaves the depth buffer too coarse at this scale: near-coplanar
        // surfaces quantize to identical depths, and the order of the indirect draw (atomically
        // allocated, so it changes every frame) then decides the winner, which reads as flicker
        // on a static camera. Keep the far/near ratio modest instead.
        camera.camera.nearClip = Math.max(radius * 0.001, 0.01);
        Object.assign(cc, {
            moveSpeed: radius * 0.1,
            moveFastSpeed: radius * 0.5,
            moveSlowSpeed: radius * 0.02
        });
        cc.reset(bounds.center, position);
        framed = true;
    }
});

app.start();
