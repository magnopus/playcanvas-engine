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
    SHADERPASS_WORLDNORMAL,
    ScriptComponentSystem,
    StandardMaterial,
    TouchDevice,
    Vec3,
    createGraphicsDevice
} from 'playcanvas';
import { CameraControls } from 'playcanvas/scripts/esm/camera-controls.mjs';

import { data, deviceType } from 'examples/context';

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('application-canvas'));
window.focus();

const BASE = './assets/meshlets/caldera/';

/**
 * @typedef {{ name: string, label: string, url: string, baked: boolean, collision?: string,
 * min: number[], max: number[] }} Region
 */
/** @type {{ regions: Region[] }} */
const manifest = await fetch(`${BASE}caldera_scene.json`).then(r => r.json());
const regions = manifest.regions.filter(r => r.baked);

const device = await createGraphicsDevice(canvas, { deviceTypes: [deviceType], antialias: false });
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
createOptions.resourceHandlers = [ContainerHandler];

const app = new AppBase(canvas);
app.init(createOptions);

app.setCanvasFillMode(FILLMODE_FILL_WINDOW);
app.setCanvasResolution(RESOLUTION_AUTO);

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
app.systems.meshlet.poolBytes = 2048 * 1024 * 1024;
app.systems.meshlet.occlusion = true;

const scene = new Entity('Caldera');
app.root.addChild(scene);

// regions arrive independently; the meshlet world rebuilds once their loads go quiet
for (const region of regions) {
    loadContainer(region.label, BASE + region.url).then((asset) => {
        const entity = new Entity(region.label);
        entity.addComponent('meshlet', { asset });
        scene.addChild(entity);
    }).catch((err) => {
        console.error(`Caldera: failed to load ${region.url}`, err);
    });
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

let statFrames = 0;
app.on('framerender', () => {
    const director = app.systems.meshlet.director;
    const world = director?.world;
    if (!world?.finalized) return;
    const visualization = data.get('data.visualization');
    world.setColorMode(visualization === 'meshlet' ? 2 : visualization === 'lod' ? 1 : 0);
    const view = director.views.get(camera.camera);
    if (view?.lastDemand && ++statFrames % 30 === 0) {
        const tris = Math.round(view.lastDemand.indices.reduce((a, b) => a + b, 0) / 3);
        const res = director.residency;
        data.set('data.stats', `${view.renderedMeshlets} meshlets, cut ${(tris / 1e6).toFixed(2)}M tris, ` +
            `${res ? res.residentPages : 0}/${world.totalPages} pages`);
    }
});

app.start();
