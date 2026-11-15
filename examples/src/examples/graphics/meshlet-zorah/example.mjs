// @config
//
// The Zorah courtyard (NVIDIA RTX Kit sample geometry), streamed through the GPU-driven meshlet
// pipeline. The export carries geometry only, no textures, so the look comes from the lighting: a
// low warm sun with four cascaded shadows, a procedural sky for ambient light, contact ambient
// occlusion, and a stone / foliage / metal palette picked from the material names. See
// assets/meshlets/README.md for the local dataset setup.
//
// Fly with WASD and mouse look; Shift/Ctrl changes speed.
//   ?scene=all|22m|default manifest in the bake folder (default: all 19 chunks; `default` is the
//                          5-chunk scene.json, for machines with less memory)
//   ?pool=<MiB>            geometry budget override
//   ?tint=0                leave the materials as baked (white)
//   ?mountains=A,B,E|none  which background terrain pieces to keep (default all but the F1 sheets)
//   ?view=start            begin at the manifest's start pose instead of the hero view
//   ?scale=<0.25..1>       scene render scale, TAA upscales to the output (default 0.6)
//
// @flag WEBGL_DISABLED
// @flag WEBGPU_BARE_DISABLED
// @flag HIDDEN

import {
    AppBase,
    AppOptions,
    Asset,
    AssetListLoader,
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
    Picker,
    RenderComponentSystem,
    RESOLUTION_AUTO,
    SSAOTYPE_COMBINE,
    BoundingBox,
    SHADOWUPDATE_REALTIME,
    SHADOWUPDATE_THISFRAME,
    SSAOTYPE_NONE,
    ScriptComponentSystem,
    TONEMAP_ACES2,
    TONEMAP_NONE,
    TouchDevice,
    Vec3,
    createGraphicsDevice,
    math
} from 'playcanvas';
import { CameraControls } from 'playcanvas/scripts/esm/camera-controls.mjs';
import { ProceduralSky } from 'playcanvas/scripts/esm/sky/procedural-sky.mjs';

import { data, deviceType } from 'examples/context';

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('application-canvas'));
window.focus();

const params = new URLSearchParams(location.search);
const sceneName = params.get('scene');
const manifestFile = sceneName === 'default' ? 'scene.json' : `scene.${(sceneName ?? 'all').replace(/[^\w.-]/g, '')}.json`;
const manifestUrl = new URL(`./assets/meshlets/zorah/${manifestFile}`, document.baseURI);
const response = await fetch(manifestUrl);
if (!response.ok) {
    throw new Error(`Cannot load Zorah (${response.status}) from ${manifestUrl.pathname}; see assets/meshlets/README.md.`);
}
const scene = await response.json();
if (scene.version !== 1 || !scene.entities?.length) {
    throw new Error('Invalid Zorah scene manifest.');
}

// The NVIDIA sample's own camera and sun (zorah_main_public.v2.cfg): a view down the courtyard
// into the low warm light
const HERO = { position: [55.34, 6.45, 0.32], target: [46.38, 5.8, 0.53] };
const SUN = { toSun: new Vec3(0.6, 0.7, 0.36).normalize(), color: new Color(1, 0.8, 0.5) };

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
options.resourceHandlers = [ContainerHandler];

const app = new AppBase(canvas);
app.init(options);
app.setCanvasFillMode(FILLMODE_FILL_WINDOW);
app.setCanvasResolution(RESOLUTION_AUTO);
const resize = () => app.resizeCanvas();
window.addEventListener('resize', resize);
app.on('destroy', () => window.removeEventListener('resize', resize));

// the manifests' budgets are for the bake's size alone: the full scene deferred thousands of
// instances at 10 GiB, so everything but the 5-chunk scene gets 16 GiB
const defaultPoolMiB = sceneName === 'default' ? scene.meshletPoolMiB ?? 4096 : 16384;
app.systems.meshlet.poolBytes = (Number(params.get('pool')) || defaultPoolMiB) * 1024 * 1024;
app.systems.meshlet.occlusion = true;
app.systems.meshlet.cssPixels = true;
app.systems.meshlet.deferRebuildWhileLoading = true;

// Manifest model URLs are relative to assets/meshlets, not the manifest directory.
const assetBase = new URL('../', manifestUrl);
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
    new AssetListLoader([...assets.values()], app.assets).load((error) => {
        if (error) reject(error);
        else resolve();
    });
});

/**
 * Small deterministic variation per material name, so neighbouring parts that share a base
 * colour still read as separate pieces.
 *
 * @param {string} name - Material name.
 * @returns {number} A value in [-1, 1].
 */
const nameJitter = (name) => {
    let h = 2166136261;
    for (let i = 0; i < name.length; i++) h = Math.imul(h ^ name.charCodeAt(i), 16777619);
    return ((h >>> 0) % 2001) / 1000 - 1;
};

// [pattern, linear base colour, metallic, roughness, jitter]; the first match wins. Everything
// else is limestone. These are an art-directed guess from the names: the export has no textures.
const PALETTE = [
    [/Mirror|Emissive|Glass|StainedGlass|WaterMID/i, null],
    [/Gold|Gilt/i, [0.8, 0.5, 0.12], 1, 0.28, 0.1],
    [/Copper|RoofOxid/i, [0.1, 0.28, 0.22], 0.6, 0.5, 0.1],
    [/Mountains/i, [0.22, 0.19, 0.16], 0, 1, 0.1],
    [/Bush|Leaves|Leaf|Grass|Hedge|Flower|Plant|Palm|Dryopteris|Dypsis|Howea|Strelitzia|Moss/i, [0.045, 0.11, 0.03], 0, 0.7, 0.35],
    [/Bark|Trunk|Root|Branch|Stem|Firewood|Wood/i, [0.1, 0.065, 0.04], 0, 0.85, 0.2],
    [/Soil|Dirt|Dry|Mud/i, [0.07, 0.05, 0.035], 0, 1, 0.15],
    [/Metal|Iron|Grate|Rail|Gate|Frame|Lamp|Armour|Spitzer|Jensen/i, [0.035, 0.035, 0.04], 0.85, 0.42, 0.1],
    [/DarkGrey/i, [0.06, 0.06, 0.065], 0, 0.35, 0.05],
    [/StoneFloor|Tile|Floor/i, [0.26, 0.21, 0.16], 0, 0.75, 0.08],
    [/Marble/i, [0.42, 0.4, 0.37], 0, 0.3, 0.05],
    [/Roof|Ceiling|Cupola|Dome/i, [0.24, 0.19, 0.14], 0, 0.7, 0.12],
    [/Candle|Incense|Tarp|Awning|Cloth|Banner|Cape|Ribbon|Kite|Flag/i, [0.4, 0.09, 0.07], 0, 0.9, 0.3]
];
const HIDDEN = /EasyFog|Subuvreflections/i;
const LIMESTONE = [0.3, 0.245, 0.18];

/**
 * Writes the palette into a resource's baked material table, before the world uploads it.
 *
 * @param {object} resource - The container's meshlet resource and glTF data.
 * @param {object} resource.meshlets - Parsed meshlet resources.
 * @param {object} resource.data - The parsed glTF.
 */
const tintMaterials = (resource) => {
    const meshlet = resource.meshlets?.[0];
    const table = meshlet?.materialTable;
    const materials = resource.data?.materials;
    // record layout (MATERIAL_RECORD): rgba base colour at 0, metallic at 8, roughness at 9
    if (!table || !materials || materials.length < meshlet.materialCount) {
        console.warn('Zorah: cannot tint', { table: !!table, materials: materials?.length, count: meshlet?.materialCount });
        return;
    }
    const f32 = new Float32Array(table.buffer, table.byteOffset, table.length);
    for (let i = 0; i < meshlet.materialCount; i++) {
        const name = materials[i]?.name ?? '';
        const record = i * 32;
        // fog and reflection cards are effects for the sample's own renderer, and draw as opaque
        // white sheets here: alpha 0 under an alpha mask discards them
        if (HIDDEN.test(name)) {
            f32[record + 3] = 0;
            table[record + 11] |= 2; // MATERIAL_FLAG_ALPHA_MASK
            continue;
        }
        const hit = PALETTE.find(([pattern]) => pattern.test(name));
        if (hit && !hit[1]) continue;
        const [, color = LIMESTONE, metallic = 0, roughness = 0.8, jitter = 0.08] = hit ?? [];
        // decals sit on other surfaces, so they take the surface colour exactly
        const k = /Decal/i.test(name) ? 1 : 1 + jitter * nameJitter(name);
        f32[record] = color[0] * k;
        f32[record + 1] = color[1] * k;
        f32[record + 2] = color[2] * k;
        f32[record + 8] = metallic;
        f32[record + 9] = roughness;
    }
};
// The background terrain pieces (MI_Mountains_Background_<L>1) are 13 km tiles. The two F1 tiles are
// thin sheets whose surface cuts through the courtyard, burying its floor and the lower walls, so
// they are dropped. ?mountains=A,B,E keeps only those letters, ?mountains=none drops them all
const keepMountains = new Set((params.get('mountains') ?? 'A,B,C,D,E,G').toUpperCase().split(','));
for (const asset of assets.values()) {
    const resource = asset.resource.meshlets?.[0];
    const materials = asset.resource.data?.materials;
    if (!resource || !materials) continue;
    const letters = resource.primitives.map(p => /Mountains_Background_([A-Z])/i.exec(materials[p.materialIndex]?.name ?? '')?.[1]?.toUpperCase());
    resource.instances = resource.instances.filter((inst) => {
        const letter = letters[inst.primIndex];
        return !letter || keepMountains.has(letter);
    });
}
if (params.get('tint') !== '0') {
    for (const asset of assets.values()) tintMaterials(asset.resource);
}

/**
 * The blended surfaces draw through the ordinary renderer with the export's bare glTF material,
 * an opaque white. Fog and reflection cards are dropped, and water and glass get a translucent
 * look.
 *
 * @param {Entity} root - The instantiated render entity.
 */
const styleInlineSurfaces = (root) => {
    const seen = new Set();
    for (const render of root.findComponents('render')) {
        const instances = render.meshInstances;
        if (instances.length && instances.every(mi => HIDDEN.test(mi.material.name))) {
            render.enabled = false;
            continue;
        }
        for (const { material } of instances) {
            if (seen.has(material)) continue;
            seen.add(material);
            const water = /Water/i.test(material.name);
            if (!water && !/Glass/i.test(material.name)) continue;
            material.diffuse = water ? new Color(0.02, 0.12, 0.15) : new Color(0.7, 0.8, 0.85);
            material.metalness = 0;
            material.gloss = 0.95;
            material.opacity = water ? 0.75 : 0.25;
            material.blendType = BLEND_NORMAL;
            material.depthWrite = false;
            material.update();
        }
    }
};

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
            const inline = asset.resource.instantiateRenderEntity();
            styleInlineSurfaces(inline);
            child.addChild(inline);
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

// The scene's extent from the manifest, leaving out the kilometre-wide background terrain. The
// static shadow covers exactly this box, so it is one fixed map whatever the camera does.
const sceneBounds = new BoundingBox();
let anyBounds = false;
for (const entry of scene.entities) {
    const { min, max } = entry.bounds ?? {};
    if (!min || Math.max(max[0] - min[0], max[2] - min[2]) > 500) continue;
    const box = new BoundingBox(new Vec3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2),
        new Vec3((max[0] - min[0]) / 2, (max[1] - min[1]) / 2, (max[2] - min[2]) / 2));
    if (anyBounds) sceneBounds.add(box);
    else sceneBounds.copy(box);
    anyBounds = true;
}

// The sun: a low warm key light. Its shadows carry most of the look on untextured geometry. By
// default they are one 8k map fitted to the scene bounds and rendered once, then again only when
// streaming changes what the meshlets cast (see MeshletShadowRenderer): nothing about it follows
// the camera, so there is nothing to swim or flicker as it moves. The realtime alternative is four
// camera-fitted cascades.
const sun = new Entity('Sun');
sun.addComponent('light', {
    type: 'directional',
    color: SUN.color,
    intensity: 1.7,
    castShadows: true,
    numCascades: 1,
    shadowResolution: 8192,
    shadowDistance: 300,
    shadowBias: 0.2,
    normalOffsetBias: 0.05,
    shadowBounds: anyBounds ? sceneBounds : null,
    shadowUpdateMode: SHADOWUPDATE_THISFRAME
});
// a directional light shines down its local -Y, so +Y points at the sun
sun.setRotation(sun.getRotation().setFromDirections(Vec3.UP, SUN.toSun));
app.root.addChild(sun);
app.systems.meshlet.shadows = true;

const camera = new Entity('Camera');
camera.addComponent('camera', {
    clearColor: new Color(0.45, 0.6, 0.8),
    fov: 60,
    nearClip: 0.05,
    farClip: scene.farClip
});
camera.addComponent('script');
app.root.addChild(camera);
const controls = /** @type {CameraControls} */ (camera.script.create(CameraControls));
Object.assign(controls, { moveSpeed: 4, moveFastSpeed: 20, moveSlowSpeed: 1 });

/**
 * @param {number[]} position - Eye position.
 * @param {number[]} target - Point to look at.
 */
const lookFrom = (position, target) => {
    const eye = new Vec3(...position);
    camera.setPosition(eye);
    camera.lookAt(new Vec3(...target));
    controls.reset(new Vec3(...target), eye);
};
if (params.get('view') === 'start') {
    const start = new Vec3(...scene.start.position);
    lookFrom(scene.start.position, start.clone().add(new Vec3(0, 0, -10)).toArray());
} else {
    lookFrom(HERO.position, HERO.target);
}

// Cameraframe supplies the scene-depth attachment the two-phase occlusion reads, and the
// compose-pass effects that stand in for the missing material detail.
const cameraFrame = new CameraFrame(app, camera.camera);
cameraFrame.rendering.samples = 1;
// an HDR canvas takes the scene values as they are, tone mapping would clip them
cameraFrame.rendering.toneMapping = device.isHdr ? TONEMAP_NONE : TONEMAP_ACES2;
cameraFrame.bloom.enabled = false;
cameraFrame.taa.enabled = true;
// The scene renders at a fraction of the output and TAA accumulates the jittered frames back to full
// resolution, so the meshlet raster, the lit shading and the HZB all run on fewer pixels (the DAG
// cut measures against the scaled scene target too). ?scale=1 renders at full resolution.
const renderScale = math.clamp(parseFloat(params.get('scale') ?? '0.6') || 0.6, 0.25, 1);
cameraFrame.rendering.renderTargetScale = renderScale;
cameraFrame.taa.upscale = renderScale < 1;
cameraFrame.rendering.sharpness = renderScale < 1 ? 0.5 : 0;
// contact shadows in the corners and under ledges, from the depth the meshlets write
// off by default: at the export's coarse LODs and without textures it reads as dark smudges around
// carved detail. ?ssao=1 starts with it on
const ssaoOn = params.get('ssao') === '1';
Object.assign(cameraFrame.ssao, {
    type: ssaoOn ? SSAOTYPE_COMBINE : SSAOTYPE_NONE,
    radius: 0.6,
    intensity: 0.45,
    samples: 16,
    scale: 1
});
cameraFrame.update();
app.on('destroy', () => cameraFrame.destroy());

// A procedural sky lights the scene through its env-atlas, in place of a flat ambient, and puts
// the sun disk where the light shines from.
const sky = new Entity('ProceduralSky');
sky.addComponent('script');
const skyScript = /** @type {ProceduralSky} */ (sky.script.create(ProceduralSky));
skyScript.elevation = Math.asin(SUN.toSun.y) * math.RAD_TO_DEG;
skyScript.azimuth = Math.atan2(SUN.toSun.x, SUN.toSun.z) * math.RAD_TO_DEG;
skyScript.luminance = 0.2;
app.root.addChild(sky);
app.scene.ambientLight = new Color(0, 0, 0);

// orbit pivot: the world point under the screen centre, picked once when an orbit drag starts.
// The pick runs from postrender, as meshlet draw args are cleared at the end of each frame.
const picker = new Picker(app, Math.round(device.width * 0.5), Math.round(device.height * 0.5), true);
const pivotTarget = new Vec3();
let pickRequested = false;
let pickInFlight = false;
window.addEventListener('pointerdown', (e) => {
    if (e.button === 0 || e.button === 1) pickRequested = true;
});
app.on('destroy', () => picker.destroy());
app.on('postrender', () => {
    if (!pickRequested || pickInFlight) return;
    pickRequested = false;
    pickInFlight = true;
    picker.prepare(camera.camera, app.scene, [app.scene.layers.getLayerByName('World')]);
    picker.getWorldPointAsync(picker.width * 0.5, picker.height * 0.5).then((point) => {
        if (point) {
            pivotTarget.copy(point);
            controls.focusPoint = pivotTarget;
        }
    }).finally(() => {
        pickInFlight = false;
    });
});

data.set('data', {
    threshold: 4,
    sunIntensity: sun.light.intensity,
    skyLuminance: skyScript.luminance,
    shadows: true,
    staticShadows: true,
    ssao: ssaoOn,
    taa: true,
    renderScale,
    occlusion: true,
    occlusionAdmission: true,
    teleport: 'hero',
    stats: '',
    statsRoots: '',
    statsDemand: '',
    statsPressure: '',
    statsPages: ''
});
app.systems.meshlet.dagPixelThreshold = data.get('data.threshold');

data.on('data.threshold:set', (/** @type {number} */ value) => {
    app.systems.meshlet.dagPixelThreshold = value;
});
data.on('data.sunIntensity:set', (/** @type {number} */ value) => {
    sun.light.intensity = value;
});
data.on('data.skyLuminance:set', (/** @type {number} */ value) => {
    skyScript.luminance = value;
});
// Static: one 8k map fitted to the scene, cached. Realtime: four cascades fitted to the camera,
// the far two on alternate frames.
const setShadowMode = (/** @type {boolean} */ isStatic) => {
    const light = sun.light;
    light.shadowResolution = isStatic ? 8192 : 2048;
    light.numCascades = isStatic ? 1 : 4;
    light.cascadeUpdateIntervals = isStatic ? null : [1, 1, 2, 2];
    light.shadowBounds = isStatic && anyBounds ? sceneBounds : null;
    light.shadowUpdateMode = isStatic ? SHADOWUPDATE_THISFRAME : SHADOWUPDATE_REALTIME;
};
data.on('data.shadows:set', (/** @type {boolean} */ value) => {
    app.systems.meshlet.shadows = value;
    sun.light.castShadows = value;
    if (value) setShadowMode(data.get('data.staticShadows'));
});
data.on('data.staticShadows:set', setShadowMode);
data.on('data.ssao:set', (/** @type {boolean} */ value) => {
    cameraFrame.ssao.type = value ? SSAOTYPE_COMBINE : SSAOTYPE_NONE;
    cameraFrame.update();
});
// upscaling is part of the TAA resolve, so without TAA a reduced scale is a bilinear stretch
const applyResolution = () => {
    const taa = data.get('data.taa');
    const scale = data.get('data.renderScale');
    cameraFrame.taa.enabled = taa;
    cameraFrame.taa.upscale = taa && scale < 1;
    cameraFrame.rendering.renderTargetScale = scale;
    cameraFrame.rendering.sharpness = scale < 1 ? 0.5 : 0;
    cameraFrame.update();
};
data.on('data.taa:set', applyResolution);
data.on('data.renderScale:set', applyResolution);
data.on('data.occlusion:set', (/** @type {boolean} */ value) => {
    app.systems.meshlet.occlusion = value;
});
data.on('data.occlusionAdmission:set', (/** @type {boolean} */ value) => {
    app.systems.meshlet.occlusionAdmission = value;
});
data.on('data.teleport:set', (/** @type {string} */ name) => {
    if (name === 'hero') {
        lookFrom(HERO.position, HERO.target);
        return;
    }
    const spot = scene.teleports?.find(t => t.name === name);
    if (!spot) return;
    // keep looking along the current heading from the new spot
    const forward = camera.forward.clone();
    lookFrom(spot.position, new Vec3(...spot.position).add(forward).toArray());
});

// Budget readout, once a second, summed over the meshlet worlds. It shows which limit a coarse
// view is running into: instances the root budget rejected (missing), the cut's index demand
// against the budget (LOD pressure coarsens everything), or pages still streaming in. Occlusion
// does not reduce the first two: roots are admitted, and demand counted, before the occlusion test.
const count = (/** @type {number} */ n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : `${n}`);
const sum3 = (/** @type {number[]} */ a) => a[0] + a[1] + a[2];
let statsAt = 0;
app.on('update', () => {
    const time = performance.now();
    if (time - statsAt < 1000) return;
    statsAt = time;
    const budget = {
        worlds: 0,
        admitted: 0,
        rejected: 0,
        rejectedInstances: 0,
        unready: 0,
        candidates: 0,
        subpixel: 0,
        occluded: 0,
        demand: 0,
        capacity: 0,
        ceiling: 0,
        pressure: 1,
        maxPressure: 1,
        drawn: 0,
        resident: 0,
        slots: 0,
        missing: 0,
        rootShortfall: 0,
        /** @type {object[]} */
        perWorld: []
    };
    for (const director of app.systems.meshlet.directors) {
        const view = director.views?.values().next().value;
        if (!view) continue;
        const roots = view.lastRoots;
        budget.worlds++;
        budget.admitted += sum3(roots.admitted);
        budget.rejected += sum3(roots.rejected);
        budget.rejectedInstances += roots.rejectedInstances;
        budget.unready += roots.unready;
        budget.candidates += roots.candidates;
        budget.subpixel += roots.subpixel;
        budget.occluded += roots.occluded;
        budget.demand += view.lastDemand ? sum3(view.lastDemand.indices) : 0;
        budget.capacity += view.indexTotal();
        budget.ceiling += view.indexShare > 0 ? view.indexShare : director.world.indexCeiling;
        budget.pressure = Math.max(budget.pressure, director.budget.pressureScale);
        budget.maxPressure = director.budget.maxScale;
        budget.drawn += view.renderedMeshlets ?? 0;
        budget.resident += director.residency?.residentPages ?? 0;
        budget.slots += director.world.poolSlots ?? 0;
        budget.missing += director.residency?.lastMissingWanted ?? 0;
        budget.rootShortfall += director.residency?.rootShortfallPages ?? 0;
        budget.perWorld.push({
            meshlets: director.world.totalMeshlets,
            instances: director.world.instanceCount,
            inView: roots.candidates,
            occluded: roots.occluded,
            admitted: sum3(roots.admitted),
            rejected: sum3(roots.rejected),
            demand: view.lastDemand ? sum3(view.lastDemand.indices) : 0,
            ceiling: view.indexShare > 0 ? view.indexShare : director.world.indexCeiling,
            buffers: view.indexTotal(),
            pressure: director.budget.pressureScale,
            drawn: view.renderedMeshlets ?? 0
        });
    }
    // for reading from the console and profiling scripts: window.__zorahBudget / __zorahData
    window.__zorahBudget = budget;
    window.__zorahData = data;
    window.__zorahCameraFrame = cameraFrame;
    const threshold = app.systems.meshlet.dagPixelThreshold;
    data.set('data.statsRoots', `admitted ${count(budget.admitted)} idx, rejected ${count(budget.rejected)} idx ` +
        `(${budget.rejectedInstances} instances missing), unready ${budget.unready}, skipped as occluded ${budget.occluded}`);
    data.set('data.statsDemand', `cut ${count(budget.demand)} of ${count(budget.ceiling)} idx ` +
        `(${(budget.demand / Math.max(budget.ceiling, 1)).toFixed(2)}x), buffers ${count(budget.capacity)}`);
    data.set('data.statsPressure', `${budget.pressure.toFixed(2)}x (error ${(threshold * budget.pressure).toFixed(1)} px of ${threshold})`);
    data.set('data.statsPages', `${count(budget.resident)} of ${count(budget.slots)} resident, ${budget.missing} missing, ` +
        `${budget.rootShortfall} root short`);
    data.set('data.stats', `${count(budget.drawn)} meshlets drawn, ${count(budget.candidates)} instances in view ` +
        `(${count(budget.subpixel)} too small), ${budget.worlds} worlds`);
});

app.start();
