// @config
//
// The Zorah courtyard (NVIDIA RTX Kit sample geometry) as streamed meshlets (WebGPU only). The
// export carries geometry only, no textures, so the look comes from the lighting: a shadowed sun,
// an HDRI sky for ambient light, SSAO, and a stone / foliage / metal palette picked from the
// material names. Fly with WASD and mouse look; Shift/Ctrl changes speed. Needs a local bake, see
// examples/assets/meshlets/README.md.
//
// @flag WEBGL_DISABLED
// @flag WEBGPU_BARE_DISABLED
// @flag HIDDEN

// URL parameters:
//   ?scene=all|<name>|default  manifest in the bake folder: scene.<name>.json, or scene.json for
//                              `default` (default: all)
//   ?pool=<MiB>                geometry budget override
//   ?tint=0                    leave the materials as baked (white)
//   ?mountains=A,B,E|all       background terrain pieces to keep (default none)
//   ?view=start                begin at the manifest's start pose instead of the hero view
//   ?scale=<0.25..1>           scene render scale, TAA upscales to the output (default 0.6)
//   ?ssao=0                    start with SSAO off

import {
    ADDRESS_CLAMP_TO_EDGE,
    ADDRESS_REPEAT,
    AppBase,
    AppOptions,
    Asset,
    AssetListLoader,
    BLEND_NORMAL,
    CameraComponentSystem,
    CameraFrame,
    Color,
    ContainerHandler,
    EnvLighting,
    Entity,
    FILLMODE_FILL_WINDOW,
    FILTER_NEAREST,
    Keyboard,
    LightComponentSystem,
    MeshletComponentSystem,
    Mouse,
    PIXELFORMAT_RGBA8,
    Picker,
    RenderComponentSystem,
    RESOLUTION_AUTO,
    SSAOTYPE_COMBINE,
    BoundingBox,
    SHADOWUPDATE_REALTIME,
    SHADOWUPDATE_THISFRAME,
    SSAOTYPE_NONE,
    ScriptComponentSystem,
    TEXTURELOCK_READ,
    TEXTURETYPE_RGBE,
    TONEMAP_ACES2,
    TONEMAP_NONE,
    Texture,
    TextureHandler,
    TouchDevice,
    Vec3,
    createGraphicsDevice,
    math
} from 'playcanvas';
import { CameraControls } from 'playcanvas/scripts/esm/camera-controls.mjs';

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

// The NVIDIA sample's own camera (zorah_main_public.v2.cfg): a view down the courtyard into the low
// warm light
const HERO = { position: [55.34, 6.45, 0.32], target: [46.38, 5.8, 0.53] };

// The sky and ambient light come from the Rhodes Memorial HDRI (Poly Haven, CC0), and the sun light
// from its sun disk (see splitSun). The intensity scales the sky, and the sun light with it.
const HDRI = { url: './assets/hdri/rhodes_memorial_4k.hdr', intensity: 1.2, sunLuminance: 20 };

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

// the manifests' budgets only cover the bake's size, too little for the larger scenes
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
const hdriAsset = new Asset('hdri', 'texture', { url: HDRI.url }, { mipmaps: false });
await new Promise((resolve, reject) => {
    new AssetListLoader([...assets.values(), hdriAsset], app.assets).load((error) => {
        if (error) reject(error);
        else resolve();
    });
});

/**
 * Splits an RGBE equirect into the sky and its sun: a copy with every texel's luminance clamped,
 * which takes the sun disk out of it, and the direction, colour and irradiance of the light the
 * clamp removed. The directional light supplies the sun, so in the env-atlas the disk would light
 * everything a second time (three quarters of this HDRI's energy is in it).
 *
 * @param {Texture} source - The RGBE equirect.
 * @param {number} maxLuminance - The luminance to clamp to.
 * @returns {{ sky: Texture, toSun: Vec3, color: Color, irradiance: number }} The clamped copy, and
 * the removed light: its direction, its colour normalized to a maximum of 1, and its irradiance.
 */
const splitSun = (source, maxLuminance) => {
    const src = source.lock({ mode: TEXTURELOCK_READ });
    source.unlock();
    const dst = new Uint8Array(src);
    const { width, height } = source;
    const removed = [0, 0, 0];
    const toSun = new Vec3();
    for (let y = 0; y < height; y++) {
        // texel rows run from straight up to straight down, columns around atan(x, z)
        const elevation = (0.5 - (y + 0.5) / height) * Math.PI;
        const solidAngle = Math.cos(elevation) * (Math.PI / height) * (2 * Math.PI / width);
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            const e = dst[i + 3];
            if (!e) continue;
            const f = 2 ** (e - 136);
            const lum = (0.2126 * dst[i] + 0.7152 * dst[i + 1] + 0.0722 * dst[i + 2]) * f;
            if (lum <= maxLuminance) continue;

            // the light above the clamp, weighted by the texel's solid angle
            const cut = (1 - maxLuminance / lum) * f * solidAngle;
            removed[0] += dst[i] * cut;
            removed[1] += dst[i + 1] * cut;
            removed[2] += dst[i + 2] * cut;
            const azimuth = ((x + 0.5) / width * 2 - 1) * Math.PI;
            const weight = (lum - maxLuminance) * solidAngle;
            toSun.x += Math.cos(elevation) * Math.sin(azimuth) * weight;
            toSun.y += Math.sin(elevation) * weight;
            toSun.z += Math.cos(elevation) * Math.cos(azimuth) * weight;

            const k = maxLuminance / lum * f;
            const r = dst[i] * k, g = dst[i + 1] * k, b = dst[i + 2] * k;
            const exponent = Math.floor(Math.log2(Math.max(r, g, b))) + 1;
            const scale = 256 / 2 ** exponent;
            dst[i] = r * scale;
            dst[i + 1] = g * scale;
            dst[i + 2] = b * scale;
            dst[i + 3] = exponent + 128;
        }
    }
    // the skybox and the env-atlas look the equirect up with x mirrored
    toSun.x = -toSun.x;
    const peak = Math.max(...removed);
    return {
        sky: new Texture(device, {
            name: 'hdri-sky',
            width,
            height,
            format: PIXELFORMAT_RGBA8,
            type: TEXTURETYPE_RGBE,
            addressU: ADDRESS_REPEAT,
            addressV: ADDRESS_CLAMP_TO_EDGE,
            minFilter: FILTER_NEAREST,
            magFilter: FILTER_NEAREST,
            mipmaps: false,
            levels: [dst]
        }),
        toSun: toSun.normalize(),
        color: new Color(removed[0] / peak, removed[1] / peak, removed[2] / peak),
        irradiance: 0.2126 * removed[0] + 0.7152 * removed[1] + 0.0722 * removed[2]
    };
};

// The HDRI is the visible sky, sun disk included, and lights the scene through its env-atlas in
// place of a flat ambient, without its sun. The sun light replaces that, in the disk's direction
// and colour. A surface lit by the disk would receive albedo * irradiance / pi, and the light gives
// albedo * intensity, so its intensity is the irradiance / pi, at the sky's scale.
const hdri = hdriAsset.resource;
const SUN = splitSun(hdri, HDRI.sunLuminance);
app.scene.skybox = EnvLighting.generateSkyboxCubemap(hdri);
const lightingSource = EnvLighting.generateLightingSource(SUN.sky);
app.scene.envAtlas = EnvLighting.generateAtlas(lightingSource);
lightingSource.destroy();
SUN.sky.destroy();
app.scene.skyboxIntensity = HDRI.intensity;
app.scene.ambientLight = new Color(0, 0, 0);
app.on('destroy', () => {
    app.scene.skybox?.destroy();
    app.scene.envAtlas?.destroy();
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
// The background terrain tiles (MI_Mountains_Background_<L>1) are dropped by default, as they do
// not fit the pool next to the courtyard and the HDRI supplies the horizon. ?mountains=all keeps all
// but the F1 tiles, whose surface cuts through the courtyard.
const mountainParam = (params.get('mountains') ?? 'none').toUpperCase();
const keepMountains = new Set((mountainParam === 'ALL' ? 'A,B,C,D,E,G' : mountainParam).split(','));
for (const asset of assets.values()) {
    const resource = asset.resource.meshlets?.[0];
    const materials = asset.resource.data?.materials;
    if (!resource || !materials) continue;
    const letters = resource.primitives.map(p => /Mountains_Background_([A-Z])/i.exec(materials[p.materialIndex]?.name ?? '')?.[1]?.toUpperCase());
    resource.instances = resource.instances.filter((primIndex) => {
        const letter = letters[primIndex];
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
            const inline = asset.resource.instantiateRenderEntity({ skipBlendedShadowCasters: true });
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

// The sun: the key light. Its shadows carry most of the look on untextured geometry. By
// default they are one 8k map fitted to the scene bounds and rendered once, then again only when
// streaming changes what the meshlets cast (see MeshletShadowRenderer): nothing about it follows
// the camera, so there is nothing to swim or flicker as it moves. The realtime alternative is four
// camera-fitted cascades.
const sun = new Entity('Sun');
sun.addComponent('light', {
    type: 'directional',
    color: SUN.color,
    intensity: SUN.irradiance / Math.PI * HDRI.intensity,
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

// CameraFrame supplies the scene-depth attachment the two-phase occlusion reads, and the
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
// SSAO in combine mode: the meshlet material does not sample the AO texture the 'lighting' mode
// relies on. A lower power than the default keeps the carving's creases from crushing to black.
const ssaoOn = params.get('ssao') !== '0';
const SSAO = { radius: 0.6, intensity: 0.6, power: 1.5 };
Object.assign(cameraFrame.ssao, {
    type: ssaoOn ? SSAOTYPE_COMBINE : SSAOTYPE_NONE,
    ...SSAO,
    // a new sampling pattern each frame, which TAA averages away: the fixed per-pixel noise
    // otherwise survives the blur as a pattern on the carving
    randomize: true,
    samples: 16,
    scale: 1
});
cameraFrame.update();
app.on('destroy', () => cameraFrame.destroy());

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
    skyIntensity: app.scene.skyboxIntensity,
    shadows: true,
    staticShadows: true,
    ssao: ssaoOn,
    ssaoRadius: SSAO.radius,
    ssaoIntensity: SSAO.intensity,
    ssaoPower: SSAO.power,
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
data.on('data.skyIntensity:set', (/** @type {number} */ value) => {
    app.scene.skyboxIntensity = value;
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
for (const key of ['Radius', 'Intensity', 'Power']) {
    data.on(`data.ssao${key}:set`, (/** @type {number} */ value) => {
        cameraFrame.ssao[key.toLowerCase()] = value;
        cameraFrame.update();
    });
}
// upscaling is part of the TAA resolve, so without TAA a reduced scale is a bilinear stretch
const applyResolution = () => {
    const taa = data.get('data.taa');
    const scale = data.get('data.renderScale');
    cameraFrame.taa.enabled = taa;
    cameraFrame.ssao.randomize = taa;
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
        drawn: 0,
        resident: 0,
        slots: 0,
        missing: 0,
        rootShortfall: 0
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
        budget.drawn += view.renderedMeshlets ?? 0;
        budget.resident += director.residency?.residentPages ?? 0;
        budget.slots += director.world.poolSlots ?? 0;
        budget.missing += director.residency?.lastMissingWanted ?? 0;
        budget.rootShortfall += director.residency?.rootShortfallPages ?? 0;
    }
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
