// @config
//
// The JungleRuins scene as streamed meshlets: an 8 km terrain under millions of scattered trees,
// saplings and seedlings, with the pyramid and the banyan forest in the 1 km cinematic core. Each
// group (terrain, static, trees, plants) is its own bake; every scatter is one prototype mesh
// with an EXT_mesh_gpu_instancing node, so all copies of a plant share one set of pages. See
// examples/utils/jungle/usd2glb.py for building the assets.
//
// @flag WEBGL_DISABLED
// @flag WEBGPU_BARE_DISABLED
// @flag HIDDEN

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
    DISPLAYFORMAT_HDR,
    Entity,
    EnvLighting,
    FILLMODE_FILL_WINDOW,
    FILTER_LINEAR,
    FILTER_LINEAR_MIPMAP_LINEAR,
    Keyboard,
    LightComponentSystem,
    math,
    MeshletComponentSystem,
    Mouse,
    Picker,
    PIXELFORMAT_RGBA16F,
    Quat,
    RenderComponentSystem,
    RESOLUTION_AUTO,
    SHADERPASS_WORLDNORMAL,
    ShaderChunks,
    SSAOTYPE_COMBINE,
    ScriptComponentSystem,
    TONEMAP_ACES2,
    TONEMAP_NONE,
    Texture,
    TextureHandler,
    TouchDevice,
    Vec3,
    createGraphicsDevice
} from 'playcanvas';
import { CameraControls } from 'playcanvas/scripts/esm/camera-controls.mjs';
import { ProceduralSky } from 'playcanvas/scripts/esm/sky/procedural-sky.mjs';

import { data, deviceType } from 'examples/context';

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('application-canvas'));
window.focus();

// KTX2 textures in the bake are Basis-compressed, so the Basis transcoder must be ready before load
basisInitialize({
    glueUrl: './assets/wasm/basis/basis.wasm.js',
    wasmUrl: './assets/wasm/basis/basis.wasm.wasm',
    fallbackUrl: './assets/wasm/basis/basis.js'
});

const params = new URLSearchParams(location.search);
// ?base=<dir>/ loads a different bake folder
const BASE = params.get('base') ?? './assets/meshlets/jungle/';

/**
 * @typedef {{ name: string, label: string, url: string, min: number[], max: number[],
 * instances?: number }} Group
 */
/** @type {{ groups: Group[], start?: { pos: number[], look: number[] } }} */
const manifest = await fetch(`${BASE}jungle_scene.json`).then(r => r.json());
// ?groups=terrain,trees loads only those
const only = params.get('groups')?.split(',');
// ?moss=0 leaves the pyramid's moss out: about 2M instances, most of the scatter's cost
const groups = manifest.groups.filter(g => (!only || only.includes(g.name)) && !(g.name === 'moss' && params.get('moss') === '0'));

const device = await createGraphicsDevice(canvas, {
    deviceTypes: [deviceType],
    antialias: false,
    reverseZ: params.get('reverseZ') !== '0',
    // a 16 bit float canvas on displays that support HDR, silently LDR elsewhere; ?hdr=0 forces LDR
    displayFormat: params.get('hdr') === '0' ? undefined : DISPLAYFORMAT_HDR
});
device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);

// Canopy light: realtime shadows cannot afford the forest, so canopy.py bakes the leaf area into
// static maps - along the sun ray (leaf depth and the slab of distances it spans) for the direct
// light, and straight down for the sky. Hooked into every lit shader through the global chunks:
// the sun term scales directional lights, the sky term the ambient and environment reflections.
// ?canopy=0 disables it; ?canopySun / ?canopySky (0..1) set the strengths, ?canopyK the extinction.
const canopyVisibilityWGSL = /* wgsl */ `
#if !defined(PICK_PASS) && !defined(PREPASS_PASS) && !defined(SHADOW_PASS)
    var canopySunCore: texture_2d<f32>;
    var canopySunCoreSampler: sampler;
    var canopySunFar: texture_2d<f32>;
    var canopySunFarSampler: sampler;
    var canopySky: texture_2d<f32>;
    var canopySkySampler: sampler;
    // per map: (u / span, -origin.x / span), (v / span, -origin.y / span)
    uniform canopySunCoreU: vec4f;
    uniform canopySunCoreV: vec4f;
    uniform canopySunFarU: vec4f;
    uniform canopySunFarV: vec4f;
    uniform canopySkyU: vec4f;
    uniform canopySkyV: vec4f;
    uniform canopySunAxis: vec4f;
    // extinction, sun strength, sky strength, sky mip
    uniform canopyParams: vec4f;
    uniform canopyEnvScale: f32;

    fn canopyUv(rowU: vec4f, rowV: vec4f, p: vec3f) -> vec2f {
        return vec2f(dot(rowU.xyz, p) + rowU.w, dot(rowV.xyz, p) + rowV.w);
    }

    fn canopyInside(uv: vec2f) -> bool {
        return all(uv > vec2f(0.0)) && all(uv < vec2f(1.0));
    }

    // leaf depth between distance d along the map axis and the light: the texel stores depth and
    // depth-premultiplied slab bounds, so filtering keeps them consistent
    fn canopyDepthAbove(s: vec4f, d: f32) -> f32 {
        if (s.r < 1e-3) {
            return 0.0;
        }
        let lo = s.g / s.r;
        let hi = s.b / s.r;
        return s.r * saturate((hi - d) / max(hi - lo, 0.1));
    }

    fn canopySunVisibility() -> f32 {
        let p = vPositionW;
        let d = dot(uniform.canopySunAxis.xyz, p);
        var depth = 0.0;
        let uvCore = canopyUv(uniform.canopySunCoreU, uniform.canopySunCoreV, p);
        if (canopyInside(uvCore)) {
            depth = canopyDepthAbove(textureSampleLevel(canopySunCore, canopySunCoreSampler, uvCore, 0.0), d);
        } else {
            let uvFar = canopyUv(uniform.canopySunFarU, uniform.canopySunFarV, p);
            if (canopyInside(uvFar)) {
                depth = canopyDepthAbove(textureSampleLevel(canopySunFar, canopySunFarSampler, uvFar, 0.0), d);
            }
        }
        return mix(1.0, exp(-uniform.canopyParams.x * depth), uniform.canopyParams.y);
    }

    fn canopySkyVisibility() -> f32 {
        let p = vPositionW;
        let uv = canopyUv(uniform.canopySkyU, uniform.canopySkyV, p);
        if (!canopyInside(uv)) {
            return 1.0;
        }
        let depth = canopyDepthAbove(textureSampleLevel(canopySky, canopySkySampler, uv, uniform.canopyParams.w), p.y);
        return mix(1.0, exp(-uniform.canopyParams.x * depth), uniform.canopyParams.z);
    }
#endif
`;

/**
 * @typedef {{ name: string, u: number[], v: number[], axis: number[], origin: number[], span: number,
 * size: number, offset: number, levels: number }} CanopyMap
 */
/** @type {{ maps: CanopyMap[], extinction: number, toSun: number[] }|null} */
const canopy = params.get('canopy') === '0' ? null :
    await fetch(`${BASE}jungle_canopy.json`).then(r => (r.ok ? r.json() : null)).catch(() => null);
if (canopy) {
    const bin = await fetch(`${BASE}jungle_canopy.bin`).then(r => r.arrayBuffer());
    for (const map of canopy.maps) {
        const levels = [];
        let offset = map.offset;
        for (let l = 0, size = map.size; l < map.levels; l++, size = Math.max(1, size >> 1)) {
            levels.push(new Uint16Array(bin, offset, size * size * 4));
            offset += size * size * 8;
        }
        const texture = new Texture(device, {
            name: `canopy-${map.name}`,
            width: map.size,
            height: map.size,
            format: PIXELFORMAT_RGBA16F,
            mipmaps: true,
            levels,
            addressU: ADDRESS_CLAMP_TO_EDGE,
            addressV: ADDRESS_CLAMP_TO_EDGE,
            minFilter: FILTER_LINEAR_MIPMAP_LINEAR,
            magFilter: FILTER_LINEAR
        });
        const id = `canopy${map.name[0].toUpperCase()}${map.name.slice(1)}`;
        const inv = 1 / map.span;
        device.scope.resolve(id).setValue(texture);
        device.scope.resolve(`${id}U`).setValue(new Float32Array([...map.u.map(c => c * inv), -map.origin[0] * inv]));
        device.scope.resolve(`${id}V`).setValue(new Float32Array([...map.v.map(c => c * inv), -map.origin[1] * inv]));
    }
    device.scope.resolve('canopySunAxis').setValue(new Float32Array([...canopy.toSun, 0]));
}

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

// the engine chunks are registered by app.init; patch them before anything compiles
if (canopy) {
    const chunks = ShaderChunks.get(device, 'wgsl');
    chunks.set('litUserDeclarationPS', canopyVisibilityWGSL);
    chunks.set('lightFunctionLightPS', chunks.get('lightFunctionLightPS').replace('    // Relief self shadowing', `
    #if LIGHT{i}TYPE == DIRECTIONAL
        dAtten = dAtten * canopySunVisibility();
    #endif

    // Relief self shadowing`));
    chunks.set('ambientPS', `${chunks.get('ambientPS').replace('fn addAmbient(', 'fn addAmbientOpen(')}
fn addAmbient(worldNormal: vec3f) {
    addAmbientOpen(worldNormal);
    dDiffuseLight = dDiffuseLight * (canopySkyVisibility() * uniform.canopyEnvScale);
}
`);
    for (const name of ['reflectionEnvPS', 'reflectionEnvHQPS']) {
        chunks.set(name, `${chunks.get(name).replace('fn addReflection(', 'fn addReflectionOpen(')}
fn addReflection(reflDir: vec3f, gloss: f32) {
    addReflectionOpen(reflDir, gloss);
    dReflection = vec4f(dReflection.rgb * (canopySkyVisibility() * uniform.canopyEnvScale), dReflection.a);
}
`);
    }
}

app.setCanvasFillMode(FILLMODE_FILL_WINDOW);
app.setCanvasResolution(RESOLUTION_AUTO);

const resize = () => app.resizeCanvas();
window.addEventListener('resize', resize);
app.on('destroy', () => {
    window.removeEventListener('resize', resize);
});

// ?pool=<MiB> geometry budget shared by every group; ?worldMeshlets=<millions> splits the content
// across worlds of at most that many meshlets
app.systems.meshlet.poolBytes = (Number(params.get('pool')) || 3072) * 1024 * 1024;
if (params.get('worldMeshlets')) app.systems.meshlet.maxWorldMeshlets = Number(params.get('worldMeshlets')) * 1e6;
app.systems.meshlet.occlusion = true;
app.systems.meshlet.cssPixels = params.get('cssPixels') !== '0';
app.systems.meshlet.deferRebuildWhileLoading = params.get('progressive') !== '1';

// Impostors: the meshlet LOD erases distant leaf cards, so beyond ?impostorPx (the radius, in
// pixels, a tree projects to) each tree is one billboard sampling views baked by
// examples/utils/jungle (protos.py, bake_server.py, the meshlet-impostor-bake example).
// ?impostors=0 turns them off.
const impostorManifest = params.get('impostors') === '0' ? null :
    await fetch(`${BASE}jungle_impostors.json`).then(r => (r.ok ? r.json() : null)).catch(() => null);
if (impostorManifest) {
    const atlas = async (/** @type {string} */ kind, /** @type {boolean} */ srgb) => {
        // one UASTC KTX2 per layer (bake_server.py), transcoded to the device's block format by the
        // engine's own transcoder
        const transcode = app.systems.meshlet.group.transcode;
        const layers = await Promise.all(impostorManifest.prototypes.map(async (/** @type {{ layer: number }} */ p) => {
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
        grid: impostorManifest.grid,
        prototypes: Object.fromEntries(impostorManifest.prototypes.map((/** @type {{ name: string, layer: number, center: number[], radius: number, half: number[] }} */ p) => [p.name, {
            layer: p.layer, center: p.center, radius: p.radius, half: p.half
        }]))
    };
}

const scene = new Entity('Jungle');
app.root.addChild(scene);
for (const group of groups) {
    const asset = new Asset(group.label, 'container', { url: BASE + group.url });
    asset.once('error', err => console.error(`Jungle: failed to load ${group.url}`, err));
    app.assets.add(asset);
    const entity = new Entity(group.label);
    entity.addComponent('meshlet', { asset });
    scene.addChild(entity);
}

// the reference render's dome HDR (sky.py): skybox, clamped lighting copy, and the glow it clamped
// away as the sun. ?sky=procedural (or a bake without it) uses the procedural sky and a fixed sun.
/** @type {{ toSun: number[], sunColor: number[], sunIntensity: number }|null} */
const hdrSky = params.get('sky') === 'procedural' ? null :
    await fetch(`${BASE}jungle_sky.json`).then(r => (r.ok ? r.json() : null)).catch(() => null);

const light = new Entity('Sun');
light.addComponent('light', {
    type: 'directional',
    color: hdrSky ? new Color(...hdrSky.sunColor) : new Color(1, 0.95, 0.85),
    intensity: hdrSky ? hdrSky.sunIntensity : 2.5
});
if (hdrSky) {
    // a directional light shines down its local -Y
    light.setRotation(new Quat().setFromDirections(Vec3.UP, new Vec3(hdrSky.toSun)));
} else {
    light.setEulerAngles(45, 30, 0);
}
app.root.addChild(light);

// ?shadows=1: four cascades from the sun, meshlets included
if (params.get('shadows') === '1') {
    app.systems.meshlet.shadows = true;
    light.light.castShadows = true;
    light.light.numCascades = 4;
    light.light.shadowResolution = 2048;
    light.light.shadowDistance = Number(params.get('shadowDistance')) || 400;
    light.light.cascadeUpdateIntervals = [1, 1, 2, 2];
}

const camera = new Entity('Camera');
camera.addComponent('camera', {
    clearColor: new Color(0.55, 0.68, 0.82),
    nearClip: 0.1,
    farClip: 12000
});
camera.addComponent('script');
app.root.addChild(camera);
const cc = /** @type {CameraControls} */ (camera.script.create(CameraControls));
Object.assign(cc, {
    moveSpeed: 15,
    moveFastSpeed: 120,
    moveSlowSpeed: 3
});

// CameraFrame supplies the scene-depth attachment the two-phase occlusion reads
const cameraFrame = new CameraFrame(app, camera.camera);
cameraFrame.rendering.samples = 1;
cameraFrame.bloom.enabled = false;
// an HDR canvas takes the scene values as they are, tone mapping would clip them
cameraFrame.rendering.toneMapping = device.isHdr ? TONEMAP_NONE : TONEMAP_ACES2;
// the foliage is all alpha-tested cutouts, so edges alias hard: ?taa=0 turns the temporal resolve off
cameraFrame.taa.enabled = params.get('taa') !== '0';
// ?ssao=1 darkens contact areas in the compose pass, from the scene depth the meshlets write
if (params.get('ssao') === '1') {
    Object.assign(cameraFrame.ssao, {
        type: SSAOTYPE_COMBINE,
        radius: Number(params.get('ssaoRadius')) || 4,
        intensity: Number(params.get('ssaoIntensity')) || 0.8,
        scale: 0.5
    });
}
cameraFrame.update();
app.on('destroy', () => cameraFrame.destroy());

if (hdrSky) {
    /**
     * @param {string} file - HDR file in the bake folder.
     * @returns {Promise<Texture>} The equirect texture.
     */
    const loadHdr = file => new Promise((resolve, reject) => {
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
    skyTexture.destroy();
    envTexture.destroy();
} else {
    // a procedural sky lights the scene through its env-atlas; the sun disk sits where the light
    // shines from
    const sky = new Entity('ProceduralSky');
    sky.addComponent('script');
    const skyScript = /** @type {ProceduralSky} */ (sky.script.create(ProceduralSky));
    const toSun = light.up;
    skyScript.elevation = Math.asin(toSun.y) * math.RAD_TO_DEG;
    skyScript.azimuth = Math.atan2(toSun.x, toSun.z) * math.RAD_TO_DEG;
    skyScript.luminance = 0.4;
    app.root.addChild(sky);
}
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
            cc.focusPoint = pivotTarget;
        }
    }).finally(() => {
        pickInFlight = false;
    });
});

data.set('data', {
    visualization: 'material',
    occlusion: true,
    threshold: Number(params.get('threshold') ?? 6),
    impostorPx: impostorManifest ? Number(params.get('impostorPx') ?? 128) : 0,
    impostorTint: false,
    taa: cameraFrame.taa.enabled,
    impostorBrightness: Number(params.get('impostorBrightness') ?? 0.85),
    impostorOcclusion: Number(params.get('impostorAo') ?? 0.1),
    stats: ''
});
data.on('data.taa:set', (/** @type {boolean} */ value) => {
    cameraFrame.taa.enabled = value;
    cameraFrame.update();
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
app.systems.meshlet.dagPixelThreshold = data.get('data.threshold');
app.systems.meshlet.impostorPixels = data.get('data.impostorPx');
data.on('data.impostorPx:set', (/** @type {number} */ value) => {
    app.systems.meshlet.impostorPixels = value;
});

// canopy strengths, live from the panel (seeded from the query parameters)
const canopyParams = new Float32Array(4);
const updateCanopy = () => {
    canopyParams[0] = data.get('data.canopyK');
    canopyParams[1] = data.get('data.canopySun');
    canopyParams[2] = data.get('data.canopySky');
    canopyParams[3] = Number(params.get('canopySkyMip') ?? 3);
    device.scope.resolve('canopyParams').setValue(canopyParams);
};
data.set('data.canopyK', Number(params.get('canopyK') ?? canopy?.extinction ?? 1.5));
data.set('data.canopySun', canopy ? Number(params.get('canopySun') ?? 1) : 0);
data.set('data.canopySky', canopy ? Number(params.get('canopySky') ?? 1) : 0);
updateCanopy();
data.on('data.canopyK:set', updateCanopy);
data.on('data.canopySun:set', updateCanopy);
data.on('data.canopySky:set', updateCanopy);

// sun / sky / exposure, live from the panel. The dome HDR alone gives a dusk-like key light: its
// glow carries ~0.3 against ~0.7 of sky, while the reference renders show a clear-day sun with
// hard shadows, so the default sun is boosted to about twice the sky on the ground.
const envScale = new Float32Array(1);
const updateLighting = () => {
    light.light.intensity = data.get('data.sun');
    envScale[0] = data.get('data.sky');
    device.scope.resolve('canopyEnvScale').setValue(envScale);
    app.scene.skyboxIntensity = data.get('data.sky');
    app.scene.exposure = data.get('data.exposure');
};
data.set('data.sun', Number(params.get('sun') ?? (hdrSky ? 3.5 : 2.5)));
data.set('data.sky', Number(params.get('skyScale') ?? 1));
data.set('data.exposure', Number(params.get('exposure') ?? (hdrSky ? 0.6 : 1)));
updateLighting();
data.on('data.sun:set', updateLighting);
data.on('data.sky:set', updateLighting);
data.on('data.exposure:set', updateLighting);

/**
 * @param {string} key - Query parameter holding `x,y,z`.
 * @returns {Vec3|null} The parsed vector, or null when absent or malformed.
 */
const vecParam = (key) => {
    const v = params.get(key)?.split(',').map(Number);
    return v?.length === 3 && v.every(Number.isFinite) ? new Vec3(v) : null;
};
// ?pos=x,y,z&look=x,y,z (glTF metres); the default looks over the pyramid
const startPos = vecParam('pos') ?? new Vec3(manifest.start?.pos ?? [480, 60, -760]);
const startLook = vecParam('look') ?? new Vec3(manifest.start?.look ?? [544, 5, -810]);
cc.reset(startLook, startPos);

let statFrames = 0;
const impostorDebugApplied = new WeakMap();
app.on('framerender', () => {
    const directors = app.systems.meshlet.directors.filter(d => d.world.finalized);
    if (!directors.length) return;
    // ?impostorDebug=N: 1 magenta tint, 2 albedo, 3 normal, 4 alpha; the panel toggle tints
    const impostorDebug = data.get('data.impostorTint') ? 1 : Number(params.get('impostorDebug') ?? 0);
    const impostorBrightness = data.get('data.impostorBrightness');
    const impostorOcclusion = data.get('data.impostorOcclusion');
    for (const director of directors) {
        // a rebuilt world brings a fresh material, so the value is tracked per material
        const material = director.world.impostorMaterial;
        const key = `${impostorDebug},${impostorBrightness},${impostorOcclusion}`;
        if (material && impostorDebugApplied.get(material) !== key) {
            impostorDebugApplied.set(material, key);
            material.setParameter('impostorDebug', impostorDebug);
            // crown normal blend 0, lower-crown darkening, albedo scale
            material.setParameter('impostorShading', [0, impostorOcclusion, impostorBrightness, 0]);
        }
    }
    const visualization = data.get('data.visualization');
    const colorMode = visualization === 'meshlet' ? 2 : visualization === 'lod' ? 1 : 0;
    let meshlets = 0, indices = 0, resident = 0, pages = 0, instances = 0, demand = false;
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
        instances += director.world.instanceCount;
    }
    if (demand && ++statFrames % 30 === 0) {
        const worlds = directors.length > 1 ? ` (${directors.length} worlds)` : '';
        data.set('data.stats', `${meshlets} meshlets, cut ${(indices / 3e6).toFixed(2)}M tris, ` +
            `${(instances / 1e6).toFixed(2)}M instances, ${resident}/${pages} pages${worlds}`);
    }
});

app.start();
