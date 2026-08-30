// @config
//
// GPU-driven meshlet rendering: a grid of Stanford bunnies, fully resident, culled and
// LOD-selected on the GPU every frame - instance frustum cull, per-cluster DAG cut (crack-free,
// screen-space error driven), cone backface cull, GPU-generated index buffer, indirect draw.
// All bunnies share one set of geometry pages, so a 24x24 grid costs the same memory as one
// bunny; the draw index buffer starts small and grows to the observed demand. Colour by meshlet
// to inspect the clusters and watch the DAG cut coarsen with distance and the error threshold.
// Click a bunny to select it: the GPU picker identifies the individual instance behind the
// pixel, and that same index outlines it - shift-click to add to the selection.
//
// @flag WEBGL_DISABLED

import {
    ADDRESS_CLAMP_TO_EDGE,
    AppBase,
    AppOptions,
    Asset,
    AssetListLoader,
    CameraComponentSystem,
    Color,
    ContainerHandler,
    Entity,
    FILLMODE_FILL_WINDOW,
    FILTER_LINEAR,
    FOG_LINEAR,
    FOG_NONE,
    LAYERID_SKYBOX,
    LAYERID_IMMEDIATE,
    LAYERID_WORLD,
    LightComponentSystem,
    Mat4,
    CameraFrame,
    EVENT_MOUSEDOWN,
    EVENT_MOUSEMOVE,
    Layer,
    MeshletComponentSystem,
    Mouse,
    OutlineRenderer,
    Picker,
    PIXELFORMAT_BGRA8,
    PIXELFORMAT_DEPTH,
    Quat,
    RESOLUTION_AUTO,
    RenderComponentSystem,
    RenderTarget,
    StandardMaterial,
    TEXTURETYPE_RGBP,
    TONEMAP_LINEAR,
    TRACEID_GPU_TIMINGS,
    Texture,
    TextureHandler,
    Tracing,
    Vec3,
    createGraphicsDevice
} from 'playcanvas';

import { data, deviceType } from 'examples/context';

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('application-canvas'));
window.focus();

const assets = {
    bunny: new Asset('bunny', 'container', { url: './assets/meshlets/bunny-v2.glb' }),
    helipad: new Asset(
        'helipad-env-atlas',
        'texture',
        { url: './assets/cubemaps/helipad-env-atlas.png' },
        { type: TEXTURETYPE_RGBP, mipmaps: false }
    )
};

// The example runs in an iframe and the example browser does not forward its query string, so
// merge the top window's params in (lowest priority) - ?grid=24&aa=0 then works on the outer
// /#/graphics/meshlet-inspect URL as well as on the /iframe/ one.
const params = new URLSearchParams(location.search);
try {
    if (window.top !== window) {
        new URLSearchParams(window.top.location.search).forEach((value, key) => {
            if (!params.has(key)) params.set(key, value);
        });
    }
} catch (_e) {
    // cross-origin parent - this frame's own params only
}

// No backbuffer MSAA: anti-aliasing is chosen from the panel's AA dropdown and applied on the
// CameraFrame's scene target instead, where it can change at runtime (the backbuffer's sample
// count is fixed at device creation).
const gfxOptions = {
    deviceTypes: [deviceType],
    antialias: false
};

const device = await createGraphicsDevice(canvas, gfxOptions);
device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);

const createOptions = new AppOptions();
createOptions.graphicsDevice = device;
createOptions.componentSystems = [
    RenderComponentSystem,
    CameraComponentSystem,
    LightComponentSystem,
    MeshletComponentSystem
];
createOptions.resourceHandlers = [TextureHandler, ContainerHandler];

const app = new AppBase(canvas);
app.init(createOptions);

app.setCanvasFillMode(FILLMODE_FILL_WINDOW);
app.setCanvasResolution(RESOLUTION_AUTO);

const resize = () => app.resizeCanvas();
window.addEventListener('resize', resize);
app.on('destroy', () => {
    window.removeEventListener('resize', resize);
});

await new Promise((resolve) => {
    new AssetListLoader(Object.values(assets), app.assets).load(resolve);
});

app.start();

const resource = assets.bunny.resource.meshlets[0];

// fully resident: fetch all shards once, reused across rebuilds
const shardBuffers = await Promise.all(
    resource.manifest.blobs.map((blob) => fetch(`./assets/meshlets/${blob.uri}`).then((r) => r.arrayBuffer()))
);

const SPACING = 0.28;
const buildInstances = (n) => {
    const instances = [];
    const m = new Mat4();
    const pos = new Vec3();
    const rot = new Quat();
    const half = ((n - 1) * SPACING) / 2;
    for (let x = 0; x < n; x++) {
        for (let z = 0; z < n; z++) {
            pos.set(x * SPACING - half, 0, z * SPACING - half);
            rot.setFromEulerAngles(0, (x * 31 + z * 17) % 360, 0);
            m.setTRS(pos, rot, Vec3.ONE);
            instances.push({ primIndex: 0, matrix: new Float32Array(m.data) });
        }
    }
    return instances;
};

// the component system owns the director (and wires the KTX2 transcoder); this example
// drives the director directly instead of adding meshlet components
const director = app.systems.meshlet.director;
director.world.initialIndices = 1024 * 1024;
// debug hooks: the director and the panel observer, so this can be driven headlessly
window.__meshletDirector = director;
window.__data = data;

// ?opaque=1 clears the material's doubleSided flag, moving the asset from the two-sided bucket
// to the fully opaque one. The bunny is a closed solid, so its doubleSided export flag is a
// lie - and backface culling is worth roughly another 20% of the draw on top of what dropping
// the alpha test already recovers. A fragment-cost A/B, and a demonstration of what the asset
// is costing itself.
if (params.get('opaque') === '1' && resource.materialTable) {
    // baked material records - MATERIAL_RECORD_U32S words per row, flags word, double-sided
    // bit (src/scene/meshlet/constants.js; the constants are not exported from the engine)
    const MATERIAL_RECORD_U32S = 32;
    const MATERIAL_RECORD_FLAGS = 11;
    const MATERIAL_FLAG_DOUBLE_SIDED = 1;
    for (let row = 0; row < resource.materialCount; row++) {
        resource.materialTable[row * MATERIAL_RECORD_U32S + MATERIAL_RECORD_FLAGS] &= ~MATERIAL_FLAG_DOUBLE_SIDED;
    }
}

// ?timers=1 logs per-pass GPU timings to the console (dev build only)
if (params.get('timers') === '1') {
    Tracing.set(TRACEID_GPU_TIMINGS, true);
    device.gpuProfiler.enabled = true;
}

data.set('data', {
    gridSize: parseInt(params.get('grid') ?? '8', 10),
    colorMode: parseInt(params.get('color') ?? '0', 10),
    threshold: parseFloat(params.get('threshold') ?? '1'),
    sunLight: params.get('lights') !== '0' && params.get('sun') !== '0',
    spotLight: params.get('lights') !== '0' && params.get('spot') !== '0',
    omniLights: params.get('lights') !== '0' && params.get('omni') !== '0',
    ibl: params.get('ibl') !== '0',
    fog: params.get('fog') ?? 'none',
    secondView: params.get('view2') === '1',
    selection: '',
    shadows: params.get('meshletshadows') !== '0',
    aaMode: 'none'
});

// meshlet shadow casting is opt-in per scene (one single-phase cull chain per cascade)
director.shadowsEnabled = true;
// 3 cascades + two omnis at six faces each + one spot
director.shadowRenderer.maxShadowViews = 24;
director.shadowBudgetCascades = 16;
const numCascades = Math.max(1, Math.min(4, parseInt(params.get('cascades') ?? '1', 10)));

let builtSize = 0;
let gridExtent = 1;
const rebuild = (n) => {
    builtSize = n;
    gridExtent = Math.max(n, 2) * SPACING;
    director.rebuild((world) => {
        world.addResource(resource, null, shardBuffers, buildInstances(n));
    });
};
rebuild(data.get('data.gridSize'));

const camera = new Entity();
camera.addComponent('camera', {
    clearColor: new Color(0.08, 0.09, 0.11),
    nearClip: 0.01,
    farClip: 500
});
app.root.addChild(camera);
director.cameraComponent = camera.camera;

// second view: an independent camera parked close to a corner of the grid, rendering into a
// texture shown as an inset - its own LOD cut (finer, it is nearer) and visibility state,
// streaming served by the shared world
const view2Color = new Texture(device, {
    name: 'View2Color',
    width: 512,
    height: 384,
    format: PIXELFORMAT_BGRA8,
    mipmaps: false,
    minFilter: FILTER_LINEAR,
    magFilter: FILTER_LINEAR,
    addressU: ADDRESS_CLAMP_TO_EDGE,
    addressV: ADDRESS_CLAMP_TO_EDGE
});
const view2Depth = new Texture(device, {
    name: 'View2Depth',
    width: 512,
    height: 384,
    format: PIXELFORMAT_DEPTH,
    mipmaps: false
});
const view2Target = new RenderTarget({ colorBuffer: view2Color, depthBuffer: view2Depth });
const camera2 = new Entity('View2Camera');
camera2.addComponent('camera', {
    clearColor: new Color(0.05, 0.08, 0.06),
    nearClip: 0.01,
    farClip: 500,
    priority: -1,
    // no UI layer - the inset plane samples this camera's output and must not be drawn by it
    layers: [LAYERID_WORLD, LAYERID_SKYBOX]
});
camera2.camera.renderTarget = view2Target;
camera2.enabled = false;
app.root.addChild(camera2);

// inset display: an unlit plane parented to the main camera
const insetMaterial = new StandardMaterial();
insetMaterial.useLighting = false;
insetMaterial.emissiveMap = view2Color;
insetMaterial.emissive.set(1, 1, 1);
insetMaterial.diffuse.set(0, 0, 0);
insetMaterial.update();
const inset = new Entity('View2Inset');
inset.addComponent('render', {
    type: 'plane',
    material: insetMaterial,
    castShadows: false,
    layers: [LAYERID_IMMEDIATE]
});
inset.enabled = false;
app.root.addChild(inset);

// engine lighting: IBL env atlas + a shadowed sun + two orbiting point lights (clustered)
app.scene.envAtlas = assets.helipad.resource;
app.scene.skyboxIntensity = 0.6;
app.scene.ambientLight = new Color(0.03, 0.03, 0.035);

const sun = new Entity('Sun');
sun.addComponent('light', {
    type: 'directional',
    color: new Color(1, 0.96, 0.88),
    intensity: 1.6,
    castShadows: true,
    shadowBias: 0.05,
    normalOffsetBias: 0.02,
    shadowDistance: 12,
    shadowResolution: 2048,
    numCascades: numCascades
});
sun.setEulerAngles(48, 35, 0);
app.root.addChild(sun);

// ?localshadows=0 turns the point/spot shadows off. An omni costs SIX meshlet shadow views -
// one cull chain per cube face - so it is the honest stress case for local shadows.
const localShadows = params.get('localshadows') !== '0';

const makeOmni = (color) => {
    const e = new Entity('Omni');
    e.addComponent('light', {
        type: 'omni',
        color: color,
        intensity: 1.2,
        range: 2.5,
        castShadows: localShadows,
        shadowBias: 0.05,
        normalOffsetBias: 0.02,
        shadowResolution: 512
    });
    app.root.addChild(e);
    return e;
};

// a spot light for the single-face local case
const spot = new Entity('Spot');
spot.addComponent('light', {
    type: 'spot',
    color: new Color(0.6, 1, 0.7),
    intensity: 4,
    range: 8,
    innerConeAngle: 14,
    outerConeAngle: 24,
    castShadows: localShadows,
    shadowBias: 0.05,
    normalOffsetBias: 0.02,
    shadowResolution: 1024
});
app.root.addChild(spot);
const omniA = makeOmni(new Color(1.0, 0.35, 0.2));
const omniB = makeOmni(new Color(0.2, 0.5, 1.0));

// receiver plane: without one the only evidence of meshlet shadows is bunny-on-bunny
// self-shadowing, which is too subtle to diff. ?ground=0 removes it.
if (params.get('ground') !== '0') {
    const groundMaterial = new StandardMaterial();
    groundMaterial.diffuse.set(0.62, 0.62, 0.66);
    groundMaterial.useMetalness = true;
    groundMaterial.metalness = 0;
    groundMaterial.gloss = 0.2;
    groundMaterial.update();
    const ground = new Entity('Ground');
    ground.addComponent('render', { type: 'plane', material: groundMaterial, castShadows: false });
    ground.setLocalScale(60, 1, 60);
    ground.setLocalPosition(0, -0.001, 0);
    app.root.addChild(ground);
}

// reference StandardMaterial sphere - lit meshlets must match this shading
const refMaterial = new StandardMaterial();
refMaterial.diffuse.set(1, 1, 1);
refMaterial.useMetalness = true;
refMaterial.metalness = 0;
refMaterial.gloss = 0;
refMaterial.update();
const refSphere = new Entity('RefSphere');
refSphere.addComponent('render', { type: 'sphere', material: refMaterial, castShadows: true });
refSphere.setLocalScale(0.12, 0.12, 0.12);
refSphere.setLocalPosition(0, 0.3, 0);
app.root.addChild(refSphere);

// The camera renders through a CameraFrame (bloom/tonemap post chain). A camera with frame
// passes never reaches the renderer's render-action path, so the meshlet passes are spliced
// into the CameraFrame's own scene chain instead - see MeshletDirector#buildCameraFramePasses.
// ?cameraframe=0 drops it to exercise the raw render-action path (no post, no AA dropdown).
//
// Anti-aliasing comes from the panel's AA dropdown, applied on the CameraFrame's scene target.
// Mind the MSAA option: meshlet geometry is micro-triangle dense, so nearly every pixel sits
// on a triangle edge and shades several samples - measured here as roughly quadrupling the
// per-fragment lighting cost (local clustered lights went from ~1.5 ms to ~12 ms a frame at
// 4x on a 5M-pixel canvas). TAA pays a flat post-process cost instead.
let cameraFrame = null;
if (params.get('cameraframe') !== '0') {
    cameraFrame = new CameraFrame(app, camera.camera);
    cameraFrame.rendering.toneMapping = TONEMAP_LINEAR;
    // off by default - even a subtle bloom reads as a washed-out veil over this bright scene
    cameraFrame.bloom.intensity = parseFloat(params.get('bloom') ?? '0');
    // two-phase occlusion needs a sampleable scene depth; ?scenedepth=1 asks the CameraFrame
    // for one
    if (params.get('scenedepth') === '1') cameraFrame.rendering.sceneDepthMap = true;
    cameraFrame.update();
}

// SMAA is deliberately not offered: the engine's SMAA chain has a gamma handling bug
// unrelated to meshlets (compose hands it a gamma-encoded intermediate whose round trip does
// not survive every target setup), so the choices are None / TAA / MSAA 4x.
const applyAaMode = () => {
    if (!cameraFrame) return;
    const mode = data.get('data.aaMode') ?? 'none';
    cameraFrame.taa.enabled = mode === 'taa';
    cameraFrame.rendering.samples = mode === 'msaa' ? 4 : 1;
    // counteract TAA's blur; leave the image untouched otherwise
    cameraFrame.rendering.sharpness = mode === 'taa' ? 0.5 : 0;
    cameraFrame.update();
};
data.on('data.aaMode:set', applyAaMode);
applyAaMode();

// Click to select. This is the editor round trip in miniature: the GPU picker returns a per
// INSTANCE record - one primitive at one transform, which is what a glTF submesh becomes here -
// and that instance index is exactly what selects it for outlining.
//
// Meshlets need both halves wired by hand, because their mesh instances live outside the layer
// system: the picker gets them from MeshletDirector#preparePicking, and the outline renderer
// gets the whole geometry once (on a layer the scene camera does NOT render, or the main camera
// would draw the world a second time) with a per-instance flag deciding what actually
// rasterises. Ordinary meshes still go through OutlineRenderer#addEntity - the reference sphere
// is wired that way, in a different colour, so the two paths can be told apart.
const outlineLayer = new Layer({ name: 'MeshletOutline' });
app.scene.layers.push(outlineLayer);
const outlineRenderer = new OutlineRenderer(app, outlineLayer);
director.setOutlineLayer(outlineLayer, new Color(1, 0.6, 0.1), new Color(1, 1, 1));

const picker = new Picker(app, 512, 320);
const worldLayer = app.scene.layers.getLayerByName('World');
let sphereSelected = false;
let sphereHovered = false;

/** @type {Set<number>} - selected meshlet instance indices. */
const selection = new Set();
/** @type {number} - hovered meshlet instance, or -1. */
let hovered = -1;

const applySelection = () => {
    director.clearOutlines();
    director.clearOutlines(true);
    if (selection.size) director.setOutlined([...selection]);
    // hover is a separate flag with its own colour, so hovering never disturbs the selection
    if (hovered >= 0 && !selection.has(hovered)) director.setHovered([hovered]);

    // OutlineRenderer colours per entity, so the sphere re-registers to change colour
    if (sphereSelected || sphereHovered) {
        outlineRenderer.addEntity(refSphere, sphereSelected ? new Color(0.2, 0.8, 1) : Color.WHITE);
    } else {
        outlineRenderer.removeEntity(refSphere);
    }

    const parts = [];
    if (selection.size) parts.push(`${selection.size} meshlet instance${selection.size > 1 ? 's' : ''}`);
    if (sphereSelected) parts.push('sphere');
    const sel = parts.length ? parts.join(' + ') : 'nothing';
    const hov = hovered >= 0 ? `instance ${hovered}` : sphereHovered ? 'sphere' : 'none';
    data.set('data.selection', `${sel}  (hover: ${hov})`);
};

/**
 * Picks at a canvas coordinate and updates the selection. Exposed on window so the headless
 * checks can drive it without synthesising mouse events.
 *
 * @param {number} x - Canvas x, in CSS pixels.
 * @param {number} y - Canvas y, in CSS pixels.
 * @param {boolean} [add] - True to add to the selection rather than replace it.
 * @returns {Promise<string>} What was hit, for logging.
 */
// A click cannot pick immediately. The meshlet draw is INDIRECT, and its draw arguments are
// written by the cull that runs at the start of each frame - so a picker prepared during input
// handling renders the meshlets with arguments that are stale or already cleared, and finds
// nothing. Queue the request and service it from postrender, once the frame's cull has run.
/** @type {{ x: number, y: number, add: boolean, resolve: Function }|null} */
let pendingPick = null;

let pickBusy = false;

app.on('postrender', () => {
    if (!pendingPick || pickBusy) return;
    const { x, y, add, hover, resolve } = pendingPick;
    pendingPick = null;
    pickBusy = true;

    // the picker renders at its own resolution, so scale the canvas coordinate into it
    const sx = Math.floor((x / canvas.clientWidth) * picker.width);
    const sy = Math.floor((y / canvas.clientHeight) * picker.height);
    picker.prepare(camera.camera, app.scene);
    picker
        .getSelectionAsync(sx, sy, 1, 1)
        .then((hits) => {
            pickBusy = false;
            const hit = hits[0];
            let description = 'nothing';

            if (hover) {
                hovered = hit?.resource ? hit.instanceIndex : -1;
                sphereHovered = hit?.node === refSphere;
                if (hit?.resource) description = `meshlet instance ${hit.instanceIndex}`;
                else if (sphereHovered) description = 'reference sphere';
            } else {
                if (!add) {
                    selection.clear();
                    sphereSelected = false;
                }
                if (hit?.resource) {
                    // a meshlet: the record says which instance, and which primitive of the resource
                    if (selection.has(hit.instanceIndex)) selection.delete(hit.instanceIndex);
                    else selection.add(hit.instanceIndex);
                    description = `meshlet instance ${hit.instanceIndex}, primitive ${hit.primIndex}`;
                } else if (hit?.node === refSphere) {
                    sphereSelected = !sphereSelected;
                    description = 'reference sphere (an ordinary mesh)';
                }
            }
            applySelection();
            resolve(description);
        })
        .catch(() => {
            pickBusy = false;
            resolve('pick failed');
        });
});

/**
 * Queues a pick at a canvas coordinate; resolves once the selection has been updated. Exposed
 * on window so the headless checks can drive it without synthesising mouse events.
 *
 * @param {number} x - Canvas x, in CSS pixels.
 * @param {number} y - Canvas y, in CSS pixels.
 * @param {boolean} [add] - True to add to the selection rather than replace it.
 * @param {boolean} [hover] - True to update the hover highlight instead of the selection.
 * @returns {Promise<string>} What was hit.
 */
const selectAt = (x, y, add = false, hover = false) =>
    new Promise((resolve) => {
        // a click supersedes a queued hover, never the other way round
        if (pendingPick && hover && !pendingPick.hover) {
            resolve('skipped');
            return;
        }
        pendingPick?.resolve('superseded');
        pendingPick = { x, y, add, hover, resolve };
    });
window.__selectAt = selectAt;
window.__hoverAt = (x, y) => selectAt(x, y, false, true);

const mouse = new Mouse(canvas);
mouse.on(EVENT_MOUSEDOWN, (event) => {
    selectAt(event.x, event.y, event.shiftKey || event.ctrlKey || event.metaKey);
});
// hover: one pick per frame at most, and only when the pointer has actually moved
let hoverX = -1;
let hoverY = -1;
mouse.on(EVENT_MOUSEMOVE, (event) => {
    if (event.x === hoverX && event.y === hoverY) return;
    hoverX = event.x;
    hoverY = event.y;
    selectAt(event.x, event.y, false, true);
});
canvas.addEventListener('pointerleave', () => {
    if (hovered < 0 && !sphereHovered) return;
    hovered = -1;
    sphereHovered = false;
    applySelection();
});

app.on('update', () => outlineRenderer.frameUpdate(camera, worldLayer, true));
applySelection();

// ?freeze=<seconds> stops the animation at a fixed time for reproducible screenshots
const freezeAt = params.has('freeze') ? parseFloat(params.get('freeze')) : -1;

let angle = 0;
let time = 0;
let sizeChangedAt = -1;

app.on('update', (dt) => {
    // clamp the final step so `time` lands EXACTLY on freezeAt: everything animated here is a
    // function of accumulated time, so two runs then produce identical frames regardless of
    // how the real frame times fell
    if (freezeAt >= 0) dt = Math.max(Math.min(dt, freezeAt - time), 0);
    time += dt;

    // debounce the grid slider - a resize rebuilds the world
    const wanted = Math.round(data.get('data.gridSize'));
    if (wanted !== builtSize) {
        if (sizeChangedAt < 0) sizeChangedAt = time;
        if (time - sizeChangedAt > 0.35) {
            sizeChangedAt = -1;
            rebuild(wanted);
        }
    } else {
        sizeChangedAt = -1;
    }

    angle += dt * 0.15;
    const r = gridExtent * (0.55 + 0.25 * Math.sin(angle * 0.6));
    camera.setLocalPosition(Math.sin(angle) * r, gridExtent * 0.35, Math.cos(angle) * r);
    camera.lookAt(0, 0.06, 0);

    // orbit the point lights through the grid
    omniA.setLocalPosition(Math.sin(time * 0.7) * gridExtent * 0.4, 0.25, Math.cos(time * 0.7) * gridExtent * 0.4);
    omniB.setLocalPosition(Math.cos(time * 0.5) * gridExtent * 0.3, 0.15, Math.sin(time * 0.5) * gridExtent * 0.3);

    // per-type toggles so the cost of each light type can be measured on its own. Note the
    // draw cost keeps climbing for tens of seconds after a load or rebuild while pages stream
    // in and the LOD refines - compare toggles only once the frame time has settled.
    sun.enabled = !!data.get('data.sunLight');
    spot.enabled = !!data.get('data.spotLight');
    const omnisOn = !!data.get('data.omniLights');
    omniA.enabled = omnisOn;
    omniB.enabled = omnisOn;
    // sweep the spot ACROSS the grid rather than around it: a small orbit radius high above
    // the bunnies, aimed at a point circling the other way, so the cone crosses the field
    const spotAngle = time * 0.35;
    spot.setLocalPosition(
        Math.cos(spotAngle) * gridExtent * 0.18,
        gridExtent * 0.9,
        Math.sin(spotAngle) * gridExtent * 0.18
    );
    spot.lookAt(
        Math.cos(spotAngle + Math.PI) * gridExtent * 0.3,
        0.05,
        Math.sin(spotAngle + Math.PI) * gridExtent * 0.3
    );
    // lookAt aims the entity's -Z at the target, but a spot light emits down its -Y (the same
    // convention the shadow renderer undoes with rotateLocal(-90, 0, 0) to build its camera).
    // Without this the cone points 90 degrees off and the pool lands beside the target.
    spot.rotateLocal(90, 0, 0);
    sun.light.shadowDistance = gridExtent * 2.5;
    app.scene.envAtlas = data.get('data.ibl') ? assets.helipad.resource : null;

    const fogMode = data.get('data.fog');
    app.scene.fog.type = fogMode === 'linear' ? FOG_LINEAR : FOG_NONE;
    if (fogMode === 'linear') {
        app.scene.fog.color = new Color(0.08, 0.09, 0.11);
        app.scene.fog.start = gridExtent * 0.3;
        app.scene.fog.end = gridExtent * 1.6;
    }

    // second-view toggle: two cameras -> two independent views on one director
    const wantView2 = !!data.get('data.secondView');
    if (wantView2 !== camera2.enabled) {
        camera2.enabled = wantView2;
        inset.enabled = wantView2;
        director.cameras = wantView2 ? [camera.camera, camera2.camera] : [camera.camera];
    }
    if (wantView2) {
        const half = ((builtSize - 1) * SPACING) / 2;
        camera2.setLocalPosition(half + 0.35, 0.18, half + 0.35);
        camera2.lookAt(half, 0.08, half);
        // billboard screen standing on the far side of the grid (opposite the orbiting
        // camera), always in frame and always facing it
        const d = gridExtent * 0.62;
        inset.setLocalScale(gridExtent * 0.42, 1, gridExtent * 0.32);
        inset.setLocalPosition(-Math.sin(angle) * d, gridExtent * 0.16, -Math.cos(angle) * d);
        // lookAt points the entity's -Z at the camera; the plane's face is +Y, so pitch
        // it -90 to bring the face onto -Z (toward the camera)
        inset.lookAt(camera.getPosition());
        inset.rotateLocal(-90, 0, 0);
    }

    director.dagPixelThreshold = data.get('data.threshold');
    director.world.setColorMode(data.get('data.colorMode'));
    director.shadowsEnabled = !!data.get('data.shadows');

    // numeric oracle for the headless gates: per-cascade index demand. A cascade reporting 0
    // means its cull emitted nothing - the failure mode of a wrong LOD projection, which is
    // visually indistinguishable from "subtle" but obvious here.
    window.__meshletShadowStats = director.shadowRenderer.views.map((v) => ({
        light: v.light._node?.name ?? '?',
        type: v.light._type,
        face: v.face,
        orthoScale: v.culler.orthoScale,
        demand: v.lastDemand,
        indexCapacity: v.indexCapacity.slice()
    }));
});
