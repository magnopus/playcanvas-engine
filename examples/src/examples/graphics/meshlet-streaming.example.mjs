// @config
//
// GPU-driven meshlet rendering: the Bistro (100k meshlets, 166 MB of geometry pages) streams on
// demand over HTTP Range requests from its pinned root pages - the GPU cull pass marks the
// pages its LOD cut needs, missing clusters render their nearest resident ancestor, and the
// scene refines coarse to fine with no holes. A grid of Stanford bunnies (a separately baked
// asset sharing the world) sits in the street. Two-phase HZB occlusion culling can be toggled:
// watch the triangles-drawn counter drop when the buildings occlude the street behind them.
//
// @flag WEBGL_DISABLED
// @flag HIDDEN

import {
    ADDRESS_CLAMP_TO_EDGE,
    BLEND_NORMAL,
    FILLMODE_NONE,
    FOG_LINEAR,
    FOG_NONE,
    RESOLUTION_AUTO,
    AppBase,
    AppOptions,
    Asset,
    AssetListLoader,
    CameraComponentSystem,
    Color,
    ContainerHandler,
    Entity,
    FILTER_NEAREST,
    LightComponentSystem,
    Mat4,
    MeshletComponentSystem,
    PIXELFORMAT_BGRA8,
    PIXELFORMAT_DEPTH,
    Quat,
    RenderComponentSystem,
    RenderTarget,
    StandardMaterial,
    TEXTURETYPE_RGBP,
    TRACEID_GPU_TIMINGS,
    Texture,
    TextureHandler,
    Tracing,
    Vec3,
    basisInitialize,
    createGraphicsDevice,
    math
} from 'playcanvas';

import { data, deviceType } from 'examples/context';

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('application-canvas'));
window.focus();

// the Bistro's streamed textures are KTX2/Basis - initialize the transcoder
basisInitialize({
    glueUrl: './assets/wasm/basis/basis.wasm.js',
    wasmUrl: './assets/wasm/basis/basis.wasm.wasm',
    fallbackUrl: './assets/wasm/basis/basis.js'
});

const assets = {
    bistro: new Asset('bistro', 'container', { url: './assets/meshlets/bistro-v2.glb' }),
    bunny: new Asset('bunny', 'container', { url: './assets/meshlets/bunny-v2.glb' }),
    helipad: new Asset(
        'helipad-env-atlas',
        'texture',
        { url: './assets/cubemaps/helipad-env-atlas.png' },
        { type: TEXTURETYPE_RGBP, mipmaps: false }
    )
};

const gfxOptions = {
    deviceTypes: [deviceType]
};

const device = await createGraphicsDevice(canvas, gfxOptions);
device.maxPixelRatio = 1;

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

await new Promise((resolve) => {
    new AssetListLoader(Object.values(assets), app.assets).load(resolve);
});

app.setCanvasFillMode(FILLMODE_NONE);
app.setCanvasResolution(RESOLUTION_AUTO);
app.start();
app.resizeCanvas();

// the example runs in an iframe and the example browser does not forward its query string, so
// merge the top window's params in (lowest priority) - the ?flags then work on the outer
// /#/graphics/meshlet-streaming URL as well as on the /iframe/ one
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

// GPU timing: the panel reports whole-frame GPU cost, which is what the occlusion toggle is
// supposed to move. ?timers=1 additionally logs the per-pass breakdown to the console.
device.gpuProfiler.enabled = true;
if (params.get('timers') === '1') {
    Tracing.set(TRACEID_GPU_TIMINGS, true);
}

const bistroResource = assets.bistro.resource.meshlets[0];
const bunnyResource = assets.bunny.resource.meshlets[0];

data.set('data', {
    colorMode: parseInt(params.get('color') ?? '0', 10),
    gridSize: parseInt(params.get('grid') ?? '4', 10),
    threshold: parseFloat(params.get('threshold') ?? '1'),
    occlusion: params.get('occlusion') !== '0',
    lights: params.get('lights') !== '0',
    ibl: params.get('ibl') !== '0',
    fog: params.get('fog') ?? 'none',
    slab: params.get('slab') === '1',
    occluder: params.get('occluder') !== '0',
    bunnies: true,
    texPoolMb: parseInt(params.get('texpool') ?? '96', 10),
    // Geometry VRAM budget - the WHOLE geometry side, not just the page pool. The Bistro needs
    // ~300 MB of pages alone; 128 (the old page-pool-only default) leaves it thousands of pages
    // short, which shows as holes rather than lower detail.
    geoPoolMb: parseInt(params.get('geopool') ?? '512', 10),
    shadows: params.get('shadows') !== '0',
    cascades: Math.max(1, Math.min(4, parseInt(params.get('cascades') ?? '2', 10))),
    shadowDist: parseFloat(params.get('shadowdist') ?? '0'),
    stats: '',
    texStats: '',
    budgetWarning: '',
    geoPct: 0,
    geoBar: '',
    texPct: 0,
    texBar: ''
});

// the component system owns the director (and wires the KTX2 transcoder); this example
// drives the director directly instead of adding meshlet components
const director = app.systems.meshlet.director;
// set before the first build - the budget reserves per-view state, cascades included
director.shadowsEnabled = !!data.get('data.shadows');
director.world.poolBytes = Math.round(data.get('data.geoPoolMb')) * 1024 * 1024;
director.world.initialIndices = 2 * 1024 * 1024;
// surface an unachievable budget in the panel rather than leaving the user to wonder why the
// scene has holes in it
director.onBudgetExceeded = (info) => {
    data.set(
        'data.budgetWarning',
        `budget too small: ${info.pagesMissing} pages short, try ${Math.ceil(info.suggestedBudgetBytes / 1048576)} MB`
    );
};
// debug hook: the director is reachable from the console / headless probes
window.__meshletDirector = director;
window.__data = data; // ...and the panel observer, so the controls can be driven headlessly

// The bunny field is a fixed-width column of rows that grows AWAY from the camera: its front
// row is pinned to the street, and raising the count adds rows down the street rather than
// spreading outwards. That keeps the nearest bunny at a fixed distance, so the occluder wall
// in front of it neither creeps towards the camera nor swells to fill the screen as the
// count rises - only the field's (shrinking) angular extent changes.
const BUNNY_SCALE = 15;
const BUNNY_SPACING = 2.6;
const BUNNY_HEIGHT = BUNNY_SCALE * 0.15;
const FIELD_COLUMNS = 10; // as wide as the street will take; the rest becomes depth
const streetCenter = new Vec3();
const fieldRight = new Vec3();
const fieldAway = new Vec3();
/** @type {Vec3[]} - the field's 8 bounding corners, for sizing the occluder wall */
const fieldCorners = [];

const buildBunnyInstances = (n) => {
    const instances = [];
    const m = new Mat4();
    const pos = new Vec3();
    const rot = new Quat();
    const scale = new Vec3(BUNNY_SCALE, BUNNY_SCALE, BUNNY_SCALE);
    const count = n * n;
    const cols = Math.min(n, FIELD_COLUMNS);
    const rows = Math.ceil(count / cols);
    const latOf = (c) => (c - (cols - 1) / 2) * BUNNY_SPACING;

    for (let i = 0; i < count; i++) {
        const c = i % cols;
        const r = Math.floor(i / cols);
        const lat = latOf(c);
        const dep = r * BUNNY_SPACING;
        pos.set(
            streetCenter.x + fieldRight.x * lat + fieldAway.x * dep,
            streetCenter.y,
            streetCenter.z + fieldRight.z * lat + fieldAway.z * dep
        );
        rot.setFromEulerAngles(0, (c * 31 + r * 17) % 360, 0);
        m.setTRS(pos, rot, scale);
        instances.push({ primIndex: 0, matrix: new Float32Array(m.data) });
    }

    // bounding corners of the whole field (plus one bunny's footprint), used by the wall
    fieldCorners.length = 0;
    const pad = BUNNY_SPACING * 0.5;
    for (const lat of [latOf(0) - pad, latOf(cols - 1) + pad]) {
        for (const dep of [-pad, (rows - 1) * BUNNY_SPACING + pad]) {
            for (const y of [0, BUNNY_HEIGHT]) {
                fieldCorners.push(
                    new Vec3(
                        streetCenter.x + fieldRight.x * lat + fieldAway.x * dep,
                        streetCenter.y + y,
                        streetCenter.z + fieldRight.z * lat + fieldAway.z * dep
                    )
                );
            }
        }
    }
    return instances;
};

let builtSize = -1;
let rebuildCount = -1; // first build is not a "re"-build
let bunniesHidden = false;
const rebuild = (n) => {
    builtSize = n;
    rebuildCount++;
    bunniesHidden = false;
    director.rebuild((world) => {
        world.addStreamedResource(bistroResource, null, './assets/meshlets');
        if (n > 0) {
            world.addStreamedResource(bunnyResource, null, './assets/meshlets', buildBunnyInstances(n));
        }
    });
};
rebuild(0);
const world = director.world;

const center = world.worldBounds.center.clone();
const radius = world.worldBounds.halfExtents.length();
// the Bistro is authored with the street at y = 0; the world bounds are sphere-inflated and
// useless for ground placement, so drop the grid onto the street in front of the fixed camera
const camA = 0.6;
const camR = radius * 0.3;
streetCenter.set(center.x + Math.sin(camA) * camR * 0.45, 0, center.z + Math.cos(camA) * camR * 0.45);
// the field's basis: "away" points from the fixed camera vantage down the street, "right" is
// its horizontal perpendicular. Rows march along away, columns along right.
fieldAway.set(-Math.sin(camA), 0, -Math.cos(camA)).normalize();
fieldRight.cross(fieldAway, Vec3.UP).normalize();
rebuild(data.get('data.gridSize'));

// occlusion needs a sampleable depth texture: render the camera into an offscreen target and
// blit the color to the backbuffer at the end of the frame
const width = device.width;
const height = device.height;
const colorTexture = new Texture(device, {
    name: 'SceneColor',
    width,
    height,
    format: PIXELFORMAT_BGRA8,
    mipmaps: false,
    minFilter: FILTER_NEAREST,
    magFilter: FILTER_NEAREST,
    addressU: ADDRESS_CLAMP_TO_EDGE,
    addressV: ADDRESS_CLAMP_TO_EDGE
});
const depthTexture = new Texture(device, {
    name: 'SceneDepth',
    width,
    height,
    format: PIXELFORMAT_DEPTH,
    mipmaps: false,
    minFilter: FILTER_NEAREST,
    magFilter: FILTER_NEAREST,
    addressU: ADDRESS_CLAMP_TO_EDGE,
    addressV: ADDRESS_CLAMP_TO_EDGE
});
const renderTarget = new RenderTarget({ colorBuffer: colorTexture, depthBuffer: depthTexture });

const camera = new Entity();
camera.addComponent('camera', {
    clearColor: new Color(0.08, 0.09, 0.11),
    nearClip: Math.max(radius * 0.001, 0.01),
    farClip: radius * 10
});
camera.camera.renderTarget = renderTarget;
app.root.addChild(camera);
director.cameraComponent = camera.camera;

// engine lighting: IBL env atlas + a shadowed sun over the street
app.scene.envAtlas = assets.helipad.resource;
app.scene.skyboxIntensity = 0.4;
app.scene.ambientLight = new Color(0.02, 0.02, 0.025);

const sun = new Entity('Sun');
sun.addComponent('light', {
    type: 'directional',
    color: new Color(1, 0.95, 0.85),
    intensity: 1.2,
    castShadows: true,
    shadowBias: 0.2,
    normalOffsetBias: 0.5,
    shadowResolution: 2048
});
sun.setEulerAngles(55, 25, 0);
app.root.addChild(sun);

// reference StandardMaterial sphere in the street - lit meshlets must match its shading
const refMaterial = new StandardMaterial();
refMaterial.diffuse.set(1, 1, 1);
refMaterial.useMetalness = true;
refMaterial.metalness = 0;
refMaterial.gloss = 0;
refMaterial.update();
const refSphere = new Entity('RefSphere');
refSphere.addComponent('render', { type: 'sphere', material: refMaterial, castShadows: true });
refSphere.setLocalScale(2, 2, 2);
refSphere.setLocalPosition(streetCenter.x + 8, 2, streetCenter.z + 4);
app.root.addChild(refSphere);

// transparent test slab straddling the street - verifies draw ordering: it must composite
// OVER the meshlets behind it and UNDER anything in front (meshlets draw before transparents)
const slabMaterial = new StandardMaterial();
slabMaterial.diffuse.set(0.2, 0.6, 1.0);
slabMaterial.blendType = BLEND_NORMAL;
slabMaterial.opacity = 0.45;
slabMaterial.update();
const slab = new Entity('TransparentSlab');
slab.addComponent('render', { type: 'box', material: slabMaterial });
slab.setLocalScale(14, 8, 0.4);
slab.setLocalPosition(streetCenter.x, 4, streetCenter.z - 6);
app.root.addChild(slab);

// occluder wall - a REGULAR render component, not a meshlet. The meshlet pass order draws
// meshlet depth first, then the scene's opaque geometry, and only then builds the HZB, so
// ordinary scene meshes occlude meshlets too. This one slides across in front of the bunny
// grid: with occlusion on, the whole grid stops being drawn a frame later; with it off, every
// bunny still pays full vertex and pixel cost behind the wall.
// unlit on purpose: the wall is a measuring instrument, and a lit surface covering this much
// of the screen would cost more to shade than the bunnies it hides save
const wallMaterial = new StandardMaterial();
wallMaterial.useLighting = false;
wallMaterial.diffuse.set(0, 0, 0);
wallMaterial.emissive.set(0.22, 0.2, 0.19);
wallMaterial.update();
const occluder = new Entity('OccluderWall');
occluder.addComponent('render', { type: 'box', material: wallMaterial, castShadows: false });
app.root.addChild(occluder);

app.on('postrender', () => {
    device.copyRenderTarget(renderTarget, null, true, false);
});

// ?wall=closed|open parks the occluder instead of sliding it
const wallMode = params.get('wall') ?? 'slide';

const _gridPos = new Vec3();
const _wallDir = new Vec3();
const _wallRight = new Vec3();
const _wallUp = new Vec3();
const _tmpVec = new Vec3();

// how far the field reaches down the street for the current count
const fieldDepth = () => {
    const count = builtSize * builtSize;
    const cols = Math.min(builtSize, FIELD_COLUMNS);
    return cols > 0 ? (Math.ceil(count / cols) - 1) * BUNNY_SPACING : 0;
};

let angle = 0;
let time = 0;
let sizeChangedAt = -1;
let statFrames = 0;

app.on('update', (dt) => {
    time += dt;

    // debounce the bunny + pool sliders - all three rebuild the world (geometry pages and
    // texture slots survive a rebuild unless their own budget changed)
    const wanted = Math.round(data.get('data.gridSize'));
    const wantedTexPool = Math.round(data.get('data.texPoolMb')) * 1024 * 1024;
    const wantedGeoPool = Math.round(data.get('data.geoPoolMb')) * 1024 * 1024;
    if (
        wanted !== builtSize ||
        wantedTexPool !== director.world.texturePoolBytes ||
        wantedGeoPool !== director.world.poolBytes
    ) {
        if (sizeChangedAt < 0) sizeChangedAt = time;
        if (time - sizeChangedAt > 0.4) {
            sizeChangedAt = -1;
            director.world.texturePoolBytes = wantedTexPool;
            director.world.poolBytes = wantedGeoPool;
            rebuild(wanted);
        }
    } else {
        sizeChangedAt = -1;
    }

    // ?freeze=1 stops the camera sway (isolates temporal artifacts from motion)
    if (params.get('freeze') !== '1') angle += dt * 0.25;
    // fixed low vantage outside the block (an orbit would clip through buildings); the gentle
    // look sway keeps disocclusion happening so phase 2 stays visibly active
    const a = 0.6;
    const r = radius * 0.3;
    camera.setLocalPosition(center.x + Math.sin(a) * r, center.y + radius * 0.02, center.z + Math.cos(a) * r);
    // look at the bunny grid down the street (the occlusion test's subject), with a gentle
    // sway so disocclusion keeps happening and phase 2 stays visibly active
    camera.lookAt(
        streetCenter.x + Math.sin(angle) * radius * 0.02,
        2.0 + Math.cos(angle * 0.7) * radius * 0.008,
        streetCenter.z + Math.cos(angle * 0.8) * radius * 0.02
    );

    // ?near=1 parks the camera by the shopfront - exercises fine texture streaming
    if (params.get('near') === '1') {
        camera.setLocalPosition(streetCenter.x + 6, 2.5, streetCenter.z + 6);
        camera.lookAt(streetCenter.x + 20, 3, streetCenter.z - 2);
    }

    // slide the occluder wall across the camera-to-grid line. It is sized from the grid's
    // angular extent so a full slide hides every bunny, and dwells at both ends (the smooth
    // clamp) so the triangle and GPU-time counters settle before it moves again.
    const occluderOn = data.get('data.occluder');
    occluder.enabled = occluderOn && builtSize > 0 && fieldCorners.length > 0;
    if (occluder.enabled) {
        const camPos = camera.getPosition();

        // view basis aimed at the field's centre
        _gridPos
            .set(streetCenter.x, BUNNY_HEIGHT * 0.5, streetCenter.z)
            .add(_tmpVec.copy(fieldAway).mulScalar(fieldDepth() * 0.5));
        _wallDir.sub2(_gridPos, camPos).normalize();
        _wallRight.cross(_wallDir, Vec3.UP).normalize();
        _wallUp.cross(_wallRight, _wallDir).normalize();

        // The field is deep, not a point: the wall must stand clear of its NEAREST corner or
        // the front rows end up in front of it and can never be occluded. Size then comes from
        // projecting every corner onto the wall plane, which covers the whole field exactly -
        // and because the field grows away from the camera, adding rows barely moves it.
        let nearFwd = Infinity;
        for (const c of fieldCorners) {
            _tmpVec.sub2(c, camPos);
            nearFwd = Math.min(nearFwd, _tmpVec.dot(_wallDir));
        }
        const wallDist = Math.max(nearFwd - BUNNY_SPACING, 1);

        let halfW = 0;
        let halfH = 0;
        for (const c of fieldCorners) {
            _tmpVec.sub2(c, camPos);
            const fwd = Math.max(_tmpVec.dot(_wallDir), 0.001);
            const k = wallDist / fwd;
            halfW = Math.max(halfW, Math.abs(_tmpVec.dot(_wallRight)) * k);
            halfH = Math.max(halfH, Math.abs(_tmpVec.dot(_wallUp)) * k);
        }
        halfW *= 1.08;
        halfH *= 1.08;

        // dwell open / dwell closed: a sine pushed past +-1 and clamped. ?wall=closed|open
        // parks it, which is how the occlusion win is measured with a steady counter.
        const cover =
            wallMode === 'closed'
                ? 1
                : wallMode === 'open'
                  ? 0
                  : math.clamp(Math.sin(time * 0.55) * 1.9, -1, 1) * 0.5 + 0.5;
        const slide = (1 - cover) * (halfW * 2 + wallDist);
        occluder.setLocalScale(halfW * 2, halfH * 2, 0.5);
        occluder.setPosition(
            camPos.x + _wallDir.x * wallDist + _wallRight.x * slide,
            camPos.y + _wallDir.y * wallDist + _wallRight.y * slide,
            camPos.z + _wallDir.z * wallDist + _wallRight.z * slide
        );
        occluder.lookAt(camPos);
    }

    // bunny visibility = per-instance hide bit, no rebuild and no re-streaming
    const bunniesVisible = data.get('data.bunnies');
    if (builtSize > 0 && director.world.placements.length > 1 && bunniesHidden === bunniesVisible) {
        bunniesHidden = !bunniesVisible;
        director.world.setPlacementHidden(1, bunniesHidden);
    }

    slab.enabled = data.get('data.slab');
    sun.enabled = data.get('data.lights');

    // meshlet shadow casting. Cascade count and shadow distance together set the light-space
    // texel rate, which is what decides whether cast detail is visible at all - a single
    // cascade stretched over the whole block gives each shopfront a couple of texels.
    director.shadowsEnabled = !!data.get('data.shadows');
    sun.light.numCascades = Math.round(data.get('data.cascades'));
    const dist = data.get('data.shadowDist');
    sun.light.shadowDistance = dist > 0 ? dist : radius * 0.6;

    app.scene.envAtlas = data.get('data.ibl') ? assets.helipad.resource : null;
    const fogMode = data.get('data.fog');
    app.scene.fog.type = fogMode === 'linear' ? FOG_LINEAR : FOG_NONE;
    if (fogMode === 'linear') {
        app.scene.fog.color = new Color(0.55, 0.6, 0.7);
        app.scene.fog.start = radius * 0.1;
        app.scene.fog.end = radius * 0.9;
    }

    director.occlusionEnabled = data.get('data.occlusion');
    director.dagPixelThreshold = data.get('data.threshold');
    director.world.setColorMode(data.get('data.colorMode'));

    // The view already reads its own index-demand counters back each frame to drive buffer
    // growth - reuse that rather than mapping the counters a second time. With two-phase on
    // this is the occlusion-culled set, so toggling occlusion makes the count visibly drop.
    const view = director.views.get(camera.camera);
    if (view?.lastDemand && ++statFrames % 30 === 0) {
        const tris = Math.round(view.lastDemand.indices.reduce((a, b) => a + b, 0) / 3);
        const res = director.residency;
        const gpuMs = device.gpuProfiler._frameTime;
        data.set(
            'data.stats',
            `${(tris / 1e6).toFixed(2)}M tris${gpuMs > 0 ? `, ${gpuMs.toFixed(2)} ms GPU` : ''}, ` +
                `${res ? res.residentPages : 0}/${director.world.totalPages} pages, ` +
                `${res ? res.fetchedPages : 0} fetched, ${Math.max(rebuildCount, 0)} rebuilds`
        );
        const tex = director.world.textures;
        if (tex) {
            data.set(
                'data.texStats',
                `${tex.slotsUsed}/${tex.slotsTotal} fine slots, ${tex.fineFetches} fetched, ` +
                    `${tex.fineEvictions} evicted, bias ${tex.mipBias.toFixed(2)}`
            );
        }
        // pool occupancy bars: bytes held vs the byte budget, plus the geometry
        // pressure scale (the quality cost of fitting the budget; 1 = full quality)
        const mb = (b) => (b / 1048576).toFixed(0);
        const pageBytes = director.world.pageSizeBytes;
        // whole geometry budget, not just the page pool - storage + index buffers is what the
        // meshlet pipeline actually holds for geometry
        const geoUsed = device._vram.sb + device._vram.ib;
        const geoMax = director.world.poolBytes;
        const pool = (res ? res.residentPages : 0) * pageBytes;
        data.set('data.geoPct', geoMax > 0 ? Math.min((100 * geoUsed) / geoMax, 100) : 0);
        data.set(
            'data.geoBar',
            `${mb(geoUsed)}/${mb(geoMax)} MB geo VRAM (${mb(pool)} MB pool), ` +
                `pressure x${director.budget.pressureScale.toFixed(1)}`
        );
        // tails + fine slots against the whole texture budget, matching the geo bar
        const texUsed = tex ? tex.bytesUsed : 0;
        data.set('data.texPct', tex && tex.poolBytes > 0 ? Math.min((100 * texUsed) / tex.poolBytes, 100) : 0);
        data.set(
            'data.texBar',
            `${mb(texUsed)}/${mb(tex ? tex.poolBytes : 0)} MB tex VRAM (${mb(tex ? tex.tailBytes : 0)} MB tails)`
        );
    }
});
