// @config
//
// Bakes the octahedral impostor atlases for the JungleRuins tree prototypes. Every prototype GLB
// (examples/utils/jungle/protos.py) is drawn from a grid of hemi-octahedral view directions with
// an orthographic camera framed on its bounding sphere, once with the engine's albedo debug pass
// and once with its world normal pass (the prototype sits unrotated, so world is object space).
// The views are supersampled, composed into one atlas per map and posted to
// examples/utils/jungle/bake_server.py, which dilates the edges and writes the PNGs.
//
// @flag WEBGL_DISABLED
// @flag WEBGPU_BARE_DISABLED
// @flag HIDDEN
// @flag HIDDEN_IN_DEV

// the bake is sequential by nature: one render target, read back view by view
/* eslint-disable no-await-in-loop */
import {
    AppBase,
    AppOptions,
    Asset,
    CameraComponentSystem,
    Color,
    ContainerHandler,
    Entity,
    FILTER_NEAREST,
    GAMMA_NONE,
    PIXELFORMAT_RGBA8,
    PROJECTION_ORTHOGRAPHIC,
    RenderComponentSystem,
    RenderTarget,
    SHADERPASS_ALBEDO,
    SHADERPASS_WORLDNORMAL,
    TONEMAP_NONE,
    Texture,
    TextureHandler,
    Vec3,
    createGraphicsDevice
} from 'playcanvas';

import { deviceType } from 'examples/context';

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('application-canvas'));
const params = new URLSearchParams(location.search);
// ?base=<dir>/ holds protos.json and the prototype GLBs; ?server= is bake_server.py
const BASE = params.get('base') ?? './assets/meshlets/jungle-protos/';
const SERVER = params.get('server') ?? 'http://localhost:5600';
// Views per side, pixels per view (tall: a tree's silhouette is about 0.6 as wide as it is high, so
// a square tile would waste most of its width) and supersampling per axis. Leaf textures are only
// 17-37% opaque, so minified alpha falls under the 0.5 cutoff and the foliage vanishes: each view
// is rendered large enough that the cutout stays sharp, then box filtered.
const GRID = Number(params.get('grid') ?? 8);
const TILE_W = Number(params.get('tw') ?? 240);
const TILE_H = Number(params.get('th') ?? 384);
const SS = Number(params.get('ss') ?? 6);
const only = params.get('only')?.split(',');
// Normals: the box mean over a texel's thousands of randomly angled leaf cards points about the
// same way everywhere and shades the tree flat, so by default each texel keeps the normal of its
// central samples only (one leaf's direction); ?normals=mean restores the box mean.
const NORMAL_CENTRE = params.get('normals') !== 'mean';
const ATLAS_W = GRID * TILE_W;
const ATLAS_H = GRID * TILE_H;
const TILE_SS_W = TILE_W * SS;
const TILE_SS_H = TILE_H * SS;

const status = document.createElement('pre');
status.style.cssText = 'position:fixed;left:8px;top:8px;margin:0;color:#fff;font:12px monospace;z-index:9';
document.body.appendChild(status);
const log = (/** @type {string} */ text) => {
    status.textContent = text;
    console.log(text);
};

const device = await createGraphicsDevice(canvas, { deviceTypes: [deviceType], antialias: false });
device.maxPixelRatio = 1;

const createOptions = new AppOptions();
createOptions.graphicsDevice = device;
createOptions.componentSystems = [CameraComponentSystem, RenderComponentSystem];
createOptions.resourceHandlers = [ContainerHandler, TextureHandler];
const app = new AppBase(canvas);
app.init(createOptions);
app.autoRender = false;
app.start();

const colorBuffer = new Texture(device, {
    name: 'impostor-tile',
    width: TILE_SS_W,
    height: TILE_SS_H,
    format: PIXELFORMAT_RGBA8,
    mipmaps: false,
    minFilter: FILTER_NEAREST,
    magFilter: FILTER_NEAREST
});
const renderTarget = new RenderTarget({ colorBuffer, depth: true, samples: 1 });

const camera = new Entity('Camera');
camera.addComponent('camera', {
    projection: PROJECTION_ORTHOGRAPHIC,
    orthoHeight: 1,
    nearClip: 0.01,
    farClip: 10,
    clearColor: new Color(0, 0, 0, 0),
    renderTarget,
    toneMapping: TONEMAP_NONE,
    gammaCorrection: GAMMA_NONE
});
app.root.addChild(camera);

/**
 * The view direction (towards the camera, object space) of an atlas frame: the hemi-octahedral
 * decode of the frame's grid position, y up. The runtime billboard must use the same mapping.
 *
 * @param {number} i - Frame column.
 * @param {number} j - Frame row.
 * @returns {Vec3} Unit direction.
 */
const frameDirection = (i, j) => {
    const gx = i / (GRID - 1);
    const gy = j / (GRID - 1);
    const x = gx - gy;
    const z = -1 + gx + gy;
    return new Vec3(x, 1 - Math.abs(x) - Math.abs(z), z).normalize();
};

const frame = () => new Promise((resolve) => {
    app.once('frameend', resolve);
    app.renderNextFrame = true;
});

/**
 * @param {number} c - Linear channel 0..1.
 * @returns {number} sRGB encoded channel.
 */
const encodeSrgb = c => (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);

/**
 * Renders the current view and resolves it to TILE_W x TILE_H: premultiplied box filter, then for
 * the normal pass a renormalise.
 *
 * @param {boolean} normals - True for the world normal pass.
 * @returns {Promise<Uint8Array>} RGBA8 tile.
 */
const renderTile = async (normals) => {
    camera.camera.setShaderPass(normals ? SHADERPASS_WORLDNORMAL : SHADERPASS_ALBEDO);
    await frame();
    const raw = await colorBuffer.read(0, 0, TILE_SS_W, TILE_SS_H, { renderTarget, immediate: true });
    const out = new Uint8Array(TILE_W * TILE_H * 4);
    const area = SS * SS;
    for (let y = 0; y < TILE_H; y++) {
        for (let x = 0; x < TILE_W; x++) {
            let r = 0, g = 0, b = 0, a = 0;
            let cr = 0, cg = 0, cb = 0, ca = 0;
            for (let sy = 0; sy < SS; sy++) {
                for (let sx = 0; sx < SS; sx++) {
                    const p = ((y * SS + sy) * TILE_SS_W + x * SS + sx) * 4;
                    const w = raw[p + 3];
                    r += raw[p] * w;
                    g += raw[p + 1] * w;
                    b += raw[p + 2] * w;
                    a += w;
                    if (Math.abs(sy - (SS - 1) / 2) <= 0.5 && Math.abs(sx - (SS - 1) / 2) <= 0.5) {
                        cr += raw[p] * w;
                        cg += raw[p + 1] * w;
                        cb += raw[p + 2] * w;
                        ca += w;
                    }
                }
            }
            if (normals && NORMAL_CENTRE && ca > 0) {
                r = cr / ca * a;
                g = cg / ca * a;
                b = cb / ca * a;
            }
            const o = (y * TILE_W + x) * 4;
            if (a > 0) {
                r /= a;
                g /= a;
                b /= a;
                if (!normals) {
                    // the albedo pass writes linear values; store sRGB (8 bit linear bands the darks)
                    r = encodeSrgb(r / 255) * 255;
                    g = encodeSrgb(g / 255) * 255;
                    b = encodeSrgb(b / 255) * 255;
                } else {
                    const nx = r / 127.5 - 1, ny = g / 127.5 - 1, nz = b / 127.5 - 1;
                    const len = Math.hypot(nx, ny, nz) || 1;
                    r = (nx / len * 0.5 + 0.5) * 255;
                    g = (ny / len * 0.5 + 0.5) * 255;
                    b = (nz / len * 0.5 + 0.5) * 255;
                }
                out[o] = r + 0.5;
                out[o + 1] = g + 0.5;
                out[o + 2] = b + 0.5;
                out[o + 3] = a / area + 0.5;
            }
        }
    }
    return out;
};

/**
 * @param {string} name - Output file name.
 * @param {Uint8Array} atlas - RGBA8 atlas.
 */
const post = async (name, atlas) => {
    const res = await fetch(`${SERVER}/atlas?name=${name}&w=${ATLAS_W}&h=${ATLAS_H}&grid=${GRID}`, { method: 'POST', body: atlas });
    if (!res.ok) throw new Error(`bake server rejected ${name}`);
};

const protos = /** @type {Array<{ name: string, file: string, center: number[], radius: number, extent: number[] }>} */ (
    await fetch(`${BASE}protos.json`).then(r => r.json()));
const selected = protos.filter(p => !only || only.includes(p.name));
const manifest = {
    version: 2, grid: GRID, hemi: true, atlasWidth: ATLAS_W, atlasHeight: ATLAS_H, prototypes: []
};

for (const [layer, proto] of protos.entries()) {
    if (!selected.includes(proto)) continue;
    log(`${proto.name}: loading (${layer + 1}/${protos.length})`);
    const asset = new Asset(proto.name, 'container', { url: BASE + proto.file });
    await new Promise((resolve, reject) => {
        asset.once('load', resolve);
        asset.once('error', reject);
        app.assets.add(asset);
        app.assets.load(asset);
    });
    const model = asset.resource.instantiateRenderEntity();
    // a unit bounding sphere about the origin: every view then frames the same orthoHeight
    const scale = 1 / proto.radius;
    // frame the silhouette the views actually reach (protos.py), not the whole sphere: half height
    // in unit-sphere units, wide enough for the half width at the tile's aspect, plus a margin.
    // The runtime billboard spans exactly this box, so it goes into the manifest.
    const halfH = Math.max(proto.extent[1], proto.extent[0] * TILE_H / TILE_W) / proto.radius * 1.03;
    const halfW = halfH * TILE_W / TILE_H;
    camera.camera.orthoHeight = halfH;
    model.setLocalScale(scale, scale, scale);
    model.setLocalPosition(-proto.center[0] * scale, -proto.center[1] * scale, -proto.center[2] * scale);
    app.root.addChild(model);

    const albedo = new Uint8Array(ATLAS_W * ATLAS_H * 4);
    const normal = new Uint8Array(ATLAS_W * ATLAS_H * 4);
    for (let j = 0; j < GRID; j++) {
        for (let i = 0; i < GRID; i++) {
            const dir = frameDirection(i, j);
            camera.setPosition(dir.x * 3, dir.y * 3, dir.z * 3);
            camera.lookAt(Vec3.ZERO, Math.abs(dir.y) > 0.999 ? Vec3.BACK : Vec3.UP);
            for (const [normals, atlas] of [[false, albedo], [true, normal]]) {
                const tile = await renderTile(normals);
                for (let y = 0; y < TILE_H; y++) {
                    atlas.set(tile.subarray(y * TILE_W * 4, (y + 1) * TILE_W * 4), ((j * TILE_H + y) * ATLAS_W + i * TILE_W) * 4);
                }
            }
        }
        log(`${proto.name}: row ${j + 1}/${GRID}`);
    }
    await post(`impostor_albedo_${layer}.png`, albedo);
    await post(`impostor_normal_${layer}.png`, normal);
    manifest.prototypes.push({
        name: proto.name,
        layer,
        center: proto.center,
        radius: proto.radius,
        half: [halfW * proto.radius, halfH * proto.radius]
    });

    model.destroy();
    asset.unload();
    app.assets.remove(asset);
}

await fetch(`${SERVER}/manifest`, { method: 'POST', body: JSON.stringify(manifest) });
log(`done: ${manifest.prototypes.length} prototypes`);
// @ts-ignore
window.__bakeDone = true;
