import { expect } from 'chai';

import { Mat4 } from '../../../src/core/math/mat4.js';
import { Vec3 } from '../../../src/core/math/vec3.js';
import { BoundingBox } from '../../../src/core/shape/bounding-box.js';
import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import { LAYERID_WORLD, LIGHTTYPE_DIRECTIONAL, LIGHTTYPE_OMNI, LIGHTTYPE_SPOT, SHADOWUPDATE_NONE, SHADOWUPDATE_REALTIME, SHADOWUPDATE_THISFRAME } from '../../../src/scene/constants.js';
import { GraphNode } from '../../../src/scene/graph-node.js';
import { ShaderMaterial } from '../../../src/scene/materials/shader-material.js';
import { CULL_FLAG_NO_TEXEL_RATE, CULL_FLAG_SHADOW_VIEW, MESHLET_BUCKET_COUNT, WORK_ITEM_U32S } from '../../../src/scene/meshlet/constants.js';
import { MeshletCullShaders } from '../../../src/scene/meshlet/meshlet-cull-shaders.js';
import { MeshletShadowRenderer } from '../../../src/scene/meshlet/meshlet-shadow-renderer.js';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

// null device dressed up as compute-capable, as in meshlet-culler.test.mjs
const makeDevice = () => {
    const device = new NullGraphicsDevice(document.createElement('canvas'));
    device.supportsCompute = true;
    device.createComputeImpl = () => ({ destroy() {} });
    device.createBufferImpl = () => ({
        allocate() {}, write() {}, destroy() {}, loseContext() {}, buffer: {}
    });
    device.getCommandEncoder = () => ({ copyBufferToBuffer() {}, clearBuffer() {} });
    device.computeDispatch = () => {};
    device.getIndirectDrawSlot = () => 0;
    device.getIndirectDispatchSlot = () => 0;
    return device;
};

const PAIRS = 1000;
const materials = () => Array.from({ length: MESHLET_BUCKET_COUNT }, () => new ShaderMaterial());

// the world spans x, z in [-100, 100] and y in [0, 100]
const makeWorld = () => {
    const lit = materials();
    return {
        finalized: true,
        totalPairs: PAIRS,
        workItemCapacity: 50,
        cut: { rootIndices: [0, 0, 0], rootRecords: 0, stageLevels: [] },
        cells: { cellCount: 1, boundsBuffer: {}, instancesBuffer: {} },
        recordCapacity: 4000,
        initialRecords: 100,
        instanceCount: 7,
        pageSizeBytes: 65536,
        totalPages: 3,
        indexCapacity: [300, 60, 30],
        indexWorst: [3000, 600, 300],
        indexCeiling: 100000,
        maxIndices: 0,
        dummyBits: { destroy() {} },
        objectDataBuffer: {},
        meshletDataBuffer: {},
        residencyBuffer: {},
        requestsBuffer: {},
        pagePool: {},
        litMaterials: lit,
        bucketMaterials: lit,
        worldBounds: new BoundingBox(new Vec3(0, 50, 0), new Vec3(100, 50, 100))
    };
};

const makeLayer = () => ({
    id: LAYERID_WORLD,
    casters: [],
    addShadowCasters(instances) {
        this.casters.push(...instances);
    },
    removeShadowCasters(instances) {
        this.casters = this.casters.filter(mi => !instances.includes(mi));
    }
});

let nextLightId = 1;
const makeLight = ({ type = LIGHTTYPE_DIRECTIONAL, faces = 3, position = new Vec3(), forward = new Vec3(0, -1, 0), range = 10, cascades = null, renderData = {} } = {}) => ({
    id: nextLightId++,
    enabled: true,
    castShadows: true,
    visibleThisFrame: true,
    _type: type,
    numShadowFaces: faces,
    attenuationEnd: range,
    _shadowResolution: 2048,
    _shadowCascadeDistances: cascades,
    _node: { name: 'light', forward, getPosition: () => position },
    getRenderData: (camera, face) => renderData[face] ?? null
});

// a scene camera whose cascade slices are 20-unit-wide boxes between the two clip distances
const HALF_WIDTH = 10;
const makeSceneCamera = () => ({
    _nearClip: 0.5,
    node: { getWorldTransform: () => new Mat4() },
    getFrustumCorners(near, far) {
        const pts = [];
        for (const z of [-near, -far]) {
            for (const [x, y] of [[1, -1], [1, 1], [-1, 1], [-1, -1]]) {
                pts.push(new Vec3(x * HALF_WIDTH, y * HALF_WIDTH, z));
            }
        }
        return pts;
    }
});

describe('MeshletShadowRenderer', function () {

    let device;
    let shaders;
    let world;
    let director;
    let layer;
    let comp;
    let sceneCamera;
    let renderer;

    beforeEach(function () {
        jsdomSetup();
        device = makeDevice();
        shaders = new MeshletCullShaders(device);
        world = makeWorld();
        sceneCamera = makeSceneCamera();
        director = {
            world,
            renderer: { culler: { cameraDirShadowLights: new Map() }, localLights: [] },
            cullShaders: shaders,
            takeShadowCapCarry: () => null,
            shadowInitialIndexScale: 0.5,
            dagPixelThreshold: 2,
            pixelScale: 1,
            shadowThresholdScale: 1.5,
            budget: { pressureScale: 1.25 }
        };
        layer = makeLayer();
        comp = { getLayerById: id => (id === LAYERID_WORLD ? layer : null) };
        renderer = new MeshletShadowRenderer(device, director);
    });

    afterEach(function () {
        renderer.destroy();
        shaders.destroy();
        device.destroy();
        device = null;
        jsdomTeardown();
    });

    it('builds one caster view per face for the first camera\'s shadow-casting lights', function () {
        const sun = makeLight({ faces: 3 });
        const off = makeLight({ faces: 2 });
        off.castShadows = false;
        const otherCameraSun = makeLight({ faces: 1 });
        director.renderer.culler.cameraDirShadowLights.set(sceneCamera, [sun, off]);
        director.renderer.culler.cameraDirShadowLights.set({}, [otherCameraSun]);

        renderer.syncLights(comp);
        expect([...renderer.entries.keys()]).to.deep.equal([sun]);
        const entry = renderer.entries.get(sun);
        expect(entry.camera).to.equal(sceneCamera);
        expect(entry.views).to.have.lengthOf(3);
        expect(entry.views.map(v => v.face)).to.deep.equal([0, 1, 2]);
        expect(entry.views.every(v => v.singlePhase && v.light === sun && v.sceneCamera === sceneCamera)).to.equal(true);
        expect(layer.casters, 'three buckets per face registered as casters').to.have.lengthOf(3 * MESHLET_BUCKET_COUNT);
        expect(entry.views[0].indexCapacity, 'shadow views start at the scaled budget').to.deep.equal([150, 30, 15]);

        // one claim plane and one work-item buffer shared by every face
        const claim = entry.views[0].claimBitsBuffer;
        expect(entry.views.every(v => v.claimBitsBuffer === claim && v.workItemsBuffer === renderer._workItems)).to.equal(true);
        // pair bits, admission bits, then one off-frustum refinement flag per pair
        expect(claim.byteSize).to.equal((Math.ceil((PAIRS * 2 + 7) / 32) + 7 + 2 * Math.ceil(7 / 32)) * 4, 'transient bits, then two persistent admission regions');
        expect(renderer._workItems.byteSize).to.equal(50 * WORK_ITEM_U32S * 4);
        expect(renderer.views).to.have.lengthOf(3);
    });

    it('retires entries whose light left, stopped casting or changed its face count', function () {
        const sun = makeLight({ faces: 3 });
        director.renderer.culler.cameraDirShadowLights.set(sceneCamera, [sun]);
        renderer.syncLights(comp);
        const first = renderer.entries.get(sun);

        sun.numShadowFaces = 2;
        renderer.syncLights(comp);
        expect(renderer.entries.get(sun), 'recreated with the new face count').to.not.equal(first);
        expect(renderer.entries.get(sun).views).to.have.lengthOf(2);
        expect(layer.casters).to.have.lengthOf(2 * MESHLET_BUCKET_COUNT);

        sun.castShadows = false;
        renderer.syncLights(comp);
        expect(renderer.entries.size).to.equal(0);
        expect(layer.casters, 'casters unregistered').to.have.lengthOf(0);
    });

    it('caps the total view count and refuses a light that does not fit', function () {
        renderer.maxShadowViews = 4;
        const sun = makeLight({ faces: 3 });
        const omni = makeLight({ type: LIGHTTYPE_OMNI, faces: 6 });
        director.renderer.culler.cameraDirShadowLights.set(sceneCamera, [sun]);
        director.renderer.localLights = [omni];
        renderer.syncLights(comp);
        expect(renderer.entries.has(sun)).to.equal(true);
        expect(renderer.entries.has(omni), 'six faces do not fit beside three').to.equal(false);
    });

    it('serves local lights only while visible and only when enabled', function () {
        const spot = makeLight({ type: LIGHTTYPE_SPOT, faces: 1 });
        director.renderer.localLights = [spot];
        renderer.localLights = false;
        renderer.syncLights(comp);
        expect(renderer.entries.size).to.equal(0);

        renderer.localLights = true;
        renderer.syncLights(comp);
        expect(renderer.entries.get(spot).camera, 'local shadow data is camera-independent').to.equal(null);
        spot.visibleThisFrame = false;
        renderer.syncLights(comp);
        expect(renderer.entries.size).to.equal(0);
    });

    it('does nothing without a World layer or a finalized world', function () {
        director.renderer.culler.cameraDirShadowLights.set(sceneCamera, [makeLight()]);
        renderer.syncLights({ getLayerById: () => null });
        expect(renderer.entries.size).to.equal(0);
        world.finalized = false;
        renderer.syncLights(comp);
        expect(renderer.entries.size).to.equal(0);
    });

    it('bounds a local light\'s casters by its attenuation sphere clipped to the world', function () {
        const spot = makeLight({ type: LIGHTTYPE_SPOT, faces: 1, position: new Vec3(95, 5, 0), range: 10 });
        const box = renderer._casterBounds({ light: spot, camera: null }, { face: 0 });
        expect(box.getMin().equals(new Vec3(85, 0, -10))).to.equal(true);
        expect(box.getMax().equals(new Vec3(100, 15, 10))).to.equal(true);

        const outside = makeLight({ type: LIGHTTYPE_SPOT, faces: 1, position: new Vec3(500, 0, 0), range: 10 });
        expect(renderer._casterBounds({ light: outside, camera: null }, { face: 0 })).to.equal(world.worldBounds);
    });

    it('bounds a cascade by its frustum slice padded along the light, or the world before the first fit', function () {
        const sun = makeLight({ cascades: null });
        const shadowNode = { forward: new Vec3(0, -1, 0) };
        const view = { face: 1, shadowCamera: { _node: shadowNode } };
        expect(renderer._casterBounds({ light: sun, camera: sceneCamera }, view), 'no split distances yet').to.equal(world.worldBounds);

        // slice 1 spans z in [-60, -20]: centre (0, 0, -40), radius |(10, 10, -20) - (0, 0, -40)|
        sun._shadowCascadeDistances = [20, 60, 100];
        const radius = Math.hypot(HALF_WIDTH, HALF_WIDTH, 20);
        let box = renderer._casterBounds({ light: sun, camera: sceneCamera }, view);
        expect(box.center.x).to.be.closeTo(0, 1e-6);
        expect(box.center.z).to.be.closeTo(-40, 1e-6);
        expect(box.halfExtents.x, 'a vertical light pads by the radius only').to.be.closeTo(radius, 1e-6);
        expect(box.halfExtents.z).to.be.closeTo(radius, 1e-6);
        expect([box.center.y, box.halfExtents.y], 'the full world height').to.deep.equal([50, 50]);

        // a 45 degree light reaches a further world-height along the ground, clipped to the world
        shadowNode.forward = new Vec3(Math.SQRT1_2, -Math.SQRT1_2, 0);
        box = renderer._casterBounds({ light: sun, camera: sceneCamera }, view);
        expect(box.getMax().x, 'radius + 100 clipped to the world edge').to.be.closeTo(100, 1e-6);
        expect(box.getMin().z).to.be.closeTo(-100, 1e-6);
        expect(box.getMax().z).to.be.closeTo(-40 + radius + 100, 1e-6);
    });

    it('holds a cached shadow map\'s re-render while streamed content keeps changing', function () {
        const sun = makeLight({ faces: 1 });
        sun.shadowUpdateMode = SHADOWUPDATE_NONE;
        director.renderer.culler.cameraDirShadowLights.set(sceneCamera, [sun]);
        world.contentVersion = 0;
        renderer.staticSettleMs = Infinity;
        renderer.staticMaxWaitMs = Infinity;

        renderer.syncLights(comp);
        world.contentVersion++;
        renderer.syncLights(comp);
        expect(sun.shadowUpdateMode, 'content changed but has not settled').to.equal(SHADOWUPDATE_NONE);

        // once it has been still long enough, one re-render is requested
        renderer.staticSettleMs = 0;
        renderer.syncLights(comp);
        expect(sun.shadowUpdateMode).to.equal(SHADOWUPDATE_THISFRAME);

        // and nothing more is requested until the content changes again
        sun.shadowUpdateMode = SHADOWUPDATE_NONE;
        renderer.syncLights(comp);
        expect(sun.shadowUpdateMode).to.equal(SHADOWUPDATE_NONE);

        // content that never settles still re-renders after the maximum wait
        renderer.staticSettleMs = Infinity;
        world.contentVersion++;
        renderer.syncLights(comp);
        expect(sun.shadowUpdateMode).to.equal(SHADOWUPDATE_NONE);
        renderer.staticMaxWaitMs = 0;
        renderer.syncLights(comp);
        expect(sun.shadowUpdateMode).to.equal(SHADOWUPDATE_THISFRAME);
    });

    it('re-renders a cached shadow map for new content only when its last render was incomplete', function () {
        const sun = makeLight({ faces: 1 });
        sun.shadowUpdateMode = SHADOWUPDATE_NONE;
        director.renderer.culler.cameraDirShadowLights.set(sceneCamera, [sun]);
        world.contentVersion = 0;
        renderer.staticSettleMs = 0;
        renderer.staticMaxWaitMs = 0;
        renderer.syncLights(comp);
        sun.shadowUpdateMode = SHADOWUPDATE_NONE;
        const [view] = renderer.entries.get(sun).views;
        const roots = { admitted: [10, 0, 0], rejected: [0, 0, 0], candidates: 1, rejectedInstances: 0, unready: 0, subpixel: 0, occluded: 0, waiting: 0 };
        view.lastDemand = { indices: [10, 0, 0], records: 1, cutQueue: 0 };
        view.lastRoots = roots;
        const changed = () => {
            world.contentVersion++;
            renderer.syncLights(comp);
            const mode = sun.shadowUpdateMode;
            sun.shadowUpdateMode = SHADOWUPDATE_NONE;
            return mode;
        };

        expect(changed(), 'a complete cut renders the same casters again').to.equal(SHADOWUPDATE_NONE);
        roots.waiting = 3;
        expect(changed(), 'groups waited on pages').to.equal(SHADOWUPDATE_THISFRAME);
        roots.waiting = 0;
        roots.unready = 2;
        expect(changed(), 'roots were not resident').to.equal(SHADOWUPDATE_THISFRAME);
        roots.unready = 0;

        // casters dropped for capacity are worth another render only while the buffer can grow
        roots.rejectedInstances = 5;
        view.canGrowIndices = () => true;
        expect(changed()).to.equal(SHADOWUPDATE_THISFRAME);
        view.canGrowIndices = () => false;
        expect(changed(), 'at the ceiling').to.equal(SHADOWUPDATE_NONE);

        roots.rejectedInstances = 0;
        roots.candidates = 0;
        expect(changed(), 'an empty cut, rendered before anything could be admitted').to.equal(SHADOWUPDATE_THISFRAME);
        roots.candidates = 1;

        // the counters of the last render still on their way: decided once they arrive
        roots.waiting = 1;
        view._countersReadBusy = true;
        expect(changed()).to.equal(SHADOWUPDATE_NONE);
        view._countersReadBusy = false;
        renderer.syncLights(comp);
        expect(sun.shadowUpdateMode).to.equal(SHADOWUPDATE_THISFRAME);
    });

    it('serves a light with shadow bounds one camera-independent view fitted to the bounds', function () {
        const sun = makeLight({ faces: 1, renderData: { 0: { shadowCamera: { _node: { forward: new Vec3(0, -1, 0) } } } } });
        sun._shadowBounds = new BoundingBox(new Vec3(10, 5, -20), new Vec3(30, 5, 15));
        director.renderer.culler.cameraDirShadowLights.set(sceneCamera, [sun]);
        renderer.syncLights(comp);

        const entry = renderer.entries.get(sun);
        expect(entry.camera, 'the shadow is shared by all cameras').to.equal(null);
        expect(entry.views).to.have.lengthOf(1);
        expect(entry.views[0].sceneCamera).to.equal(null);

        // the bounds stand in for the camera's slice: padded by their radius, the full world height
        const radius = Math.hypot(30, 5, 15);
        const box = renderer._casterBounds({ light: sun, camera: null }, entry.views[0]);
        expect(box.center.x).to.be.closeTo(10, 1e-6);
        expect(box.center.z).to.be.closeTo(-20, 1e-6);
        expect(box.halfExtents.x).to.be.closeTo(radius, 1e-6);
        expect([box.center.y, box.halfExtents.y]).to.deep.equal([50, 50]);

        // dropping the bounds returns the light to per-camera cascades
        sun._shadowBounds = null;
        sun.numShadowFaces = 3;
        renderer.syncLights(comp);
        expect(renderer.entries.get(sun).camera).to.equal(sceneCamera);
    });

    it('keeps casters whose shadow can reach a receiver in the cascade slice', function () {
        // cascade 0 of the scene camera covers x, y in [-10, 10], z in [-20, -0.5]
        const sun = makeLight({ cascades: [20] });
        sun.cascadeBlend = 0;
        sun._normalOffsetBias = 0;
        const shadowNode = { forward: new Vec3() };
        const view = { face: 0, shadowCamera: { _node: shadowNode } };
        const culler = { casterPlanes: new Float32Array(48), casterPlaneCount: 0 };
        const casts = (center, radius) => {
            renderer._setCasterVolume({ light: sun, camera: sceneCamera }, view, 32, culler);
            const p = culler.casterPlanes;
            for (let i = 0; i < culler.casterPlaneCount; i++) {
                const o = i * 4;
                if (p[o] * center.x + p[o + 1] * center.y + p[o + 2] * center.z + p[o + 3] < -radius) return false;
            }
            return true;
        };

        // the whole meshlet world inside the slice: every slice face clips away against it
        world.worldBounds = new BoundingBox(new Vec3(0, 0, -10), new Vec3(1, 1, 1));
        shadowNode.forward.set(0, -1, 0);
        expect(casts(new Vec3(0, 0, -10), 1), 'world wholly inside the slice').to.equal(true);
        expect(culler.casterPlaneCount, 'still culled by a volume').to.be.greaterThan(1);

        // a caster outside the slice whose shadow falls into it, on ground outside the world
        world.worldBounds = new BoundingBox(new Vec3(30, 5, -10), new Vec3(1, 1, 1));
        shadowNode.forward.set(-1, -0.5, 0).normalize();
        expect(casts(new Vec3(30, 5, -10), 1), 'shadow thrown into the slice').to.equal(true);

        // the same caster lit from the other side throws its shadow away from the slice
        shadowNode.forward.set(1, -0.5, 0).normalize();
        expect(casts(new Vec3(30, 5, -10), 1), 'shadow thrown away from the slice').to.equal(false);
    });

    it('derives the LOD projection from the face\'s shadow-map pixel height', function () {
        const sun = makeLight();
        const spot = makeLight({ type: LIGHTTYPE_SPOT, faces: 1 });
        const view = { renderData: { shadowViewport: { w: 0.5 } } };
        let projection = renderer._projection(sun, view, { renderTarget: { height: 2048 }, orthoHeight: 32 });
        expect(projection, 'ortho: 1024 texels over 64 units').to.deep.equal({ orthoScale: 16, projScale: 1 });
        projection = renderer._projection(sun, { renderData: null }, { renderTarget: null, orthoHeight: 32 });
        expect(projection, 'no target yet: nominal resolution, full viewport').to.deep.equal({ orthoScale: 32, projScale: 1 });
        expect(renderer._projection(sun, view, { renderTarget: null, orthoHeight: 0 })).to.equal(null);
        projection = renderer._projection(spot, view, { renderTarget: { height: 2048 }, fov: 90 });
        expect(projection.orthoScale).to.equal(0);
        expect(projection.projScale, 'perspective: half the pixels over tan(45)').to.be.closeTo(512, 1e-9);
    });

    it('culls each face against the final shadow camera state with the light-space LOD projection', function () {
        const shadowNode = new GraphNode();
        shadowNode.setPosition(3, 40, -7);
        const projectionMatrix = new Mat4().setOrtho(-32, 32, -32, 32, 0.1, 200);
        const shadowCamera = { _node: shadowNode, projectionMatrix, orthoHeight: 32, renderTarget: { height: 2048 } };
        const renderData = { 0: { shadowCamera, shadowViewport: { w: 1 } } };
        const sun = makeLight({ faces: 2, renderData });
        director.renderer.culler.cameraDirShadowLights.set(sceneCamera, [sun]);
        renderer.syncLights(comp);

        const calls = [];
        renderer.entries.get(sun).views.forEach((view) => {
            view.culler.beginFrame = (...args) => calls.push({ view, args });
            view.readbackPool.read = () => new Promise(() => {});
        });
        renderer.cull();

        expect(calls, 'face 1 has no shadow camera yet').to.have.lengthOf(1);
        const { view, args } = calls[0];
        expect(view.face).to.equal(0);
        const culler = view.culler;
        expect(culler.twoPhase).to.equal(false);
        expect(culler.orthoScale).to.equal(2048 / 64);
        expect(culler.cullFlags).to.equal(CULL_FLAG_NO_TEXEL_RATE | CULL_FLAG_SHADOW_VIEW);
        expect(culler.viewDir.equals(shadowNode.forward)).to.equal(true);
        expect(culler.dagPixelThreshold).to.equal(3);
        expect(culler.pressureScale).to.equal(1.25);
        expect(view._countersReadBusy, 'demand readback issued').to.equal(true);

        const [planes, position, projScale, viewProj, hzb] = args;
        expect(planes).to.have.lengthOf(24);
        expect(position.equals(shadowNode.getPosition())).to.equal(true);
        expect(projScale).to.equal(1);
        const expected = new Mat4().mul2(projectionMatrix, shadowNode.getWorldTransform().clone().invert());
        expect(Array.from(viewProj)).to.deep.equal(Array.from(expected.data));
        expect(hzb).to.equal(null);
    });

    it('skips the cull for a cached shadow map and for cascades overridden to not update', function () {
        const shadowCamera = {
            _node: new GraphNode(),
            projectionMatrix: new Mat4().setOrtho(-32, 32, -32, 32, 0.1, 200),
            orthoHeight: 32,
            renderTarget: { height: 2048 }
        };
        const shadowViewport = { w: 1 };
        const renderData = { 0: { shadowCamera, shadowViewport }, 1: { shadowCamera, shadowViewport } };
        const sun = makeLight({ faces: 2, renderData });
        director.renderer.culler.cameraDirShadowLights.set(sceneCamera, [sun]);
        renderer.syncLights(comp);

        const faces = [];
        renderer.entries.get(sun).views.forEach((view) => {
            view.culler.beginFrame = () => faces.push(view.face);
            view.readbackPool.read = () => new Promise(() => {});
        });
        const cullWith = (mode, overrides = null) => {
            sun.shadowUpdateMode = mode;
            sun.shadowUpdateOverrides = overrides;
            faces.length = 0;
            renderer.cull();
            return faces.slice();
        };

        expect(cullWith(SHADOWUPDATE_REALTIME)).to.deep.equal([0, 1]);
        expect(cullWith(SHADOWUPDATE_THISFRAME), 'a requested update still culls').to.deep.equal([0, 1]);
        expect(cullWith(SHADOWUPDATE_NONE), 'cached map').to.deep.equal([]);
        expect(cullWith(SHADOWUPDATE_REALTIME, [SHADOWUPDATE_REALTIME, SHADOWUPDATE_NONE])).to.deep.equal([0]);
    });

    it('releases every view, caster and shared buffer on reset', function () {
        const sun = makeLight({ faces: 2 });
        director.renderer.culler.cameraDirShadowLights.set(sceneCamera, [sun]);
        renderer.syncLights(comp);
        expect(layer.casters).to.have.lengthOf(2 * MESHLET_BUCKET_COUNT);
        renderer.reset();
        expect(renderer.entries.size).to.equal(0);
        expect(layer.casters).to.have.lengthOf(0);
        expect(renderer._claimBits).to.equal(null);
        expect(renderer._workItems).to.equal(null);
    });
});
