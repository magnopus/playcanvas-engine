import { Debug } from '../../core/debug.js';
import { MESHLET_BUCKET_COUNT } from './constants.js';
import { FramePassMeshletCompute } from './frame-pass-meshlet-compute.js';
import { MeshletCullShaders } from './meshlet-cull-shaders.js';
import { MeshletDirector } from './meshlet-director.js';
import { MeshletHzb } from './meshlet-hzb.js';
import { RenderPassMeshletDraw } from './render-pass-meshlet-draw.js';
import { MAX_CONCURRENT_FETCHES, MeshletFetchScheduler } from './streaming/meshlet-fetch-scheduler.js';

/**
 * @import { ByteCache } from '../../platform/net/byte-cache.js'
 * @import { CameraComponent } from '../../framework/components/camera/component.js'
 * @import { Color } from '../../core/math/color.js'
 * @import { ForwardRenderer } from '../renderer/forward-renderer.js'
 * @import { FrameGraph } from '../frame-graph.js'
 * @import { FramePass } from '../../platform/graphics/frame-pass.js'
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { Layer } from '../layer.js'
 * @import { LayerComposition } from '../composition/layer-composition.js'
 * @import { MeshInstance } from '../mesh-instance.js'
 * @import { MeshletTranscodeFn } from './textures/meshlet-ktx2.js'
 * @import { MeshletView } from './meshlet-view.js'
 * @import { RenderTarget } from '../../platform/graphics/render-target.js'
 */

/**
 * Per-camera state the group shares across its worlds: one HZB, and one pass of each kind that
 * covers every world's view of the camera.
 *
 * @ignore
 */
class MeshletGroupCamera {
    /** @type {MeshletHzb} */
    hzb;

    /** @type {RenderPassMeshletDraw} */
    drawPass1;

    /** @type {RenderPassMeshletDraw} */
    drawPass2;

    /** @type {FramePassMeshletCompute} */
    hzbMipsPass;

    /** @type {FramePassMeshletCompute} */
    cullPhase2Pass;

    /** @type {MeshletView[]} - this frame's views of the camera, one per world that has content. */
    views = [];

    /** @type {boolean} - any view culls two-phase this frame (they all decide alike). */
    useOcclusion = false;

    /** @type {boolean} - passes already emitted this frame (render-action path). */
    passesAdded = false;

    /**
     * @param {GraphicsDevice} device - The graphics device.
     * @param {ForwardRenderer|null} renderer - The forward renderer.
     */
    constructor(device, renderer) {
        this.hzb = new MeshletHzb(device);
        this.drawPass1 = new RenderPassMeshletDraw(device, renderer, 'MeshletDrawPhase1');
        this.drawPass2 = new RenderPassMeshletDraw(device, renderer, 'MeshletDrawPhase2');
        this.hzbMipsPass = new FramePassMeshletCompute(device, 'MeshletHzbMips', () => this.hzb.buildMips());
        // every world's phase 2 reads the same pyramid, built once from all of their phase-1 draws
        this.cullPhase2Pass = new FramePassMeshletCompute(device, 'MeshletCullPhase2', () => {
            for (const view of this.views) {
                if (view.useOcclusion && view.culler) view.culler.dispatchPhase2();
            }
        });
    }

    destroy() {
        this.hzb.destroy();
        this.views = [];
    }

    /**
     * Collects the views of every world for this frame and points the draw passes at their
     * mesh instances: phase-1 buckets in one pass, phase-2 buckets in the other.
     *
     * @param {MeshletDirector[]} directors - The group's directors.
     * @param {CameraComponent} cameraComponent - The camera.
     */
    gather(directors, cameraComponent) {
        const views = this.views;
        views.length = 0;
        let useOcclusion = false;
        for (const director of directors) {
            const view = director.views.get(cameraComponent);
            if (!view) continue;
            views.push(view);
            useOcclusion ||= view.useOcclusion;
        }
        this.useOcclusion = useOcclusion;

        const mis1 = this.drawPass1.meshInstances;
        const mis2 = this.drawPass2.meshInstances;
        mis1.length = 0;
        mis2.length = 0;
        for (const view of views) {
            const mis = view.meshInstances;
            for (let i = 0; i < MESHLET_BUCKET_COUNT; i++) mis1.push(mis[i]);
            if (view.useOcclusion) {
                for (let i = MESHLET_BUCKET_COUNT; i < MESHLET_BUCKET_COUNT * 2; i++) mis2.push(mis[i]);
            }
        }
    }
}

/**
 * Drives several meshlet worlds as one pipeline. A single {@link MeshletWorld} keeps every
 * meshlet record, object row and per-pair bit of its scene resident in one buffer each, so a
 * large enough scene outgrows what the browser can allocate or WebGPU can bind. The group splits
 * the scene across worlds, one {@link MeshletDirector} each, and shares everything that does not
 * scale with the content:
 *
 * - one fetch scheduler, so the concurrency cap and request priorities span every world
 * - one set of compiled cull shaders
 * - per camera, one HZB and one draw pass per phase. Every world's phase-1 cull is encoded back
 *   to back ahead of the frame graph (each tests against last frame's pyramid); a single phase-1
 *   pass draws them all; the pyramid is built once from that depth plus the scene's opaque
 *   layers; then every world's phase-2 cull runs against it before one phase-2 draw.
 *
 * Each world still has its own page pool, residency, LOD budget controller, index budget and
 * shadow views. The renderer talks to the group (it registers itself as the renderer's
 * `meshletDirector`); the component system assigns resources to the worlds.
 *
 * @ignore
 */
class MeshletDirectorGroup {
    /** @type {GraphicsDevice} */
    device;

    /** @type {ForwardRenderer|null} */
    renderer = null;

    /** @type {MeshletDirector[]} - one per world; never empty. */
    directors = [];

    /** @type {MeshletFetchScheduler} */
    fetchScheduler = new MeshletFetchScheduler();

    /** @type {MeshletCullShaders} */
    cullShaders;

    /**
     * Cap on requests in flight while any world is fast-loading (see
     * {@link MeshletDirector#fastLoad}).
     *
     * @type {number}
     */
    fastLoadConcurrentFetches = 64;

    /** @type {Map<CameraComponent, MeshletGroupCamera>} @private */
    _cameras = new Map();

    /** @type {number} @private */
    _maxConcurrentFetches = MAX_CONCURRENT_FETCHES;

    /**
     * Settings applied to every director, including ones created later. Mirrors the director
     * properties of the same names.
     *
     * @private
     */
    _settings = {
        dagPixelThreshold: 1,
        occlusionEnabled: true,
        cameras: /** @type {CameraComponent[]} */ ([]),
        shadowsEnabled: false,
        shadowThresholdScale: 4,
        rootMinPixels: 3,
        shadowRootMinPixels: 2,
        shadowMatchCameraLod: false,
        cssPixels: true,
        fastLoad: true,
        forceSubmitBoundaries: false,
        transcode: /** @type {MeshletTranscodeFn|null} */ (null)
    };

    /** @type {{ layer: Layer|null, color: Color|null, hoverColor: Color|null }} @private */
    _outline = { layer: null, color: null, hoverColor: null };

    /**
     * @param {GraphicsDevice} device - The graphics device.
     */
    constructor(device) {
        this.device = device;
        this.cullShaders = new MeshletCullShaders(device);
        this.resize(1);
    }

    destroy() {
        if (this.renderer && this.renderer.meshletDirector === this) {
            this.renderer.meshletDirector = null;
        }
        this.directors.forEach(d => d.destroy());
        this.directors = [];
        this._cameras.forEach(c => c.destroy());
        this._cameras.clear();
        this.fetchScheduler.clear();
        this.cullShaders.destroy();
    }

    /**
     * Sets the number of worlds. New directors take the group's settings; surplus ones are
     * destroyed with their worlds.
     *
     * @param {number} count - World count, at least 1.
     */
    resize(count) {
        count = Math.max(count, 1);
        while (this.directors.length > count) {
            this.directors.pop().destroy();
        }
        while (this.directors.length < count) {
            const director = new MeshletDirector(this.device, {
                fetchScheduler: this.fetchScheduler,
                cullShaders: this.cullShaders
            });
            director.group = this;
            director.renderer = this.renderer;
            this._applySettings(director);
            if (this._outline.layer) {
                director.setOutlineLayer(this._outline.layer, this._outline.color, this._outline.hoverColor);
            }
            this.directors.push(director);
        }
    }

    /**
     * @param {MeshletDirector} director - The director to configure.
     * @private
     */
    _applySettings(director) {
        const s = this._settings;
        director.dagPixelThreshold = s.dagPixelThreshold;
        director.occlusionEnabled = s.occlusionEnabled;
        director.cameras = s.cameras;
        director.shadowsEnabled = s.shadowsEnabled;
        director.shadowThresholdScale = s.shadowThresholdScale;
        director.rootMinPixels = s.rootMinPixels;
        director.shadowRootMinPixels = s.shadowRootMinPixels;
        director.shadowMatchCameraLod = s.shadowMatchCameraLod;
        director.cssPixels = s.cssPixels;
        director.fastLoad = s.fastLoad;
        director.forceSubmitBoundaries = s.forceSubmitBoundaries;
        director.transcode = s.transcode;
    }

    /**
     * @param {string} name - Setting name.
     * @param {*} value - New value.
     * @private
     */
    _set(name, value) {
        this._settings[name] = value;
        for (const director of this.directors) director[name] = value;
    }

    /** @type {number} */
    set dagPixelThreshold(value) {
        this._set('dagPixelThreshold', value);
    }

    get dagPixelThreshold() {
        return this._settings.dagPixelThreshold;
    }

    /** @type {boolean} */
    set occlusionEnabled(value) {
        this._set('occlusionEnabled', !!value);
    }

    get occlusionEnabled() {
        return this._settings.occlusionEnabled;
    }

    /** @type {CameraComponent[]} */
    set cameras(value) {
        this._set('cameras', value ?? []);
    }

    get cameras() {
        return this._settings.cameras;
    }

    /** @type {CameraComponent|null} */
    set cameraComponent(camera) {
        this.cameras = camera ? [camera] : [];
    }

    get cameraComponent() {
        return this._settings.cameras[0] ?? null;
    }

    /** @type {boolean} */
    set shadowsEnabled(value) {
        this._set('shadowsEnabled', !!value);
    }

    get shadowsEnabled() {
        return this._settings.shadowsEnabled;
    }

    /** @type {number} */
    set shadowThresholdScale(value) {
        this._set('shadowThresholdScale', value);
    }

    get shadowThresholdScale() {
        return this._settings.shadowThresholdScale;
    }

    /** @type {number} */
    set rootMinPixels(value) {
        this._set('rootMinPixels', value);
    }

    get rootMinPixels() {
        return this._settings.rootMinPixels;
    }

    /** @type {boolean} */
    set cssPixels(value) {
        this._set('cssPixels', !!value);
    }

    get cssPixels() {
        return this._settings.cssPixels;
    }

    /** @type {number} */
    set shadowRootMinPixels(value) {
        this._set('shadowRootMinPixels', value);
    }

    get shadowRootMinPixels() {
        return this._settings.shadowRootMinPixels;
    }

    /** @type {boolean} */
    set shadowMatchCameraLod(value) {
        this._set('shadowMatchCameraLod', value);
    }

    get shadowMatchCameraLod() {
        return this._settings.shadowMatchCameraLod;
    }

    /** @type {boolean} */
    set fastLoad(value) {
        this._set('fastLoad', !!value);
    }

    get fastLoad() {
        return this._settings.fastLoad;
    }

    /** @type {boolean} */
    set forceSubmitBoundaries(value) {
        this._set('forceSubmitBoundaries', !!value);
    }

    get forceSubmitBoundaries() {
        return this._settings.forceSubmitBoundaries;
    }

    /** @type {MeshletTranscodeFn|null} */
    set transcode(value) {
        this._set('transcode', value);
    }

    get transcode() {
        return this._settings.transcode;
    }

    /**
     * Cap on sidecar requests in flight across every world. See
     * {@link MeshletDirector#maxConcurrentFetches}.
     *
     * @type {number}
     */
    set maxConcurrentFetches(value) {
        this._maxConcurrentFetches = Math.max(1, value | 0);
        this._applyFetchConcurrency();
    }

    get maxConcurrentFetches() {
        return this._maxConcurrentFetches;
    }

    /** @type {ByteCache|null} */
    set byteCache(value) {
        this.fetchScheduler.byteCache = value;
    }

    get byteCache() {
        return this.fetchScheduler.byteCache;
    }

    /**
     * The shared scheduler loads fast while any world is fast-loading.
     *
     * @ignore
     */
    _applyFetchConcurrency() {
        const fast = this.directors.some(d => !!d.residency?.fastLoad);
        this.fetchScheduler.maxConcurrent = fast ?
            Math.max(this._maxConcurrentFetches, this.fastLoadConcurrentFetches) :
            this._maxConcurrentFetches;
    }

    /**
     * Whether every world's content is playable (see {@link MeshletDirector#playable}). A world
     * with nothing built has nothing to wait for.
     *
     * @type {boolean}
     */
    get playable() {
        return this.directors.every(d => !d.world.finalized || d.playable);
    }

    /**
     * The camera state shared by every world's view of a camera, created on first use.
     *
     * @param {CameraComponent} cameraComponent - The camera.
     * @returns {MeshletGroupCamera} The shared state.
     * @private
     */
    _cameraState(cameraComponent) {
        let state = this._cameras.get(cameraComponent);
        if (!state) {
            state = new MeshletGroupCamera(this.device, this.renderer);
            this._cameras.set(cameraComponent, state);
        }
        return state;
    }

    /**
     * The HZB every world's view of a camera culls against. Directors call this while setting
     * up their views each frame.
     *
     * @param {CameraComponent} cameraComponent - The camera.
     * @returns {MeshletHzb} The shared pyramid.
     * @ignore
     */
    hzbFor(cameraComponent) {
        return this._cameraState(cameraComponent).hzb;
    }

    /**
     * Registers this group with the forward renderer, which ticks it each frame and inserts its
     * passes into the frame graph.
     *
     * @param {ForwardRenderer} renderer - The renderer (app.renderer).
     */
    bindRenderer(renderer) {
        this.renderer = renderer;
        renderer.meshletDirector = this;
        for (const director of this.directors) director.renderer = renderer;
    }

    /**
     * Gives every world a consecutive range of the group's instance index space - the index a
     * pick returns and {@link setOutlined} takes. Call after the worlds rebuild.
     */
    assignInstanceOffsets() {
        let offset = 0;
        for (const director of this.directors) {
            const world = director.world;
            if (world.instanceOffset !== offset) {
                world.instanceOffset = offset;
                world._pickRecords = null;
            }
            offset += world.finalized ? world.instanceCount : 0;
        }
    }

    /**
     * Per-frame tick, called by the forward renderer before the frame graph is built. Each world
     * runs its streaming update and encodes its phase-1 cull, one after another; the shared
     * per-camera passes are then pointed at this frame's views.
     *
     * @param {LayerComposition} [comp] - The layer composition.
     */
    update(comp) {
        for (const director of this.directors) {
            director.update(comp);
            // with occlusion off nothing asks for the camera's HZB, so the state is made here
            for (const cameraComponent of director.views.keys()) this._cameraState(cameraComponent);
        }

        this._cameras.forEach((state, cameraComponent) => {
            state.passesAdded = false;
            state.gather(this.directors, cameraComponent);
            // no world has a view of this camera any more
            if (!state.views.length) {
                state.destroy();
                this._cameras.delete(cameraComponent);
            }
        });
    }

    /**
     * See {@link MeshletDirector#updateShadowLights}.
     *
     * @param {LayerComposition} comp - The layer composition.
     */
    updateShadowLights(comp) {
        for (const director of this.directors) director.updateShadowLights(comp);
    }

    /**
     * See {@link MeshletDirector#updateShadows}.
     */
    updateShadows() {
        for (const director of this.directors) director.updateShadows();
    }

    /** @type {MeshInstance[]} - every world's mesh instances (shader-update sweep). */
    get allMeshInstances() {
        const mis = [];
        for (const director of this.directors) mis.push(...director.allMeshInstances);
        return mis;
    }

    /**
     * See {@link MeshletDirector#isLightLayer}.
     *
     * @param {Layer} layer - The layer to test.
     * @returns {boolean} True for the meshlet light layer.
     */
    isLightLayer(layer) {
        return this.directors.some(d => d.isLightLayer(layer));
    }

    /**
     * See {@link MeshletDirector#preparePicking}. Pick ids are reserved globally, so the worlds'
     * mappings never collide.
     *
     * @param {CameraComponent} cameraComponent - The camera being picked through.
     * @param {Map<number, object>} mapping - Picker id -> object map.
     * @returns {MeshInstance[]} Instances to draw into the pick buffer.
     */
    preparePicking(cameraComponent, mapping) {
        const mis = [];
        for (const director of this.directors) mis.push(...director.preparePicking(cameraComponent, mapping));
        return mis;
    }

    /**
     * See {@link MeshletDirector#wantsSceneDepth}.
     *
     * @param {CameraComponent} cameraComponent - The camera.
     * @returns {boolean} True when the frame should render the scene depth.
     */
    wantsSceneDepth(cameraComponent) {
        return this.directors.some(d => d.wantsSceneDepth(cameraComponent));
    }

    /**
     * See {@link MeshletDirector#setOutlineLayer}.
     *
     * @param {Layer|null} layer - The outline renderer's layer, or null to unregister.
     * @param {Color|null} [color] - Selection outline colour.
     * @param {Color|null} [hoverColor] - Hover outline colour.
     */
    setOutlineLayer(layer, color = null, hoverColor = null) {
        this._outline = { layer: layer ?? null, color, hoverColor };
        for (const director of this.directors) director.setOutlineLayer(layer, color, hoverColor);
    }

    /**
     * Selects instances for outlining, by group-wide instance index (see
     * {@link assignInstanceOffsets}); each world takes the indices in its range.
     *
     * @param {number[]|Set<number>} instanceIndices - Instance indices.
     * @param {boolean} [outlined] - True to add, false to remove. Defaults to true.
     */
    setOutlined(instanceIndices, outlined = true) {
        for (const director of this.directors) director.setOutlined(instanceIndices, outlined);
    }

    /**
     * See {@link MeshletDirector#setHovered}.
     *
     * @param {number[]|Set<number>} instanceIndices - Instance indices.
     * @param {boolean} [hovered] - True to add, false to remove. Defaults to true.
     */
    setHovered(instanceIndices, hovered = true) {
        for (const director of this.directors) director.setHovered(instanceIndices, hovered);
    }

    /**
     * See {@link MeshletDirector#clearOutlines}.
     *
     * @param {boolean} [hover] - Clear the hover state rather than the selection.
     */
    clearOutlines(hover = false) {
        for (const director of this.directors) director.clearOutlines(hover);
    }

    /**
     * The CameraFrame variant of {@link buildCameraPasses} - see
     * {@link MeshletDirector#buildCameraFramePasses}. The shared passes cover every world.
     *
     * @param {CameraComponent} cameraComponent - The camera being set up.
     * @param {RenderTarget} renderTarget - The CameraFrame's scene render target.
     * @param {boolean} clears - True when this camera's first pass should clear.
     * @returns {{ before: FramePass[], middle: FramePass[] }|null} Passes to splice, or null
     * when no world drives this camera.
     */
    buildCameraFramePasses(cameraComponent, renderTarget, clears) {
        const state = this._cameras.get(cameraComponent);
        if (!state || !state.views.length) {
            return null;
        }
        const lightLayer = this._lightLayer();
        const camera = cameraComponent.camera;
        state.drawPass1.setup(cameraComponent, renderTarget, lightLayer, {
            clearColor: clears && camera.clearColorBuffer,
            clearDepth: clears && camera.clearDepthBuffer,
            clearStencil: clears && camera.clearStencilBuffer
        });

        const middle = [];
        if (state.useOcclusion) {
            state.drawPass2.setup(cameraComponent, renderTarget, lightLayer);
            middle.push(state.hzb.mip0Pass, state.hzbMipsPass, state.cullPhase2Pass, state.drawPass2);
        }
        state.passesAdded = true;
        return { before: [state.drawPass1], middle };
    }

    /**
     * See {@link MeshletDirector#buildCameraPasses}. Emits one phase-1 draw for every world, the
     * opaque scene passes, then - with occlusion - one HZB build, every world's phase-2 cull and
     * one phase-2 draw, then the transparent scene passes.
     *
     * @param {FrameGraph} frameGraph - The frame graph being built.
     * @param {LayerComposition} layerComposition - The layer composition.
     * @param {RenderTarget|null} renderTarget - The block's render target.
     * @param {number} startIndex - First render action index of the block.
     * @param {number} endIndex - Last render action index of the block.
     * @param {CameraComponent} cameraComponent - The camera of the block.
     * @returns {boolean} True when the group emitted the block's passes.
     */
    buildCameraPasses(frameGraph, layerComposition, renderTarget, startIndex, endIndex, cameraComponent) {
        const state = this._cameras.get(cameraComponent);
        if (!state || !state.views.length || state.passesAdded) {
            return false;
        }
        if (!cameraComponent.camera.fullSizeClearRect) {
            Debug.warnOnce('MeshletDirectorGroup: camera does not clear the full viewport; meshlet draws fall back to post-scene ordering (incorrect compositing with scene geometry).');
            return false;
        }
        state.passesAdded = true;

        const renderActions = layerComposition._renderActions;
        const renderer = this.renderer;
        const lightLayer = this._lightLayer();

        const firstAction = renderActions[startIndex];
        state.drawPass1.setup(cameraComponent, renderTarget, lightLayer, {
            clearColor: firstAction.clearColor,
            clearDepth: firstAction.clearDepth,
            clearStencil: firstAction.clearStencil
        });
        frameGraph.addRenderPass(state.drawPass1);

        let splitIndex = endIndex + 1;
        for (let j = startIndex; j <= endIndex; j++) {
            if (renderActions[j].transparent) {
                splitIndex = j;
                break;
            }
        }
        if (splitIndex > startIndex) {
            renderer.addMainRenderPass(frameGraph, layerComposition, renderTarget, startIndex, splitIndex - 1, true);
        }
        if (state.useOcclusion) {
            frameGraph.addRenderPass(state.hzb.mip0Pass);
            frameGraph.addRenderPass(state.hzbMipsPass);
            frameGraph.addRenderPass(state.cullPhase2Pass);
            state.drawPass2.setup(cameraComponent, renderTarget, lightLayer);
            frameGraph.addRenderPass(state.drawPass2);
        }
        if (splitIndex <= endIndex) {
            renderer.addMainRenderPass(frameGraph, layerComposition, renderTarget, splitIndex, endIndex, true);
        }
        return true;
    }

    /**
     * @returns {Layer|null} The layer whose lights shade the meshlet draws.
     * @private
     */
    _lightLayer() {
        for (const director of this.directors) {
            if (director._lightLayer) return director._lightLayer;
        }
        return null;
    }
}

export { MeshletDirectorGroup };
