import { Debug } from '../../../core/debug.js';
import { basisTranscode } from '../../handlers/basis.js';
import { MeshletDirector } from '../../../scene/meshlet/meshlet-director.js';
import { ComponentSystem } from '../system.js';
import { MeshletComponent } from './component.js';

/**
 * @import { AppBase } from '../../app-base.js'
 * @import { CameraComponent } from '../camera/component.js'
 */

const _properties = ['asset', 'resource', 'baseUrl'];

/**
 * Allows an Entity to render streamed meshlet assets through the GPU-driven meshlet pipeline.
 * WebGPU only - on other devices meshlet components are inert.
 *
 * The system owns the shared `MeshletDirector` and its pipeline-wide settings:
 *
 * ```javascript
 * app.systems.meshlet.poolBytes = 256 * 1024 * 1024; // geometry page pool budget
 * app.systems.meshlet.dagPixelThreshold = 1;         // LOD screen-space error target
 * app.systems.meshlet.occlusion = true;              // two-phase HZB occlusion culling
 * app.systems.meshlet.shadows = true;                // meshlets cast shadows
 * ```
 *
 * Adding or removing meshlet components (or changing their assets) rebuilds the meshlet world
 * at the start of the next rendered frame - and the rebuild RETAINS streaming state for
 * resources that persist across it, so unchanged assets do not re-download. Components sharing
 * one asset are deduplicated into a single set of streamed pages. Enabling/disabling a
 * component (or its entity) and entity transform changes are cheap per-frame updates with no
 * rebuild at all.
 *
 * @category Graphics
 */
class MeshletComponentSystem extends ComponentSystem {
    /**
     * The shared meshlet director, or null on non-WebGPU devices.
     *
     * @type {MeshletDirector|null}
     */
    director = null;

    /** @private */
    _dirty = false;

    /**
     * Components contributing to the current world, in placement order.
     *
     * @type {MeshletComponent[]}
     * @private
     */
    _activeComponents = [];

    /** @private */
    _poolBytes = 256 * 1024 * 1024;

    /** @private */
    _maxDrawIndices = 0;

    /**
     * Create a new MeshletComponentSystem.
     *
     * @param {AppBase} app - The Application.
     * @ignore
     */
    constructor(app) {
        super(app);

        this.id = 'meshlet';
        this.ComponentType = MeshletComponent;

        if (app.graphicsDevice.isWebGPU) {
            this.director = new MeshletDirector(app.graphicsDevice);
            this.director.world.poolBytes = this._poolBytes;
            this.director.world.initialIndices = 2 * 1024 * 1024;
            this.director.world.maxIndices = this._maxDrawIndices;
            this.director.bindRenderer(app.renderer);
            // streamed textures are KTX2; the scene layer cannot import the Basis handler, so
            // the transcoder is injected here (basisInitialize() decides the worker options)
            this.director.transcode = basisTranscode;
        } else {
            Debug.warnOnce('MeshletComponentSystem: meshlet rendering requires WebGPU; meshlet components will not render.');
        }

        this.on('beforeremove', this.onBeforeRemove, this);

        // rebuild the world and sync entity transforms after the app update, before rendering
        this.app.on('framerender', this._onFrameRender, this);
    }

    /**
     * Sets the geometry page pool byte budget. Under pool pressure the LOD cut coarsens until
     * its working set fits, so a small budget trades quality for memory instead of failing.
     * Takes effect on the next world rebuild. Defaults to 256 MiB.
     *
     * @type {number}
     */
    set poolBytes(value) {
        this._poolBytes = value;
        if (this.director && this.director.world.poolBytes !== value) {
            this.director.world.poolBytes = value;
            this._markDirty();
        }
    }

    get poolBytes() {
        return this._poolBytes;
    }

    /**
     * Sets the fine-texture slot-pool byte budget. Under pressure the texture mip bias rises,
     * pushing more textures onto their always-resident tails, so a small budget trades
     * sharpness for memory instead of failing. Takes effect on the next world rebuild.
     * Defaults to 96 MiB.
     *
     * @type {number}
     */
    set texturePoolBytes(value) {
        if (this.director && this.director.world.texturePoolBytes !== value) {
            this.director.world.texturePoolBytes = value;
            this._markDirty();
        }
    }

    get texturePoolBytes() {
        return this.director?.world.texturePoolBytes ?? 0;
    }

    /**
     * Sets an optional hard ceiling on the GPU-written draw index buffer, in indices (0 =
     * unbounded, the default). The buffer starts small and grows to the observed per-frame
     * demand, which the LOD cut bounds by screen area; set a ceiling to hard-cap that memory -
     * the cull shader clamps safely on overflow. Takes effect on the next world rebuild.
     *
     * @type {number}
     */
    set maxDrawIndices(value) {
        this._maxDrawIndices = value;
        if (this.director && this.director.world.maxIndices !== value) {
            this.director.world.maxIndices = value;
            this._markDirty();
        }
    }

    get maxDrawIndices() {
        return this._maxDrawIndices;
    }

    /**
     * Sets the screen-space error threshold in pixels for the DAG LOD cut. Defaults to 1.
     *
     * @type {number}
     */
    set dagPixelThreshold(value) {
        if (this.director) this.director.dagPixelThreshold = value;
    }

    get dagPixelThreshold() {
        return this.director?.dagPixelThreshold ?? 1;
    }

    /**
     * Sets whether two-phase HZB occlusion culling is enabled. Requires the camera to render
     * into a render target with a depth texture. Defaults to false.
     *
     * @type {boolean}
     */
    set occlusion(value) {
        if (this.director) this.director.occlusionEnabled = value;
    }

    get occlusion() {
        return this.director?.occlusionEnabled ?? false;
    }

    /**
     * Sets the camera meshlets are culled and drawn for. When null (the default), the first
     * camera of the layer composition is used.
     *
     * @type {CameraComponent|null}
     */
    set camera(value) {
        if (this.director) this.director.cameraComponent = value;
    }

    get camera() {
        return this.director?.cameraComponent ?? null;
    }

    /**
     * Sets the cameras meshlets are culled and drawn for - one independent LOD cut and
     * occlusion state per camera, streaming against the union of their demand. Empty (the
     * default) means the first camera of the layer composition. Capped at the director's
     * maxViews (2 by default).
     *
     * @type {CameraComponent[]}
     */
    set cameras(value) {
        if (this.director) this.director.cameras = value ?? [];
    }

    get cameras() {
        return this.director?.cameras ?? [];
    }

    /**
     * Whether meshlet geometry casts shadows - directional cascades and local (spot / omni)
     * shadow faces. Off by default: every shadow face costs one extra cull pass per frame.
     *
     * @type {boolean}
     */
    set shadows(value) {
        if (this.director) this.director.shadowsEnabled = !!value;
    }

    get shadows() {
        return this.director?.shadowsEnabled ?? false;
    }

    /**
     * Multiplier on {@link dagPixelThreshold} for shadow views. Shadow-map texels are coarser
     * than screen pixels, so casters can use a coarser LOD cut than the camera. Defaults to 4.
     *
     * @type {number}
     */
    set shadowThresholdScale(value) {
        if (this.director) this.director.shadowThresholdScale = value;
    }

    get shadowThresholdScale() {
        return this.director?.shadowThresholdScale ?? 4;
    }

    initializeComponentData(component, data, properties) {
        for (let i = 0; i < _properties.length; i++) {
            if (data.hasOwnProperty(_properties[i])) {
                component[_properties[i]] = data[_properties[i]];
            }
        }

        super.initializeComponentData(component, data);
    }

    cloneComponent(entity, clone) {
        const meshletComponent = entity.meshlet;
        return this.addComponent(clone, {
            enabled: meshletComponent.enabled,
            asset: meshletComponent.asset,
            resource: meshletComponent.resource,
            baseUrl: meshletComponent.baseUrl
        });
    }

    onBeforeRemove(entity, component) {
        component.onBeforeRemove();
    }

    /** @ignore */
    _markDirty() {
        this._dirty = true;
    }

    /** @private */
    _onFrameRender() {
        if (!this.director) return;
        if (this._dirty) {
            this._rebuild();
        }
        this._syncTransforms();
    }

    /** @private */
    _rebuild() {
        this._dirty = false;

        // group components by (resource, baseUrl): each unique resource is added ONCE, with a
        // merged instance list - N components share one set of pages, records and textures.
        // Disabled components are included (hidden via the per-instance hide bit) so a later
        // enable is a buffer write, not a rebuild.
        const groups = [];
        for (const guid in this.store) {
            const component = this.store[guid].entity.meshlet;
            if (!component?._effectiveResource) continue;
            if (!component._effectiveBaseUrl) {
                Debug.warnOnce('MeshletComponent: no stream base URL - assign an asset with a file URL, or set baseUrl with the resource.');
                continue;
            }
            const resource = component._effectiveResource;
            const baseUrl = component._effectiveBaseUrl;
            let group = groups.find(g => g.resource === resource && g.baseUrl === baseUrl);
            if (!group) {
                group = { resource, baseUrl, components: [] };
                groups.push(group);
            }
            group.components.push(component);
        }

        const components = [];
        this.director.rebuild((world) => {
            for (let gi = 0; gi < groups.length; gi++) {
                const { resource, baseUrl, components: members } = groups[gi];
                const per = resource.instances.length;

                // one placement per group; each member owns a sub-range of its instances
                let merged = null;
                if (members.length > 1) {
                    merged = [];
                    for (let mi = 0; mi < members.length; mi++) merged.push(...resource.instances);
                }
                for (let mi = 0; mi < members.length; mi++) {
                    const component = members[mi];
                    component._placementIndex = gi;
                    component._subBase = mi * per;
                    component._subCount = per;
                    component._transformDirty = true;
                    component._hiddenApplied = null;
                    components.push(component);
                }
                world.addStreamedResource(resource, null, baseUrl, merged);
            }
        });
        this._activeComponents = components;
    }

    /** @private */
    _syncTransforms() {
        const world = this.director.world;
        if (!world.finalized) return;
        const components = this._activeComponents;
        for (let i = 0; i < components.length; i++) {
            const component = components[i];
            const wt = component.entity.getWorldTransform();
            if (component._transformDirty || !component._lastTransform.equals(wt)) {
                component._transformDirty = false;
                component._lastTransform.copy(wt);
                world.setPlacementTransform(component._placementIndex, wt, component._subBase, component._subCount);
            }
            const hidden = !(component.enabled && component.entity.enabled);
            if (component._hiddenApplied !== hidden) {
                component._hiddenApplied = hidden;
                world.setPlacementHidden(component._placementIndex, hidden, component._subBase, component._subCount);
            }
        }
    }

    destroy() {
        super.destroy();
        this.app.off('framerender', this._onFrameRender, this);
        this.director?.destroy();
        this.director = null;
    }
}

export { MeshletComponentSystem };
