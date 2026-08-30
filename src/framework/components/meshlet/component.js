import { Mat4 } from '../../../core/math/mat4.js';
import { AssetReference } from '../../asset/asset-reference.js';
import { Component } from '../component.js';

/**
 * @import { Asset } from '../../asset/asset.js'
 * @import { Entity } from '../../entity.js'
 * @import { MeshletComponentSystem } from './system.js'
 * @import { MeshletResource } from '../../../scene/meshlet/meshlet-resource.js'
 */

/**
 * The MeshletComponent renders a streamed meshlet asset (`MAG_meshlets_gpu` /
 * `MAG_meshlets_stream` v2, baked by gltf-tools) through the GPU-driven meshlet pipeline:
 * per-cluster DAG LOD, on-demand geometry page streaming, optional two-phase HZB occlusion
 * culling. WebGPU only.
 *
 * Assign a `container` asset whose GLB carries the meshlet extension; geometry pages stream
 * over HTTP Range requests relative to the asset's URL. The entity's world transform applies
 * to the asset's placements and may change dynamically.
 *
 * ```javascript
 * const entity = new Entity();
 * entity.addComponent('meshlet', { asset: containerAsset });
 * ```
 *
 * Pipeline-wide settings (page pool budget, LOD threshold, occlusion) live on the system:
 * {@link MeshletComponentSystem}, accessible as `app.systems.meshlet`.
 *
 * @hideconstructor
 * @category Graphics
 */
class MeshletComponent extends Component {
    /**
     * @type {AssetReference}
     * @private
     */
    _assetReference;

    /**
     * Direct resource reference (bypasses the asset system).
     *
     * @type {MeshletResource|null}
     * @private
     */
    _resource = null;

    /**
     * Base URL for page streaming when using a direct resource.
     *
     * @type {string|null}
     * @private
     */
    _baseUrl = null;

    /**
     * ObjectData placement index in the current world, managed by the system. Components
     * sharing a resource share a placement; each owns a sub-range of its instances.
     *
     * @type {number}
     * @ignore
     */
    _placementIndex = -1;

    /** @ignore */
    _subBase = 0;

    /** @ignore */
    _subCount = 0;

    /** @ignore */
    _transformDirty = true;

    /**
     * Hidden state currently baked into the world's objectData rows (null = unknown).
     *
     * @type {boolean|null}
     * @ignore
     */
    _hiddenApplied = null;

    /**
     * The entity world transform baked into the current world's objectData rows.
     *
     * @type {Mat4}
     * @ignore
     */
    _lastTransform = new Mat4();

    /**
     * Create a new MeshletComponent.
     *
     * @param {MeshletComponentSystem} system - The ComponentSystem that created this Component.
     * @param {Entity} entity - The Entity that this Component is attached to.
     */
    constructor(system, entity) {
        super(system, entity);

        this._assetReference = new AssetReference(
            'asset',
            this,
            system.app.assets, {
                add: this._onAssetAdded,
                load: this._onAssetLoad,
                remove: this._onAssetRemove,
                unload: this._onAssetUnload
            },
            this
        );
    }

    /**
     * Sets the `container` asset carrying the meshlet extension.
     *
     * @type {Asset|number|null}
     */
    set asset(value) {
        const id = value?.id ?? value;
        if (this._assetReference.id === id) return;
        this._assetReference.id = id;
        const asset = this._assetReference.asset;
        if (asset && !asset.resource && this.enabled && this.entity.enabled) {
            this.system.app.assets.load(asset);
        }
        this.system._markDirty();
    }

    /**
     * Gets the asset id.
     *
     * @type {number|null}
     */
    get asset() {
        return this._assetReference.id;
    }

    /**
     * Sets a meshlet resource directly, bypassing the asset system. Also set {@link baseUrl} so
     * pages can stream. Ignored when an asset is assigned.
     *
     * @type {MeshletResource|null}
     */
    set resource(value) {
        if (this._resource === value) return;
        this._resource = value;
        this.system._markDirty();
    }

    get resource() {
        return this._resource;
    }

    /**
     * Sets the URL directory page streaming resolves the manifest's blob URIs against. Derived
     * from the asset's URL when an asset is used; required with a direct {@link resource}.
     *
     * @type {string|null}
     */
    set baseUrl(value) {
        if (this._baseUrl === value) return;
        this._baseUrl = value;
        this.system._markDirty();
    }

    get baseUrl() {
        return this._baseUrl;
    }

    /**
     * The resource this component contributes to the meshlet world, or null when not loaded.
     *
     * @type {MeshletResource|null}
     * @ignore
     */
    get _effectiveResource() {
        const containerResource = this._assetReference.asset?.resource;
        return containerResource?.meshlets?.[0] ?? this._resource;
    }

    /**
     * The stream base URL for the effective resource.
     *
     * @type {string|null}
     * @ignore
     */
    get _effectiveBaseUrl() {
        if (this._baseUrl !== null) return this._baseUrl;
        const url = this._assetReference.asset?.file?.url;
        if (!url) return null;
        const slash = url.lastIndexOf('/');
        return slash >= 0 ? url.substring(0, slash) : '.';
    }

    _onAssetAdded(asset) {
        if (!asset.resource && this.enabled && this.entity.enabled) {
            this.system.app.assets.load(asset);
        }
    }

    _onAssetLoad() {
        this.system._markDirty();
    }

    _onAssetRemove() {
        this.system._markDirty();
    }

    _onAssetUnload() {
        this.system._markDirty();
    }

    onEnable() {
        const asset = this._assetReference.asset;
        if (asset && !asset.resource) {
            this.system.app.assets.load(asset);
        }
        // no rebuild: the system's per-frame sync flips this component's hide bit
    }

    onDisable() {
        // no rebuild: the system's per-frame sync flips this component's hide bit
    }

    onBeforeRemove() {
        this.asset = null;
        this._resource = null;
        this.system._markDirty();
    }
}

export { MeshletComponent };
