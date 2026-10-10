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
 * Load state of a meshlet component's streamed content.
 *
 * @typedef {object} MeshletStreamStatus
 * @property {boolean} playable - Roots, playable floor and texture tails are resident (or the
 * stream was given up on - see `failed`).
 * @property {boolean} failed - The stream was given up on, or the component was removed before
 * it became playable.
 * @property {number} progress - 0..1 toward playable, by bytes.
 * @property {number} totalBytes - Bytes needed to become playable: root and floor pages plus
 * texture tails. 0 until the world has placed the component.
 * @property {number} loadedBytes - Bytes of those already resident.
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
 * A mixed container can also retain regular primitives, such as alpha-blended windows. Render
 * these by adding `containerAsset.resource.instantiateRenderEntity()` as a child of this
 * entity. The glTF parser excludes streamed primitives from that regular render hierarchy;
 * the meshlet component itself only draws the streamed primitives.
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
     * Fired while this component's streamed content loads, with its progress (0..1) toward
     * {@link MeshletComponent#playable} and the bytes that progress weighs.
     *
     * @event
     * @example
     * entity.meshlet.on('progress', (progress, totalBytes) => {
     *     console.log(`${Math.round(progress * 100)}% of ${totalBytes} bytes`);
     * });
     */
    static EVENT_PROGRESS = 'progress';

    /**
     * Fired when this component's content becomes {@link MeshletComponent#playable}.
     *
     * @event
     * @example
     * entity.meshlet.on('playable', () => {
     *     console.log('streamed model ready');
     * });
     */
    static EVENT_PLAYABLE = 'playable';

    /**
     * @type {AssetReference}
     * @private
     */
    _assetReference;

    /**
     * The resource the last world rebuild placed for this component.
     *
     * @type {MeshletResource|null}
     * @ignore
     */
    _placedResource = null;

    /**
     * Pending {@link whenPlayable} resolvers.
     *
     * @type {Array<(status: MeshletStreamStatus) => void>}
     * @ignore
     */
    _playableWaiters = [];

    /** @type {number} - progress last reported through {@link EVENT_PROGRESS}. @ignore */
    _reportedProgress = -1;

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
     * @type {boolean}
     * @private
     */
    _castShadows = true;

    /**
     * Shadow-casting state currently baked into the world's objectData rows (null = unknown).
     *
     * @type {boolean|null}
     * @ignore
     */
    _castShadowsApplied = null;

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
     * Whether this component is in the system's transform sync queue.
     *
     * @type {boolean}
     * @ignore
     */
    _transformQueued = false;

    /**
     * Whether this component is in the system's hidden / shadow-casting sync queue.
     *
     * @type {boolean}
     * @ignore
     */
    _stateQueued = false;

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

        // the entity reports its own world-transform changes, so the per-frame sync visits
        // only moved components rather than polling every one
        entity._worldDirtyListener = this;
    }

    /**
     * Called by the entity whenever its world transform (or an ancestor's) is dirtied.
     *
     * @ignore
     */
    onWorldDirty() {
        if (this._transformQueued) return;
        this._transformQueued = true;
        this.system._transformQueue.push(this);
    }

    /**
     * Queues a re-check of this component's hidden and shadow-casting state.
     *
     * @ignore
     */
    _queueState() {
        if (this._stateQueued) return;
        this._stateQueued = true;
        this.system._stateQueue.push(this);
    }

    /**
     * Sets whether this component's instances cast shadows. Per instance: applied by the
     * system's per-frame sync as a buffer write, no rebuild. Shadow passes run only while the
     * system's {@link MeshletComponentSystem#shadows} is on. Defaults to true.
     *
     * @type {boolean}
     */
    set castShadows(value) {
        value = value !== false;
        if (this._castShadows === value) return;
        this._castShadows = value;
        this._queueState();
    }

    /**
     * Gets whether this component's instances cast shadows.
     *
     * @type {boolean}
     */
    get castShadows() {
        return this._castShadows;
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
     * The stream base URL for the effective resource. Null when there is none to stream from:
     * no asset URL, or an asset loaded from a `blob:` or `data:` URL - those cannot be a base
     * for the package's relative sidecar URIs, so such a component stays inert rather than
     * failing every URL it resolves.
     *
     * @type {string|null}
     * @ignore
     */
    get _effectiveBaseUrl() {
        if (this._baseUrl !== null) return this._baseUrl;
        const url = this._assetReference.asset?.file?.url;
        if (!url || /^(?:blob|data):/i.test(url)) return null;
        const slash = url.lastIndexOf('/');
        return slash >= 0 ? url.substring(0, slash) : '.';
    }

    /**
     * How the effective resource's sidecars (geometry shards, texture containers) are fetched.
     * With an asset, each URI resolves through the application's URL resolver relative to the
     * asset's file URL - the same route the asset itself loaded by - so a host that rewrites or
     * signs a package's URLs serves the sidecars too, and requests carry cookies when the asset
     * was loaded with `crossOrigin: 'use-credentials'`. A direct resource with an explicit
     * {@link baseUrl} appends the URI to it and sends no credentials (null).
     *
     * @type {import('../../../scene/meshlet/meshlet-world.js').MeshletFetchOptions|null}
     * @ignore
     */
    get _effectiveFetchOptions() {
        if (this._baseUrl !== null) return null;
        const asset = this._assetReference.asset;
        const fileUrl = asset?.file?.url;
        if (!fileUrl) return null;
        const app = this.system.app;
        return {
            resolveUrl: typeof app.resolveUrl === 'function' ? uri => app.resolveUrl(uri, { baseUrl: fileUrl }).load : null,
            credentials: asset.options?.crossOrigin === 'use-credentials' ? 'include' : null
        };
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
        this._queueState();
    }

    onDisable() {
        // no rebuild: the system's per-frame sync flips this component's hide bit
        this._queueState();
    }

    /**
     * Load state of this component's streamed content: its root pages, its playable floor (the
     * coarsest LOD free of sloppy simplification) and its texture tails. Not playable while the
     * asset loads or before a world rebuild has placed it.
     *
     * @type {MeshletStreamStatus}
     */
    get streamStatus() {
        return this.system._componentStatus(this);
    }

    /**
     * Whether this component's streamed content is resident down to its playable floor, with
     * its texture tails. Finer detail keeps streaming afterwards.
     *
     * @type {boolean}
     */
    get playable() {
        return this.streamStatus.playable;
    }

    /**
     * Resolves once this component's content is {@link playable} - at once if it is - or the
     * component is removed. Fires {@link EVENT_PROGRESS} meanwhile.
     *
     * @returns {Promise<MeshletStreamStatus>} The status when settled; `failed` is set when the
     * stream was given up on or the component was removed first.
     * @example
     * const status = await entity.meshlet.whenPlayable();
     */
    whenPlayable() {
        const status = this.streamStatus;
        if (status.playable) return Promise.resolve(status);
        return new Promise((resolve) => {
            this._playableWaiters.push(resolve);
            this.system._watchPlayable(this);
        });
    }

    /**
     * Settles every {@link whenPlayable} waiter.
     *
     * @param {MeshletStreamStatus} status - The status to resolve with.
     * @ignore
     */
    _settlePlayable(status) {
        const waiters = this._playableWaiters;
        this._playableWaiters = [];
        this._reportedProgress = -1;
        for (const resolve of waiters) resolve(status);
    }

    onBeforeRemove() {
        if (this.entity._worldDirtyListener === this) this.entity._worldDirtyListener = null;
        this.system._unwatchPlayable(this);
        this._settlePlayable({ playable: false, failed: true, progress: 0, totalBytes: 0, loadedBytes: 0 });
        this.asset = null;
        this._resource = null;
        this.system._markDirty();
    }
}

export { MeshletComponent };
