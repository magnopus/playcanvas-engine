import { Debug } from '../../../core/debug.js';
import { restoreMeshletRecords } from '../../parsers/glb/extensions/mag-meshlets.js';

/**
 * @import { MeshletComponent } from './component.js'
 * @import { MeshletResource } from '../../../scene/meshlet/meshlet-resource.js'
 */

/**
 * Releases the CPU meshlet records of resources whose worlds have uploaded them, and fetches them
 * back from the resource's GLB when a rebuild or a reader needs them again.
 *
 * @ignore
 */
class MeshletRecordStore {
    /**
     * Where each resource whose records were released reloads them from.
     *
     * @type {Map<MeshletResource, { url: string, credentials: RequestCredentials|null }>}
     */
    sources = new Map();

    /**
     * Record restores in flight (or waiting to retry after a failure), by resource.
     *
     * @type {Map<MeshletResource, Promise<void>>}
     * @private
     */
    _restoring = new Map();

    /**
     * Consecutive restore failures, by resource, for the retry backoff.
     *
     * @type {Map<MeshletResource, number>}
     * @private
     */
    _failures = new Map();

    /**
     * Called when a restore settles, successfully or ready to retry. Null once destroyed.
     *
     * @type {(() => void)|null}
     * @private
     */
    _onSettled;

    /**
     * @param {() => void} onSettled - Called when a restore settles.
     */
    constructor(onSettled) {
        this._onSettled = onSettled;
    }

    /**
     * Whether any restore is pending.
     *
     * @type {boolean}
     */
    get restoring() {
        return this._restoring.size > 0;
    }

    /**
     * @param {MeshletResource} resource - The resource.
     * @returns {boolean} Whether a restore of its records is pending.
     */
    isRestoring(resource) {
        return this._restoring.has(resource);
    }

    /**
     * Releases a resource group's meshlet records when they can be fetched back: the resource
     * came from an asset with a file URL. A direct resource has nowhere to reload from.
     *
     * @param {{ resource: MeshletResource, components: MeshletComponent[] }} group - The group.
     */
    release(group) {
        const component = group.components[0];
        const url = component._assetReference.asset?.getFileUrl();
        if (!url || !group.resource.recordsResident) return;
        this.sources.set(group.resource, { url, credentials: component._effectiveFetchOptions?.credentials ?? null });
        group.resource.releaseRecords();
    }

    /**
     * Starts fetching a released resource's records back, once.
     *
     * @param {MeshletResource} resource - The resource.
     */
    restore(resource) {
        if (this._restoring.has(resource)) return;
        const source = this.sources.get(resource);
        if (!source) {
            Debug.error('MeshletComponentSystem: a resource\'s meshlet records were released with no source to restore them from.');
            return;
        }
        const settle = () => {
            if (!this._onSettled) return;
            this._restoring.delete(resource);
            this._onSettled();
        };
        const promise = restoreMeshletRecords(source.url, resource, source.credentials).then(() => {
            this._failures.delete(resource);
            settle();
        }, (err) => {
            // the world keeps its last build; the rebuild retries the restore after a backoff
            const failures = (this._failures.get(resource) ?? 0) + 1;
            this._failures.set(resource, failures);
            const delayMs = Math.min(1000 * 2 ** (failures - 1), 30000);
            Debug.error(`MeshletComponentSystem: restoring meshlet records from ${source.url} failed, retrying in ${delayMs} ms`, err);
            return new Promise((resolve) => {
                setTimeout(resolve, delayMs);
            }).then(settle);
        });
        this._restoring.set(resource, promise);
    }

    /**
     * Forgets the sources of resources no world places any more, so they can be freed.
     *
     * @param {Set<MeshletResource>} placed - The resources placed by the current worlds.
     */
    retain(placed) {
        for (const resource of this.sources.keys()) {
            if (!placed.has(resource) && !this._restoring.has(resource)) {
                this.sources.delete(resource);
                this._failures.delete(resource);
            }
        }
    }

    /**
     * Runs `callback` with every released resource's records fetched back, and releases the ones
     * it fetched again afterwards, unless a rebuild has taken them over meanwhile.
     *
     * @template T
     * @param {() => (T|Promise<T>)} callback - Reads the records.
     * @returns {Promise<T>} What the callback returns.
     */
    async withRecords(callback) {
        const restored = [];
        const pending = [];
        for (const [resource, source] of this.sources) {
            if (resource.recordsResident || resource.primitives.length === 0) continue;
            restored.push(resource);
            pending.push(restoreMeshletRecords(source.url, resource, source.credentials));
        }
        try {
            await Promise.all(pending);
            return await callback();
        } finally {
            for (const resource of restored) {
                if (!this._restoring.has(resource)) resource.releaseRecords();
            }
        }
    }

    destroy() {
        this._onSettled = null;
        this.sources.clear();
        this._failures.clear();
    }
}

export { MeshletRecordStore };
