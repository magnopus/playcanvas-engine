import { deserializeCutTables, serializeCutTables } from '../../../scene/meshlet/meshlet-cut-table.js';

/**
 * @import { ByteCache } from '../../../platform/net/byte-cache.js'
 * @import { MeshletComponent } from './component.js'
 * @import { MeshletResource } from '../../../scene/meshlet/meshlet-resource.js'
 */

/**
 * Keeps each resource's meshlet cut tables in the byte cache, so a later load reads them instead
 * of deriving them again. A world build derives them from the meshlet records, which for a large
 * scene is seconds of main-thread work; they depend on the records alone, so they are saved per
 * resource, once, and the next session's build finds them waiting.
 *
 * The cache is asynchronous and the build is not, so a rebuild first asks for the tables of the
 * resources it will place ({@link request}) and stays pending until they arrive. Tables a build
 * had to compute are saved by {@link finish}.
 *
 * @ignore
 */
class MeshletCutTableStore {
    /**
     * Resources with fewer meshlets than this are not saved or loaded: their tables are built
     * faster than a rebuild could wait for them to be read back (a Caldera region's are about
     * 150k meshlets and gained nothing), so only a very large resource's are worth the trip.
     *
     * @type {number}
     */
    minMeshlets = 500000;

    /** @type {{ requested: number, loaded: number, saved: number, savedBytes: number }} - counters, for tuning. */
    stats = { requested: 0, loaded: 0, saved: 0, savedBytes: 0 };

    /**
     * Loads in flight, by resource.
     *
     * @type {Map<MeshletResource, Promise<void>>}
     * @private
     */
    _loading = new Map();

    /**
     * Resources whose tables were already asked for since they last went through a build.
     *
     * @type {WeakSet<MeshletResource>}
     * @private
     */
    _asked = new WeakSet();

    /**
     * Called when a load settles. Null once destroyed.
     *
     * @type {(() => void)|null}
     * @private
     */
    _onSettled;

    /**
     * @param {() => void} onSettled - Called when a load settles.
     */
    constructor(onSettled) {
        this._onSettled = onSettled;
    }

    /**
     * Whether any load is pending.
     *
     * @type {boolean}
     */
    get loading() {
        return this._loading.size > 0;
    }

    /**
     * The byte cache key of a resource's tables, or null when it has nothing stable to be keyed
     * by (a resource that did not come from an asset). The cache assumes the content behind a key
     * never changes, as it does for the asset's own file; the bake id and meshlet count keep a
     * rebake under the same URL from matching, and each table also carries a fingerprint of its
     * records.
     *
     * @param {{ resource: MeshletResource, components: MeshletComponent[] }} group - The group.
     * @returns {string|null} The key.
     */
    keyOf(group) {
        const url = group.components[0]?._assetReference.asset?.getFileUrl();
        if (!url) return null;
        const { manifest, totalMeshlets } = group.resource;
        return `${url}#meshlet-cut-1:${manifest?.bakeId ?? ''}:${totalMeshlets}`;
    }

    /**
     * Starts loading a group's saved tables, once per build.
     *
     * @param {{ resource: MeshletResource, components: MeshletComponent[] }} group - The group.
     * @param {ByteCache|null} cache - The byte cache, if any.
     * @returns {boolean} True while its load is pending.
     */
    request(group, cache) {
        const resource = group.resource;
        if (this._loading.has(resource)) return true;
        if (!cache || this._asked.has(resource) || resource.totalMeshlets < this.minMeshlets) return false;
        // tables built as the resource arrived (MeshletResource#prepare) need no cache read
        if (resource.primitives.every(prim => prim._cutTable)) return false;
        this._asked.add(resource);
        const key = this.keyOf(group);
        if (!key || !cache.accepts(key)) return false;

        this.stats.requested++;
        const promise = cache.ready.then(() => cache.getFile(key)).then((hit) => {
            const tables = hit && deserializeCutTables(hit.data, resource.primitives.length);
            if (tables) {
                this.stats.loaded++;
                resource.primitives.forEach((prim, i) => {
                    prim._cutTable = tables[i];
                });
            }
        }).catch(() => {}).finally(() => {
            if (!this._onSettled) return;
            this._loading.delete(resource);
            this._onSettled();
        });
        this._loading.set(resource, promise);
        return true;
    }

    /**
     * Saves the tables a build computed and drops the ones it loaded, which have done their job.
     * The next build of the resource asks again, and finds what was just saved.
     *
     * @param {Array<{ resource: MeshletResource, components: MeshletComponent[] }>} groups - The
     * groups the build placed.
     * @param {ByteCache|null} cache - The byte cache, if any.
     */
    finish(groups, cache) {
        for (const group of groups) {
            const prims = group.resource.primitives;
            const key = group.resource.totalMeshlets >= this.minMeshlets ? this.keyOf(group) : null;
            if (key && cache?.accepts(key) && prims.some(prim => prim._builtCutTable)) {
                const blob = serializeCutTables(prims.map(prim => prim._builtCutTable ?? prim._cutTable ?? null));
                // the blob must stay attached until the write finishes: nothing else holds it
                cache.putFile(key, blob, 'application/octet-stream');
                this.stats.saved++;
                this.stats.savedBytes += blob.byteLength;
            }
            for (const prim of prims) {
                prim._cutTable = null;
                prim._builtCutTable = null;
            }
            this._asked.delete(group.resource);
        }
    }

    destroy() {
        this._onSettled = null;
        this._loading.clear();
    }
}

export { MeshletCutTableStore };
