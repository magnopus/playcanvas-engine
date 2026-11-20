import { ByteStore } from './byte-store.js';

/**
 * Origin the synthetic request URLs are keyed under. The `.invalid` TLD never resolves, so the
 * entries can never be confused with real responses.
 */
const KEY_ORIGIN = 'https://byte-store.invalid/';

/**
 * A {@link ByteStore} in the browser's Cache Storage (the Cache API used by service workers).
 * Persistent across sessions and available in every browser that supports service workers,
 * which makes it the fallback where an {@link OpfsByteStore} is not supported.
 *
 * @category Asset
 */
class CacheStorageByteStore extends ByteStore {
    /**
     * Name of the cache in Cache Storage the values are stored in.
     *
     * @type {string}
     */
    cacheName;

    /**
     * @type {Cache|null}
     * @private
     */
    _cache = null;

    /**
     * Creates a new CacheStorageByteStore.
     *
     * @param {string} [cacheName] - Name of the cache the values are stored in. Defaults to
     * 'playcanvas-byte-cache'. Stores with different names are independent.
     */
    constructor(cacheName = 'playcanvas-byte-cache') {
        super();
        this.cacheName = cacheName;
    }

    /**
     * Whether the browser supports this store.
     *
     * @returns {boolean} True when Cache Storage is available in this context.
     */
    static isSupported() {
        return typeof caches !== 'undefined';
    }

    async open() {
        this._cache = await caches.open(this.cacheName);
    }

    async read(name) {
        try {
            const response = await this._cache.match(KEY_ORIGIN + name);
            return response ? await response.arrayBuffer() : null;
        } catch (err) {
            return null;
        }
    }

    write(name, data) {
        return this._cache.put(KEY_ORIGIN + name, new Response(data));
    }

    async remove(name) {
        await this._cache.delete(KEY_ORIGIN + name);
    }

    async list() {
        const requests = await this._cache.keys();
        return requests.map(request => request.url.slice(KEY_ORIGIN.length));
    }

    async clear() {
        await caches.delete(this.cacheName);
        this._cache = await caches.open(this.cacheName);
    }
}

export { CacheStorageByteStore };
