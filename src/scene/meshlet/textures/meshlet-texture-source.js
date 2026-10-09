import { FETCH_PRIORITY_FINE, FETCH_PRIORITY_TAILS, defaultFetchScheduler } from '../streaming/meshlet-fetch-scheduler.js';

/**
 * HTTP byte-range access to a MAG_texture_streaming v2 container (see the gltf-tools spec):
 * one packed file per texture array, addressed by the manifest's absolute per-(layer, mip)
 * byte ranges. The shared tail region (every layer's coarse mips) is one request; each finer
 * mip band is one more, all through the world's fetch scheduler (which caps concurrency and
 * merges neighbouring bands). Region fetches are promise-cached by (url, offset) so concurrent
 * consumers of one region share a single request.
 *
 * @ignore
 */
class MeshletTextureSource {
    /** @type {Map<string, Promise<ArrayBuffer>>} */
    _regions = new Map();

    /** @type {Map<string, string>} - container URI -> fetch URL, resolved once. */
    _urls = new Map();

    /**
     * @param {string} baseUrl - Directory the manifest's container URIs are relative to.
     * @param {import('../meshlet-world.js').MeshletFetchOptions|null} [fetchOptions] - How the
     * containers are fetched - see MeshletPageFetcher.
     * @param {import('../streaming/meshlet-fetch-scheduler.js').MeshletFetchScheduler|null} [scheduler] - The
     * scheduler requests go through; null selects the page-wide default.
     */
    constructor(baseUrl, fetchOptions = null, scheduler = null) {
        this.baseUrl = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
        this.resolveUrl = fetchOptions?.resolveUrl ?? null;
        this.credentials = fetchOptions?.credentials ?? null;
        this.scheduler = scheduler ?? defaultFetchScheduler();
    }

    /**
     * The URL a container URI is fetched from.
     *
     * @param {string} uri - Container URI relative to the base URL.
     * @returns {string} The URL.
     * @private
     */
    _url(uri) {
        let url = this._urls.get(uri);
        if (url === undefined) {
            url = this.resolveUrl ? this.resolveUrl(uri) : this.baseUrl + uri;
            this._urls.set(uri, url);
        }
        return url;
    }

    /**
     * Fetches one byte range of a container. Falls back to slicing a full-body response when
     * the host ignores the Range header (status 200 instead of 206).
     *
     * @param {string} uri - Container URI relative to the base URL.
     * @param {number} byteOffset - Absolute offset into the container.
     * @param {number} byteLength - Length of the region.
     * @param {number} [priority] - Scheduler priority (FETCH_PRIORITY_*).
     * @param {(() => boolean)|null} [stillWanted] - Polled before the request is sent; false
     * drops it (the promise resolves null).
     * @returns {Promise<ArrayBuffer|null>} The region's bytes, or null when dropped.
     */
    fetchRegion(uri, byteOffset, byteLength, priority = FETCH_PRIORITY_FINE, stillWanted = null) {
        const url = this._url(uri);
        const key = `${url}@${byteOffset}+${byteLength}`;
        let promise = this._regions.get(key);
        if (!promise) {
            promise = this.scheduler.fetchRange(url, byteOffset, byteLength, {
                credentials: this.credentials,
                priority,
                stillWanted
            }).then((bytes) => {
                // a dropped or failed request must not stand in for a later, wanted one
                if (!bytes) this._regions.delete(key);
                return bytes;
            }, (err) => {
                this._regions.delete(key);
                throw err;
            });
            this._regions.set(key, promise);
        }
        return promise;
    }

    /**
     * Fetches a manifest array's shared tail region.
     *
     * @param {object} array - The manifest array (containerVersion 2).
     * @returns {Promise<ArrayBuffer|null>} The tail region's bytes, or null when dropped.
     */
    fetchTail(array) {
        return this.fetchRegion(array.container.uri, array.tail.byteOffset, array.tail.byteLength, FETCH_PRIORITY_TAILS);
    }

    /**
     * Slices one (layer, mip) file out of a fetched region.
     *
     * @param {ArrayBuffer} regionBuffer - The region's bytes.
     * @param {number} regionOffset - The region's absolute container offset.
     * @param {object} array - The manifest array.
     * @param {number} layer - Layer index.
     * @param {number} mip - Mip level (source mip space).
     * @returns {ArrayBuffer|null} The mip file's bytes, or null when outside the region.
     */
    sliceMip(regionBuffer, regionOffset, array, layer, mip) {
        const range = array.mipRanges?.[layer]?.[mip];
        if (!range) return null;
        const [offset, length] = range;
        const local = offset - regionOffset;
        if (local < 0 || local + length > regionBuffer.byteLength) return null;
        return regionBuffer.slice(local, local + length);
    }

    destroy() {
        this._regions.clear();
    }
}

export { MeshletTextureSource };
