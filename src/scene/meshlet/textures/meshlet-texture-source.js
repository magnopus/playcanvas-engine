/**
 * HTTP byte-range access to a MAG_texture_streaming v2 container (see the gltf-tools spec):
 * one packed file per texture array, addressed by the manifest's absolute per-(layer, mip)
 * byte ranges. The shared tail region (every layer's coarse mips) is one request; each finer
 * mip band is one more. Region fetches are promise-cached by (url, offset) so concurrent
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
     */
    constructor(baseUrl, fetchOptions = null) {
        this.baseUrl = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
        this.resolveUrl = fetchOptions?.resolveUrl ?? null;
        this.credentials = fetchOptions?.credentials ?? null;
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
     * @returns {Promise<ArrayBuffer>} The region's bytes.
     */
    fetchRegion(uri, byteOffset, byteLength) {
        const url = this._url(uri);
        const key = `${url}@${byteOffset}+${byteLength}`;
        let promise = this._regions.get(key);
        if (!promise) {
            /** @type {RequestInit} */
            const init = { headers: { Range: `bytes=${byteOffset}-${byteOffset + byteLength - 1}` } };
            if (this.credentials) init.credentials = this.credentials;
            promise = fetch(url, init).then((response) => {
                if (!response.ok && response.status !== 206) {
                    throw new Error(`MeshletTextureSource: ${response.status} fetching ${url}`);
                }
                return response.arrayBuffer().then((buffer) => {
                    // status 200 = host ignored the Range header and sent the whole file
                    if (response.status === 200 && buffer.byteLength > byteLength) {
                        return buffer.slice(byteOffset, byteOffset + byteLength);
                    }
                    return buffer;
                });
            });
            this._regions.set(key, promise);
        }
        return promise;
    }

    /**
     * Fetches a manifest array's shared tail region.
     *
     * @param {object} array - The manifest array (containerVersion 2).
     * @returns {Promise<ArrayBuffer>} The tail region's bytes.
     */
    fetchTail(array) {
        return this.fetchRegion(array.container.uri, array.tail.byteOffset, array.tail.byteLength);
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
