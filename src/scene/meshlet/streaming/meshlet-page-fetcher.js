import { PAGE_TABLE, PAGE_TABLE_FIELDS } from '../constants.js';
import { FETCH_PRIORITY_PAGES, FETCH_PRIORITY_ROOTS, defaultFetchScheduler } from './meshlet-fetch-scheduler.js';

/**
 * HTTP Range fetching of meshlet pages with run coalescing: wanted pages are sorted by shard
 * offset and merged into contiguous byte runs (small gaps fetched and discarded are cheaper
 * than extra requests), each run one Range request through the world's fetch scheduler.
 *
 * @ignore
 */

/** Gap between wanted pages that still merges into one run - fetching and discarding up to this much is cheaper than another request. */
export const RUN_GAP_BYTES = 512 * 1024;

/** Largest single run, so one completion never carries more than this into the install queue. */
export const MAX_RUN_BYTES = 16 * 1024 * 1024;

class MeshletPageFetcher {
    /**
     * @param {object} manifest - The resource's stream manifest.
     * @param {string} baseUrl - URL directory the manifest's blob URIs are relative to.
     * @param {import('../meshlet-world.js').MeshletFetchOptions|null} [fetchOptions] - How the
     * sidecars are fetched: `resolveUrl` maps a manifest URI to the URL to fetch in place of
     * appending it to the base URL (the application's URL resolver, so hosts that store a
     * package's files behind rewritten or signed URLs can serve them), `credentials` is the fetch
     * credentials mode (cookies for hosts that need them).
     * @param {import('./meshlet-fetch-scheduler.js').MeshletFetchScheduler|null} [scheduler] - The
     * scheduler requests go through; null selects the page-wide default.
     */
    constructor(manifest, baseUrl, fetchOptions = null, scheduler = null) {
        this.manifest = manifest;
        this.baseUrl = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
        this.resolveUrl = fetchOptions?.resolveUrl ?? null;
        this.credentials = fetchOptions?.credentials ?? null;
        this.scheduler = scheduler ?? defaultFetchScheduler();

        /** @type {string[]} - per-blob fetch URLs, resolved once. */
        this._urls = [];
    }

    _blobUrl(blobIndex) {
        let url = this._urls[blobIndex];
        if (url === undefined) {
            const uri = this.manifest.blobs[blobIndex].uri;
            url = this.resolveUrl ? this.resolveUrl(uri) : this.baseUrl + uri;
            this._urls[blobIndex] = url;
        }
        return url;
    }

    /**
     * Fetches one byte range of a shard.
     *
     * @param {number} blobIndex - Index into the manifest's blobs.
     * @param {number} offset - Byte offset.
     * @param {number} length - Byte length.
     * @returns {Promise<ArrayBuffer|null>} The bytes, or null when the scheduler dropped the request.
     */
    fetchRange(blobIndex, offset, length) {
        return this.scheduler.fetchRange(this._blobUrl(blobIndex), offset, length, {
            credentials: this.credentials,
            priority: FETCH_PRIORITY_PAGES
        });
    }

    /**
     * Fetches the whole shard (used for the eager roots blob).
     *
     * @param {number} blobIndex - Index into the manifest's blobs.
     * @returns {Promise<ArrayBuffer|null>} The bytes, or null when the scheduler dropped the request.
     */
    fetchBlob(blobIndex) {
        return this.scheduler.fetchAll(this._blobUrl(blobIndex), {
            credentials: this.credentials,
            priority: FETCH_PRIORITY_ROOTS
        });
    }

    /**
     * Coalesces local page indices into fetch runs.
     *
     * @param {number[]} localPages - Resource-local page indices.
     * @returns {Array<{ blob: number, offset: number, length: number, pages: Array<{ localPage: number, byteOffset: number }> }>} Runs.
     */
    buildRuns(localPages) {
        const table = this.manifest.pageTable;
        const pageSize = this.manifest.pageSizeBytes;
        const entries = localPages.map((localPage) => {
            const entry = localPage * PAGE_TABLE_FIELDS;
            return {
                localPage,
                blob: table[entry + PAGE_TABLE.BLOB],
                // 64-bit shard offsets arrive as two u32 words
                offset: table[entry + PAGE_TABLE.OFFSET_HI] * 0x100000000 + table[entry + PAGE_TABLE.OFFSET_LO]
            };
        });
        entries.sort((a, b) => (a.blob - b.blob) || (a.offset - b.offset));

        const runs = [];
        let run = null;
        for (const page of entries) {
            const end = page.offset + pageSize;
            if (run && run.blob === page.blob &&
                page.offset - (run.offset + run.length) <= RUN_GAP_BYTES &&
                end - run.offset <= MAX_RUN_BYTES) {
                run.length = end - run.offset;
                run.pages.push({ localPage: page.localPage, byteOffset: page.offset - run.offset });
            } else {
                run = { blob: page.blob, offset: page.offset, length: pageSize, pages: [{ localPage: page.localPage, byteOffset: 0 }] };
                runs.push(run);
            }
        }
        return runs;
    }
}

export { MeshletPageFetcher };
