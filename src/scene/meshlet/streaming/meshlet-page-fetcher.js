import { PAGE_TABLE, PAGE_TABLE_FIELDS } from '../constants.js';

/**
 * HTTP Range fetching of meshlet pages with run coalescing: wanted pages are sorted by shard
 * offset and merged into contiguous byte runs (small gaps fetched and discarded are cheaper
 * than extra requests), each run one Range request.
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
     */
    constructor(manifest, baseUrl) {
        this.manifest = manifest;
        this.baseUrl = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
    }

    _blobUrl(blobIndex) {
        return this.baseUrl + this.manifest.blobs[blobIndex].uri;
    }

    /**
     * Fetches one byte range of a shard.
     *
     * @param {number} blobIndex - Index into the manifest's blobs.
     * @param {number} offset - Byte offset.
     * @param {number} length - Byte length.
     * @returns {Promise<ArrayBuffer>} The bytes.
     */
    async fetchRange(blobIndex, offset, length) {
        const response = await fetch(this._blobUrl(blobIndex), {
            headers: { Range: `bytes=${offset}-${offset + length - 1}` }
        });
        if (!response.ok && response.status !== 206) {
            throw new Error(`meshlet shard fetch failed (${response.status}): ${this._blobUrl(blobIndex)}`);
        }
        const buffer = await response.arrayBuffer();
        if (response.status === 200 && buffer.byteLength > length) {
            // server ignored the Range header and returned the whole shard
            return buffer.slice(offset, offset + length);
        }
        return buffer;
    }

    /**
     * Fetches the whole shard (used for the eager roots blob).
     *
     * @param {number} blobIndex - Index into the manifest's blobs.
     * @returns {Promise<ArrayBuffer>} The bytes.
     */
    async fetchBlob(blobIndex) {
        const response = await fetch(this._blobUrl(blobIndex));
        if (!response.ok) {
            throw new Error(`meshlet shard fetch failed (${response.status}): ${this._blobUrl(blobIndex)}`);
        }
        return response.arrayBuffer();
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
