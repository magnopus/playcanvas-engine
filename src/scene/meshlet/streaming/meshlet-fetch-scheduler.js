/**
 * Fetch priorities, lowest first. Root shards gate rendering anything, geometry pages gate the
 * LOD cut, texture tails gate a texture showing at all, fine mips only sharpen one.
 */
export const FETCH_PRIORITY_ROOTS = 0;
export const FETCH_PRIORITY_PAGES = 1;
export const FETCH_PRIORITY_TAILS = 2;
export const FETCH_PRIORITY_FINE = 3;

/** Default cap on sidecar requests in flight. */
export const MAX_CONCURRENT_FETCHES = 16;

/** Queued byte ranges on one URL closer than this merge into a single Range request. */
export const COALESCE_GAP_BYTES = 256 * 1024;

/** Largest merged Range request. */
export const MAX_COALESCED_BYTES = 8 * 1024 * 1024;

/**
 * @typedef {object} MeshletFetchRequestOptions
 * @property {string|null} [credentials] - Fetch credentials mode, or null for the default.
 * @property {number} [priority] - One of the FETCH_PRIORITY_* values; lower dispatches first.
 * @property {(() => boolean)|null} [stillWanted] - Polled when the request reaches the front of
 * the queue; a false answer drops the request unsent and resolves it with null.
 */

/**
 * The HTTP scheduler for a meshlet world's sidecar fetches: geometry shards and texture
 * containers. Every request queues here and at most {@link maxConcurrent} are in flight at a
 * time, in priority order, so a large scene's first frames - hundreds of resources each wanting
 * a root shard, its texture tails and a burst of pages at once - do not pile thousands of
 * fetch() calls into the browser, which fails them with ERR_INSUFFICIENT_RESOURCES instead of
 * queueing them. Queued byte ranges on one URL that lie within {@link COALESCE_GAP_BYTES} of
 * each other go out as one Range request and are sliced apart on arrival, and a queued request
 * whose `stillWanted` callback has turned false by the time a slot frees is dropped instead of
 * fetched (a fine mip whose slot was evicted while it waited).
 *
 * @ignore
 */
class MeshletFetchScheduler {
    /** @type {number} - cap on requests in flight. */
    maxConcurrent;

    /** @type {number} - requests in flight. */
    active = 0;

    /** @type {number} - requests sent. */
    sent = 0;

    /** @type {number[]} - requests sent per FETCH_PRIORITY_* value. */
    sentByPriority = [0, 0, 0, 0];

    /** @type {number} - queued requests dropped as no longer wanted. */
    dropped = 0;

    /** @type {number} - queued requests merged into another request. */
    coalesced = 0;

    /**
     * Pending requests in dispatch order. `offset` is -1 for a whole-file request.
     *
     * @type {Array<{ url: string, offset: number, length: number, credentials: string|null,
     * priority: number, stillWanted: (() => boolean)|null, resolve: Function, reject: Function }>}
     * @private
     */
    _queue = [];

    /**
     * @param {number} [maxConcurrent] - Cap on requests in flight.
     */
    constructor(maxConcurrent = MAX_CONCURRENT_FETCHES) {
        this.maxConcurrent = maxConcurrent;
    }

    /** @type {number} - requests waiting for a slot. */
    get queued() {
        return this._queue.length;
    }

    /**
     * Fetches a whole file.
     *
     * @param {string} url - The URL.
     * @param {MeshletFetchRequestOptions} [options] - Request options.
     * @returns {Promise<ArrayBuffer|null>} The bytes, or null when dropped.
     */
    fetchAll(url, options = {}) {
        return this._enqueue(url, -1, 0, options);
    }

    /**
     * Fetches one byte range of a file. A host that ignores the Range header and answers with
     * the whole file is sliced to the range.
     *
     * @param {string} url - The URL.
     * @param {number} offset - First byte.
     * @param {number} length - Byte count.
     * @param {MeshletFetchRequestOptions} [options] - Request options.
     * @returns {Promise<ArrayBuffer|null>} The range's bytes, or null when dropped.
     */
    fetchRange(url, offset, length, options = {}) {
        return this._enqueue(url, offset, length, options);
    }

    /**
     * Drops every queued request (each resolves null). Requests in flight complete.
     */
    clear() {
        const queue = this._queue;
        this._queue = [];
        this.dropped += queue.length;
        for (const req of queue) req.resolve(null);
    }

    /**
     * @param {string} url - The URL.
     * @param {number} offset - First byte, or -1 for the whole file.
     * @param {number} length - Byte count.
     * @param {MeshletFetchRequestOptions} options - Request options.
     * @returns {Promise<ArrayBuffer|null>} The bytes, or null when dropped.
     * @private
     */
    _enqueue(url, offset, length, { credentials = null, priority = FETCH_PRIORITY_FINE, stillWanted = null }) {
        return new Promise((resolve, reject) => {
            const req = { url, offset, length, credentials, priority, stillWanted, resolve, reject };
            // stable priority order: behind every queued request of equal or higher priority
            const queue = this._queue;
            let i = queue.length;
            while (i > 0 && queue[i - 1].priority > priority) i--;
            queue.splice(i, 0, req);
            this._pump();
        });
    }

    /** @private */
    _pump() {
        while (this.active < this.maxConcurrent && this._queue.length) {
            const head = this._queue.shift();
            if (head.stillWanted && !head.stillWanted()) {
                this.dropped++;
                head.resolve(null);
                continue;
            }
            const batch = [head];
            if (head.offset >= 0) this._gather(head, batch);
            this._dispatch(batch);
        }
    }

    /**
     * Pulls queued ranges on the head request's URL into its batch while they fall within the
     * gap of the merged span and the span stays under the size cap. The queue is in dispatch
     * order, so what merges is what would have gone out next anyway.
     *
     * @param {object} head - The request being dispatched.
     * @param {object[]} batch - Its batch, extended in place.
     * @private
     */
    _gather(head, batch) {
        let start = head.offset;
        let end = head.offset + head.length;
        const queue = this._queue;
        for (let i = 0; i < queue.length;) {
            const req = queue[i];
            if (req.offset < 0 || req.url !== head.url || req.credentials !== head.credentials) {
                i++;
                continue;
            }
            const reqEnd = req.offset + req.length;
            // positive when disjoint on either side, negative on overlap
            const gap = Math.max(start - reqEnd, req.offset - end);
            const spanStart = Math.min(start, req.offset);
            const spanEnd = Math.max(end, reqEnd);
            if (gap > COALESCE_GAP_BYTES || spanEnd - spanStart > MAX_COALESCED_BYTES) {
                i++;
                continue;
            }
            queue.splice(i, 1);
            if (req.stillWanted && !req.stillWanted()) {
                this.dropped++;
                req.resolve(null);
                continue;
            }
            batch.push(req);
            this.coalesced++;
            start = spanStart;
            end = spanEnd;
        }
    }

    /**
     * Sends one request for a batch and hands each member its slice of the response.
     *
     * @param {object[]} batch - Requests sharing one URL; ranges, or a single whole-file request.
     * @private
     */
    _dispatch(batch) {
        this.active++;
        this.sent++;
        const head = batch[0];
        this.sentByPriority[head.priority] = (this.sentByPriority[head.priority] ?? 0) + 1;
        const ranged = head.offset >= 0;
        let start = 0;
        let end = 0;
        /** @type {RequestInit} */
        const init = {};
        if (head.credentials) init.credentials = head.credentials;
        if (ranged) {
            start = Infinity;
            for (const req of batch) {
                start = Math.min(start, req.offset);
                end = Math.max(end, req.offset + req.length);
            }
            init.headers = { Range: `bytes=${start}-${end - 1}` };
        }

        fetch(head.url, init).then((response) => {
            if (!response.ok && response.status !== 206) {
                throw new Error(`meshlet sidecar fetch failed (${response.status}): ${head.url}`);
            }
            return response.arrayBuffer().then((buffer) => {
                if (!ranged) {
                    head.resolve(buffer);
                    return;
                }
                // a 200 carrying more than the span is a host that ignored the Range header and
                // sent the whole file: the buffer then starts at byte 0, not at `start`
                const base = (response.status === 200 && buffer.byteLength > end - start) ? 0 : start;
                for (const req of batch) {
                    const local = req.offset - base;
                    if (local < 0 || local >= buffer.byteLength) {
                        req.reject(new Error(`meshlet sidecar fetch short (${buffer.byteLength} bytes for ${start}-${end - 1}): ${head.url}`));
                    } else if (batch.length === 1 && base === start && buffer.byteLength === req.length) {
                        req.resolve(buffer);
                    } else {
                        req.resolve(buffer.slice(local, local + req.length));
                    }
                }
            });
        }).catch((err) => {
            for (const req of batch) req.reject(err);
        }).finally(() => {
            this.active--;
            this._pump();
        });
    }
}

let _defaultScheduler = null;

/**
 * The scheduler used when a world has none assigned - one per page, since the browser's
 * request limit is per page.
 *
 * @returns {MeshletFetchScheduler} The shared scheduler.
 */
function defaultFetchScheduler() {
    _defaultScheduler ??= new MeshletFetchScheduler();
    return _defaultScheduler;
}

export { MeshletFetchScheduler, defaultFetchScheduler };
