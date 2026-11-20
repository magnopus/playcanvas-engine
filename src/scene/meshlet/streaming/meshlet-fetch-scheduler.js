import { Debug } from '../../../core/debug.js';

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
 * Share of the concurrency cap each priority may hold while something less urgent is waiting:
 * root shards take whatever they need, pages half, texture tails and fine mips a quarter each.
 * A priority only exceeds its share when no lower priority has anything queued, so a scene
 * that streams pages continuously (a pool smaller than its working set) still lets its fine
 * mips through instead of starving them behind the page traffic.
 */
const PRIORITY_SHARE = [1, 0.5, 0.25, 0.25];

/**
 * @typedef {object} MeshletFetchRequestOptions
 * @property {string|null} [credentials] - Fetch credentials mode, or null for the default.
 * @property {string|null} [cacheVersion] - Versions the request's {@link MeshletFetchScheduler#byteCache}
 * key: the bytes are cached under `<url>#v=<cacheVersion>`, so content replaced behind an
 * unversioned URL (a rebake) misses instead of being served stale. Null keys by the URL alone.
 * @property {number} [priority] - One of the FETCH_PRIORITY_* values; lower dispatches first.
 * @property {(() => boolean)|null} [stillWanted] - Polled when the request reaches the front of
 * the queue; a false answer drops the request unsent and resolves it with null.
 * @property {boolean} [view] - {@link MeshletFetchScheduler#fetchRange} only: resolve with a
 * Uint8Array over the response in place of an ArrayBuffer of the range's own, saving a copy when
 * the request went out wider than the range (merged with neighbours, or widened to the byte
 * cache's chunks). The view may share its buffer with other requests' views, so it must not be
 * transferred.
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
 * fetched (a fine mip whose slot was evicted while it waited). Priority is by share, not
 * strict: see {@link PRIORITY_SHARE}. A URL whose host answers a Range request with the whole
 * file is kept in memory and its later ranges are served from it.
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

    /** @type {number} - requests answered from {@link byteCache} without a fetch. */
    cacheHits = 0;

    /**
     * Persistent cache in front of the network, or null for none. A request whose bytes are all
     * stored is answered from it without queueing; fetched ranges are widened to the cache's
     * chunk boundaries and stored in it.
     *
     * @type {import('../../../platform/net/byte-cache.js').ByteCache|null}
     */
    byteCache = null;

    /**
     * Pending requests, one FIFO per priority. `offset` is -1 for a whole-file request.
     *
     * @type {Array<Array<{ url: string, cacheUrl: string, offset: number, length: number, credentials: string|null,
     * priority: number, stillWanted: (() => boolean)|null, view: boolean, resolve: Function, reject: Function }>>}
     * @private
     */
    _queues = [[], [], [], []];

    /**
     * Whole files of hosts that ignore the Range header, by cache URL. Each such response
     * carries the whole file, so keeping the first one costs less than downloading it again
     * for every later range.
     *
     * @type {Map<string, ArrayBuffer>}
     * @private
     */
    _wholeFiles = new Map();

    /** @type {number[]} - requests in flight per priority. @private */
    _activeByPriority = [0, 0, 0, 0];

    /**
     * @param {number} [maxConcurrent] - Cap on requests in flight.
     */
    constructor(maxConcurrent = MAX_CONCURRENT_FETCHES) {
        this.maxConcurrent = maxConcurrent;
    }

    /** @type {number} - requests waiting for a slot. */
    get queued() {
        return this._queues.reduce((n, q) => n + q.length, 0);
    }

    /**
     * Fetches a whole file.
     *
     * @param {string} url - The URL.
     * @param {MeshletFetchRequestOptions} [options] - Request options.
     * @returns {Promise<ArrayBuffer|null>} The bytes, or null when dropped.
     */
    fetchAll(url, options = {}) {
        return /** @type {Promise<ArrayBuffer|null>} */ (this._enqueue(url, -1, 0, options));
    }

    /**
     * Fetches one byte range of a file. A host that ignores the Range header and answers with
     * the whole file is sliced to the range. Rejects when the response ends before the range
     * does.
     *
     * @param {string} url - The URL.
     * @param {number} offset - First byte.
     * @param {number} length - Byte count.
     * @param {MeshletFetchRequestOptions} [options] - Request options.
     * @returns {Promise<ArrayBuffer|Uint8Array|null>} The range's bytes (a Uint8Array when
     * `options.view` is set), or null when dropped.
     */
    fetchRange(url, offset, length, options = {}) {
        return this._enqueue(url, offset, length, options);
    }

    /**
     * Drops every queued request (each resolves null) and the whole files kept for hosts that
     * ignore Range. Requests in flight complete.
     */
    clear() {
        const queues = this._queues;
        this._queues = [[], [], [], []];
        this._wholeFiles.clear();
        for (const queue of queues) {
            this.dropped += queue.length;
            for (const req of queue) req.resolve(null);
        }
    }

    /**
     * @param {string} url - The URL.
     * @param {number} offset - First byte, or -1 for the whole file.
     * @param {number} length - Byte count.
     * @param {MeshletFetchRequestOptions} options - Request options.
     * @returns {Promise<ArrayBuffer|Uint8Array|null>} The bytes, or null when dropped.
     * @private
     */
    _enqueue(url, offset, length, options) {
        const cacheUrl = options.cacheVersion ? `${url}#v=${options.cacheVersion}` : url;
        const view = offset >= 0 && !!options.view;
        const whole = offset >= 0 ? this._wholeFiles.get(cacheUrl) : undefined;
        if (whole) {
            return new Promise((resolve, reject) => {
                this._resolveRange({ url, offset, length, view, resolve, reject }, whole, 0, false);
            });
        }
        const cache = this.byteCache;
        if (cache && (offset < 0 ? cache.hasFile(cacheUrl) : cache.hasRange(cacheUrl, offset, length))) {
            const read = offset < 0 ?
                cache.getFile(cacheUrl).then(hit => hit?.data ?? null) :
                cache.getRange(cacheUrl, offset, length);
            return read.then((bytes) => {
                if (!bytes) return this._queue(url, offset, length, options);
                this.cacheHits++;
                return view ? new Uint8Array(bytes) : bytes;
            });
        }
        return this._queue(url, offset, length, options);
    }

    /**
     * @param {string} url - The URL.
     * @param {number} offset - First byte, or -1 for the whole file.
     * @param {number} length - Byte count.
     * @param {MeshletFetchRequestOptions} options - Request options.
     * @returns {Promise<ArrayBuffer|Uint8Array|null>} The bytes, or null when dropped.
     * @private
     */
    _queue(url, offset, length, { credentials = null, priority = FETCH_PRIORITY_FINE, stillWanted = null, cacheVersion = null, view = false }) {
        return new Promise((resolve, reject) => {
            const p = Math.min(Math.max(priority | 0, 0), this._queues.length - 1);
            const cacheUrl = cacheVersion ? `${url}#v=${cacheVersion}` : url;
            this._queues[p].push({ url, cacheUrl, offset, length, credentials, priority: p, stillWanted, view: offset >= 0 && view, resolve, reject });
            this._pump();
        });
    }

    /** @private */
    _pump() {
        while (this.active < this.maxConcurrent) {
            const p = this._next();
            if (p < 0) break;
            const queue = this._queues[p];
            const head = queue.shift();
            if (head.stillWanted && !head.stillWanted()) {
                this.dropped++;
                head.resolve(null);
                continue;
            }
            // queued before an earlier response showed the host ignores Range
            const whole = head.offset >= 0 ? this._wholeFiles.get(head.cacheUrl) : undefined;
            if (whole) {
                this._resolveRange(head, whole, 0, false);
                continue;
            }
            const batch = [head];
            if (head.offset >= 0) this._gather(head, batch, queue);
            this._dispatch(batch);
        }
    }

    /**
     * The priority to dispatch from next: the most urgent non-empty queue still under its share
     * of the cap; when every waiting priority is at its share, the most urgent one takes the
     * spare slot.
     *
     * @returns {number} The priority, or -1 when nothing is queued.
     * @private
     */
    _next() {
        let first = -1;
        for (let p = 0; p < this._queues.length; p++) {
            if (!this._queues[p].length) continue;
            if (first < 0) first = p;
            const share = Math.max(1, Math.ceil(this.maxConcurrent * PRIORITY_SHARE[p]));
            if (this._activeByPriority[p] < share) return p;
        }
        return first;
    }

    /**
     * Pulls queued ranges on the head request's URL into its batch while they fall within the
     * gap of the merged span and the span stays under the size cap. The queue is in dispatch
     * order, so what merges is what would have gone out next anyway.
     *
     * @param {object} head - The request being dispatched.
     * @param {object[]} batch - Its batch, extended in place.
     * @param {object[]} queue - The head's priority queue.
     * @private
     */
    _gather(head, batch, queue) {
        let start = head.offset;
        let end = head.offset + head.length;
        for (let i = 0; i < queue.length;) {
            const req = queue[i];
            if (req.offset < 0 || req.url !== head.url || req.cacheUrl !== head.cacheUrl || req.credentials !== head.credentials) {
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
        this.sentByPriority[head.priority]++;
        this._activeByPriority[head.priority]++;
        const ranged = head.offset >= 0;
        const cacheUrl = head.cacheUrl;
        const cache = this.byteCache?.accepts(cacheUrl) ? this.byteCache : null;
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
            if (cache) {
                // whole chunks only, so every fetched byte can be stored
                const aligned = cache.alignRange(cacheUrl, start, end - start);
                start = aligned.offset;
                end = aligned.offset + aligned.length;
            }
            init.headers = { Range: `bytes=${start}-${end - 1}` };
        }

        fetch(head.url, init).then((response) => {
            if (!response.ok && response.status !== 206) {
                throw Object.assign(new Error(`meshlet sidecar fetch failed (${response.status}): ${head.url}`), { status: response.status });
            }
            return response.arrayBuffer().then((buffer) => {
                if (!ranged) {
                    if (cache && response.status === 200) cache.putFile(cacheUrl, buffer);
                    head.resolve(buffer);
                    return;
                }
                // a 200 to a Range request is a host that ignored the header and sent the whole
                // file: the buffer then starts at byte 0, not at `start`
                const whole = response.status === 200;
                if (whole) {
                    Debug.warnOnce(`MeshletFetchScheduler: ${head.url} ignores Range requests; it is kept in memory whole and its ranges are served from there`);
                    this._wholeFiles.set(cacheUrl, buffer);
                }
                // store before resolving: a consumer may transfer the buffer
                if (cache) {
                    if (whole) {
                        cache.putFile(cacheUrl, buffer);
                    } else {
                        // a short response ran into the end of the file (the widened range may
                        // reach past it)
                        cache.putRange(cacheUrl, start, buffer, buffer.byteLength < end - start);
                    }
                }
                const exclusive = batch.length === 1 && !whole;
                for (const req of batch) this._resolveRange(req, buffer, whole ? 0 : start, exclusive);
            });
        }).catch((err) => {
            for (const req of batch) req.reject(err);
        }).finally(() => {
            this.active--;
            this._activeByPriority[head.priority]--;
            this._pump();
        });
    }

    /**
     * Settles a range request from a response that holds it.
     *
     * @param {{ url: string, offset: number, length: number, view: boolean, resolve: Function, reject: Function }} req - The request.
     * @param {ArrayBuffer} buffer - The response's bytes.
     * @param {number} base - File offset of the buffer's first byte.
     * @param {boolean} exclusive - The buffer belongs to this request alone, so a response of
     * exactly its range may be handed over as is.
     * @private
     */
    _resolveRange(req, buffer, base, exclusive) {
        const local = req.offset - base;
        if (local < 0 || local + req.length > buffer.byteLength) {
            req.reject(new Error(`meshlet sidecar fetch short: ${req.offset}-${req.offset + req.length - 1} requested, ${base}-${base + buffer.byteLength - 1} received: ${req.url}`));
        } else if (req.view) {
            req.resolve(new Uint8Array(buffer, local, req.length));
        } else if (exclusive && local === 0 && buffer.byteLength === req.length) {
            req.resolve(buffer);
        } else {
            req.resolve(buffer.slice(local, local + req.length));
        }
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
