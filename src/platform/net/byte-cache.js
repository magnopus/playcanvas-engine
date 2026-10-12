import { Debug } from '../../core/debug.js';

/**
 * @import { ByteStore } from './byte-store.js'
 */

/** Name of the stored index; chunk names never contain a '-', so it cannot collide with one. */
const INDEX_NAME = 'index-v1';

const INDEX_VERSION = 1;

/** Size an eviction pass shrinks the cache to, as a share of its limit, so each pass buys headroom. */
const EVICT_TARGET = 0.9;

/**
 * 53-bit string hash (cyrb53), rendered in base 36. Wide enough that a collision between two cached
 * URLs is negligible; the cache still refuses the second of two keys that do collide.
 *
 * @param {string} str - The string.
 * @returns {string} The hash.
 */
const hashKey = (str) => {
    let h1 = 0xdeadbeef;
    let h2 = 0x41c6ce57;
    for (let i = 0; i < str.length; i++) {
        const ch = str.charCodeAt(i);
        h1 = Math.imul(h1 ^ ch, 2654435761);
        h2 = Math.imul(h2 ^ ch, 1597334677);
    }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
};

/**
 * Default cache key: the URL itself, except for data and blob URLs, which are already local.
 *
 * @param {string} url - The URL.
 * @returns {string|null} The key, or null to leave the URL uncached.
 */
const defaultKey = url => (/^(?:data|blob):/i.test(url) ? null : url);

/**
 * @param {ArrayBuffer|ArrayBufferView} data - The bytes.
 * @returns {Uint8Array} A byte view of the same memory.
 */
const toBytes = data => (ArrayBuffer.isView(data) ?
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength) :
    new Uint8Array(data));

/**
 * One cached URL: which of its chunks are stored and, once a response has reached its end, its
 * total size.
 *
 * @ignore
 */
class ByteCacheEntry {
    /** @type {string} */
    key;

    /** @type {string} - hash of the key, the prefix of its chunks' names in the store. */
    name;

    /** @type {number} - total size in bytes, or -1 while unknown. */
    size = -1;

    /** @type {string} - the response's content type, or '' when unknown. */
    type = '';

    /** @type {Set<number>} - indices of the stored chunks. */
    chunks = new Set();

    /** @type {number} - bytes stored. */
    bytes = 0;

    /** @type {number} - when the entry was last read or written, for LRU eviction. */
    used = 0;

    /**
     * @param {string} key - The cache key.
     * @param {string} name - The hash of the key.
     */
    constructor(key, name) {
        this.key = key;
        this.name = name;
    }
}

/**
 * @typedef {object} ByteCacheOptions
 * @property {number} [chunkSize] - Granularity of storage in bytes. Every URL is stored as aligned
 * chunks of this size, so byte ranges fetched at different offsets on different visits still hit.
 * Defaults to 256 KiB. Changing it discards an existing cache.
 * @property {number} [maxBytes] - Size the cache is kept under; the least recently used URLs are
 * evicted beyond it. Defaults to 2 GiB. Lowered automatically when the store runs out of quota.
 * @property {number} [flushDelay] - Milliseconds the index is held in memory after a change before
 * it is written to the store. Defaults to 2000.
 * @property {(url: string) => (string|null)} [key] - Maps a URL to its cache key, or to null to
 * leave it uncached. Use it to restrict caching to some hosts, or to strip query parameters that
 * change between requests without changing the content (signatures, session tokens). Defaults to
 * the URL itself for every URL except data and blob URLs.
 */

/**
 * A persistent cache of fetched bytes, keyed by URL. It holds whole files and byte ranges alike:
 * every URL is stored as aligned chunks of {@link ByteCache#chunkSize} bytes, so a range is
 * served from the cache when every chunk it touches has been stored, by whichever earlier
 * requests stored them.
 *
 * The cache assumes the content behind a URL never changes - use versioned URLs, or a
 * {@link ByteCacheOptions#key} function that folds the version into the key. Least recently used
 * URLs are evicted once the cache grows past {@link ByteCache#maxBytes}.
 *
 * Storage is delegated to a {@link ByteStore}: an {@link OpfsByteStore} or
 * {@link CacheStorageByteStore} persists across sessions, a {@link MemoryByteStore} lasts for the
 * page's lifetime. If the store fails to open, the cache disables itself and every lookup misses.
 *
 * Attach a cache to {@link ResourceLoader#byteCache} to cache asset downloads, and to
 * {@link MeshletDirector#byteCache} to cache streamed meshlet geometry and textures. One cache may
 * serve both.
 *
 * @category Asset
 * @example
 * const store = pc.OpfsByteStore.isSupported() ?
 *     new pc.OpfsByteStore('my-app-cache') :
 *     new pc.CacheStorageByteStore('my-app-cache');
 * const cache = new pc.ByteCache(store, { maxBytes: 4 * 1024 * 1024 * 1024 });
 * app.loader.byteCache = cache;
 * director.byteCache = cache;
 *
 * // ask the browser not to evict the cache under storage pressure
 * navigator.storage.persist();
 */
class ByteCache {
    /**
     * The storage backend.
     *
     * @type {ByteStore}
     */
    store;

    /**
     * Granularity of storage in bytes.
     *
     * @type {number}
     */
    chunkSize;

    /**
     * Size the cache is kept under, in bytes.
     *
     * @type {number}
     */
    maxBytes;

    /**
     * Milliseconds the index is held in memory after a change before it is written to the store.
     *
     * @type {number}
     */
    flushDelay;

    /**
     * Resolves once the stored index has been loaded. Until then every lookup misses; writes made
     * before it resolves are applied once it does.
     *
     * @type {Promise<void>}
     */
    ready;

    /**
     * Number of lookups served from the cache.
     *
     * @type {number}
     */
    hits = 0;

    /**
     * Bytes served from the cache.
     *
     * @type {number}
     */
    bytesRead = 0;

    /**
     * Bytes written to the cache.
     *
     * @type {number}
     */
    bytesWritten = 0;

    /**
     * @type {(url: string) => (string|null)}
     * @private
     */
    _key;

    /**
     * @type {Map<string, ByteCacheEntry>}
     * @private
     */
    _entries = new Map();

    /**
     * Entries by name, to map stored chunk names back to entries and to refuse a second key whose
     * hash collides with a cached one.
     *
     * @type {Map<string, ByteCacheEntry>}
     * @private
     */
    _names = new Map();

    /**
     * Names of entries evicted since the index was last written, so merging another tab's index
     * does not bring them back.
     *
     * @type {Set<string>}
     * @private
     */
    _removed = new Set();

    /**
     * Chunk names with a write in flight.
     *
     * @type {Set<string>}
     * @private
     */
    _writing = new Set();

    /** @private */
    _totalBytes = 0;

    /** @private */
    _loaded = false;

    /** @private */
    _disabled = false;

    /** @private */
    _dirty = false;

    /**
     * @type {ReturnType<typeof setTimeout>|null}
     * @private
     */
    _flushTimer = null;

    /**
     * @type {Promise<void>}
     * @private
     */
    _flushing = Promise.resolve();

    /**
     * Creates a new ByteCache and starts loading its index from the store.
     *
     * @param {ByteStore} store - The storage backend.
     * @param {ByteCacheOptions} [options] - Options.
     */
    constructor(store, options = {}) {
        this.store = store;
        this.chunkSize = Math.max(1, (options.chunkSize ?? 256 * 1024) | 0);
        this.maxBytes = options.maxBytes ?? 2 * 1024 * 1024 * 1024;
        this.flushDelay = options.flushDelay ?? 2000;
        this._key = options.key ?? defaultKey;
        this.ready = this._load();
    }

    /**
     * Total bytes stored.
     *
     * @type {number}
     */
    get totalBytes() {
        return this._totalBytes;
    }

    /**
     * Number of URLs with at least one chunk stored.
     *
     * @type {number}
     */
    get entryCount() {
        return this._entries.size;
    }

    /**
     * Whether a URL is cacheable: the cache is usable and the key function accepts the URL.
     *
     * @param {string} url - The URL.
     * @returns {boolean} True when responses for the URL are stored.
     */
    accepts(url) {
        return !this._disabled && this._key(url) !== null;
    }

    /**
     * Whether every byte of a range is stored, so {@link ByteCache#getRange} would hit.
     *
     * @param {string} url - The URL.
     * @param {number} offset - First byte.
     * @param {number} length - Byte count.
     * @returns {boolean} True when the range is stored.
     */
    hasRange(url, offset, length) {
        const entry = this._lookup(url);
        if (!entry || length <= 0 || (entry.size >= 0 && offset + length > entry.size)) {
            return false;
        }
        const last = Math.floor((offset + length - 1) / this.chunkSize);
        for (let i = Math.floor(offset / this.chunkSize); i <= last; i++) {
            if (!entry.chunks.has(i)) return false;
        }
        return true;
    }

    /**
     * Whether a whole file is stored, so {@link ByteCache#getFile} would hit.
     *
     * @param {string} url - The URL.
     * @returns {boolean} True when the file is stored.
     */
    hasFile(url) {
        const entry = this._lookup(url);
        return !!entry && entry.size >= 0 && entry.chunks.size === Math.ceil(entry.size / this.chunkSize);
    }

    /**
     * Reads a byte range.
     *
     * @param {string} url - The URL.
     * @param {number} offset - First byte.
     * @param {number} length - Byte count.
     * @returns {Promise<ArrayBuffer|null>} A new buffer holding the range, or null when any of it
     * is not stored. Never rejects.
     */
    async getRange(url, offset, length) {
        if (!this.hasRange(url, offset, length)) return null;
        const entry = /** @type {ByteCacheEntry} */ (this._lookup(url));
        const chunkSize = this.chunkSize;
        const first = Math.floor(offset / chunkSize);
        const last = Math.floor((offset + length - 1) / chunkSize);

        const reads = [];
        for (let i = first; i <= last; i++) {
            reads.push(this.store.read(this._chunkName(entry, i)).catch(() => null));
        }
        const parts = await Promise.all(reads);

        // the entry may have been evicted or reset while the reads were in flight
        if (this._entries.get(entry.key) !== entry) return null;

        const out = new Uint8Array(length);
        const end = offset + length;
        for (let i = first; i <= last; i++) {
            const part = parts[i - first];
            if (!part || part.byteLength !== this._chunkLength(entry, i)) {
                // lost behind our back: another tab evicted it, or the browser cleared storage
                this._dropChunk(entry, i);
                return null;
            }
            const chunkStart = i * chunkSize;
            const from = Math.max(offset, chunkStart);
            const to = Math.min(end, chunkStart + part.byteLength);
            out.set(new Uint8Array(part, from - chunkStart, to - from), from - offset);
        }

        this._touch(entry);
        this.hits++;
        this.bytesRead += length;
        return out.buffer;
    }

    /**
     * Reads a whole file.
     *
     * @param {string} url - The URL.
     * @returns {Promise<{ data: ArrayBuffer, type: string }|null>} The file's bytes and content
     * type ('' when unknown), or null when it is not stored. Never rejects.
     */
    async getFile(url) {
        if (!this.hasFile(url)) return null;
        const entry = /** @type {ByteCacheEntry} */ (this._lookup(url));
        if (entry.size === 0) {
            this._touch(entry);
            this.hits++;
            return { data: new ArrayBuffer(0), type: entry.type };
        }
        const data = await this.getRange(url, 0, entry.size);
        return data ? { data, type: entry.type } : null;
    }

    /**
     * Stores the bytes of a response to a range request. Only the chunks the range covers
     * completely are stored; use {@link ByteCache#alignRange} when making the request so none of
     * it is wasted. The bytes are copied before this returns, so the caller is free to transfer
     * or modify them.
     *
     * @param {string} url - The URL.
     * @param {number} offset - Offset of the first byte.
     * @param {ArrayBuffer|ArrayBufferView} data - The bytes.
     * @param {boolean} [eof] - True when the bytes end at the end of the file (the server returned
     * fewer than were requested), which fixes the file's size.
     * @param {string} [type] - The response's content type.
     * @returns {Promise<void>} Resolves when the bytes are written. Never rejects.
     */
    putRange(url, offset, data, eof = false, type = '') {
        const key = this._disabled ? null : this._key(url);
        if (key === null) return Promise.resolve();
        const bytes = toBytes(data);
        const end = offset + bytes.byteLength;
        const known = this._loaded ? this._entries.get(key) : null;
        const size = eof ? end : (known?.size ?? -1);
        const chunkSize = this.chunkSize;

        // slice the chunks to store now: the caller may transfer or reuse the buffer once we return
        const slices = [];
        for (let i = Math.ceil(offset / chunkSize); i * chunkSize < end; i++) {
            const chunkStart = i * chunkSize;
            const chunkEnd = size >= 0 ? Math.min(chunkStart + chunkSize, size) : chunkStart + chunkSize;
            if (chunkEnd > end) break;
            if (known && known.size === size && known.chunks.has(i)) continue;
            slices.push({ index: i, bytes: bytes.slice(chunkStart - offset, chunkEnd - offset) });
        }
        if (!slices.length && !eof && !type) return Promise.resolve();

        return this.ready.then(() => this._store(key, slices, eof ? end : -1, type));
    }

    /**
     * Stores a whole file. The bytes are copied before this returns.
     *
     * @param {string} url - The URL.
     * @param {ArrayBuffer|ArrayBufferView} data - The file's bytes.
     * @param {string} [type] - The response's content type.
     * @returns {Promise<void>} Resolves when the bytes are written. Never rejects.
     */
    putFile(url, data, type = '') {
        return this.putRange(url, 0, data, true, type);
    }

    /**
     * Widens a byte range to whole chunks, clamped to the file's size where it is known. Fetching
     * the widened range lets {@link ByteCache#putRange} store all of it.
     *
     * @param {string} url - The URL.
     * @param {number} offset - First byte.
     * @param {number} length - Byte count.
     * @returns {{ offset: number, length: number }} The widened range.
     */
    alignRange(url, offset, length) {
        const chunkSize = this.chunkSize;
        const start = Math.floor(offset / chunkSize) * chunkSize;
        let end = Math.ceil((offset + length) / chunkSize) * chunkSize;
        const entry = this._lookup(url);
        if (entry && entry.size >= 0) end = Math.max(offset + length, Math.min(end, entry.size));
        return { offset: start, length: end - start };
    }

    /**
     * Removes one URL from the cache.
     *
     * @param {string} url - The URL.
     */
    remove(url) {
        const entry = this._lookup(url);
        if (entry) {
            this._removeEntry(entry);
            this._scheduleFlush();
        }
    }

    /**
     * Removes everything from the cache and its store.
     *
     * @returns {Promise<void>} Resolves when the store is empty.
     */
    async clear() {
        await this.ready;
        this._entries.clear();
        this._names.clear();
        this._removed.clear();
        this._totalBytes = 0;
        this._dirty = false;
        if (!this._disabled) await this.store.clear().catch(() => {});
    }

    /**
     * Writes the index to the store now instead of after {@link ByteCache#flushDelay}. Call it
     * when the page is being hidden so the most recent writes are found next session.
     *
     * @returns {Promise<void>} Resolves when the index is written. Never rejects.
     */
    flush() {
        if (this._flushTimer !== null) {
            clearTimeout(this._flushTimer);
            this._flushTimer = null;
        }
        this._flushing = this._flushing.then(() => this._writeIndex());
        return this._flushing;
    }

    /**
     * Flushes the index and stops using the store. Lookups miss and writes are dropped afterwards.
     *
     * @returns {Promise<void>} Resolves once the index is written.
     */
    destroy() {
        const flushed = this.ready.then(() => this.flush());
        this._disabled = true;
        return flushed;
    }

    /**
     * @param {string} url - The URL.
     * @returns {ByteCacheEntry|null} The URL's entry, or null when nothing of it is stored.
     * @private
     */
    _lookup(url) {
        if (!this._loaded || this._disabled) return null;
        const key = this._key(url);
        return key === null ? null : (this._entries.get(key) ?? null);
    }

    /**
     * @param {ByteCacheEntry} entry - The entry.
     * @param {number} index - Chunk index.
     * @returns {string} The chunk's name in the store.
     * @private
     */
    _chunkName(entry, index) {
        return `${entry.name}.${index}`;
    }

    /**
     * @param {ByteCacheEntry} entry - The entry.
     * @param {number} index - Chunk index.
     * @returns {number} The chunk's length: a whole chunk, except the last of a file of known size.
     * @private
     */
    _chunkLength(entry, index) {
        const start = index * this.chunkSize;
        return entry.size >= 0 ? Math.min(this.chunkSize, entry.size - start) : this.chunkSize;
    }

    /**
     * @param {ByteCacheEntry} entry - The entry.
     * @private
     */
    _touch(entry) {
        entry.used = Date.now();
        this._scheduleFlush();
    }

    /**
     * Records chunks sliced by {@link ByteCache#putRange} and writes them to the store.
     *
     * @param {string} key - The cache key.
     * @param {Array<{ index: number, bytes: Uint8Array }>} slices - The chunks.
     * @param {number} size - The file's size when the bytes reached its end, otherwise -1.
     * @param {string} type - The content type, or ''.
     * @returns {Promise<void>} Resolves when the chunks are written.
     * @private
     */
    async _store(key, slices, size, type) {
        if (this._disabled) return;

        let entry = this._entries.get(key);
        if (entry && size >= 0 && entry.size >= 0 && entry.size !== size) {
            // the content behind the key changed size: what is stored is stale
            this._removeEntry(entry);
            entry = undefined;
        }
        if (!entry) {
            const name = hashKey(key);
            if (this._names.has(name)) {
                Debug.warnOnce(`ByteCache: key hash collision, not caching ${key}`);
                return;
            }
            entry = new ByteCacheEntry(key, name);
            this._entries.set(key, entry);
            this._names.set(name, entry);
            this._removed.delete(name);
        }
        if (size >= 0 && entry.size < 0) {
            entry.size = size;
            // drop chunks stored before the size was known that the size shows were too long
            for (const index of entry.chunks) {
                if (index * this.chunkSize >= size) this._dropChunk(entry, index);
            }
        }
        if (type) entry.type = type;
        this._touch(entry);

        const writes = [];
        for (const { index, bytes } of slices) {
            const name = this._chunkName(entry, index);
            if (entry.chunks.has(index) || this._writing.has(name) || bytes.byteLength !== this._chunkLength(entry, index)) {
                continue;
            }
            this._writing.add(name);
            writes.push(this.store.write(name, bytes).then(() => {
                // only count it if the entry survived the write
                if (this._entries.get(key) === entry && !entry.chunks.has(index)) {
                    entry.chunks.add(index);
                    entry.bytes += bytes.byteLength;
                    this._totalBytes += bytes.byteLength;
                    this.bytesWritten += bytes.byteLength;
                } else {
                    this.store.remove(name).catch(() => {});
                }
            }, (err) => {
                this._onWriteError(err);
            }).finally(() => {
                this._writing.delete(name);
            }));
        }
        await Promise.all(writes);
        this._evict(this.maxBytes);
        this._scheduleFlush();
    }

    /**
     * @param {any} err - The error a store write rejected with.
     * @private
     */
    _onWriteError(err) {
        if (err?.name === 'QuotaExceededError') {
            // the quota is below the configured limit: settle under what actually fit
            this.maxBytes = Math.max(this.chunkSize, Math.floor(this._totalBytes * EVICT_TARGET));
            Debug.warnOnce(`ByteCache: storage quota reached, limiting the cache to ${this.maxBytes} bytes`);
            this._evict(this.maxBytes);
        } else {
            Debug.warnOnce('ByteCache: store write failed', err);
        }
    }

    /**
     * Evicts least recently used entries until the cache is under its target size.
     *
     * @param {number} limit - Size that triggers eviction.
     * @private
     */
    _evict(limit) {
        if (this._totalBytes <= limit) return;
        const target = limit * EVICT_TARGET;
        const entries = [...this._entries.values()].sort((a, b) => a.used - b.used);
        for (const entry of entries) {
            if (this._totalBytes <= target) break;
            this._removeEntry(entry);
        }
        this._scheduleFlush();
    }

    /**
     * @param {ByteCacheEntry} entry - The entry to remove, with its chunks.
     * @private
     */
    _removeEntry(entry) {
        this._entries.delete(entry.key);
        this._names.delete(entry.name);
        this._removed.add(entry.name);
        this._totalBytes -= entry.bytes;
        for (const index of entry.chunks) {
            this.store.remove(this._chunkName(entry, index)).catch(() => {});
        }
        entry.chunks.clear();
        entry.bytes = 0;
        this._dirty = true;
    }

    /**
     * @param {ByteCacheEntry} entry - The entry.
     * @param {number} index - The chunk to forget.
     * @private
     */
    _dropChunk(entry, index) {
        if (entry.chunks.delete(index)) {
            const length = this._chunkLength(entry, index);
            entry.bytes -= length;
            this._totalBytes -= length;
            this.store.remove(this._chunkName(entry, index)).catch(() => {});
            this._scheduleFlush();
        }
    }

    /** @private */
    _scheduleFlush() {
        this._dirty = true;
        if (this._flushTimer === null && !this._disabled) {
            this._flushTimer = setTimeout(() => this.flush(), this.flushDelay);
        }
    }

    /**
     * Loads the index, then deletes stored chunks it does not reference (left behind by a session
     * that ended before writing its index).
     *
     * @returns {Promise<void>} Resolves when the cache is usable, or has disabled itself.
     * @private
     */
    async _load() {
        try {
            await this.store.open();
            const index = await this._readIndex();
            if (index && index.chunkSize !== this.chunkSize) {
                await this.store.clear();
            } else if (index) {
                this._adopt(index.entries);
            }

            const names = await this.store.list();
            for (const name of names) {
                if (name !== INDEX_NAME && !this._references(name)) {
                    this.store.remove(name).catch(() => {});
                }
            }
        } catch (err) {
            Debug.warn('ByteCache: store unavailable, caching disabled', err);
            this._disabled = true;
        }
        this._loaded = true;
    }

    /**
     * @returns {Promise<{ chunkSize: number, entries: Array<object> }|null>} The stored index, or
     * null when there is none or it is unreadable.
     * @private
     */
    async _readIndex() {
        const data = await this.store.read(INDEX_NAME);
        if (!data) return null;
        try {
            const index = JSON.parse(new TextDecoder().decode(data));
            return index?.version === INDEX_VERSION && Array.isArray(index.entries) ? index : null;
        } catch (err) {
            return null;
        }
    }

    /**
     * Adds stored entries this cache does not know, and merges the chunks of those it does. The
     * stored index may hold another tab's writes.
     *
     * @param {Array<{ k: string, n: string, s: number, t: string, c: number[], u: number }>} records - Serialized entries.
     * @private
     */
    _adopt(records) {
        for (const record of records) {
            if (typeof record?.k !== 'string' || this._removed.has(record.n)) continue;
            let entry = this._entries.get(record.k);
            if (!entry) {
                if (this._names.has(record.n)) continue;
                entry = new ByteCacheEntry(record.k, record.n);
                entry.size = record.s;
                entry.type = record.t ?? '';
                this._entries.set(entry.key, entry);
                this._names.set(entry.name, entry);
            } else if (entry.size !== record.s && entry.size >= 0 && record.s >= 0) {
                continue;
            }
            entry.used = Math.max(entry.used, record.u ?? 0);
            for (const index of record.c) {
                if (!entry.chunks.has(index)) {
                    entry.chunks.add(index);
                    const length = this._chunkLength(entry, index);
                    entry.bytes += length;
                    this._totalBytes += length;
                }
            }
        }
    }

    /**
     * @param {string} name - A name in the store.
     * @returns {boolean} True when it is a chunk the index references.
     * @private
     */
    _references(name) {
        const dot = name.lastIndexOf('.');
        const entry = dot < 0 ? undefined : this._names.get(name.slice(0, dot));
        return !!entry && entry.chunks.has(Number(name.slice(dot + 1)));
    }

    /**
     * Merges the stored index (another tab may have written it) and writes the result.
     *
     * @returns {Promise<void>} Resolves when written. Never rejects.
     * @private
     */
    async _writeIndex() {
        if (!this._dirty || !this._loaded || this._disabled) return;
        this._dirty = false;
        try {
            const stored = await this._readIndex();
            if (stored && stored.chunkSize === this.chunkSize) this._adopt(stored.entries);
            this._evict(this.maxBytes);
            this._dirty = false;

            const entries = [];
            for (const entry of this._entries.values()) {
                if (entry.chunks.size) {
                    entries.push({ k: entry.key, n: entry.name, s: entry.size, t: entry.type, c: [...entry.chunks], u: entry.used });
                }
            }
            const json = JSON.stringify({ version: INDEX_VERSION, chunkSize: this.chunkSize, entries });
            await this.store.write(INDEX_NAME, new TextEncoder().encode(json));
            this._removed.clear();
        } catch (err) {
            Debug.warnOnce('ByteCache: failed to write index', err);
            this._dirty = true;
        }
    }
}

export { ByteCache };
