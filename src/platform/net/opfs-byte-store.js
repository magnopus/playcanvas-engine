import { Debug } from '../../core/debug.js';
import { ByteStore } from './byte-store.js';

/** Bytes handed to the file writer per step of {@link OpfsByteStore#writeLarge}. */
const LARGE_WRITE_WINDOW = 8 * 1024 * 1024;

/** Size at which the worker starts a new pack file. */
const PACK_BYTES = 1024 * 1024 * 1024;

/**
 * The store's worker. Stringified into a blob, so it must not reference anything outside itself.
 *
 * Values are appended as records to a few large pack files, whose synchronous access handles are
 * opened once, when the store opens, and kept open. Opening a handle costs milliseconds (and was
 * seen never to settle under load), while reading or writing through an open one runs at
 * gigabytes a second, so every read and write after the open is one synchronous call at a known
 * offset. Where each value lives is held in memory, rebuilt at open by scanning the records'
 * headers.
 *
 * A record is a 16-byte header (magic, flags, name length, 64-bit data length), the name, then the
 * data. Removing a value appends a tombstone record. Packs only ever grow at their end: a crash
 * can leave a record cut short at the end of a pack, which the scan truncates away. Space held by
 * replaced or removed values is reclaimed by compacting the oldest pack once enough of the total
 * is dead: its live values are appended anew and the pack is deleted. Compacting only the oldest
 * pack keeps a tombstone from being dropped while an older record it cancels still exists.
 *
 * Operations run one at a time, in the order they arrived; compaction runs between them in steps.
 *
 * @param {object} scope - The worker's global scope.
 * @param {number} packBytes - Size at which a new pack is started.
 * @param {number} [compactMinDead] - Dead bytes below which nothing is compacted.
 * @ignore
 */
function OpfsPackWorker(scope, packBytes, compactMinDead = 256 * 1024 * 1024) {
    const HEADER = 16;
    const MAGIC = 0x31534b50; // 'PKS1'
    const TOMBSTONE = 1;
    const COMPACT_STEP_BYTES = 64 * 1024 * 1024;
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();

    let root = null;
    let dir = null;
    let directoryName = '';
    /** @type {{ n: number, file: string, handle: any, size: number, live: number, names: Set<string> }[]} */
    let packs = [];
    /** @type {Map<string, { pack: any, offset: number, length: number }>} */
    const entries = new Map();
    let compacting = false;

    // every operation and compaction step runs after the previous one has finished
    let chain = Promise.resolve();
    const serial = (fn) => {
        const result = chain.then(fn);
        chain = result.catch(() => {});
        return result;
    };

    const readFully = (handle, view, at) => {
        let done = 0;
        while (done < view.byteLength) {
            const n = handle.read(view.subarray(done), { at: at + done });
            if (n <= 0) return false;
            done += n;
        }
        return true;
    };

    const writeFully = (handle, view, at) => {
        let done = 0;
        while (done < view.byteLength) {
            const n = handle.write(view.subarray(done), { at: at + done });
            if (n <= 0) throw new Error('OpfsByteStore: short write');
            done += n;
        }
    };

    const forget = (name) => {
        const entry = entries.get(name);
        if (!entry) return;
        entry.pack.live -= entry.length;
        entry.pack.names.delete(name);
        entries.delete(name);
    };

    const remember = (name, pack, offset, length) => {
        forget(name);
        entries.set(name, { pack, offset, length });
        pack.live += length;
        pack.names.add(name);
    };

    const openPack = async (n) => {
        const file = `pack-${n}.dat`;
        const fileHandle = await dir.getFileHandle(file, { create: true });
        const handle = await fileHandle.createSyncAccessHandle();
        const pack = { n, file, handle, size: handle.getSize(), live: 0, names: new Set() };
        packs.push(pack);
        return pack;
    };

    const scan = (pack) => {
        const head = new Uint8Array(HEADER);
        const view = new DataView(head.buffer);
        let pos = 0;
        while (pos + HEADER <= pack.size) {
            if (!readFully(pack.handle, head, pos) || view.getUint32(0, true) !== MAGIC) break;
            const flags = view.getUint16(4, true);
            const nameLength = view.getUint16(6, true);
            const length = view.getUint32(8, true) + view.getUint32(12, true) * 4294967296;
            const end = pos + HEADER + nameLength + length;
            if (end > pack.size) break;
            const nameBytes = new Uint8Array(nameLength);
            if (!readFully(pack.handle, nameBytes, pos + HEADER)) break;
            const name = decoder.decode(nameBytes);
            if (flags & TOMBSTONE) {
                forget(name);
            } else {
                remember(name, pack, pos + HEADER + nameLength, length);
            }
            pos = end;
        }
        // a record cut short, by a crash or a failed write
        if (pos < pack.size) {
            pack.handle.truncate(pos);
            pack.size = pos;
        }
    };

    const append = async (name, data, flags) => {
        const nameBytes = encoder.encode(name);
        const length = data ? data.byteLength : 0;
        const recordBytes = HEADER + nameBytes.length + length;
        let pack = packs[packs.length - 1];
        if (pack.size > 0 && pack.size + recordBytes > packBytes) pack = await openPack(pack.n + 1);
        const head = new Uint8Array(HEADER + nameBytes.length);
        const view = new DataView(head.buffer);
        view.setUint32(0, MAGIC, true);
        view.setUint16(4, flags, true);
        view.setUint16(6, nameBytes.length, true);
        view.setUint32(8, length % 4294967296, true);
        view.setUint32(12, Math.floor(length / 4294967296), true);
        head.set(nameBytes, HEADER);
        const at = pack.size;
        try {
            writeFully(pack.handle, head, at);
            if (length) writeFully(pack.handle, data, at + head.byteLength);
        } catch (err) {
            pack.handle.truncate(at);
            throw err;
        }
        pack.size = at + recordBytes;
        return { pack, offset: at + head.byteLength, length };
    };

    const compactStep = async () => {
        const oldest = packs[0];
        if (packs.length < 2) {
            compacting = false;
            return;
        }
        let moved = 0;
        for (const name of oldest.names) {
            const entry = entries.get(name);
            const data = new Uint8Array(entry.length);
            if (!readFully(oldest.handle, data, entry.offset)) {
                forget(name);
                continue;
            }
            const written = await append(name, data, 0);
            remember(name, written.pack, written.offset, written.length);
            moved += entry.length;
            if (moved >= COMPACT_STEP_BYTES) {
                // let queued operations run before the next step
                setTimeout(() => serial(compactStep).catch(() => {
                    compacting = false;
                }), 0);
                return;
            }
        }
        packs.shift();
        oldest.handle.close();
        await dir.removeEntry(oldest.file).catch(() => {});
        compacting = false;
        maybeCompact();
    };

    const maybeCompact = () => {
        if (compacting || packs.length < 2) return;
        let total = 0;
        let live = 0;
        for (const pack of packs) {
            total += pack.size;
            live += pack.live;
        }
        const dead = total - live;
        if (dead < Math.max(compactMinDead, total * 0.25)) return;
        compacting = true;
        setTimeout(() => serial(compactStep).catch(() => {
            compacting = false;
        }), 0);
    };

    // deletes the given entries a few at a time, outside the operation chain: none is a pack
    const removeStray = (names) => {
        const lane = async () => {
            while (names.length) {
                const name = names.pop();
                await dir.removeEntry(name, { recursive: true }).catch(() => {}); // eslint-disable-line no-await-in-loop
            }
        };
        for (let i = 0; i < 4; i++) lane();
    };

    const ops = {
        open: async (msg) => {
            if (typeof FileSystemFileHandle === 'undefined' || !('createSyncAccessHandle' in FileSystemFileHandle.prototype)) {
                throw new Error('OPFS synchronous access handles are not available in workers');
            }
            directoryName = msg.directoryName;
            root = await navigator.storage.getDirectory();
            dir = await root.getDirectoryHandle(directoryName, { create: true });
            const numbers = [];
            const stray = [];
            for await (const [name, handle] of dir.entries()) {
                const match = /^pack-(\d+)\.dat$/.exec(name);
                if (match && handle.kind === 'file') {
                    numbers.push(Number(match[1]));
                } else {
                    stray.push(name);
                }
            }
            numbers.sort((a, b) => a - b);
            for (const n of numbers) scan(await openPack(n)); // eslint-disable-line no-await-in-loop
            if (!packs.length) await openPack(0);
            // anything else in the directory is not ours: one-file-per-value data left by the
            // main-thread fallback, say. Deleting costs milliseconds a file, so it runs alongside
            // the operations rather than holding up the open.
            removeStray(stray);
            maybeCompact();
            return null;
        },
        read: (msg) => {
            const entry = entries.get(msg.name);
            if (!entry) return null;
            const buffer = new ArrayBuffer(entry.length);
            return readFully(entry.pack.handle, new Uint8Array(buffer), entry.offset) ? buffer : null;
        },
        readRange: (msg) => {
            const entry = entries.get(msg.name);
            if (!entry || msg.offset + msg.length > entry.length) return null;
            const buffer = new ArrayBuffer(msg.length);
            return readFully(entry.pack.handle, new Uint8Array(buffer), entry.offset + msg.offset) ? buffer : null;
        },
        write: async (msg) => {
            const written = await append(msg.name, msg.data, 0);
            remember(msg.name, written.pack, written.offset, written.length);
            maybeCompact();
            return null;
        },
        remove: async (msg) => {
            if (!entries.has(msg.name)) return null;
            forget(msg.name);
            await append(msg.name, null, TOMBSTONE);
            maybeCompact();
            return null;
        },
        list: () => [...entries.keys()],
        clear: async () => {
            for (const pack of packs) pack.handle.close();
            packs = [];
            entries.clear();
            await root.removeEntry(directoryName, { recursive: true }).catch(() => {});
            dir = await root.getDirectoryHandle(directoryName, { create: true });
            await openPack(0);
            return null;
        }
    };

    scope.onmessage = (event) => {
        const msg = event.data;
        if (msg.returnId) {
            // a lent buffer: the page waits for it, so it is copied here and handed straight back
            // rather than after the writes queued ahead of this one
            const lent = msg.data;
            msg.data = new Uint8Array(lent.slice(0));
            scope.postMessage({ id: msg.returnId, result: lent }, [lent]);
        }
        serial(() => ops[msg.op](msg)).then((result) => {
            scope.postMessage({ id: msg.id, result }, result instanceof ArrayBuffer ? [result] : []);
        }, (error) => {
            scope.postMessage({ id: msg.id, error: { name: error?.name ?? 'Error', message: String(error?.message ?? error) } });
        });
    };
}

/**
 * A {@link ByteStore} in the Origin Private File System, inside a directory of the origin's
 * private file system. Persistent across sessions and suited to large amounts of data, but
 * subject to the browser's storage quota and eviction - call `navigator.storage.persist()` to ask
 * the browser not to evict it under storage pressure.
 *
 * By default a worker does the file I/O through synchronous access handles, off the main thread.
 * It appends the values to a few large pack files whose handles it keeps open, as OPFS charges
 * milliseconds per file opened but moves the bytes of an open file at gigabytes a second. The
 * pack files are held exclusively, so only one page at a time can use a directory: in another,
 * {@link OpfsByteStore#open} rejects and a {@link ByteCache} over the store disables itself.
 *
 * Where a worker cannot be started, or does not support synchronous access handles, the store
 * keeps one file per value and does the I/O on the main thread through
 * `FileSystemFileHandle.createWritable`, which is much slower for many small values. The two
 * layouts do not share data: opening a directory in one discards what the other stored there.
 *
 * Requires `FileSystemFileHandle.createWritable` (see {@link OpfsByteStore.isSupported}); where
 * it is missing, use a {@link CacheStorageByteStore} instead.
 *
 * @category Asset
 */
class OpfsByteStore extends ByteStore {
    /**
     * Name of the directory, at the root of the origin's private file system, the values are
     * stored in.
     *
     * @type {string}
     */
    directoryName;

    /**
     * Whether {@link OpfsByteStore#open} tries to start a worker for the file I/O.
     *
     * @type {boolean}
     */
    useWorker;

    /**
     * @type {FileSystemDirectoryHandle|null}
     * @private
     */
    _root = null;

    /**
     * @type {FileSystemDirectoryHandle|null}
     * @private
     */
    _dir = null;

    /**
     * @type {Worker|null}
     * @private
     */
    _worker = null;

    /**
     * Calls to the worker awaiting a reply, by id.
     *
     * @type {Map<number, { resolve: (value: any) => void, reject: (reason: any) => void }>}
     * @private
     */
    _calls = new Map();

    /** @private */
    _nextCallId = 1;

    /**
     * Creates a new OpfsByteStore.
     *
     * @param {string} [directoryName] - Name of the directory the values are stored in. Defaults
     * to 'playcanvas-byte-cache'. Stores with different names are independent.
     * @param {object} [options] - Options.
     * @param {boolean} [options.worker] - Whether to do the file I/O in a worker. Defaults to true.
     */
    constructor(directoryName = 'playcanvas-byte-cache', options = {}) {
        super();
        this.directoryName = directoryName;
        this.useWorker = options.worker ?? true;
    }

    /**
     * Whether the browser supports this store.
     *
     * @returns {boolean} True when the Origin Private File System is writable from this context.
     */
    static isSupported() {
        return typeof navigator !== 'undefined' && !!navigator.storage?.getDirectory &&
            typeof FileSystemFileHandle !== 'undefined' && 'createWritable' in FileSystemFileHandle.prototype;
    }

    /**
     * Whether the file I/O runs in a worker. Known once {@link OpfsByteStore#open} resolves.
     *
     * @type {boolean}
     */
    get usingWorker() {
        return !!this._worker;
    }

    async open() {
        if (this.useWorker && typeof Worker !== 'undefined' && typeof Blob !== 'undefined' &&
            typeof URL !== 'undefined' && typeof URL.createObjectURL === 'function') {
            try {
                await this._startWorker();
                return;
            } catch (err) {
                this._stopWorker(err);
                // another page holds the pack files: falling back would write the other layout
                // into the same directory, so this page does without
                if (err?.name === 'NoModificationAllowedError') throw err;
                Debug.warnOnce(`OpfsByteStore: no worker for the file I/O, using the main thread (${err?.message ?? err})`);
            }
        }
        this._root = await navigator.storage.getDirectory();
        this._dir = await this._root.getDirectoryHandle(this.directoryName, { create: true });
    }

    async read(name) {
        if (this._worker) return this._call({ op: 'read', name }).catch(() => null);
        try {
            const handle = await this._dir.getFileHandle(name);
            const file = await handle.getFile();
            return await file.arrayBuffer();
        } catch (err) {
            return null;
        }
    }

    async readRange(name, offset, length) {
        if (this._worker) return this._call({ op: 'readRange', name, offset, length }).catch(() => null);
        try {
            const handle = await this._dir.getFileHandle(name);
            const file = await handle.getFile();
            if (file.size < offset + length) return null;
            return await file.slice(offset, offset + length).arrayBuffer();
        } catch (err) {
            return null;
        }
    }

    writeLent(name, buffer) {
        if (!this._worker) return super.writeLent(name, buffer);
        // the copy is taken in the worker, off the main thread
        const returnId = this._nextCallId++;
        const returned = new Promise((resolve, reject) => {
            this._calls.set(returnId, { resolve, reject });
        });
        const stored = this._call({ op: 'write', name, data: buffer, returnId }, [buffer]).then(() => {});
        return { returned, stored };
    }

    async writeLarge(name, data) {
        // the worker writes a value in one step off the main thread, from a copy taken now: the
        // caller keeps using its buffer
        if (this._worker) return this._writeInWorker(name, data.slice());
        const size = data.byteLength;
        const handle = await this._dir.getFileHandle(name, { create: true });
        const writable = await handle.createWritable();
        try {
            for (let offset = 0; offset < size; offset += LARGE_WRITE_WINDOW) {
                // the caller's buffer was transferred away mid-write: do not store a truncated value
                if (data.byteLength !== size) throw new Error('OpfsByteStore: buffer detached while writing');
                await writable.write(data.subarray(offset, Math.min(offset + LARGE_WRITE_WINDOW, size))); // eslint-disable-line no-await-in-loop
            }
            await writable.close();
        } catch (err) {
            await writable.abort().catch(() => {});
            throw err;
        }
    }

    async write(name, data) {
        // the caller hands the bytes over (see ByteStore#write): a view of a whole buffer is
        // transferred as it is, anything else is copied first
        if (this._worker) {
            const whole = data.byteOffset === 0 && data.byteLength === data.buffer.byteLength && data.buffer instanceof ArrayBuffer;
            return this._writeInWorker(name, whole ? data : data.slice());
        }
        const handle = await this._dir.getFileHandle(name, { create: true });
        const writable = await handle.createWritable();
        try {
            await writable.write(data);
            await writable.close();
        } catch (err) {
            await writable.abort().catch(() => {});
            throw err;
        }
    }

    async remove(name) {
        if (this._worker) {
            await this._call({ op: 'remove', name }).catch(() => {});
            return;
        }
        try {
            await this._dir.removeEntry(name);
        } catch (err) {
            // already gone
        }
    }

    async list() {
        if (this._worker) return this._call({ op: 'list' });
        const names = [];
        // @ts-ignore - async iteration of directory handles is missing from older DOM typings
        for await (const name of this._dir.keys()) {
            names.push(name);
        }
        return names;
    }

    async clear() {
        if (this._worker) {
            await this._call({ op: 'clear' });
            return;
        }
        await this._root.removeEntry(this.directoryName, { recursive: true }).catch(() => {});
        this._dir = await this._root.getDirectoryHandle(this.directoryName, { create: true });
    }

    /**
     * Stops the store's worker, if it has one, which releases its files to other pages.
     * Operations still in flight fail, and the store cannot be used afterwards.
     */
    destroy() {
        this._stopWorker(new Error('OpfsByteStore: destroyed'));
        this._dir = null;
        this._root = null;
    }

    /**
     * Starts the worker and opens the directory in it.
     *
     * @returns {Promise<void>} Resolves when the worker is ready; rejects when it cannot be used.
     * @private
     */
    async _startWorker() {
        const code = `(${OpfsPackWorker.toString()})(self, ${PACK_BYTES});\n`;
        const url = URL.createObjectURL(new Blob([code], { type: 'application/javascript' }));
        try {
            const worker = this._worker = new Worker(url);
            worker.onmessage = (event) => {
                const { id, result, error } = event.data;
                const call = this._calls.get(id);
                if (!call) return;
                this._calls.delete(id);
                if (error) {
                    // keep the name: the cache reacts to a QuotaExceededError
                    const err = new Error(error.message);
                    err.name = error.name;
                    call.reject(err);
                } else {
                    call.resolve(result);
                }
            };
            // a worker that fails to load or dies takes its calls with it
            worker.onerror = (event) => {
                event.preventDefault?.();
                this._stopWorker(new Error(`OpfsByteStore: worker failed (${event.message ?? 'unknown error'})`));
            };
            await this._call({ op: 'open', directoryName: this.directoryName });
        } finally {
            URL.revokeObjectURL(url);
        }
    }

    /**
     * Terminates the worker and fails its pending calls.
     *
     * @param {Error} reason - Why the calls fail.
     * @private
     */
    _stopWorker(reason) {
        this._worker?.terminate();
        this._worker = null;
        const calls = [...this._calls.values()];
        this._calls.clear();
        for (const call of calls) call.reject(reason);
    }

    /**
     * Writes a value in the worker, transferring its buffer: no copy is made, and the buffer is
     * detached once this returns.
     *
     * @param {string} name - The value's name.
     * @param {Uint8Array} data - The bytes, a view of the whole of a buffer the store now owns.
     * @returns {Promise<void>} Resolves when the value is stored.
     * @private
     */
    async _writeInWorker(name, data) {
        await this._call({ op: 'write', name, data }, [data.buffer]);
    }

    /**
     * Sends an operation to the worker.
     *
     * @param {object} message - The operation.
     * @param {Transferable[]} [transfer] - Buffers to transfer with it.
     * @returns {Promise<any>} The operation's result.
     * @private
     */
    _call(message, transfer = []) {
        const worker = this._worker;
        if (!worker) return Promise.reject(new Error('OpfsByteStore: the worker is not running'));
        return new Promise((resolve, reject) => {
            const id = this._nextCallId++;
            this._calls.set(id, { resolve, reject });
            worker.postMessage({ ...message, id }, transfer);
        });
    }
}

export { OpfsByteStore, OpfsPackWorker };
