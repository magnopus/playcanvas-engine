import { ByteStore } from './byte-store.js';

/** Bytes handed to the file writer per step of {@link OpfsByteStore#writeLarge}. */
const LARGE_WRITE_WINDOW = 8 * 1024 * 1024;

/**
 * A {@link ByteStore} in the Origin Private File System: one file per value inside a directory of
 * the origin's private file system. Persistent across sessions and suited to large values, but
 * subject to the browser's storage quota and eviction - call `navigator.storage.persist()` to ask
 * the browser not to evict it under storage pressure.
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
     * Creates a new OpfsByteStore.
     *
     * @param {string} [directoryName] - Name of the directory the values are stored in. Defaults
     * to 'playcanvas-byte-cache'. Stores with different names are independent.
     */
    constructor(directoryName = 'playcanvas-byte-cache') {
        super();
        this.directoryName = directoryName;
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

    async open() {
        this._root = await navigator.storage.getDirectory();
        this._dir = await this._root.getDirectoryHandle(this.directoryName, { create: true });
    }

    async read(name) {
        try {
            const handle = await this._dir.getFileHandle(name);
            const file = await handle.getFile();
            return await file.arrayBuffer();
        } catch (err) {
            return null;
        }
    }

    async readRange(name, offset, length) {
        try {
            const handle = await this._dir.getFileHandle(name);
            const file = await handle.getFile();
            if (file.size < offset + length) return null;
            return await file.slice(offset, offset + length).arrayBuffer();
        } catch (err) {
            return null;
        }
    }

    async writeLarge(name, data) {
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
        try {
            await this._dir.removeEntry(name);
        } catch (err) {
            // already gone
        }
    }

    async list() {
        const names = [];
        // @ts-ignore - async iteration of directory handles is missing from older DOM typings
        for await (const name of this._dir.keys()) {
            names.push(name);
        }
        return names;
    }

    async clear() {
        await this._root.removeEntry(this.directoryName, { recursive: true }).catch(() => {});
        this._dir = await this._root.getDirectoryHandle(this.directoryName, { create: true });
    }
}

export { OpfsByteStore };
