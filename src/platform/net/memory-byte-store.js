import { ByteStore } from './byte-store.js';

/**
 * A {@link ByteStore} held in memory. Its contents last as long as the page, so a
 * {@link ByteCache} over it saves refetching within one session only (leaving and re-entering a
 * scene, for example). Mostly useful for tests and as a fallback where no persistent store is
 * available.
 *
 * @category Asset
 */
class MemoryByteStore extends ByteStore {
    /**
     * @type {Map<string, Uint8Array>}
     * @private
     */
    _values = new Map();

    read(name) {
        const value = this._values.get(name);
        // hand out a copy so a caller transferring or modifying it cannot corrupt the store
        return Promise.resolve(value ? value.slice().buffer : null);
    }

    write(name, data) {
        this._values.set(name, data);
        return Promise.resolve();
    }

    remove(name) {
        this._values.delete(name);
        return Promise.resolve();
    }

    list() {
        return Promise.resolve([...this._values.keys()]);
    }

    clear() {
        this._values.clear();
        return Promise.resolve();
    }
}

export { MemoryByteStore };
