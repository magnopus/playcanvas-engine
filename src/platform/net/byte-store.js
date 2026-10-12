/**
 * A flat, named blob store - the storage backend of a {@link ByteCache}. Names are short ASCII
 * strings chosen by the cache; values are raw bytes. Every method is asynchronous and a failure
 * to find a name is reported as a null read, never a rejection.
 *
 * This base class stores nothing: every read misses and every write is discarded. Subclass it to
 * add a backend.
 *
 * @category Asset
 */
class ByteStore {
    /**
     * Prepares the store for use. Called once by the owning cache before any other method.
     *
     * @returns {Promise<void>} Resolves when the store is ready; rejects when it cannot be used.
     */
    open() {
        return Promise.resolve();
    }

    /**
     * Reads a value.
     *
     * @param {string} name - The value's name.
     * @returns {Promise<ArrayBuffer|null>} The bytes, or null when the name is not stored.
     */
    read(name) {
        return Promise.resolve(null);
    }

    /**
     * Writes a value, replacing any value already stored under the name.
     *
     * @param {string} name - The value's name.
     * @param {Uint8Array} data - The bytes. The store may keep a reference, so the caller must
     * not modify them afterwards.
     * @returns {Promise<void>} Resolves when the value is stored.
     */
    write(name, data) {
        return Promise.resolve();
    }

    /**
     * Removes a value. Removing a name that is not stored is not an error.
     *
     * @param {string} name - The value's name.
     * @returns {Promise<void>} Resolves when the value is removed.
     */
    remove(name) {
        return Promise.resolve();
    }

    /**
     * Lists the names of every stored value.
     *
     * @returns {Promise<string[]>} The names.
     */
    list() {
        return Promise.resolve([]);
    }

    /**
     * Removes every stored value.
     *
     * @returns {Promise<void>} Resolves when the store is empty.
     */
    clear() {
        return Promise.resolve();
    }
}

export { ByteStore };
