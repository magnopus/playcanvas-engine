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
     * @param {Uint8Array} data - The bytes. The store may keep a reference, or transfer the
     * buffer to a worker (detaching it), so the caller must not use them afterwards.
     * @returns {Promise<void>} Resolves when the value is stored.
     */
    write(name, data) {
        return Promise.resolve();
    }

    /**
     * Reads part of a value.
     *
     * @param {string} name - The value's name.
     * @param {number} offset - First byte.
     * @param {number} length - Byte count.
     * @returns {Promise<ArrayBuffer|null>} The bytes, or null when the name is not stored or holds
     * fewer than `offset + length` bytes. The base implementation reads the whole value; stores
     * that can read a range cheaply override it.
     */
    readRange(name, offset, length) {
        return this.read(name).then((data) => {
            return data && data.byteLength >= offset + length ? data.slice(offset, offset + length) : null;
        });
    }

    /**
     * Writes a very large value. The store should write it in steps that yield to the event loop
     * rather than copy it at once, and the value only counts as stored when the promise resolves.
     * The caller must leave `data` attached (not transfer its buffer) until then. The base
     * implementation copies the bytes and calls {@link ByteStore#write}.
     *
     * @param {string} name - The value's name.
     * @param {Uint8Array} data - The bytes.
     * @returns {Promise<void>} Resolves when the value is stored.
     */
    writeLarge(name, data) {
        return this.write(name, data.slice());
    }

    /**
     * Writes a very large value from a buffer the caller lends to the store, for a caller that
     * needs the bytes itself straight afterwards. The store may transfer the buffer (to a worker
     * that copies it there, sparing the main thread the copy), so the caller must use the buffer
     * `returned` resolves with in place of the one it passed. The base implementation hands the
     * same buffer back at once and writes it with {@link ByteStore#writeLarge}.
     *
     * @param {string} name - The value's name.
     * @param {ArrayBuffer} buffer - The bytes.
     * @returns {{ returned: Promise<ArrayBuffer>, stored: Promise<void> }} `returned` resolves with
     * the bytes for the caller, and rejects if the store lost them; `stored` resolves when the
     * value is stored.
     */
    writeLent(name, buffer) {
        return { returned: Promise.resolve(buffer), stored: this.writeLarge(name, new Uint8Array(buffer)) };
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
