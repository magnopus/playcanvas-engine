/**
 * A sequential numeric ID generator. Each instance maintains its own independent counter,
 * allowing separate ID spaces for different purposes.
 *
 * @ignore
 */
class NumericIds {
    /** @type {number} */
    _counter = 0;

    /**
     * Get the next unique ID.
     *
     * @returns {number} A unique sequential ID.
     */
    get() {
        return this._counter++;
    }

    /**
     * Reserve a contiguous block of IDs. Used where one object owns many identities that must
     * stay distinct - GPU picking of instanced geometry, where the id is per instance but comes
     * from the same space as every other pickable thing.
     *
     * @param {number} count - How many consecutive IDs to reserve.
     * @returns {number} The first ID of the block.
     */
    reserve(count) {
        const first = this._counter;
        this._counter += Math.max(count, 0);
        return first;
    }
}

export { NumericIds };
