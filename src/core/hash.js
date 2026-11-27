/**
 * Calculates simple hash value of a string. Designed for performance, not perfect.
 *
 * @param {string} str - String.
 * @returns {number} Hash value.
 */
function hashCode(str) {
    if (str === null || str === undefined) {
        return 0;
    }
    let hash = 0;
    for (let i = 0, len = str.length; i < len; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        // Convert to 32bit integer
        hash |= 0;
    }
    return hash;
}

/**
 * Calculates simple 32bit hash value of an array of 32bit integer numbers. Designed for
 * performance, but provides good distribution with small number of collisions. Based on
 * FNV-1a non-cryptographic hash function.
 *
 * @param {number[]|Uint32Array} array - Array of 32bit integer numbers to hash.
 * @returns {number} 32bit unsigned integer hash value.
 */
function hash32Fnv1a(array) {
    const prime = 16777619;
    let hash = 2166136261;

    for (let i = 0; i < array.length; i++) {
        hash ^= array[i];

        // a 32bit integer multiply, as the hash function specifies - a floating point one exceeds
        // the 53bit mantissa and so is not the 32bit product, and is slower
        hash = Math.imul(hash, prime);
    }
    return hash >>> 0; // Ensure non-negative integer
}

/**
 * Calculates a 53bit hash (cyrb53) of a sequence of strings and arrays of 32bit integer
 * numbers, rendered in base 36. Strings contribute their UTF-16 code units. Wide enough that a
 * collision between two keys is negligible, where a 32bit hash collides within tens of thousands.
 *
 * @param {...(string|ArrayLike<number>)} parts - The strings and arrays to hash, in order.
 * @returns {string} The hash.
 * @ignore
 */
function hash53(...parts) {
    let h1 = 0xdeadbeef;
    let h2 = 0x41c6ce57;
    for (let p = 0; p < parts.length; p++) {
        const part = parts[p];
        const isString = typeof part === 'string';
        for (let i = 0; i < part.length; i++) {
            const word = isString ? part.charCodeAt(i) : part[i] >>> 0;
            h1 = Math.imul(h1 ^ word, 2654435761);
            h2 = Math.imul(h2 ^ word, 1597334677);
        }
    }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
}

export { hashCode, hash32Fnv1a, hash53 };
