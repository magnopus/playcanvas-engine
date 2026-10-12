/**
 * Bake stamps: a short id per streamed resource that changes whenever the bake does, used to
 * version the persistent cache keys of its sidecars (geometry shards, texture containers).
 * Sidecar URLs are often unversioned (`<name>_meshlets/pages_0.dat` next to the GLB), so a
 * rebake written over the same files would otherwise be served the previous bake's cached
 * pages - which the new GLB's page table reads as garbage.
 *
 * @ignore
 */

/** @type {WeakMap<object, string>} */
const stamps = new WeakMap();

/**
 * 53-bit hash (cyrb53) folded over 32-bit words, rendered in base 36.
 *
 * @param {ArrayLike<number>} words - The words.
 * @param {number} [h1] - Running state, word 1.
 * @param {number} [h2] - Running state, word 2.
 * @returns {[number, number]} The new state.
 */
const fold = (words, h1 = 0xdeadbeef, h2 = 0x41c6ce57) => {
    for (let i = 0; i < words.length; i++) {
        const w = words[i] >>> 0;
        h1 = Math.imul(h1 ^ w, 2654435761);
        h2 = Math.imul(h2 ^ w, 1597334677);
    }
    return [h1, h2];
};

/**
 * @param {string} str - A string.
 * @returns {Uint32Array} Its UTF-16 code units, one per word.
 */
const codeUnits = (str) => {
    const out = new Uint32Array(str.length);
    for (let i = 0; i < str.length; i++) out[i] = str.charCodeAt(i);
    return out;
};

/**
 * The bake stamp of a streamed resource: the bake's own `bakeId` from its stream manifest when
 * the baker wrote one, otherwise a hash of what any rebake changes - the page table (every
 * page's shard, offset and length), the shard list and the texture containers' layout.
 *
 * @param {{ manifest: object, textureManifest?: object|null }} resource - The MeshletResource.
 * @returns {string} The stamp.
 */
function meshletBakeStamp(resource) {
    let stamp = stamps.get(resource) ?? '';
    if (!stamp) {
        const manifest = resource.manifest;
        if (typeof manifest.bakeId === 'string' && manifest.bakeId) {
            stamp = manifest.bakeId;
        } else {
            let state = fold(manifest.pageTable ?? []);
            state = fold(codeUnits(JSON.stringify(manifest.blobs ?? [])), ...state);
            if (resource.textureManifest) {
                state = fold(codeUnits(JSON.stringify(resource.textureManifest)), ...state);
            }
            let [h1, h2] = state;
            h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
            h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
            stamp = (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
        }
        stamps.set(resource, stamp);
    }
    return stamp;
}

export { meshletBakeStamp };
