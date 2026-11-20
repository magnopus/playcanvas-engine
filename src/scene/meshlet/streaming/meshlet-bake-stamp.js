import { hash53 } from '../../../core/hash.js';

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
            stamp = hash53(
                manifest.pageTable ?? [],
                JSON.stringify(manifest.blobs ?? []),
                resource.textureManifest ? JSON.stringify(resource.textureManifest) : ''
            );
        }
        stamps.set(resource, stamp);
    }
    return stamp;
}

export { meshletBakeStamp };
