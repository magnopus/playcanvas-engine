/**
 * @import { CameraComponent } from '../../framework/components/camera/component.js'
 * @import { MeshletResidency } from './streaming/meshlet-residency.js'
 * @import { MeshletShadowRenderer } from './meshlet-shadow-renderer.js'
 * @import { MeshletView } from './meshlet-view.js'
 * @import { MeshletWorld } from './meshlet-world.js'
 * @import { StorageBuffer } from '../../platform/graphics/storage-buffer.js'
 */

// Margin over the pair-count ratio when pre-sizing carried index capacities for a scene that
// grew: enough to lead the demand readback by a step change, not a scene-size ratio.
const CARRY_GROWTH_MARGIN = 1.35;

/**
 * The same placement list (an instance list is compared by identity, a placement array by its
 * entries).
 *
 * @param {*} a - A placement list.
 * @param {*} b - Another.
 * @returns {boolean} True when they place the same instances.
 */
const sameInstances = (a, b) => {
    if (a === b) return true;
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
};

/**
 * What a director's rebuild hands from the views of the previous world to those of the new
 * one: each camera's persistent visibility bits and each view's demand-grown index capacities.
 * A recreated view would otherwise start from scratch - its first phase 2 re-discovering the
 * whole frustum, its draws clamped at the initial capacity until the demand readback catches up
 * (see MeshletView#growAt) - and the page-demand burst either causes evicts the hot working set.
 *
 * Every entry is handed over once: views take theirs as they are created.
 *
 * @ignore
 */
class MeshletRebuildCarry {
    /**
     * Index capacities per bucket, by camera.
     *
     * @type {Map<CameraComponent, number[]>}
     */
    caps = new Map();

    /**
     * Shadow views' index capacities per bucket, keyed `lightId:face`. Shadow views hold buffers
     * sized to the old world (claim bits are one bit per instance-meshlet pair), so they are
     * rebuilt rather than carried, keeping only their capacities.
     *
     * @type {Map<string, number[]>}
     */
    shadowCaps = new Map();

    /**
     * Visibility-bit buffers by camera, GPU-copied into the recreated views' fresh buffers.
     *
     * @type {Map<CameraComponent, StorageBuffer>}
     */
    visBits = new Map();

    /** @type {number} - leading words of {@link visBits} that are valid in the new world. */
    visWords = 0;

    /**
     * Records the previous world's view capacities.
     *
     * @param {Map<CameraComponent, MeshletView>} views - Its camera views.
     * @param {MeshletShadowRenderer|null} shadowRenderer - Its shadow views.
     */
    captureCapacities(views, shadowRenderer) {
        views.forEach((view, camera) => this.caps.set(camera, view.indexCapacity.slice()));
        shadowRenderer?.entries.forEach((entry, light) => {
            entry.views.forEach((view) => {
                this.shadowCaps.set(`${light.id}:${view.face}`, view.indexCapacity.slice());
            });
        });
    }

    /**
     * Scales the carried capacities to the new world. Index demand roughly follows the pair
     * count, but it is screen-bound - the cut coarsens with distance and the frustum limits what
     * is in view - so a grown scene's carry is raised by a bounded margin (pre-sizing only has to
     * cover a step change), while a shrunk scene's is scaled down in proportion. A flat
     * multiplier would compound over same-size rebuilds (a budget change, say) into unbounded
     * buffers.
     *
     * @param {number} pairRatio - The new world's instance-meshlet pairs over the previous one's.
     * @param {number} maxScale - Most a capacity may grow by.
     */
    scaleCapacities(pairRatio, maxScale) {
        const scale = pairRatio > 1 ? Math.min(pairRatio * CARRY_GROWTH_MARGIN, maxScale) : pairRatio;
        const scaleCaps = (/** @type {number[]} */ caps) => {
            for (let b = 0; b < caps.length; b++) caps[b] = Math.ceil(caps[b] * scale);
        };
        this.caps.forEach(scaleCaps);
        this.shadowCaps.forEach(scaleCaps);
    }

    /**
     * Decides what the new (pending, not yet finalized) world inherits from the previous
     * streamed one, and hands it over through the world's adopt* fields.
     *
     * Leading resources that are identical in both builds (same resource, base URL and order)
     * keep their page space; the stricter pair prefix - instance lists unchanged too - also keeps
     * the views' visibility bits. The page pool carries when the budget is unchanged (the world
     * keeps a pool large enough and copies the slots of one that grew) or the pool would not
     * shrink - a budget cut starts cold. The texture system carries when the textured resources
     * are an unchanged prefix of the new build's and every one after it fits the family arrays
     * as sized: those are appended in place, so a scene whose assets arrive one by one does not
     * re-download its loaded tails per arrival.
     *
     * @param {MeshletWorld} world - The new world.
     * @param {MeshletWorld} prevWorld - The finalized, streamed world being replaced.
     * @param {MeshletResidency} prevResidency - Its residency.
     * @param {Map<CameraComponent, MeshletView>} prevViews - Its views; their visibility-bit
     * buffers are taken over.
     * @returns {object|null} The slot state to carry into the new {@link MeshletResidency}, or
     * null when the pool does not carry.
     */
    adoptStreamingState(world, prevWorld, prevResidency, prevViews) {
        const pending = world._pending;
        const prevEntries = prevWorld.buildEntries;
        const { poolBytes, texturePoolBytes } = world;

        const repeatOf = transform => (Array.isArray(transform) ? transform.length : 1);
        let prefix = 0;
        let adoptedPages = 0;
        let carriedPairs = 0;
        let pairPrefixIntact = true;
        while (prefix < pending.length && prefix < prevEntries.length) {
            const n = pending[prefix];
            const o = prevEntries[prefix];
            if (n.resource !== o.resource || (n.baseUrl ?? null) !== o.baseUrl ||
                !o.streamed || !!n.shardBuffers) break;
            if (pairPrefixIntact && sameInstances(n.instances ?? n.resource.instances, o.instances) && repeatOf(n.transform) === o.repeat) {
                carriedPairs += prevWorld.placements[prefix].pairCount;
            } else {
                pairPrefixIntact = false;
            }
            adoptedPages += o.pageCount;
            prefix++;
        }

        // predict the new pool's slot count the way finalize will: the budget covers more than
        // the pool, so with the budget unchanged the previous world's resolved pool is the
        // comparable figure
        let newTotalPages = 0;
        for (const { resource } of pending) newTotalPages += resource.manifest.pageCount;
        const newPagePoolBytes = poolBytes === prevWorld.poolBytes ? prevWorld.pagePoolBytes : poolBytes;
        let newPoolSlots = newPagePoolBytes > 0 ?
            Math.max(Math.floor(newPagePoolBytes / prevWorld.pageSizeBytes), 1) : newTotalPages;
        newPoolSlots = Math.min(newPoolSlots, newTotalPages, prevWorld.maxPoolSlots());

        let carriedSlots = null;
        if (prefix > 0 && (poolBytes === prevWorld.poolBytes || newPoolSlots >= prevWorld.poolSlots) &&
            pending[0].resource.manifest.pageSizeBytes === prevWorld.pageSizeBytes) {
            world.adoptPagePool = prevWorld.pagePool;
            prevWorld.pagePool = null;
            world.adoptResidency = prevWorld.residency;
            world.adoptedPages = adoptedPages;
            if (carriedPairs > 0) {
                // one bit per pair, 32 to a word: only whole words of the intact prefix carry
                this.visWords = Math.floor(carriedPairs / 32);
                prevViews.forEach((view, camera) => {
                    if (view.visBitsBuffer) {
                        this.visBits.set(camera, view.visBitsBuffer);
                        view.visBitsBuffer = null;
                    }
                });
            }
            carriedSlots = {
                slotPage: prevResidency.slotPage,
                slotPinned: prevResidency.slotPinned,
                slotLastUsed: prevResidency.slotLastUsed,
                frame: prevResidency.frame
            };
        }

        // the texture budget compares against what the live system was built with
        // (prevWorld.texturePoolBytes may already hold the caller's new value)
        const prevTextures = prevWorld.textures;
        if (prevTextures) {
            const prevTex = prevEntries.filter(e => e.resource.textureManifest);
            const newTex = pending.filter(e => e.resource.textureManifest);
            const prefixSame = prevTex.length <= newTex.length &&
                prevTex.every((e, i) => e.resource === newTex[i].resource && e.baseUrl === (newTex[i].baseUrl ?? null));
            const appendable = prefixSame &&
                newTex.slice(prevTex.length).every(e => prevTextures.canAppend(e.resource.textureManifest));
            if (appendable && prevTextures.poolBytes === texturePoolBytes &&
                prevTextures.maxFineSize === world.maxFineTextureSize) {
                world.adoptTextures = prevTextures;
                world.adoptTexturesPrefix = prevTex.length;
                prevWorld.textures = null;
            }
        }
        return carriedSlots;
    }

    /**
     * Hands over a camera's carried index capacities.
     *
     * @param {CameraComponent} camera - The camera.
     * @returns {number[]|null} The capacities, or null.
     */
    takeCaps(camera) {
        const caps = this.caps.get(camera) ?? null;
        this.caps.delete(camera);
        return caps;
    }

    /**
     * Hands over a shadow view's carried index capacities.
     *
     * @param {number} lightId - The light's id.
     * @param {number} face - The shadow face index.
     * @returns {number[]|null} The capacities, or null.
     */
    takeShadowCaps(lightId, face) {
        const key = `${lightId}:${face}`;
        const caps = this.shadowCaps.get(key) ?? null;
        this.shadowCaps.delete(key);
        return caps;
    }

    /**
     * Hands over a camera's carried visibility bits. The caller owns the buffer from here; it
     * must outlive the copy's submission.
     *
     * @param {CameraComponent} camera - The camera.
     * @returns {{ buffer: StorageBuffer, words: number }|null} The bits, or null.
     */
    takeVisBits(camera) {
        const buffer = this.visBits.get(camera);
        if (!buffer) return null;
        this.visBits.delete(camera);
        return { buffer, words: this.visWords };
    }

    /**
     * Releases the visibility bits no view took.
     */
    destroy() {
        this.visBits.forEach(buffer => buffer.destroy());
        this.visBits.clear();
    }
}

export { MeshletRebuildCarry };
