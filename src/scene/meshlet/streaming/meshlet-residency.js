import { Debug } from '../../../core/debug.js';
import { WebgpuReadbackPool } from '../../../platform/graphics/webgpu/webgpu-readback-pool.js';
import { PAGE_FLAG_ROOT, PAGE_NOT_RESIDENT, PAGE_REQUEST, PAGE_TABLE, PAGE_TABLE_FIELDS } from '../constants.js';
import { MeshletPageFetcher } from './meshlet-page-fetcher.js';

/**
 * @import { GraphicsDevice } from '../../../platform/graphics/graphics-device.js'
 * @import { MeshletWorld } from '../meshlet-world.js'
 */

/**
 * Concurrent fetch runs. A soft cap: it gates ENTRY to the pump, which then issues every run the
 * queued pages coalesce into, so a burst can briefly exceed it; the next readback (one to two
 * frames) is when a backlog gets another look.
 */
export const MAX_IN_FLIGHT_RUNS = 8;

/**
 * Streaming residency for a meshlet world: reads back the GPU's per-frame page request marks
 * (one to two frames latent), coalesces missing pages into HTTP Range runs, uploads arrived
 * pages into free or LRU-evicted pool slots and maintains the page-to-slot residency map. Root
 * pages are fetched eagerly and pinned, so the cull shader's ancestor fallback always
 * terminates on a resident cluster.
 *
 * @ignore
 */
class MeshletResidency {
    /** @type {GraphicsDevice} */
    device;

    /** @type {MeshletWorld} */
    world;

    /** @type {WebgpuReadbackPool} */
    readbackPool;

    /** @type {Uint32Array} - slot -> global page (PAGE_NOT_RESIDENT when free). */
    slotPage;

    /** @type {Uint32Array} - slot -> frame the page was last wanted. */
    slotLastUsed;

    /** @type {Uint8Array} - slot -> pinned flag (roots). */
    slotPinned;

    /** @type {number[]} - free slot stack. */
    freeSlots = [];

    /** @type {Set<number>} - global pages currently being fetched. */
    inFlight = new Set();

    /** @type {number} - concurrent run counter. */
    _activeRuns = 0;

    /** @type {number[]} - global pages waiting for a free run slot. */
    _queue = [];

    /** @type {Array<{ globalPage: number, words: Uint32Array }>} - fetched, awaiting install. */
    _arrived = [];

    _arrivedHead = 0;

    /**
     * Upload budget per frame for freshly streamed pages. Arrivals are lumpy - one coalesced
     * range can carry hundreds of pages - so they install over several frames instead of
     * spiking one. Raise it to fill faster, lower it for smoother frames.
     *
     * @type {number}
     */
    maxInstallBytesPerFrame = 4 * 1024 * 1024;

    frame = 0;

    _readbackBusy = false;

    _requestData = null;

    _residencyDirty = false;

    rootsResident = false;

    // stats
    residentPages = 0;

    fetchedPages = 0;

    evictedPages = 0;

    droppedNoSlot = 0;

    /** @type {number} - missing pages the last processed readback wanted (pressure signal). */
    lastMissingWanted = 0;

    /**
     * Receives the texture-mip feedback marks (one u32 per material row) from each completed
     * request readback - the texture residency's demand input.
     *
     * @type {((marks: Uint32Array) => void)|null}
     */
    onTexelRateMarks = null;

    /**
     * @param {GraphicsDevice} device - The graphics device.
     * @param {MeshletWorld} world - The finalized streamed world.
     * @param {object|null} [carry] - A previous residency's slot state ({ slotPage, slotPinned,
     * slotLastUsed }), adopted when the world carried its page pool across a rebuild. Slots
     * referencing pages the new world no longer has are freed.
     */
    constructor(device, world, carry = null) {
        this.maxInstallBytesPerFrame = world.maxInstallBytesPerFrame ?? this.maxInstallBytesPerFrame;
        this.device = device;
        this.world = world;
        this.readbackPool = new WebgpuReadbackPool(device);

        const slots = world.poolSlots;
        if (carry && carry.slotPage.length === slots) {
            this.slotPage = carry.slotPage;
            this.slotPinned = carry.slotPinned;
            this.slotLastUsed = carry.slotLastUsed;
            // the LRU compares against slotLastUsed - restarting the frame counter at 0 would
            // make every carried slot look ancient and the freshly touched ones look oldest
            this.frame = carry.frame;
            for (let s = slots - 1; s >= 0; s--) {
                if (this.slotPage[s] !== PAGE_NOT_RESIDENT && this.slotPage[s] >= world.totalPages) {
                    // page belonged to a dropped resource
                    this.slotPage[s] = PAGE_NOT_RESIDENT;
                    this.slotPinned[s] = 0;
                }
                if (this.slotPage[s] === PAGE_NOT_RESIDENT) {
                    this.slotPinned[s] = 0;
                    this.freeSlots.push(s);
                } else {
                    this.residentPages++;
                }
            }
        } else {
            this.slotPage = new Uint32Array(slots).fill(PAGE_NOT_RESIDENT);
            this.slotLastUsed = new Uint32Array(slots);
            this.slotPinned = new Uint8Array(slots);
            for (let s = slots - 1; s >= 0; s--) this.freeSlots.push(s);
        }

        this._requestData = new Uint32Array(world.totalPages + (world.materialRowCount ?? 0));

        // per-resource fetchers
        this._streams = world.streamInfo.map(info => ({
            ...info,
            fetcher: new MeshletPageFetcher(info.resource.manifest, info.baseUrl)
        }));
    }

    destroy() {
        // in-flight fetch completions and pending readbacks check this before touching the
        // (possibly rebuilt) world's buffers
        this._destroyed = true;
        this.readbackPool.destroy();
    }

    _streamOf(globalPage) {
        // streams are ordered by pageBase
        for (let i = this._streams.length - 1; i >= 0; i--) {
            if (globalPage >= this._streams[i].pageBase) return this._streams[i];
        }
        return null;
    }

    /**
     * Fetches every root page (blob 0 of each resource) and pins it. Resolves when the coarse
     * fallback set is resident and the world can render.
     *
     * @returns {Promise<void>} Resolves when roots are resident.
     */
    async loadRoots() {
        await Promise.all(this._streams.map(async (stream) => {
            const manifest = stream.resource.manifest;
            // carried across a rebuild: skip streams whose roots are all still resident
            const world = this.world;
            if (manifest.rootPages.every(p => world.residency[stream.pageBase + p] !== PAGE_NOT_RESIDENT)) {
                return;
            }
            const bytes = await stream.fetcher.fetchBlob(0);
            const table = manifest.pageTable;
            for (const localPage of manifest.rootPages) {
                const entry = localPage * PAGE_TABLE_FIELDS;
                Debug.assert(table[entry + PAGE_TABLE.BLOB] === 0 && (table[entry + PAGE_TABLE.FLAGS] & PAGE_FLAG_ROOT),
                    'MeshletResidency: manifest rootPages entry is not in the roots blob');
                const offset = table[entry + PAGE_TABLE.OFFSET_HI] * 0x100000000 + table[entry + PAGE_TABLE.OFFSET_LO];
                this._installPage(stream.pageBase + localPage, new Uint32Array(bytes, offset, manifest.pageSizeBytes / 4), true);
            }
        }));
        if (this._destroyed) return;
        this._flushResidency();
        this.rootsResident = true;
    }

    _allocSlot(protectedSet) {
        if (this.freeSlots.length) {
            return this.freeSlots.pop();
        }
        // evict the least recently wanted non-pinned, non-protected slot
        let best = -1;
        let bestUsed = Infinity;
        for (let s = 0; s < this.slotPage.length; s++) {
            if (this.slotPinned[s]) continue;
            if (protectedSet?.has(this.slotPage[s])) continue;
            if (this.slotLastUsed[s] < bestUsed) {
                bestUsed = this.slotLastUsed[s];
                best = s;
            }
        }
        if (best >= 0) {
            const page = this.slotPage[best];
            this.world.residency[page] = PAGE_NOT_RESIDENT;
            this.slotPage[best] = PAGE_NOT_RESIDENT;
            this.residentPages--;
            this.evictedPages++;
            this._residencyDirty = true;
        }
        return best;
    }

    _installPage(globalPage, pageWords, pinned) {
        if (this._destroyed) return;
        const world = this.world;
        if (world.residency[globalPage] !== PAGE_NOT_RESIDENT) {
            return; // already resident (double fetch)
        }
        const slot = this._allocSlot(this._protected);
        if (slot < 0) {
            this.droppedNoSlot++;
            return;
        }
        world.pagePool.write(slot * world.pageSizeBytes, pageWords);
        world.residency[globalPage] = slot;
        this.slotPage[slot] = globalPage;
        this.slotPinned[slot] = pinned ? 1 : 0;
        this.slotLastUsed[slot] = this.frame;
        this.residentPages++;
        if (!pinned) this.fetchedPages++;
        this._residencyDirty = true;
    }

    /**
     * Installs pages that have arrived, up to this frame's byte budget. Pages stay in
     * {@link inFlight} until installed, so the demand pass does not re-request them.
     *
     * @private
     */
    _drainArrived() {
        const pageBytes = this.world.pageSizeBytes;
        const budget = Math.max(this.maxInstallBytesPerFrame, pageBytes);
        let spent = 0;
        while (this._arrivedHead < this._arrived.length && spent < budget) {
            const { globalPage, words } = this._arrived[this._arrivedHead++];
            this.inFlight.delete(globalPage);
            this._installPage(globalPage, words, false);
            spent += pageBytes;
        }
        if (this._arrivedHead > 0 && this._arrivedHead === this._arrived.length) {
            this._arrived.length = 0;
            this._arrivedHead = 0;
        } else if (this._arrivedHead > 1024) {
            this._arrived = this._arrived.slice(this._arrivedHead);
            this._arrivedHead = 0;
        }
    }

    _flushResidency() {
        if (this._residencyDirty) {
            this.world.residencyBuffer.write(0, this.world.residency);
            this._residencyDirty = false;
        }
    }

    _pump() {
        while (this._activeRuns < MAX_IN_FLIGHT_RUNS && this._queue.length) {
            // take a batch of queued pages per stream and coalesce into runs
            const byStream = new Map();
            for (const page of this._queue) {
                const stream = this._streamOf(page);
                let list = byStream.get(stream);
                if (!list) {
                    list = []; byStream.set(stream, list);
                }
                list.push(page - stream.pageBase);
            }
            this._queue.length = 0;

            for (const [stream, localPages] of byStream) {
                const runs = stream.fetcher.buildRuns(localPages);
                for (const run of runs) {
                    this._activeRuns++;
                    stream.fetcher.fetchRange(run.blob, run.offset, run.length).then((bytes) => {
                        const pageWordsSize = stream.resource.manifest.pageSizeBytes / 4;
                        // Queue the arrivals; the frame tick installs them against a byte
                        // budget. Runs coalesce contiguous pages, so a single completion can
                        // carry hundreds of pages - installing them all here would push
                        // megabytes of uploads into one frame and show up as a hitch.
                        for (const { localPage, byteOffset } of run.pages) {
                            const globalPage = stream.pageBase + localPage;
                            this._arrived.push({
                                globalPage,
                                words: new Uint32Array(bytes, byteOffset, pageWordsSize)
                            });
                        }
                    }).catch((err) => {
                        Debug.error(`MeshletResidency: run fetch failed: ${err.message}`);
                        for (const { localPage } of run.pages) {
                            this.inFlight.delete(stream.pageBase + localPage);
                        }
                    }).finally(() => {
                        this._activeRuns--;
                    });
                }
            }
        }
    }

    /**
     * Per-frame tick: upload freshly arrived residency changes, process the completed request
     * readback, queue fetches and kick the next readback. Call before the culler dispatches.
     */
    frameUpdate() {
        this.frame++;
        const world = this.world;

        this._drainArrived();
        this._flushResidency();

        if (!this.rootsResident) {
            return;
        }

        // kick a request-marks readback when the previous one completed. The copy is recorded
        // on the current command encoder ahead of this frame's cull dispatch, so it captures
        // LAST frame's marks; the buffer is then cleared in-encoder for the coming frame.
        if (!this._readbackBusy) {
            this._readbackBusy = true;
            const requestsSize = (world.totalPages + (world.materialRowCount ?? 0)) * 4;
            this.readbackPool.read(world.requestsBuffer, 0, requestsSize, this._requestData).then((data) => {
                this._readbackBusy = false;
                this._processRequests(data);
                // texture-mip feedback marks live after the page marks - hand them to the
                // texture residency (same readback, same clear)
                if (!this._destroyed && world.materialRowCount) {
                    this.onTexelRateMarks?.(data.subarray(world.totalPages));
                }
            }).catch(() => {
                this._readbackBusy = false;
            });
            const encoder = this.device.getCommandEncoder();
            encoder.clearBuffer(world.requestsBuffer.impl.buffer, 0, requestsSize);
        }
    }

    _processRequests(marks) {
        if (this._destroyed) return;
        const world = this.world;
        const wanted = this._protected ?? new Set();
        wanted.clear();

        // PAGE_REQUEST.USED = resident page the GPU used this frame (touch + protect),
        // PAGE_REQUEST.MISSING = missing page the cut wanted (fetch candidate)
        const missing = [];
        let usedResident = 0;
        for (let p = 0; p < world.totalPages; p++) {
            const mark = marks[p];
            if (mark === PAGE_REQUEST.NONE) continue;
            wanted.add(p);
            if (mark === PAGE_REQUEST.USED) {
                const slot = world.residency[p];
                if (slot !== PAGE_NOT_RESIDENT) {
                    this.slotLastUsed[slot] = this.frame;
                    usedResident++;
                }
            } else if (world.residency[p] === PAGE_NOT_RESIDENT && !this.inFlight.has(p)) {
                // in flight = queued, being fetched, or arrived and awaiting install: the GPU may
                // keep marking such a page every frame, and it is never requested twice
                missing.push(p);
            }
        }
        this._protected = wanted;
        this.lastMissingWanted = missing.length;

        // only fetch what can actually be installed: free slots plus cold (unpinned, unused)
        // slots. Fetching beyond that would evict hot pages or be dropped on arrival - the
        // starved-pool steady state is the coarser ancestor fallback, not fetch churn.
        let pinnedCount = 0;
        for (let s = 0; s < this.slotPinned.length; s++) pinnedCount += this.slotPinned[s];
        const evictable = Math.max(this.slotPage.length - pinnedCount - usedResident, 0);
        const budget = Math.max(this.freeSlots.length + evictable - this.inFlight.size, 0);
        for (let i = 0; i < Math.min(missing.length, budget); i++) {
            this.inFlight.add(missing[i]);
            this._queue.push(missing[i]);
        }
        this._pump();
    }
}

export { MeshletResidency };
