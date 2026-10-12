import { Debug } from '../../../core/debug.js';
import { WebgpuReadbackPool } from '../../../platform/graphics/webgpu/webgpu-readback-pool.js';
import { PAGE_NOT_RESIDENT, PAGE_REQUEST } from '../constants.js';
import { FETCH_PRIORITY_ROOTS } from './meshlet-fetch-scheduler.js';
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
 * Minimum pool slots kept free of pinned roots for demand pages, so a pool too small for every
 * root still streams refinement.
 */
const DEMAND_SLOT_RESERVE = 64;

/** Consecutive settled request readbacks that make a view count as converged. */
const SETTLED_READBACKS = 2;

/**
 * Streaming residency for a meshlet world: reads back the GPU's per-frame page request marks
 * (one to two frames latent), coalesces missing pages into HTTP Range runs, uploads arrived
 * pages into free or LRU-evicted pool slots and maintains the page-to-slot residency map.
 *
 * Every resource's root pages (its coarsest clusters, `cut.rootPages`) are fetched once at load
 * and pinned for the resource's lifetime, as in Nanite: a few pages per unique mesh, whatever the
 * placement count. Any placement's coarse cut is then always drawable, so admission needs no
 * per-view page accounting.
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

    /**
     * Load as fast as possible: arrived pages install without the per-frame byte budget and more
     * fetch runs may be in flight. Set by the director until the world is playable, while a
     * loading screen hides the frame-time spikes this causes.
     *
     * @type {boolean}
     */
    fastLoad = false;

    /** @type {boolean} - {@link loadFloor} has run. */
    floorRequested = false;

    /** @type {number} - floor pages left unrequested because the pool cannot hold them. */
    floorShortfallPages = 0;

    /**
     * Floor pages requested but not yet installed (or failed).
     *
     * @type {Set<number>}
     * @private
     */
    _floorPending = new Set();

    /** @type {number} - consecutive request readbacks with nothing missing or in flight. @private */
    _settledReadbacks = 0;

    frame = 0;

    _readbackBusy = false;

    _requestData = null;

    _residencyDirty = false;

    /** @type {boolean} - demand streaming initialized (root pages may still be arriving). */
    rootsResident = false;

    /** @type {boolean} - every pinnable root page is installed. */
    rootsLoaded = false;

    /** @type {number} - root pages that could not be pinned because the pool is too small. */
    rootShortfallPages = 0;

    /** @type {Uint8Array} - global page -> 1 when it is a pinned root page. */
    _isRoot;

    /** @type {Array<{ globalPage: number, words: Uint32Array }>} - root pages awaiting install. */
    _arrivedRoots = [];

    _arrivedRootsHead = 0;

    /** @type {number} - root fetch runs still outstanding. */
    _rootRunsPending = 0;

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
     * referencing pages the new world no longer has are freed; slots the pool grew by start free.
     */
    constructor(device, world, carry = null) {
        this.maxInstallBytesPerFrame = world.maxInstallBytesPerFrame ?? this.maxInstallBytesPerFrame;
        this.device = device;
        this.world = world;
        this.readbackPool = new WebgpuReadbackPool(device);

        const slots = world.poolSlots;
        // Root pages pinned for good, up to what the pool can hold beside a demand reserve; the
        // world's budget sizes the pool to fit them, so a shortfall means the budget is too small.
        const rootPages = world.cut?.rootPages ?? [];
        // the world's budget sizes the pool for every root plus this reserve; a pool too small
        // for both (a budget below the fixed costs) still keeps half its slots for demand
        const reserve = Math.min(DEMAND_SLOT_RESERVE, Math.max(world.totalPages - rootPages.length, 0), Math.floor(slots / 2));
        const pinnable = Math.min(rootPages.length, Math.max(slots - reserve, 0));
        this._isRoot = new Uint8Array(world.totalPages);
        for (let i = 0; i < pinnable; i++) this._isRoot[rootPages[i]] = 1;
        this.rootShortfallPages = rootPages.length - pinnable;
        if (this.rootShortfallPages) {
            Debug.warnOnce(`MeshletResidency: the page pool (${slots} slots) cannot pin all ${rootPages.length} root pages; ${this.rootShortfallPages} stream on demand instead.`);
        }
        this._compareRequests = (a, b) => this._isRoot[b] - this._isRoot[a];

        if (carry && carry.slotPage.length <= slots) {
            this.slotPage = new Uint32Array(slots).fill(PAGE_NOT_RESIDENT);
            this.slotPage.set(carry.slotPage);
            this.slotPinned = new Uint8Array(slots);
            this.slotPinned.set(carry.slotPinned);
            this.slotLastUsed = new Uint32Array(slots);
            this.slotLastUsed.set(carry.slotLastUsed);
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
                    // pins follow the new world's roots: a carried pin on a page that is no
                    // longer a root is released, a carried page that is now a root is pinned
                    this.slotPinned[s] = this._isRoot[this.slotPage[s]];
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

        /**
         * Every streamed resource's bake marks its sloppy clusters, so the floor is exact. Older
         * bakes cannot tell sloppy levels apart, so readiness falls back to the view's cut
         * having converged.
         *
         * @type {boolean}
         */
        this.sloppyAware = world.streamInfo.length > 0 && world.streamInfo.every(info => info.resource.manifest.sloppyFlags === true);

        // per-resource fetchers
        this._streams = world.streamInfo.map((info, index) => ({
            ...info,
            index,
            fetcher: new MeshletPageFetcher(info.resource.manifest, info.baseUrl, info.fetchOptions ?? null, this.world.fetchScheduler),
            // fetch failure state: a stream whose shard is gone for good (404-class) or keeps
            // failing is dead - its pages are never requested again, so a broken package costs
            // one error line instead of a request per readback for the rest of the session
            failures: 0,
            retryAfterFrame: 0,
            dead: false,
            // per-resource load accounting (pages): roots and playable floor, required vs still
            // outstanding; pages already resident count as required and done
            rootsTotal: 0,
            rootsPending: 0,
            floorTotal: 0,
            floorPending: 0,
            // consecutive readbacks since the roots landed with none of this stream's pages
            // missing or in flight - the convergence test for bakes without sloppy flags
            settledReadbacks: 0
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
     * Starts streaming: fetches every resource's pinnable root pages (skipping those a carried
     * pool already holds) at root priority, and enables demand request processing at once - the
     * GPU simply cannot admit an instance until its roots land.
     *
     * @returns {Promise<void>} Resolves when every root fetch has settled (failed streams are
     * reported and marked dead; their placements stay unadmitted).
     */
    loadRoots() {
        this.rootsResident = true;
        const world = this.world;
        const byStream = new Map();
        for (let page = 0; page < this._isRoot.length; page++) {
            if (!this._isRoot[page]) continue;
            const stream = this._streamOf(page);
            if (!stream) continue;
            stream.rootsTotal++;
            if (world.residency[page] !== PAGE_NOT_RESIDENT || stream.dead) continue;
            stream.rootsPending++;
            let list = byStream.get(stream);
            if (!list) byStream.set(stream, list = []);
            list.push(page - stream.pageBase);
            this.inFlight.add(page);
        }
        const fetches = [];
        for (const [stream, localPages] of byStream) {
            for (const run of stream.fetcher.buildRuns(localPages)) {
                this._rootRunsPending++;
                fetches.push(stream.fetcher.fetchRange(run.blob, run.offset, run.length, FETCH_PRIORITY_ROOTS).then((bytes) => {
                    if (!bytes) {
                        for (const { localPage } of run.pages) this._settleFailedRoot(stream.pageBase + localPage);
                        return;
                    }
                    const pageWordsSize = stream.resource.manifest.pageSizeBytes / 4;
                    for (const { localPage, byteOffset } of run.pages) {
                        this._arrivedRoots.push({
                            globalPage: stream.pageBase + localPage,
                            words: new Uint32Array(bytes, byteOffset, pageWordsSize)
                        });
                    }
                }).catch((err) => {
                    // roots are required: no retries against a backoff, the stream is dead
                    stream.failures = Math.max(stream.failures, 3);
                    this._failStream(stream, err);
                    for (const { localPage } of run.pages) this._settleFailedRoot(stream.pageBase + localPage);
                }).finally(() => {
                    this._rootRunsPending--;
                }));
            }
        }
        if (!fetches.length) this.rootsLoaded = true;
        return Promise.all(fetches).then(() => undefined);
    }

    /**
     * Starts fetching the playable floor (see {@link MeshletCutData#floorPages}) beside the
     * roots, without waiting for the GPU to request it level by level. Pages already resident or
     * in flight are skipped; pages beyond what the pool can hold beside the roots and a demand
     * reserve are left to demand streaming, coarsest kept first.
     */
    loadFloor() {
        this.floorRequested = true;
        const world = this.world;
        const floor = world.cut?.floorPages ?? [];
        let rootsMissing = 0;
        for (const page of world.cut?.rootPages ?? []) {
            if (this._isRoot[page] && world.residency[page] === PAGE_NOT_RESIDENT) rootsMissing++;
        }
        const reserve = Math.min(DEMAND_SLOT_RESERVE, Math.floor(this.slotPage.length / 4));
        const capacity = Math.max(this.freeSlots.length - rootsMissing - reserve, 0);
        let requested = 0;
        for (const page of floor) {
            if (this._isRoot[page]) continue;
            const stream = this._streamOf(page);
            if (!stream) continue;
            if (world.residency[page] !== PAGE_NOT_RESIDENT) {
                stream.floorTotal++;
                continue;
            }
            if (stream.dead || this.inFlight.has(page)) continue;
            if (requested >= capacity) {
                this.floorShortfallPages++;
                continue;
            }
            requested++;
            stream.floorTotal++;
            stream.floorPending++;
            this._floorPending.add(page);
            this.inFlight.add(page);
            this._queue.push(page);
        }
        if (this.floorShortfallPages) {
            Debug.warnOnce(`MeshletResidency: the page pool cannot hold the playable floor; ${this.floorShortfallPages} floor page(s) stream on demand instead. Raise the geometry budget (poolBytes).`);
        }
        this._pump();
    }

    /**
     * Every requested floor page has been installed or given up on.
     *
     * @type {boolean}
     */
    get floorLoaded() {
        return this.floorRequested && this._floorPending.size === 0;
    }

    /**
     * The last request readbacks since the roots loaded found nothing missing and nothing in
     * flight: the view's cut is fully resident.
     *
     * @type {boolean}
     */
    get demandSettled() {
        return this._settledReadbacks >= SETTLED_READBACKS;
    }

    /**
     * The roots and the playable floor are resident (on bakes without sloppy flags: the roots,
     * and the current view's cut has converged).
     *
     * @type {boolean}
     */
    get playable() {
        if (!this.rootsLoaded || !this.floorLoaded) return false;
        // bakes with sloppy flags are done once their floor is in; only older bakes wait for
        // their own share of the view's cut, so one old asset cannot hold up the rest
        for (const stream of this._streams) {
            if (!stream.dead && stream.resource.manifest.sloppyFlags !== true && stream.settledReadbacks < SETTLED_READBACKS) {
                return false;
            }
        }
        return true;
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
            this.world.contentVersion++;
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
            if (pinned) this.slotPinned[world.residency[globalPage]] = 1;
            return; // already resident (double fetch)
        }
        const slot = this._allocSlot(this._protected);
        if (slot < 0) {
            this.droppedNoSlot++;
            return;
        }
        world.pagePool.write(slot * world.pageSizeBytes, pageWords);
        world.residency[globalPage] = slot;
        world.contentVersion++;
        this.slotPage[slot] = globalPage;
        this.slotPinned[slot] = pinned || this._isRoot[globalPage] ? 1 : 0;
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
        let budget = this.fastLoad ? Infinity : Math.max(this.maxInstallBytesPerFrame, pageBytes);
        let spent = 0;
        // roots first, at up to four times the budget while they load: nothing of a placement
        // draws until they are in, and they are a bounded one-off (a few pages per unique mesh)
        if (!this.rootsLoaded) {
            const rootBudget = budget * 4;
            while (this._arrivedRootsHead < this._arrivedRoots.length && spent < rootBudget) {
                const { globalPage, words } = this._arrivedRoots[this._arrivedRootsHead++];
                this._settleFailedRoot(globalPage);
                this._installPage(globalPage, words, true);
                spent += pageBytes;
            }
            if (this._arrivedRootsHead === this._arrivedRoots.length) {
                this._arrivedRoots.length = 0;
                this._arrivedRootsHead = 0;
                if (this._rootRunsPending === 0) this.rootsLoaded = true;
            }
            budget = Math.max(budget - spent, 0);
            spent = 0;
        }
        while (this._arrivedHead < this._arrived.length && spent < budget) {
            const { globalPage, words } = this._arrived[this._arrivedHead++];
            this._settleFloor(globalPage);
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
        const maxRuns = this.fastLoad ? MAX_IN_FLIGHT_RUNS * 4 : MAX_IN_FLIGHT_RUNS;
        while (this._activeRuns < maxRuns && this._queue.length) {
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
                        if (!bytes) {
                            // scheduler cleared (director destroyed): nothing arrives
                            for (const { localPage } of run.pages) this._settleFailed(stream.pageBase + localPage);
                            return;
                        }
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
                        this._failStream(stream, err);
                        for (const { localPage } of run.pages) {
                            this._settleFailed(stream.pageBase + localPage);
                        }
                    }).finally(() => {
                        this._activeRuns--;
                    });
                }
            }
        }
    }

    /**
     * Forgets a page whose fetch was dropped or failed: demand may request it again, and a floor
     * page stops holding up {@link floorLoaded}.
     *
     * @param {number} globalPage - The page.
     * @private
     */
    _settleFailed(globalPage) {
        this.inFlight.delete(globalPage);
        this._settleFloor(globalPage);
    }

    /**
     * Stops a floor page holding up its resource and {@link floorLoaded}.
     *
     * @param {number} globalPage - The page.
     * @private
     */
    _settleFloor(globalPage) {
        if (this._floorPending.delete(globalPage)) {
            const stream = this._streamOf(globalPage);
            if (stream) stream.floorPending--;
        }
    }

    /**
     * Forgets a root page that arrived (it installs next) or whose fetch failed.
     *
     * @param {number} globalPage - The page.
     * @private
     */
    _settleFailedRoot(globalPage) {
        if (this.inFlight.delete(globalPage)) {
            const stream = this._streamOf(globalPage);
            if (stream) stream.rootsPending--;
        }
    }

    /**
     * Load state of one streamed resource: its root pages and its playable floor.
     *
     * @param {object} resource - The resource (a MeshletResource in this world).
     * @returns {{ playable: boolean, failed: boolean, totalBytes: number, loadedBytes: number }|null}
     * Whether its roots and floor are resident (on bakes without sloppy flags, also that the
     * view's cut has converged), whether its stream was given up on, and its required and
     * resident page bytes; null when the resource is not in this world.
     */
    streamStatus(resource) {
        const stream = this._streams.find(s => s.resource === resource);
        if (!stream) return null;
        const pageBytes = this.world.pageSizeBytes;
        const total = stream.rootsTotal + stream.floorTotal;
        const pending = stream.rootsPending + stream.floorPending;
        const loaded = this.rootsResident && this.floorRequested && pending === 0;
        const converged = stream.resource.manifest.sloppyFlags === true || stream.settledReadbacks >= SETTLED_READBACKS;
        return {
            playable: stream.dead || (loaded && converged),
            failed: stream.dead,
            totalBytes: total * pageBytes,
            loadedBytes: (total - pending) * pageBytes
        };
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
            // marks recorded before the roots landed say nothing about convergence: the GPU
            // admits no instance until its roots are resident
            const afterRoots = this.rootsLoaded;
            const requestsSize = (world.totalPages + (world.materialRowCount ?? 0)) * 4;
            this.readbackPool.read(world.requestsBuffer, 0, requestsSize, this._requestData).then((data) => {
                this._readbackBusy = false;
                this._processRequests(data, afterRoots);
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

    /**
     * Records a failed fetch on a stream: a 404-class status kills it outright, anything else
     * backs it off exponentially and kills it after repeated failures.
     *
     * @param {object} stream - The stream.
     * @param {Error} err - The failure.
     * @private
     */
    _failStream(stream, err) {
        const status = /\((\d{3})\)/.exec(err?.message ?? '')?.[1];
        stream.failures++;
        if (status === '404' || status === '403' || status === '410' || stream.failures >= 4) {
            if (!stream.dead) {
                stream.dead = true;
                console.error(`MeshletResidency: giving up on ${stream.resource.manifest.blobs[0]?.uri ?? 'a resource'} after ${stream.failures} failed fetch(es): ${err?.message ?? err}`);
            }
            return;
        }
        stream.retryAfterFrame = this.frame + 60 * (1 << stream.failures);
    }

    /** @type {number} - streams given up on after fetch failures. */
    get deadStreams() {
        return this._streams.reduce((n, s) => n + (s.dead ? 1 : 0), 0);
    }

    /**
     * Per-stream form of the convergence test: a stream is settled when none of its pages is
     * missing or in flight (or nothing missing could fit in the pool).
     *
     * @param {number[]} missing - Pages the cut wants that are not resident, ascending by stream.
     * @param {number} budget - Slots available for fetching this readback.
     * @param {boolean} afterRoots - The readback was kicked after the roots landed.
     * @private
     */
    _updateStreamSettled(missing, budget, afterRoots) {
        const streams = this._streams;
        if (!streams.length) return;
        const busy = new Uint8Array(streams.length);
        const markBusy = (page) => {
            const stream = this._streamOf(page);
            if (stream) busy[stream.index] = 1;
        };
        if (budget > 0) for (const page of missing) markBusy(page);
        for (const page of this.inFlight) markBusy(page);
        for (let i = 0; i < streams.length; i++) {
            streams[i].settledReadbacks = (afterRoots && !busy[i]) ? streams[i].settledReadbacks + 1 : 0;
        }
    }

    _processRequests(marks, afterRoots = true) {
        if (this._destroyed) return;
        const world = this.world;
        const wanted = this._protected ?? new Set();
        wanted.clear();

        // PAGE_REQUEST.USED = resident page the GPU used this frame (touch + protect),
        // PAGE_REQUEST.MISSING = missing page the cut wanted (fetch candidate)
        const missing = [];
        const streams = this._streams;
        let si = 0; // streams are ordered by pageBase, pages ascend: one pass tracks the stream
        for (let p = 0; p < world.totalPages; p++) {
            const mark = marks[p];
            if (mark === PAGE_REQUEST.NONE) continue;
            wanted.add(p);
            if (mark === PAGE_REQUEST.USED) {
                const slot = world.residency[p];
                if (slot !== PAGE_NOT_RESIDENT) {
                    this.slotLastUsed[slot] = this.frame;
                }
            } else if (world.residency[p] === PAGE_NOT_RESIDENT && !this.inFlight.has(p)) {
                // in flight = queued, being fetched, or arrived and awaiting install: the GPU may
                // keep marking such a page every frame, and it is never requested twice
                while (si + 1 < streams.length && p >= streams[si + 1].pageBase) si++;
                const stream = streams[si];
                if (stream.dead || stream.retryAfterFrame > this.frame) continue;
                missing.push(p);
            }
        }
        this._protected = wanted;
        this.lastMissingWanted = missing.length;
        missing.sort(this._compareRequests);

        // only fetch what can actually be installed: free slots plus cold (unpinned, unused)
        // slots. Fetching beyond that would evict hot pages or be dropped on arrival - the
        // starved-pool steady state is the coarser ancestor fallback, not fetch churn.
        let evictable = 0;
        for (let s = 0; s < this.slotPage.length; s++) {
            const page = this.slotPage[s];
            if (page !== PAGE_NOT_RESIDENT && !this.slotPinned[s] && !wanted.has(page)) evictable++;
        }
        const budget = Math.max(this.freeSlots.length + evictable - this.inFlight.size, 0);
        // converged: nothing in flight, and nothing missing - or nothing that could fit (a pool
        // too small for the cut settles at the coarser fallback and must not wait forever)
        const settled = afterRoots && this.inFlight.size === 0 && (missing.length === 0 || budget === 0);
        this._settledReadbacks = settled ? this._settledReadbacks + 1 : 0;
        this._updateStreamSettled(missing, budget, afterRoots);
        for (let i = 0; i < Math.min(missing.length, budget); i++) {
            this.inFlight.add(missing[i]);
            this._queue.push(missing[i]);
        }
        this._pump();
    }
}

export { MeshletResidency };
