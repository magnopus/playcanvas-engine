import { expect } from 'chai';

import { PAGE_NOT_RESIDENT, PAGE_REQUEST } from '../../../src/scene/meshlet/constants.js';
import { FETCH_PRIORITY_ROOTS } from '../../../src/scene/meshlet/streaming/meshlet-fetch-scheduler.js';
import { MeshletResidency } from '../../../src/scene/meshlet/streaming/meshlet-residency.js';

const PAGE = 256;

// A fake streamed world with `slots` pool slots and `pages` pages in one stream; the pool
// records its writes so installs can be checked.
const makeWorld = (slots, pages, installBytes = 4 * PAGE) => ({
    poolSlots: slots,
    totalPages: pages,
    materialRowCount: 0,
    pageSizeBytes: PAGE,
    maxInstallBytesPerFrame: installBytes,
    residency: new Uint32Array(pages).fill(PAGE_NOT_RESIDENT),
    pagePool: { writes: [],
        write(offset, data) {
            this.writes.push({ offset, page: data[0] });
        } },
    residencyBuffer: { flushes: 0,
        write() {
            this.flushes++;
        } },
    requestsBuffer: { impl: { buffer: {} } },
    streamInfo: [{ resource: { manifest: { pageTable: new Uint32Array(pages * 8), pageSizeBytes: PAGE, rootPages: [0], blobs: [{ uri: 'r.dat' }] } }, pageBase: 0, baseUrl: 'x/' }]
});
const device = { getCommandEncoder: () => ({ clearBuffer() {} }), createBufferImpl: () => ({ allocate() {}, destroy() {}, buffer: null }) };
const words = page => Uint32Array.from({ length: PAGE / 4 }, (_, i) => (i === 0 ? page : 0));

describe('MeshletResidency', function () {

    let savedError;

    beforeEach(function () {
        savedError = console.error;
        console.error = () => {};
    });

    afterEach(function () {
        console.error = savedError;
    });

    it('starts with every slot free and installs pages into slots, lowest first', function () {
        const world = makeWorld(3, 10);
        const residency = new MeshletResidency(device, world);
        expect(residency.freeSlots).to.have.lengthOf(3);
        residency._installPage(7, words(7), false);
        residency._installPage(2, words(2), true);
        expect(world.residency[7]).to.equal(0);
        expect(world.residency[2]).to.equal(1);
        expect(world.pagePool.writes).to.deep.equal([{ offset: 0, page: 7 }, { offset: PAGE, page: 2 }]);
        expect(residency.residentPages).to.equal(2);
        expect(residency.fetchedPages, 'pinned roots are not counted as fetched').to.equal(1);
        expect(residency.slotPinned[1]).to.equal(1);
        residency._installPage(7, words(7), false);
        expect(residency.residentPages, 'a double install is ignored').to.equal(2);
        residency.destroy();
    });

    it('evicts the least recently wanted slot that is neither pinned nor wanted this frame', function () {
        const world = makeWorld(3, 10);
        const residency = new MeshletResidency(device, world);
        residency._installPage(0, words(0), true);    // pinned root
        residency.frame = 5;
        residency._installPage(1, words(1), false);
        residency.frame = 3;
        residency._installPage(2, words(2), false);   // older
        residency.frame = 9;

        residency._protected = new Set([2]);          // page 2 is wanted this frame
        residency._installPage(3, words(3), false);
        expect(world.residency[1], 'page 1 lost its slot').to.equal(PAGE_NOT_RESIDENT);
        expect(world.residency[3]).to.equal(1);
        expect(residency.evictedPages).to.equal(1);

        residency._protected = new Set([2, 3]);
        residency._installPage(4, words(4), false);
        expect(residency.droppedNoSlot, 'everything pinned or protected: dropped').to.equal(1);
        expect(world.residency[4]).to.equal(PAGE_NOT_RESIDENT);
        residency.destroy();
    });

    it('fetches every root page at root priority and pins it for good', async function () {
        const world = makeWorld(4, 6);
        world.cut = { rootPages: [0, 2] };
        const residency = new MeshletResidency(device, world);
        const fetches = [];
        residency._streams[0].fetcher = {
            buildRuns: pages => pages.map((p, i) => ({ blob: 0, offset: p * PAGE, length: PAGE, pages: [{ localPage: p, byteOffset: 0 }] })),
            fetchRange: (blob, offset, length, priority) => {
                fetches.push({ page: offset / PAGE, priority });
                const bytes = new ArrayBuffer(PAGE);
                new Uint32Array(bytes)[0] = offset / PAGE;
                return Promise.resolve(bytes);
            }
        };
        await residency.loadRoots();
        expect(fetches).to.deep.equal([{ page: 0, priority: FETCH_PRIORITY_ROOTS }, { page: 2, priority: FETCH_PRIORITY_ROOTS }]);
        expect(residency.rootsResident, 'demand streaming starts at once').to.equal(true);
        expect(residency.rootsLoaded).to.equal(false);
        residency._drainArrived();
        expect(residency.rootsLoaded).to.equal(true);
        expect(world.residency[0]).to.not.equal(PAGE_NOT_RESIDENT);
        expect(world.residency[2]).to.not.equal(PAGE_NOT_RESIDENT);
        expect(residency.slotPinned[world.residency[0]]).to.equal(1);
        expect(residency.slotPinned[world.residency[2]]).to.equal(1);
        // demand pages evict around the roots, never through them
        for (const page of [1, 3, 4, 5]) residency._installPage(page, words(page), false);
        expect(world.residency[0]).to.not.equal(PAGE_NOT_RESIDENT);
        expect(world.residency[2]).to.not.equal(PAGE_NOT_RESIDENT);
        residency.destroy();
    });

    it('pins a demand-installed root page too', function () {
        const world = makeWorld(4, 6);
        world.cut = { rootPages: [3] };
        const residency = new MeshletResidency(device, world);
        residency._installPage(3, words(3), false);
        expect(residency.slotPinned[world.residency[3]]).to.equal(1);
        residency.destroy();
    });

    it('pins only the roots the pool can hold beside a demand reserve, and reports the rest', function () {
        const world = makeWorld(4, 10);
        world.cut = { rootPages: [0, 1, 2, 3, 4] };
        const residency = new MeshletResidency(device, world);
        // a pool below roots + demand reserve keeps half its slots for demand
        expect(residency.rootShortfallPages).to.equal(3);
        const roomy = makeWorld(8, 7);
        roomy.cut = { rootPages: [0, 1, 2, 3, 4] };
        const fits = new MeshletResidency(device, roomy);
        expect(fits.rootShortfallPages, '8 slots - reserve of 2 non-root pages = 6 pinnable').to.equal(0);
        residency.destroy();
        fits.destroy();
    });

    it('marks a stream dead when its roots cannot be fetched', async function () {
        const world = makeWorld(4, 6);
        world.cut = { rootPages: [0] };
        const residency = new MeshletResidency(device, world);
        residency._streams[0].fetcher = {
            buildRuns: pages => [{ blob: 0, offset: 0, length: PAGE, pages: pages.map(p => ({ localPage: p, byteOffset: 0 })) }],
            fetchRange: () => Promise.reject(new Error('404'))
        };
        await residency.loadRoots();
        expect(residency._streams[0].dead).to.equal(true);
        expect(residency.inFlight.size).to.equal(0);
        residency.destroy();
    });

    it('turns request marks into touches, protection and a bounded fetch queue', function () {
        const world = makeWorld(4, 10);
        const residency = new MeshletResidency(device, world);
        residency._installPage(0, words(0), true);
        residency._installPage(1, words(1), false);
        residency.frame = 20;
        const queued = [];
        residency._streams[0].fetcher = {
            buildRuns: pages => [{ blob: 0, offset: 0, length: pages.length * PAGE, pages: pages.map((p, i) => ({ localPage: p, byteOffset: i * PAGE })) }],
            fetchRange: (blob, offset, length) => {
                queued.push(length / PAGE);
                return new Promise(() => {});
            }
        };

        const marks = new Uint32Array(10);
        marks[1] = PAGE_REQUEST.USED;
        marks[5] = PAGE_REQUEST.MISSING;
        marks[6] = PAGE_REQUEST.MISSING;
        marks[7] = PAGE_REQUEST.MISSING;
        marks[8] = PAGE_REQUEST.MISSING;
        residency._processRequests(marks);

        expect(residency.slotLastUsed[world.residency[1]], 'used page touched').to.equal(20);
        expect(residency._protected.has(1)).to.equal(true);
        expect(residency.lastMissingWanted).to.equal(4);
        // Two free slots; the occupied slots are pinned or used, so neither is evictable.
        expect(residency.inFlight.size).to.equal(2);
        expect(queued).to.deep.equal([2]);

        // a page already in flight is not requested again
        residency._processRequests(marks);
        expect(residency.inFlight.size).to.equal(2);
        residency.destroy();
    });

    it('installs arrived pages against the per-frame byte budget', function () {
        const world = makeWorld(8, 10, 2 * PAGE);
        const residency = new MeshletResidency(device, world);
        for (let p = 3; p < 8; p++) {
            residency.inFlight.add(p);
            residency._arrived.push({ globalPage: p, words: words(p) });
        }
        residency.frameUpdate();
        expect(residency.residentPages, 'two pages per frame at this budget').to.equal(2);
        expect(residency.inFlight.size, 'installed pages leave the in-flight set').to.equal(3);
        expect(world.residencyBuffer.flushes).to.equal(1);
        residency.frameUpdate();
        residency.frameUpdate();
        expect(residency.residentPages).to.equal(5);
        expect(residency._arrived).to.have.lengthOf(0);
        residency.destroy();
    });

    it('carries slot state across a rebuild, freeing pages the new world no longer has', function () {
        const world = makeWorld(4, 10);
        const first = new MeshletResidency(device, world);
        first._installPage(0, words(0), true);
        first._installPage(9, words(9), false);
        first.frame = 40;
        first.slotLastUsed[world.residency[9]] = 40;

        const smaller = makeWorld(4, 6);   // pages 6..9 dropped
        smaller.cut = { rootPages: [0] };  // pins follow the new world's roots
        smaller.residency.set(world.residency.subarray(0, 6));
        const carried = new MeshletResidency(device, smaller, { slotPage: first.slotPage, slotPinned: first.slotPinned, slotLastUsed: first.slotLastUsed, frame: first.frame });
        expect(carried.frame, 'the LRU clock continues').to.equal(40);
        expect(carried.residentPages, 'only the root survives').to.equal(1);
        expect(carried.freeSlots).to.have.lengthOf(3);
        expect(carried.slotPinned[world.residency[0]]).to.equal(1);
        first.destroy();
        carried.destroy();
    });

    describe('playable floor and fast load', function () {

        // a fetcher whose runs are single pages, resolving at once
        const instantFetcher = fetched => ({
            buildRuns: pages => pages.map(p => ({ blob: 0, offset: p * PAGE, length: PAGE, pages: [{ localPage: p, byteOffset: 0 }] })),
            fetchRange: (blob, offset) => {
                fetched?.push(offset / PAGE);
                const bytes = new ArrayBuffer(PAGE);
                new Uint32Array(bytes)[0] = offset / PAGE;
                return Promise.resolve(bytes);
            }
        });
        const settle = () => new Promise((resolve) => {
            setTimeout(resolve, 0);
        });

        it('fetches the floor beside the roots and becomes playable once it is installed', async function () {
            const world = makeWorld(32, 10);
            world.streamInfo[0].resource.manifest.sloppyFlags = true;
            world.cut = { rootPages: [0], floorPages: [0, 5, 3, 7] };
            const residency = new MeshletResidency(device, world);
            const fetched = [];
            residency._streams[0].fetcher = instantFetcher(fetched);
            expect(residency.sloppyAware).to.equal(true);

            residency.loadRoots();
            residency.loadFloor();
            expect(fetched, 'roots are fetched by loadRoots, the floor without them').to.deep.equal([0, 5, 3, 7]);
            expect(residency.floorLoaded).to.equal(false);
            await settle();
            residency._drainArrived();
            expect(residency.rootsLoaded).to.equal(true);
            expect(residency.floorLoaded).to.equal(true);
            expect(residency.playable).to.equal(true);
            for (const page of [5, 3, 7]) expect(world.residency[page]).to.not.equal(PAGE_NOT_RESIDENT);
            residency.destroy();
        });

        it('leaves floor pages beyond the pool to demand streaming, keeping the coarsest', function () {
            // 8 slots: one root, a demand reserve of 2 (a quarter of the pool), 5 for the floor
            const world = makeWorld(8, 20);
            world.streamInfo[0].resource.manifest.sloppyFlags = true;
            world.cut = { rootPages: [0], floorPages: [0, 9, 8, 7, 6, 5, 4, 3, 2] };
            const residency = new MeshletResidency(device, world);
            const fetched = [];
            residency._streams[0].fetcher = instantFetcher(fetched);
            const savedWarn = console.warn;
            console.warn = () => {};
            residency.loadRoots();
            residency.loadFloor();
            console.warn = savedWarn;
            expect(fetched.slice(1)).to.deep.equal([9, 8, 7, 6, 5]);
            expect(residency.floorShortfallPages).to.equal(3);
            residency.destroy();
        });

        it('stops waiting for a floor page whose fetch failed', async function () {
            const world = makeWorld(32, 10);
            world.streamInfo[0].resource.manifest.sloppyFlags = true;
            world.cut = { rootPages: [], floorPages: [4] };
            const residency = new MeshletResidency(device, world);
            residency._streams[0].fetcher = { ...instantFetcher(), fetchRange: () => Promise.reject(new Error('boom (500)')) };
            residency.loadRoots();
            residency.loadFloor();
            await settle();
            expect(residency.floorLoaded).to.equal(true);
            residency.destroy();
        });

        it('installs every arrived page in one frame while fast loading', function () {
            const world = makeWorld(32, 20, PAGE);
            const residency = new MeshletResidency(device, world);
            for (let p = 3; p < 13; p++) {
                residency.inFlight.add(p);
                residency._arrived.push({ globalPage: p, words: words(p) });
            }
            residency.fastLoad = true;
            residency._drainArrived();
            expect(residency.residentPages).to.equal(10);
            residency.destroy();
        });

        it('reports each resource\'s roots and floor separately', async function () {
            const world = makeWorld(32, 20);
            const manifest = pages => ({ pageTable: new Uint32Array(pages * 8), pageSizeBytes: PAGE, blobs: [{ uri: 'r.dat' }], sloppyFlags: true });
            const a = { manifest: manifest(10) };
            const b = { manifest: manifest(10) };
            world.streamInfo = [{ resource: a, pageBase: 0, baseUrl: 'a/' }, { resource: b, pageBase: 10, baseUrl: 'b/' }];
            world.cut = { rootPages: [0, 10], floorPages: [0, 10, 3, 4, 13] };
            const residency = new MeshletResidency(device, world);
            // stream b answers at once, stream a never does
            residency._streams[0].fetcher = { ...instantFetcher(), fetchRange: () => new Promise(() => {}) };
            residency._streams[1].fetcher = { ...instantFetcher(), fetchRange: (blob, offset) => instantFetcher().fetchRange(blob, offset + 10 * PAGE) };
            residency.loadRoots();
            residency.loadFloor();
            expect(residency.streamStatus(a)).to.deep.equal({ playable: false, failed: false, totalBytes: 3 * PAGE, loadedBytes: 0 });
            await settle();
            residency._drainArrived();
            expect(residency.streamStatus(b)).to.deep.equal({ playable: true, failed: false, totalBytes: 2 * PAGE, loadedBytes: 2 * PAGE });
            expect(residency.streamStatus(a).playable).to.equal(false);
            expect(residency.streamStatus({})).to.equal(null);
            residency.destroy();
        });

        it('counts a resource whose stream died as settled and failed', async function () {
            const world = makeWorld(32, 10);
            world.streamInfo[0].resource.manifest.sloppyFlags = true;
            world.cut = { rootPages: [0], floorPages: [0, 4] };
            const residency = new MeshletResidency(device, world);
            residency._streams[0].fetcher = { ...instantFetcher(), fetchRange: () => Promise.reject(new Error('gone (404)')) };
            residency.loadRoots();
            residency.loadFloor();
            await settle();
            const status = residency.streamStatus(world.streamInfo[0].resource);
            expect(status.playable).to.equal(true);
            expect(status.failed).to.equal(true);
            residency.destroy();
        });

        it('lets an old bake wait for its own pages only, not hold up flagged bakes', function () {
            const world = makeWorld(32, 20);
            const manifest = (pages, sloppyFlags) => ({ pageTable: new Uint32Array(pages * 8), pageSizeBytes: PAGE, blobs: [{ uri: 'r.dat' }], sloppyFlags });
            const flagged = { manifest: manifest(10, true) };
            const old = { manifest: manifest(10, false) };
            world.streamInfo = [{ resource: flagged, pageBase: 0, baseUrl: 'a/' }, { resource: old, pageBase: 10, baseUrl: 'b/' }];
            world.cut = { rootPages: [], floorPages: [] };
            const residency = new MeshletResidency(device, world);
            for (const stream of residency._streams) stream.fetcher = { ...instantFetcher(), fetchRange: () => new Promise(() => {}) };
            residency.loadRoots();
            residency.loadFloor();

            // the flagged bake's page 3 is still wanted; the old bake has nothing outstanding
            const marks = new Uint32Array(20);
            marks[3] = PAGE_REQUEST.MISSING;
            residency._processRequests(marks, true);
            residency._processRequests(marks, true);
            expect(residency.streamStatus(old).playable, 'old bake settled on its own pages').to.equal(true);
            expect(residency.streamStatus(flagged).playable, 'flagged bake needs only its floor').to.equal(true);
            expect(residency.playable).to.equal(true);

            // now the old bake wants a page
            const marks2 = new Uint32Array(20);
            marks2[14] = PAGE_REQUEST.MISSING;
            residency._processRequests(marks2, true);
            expect(residency.streamStatus(old).playable).to.equal(false);
            expect(residency.playable).to.equal(false);
            residency.destroy();
        });

        it('waits for the view to converge on bakes without sloppy flags', function () {
            const world = makeWorld(32, 10);
            world.cut = { rootPages: [], floorPages: [] };
            const residency = new MeshletResidency(device, world);
            residency._streams[0].fetcher = instantFetcher();
            residency.loadRoots();
            residency.loadFloor();
            expect(residency.sloppyAware).to.equal(false);
            expect(residency.rootsLoaded && residency.floorLoaded).to.equal(true);

            const marks = new Uint32Array(10);
            residency._processRequests(marks, false);
            residency._processRequests(marks, false);
            expect(residency.playable, 'readbacks from before the roots landed do not count').to.equal(false);

            marks[6] = PAGE_REQUEST.MISSING;
            residency._processRequests(marks, true);
            expect(residency.playable, 'a missing page was requested').to.equal(false);
            residency._processRequests(marks, true);
            expect(residency.playable, 'still in flight').to.equal(false);

            residency.inFlight.clear();
            world.residency[6] = 0;
            marks[6] = PAGE_REQUEST.USED;
            residency._processRequests(marks, true);
            residency._processRequests(marks, true);
            expect(residency.playable).to.equal(true);
            residency.destroy();
        });
    });
});
