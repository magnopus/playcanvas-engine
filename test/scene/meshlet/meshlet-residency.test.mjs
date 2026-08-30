import { expect } from 'chai';

import { PAGE_NOT_RESIDENT, PAGE_REQUEST } from '../../../src/scene/meshlet/constants.js';
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
        // budget = free slots (2) + evictable (4 slots - 1 pinned - 1 used) - in flight (0) = 4
        expect(residency.inFlight.size).to.equal(4);
        expect(queued).to.deep.equal([4]);

        // a page already in flight is not requested again
        residency._processRequests(marks);
        expect(residency.inFlight.size).to.equal(4);
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
        smaller.residency.set(world.residency.subarray(0, 6));
        const carried = new MeshletResidency(device, smaller, { slotPage: first.slotPage, slotPinned: first.slotPinned, slotLastUsed: first.slotLastUsed, frame: first.frame });
        expect(carried.frame, 'the LRU clock continues').to.equal(40);
        expect(carried.residentPages, 'only the root survives').to.equal(1);
        expect(carried.freeSlots).to.have.lengthOf(3);
        expect(carried.slotPinned[world.residency[0]]).to.equal(1);
        first.destroy();
        carried.destroy();
    });
});
