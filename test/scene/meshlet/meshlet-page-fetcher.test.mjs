import { expect } from 'chai';

import { PAGE_TABLE, PAGE_TABLE_FIELDS } from '../../../src/scene/meshlet/constants.js';
import { MAX_RUN_BYTES, MeshletPageFetcher, RUN_GAP_BYTES } from '../../../src/scene/meshlet/streaming/meshlet-page-fetcher.js';

const PAGE = 64 * 1024;

// a manifest whose page p sits in blob `blobOf(p)` at byte offset `offsetOf(p)`
const makeManifest = (pages, blobOf, offsetOf) => {
    const pageTable = new Uint32Array(pages * PAGE_TABLE_FIELDS);
    for (let p = 0; p < pages; p++) {
        const entry = p * PAGE_TABLE_FIELDS;
        const offset = offsetOf(p);
        pageTable[entry + PAGE_TABLE.BLOB] = blobOf(p);
        pageTable[entry + PAGE_TABLE.OFFSET_LO] = offset % 0x100000000;
        pageTable[entry + PAGE_TABLE.OFFSET_HI] = Math.floor(offset / 0x100000000);
        pageTable[entry + PAGE_TABLE.BYTE_LENGTH] = PAGE;
    }
    return {
        blobs: [{ uri: 'pages_roots.dat' }, { uri: 'pages_0.dat' }],
        pageTable,
        pageCount: pages,
        pageSizeBytes: PAGE
    };
};

describe('MeshletPageFetcher', function () {

    describe('buildRuns', function () {

        it('coalesces contiguous pages into one run in shard order, with page offsets relative to the run', function () {
            const fetcher = new MeshletPageFetcher(makeManifest(6, () => 1, p => p * PAGE), 'assets/x');
            const runs = fetcher.buildRuns([3, 1, 2]);
            expect(runs).to.have.lengthOf(1);
            expect(runs[0].blob).to.equal(1);
            expect(runs[0].offset).to.equal(PAGE);
            expect(runs[0].length).to.equal(3 * PAGE);
            expect(runs[0].pages).to.deep.equal([
                { localPage: 1, byteOffset: 0 }, { localPage: 2, byteOffset: PAGE }, { localPage: 3, byteOffset: 2 * PAGE }
            ]);
        });

        it('bridges a gap up to RUN_GAP_BYTES but starts a new run beyond it', function () {
            const gap = RUN_GAP_BYTES;
            const offsets = [0, PAGE + gap, 2 * PAGE + gap + gap + 1];
            const fetcher = new MeshletPageFetcher(makeManifest(3, () => 0, p => offsets[p]), 'assets/x/');
            const runs = fetcher.buildRuns([0, 1, 2]);
            expect(runs).to.have.lengthOf(2);
            expect(runs[0].pages.map(p => p.localPage), 'page 1 rides along after the gap').to.deep.equal([0, 1]);
            expect(runs[0].length).to.equal(2 * PAGE + gap);
            expect(runs[1].pages.map(p => p.localPage)).to.deep.equal([2]);
        });

        it('caps a run at MAX_RUN_BYTES', function () {
            const pages = MAX_RUN_BYTES / PAGE + 1;
            const fetcher = new MeshletPageFetcher(makeManifest(pages, () => 0, p => p * PAGE), 'x');
            const runs = fetcher.buildRuns(Array.from({ length: pages }, (_, p) => p));
            expect(runs).to.have.lengthOf(2);
            expect(runs[0].length).to.equal(MAX_RUN_BYTES);
            expect(runs[1].pages).to.have.lengthOf(1);
        });

        it('never merges across shards', function () {
            const fetcher = new MeshletPageFetcher(makeManifest(4, p => (p < 2 ? 0 : 1), p => (p % 2) * PAGE), 'x');
            const runs = fetcher.buildRuns([0, 1, 2, 3]);
            expect(runs.map(r => r.blob)).to.deep.equal([0, 1]);
            expect(runs.map(r => r.pages.length)).to.deep.equal([2, 2]);
        });

        it('reads 64-bit shard offsets from the two table words', function () {
            const big = 5 * 0x100000000 + 3 * PAGE;
            const fetcher = new MeshletPageFetcher(makeManifest(1, () => 0, () => big), 'x');
            expect(fetcher.buildRuns([0])[0].offset).to.equal(big);
        });
    });

    describe('fetchRange', function () {

        let savedFetch;
        let calls;

        beforeEach(function () {
            savedFetch = globalThis.fetch;
            calls = [];
        });

        afterEach(function () {
            globalThis.fetch = savedFetch;
        });

        const respond = (status, byteLength) => (url, init) => {
            calls.push({ url, range: init?.headers?.Range });
            return Promise.resolve({ ok: status < 300, status, arrayBuffer: () => Promise.resolve(new ArrayBuffer(byteLength)) });
        };

        it('requests the byte range of the page\'s shard relative to the base URL', async function () {
            globalThis.fetch = respond(206, PAGE);
            const fetcher = new MeshletPageFetcher(makeManifest(1, () => 1, () => 0), 'assets/meshlets');
            const bytes = await fetcher.fetchRange(1, 3 * PAGE, PAGE);
            expect(calls[0].url).to.equal('assets/meshlets/pages_0.dat');
            expect(calls[0].range).to.equal(`bytes=${3 * PAGE}-${4 * PAGE - 1}`);
            expect(bytes.byteLength).to.equal(PAGE);
        });

        it('slices the range out of a whole-file response when the host ignores Range', async function () {
            globalThis.fetch = respond(200, 8 * PAGE);
            const fetcher = new MeshletPageFetcher(makeManifest(1, () => 0, () => 0), 'x');
            const bytes = await fetcher.fetchRange(0, 2 * PAGE, PAGE);
            expect(bytes.byteLength).to.equal(PAGE);
        });

        it('throws on an error status', async function () {
            globalThis.fetch = respond(404, 0);
            const fetcher = new MeshletPageFetcher(makeManifest(1, () => 0, () => 0), 'x');
            let error = null;
            try {
                await fetcher.fetchRange(0, 0, PAGE);
            } catch (e) {
                error = e;
            }
            expect(error?.message).to.match(/404/);
        });
    });
});
