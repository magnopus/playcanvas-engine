import { expect } from 'chai';

import { MeshletFetchScheduler } from '../../../src/scene/meshlet/streaming/meshlet-fetch-scheduler.js';

const URL = 'https://example.com/pages_0.dat';
const FILE = Uint8Array.from({ length: 1000 }, (_, i) => i & 0xff);

describe('MeshletFetchScheduler', function () {

    let savedFetch;
    let savedWarn;
    let calls;

    // a host serving FILE: honouring Range (clamped to the file's end) unless told to ignore it
    const host = ({ ignoreRange = false, status = 0 } = {}) => (url, init) => {
        const range = init?.headers?.Range;
        calls.push(range ?? 'all');
        if (status) {
            return Promise.resolve({ ok: false, status, arrayBuffer: () => Promise.resolve(new ArrayBuffer(0)) });
        }
        let body = FILE;
        let code = 200;
        if (range && !ignoreRange) {
            const [, a, b] = /bytes=(\d+)-(\d+)/.exec(range);
            body = FILE.slice(Number(a), Math.min(Number(b) + 1, FILE.length));
            code = 206;
        }
        return Promise.resolve({ ok: true, status: code, arrayBuffer: () => Promise.resolve(body.slice().buffer) });
    };

    const rejection = async (promise) => {
        try {
            await promise;
        } catch (err) {
            return err;
        }
        return null;
    };

    beforeEach(function () {
        savedFetch = globalThis.fetch;
        savedWarn = console.warn;
        console.warn = () => {};
        calls = [];
    });

    afterEach(function () {
        globalThis.fetch = savedFetch;
        console.warn = savedWarn;
    });

    it('resolves merged ranges as views of the one response when asked, as copies otherwise', async function () {
        globalThis.fetch = host();
        const scheduler = new MeshletFetchScheduler(1);
        // the first request takes the only slot, so the next two queue and merge
        const first = scheduler.fetchRange(URL, 0, 10);
        const a = scheduler.fetchRange(URL, 100, 50, { view: true });
        const b = scheduler.fetchRange(URL, 200, 50);
        const [, viewA, copyB] = await Promise.all([first, a, b]);
        expect(calls).to.deep.equal(['bytes=0-9', 'bytes=100-249']);
        expect(viewA).to.be.instanceOf(Uint8Array);
        expect(viewA.byteOffset).to.equal(0);
        expect(viewA.buffer.byteLength, 'shares the merged response').to.equal(150);
        expect(viewA).to.deep.equal(FILE.slice(100, 150));
        expect(ArrayBuffer.isView(copyB), 'an ArrayBuffer of its own').to.equal(false);
        expect(copyB.byteLength).to.equal(50);
        expect(new Uint8Array(copyB)).to.deep.equal(FILE.slice(200, 250));
    });

    it('rejects a range the response ends before', async function () {
        globalThis.fetch = host();
        const scheduler = new MeshletFetchScheduler();
        const err = await rejection(scheduler.fetchRange(URL, 990, 20, { view: true }));
        expect(err?.message).to.match(/short/);
    });

    it('tags an HTTP error with its status', async function () {
        globalThis.fetch = host({ status: 503 });
        const scheduler = new MeshletFetchScheduler();
        const err = await rejection(scheduler.fetchRange(URL, 0, 10));
        expect(err?.status).to.equal(503);
    });

    it('keeps the whole file of a host that ignores Range and serves later ranges from it', async function () {
        globalThis.fetch = host({ ignoreRange: true });
        const scheduler = new MeshletFetchScheduler();
        const first = await scheduler.fetchRange(URL, 100, 50);
        expect(new Uint8Array(first)).to.deep.equal(FILE.slice(100, 150));
        const view = await scheduler.fetchRange(URL, 500, 20, { view: true });
        const copy = await scheduler.fetchRange(URL, 600, 20);
        expect(calls.length, 'one download').to.equal(1);
        expect(view).to.deep.equal(FILE.slice(500, 520));
        expect(new Uint8Array(copy)).to.deep.equal(FILE.slice(600, 620));
        // a copy, so a consumer that transfers it cannot detach the kept file
        expect(copy.byteLength).to.equal(20);

        scheduler.clear();
        await scheduler.fetchRange(URL, 0, 10);
        expect(calls.length, 'cleared: fetched again').to.equal(2);
    });

    it('serves ranges queued before the host turned out to ignore Range from the kept file', async function () {
        globalThis.fetch = host({ ignoreRange: true });
        const scheduler = new MeshletFetchScheduler(1);
        const first = scheduler.fetchRange(URL, 0, 10);
        // far apart, so it does not merge into the first request
        const queued = scheduler.fetchRange(URL, 900, 10, { view: true });
        const [, bytes] = await Promise.all([first, queued]);
        expect(calls.length).to.equal(1);
        expect(bytes).to.deep.equal(FILE.slice(900, 910));
    });
});
