import { expect } from 'chai';

import { ByteCache } from '../../../src/platform/net/byte-cache.js';
import { MemoryByteStore } from '../../../src/platform/net/memory-byte-store.js';
import { MeshletFetchScheduler } from '../../../src/scene/meshlet/streaming/meshlet-fetch-scheduler.js';

const CHUNK = 64;
const URL = 'https://example.com/pages_0.dat';
const FILE = Uint8Array.from({ length: 1000 }, (_, i) => i & 0xff);

describe('MeshletFetchScheduler with a ByteCache', function () {

    let savedFetch;
    let calls;

    beforeEach(function () {
        savedFetch = globalThis.fetch;
        calls = [];
        // a host honouring Range requests, clamped to the file's end like a real server
        globalThis.fetch = (url, init) => {
            const range = init?.headers?.Range;
            calls.push(range ?? 'all');
            let body = FILE;
            let status = 200;
            if (range) {
                const [, a, b] = /bytes=(\d+)-(\d+)/.exec(range);
                body = FILE.slice(Number(a), Math.min(Number(b) + 1, FILE.length));
                status = 206;
            }
            return Promise.resolve({ ok: true, status, arrayBuffer: () => Promise.resolve(body.slice().buffer) });
        };
    });

    afterEach(function () {
        globalThis.fetch = savedFetch;
    });

    const makeScheduler = async () => {
        const cache = new ByteCache(new MemoryByteStore(), { chunkSize: CHUNK, flushDelay: 1e6 });
        await cache.ready;
        const scheduler = new MeshletFetchScheduler();
        scheduler.byteCache = cache;
        return { scheduler, cache };
    };

    // writes land asynchronously after the fetch resolves
    const settle = () => new Promise((resolve) => {
        setTimeout(resolve, 0);
    });

    it('widens a range request to the cache\'s chunks and still returns the requested bytes', async function () {
        const { scheduler } = await makeScheduler();
        const bytes = await scheduler.fetchRange(URL, 100, 50);
        expect(calls).to.deep.equal(['bytes=64-191']);
        expect(new Uint8Array(bytes)).to.deep.equal(FILE.slice(100, 150));
    });

    it('resolves a view into the chunk-aligned response instead of copying the range out', async function () {
        const { scheduler } = await makeScheduler();
        const bytes = await scheduler.fetchRange(URL, 100, 50, { view: true });
        expect(bytes).to.be.instanceOf(Uint8Array);
        expect(bytes.buffer.byteLength, 'the widened 64-191 response').to.equal(128);
        expect(bytes.byteOffset).to.equal(36);
        expect(bytes).to.deep.equal(FILE.slice(100, 150));
    });

    it('answers a repeated range from the cache without a fetch', async function () {
        const { scheduler } = await makeScheduler();
        await scheduler.fetchRange(URL, 100, 50);
        await settle();
        const bytes = await scheduler.fetchRange(URL, 70, 100);
        expect(calls.length).to.equal(1);
        expect(scheduler.cacheHits).to.equal(1);
        expect(new Uint8Array(bytes)).to.deep.equal(FILE.slice(70, 170));
    });

    it('learns the file size from a range that runs past the end', async function () {
        const { scheduler, cache } = await makeScheduler();
        await scheduler.fetchRange(URL, 980, 20);
        await settle();
        expect(cache.hasRange(URL, 960, 40)).to.equal(true);
        expect(cache.alignRange(URL, 980, 20)).to.deep.equal({ offset: 960, length: 40 });
    });

    it('keys cached bytes by the cache version, so a rebake behind the same URL misses', async function () {
        const { scheduler } = await makeScheduler();
        await scheduler.fetchRange(URL, 0, 64, { cacheVersion: 'bake-a' });
        await settle();
        await scheduler.fetchRange(URL, 0, 64, { cacheVersion: 'bake-a' });
        expect(calls.length, 'same bake: served from the cache').to.equal(1);
        await scheduler.fetchRange(URL, 0, 64, { cacheVersion: 'bake-b' });
        expect(calls.length, 'new bake: fetched again').to.equal(2);
    });

    it('caches a whole-file fetch for later ranges', async function () {
        const { scheduler } = await makeScheduler();
        await scheduler.fetchAll(URL);
        await settle();
        const bytes = await scheduler.fetchRange(URL, 500, 300);
        expect(calls).to.deep.equal(['all']);
        expect(new Uint8Array(bytes)).to.deep.equal(FILE.slice(500, 800));
    });

    it('caps the cache reads in flight and serves the most urgent first', async function () {
        const { scheduler, cache } = await makeScheduler();
        await scheduler.fetchAll(URL);
        await settle();
        expect(cache.hasFile(URL)).to.equal(true);

        scheduler.maxConcurrentCacheReads = 2;
        let active = 0;
        let peak = 0;
        const originalGetRange = cache.getRange.bind(cache);
        cache.getRange = async (...args) => {
            active++;
            peak = Math.max(peak, active);
            await settle();
            const result = await originalGetRange(...args);
            active--;
            return result;
        };
        const order = [];
        const requests = [
            scheduler.fetchRange(URL, 0, 10, { priority: 3 }).then(() => order.push('fine-a')),
            scheduler.fetchRange(URL, 10, 10, { priority: 3 }).then(() => order.push('fine-b')),
            scheduler.fetchRange(URL, 20, 10, { priority: 3 }).then(() => order.push('fine-c')),
            scheduler.fetchRange(URL, 30, 10, { priority: 0 }).then(() => order.push('roots'))
        ];
        await Promise.all(requests);
        expect(peak).to.equal(2);
        // the two reads already started finish first; the roots read overtakes the waiting fine one
        expect(order.indexOf('roots')).to.be.lessThan(order.indexOf('fine-c'));
        expect(calls).to.deep.equal(['all']);
    });

    it('does not read a cached range whose request is no longer wanted', async function () {
        const { scheduler, cache } = await makeScheduler();
        await scheduler.fetchAll(URL);
        await settle();
        scheduler.maxConcurrentCacheReads = 1;
        let wanted = true;
        const first = scheduler.fetchRange(URL, 0, 10);
        const second = scheduler.fetchRange(URL, 10, 10, { stillWanted: () => wanted });
        wanted = false;
        await first;
        expect(await second).to.equal(null);
        expect(cache.hits).to.be.lessThan(3);
    });
});
