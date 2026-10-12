import { expect } from 'chai';

import { ByteCache } from '../../../src/platform/net/byte-cache.js';
import { MemoryByteStore } from '../../../src/platform/net/memory-byte-store.js';

const CHUNK = 16;
const URL_A = 'https://example.com/a.bin';
const URL_B = 'https://example.com/b.bin';

/**
 * @param {number} length - Byte count.
 * @param {number} [offset] - Value of the first byte; each following byte counts up from it.
 * @returns {Uint8Array} Bytes whose values identify their position.
 */
const bytes = (length, offset = 0) => Uint8Array.from({ length }, (_, i) => (offset + i) & 0xff);

const makeCache = async (store = new MemoryByteStore(), options = {}) => {
    const cache = new ByteCache(store, { chunkSize: CHUNK, flushDelay: 1e6, ...options });
    await cache.ready;
    return cache;
};

describe('ByteCache', function () {

    describe('#putFile / #getFile', function () {

        it('round-trips a whole file and its content type', async function () {
            const cache = await makeCache();
            await cache.putFile(URL_A, bytes(40), 'model/gltf-binary');
            expect(cache.hasFile(URL_A)).to.equal(true);
            const hit = await cache.getFile(URL_A);
            expect(new Uint8Array(hit.data)).to.deep.equal(bytes(40));
            expect(hit.type).to.equal('model/gltf-binary');
            expect(cache.totalBytes).to.equal(40);
            expect(cache.hits).to.equal(1);
        });

        it('round-trips an empty file', async function () {
            const cache = await makeCache();
            await cache.putFile(URL_A, new Uint8Array(0));
            const hit = await cache.getFile(URL_A);
            expect(hit.data.byteLength).to.equal(0);
        });

        it('misses a file never stored', async function () {
            const cache = await makeCache();
            expect(cache.hasFile(URL_A)).to.equal(false);
            expect(await cache.getFile(URL_A)).to.equal(null);
        });

        it('copies the bytes, so the caller may reuse its buffer at once', async function () {
            const cache = await makeCache();
            const data = bytes(32);
            const stored = cache.putFile(URL_A, data);
            data.fill(0);
            await stored;
            expect(new Uint8Array((await cache.getFile(URL_A)).data)).to.deep.equal(bytes(32));
        });

        it('does not cache URLs the key function rejects', async function () {
            const cache = await makeCache(undefined, { key: url => (url.includes('example.com') ? url : null) });
            await cache.putFile('https://other.com/x.bin', bytes(8));
            expect(cache.accepts('https://other.com/x.bin')).to.equal(false);
            expect(cache.hasFile('https://other.com/x.bin')).to.equal(false);
        });

        it('does not cache data or blob URLs by default', async function () {
            const cache = await makeCache();
            expect(cache.accepts('data:application/octet-stream;base64,AAAA')).to.equal(false);
            expect(cache.accepts('blob:https://example.com/1234')).to.equal(false);
        });

        it('shares an entry between URLs the key function maps together', async function () {
            const cache = await makeCache(undefined, { key: url => url.split('?')[0] });
            await cache.putFile(`${URL_A}?sig=1`, bytes(20));
            expect(cache.hasFile(`${URL_A}?sig=2`)).to.equal(true);
        });
    });

    describe('#putRange / #getRange', function () {

        it('stores only the chunks a range covers completely', async function () {
            const cache = await makeCache();
            // bytes 8..40 cover chunk 1 (16..32) only
            await cache.putRange(URL_A, 8, bytes(32, 8));
            expect(cache.hasRange(URL_A, 16, 16)).to.equal(true);
            expect(cache.hasRange(URL_A, 8, 16)).to.equal(false);
            expect(cache.hasRange(URL_A, 32, 8)).to.equal(false);
            expect(cache.totalBytes).to.equal(16);
        });

        it('serves a range spanning chunks stored by different requests', async function () {
            const cache = await makeCache();
            await cache.putRange(URL_A, 0, bytes(16));
            await cache.putRange(URL_A, 16, bytes(32, 16));
            const data = await cache.getRange(URL_A, 5, 30);
            expect(new Uint8Array(data)).to.deep.equal(bytes(30, 5));
        });

        it('stores the partial last chunk once a short response fixes the size', async function () {
            const cache = await makeCache();
            await cache.putRange(URL_A, 32, bytes(10, 32), true);
            expect(cache.hasRange(URL_A, 32, 10)).to.equal(true);
            expect(cache.hasRange(URL_A, 32, 11)).to.equal(false);
            await cache.putRange(URL_A, 0, bytes(32));
            expect(cache.hasFile(URL_A)).to.equal(true);
            expect(new Uint8Array((await cache.getFile(URL_A)).data)).to.deep.equal(bytes(42));
        });

        it('serves a range of a file stored whole', async function () {
            const cache = await makeCache();
            await cache.putFile(URL_A, bytes(50));
            expect(new Uint8Array(await cache.getRange(URL_A, 20, 25))).to.deep.equal(bytes(25, 20));
        });

        it('replaces an entry whose size changed', async function () {
            const cache = await makeCache();
            await cache.putFile(URL_A, bytes(40));
            await cache.putFile(URL_A, bytes(20, 100));
            expect(new Uint8Array((await cache.getFile(URL_A)).data)).to.deep.equal(bytes(20, 100));
            expect(cache.totalBytes).to.equal(20);
        });

        it('forgets a chunk that vanished from the store', async function () {
            const store = new MemoryByteStore();
            const cache = await makeCache(store);
            await cache.putFile(URL_A, bytes(48));
            const names = (await store.list()).filter(name => name.endsWith('.1'));
            await store.remove(names[0]);
            expect(await cache.getRange(URL_A, 0, 48)).to.equal(null);
            expect(cache.hasRange(URL_A, 16, 16)).to.equal(false);
            expect(cache.hasRange(URL_A, 0, 16)).to.equal(true);
            expect(cache.totalBytes).to.equal(32);
        });
    });

    describe('#alignRange', function () {

        it('widens a range to whole chunks', async function () {
            const cache = await makeCache();
            expect(cache.alignRange(URL_A, 20, 20)).to.deep.equal({ offset: 16, length: 32 });
        });

        it('clamps to a known file size', async function () {
            const cache = await makeCache();
            await cache.putRange(URL_A, 32, bytes(10, 32), true);
            expect(cache.alignRange(URL_A, 20, 20)).to.deep.equal({ offset: 16, length: 26 });
        });
    });

    describe('eviction', function () {

        it('evicts the least recently used URLs beyond maxBytes', async function () {
            const cache = await makeCache(undefined, { maxBytes: 64 });
            await cache.putFile(URL_A, bytes(48));
            await new Promise((resolve) => {
                setTimeout(resolve, 2);
            });
            await cache.putFile(URL_B, bytes(48));
            expect(cache.hasFile(URL_A)).to.equal(false);
            expect(cache.hasFile(URL_B)).to.equal(true);
            expect(cache.totalBytes).to.equal(48);
        });

        it('lowers maxBytes to what fits when the store runs out of quota', async function () {
            const store = new MemoryByteStore();
            const write = store.write.bind(store);
            let budget = 64;
            store.write = (name, data) => {
                if (name.startsWith('index')) return write(name, data);
                budget -= data.byteLength;
                if (budget < 0) {
                    const err = new Error('full');
                    err.name = 'QuotaExceededError';
                    return Promise.reject(err);
                }
                return write(name, data);
            };
            const cache = await makeCache(store);
            await cache.putFile(URL_A, bytes(48));
            await cache.putFile(URL_B, bytes(48));
            expect(cache.maxBytes).to.be.below(96);
            expect(cache.totalBytes).to.be.at.most(cache.maxBytes);
        });
    });

    describe('persistence', function () {

        it('finds what an earlier cache on the same store flushed', async function () {
            const store = new MemoryByteStore();
            const first = await makeCache(store);
            await first.putFile(URL_A, bytes(40), 'application/octet-stream');
            await first.putRange(URL_B, 16, bytes(16, 16));
            await first.flush();

            const second = await makeCache(store);
            expect(second.hasFile(URL_A)).to.equal(true);
            expect(second.hasRange(URL_B, 16, 16)).to.equal(true);
            expect(second.totalBytes).to.equal(56);
            expect((await second.getFile(URL_A)).type).to.equal('application/octet-stream');
        });

        it('deletes chunks no index references', async function () {
            const store = new MemoryByteStore();
            const first = await makeCache(store);
            await first.putFile(URL_A, bytes(40));
            // never flushed: the chunks are orphans to the next session

            const second = await makeCache(store);
            expect(second.hasFile(URL_A)).to.equal(false);
            await new Promise((resolve) => {
                setTimeout(resolve, 0);
            });
            expect(await store.list()).to.deep.equal([]);
        });

        it('merges entries another cache on the same store flushed', async function () {
            const store = new MemoryByteStore();
            const tabA = await makeCache(store);
            const tabB = await makeCache(store);
            await tabA.putFile(URL_A, bytes(20));
            await tabA.flush();
            await tabB.putFile(URL_B, bytes(20));
            await tabB.flush();

            expect(tabB.hasFile(URL_A)).to.equal(true);
            const third = await makeCache(store);
            expect(third.hasFile(URL_A)).to.equal(true);
            expect(third.hasFile(URL_B)).to.equal(true);
        });

        it('discards a cache stored with a different chunk size', async function () {
            const store = new MemoryByteStore();
            const first = await makeCache(store);
            await first.putFile(URL_A, bytes(40));
            await first.flush();

            const second = await makeCache(store, { chunkSize: CHUNK * 2 });
            expect(second.hasFile(URL_A)).to.equal(false);
            expect(await store.list()).to.deep.equal([]);
        });

        it('disables itself when the store fails to open', async function () {
            const store = new MemoryByteStore();
            store.open = () => Promise.reject(new Error('unavailable'));
            const cache = await makeCache(store);
            expect(cache.accepts(URL_A)).to.equal(false);
            await cache.putFile(URL_A, bytes(8));
            expect(cache.hasFile(URL_A)).to.equal(false);
        });
    });

    describe('#clear', function () {

        it('empties the cache and its store', async function () {
            const store = new MemoryByteStore();
            const cache = await makeCache(store);
            await cache.putFile(URL_A, bytes(40));
            await cache.flush();
            await cache.clear();
            expect(cache.hasFile(URL_A)).to.equal(false);
            expect(cache.totalBytes).to.equal(0);
            expect(await store.list()).to.deep.equal([]);
        });
    });
});
