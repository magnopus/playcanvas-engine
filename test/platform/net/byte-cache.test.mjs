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

const settle = () => new Promise((resolve) => {
    setTimeout(resolve, 0);
});

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

        it('stores a file larger than maxFileBytes in windows, the size only once all are stored', async function () {
            const cache = await makeCache(undefined, { maxFileBytes: 32, chunkSize: 1024 * 1024 });
            // 20 MiB: three windows of 8 MiB
            const data = new Uint8Array(20 * 1024 * 1024 + 5);
            data.forEach((_, i) => {
                data[i] = i & 0xff;
            });
            const stored = cache.putFile(URL_A, data);
            expect(cache.hasFile(URL_A)).to.equal(false);
            await stored;
            expect(cache.hasFile(URL_A)).to.equal(true);
            const hit = await cache.getFile(URL_A);
            expect(hit.data.byteLength).to.equal(data.byteLength);
            expect(new Uint8Array(hit.data, 8 * 1024 * 1024 - 3, 8)).to.deep.equal(data.subarray(8 * 1024 * 1024 - 3, 8 * 1024 * 1024 + 5));
        });

        it('does not serve a large file whose buffer was transferred away before it was stored', async function () {
            const cache = await makeCache(undefined, { maxFileBytes: 32, chunkSize: 1024 * 1024 });
            const data = new Uint8Array(20 * 1024 * 1024);
            const stored = cache.putFile(URL_A, data);
            structuredClone(data.buffer, { transfer: [data.buffer] });
            await stored;
            expect(cache.hasFile(URL_A)).to.equal(false);
        });

        it('serves ranges of a large stored file, which is one value in the store', async function () {
            const store = new MemoryByteStore();
            const cache = await makeCache(store, { maxFileBytes: 32 });
            await cache.putFile(URL_A, bytes(100));
            expect((await store.list()).filter(n => n.endsWith('.w'))).to.have.lengthOf(1);
            expect(cache.hasRange(URL_A, 40, 30)).to.equal(true);
            expect(new Uint8Array(await cache.getRange(URL_A, 40, 30))).to.deep.equal(bytes(30, 40));
            expect(cache.hasRange(URL_A, 90, 20)).to.equal(false);
        });

        it('finds a large stored file again after a reload, and the sweep keeps its value', async function () {
            const store = new MemoryByteStore();
            const first = await makeCache(store, { maxFileBytes: 32 });
            await first.putFile(URL_A, bytes(100));
            await first.flush();
            const second = await makeCache(store, { maxFileBytes: 32 });
            await settle();
            expect(second.hasFile(URL_A)).to.equal(true);
            expect(new Uint8Array((await second.getFile(URL_A)).data)).to.deep.equal(bytes(100));
            expect(second.totalBytes).to.equal(100);
        });

        it('evicts a large stored file with its value', async function () {
            const store = new MemoryByteStore();
            const cache = await makeCache(store, { maxFileBytes: 32, maxBytes: 150 });
            await cache.putFile(URL_A, bytes(100));
            await cache.putFile(URL_B, bytes(100));
            await settle();
            expect(cache.hasFile(URL_A)).to.equal(false);
            expect(cache.hasFile(URL_B)).to.equal(true);
            expect((await store.list()).filter(n => n.endsWith('.w'))).to.have.lengthOf(1);
        });

        it('whenStored waits for a large file to be written', async function () {
            const cache = await makeCache(undefined, { maxFileBytes: 32 });
            cache.putFile(URL_A, bytes(100));
            await cache.whenStored();
            expect(cache.hasFile(URL_A)).to.equal(true);
        });

        it('copies a file up to maxFileBytes before returning', async function () {
            const cache = await makeCache(undefined, { maxFileBytes: 32 });
            const data = bytes(32);
            const stored = cache.putFile(URL_B, data);
            data.fill(0);
            await stored;
            expect(new Uint8Array((await cache.getFile(URL_B)).data)).to.deep.equal(bytes(32));
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

        it('writes the index on destroy, then stops serving lookups', async function () {
            const store = new MemoryByteStore();
            const first = await makeCache(store);
            await first.putFile(URL_A, bytes(40));
            await first.destroy();
            expect(first.hasFile(URL_A)).to.equal(false);

            const second = await makeCache(store);
            expect(second.hasFile(URL_A)).to.equal(true);
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

        it('is ready once the index loads, without waiting for the orphan sweep', async function () {
            const store = new MemoryByteStore();
            const first = await makeCache(store);
            await first.putFile(URL_A, bytes(40));
            await first.flush();

            let releaseList;
            const list = store.list.bind(store);
            store.list = () => new Promise((resolve) => {
                releaseList = () => resolve(list());
            });
            const second = await makeCache(store);
            expect(second.hasFile(URL_A), 'served while the sweep is still listing').to.equal(true);
            releaseList();
            await settle();
            expect(second.hasFile(URL_A)).to.equal(true);
            expect(await second.getFile(URL_A)).to.not.equal(null);
        });

        it('keeps chunks written while the sweep runs, even over an orphan of the same name', async function () {
            const store = new MemoryByteStore();
            const first = await makeCache(store);
            await first.putFile(URL_A, bytes(40));
            // never flushed: URL_A's chunks are orphans to the next session

            // the sweep's listing arrives at once, its deletions only when released
            const removals = [];
            const remove = store.remove.bind(store);
            store.remove = name => new Promise((resolve) => {
                removals.push(() => resolve(remove(name)));
            });
            const second = await makeCache(store);
            await settle();
            expect(removals.length, 'every orphan is being deleted').to.equal(3);

            const stored = second.putFile(URL_A, bytes(40, 7));
            for (const release of removals) release();
            await stored;
            await settle();
            expect(second.hasFile(URL_A)).to.equal(true);
            expect(new Uint8Array((await second.getFile(URL_A)).data)).to.deep.equal(bytes(40, 7));
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

    describe('store backlog', function () {

        /**
         * A store whose writes wait until released, standing in for one slower than the network.
         */
        class SlowStore extends MemoryByteStore {
            gate = null;

            release = () => {};

            hold() {
                this.gate = new Promise((resolve) => {
                    this.release = resolve;
                });
            }

            async write(name, data) {
                if (this.gate && name !== 'index-v1') await this.gate;
                return super.write(name, data);
            }
        }

        it('drops a response while the store is maxPendingBytes behind, then caches again', async function () {
            const store = new SlowStore();
            const cache = await makeCache(store, { maxPendingBytes: 48 });
            store.hold();
            const first = cache.putFile(URL_A, bytes(32));
            // 32 pending + 32 more is over the cap
            await cache.putFile(URL_B, bytes(32));
            expect(cache.droppedWrites).to.equal(1);
            store.release();
            await first;
            expect(cache.hasFile(URL_A)).to.equal(true);
            expect(cache.hasFile(URL_B)).to.equal(false);
            // caught up: stored again
            await cache.putFile(URL_B, bytes(32));
            expect(cache.hasFile(URL_B)).to.equal(true);
        });

        it('admits a write larger than the cap when nothing is pending', async function () {
            const cache = await makeCache(new MemoryByteStore(), { maxPendingBytes: 16 });
            await cache.putFile(URL_A, bytes(40));
            expect(cache.hasFile(URL_A)).to.equal(true);
            expect(cache.droppedWrites).to.equal(0);
        });

        it('counts the bytes of chunks the store detached', async function () {
            // a store that takes ownership the way a worker store does: by transferring the buffer
            class DetachingStore extends MemoryByteStore {
                write(name, data) {
                    const copy = data.slice();
                    structuredClone(data.buffer, { transfer: [data.buffer] });
                    return super.write(name, copy);
                }
            }
            const cache = await makeCache(new DetachingStore());
            await cache.putFile(URL_A, bytes(40));
            expect(cache.totalBytes).to.equal(40);
            const hit = await cache.getFile(URL_A);
            expect(new Uint8Array(hit.data)).to.deep.equal(bytes(40));
        });
    });

    describe('#lendFile', function () {

        /**
         * A store that takes a lent buffer and hands back a copy, as the worker store does.
         */
        class TakingStore extends MemoryByteStore {
            lost = false;

            writeLent(name, buffer) {
                const copy = buffer.slice(0);
                const taken = structuredClone(buffer, { transfer: [buffer] });
                return {
                    returned: this.lost ? Promise.reject(new Error('lost')) : Promise.resolve(taken),
                    stored: this.write(name, new Uint8Array(copy))
                };
            }
        }

        it('stores a large file and hands back a buffer with the same bytes', async function () {
            const cache = await makeCache(new TakingStore(), { maxFileBytes: 32 });
            const lent = bytes(100).buffer;
            const returned = await cache.lendFile(URL_A, lent, 'model/gltf-binary');
            expect(lent.byteLength).to.equal(0);
            expect(new Uint8Array(returned)).to.deep.equal(bytes(100));
            await cache.whenStored();
            const hit = await cache.getFile(URL_A);
            expect(new Uint8Array(hit.data)).to.deep.equal(bytes(100));
            expect(hit.type).to.equal('model/gltf-binary');
        });

        it('stores a small file the usual way, returning the same buffer', async function () {
            const cache = await makeCache(new TakingStore(), { maxFileBytes: 32 });
            const lent = bytes(20).buffer;
            expect(await cache.lendFile(URL_A, lent)).to.equal(lent);
            await cache.whenStored();
            expect(cache.hasFile(URL_A)).to.equal(true);
        });

        it('returns the buffer untouched when the file is already stored', async function () {
            const cache = await makeCache(new TakingStore(), { maxFileBytes: 32 });
            await cache.lendFile(URL_A, bytes(100).buffer);
            await cache.whenStored();
            const lent = bytes(100).buffer;
            expect(await cache.lendFile(URL_A, lent)).to.equal(lent);
        });

        it('rejects when the store loses the buffer', async function () {
            const store = new TakingStore();
            store.lost = true;
            const cache = await makeCache(store, { maxFileBytes: 32 });
            let error = null;
            await cache.lendFile(URL_A, bytes(100).buffer).catch((err) => {
                error = err;
            });
            expect(error?.message).to.equal('lost');
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
