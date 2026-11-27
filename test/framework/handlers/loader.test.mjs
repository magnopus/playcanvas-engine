import { expect } from 'chai';

import { ResourceLoader } from '../../../src/framework/handlers/loader.js';
import { ByteCache } from '../../../src/platform/net/byte-cache.js';
import { http } from '../../../src/platform/net/http.js';
import { MemoryByteStore } from '../../../src/platform/net/memory-byte-store.js';

describe('ResourceLoader', function () {

    afterEach(function () {
        http.byteCache = null;
    });

    describe('#byteCache', function () {

        it('sets the cache on the shared HTTP layer', function () {
            const loader = new ResourceLoader(null);
            const cache = new ByteCache(new MemoryByteStore());
            loader.byteCache = cache;
            expect(http.byteCache).to.equal(cache);
            expect(loader.byteCache).to.equal(cache);
        });

        it('clears the cache it set when destroyed', function () {
            const loader = new ResourceLoader(null);
            loader.byteCache = new ByteCache(new MemoryByteStore());
            loader.destroy();
            expect(http.byteCache).to.equal(null);
        });

        it('leaves a cache set by another loader when destroyed', function () {
            const first = new ResourceLoader(null);
            const second = new ResourceLoader(null);
            first.byteCache = new ByteCache(new MemoryByteStore());
            const cache = new ByteCache(new MemoryByteStore());
            second.byteCache = cache;
            first.destroy();
            expect(http.byteCache).to.equal(cache);
        });
    });
});
