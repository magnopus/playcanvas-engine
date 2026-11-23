import { expect } from 'chai';

import { MeshletCutTableStore } from '../../../../src/framework/components/meshlet/meshlet-cut-table-store.js';
import { ByteCache } from '../../../../src/platform/net/byte-cache.js';
import { MemoryByteStore } from '../../../../src/platform/net/memory-byte-store.js';
import { buildCutTable } from '../../../../src/scene/meshlet/meshlet-cut-table.js';
import { createCutFixture } from '../../../scene/meshlet/meshlet-cut-fixture.mjs';

const makeGroup = (url = 'https://example.com/a.glb') => ({
    resource: {
        primitives: [{ _cutTable: null, _builtCutTable: null }, { _cutTable: null, _builtCutTable: null }],
        manifest: { bakeId: 'bake-1' },
        totalMeshlets: 20
    },
    components: [{ _assetReference: { asset: { getFileUrl: () => url } } }]
});

const waitFor = async (condition) => {
    for (let i = 0; i < 100 && !condition(); i++) {
        await new Promise((resolve) => { // eslint-disable-line no-await-in-loop
            setTimeout(resolve, 1);
        });
    }
};

describe('MeshletCutTableStore', function () {
    let cache;
    let store;
    let settled;

    beforeEach(async function () {
        cache = new ByteCache(new MemoryByteStore(), { chunkSize: 64, flushDelay: 1e6 });
        await cache.ready;
        settled = 0;
        store = new MeshletCutTableStore(() => {
            settled++;
        });
        store.minMeshlets = 10;
    });

    it('saves the tables a build computed, then a later request hands them back', async function () {
        const { meshlets } = createCutFixture();
        const built = buildCutTable(meshlets);

        const first = makeGroup();
        first.resource.primitives[1]._builtCutTable = built;
        store.finish([first], cache);
        await cache.whenStored();
        expect(first.resource.primitives[1]._builtCutTable, 'dropped once saved').to.equal(null);

        const second = makeGroup();
        expect(store.request(second, cache)).to.equal(true);
        expect(store.loading).to.equal(true);
        await waitFor(() => !store.loading);
        expect(settled).to.equal(1);
        expect(second.resource.primitives[0]._cutTable).to.equal(null);
        const loaded = second.resource.primitives[1]._cutTable;
        expect(loaded.fingerprint).to.equal(built.fingerprint);
        expect(Array.from(loaded.words)).to.deep.equal(Array.from(built.words));

        // the build used them: finish drops them without writing again
        const written = cache.bytesWritten;
        store.finish([second], cache);
        expect(second.resource.primitives[1]._cutTable).to.equal(null);
        await cache.whenStored();
        expect(cache.bytesWritten).to.equal(written);
    });

    it('asks once per build, and finds nothing for a resource that was never saved', async function () {
        const group = makeGroup();
        expect(store.request(group, cache)).to.equal(true);
        await waitFor(() => !store.loading);
        expect(store.request(group, cache), 'already asked').to.equal(false);
        expect(group.resource.primitives[0]._cutTable).to.equal(null);
    });

    it('leaves a small resource alone: neither saved nor loaded', async function () {
        store.minMeshlets = 1000;
        const { meshlets } = createCutFixture();
        const group = makeGroup();
        group.resource.primitives[0]._builtCutTable = buildCutTable(meshlets);
        store.finish([group], cache);
        await cache.whenStored();
        expect(cache.entryCount).to.equal(0);
        expect(store.request(makeGroup(), cache)).to.equal(false);
    });

    it('does nothing without a byte cache or a stable key', function () {
        expect(store.request(makeGroup(), null)).to.equal(false);
        const noUrl = makeGroup(null);
        expect(store.keyOf(noUrl)).to.equal(null);
        expect(store.request(noUrl, cache)).to.equal(false);
        expect(store.loading).to.equal(false);
    });

    it('does not use tables saved for another bake of the same file', async function () {
        const { meshlets } = createCutFixture();
        const old = makeGroup();
        old.resource.primitives[0]._builtCutTable = buildCutTable(meshlets);
        store.finish([old], cache);
        await cache.whenStored();

        const rebaked = makeGroup();
        rebaked.resource.manifest.bakeId = 'bake-2';
        expect(store.request(rebaked, cache)).to.equal(true);
        await waitFor(() => !store.loading);
        expect(rebaked.resource.primitives[0]._cutTable).to.equal(null);
    });
});
