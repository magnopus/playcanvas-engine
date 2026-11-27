import { expect } from 'chai';

import { meshletBakeStamp } from '../../../src/scene/meshlet/streaming/meshlet-bake-stamp.js';

describe('meshletBakeStamp', function () {

    const resource = (pageTable, extra = {}) => ({
        manifest: { pageTable: Uint32Array.from(pageTable), blobs: [{ uri: 'x_meshlets/pages_0.dat' }], ...extra },
        textureManifest: null
    });

    it('uses the bake id when the bake has one', function () {
        expect(meshletBakeStamp(resource([1, 2, 3], { bakeId: 'abc-123' }))).to.equal('abc-123');
    });

    it('hashes the page table of bakes without one: equal bakes match, any change differs', function () {
        const a = meshletBakeStamp(resource([0, 0, 65536, 1, 0, 65536]));
        expect(meshletBakeStamp(resource([0, 0, 65536, 1, 0, 65536]))).to.equal(a);
        expect(meshletBakeStamp(resource([0, 0, 65536, 1, 65536, 65536]))).to.not.equal(a);
    });

    it('includes the texture containers', function () {
        const plain = resource([7]);
        const textured = { ...resource([7]), textureManifest: { arrays: [{ container: { uri: 't.ktx2' } }] } };
        expect(meshletBakeStamp(textured)).to.not.equal(meshletBakeStamp(plain));
    });
});
