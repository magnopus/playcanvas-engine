import { expect } from 'chai';

import { Mat4 } from '../../../src/core/math/mat4.js';
import { MeshletInstanceList, MeshletInstanceListBuilder } from '../../../src/scene/meshlet/meshlet-instance-list.js';

const translation = (x, y, z) => {
    const m = new Mat4();
    m.data[12] = x;
    m.data[13] = y;
    m.data[14] = z;
    return m.data;
};

describe('MeshletInstanceList', function () {

    it('stores one transform per node, shared by the primitives that use it', function () {
        const builder = new MeshletInstanceListBuilder(1);
        const lightmap = { texture: { arrayId: 0, layer: 0 } };
        const a = builder.addTransform(translation(1, 2, 3));
        const b = builder.addTransform(translation(4, 5, 6), lightmap);
        builder.add(0, a);
        builder.add(1, a);
        builder.add(0, b);
        const list = MeshletInstanceList.from(builder);
        expect(list.count).to.equal(3);
        expect(list.transforms).to.have.lengthOf(2 * 12);
        expect(Array.from(list.primIndex)).to.deep.equal([0, 1, 0]);
        const m = list.getMatrix(2, new Mat4());
        expect(Array.from(m.data)).to.deep.equal(Array.from(translation(4, 5, 6)));
        expect(list.getLightmap(0)).to.equal(null);
        expect(list.getLightmap(2)).to.equal(lightmap);
    });

    it('packs placement arrays, sharing transforms between placements of one matrix', function () {
        const shared = translation(1, 0, 0);
        const list = MeshletInstanceList.from([
            { primIndex: 0, matrix: shared },
            { primIndex: 1, matrix: shared },
            { primIndex: 0, matrix: translation(2, 0, 0) }
        ]);
        expect(list.transforms).to.have.lengthOf(2 * 12);
        expect(Array.from(list.transformIndex)).to.deep.equal([0, 0, 1]);
        expect(list.get(2)).to.deep.equal({ primIndex: 0, matrix: translation(2, 0, 0), lightmap: null });
        expect(MeshletInstanceList.from(list), 'a list is returned as is').to.equal(list);
        expect(MeshletInstanceList.from(null).count).to.equal(0);
    });

    it('filters and slices without copying the transforms', function () {
        const list = MeshletInstanceList.from([0, 1, 2, 3].map(i => ({ primIndex: i % 2, matrix: translation(i, 0, 0) })));
        const odd = list.filter(primIndex => primIndex === 1);
        expect(odd.count).to.equal(2);
        expect(odd.transforms).to.equal(list.transforms);
        expect(odd.get(1).matrix[12]).to.equal(3);
        const tail = list.slice(2);
        expect(Array.from(tail.transformIndex)).to.deep.equal([2, 3]);
    });
});
