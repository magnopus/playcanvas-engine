import { expect } from 'chai';

import { NumericIds } from '../../src/core/numeric-ids.js';

describe('NumericIds', function () {

    it('hands out sequential ids from zero', function () {
        const ids = new NumericIds();
        expect([ids.get(), ids.get(), ids.get()]).to.deep.equal([0, 1, 2]);
    });

    it('keeps independent counters per instance', function () {
        const a = new NumericIds();
        const b = new NumericIds();
        a.get();
        a.get();
        expect(b.get()).to.equal(0);
    });

    it('reserves a contiguous block and continues after it', function () {
        const ids = new NumericIds();
        expect(ids.get()).to.equal(0);
        const first = ids.reserve(3);
        expect(first).to.equal(1);
        expect(ids.get(), 'next id follows the block').to.equal(4);
        expect(ids.reserve(2)).to.equal(5);
        expect(ids.get()).to.equal(7);
    });

    it('treats a non-positive reservation as empty', function () {
        const ids = new NumericIds();
        expect(ids.reserve(0)).to.equal(0);
        expect(ids.reserve(-5)).to.equal(0);
        expect(ids.get()).to.equal(0);
    });
});
