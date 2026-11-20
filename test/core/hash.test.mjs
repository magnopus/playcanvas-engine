import { expect } from 'chai';

import { hashCode, hash32Fnv1a, hash53 } from '../../src/core/hash.js';

describe('hashCode', function () {

    it('returns 0 for the empty string', function () {
        expect(hashCode('')).to.equal(0);
    });

    it('returns the same hash for the same string', function () {
        expect(hashCode('abc')).to.equal(hashCode('abc'));
    });

    it('returns different hashes for different strings', function () {
        expect(hashCode('abc')).to.not.equal(hashCode('def'));
    });

});

describe('[1, 2, 3]', function () {

    it('returns the same hash for the same arrays', function () {
        expect(hash32Fnv1a([1, 2, 3])).to.equal(hash32Fnv1a([1, 2, 3]));
    });

    it('returns different hashes for different arrays', function () {
        expect(hash32Fnv1a([1, 2, 3])).to.not.equal(hash32Fnv1a([3, 2, 1]));
    });

    it('returns a 32bit unsigned integer', function () {
        const hash = hash32Fnv1a([0xffffffff, 0x80000000, 12345]);
        expect(Number.isInteger(hash)).to.equal(true);
        expect(hash).to.be.at.least(0);
        expect(hash).to.be.below(2 ** 32);
    });

    it('multiplies in 32bit integer arithmetic', function () {
        // FNV-1a over 32bit words, computed exactly - a floating point multiply exceeds the 53bit
        // mantissa for these values, and so differs from it
        const reference = (values) => {
            let hash = 2166136261n;
            for (const value of values) {
                hash = ((hash ^ BigInt(value)) * 16777619n) & 0xffffffffn;
            }
            return Number(hash);
        };
        const values = [0xffffffff, 0x80000000, 0x9e3779b9, 12345, 7];
        expect(hash32Fnv1a(values)).to.equal(reference(values));
        expect(hash32Fnv1a(new Uint32Array(values))).to.equal(reference(values));
    });

});

describe('hash53', function () {

    // the reference cyrb53 of a string (seed 0), as the byte cache and bake stamps used before
    const cyrb53 = (str) => {
        let h1 = 0xdeadbeef, h2 = 0x41c6ce57;
        for (let i = 0; i < str.length; i++) {
            const ch = str.charCodeAt(i);
            h1 = Math.imul(h1 ^ ch, 2654435761);
            h2 = Math.imul(h2 ^ ch, 1597334677);
        }
        h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
        h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
        return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
    };

    it('matches the reference cyrb53 of a string, so stored keys stay valid', function () {
        for (const str of ['', 'a', 'https://example.com/a.bin', 'ünïcödé']) {
            expect(hash53(str), str).to.equal(cyrb53(str));
        }
    });

    it('hashes a sequence of parts as their concatenation', function () {
        expect(hash53('ab', 'c')).to.equal(hash53('abc'));
        expect(hash53('ab', [99])).to.equal(hash53('abc'));
    });

    it('hashes arrays by their 32bit words', function () {
        expect(hash53([1, 2, 3])).to.equal(hash53(new Uint32Array([1, 2, 3])));
        expect(hash53([1, 2, 3])).to.not.equal(hash53([1, 2, 4]));
    });
});
