import { expect } from 'chai';

import { FloatPacking } from '../../../src/core/math/float-packing.js';

describe('FloatPacking', function () {

    describe('#float2Half', function () {

        it('packs signed zero', function () {
            expect(FloatPacking.float2Half(0)).to.equal(0x0000);
            expect(FloatPacking.float2Half(-0)).to.equal(0x8000);
        });

        it('packs normal values', function () {
            expect(FloatPacking.float2Half(1)).to.equal(0x3c00);
            expect(FloatPacking.float2Half(1.5)).to.equal(0x3e00);
            expect(FloatPacking.float2Half(-2)).to.equal(0xc000);
            expect(FloatPacking.float2Half(2 ** -14)).to.equal(0x0400);
            expect(FloatPacking.float2Half(65504)).to.equal(0x7bff);
        });

        it('packs denormal values', function () {
            expect(FloatPacking.float2Half(2 ** -24)).to.equal(0x0001);
            expect(FloatPacking.float2Half(-(2 ** -15))).to.equal(0x8200);
        });

        it('rounds values below the smallest denormal to the nearest representable value', function () {
            expect(FloatPacking.float2Half(1.5 * 2 ** -25)).to.equal(0x0001);
            expect(FloatPacking.float2Half(-1.5 * 2 ** -25)).to.equal(0x8001);
            expect(FloatPacking.float2Half(2 ** -26)).to.equal(0x0000);
            expect(FloatPacking.float2Half(-(2 ** -26))).to.equal(0x8000);
        });

        it('rounds the tie halfway to the smallest denormal to signed zero', function () {
            expect(FloatPacking.float2Half(2 ** -25)).to.equal(0x0000);
            expect(FloatPacking.float2Half(-(2 ** -25))).to.equal(0x8000);
        });

        it('packs values too large for a half float as infinity of the same sign', function () {
            expect(FloatPacking.float2Half(65520)).to.equal(0x7c00);
            expect(FloatPacking.float2Half(65536)).to.equal(0x7c00);
            expect(FloatPacking.float2Half(70000)).to.equal(0x7c00);
            expect(FloatPacking.float2Half(-70000)).to.equal(0xfc00);
            expect(FloatPacking.float2Half(1e10)).to.equal(0x7c00);
            expect(FloatPacking.float2Half(-1e10)).to.equal(0xfc00);
        });

        it('packs infinity', function () {
            expect(FloatPacking.float2Half(Infinity)).to.equal(0x7c00);
            expect(FloatPacking.float2Half(-Infinity)).to.equal(0xfc00);
        });

        it('packs NaN as a half float NaN', function () {
            const half = FloatPacking.float2Half(NaN);
            expect(half & 0x7c00).to.equal(0x7c00);
            expect(half & 0x03ff).to.not.equal(0);
        });

        it('always returns a 16-bit value', function () {
            const values = [0, 1, -1, 65504, 65520, 70000, -70000, 123456789, -1e30, Infinity, -Infinity, NaN];
            for (const value of values) {
                const half = FloatPacking.float2Half(value);
                expect(half).to.be.at.least(0);
                expect(half).to.be.at.most(0xffff);
            }
        });
    });

    describe('#half2Float', function () {

        it('decodes normal values, signed zeros and infinities', function () {
            expect(FloatPacking.half2Float(0x3e00)).to.equal(1.5);
            expect(FloatPacking.half2Float(0xc000)).to.equal(-2);
            expect(FloatPacking.half2Float(0x0000)).to.equal(0);
            expect(Object.is(FloatPacking.half2Float(0x8000), -0)).to.equal(true);
            expect(FloatPacking.half2Float(0x7c00)).to.equal(Infinity);
            expect(FloatPacking.half2Float(0xfc00)).to.equal(-Infinity);
            expect(FloatPacking.half2Float(0x7e00)).to.be.NaN;
        });

        it('decodes denormals', function () {
            expect(FloatPacking.half2Float(0x0001)).to.equal(2 ** -24);
        });

        it('round-trips the values float2Half packs exactly', function () {
            for (const value of [0.5, 1, 65504, -0.25, 6.103515625e-5]) {
                expect(FloatPacking.half2Float(FloatPacking.float2Half(value)), String(value)).to.equal(value);
            }
        });
    });
});
