import { expect } from 'chai';

import { WebgpuQuerySet } from '../../../../src/platform/graphics/webgpu/webgpu-query-set.js';

// Exercises WebgpuQuerySet#request against a fake staging buffer: it reads begin/end timestamp
// pairs, reports per-pass durations and the whole-frame span. A slot whose pass never executed
// holds zeros, which must not drag the span down to the absolute clock value - or, when the
// query was written in some earlier frame and kept that value, a stale timestamp that must not
// pin the span's start in the past.
describe('WebgpuQuerySet#request', function () {

    let savedMapMode;

    before(function () {
        savedMapMode = globalThis.GPUMapMode;
        globalThis.GPUMapMode = { READ: 1 };
    });

    after(function () {
        globalThis.GPUMapMode = savedMapMode;
    });

    const request = (pairs) => {
        const timestamps = new BigInt64Array(pairs.flat().map(BigInt));
        let unmapped = 0;
        const stagingBuffer = {
            mapAsync: () => Promise.resolve(),
            getMappedRange: () => timestamps.buffer,
            unmap: () => unmapped++
        };
        const fake = { activeStagingBuffer: stagingBuffer, stagingBuffers: [] };
        return WebgpuQuerySet.prototype.request.call(fake, pairs.length, 7).then((result) => {
            expect(unmapped).to.equal(1);
            expect(fake.stagingBuffers).to.deep.equal([stagingBuffer]);
            return result;
        });
    };

    const NS = 1000000; // timestamps are nanoseconds, results milliseconds

    it('reports per-pass durations and the frame span across passes', async function () {
        const result = await request([[10 * NS, 14 * NS], [12 * NS, 20 * NS]]);
        expect(result.renderVersion).to.equal(7);
        expect(result.timings).to.deep.equal([4, 8]);
        expect(result.frameTime).to.equal(10);
    });

    it('ignores a slot whose pass never executed when measuring the frame span', async function () {
        const result = await request([[10 * NS, 14 * NS], [0, 0], [12 * NS, 20 * NS]]);
        expect(result.timings).to.deep.equal([4, 0, 8]);
        expect(result.frameTime, 'zero timestamps must not become the span start').to.equal(10);
    });

    it('reports a zero frame time when no pass executed', async function () {
        const result = await request([[0, 0], [0, 0]]);
        expect(result.timings).to.deep.equal([0, 0]);
        expect(result.frameTime).to.equal(0);
    });

    it('ignores a pair with a stale timestamp when measuring the frame span', async function () {
        // second pass: begin written this frame, end still holding a value from ten seconds
        // ago (a query stays available once written, so a skipped pass-boundary write leaves
        // the old value in place rather than zero)
        const now = 20000 * NS;
        const result = await request([
            [now + 10 * NS, now + 14 * NS],
            [now + 12 * NS, now - 10000 * NS],
            [now + 12 * NS, now + 20 * NS]
        ]);
        expect(result.timings, 'a stale pair must not report a duration').to.deep.equal([4, 0, 8]);
        expect(result.frameTime, 'a stale timestamp must not become the span start').to.equal(10);
    });
});
