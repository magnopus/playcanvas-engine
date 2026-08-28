import { DebugHelper } from '../../../core/debug.js';

/**
 * A wrapper over the GpuQuerySet object, allowing timestamp and occlusion queries. The results
 * are copied back using staging buffers to avoid blocking.
 */
class WebgpuQuerySet {
    /**
     * @type {GPUQuerySet}
     */
    querySet;

    stagingBuffers = [];

    activeStagingBuffer = null;

    /** @type {number} */
    bytesPerSlot;

    constructor(device, isTimestamp, capacity) {
        this.device = device;
        this.capacity = capacity;
        this.bytesPerSlot = isTimestamp ? 8 : 4;

        // query set
        const wgpu = device.wgpu;
        this.querySet = wgpu.createQuerySet({
            type: isTimestamp ? 'timestamp' : 'occlusion',
            count: capacity
        });
        DebugHelper.setLabel(this.querySet, `QuerySet-${isTimestamp ? 'Timestamp' : 'Occlusion'}`);

        // gpu buffer for query results GPU writes to
        this.queryBuffer = wgpu.createBuffer({
            size: this.bytesPerSlot * capacity,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        DebugHelper.setLabel(this.queryBuffer, 'QueryGpuBuffer');
    }

    destroy() {
        this.device.deferDestroy(this.querySet);
        this.querySet = null;

        this.device.deferDestroy(this.queryBuffer);
        this.queryBuffer = null;

        this.activeStagingBuffer = null;

        this.stagingBuffers.forEach((stagingBuffer) => {
            stagingBuffer.destroy();
        });
        this.stagingBuffers = null;
    }

    getStagingBuffer() {
        let stagingBuffer = this.stagingBuffers.pop();
        if (!stagingBuffer) {
            stagingBuffer = this.device.wgpu.createBuffer({
                size: this.queryBuffer.size,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            DebugHelper.setLabel(stagingBuffer, 'QueryStagingBuffer');
        }
        return stagingBuffer;
    }

    resolve(count) {
        const device = this.device;
        const commandEncoder = device.getCommandEncoder();

        // copy times to the gpu buffer
        commandEncoder.resolveQuerySet(this.querySet, 0, count, this.queryBuffer, 0);

        // copy the gpu buffer to the staging buffer
        const activeStagingBuffer = this.getStagingBuffer();
        this.activeStagingBuffer = activeStagingBuffer;

        commandEncoder.copyBufferToBuffer(this.queryBuffer, 0, activeStagingBuffer, 0, this.bytesPerSlot * count);
    }

    request(count, renderVersion) {
        const stagingBuffer = this.activeStagingBuffer;
        this.activeStagingBuffer = null;

        return stagingBuffer.mapAsync(GPUMapMode.READ).then(() => {

            // timestamps in nanoseconds. Note that this array is valid only till we unmap the staging buffer.
            const srcTimings = new BigInt64Array(stagingBuffer.getMappedRange());

            // A pass-boundary timestamp is not guaranteed to be written every frame. On Metal the
            // end-of-pass sample is taken at the fragment-stage boundary, so a pass that issues no
            // fragment work never writes it: the query then resolves either to zero (if it was
            // never written) or to whatever old value the query last held (a query stays
            // "available" once written, e.g. before a pass-layout change re-mapped its slot).
            // Such a stale value can be seconds old, and letting it into the frame span pins
            // minTime in the past - frameTime then grows by a frame's worth every frame, looking
            // exactly like an accumulating timer. A frame's real timestamps all lie close to the
            // frame's latest one, so anything older than this window is treated as unwritten.
            const staleWindowNs = 1000000000n; // 1 second

            // find the latest timestamp of the frame first, as the recency reference
            let maxTime = null;
            for (let i = 0; i < count * 2; i++) {
                const t = srcTimings[i];
                if (t !== 0n && (maxTime === null || t > maxTime)) maxTime = t;
            }

            // Convert each begin/end pair to a per-pass duration in ms.
            //
            // Durations are clamped to be non-negative: on some GPUs the begin/end timestamps of a
            // pass can be reported slightly out of order (begin > end), giving a small negative
            // duration. This is a driver / architecture timing artifact rather than a real
            // measurement, so we clamp it to zero instead of letting it corrupt the totals.
            const timings = [];
            let minTime = null;
            for (let i = 0; i < count; i++) {
                const begin = srcTimings[i * 2];
                const end = srcTimings[i * 2 + 1];

                // Skip pairs with an unwritten timestamp: zero is not a reading a monotonic GPU
                // clock can produce, and a stale reading (see above) is older than any real
                // timestamp of this frame. Their duration is reported as zero as well - a stale
                // begin with a fresh end would otherwise yield a seconds-long pass time.
                const stale = begin === 0n || end === 0n ||
                    maxTime - begin > staleWindowNs || maxTime - end > staleWindowNs;
                timings.push(stale ? 0 : Math.max(0, Number(end - begin) * 0.000001));
                if (stale) continue;

                // track the earliest timestamp across all passes (used for frameTime below). Both
                // begin and end are considered to stay robust to the out-of-order case described
                // above.
                if (minTime === null || begin < minTime) minTime = begin;
                if (end < minTime) minTime = end;
            }

            // Frame GPU time is reported as the span from the earliest begin to the latest end
            // timestamp across all passes, NOT the sum of per-pass durations.
            //
            // On tile-based and other heavily pipelined GPU architectures (Apple Silicon is one
            // well known example, but it is not unique to it) consecutive passes overlap in time -
            // the vertex stage of one pass runs concurrently with the fragment stage of the
            // previous one, so their timestamp intervals overlap. Summing per-pass durations then
            // massively overestimates the real frame cost: the total grows with the number of
            // passes even while the GPU sits mostly idle. The span stays within the actual frame
            // time and gives a realistic figure regardless of architecture.
            //
            // Note the span also includes any GPU idle bubbles between passes (e.g. while the GPU
            // waits on the CPU or on dependencies), so on those frames it can read slightly higher
            // than the pure GPU work. This is the same property the WebGL profiler has, as it
            // measures the whole frame with a single query that likewise spans those gaps.
            const frameTime = minTime !== null ? Number(maxTime - minTime) * 0.000001 : 0;

            stagingBuffer.unmap();
            this.stagingBuffers?.push(stagingBuffer);

            return {
                renderVersion,
                timings,
                frameTime
            };
        });
    }
}

export { WebgpuQuerySet };
