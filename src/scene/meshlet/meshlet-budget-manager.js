/**
 * @import { MeshletResidency } from './streaming/meshlet-residency.js'
 */

/**
 * Memory-pressure feedback for the meshlet pipeline - the crash-avoidance mechanism. When the
 * page pool cannot hold the pages the current LOD cut wants, the effective DAG error threshold
 * is raised, coarsening the cut: coarser clusters want fewer, coarser pages and the working set
 * converges under the budget instead of thrashing.
 *
 * The signal is churn, not occupancy: a full pool whose pages are stable is the designed
 * saturated state (the LRU holds the winners; nothing needs to move), while pages cycling
 * through the pool every frame mean the cut genuinely exceeds the budget. Occupancy stays at
 * 100% forever once the pool fills - nothing proactively frees slots - so any controller
 * keyed on occupancy ratchets up on the first transient miss (camera motion guarantees them)
 * and can never decay. Hard allocation failures (an install finding no evictable slot) raise
 * fast; sustained eviction churn raises gently; a quiet pool decays the scale back toward 1.
 *
 * @ignore
 */
class MeshletBudgetManager {
    /** @type {number} - multiplier applied to the user's dagPixelThreshold, in [1, maxScale]. */
    pressureScale = 1;

    /** @type {number} - a 64x coarser cut is already near-root; beyond that is unrecoverable. */
    maxScale = 64;

    /** @type {number} - evictions per frame tolerated as camera-motion turnover, not thrash. */
    churnTolerance = 2;

    /** @type {number} - per-frame scale growth on a hard denial or index starvation. */
    raiseFast = 1.05;

    /** @type {number} - per-frame scale growth on sustained churn or a wedged pool. */
    raiseSlow = 1.02;

    /** @type {number} - per-frame scale decay while nothing is evicted, denied or index-tight. */
    decay = 0.995;

    _lastDropped = 0;

    _lastEvicted = 0;

    _lastFetched = 0;

    /** @type {number} - consecutive frames the wedge signature has held. */
    _wedgeFrames = 0;

    /** @type {number} - frames a wedge must persist before it counts (readback gaps are 1-2). */
    wedgeFrames = 60;

    /**
     * Fraction of a view's index ceiling at which the cut is treated as too fine for the
     * budget. The ceiling is a hard wall - once demand reaches it the cull clamps, and the
     * clusters that win the reservation race change every frame, which reads as wild flicker.
     * Coarsening the cut is the graceful way to fit; clamping is not.
     *
     * @type {number}
     */
    indexStarvedAt = 0.85;

    /**
     * Fraction of the ceiling below which index pressure is considered resolved and quality may
     * be restored. The gap to {@link indexStarvedAt} is what stops the controller oscillating:
     * decaying the moment demand drops under the starvation line would push it straight back
     * over, and a cut that alternates between two levels every few frames is precisely the LOD
     * flicker the pressure system exists to avoid.
     *
     * @type {number}
     */
    indexEasedAt = 0.6;

    /**
     * Per-frame tick.
     *
     * @param {MeshletResidency|null} residency - The streaming residency, or null (resident world).
     * @param {number} [indexRatio] - Peak index demand across views as a fraction of the budget
     * ceiling. Above {@link indexStarvedAt} the cut cannot fit and is coarsened - the same
     * response as a hard page denial, because it is the same problem. Between the two
     * thresholds the scale holds.
     */
    update(residency, indexRatio = 0) {
        if (indexRatio > this.indexStarvedAt) {
            // applies to resident worlds too - index buffers are budgeted there as well
            this.pressureScale = Math.min(this.pressureScale * this.raiseFast, this.maxScale);
            return;
        }
        const indexTight = indexRatio > this.indexEasedAt;
        if (!residency) {
            if (!indexTight) this.pressureScale = 1;
            return;
        }

        const droppedDelta = residency.droppedNoSlot - this._lastDropped;
        this._lastDropped = residency.droppedNoSlot;
        const evictedDelta = residency.evictedPages - this._lastEvicted;
        this._lastEvicted = residency.evictedPages;
        const fetchedDelta = residency.fetchedPages - this._lastFetched;
        this._lastFetched = residency.fetchedPages;

        // A pool is only wedged when demand is outstanding AND nothing is moving: no
        // evictions, nothing in flight, and no fetches landing. Without the fetch test a
        // merely SLOW pool reads as wedged - a huge asset streams a few pages per frame, so
        // most frames see zero evictions and an empty in-flight set between requests, and the
        // scale ratchets to maximum while the pool is in fact making steady progress.
        const missing = residency.lastMissingWanted ?? 0;
        const wedgedNow = missing > 0 && evictedDelta === 0 && fetchedDelta === 0 &&
            residency.inFlight.size === 0;
        // A real wedge is permanent; the same signature also appears for a frame or two in
        // the gap between demand readbacks, when a page is briefly outstanding with nothing
        // in flight yet. Requiring the condition to PERSIST separates them - without this a
        // single transient page ratchets the scale up 1.02 per frame against a 0.995 decay,
        // which drives quality to nothing on a pool that is half empty and evicting nothing.
        this._wedgeFrames = wedgedNow ? this._wedgeFrames + 1 : 0;
        const wedged = this._wedgeFrames >= this.wedgeFrames;

        if (droppedDelta > 0) {
            // hard denial - an install found nothing evictable; coarsen fast
            this.pressureScale = Math.min(this.pressureScale * this.raiseFast, this.maxScale);
        } else if (evictedDelta > this.churnTolerance || wedged) {
            // sustained churn (the cut exceeds the pool) or a genuinely wedged pool
            this.pressureScale = Math.min(this.pressureScale * this.raiseSlow, this.maxScale);
        } else if (evictedDelta === 0 && droppedDelta === 0 && !indexTight) {
            // nothing is being evicted or denied: there is no memory pressure, whether or not
            // pages are still arriving. Restore quality until churn actually resumes.
            this.pressureScale = Math.max(this.pressureScale * this.decay, 1);
        }
        // otherwise hold - the hysteresis band that stops LOD breathing
    }
}

export { MeshletBudgetManager };
