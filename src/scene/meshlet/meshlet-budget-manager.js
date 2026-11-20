/**
 * @import { MeshletResidency } from './streaming/meshlet-residency.js'
 * @import { MeshletShadowView } from './meshlet-shadow-view.js'
 * @import { MeshletView } from './meshlet-view.js'
 */

// Every view keeps at least this fraction of an equal share of the index budget, so a view
// that is momentarily empty can grow back without waiting for a redistribution.
const SHARE_FLOOR_FRACTION = 0.25;

// Occupancy camera views are budgeted at before shadow views get any index budget: under
// indexEasedAt, so a camera that is given its full want never reads as tight.
const CAMERA_SHARE_OCCUPANCY = 0.5;

// Shadow LOD controller (see MeshletBudgetManager#stepShadowPressure): the per-frame step is the
// demand ratio's distance outside the hold band raised to this power, and limited to these
// bounds - about x2 a second up and x0.6 a second down at 60 fps, against x18 and x0.74 for the
// camera's controller.
const SHADOW_PRESSURE_GAIN = 0.1;
const SHADOW_PRESSURE_MAX_RAISE = 1.012;
const SHADOW_PRESSURE_MAX_DECAY = 0.992;

/** @type {never[]} */
const _noViews = [];

/**
 * A view's index-budget want: its demand at the given target occupancy, above the share floor.
 *
 * @param {{ indexDemand: () => number }} view - A camera or shadow view.
 * @param {number} floor - The share floor every view gets.
 * @param {number} occupancy - Target occupancy of the share (1 = the demand itself).
 * @returns {number} The want, in indices.
 */
const viewWant = (view, floor, occupancy) => Math.max(view.indexDemand() / occupancy - floor, 0);

/**
 * The most indices a shadow face may draw: a number per texel of its part of the shadow map,
 * or 0 when that is not known yet.
 *
 * @param {MeshletShadowView} view - The shadow face.
 * @param {number} indicesPerTexel - Indices per texel, or 0 for no cap.
 * @returns {number} The cap, in indices.
 */
const shadowIndexCap = (view, indicesPerTexel) => {
    if (!(indicesPerTexel > 0)) return 0;
    const resolution = view.light?._shadowResolution ?? 0;
    const viewport = view.renderData?.shadowViewport;
    const texels = resolution * resolution * (viewport ? viewport.z * viewport.w : 1);
    return texels > 0 ? Math.floor(texels * indicesPerTexel) : 0;
};

/**
 * Memory-pressure feedback for one meshlet world - the crash-avoidance mechanism. When the
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
 * It also shares the world's index budget between its views ({@link distributeIndexBudget}),
 * runs the shadow views' own LOD controllers, and detects a budget the scene cannot meet at all
 * ({@link checkFeasible}).
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

    /**
     * The largest shadow face's own LOD pressure (see {@link updateShadowPressure}), on top of
     * {@link pressureScale}.
     *
     * @type {number}
     */
    shadowPressureScale = 1;

    /** @type {number} - worst shadow-view demand/share ratio from the last distribution. */
    shadowIndexRatio = 0;

    /** @type {number} - consecutive frames the budget has been infeasible. @private */
    _infeasibleFrames = 0;

    /** @type {number} - residency counters as of the last update, for per-frame deltas. @private */
    _lastDropped = 0;

    /** @private */
    _lastEvicted = 0;

    /** @private */
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
     * Forgets the residency counters the per-frame deltas are taken against, for a rebuilt
     * world whose new residency counts from zero. The pressure itself carries over.
     */
    reset() {
        this._lastDropped = 0;
        this._lastEvicted = 0;
        this._lastFetched = 0;
    }

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
            // ease back like a streamed world: snapping to 1 re-creates the overflow at once
            if (!indexTight) this.pressureScale = Math.max(this.pressureScale * this.decay, 1);
            return;
        }

        const droppedDelta = residency.droppedNoSlot - this._lastDropped;
        this._lastDropped = residency.droppedNoSlot;
        const evictedDelta = residency.evictedPages - this._lastEvicted;
        this._lastEvicted = residency.evictedPages;
        const fetchedDelta = residency.fetchedPages - this._lastFetched;
        this._lastFetched = residency.fetchedPages;

        // A pool is only wedged when demand is outstanding and nothing is moving: no
        // evictions, nothing in flight, and no fetches landing. Without the fetch test a
        // merely slow pool reads as wedged - a huge asset streams a few pages per frame, so
        // most frames see zero evictions and an empty in-flight set between requests, and the
        // scale ratchets to maximum while the pool is in fact making steady progress.
        const missing = residency.lastMissingWanted ?? 0;
        const wedgedNow = missing > 0 && evictedDelta === 0 && fetchedDelta === 0 &&
            residency.inFlight.size === 0;
        // A real wedge is permanent; the same signature also appears for a frame or two in
        // the gap between demand readbacks, when a page is briefly outstanding with nothing
        // in flight yet. Requiring the condition to persist separates them: a single transient
        // page would otherwise ratchet the scale up faster than it decays, on a pool that is
        // half empty and evicting nothing.
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

    /**
     * Shares a world's index budget across its live views, camera views first, and returns the
     * worst camera-view pressure.
     *
     * Every view keeps a growth floor. Camera views are then budgeted for their demand at
     * CAMERA_SHARE_OCCUPANCY, and shadow views split what is left in proportion to their
     * demand. A plain proportional split would let far cascades - which cover most of the world,
     * so their root floor alone can exceed the camera's demand - take most of the budget, and
     * because index pressure feeds the camera's LOD controller, the shadows would coarsen and
     * drop what the camera sees. Shadow views report to their own controllers instead
     * ({@link updateShadowPressure}).
     *
     * @param {number} ceiling - The world's index ceiling.
     * @param {Map<*, MeshletView>} cameraViews - The camera views.
     * @param {MeshletShadowView[]|null} shadowViews - The shadow views.
     * @param {number} shadowIndicesPerTexel - A shadow face's index cap per texel of its part of
     * the shadow map, or 0 for none.
     * @returns {number} Highest demand/share ratio across camera views.
     */
    distributeIndexBudget(ceiling, cameraViews, shadowViews, shadowIndicesPerTexel) {
        this.shadowIndexRatio = 0;
        if (!(ceiling > 0) || !Number.isFinite(ceiling)) return 0;

        const shadows = shadowViews ?? _noViews;
        const count = cameraViews.size + shadows.length;
        if (!count) return 0;

        // Charge the growth floor before distributing the rest so shares stay inside the budget.
        const floor = Math.floor((ceiling / count) * SHARE_FLOOR_FRACTION);
        const remaining = ceiling - floor * count;
        // A camera is budgeted for its whole demand at the target occupancy, so it always has
        // headroom. Budgeting only the part above the floor would strand a coarsened camera
        // whose demand has fallen under the floor at ~0.8 occupancy, where the controller holds
        // and never restores quality. Without shadow views nothing competes, and cameras split
        // the whole budget by demand above the floor.
        const cameraOccupancy = shadows.length ? CAMERA_SHARE_OCCUPANCY : 1;

        let cameraWant = 0;
        for (const v of cameraViews.values()) cameraWant += viewWant(v, floor, cameraOccupancy);
        const cameraGive = shadows.length ? Math.min(cameraWant, remaining) : remaining;
        let shadowWant = 0;
        for (let i = 0; i < shadows.length; i++) shadowWant += viewWant(shadows[i], floor, 1);
        const shadowGive = remaining - cameraGive;

        let worst = 0;
        for (const v of cameraViews.values()) {
            v.indexShare = floor + (cameraWant > 0 ?
                Math.floor(cameraGive * viewWant(v, floor, cameraOccupancy) / cameraWant) :
                Math.floor(cameraGive / cameraViews.size));
            worst = Math.max(worst, v.indexPressure());
        }
        for (let i = 0; i < shadows.length; i++) {
            const v = shadows[i];
            v.indexShare = floor + (shadowWant > 0 ?
                Math.floor(shadowGive * viewWant(v, floor, 1) / shadowWant) :
                Math.floor(shadowGive / shadows.length));
            // a face never gets more than its shadow map can resolve: what is left of a large
            // memory budget is not a reason to draw triangles far below a texel
            const cap = shadowIndexCap(v, shadowIndicesPerTexel);
            if (cap > 0) v.indexShare = Math.min(v.indexShare, cap);
            this.shadowIndexRatio = Math.max(this.shadowIndexRatio, v.indexPressure());
        }
        return worst;
    }

    /**
     * Steps every shadow face's own LOD pressure from its demand against its share, and reports
     * the largest as {@link shadowPressureScale}. Per face, so a far cascade over its budget
     * coarsens alone instead of taking the near cascades - where coarse casters show as
     * blotches - with it.
     *
     * @param {MeshletShadowView[]|null} shadowViews - The shadow views.
     */
    updateShadowPressure(shadowViews) {
        let worst = 1;
        const shadows = shadowViews ?? _noViews;
        for (let i = 0; i < shadows.length; i++) {
            const view = shadows[i];
            view.pressureScale = this.stepShadowPressure(view.pressureScale, view.indexPressure());
            worst = Math.max(worst, view.pressureScale);
        }
        this.shadowPressureScale = worst;
    }

    /**
     * One step of a shadow face's LOD controller: the camera controller's index response (raise
     * past indexStarvedAt, decay under indexEasedAt, hold between), but proportional and rate
     * limited. The scale moves every caster of the face at once, and demand lags a change by a
     * readback or two, so a fast ramp overshoots and decays back - a shadow LOD sawtooth seen as
     * popping and uneven frame times.
     *
     * @param {number} scale - The face's pressure scale.
     * @param {number} ratio - Its demand/share ratio.
     * @returns {number} The new scale.
     */
    stepShadowPressure(scale, ratio) {
        let step = 1;
        if (ratio > this.indexStarvedAt) {
            step = Math.min((ratio / this.indexStarvedAt) ** SHADOW_PRESSURE_GAIN, SHADOW_PRESSURE_MAX_RAISE);
        } else if (ratio < this.indexEasedAt) {
            step = Math.max((ratio / this.indexEasedAt) ** SHADOW_PRESSURE_GAIN, SHADOW_PRESSURE_MAX_DECAY);
        }
        return Math.min(Math.max(scale * step, 1), this.maxScale);
    }

    /**
     * Detects a budget the scene cannot meet. At the DAG roots there is nothing coarser to
     * select, so demand still unmet at maximum pressure for `frames` consecutive frames means
     * the working set does not fit; the streak then restarts.
     *
     * @param {number} indexRatio - Worst camera-view demand/share ratio.
     * @param {number} pagesShort - Missing pages the cut wanted.
     * @param {number} frames - Consecutive infeasible frames that make a report.
     * @returns {boolean} True when the budget should be reported as infeasible.
     */
    checkFeasible(indexRatio, pagesShort, frames) {
        const pinned = this.pressureScale >= this.maxScale;
        const short = indexRatio > 1 || pagesShort > 0;
        this._infeasibleFrames = (pinned && short) ? this._infeasibleFrames + 1 : 0;
        if (this._infeasibleFrames < frames) return false;
        this._infeasibleFrames = 0;
        return true;
    }
}

export { MeshletBudgetManager };
