import { OBJECT_DATA as O, OBJECT_DATA_U32S, OBJECT_FLAG_HIDDEN, OBJECT_FLAG_NO_SHADOW } from './constants.js';

const RADIX_BITS = 11;
const RADIX_SIZE = 1 << RADIX_BITS;
const RADIX_MASK = RADIX_SIZE - 1;

// update inputs remembered to detect a repeat: 24 frustum plane floats, camera xyz, three bucket
// capacities, record capacity, shadow flag, pool slots, world object and root page versions
const INPUT_COUNT = 24 + 3 + 3 + 1 + 1 + 1 + 2;

const storeInput = (inputs, k, value) => {
    if (inputs[k] === value) return false;
    inputs[k] = value;
    return true;
};

/**
 * Bounded admission of complete coarse instances. Nearby instances enter first with distance hysteresis
 * for existing visible instances, once their full coarse set fits. Refinement
 * remains GPU-driven. No per-frame storage grows with the total triangle count.
 * @ignore
 */
class MeshletRootSelection {
    constructor(world) {
        this.world = world;
        world.rootSelections?.add(this);
        this.wanted = new Float32Array(world.instanceCount);
        this.distance = new Float32Array(world.instanceCount);
        this.previous = new Uint8Array(world.instanceCount);
        this.pages = new Uint8Array(world.totalPages);
        this._occupied = new Uint8Array(world.totalPages);
        this.order = [];
        // sort keys: distance with hysteresis for previously admitted instances
        this._keys = new Float32Array(world.instanceCount);
        this._keyBits = new Uint32Array(this._keys.buffer);
        this._orderA = new Uint32Array(world.instanceCount);
        this._orderB = new Uint32Array(world.instanceCount);
        this._histogram = new Uint32Array(RADIX_SIZE);
        // Inputs of the last full pass. A pass whose admissions equal the previous frame's has
        // reached the hysteresis fixed point, so while the inputs repeat the result cannot change
        // and the per-instance pass is skipped - a static camera over a static scene costs a
        // compare per view instead of a cull of every instance.
        this._inputs = new Float64Array(INPUT_COUNT);
        this._settled = false;
        this._pagesPrev = new Uint8Array(world.totalPages);
        world.rootPagesVersion = (world.rootPagesVersion ?? 0) + 1;
        this.indices = [0, 0, 0];
        this.records = 0;
        this.deferred = 0;
        this.requestedIndices = 0;
        this.requestedByBucket = [0, 0, 0];
        this.requestedRecords = 0;
        this.recordDemand = 0;
        this.capacity = [0, 0, 0];
    }

    /**
     * Select complete visible roots within the current draw buffers and geometry pool.
     * @param {Float32Array} planes - World frustum planes.
     * @param {import('../../core/math/vec3.js').Vec3} camera - Camera position.
     * @param {number[]} capacities - Index capacities by bucket.
     * @param {number} recordCapacity - Draw record capacity.
     * @param {boolean} shadow - Whether to exclude non-casters.
     */
    update(planes, camera, capacities, recordCapacity, shadow) {
        const world = this.world;
        if (!this._storeInputs(planes, camera, capacities, recordCapacity, shadow) && this._settled) return;
        this._select(planes, camera, capacities, recordCapacity, shadow);

        const wanted = this.wanted, previous = this.previous;
        let settled = true;
        for (let i = 0; i < wanted.length; i++) {
            if (wanted[i] !== previous[i]) {
                settled = false;
                break;
            }
        }
        this._settled = settled;

        // other views account for this view's pages, so they must rerun when these change
        const pages = this.pages, pagesPrev = this._pagesPrev;
        for (let p = 0; p < pages.length; p++) {
            if (pages[p] !== pagesPrev[p]) {
                pagesPrev.set(pages);
                world.rootPagesVersion = (world.rootPagesVersion ?? 0) + 1;
                // this view's own change is not a reason for it to rerun
                this._inputs[INPUT_COUNT - 1] = world.rootPagesVersion;
                break;
            }
        }
    }

    /**
     * Records the inputs of this update.
     *
     * @param {Float32Array} planes - World frustum planes.
     * @param {import('../../core/math/vec3.js').Vec3} camera - Camera position.
     * @param {number[]} capacities - Index capacities by bucket.
     * @param {number} recordCapacity - Draw record capacity.
     * @param {boolean} shadow - Whether to exclude non-casters.
     * @returns {boolean} True when any input differs from the previous update.
     * @private
     */
    _storeInputs(planes, camera, capacities, recordCapacity, shadow) {
        const inputs = this._inputs;
        const world = this.world;
        let changed = false;
        let k = 0;
        for (let p = 0; p < 24; p++) changed = storeInput(inputs, k++, planes[p]) || changed;
        changed = storeInput(inputs, k++, camera.x) || changed;
        changed = storeInput(inputs, k++, camera.y) || changed;
        changed = storeInput(inputs, k++, camera.z) || changed;
        for (let b = 0; b < 3; b++) changed = storeInput(inputs, k++, capacities[b]) || changed;
        changed = storeInput(inputs, k++, recordCapacity) || changed;
        changed = storeInput(inputs, k++, shadow ? 1 : 0) || changed;
        changed = storeInput(inputs, k++, world.poolSlots) || changed;
        changed = storeInput(inputs, k++, world.objectVersion ?? 0) || changed;
        changed = storeInput(inputs, k++, world.rootPagesVersion ?? 0) || changed;
        return changed;
    }

    /**
     * The full selection pass over every instance.
     *
     * @param {Float32Array} planes - World frustum planes.
     * @param {import('../../core/math/vec3.js').Vec3} camera - Camera position.
     * @param {number[]} capacities - Index capacities by bucket.
     * @param {number} recordCapacity - Draw record capacity.
     * @param {boolean} shadow - Whether to exclude non-casters.
     * @private
     */
    _select(planes, camera, capacities, recordCapacity, shadow) {
        const world = this.world;
        const objects = world.objectDataCpu;
        const f = world.objectDataCpuF;
        const order = this.order;
        order.length = 0;
        this.previous.set(this.wanted);
        this.wanted.fill(0);
        this.pages.fill(0);
        this.indices.fill(0);
        this.records = 0;
        this.deferred = 0;
        this.requestedIndices = 0;
        this.requestedByBucket.fill(0);
        this.requestedRecords = 0;
        this.recordDemand = 0;
        const totalCapacity = capacities[0] + capacities[1] + capacities[2];
        let usedIndices = 0;
        const occupied = this._occupied;
        occupied.fill(0);
        let pageCount = 0;
        for (const other of world.rootSelections ?? []) {
            if (other === this) continue;
            for (let p = 0; p < occupied.length; p++) {
                if (other.pages[p] && !occupied[p]) {
                    occupied[p] = 1;
                    pageCount++;
                }
            }
        }
        const roots = world.cut.instanceRoots;
        const pages = this.pages;
        // Totals of every visible root, with its page union marked in `pages`. When they fit,
        // admission order is irrelevant and the distance sort (the dominant CPU cost with tens
        // of thousands of placements) is skipped.
        let visibleIndices = 0, visibleRecords = 0, visiblePages = 0;
        for (let i = 0; i < world.instanceCount; i++) {
            const row = i * OBJECT_DATA_U32S;
            const flags = objects[row + O.FLAGS];
            if ((flags & OBJECT_FLAG_HIDDEN) || (shadow && (flags & OBJECT_FLAG_NO_SHADOW))) continue;
            const x = f[row + O.SPHERE], y = f[row + O.SPHERE + 1], z = f[row + O.SPHERE + 2];
            const cx = f[row] * x + f[row + 4] * y + f[row + 8] * z + f[row + 12];
            const cy = f[row + 1] * x + f[row + 5] * y + f[row + 9] * z + f[row + 13];
            const cz = f[row + 2] * x + f[row + 6] * y + f[row + 10] * z + f[row + 14];
            const radius = f[row + O.SPHERE + 3] * f[row + O.MAX_SCALE];
            let visible = true;
            for (let p = 0; p < 24; p += 4) {
                if (planes[p] * cx + planes[p + 1] * cy + planes[p + 2] * cz + planes[p + 3] < -radius) {
                    visible = false;
                    break;
                }
            }
            if (visible) {
                const dx = cx - camera.x, dy = cy - camera.y, dz = cz - camera.z;
                this.distance[i] = Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz) - radius, 0);
                order.push(i);
                const root = roots[i];
                visibleIndices += root.cost[0];
                visibleRecords += root.cost[1];
                for (const page of root.pages) {
                    if (!pages[page]) {
                        pages[page] = 1;
                        if (!occupied[page]) visiblePages++;
                    }
                }
            }
        }
        if (visibleIndices <= totalCapacity && visibleRecords <= recordCapacity && pageCount + visiblePages <= world.poolSlots) {
            for (const i of order) {
                const root = roots[i];
                this.wanted[i] = 1;
                this.indices[root.bucket] += root.cost[0];
                this.requestedByBucket[root.bucket] += root.cost[0];
            }
            for (let p = 0; p < pages.length; p++) occupied[p] |= pages[p];
            this.requestedIndices = usedIndices = visibleIndices;
            this.requestedRecords = this.records = this.recordDemand = visibleRecords;
            this._partition(capacities, totalCapacity, usedIndices);
            return;
        }
        pages.fill(0);
        const distance = this.distance, previous = this.previous, keys = this._keys;
        for (const i of order) keys[i] = distance[i] * (previous[i] ? 0.8 : 1);
        const sorted = this._radixSort(order);
        for (let k = 0; k < sorted.length; k++) {
            const i = sorted[k];
            const root = roots[i];
            this.requestedIndices += root.cost[0];
            this.requestedRecords += root.cost[1];
            const bucket = root.bucket;
            this.requestedByBucket[bucket] += root.cost[0];
            if (usedIndices + root.cost[0] > totalCapacity) {
                this.deferred++;
                continue;
            }
            let additional = 0;
            for (const page of root.pages) additional += 1 - occupied[page];
            if (pageCount + additional > world.poolSlots) {
                this.deferred++;
                continue;
            }
            // Grow records only for a candidate that could otherwise enter this working
            // set, not every visible root that the index/page budget already excludes.
            this.recordDemand = Math.max(this.recordDemand, this.records + root.cost[1]);
            if (this.records + root.cost[1] > recordCapacity) {
                this.deferred++;
                continue;
            }
            this.wanted[i] = 1;
            this.indices[bucket] += root.cost[0];
            usedIndices += root.cost[0];
            this.records += root.cost[1];
            pageCount += additional;
            for (const page of root.pages) {
                this.pages[page] = 1;
                occupied[page] = 1;
            }
        }

        this._partition(capacities, totalCapacity, usedIndices);
    }

    /**
     * Stable LSD radix sort of instance indices by their key. Non-negative float bits order
     * like the floats, and `order` is ascending, so ties keep index order - the same result
     * as sorting by (key, index), without a comparator sort costing milliseconds per view
     * over tens of thousands of instances.
     *
     * @param {number[]} order - Visible instance indices, ascending.
     * @returns {Uint32Array} The indices ordered by key.
     * @private
     */
    _radixSort(order) {
        const count = order.length;
        const bits = this._keyBits;
        const histogram = this._histogram;
        let src = this._orderA.subarray(0, count);
        let dst = this._orderB.subarray(0, count);
        for (let k = 0; k < count; k++) src[k] = order[k];
        for (let shift = 0; shift < 32; shift += RADIX_BITS) {
            histogram.fill(0);
            for (let k = 0; k < count; k++) histogram[(bits[src[k]] >>> shift) & RADIX_MASK]++;
            let sum = 0;
            for (let b = 0; b < RADIX_SIZE; b++) {
                const c = histogram[b];
                histogram[b] = sum;
                sum += c;
            }
            for (let k = 0; k < count; k++) {
                const i = src[k];
                dst[histogram[(bits[i] >>> shift) & RADIX_MASK]++] = i;
            }
            const swap = src;
            src = dst;
            dst = swap;
        }
        return src;
    }

    /**
     * Bucket partitions are not independent allocations. Repartition the same buffer so
     * yesterday's opaque demand cannot prevent today's two-sided coarse admission.
     *
     * @param {number[]} capacities - Index capacities by bucket.
     * @param {number} totalCapacity - Sum of the capacities.
     * @param {number} usedIndices - Indices admitted this frame.
     * @private
     */
    _partition(capacities, totalCapacity, usedIndices) {
        let extraWeight = 0;
        for (let b = 0; b < 3; b++) extraWeight += Math.max(capacities[b] - this.indices[b], 0);
        const spare = totalCapacity - usedIndices;
        let assigned = 0;
        for (let b = 0; b < 3; b++) {
            const weight = Math.max(capacities[b] - this.indices[b], 0);
            const extra = b === 2 ? spare - assigned : Math.floor(spare * weight / Math.max(extraWeight, 1));
            this.capacity[b] = this.indices[b] + extra;
            assigned += extra;
        }
    }

    destroy() {
        this.world.rootSelections?.delete(this);
        this.world.rootPagesVersion = (this.world.rootPagesVersion ?? 0) + 1;
    }
}

export { MeshletRootSelection };
