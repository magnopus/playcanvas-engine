import { OBJECT_DATA as O, OBJECT_DATA_U32S, OBJECT_FLAG_HIDDEN, OBJECT_FLAG_NO_SHADOW } from './constants.js';

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
        this.indices = [0, 0, 0];
        this.records = 0;
        this.deferred = 0;
        this.requestedIndices = 0;
        this.requestedByBucket = [0, 0, 0];
        this.requestedRecords = 0;
        this.recordDemand = 0;
        this.capacity = [0, 0, 0];
        this._compare = (a, b) => this.distance[a] * (this.previous[a] ? 0.8 : 1) - this.distance[b] * (this.previous[b] ? 0.8 : 1) || a - b;
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
                this.distance[i] = Math.max(Math.hypot(cx - camera.x, cy - camera.y, cz - camera.z) - radius, 0);
                order.push(i);
            }
        }
        order.sort(this._compare);
        for (const i of order) {
            const root = world.cut.instanceRoots[i];
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

        // Bucket partitions are not independent allocations. Repartition the same buffer
        // so yesterday's opaque demand cannot prevent today's two-sided coarse admission.
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
    }
}

export { MeshletRootSelection };
