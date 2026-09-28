import { OBJECT_DATA as O, OBJECT_DATA_U32S, OBJECT_FLAG_HIDDEN, OBJECT_FLAG_NO_SHADOW } from './constants.js';

const RADIX_BITS = 11;
const RADIX_SIZE = 1 << RADIX_BITS;
const RADIX_MASK = RADIX_SIZE - 1;

// update inputs remembered to detect a repeat: three bucket capacities, record capacity, shadow
// flag, pool slots, world object and root page versions (the view is compared with tolerance)
const INPUT_COUNT = 3 + 1 + 1 + 1 + 2;

// A pass whose admissions still differ from the previous one (hysteresis converging, or budget
// contention) is rerun at least every this many updates even inside the tolerance.
const UNSETTLED_RERUN = 4;

// A pass admits against a frustum widened, per instance, by ADMIT_MOVE plus its distance times
// sin(ADMIT_ANGLE). A camera translation T shifts every plane distance by at most T and a turn of
// θ moves a point at distance d by up to d·sin θ, so the pass keeps covering the view while
// T <= ADMIT_MOVE + nearest·(sin ADMIT_ANGLE - sin θ), where `nearest` is the distance of the
// closest instance the pass saw: far above the island that allows long travel, at street level
// about ADMIT_MOVE. Without this every frame of a moving camera reruns the pass over every
// instance, once per view (camera and shadow faces).
const ADMIT_MOVE = 12;
const ADMIT_ANGLE = 10;
const ADMIT_SIN = Math.sin(ADMIT_ANGLE * Math.PI / 180);
// A pass runs as a job spread over frames (at most JOB_BUDGET_MS of it per update) while the
// previous result stays in use. The next job starts once the view has used this fraction of the
// coverage, leaving the rest for the frames it takes; a view that outruns the coverage finishes
// the running job at once.
const RERUN_FRACTION = 1 / 3;
const JOB_BUDGET_MS = 1.5;
// instances a job scans between checks of its time budget
const JOB_YIELD_EVERY = 16384;

// approximate cell count of the instance grid the selection culls against before instances
const CELL_TARGET = 4096;

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
        // per-pass distance and sort key, indexed by grid position (see _buildSpatial)
        this.distance = new Float32Array(world.instanceCount);
        this.previous = new Uint8Array(world.instanceCount);
        this.pages = new Uint8Array(world.totalPages);
        this._occupied = new Uint8Array(world.totalPages);
        // the cut's root descriptors flattened to typed arrays (see _flattenRoots)
        this._flatSource = null;
        this._rootOf = null;
        this._indexCost = null;
        this._recordCost = null;
        this._rootBucket = null;
        this._pageStart = null;
        this._pageList = null;
        this._rootStamp = null;
        this._stamp = 0;
        this._visible = null;
        // packed world-space instance spheres, flags and their grid (see _buildSpatial), and the
        // objectVersion they were built at
        this._spheres = null;
        this._flags = null;
        this._posRoot = null;
        this._posToId = null;
        this._cellStart = null;
        this._cellSpheres = null;
        this._largeStart = 0;
        this._spheresVersion = -1;
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
        // view and nearest-instance distance of the committed pass, for the coverage test (see _coverageUsed)
        this._passPlanes = new Float32Array(24);
        this._passCamera = new Float64Array(3);
        this._hasPass = false;
        this._passNearest = 0;
        this._unsettledSkips = 0;
        // the running pass (a generator, see _select), its view and inputs, and its output
        // buffers - committed to wanted/pages only once it completes
        this._job = null;
        this._jobPlanes = new Float32Array(24);
        this._jobCamera = { x: 0, y: 0, z: 0 };
        this._jobWanted = new Float32Array(world.instanceCount);
        this._jobPages = new Uint8Array(world.totalPages);
        // a soft input changed while a job ran: run another once it completes
        this._rerun = false;
        this._settled = false;
        this._pagesPrev = new Uint8Array(world.totalPages);
        world.rootPagesVersion = (world.rootPagesVersion ?? 0) + 1;
        // bumped whenever `wanted` changes, so the culler re-uploads its per-instance tail only
        // then - a megabyte-scale write per view per frame on scenes with ~1M instances
        this.wantedVersion = 0;
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
        const { hard, soft } = this._storeInputs(capacities, recordCapacity, shadow);
        // object data, shadow flag or pool size invalidate a running job; capacities and other
        // views' pages only call for another pass after it
        if (hard) this._job = null;
        if (soft) this._rerun = true;
        // how much of the committed pass's coverage the view has used
        const used = this._coverageUsed(planes, camera);
        if (!this._job) {
            const due = hard || this._rerun || !this._hasPass || used > RERUN_FRACTION ||
                (!this._settled && ++this._unsettledSkips >= UNSETTLED_RERUN);
            if (!due) return;
            this._rerun = false;
            this._unsettledSkips = 0;
            this._jobPlanes.set(planes);
            this._jobCamera.x = camera.x;
            this._jobCamera.y = camera.y;
            this._jobCamera.z = camera.z;
            this._job = this._select(this._jobPlanes, this._jobCamera, capacities.slice(), recordCapacity, shadow);
        }
        // the committed result must cover the current view: the first pass, or a view that
        // outran the coverage, completes at once
        const sync = !this._hasPass || used >= 1;
        const deadline = sync ? Infinity : performance.now() + JOB_BUDGET_MS;
        let step;
        do {
            step = this._job.next();
        } while (!step.done && performance.now() < deadline);
        if (!step.done) return;
        this._job = null;
        this._commit(step.value);
    }

    /**
     * Makes a completed job's result current.
     *
     * @param {object} result - The job's return value (see _select).
     * @private
     */
    _commit(result) {
        const world = this.world;
        this.previous.set(this.wanted);
        this.wanted.set(result.wanted);
        this.pages.set(result.pages);
        for (let b = 0; b < 3; b++) {
            this.indices[b] = result.indices[b];
            this.requestedByBucket[b] = result.requestedByBucket[b];
        }
        this.records = result.records;
        this.deferred = result.deferred;
        this.requestedIndices = result.requestedIndices;
        this.requestedRecords = result.requestedRecords;
        this.recordDemand = result.recordDemand;
        this._partition(result.capacities, result.totalCapacity, result.usedIndices);
        // The culler feeds the capacities this pass partitioned straight back in as the next
        // update's input. Expect them, so only an external change (the draw buffers growing)
        // reruns - otherwise every pass that shifts admissions a little invalidates itself.
        for (let b = 0; b < 3; b++) this._inputs[b] = this.capacity[b];
        this._passPlanes.set(this._jobPlanes);
        this._passCamera[0] = this._jobCamera.x;
        this._passCamera[1] = this._jobCamera.y;
        this._passCamera[2] = this._jobCamera.z;
        this._passNearest = result.nearest;
        this._hasPass = true;

        const wanted = this.wanted, previous = this.previous;
        let settled = true;
        for (let i = 0; i < wanted.length; i++) {
            if (wanted[i] !== previous[i]) {
                settled = false;
                break;
            }
        }
        this._settled = settled;
        if (!settled) this.wantedVersion++;

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
     * @returns {{hard: boolean, soft: boolean}} Whether an input that invalidates a running pass
     * (hard) or one that only calls for another pass (soft) differs from the previous update.
     * @private
     */
    _storeInputs(capacities, recordCapacity, shadow) {
        const inputs = this._inputs;
        const world = this.world;
        let soft = false, hard = false;
        let k = 0;
        for (let b = 0; b < 3; b++) soft = storeInput(inputs, k++, capacities[b]) || soft;
        soft = storeInput(inputs, k++, recordCapacity) || soft;
        hard = storeInput(inputs, k++, shadow ? 1 : 0) || hard;
        hard = storeInput(inputs, k++, world.poolSlots) || hard;
        hard = storeInput(inputs, k++, world.objectVersion ?? 0) || hard;
        soft = storeInput(inputs, k++, world.rootPagesVersion ?? 0) || soft;
        return { hard, soft };
    }

    /**
     * How much of the committed pass's coverage the view has used (see ADMIT_MOVE): 0 at the
     * pass's own view, 1 where the admitted margin runs out (Infinity past a turn it cannot
     * absorb). Plane offsets are compared relative to the camera, so a pure rotation leaves them
     * unchanged however far the camera is from the origin, while a clip-distance change counts as
     * translation.
     *
     * @param {Float32Array} planes - World frustum planes.
     * @param {import('../../core/math/vec3.js').Vec3} camera - Camera position.
     * @returns {number} The fraction of the coverage used.
     * @private
     */
    _coverageUsed(planes, camera) {
        if (!this._hasPass) return Infinity;
        const pc = this._passCamera, pp = this._passPlanes;
        const dx = camera.x - pc[0], dy = camera.y - pc[1], dz = camera.z - pc[2];
        let move = Math.sqrt(dx * dx + dy * dy + dz * dz);
        let minCos = 1;
        for (let p = 0; p < 24; p += 4) {
            const nx = planes[p], ny = planes[p + 1], nz = planes[p + 2];
            const ox = pp[p], oy = pp[p + 1], oz = pp[p + 2];
            // degenerate (zeroed) planes must stay zeroed
            const lenSq = nx * nx + ny * ny + nz * nz, oldSq = ox * ox + oy * oy + oz * oz;
            if ((lenSq === 0) !== (oldSq === 0)) return Infinity;
            if (lenSq === 0) continue;
            minCos = Math.min(minCos, (nx * ox + ny * oy + nz * oz) / Math.sqrt(lenSq * oldSq));
            const rel = planes[p + 3] + nx * camera.x + ny * camera.y + nz * camera.z;
            const oldRel = pp[p + 3] + ox * pc[0] + oy * pc[1] + oz * pc[2];
            move = Math.max(move, Math.abs(rel - oldRel));
        }
        if (minCos < 0) return Infinity;
        const sinTurn = Math.sqrt(Math.max(0, 1 - minCos * minCos));
        if (sinTurn >= ADMIT_SIN) return Infinity;
        // the turn consumes the angular margin; translation must fit in what remains of it at
        // the nearest instance, plus the fixed margin
        const turn = sinTurn / ADMIT_SIN;
        return Math.max(turn, move / (ADMIT_MOVE + this._passNearest * (ADMIT_SIN - sinTurn)));
    }

    /**
     * World-space bounding spheres of every instance (cx, cy, cz, r) and their visibility flags,
     * packed tight and bucketed into a coarse grid of cells, each with a bounding sphere. A pass
     * then rejects or accepts whole cells against the frustum, and reads 17 bytes per instance
     * in cells it must test instead of a 128-byte object row. Rebuilt only when the world's
     * object data changes.
     *
     * @private
     */
    _buildSpatial() {
        const world = this.world;
        const version = world.objectVersion ?? 0;
        if (this._spheres && this._spheresVersion === version) return;
        const f = world.objectDataCpuF;
        const objects = world.objectDataCpu;
        const count = world.instanceCount;
        const spheres = new Float32Array(count * 4);
        const flags = new Uint8Array(count);
        let minX = Infinity, minZ = Infinity, maxX = -Infinity, maxZ = -Infinity;
        for (let i = 0; i < count; i++) {
            const row = i * OBJECT_DATA_U32S;
            const x = f[row + O.SPHERE], y = f[row + O.SPHERE + 1], z = f[row + O.SPHERE + 2];
            const cx = f[row] * x + f[row + 4] * y + f[row + 8] * z + f[row + 12];
            const cz = f[row + 2] * x + f[row + 6] * y + f[row + 10] * z + f[row + 14];
            spheres[i * 4] = cx;
            spheres[i * 4 + 1] = f[row + 1] * x + f[row + 5] * y + f[row + 9] * z + f[row + 13];
            spheres[i * 4 + 2] = cz;
            spheres[i * 4 + 3] = f[row + O.SPHERE + 3] * f[row + O.MAX_SCALE];
            const fl = objects[row + O.FLAGS];
            flags[i] = ((fl & OBJECT_FLAG_HIDDEN) ? 1 : 0) | ((fl & OBJECT_FLAG_NO_SHADOW) ? 2 : 0);
            if (cx < minX) minX = cx;
            if (cx > maxX) maxX = cx;
            if (cz < minZ) minZ = cz;
            if (cz > maxZ) maxZ = cz;
        }

        // grid over the horizontal extent, about CELL_TARGET cells; instances larger than a cell
        // would bloat their cell's sphere, so they are tested on their own
        const extent = Math.max(maxX - minX, maxZ - minZ, 1e-3);
        const cellSize = extent / Math.sqrt(CELL_TARGET);
        const cols = Math.max(Math.ceil((maxX - minX) / cellSize), 1);
        const rows = Math.max(Math.ceil((maxZ - minZ) / cellSize), 1);
        const cellCount = cols * rows;
        const cellOf = new Int32Array(count);
        const cellStart = new Uint32Array(cellCount + 1);
        let largeCount = 0;
        for (let i = 0; i < count; i++) {
            if (spheres[i * 4 + 3] > cellSize) {
                cellOf[i] = -1;
                largeCount++;
                continue;
            }
            const cx = Math.min(Math.floor((spheres[i * 4] - minX) / cellSize), cols - 1);
            const cz = Math.min(Math.floor((spheres[i * 4 + 2] - minZ) / cellSize), rows - 1);
            cellOf[i] = cz * cols + cx;
            cellStart[cellOf[i] + 1]++;
        }
        for (let c = 0; c < cellCount; c++) cellStart[c + 1] += cellStart[c];
        const cellItems = new Uint32Array(count - largeCount);
        const large = new Uint32Array(largeCount);
        const fill = cellStart.slice(0, cellCount);
        let l = 0;
        for (let i = 0; i < count; i++) {
            if (cellOf[i] < 0) large[l++] = i;
            else cellItems[fill[cellOf[i]]++] = i;
        }
        // cell bounding spheres: centre of the members' box, radius covering every member sphere
        // (computed on instance ids, before the reorder below)
        const cellSpheres = new Float32Array(cellCount * 4);
        for (let c = 0; c < cellCount; c++) {
            const a = cellStart[c], b = cellStart[c + 1];
            if (a === b) continue;
            let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
            for (let k = a; k < b; k++) {
                const s = cellItems[k] * 4;
                x0 = Math.min(x0, spheres[s]);
                x1 = Math.max(x1, spheres[s]);
                y0 = Math.min(y0, spheres[s + 1]);
                y1 = Math.max(y1, spheres[s + 1]);
                z0 = Math.min(z0, spheres[s + 2]);
                z1 = Math.max(z1, spheres[s + 2]);
            }
            const mx = (x0 + x1) * 0.5, my = (y0 + y1) * 0.5, mz = (z0 + z1) * 0.5;
            let radius = 0;
            for (let k = a; k < b; k++) {
                const s = cellItems[k] * 4;
                const dx = spheres[s] - mx, dy = spheres[s + 1] - my, dz = spheres[s + 2] - mz;
                radius = Math.max(radius, Math.sqrt(dx * dx + dy * dy + dz * dz) + spheres[s + 3]);
            }
            cellSpheres.set([mx, my, mz, radius], c * 4);
        }
        // Reorder into cell order - cell c owns positions [cellStart[c], cellStart[c + 1]), the
        // large instances follow - so a pass streams its cells' spheres, flags and roots
        // sequentially. Passes work on positions; posToId maps back to instance ids.
        const posToId = new Uint32Array(count);
        posToId.set(cellItems, 0);
        posToId.set(large, cellItems.length);
        const posSpheres = new Float32Array(count * 4), posFlags = new Uint8Array(count);
        const posRoot = new Uint32Array(count);
        const rootOf = this._rootOf;
        for (let k = 0; k < count; k++) {
            const i = posToId[k];
            posSpheres[k * 4] = spheres[i * 4];
            posSpheres[k * 4 + 1] = spheres[i * 4 + 1];
            posSpheres[k * 4 + 2] = spheres[i * 4 + 2];
            posSpheres[k * 4 + 3] = spheres[i * 4 + 3];
            posFlags[k] = flags[i];
            posRoot[k] = rootOf[i];
        }
        this._spheres = posSpheres;
        this._flags = posFlags;
        this._posRoot = posRoot;
        this._posToId = posToId;
        this._cellStart = cellStart;
        this._cellSpheres = cellSpheres;
        this._largeStart = cellItems.length;
        this._spheresVersion = version;
    }

    /**
     * Flattens the cut's per-instance root descriptors into typed arrays. Instances of one
     * primitive share a descriptor, so costs and pages are stored once per descriptor (pages as
     * a CSR list) and instances map to it by id.
     *
     * @private
     */
    _flattenRoots() {
        const roots = this.world.cut.instanceRoots;
        if (this._flatSource === roots) return;
        const ids = new Map();
        const rootOf = new Uint32Array(roots.length);
        const unique = [];
        for (let i = 0; i < roots.length; i++) {
            let id = ids.get(roots[i]);
            if (id === undefined) {
                id = unique.length;
                ids.set(roots[i], id);
                unique.push(roots[i]);
            }
            rootOf[i] = id;
        }
        const n = unique.length;
        const indexCost = new Float64Array(n), recordCost = new Float64Array(n);
        const bucket = new Uint8Array(n), pageStart = new Uint32Array(n + 1);
        for (let r = 0; r < n; r++) pageStart[r + 1] = pageStart[r] + unique[r].pages.length;
        const pageList = new Uint32Array(pageStart[n]);
        for (let r = 0; r < n; r++) {
            const root = unique[r];
            indexCost[r] = root.cost[0];
            recordCost[r] = root.cost[1];
            bucket[r] = root.bucket;
            pageList.set(root.pages, pageStart[r]);
        }
        this._flatSource = roots;
        this._spheresVersion = -1;
        this._rootOf = rootOf;
        this._indexCost = indexCost;
        this._recordCost = recordCost;
        this._rootBucket = bucket;
        this._pageStart = pageStart;
        this._pageList = pageList;
        this._rootStamp = new Uint32Array(n);
        this._stamp = 0;
        this._visible = new Uint32Array(roots.length);
    }

    /**
     * The full selection pass over every instance, as a job: a generator that yields every
     * JOB_YIELD_EVERY instances (and between sort passes) and returns the result, written to
     * the job buffers - never to the committed wanted/pages other views and the culler read.
     *
     * @param {Float32Array} planes - World frustum planes.
     * @param {import('../../core/math/vec3.js').Vec3} camera - Camera position.
     * @param {number[]} capacities - Index capacities by bucket.
     * @param {number} recordCapacity - Draw record capacity.
     * @param {boolean} shadow - Whether to exclude non-casters.
     * @returns {Generator<undefined, object>} The job; its return value is the result.
     * @private
     */
    *_select(planes, camera, capacities, recordCapacity, shadow) {
        const world = this.world;
        this._flattenRoots();
        this._buildSpatial();
        const spheres = this._spheres;
        const indexCost = this._indexCost, recordCost = this._recordCost;
        const rootBucket = this._rootBucket, pageStart = this._pageStart, pageList = this._pageList;
        const rootStamp = this._rootStamp;
        // a fresh stamp per walk marks descriptors whose pages were already walked this pass
        const nextStamp = () => {
            if (++this._stamp === 0xffffffff) {
                rootStamp.fill(0);
                this._stamp = 1;
            }
            return this._stamp;
        };
        const wanted = this._jobWanted, pages = this._jobPages, distance = this.distance;
        // hysteresis favours what the committed result admits
        const committed = this.wanted;
        const indices = [0, 0, 0], requestedByBucket = [0, 0, 0];
        wanted.fill(0);
        pages.fill(0);
        let records = 0, deferred = 0, requestedIndices = 0, requestedRecords = 0, recordDemand = 0;
        let work = 0;
        // distance of the closest instance seen, rejected or not (see _coverageUsed)
        let nearest = Infinity;
        const totalCapacity = capacities[0] + capacities[1] + capacities[2];
        let usedIndices = 0;
        const occupied = this._occupied;
        occupied.fill(0);
        let pageCount = 0;
        for (const other of world.rootSelections ?? []) {
            if (other === this) continue;
            const otherPages = other.pages;
            for (let p = 0; p < occupied.length; p++) {
                if (otherPages[p] && !occupied[p]) {
                    occupied[p] = 1;
                    pageCount++;
                }
            }
        }
        // Totals of every visible root, with its page union marked in `pages`. When they fit,
        // admission order is irrelevant and the distance sort (the dominant CPU cost with tens
        // of thousands of placements) is skipped.
        const visible = this._visible;
        let visibleCount = 0;
        let visibleIndices = 0, visibleRecords = 0, visiblePages = 0;
        const camX = camera.x, camY = camera.y, camZ = camera.z;
        let stamp = nextStamp();
        const reject = shadow ? 3 : 1;
        const flags = this._flags;
        const p0 = planes[0], p1 = planes[1], p2 = planes[2], p3 = planes[3];
        const p4 = planes[4], p5 = planes[5], p6 = planes[6], p7 = planes[7];
        const p8 = planes[8], p9 = planes[9], p10 = planes[10], p11 = planes[11];
        const p12 = planes[12], p13 = planes[13], p14 = planes[14], p15 = planes[15];
        const p16 = planes[16], p17 = planes[17], p18 = planes[18], p19 = planes[19];
        const p20 = planes[20], p21 = planes[21], p22 = planes[22], p23 = planes[23];
        const cellStart = this._cellStart, cellSpheres = this._cellSpheres;
        const posRoot = this._posRoot, posToId = this._posToId;
        const cellCount = cellStart.length - 1;
        // cells first, then the instances too large for a cell; hot loop kept closure-free
        for (let c = 0; c <= cellCount; c++) {
            let start, end, inside = false;
            if (c < cellCount) {
                start = cellStart[c];
                end = cellStart[c + 1];
                if (start === end) continue;
                const q = c * 4;
                const x = cellSpheres[q], y = cellSpheres[q + 1], z = cellSpheres[q + 2], cr = cellSpheres[q + 3];
                // widen by how far the view may move or turn before a newer pass lands (ADMIT_MOVE)
                const cdx = x - camX, cdy = y - camY, cdz = z - camZ;
                const cellMargin = ADMIT_MOVE + (Math.sqrt(cdx * cdx + cdy * cdy + cdz * cdz) + cr) * ADMIT_SIN;
                const d = Math.min(
                    p0 * x + p1 * y + p2 * z + p3, p4 * x + p5 * y + p6 * z + p7,
                    p8 * x + p9 * y + p10 * z + p11, p12 * x + p13 * y + p14 * z + p15,
                    p16 * x + p17 * y + p18 * z + p19, p20 * x + p21 * y + p22 * z + p23);
                if (d < -(cr + cellMargin)) {
                    nearest = Math.min(nearest, Math.max(Math.sqrt(cdx * cdx + cdy * cdy + cdz * cdz) - cr, 0));
                    continue;
                }
                inside = d >= cr;
            } else {
                start = this._largeStart;
                end = world.instanceCount;
            }
            for (let k = start; k < end; k++) {
                if (++work === JOB_YIELD_EVERY) {
                    work = 0;
                    yield;
                }
                if (flags[k] & reject) continue;
                const s = k * 4;
                const cx = spheres[s], cy = spheres[s + 1], cz = spheres[s + 2], radius = spheres[s + 3];
                const dx = cx - camX, dy = cy - camY, dz = cz - camZ;
                const centre = Math.sqrt(dx * dx + dy * dy + dz * dz);
                const gap = Math.max(centre - radius, 0);
                if (gap < nearest) nearest = gap;
                if (!inside) {
                    const reach = -(radius + ADMIT_MOVE + (centre + radius) * ADMIT_SIN);
                    if (p0 * cx + p1 * cy + p2 * cz + p3 < reach ||
                        p4 * cx + p5 * cy + p6 * cz + p7 < reach ||
                        p8 * cx + p9 * cy + p10 * cz + p11 < reach ||
                        p12 * cx + p13 * cy + p14 * cz + p15 < reach ||
                        p16 * cx + p17 * cy + p18 * cz + p19 < reach ||
                        p20 * cx + p21 * cy + p22 * cz + p23 < reach) continue;
                }
                distance[k] = gap;
                visible[visibleCount++] = k;
                const r = posRoot[k];
                visibleIndices += indexCost[r];
                visibleRecords += recordCost[r];
                if (rootStamp[r] !== stamp) {
                    rootStamp[r] = stamp;
                    for (let e = pageStart[r + 1], m = pageStart[r]; m < e; m++) {
                        const page = pageList[m];
                        if (!pages[page]) {
                            pages[page] = 1;
                            if (!occupied[page]) visiblePages++;
                        }
                    }
                }
            }
        }
        const order = visible.subarray(0, visibleCount);
        if (visibleIndices <= totalCapacity && visibleRecords <= recordCapacity && pageCount + visiblePages <= world.poolSlots) {
            for (let k = 0; k < visibleCount; k++) {
                const r = posRoot[order[k]];
                wanted[posToId[order[k]]] = 1;
                indices[rootBucket[r]] += indexCost[r];
                requestedByBucket[rootBucket[r]] += indexCost[r];
            }
            return {
                wanted,
                pages,
                indices,
                requestedByBucket,
                capacities,
                totalCapacity,
                usedIndices: visibleIndices,
                records: visibleRecords,
                deferred: 0,
                requestedIndices: visibleIndices,
                requestedRecords: visibleRecords,
                recordDemand: visibleRecords,
                nearest
            };
        }
        pages.fill(0);
        const keys = this._keys;
        for (let k = 0; k < visibleCount; k++) {
            const pos = order[k];
            keys[pos] = distance[pos] * (committed[posToId[pos]] ? 0.8 : 1);
        }
        yield;
        const sorted = yield* this._radixSort(order);
        // descriptors whose pages this pass already made resident: later instances add none
        stamp = nextStamp();
        for (let k = 0; k < sorted.length; k++) {
            if (++work === JOB_YIELD_EVERY) {
                work = 0;
                yield;
            }
            const pos = sorted[k];
            const r = posRoot[pos];
            const cost = indexCost[r], rootRecords = recordCost[r], bucket = rootBucket[r];
            requestedIndices += cost;
            requestedRecords += rootRecords;
            requestedByBucket[bucket] += cost;
            if (usedIndices + cost > totalCapacity) {
                deferred++;
                continue;
            }
            let additional = 0;
            if (rootStamp[r] !== stamp) {
                for (let p = pageStart[r], e = pageStart[r + 1]; p < e; p++) additional += 1 - occupied[pageList[p]];
            }
            if (pageCount + additional > world.poolSlots) {
                deferred++;
                continue;
            }
            // Grow records only for a candidate that could otherwise enter this working
            // set, not every visible root that the index/page budget already excludes.
            recordDemand = Math.max(recordDemand, records + rootRecords);
            if (records + rootRecords > recordCapacity) {
                deferred++;
                continue;
            }
            wanted[posToId[pos]] = 1;
            indices[bucket] += cost;
            usedIndices += cost;
            records += rootRecords;
            pageCount += additional;
            if (rootStamp[r] !== stamp) {
                rootStamp[r] = stamp;
                for (let p = pageStart[r], e = pageStart[r + 1]; p < e; p++) {
                    const page = pageList[p];
                    pages[page] = 1;
                    occupied[page] = 1;
                }
            }
        }

        return {
            wanted,
            pages,
            indices,
            requestedByBucket,
            capacities,
            totalCapacity,
            usedIndices,
            records,
            deferred,
            requestedIndices,
            requestedRecords,
            recordDemand,
            nearest
        };
    }

    /**
     * Stable LSD radix sort of instance positions (see _buildSpatial) by their key. Non-negative
     * float bits order like the floats, and ties keep `order`'s cell order, without a
     * comparator sort costing milliseconds per view over tens of thousands of instances.
     *
     * A generator that yields after each pass, for the selection job.
     *
     * @param {ArrayLike<number>} order - Visible instance positions, in cell order.
     * @returns {Generator<undefined, Uint32Array>} The indices ordered by key, as its return value.
     * @private
     */
    *_radixSort(order) {
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
            yield;
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
