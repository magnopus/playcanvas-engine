import { BUFFERUSAGE_COPY_DST } from '../../platform/graphics/constants.js';
import { StorageBuffer } from '../../platform/graphics/storage-buffer.js';
import { MESHLET_CELL_SIZE, OBJECT_DATA, OBJECT_DATA_U32S } from './constants.js';

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 */

// Morton bits per axis. 8 bits order a few-km world at ~10 m resolution, plenty for grouping
// 64 neighbours, and keep the sort key an exact double (see _order).
const MORTON_BITS = 8;

// Instance index bits in the sort key - the same ceiling as the root list's instance field.
const INDEX_BITS = 27;
const INDEX_SCALE = 2 ** INDEX_BITS;

// Radius classes, one per octave: class = floor(log2(radius)) - CLASS_MIN_LOG2, clamped.
// Instances are grouped by class before position, so a cell's members are within 2x of each
// other's size. A cell's sphere encloses its largest member and the cell cull's size test uses
// that member's radius, so one large instance in a cell of small ones would keep all of them
// visible to every view that sees it - a 1 km terrain tile among crates, or a boulder among
// pebbles: with coarse classes, a far cell of clutter survives the size cut on its one big
// member and every small one is still classified each frame.
const CLASS_MIN_LOG2 = -8;
const CLASS_COUNT = 24;

// spreads the low 8 bits of v so there are two zero bits between each
const spread = (/** @type {number} */ v) => {
    v = (v | (v << 8)) & 0x00F00F;
    v = (v | (v << 4)) & 0x0C30C3;
    v = (v | (v << 2)) & 0x249249;
    return v;
};

/**
 * Groups the world's instances into spatial cells of up to {@link MESHLET_CELL_SIZE}, so a
 * view culls cells first and runs its per-instance and per-task passes over the visible
 * cells only. Without them every view - each shadow cascade included - classifies every
 * instance and visits every cut task of the world, and on a large world that sweep, not the
 * drawing, dominates the shadow cost.
 *
 * Instances are ordered by radius class, then Morton code of their world-space centre, and
 * chunked: cell `c` holds `instanceOrder[c * MESHLET_CELL_SIZE ...]`. Only the order changes -
 * instance ids, and so objectData, claim bits and pick ids, are untouched. The cut tasks of
 * each LOD level are laid out in the same order (see {@link MeshletCutData}), so a cell's tasks
 * at a level are one contiguous range.
 *
 * Culling by cell is exact: an instance in a culled cell fails the instance frustum test too,
 * so it could never have been admitted and its tasks would only have early-outed.
 *
 * @ignore
 */
class MeshletCells {
    /** @type {GraphicsDevice} */
    device;

    /** @type {number} */
    cellCount = 0;

    /** @type {Uint32Array} - instance ids, cell by cell. */
    instanceOrder;

    /** @type {Uint32Array} - the cell each instance belongs to. */
    instanceCell;

    /** @type {Float32Array} - per cell, the world-space bounding sphere (x, y, z, radius). */
    spheres;

    /**
     * Per cell, the radius of its largest member. A view whose root admission drops instances
     * below a projected size culls a whole cell when even this member, at the cell's nearest
     * point, is below it - a far cell of small props then costs no per-instance work at all.
     *
     * @type {Float32Array}
     */
    maxRadii;

    /** @type {StorageBuffer|null} - {@link maxRadii} on the GPU. */
    radiiBuffer = null;

    /** @type {StorageBuffer|null} - {@link instanceOrder} on the GPU. */
    instancesBuffer = null;

    /** @type {StorageBuffer|null} - {@link spheres} on the GPU. */
    boundsBuffer = null;

    /**
     * @param {GraphicsDevice} device - The graphics device.
     * @param {Float32Array} objectDataF - The world's objectData rows, as floats.
     * @param {number} instanceCount - The instance count.
     */
    constructor(device, objectDataF, instanceCount) {
        this.device = device;
        this.cellCount = Math.ceil(instanceCount / MESHLET_CELL_SIZE);

        const centers = new Float32Array(instanceCount * 4);
        for (let i = 0; i < instanceCount; i++) {
            MeshletCells._instanceSphere(objectDataF, i, centers, i * 4);
        }

        this.instanceOrder = MeshletCells._order(centers, instanceCount);
        this.instanceCell = new Uint32Array(instanceCount);
        for (let p = 0; p < instanceCount; p++) {
            this.instanceCell[this.instanceOrder[p]] = Math.floor(p / MESHLET_CELL_SIZE);
        }

        this.spheres = new Float32Array(Math.max(this.cellCount, 1) * 4);
        this.maxRadii = new Float32Array(Math.max(this.cellCount, 1));
        for (let c = 0; c < this.cellCount; c++) {
            this._fitCell(c, centers);
        }

        this.instancesBuffer = new StorageBuffer(device, Math.max(this.instanceOrder.byteLength, 16), BUFFERUSAGE_COPY_DST);
        if (instanceCount) this.instancesBuffer.write(0, this.instanceOrder);
        this.boundsBuffer = new StorageBuffer(device, this.spheres.byteLength, BUFFERUSAGE_COPY_DST);
        this.boundsBuffer.write(0, this.spheres);
        this.radiiBuffer = new StorageBuffer(device, Math.max(this.maxRadii.byteLength, 16), BUFFERUSAGE_COPY_DST);
        this.radiiBuffer.write(0, this.maxRadii);
    }

    destroy() {
        this.instancesBuffer?.destroy();
        this.instancesBuffer = null;
        this.boundsBuffer?.destroy();
        this.boundsBuffer = null;
        this.radiiBuffer?.destroy();
        this.radiiBuffer = null;
    }

    /**
     * Refits the cells holding a range of instances after their transforms changed. Membership
     * is kept - an instance that moves far just loosens its cell - so the cut task layout stays
     * valid without a rebuild.
     *
     * @param {Float32Array} objectDataF - The world's objectData rows, as floats.
     * @param {number} first - First instance whose row changed.
     * @param {number} count - Number of instances.
     */
    refit(objectDataF, first, count) {
        if (count <= 0) return;
        let lo = Infinity;
        let hi = -1;
        const touched = new Set();
        for (let i = first; i < first + count; i++) {
            touched.add(this.instanceCell[i]);
        }
        const centers = new Float32Array(MESHLET_CELL_SIZE * 4);
        touched.forEach((cell) => {
            const start = cell * MESHLET_CELL_SIZE;
            const end = Math.min(start + MESHLET_CELL_SIZE, this.instanceOrder.length);
            for (let p = start; p < end; p++) {
                MeshletCells._instanceSphere(objectDataF, this.instanceOrder[p], centers, (p - start) * 4);
            }
            this._fitSphere(cell, centers, end - start);
            lo = Math.min(lo, cell);
            hi = Math.max(hi, cell);
        });
        if (hi >= lo) {
            this.boundsBuffer?.write(lo * 16, this.spheres, lo * 4, (hi - lo + 1) * 4);
            this.radiiBuffer?.write(lo * 4, this.maxRadii, lo, hi - lo + 1);
        }
    }

    /**
     * Fits cell `cell` from the per-instance spheres of the whole world.
     *
     * @param {number} cell - The cell.
     * @param {Float32Array} centers - Per-instance world spheres, indexed by instance id.
     * @private
     */
    _fitCell(cell, centers) {
        const start = cell * MESHLET_CELL_SIZE;
        const end = Math.min(start + MESHLET_CELL_SIZE, this.instanceOrder.length);
        const local = new Float32Array((end - start) * 4);
        for (let p = start; p < end; p++) {
            local.set(centers.subarray(this.instanceOrder[p] * 4, this.instanceOrder[p] * 4 + 4), (p - start) * 4);
        }
        this._fitSphere(cell, local, end - start);
    }

    /**
     * Writes the sphere enclosing `count` member spheres: centred on their AABB, radius reaching
     * the farthest member's surface.
     *
     * @param {number} cell - The cell.
     * @param {Float32Array} members - Member spheres, packed.
     * @param {number} count - Member count.
     * @private
     */
    _fitSphere(cell, members, count) {
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        for (let m = 0; m < count; m++) {
            const x = members[m * 4], y = members[m * 4 + 1], z = members[m * 4 + 2], r = members[m * 4 + 3];
            minX = Math.min(minX, x - r); maxX = Math.max(maxX, x + r);
            minY = Math.min(minY, y - r); maxY = Math.max(maxY, y + r);
            minZ = Math.min(minZ, z - r); maxZ = Math.max(maxZ, z + r);
        }
        const cx = (minX + maxX) * 0.5, cy = (minY + maxY) * 0.5, cz = (minZ + maxZ) * 0.5;
        let radius = 0;
        let maxRadius = 0;
        for (let m = 0; m < count; m++) {
            const dx = members[m * 4] - cx, dy = members[m * 4 + 1] - cy, dz = members[m * 4 + 2] - cz;
            radius = Math.max(radius, Math.sqrt(dx * dx + dy * dy + dz * dz) + members[m * 4 + 3]);
            maxRadius = Math.max(maxRadius, members[m * 4 + 3]);
        }
        this.maxRadii[cell] = maxRadius * 1.001 + 1e-3;
        const o = cell * 4;
        this.spheres[o] = cx;
        this.spheres[o + 1] = cy;
        this.spheres[o + 2] = cz;
        // a hair of slack against float drift between this fit and the shader's own transform
        this.spheres[o + 3] = radius * 1.001 + 1e-3;
    }

    /**
     * World-space bounding sphere of one instance, as the cull shaders compute it.
     *
     * @param {Float32Array} objectDataF - The objectData rows, as floats.
     * @param {number} instance - The instance.
     * @param {Float32Array} out - Destination.
     * @param {number} offset - Destination offset.
     * @private
     */
    static _instanceSphere(objectDataF, instance, out, offset) {
        const row = instance * OBJECT_DATA_U32S;
        const m = OBJECT_DATA.MATRIX + row;
        const s = OBJECT_DATA.SPHERE + row;
        const x = objectDataF[s], y = objectDataF[s + 1], z = objectDataF[s + 2], r = objectDataF[s + 3];
        out[offset] = objectDataF[m] * x + objectDataF[m + 4] * y + objectDataF[m + 8] * z + objectDataF[m + 12];
        out[offset + 1] = objectDataF[m + 1] * x + objectDataF[m + 5] * y + objectDataF[m + 9] * z + objectDataF[m + 13];
        out[offset + 2] = objectDataF[m + 2] * x + objectDataF[m + 6] * y + objectDataF[m + 10] * z + objectDataF[m + 14];
        out[offset + 3] = r * objectDataF[row + OBJECT_DATA.MAX_SCALE];
    }

    /**
     * Instance ids sorted by radius class, then Morton code of their centre. Instances are
     * bucketed by class, then each bucket is sorted by a key packing Morton code (24 bits) and
     * instance id (27 bits) into an exact double, so a plain numeric sort does it.
     *
     * @param {Float32Array} centers - Per-instance world spheres.
     * @param {number} count - Instance count.
     * @returns {Uint32Array} The order.
     * @private
     */
    static _order(centers, count) {
        if (count >= INDEX_SCALE) {
            throw new Error(`MeshletCells: ${count} instances exceed the ${INDEX_SCALE} instance limit.`);
        }
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        for (let i = 0; i < count; i++) {
            minX = Math.min(minX, centers[i * 4]); maxX = Math.max(maxX, centers[i * 4]);
            minY = Math.min(minY, centers[i * 4 + 1]); maxY = Math.max(maxY, centers[i * 4 + 1]);
            minZ = Math.min(minZ, centers[i * 4 + 2]); maxZ = Math.max(maxZ, centers[i * 4 + 2]);
        }
        const cells = (1 << MORTON_BITS) - 1;
        const quantize = (/** @type {number} */ v, /** @type {number} */ lo, /** @type {number} */ hi) => (hi > lo ? Math.min(cells, Math.max(0, Math.floor((v - lo) / (hi - lo) * cells))) : 0);

        const classes = new Uint8Array(count);
        const classStarts = new Uint32Array(CLASS_COUNT + 1);
        for (let i = 0; i < count; i++) {
            const radius = centers[i * 4 + 3];
            const cls = radius > 0 ? Math.min(CLASS_COUNT - 1, Math.max(0, Math.floor(Math.log2(radius)) - CLASS_MIN_LOG2)) : 0;
            classes[i] = cls;
            classStarts[cls + 1]++;
        }
        for (let c = 0; c < CLASS_COUNT; c++) classStarts[c + 1] += classStarts[c];

        const keys = new Float64Array(count);
        const cursor = classStarts.slice(0, CLASS_COUNT);
        for (let i = 0; i < count; i++) {
            const morton = spread(quantize(centers[i * 4], minX, maxX)) |
                (spread(quantize(centers[i * 4 + 1], minY, maxY)) << 1) |
                (spread(quantize(centers[i * 4 + 2], minZ, maxZ)) << 2);
            keys[cursor[classes[i]]++] = morton * INDEX_SCALE + i;
        }
        for (let c = 0; c < CLASS_COUNT; c++) {
            keys.subarray(classStarts[c], classStarts[c + 1]).sort();
        }

        const order = new Uint32Array(count);
        for (let p = 0; p < count; p++) {
            order[p] = keys[p] % INDEX_SCALE;
        }
        return order;
    }
}

export { MeshletCells };
