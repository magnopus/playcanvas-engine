/**
 * @import { Mat4 } from '../../core/math/mat4.js'
 * @import { MeshletLightmap } from './meshlet-resource.js'
 */

// floats per stored transform: the affine part of a column-major 4x4 matrix
const TRANSFORM_FLOATS = 12;

/**
 * Collects placements for a {@link MeshletInstanceList} without knowing their count up front;
 * {@link MeshletInstanceList.from} turns it into a list.
 *
 * @ignore
 */
class MeshletInstanceListBuilder {
    count = 0;

    transformCount = 0;

    /** @type {Array<MeshletLightmap|null>|null} */
    lightmaps = null;

    /**
     * @param {number} [capacity] - Expected instance count.
     */
    constructor(capacity = 16) {
        capacity = Math.max(capacity, 1);
        this.primIndex = new Uint32Array(capacity);
        this.transformIndex = new Uint32Array(capacity);
        this.transforms = new Float32Array(capacity * TRANSFORM_FLOATS);
    }

    /**
     * Adds a transform.
     *
     * @param {ArrayLike<number>} matrix - Column-major 4x4 matrix.
     * @param {MeshletLightmap|null} [lightmap] - Its lightmap.
     * @returns {number} The transform index.
     */
    addTransform(matrix, lightmap = null) {
        const index = this.transformCount++;
        if (this.transformCount * TRANSFORM_FLOATS > this.transforms.length) {
            const grown = new Float32Array(this.transforms.length * 2);
            grown.set(this.transforms);
            this.transforms = grown;
        }
        const t = index * TRANSFORM_FLOATS;
        const d = this.transforms;
        d[t] = matrix[0]; d[t + 1] = matrix[1]; d[t + 2] = matrix[2];
        d[t + 3] = matrix[4]; d[t + 4] = matrix[5]; d[t + 5] = matrix[6];
        d[t + 6] = matrix[8]; d[t + 7] = matrix[9]; d[t + 8] = matrix[10];
        d[t + 9] = matrix[12]; d[t + 10] = matrix[13]; d[t + 11] = matrix[14];
        if (lightmap && !this.lightmaps) this.lightmaps = new Array(index).fill(null);
        this.lightmaps?.push(lightmap ?? null);
        return index;
    }

    /**
     * A transform's lightmap.
     *
     * @param {number} transform - The transform.
     * @returns {MeshletLightmap|null} The lightmap, or null.
     */
    lightmapOf(transform) {
        return this.lightmaps ? this.lightmaps[transform] : null;
    }

    /**
     * Adds an instance.
     *
     * @param {number} primIndex - The primitive.
     * @param {number} transform - The transform index.
     */
    add(primIndex, transform) {
        if (this.count === this.primIndex.length) {
            const grow = (/** @type {Uint32Array} */ a) => {
                const grown = new Uint32Array(a.length * 2);
                grown.set(a);
                return grown;
            };
            this.primIndex = grow(this.primIndex);
            this.transformIndex = grow(this.transformIndex);
        }
        this.primIndex[this.count] = primIndex;
        this.transformIndex[this.count] = transform;
        this.count++;
    }

    /**
     * What was added, trimmed to size.
     *
     * @returns {{ primIndex: Uint32Array, transformIndex: Uint32Array, transforms: Float32Array, lightmaps: Array<MeshletLightmap|null>|null }} The arrays.
     */
    finish() {
        return {
            primIndex: this.primIndex.slice(0, this.count),
            transformIndex: this.transformIndex.slice(0, this.count),
            transforms: this.transforms.slice(0, this.transformCount * TRANSFORM_FLOATS),
            lightmaps: this.lightmaps
        };
    }
}

/**
 * The placements of a meshlet resource's primitives, packed into typed arrays. Each instance
 * is one primitive at one transform. The primitives of one glTF node share its transform (and
 * an EXT_mesh_gpu_instancing scatter entry is one transform for every primitive of its mesh),
 * so transforms are stored once and instances index them.
 *
 * A scattered scene has millions of instances. One JS object and one Float32Array per
 * instance cost several hundred bytes each in the heap; here an instance is 8 bytes and its
 * transform 48 bytes, shared by the primitives that use it.
 *
 * @ignore
 */
class MeshletInstanceList {
    /** @type {number} - instance count. */
    count;

    /** @type {Uint32Array} - per instance, the resource primitive it places. */
    primIndex;

    /** @type {Uint32Array} - per instance, its transform in {@link transforms}. */
    transformIndex;

    /**
     * The transforms, {@link TRANSFORM_FLOATS} floats each: the affine part of a column-major
     * matrix (columns 0-3, rows 0-2).
     *
     * @type {Float32Array}
     */
    transforms;

    /**
     * Per transform, its EPIC lightmap, or null when no transform has one.
     *
     * @type {Array<MeshletLightmap|null>|null}
     */
    lightmaps;

    /**
     * @param {Uint32Array} primIndex - Per instance, the primitive.
     * @param {Uint32Array} transformIndex - Per instance, the transform.
     * @param {Float32Array} transforms - Packed affine transforms.
     * @param {Array<MeshletLightmap|null>|null} [lightmaps] - Per transform lightmaps.
     */
    constructor(primIndex, transformIndex, transforms, lightmaps = null) {
        this.count = primIndex.length;
        this.primIndex = primIndex;
        this.transformIndex = transformIndex;
        this.transforms = transforms;
        this.lightmaps = lightmaps;
    }

    /**
     * An empty list.
     *
     * @returns {MeshletInstanceList} The list.
     */
    static empty() {
        return new MeshletInstanceList(new Uint32Array(0), new Uint32Array(0), new Float32Array(0));
    }

    /**
     * Returns the source as a list: a list is returned as is, a builder's placements are
     * wrapped, and an array of `{ primIndex, matrix, lightmap }` placements (column-major
     * 16-float matrices) is packed. Placements that share one matrix object share one transform.
     *
     * @param {MeshletInstanceList|MeshletInstanceListBuilder|Array<{ primIndex: number, matrix: ArrayLike<number>, lightmap?: MeshletLightmap|null }>|null|undefined} source - The placements.
     * @returns {MeshletInstanceList} The list.
     */
    static from(source) {
        if (source instanceof MeshletInstanceList) return source;
        if (!source) return MeshletInstanceList.empty();
        if (source instanceof MeshletInstanceListBuilder) {
            const { primIndex, transformIndex, transforms, lightmaps } = source.finish();
            return new MeshletInstanceList(primIndex, transformIndex, transforms, lightmaps);
        }
        const builder = new MeshletInstanceListBuilder(source.length);
        const shared = new Map();
        for (const { primIndex, matrix, lightmap } of source) {
            let transform = shared.get(matrix);
            if (transform === undefined || builder.lightmapOf(transform) !== (lightmap ?? null)) {
                transform = builder.addTransform(matrix, lightmap ?? null);
                shared.set(matrix, transform);
            }
            builder.add(primIndex, transform);
        }
        return MeshletInstanceList.from(builder);
    }

    /** @type {number} - alias of {@link count}. */
    get length() {
        return this.count;
    }

    /**
     * Writes an instance's transform into a matrix.
     *
     * @param {number} index - The instance.
     * @param {Mat4} out - Receives the column-major matrix.
     * @returns {Mat4} The matrix.
     */
    getMatrix(index, out) {
        const src = this.transforms;
        const t = this.transformIndex[index] * TRANSFORM_FLOATS;
        const d = out.data;
        d[0] = src[t]; d[1] = src[t + 1]; d[2] = src[t + 2]; d[3] = 0;
        d[4] = src[t + 3]; d[5] = src[t + 4]; d[6] = src[t + 5]; d[7] = 0;
        d[8] = src[t + 6]; d[9] = src[t + 7]; d[10] = src[t + 8]; d[11] = 0;
        d[12] = src[t + 9]; d[13] = src[t + 10]; d[14] = src[t + 11]; d[15] = 1;
        return out;
    }

    /**
     * An instance's lightmap.
     *
     * @param {number} index - The instance.
     * @returns {MeshletLightmap|null} The lightmap, or null.
     */
    getLightmap(index) {
        return this.lightmaps ? this.lightmaps[this.transformIndex[index]] : null;
    }

    /**
     * One instance as a placement object - for inspection; it allocates.
     *
     * @param {number} index - The instance.
     * @returns {{ primIndex: number, matrix: Float32Array, lightmap: MeshletLightmap|null }} The placement.
     */
    get(index) {
        const matrix = new Float32Array(16);
        this.getMatrix(index, /** @type {Mat4} */ ({ data: matrix }));
        return { primIndex: this.primIndex[index], matrix, lightmap: this.getLightmap(index) };
    }

    /**
     * The instances a predicate keeps, sharing this list's transforms.
     *
     * @param {(primIndex: number, index: number) => boolean} predicate - Called per instance.
     * @returns {MeshletInstanceList} The kept instances.
     */
    filter(predicate) {
        const keep = [];
        for (let i = 0; i < this.count; i++) {
            if (predicate(this.primIndex[i], i)) keep.push(i);
        }
        const primIndex = new Uint32Array(keep.length);
        const transformIndex = new Uint32Array(keep.length);
        keep.forEach((i, k) => {
            primIndex[k] = this.primIndex[i];
            transformIndex[k] = this.transformIndex[i];
        });
        return new MeshletInstanceList(primIndex, transformIndex, this.transforms, this.lightmaps);
    }

    /**
     * A range of the instances, sharing this list's transforms.
     *
     * @param {number} [start] - First instance.
     * @param {number} [end] - One past the last instance.
     * @returns {MeshletInstanceList} The range.
     */
    slice(start = 0, end = this.count) {
        return new MeshletInstanceList(this.primIndex.slice(start, end), this.transformIndex.slice(start, end), this.transforms, this.lightmaps);
    }
}

export { MeshletInstanceList, MeshletInstanceListBuilder };
