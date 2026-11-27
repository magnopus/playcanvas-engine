import { Debug } from '../../core/debug.js';
import { FloatPacking } from '../../core/math/float-packing.js';
import {
    MATERIAL_FLAG_ALPHA_MASK, MATERIAL_FLAG_DOUBLE_SIDED, MATERIAL_RECORD, MATERIAL_RECORD_U32S, MATERIAL_SLOT_ABSENT,
    MATERIAL_TEXTURE_SLOTS, MESHLET_BUCKET_MASKED, MESHLET_BUCKET_OPAQUE, MESHLET_BUCKET_OPAQUE_TWO_SIDED,
    MESHLET_FLAG_ALPHA_MASKED, MESHLET_FLAG_TWO_SIDED, TEXEL_RATE_PER_MIP
} from './constants.js';

// texIndex field of a material slot word (texIndex:16 | arrayId:8 | texCoord:2)
const TEX_INDEX_MASK = 0xFFFF;

/**
 * The draw bucket of a meshlet's flags. A flag set that cannot tell alpha masking and
 * two-sidedness apart reads as masked, which is the safe way to be wrong.
 *
 * @param {number} flags - The meshlet's MESHLET_FLAG_* bits.
 * @returns {number} The MESHLET_BUCKET_* the meshlet draws in.
 * @ignore
 */
const bucketOfMeshletFlags = flags => ((flags & MESHLET_FLAG_ALPHA_MASKED) ? MESHLET_BUCKET_MASKED :
    ((flags & MESHLET_FLAG_TWO_SIDED) ? MESHLET_BUCKET_OPAQUE_TWO_SIDED : MESHLET_BUCKET_OPAQUE));

/**
 * The meshlet flags that select a draw bucket - the inverse of {@link bucketOfMeshletFlags}.
 *
 * @param {number} bucket - A MESHLET_BUCKET_*.
 * @returns {number} Its MESHLET_FLAG_ALPHA_MASKED and MESHLET_FLAG_TWO_SIDED bits.
 * @ignore
 */
const meshletFlagsOfBucket = bucket => (bucket === MESHLET_BUCKET_MASKED ? MESHLET_FLAG_ALPHA_MASKED : 0) |
    (bucket === MESHLET_BUCKET_OPAQUE ? 0 : MESHLET_FLAG_TWO_SIDED);

/**
 * A meshlet world's material table: MATERIAL_RECORD_U32S words per material row, the rows of
 * every resource concatenated in add order. A resource with a baked table contributes its
 * records verbatim (plus a synthesized fallback row when any primitive has no material); one
 * without gets a synthesized row per primitive.
 *
 * @ignore
 */
class MeshletMaterialTable {
    /** @type {Uint32Array} */
    data;

    /** @type {Float32Array} - float view of {@link data}. */
    f32;

    /** @type {number} */
    rowCount;

    /**
     * @param {number} rowCount - Material rows.
     */
    constructor(rowCount) {
        this.rowCount = rowCount;
        this.data = new Uint32Array(Math.max(rowCount, 1) * MATERIAL_RECORD_U32S);
        this.f32 = new Float32Array(this.data.buffer);
    }

    /**
     * Copies a resource's baked records into place.
     *
     * @param {Uint32Array} records - The baked table.
     * @param {number} firstRow - Row of the first record.
     */
    setRecords(records, firstRow) {
        this.data.set(records, firstRow * MATERIAL_RECORD_U32S);
    }

    /**
     * Writes a synthesized record: matte, the given base colour (white without one), no textures.
     *
     * @param {number} row - The row.
     * @param {number[]|null} baseColor - Base colour rgba, or null.
     */
    writeFallback(row, baseColor) {
        const base = row * MATERIAL_RECORD_U32S;
        const f32 = this.f32;
        for (let c = 0; c < 4; c++) f32[base + MATERIAL_RECORD.BASE_COLOR + c] = baseColor?.[c] ?? 1;
        f32[base + MATERIAL_RECORD.EMISSIVE_STRENGTH] = 1;
        f32[base + MATERIAL_RECORD.METALLIC] = 0;
        f32[base + MATERIAL_RECORD.ROUGHNESS] = 1;
        f32[base + MATERIAL_RECORD.ALPHA_CUTOFF] = 0.5;
        for (let slot = 0; slot < MATERIAL_TEXTURE_SLOTS; slot++) {
            this.data[base + MATERIAL_RECORD.SLOT_WORDS + slot] = MATERIAL_SLOT_ABSENT;
        }
    }

    /**
     * Rebases a range of rows' slot texture indices from their resource's own texture space into
     * the world's flat one, like a page index into the page pool.
     *
     * @param {number} firstRow - First row of the resource.
     * @param {number} rowCount - Rows of the resource.
     * @param {number} textureBase - Flat index of the resource's first texture.
     */
    rebaseTextures(firstRow, rowCount, textureBase) {
        const data = this.data;
        for (let row = firstRow; row < firstRow + rowCount; row++) {
            const base = row * MATERIAL_RECORD_U32S + MATERIAL_RECORD.SLOT_WORDS;
            for (let slot = 0; slot < MATERIAL_TEXTURE_SLOTS; slot++) {
                const word = data[base + slot];
                if ((word & TEX_INDEX_MASK) === MATERIAL_SLOT_ABSENT) continue;
                // the index is 16 bits with the top value meaning absent: a world past that drops
                // the slot rather than wrap onto another texture
                const rebased = (word & TEX_INDEX_MASK) + textureBase;
                if (rebased >= MATERIAL_SLOT_ABSENT) {
                    Debug.errorOnce(`MeshletWorld: more than ${MATERIAL_SLOT_ABSENT} streamed textures in one world; textures past the limit are dropped.`);
                }
                data[base + slot] = (word & ~TEX_INDEX_MASK) | Math.min(rebased, MATERIAL_SLOT_ABSENT);
            }
        }
    }

    /**
     * The draw bucket of a row. Alpha masking and two-sidedness are separate flags of the
     * record: folding them into one would put opaque geometry in the alpha-tested bucket and
     * hand it a `discard` that defeats early-Z.
     *
     * @param {number} row - The row.
     * @returns {number} The MESHLET_BUCKET_* of the row's primitives.
     */
    bucket(row) {
        const flags = this.data[row * MATERIAL_RECORD_U32S + MATERIAL_RECORD.FLAGS];
        return (flags & MATERIAL_FLAG_ALPHA_MASK) ? MESHLET_BUCKET_MASKED :
            ((flags & MATERIAL_FLAG_DOUBLE_SIDED) ? MESHLET_BUCKET_OPAQUE_TWO_SIDED : MESHLET_BUCKET_OPAQUE);
    }

    /**
     * Per (row, slot), the flat texture index the slot samples (-1 when absent) and its tiling
     * bias: TEXEL_RATE_PER_MIP * log2 of the slot's KHR_texture_transform scale. The cull pass
     * measures texel rate in pre-transform UV space, but sampling applies the scale to the
     * derivatives, so a tiled texture resolves that much coarser than its mark suggests - the
     * texture streamer subtracts the bias from the mark.
     *
     * @returns {{ rowTex: Int32Array, rowTilingBias: Int16Array }} The maps.
     */
    slotTextureMaps() {
        const data = this.data;
        const rowTex = new Int32Array(this.rowCount * MATERIAL_TEXTURE_SLOTS).fill(-1);
        const rowTilingBias = new Int16Array(this.rowCount * MATERIAL_TEXTURE_SLOTS);
        for (let row = 0; row < this.rowCount; row++) {
            const base = row * MATERIAL_RECORD_U32S;
            for (let s = 0; s < MATERIAL_TEXTURE_SLOTS; s++) {
                const word = data[base + MATERIAL_RECORD.SLOT_WORDS + s];
                if ((word & TEX_INDEX_MASK) === MATERIAL_SLOT_ABSENT) continue;
                rowTex[row * MATERIAL_TEXTURE_SLOTS + s] = word & TEX_INDEX_MASK;
                const offset = data[base + MATERIAL_RECORD.SLOT_TRANSFORMS + s * 2];
                const scale = data[base + MATERIAL_RECORD.SLOT_TRANSFORMS + s * 2 + 1];
                if ((offset | scale) !== 0) {
                    const maxScale = Math.max(Math.abs(FloatPacking.half2Float(scale & 0xFFFF)), Math.abs(FloatPacking.half2Float(scale >>> 16)));
                    if (maxScale > 0 && Number.isFinite(maxScale)) {
                        rowTilingBias[row * MATERIAL_TEXTURE_SLOTS + s] = Math.round(TEXEL_RATE_PER_MIP * Math.log2(maxScale));
                    }
                }
            }
        }
        return { rowTex, rowTilingBias };
    }
}

export { MeshletMaterialTable, bucketOfMeshletFlags, meshletFlagsOfBucket };
