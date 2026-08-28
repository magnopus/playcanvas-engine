import { Debug } from '../../../core/debug.js';
import {
    ADDRESS_REPEAT, BUFFERUSAGE_COPY_DST, FILTER_LINEAR, FILTER_LINEAR_MIPMAP_LINEAR,
    PIXELFORMAT_RGBA8, pixelFormatInfo
} from '../../../platform/graphics/constants.js';
import { StorageBuffer } from '../../../platform/graphics/storage-buffer.js';
import { Texture } from '../../../platform/graphics/texture.js';
import {
    MATERIAL_SLOT_ABSENT, MATERIAL_TEXTURE_SLOTS, MESHLET_TEX_NO_MINLOD, TEXEL_RATE_PER_MIP, TEX_RESIDENCY_U32S
} from '../constants.js';
import { transcodeKtx2 } from './meshlet-ktx2.js';
import { MeshletTextureSource } from './meshlet-texture-source.js';

/**
 * Streamed-texture state of a meshlet world.
 *
 * Two tiers per texture family (srgb / srgba / normal / linear - a family is one transcode
 * format):
 * - The always-resident TAIL: every texture gets one layer holding its manifest tail mips
 *   (<= 256 px). One HTTP range request per source array loads it; it is the terminal
 *   fallback, so no pinning is ever needed.
 * - The demand-streamed FINE pool: one slot-pool array per family, sized to the family's
 *   largest source resolution and holding only the mips ABOVE the tail. The cull pass feeds
 *   back per-material texel rates (texelRateMarks); {@link processMarks} converts them to desired
 *   source mips, acquires LRU slots and fetches exactly the (layer, mip) byte ranges needed.
 *   VRAM is bounded by {@link poolBytes} - under pressure {@link mipBias} rises, pushing more
 *   textures under the tail top, which frees slots (same convergence as the geometry cut).
 *
 * Per-texture GPU state lives in the texResidency buffer, 2 u32 per flat texture index:
 * - word0: fineSlotLayer:16 (0xFFFF = tail-only) | familyIndex:8 | sizeBias:8
 *   (sizeBias = log2(familySlotSize / srcSize) - the level offset of a smaller texture
 *   inside its family arrays)
 * - word1: minLod:16 (finest resident mip in SOURCE-mip space; 0x7FFF = nothing resident
 *   yet - the shader falls back to material factors) | tailLayer:16
 *
 * The virtual mip chain the shader walks is [fine slot levels | tail levels] with the
 * sizeBias applied, so slot and tail sampling are seamless.
 *
 * @ignore
 */

const MESHLET_TEX_FAMILIES = ['srgb', 'srgba', 'normal', 'linear'];
const FAMILY_IS_SRGB = [true, true, false, false];
const TAIL_MAX_SIZE = 256;
const MIN_MIP_SIZE = 4;

const familyOfArray = (array) => {
    const prefix = String(array.name ?? '').split('_')[0];
    const index = MESHLET_TEX_FAMILIES.indexOf(prefix);
    if (index >= 0) return index;
    return array.type === 'normal' ? 2 : (array.type === 'srgb' ? 0 : 3);
};

const log2i = v => Math.round(Math.log2(v));

// processMarks bookkeeping and tunables
const MAX_SOURCE_MIPS = 32;   // fineMask is one bit per source mip; also the (texture, mip) key stride
const CHURN_THRESHOLD = 2;    // evictions per frame above which the fine pool is thrashing
const MIP_BIAS_ATTACK = 0.5;  // mips per frame the pressure bias rises while thrashing
const MIP_BIAS_DECAY = 0.05;  // mips per frame it relaxes once evictions stop - 10x slower, hysteresis
const MIP_BIAS_MAX = 8;       // 8 levels coarser than any top is already under every tail

class MeshletTextures {
    /** @type {Array<{ family: number, srcSize: number, array: object, layer: number, source: MeshletTextureSource, tailLayer: number, tailTop: number, tailStart: number, sizeBias: number, fineMask: number, slot: number }>} */
    textures = [];

    /**
     * Per family: tail + fine-pool state. size/levels describe the tail array; slotSize/
     * slotLevels the fine pool (slotLevels 0 = family has no above-tail mips).
     *
     * @type {Array<object>}
     */
    families = [];

    /** @type {StorageBuffer|null} */
    residencyBuffer = null;

    /** @type {((family: number, tail: Texture, fine: Texture|null) => void)|null} */
    onFamilyTexturesReady = null;

    /**
     * Total texture VRAM budget in bytes - the always-resident tail arrays AND the demand-
     * streamed fine slot pool, not the pool alone. Budgeting only the pool understates what is
     * held: the tails are typically the same size again, so a "96 MB" pool sat inside ~186 MB
     * of texture VRAM. The tails are sized by the asset and cannot be traded away, so they come
     * off the top and the fine pool gets what remains, split across families by demand.
     *
     * @type {number}
     */
    poolBytes = 96 * 1024 * 1024;

    /** @type {number} - bytes held by the resident tail arrays created so far. */
    tailBytes = 0;

    /** @type {number} - pressure output: added to every desired source mip (0 = no pressure). */
    mipBias = 0;

    // stats
    fineFetches = 0;

    fineEvictions = 0;

    slotDenials = 0;

    wantedAboveTail = 0;

    _dirty = false;

    _destroyed = false;

    _frame = 0;

    _lastEvictions = 0;

    /**
     * @param {import('../../../platform/graphics/graphics-device.js').GraphicsDevice} device - The device.
     * @param {import('./meshlet-ktx2.js').MeshletTranscodeFn|null} [transcode] - The KTX2
     * transcoder (see {@link transcodeKtx2}); required before any texture can be uploaded.
     */
    constructor(device, transcode = null) {
        this.device = device;
        this.transcode = transcode;
        this._sources = [];
        this._arrays = []; // { array, source, firstTexIndex }
        this._rowTex = null; // Int32Array rows*4, -1 = slot absent
        this._rowTilingBias = null; // Int16Array rows*4, per-slot texture-transform tiling bias
        this._inFlight = new Set(); // texIndex * MAX_SOURCE_MIPS + sourceMip
    }

    /**
     * Registers a resource's texture manifest. Call during world finalize, in resource order -
     * the returned base is the world-global flat index of the resource's first texture (the
     * material-record slot words rebase by it).
     *
     * @param {object} textureManifest - The parsed MAG_texture_streaming manifest ({ arrays }).
     * @param {string} baseUrl - Directory the manifest's container URIs are relative to.
     * @returns {number} The resource's texture base index.
     */
    addResource(textureManifest, baseUrl) {
        const base = this.textures.length;
        const source = new MeshletTextureSource(baseUrl);
        this._sources.push(source);
        for (const array of textureManifest.arrays ?? []) {
            if (array.containerVersion !== 2) {
                Debug.warnOnce(`MeshletTextures: array '${array.name}' has an unsupported containerVersion (${array.containerVersion}) and is skipped.`);
                continue;
            }
            const family = familyOfArray(array);
            const srcSize = Math.max(array.width ?? 1, array.height ?? 1);
            const tailTop = Math.min(srcSize >> array.tailMip, TAIL_MAX_SIZE);
            this._arrays.push({ array, source, firstTexIndex: this.textures.length });
            for (let layer = 0; layer < array.layers.length; layer++) {
                this.textures.push({
                    family,
                    srcSize,
                    array,
                    layer,
                    source,
                    tailLayer: 0,
                    tailTop,
                    tailStart: log2i(srcSize / tailTop), // source mip where the tail begins
                    sizeBias: 0,                         // level offset in family arrays (finalize)
                    fineMask: 0,                         // bit per source mip resident in the slot
                    slot: -1
                });
            }
        }
        return base;
    }

    /** @type {number} - registered textures across resources. */
    get textureCount() {
        return this.textures.length;
    }

    /**
     * Provides the material-row - texture-slot map (rows x 4 flat texture indices, -1 =
     * absent), used to turn per-material feedback marks into per-texture demand, plus a
     * per-slot tiling bias: 16 * log2 of the slot's KHR_texture_transform scale, SUBTRACTED
     * from that slot's mark. The cull measures texel rate in mesh UV space, but a tiled
     * texture repeats across that span, so the sampler lands on coarser mips than the mark
     * suggests - without the correction the streamer fetches mips nothing samples.
     *
     * @param {Int32Array} rowTex - The map.
     * @param {Int16Array} [rowTilingBias] - Per-slot tiling bias, subtracted from that slot's mark.
     */
    setMaterialSlotMap(rowTex, rowTilingBias = null) {
        this._rowTex = rowTex;
        this._rowTilingBias = rowTilingBias;
    }

    /**
     * Builds the residency buffer and family placeholders, assigns tail layers and size
     * biases, and starts the async tail load. Call once after every resource is added.
     */
    finalize() {
        const device = this.device;

        const familyLayers = [0, 0, 0, 0];
        const familyTailSize = [0, 0, 0, 0];
        const familySlotSize = [0, 0, 0, 0];
        const familyFineTex = [0, 0, 0, 0];
        for (const tex of this.textures) {
            tex.tailLayer = familyLayers[tex.family]++;
            familyTailSize[tex.family] = Math.max(familyTailSize[tex.family], tex.tailTop);
            familySlotSize[tex.family] = Math.max(familySlotSize[tex.family], tex.srcSize);
            if (tex.srcSize > tex.tailTop) familyFineTex[tex.family]++;
        }

        for (let f = 0; f < MESHLET_TEX_FAMILIES.length; f++) {
            const tailSize = Math.max(familyTailSize[f], MIN_MIP_SIZE);
            const slotSize = Math.max(familySlotSize[f], tailSize);
            const placeholder = new Texture(device, {
                name: `MeshletTexPlaceholder-${MESHLET_TEX_FAMILIES[f]}`,
                width: 4,
                height: 4,
                arrayLength: 1,
                format: PIXELFORMAT_RGBA8,
                mipmaps: false,
                levels: [[new Uint8Array(64).fill(255)]]
            });
            this.families.push({
                tail: null,
                fine: null,
                placeholder,
                layerCount: familyLayers[f],
                fineDemandTex: familyFineTex[f],
                size: tailSize,
                levels: log2i(tailSize / MIN_MIP_SIZE) + 1,
                slotSize: slotSize,
                slotLevels: log2i(slotSize / tailSize), // fine levels above the tail top
                slotCount: 0,
                slotTex: null,      // Int32Array slot -> texIndex (-1 free)
                slotLastUsed: null, // Uint32Array
                freeSlots: [],
                pendingFree: []
            });
        }

        for (const tex of this.textures) {
            tex.sizeBias = log2i(this.families[tex.family].slotSize / tex.srcSize);
        }

        // residency: everything starts "nothing resident" - the shader uses factors only
        const residency = new Uint32Array(Math.max(this.textures.length, 1) * TEX_RESIDENCY_U32S);
        for (let i = 0; i < this.textures.length; i++) {
            const tex = this.textures[i];
            residency[i * TEX_RESIDENCY_U32S] = MATERIAL_SLOT_ABSENT | (tex.family << 16) | (tex.sizeBias << 24);
            residency[i * TEX_RESIDENCY_U32S + 1] = MESHLET_TEX_NO_MINLOD | (tex.tailLayer << 16);
        }
        this.residencyCpu = residency;
        this.residencyBuffer = new StorageBuffer(device, residency.byteLength, BUFFERUSAGE_COPY_DST);
        this.residencyBuffer.write(0, residency);

        this._loadTails();
    }

    /**
     * The textures bound for a family right now - real arrays once the format is known,
     * placeholders before that.
     *
     * @param {number} family - Family index.
     * @returns {{ tail: Texture, fine: Texture }} The textures to bind.
     */
    familyTextures(family) {
        const info = this.families[family];
        return { tail: info.tail ?? info.placeholder, fine: info.fine ?? info.placeholder };
    }

    /**
     * Uploads the residency buffer when the CPU mirror changed. Call once per frame.
     */
    flush() {
        if (this._dirty && !this._destroyed) {
            this._dirty = false;
            this.residencyBuffer.write(0, this.residencyCpu);
        }
    }

    /** @type {number} - fine slots in use across families. */
    get slotsUsed() {
        let used = 0;
        for (const fam of this.families) {
            if (fam.slotTex) for (let s = 0; s < fam.slotTex.length; s++) used += fam.slotTex[s] >= 0 ? 1 : 0;
        }
        return used;
    }

    /** @type {number} - total fine slots across families. */
    get slotsTotal() {
        return this.families.reduce((n, fam) => n + fam.slotCount, 0);
    }

    /**
     * Bytes the fine pool may use: the budget less the resident tails. Families are created as
     * their formats land, so a family created later sees the tails allocated before it - the
     * split converges as the asset loads and is re-derived on the next rebuild.
     *
     * @type {number}
     */
    get finePoolBytes() {
        return Math.max(this.poolBytes - this.tailBytes, 0);
    }

    /** @type {number} - total texture VRAM held: resident tails plus occupied fine slots. */
    get bytesUsed() {
        return this.tailBytes + this.poolBytesUsed;
    }

    /** @type {number} - bytes of the fine-slot budget currently held by occupied slots. */
    get poolBytesUsed() {
        let bytes = 0;
        for (const fam of this.families) {
            if (!fam.slotTex) continue;
            for (let s = 0; s < fam.slotTex.length; s++) {
                if (fam.slotTex[s] >= 0) bytes += fam.slotBytes;
            }
        }
        return bytes;
    }

    /**
     * Converts a frame's per-material texel-rate marks from the cull pass into per-texture
     * fine-mip demand: acquire slots, fetch missing (layer, mip) ranges, touch the LRU, release
     * slots whose demand fell back under the tail, and update the pressure bias.
     *
     * A mark is `TEXEL_RATE_PER_MIP * log2(screen pixels / mesh-space UV extent)` - how many
     * levels below a texture's top the sampler will land, in sixteenths. Worked example, a
     * 1024 px texture (log2 = 10) whose tail holds mips 2 and up (tailStart 2): a mark of 144
     * is 144 / 16 = 9 levels down, so the sampler wants mip 10 - 9 = 1, which is above the
     * tail - fetch mip 1 (and nothing finer). A mark of 96 wants mip 4, inside the tail - the
     * texture keeps no fine slot. Under pressure mipBias adds to the wanted mip, so a rising
     * bias pushes textures back under their tails and frees their slots.
     *
     * @param {Uint32Array} texelRateMarks - One mark per material row.
     */
    processMarks(texelRateMarks) {
        if (this._destroyed || !this._rowTex) return;
        this._frame++;

        // per-texture desired texel rate = max across the materials referencing it, with each
        // slot's texture-transform bias applied (tiled textures resolve coarser than the
        // pre-transform mark suggests)
        const desiredTexelRate = this._desiredTexelRate ??= new Uint16Array(this.textures.length);
        desiredTexelRate.fill(0);
        const rows = Math.min(texelRateMarks.length, this._rowTex.length / MATERIAL_TEXTURE_SLOTS);
        for (let row = 0; row < rows; row++) {
            const texelRate = texelRateMarks[row];
            if (!texelRate) continue;
            for (let s = 0; s < MATERIAL_TEXTURE_SLOTS; s++) {
                const texIndex = this._rowTex[row * MATERIAL_TEXTURE_SLOTS + s];
                if (texIndex < 0) continue;
                const slotTexelRate = Math.max(texelRate - (this._rowTilingBias ? this._rowTilingBias[row * MATERIAL_TEXTURE_SLOTS + s] : 0), 0);
                if (slotTexelRate > desiredTexelRate[texIndex]) desiredTexelRate[texIndex] = slotTexelRate;
            }
        }

        // recycle last frame's evictions now - the residency rewrite has been on the GPU for
        // a frame, so nothing samples the recycled slot's old content any more
        for (const fam of this.families) {
            if (fam.pendingFree.length) {
                fam.freeSlots.push(...fam.pendingFree);
                fam.pendingFree.length = 0;
            }
        }

        let wantedAboveTail = 0;
        let denials = 0;
        for (let i = 0; i < this.textures.length; i++) {
            const tex = this.textures[i];
            const texelRate = desiredTexelRate[i];
            const desiredMip = this._desiredMip(tex, texelRate);
            // seen this frame, has mips above its tail at all, and wants one of them
            const wantsFine = texelRate > 0 && tex.tailStart > 0 && desiredMip < tex.tailStart;
            if (!wantsFine) continue;
            wantedAboveTail++;
            const fam = this.families[tex.family];
            if (!fam.fine) {
                denials++; continue;
            } // pool not created yet (or budget 0)

            if (tex.slot < 0) {
                const slot = this._acquireSlot(fam, desiredTexelRate);
                if (slot < 0) {
                    denials++; continue;
                }
                tex.slot = slot;
                fam.slotTex[slot] = i;
                this.residencyCpu[i * TEX_RESIDENCY_U32S] = (slot & 0xFFFF) | (tex.family << 16) | (tex.sizeBias << 24);
                this._dirty = true;
            }
            fam.slotLastUsed[tex.slot] = this._frame;

            // fetch missing fine mips, coarse to fine
            for (let m = tex.tailStart - 1; m >= desiredMip; m--) {
                const bit = 1 << m;
                const key = i * MAX_SOURCE_MIPS + m;
                if ((tex.fineMask & bit) || this._inFlight.has(key)) continue;
                this._inFlight.add(key);
                this._fetchFineMip(i, m);
            }
        }
        this.wantedAboveTail = wantedAboveTail;
        this.slotDenials = denials;

        // pressure: eviction CHURN means the hot set exceeds the pool and slots ping-pong -
        // bias mips coarser (fast) until the churn stops, then decay slowly (hysteresis).
        // Steady denials at full occupancy without churn are the expected saturated state:
        // the held slots keep full sharpness and the rest stay on their tails.
        const churn = this.fineEvictions - this._lastEvictions;
        this._lastEvictions = this.fineEvictions;
        if (churn > CHURN_THRESHOLD) {
            this.mipBias = Math.min(this.mipBias + MIP_BIAS_ATTACK, MIP_BIAS_MAX);
        } else if (churn === 0) {
            this.mipBias = Math.max(this.mipBias - MIP_BIAS_DECAY, 0);
        }
    }

    /**
     * The source mip a texture's sampler lands on for a texel rate, under the current pressure
     * bias: `log2(size) - texelRate / TEXEL_RATE_PER_MIP + mipBias`, floored and clamped at 0
     * (a mark asking for more texels than the texture has still means its top mip).
     *
     * @param {object} tex - The texture entry.
     * @param {number} texelRate - Its desired texel rate this frame.
     * @returns {number} The wanted source mip.
     * @private
     */
    _desiredMip(tex, texelRate) {
        return Math.max(Math.floor(log2i(tex.srcSize) - texelRate / TEXEL_RATE_PER_MIP + this.mipBias), 0);
    }

    /**
     * @param {object} fam - The family.
     * @param {Uint16Array} desiredTexelRate - This frame's per-texture demand (protects hot slots).
     * @returns {number} A free or LRU-evicted slot, or -1.
     * @private
     */
    _acquireSlot(fam, desiredTexelRate) {
        if (fam.freeSlots.length) return fam.freeSlots.pop();
        let best = -1;
        let bestUsed = 0xFFFFFFFF;
        for (let s = 0; s < fam.slotCount; s++) {
            const holder = fam.slotTex[s];
            if (holder >= 0 && desiredTexelRate[holder] > 0) continue; // wanted this frame
            if (fam.slotLastUsed[s] < bestUsed) {
                bestUsed = fam.slotLastUsed[s];
                best = s;
            }
        }
        if (best < 0) return -1;
        const victim = fam.slotTex[best];
        if (victim >= 0) {
            const tex = this.textures[victim];
            tex.slot = -1;
            tex.fineMask = 0;
            this.residencyCpu[victim * TEX_RESIDENCY_U32S] = MATERIAL_SLOT_ABSENT | (tex.family << 16) | (tex.sizeBias << 24);
            const tailMin = this.residencyCpu[victim * TEX_RESIDENCY_U32S + 1] & 0xFFFF;
            if (tailMin !== MESHLET_TEX_NO_MINLOD) {
                this.residencyCpu[victim * TEX_RESIDENCY_U32S + 1] = (tex.tailStart & 0xFFFF) | (tex.tailLayer << 16);
            }
            this._dirty = true;
            this.fineEvictions++;
            fam.slotTex[best] = -1;
            // The slot is reused only NEXT frame, after the residency rewrite reached the GPU:
            // it goes through pendingFree, and this frame's requester is denied. Handing it out
            // now would leave it queued in pendingFree as well, and the next frame would give
            // the same slot to a second texture.
            fam.pendingFree.push(best);
            return -1;
        }
        return best;
    }

    /**
     * Fetches, transcodes and uploads one fine (texture, source mip) into the texture's fine
     * slot, then refines minLod when the resident chain grew contiguously finer.
     *
     * @param {number} texIndex - Flat texture index.
     * @param {number} sourceMip - Source-space mip to load.
     * @private
     */
    _fetchFineMip(texIndex, sourceMip) {
        const tex = this.textures[texIndex];
        // source-mip m maps to manifest mip (m + array.tailMip - tailStart); with the packer's
        // absolute tail cap the two spaces coincide and this is just m
        const manifestMip = sourceMip + (tex.array.tailMip - tex.tailStart);
        const r = tex.array.mipRanges?.[tex.layer]?.[manifestMip];
        const key = texIndex * MAX_SOURCE_MIPS + sourceMip;
        if (!r) {
            this._inFlight.delete(key);
            return;
        }
        tex.source.fetchRegion(tex.array.container.uri, r[0], r[1]).then((bytes) => {
            if (this._destroyed) return null;
            return transcodeKtx2(this.transcode, this.device, bytes.slice(0));
        }).then((result) => {
            this._inFlight.delete(key);
            if (!result || this._destroyed) return;
            const fam = this.families[tex.family];
            // the slot may have been evicted while the fetch was in flight
            if (tex.slot < 0 || fam.slotTex[tex.slot] !== texIndex || !fam.fine) return;
            const level = sourceMip + tex.sizeBias;
            fam.fine.impl.uploadTypedArrayData(this.device, new Uint8Array(result.levels[0]), level, tex.slot);
            tex.fineMask |= 1 << sourceMip;
            this.fineFetches++;
            // minLod refines only contiguously from the tail downwards - a gap would make the
            // shader sample an unwritten level
            let minLod = tex.tailStart;
            while (minLod > 0 && (tex.fineMask & (1 << (minLod - 1)))) minLod--;
            const cur = this.residencyCpu[texIndex * TEX_RESIDENCY_U32S + 1] & 0xFFFF;
            if (cur !== MESHLET_TEX_NO_MINLOD && minLod < cur) {
                this.residencyCpu[texIndex * TEX_RESIDENCY_U32S + 1] = (minLod & 0xFFFF) | (tex.tailLayer << 16);
                this._dirty = true;
            }
        }).catch((err) => {
            this._inFlight.delete(key);
            Debug.warnOnce(`MeshletTextures: fine mip fetch failed (${tex.array.name} layer ${tex.layer} mip ${sourceMip}): ${err.message}`);
        });
    }

    /**
     * @param {number} family - Family index.
     * @param {number} mipSize - A mip's pixel size.
     * @returns {number} The tail-array level holding that size.
     * @private
     */
    _tailLevel(family, mipSize) {
        return log2i(this.families[family].size) - log2i(mipSize);
    }

    /**
     * Creates a family's tail array and fine slot pool once its transcode format is known.
     * The fine pool's slot count comes from the byte budget, split across families by their
     * fine-texture demand.
     *
     * @param {number} family - Family index.
     * @param {number} format - Engine pixel format (linear id) from the first transcode.
     * @private
     */
    _createFamilyTextures(family, format) {
        const info = this.families[family];
        if (info.tail || this._destroyed) return;
        const device = this.device;
        const name = MESHLET_TEX_FAMILIES[family];

        info.tail = new Texture(device, {
            name: `MeshletTail-${name}`,
            width: info.size,
            height: info.size,
            arrayLength: Math.max(info.layerCount, 1),
            format: format,
            srgb: FAMILY_IS_SRGB[family],
            mipmaps: true,
            addressU: ADDRESS_REPEAT,
            addressV: ADDRESS_REPEAT,
            minFilter: FILTER_LINEAR_MIPMAP_LINEAR,
            magFilter: FILTER_LINEAR
        });

        this.tailBytes += info.tail.gpuSize;

        if (info.slotLevels > 0 && info.fineDemandTex > 0 && this.finePoolBytes > 0) {
            // bytes of one slot = its above-tail mip chain in the transcoded block format
            const blockSize = pixelFormatInfo.get(format)?.blockSize ?? 16;
            let slotBytes = 0;
            for (let l = 0; l < info.slotLevels; l++) {
                const size = info.slotSize >> l;
                slotBytes += Math.ceil(size / 4) * Math.ceil(size / 4) * blockSize;
            }
            const totalDemand = this.families.reduce((n, f) => n + f.fineDemandTex, 0);
            const share = this.finePoolBytes * (info.fineDemandTex / Math.max(totalDemand, 1));
            info.slotCount = Math.max(Math.min(Math.floor(share / slotBytes), info.fineDemandTex), 1);
            info.slotBytes = slotBytes;
            info.slotTex = new Int32Array(info.slotCount).fill(-1);
            info.slotLastUsed = new Uint32Array(info.slotCount);
            for (let s = info.slotCount - 1; s >= 0; s--) info.freeSlots.push(s);

            info.fine = new Texture(device, {
                name: `MeshletFine-${name}`,
                width: info.slotSize,
                height: info.slotSize,
                arrayLength: info.slotCount,
                format: format,
                srgb: FAMILY_IS_SRGB[family],
                mipmaps: true,
                addressU: ADDRESS_REPEAT,
                addressV: ADDRESS_REPEAT,
                minFilter: FILTER_LINEAR_MIPMAP_LINEAR,
                magFilter: FILTER_LINEAR
            });
        }

        this.onFamilyTexturesReady?.(family, info.tail, info.fine);
    }

    /**
     * Fetches every array's tail region, transcodes each (layer, mip) and uploads it into the
     * family tail array; a texture's residency entry goes live when all its mips landed.
     *
     * @private
     */
    _loadTails() {
        for (const { array, source, firstTexIndex } of this._arrays) {
            source.fetchTail(array).then((tailBuffer) => {
                if (this._destroyed) return;
                const mipCount = array.layers[0].length;
                for (let layer = 0; layer < array.layers.length; layer++) {
                    const texIndex = firstTexIndex + layer;
                    const tex = this.textures[texIndex];
                    const jobs = [];
                    for (let mip = array.tailMip; mip < mipCount; mip++) {
                        const mipSize = Math.max(tex.srcSize >> mip, 1);
                        if (mipSize > TAIL_MAX_SIZE || mipSize < MIN_MIP_SIZE) continue;
                        const bytes = source.sliceMip(tailBuffer, array.tail.byteOffset, array, layer, mip);
                        if (!bytes) continue;
                        jobs.push(transcodeKtx2(this.transcode, this.device, bytes).then((result) => {
                            if (this._destroyed) return;
                            this._createFamilyTextures(tex.family, result.format);
                            const level = this._tailLevel(tex.family, mipSize);
                            this.families[tex.family].tail.impl.uploadTypedArrayData(this.device, new Uint8Array(result.levels[0]), level, tex.tailLayer);
                        }));
                    }
                    Promise.all(jobs).then(() => {
                        if (this._destroyed || jobs.length === 0) return;
                        // all tail mips resident: minLod = the tail's finest source mip
                        this.residencyCpu[texIndex * TEX_RESIDENCY_U32S + 1] = (tex.tailStart & 0xFFFF) | (tex.tailLayer << 16);
                        this._dirty = true;
                    }).catch((err) => {
                        Debug.warnOnce(`MeshletTextures: tail load failed for '${array.name}' layer ${layer}: ${err.message}`);
                    });
                }
            }).catch((err) => {
                Debug.warnOnce(`MeshletTextures: tail fetch failed for '${array.name}': ${err.message}`);
            });
        }
    }

    destroy() {
        this._destroyed = true;
        this._sources.forEach(s => s.destroy());
        this._sources = [];
        this.families.forEach((f) => {
            f.tail?.destroy();
            f.fine?.destroy();
            f.placeholder.destroy();
        });
        this.families = [];
        this.residencyBuffer?.destroy();
        this.residencyBuffer = null;
    }
}

export { MeshletTextures, MESHLET_TEX_FAMILIES };
