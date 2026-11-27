import { Debug } from '../../core/debug.js';
import {
    claimWordCount, MATERIAL_RECORD_U32S, MESHLET_DATA_U32S, OBJECT_DATA_U32S, RECORD_U32S, ROOT_LIST_U32S
} from './constants.js';
import { DEMAND_SLOT_RESERVE } from './streaming/meshlet-residency.js';

const BYTES_PER_MB = 1024 * 1024;

// Least the draw index budget is resolved to, whatever the geometry budget leaves for it: 4M
// indices. Root admission takes instances nearest first, so any real capacity draws the closest
// ones; a budget that leaves the indices nothing would admit no instance at all and draw an
// empty scene.
const MIN_INDEX_BUDGET_BYTES = 16 * 1024 * 1024;

// Bytes per record a view holds: the record itself and its selected-meshlet entry.
const RECORD_BYTES = RECORD_U32S * 4 + 8;

// Bytes per cut queue entry: an (instance, group row) pair.
const CUT_QUEUE_ENTRY_BYTES = 8;

// Least each view's records and cut queue are resolved to under a budget: enough for a few
// thousand instances' roots and their first refinements.
const MIN_VIEW_RECORDS = 16 * 1024;
const MIN_VIEW_CUT_QUEUE = 16 * 1024;

// Shares of the draw working set given to all views' records and cut queues, the rest going to
// the indices. A drawn meshlet costs hundreds of indices (four bytes each) against one 24-byte
// record and a cut task for its group, so these are small.
const RECORD_BUDGET_SHARE = 0.05;
const CUT_QUEUE_BUDGET_SHARE = 0.03;

// A shadow view's weight against a camera view's in that split. A cascade cuts far coarser
// than the camera it shadows, and an even split hands a scene with a dozen shadow views a
// camera share too small to refine anything near.
const SHADOW_VIEW_DRAW_WEIGHT = 0.25;

/**
 * Root list entries a view holds: one per instance. Admission ranks every instance in view, so
 * this scales with the scene, not the budget.
 *
 * @param {number} instanceCount - The world's instances.
 * @returns {number} Entries.
 * @ignore
 */
const rootListItems = instanceCount => Math.max(instanceCount, 4);

/**
 * Words of a camera view's persistent visibility bits: one bit per instance-meshlet pair and
 * per instance, and a few words at least so a tiny scene still binds a real buffer.
 *
 * @param {number} pairs - The world's instance-meshlet pairs.
 * @param {number} instances - The world's instances.
 * @returns {number} Words.
 * @ignore
 */
const visWordCount = (pairs, instances) => Math.max(Math.ceil((pairs + instances) / 32), 4);

/**
 * Scene totals a geometry budget is resolved against.
 *
 * @typedef {object} MeshletBudgetCounts
 * @property {number} totalPages - Pages across resources.
 * @property {number} totalMeshlets - Meshlets across resources.
 * @property {number} totalInstances - Instances across placements.
 * @property {number} totalPrimitiveRows - Primitive rows of objectData.
 * @property {number} totalMaterialRows - Material table rows.
 * @property {number} pairs - Instance-meshlet pairs.
 * @property {number} lightmapBytes - Bytes of the per-instance lightmap table.
 * @property {number} rootPages - Pages the residency pins for the resources' roots.
 * @ignore
 */

/**
 * A geometry budget resolved into its parts.
 *
 * @typedef {object} MeshletResolvedBudget
 * @property {number} pagePoolBytes - Bytes for the page pool.
 * @property {number} indexBudgetTotal - Draw indices shared by every view.
 * @property {number} recordCeiling - Records each camera view may grow to.
 * @property {number} cutQueueCeiling - Cut queue entries each camera view may grow to.
 * @property {number} shadowRecordCeiling - Records each shadow view may grow to.
 * @property {number} shadowCutQueueCeiling - Cut queue entries each shadow view may grow to.
 * @property {{ budget: number, fixed: number, perView: number, pagePool: number, indices: number, records: number, cutQueue: number, rootPages: number }} breakdown -
 * Where the budget went, in bytes (root pages as a count).
 * @ignore
 */

/**
 * Splits a world's geometry budget into the page pool and the draw working set: the shared
 * index pool and each view's record and cut-queue ceilings. Nothing a view allocates later grows
 * past these, so the budget bounds the pipeline's memory rather than its starting point - what
 * does not fit is not drawn, farthest first (see MeshletWorld#rootBudgetFraction).
 *
 * The world tables and each view's scene-sized state (claim and visibility bits, one per
 * instance-meshlet pair, and the root list, one entry per instance) are exactly computable from
 * the scene's counts and come off the top. A shadow view costs far less than a camera one - it
 * borrows the claim plane and root list from its siblings and binds a placeholder for the
 * visibility bits it never touches - so the two are counted separately.
 *
 * @param {object} settings - The world's budget settings.
 * @param {number} settings.poolBytes - The geometry budget in bytes.
 * @param {number} settings.indexBudgetFraction - Share of what is left after the fixed and
 * per-view costs that goes to the draw working set.
 * @param {number} settings.cameraViews - Camera views to budget for.
 * @param {number} settings.shadowViews - Shadow views to budget for.
 * @param {number} settings.pageSizeBytes - Bytes per page.
 * @param {MeshletBudgetCounts} counts - Scene totals.
 * @returns {MeshletResolvedBudget|null} The resolved budget, or null when unbudgeted (size the
 * page pool to every page).
 * @ignore
 */
function resolveMeshletBudget(settings, counts) {
    const { poolBytes, indexBudgetFraction, pageSizeBytes } = settings;
    if (!(poolBytes > 0)) return null;
    const { totalPages, totalMeshlets, totalInstances, totalPrimitiveRows, totalMaterialRows, pairs, lightmapBytes, rootPages } = counts;

    const pairWords = visWordCount(pairs, totalInstances);
    const claimWords = claimWordCount(pairs, totalInstances);
    // Group headers and membership lists, packed selection links, per-instance root offsets
    // and cell tables.
    const cutBytes = totalMeshlets * (56 + 8) + totalInstances * 44;
    const fixed = cutBytes + lightmapBytes + totalMeshlets * MESHLET_DATA_U32S * 4 +      // meshletData
        (totalInstances + totalPrimitiveRows) * OBJECT_DATA_U32S * 4 +  // objectData
        totalPages * 4 +                                       // residency
        (totalPages + totalMaterialRows) * 4 +                 // requests + texel-rate marks
        totalMaterialRows * MATERIAL_RECORD_U32S * 4;          // material table
    const rootListBytes = rootListItems(totalInstances) * ROOT_LIST_U32S * 4;
    // camera view: own claim plane, own visibility bits, own root list
    const perCamera = (claimWords + pairWords) * 4 + rootListBytes;
    // shadow views: one claim plane and root list shared by every cascade
    const shadows = settings.shadowViews;
    const shared = shadows > 0 ? claimWords * 4 + rootListBytes : 0;

    const views = Math.max(settings.cameraViews, 1);
    const reserved = fixed + perCamera * views + shared;
    let available = poolBytes - reserved;

    // The pool must hold every pinned root page plus the residency's demand reserve, and the
    // draw working set enough for the nearest instances' roots - without either nothing draws. A
    // budget that cannot even hold the fixed cost is not a budget: keep going with those
    // minimums rather than allocating nothing, and say so.
    const viewCount = views + shadows;
    const minRecordBytes = MIN_VIEW_RECORDS * RECORD_BYTES;
    const minQueueBytes = MIN_VIEW_CUT_QUEUE * CUT_QUEUE_ENTRY_BYTES;
    const minDraw = MIN_INDEX_BUDGET_BYTES + viewCount * (minRecordBytes + minQueueBytes);
    const minPool = Math.min(totalPages, rootPages + Math.min(DEMAND_SLOT_RESERVE, Math.max(totalPages - rootPages, 0))) * pageSizeBytes;
    const minimum = minPool + minDraw;
    if (available < minimum) {
        Debug.warnOnce(`MeshletWorld: geometry budget ${(poolBytes / BYTES_PER_MB).toFixed(0)} MB is below the ${((reserved + minimum) / BYTES_PER_MB).toFixed(0)} MB this scene needs for its fixed and per-view buffers, its ${rootPages} root pages and a minimal draw working set; falling back to those minimums, so only the nearest instances draw.`);
        available = minimum;
    }

    let drawBytes = Math.floor(available * indexBudgetFraction);
    let pagePool = available - drawBytes;
    if (pagePool < minPool) {
        // the roots come before the draw working set
        drawBytes -= minPool - pagePool;
        pagePool = minPool;
    }
    if (drawBytes < minDraw) {
        // and the minimal draw working set before the rest of the page pool
        const take = Math.min(minDraw - drawBytes, pagePool - minPool);
        pagePool -= take;
        drawBytes = Math.max(drawBytes + take, minDraw);
    }
    // The pool never allocates more slots than the scene has pages (many instances of a small
    // asset fit whole): hand the rest to the draw working set instead of leaving it unused.
    const allPages = totalPages * pageSizeBytes;
    if (allPages > 0 && pagePool > allPages) {
        drawBytes += pagePool - allPages;
        pagePool = allPages;
    }

    // Each view's records and cut queue take a small share of the draw working set; the
    // indices, by far the largest per drawn meshlet, take the rest as one pool the director
    // shares between the views by demand.
    const weights = views + shadows * SHADOW_VIEW_DRAW_WEIGHT;
    const recordBytes = Math.max(Math.floor(drawBytes * RECORD_BUDGET_SHARE / weights), minRecordBytes);
    const queueBytes = Math.max(Math.floor(drawBytes * CUT_QUEUE_BUDGET_SHARE / weights), minQueueBytes);
    const shadowRecordBytes = Math.max(Math.floor(recordBytes * SHADOW_VIEW_DRAW_WEIGHT), minRecordBytes);
    const shadowQueueBytes = Math.max(Math.floor(queueBytes * SHADOW_VIEW_DRAW_WEIGHT), minQueueBytes);
    const records = recordBytes * views + shadowRecordBytes * shadows;
    const cutQueue = queueBytes * views + shadowQueueBytes * shadows;
    const indexBytes = Math.max(drawBytes - records - cutQueue, MIN_INDEX_BUDGET_BYTES);
    return {
        pagePoolBytes: pagePool,
        indexBudgetTotal: Math.max(Math.floor(indexBytes / 4), 3),
        recordCeiling: Math.floor(recordBytes / RECORD_BYTES),
        cutQueueCeiling: Math.floor(queueBytes / CUT_QUEUE_ENTRY_BYTES),
        shadowRecordCeiling: Math.floor(shadowRecordBytes / RECORD_BYTES),
        shadowCutQueueCeiling: Math.floor(shadowQueueBytes / CUT_QUEUE_ENTRY_BYTES),
        breakdown: {
            budget: poolBytes,
            fixed,
            perView: reserved - fixed,
            pagePool,
            indices: indexBytes,
            records,
            cutQueue,
            rootPages
        }
    };
}

export { resolveMeshletBudget, rootListItems, visWordCount };
