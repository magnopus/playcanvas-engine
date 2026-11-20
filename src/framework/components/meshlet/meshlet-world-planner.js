import { Debug } from '../../../core/debug.js';
import { MESHLET_DATA_U32S } from '../../../scene/meshlet/constants.js';

/**
 * @import { GraphicsDevice } from '../../../platform/graphics/graphics-device.js'
 * @import { WebgpuGraphicsDevice } from '../../../platform/graphics/webgpu/webgpu-graphics-device.js'
 * @import { MeshletDirector } from '../../../scene/meshlet/meshlet-director.js'
 * @import { MeshletFetchOptions, MeshletWorld } from '../../../scene/meshlet/meshlet-world.js'
 * @import { MeshletResource } from '../../../scene/meshlet/meshlet-resource.js'
 * @import { MeshletComponent } from './component.js'
 */

/**
 * The components sharing one resource and stream base URL, placed in a world as one placement.
 *
 * @typedef {object} MeshletResourceGroup
 * @property {MeshletResource} resource - The shared resource.
 * @property {string} baseUrl - The stream base URL.
 * @property {MeshletFetchOptions|null} fetchOptions - How the resource's sidecars are fetched.
 * @property {MeshletComponent[]} components - The components, in placement order.
 * @ignore
 */

/**
 * What a world is built with, kept to tell whether a later rebuild can leave it alone.
 *
 * @typedef {object} MeshletPartitionState
 * @property {MeshletResourceGroup[]} groups - The world's resource groups.
 * @property {number} poolBytes - The world's share of the geometry budget.
 * @property {number} texturePoolBytes - The world's share of the texture budget.
 * @property {string} settings - The world settings key (see {@link worldSettingsKey}).
 * @ignore
 */

/**
 * The world budgets and limits the system builds with.
 *
 * @typedef {object} MeshletWorldSettings
 * @property {number} poolBytes - The geometry page pool budget, divided between worlds.
 * @property {number} texturePoolBytes - The texture slot pool budget, divided between worlds.
 * @property {number} maxFineTextureSize - Largest texture size streamed at full resolution.
 * @property {number} maxDrawIndices - Draw index buffer ceiling, 0 for none.
 * @property {number} maxWorldMeshlets - The most meshlets one world holds.
 * @ignore
 */

// draw indices a fresh world's views start with; demand grows them from there
const INITIAL_DRAW_INDICES = 2 * 1024 * 1024;

/**
 * Groups components by (resource, baseUrl): each unique resource is added to a world once, with
 * a merged instance list, so N components share one set of pages, records and textures.
 * Disabled components are included (hidden via the per-instance hide bit) so a later enable is a
 * buffer write, not a rebuild.
 *
 * @param {Iterable<MeshletComponent>} components - The components.
 * @returns {MeshletResourceGroup[]} The groups, in component order.
 * @ignore
 */
const groupComponents = (components) => {
    const groups = [];
    /** @type {Map<MeshletResource, Map<string, MeshletResourceGroup>>} */
    const groupIndex = new Map();
    for (const component of components) {
        const resource = component._effectiveResource;
        if (!resource) continue;
        const baseUrl = component._effectiveBaseUrl;
        if (!baseUrl) {
            Debug.warnOnce('MeshletComponent: no stream base URL - assign an asset with an http(s) file URL (not a blob: or data: URL), or set baseUrl with the resource.');
            continue;
        }
        let byUrl = groupIndex.get(resource);
        if (!byUrl) {
            byUrl = new Map();
            groupIndex.set(resource, byUrl);
        }
        let group = byUrl.get(baseUrl);
        if (!group) {
            group = { resource, baseUrl, fetchOptions: component._effectiveFetchOptions, components: [] };
            byUrl.set(baseUrl, group);
            groups.push(group);
        }
        group.components.push(component);
    }
    return groups;
};

/**
 * The meshlet limit per world: `maxWorldMeshlets`, clamped so a world's meshlet records fit one
 * storage binding.
 *
 * @param {GraphicsDevice} device - The graphics device.
 * @param {number} maxWorldMeshlets - The configured limit.
 * @returns {number} Meshlets per world.
 * @ignore
 */
const worldMeshletLimit = (device, maxWorldMeshlets) => {
    // WebGPU devices expose their adapter limits; others have none to clamp to
    const limits = device.isWebGPU ? /** @type {WebgpuGraphicsDevice} */ (device).limits : null;
    const bytes = Math.min(
        limits?.maxBufferSize ?? Number.MAX_SAFE_INTEGER,
        limits?.maxStorageBufferBindingSize ?? Number.MAX_SAFE_INTEGER
    );
    return Math.max(Math.min(maxWorldMeshlets, Math.floor(bytes / (MESHLET_DATA_U32S * 4))), 1);
};

/**
 * @param {MeshletWorldSettings} settings - The world settings.
 * @returns {string} The world settings a partition is built with, other than its budget shares.
 * @ignore
 */
const worldSettingsKey = settings => `${settings.poolBytes}|${settings.texturePoolBytes}|${settings.maxFineTextureSize}|${settings.maxDrawIndices}`;

/**
 * Whether a world can keep its current build: the same resource groups with the same
 * components in the same order, the same world settings, and a built world. Its budget share is
 * not compared: shares follow the scene total, which every added resource moves, and
 * rebuilding every world (re-fetching released records) for a slightly different page pool
 * would cost far more than the pool difference. A kept world keeps the share it was built
 * with; changing the budget setting itself rebuilds them all.
 *
 * @param {MeshletPartitionState|undefined} previous - The partition's state at its last build.
 * @param {MeshletPartitionState} next - The state it would be built with now.
 * @param {MeshletDirector} director - The partition's director.
 * @returns {boolean} True when the world needs no rebuild.
 * @ignore
 */
const canKeepPartition = (previous, next, director) => {
    if (!previous || !director.world.finalized) return false;
    if (previous.settings !== next.settings) return false;
    if (previous.groups.length !== next.groups.length) return false;
    for (let i = 0; i < next.groups.length; i++) {
        const a = previous.groups[i], b = next.groups[i];
        if (a.resource !== b.resource || a.baseUrl !== b.baseUrl || a.components.length !== b.components.length) return false;
        for (let c = 0; c < b.components.length; c++) {
            if (a.components[c] !== b.components[c]) return false;
        }
    }
    return true;
};

/**
 * Plans the worlds for a set of resource groups. The groups are packed into worlds in order,
 * starting a new world when the next resource would take the current one past the meshlet
 * limit; a resource larger than the limit gets a world of its own. Each world's budget share is
 * its fraction of the scene's meshlets (textured meshlets for the texture pool): the fixed
 * per-meshlet tables dominate a world's cost, and its page count follows its meshlets.
 *
 * @param {MeshletResourceGroup[]} groups - The resource groups, in placement order.
 * @param {MeshletWorldSettings} settings - The world settings.
 * @param {number} limit - Meshlets per world (see {@link worldMeshletLimit}).
 * @param {MeshletPartitionState[]} previous - Each world's state at its last build.
 * @param {MeshletDirector[]} directors - The current worlds' directors.
 * @returns {{ plans: Array<{ groups: MeshletResourceGroup[], state: MeshletPartitionState, keep: boolean }>, totalMeshlets: number }}
 * One plan per world, and the scene's meshlet count.
 * @ignore
 */
const planWorlds = (groups, settings, limit, previous, directors) => {
    const partitions = [{ groups: [], meshlets: 0, texturedMeshlets: 0 }];
    for (const group of groups) {
        const meshlets = group.resource.totalMeshlets;
        let partition = partitions[partitions.length - 1];
        if (partition.groups.length && partition.meshlets + meshlets > limit) {
            partition = { groups: [], meshlets: 0, texturedMeshlets: 0 };
            partitions.push(partition);
        }
        partition.groups.push(group);
        partition.meshlets += meshlets;
        if (group.resource.textureManifest) partition.texturedMeshlets += meshlets;
    }
    let totalMeshlets = 0;
    let totalTextured = 0;
    for (const p of partitions) {
        totalMeshlets += p.meshlets;
        totalTextured += p.texturedMeshlets;
    }

    const split = partitions.length > 1;
    const settingsKey = worldSettingsKey(settings);
    const plans = partitions.map((partition, pi) => {
        const poolBytes = split && settings.poolBytes > 0 ?
            Math.floor(settings.poolBytes * partition.meshlets / Math.max(totalMeshlets, 1)) :
            settings.poolBytes;
        const texturePoolBytes = split ?
            Math.floor(settings.texturePoolBytes * partition.texturedMeshlets / Math.max(totalTextured, 1)) :
            settings.texturePoolBytes;
        const state = {
            groups: partition.groups.map(g => ({ ...g, components: g.components.slice() })),
            poolBytes,
            texturePoolBytes,
            settings: settingsKey
        };
        const director = directors[pi];
        return { groups: partition.groups, state, keep: !!director && canKeepPartition(previous[pi], state, director) };
    });
    return { plans, totalMeshlets };
};

/**
 * Applies the world settings to a world about to be built.
 *
 * @param {MeshletWorld} world - The world.
 * @param {MeshletWorldSettings} settings - The world settings.
 * @param {number} poolBytes - The world's share of the geometry budget.
 * @param {number} texturePoolBytes - The world's share of the texture budget.
 * @ignore
 */
const applyWorldSettings = (world, settings, poolBytes, texturePoolBytes) => {
    world.poolBytes = poolBytes;
    world.texturePoolBytes = texturePoolBytes;
    world.maxFineTextureSize = settings.maxFineTextureSize;
    world.initialIndices = INITIAL_DRAW_INDICES;
    world.maxIndices = settings.maxDrawIndices;
};

/**
 * Adds one world's resource groups, placing their components in it.
 *
 * @param {MeshletWorld} world - The world.
 * @param {MeshletResourceGroup[]} groups - The world's groups.
 * @param {number} partition - The world's index in the director group.
 * @param {MeshletComponent[]} components - Receives the placed components.
 * @ignore
 */
const buildPartition = (world, groups, partition, components) => {
    for (let gi = 0; gi < groups.length; gi++) {
        const { resource, baseUrl, fetchOptions, components: members } = groups[gi];
        const per = resource.instances.count;

        // one placement per group: the resource's instance list repeated once per member (one
        // root transform each), so each member owns a run of the placement's instances. The
        // build bakes each member's current transform and the default state (shown, casting),
        // so the first sync after it only rewrites what actually differs.
        const transforms = [];
        for (let mi = 0; mi < members.length; mi++) {
            const component = members[mi];
            component._partition = partition;
            component._placementIndex = gi;
            component._placedResource = resource;
            component._subBase = mi * per;
            component._subCount = per;
            const wt = component.entity.getWorldTransform();
            component._lastTransform.copy(wt);
            component._transformDirty = false;
            transforms.push(component._lastTransform);
            component._hiddenApplied = false;
            component._castShadowsApplied = true;
            component.onWorldDirty();
            component._queueState();
            components.push(component);
        }
        world.addStreamedResource(resource, transforms, baseUrl, null, fetchOptions);
    }
};

export { applyWorldSettings, buildPartition, canKeepPartition, groupComponents, planWorlds, worldMeshletLimit, worldSettingsKey };
