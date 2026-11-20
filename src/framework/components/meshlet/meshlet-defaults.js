import { MESHLET_DIRECTOR_SETTINGS } from '../../../scene/meshlet/meshlet-director.js';
import { MAX_CONCURRENT_FETCHES } from '../../../scene/meshlet/streaming/meshlet-fetch-scheduler.js';

const D = MESHLET_DIRECTOR_SETTINGS;

/**
 * Default values of the {@link MeshletComponentSystem} settings. The pipeline settings come from
 * the director's own defaults, which the group applies once the pipeline exists; the system
 * reports these until then, and on devices without WebGPU. Not to be modified.
 *
 * @ignore
 */
const MESHLET_DEFAULTS = {
    // world budgets, owned by the system
    poolBytes: 256 * 1024 * 1024,
    texturePoolBytes: 96 * 1024 * 1024,
    maxFineTextureSize: 2048,
    maxDrawIndices: 0,
    maxWorldMeshlets: 12 * 1024 * 1024,

    // pipeline settings, owned by the director group
    maxConcurrentFetches: MAX_CONCURRENT_FETCHES,
    dagPixelThreshold: D.dagPixelThreshold,
    rootHysteresis: D.rootHysteresis,
    bucketThresholdScales: D.bucketThresholdScales,
    bucketRootMinScales: D.bucketRootMinScales,
    occlusion: D.occlusionEnabled,
    occlusionAdmission: D.occlusionAdmission,
    shadows: D.shadowsEnabled,
    shadowThresholdScale: D.shadowThresholdScale,
    rootMinPixels: D.rootMinPixels,
    cssPixels: D.cssPixels,
    shadowRootMinPixels: D.shadowRootMinPixels,
    fastLoad: D.fastLoad
};

export { MESHLET_DEFAULTS };
