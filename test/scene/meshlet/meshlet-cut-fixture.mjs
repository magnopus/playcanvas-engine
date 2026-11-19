import { MESHLET_DATA as M, MESHLET_DATA_U32S, MESHLET_NO_PARENT } from '../../../src/scene/meshlet/constants.js';

// Cross-linked DAG: both middle replacements depend on BOTH root refinements.
// Coverage bits denote disjoint surface patches, independent of triangle count.
export const coverage = [3, 12, 48, 192, 5, 10, 80, 160, 85, 170];
export const createCutFixture = () => {
    const meshlets = new Uint32Array(10 * MESHLET_DATA_U32S);
    const f = new Float32Array(meshlets.buffer);
    const parents = [4, 4, 6, 6, 8, 9, 8, 9, MESHLET_NO_PARENT, MESHLET_NO_PARENT];
    for (let m = 0; m < 10; m++) {
        const row = m * MESHLET_DATA_U32S;
        const level = m < 4 ? 0 : (m < 8 ? 1 : 2);
        meshlets[row + M.PARENT] = parents[m];
        meshlets[row + M.PAGE] = m;
        meshlets[row + M.TRIANGLE_COUNT] = m < 4 ? 2 : 1;
        meshlets[row + M.LOD_LEVEL] = level;
        f[row + M.CLUSTER_ERROR] = level;
        f[row + M.PARENT_ERROR] = level === 2 ? 1e30 : level + 1;
        f[row + M.GROUP_SPHERE + 3] = 1;
        f[row + M.PARENT_SPHERE + 3] = 1;
        f[row + M.SPHERE + 3] = 1;
    }
    // one instance of the primitive, its meshlets from world-global id 0
    const instances = { count: 1, first: () => 0, meshletCount: () => 10 };
    return { meshlets, instances };
};
