import { expect } from 'chai';

import { Vec3 } from '../../../src/core/math/vec3.js';
import { OBJECT_DATA as O, OBJECT_DATA_U32S } from '../../../src/scene/meshlet/constants.js';
import { MeshletRootSelection } from '../../../src/scene/meshlet/meshlet-root-selection.js';

const fixture = () => {
    const objects = new Uint32Array(4 * OBJECT_DATA_U32S);
    const f = new Float32Array(objects.buffer);
    for (let i = 0; i < 4; i++) {
        const r = i * OBJECT_DATA_U32S;
        f[r] = f[r + 5] = f[r + 10] = f[r + 15] = 1;
        f[r + O.MAX_SCALE] = f[r + O.SPHERE + 3] = 1;
        f[r + 12] = i === 3 ? 100 : (i + 1) * 2;
    }
    const world = {
        instanceCount: 4,
        totalPages: 4,
        poolSlots: 2,
        objectDataCpu: objects,
        objectDataCpuF: f,
        rootSelections: new Set(),
        cut: { instanceRoots: Array.from({ length: 4 }, (_, i) => ({ bucket: 0, cost: [3, 1], pages: [i] })) }
    };
    const planes = new Float32Array(24);
    planes[0] = -1;
    planes[3] = 10;
    return { world, planes };
};

describe('MeshletRootSelection', function () {
    it('admits complete nearby instances within draw and page budgets, excluding off-screen roots', function () {
        const { world, planes } = fixture();
        const selection = new MeshletRootSelection(world);
        selection.update(planes, Vec3.ZERO, [6, 0, 0], 2, false);
        expect(Array.from(selection.wanted)).to.deep.equal([1, 1, 0, 0]);
        expect(Array.from(selection.pages)).to.deep.equal([1, 1, 0, 0]);
        expect(selection.indices).to.deep.equal([6, 0, 0]);
        expect(selection.records).to.equal(2);
        expect(selection.deferred).to.equal(1);
        expect(selection.requestedByBucket).to.deep.equal([9, 0, 0]);
        expect(selection.requestedRecords).to.equal(3);
        expect(selection.recordDemand).to.equal(2);
        selection.destroy();
        expect(world.rootSelections.size).to.equal(0);
    });

    it('requests record growth for otherwise affordable candidates', function () {
        const { world, planes } = fixture();
        const selection = new MeshletRootSelection(world);
        selection.update(planes, Vec3.ZERO, [6, 0, 0], 0, false);
        expect(Array.from(selection.wanted)).to.deep.equal([0, 0, 0, 0]);
        expect(selection.recordDemand).to.equal(1);
        selection.destroy();
    });

    it('counts shared geometry once while keeping per-placement draw costs', function () {
        const { world, planes } = fixture();
        world.poolSlots = 1;
        world.cut.instanceRoots[1].pages = [0];
        const selection = new MeshletRootSelection(world);
        selection.update(planes, Vec3.ZERO, [6, 0, 0], 2, false);
        expect(Array.from(selection.wanted)).to.deep.equal([1, 1, 0, 0]);
        expect(Array.from(selection.pages)).to.deep.equal([1, 0, 0, 0]);
        selection.destroy();
    });

    it('repartitions existing draw storage when visible demand changes material buckets', function () {
        const { world, planes } = fixture();
        world.cut.instanceRoots[0].bucket = 1;
        const selection = new MeshletRootSelection(world);
        selection.update(planes, Vec3.ZERO, [6, 0, 0], 2, false);
        expect(Array.from(selection.wanted)).to.deep.equal([1, 1, 0, 0]);
        expect(selection.capacity).to.deep.equal([3, 3, 0]);
        selection.destroy();
    });

    it('replaces distant demand when the camera approaches new geometry', function () {
        const { world, planes } = fixture();
        const selection = new MeshletRootSelection(world);
        selection.update(planes, Vec3.ZERO, [3, 0, 0], 1, false);
        selection.update(planes, new Vec3(6, 0, 0), [3, 0, 0], 1, false);
        expect(Array.from(selection.wanted)).to.deep.equal([0, 0, 1, 0]);
        expect(Array.from(selection.pages)).to.deep.equal([0, 0, 1, 0]);
        selection.destroy();
    });

    it('skips the instance pass while its inputs repeat', function () {
        const { world, planes } = fixture();
        const selection = new MeshletRootSelection(world);
        let passes = 0;
        const select = selection._select;
        selection._select = function (...args) {
            passes++;
            return select.apply(this, args);
        };
        selection.update(planes, Vec3.ZERO, [6, 0, 0], 2, false);
        selection.update(planes, Vec3.ZERO, [6, 0, 0], 2, false);
        const settledPasses = passes;
        selection.update(planes, Vec3.ZERO, [6, 0, 0], 2, false);
        expect(passes).to.equal(settledPasses);
        expect(Array.from(selection.wanted)).to.deep.equal([1, 1, 0, 0]);

        // a moved instance reruns the pass
        const f = world.objectDataCpuF;
        f[12] = 100;
        world.objectVersion = 1;
        selection.update(planes, Vec3.ZERO, [6, 0, 0], 2, false);
        expect(passes).to.equal(settledPasses + 1);
        expect(Array.from(selection.wanted)).to.deep.equal([0, 1, 1, 0]);
        selection.destroy();
    });

    it('reuses a pass while the view stays within tolerance, admitting what the tolerance could bring into view', function () {
        const { world, planes } = fixture();
        const f = world.objectDataCpuF;
        // just outside the exact frustum (x <= 10 + r), inside the widened one
        f[2 * OBJECT_DATA_U32S + 12] = 11.5;
        // room for every root, so only the frustum decides
        world.poolSlots = 4;
        const selection = new MeshletRootSelection(world);
        let passes = 0;
        const select = selection._select;
        selection._select = function (...args) {
            passes++;
            return select.apply(this, args);
        };
        selection.update(planes, Vec3.ZERO, [12, 0, 0], 4, false);
        selection.update(planes, Vec3.ZERO, [12, 0, 0], 4, false);
        expect(Array.from(selection.wanted)).to.deep.equal([1, 1, 1, 0]);
        const settled = passes;

        // a small move reuses the pass
        selection.update(planes, new Vec3(0.5, 0, 0), [12, 0, 0], 4, false);
        expect(passes).to.equal(settled);

        // beyond the move tolerance, or a changed clip distance, reruns it
        selection.update(planes, new Vec3(6, 0, 0), [12, 0, 0], 4, false);
        expect(passes).to.equal(settled + 1);
        selection.update(planes, new Vec3(6, 0, 0), [12, 0, 0], 4, false);
        const moved = passes;
        const clipped = planes.slice();
        clipped[3] = 0;
        selection.update(clipped, new Vec3(6, 0, 0), [12, 0, 0], 4, false);
        expect(passes).to.equal(moved + 1);
        selection.destroy();
    });

    it('does not rerun when fed back the capacities it partitioned', function () {
        const { world, planes } = fixture();
        const selection = new MeshletRootSelection(world);
        let passes = 0;
        const select = selection._select;
        selection._select = function (...args) {
            passes++;
            return select.apply(this, args);
        };
        // as the culler does: each update's capacities are the previous update's partition
        let capacities = [9, 3, 0];
        // long enough for the first, unsettled result to be rerun and settle
        for (let frame = 0; frame < 10; frame++) {
            selection.update(planes, Vec3.ZERO, capacities, 2, false);
            capacities = selection.capacity.slice();
        }
        const settled = passes;
        for (let frame = 0; frame < 4; frame++) {
            selection.update(planes, Vec3.ZERO, capacities, 2, false);
            capacities = selection.capacity.slice();
        }
        expect(passes).to.equal(settled);
        // an external change (the draw buffers grew) still reruns
        selection.update(planes, Vec3.ZERO, [capacities[0] + 6, capacities[1], capacities[2]], 2, false);
        expect(passes).to.equal(settled + 1);
        selection.destroy();
    });

    it('reruns when another view changes its pages', function () {
        const { world, planes } = fixture();
        world.poolSlots = 1;
        const a = new MeshletRootSelection(world);
        a.update(planes, Vec3.ZERO, [9, 0, 0], 3, false);
        a.update(planes, Vec3.ZERO, [9, 0, 0], 3, false);
        expect(Array.from(a.wanted)).to.deep.equal([1, 0, 0, 0]);
        const b = new MeshletRootSelection(world);
        b.pages.set([0, 0, 0, 1]);
        world.rootPagesVersion++;
        a.update(planes, Vec3.ZERO, [9, 0, 0], 3, false);
        expect(Array.from(a.wanted)).to.deep.equal([0, 0, 0, 0]);
        a.destroy();
        b.destroy();
    });

    it('accounts for the pinned working set of other views', function () {
        const { world, planes } = fixture();
        world.poolSlots = 1;
        const other = { pages: new Uint8Array([0, 0, 1, 0]) };
        world.rootSelections.add(other);
        const selection = new MeshletRootSelection(world);
        selection.update(planes, Vec3.ZERO, [9, 0, 0], 3, false);
        expect(Array.from(selection.wanted)).to.deep.equal([0, 0, 1, 0]);
        selection.destroy();
    });
});
