import { expect } from 'chai';

import { MeshletBudgetManager } from '../../../src/scene/meshlet/meshlet-budget-manager.js';

// A fake streaming residency: the manager only reads counters and the in-flight set.
const makeResidency = () => ({ droppedNoSlot: 0, evictedPages: 0, fetchedPages: 0, lastMissingWanted: 0, inFlight: new Set() });

describe('MeshletBudgetManager', function () {

    it('coarsens fast while index demand is starved, whether or not the world streams', function () {
        const manager = new MeshletBudgetManager();
        manager.update(null, 0.9);
        expect(manager.pressureScale).to.be.closeTo(manager.raiseFast, 1e-9);
        manager.update(makeResidency(), 0.9);
        expect(manager.pressureScale).to.be.closeTo(manager.raiseFast ** 2, 1e-9);
    });

    it('holds in the band between eased and starved, and resets a resident world once eased', function () {
        const manager = new MeshletBudgetManager();
        manager.pressureScale = 4;
        manager.update(null, 0.7);
        expect(manager.pressureScale, 'held').to.equal(4);
        manager.update(null, 0.5);
        expect(manager.pressureScale, 'resident worlds have no churn to wait for').to.equal(1);
    });

    it('coarsens fast on a hard denial and gently on eviction churn', function () {
        const manager = new MeshletBudgetManager();
        const residency = makeResidency();
        manager.update(residency);
        residency.droppedNoSlot = 1;
        manager.update(residency);
        expect(manager.pressureScale).to.be.closeTo(manager.raiseFast, 1e-9);
        residency.evictedPages += manager.churnTolerance + 1;
        manager.update(residency);
        expect(manager.pressureScale).to.be.closeTo(manager.raiseFast * manager.raiseSlow, 1e-9);
        residency.evictedPages += manager.churnTolerance;   // tolerated turnover: hold
        manager.update(residency);
        expect(manager.pressureScale).to.be.closeTo(manager.raiseFast * manager.raiseSlow, 1e-9);
    });

    it('decays toward 1 only while nothing is evicted, denied or index-tight', function () {
        const manager = new MeshletBudgetManager();
        const residency = makeResidency();
        manager.pressureScale = 2;
        manager.update(residency, 0);
        expect(manager.pressureScale).to.be.closeTo(2 * manager.decay, 1e-9);
        manager.update(residency, 0.7);
        expect(manager.pressureScale, 'index-tight holds').to.be.closeTo(2 * manager.decay, 1e-9);
        manager.pressureScale = 1.001;
        manager.update(residency, 0);
        expect(manager.pressureScale, 'floored at 1').to.equal(1);
    });

    it('treats a pool as wedged only after the signature persists for wedgeFrames', function () {
        const manager = new MeshletBudgetManager();
        const residency = makeResidency();
        residency.lastMissingWanted = 5;   // outstanding demand, nothing moving, nothing in flight
        manager.update(residency);
        for (let i = 2; i < manager.wedgeFrames; i++) manager.update(residency);   // wedgeFrames - 1 ticks in total
        expect(manager.pressureScale, 'transient readback gaps look wedged too').to.equal(1);
        manager.update(residency);
        expect(manager.pressureScale).to.be.closeTo(manager.raiseSlow, 1e-9);
        // any movement resets the streak
        residency.fetchedPages = 1;
        manager.update(residency);
        expect(manager._wedgeFrames).to.equal(0);
    });

    it('never exceeds maxScale', function () {
        const manager = new MeshletBudgetManager();
        for (let i = 0; i < 500; i++) manager.update(null, 1);
        expect(manager.pressureScale).to.equal(manager.maxScale);
    });
});
