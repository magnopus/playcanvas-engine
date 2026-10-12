import { expect } from 'chai';

import { EventHandler } from '../../src/core/event-handler.js';
import { Light } from '../../src/scene/light.js';

describe('Light shader key', function () {
    let light;

    beforeEach(function () {
        light = new Light(new EventHandler(), false);
    });

    afterEach(function () {
        light.destroy();
    });

    it('distinguishes single-cascade and multi-cascade shaders', function () {
        light.updateKey();
        const singleCascadeKey = light.key;

        for (const cascadeCount of [2, 3, 4]) {
            light.numCascades = cascadeCount;
            expect(light.key).not.to.equal(singleCascadeKey);

            light.numCascades = 1;
            expect(light.key).to.equal(singleCascadeKey);
        }
    });

    it('reuses the multi-cascade shader across cascade counts', function () {
        light.numCascades = 2;
        const multiCascadeKey = light.key;

        for (const cascadeCount of [3, 4, 2]) {
            light.numCascades = cascadeCount;
            expect(light.key).to.equal(multiCascadeKey);
        }
    });
});
