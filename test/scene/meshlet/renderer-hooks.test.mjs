import { expect } from 'chai';

import { createApp } from '../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

describe('ForwardRenderer meshlet hooks', function () {

    let app;

    beforeEach(function () {
        jsdomSetup();
        app = createApp();
    });

    afterEach(function () {
        app.destroy();
        app = null;
        jsdomTeardown();
    });

    it('has no director until one is bound', function () {
        expect(app.renderer.meshletDirector).to.equal(null);
    });

    it('strips the camera clears from a main render pass only when asked', function () {
        const renderer = app.renderer;
        const camera = {};
        const layer = { id: 1 };
        const renderAction = (transparent = false) => ({
            camera,
            layer,
            transparent,
            renderTarget: null,
            clearColor: true,
            clearDepth: true,
            clearStencil: true,
            firstCameraUse: true,
            lastCameraUse: true
        });
        const layerComposition = { _renderActions: [renderAction(), renderAction(true)] };
        const frameGraph = {
            passes: [],
            addRenderPass(pass) {
                this.passes.push(pass);
            }
        };

        renderer.addMainRenderPass(frameGraph, layerComposition, null, 0, 1);
        const kept = frameGraph.passes[0].layerRenderSteps;
        expect(kept).to.have.lengthOf(2);
        expect(kept.map(s => [s.clearColor, s.clearDepth, s.clearStencil])).to.deep.equal([
            [true, true, true], [true, true, true]
        ]);

        // another pass (meshlet draw phase 1) owns the camera clear; these load
        renderer.addMainRenderPass(frameGraph, layerComposition, null, 0, 1, true);
        const stripped = frameGraph.passes[1].layerRenderSteps;
        expect(stripped.map(s => [s.clearColor, s.clearDepth, s.clearStencil])).to.deep.equal([
            [false, false, false], [false, false, false]
        ]);
        expect(stripped.map(s => s.transparent), 'only the clears change').to.deep.equal([false, true]);
        expect(stripped.map(s => s.layer)).to.deep.equal([layer, layer]);
        frameGraph.passes.forEach(pass => pass.destroy());
    });
});
