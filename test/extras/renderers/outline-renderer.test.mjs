import { expect } from 'chai';

import { OutlineRenderer } from '../../../src/extras/renderers/outline-renderer.js';
import { BlendState } from '../../../src/platform/graphics/blend-state.js';
import { BLENDEQUATION_ADD, BLENDMODE_ONE_MINUS_SRC_ALPHA, BLENDMODE_SRC_ALPHA } from '../../../src/platform/graphics/constants.js';

describe('OutlineRenderer', function () {
    it('blends only into scene colour and adapts when the target changes', function () {
        const renderer = Object.create(OutlineRenderer.prototype);
        const states = [];
        let draws = 0;
        const device = {
            renderTarget: { colorBufferCount: 2 },
            scope: { resolve: () => ({ setValue: () => {} }) },
            setDrawStates: state => states.push(state)
        };
        renderer.app = { graphicsDevice: device };
        renderer.rt = { colorBuffer: {} };
        renderer.blendState = new BlendState(true, BLENDEQUATION_ADD, BLENDMODE_SRC_ALPHA, BLENDMODE_ONE_MINUS_SRC_ALPHA);
        renderer.quadRenderer = { render: () => draws++ };

        renderer.blendOutlines();
        device.renderTarget = null;
        renderer.blendOutlines();
        device.renderTarget = { colorBufferCount: 3 };
        renderer.blendOutlines();

        const attachment = new BlendState();
        states[0].getAttachment(0, attachment);
        expect(attachment.key).to.equal(renderer.blendState.key);
        states[0].getAttachment(1, attachment);
        expect(attachment.allWrite).to.equal(0);
        expect(states[1]).to.equal(renderer.blendState);
        states[2].getAttachment(2, attachment);
        expect(attachment.allWrite).to.equal(0);
        expect(renderer.blendState.hasAttachmentOverrides).to.be.false;
        expect(draws).to.equal(3);
    });
});
