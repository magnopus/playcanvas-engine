import { expect } from 'chai';

import { TranslateGizmo } from '../../../src/extras/gizmo/translate-gizmo.js';
import { Application } from '../../../src/framework/application.js';
import { Entity } from '../../../src/framework/entity.js';
import { getApplication } from '../../../src/framework/globals.js';
import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

describe('TranslateGizmo', function () {
    let gizmoApp;
    let otherApp;

    const createApp = (id) => {
        const canvas = document.createElement('canvas');
        canvas.id = id;
        return new Application(canvas, { graphicsDevice: new NullGraphicsDevice(canvas) });
    };

    beforeEach(function () {
        jsdomSetup();
        gizmoApp = createApp('gizmo-app');
        // A second application on the page (e.g. a model viewer) becomes the current application.
        otherApp = createApp('other-app');
    });

    afterEach(function () {
        otherApp?.destroy();
        gizmoApp?.destroy();
        jsdomTeardown();
    });

    it('creates all its entities in its camera\'s application, not the current one', function () {
        expect(getApplication() === otherApp).to.equal(true);

        const camera = new Entity('camera', gizmoApp);
        camera.addComponent('camera');
        gizmoApp.root.addChild(camera);

        const gizmo = new TranslateGizmo(camera.camera, TranslateGizmo.createLayer(gizmoApp));

        const entities = [];
        gizmo.root.forEach(node => entities.push(node));
        expect(entities.length).to.be.greaterThan(1);
        const inOtherApp = entities.filter(entity => entity._app !== gizmoApp).map(entity => entity.name);
        expect(inOtherApp).to.deep.equal([]);

        // Destroying the other application leaves the gizmo usable.
        otherApp.destroy();
        otherApp = null;
        expect(() => gizmo.destroy()).not.to.throw();
    });
});
