import { expect } from 'chai';

import { Entity } from '../../../../src/framework/entity.js';
import { createApp } from '../../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../../jsdom.mjs';

describe('MeshletComponent', function () {

    let app;

    beforeEach(function () {
        jsdomSetup();
        app = createApp();
    });

    afterEach(function () {
        app?.destroy();
        app = null;
        jsdomTeardown();
    });

    it('registers its system and stays inert without WebGPU', function () {
        expect(app.systems.meshlet).to.exist;
        expect(app.systems.meshlet.director, 'null device: no director').to.equal(null);
        const e = new Entity();
        e.addComponent('meshlet');
        app.root.addChild(e);
        expect(e.meshlet).to.exist;
        expect(e.meshlet.enabled).to.equal(true);
        expect(e.meshlet.asset).to.equal(null);
        expect(e.meshlet.resource).to.equal(null);
        expect(e.meshlet.baseUrl).to.equal(null);
        app.systems.meshlet._onFrameRender();
        e.destroy();
    });

    it('round-trips its properties and initializes them from data', function () {
        const resource = { instances: [] };
        const e = new Entity();
        e.addComponent('meshlet', { resource, baseUrl: 'http://assets.test/bistro' });
        expect(e.meshlet.resource).to.equal(resource);
        expect(e.meshlet.baseUrl).to.equal('http://assets.test/bistro');
        expect(e.meshlet._effectiveResource, 'no asset: the direct resource').to.equal(resource);
        expect(e.meshlet._effectiveBaseUrl).to.equal('http://assets.test/bistro');

        e.meshlet.resource = null;
        expect(e.meshlet.resource).to.equal(null);
        expect(app.systems.meshlet._dirty, 'changes mark the system dirty').to.equal(true);
        e.destroy();
    });

    it('exposes the pipeline defaults on the system without a director', function () {
        const system = app.systems.meshlet;
        expect(system.dagPixelThreshold).to.equal(1);
        expect(system.occlusion).to.equal(false);
        expect(system.shadows).to.equal(false);
        expect(system.shadowThresholdScale).to.equal(4);
        expect(system.camera).to.equal(null);
        expect(system.cameras).to.deep.equal([]);
        expect(system.texturePoolBytes).to.equal(0);
        expect(system.poolBytes).to.equal(256 * 1024 * 1024);
        system.poolBytes = 128 * 1024 * 1024;
        expect(system.poolBytes).to.equal(128 * 1024 * 1024);
        system.maxDrawIndices = 5000;
        expect(system.maxDrawIndices).to.equal(5000);
        system.occlusion = true;
        expect(system.occlusion, 'no director to hold it').to.equal(false);
    });

    it('clones the component properties onto a cloned entity', function () {
        const resource = { instances: [] };
        const e = new Entity();
        e.addComponent('meshlet', { resource, baseUrl: 'http://assets.test/x' });
        app.root.addChild(e);
        const clone = e.clone();
        expect(clone.meshlet).to.exist;
        expect(clone.meshlet.resource).to.equal(resource);
        expect(clone.meshlet.baseUrl).to.equal('http://assets.test/x');
        e.destroy();
        clone.destroy();
    });

    it('clears its references when removed', function () {
        const e = new Entity();
        e.addComponent('meshlet', { resource: { instances: [] } });
        app.root.addChild(e);
        e.removeComponent('meshlet');
        expect(e.meshlet).to.not.exist;
        e.destroy();
    });
});
