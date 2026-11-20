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

    it('creates no pipeline and warns nothing until a meshlet component is added', function () {
        const system = app.systems.meshlet;
        const warn = console.warn;
        const warnings = [];
        console.warn = (...args) => warnings.push(args.join(' '));
        try {
            expect(system._initialized, 'nothing at app creation').to.equal(false);
            expect(app.hasEvent('framerender') && app._callbacks.get('framerender')?.some(cb => cb.scope === system)).to.not.equal(true);
            const e = new Entity();
            e.addComponent('meshlet');
            app.root.addChild(e);
            expect(system._initialized).to.equal(true);
            // the pipeline exists only on WebGPU (the suite also runs on a real WebGPU device)
            if (app.graphicsDevice.isWebGPU) {
                expect(system.group).to.not.equal(null);
            } else {
                expect(system.group, 'null device: no pipeline').to.equal(null);
            }
            e.destroy();
        } finally {
            console.warn = warn;
        }
        expect(warnings.filter(w => w.includes('requires WebGPU'))).to.have.lengthOf.at.most(1);
    });

    it('registers its system and stays inert without WebGPU', function () {
        if (app.graphicsDevice.isWebGPU) this.skip();
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
        if (app.graphicsDevice.isWebGPU) this.skip();
        const system = app.systems.meshlet;
        expect(system.dagPixelThreshold).to.equal(2);
        expect(system.bucketThresholdScales).to.deep.equal([4, 1, 1]);
        expect(system.occlusion).to.equal(true);
        expect(system.shadows).to.equal(false);
        expect(system.shadowThresholdScale).to.equal(4);
        expect(system.camera).to.equal(null);
        expect(system.cameras).to.deep.equal([]);
        expect(system.texturePoolBytes).to.equal(96 * 1024 * 1024);
        expect(system.maxWorldMeshlets).to.equal(12 * 1024 * 1024);
        expect(system.directors).to.deep.equal([]);
        expect(system.poolBytes).to.equal(256 * 1024 * 1024);
        system.poolBytes = 128 * 1024 * 1024;
        expect(system.poolBytes).to.equal(128 * 1024 * 1024);
        system.maxDrawIndices = 5000;
        expect(system.maxDrawIndices).to.equal(5000);
        system.occlusion = false;
        expect(system.occlusion, 'no director to hold it').to.equal(true);
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

    it('queues only components whose world transform was dirtied', function () {
        const system = app.systems.meshlet;
        const parent = new Entity();
        const moved = new Entity();
        const still = new Entity();
        moved.addComponent('meshlet');
        still.addComponent('meshlet');
        parent.addChild(moved);
        app.root.addChild(parent);
        app.root.addChild(still);
        app.root.syncHierarchy();
        system._transformQueue.forEach((c) => {
            c._transformQueued = false;
        });
        system._transformQueue.length = 0;

        parent.setLocalPosition(1, 2, 3);
        moved.setLocalPosition(4, 5, 6);
        expect(system._transformQueue, 'an ancestor move queues the descendant once').to.deep.equal([moved.meshlet]);

        moved.removeComponent('meshlet');
        expect(moved._worldDirtyListener, 'removal detaches the listener').to.equal(null);
        parent.destroy();
        still.destroy();
    });

    it('clears its references when removed', function () {
        const e = new Entity();
        e.addComponent('meshlet', { resource: { instances: [] } });
        app.root.addChild(e);
        e.removeComponent('meshlet');
        expect(e.meshlet).to.not.exist;
        e.destroy();
    });

    describe('#suspended', function () {

        it('defaults to false and round-trips', function () {
            const system = app.systems.meshlet;
            expect(system.suspended).to.equal(false);
            system.suspended = true;
            expect(system.suspended).to.equal(true);
            system.suspended = false;
            expect(system.suspended).to.equal(false);
        });

        it('does not rebuild worlds while set', function () {
            const system = app.systems.meshlet;
            let rebuilds = 0;
            const rebuild = system._rebuild;
            system._rebuild = function (...args) {
                rebuilds++;
                return rebuild.apply(this, args);
            };
            system._group = system._group ?? {};
            const group = system._group;
            const wasDirty = system._dirty;
            system.suspended = true;
            system._dirty = true;
            system._onFrameRender();
            system.flushRebuild();
            system._dirty = wasDirty;
            system.suspended = false;
            system._rebuild = rebuild;
            if (group === system._group && !group.directors) system._group = null;
            expect(rebuilds).to.equal(0);
        });
    });

    describe('#withMeshletRecords', function () {

        it('returns what the callback returns and leaves resident records alone', async function () {
            const system = app.systems.meshlet;
            const resident = {
                recordsResident: true,
                primitives: [{}],
                releaseRecords() {
                    throw new Error('released a resource it did not restore');
                }
            };
            system._records.sources.set(resident, { url: 'a.glb', credentials: null });
            expect(await system.withMeshletRecords(() => 42)).to.equal(42);
            expect(await system.withMeshletRecords(() => Promise.resolve('later'))).to.equal('later');
        });

        it('lets the callback\'s error through', async function () {
            let thrown = null;
            try {
                await app.systems.meshlet.withMeshletRecords(() => {
                    throw new Error('read failed');
                });
            } catch (e) {
                thrown = e;
            }
            expect(thrown?.message).to.equal('read failed');
        });
    });
});
