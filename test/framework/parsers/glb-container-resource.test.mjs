import { expect } from 'chai';

import { Asset } from '../../../src/framework/asset/asset.js';
import { GlbContainerResource } from '../../../src/framework/parsers/glb-container-resource.js';
import { BLEND_NORMAL } from '../../../src/scene/constants.js';
import { GraphNode } from '../../../src/scene/graph-node.js';
import { StandardMaterial } from '../../../src/scene/materials/standard-material.js';
import { Mesh } from '../../../src/scene/mesh.js';
import { Render } from '../../../src/scene/render.js';
import { createApp } from '../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

describe('GlbContainerResource', function () {

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

    describe('#instantiateRenderEntity', function () {

        it('does not cast shadows from blended or transmissive primitives', function () {
            const opaque = new StandardMaterial();
            const blended = new StandardMaterial();
            blended.blendType = BLEND_NORMAL;
            const transmissive = new StandardMaterial();
            transmissive.refraction = 1;
            const materials = [opaque, blended, transmissive];

            // one node and one single-primitive mesh per material
            const root = new GraphNode('root');
            const nodes = [];
            const renders = [];
            const meshDefaultMaterials = {};
            for (let i = 0; i < materials.length; i++) {
                const mesh = new Mesh(app.graphicsDevice);
                meshDefaultMaterials[mesh.id] = i;
                const render = new Render();
                render.meshes = [mesh];
                renders.push(render);
                const node = new GraphNode(`node${i}`);
                root.addChild(node);
                nodes.push(node);
            }

            const data = {
                gltf: { nodes: materials.map((_, i) => ({ mesh: i })) },
                nodes,
                scenes: [root],
                renders,
                materials,
                meshDefaultMaterials,
                animations: [],
                textures: [],
                skins: [],
                nodeInstancingMap: new Map()
            };
            const container = new GlbContainerResource(data, new Asset('test', 'container', { url: '' }), app.assets, opaque);
            const entity = container.instantiateRenderEntity();

            const castShadow = name => entity.findByName(name).render.meshInstances[0].castShadow;
            expect(castShadow('node0'), 'opaque').to.equal(true);
            expect(castShadow('node1'), 'blended').to.equal(false);
            expect(castShadow('node2'), 'transmissive').to.equal(false);
        });

    });

});
