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

        const createContainer = () => {
            const opaque = new StandardMaterial();
            const blended = new StandardMaterial();
            blended.blendType = BLEND_NORMAL;
            const transmissive = new StandardMaterial();
            transmissive.refraction = 1;
            const masked = new StandardMaterial();
            masked.blendType = BLEND_NORMAL;
            masked.alphaTest = 0.5;
            const materials = [opaque, blended, transmissive, masked];

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
            return new GlbContainerResource(data, new Asset('test', 'container', { url: '' }), app.assets, opaque);
        };

        const castShadow = (entity, name) => entity.findByName(name).render.meshInstances[0].castShadows;

        it('casts shadows from every primitive by default', function () {
            const entity = createContainer().instantiateRenderEntity();
            expect(castShadow(entity, 'node0'), 'opaque').to.equal(true);
            expect(castShadow(entity, 'node1'), 'blended').to.equal(true);
            expect(castShadow(entity, 'node2'), 'transmissive').to.equal(true);
            expect(castShadow(entity, 'node3'), 'masked').to.equal(true);
        });

        it('skips blended and transmissive casters with skipBlendedShadowCasters', function () {
            const entity = createContainer().instantiateRenderEntity({ skipBlendedShadowCasters: true });
            expect(castShadow(entity, 'node0'), 'opaque').to.equal(true);
            expect(castShadow(entity, 'node1'), 'blended').to.equal(false);
            expect(castShadow(entity, 'node2'), 'transmissive').to.equal(false);
            expect(castShadow(entity, 'node3'), 'masked').to.equal(true);
        });

    });

});
