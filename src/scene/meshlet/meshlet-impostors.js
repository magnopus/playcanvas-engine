import {
    BUFFER_STATIC, CULLFACE_NONE, INDEXFORMAT_UINT16, PRIMITIVE_TRIANGLES
} from '../../platform/graphics/constants.js';
import { IndexBuffer } from '../../platform/graphics/index-buffer.js';
import { Mesh } from '../mesh.js';
import { StandardMaterial } from '../materials/standard-material.js';
import { OBJECT_IMPOSTOR_LAYER_SHIFT } from './constants.js';
import { meshletObjectDataWGSL, meshletStructsWGSL } from './shaders/meshlet-page-wgsl.js';

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { Texture } from '../../platform/graphics/texture.js'
 */

/**
 * The octahedral impostors of a meshlet scene: pre-baked views of whole prototypes (trees) that
 * draw in place of their meshlets once the prototype is small on screen. Distant foliage cannot
 * live in the meshlet LOD - its cut erases the leaf cards (each is a tiny geometric error however
 * much of the canopy it makes up) - so beyond the switch size a tree is one camera-facing quad
 * sampling the atlas.
 *
 * Prototypes are matched to the glTF meshes of the loaded assets by mesh name. All primitives of
 * a mesh form one tree, and share one transform: the first of them emits the billboard and the
 * rest are dropped.
 *
 * The atlas holds one layer per prototype: a `grid` x `grid` set of orthographic views of the
 * prototype, one hemi-octahedral direction each (y up; frame (i, j) looks from
 * `hemiOctDecode(i / (grid - 1), j / (grid - 1))`, with up along +Y), each framed on the box
 * `half` and packed in a tile of layer width / grid by layer height / grid. The nearest view is
 * drawn as it is - there is no blending between views.
 *
 * @typedef {object} MeshletImpostors
 * @property {Texture} albedo - 2D texture array, one layer per prototype: sRGB colour, alpha is
 * coverage. Mipmapped (the shader uses the first few).
 * @property {Texture} normal - 2D texture array, the same layout: object space normal * 0.5 + 0.5.
 * @property {number} grid - Views per side of a layer.
 * @property {Object<string, { layer: number, center: number[], radius: number, half?: number[] }>} prototypes -
 * Per glTF mesh name: atlas layer, the bounding sphere (instance space) that decides when the tree
 * becomes a billboard, and `half`, the half width and half height (instance space) of the box
 * every view framed (defaults to the sphere's radius).
 * @ignore
 */

/**
 * The impostor draw's shader chunks: a StandardMaterial whose vertex stage builds a camera-facing
 * quad around one billboard of the impostor list, and whose fragment frontend reads the surface
 * from the nearest baked view. The engine's own lit backend lights it.
 *
 * Per fragment, the view ray is intersected with that view's plane (rather than mapping the
 * quad's own uv into it), which keeps the content steady while the camera moves inside the
 * view's range.
 *
 * @param {object} options - Atlas layout.
 * @param {number} options.grid - Views per side.
 * @param {number} options.tilePixels - Pixels per view across.
 * @returns {object} The chunk override map for StandardMaterial.getShaderChunks('wgsl').add().
 * @ignore
 */
function buildMeshletImpostorChunks({ grid, tilePixels }) {
    const GRID = grid.toFixed(1);
    const TILE = tilePixels.toFixed(1);

    const transformCoreVS = /* wgsl */ `
        var<private> vertex_position: vec4f;
        var<private> vertex_normal: vec3f;

        uniform matrix_viewProjection: mat4x4f;
        #ifndef VIEWMATRIX
        #define VIEWMATRIX
            uniform matrix_view: mat4x4f;
        #endif

        ${meshletObjectDataWGSL}
        var<storage, read> impostorList : array<u32>;
        var<storage, read> impostorProtos : array<vec4f>;

        fn getModelMatrix() -> mat4x4f {
            return mat4x4f(vec4f(1.0, 0.0, 0.0, 0.0), vec4f(0.0, 1.0, 0.0, 0.0), vec4f(0.0, 0.0, 1.0, 0.0), vec4f(0.0, 0.0, 0.0, 1.0));
        }

        fn getLocalPosition(vertexPosition: vec3f) -> vec3f {
            return vertexPosition;
        }
    `;

    const normalCoreVS = /* wgsl */ `
        fn getLocalNormal(vertexNormal: vec3f) -> vec3f {
            return vertexNormal;
        }

        fn getNormalMatrix(modelMatrix: mat4x4f) -> mat3x3f {
            // the cofactor matrix: the inverse transpose up to scale, so normals stay
            // perpendicular under non-uniform scale; the determinant's sign keeps mirrored
            // instances facing out
            let m0 = modelMatrix[0].xyz;
            let m1 = modelMatrix[1].xyz;
            let m2 = modelMatrix[2].xyz;
            let c0 = cross(m1, m2);
            let s = select(1.0, -1.0, dot(m0, c0) < 0.0);
            return mat3x3f(c0 * s, cross(m2, m0) * s, cross(m0, m1) * s);
        }
    `;

    // the instance's rotation, centre and radius travel as flat varyings: the fragment stage
    // reads no storage
    const litEngineDeclarationVS = /* wgsl */ `
        ${meshletStructsWGSL}
        varying @interpolate(flat) vImpostorRot0: vec3f;
        varying @interpolate(flat) vImpostorRot1: vec3f;
        varying @interpolate(flat) vImpostorRot2: vec3f;
        varying @interpolate(flat) vImpostorCenter: vec3f;
        varying @interpolate(flat) vImpostorHalf: vec2f;
        varying @interpolate(flat) vImpostorLayer: u32;
        varying @interpolate(flat) vImpostorFade: f32;
        varying @interpolate(flat) vImpostorPickId: u32;
        var<private> dImpostorRot0: vec3f;
        var<private> dImpostorRot1: vec3f;
        var<private> dImpostorRot2: vec3f;
        var<private> dImpostorCenter: vec3f;
        var<private> dImpostorHalf: vec2f;
        var<private> dImpostorLayer: u32;
        var<private> dImpostorFade: f32;
        var<private> dImpostorPickId: u32;
    `;

    const litEngineMainStartVS = /* wgsl */ `
        let impostorInstance = impostorList[pcInstanceIndex * 2u];
        dImpostorFade = bitcast<f32>(impostorList[pcInstanceIndex * 2u + 1u]);
        let impostorObject = objectData[impostorInstance];
        dImpostorPickId = impostorObject.pickId;
        let impostorMatrix = impostorObject.worldMatrix;
        dImpostorLayer = impostorObject.flags >> ${OBJECT_IMPOSTOR_LAYER_SHIFT}u;
        // per layer: [centre xyz, bounding radius] then [half width, half height, 0, 0] of the box
        // the views were framed on (prototype units, the instance's scale multiplies them)
        let impostorProto = impostorProtos[dImpostorLayer * 2u];
        dImpostorCenter = (impostorMatrix * vec4f(impostorProto.xyz, 1.0)).xyz;
        dImpostorHalf = impostorProtos[dImpostorLayer * 2u + 1u].xy * impostorObject.maxScale;
        dImpostorRot0 = normalize(impostorMatrix[0].xyz);
        dImpostorRot1 = normalize(impostorMatrix[1].xyz);
        dImpostorRot2 = normalize(impostorMatrix[2].xyz);

        // a quad facing the camera, spanning the box every baked view framed
        let impostorRight = vec3f(uniform.matrix_view[0][0], uniform.matrix_view[1][0], uniform.matrix_view[2][0]);
        let impostorUp = vec3f(uniform.matrix_view[0][1], uniform.matrix_view[1][1], uniform.matrix_view[2][1]);
        let impostorCorner = vec2f(f32(pcVertexIndex & 1u), f32((pcVertexIndex >> 1u) & 1u)) * 2.0 - 1.0;
        vertex_position = vec4f(dImpostorCenter + (impostorRight * impostorCorner.x * dImpostorHalf.x + impostorUp * impostorCorner.y * dImpostorHalf.y), 1.0);
        vertex_normal = cross(impostorRight, impostorUp);
    `;

    const litEngineMainEndVS = /* wgsl */ `
        output.vImpostorRot0 = dImpostorRot0;
        output.vImpostorRot1 = dImpostorRot1;
        output.vImpostorRot2 = dImpostorRot2;
        output.vImpostorCenter = dImpostorCenter;
        output.vImpostorHalf = dImpostorHalf;
        output.vImpostorLayer = dImpostorLayer;
        output.vImpostorFade = dImpostorFade;
        output.vImpostorPickId = dImpostorPickId;
    `;

    const litEngineDeclarationPS = /* wgsl */ `
        ${meshletStructsWGSL}
        varying @interpolate(flat) vImpostorRot0: vec3f;
        varying @interpolate(flat) vImpostorRot1: vec3f;
        varying @interpolate(flat) vImpostorRot2: vec3f;
        varying @interpolate(flat) vImpostorCenter: vec3f;
        varying @interpolate(flat) vImpostorHalf: vec2f;
        varying @interpolate(flat) vImpostorLayer: u32;
        varying @interpolate(flat) vImpostorFade: f32;
        varying @interpolate(flat) vImpostorPickId: u32;

        #ifdef PICK_PASS
        fn getPickOutput() -> vec4f {
            return encodePickOutput(vImpostorPickId);
        }
        #endif

        var impostorAlbedo: texture_2d_array<f32>;
        var impostorAlbedoSampler: sampler;
        var impostorNormal: texture_2d_array<f32>;
        var impostorNormalSampler: sampler;
        uniform impostorDebug: f32;
        // x: weight of the crown-shaped normal blended over the baked one, y: darkening towards
        // the foot of the crown, z: albedo scale (a real crown's leaves face every way, most of
        // them away from the sun, so its average sits well under that of one lit surface)
        uniform impostorShading: vec4f;

        var<private> dImpostorAlbedo: vec3f;
        var<private> dImpostorAlpha: f32;
        var<private> dImpostorNormal: vec3f;

        // direction (towards the camera, object space) of the baked view at frame = (column, row)
        fn impostorFrameDirection(frame: vec2f) -> vec3f {
            let g = frame / (${GRID} - 1.0);
            let x = g.x - g.y;
            let z = -1.0 + g.x + g.y;
            return normalize(vec3f(x, 1.0 - abs(x) - abs(z), z));
        }

        #if !defined(PICK_PASS) && !defined(PREPASS_PASS) && !defined(SHADOW_PASS)
        fn impostorEvaluate() {
            dImpostorAlpha = 0.0;
            dImpostorAlbedo = vec3f(0.5);
            dImpostorNormal = vec3f(0.0, 1.0, 0.0);

            // mip: texels of a view tile per screen pixel. Derivatives first - they are only legal
            // in uniform control flow, and the early returns below are not.
            let pixelWorld = max(length(dpdx(vPositionW)), length(dpdy(vPositionW)));
            let lod = clamp(log2(max(pixelWorld * ${TILE} / (2.0 * vImpostorHalf.x), 1e-3)), 0.0, 3.0);

            let rotation = mat3x3f(vImpostorRot0, vImpostorRot1, vImpostorRot2);
            let inverse = transpose(rotation);
            let cameraPosition = uniform.view_position;

            // the baked view nearest the camera: its direction in object space, on the
            // hemi-octahedral grid (the bake covers the upper hemisphere only)
            var view = normalize(inverse * (cameraPosition - vImpostorCenter));
            view = normalize(vec3f(view.x, max(view.y, 0.0) + 1e-4, view.z));
            let oct = view / (abs(view.x) + abs(view.y) + abs(view.z));
            let grid = vec2f((1.0 + oct.x + oct.z) * 0.5, (1.0 + oct.z - oct.x) * 0.5) * (${GRID} - 1.0);
            let frame = clamp(floor(grid + 0.5), vec2f(0.0), vec2f(${GRID} - 1.0));

            // where the pixel ray meets that view's plane (through the centre, normal = the view
            // direction) is where the view saw this part of the tree
            let o = inverse * (vPositionW - vImpostorCenter);
            let d = inverse * normalize(vPositionW - cameraPosition);
            let f = impostorFrameDirection(frame);
            let denominator = dot(d, f);
            if (abs(denominator) < 1e-4) { return; }
            let refUp = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(f.y) > 0.999);
            let right = normalize(cross(refUp, f));
            let up = cross(f, right);
            let hit = o - d * (dot(o, f) / denominator);
            let tile = vec2f(dot(hit, right) / vImpostorHalf.x, dot(hit, up) / vImpostorHalf.y) * 0.5 + 0.5;
            if (any(tile < vec2f(0.0)) || any(tile > vec2f(1.0))) { return; }

            let uv = (frame + vec2f(tile.x, 1.0 - tile.y)) / ${GRID};
            let layer = i32(vImpostorLayer);
            let albedo = textureSampleLevel(impostorAlbedo, impostorAlbedoSampler, uv, layer, lod);
            var normal = textureSampleLevel(impostorNormal, impostorNormalSampler, uv, layer, lod).xyz * 2.0 - 1.0;
            if (dot(normal, normal) < 1e-8) { normal = vec3f(0.0, 1.0, 0.0); }

            // The baked normal is the average over the thousands of randomly angled leaf cards
            // under each texel, so it points about the same way everywhere and the tree shades
            // flat. Blend in the normal of the crown as a whole (an ellipsoid over the framed
            // box) for volume, and darken the lower crown, which the upper one shades.
            let local = tile * 2.0 - 1.0;
            let depth = sqrt(max(1.0 - local.x * local.x - local.y * local.y, 0.0));
            let crown = normalize(vec3f(local.x / vImpostorHalf.x, local.y / vImpostorHalf.y, depth / vImpostorHalf.x));
            let crownObject = right * crown.x + up * crown.y + f * crown.z;
            normal = normalize(mix(normalize(normal), crownObject, uniform.impostorShading.x));
            let height = saturate(tile.y * 1.4);
            let occlusion = mix(1.0 - uniform.impostorShading.y, 1.0, pow(height, 0.8));

            // dithered in over the meshlets (interleaved gradient noise)
            let noise = fract(52.9829189 * fract(dot(pcPosition.xy, vec2f(0.06711056, 0.00583715))));
            dImpostorAlpha = select(0.0, albedo.a, noise < vImpostorFade);
            dImpostorAlbedo = albedo.rgb * occlusion * uniform.impostorShading.z;
            dImpostorNormal = normalize(rotation * normal);
        }
        #else
        fn impostorEvaluate() {
            dImpostorAlpha = 1.0;
            dImpostorAlbedo = vec3f(0.5);
            dImpostorNormal = vec3f(0.0, 1.0, 0.0);
        }
        #endif
    `;

    // the sample is taken once, at the top of the fragment: the frontend functions below only
    // read it, and derivatives are only legal in uniform control flow
    const litEngineMainStartPS = /* wgsl */ `
        impostorEvaluate();
    `;

    // impostorDebug: 1 tints, 2 shows the albedo unlit, 3 the normal, 4 the alpha
    const diffusePS = /* wgsl */ `
        fn getAlbedo() {
            dAlbedo = select(dImpostorAlbedo, vec3f(0.0), uniform.impostorDebug > 1.5);
        }
    `;

    const opacityPS = /* wgsl */ `
        fn getOpacity() {
            dAlpha = dImpostorAlpha;
        }
    `;

    const metalnessPS = /* wgsl */ `
        fn getMetalness() {
            dMetalness = 0.0;
        }
    `;

    const glossPS = /* wgsl */ `
        fn getGlossiness() {
            dGlossiness = 0.15;
        }
    `;

    const emissivePS = /* wgsl */ `
        fn getEmission() {
            var emission = vec3f(0.0);
            if (uniform.impostorDebug > 0.5 && uniform.impostorDebug < 1.5) {
                emission = vec3f(0.5, 0.0, 0.5);
            } else if (uniform.impostorDebug < 2.5) {
                emission = dImpostorAlbedo;
            } else if (uniform.impostorDebug < 3.5) {
                emission = dImpostorNormal * 0.5 + 0.5;
            } else {
                emission = vec3f(dImpostorAlpha);
            }
            dEmission = select(emission, vec3f(0.0), uniform.impostorDebug < 0.5);
        }
    `;

    const alphaTestPS = /* wgsl */ `
        fn alphaTest(a: f32) {
            if (a < 0.5) {
                discard;
            }
        }
    `;

    const normalMapPS = /* wgsl */ `
        fn getNormal() {
            dNormalW = dImpostorNormal;
        }
    `;

    return {
        transformCoreVS,
        normalCoreVS,
        litEngineDeclarationVS,
        litEngineMainStartVS,
        litEngineMainEndVS,
        litEngineDeclarationPS,
        litEngineMainStartPS,
        diffusePS,
        opacityPS,
        metalnessPS,
        glossPS,
        emissivePS,
        alphaTestPS,
        normalMapPS
    };
}

/**
 * Creates the impostor draw material and binds the atlas to it. The caller binds the world's
 * `objectData` and `impostorProtos` storage buffers, and each view's `impostorList` rides on its
 * mesh instance.
 *
 * @param {MeshletImpostors} impostors - The impostor set.
 * @returns {StandardMaterial} The material.
 * @ignore
 */
function createMeshletImpostorMaterial(impostors) {
    const material = new StandardMaterial();
    material.name = 'MeshletImpostor';
    // the lit option defines; the chunk overrides supply the real surface
    material.useMetalness = true;
    material.diffuse.set(1, 1, 1);
    material.metalness = 0;
    material.gloss = 0.15;
    // any value > 0 compiles LIT_ALPHA_TEST; alphaTestPS holds the cutoff
    material.alphaTest = 0.5;
    material.cull = CULLFACE_NONE;
    material.setDefine('PICK_CUSTOM_ID', true);

    const chunks = buildMeshletImpostorChunks({
        grid: impostors.grid,
        tilePixels: impostors.albedo.width / impostors.grid
    });
    material.getShaderChunks('wgsl').add(chunks);
    material.shaderChunksVersion = '2.8';
    material.setParameter('impostorAlbedo', impostors.albedo);
    material.setParameter('impostorNormal', impostors.normal);
    material.setParameter('impostorDebug', 0);
    material.setParameter('impostorShading', [0, 0.1, 0.85, 0]);
    material.update();
    return material;
}

/**
 * The impostor draw's mesh: six indices over four corners (the vertex shader reads the corner
 * from the index) and no vertex buffer. The indirect draw sets the instance count.
 *
 * @param {GraphicsDevice} device - The graphics device.
 * @returns {Mesh} The mesh.
 * @ignore
 */
function createMeshletImpostorMesh(device) {
    const mesh = new Mesh(device);
    mesh.indexBuffer[0] = new IndexBuffer(device, INDEXFORMAT_UINT16, 6, BUFFER_STATIC, new Uint16Array([0, 1, 2, 2, 1, 3]).buffer);
    mesh.primitive[0] = { type: PRIMITIVE_TRIANGLES, base: 0, baseVertex: 0, count: 6, indexed: true };
    return mesh;
}

export { buildMeshletImpostorChunks, createMeshletImpostorMaterial, createMeshletImpostorMesh };
