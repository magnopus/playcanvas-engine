# Streamed meshlet example assets

Assets for the `graphics/meshlet-inspect` and `graphics/meshlet-streaming` examples. Each bake is
a GLB carrying the `MAG_meshlets_gpu` (per-primitive meshletData, root material table) and
`MAG_meshlets_stream` (root manifest: page table, blob URIs, page size, position grid)
extensions, plus sibling directories holding what the runtime streams over HTTP Range requests:

```
<name>.glb
<name>_meshlets/pages_roots.dat    root pages, pinned resident for the whole session
<name>_meshlets/pages_<n>.dat      the rest of the DAG, fetched on demand
<name>_textures/<array>.bin        KTX2 texture containers (MAG_texture_streaming), textured
                                   bakes only - one per texture array, mip regions fetched by range
```

URIs are resolved relative to the GLB's own URL, so the directories have to sit next to it. The
examples dev server answers Range requests (`examples/utils/vite-dev-server.mjs`); a static host
without Range support still works, at the cost of downloading whole shards.

## What is committed

| Asset | Size | Textures | Used by |
| --- | --- | --- | --- |
| `bunny-v2.glb` + `bunny-v2_meshlets/` | 2.6 MB | none | both examples |
| `bistro-v2.glb` + `bistro-v2_meshlets/` + `bistro-v2_textures/` | 40 + 291 + 329 MB | 6 KTX2 arrays | `meshlet-streaming` only |

Only the bunny is committed. The Bistro bake is ~660 MB and is ignored via `.gitignore`, so
`meshlet-streaming` (hidden from the production sidebar for that reason) needs it baked locally
before it will run. `meshlet-inspect` works from a fresh clone.

## Baking the Bistro

Bakes come from the `streamed-meshlets` transform in
[gltf-tools](https://github.com/magnopus/gltf-tools), which clusterises each mesh into a
Nanite-style meshlet DAG, quantises the geometry onto a position grid, packs it into fixed-size
pages and (with `--format ktx2`) re-encodes the textures into range-fetchable KTX2 containers:

```sh
cd gltf-tools
deno run -A bundle.ts                       # rebuilds gltf-tools-plugin.js, which the CLI loads
gltf-transform streamed-meshlets bistro2-raw.glb ../engine/examples/assets/meshlets/bistro-v2.glb \
    --stream-geometry --strip-source-geometry --format ktx2 --fallback-size 0 \
    --config ./gltf-tools-plugin.js
```

The input is the Amazon Lumberyard Bistro as a single GLB (`bistro2-raw.glb` in the gltf-tools
working tree - not `public/bunny.glb` or other stripped outputs). Run any topology-changing
transform - welding, decimation - before this one; the DAG is built from the final index buffer.
The output name determines the sibling directory names, so keep it `bistro-v2` to match the
example's asset URL. `--fallback-size 0` drops the embedded fallback images, which the meshlet
material never samples (they were ~67 MB of dead VRAM at 256 px).

Textured bakes are KTX2/Basis: the example calls `basisInitialize()` with the wasm transcoder
under `examples/assets/wasm/basis/` before loading, and the `MeshletComponentSystem` hands the
transcoder to the meshlet director.

The runtime requires manifest `version: 2`; any other version logs an error and loads nothing. If a bake
renders as holes or a flat colour, re-bake with current gltf-tools rather than chasing it in the
engine - the GPU layout contracts in `src/scene/meshlet/constants.js` mirror the tooling specs
`MAG_meshlets_gpu.md` / `MAG_meshlets_stream.md` / `MAG_texture_streaming.md` exactly, and drift
between the two shows up as garbage geometry.

Both examples are WebGPU-only (`@flag WEBGL_DISABLED`): the cull pipeline is compute plus
indirect draws.
