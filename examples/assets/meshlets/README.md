# Streamed meshlet example assets

The `graphics/meshlet-*` examples load glTF bakes from this directory. They are WebGPU-only
(`@flag WEBGL_DISABLED`): the cull pipeline is compute plus indirect draws. For how the runtime
works, see `docs/meshlet/architecture.md`.

## Bake format

Bakes come from the `streamed-meshlets` transform in Magnopus's private
[gltf-tools](https://github.com/magnopus/gltf-tools) repository. It clusterises each mesh into a
meshlet DAG, quantises the geometry onto a position grid, packs it into fixed-size pages and
(with `--format ktx2`) re-encodes the textures into range-fetchable KTX2 containers.

The format contracts are specified in gltf-tools, not here: `docs/MAG_meshlets_gpu.md`,
`docs/MAG_meshlets_stream.md` and `docs/MAG_texture_streaming.md`. The GPU layouts in
`src/scene/meshlet/constants.js` mirror those specs, and drift between the two shows up as garbage
geometry. The runtime requires manifest `version: 2`; any other version logs an error and loads
nothing. If a bake renders as holes or a flat colour, re-bake with current gltf-tools before
looking for an engine bug.

Each bake is a GLB carrying the `MAG_meshlets_gpu` (per-primitive meshlet data, root material
table) and `MAG_meshlets_stream` (page table, blob URIs, page size, position grid) extensions,
plus sibling directories the runtime streams from over HTTP Range requests:

```
<name>.glb
<name>_meshlets/pages_roots.dat    coarse pages, pinned only while needed by active instances
<name>_meshlets/pages_<n>.dat      the rest of the DAG, fetched on demand
<name>_textures/<array>.bin        KTX2 texture containers, textured bakes only: one per texture
                                   array, mip regions fetched by range
```

URIs resolve relative to the GLB's own URL, so the directories must sit next to it. The examples
dev server answers Range requests (`examples/utils/vite-dev-server.mjs`); a static host without
Range support still works, at the cost of downloading whole shards. Textured bakes are
KTX2 / Basis, so the examples call `basisInitialize()` before loading.

Primitives with `BLEND` materials stay inline in the GLB for the sorted transparent renderer:
`--strip-source-geometry` does not strip them. An example that creates only a meshlet component
omits those surfaces; `meshlet-basic` and `meshlet-zorah` also instantiate the container's
remaining render geometry (the parser leaves streamed placeholders out of it).

## What is committed

Only the bunny (`bunny-v2.glb` + `bunny-v2_meshlets/`) is committed. It is used by
`meshlet-basic` and `meshlet-inspect`, which work from a fresh clone. Every other example needs a
local bake, ignored by git (see the root `.gitignore`).

| Example | Bake |
| --- | --- |
| `meshlet-basic` | bunny; `?asset=magoffice` loads a local `magoffice_streamed.glb` bake |
| `meshlet-inspect` | bunny |
| `meshlet-streaming` | `bistro-v2.glb`, see [Bistro](#bistro) |
| `meshlet-caldera` | `caldera/`, see [Caldera](#caldera) |
| `meshlet-jungle` | `jungle/`, see [JungleRuins](#jungleruins) |
| `meshlet-zorah` | `zorah/`, see [Zorah](#zorah) |

All but `meshlet-basic` and `meshlet-inspect` are hidden from the production sidebar, and
`meshlet-streaming` from the development one too; they remain available at their direct URLs,
for example `/iframe/graphics_meshlet-caldera.html?deviceType=webgpu`.

## Bistro

The input is the Amazon Lumberyard Bistro as a single GLB. Run any topology-changing transform
(welding, decimation) before this one: the DAG is built from the final index buffer.

```sh
cd gltf-tools
deno run -A bundle.ts                       # rebuilds gltf-tools-plugin.js, which the CLI loads
gltf-transform streamed-meshlets bistro2-raw.glb <engine>/examples/assets/meshlets/bistro-v2.glb \
    --strip-source-geometry --format ktx2 --fallback-size 0 \
    --config ./gltf-tools-plugin.js
```

The output name determines the sibling directory names, so keep it `bistro-v2` to match the
example's asset URL. `--fallback-size 0` drops the embedded fallback images, which the meshlet
material never samples.

## Caldera

`graphics/meshlet-caldera` loads the [Caldera](https://github.com/Activision/caldera) map, each
region baked to its own streamed meshlet asset, into one meshlet world. **Jump to** frames a
region and **Collision** overlays the collision soups. Everything lives in the ignored
`caldera/` directory; `caldera_scene.json` lists the baked regions and their bounds, so a partial
bake still loads.

The regions are listed in `examples/utils/caldera/usd2glb.py` (`REGIONS`) and
`examples/utils/caldera/manifest.py` (`ORDER`): the named points of interest and districts
(`capital`, `airfield`, `docks`, `tile_<x>`, ...), the full-terrain tiles `st_<x>`, and a simple
whole-island `terrain`, which the manifest drops once every full tile is baked. Prefabs placed
inside a district (`hotel`, `power_station`) are listed only while their district is missing, so
they never draw twice. Build from a Caldera checkout, with a Python that has `usd-core` and
`numpy`, the glTF Transform CLI and a current gltf-tools checkout:

```sh
CALDERA=~/caldera GLTF_TOOLS=~/gltf-tools PY=/path/to/venv/bin/python \
    examples/utils/caldera/pipeline.sh terrain capital airfield docks
```

Per region, `pipeline.sh`:

1. converts the USD to a raw GLB and a collision GLB (`usd2glb.py`),
2. bakes with `streamed-meshlets --strip-source-geometry --no-tangents`,
3. collapses the per-placement nodes into `EXT_mesh_gpu_instancing` scatters (`instance_nodes.py`),
4. rewrites `caldera_scene.json` (`manifest.py`),
5. deletes the raw GLB, so the disk peak stays near one raw GLB.

Raw files go to `$CALDERA_RAW` (default `./caldera-raw`). Set `KEEPRAW=1` to keep them;
`RECONVERT=1 NOBAKE=1` only regenerates the collision GLBs. `CLUTTER=trees|all` also bakes the
terrain clutter, loaded with `?clutter=1` / `?clutter=all`. `linearize_palette.py` patches bakes
made before the palette colours were written as linear values.

Conversion notes:

- USD is Z-up inches; the GLBs are Y-up metres. Each USD prototype becomes one glTF mesh reused by
  every instance node. Unique world geometry (brushes, patches) is merged per 64 m cell and
  material, since a 12-triangle brush has nothing for the DAG to simplify. Every mesh is recentred
  on its bounds, so the asset-wide position grid is sized by the largest mesh, not by the distance
  from the map origin.
- The data set ships no materials. Faces get one of ~20 flat palette materials by keyword match on
  their material-bind subset name (`plaster`, `wood`, `foliage`, ...). Shadow-caster, decal, caulk
  and tool faces are not drawn.
- Brushes without authored normals get crease-angle (35°) normals.
- Models placed without USD instancing are deduplicated by content: repeats share one glTF mesh
  with a node per placement. Singletons under 5k triangles merge into their cell.
- The full terrain is baked per tile, to stay under the 4 GB GLB limit. Its `height` meshes also go
  into the `world` collision.
- The source engine layers blend, puddle and grime materials as extra patches on the same surface
  as their base. Drawn opaque they z-fight, so `overlays.py` drops brush triangles lying on another
  prim's surface, preferring to drop the copy whose name reads as a layer (`rvl`, `puddle`,
  `stain`, ...).

`caldera_<region>_collision.glb` holds plain world-space meshes (POSITION + indices, no
materials), for loading as regular render meshes. Its scene root has three children, each a set of
per-64 m-cell meshes:

- `player_clip`: `guide` brushes with `clip`, `clip_player*`, `clip_ai*`, `clip_lm_*`, `clip_full*`,
  `clip_stairs*`, `clip_vehicle*`, `clip_grating*` materials, plus `caulk`
- `weapon_clip`: `clip_weap*`, `clip_missile*`, `clip_nosight*` (bullets / sight only)
- `world`: visible brush and patch faces, which are solid in the source engine (water and glass
  excluded). For the terrain this is the full proxy terrain.

Props carry no collision in the data set beyond the clip brushes their prefabs place around them.
Foliage clips, mantle/ladder helpers, triggers and volumes are left out.

`npm run bench:caldera` (from the engine root) flies a fixed route through the regions in Chrome
and prints frame-time statistics; see `examples/utils/caldera/benchmark.mjs` for its options.

## JungleRuins

`graphics/meshlet-jungle` loads the JungleRuins USD scene (a Houdini / Karma export) from the
ignored `jungle/` directory: an 8 km terrain, the pyramid and banyan forest, and millions of
scattered trees and plants. Each group is its own bake, and every scatter is one prototype mesh
with an `EXT_mesh_gpu_instancing` node. Build with a Python that has `usd-core`, `numpy` and
`Pillow`:

```sh
JUNGLE_USD=/path/to/JungleRuins/USD/JungleRuins_Karma.usda \
JUNGLE_HDR=/path/to/JungleRuins/USD/kloppenheim_06_puresky_8k.hdr \
GLTF_TOOLS=~/gltf-tools PY=/path/to/python \
    examples/utils/jungle/pipeline.sh terrain static trees plants moss sky canopy
```

`usd2glb.py` converts each group (`terrain`, `static`, `trees`, `plants`, `moss`), the bake runs
per group, and `manifest.py` writes `jungle_scene.json`. `sky` converts the dome HDR into the
skybox, a clamped lighting copy and the sun direction (`sky.py`); `canopy` then bakes the leaf
maps the example uses for canopy light (`canopy.py`), so run `sky` first. Raw GLBs go to
`$JUNGLE_RAW` (default `./jungle-raw`), the bake to `$JUNGLE_OUT` (default `jungle/` here).

## Zorah

`graphics/meshlet-zorah` loads the Zorah courtyard from NVIDIA's RTX Kit sample
(`zorah_main_public.v2.gltf`, geometry only) from the ignored `zorah/` directory, usually a
symlink to the bake folder on another disk. The source is too large to bake in one piece:

1. Split it into chunks with gltf-tools `tools/split-meshopt-chunks.js` (`chunk_NNN.gltf`).
2. Bake each chunk with `streamed-meshlets` to `<bake_dir>/chunk_NNN.streamed.glb`.
3. `python3 examples/utils/zorah/dielectric.py <bake_dir>` makes the untextured materials
   dielectric (glTF's default is metallic).
4. `python3 examples/utils/zorah/manifest.py <chunk_dir> <bake_dir>` writes `scene.all.json`
   (every baked chunk) and `scene.json` (as many chunks as fit `ZORAH_MESHLET_BUDGET` meshlets)
   into the bake folder.
5. Link `<bake_dir>` as `examples/assets/meshlets/zorah`.

The example loads `scene.all.json` by default; `?scene=default` loads `scene.json`, for machines
with less memory.
