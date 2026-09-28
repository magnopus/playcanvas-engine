# Streamed meshlet example assets

## Basic demo asset and material inspection

The **Asset** dropdown in `graphics/meshlet-basic` selects Bunny, Zorah chunk 003, or
MagOffice. Switching reloads the example to release the previous asset's geometry and texture
pools. A direct office link is `/iframe/graphics_meshlet-basic.html?deviceType=webgpu&asset=magoffice`.
The **Visualisation** dropdown includes meshlet colours, LOD tiers, world normals (including
normal maps), albedo, metalness and roughness.

For the local office bake, keep `magoffice_streamed.glb`, `magoffice_streamed_meshlets/` and
`magoffice_streamed_textures/` together in this directory. These local assets are ignored by git.
The example initializes Basis for the streamed KTX2 textures and instantiates both the meshlet
component and the container's remaining regular geometry. The parser excludes streamed
placeholders from the latter, so this does not draw the streamed surfaces twice.

The inspected office bake retains 41 BLEND primitives (1,623 triangles before instancing).
`--strip-source-geometry` does not strip these: the bake leaves them inline for the sorted
transparent renderer. A loader that creates only a meshlet component will omit these surfaces.
The rebaked package has normal maps and packed `oct16_sign` tangents in its streamed pages, so
the normal-map shader can use the baked frame directly. The glTF-tools bake defaults now retain
tangents; pass `--no-tangents` only when deliberately omitting normal-map support. Runtime does
not reconstruct tangents for tangent-free pages, so those pages use geometric normals. The local
package grew from 473 to 508 pages (about 2.2 MiB more geometry storage).

## Local Zorah reproduction

`graphics/meshlet-basic` loads a local Zorah chunk through an asset link; the dropdown's Zorah
choice selects `chunk_003.streamed.glb`. The example opens on the Bunny; pick Zorah from the
dropdown or open `/iframe/graphics_meshlet-basic.html?deviceType=webgpu&asset=zorah`.
From the engine repository root, create the link with:

```sh
ln -s /Users/adrian.meredith/Documents/zorah_main_public.v2.gltf/final_streamed_meshlets_dagv2 examples/assets/meshlets/zorah
```

The local link should not be committed. It serves the GLB and its relative geometry sidecars
through the examples server, including HTTP Range requests, without copying the multi-GB bake.
Run `npm run dev` in `examples`, then open `graphics/meshlet-basic` in Chrome with WebGPU.
The camera frames the chunk automatically; orbit, pan and zoom into surfaces. Occlusion culling
is on by default; turn it off to isolate streaming gaps.
The example uses CameraFrame with single-sample scene depth and a 2 GiB geometry budget.

`final_streamed_meshlets_dagv2` is baked with the clusterlod-style DAG (group-boundary locks,
uncapped simplification, terminal stuck groups): every primitive reduces to one or a few root
meshlets, so the always-resident set is 4 pages. The previous bake
(`final_streamed_meshlets`) stalled on foliage - its coarse placements needed hundreds of
millions of indices, and most instances were deferred with a budget warning.

## Mission ISS simplification comparison

`graphics/meshlet-groups` shows three synchronized views of `~/Downloads/Mission_ISS.glb`.
Use **Compare → Group sizes** for the requested 8/16/32 bakes, or **Bake approaches (trial)**
for the original group-8 bake, relaxed internal seams, and joining compatible static geometry
before relaxing seams. Orbit, pan, zoom, LOD error and colour mode are shared. **Coarsest LOD**
forces the baked floor for inspection; it does not represent the normal screen-error selection.
The geometry pools fit each whole asset, so memory pressure does not confound the comparison.

Generate the local assets from the engine root (requires the glTF Transform CLI, the current
gltf-tools bundle and its texture encoder dependencies):

```sh
node examples/utils/bake-iss-comparison.mjs \
    ~/Downloads/Mission_ISS.glb \
    /path/to/gltf-tools/gltf-tools-plugin.js
```

Then run the examples server and open
`/iframe/graphics_meshlet-groups.html?deviceType=webgpu`. Add `&approaches=1` to open the trials.
All generated assets, logs, trial plugin copies and `comparison.json` live in the ignored
`examples/assets/meshlets/iss-group-comparison/` directory. The helper never modifies the
gltf-tools bundle or source. Trial patches deliberately fail if the bundle no longer matches
the implementation being tested. Textures are encoded once and shared across variants.

Measured baked floors for this source:

| Bake | Root triangles | Root meshlets |
| --- | ---: | ---: |
| Group 8 | 68,748 | 1,764 |
| Group 16 | 66,312 | 1,680 |
| Group 32 | 66,614 | 1,670 |
| Group 8 + permissive internal seams | 56,294 | 1,491 |
| Join + permissive seams + retain disconnected parts | 47,406 | 1,190 |

These are experiments, not new bake defaults. Seam relaxation adds meshoptimizer's `Permissive`
flag while retaining `LockBorder` and the existing attribute weights. Joining reduces 343
primitives to 28, allowing simplification across former primitive boundaries. The joined trial
also disables proximity welding (larger merged bounds otherwise increase the tolerance and
erase fine detail), and disables card decimation (the current heuristic mistakes disconnected
opaque ISS parts for foliage and removes whole solar panels at coarse LODs).

The joined trial retains 247,525 finest triangles versus 247,337 in the original bakes;
the source has 247,547. This difference comes from the welding/degenerate-removal policy, not
intentional finest-LOD simplification. An earlier joined trial reaching 38,749 root triangles
was rejected because it dropped components. A smaller baked floor alone does not establish
a runtime improvement: compare the live drawn counts at the same camera and error setting,
and inspect material appearance as well as meshlet colours. Joining also changes LOD group
bounds and granularity, which can keep more geometry detailed.

Chrome/WebGPU check at 1680 × 960, synchronized cameras and 1 px error: the initial framed
view drew 3,641 / 3,564 / 3,524 meshlets (original / permissive / joined). At ~8.2 times that
camera distance it drew 2,338 / 2,120 / 2,656. The joined trial therefore does not establish
a useful general runtime improvement. All three had a pressure scale of 1 and no captured
GPU validation errors. These are drawn counts, not GPU timing measurements.

The next experiment should build spatially local coarse proxies for disconnected detail,
with baked appearance and measured geometric/coverage error, rather than merely increasing
group size or joining all geometry sharing a material. Validate one small region first:
compare its silhouette, shading, distant coverage and selected counts across a camera sweep,
then retain the detailed representation until its replacement satisfies the error threshold.

## Isolated Zorah marble tiles

`graphics/meshlet-tiles` displays the unchanged source geometry for the four marble floor
pieces in chunk 003. Select A1–D1, switch between top and angled views, and orbit/zoom.
It renders one ordinary mesh, without meshlet selection, so the shape can be identified
before comparing proxy bakes. A neutral preview material makes relief visible; the extracted
GLBs retain their original material records.

```sh
node examples/utils/extract-zorah-tiles.mjs \
    ~/Documents/zorah_main_public.v2.gltf/zorah_main_public.v2.gltf \
    /path/to/gltf-tools/node_modules/meshoptimizer
```

The extractor reads only the referenced ranges of the ~10 GB source binary, decodes meshopt
compression, and writes the following under the ignored `assets/meshlets/zorah-tiles/` directory:

| File | Source mesh | Chunk 003 mesh | Source triangles | Placements | Size |
| --- | ---: | ---: | ---: | ---: | ---: |
| `marble-a.glb` | 358 | 82 | 9,229,706 | 95 | 160.3 MiB |
| `marble-b.glb` | 359 | 83 | 1,705,308 | 95 | 29.7 MiB |
| `marble-c.glb` | 357 | 81 | 1,175,311 | 192 | 20.4 MiB |
| `marble-d.glb` | 360 | 84 | 210,119 | 190 | 3.7 MiB |

Each GLB contains a single tile in its unchanged local coordinates. Its corresponding
`marble-*.placements.json` retains all original node names, source node indices and transforms,
referencing mesh 0 in the extracted asset. `extraction.json` records provenance, counts and
SHA-256 hashes of decoded buffer views. The source NVIDIA MIT license is copied alongside.
Geometry is not welded, simplified or normal-generated by the extractor. This geometry-only
Zorah export has POSITION and indices, but no authored normals, UVs or textures; the viewer
generates normals on import.

Tile A1 spans about 1.802 × 1.801 m with 29 mm of vertical relief. The existing streamed bake
still has 2,470 roots / 140,881 triangles for each copy. It is a candidate for a low-poly relief
surface with baked normals, with the tile outline and coverage retained. This is a different
experiment from merely changing simplification group sizes.

The captured overhead profile must not be attributed to these tiles without another measurement:
its largest selected groups were mesh 77 (hedge, 129,974 selected records), mesh 67 (tree leaves,
78,067) and mesh 1 (facade ornaments, 67,114). This extraction identifies the floor geometry;
it does not establish that it caused the earlier dominant culling cost.

Validation: all four GLBs rendered separately in Chrome/WebGPU without captured GPU validation
errors; decoded buffer hashes, index bounds, triangle counts, material records and original
placement transforms were checked after writing.

## Deferred marble A1 proxy technique

The relief-proxy experiment has been removed. Its bake method, measurements, limitations
and proposed LOD transition are recorded locally in
`~/Documents/Meshlet-Relief-Proxy-Notes.md`. The source tile extractor and inspector above
remain available; there is no automatic proxy LOD integration.

## Caldera scene

`graphics/meshlet-caldera` (hidden) loads the [Caldera](https://github.com/Activision/caldera) points
of interest and the simplified terrain, each baked to its own streamed meshlet asset in one meshlet
world. Open `/iframe/graphics_meshlet-caldera.html?deviceType=webgpu`; **Jump to** frames a region,
**Collision** overlays the collision soups. Everything lives in the ignored
`assets/meshlets/caldera/` directory; `caldera_scene.json` lists the baked regions and their bounds,
so a partial bake still loads.

| Region | Source prim (under `mp_wz_island_geo`) | Unique / instanced tris |
| --- | --- | ---: |
| Capital (contains `restaurant_01`) | `map_capital` | 37.4M / 194M |
| Airfield | `map_airfield` | 29.9M / 161M |
| Phosphate Mine | `map_phosphate_mine` | 22.9M / 123M |
| Beachhead | `map_beachhead` | 15.6M / 160M |
| Hotel | `map_tile_p/hotel_01` | 6.1M / 19.2M |
| Power Station | `map_tile_n/power_station_01` | 4.4M / 19.1M |
| Terrain (simple) | `st_main`, every tile on its `proxy` variant | 7.5M / 7.5M |
| Terrain tiles `st_b` … `st_p` (14) | `st_main/st_<x>` on its `full` variant, clutter skipped | e.g. `st_d` 2.6M / 14M |

The restaurant POI is not exported separately: it is placed inside the Capital, and a second copy
would draw twice. The bake of all seven regions is 5.3 GB of pages (Capital alone 1.3 GB) and peaks
at ~4 GB of RAM per bake. Build from a Caldera checkout, with a Python that has `usd-core` and
`numpy`, the glTF Transform CLI and a current gltf-tools bundle:

```sh
CALDERA=~/caldera GLTF_TOOLS=~/gltf-tools PY=/path/to/venv/bin/python \
    examples/utils/caldera/pipeline.sh terrain capital airfield phosphate_mine beachhead hotel power_station
```

Per region, `pipeline.sh` converts (`usd2glb.py`), bakes with
`streamed-meshlets --stream-geometry --strip-source-geometry --no-tangents`, rewrites
`caldera_scene.json` (`manifest.py`), and deletes the raw render GLB, so the disk peak stays near
one raw GLB (Capital is 1.2 GB raw). Raw files go to `$CALDERA_RAW` (default `./caldera-raw`).
Set `KEEPRAW=1` to keep them; `RECONVERT=1 NOBAKE=1` only regenerates the collision GLBs.

Conversion notes:

- USD is Z-up inches; the GLBs are Y-up metres. Each USD prototype becomes one glTF mesh reused by
  every instance node. Unique world geometry (brushes, patches) is merged per 64 m cell and material,
  since a 12-triangle brush has nothing for the DAG to simplify. Every mesh is recentred on its
  bounds, so the asset-wide position grid is sized by the largest mesh, not by the distance from the
  map origin.
- The data set ships no materials. Faces get one of ~20 flat palette materials by keyword match on
  their material-bind subset name (`plaster`, `wood`, `foliage`, ...). Shadow-caster, decal, caulk
  and tool faces are not drawn. Points are time-sampled at t=1 on some prims, and xmodel geometry
  is named `tag_origin*` after its root bone, so neither can be skipped.
- Brushes without authored normals get crease-angle (35°) normals.
- Models placed without USD instancing are deduplicated by content (local points, topology and
  materials): repeats share one glTF mesh with a node per placement. On `st_k` this takes the
  non-instanced models from 61.6M to 17.3M triangles. Singletons under 5k triangles merge into
  their cell.
- The full terrain is baked per tile (all 14 at once would be ~8 GB of raw GLB, over the 4 GB
  limit). Its `height` meshes also go into the `world` collision. Once every tile is baked the
  manifest drops the simple terrain, which would otherwise overlap it.
- Terrain clutter (grass, small rocks: 14 point instancers, 1.54M points) is skipped. Its
  prototypes live under the instancer, so the traversal prunes them rather than drawing each once
  at its authored origin.
- The source engine layers blend, puddle and grime materials as extra patches on the exact same
  surface as their base (no vertex alpha survives in the USD). Drawn opaque they z-fight, so
  `overlays.py` drops brush triangles whose centroid lies on another prim's surface within 0.1"
  (the hotel: 100k of 252k brush triangles). Of two exactly coplanar copies, the one whose name reads
  as a layer (`rvl`, `puddle`, `stain`, `grunge`, `dust`, ...) goes. Real trim sits ≥ 0.5" proud of
  its wall and is kept.

`caldera_<region>_collision.glb` holds plain world-space meshes (POSITION + indices, no materials),
for loading as regular render meshes. Its scene root has three children, each a set of per-64 m-cell
meshes:

- `player_clip`: `guide` brushes with `clip`, `clip_player*`, `clip_ai*`, `clip_lm_*`, `clip_full*`,
  `clip_stairs*`, `clip_vehicle*`, `clip_grating*` materials, plus `caulk`
- `weapon_clip`: `clip_weap*`, `clip_missile*`, `clip_nosight*` (bullets / sight only)
- `world`: visible brush and patch faces, which are solid in the source engine (water and glass
  excluded). For the terrain this is the full proxy terrain.

Props (xmodels) carry no collision in the data set beyond the clip brushes their prefabs place
around them. Foliage clips, mantle/ladder helpers, triggers and volumes are left out.

## Other examples

Assets for the `graphics/meshlet-inspect` and `graphics/meshlet-streaming` examples. Each bake is
a GLB carrying the `MAG_meshlets_gpu` (per-primitive meshletData, root material table) and
`MAG_meshlets_stream` (root manifest: page table, blob URIs, page size, position grid)
extensions, plus sibling directories holding what the runtime streams over HTTP Range requests:

```
<name>.glb
<name>_meshlets/pages_roots.dat    coarse pages, pinned only while needed by active instances
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
