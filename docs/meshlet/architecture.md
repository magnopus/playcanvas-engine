# GPU-driven meshlet renderer: architecture

This document describes how streamed meshlet assets get from a glTF file to pixels. It is aimed at engineers (and LLM agents) who need to change the pipeline. For features that were removed, and notes on how to rebuild them, see [removed-features.md](removed-features.md).

The pipeline renders Nanite-style meshlet DAGs: each primitive is a hierarchy of clusters at several levels of detail, cut per frame on the GPU by projected error, with geometry pages streamed on demand over HTTP Range requests. CPU work per frame is roughly constant in scene size. Culling, LOD selection, occlusion and index generation all run in compute, and drawing is a small fixed number of indirect draws per camera.

## Scope and platform

- WebGPU only. On any other device (WebGL2 included) `MeshletComponentSystem` logs a one-time warning, never creates the pipeline, and meshlet components stay inert: assets still load, nothing streams or draws. There is no WebGL fallback path, and none is planned. The pipeline needs storage buffers, indirect draw and dispatch, and buffer readback.
- Everything under `src/scene/meshlet` is `@ignore` internal API. The public surface is `MeshletComponent`, `MeshletComponentSystem` (`app.systems.meshlet`) and the `ByteCache` / `ByteStore` classes. `MeshletDirector`, `MeshletWorld`, `MeshletResource` and `MESHLET_COLOR_MODE` are exported from `src/index.js` for advanced use and the examples, but they are undocumented.

## The asset contract

Assets are produced by the bake in Magnopus's private `gltf-tools` repository. That repository holds the format specifications (`docs/MAG_meshlets_gpu.md`, `docs/MAG_meshlets_stream.md`, `docs/MAG_texture_streaming.md`). The parser and the layouts in `constants.js` must match those specs. Change the format in both repositories together.

A baked container is a GLB plus sidecar files:

- `MAG_meshlets_gpu` (per primitive): one accessor of resident `meshletData` records, 32 u32 per meshlet. This is the exact GPU layout, so upload is a memcpy. The primitive's inline geometry is a placeholder, and the regular glTF mesh path skips it (`handlesPrimitive`). The root block may carry a baked material table (32 u32 per glTF material).
- `MAG_meshlets_stream` (root): the stream manifest. It holds a binary page table (8 u32 per page), shard blob URIs, root page list, attribute layout, position grid, a fixed page size and an optional `bakeId`.
- `MAG_texture_streaming` (root and node): texture arrays packed into KTX2 containers with per-(layer, mip) byte ranges, plus per-node lightmap references.
- `EXT_mesh_gpu_instancing` scatters become one meshlet instance per scatter entry, all sharing one set of pages.

Version checks are strict. The manifest and the primitive extension must both be version 2, and the page size must equal `PAGE_SIZE_BYTES` (64 KiB), because every resource in a world shares one pool of page-sized slots. Mismatches log a "re-bake" error and the resource is dropped. Each streamed page begins with a 48-byte header (`PAGE_MAGIC` 'MPGS', `PAGE_FORMAT_VERSION`).

The bake stamp (`meshletBakeStamp`) is a short id per resource: the manifest's `bakeId` when present, otherwise a hash of the page table, shard list and texture layout. It is not a format version. It versions the persistent cache keys of the sidecars (`<url>#v=<stamp>`). Sidecar URLs are usually unversioned (`<name>_meshlets/pages_0.dat`), so without the stamp a rebake written over the same files would be served stale cached pages, which the new page table would read as garbage.

## Data flow

```
GLB (MAG_meshlets_gpu / _stream, MAG_texture_streaming)
  |  mag-meshlets.js (GlbResourceExtension, registered by MeshletComponentSystem)
  v
MeshletResource  (CPU only: primitives + meshletData, manifest, MeshletInstanceList,
  |               material table, texture manifest)
  v
MeshletComponent (entity -> asset)  ...grouped by (resource, baseUrl)...
  v
MeshletComponentSystem  -- debounced rebuild, transform/flag sync, playable events
  v
MeshletDirectorGroup  (one per app; shares fetch scheduler, cull shaders, per-camera HZB + draw passes)
  v  one per world
MeshletDirector  -- per-frame orchestration, budgets, views, shadows, rebuild/retention
  |-- MeshletWorld        GPU tables: page pool, residency map, meshletData, objectData,
  |                       material table, MeshletCells, MeshletCutData, MeshletTextures
  |-- MeshletResidency    request readback -> MeshletPageFetcher -> MeshletFetchScheduler -> ByteCache
  |-- MeshletBudgetManager  LOD pressure from page churn and index starvation
  |-- MeshletView (per camera)  -> MeshletCuller (compute chains), MeshletHzb, draw mesh instances
  |-- MeshletShadowRenderer -> MeshletShadowView (per light face, single-phase)
  v
RenderPassMeshletDraw (phase 1 / phase 2) + FramePassMeshletCompute (HZB mips, phase-2 cull)
```

### Parser and resource

`mag-meshlets.js` exports `magMeshlets`, a `GlbResourceExtension`. `createMeshlets` reads the manifest and the per-primitive records, walks the node hierarchy (iteratively, because bakes nest deeply and carry millions of placement nodes), and packs placements into a `MeshletInstanceList`. The list holds 8 bytes per instance and one 48-byte transform shared by all primitives of a node.

`MeshletResource` holds CPU data only. After a world is built, its `meshletData` records can be dropped (`releaseRecords`) so the parsed GLB buffer can be garbage collected. Before the resource joins another world build, `restoreMeshletRecords` re-fetches the GLB and restores them. `MeshletRecordStore` in the component system decides when to do either.

### Component and system

`MeshletComponent` points an entity at a `container` asset (or at a direct `resource` + `baseUrl`). Pages stream relative to the asset's URL, which must be http(s): `blob:` and `data:` URLs are skipped with a warning. It exposes `playable` / `progress` events and `whenPlayable()`.

`MeshletComponentSystem` owns the pipeline and its settings (`poolBytes`, `texturePoolBytes`, `dagPixelThreshold`, `occlusion`, `shadows`, `cameras`, `byteCache`, and others). Its constructor registers the glTF extension, so the meshlet code stays tree-shakeable when the system is not used. The pipeline itself is created lazily on first use. On the app's `framerender` event the system does three things:

- Rebuilds when components or assets changed. Rebuilds are debounced (`rebuildQuietMs`, `rebuildMaxWaitMs`, `deferRebuildWhileLoading`), because a rebuild re-finalizes the whole world.
- Syncs entity transforms and enabled/shadow flags into the world as cheap buffer writes, with no rebuild.
- Fires playable transitions.

The system delegates to three helpers: `MeshletWorldPlanner` (which components go into which world), `MeshletRecordStore` (releasing and restoring CPU records) and `MeshletPlayableTracker` (playable events). Setting defaults live in `MESHLET_DEFAULTS`.

A rebuild groups components by (resource, baseUrl). Components that share an asset share one set of pages, records and textures. The groups are then packed into worlds of at most `maxWorldMeshlets` meshlets each. Disabled components stay in the world with a per-instance hidden bit, so re-enabling one is a flag write.

### Director group and director

`MeshletDirectorGroup` registers itself as `renderer.meshletDirector`, so the forward renderer talks only to the group. It keeps one `MeshletDirector` per world and shares the costs that do not grow with content:

- one `MeshletFetchScheduler`, so the concurrency cap and priorities span all worlds
- one `MeshletCullShaders`
- per camera, a `MeshletGroupCamera`: one `MeshletHzb`, one phase-1 and one phase-2 `RenderPassMeshletDraw`, and the HZB-mip and phase-2-cull frame passes covering every world's view of that camera

`MeshletDirector` runs one world each frame. It ticks residency, textures and budget, creates and destroys `MeshletView`s for the active cameras (`maxViews`, default 2), encodes each view's phase-1 cull, owns the `MeshletShadowRenderer`, and provides picking and outline selection. `rebuild(build)` swaps in a new world and keeps streaming state where it can (`_adoptStreamingState`, with per-view state carried by `MeshletRebuildCarry`). The page pool, its resident pages, the LRU state, pinned roots, visibility bits and the texture system all carry over when the leading resources are unchanged, so an added asset does not re-stream the existing ones.

### World

`MeshletWorld` holds the GPU state of one set of resources. Building it is one-shot: add resources, then `finalize()`. Its responsibilities:

- **Page pool and residency.** The page pool is one storage buffer of 64 KiB slots. The residency map is page to slot (`PAGE_NOT_RESIDENT`). The requests buffer holds per-page marks (`PAGE_REQUEST`) followed by per-material texel-rate marks.
- **Tables.** `meshletData` holds every resource's records, rebased into one world-global meshlet and page space. `objectData` holds one 16-word row per instance, followed by one row per primitive. An instance row carries only what differs between copies (matrix, primitive row, flags, max scale, first pair bit). The material table (`MeshletMaterialTable`) holds one 32-word record per material: baked, or synthesized from `baseColorFactor` when the bake has none.
- **Spatial cells (`MeshletCells`).** Instances are sorted by radius class and then Morton code, and grouped into cells of `MESHLET_CELL_SIZE` (64). Every per-instance and per-task pass runs one workgroup per visible cell, so a view, each shadow cascade included, pays for what is near it rather than for the whole world.
- **Cut tables (`MeshletCutData`).** These rebuild replacement groups from the baker's level-major cluster order. They lay out per-group task rows (representative, parents, pages, owned tasks, children) for the queued cut traversal, along with `rootPages` (pinned) and `floorPages` (the "playable floor", see below).
- **Draw buckets and materials.** There are three buckets: opaque, opaque two-sided and masked. They differ only in cull mode and whether the alpha-test discard is compiled in. Keeping the discard out of the opaque buckets preserves early-Z. Each bucket has one `createMeshletLitMaterial` (a `StandardMaterial` whose vertex stage pulls from the page pool and whose frontend reads the material record, via `buildMeshletLitChunks`) and an optional debug material.
- **Budget resolution.** `resolveMeshletBudget` splits `poolBytes` (see Gotchas).
- **Textures.** `MeshletTextures`, when any resource carries a texture manifest.

### Views and culling

`MeshletView` holds everything that depends on one camera's cull result: cull params, counters, root list, claim bits (one bit per instance-meshlet pair), persistent visibility bits, cut queue, records, the GPU-written draw index buffer, six mesh instances (three buckets times two phases) pointed at indirect draw slots, and the view's `MeshletCuller`. The view grows its index, record and cut-queue capacities from read-back demand, within ceilings set by the budget.

`MeshletCuller` builds and dispatches the compute chains. The shaders are stateless and shared, and all per-view state lives on the `Compute` instances. Phase 2 uses its own `Compute` instances, because uniform writes land before the frame's command buffer: one instance dispatched twice would give both dispatches the later values.

`MeshletShadowView` is a single-phase `MeshletView` for one light face (a directional cascade, a spot frustum or an omni cube face). The face's LOD cut is driven by its own shadow-map texel rate. Root admission is ranked by the viewing camera's position, because the light's own camera sits far back and an orthographic projection has no size falloff. Faces share one claim plane and one root list (with hysteresis disabled) and bind a placeholder for the visibility bits. `MeshletShadowRenderer` registers their mesh instances as layer shadow casters, so the engine's own `ShadowRenderer` draws them. It also builds a per-cascade caster volume (`tightCasterVolume`: extra cull planes, `CULL_PARAMS.CASTER_PLANES`) that bounds casters to those that can shadow the slice's receivers. Shadows are off by default (`shadows` / `shadowsEnabled`). When they are on, spot and omni lights are included unless `shadowRenderer.localLights` is false.

### Two-phase occlusion

The persistent visibility bit of each (instance, meshlet) pair means "visible the last time it was tested":

1. Phase 1 draws the pairs whose bit is set and that pass the frustum test.
2. `MeshletHzb` builds a max-depth pyramid in linear view depth from the phase-1 depth plus the scene's opaque layers. Mip 0 is a fragment pass (depth textures cannot be storage-written), and the remaining mips are compute.
3. Phase 2 re-tests the same selected cut against the pyramid, draws only the newly visible pairs, and sets or clears their bits.

Root admission also tests instances against the previous frame's pyramid (`occlusionAdmission`), but only while that pyramid is the same texture at the same size. Otherwise occluded instances would spend capacity that visible ones need. Occlusion needs a depth source: either a render target with a depth texture, or a CameraFrame that renders linear scene depth (`wantsSceneDepth`). Without one, the view falls back to single-phase culling with a warning.

### Streaming

- `MeshletResidency` (per world). It pins every resource's root pages at load and fetches the playable floor eagerly (`loadFloor`). Each frame it reads back the request marks, queues missing pages, installs arrived pages into free or LRU-evicted slots (at most `maxInstallBytesPerFrame` per frame), and flushes the residency map. Failed streams back off exponentially, and a 404 kills the stream.
- `MeshletPageFetcher` (per resource). It sorts wanted pages by shard offset and merges them into byte runs (gaps up to 512 KiB, runs up to 16 MiB).
- `MeshletFetchScheduler` (shared). It caps requests in flight (`maxConcurrentFetches`, raised while `fastLoad` applies before playable) and serves priority classes by share: roots, then pages, then texture tails, then fine mips. It coalesces nearby ranges on one URL, drops requests whose `stillWanted` has turned false, and consults the `ByteCache`.
- `ByteCache` (`src/platform/net`). This is a persistent URL-keyed cache of aligned chunks, so ranges are served once every chunk they touch is stored, with LRU eviction past `maxBytes`. Storage goes through a `ByteStore`: `OpfsByteStore`, `CacheStorageByteStore` or `MemoryByteStore`. The same cache can also back `ResourceLoader#byteCache`.

"Playable" means the roots, the playable floor and the texture tails are resident. The playable floor is the coarsest level of each primitive that is not derived from sloppy (vertex-clustered) simplification (`MESHLET_FLAG_SLOPPY`), plus the pages needed to refine to it.

### Textures

`MeshletTextures` streams KTX2 texture arrays in four families (srgb, srgba, normal, linear). Each family is one transcode format. Each texture has two tiers:

- **Tail.** An always-resident layer holding the mips at or below 256 px, loaded with one range request per source array. The tail is the terminal fallback, so no texture needs pinning.
- **Fine pool.** An LRU slot pool per family that holds only the mips above the tail, capped at `maxFineTextureSize`.

The meshlet cull writes per-material texel-rate marks into the requests buffer, in sixteenths of a mip (`TEXEL_RATE_PER_MIP`). They arrive with the page readback, and `processMarks` turns them into desired mips, slot acquisitions and fetches. Per-texture state reaches the shader through the `texResidency` and slot-descriptor buffers. Under pressure, `mipBias` rises and frees slots. This mirrors what `MeshletBudgetManager` does for geometry. KTX2 transcoding is injected (`group.transcode = basisTranscode`), because scene code cannot import the framework's Basis handler.

## Per-frame order

CPU, in order:

1. `framerender` → `MeshletComponentSystem._onFrameRender`: a rebuild if due, a transform/flag sync, record release, and playable events.
2. `ForwardRenderer.update` → `MeshletDirectorGroup.update` → each `MeshletDirector.update`:
   - `residency.frameUpdate()`: drain and install arrived pages, flush the residency map, kick the request-marks readback if none is in flight, and clear the requests buffer in-encoder.
   - `textures.flush()`, playable state, `_distributeIndexBudget()`, `budget.update()` (LOD `pressureScale`), and the shadow LOD controller.
   - For each camera view: `syncMaterials`, `applyPendingGrowth` (buffer swaps happen here and nowhere else), `monitorIndexDemand` (kick the counters readback), frustum, projection and pixel scale, HZB setup, then `culler.beginFrame()`. `beginFrame()` uploads the cull params, clears counters and claim bits in-encoder, and encodes the phase-1 chain.
   - Then the group points each camera's shared draw passes at this frame's views.
3. `updateShadowLights` (after light visibility, before `cullComposition`): `MeshletShadowRenderer.syncLights` reconciles the face views and caster registration.
4. Frame graph build: `buildCameraPasses` (or `buildCameraFramePasses` for CameraFrame) emits phase-1 draw → opaque scene passes → [HZB mip 0 → HZB mips → phase-2 cull for every world → phase-2 draw] → transparent scene passes. Phase 1 owns the camera clear. On the render-action path, a camera that does not clear its full viewport falls back to drawing meshlets after the scene (incorrect compositing) with a warning.
5. `ForwardRenderer.cull` → `updateShadows`: one single-phase chain per shadow face, encoded once `cullComposition` has fitted the cascade cameras.

All compute encoded in steps 2 and 5 runs before any frame-graph pass. The phase-2 work runs inside the frame graph through `FramePassMeshletCompute`.

GPU phase-1 chain per view. It is one compute pass, or one pass per stage when the GPU profiler's pass detail is on:

1. **Cell cull** → indirect args (one workgroup per visible cell).
2. **Root classify**: drop instances that are hidden, non-casting (shadow views), off-frustum, or behind last frame's HZB. Rank the rest by projected size into 16 log bands, with hysteresis from last frame's admissions. Then band args and a scatter into band order.
3. **Root admit**, once per band, largest first: needs the root pages resident, reserves root indices and records against capacity (`rootBudgetFraction` under a budget), and queues root-emit chunks and root-owned cut tasks.
4. **Queued cut stages**, coarse level to fine: an args pass, then one thread per queued task. Each task decides whether its replacement group refines (error against `dagPixelThreshold × pressureScale × bucket scale`, pages resident, capacity) and queues the tasks it owns. Missing pages are marked `MISSING` in the requests buffer.
5. **Cut emit**: one thread per queue entry. A meshlet is selected when its parent group refined and its own group did not.
6. **Meshlet cull**: frustum, cone backface (backface-culled bucket only), and the occlusion phase test. Survivors append a record, reserve a bucket index range, mark their page `USED`, and write texel-rate marks.
7. **Finalize args**: one `DrawIndexedIndirect` per bucket, plus the index-write dispatch.
8. **Index write**: expands each record's u8 corner stream from the page pool into `(recordIndex << 8) | localVertex`.

The phase-2 chain is reset → meshlet cull (phase 2) → finalize args → index write.

Readbacks (both through `WebgpuReadbackPool`, at most one in flight each):

- **Request marks, per world.** These arrive one to two frames late and drive page fetches, LRU touches and texture mips.
- **Counters, per view.** Index, record and cut-queue demand, root admission statistics and the rendered count, a few frames late. These drive buffer growth and the budget-exceeded report.

## Buffer and format contracts

`constants.js` is the source of truth for every shared layout: `MESHLET_DATA`, `OBJECT_DATA`, `PRIMITIVE_DATA`, `MATERIAL_RECORD`, `CULL_PARAMS`, `MESHLET_COUNTER`, the page header and page table, flag bits, buckets and claim-buffer offsets. The JS writes flat `u32` arrays at those offsets. Several constants are interpolated directly into WGSL source.

The shaders read the world's buffers through typed structs generated from the field tables in `shaders/meshlet-page-wgsl.js` (`MESHLET_OBJECT_STRUCT`, `MESHLET_DATA_STRUCT`, `MESHLET_MATERIAL_STRUCT`, and others). The rule:

- A layout change edits the `constants.js` enum, the matching struct field table, and every JS writer together. For baked records (`meshletData`, the material record, page header and page table) it also changes the gltf-tools spec and baker.
- The struct tables follow WGSL alignment. Fields that are not 16-byte aligned are declared `array<f32, 3>` rather than `vec3f`, so nothing pads.
- `test/scene/meshlet/meshlet-cull-shaders.test.mjs` checks the struct offsets against the enums, and `constants.test.mjs` checks record sizes, counter packing and indirect-argument strides. Run `npm test` after any layout change.

## Gotchas

- **The geometry budget is a hard cap.** `poolBytes` covers the page pool, the world tables, per-view bit planes and root lists, and the draw working set (indices, records, cut queue), not just the pool. `resolveMeshletBudget` computes the fixed costs from scene counts and splits the rest between pages and indices (`indexBudgetFraction`). Views grow only up to their resolved ceilings: what does not fit is not drawn, smallest on screen first. The one exception: a budget below the scene's minimum (pinned roots plus a minimal draw set) falls back to that minimum with a warning. `texturePoolBytes` likewise covers the tails as well as the fine pool. Under sustained pressure, the LOD threshold and texture mip bias rise instead of memory growing.
- **Readback latency.** Every CPU decision based on GPU output lags by one to several frames. Never resize or swap a buffer in a readback callback, because the chain may already be encoded against the old buffer. Record the demand, then apply it in `applyPendingGrowth` at the top of the next frame. Readback copies are encoded before the frame's clears, which is why the counters and requests buffers are cleared in-encoder rather than with `queue.writeBuffer`. Playable transitions fire one frame late.
- **Under-sized buffers corrupt occlusion, not just detail.** Clamped draws punch holes in the HZB, which phase 2 reads as disocclusion. That poisons the visibility bits and triggers a page-demand burst. This is why growth triggers at `growAt` occupancy rather than at overflow, and why capacities carry across rebuilds.
- **Strides and alignment.** Pages are exactly 64 KiB. The drawn index encoding limits a meshlet to 256 vertices. Indirect argument blocks follow WebGPU's layouts (5 and 3 u32). Readback offsets and sizes must be multiples of 4. Several cull passes sit at the per-stage storage-buffer binding limit, which is why primitive rows live inside `objectData` instead of their own buffer. Adding a binding to a cull shader needs care.
- **Uniforms are per `Compute`, not per dispatch.** A `Compute` dispatched twice in one frame sees its last uniform values. Use separate instances, as phase 2 and the per-band root admission do.
- **Indirect slots.** Each culled view takes roughly `MESHLET_ROOT_BANDS + 5` indirect dispatch slots per frame. The director raises the device's slot count, so many shadow faces still fit.
- **Tree shaking.** `package.json` declares `"sideEffects": false`. Registration (the glTF extension, the renderer hook) happens in constructors and `_initialize`, never at module scope. Keep it that way, and put deprecated shims where `AGENTS.md` says.
- **Layering.** `scene/meshlet` must not import from `framework/` at runtime. It uses `@import` types for `CameraComponent` only, which is the existing documented exception. Framework services such as the Basis transcoder are injected.
- **Debug colour modes.** `MeshletWorld#setColorMode(MESHLET_COLOR_MODE.*)` selects `LIT` (0), `MESHLET` (2, a flat debug material tinted per meshlet) or `TEXTURES` (3, lit materials painted with each pixel's texture residency state). Examples pass these values as URL parameters, so keep them stable. Value 1 is unused.
- **Instance flags are the only per-instance CPU state.** Hide, shadow, outline and hover are bits in `objectData` (`OBJECT_FLAG_*`), because one indirect draw covers the whole world: there is no per-object mesh instance or uniform to toggle.
- **Rebuilds are expensive and debounced.** Transforms and flags are cheap. Adding or removing assets re-finalizes the world, and may re-fetch released GLB records first. Keep changes that are retention-compatible (same leading resources, same order) if you want streaming state to survive a rebuild.
- **Cameras.** A view projects with the vertical field of view, even for cameras set to horizontal FOV. It measures pixels against the scene render target (CameraFrame render scale included). With `cssPixels` on, the screen-space thresholds are scaled by render-target pixels per CSS pixel, so high-DPI displays do not refine with the density squared. Each director caps at `maxViews` cameras.

## Where things live

| Concern | Location |
| --- | --- |
| glTF parsing | `src/framework/parsers/glb/extensions/mag-meshlets.js` |
| Component, system | `src/framework/components/meshlet/` (`component.js`, `system.js`, `meshlet-world-planner.js`, `meshlet-record-store.js`, `meshlet-playable-tracker.js`, `meshlet-defaults.js`) |
| Layout constants | `src/scene/meshlet/constants.js` |
| Resource, placements | `src/scene/meshlet/meshlet-resource.js`, `meshlet-instance-list.js` |
| Orchestration | `src/scene/meshlet/meshlet-director-group.js`, `meshlet-director.js`, `meshlet-retention.js` |
| World GPU state | `src/scene/meshlet/meshlet-world.js`, `meshlet-material-table.js` |
| Budget split, LOD pressure | `src/scene/meshlet/meshlet-world-budget.js`, `meshlet-budget-manager.js` |
| Cells, cut tables | `src/scene/meshlet/meshlet-cells.js`, `meshlet-cut-data.js` |
| Per-camera view, culling | `src/scene/meshlet/meshlet-view.js`, `meshlet-culler.js`, `meshlet-cull-shaders.js`, `meshlet-hzb.js` |
| Shadows | `src/scene/meshlet/meshlet-shadow-renderer.js`, `meshlet-shadow-view.js`, `meshlet-caster-volume.js` |
| Passes, materials | `src/scene/meshlet/render-pass-meshlet-draw.js`, `frame-pass-meshlet-compute.js`, `meshlet-lit-material.js`, `meshlet-material.js` |
| WGSL | `src/scene/meshlet/shaders/` (page/struct views, shared cull helpers, cull, root admission, cut, lit chunks) |
| Streaming | `src/scene/meshlet/streaming/` (residency, page fetcher, fetch scheduler, bake stamp) |
| Textures | `src/scene/meshlet/textures/` (textures, texture source, KTX2) |
| Byte cache and stores | `src/platform/net/byte-cache.js`, `byte-store.js`, `opfs-byte-store.js`, `cache-storage-byte-store.js`, `memory-byte-store.js` |
| Renderer hooks | `src/scene/renderer/forward-renderer.js`, `src/extras/render-passes/frame-pass-camera-frame.js`, `src/framework/graphics/render-pass-picker.js` |
| Tests | `test/scene/meshlet/` |
| Examples, bake tooling | `examples/src/examples/graphics/meshlet-*`, `examples/utils/{caldera,jungle,zorah}`, `examples/assets/meshlets/README.md` |
