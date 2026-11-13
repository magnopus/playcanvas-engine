# Meshlet features removed to shrink the pipeline

These features were cut to reduce the meshlet code to its core. The full implementations live on
the branch `feat/gpu-driven-final-before-cutting-features`. Each section covers what the feature
did, where it lived, and what we would do differently if it comes back.

---

## Impostors (octahedral billboards for distant foliage)

**What it did.** The meshlet DAG cut erases leaf cards at distance: each card is a tiny
geometric error, however much of the canopy it makes up. So below `impostorPixels` (the projected
radius of a whole tree) a tree drew as one camera-facing quad instead. The quad sampled the
nearest of 8x8 hemi-octahedral views baked into an albedo/normal texture array, one layer per
prototype.

**Where it lived.**
- `meshlet-impostors.js`: StandardMaterial chunk overrides and the quad mesh.
- The gate in `meshletRootClassifyWGSL`.
- `meshletImpostorArgsWGSL`: the indirect draw arguments.
- `OBJECT_FLAG_IMPOSTOR` and `OBJECT_FLAG_IMPOSTOR_PRIMARY`, with the atlas layer in flag bits
  16-31.
- `CULL_PARAMS.IMPOSTOR` and `MESHLET_COUNTER.IMPOSTORS`.
- A prototype table buffer, a per-view impostor list and mesh instance, and impostor-aware cell
  bounds in `MeshletCells`.
- Examples `meshlet-impostor-bake` and `meshlet-impostor-test`, plus the bake tooling in
  `examples/utils/jungle` (`protos.py`, `bake_server.py`).

**What worked.**
- The gate must be stateless and use the whole prototype sphere through the shared instance
  matrix. Per-primitive bounds, per-primitive distance bias or per-instance hysteresis made trees
  draw as billboard and meshlets at once.
- Only the prototype's primary primitive appends the billboard.
- A short dithered fade band, where the billboard fades in over the meshlets just above the
  switch size, hides the transition.
- Use the single nearest baked view. Blending three views smeared detail.
- Fit the quad to the silhouette extent (`half`), not the bounding sphere. Tall trees otherwise
  waste most of their fill.
- The baked atlas needs supersampling (6-8x). Leaf alpha minifies below the 0.5 cutoff and the
  foliage vanishes without it.
- Atlases need coverage-preserving alpha mips.

**Hindsight: how to rebuild it.**
1. **Make the impostor the coarsest LOD of the prototype in the bake**, not a separate gate.
   If gltf-tools emitted the billboard as the root of a per-prototype DAG (a "root of roots"),
   the existing cut, root admission, budgets and streaming would handle the switch, the
   hysteresis and the page prefetch with no special shader path. That removes the separate
   draw, the list, the args kernel and the extra objectData flags.
2. **Tag prototypes in the asset.** Put a prototype id and primary flag in the glTF extension
   instead of matching on glTF mesh names at load. Name matching was fragile, and it forced the
   parser to record `meshName` and `firstOfMesh` on every primitive.
3. **Lighting.** The normal atlas held averaged leaf normals (all roughly "up"), which shaded
   flat and too bright. Bake a bent normal and a depth/thickness channel, or blend toward an
   ellipsoid crown normal, and bake ambient occlusion.
4. **Shadows.** Billboards cast none. Draw them in the shadow pass as alpha-tested quads facing
   the light, or keep the coarsest meshlet level for shadows only.
5. **Handoff quality.** Near the switch distance, mark the tree's meshlet pages as wanted, so
   swapping back to meshlets does not start from a coarse blob.
6. **Tooling.** The bake pipeline (prototype export, the browser bake app, KTX2 encode) belongs
   in gltf-tools, not in the engine's examples.

---

## `liteMaterials` (mobile lit variant)

**What it did.** It set `MESHLET_MATERIAL_LITE` on the lit bucket materials:
- Roughness and metalness came from the material factors only, so the ORM texture was not
  sampled.
- Streamed textures took one sample instead of blending fine and tail mips across the boundary
  band.

It measured about 19% less GPU time on a Quest 3 in the Bistro, where the lit fragment shader
dominates.

**Where it lived.**
- `liteMaterials` on `MeshletComponentSystem` and `MeshletWorld`.
- Two `#ifdef` blocks in `meshlet-lit-chunks-wgsl.js`.

**Hindsight.**
- Dropping ORM changes appearance. A cheaper path that keeps the look is to pack
  roughness/metalness where they are already fetched: the albedo alpha for opaque surfaces, or
  the spare channel of a two-channel normal map. This is a bake-side change.
- The fine/tail blend could be replaced by a dithered or temporal choice between the two
  samples. This keeps one fetch per pixel without a visible seam, and works on all platforms.
- A quality tier is better driven by the engine (a general material quality setting) than by a
  meshlet-only flag.

---

## Software texture anisotropy (`textureAnisotropy`)

**What it did.** With `textureAnisotropy` above 1 (`MESHLET_ANISOTROPY`), the lit shader
eigen-decomposed the UV footprint per pixel. It then took up to 2, 4 or 8 explicit-LOD samples
along the major axis, each clamped to the resident mip range.

**Where it lived.**
- `meshletTexFootprint` and the sample loop in `meshlet-lit-chunks-wgsl.js`.
- The `meshletTextureAnisotropy` uniform and the setters on the system and world.

**Hindsight.** Use hardware filtering with `textureSampleGrad` and an anisotropic sampler, so
the hardware filters the footprint in one sample instruction.
- **Residency clamp.** Sampling with gradients lets the hardware pick the mip, and samplers can
  only clamp the LOD for the whole texture, not per pixel. Clamp to the resident range by
  scaling `ddx`/`ddy` so the hardware never picks a finer level than is resident, with a margin
  of about a quarter mip for vendor rounding.
- **Start with the tail arrays.** They are always resident, so only the fixed per-texture
  `sizeBias` floor applies.
- **The fine/tail boundary** still needs two samples, because the hardware cannot filter across
  two textures.
- **Verify** with the texture-state debug view (`MESHLET_COLOR_MODE.TEXTURES`) at grazing
  angles, looking for bleed from non-resident levels.

---

## LOD-tier debug colour mode

**What it did.** `MESHLET_COLOR_MODE.LOD_TIER` (value 1) tinted each meshlet by
`meshletData.lodLevel` in the unlit debug material (`lodTint` in `meshlet-material.js`).

**Hindsight.** Raw DAG levels are not comparable across meshes: a level-3 cluster of a rock and
of a building mean very different errors. A more useful view is an **error-ratio heat map**:
each cluster's projected error divided by the threshold it was cut against, which shows where
the cut is at, below or above budget. It could be a define on the lit materials, as the
texture-state view is, rather than a separate unlit material. The remaining colour mode values
were kept stable, because examples pass them by number (`?color=2`, `?color=3`).

---

## `forceSubmitBoundaries`

**What it did.** On request, it split the cull chains into separate command buffer
submissions around every GPU-written indirect dispatch argument. Some Windows/D3D12 drivers
were seen to intermittently not see the arguments a compute pass wrote, when
`dispatchWorkgroupsIndirect` was recorded in the same command buffer. It mirrored a workaround
in `gsplat-hybrid-renderer.js`. Off by default, and plumbed through the director, the group,
the shadow renderer and every culler.

**Hindsight.**
- Confirm the bug with a minimal repro first. WebGPU's usage-scope rules make each dispatch its
  own scope, so this is a driver bug rather than an API-ordering issue.
- If it is real, apply the workaround once, centrally: in the device's `computeDispatch`, keyed
  off adapter info, so every compute user benefits. Per-feature flags that must be propagated to
  each culler are easy to miss: shadow views were broken until the flag was also passed to
  them.
