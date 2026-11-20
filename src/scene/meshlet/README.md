# src/scene/meshlet

This directory holds the GPU-driven meshlet renderer. It draws streamed `MAG_meshlets_gpu` / `MAG_meshlets_stream` assets (baked by gltf-tools) through compute-based DAG LOD selection, two-phase HZB occlusion and indirect draws, with geometry pages and textures streamed on demand.

It is WebGPU only. On WebGL2, `MeshletComponentSystem` warns once and meshlet components do not render. Everything here is internal (`@ignore`). The public entry point is `MeshletComponent` / `app.systems.meshlet` in `src/framework/components/meshlet/`.

`constants.js` defines the buffer layouts, and `shaders/meshlet-page-wgsl.js` mirrors them for the shaders. Change the two together.

- Architecture: [../../../docs/meshlet/architecture.md](../../../docs/meshlet/architecture.md)
- Removed features: [../../../docs/meshlet/removed-features.md](../../../docs/meshlet/removed-features.md)
