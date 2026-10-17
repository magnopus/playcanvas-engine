"""Write zorah/scene.json (a magnopus-web-renderer render-test scene description, like caldera's) for
every baked Zorah chunk.

Usage: python3 manifest.py [chunk_dir] [bake_dir]
    chunk_dir  the split source chunks (tools/split-meshopt-chunks.js), for world bounds
    bake_dir   the streamed-meshlet bakes, chunk_NNN.streamed.glb; scene.json is written here

The render-test meshlets scene loads it with `?scene=zorah/scene.json` once bake_dir is linked into
its public/meshlets/zorah. Rerun after more chunks are baked.

The meshlet world keeps a 128-byte record per meshlet resident in one buffer: the 19 chunks are
~46M meshlets (~6 GB), past what the browser allocates and WebGPU binds. scene.json therefore holds
as many chunks as fit ZORAH_MESHLET_BUDGET (default 10M meshlets, ~1.3 GB; Caldera is ~6M), the
mountain backdrop first; scene.all.json lists every baked chunk.
"""
import glob
import json
import os
import struct
import sys

import numpy as np

ZORAH = os.path.expanduser('~/Downloads/zorah_main_public.v2.gltf')
CHUNKS = sys.argv[1] if len(sys.argv) > 1 else f'{ZORAH}/mesh_chunks'
OUT = sys.argv[2] if len(sys.argv) > 2 else f'{ZORAH}/streamed_meshlets_v4'
# path of OUT under the render-test's /meshlets/
PREFIX = 'zorah'
BUDGET = int(float(os.environ.get('ZORAH_MESHLET_BUDGET', 10e6)))


def meshlet_count(glb):
    with open(glb, 'rb') as f:
        f.read(12)
        n, _ = struct.unpack('<II', f.read(8))
        gltf = json.loads(f.read(n))
    return sum(p['extensions']['MAG_meshlets_gpu']['meshletCount']
               for m in gltf['meshes'] for p in m['primitives'] if 'MAG_meshlets_gpu' in p.get('extensions', {}))


def bounds(gltf):
    """World bounds of every mesh node. Zorah's nodes are flat (no hierarchy), each with a matrix."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    acc = gltf['accessors']
    for node in gltf['nodes']:
        if 'mesh' not in node:
            continue
        m = np.array(node.get('matrix', np.eye(4).reshape(-1).tolist())).reshape(4, 4).T
        for p in gltf['meshes'][node['mesh']]['primitives']:
            a = acc[p['attributes']['POSITION']]
            mn, mx = np.array(a['min']), np.array(a['max'])
            corners = np.array([[x, y, z, 1] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
            w = corners @ m.T
            lo = np.minimum(lo, w[:, :3].min(axis=0))
            hi = np.maximum(hi, w[:, :3].max(axis=0))
    return lo, hi


identity = {'position': [0, 0, 0], 'rotation': [0, 0, 0, 1], 'scale': [1, 1, 1]}
entities = []
lo_all = np.full(3, np.inf)
hi_all = np.full(3, -np.inf)
for path in sorted(glob.glob(f'{CHUNKS}/chunk_*.gltf')):
    name = os.path.basename(path)[:-len('.gltf')]
    baked = f'{name}.streamed.glb'
    if not os.path.exists(f'{OUT}/{baked}'):
        continue
    with open(path) as f:
        lo, hi = bounds(json.load(f))
    lo_all = np.minimum(lo_all, lo)
    hi_all = np.maximum(hi_all, hi)
    entities.append({'id': len(entities) + 1, 'parentId': None, 'name': name, **identity,
                     'models': [{'name': name, 'url': f'{PREFIX}/{baked}', **identity, 'visible': True, 'shadowCaster': True}],
                     'bounds': {'min': lo.round(2).tolist(), 'max': hi.round(2).tolist()},
                     'meshlets': meshlet_count(f'{OUT}/{baked}')})
    print(f"{name}: {entities[-1]['meshlets'] / 1e6:.2f}M meshlets, {lo.round(1).tolist()} .. {hi.round(1).tolist()}", file=sys.stderr)

if not entities:
    sys.exit(f'no baked chunks in {OUT}')

# The town is ~170 m across; the chunk holding the mountain backdrop spans ~60 km. Place the
# start and teleports from the town's chunks only, and size the far clip to the whole thing.
TOWN_MAX_EXTENT = 1000
lo_town = np.full(3, np.inf)
hi_town = np.full(3, -np.inf)
for e in entities:
    lo, hi = np.array(e['bounds']['min']), np.array(e['bounds']['max'])
    if (hi - lo).max() < TOWN_MAX_EXTENT:
        lo_town = np.minimum(lo_town, lo)
        hi_town = np.maximum(hi_town, hi)
if not np.isfinite(lo_town).all():
    lo_town, hi_town = lo_all, hi_all

# Chunks are mesh-index ranges, not regions, so they overlap spatially: teleport across the town
# rather than per chunk. Teleports are dropped at street level (the town's floor is y ~ 0).
y = round(float(lo_town[1]) + 2, 2)
teleports = []
for label, fx, fz in [('Centre', 0.5, 0.5), ('North-west', 0.2, 0.2), ('North-east', 0.8, 0.2),
                      ('South-west', 0.2, 0.8), ('South-east', 0.8, 0.8)]:
    x = lo_town[0] + (hi_town[0] - lo_town[0]) * fx
    z = lo_town[2] + (hi_town[2] - lo_town[2]) * fz
    teleports.append({'name': label, 'position': [round(float(x), 2), y, round(float(z), 2)]})
far = float(np.abs(np.concatenate([lo_all, hi_all])).max())

# `start` is absolute world space; teleports are relative to the render-test's field root,
# which sits at z = 10 (FIELD_Z in render-test/app/scenes/meshlets.ts)
FIELD_Z = 10
sx, sy, sz = teleports[0]['position']


def write_scene(file, chosen):
    scene = {'version': 1, 'source': {'generator': 'zorah manifest.py'},
             'start': {'position': [sx, sy, round(sz + FIELD_Z, 2)], 'rotation': [0, 0, 0, 1]},
             'teleports': teleports,
             # far enough to see the mountain backdrop from the town
             'farClip': max(1000, round(far * 1.5)),
             # the budget covers each world's fixed tables (~200 bytes a meshlet: records plus LOD-cut
             # data) as well as its pages, so it grows with the content; 2 GiB on top for pages
             'meshletPoolMiB': max(4096, round(sum(e['meshlets'] for e in chosen) * 200 / 2**20) + 2048),
             'entities': [{k: v for k, v in e.items() if k != 'meshlets'} | {'id': i + 1} for i, e in enumerate(chosen)]}
    with open(f'{OUT}/{file}', 'w') as f:
        json.dump(scene, f, indent=1)
    total = sum(e['meshlets'] for e in chosen)
    print(f"{file}: {', '.join(e['name'] for e in chosen)} ({total / 1e6:.1f}M meshlets)", file=sys.stderr)


def is_backdrop(e):
    return (np.array(e['bounds']['max']) - np.array(e['bounds']['min'])).max() >= TOWN_MAX_EXTENT


budgeted = []
used = 0
for e in sorted(entities, key=lambda e: not is_backdrop(e)):
    if used + e['meshlets'] <= BUDGET:
        budgeted.append(e)
        used += e['meshlets']
write_scene(os.environ.get('ZORAH_SCENE_FILE', 'scene.json'), sorted(budgeted, key=lambda e: e['name']))
write_scene('scene.all.json', entities)
