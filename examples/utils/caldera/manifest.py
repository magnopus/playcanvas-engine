"""Write caldera_scene.json (region bounds, file names) from the raw GLBs next to the baked assets."""
import json
import os
import struct
import sys

import numpy as np

from player_collision import player_collision

RAW = os.environ.get('CALDERA_RAW', 'caldera-raw')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets/meshlets/caldera')
# hotel and power_station are prefabs inside tile_p / tile_n; once those districts are baked the
# prefab bakes would draw a second copy, so they are listed only while their district is missing
ORDER = ['terrain', 'capital', 'airfield', 'phosphate_mine', 'beachhead', 'tile_n', 'tile_p', 'arsenal',
         'tile_f', 'docks', 'caldera', 'subpen', 'hotel', 'power_station']
CONTAINED = {'hotel': 'tile_p', 'power_station': 'tile_n'}
# full-terrain tiles; the simple terrain is replaced once every tile is baked
TILES = [f'st_{t}' for t in 'bcdefghijklnop']
ORDER = ORDER + TILES
LABELS = {'terrain': 'Terrain (simple)', 'capital': 'Capital (incl. restaurant)', 'airfield': 'Airfield',
          'phosphate_mine': 'Phosphate Mine', 'beachhead': 'Beachhead', 'hotel': 'Hotel',
          'power_station': 'Power Station', 'tile_n': 'Tile N (incl. power station)',
          'tile_p': 'Tile P (incl. hotel)', 'arsenal': 'Arsenal', 'tile_f': 'Tile F', 'docks': 'Docks',
          'caldera': 'Caldera', 'subpen': 'Sub Pen',
          **{t: f'Terrain {t[3:].upper()} (full)' for t in TILES}}


def read_json(path):
    with open(path, 'rb') as f:
        magic, _, _ = struct.unpack('<III', f.read(12))
        assert magic == 0x46546C67
        n, _ = struct.unpack('<II', f.read(8))
        return json.loads(f.read(n))


def bounds(gltf):
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


regions = []
for name in ORDER:
    # bounds from the raw GLB, or its collision soup once the pipeline has deleted the raw
    # (the baked GLB only carries placeholder positions)
    raw = f'{RAW}/caldera_{name}.glb'
    if not os.path.exists(raw):
        raw = f'{RAW}/caldera_{name}_collision.glb'
    if not os.path.exists(raw):
        continue
    if name in CONTAINED and os.path.exists(f'{OUT}/caldera_{CONTAINED[name]}.glb'):
        continue
    if name == 'terrain' and all(os.path.exists(f'{OUT}/caldera_{t}.glb') for t in TILES):
        continue
    baked = f'{OUT}/caldera_{name}.glb'
    lo, hi = bounds(read_json(raw))
    entry = {'name': name, 'label': LABELS[name], 'url': f'caldera_{name}.glb',
             'baked': os.path.exists(baked), 'min': lo.round(2).tolist(), 'max': hi.round(2).tolist()}
    if os.path.exists(f'{OUT}/caldera_{name}_collision.glb'):
        entry['collision'] = f'caldera_{name}_collision.glb'
        entry['playerCollision'] = player_collision(OUT, name)
    regions.append(entry)
    print(entry, file=sys.stderr)

with open(f'{OUT}/caldera_scene.json', 'w') as f:
    json.dump({'regions': regions}, f, indent=2)

# the same regions as a magnopus-web-renderer render-test scene description (`meshlets` scene,
# `?scene=caldera/scene.json`), with the caldera directory linked into public/meshlets/caldera
identity = {'position': [0, 0, 0], 'rotation': [0, 0, 0, 1], 'scale': [1, 1, 1]}
# Positions are glTF Y-up metres, as the example uses them. Entities also carry their player
# collision and world bounds, which the scene streams in by proximity; `teleports` frame the regions.
entities = []
for n, r in enumerate(x for x in regions if x['baked']):
    e = {'id': n + 1, 'parentId': None, 'name': r['label'], **identity,
         'models': [{'name': r['name'], 'url': f"caldera/{r['url']}", **identity,
                     'visible': True, 'shadowCaster': True}],
         'bounds': {'min': r['min'], 'max': r['max']}}
    entities.append(e)
# region centres, dropped a little above the highest point so the player lands on top
teleports = [{'name': r['label'], 'position': [round((r['min'][0] + r['max'][0]) / 2, 2), round(r['max'][1] + 2, 2),
                                                 round((r['min'][2] + r['max'][2]) / 2, 2)]}
             for r in regions if r['baked'] and not r['name'].startswith('st_')]
scene = {'version': 1, 'source': {'generator': 'caldera manifest.py'},
         'start': {'position': [206.73, 232.20, -408.06], 'rotation': [0, 0, 0, 1]},
         'teleports': teleports, 'farClip': 6000,
         # ~1.2M instances: the meshlet world's fixed tables alone are ~1.4 GB, past MWR's 768 MB default
         'meshletPoolMiB': 4096, 'entities': entities}
# island-wide player collision, chunked for streaming (collision_chunks.py)
if os.path.exists(f'{OUT}/collision_chunks/index.json'):
    scene['collisionChunks'] = 'caldera/collision_chunks/index.json'
with open(f'{OUT}/scene.json', 'w') as f:
    json.dump(scene, f, indent=1)
