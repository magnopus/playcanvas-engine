"""
Write jungle_scene.json, the list of baked groups the meshlet-jungle example loads, from the
jungle_<group>*.json sidecars usd2glb.py writes next to each export (pipeline.sh copies them into
the bake folder).

  python3 manifest.py [bake dir]
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../../assets/meshlets/jungle')
LABELS = {'terrain': 'Terrain', 'static': 'Pyramid and banyan', 'trees': 'Trees', 'plants': 'Plants'}
ORDER = ['terrain', 'static', 'trees', 'plants']

groups = []
for sidecar in sorted(glob.glob(os.path.join(OUT, 'jungle_*.json'))):
    if sidecar.endswith('jungle_scene.json'):
        continue
    glb = os.path.splitext(sidecar)[0] + '.glb'
    if not os.path.exists(glb):
        continue
    with open(sidecar) as f:
        info = json.load(f)
    name = os.path.basename(glb)[len('jungle_'):-len('.glb')]
    groups.append({'name': name, 'label': LABELS.get(name, name), 'url': os.path.basename(glb), **info})
groups.sort(key=lambda g: (ORDER.index(g['group']) if g['group'] in ORDER else len(ORDER), g['name']))

# opens looking over the pyramid (glTF metres: USD x, z = -USD y)
manifest = {'groups': groups, 'start': {'pos': [470, 45, -740], 'look': [544, 8, -810]}}
with open(os.path.join(OUT, 'jungle_scene.json'), 'w') as f:
    json.dump(manifest, f, indent=1)
print(f'{len(groups)} group(s) -> {os.path.join(OUT, "jungle_scene.json")}')
