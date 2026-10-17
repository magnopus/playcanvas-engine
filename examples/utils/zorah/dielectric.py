"""Make Zorah's untextured materials dielectric in the baked meshlet material tables.

Usage: python3 dielectric.py [bake_dir] [roughness]

The Zorah export is geometry only: most of its materials are `"pbrMetallicRoughness": {}`, which
glTF reads as metallic 1 / roughness 1 - a white metal mirroring a blurred sky, so every surface
renders washed out. This rewrites the METALLIC word of each such material's record in the root
MAG_meshlets_gpu material table to 0 (and ROUGHNESS to the given value), in place: the table is
fixed-size, so the GLB layout is untouched. Materials that set metallicFactor are left alone.
Idempotent. Inline (BLEND) primitives use the glTF JSON material and are not changed.
"""
import glob
import json
import os
import struct
import sys

ZORAH = os.path.expanduser('~/Downloads/zorah_main_public.v2.gltf')
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{ZORAH}/streamed_meshlets_v4'
ROUGHNESS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8

# src/scene/meshlet/constants.js MATERIAL_RECORD
RECORD_U32S = 32
METALLIC = 8
ROUGHNESS_WORD = 9

for path in sorted(glob.glob(f'{OUT}/*.streamed.glb')):
    with open(path, 'r+b') as f:
        f.seek(12)
        json_len, _ = struct.unpack('<II', f.read(8))
        gltf = json.loads(f.read(json_len))
        bin_start = 12 + 8 + json_len + 8
        root = gltf.get('extensions', {}).get('MAG_meshlets_gpu')
        if not root or 'materialTable' not in root:
            continue
        acc = gltf['accessors'][root['materialTable']]
        view = gltf['bufferViews'][acc['bufferView']]
        base = bin_start + view.get('byteOffset', 0) + acc.get('byteOffset', 0)
        patched = 0
        for i, material in enumerate(gltf['materials'][:root['materialCount']]):
            if 'metallicFactor' in material.get('pbrMetallicRoughness', {}):
                continue
            record = base + i * RECORD_U32S * 4
            f.seek(record + METALLIC * 4)
            f.write(struct.pack('<ff', 0.0, ROUGHNESS))
            patched += 1
        print(f'{os.path.basename(path)}: {patched}/{root["materialCount"]} materials made dielectric', file=sys.stderr)
