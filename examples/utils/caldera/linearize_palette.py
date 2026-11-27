"""Decode the sRGB palette colours of a baked meshlet GLB to the linear values glTF expects.

usd2glb.py's palette is picked by eye in sRGB, but bakes before the fix wrote it verbatim as
baseColorFactor, which is linear - so under physically scaled lighting everything rendered too
bright and washed out (sand 0.72 instead of 0.48). This patches a bake in place without
rebaking: each untextured material's baseColorFactor, and its record in the MAG_meshlets_gpu
material table the meshlet world uploads verbatim. Materials marked `extras.linearPalette` (new
bakes, or files already patched) are left alone, so re-running is a no-op.

  python linearize_palette.py caldera_docks.glb [more.glb ...]
"""
import sys

import numpy as np

from instance_nodes import read_glb, write_glb
from usd2glb import srgb_to_linear

GPU_EXT = 'MAG_meshlets_gpu'
RECORD_U32S = 32  # MATERIAL_RECORD_U32S; base colour rgba is words 0-3


def linearize(path):
    gltf, binary = read_glb(path)
    table = None
    gpu = gltf.get('extensions', {}).get(GPU_EXT)
    if gpu and 'materialTable' in gpu:
        acc = gltf['accessors'][gpu['materialTable']]
        view = gltf['bufferViews'][acc['bufferView']]
        start = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
        binary = bytearray(binary)
        table = np.frombuffer(binary, np.float32, gpu['materialCount'] * RECORD_U32S, start).reshape(-1, RECORD_U32S)
    patched = 0
    for i, mat in enumerate(gltf.get('materials', [])):
        pbr = mat.get('pbrMetallicRoughness', {})
        extras = mat.setdefault('extras', {})
        if extras.get('linearPalette') or 'baseColorTexture' in pbr or 'baseColorFactor' not in pbr:
            continue
        rgb = [srgb_to_linear(c) for c in pbr['baseColorFactor'][:3]]
        pbr['baseColorFactor'][:3] = rgb
        if table is not None and i < len(table):
            table[i, :3] = rgb
        extras['linearPalette'] = True
        patched += 1
    if patched:
        write_glb(path, gltf, bytes(binary))
    print(f'{path}: {patched} material(s) linearized')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        linearize(p)
