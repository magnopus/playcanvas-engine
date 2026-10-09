"""
Derive caldera_<region>_player_collision.glb from caldera_<region>_collision.glb: only the classes a
walking player collides with (player_clip, world). The binary is rebuilt so dropped classes cost
nothing to download or to cook into a physics mesh.
"""
import json
import os
import struct
import sys

KEEP = ('player_clip', 'world')


def read_glb(path):
    with open(path, 'rb') as f:
        f.read(12)
        n, _ = struct.unpack('<II', f.read(8))
        gltf = json.loads(f.read(n))
        bl, _ = struct.unpack('<II', f.read(8))
        return gltf, f.read(bl)


def write_player(src, dst):
    gltf, blob = read_glb(src)
    nodes = gltf['nodes']
    root = nodes[gltf['scenes'][0]['nodes'][0]]
    out = {'asset': gltf['asset'], 'scene': 0, 'nodes': [], 'meshes': [], 'accessors': [], 'bufferViews': []}
    bin_parts, offset = [], 0

    def copy_accessor(i):
        nonlocal offset
        acc = dict(gltf['accessors'][i])
        view = gltf['bufferViews'][acc['bufferView']]
        data = blob[view.get('byteOffset', 0):view.get('byteOffset', 0) + view['byteLength']]
        pad = (-offset) % 4
        bin_parts.append(b'\0' * pad + data)
        offset += pad
        out['bufferViews'].append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(data),
                                   **({'target': view['target']} if 'target' in view else {})})
        offset += len(data)
        acc['bufferView'] = len(out['bufferViews']) - 1
        out['accessors'].append(acc)
        return len(out['accessors']) - 1

    groups = []
    for ci in root.get('children', []):
        grp = nodes[ci]
        if grp['name'] not in KEEP:
            continue
        kids = []
        for k in grp.get('children', []):
            node = dict(nodes[k])
            if 'mesh' in node:
                mesh = gltf['meshes'][node['mesh']]
                prims = [{'attributes': {'POSITION': copy_accessor(p['attributes']['POSITION'])},
                          'indices': copy_accessor(p['indices']), 'mode': p.get('mode', 4)}
                         for p in mesh['primitives']]
                out['meshes'].append({'name': mesh.get('name'), 'primitives': prims})
                node['mesh'] = len(out['meshes']) - 1
            out['nodes'].append(node)
            kids.append(len(out['nodes']) - 1)
        out['nodes'].append({'name': grp['name'], 'children': kids})
        groups.append(len(out['nodes']) - 1)
    out['nodes'].append({'name': root.get('name', 'collision'), 'children': groups})
    out['scenes'] = [{'nodes': [len(out['nodes']) - 1]}]
    binary = b''.join(bin_parts)
    binary += b'\0' * ((-len(binary)) % 4)
    out['buffers'] = [{'byteLength': len(binary)}]
    js = json.dumps(out, separators=(',', ':')).encode()
    js += b' ' * ((-len(js)) % 4)
    with open(dst, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack('<II', len(binary), 0x004E4942))
        f.write(binary)


def player_collision(out_dir, name):
    """Writes (if stale) and returns the player collision file name, or None without a source."""
    src = f'{out_dir}/caldera_{name}_collision.glb'
    dst = f'{out_dir}/caldera_{name}_player_collision.glb'
    if not os.path.exists(src):
        return None
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        write_player(src, dst)
    return os.path.basename(dst)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        write_player(path, path.replace('_collision.glb', '_player_collision.glb'))
