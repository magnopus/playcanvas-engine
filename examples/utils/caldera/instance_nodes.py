"""Collapse a baked meshlet GLB's per-instance nodes into EXT_mesh_gpu_instancing scatters.

The USD export writes one glTF node per placement - 1.68M across Caldera, 330 MB of the
containers' JSON - and the client parses all of it before anything can stream. The meshlet loader
(MAG_meshlets) already reads EXT_mesh_gpu_instancing, so the same placements can ride as binary
TRS accessors on one node per (parent, mesh) instead.

Only plain leaf nodes are folded: a mesh, a TRS (or nothing), no children, extensions, camera,
skin or matrix. Accessors and bufferViews are only appended, so every existing index - including
the MAG_meshlets ones - stays valid. Runs in place; re-running on a converted file is a no-op.

  python instance_nodes.py caldera_docks.glb [more.glb ...] [--dry-run]
"""
import json
import os
import struct
import sys

import numpy as np

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
EXT = 'EXT_mesh_gpu_instancing'
FLOAT = 5126
FOLDABLE_KEYS = {'mesh', 'name', 'translation', 'rotation', 'scale'}


def read_glb(path):
    with open(path, 'rb') as f:
        data = f.read()
    magic, version, _ = struct.unpack_from('<III', data, 0)
    assert magic == GLB_MAGIC and version == 2, f'{path}: not a glTF 2 binary'
    json_len, json_type = struct.unpack_from('<II', data, 12)
    assert json_type == CHUNK_JSON
    gltf = json.loads(data[20:20 + json_len])
    offset = 20 + json_len
    binary = b''
    if offset < len(data):
        bin_len, bin_type = struct.unpack_from('<II', data, offset)
        assert bin_type == CHUNK_BIN
        binary = data[offset + 8:offset + 8 + bin_len]
    return gltf, binary


def write_glb(path, gltf, binary):
    text = json.dumps(gltf, separators=(',', ':')).encode()
    text += b' ' * (-len(text) % 4)
    binary += b'\0' * (-len(binary) % 4)
    total = 12 + 8 + len(text) + (8 + len(binary) if binary else 0)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(struct.pack('<III', GLB_MAGIC, 2, total))
        f.write(struct.pack('<II', len(text), CHUNK_JSON))
        f.write(text)
        if binary:
            f.write(struct.pack('<II', len(binary), CHUNK_BIN))
            f.write(binary)
    os.replace(tmp, path)


def foldable(node):
    return 'mesh' in node and set(node) <= FOLDABLE_KEYS


def convert(gltf, binary):
    """Returns the new (gltf, binary, folded), or None when there is nothing to fold. `folded` pairs
    each new node's scatter accessors with the source TRS it was built from, for checking."""
    nodes = gltf.get('nodes', [])
    if gltf.get('animations') or gltf.get('skins'):
        return None  # these address nodes by index; leave such files alone

    parent = {}
    for i, node in enumerate(nodes):
        for child in node.get('children', []):
            parent[child] = i

    groups = {}
    for i, node in enumerate(nodes):
        if foldable(node):
            groups.setdefault((parent.get(i), node['mesh']), []).append(i)
    groups = {key: members for key, members in groups.items() if len(members) > 1}
    if not groups:
        return None

    # the scatter data, appended as three shared bufferViews (translation, rotation, scale)
    columns = {'TRANSLATION': [], 'ROTATION': [], 'SCALE': []}
    accessors = gltf.setdefault('accessors', [])
    views = gltf.setdefault('bufferViews', [])
    pending = []  # (group key, {attribute: (column, first row, count)})
    for key, members in groups.items():
        t = np.array([nodes[i].get('translation', [0, 0, 0]) for i in members], dtype=np.float64)
        r = np.array([nodes[i].get('rotation', [0, 0, 0, 1]) for i in members], dtype=np.float64)
        s = np.array([nodes[i].get('scale', [1, 1, 1]) for i in members], dtype=np.float64)
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        attrs = {}
        # attributes at their default for every member are omitted, as the extension allows
        for name, values, default in (('TRANSLATION', t, [0, 0, 0]), ('ROTATION', r, [0, 0, 0, 1]), ('SCALE', s, [1, 1, 1])):
            # translation is always kept: it carries the placement count even when every member
            # sits at the origin
            if name != 'TRANSLATION' and np.all(values == default):
                continue
            rows = sum(len(c) for c in columns[name])
            columns[name].append(values.astype(np.float32))
            attrs[name] = (rows, len(members))
        pending.append((key, attrs, (t, r, s)))

    view_index = {}
    blob = bytearray(binary)
    for name, chunks in columns.items():
        if not chunks:
            continue
        data = np.concatenate(chunks).tobytes()
        blob += b'\0' * (-len(blob) % 4)
        view_index[name] = len(views)
        views.append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': len(data)})
        blob += data

    # one node per group, at its first member's position; the other members are dropped
    replacement = {}
    dropped = set()
    folded = []
    for (key, attrs, source), members in zip(pending, groups.values()):
        width = {'TRANSLATION': 3, 'ROTATION': 4, 'SCALE': 3}
        ext_attrs = {}
        for name, (row, count) in attrs.items():
            ext_attrs[name] = len(accessors)
            accessors.append({'bufferView': view_index[name], 'byteOffset': row * width[name] * 4,
                              'componentType': FLOAT, 'count': count, 'type': 'VEC4' if width[name] == 4 else 'VEC3'})
        mesh = key[1]
        name = gltf['meshes'][mesh].get('name', f'mesh{mesh}')
        replacement[members[0]] = {'name': f'{name} x{len(members)}', 'mesh': mesh,
                                   'extensions': {EXT: {'attributes': ext_attrs}}}
        folded.append((ext_attrs, source))
        dropped.update(members[1:])

    remap = {}
    out_nodes = []
    for i, node in enumerate(nodes):
        if i in dropped:
            continue
        remap[i] = len(out_nodes)
        out_nodes.append(replacement.get(i, node))
    for node in out_nodes:
        if 'children' in node:
            node['children'] = [remap[c] for c in node['children'] if c in remap]
    for scene in gltf.get('scenes', []):
        scene['nodes'] = [remap[n] for n in scene.get('nodes', []) if n in remap]
    gltf['nodes'] = out_nodes

    gltf['buffers'][0]['byteLength'] = len(blob) + (-len(blob) % 4)
    used = gltf.setdefault('extensionsUsed', [])
    if EXT not in used:
        used.append(EXT)
    return gltf, bytes(blob), folded


def read_accessor(gltf, binary, index, width):
    a = gltf['accessors'][index]
    v = gltf['bufferViews'][a['bufferView']]
    start = v.get('byteOffset', 0) + a.get('byteOffset', 0)
    return np.frombuffer(binary, dtype=np.float32, count=a['count'] * width, offset=start).reshape(-1, width)


def placement_counts(gltf):
    """Placements per mesh, as the meshlet loader counts them (a scatter places one per entry)."""
    counts = {}
    for node in gltf['nodes']:
        if 'mesh' not in node:
            continue
        scatter = (node.get('extensions') or {}).get(EXT, {}).get('attributes')
        n = max(gltf['accessors'][v]['count'] for v in scatter.values()) if scatter else 1
        counts[node['mesh']] = counts.get(node['mesh'], 0) + n
    return counts


def check(gltf, binary, folded, counts_before):
    """Every scatter entry equals its source node's TRS (to float32), and no placement is lost."""
    assert placement_counts(gltf) == counts_before, 'placement counts per mesh changed'
    worst = 0.0
    for attrs, (t, r, s) in folded:
        for name, width, source, default in (('TRANSLATION', 3, t, None), ('ROTATION', 4, r, [0, 0, 0, 1]), ('SCALE', 3, s, [1, 1, 1])):
            if name in attrs:
                got = read_accessor(gltf, binary, attrs[name], width)
                # relative to magnitude: float32 keeps ~7 significant digits of a km-scale position
                worst = max(worst, float(np.max(np.abs(got - source) / np.maximum(1, np.abs(source)))))
            else:
                assert np.all(source == default), f'{name} dropped but not at its default'
    return sum(counts_before.values()), worst


def main(argv):
    dry = '--dry-run' in argv
    paths = [p for p in argv if not p.startswith('--')]
    for path in paths:
        size = os.path.getsize(path)
        gltf, binary = read_glb(path)
        node_count = len(gltf.get('nodes', []))
        counts = placement_counts(gltf)
        result = convert(gltf, binary)
        if result is None:
            print(f'{os.path.basename(path)}: nothing to fold')
            continue
        gltf, binary, folded = result
        count, worst = check(gltf, binary, folded, counts)
        assert worst < 1e-6, f'{path}: scatter differs from its source by {worst:.2e}'
        text = len(json.dumps(gltf, separators=(',', ':')))
        print(f'{os.path.basename(path)}: {node_count} -> {len(gltf["nodes"])} nodes, {count} placements kept '
              f'(max relative TRS error {worst:.1e}); {size / 1048576:.1f} MB -> ~{(text + len(binary) + 28) / 1048576:.1f} MB')
        if not dry:
            write_glb(path, gltf, binary)


if __name__ == '__main__':
    main(sys.argv[1:])
