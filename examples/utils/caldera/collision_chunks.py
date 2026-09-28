"""
Merge every region's player collision into an island-wide grid of chunk GLBs, for streaming into a
physics engine near the player. A whole region as one physics mesh is millions of triangles (Jolt's
wasm heap runs out with a few loaded); a chunk is a few hundred thousand at most.

Writes <out>/collision_chunks/chunk_<ix>_<iz>.glb (world-space POSITION + indices, welded) and
<out>/collision_chunks/index.json: {size, chunks: [{file, min, max, triangles}]}.
"""
import glob
import json
import os
import struct
import sys
from collections import defaultdict

import numpy as np

CHUNK = 128.0   # metres
WELD = 1e-3     # metres; brushes are exported with their own vertices


def read_glb(path):
    with open(path, 'rb') as f:
        f.read(12)
        n, _ = struct.unpack('<II', f.read(8))
        gltf = json.loads(f.read(n))
        bl, _ = struct.unpack('<II', f.read(8))
        return gltf, f.read(bl)


def accessor(gltf, blob, i, comps, dtype):
    a = gltf['accessors'][i]
    v = gltf['bufferViews'][a['bufferView']]
    off = v.get('byteOffset', 0) + a.get('byteOffset', 0)
    return np.frombuffer(blob, dtype, a['count'] * comps, off).reshape(a['count'], comps) if comps > 1 else \
        np.frombuffer(blob, dtype, a['count'], off)


def world_triangles(path):
    """All triangles of a (flat, translation-only below the root) collision GLB in world space."""
    gltf, blob = read_glb(path)
    nodes = gltf['nodes']
    out = []

    def walk(i, m):
        node = nodes[i]
        local = np.array(node['matrix']).reshape(4, 4).T if 'matrix' in node else np.eye(4)
        m = m @ local
        if 'mesh' in node:
            for p in gltf['meshes'][node['mesh']]['primitives']:
                pos = accessor(gltf, blob, p['attributes']['POSITION'], 3, np.float32).astype(np.float64)
                idx = accessor(gltf, blob, p['indices'], 1, np.uint32).reshape(-1, 3)
                pos = pos @ m[:3, :3].T + m[:3, 3]
                out.append(pos[idx].astype(np.float32))
        for c in node.get('children', []):
            walk(c, m)

    for r in gltf['scenes'][0]['nodes']:
        walk(r, np.eye(4))
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def write_chunk(path, tris):
    flat = tris.reshape(-1, 3)
    key = np.round(flat / WELD).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    pos = flat[first].astype(np.float32)
    idx = inverse.reshape(-1).astype(np.uint32)
    # drop triangles the weld collapsed
    t = idx.reshape(-1, 3)
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    idx = t.reshape(-1)
    pb, ib = pos.tobytes(), idx.tobytes()
    gltf = {
        'asset': {'version': '2.0', 'generator': 'caldera collision_chunks.py'},
        'buffers': [{'byteLength': len(pb) + len(ib)}],
        'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': len(pb), 'target': 34962},
                        {'buffer': 0, 'byteOffset': len(pb), 'byteLength': len(ib), 'target': 34963}],
        'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': len(pos), 'type': 'VEC3',
                       'min': pos.min(0).tolist(), 'max': pos.max(0).tolist()},
                      {'bufferView': 1, 'componentType': 5125, 'count': len(idx), 'type': 'SCALAR'}],
        'meshes': [{'primitives': [{'attributes': {'POSITION': 0}, 'indices': 1, 'mode': 4}]}],
        'nodes': [{'name': os.path.basename(path)[:-4], 'mesh': 0}],
        'scenes': [{'nodes': [0]}], 'scene': 0
    }
    js = json.dumps(gltf, separators=(',', ':')).encode()
    js += b' ' * ((-len(js)) % 4)
    body = pb + ib
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(body)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack('<II', len(body), 0x004E4942))
        f.write(body)
    return len(idx) // 3, pos.min(0), pos.max(0)


def build(out_dir, sources):
    buckets = defaultdict(list)
    for src in sources:
        tris = world_triangles(src)
        if not len(tris):
            continue
        c = tris.mean(axis=1)
        keys = np.floor(c[:, [0, 2]] / CHUNK).astype(np.int64)
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        keys, tris = keys[order], tris[order]
        split = np.nonzero(np.any(np.diff(keys, axis=0) != 0, axis=1))[0] + 1
        for part_keys, part in zip(np.split(keys, split), np.split(tris, split)):
            buckets[tuple(part_keys[0])].append(part)
        print(f'  {os.path.basename(src)}: {len(tris) / 1e6:.2f}M tris', flush=True)
    chunk_dir = f'{out_dir}/collision_chunks'
    os.makedirs(chunk_dir, exist_ok=True)
    for old in glob.glob(f'{chunk_dir}/chunk_*.glb'):
        os.remove(old)
    chunks, total = [], 0
    for (ix, iz), parts in sorted(buckets.items()):
        name = f'chunk_{ix}_{iz}.glb'
        n, lo, hi = write_chunk(f'{chunk_dir}/{name}', np.concatenate(parts))
        total += n
        chunks.append({'file': name, 'min': lo.round(2).tolist(), 'max': hi.round(2).tolist(), 'triangles': n})
    with open(f'{chunk_dir}/index.json', 'w') as f:
        json.dump({'size': CHUNK, 'chunks': chunks}, f, indent=1)
    worst = max(c['triangles'] for c in chunks)
    print(f'  {len(chunks)} chunks, {total / 1e6:.1f}M tris, largest {worst / 1e3:.0f}k', flush=True)


if __name__ == '__main__':
    out = sys.argv[1]
    build(out, sorted(sys.argv[2:]))
