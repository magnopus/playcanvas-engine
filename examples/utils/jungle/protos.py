"""
Exports every tree prototype of the JungleRuins scene as a small standalone GLB for the impostor
baker (meshlet-impostor-bake example): one mesh at the same frame the scene export uses (origin at
the bounding box centre, metres, Y-up), no scatter, only the materials it uses.

Writes <out>/protos/<name>.glb and <out>/protos/protos.json:
  [{ name, file, center: [0,0,0], radius, half: [x,y,z] }]
radius is the bounding sphere about the origin and half the AABB half extents, both in prototype
units; the instances' own scale multiplies them. extent is the largest half width and half height
the silhouette reaches over every baked view (hemi-octahedral grid, --grid per side), so the bake can
frame a tight tile instead of the whole sphere. The runtime billboard uses the extents the bake
actually framed (they are copied into the atlas manifest).

--test-scene also writes <out>/jungle_treetest.glb for the meshlet-impostor-test example: every
prototype twice, side by side on a row - once under its own name (so the impostor set tags it) and
once as '<name>__real' (never tagged, always meshlets), at the scale the scatter instances use.
"""
import argparse
import json
import os
import sys

import numpy as np
from pxr import Usd, UsdGeom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import usd2glb as U  # noqa: E402


def silhouette_extent(points, grid):
    """Largest |x| and |y| of the points in the view plane of any frame of the hemi-octahedral
    grid, with the bake's basis (right = up0 x dir, up = dir x right, up0 = +Y)."""
    ex = ey = 0.0
    for j in range(grid):
        for i in range(grid):
            gx, gy = i / (grid - 1), j / (grid - 1)
            x, z = gx - gy, -1 + gx + gy
            f = np.array([x, 1 - abs(x) - abs(z), z])
            f /= np.linalg.norm(f)
            right = np.cross([0.0, 1.0, 0.0], f)
            right /= np.linalg.norm(right)
            up = np.cross(f, right)
            ex = max(ex, float(np.abs(points @ right).max()))
            ey = max(ey, float(np.abs(points @ up).max()))
    return ex, ey


def write_test_scene(path, mats, tests, manifest, scale):
    """Every prototype twice in a row, 'real' behind its impostor-tagged twin, standing on y = 0."""
    info = {m['name']: m for m in manifest}
    w = U.GlbWriter(path)
    w.materials(mats.defs)
    children = []
    x = 0.0
    for name, prims, _ in sorted(tests, key=lambda t: t[0]):
        m = info[name]
        gap = m['radius'] * scale * 1.25
        # the mesh origin is the bounding box centre: lift by the half height so the tree stands on the ground
        lift = m['half'][1] * scale
        for suffix, dx in (('', gap), ('__real', -gap)):
            mi = w.mesh(name + suffix, prims)
            matrix = np.eye(4)
            matrix[:3, :3] *= scale
            matrix[:3, 3] = [x + dx, lift, 0.0]
            children.append(w.node(name + suffix, mi, matrix))
        x += 4 * gap
    w.node('jungle_treetest', children=children, root=True)
    size = w.close()
    print(f'test scene: {path} {size / 2**20:.0f} MiB, {len(tests)} prototypes', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--tex-size', type=int, default=2048)
    ap.add_argument('--grid', type=int, default=8, help='views per side of the impostor atlas')
    ap.add_argument('--test-scene', action='store_true', help='also write jungle_treetest.glb')
    ap.add_argument('--instance-scale', type=float, default=170.0, help='scale of the test scene instances')
    args = ap.parse_args()
    out = os.path.join(args.out, 'protos')
    os.makedirs(out, exist_ok=True)

    stage = Usd.Stage.Open(U.SCENE)
    cache = U.TextureCache(os.path.join(args.out, 'textures'), args.tex_size)
    mats = U.Materials(cache)
    manifest = []
    tests = []  # (name, prims with scene material indices, half extents) for --test-scene
    seen = set()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.PointInstancer):
            continue
        for target in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets():
            name = target.name
            if name in seen or U.group_of_proto(name) != 'trees':
                continue
            seen.add(name)
            proto = stage.GetPrimAtPath(target)
            proto_world = U.gf_to_np(UsdGeom.Xformable(proto).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            prims = []
            for d in Usd.PrimRange(proto, Usd.TraverseInstanceProxies()):
                if d.IsA(UsdGeom.Mesh):
                    rel = np.linalg.inv(proto_world) @ U.gf_to_np(UsdGeom.Xformable(d).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                    prims += U.mesh_primitives(d, U.CONV @ rel, mats.key)
            if not prims:
                continue
            prims, _ = U.recentre(prims)

            tests.append((name, prims, None))
            # only the materials this prototype uses, so the GLB embeds only its own textures
            used = sorted({p[0] for p in prims if p[0] is not None})
            remap = {m: i for i, m in enumerate(used)}
            prims = [(remap.get(m), pos, nor, idx, uv) for m, pos, nor, idx, uv in prims]
            allpos = np.concatenate([p[1] for p in prims])
            lo, hi = allpos.min(axis=0), allpos.max(axis=0)
            centre = (lo + hi) * 0.5
            radius = float(np.linalg.norm(allpos - centre, axis=1).max())
            extent = silhouette_extent(allpos - centre, args.grid)

            path = os.path.join(out, f'{name}.glb')
            w = U.GlbWriter(path)
            w.materials([mats.defs[m] for m in used])
            mi = w.mesh(name, prims)
            w.node(name, mi, root=True)
            w.close()
            manifest.append({'name': name, 'file': f'{name}.glb', 'center': [round(float(c), 4) for c in centre],
                             'radius': round(radius, 4), 'half': [round(float(h), 4) for h in (hi - lo) * 0.5],
                             'extent': [round(extent[0], 4), round(extent[1], 4)]})
            print(f'  {name}: radius {radius:.2f} m, half {manifest[-1]["half"]}, {len(used)} materials', flush=True)
    manifest.sort(key=lambda p: p['name'])
    with open(os.path.join(out, 'protos.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    if args.test_scene:
        write_test_scene(os.path.join(args.out, 'jungle_treetest.glb'), mats, tests, manifest, args.instance_scale)
    print(f'{len(manifest)} prototypes -> {out}', flush=True)


if __name__ == '__main__':
    main()
