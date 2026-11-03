"""
Bake the JungleRuins tree canopy into top-down maps the meshlet-jungle example shades with. Real
time shadows cannot afford the forest (each shadow view needs its own culling buffers sized by the
instance count), but the forest is a heightfield world: what darkens the floor is leaf area
overhead, which a 2D map of leaf density plus its vertical extent captures.

Per texel: optical depth (one-sided leaf area per m^2 of ground, scaled by the alpha coverage of
the leaf card textures) and the canopy slab [bottom, top] in world metres. The shader marches the
sun ray through the slabs (Beer-Lambert) and dims the sky light by the canopy above.

  python3 canopy.py --out <bake dir> [--forest-radius 750]

Writes jungle_canopy.bin - RGBA float16 mip chains of (depth, depth * bottom, depth * top, 0), the
heights premultiplied so bilinear filtering and mips stay correct - and jungle_canopy.json.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import usd2glb as U  # noqa: E402

# chunk of instance x leaf-point products transformed at once
CHUNK_POINTS = 24_000_000
MAX_DEPTH = 30.0


def alpha_coverage(path, channel, threshold):
    """Fraction of a leaf card texture that survives the alpha test."""
    try:
        im = Image.open(path)
        im.thumbnail((256, 256))
        bands = im.getbands()
        want = {'r': 'R', 'g': 'G', 'b': 'B', 'a': 'A'}.get(channel, bands[-1])
        band = im.getchannel(want if want in bands else bands[-1])
        return float((np.asarray(band, np.float32) / 255.0 >= threshold).mean())
    except OSError:
        return 0.5


class LeafMaterials:
    """material_of for usd2glb.mesh_primitives: alpha-tested materials are leaves, valued by coverage."""

    def __init__(self):
        self.cache = {}

    def __call__(self, material):
        if not material:
            return 0.0
        name = material.GetPrim().GetName()
        if name not in self.cache:
            cov = 0.0
            sh = U.surface_shader(material)
            threshold = float(U.input_value(sh, 'opacityThreshold', 0.0)) if sh else 0.0
            if threshold > 0:
                tex = U.connected_texture(sh.GetInput('opacity'))
                cov = alpha_coverage(tex[0], tex[1], threshold) if tex else 1.0
                print(f'  leaf material {name}: coverage {cov:.2f}', flush=True)
            self.cache[name] = cov
        return self.cache[name]


def leaf_points(root, root_world, leaf, voxel):
    """Leaf area of the meshes under root, binned into voxels: (centres (n, 3), area (n,)), in the
    root's local space (USD cm, cm^2)."""
    inv = np.linalg.inv(root_world)
    cs, ws = [], []
    for d in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not d.IsA(UsdGeom.Mesh):
            continue
        rel = inv @ U.gf_to_np(UsdGeom.Xformable(d).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        for cov, pos, _nor, idx, _uv in U.mesh_primitives(d, rel, leaf):
            if cov <= 0:
                continue
            t = pos.astype(np.float64)[idx.reshape(-1, 3)]
            area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
            cs.append(t.mean(axis=1))
            ws.append(area * cov)
    if not cs:
        return None
    c = np.concatenate(cs)
    w = np.concatenate(ws)
    key = np.floor(c / voxel).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    n = inverse.max() + 1
    wsum = np.bincount(inverse, w, n)
    centre = np.stack([np.bincount(inverse, w * c[:, k], n) for k in range(3)], axis=1) / np.maximum(wsum, 1e-12)[:, None]
    keep = wsum > 0
    return centre[keep], wsum[keep]


class CanopyMap:
    """Leaf area projected along `axis` (glTF, unit) onto the plane spanned by u, v: per texel the
    sums of w, w d, w d^2, where d is the leaf's distance along the axis."""

    def __init__(self, name, axis, origin, span, size):
        axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
        ref = np.array([0.0, 0.0, 1.0]) if abs(axis[1]) > 0.99 else np.array([0.0, 1.0, 0.0])
        u = np.cross(ref, axis)
        u /= np.linalg.norm(u)
        self.basis = np.stack([u, np.cross(axis, u), axis])
        self.name, self.origin, self.span, self.size = name, origin, span, size
        self.cell = span / size
        self.acc = np.zeros((3, size * size))

    def project(self, p):
        return p @ self.basis.T

    def add(self, p, w):
        q = self.project(p)
        i = np.floor((q[:, 0] - self.origin[0]) / self.cell).astype(np.int64)
        j = np.floor((q[:, 1] - self.origin[1]) / self.cell).astype(np.int64)
        ok = (i >= 0) & (i < self.size) & (j >= 0) & (j < self.size)
        flat = j[ok] * self.size + i[ok]
        w, d = w[ok], q[ok, 2]
        n = self.size * self.size
        self.acc[0] += np.bincount(flat, w, n)
        self.acc[1] += np.bincount(flat, w * d, n)
        self.acc[2] += np.bincount(flat, w * d * d, n)

    def texels(self, blur):
        acc = self.acc.reshape(3, self.size, self.size)
        for _ in range(blur):
            # [1 2 1] / 4 both ways: hides the point splatting
            acc = (np.roll(acc, 1, 1) + 2 * acc + np.roll(acc, -1, 1)) * 0.25
            acc = (np.roll(acc, 1, 2) + 2 * acc + np.roll(acc, -1, 2)) * 0.25
        w = acc[0]
        safe = np.maximum(w, 1e-9)
        mean = acc[1] / safe
        std = np.sqrt(np.maximum(acc[2] / safe - mean * mean, 0.0))
        # a uniform slab with this mean and spread; at least 1 m thick
        half = np.maximum(math.sqrt(3.0) * std, 0.5)
        depth = np.minimum(w / (self.cell * self.cell), MAX_DEPTH)
        out = np.zeros((self.size, self.size, 4), np.float32)
        out[..., 0] = depth
        out[..., 1] = depth * (mean - half)
        out[..., 2] = depth * (mean + half)
        return out

    def levels(self, blur):
        lv = [self.texels(blur)]
        while lv[-1].shape[0] > 1:
            a = lv[-1]
            lv.append(0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]))
        return lv

    def meta(self):
        return {'name': self.name, 'u': self.basis[0].tolist(), 'v': self.basis[1].tolist(),
                'axis': self.basis[2].tolist(), 'origin': list(self.origin), 'span': self.span,
                'size': self.size}


def fit(name, axis, pts, size, min_cell=0.0):
    """A square map along axis covering the projection of pts (n, 3 glTF)."""
    m = CanopyMap(name, axis, (0.0, 0.0), 1.0, size)
    q = m.project(pts)[:, :2]
    lo, hi = q.min(axis=0), q.max(axis=0)
    span = max(float(max(hi - lo)), min_cell * size)
    centre = (lo + hi) * 0.5
    return CanopyMap(name, axis, tuple((centre - span * 0.5).tolist()), span, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--forest-radius', type=float,
                    help='as usd2glb.py: RiverForest / QueenForest trees kept within this radius of the core centre')
    ap.add_argument('--sun', type=lambda v: [float(c) for c in v.split(',')],
                    help='direction to the sun, glTF x,y,z; defaults to toSun from <out>/jungle_sky.json '
                         '(sky.py), else the example\'s fallback light at euler 45, 30, 0')
    ap.add_argument('--size', type=int, default=2048, help='texels per sun map side (power of two)')
    ap.add_argument('--sky-size', type=int, default=1024, help='texels per sky map side (power of two)')
    ap.add_argument('--voxel', type=float, default=0.5, help='leaf point binning per prototype, metres')
    args = ap.parse_args()
    if args.sun is None:
        sky = os.path.join(args.out, 'jungle_sky.json')
        args.sun = json.load(open(sky))['toSun'] if os.path.exists(sky) else [0.35355, 0.70711, 0.61237]
    print(f'sun direction {args.sun}', flush=True)
    t0 = time.time()
    stage = Usd.Stage.Open(U.SCENE)
    print(f'opened {U.SCENE} in {time.time() - t0:.0f} s', flush=True)
    leaf = LeafMaterials()
    voxel = args.voxel * 100.0

    # gather tree instances first so the outer map can be sized to them
    protos = {}   # name -> (points, area) or None
    batches = []  # (name, row-vector world matrices)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.PointInstancer):
            continue
        pi = UsdGeom.PointInstancer(prim)
        targets = pi.GetPrototypesRel().GetTargets()
        idx = np.asarray(pi.GetProtoIndicesAttr().Get(), dtype=np.int64)
        if not len(idx):
            continue
        inst = None
        for pidx, target in enumerate(targets):
            name = target.name
            if U.group_of_proto(name) != 'trees':
                continue
            if inst is None:
                inst = np.asarray(pi.ComputeInstanceTransformsAtTime(Usd.TimeCode.Default(), Usd.TimeCode.Default()))
                inst = inst @ np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                pos = inst[:, 3, :3] * 0.01
            sel = idx == pidx
            if args.forest_radius is not None and name.startswith(U.FOREST):
                sel &= np.hypot(pos[:, 0] - U.CORE_CENTRE[0], pos[:, 1] - U.CORE_CENTRE[1]) <= args.forest_radius
            if not sel.any():
                continue
            if name not in protos:
                proto = stage.GetPrimAtPath(target)
                world = U.gf_to_np(UsdGeom.Xformable(proto).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                # prototypes are authored in metres and scaled to cm by the instances; bin
                # in the prototype's own units
                scale = float(np.median(np.abs(np.linalg.det(inst[sel][:, :3, :3])) ** (1.0 / 3.0)))
                protos[name] = leaf_points(proto, world, leaf, voxel / scale)
                if protos[name] is not None:
                    print(f'  prototype {name}: {len(protos[name][1])} leaf points, '
                          f'{protos[name][1].sum() * scale * scale * 1e-4:.0f} m^2 leaf area', flush=True)
            if protos[name] is not None:
                batches.append((name, inst[sel]))

    # the fixed banyan forest is plain meshes in world space
    for top in stage.GetPrimAtPath('/root').GetChildren():
        if top.GetName().startswith('STT_Fixed_BanyanForest'):
            pts = leaf_points(top, np.eye(4), leaf, voxel)
            if pts is not None:
                protos[top.GetName()] = pts
                batches.append((top.GetName(), np.eye(4)[None]))
                print(f'  {top.GetName()}: {len(pts[1])} leaf points', flush=True)

    # tree origins in glTF metres (x, z up -> y, -y -> z); leaves reach up to ~45 m above them
    origins = np.concatenate([b[:, 3, :3] for _, b in batches]) * 0.01
    origins = origins @ U.CONV[:3, :3].T * 100.0
    corners = []
    for dy in (-10.0, 50.0):
        corners.append(origins + [0.0, dy, 0.0])
    # core: the 1 km cinematic square (USD x, y 0..1024 m) at the trees' height range
    ylo, yhi = origins[:, 1].min() - 10.0, origins[:, 1].max() + 50.0
    core_box = np.array([[x, y, z] for x in (0.0, 1024.0) for y in (ylo, yhi) for z in (-1024.0, 0.0)])
    to_sun = np.asarray(args.sun, np.float64)
    maps = [
        # sun visibility: leaf depth along the sun ray, dense over the core, coarser beyond
        fit('sunCore', to_sun, core_box, args.size),
        fit('sunFar', to_sun, np.concatenate(corners), args.size),
        # sky visibility: straight down, coarse - the shader reads it blurred anyway
        fit('sky', [0.0, 1.0, 0.0], np.concatenate(corners), args.sky_size, min_cell=1.0)
    ]
    print(f'{sum(len(b) for _, b in batches)} tree instances, maps: '
          + ', '.join(f'{m.name} {m.cell:.2f} m' for m in maps), flush=True)

    conv = U.CONV[:3, :3] * 100.0   # axis swap only, the points are already metres
    for name, mats in batches:
        pts, area = protos[name]
        t1 = time.time()
        step = max(1, CHUNK_POINTS // len(pts))
        for s in range(0, len(mats), step):
            m = mats[s:s + step]
            world = (pts[None] @ m[:, :3, :3] + m[:, None, 3, :3]) * 0.01   # (n, p, 3) USD metres
            scale2 = np.abs(np.linalg.det(m[:, :3, :3])) ** (2.0 / 3.0)
            w = (area[None] * scale2[:, None] * 1e-4).reshape(-1)
            p = world.reshape(-1, 3) @ conv.T
            for cm in maps:
                cm.add(p, w)
        print(f'  splatted {name}: {len(mats)} x {len(pts)} points, {time.time() - t1:.0f} s', flush=True)

    os.makedirs(args.out, exist_ok=True)
    meta = {'maps': [], 'extinction': 1.5, 'toSun': (to_sun / np.linalg.norm(to_sun)).tolist()}
    offset = 0
    with open(os.path.join(args.out, 'jungle_canopy.bin'), 'wb') as f:
        for cm in maps:
            lv = cm.levels(blur=1)
            entry = {**cm.meta(), 'offset': offset, 'levels': len(lv)}
            for a in lv:
                b = a.astype(np.float16).tobytes()
                f.write(b)
                offset += len(b)
            meta['maps'].append(entry)
            print(f'  {cm.name}: {cm.cell:.2f} m texels, mean depth {lv[0][..., 0].mean():.2f}, '
                  f'max {lv[0][..., 0].max():.1f}', flush=True)
    with open(os.path.join(args.out, 'jungle_canopy.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'canopy -> {args.out} ({offset / 2**20:.0f} MiB), {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
