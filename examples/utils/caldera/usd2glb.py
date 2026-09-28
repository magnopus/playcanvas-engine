"""
Convert regions of the Caldera USD data set to glTF for the streamed-meshlet bake.

Per region it writes:
  <name>.glb            render geometry: one glTF mesh per USD prototype (reused by every instance
                        node), unique world geometry (brushes / patches) merged into spatial cells.
                        Flat palette materials derived from the material-bind subset names.
  <name>_collision.glb  world-space collision soup (POSITION + indices), one node per class/cell.

Coordinates: USD is Z-up inches, glTF output is Y-up metres. Every mesh is recentred on its own
bounds (offset folded into the node matrix) because the meshlet bake quantises positions on a grid
sized from the union of mesh-local bounds.
"""
import argparse
import hashlib
import json
import os
import math
import re
import struct
import sys
import time
from collections import defaultdict

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

from overlays import overlay_mask

CALDERA = os.environ.get('CALDERA', os.path.expanduser('~/caldera'))
MAP = CALDERA + '/map_source/mp_wz_island.usd'
GEO = '/world/mp_wz_island/mp_wz_island_paths/mp_wz_island_geo'

# name -> (prim path, districts forced to their full variant)
REGIONS = {
    'capital': (GEO + '/map_capital', ['map_capital']),
    'airfield': (GEO + '/map_airfield', ['map_airfield']),
    'phosphate_mine': (GEO + '/map_phosphate_mine', ['map_phosphate_mine']),
    'beachhead': (GEO + '/map_beachhead', ['map_beachhead']),
    'hotel': (GEO + '/map_tile_p/hotel_01', ['map_tile_p']),
    'power_station': (GEO + '/map_tile_n/power_station_01', ['map_tile_n']),
    'tile_n': (GEO + '/map_tile_n', ['map_tile_n']),
    'tile_p': (GEO + '/map_tile_p', ['map_tile_p']),
    'arsenal': (GEO + '/map_arsenal', ['map_arsenal']),
    'tile_f': (GEO + '/map_tile_f', ['map_tile_f']),
    'docks': (GEO + '/map_docks', ['map_docks']),
    'caldera': (GEO + '/map_caldera', ['map_caldera']),
    'subpen': (GEO + '/map_subpen', ['map_subpen']),
    'terrain': (GEO + '/st_main', []),
    # full terrain, one region per tile (heightfield, cliffs, rocks, roads; clutter skipped)
    **{f'st_{t}': (GEO + f'/st_main/st_{t}', [f'st_main/st_{t}']) for t in 'bcdefghijklnop'},
}

INCH = 0.0254
# Z-up inches -> Y-up metres: (x, y, z) -> (x, z, -y) * INCH
CONV = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
CONV[:3, :3] *= INCH
CONV_INV = np.linalg.inv(CONV)

CELL = 64.0 / INCH  # merge cell size for unique world geometry (64 m, in inches)
CREASE_COS = math.cos(math.radians(35.0))
# coplanar layer faces closer than this to another prim's surface are dropped (inches). Measured
# on the hotel, the duplicates are exactly coplanar; real trim starts ~0.5" proud of its wall.
OVERLAY_OFFSET = 0.1
# a content-deduped mesh placed once and smaller than this merges into its cell instead
DEDUPE_MERGE_TRIS = 5000
# names that read as a blend / grime layer lose the tie when two coplanar copies overlap
OVERLAY_NAME = re.compile(r'rvl|_rev\b|_rev_|puddle|stain|grunge|grime|dust|wear|leak|peel|crease|seam|'
                          r'macro|dirty|_wet|footprint|overlay|blend')

# ---------------------------------------------------------------------------------------------
# material classification

# render faces with these materials are not drawn (tool / no-draw / overlay surfaces)
RENDER_SKIP = re.compile(r'^(caulk|skip|nodraw|clip|trigger|volume|portal|shadow|sky|mount_|'
                         r'reflection|client_|lightgrid|occlusion|hint|origin|mantle|ladder|'
                         r'broadphase|vista|terrain_cutout|terrain_collision|outofbounds)'
                         r'|decal|_shadow_caster|shadowcaster')
# collision class for a material; None = not collision
COLL_PLAYER = re.compile(r'^(clip($|_player|_ai|_lm_|_full|_stairs|_vehicle|_metal|_wood|_grating|'
                         r'_concrete|_rock|_dirt|_glass|_nosight_player|_water_player)|caulk)')
COLL_WEAPON = re.compile(r'^clip_(weap|missile|bullet|shot|nosight(?!_player))')

PALETTE = [
    # (regex, name, rgb, roughness, metallic, doubleSided)
    (r'water|ocean|puddle', 'water', (0.10, 0.24, 0.30), 0.05, 0.0, False),
    (r'glass|window_pane|windshield', 'glass', (0.45, 0.55, 0.60), 0.05, 0.3, True),
    (r'foliage|leaf|leaves|grass|tree|palm|bush|shrub|plant|vine|ivy|hedge|fern|flower|moss|reed|'
     r'weed|cactus|bamboo|agave', 'foliage', (0.22, 0.34, 0.14), 0.8, 0.0, True),
    (r'sand|beach', 'sand', (0.72, 0.64, 0.48), 0.95, 0.0, False),
    (r'dirt|mud|soil|ground|terra|gravel|pebble|earth|dust', 'dirt', (0.43, 0.35, 0.26), 0.95, 0.0, False),
    (r'rock|stone|cliff|boulder|basalt|lava|volcan|cobble', 'rock', (0.40, 0.38, 0.35), 0.9, 0.0, False),
    (r'asphalt|road|tarmac|runway', 'asphalt', (0.18, 0.18, 0.19), 0.9, 0.0, False),
    (r'concrete|cement|cinder|curb|sidewalk|pavement', 'concrete', (0.55, 0.54, 0.51), 0.9, 0.0, False),
    (r'brick', 'brick', (0.52, 0.30, 0.24), 0.85, 0.0, False),
    (r'roof|shingle|tile_roof|terracotta', 'roof', (0.50, 0.26, 0.20), 0.8, 0.0, False),
    (r'marble|ceramic|porcelain|tile', 'tile', (0.78, 0.76, 0.72), 0.3, 0.0, False),
    (r'plaster|drywall|stucco|wallpaper|paint|wall|ceiling', 'plaster', (0.76, 0.72, 0.64), 0.85, 0.0, False),
    (r'wood|timber|plank|lumber|pallet|crate|bark|log', 'wood', (0.45, 0.31, 0.19), 0.75, 0.0, False),
    (r'rust|corrugat', 'rust', (0.45, 0.28, 0.18), 0.8, 0.4, False),
    (r'metal|steel|iron|alum|chrome|pipe|rail|fence|wire|cable|tin|copper|brass|mtl', 'metal',
     (0.56, 0.57, 0.58), 0.45, 0.9, False),
    (r'fabric|cloth|carpet|cushion|rug|curtain|canvas|tarp|leather|sofa|bed|pillow', 'fabric',
     (0.46, 0.25, 0.22), 0.95, 0.0, True),
    (r'rubber|tire|tyre|plastic|black|vinyl', 'rubber', (0.12, 0.12, 0.13), 0.7, 0.0, False),
    (r'paper|cardboard|book|sign|poster', 'paper', (0.70, 0.66, 0.56), 0.9, 0.0, True),
    (r'light|lamp|bulb|emissive|neon', 'light', (0.95, 0.92, 0.80), 0.3, 0.0, False),
]
PALETTE = [(re.compile(r), n, c, ro, me, ds) for r, n, c, ro, me, ds in PALETTE]
MATERIAL_NAMES = [p[1] for p in PALETTE] + ['default', 'terrain']
MATERIAL_INDEX = {n: i for i, n in enumerate(MATERIAL_NAMES)}


def classify_render(name, is_terrain):
    """Palette material index for a render material name, or -1 when the face is not drawn."""
    if is_terrain:
        return MATERIAL_INDEX['terrain']
    if not name:
        return MATERIAL_INDEX['default']
    n = name.lower()
    if RENDER_SKIP.search(n):
        return -1
    for rx, pname, *_ in PALETTE:
        if rx.search(n):
            return MATERIAL_INDEX[pname]
    return MATERIAL_INDEX['default']


def classify_collision(name):
    if not name:
        return None
    n = name.lower()
    if COLL_WEAPON.search(n):
        return 'weapon_clip'
    if COLL_PLAYER.search(n):
        return 'player_clip'
    return None


# ---------------------------------------------------------------------------------------------
# USD reading helpers

def attr_value(attr):
    """USD value at the default time, falling back to the first time sample (proxies are sampled)."""
    if not attr or not attr.HasAuthoredValue():
        return None
    v = attr.Get()
    if v is None and attr.GetNumTimeSamples():
        v = attr.Get(attr.GetTimeSamples()[0])
    return v


def gf_to_np(m):
    # Gf matrices are row-vector (v * M); transpose to column-vector convention
    return np.array(m, dtype=np.float64).T


def read_mesh(prim):
    """Returns (points (N,3) f64, face counts, face indices, faceVarying normals or None,
    per-face material names list, leftHanded) or None when empty."""
    mesh = UsdGeom.Mesh(prim)
    pts = attr_value(mesh.GetPointsAttr())
    fvc = attr_value(mesh.GetFaceVertexCountsAttr())
    fvi = attr_value(mesh.GetFaceVertexIndicesAttr())
    if pts is None or fvc is None or fvi is None or len(pts) == 0 or len(fvc) == 0:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    fvc = np.asarray(fvc, dtype=np.int64)
    fvi = np.asarray(fvi, dtype=np.int64)

    normals = None
    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar('normals')
    if pv and pv.HasAuthoredValue():
        val = attr_value(pv.GetAttr())
        if val is not None and len(val):
            val = np.asarray(val, dtype=np.float64)
            idx = attr_value(pv.GetIndicesAttr()) if pv.IsIndexed() else None
            if idx is not None:
                val = val[np.asarray(idx, dtype=np.int64)]
            normals = (pv.GetInterpolation(), val)
    else:
        na = mesh.GetNormalsAttr()
        val = attr_value(na)
        if val is not None and len(val):
            normals = (mesh.GetNormalsInterpolation(), np.asarray(val, dtype=np.float64))

    # per-face material names from materialBind subsets
    face_mat = [None] * len(fvc)
    for sub in UsdGeom.Subset.GetGeomSubsets(UsdGeom.Imageable(prim), 'face', 'materialBind'):
        ind = attr_value(sub.GetIndicesAttr())
        if ind is None:
            continue
        nm = sub.GetPrim().GetName()
        for f in ind:
            if 0 <= f < len(face_mat):
                face_mat[f] = nm

    left = mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
    return pts, fvc, fvi, normals, face_mat, left


def triangulate(fvc, keep_faces):
    """Fan triangulation over the kept faces. Returns (corner indices into the face-vertex array
    (T,3), face id per triangle)."""
    starts = np.concatenate([[0], np.cumsum(fvc)[:-1]])
    sel = np.nonzero(keep_faces & (fvc >= 3))[0]
    if len(sel) == 0:
        return np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
    counts = fvc[sel] - 2
    face_of_tri = np.repeat(sel, counts)
    base = np.repeat(starts[sel], counts)
    # k-th triangle within its face: 0..count-1
    k = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    tri = np.stack([base, base + k + 1, base + k + 2], axis=1)
    return tri, face_of_tri


def build_geometry(pts, fvc, fvi, normals, tri_corners, left, xform):
    """Expand to per-corner vertices, transform, generate normals if absent, weld.
    Returns (positions (V,3) f32, normals (V,3) f32, indices (T*3) u32)."""
    if left:
        tri_corners = tri_corners[:, [0, 2, 1]]
    corners = tri_corners.reshape(-1)
    vid = fvi[corners]
    p = pts[vid]
    p = p @ xform[:3, :3].T + xform[:3, 3]

    tri_p = p.reshape(-1, 3, 3)
    fn = np.cross(tri_p[:, 1] - tri_p[:, 0], tri_p[:, 2] - tri_p[:, 0])
    area = np.linalg.norm(fn, axis=1)
    good = area > 1e-12
    if not good.all():
        tri_p = tri_p[good]
        fn = fn[good]
        area = area[good]
        mask = np.repeat(good, 3)
        corners = corners[mask]
        vid = vid[mask]
        p = tri_p.reshape(-1, 3)
    if len(p) == 0:
        return None
    fn_unit = fn / area[:, None]

    nrm = None
    if normals is not None:
        interp, val = normals
        try:
            if interp == 'faceVarying' and len(val) == len(fvi):
                nrm = val[corners]
            elif interp in ('vertex', 'varying') and len(val) == len(pts):
                nrm = val[vid]
        except IndexError:
            nrm = None
        if nrm is not None:
            # inverse-transpose for column vectors is n @ inv(M) for row vectors
            nrm = nrm @ np.linalg.inv(xform[:3, :3])
            ln = np.linalg.norm(nrm, axis=1)
            bad = ln < 1e-8
            nrm = nrm / np.where(bad, 1, ln)[:, None]
            corner_fn = np.repeat(fn_unit, 3, axis=0)
            nrm[bad] = corner_fn[bad]
    if nrm is None:
        # crease-angle smoothing over shared source vertices, area weighted
        corner_fn = np.repeat(fn_unit, 3, axis=0)
        acc = np.zeros((len(pts), 3))
        np.add.at(acc, vid, np.repeat(fn, 3, axis=0))
        sm = acc[vid]
        ln = np.linalg.norm(sm, axis=1)
        sm = sm / np.where(ln < 1e-12, 1, ln)[:, None]
        use = (np.einsum('ij,ij->i', sm, corner_fn) >= CREASE_COS) & (ln >= 1e-12)
        nrm = np.where(use[:, None], sm, corner_fn)

    # weld identical (position, quantised normal)
    qn = np.round(nrm * 1024).astype(np.int32)
    key = np.concatenate([p.view(np.int64).reshape(len(p), 3) if p.dtype == np.float64 else p,
                          qn.astype(np.int64)], axis=1)
    uniq, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    pos = p[first].astype(np.float32)
    nor = nrm[first].astype(np.float32)
    return pos, nor, inverse.reshape(-1).astype(np.uint32)


# ---------------------------------------------------------------------------------------------
# accumulation

class GeoBucket:
    """Concatenates geometry chunks for one glTF primitive."""
    __slots__ = ('pos', 'nor', 'idx', 'nv')

    def __init__(self):
        self.pos, self.nor, self.idx, self.nv = [], [], [], 0

    def add(self, pos, nor, idx):
        self.pos.append(pos)
        if nor is not None:
            self.nor.append(nor)
        self.idx.append(idx + self.nv)
        self.nv += len(pos)

    def tris(self):
        return sum(len(i) for i in self.idx) // 3


class Exporter:
    def __init__(self, is_terrain):
        self.is_terrain = is_terrain
        self.protos = {}           # prototype path -> {material: GeoBucket}
        self.placements = defaultdict(list)  # prototype path -> [4x4 world (inches)]
        self.cells = defaultdict(lambda: defaultdict(GeoBucket))  # cell -> material -> bucket
        self.coll = defaultdict(lambda: defaultdict(GeoBucket))   # class -> cell -> bucket
        self.unknown_guide = defaultdict(int)
        self.stats = defaultdict(int)
        self.xcache = UsdGeom.XformCache(Usd.TimeCode.EarliestTime())
        # brush / patch render geometry, per prototype (None = unique), resolved for coplanar
        # overlay layers once the traversal has seen all of it
        self.deferred = defaultdict(list)
        # non-instanced, non-brush meshes by content hash -> (first mesh data, [world matrices])
        self.dedupe = {}
        self.overlay_dropped = 0
        self.overlay_total = 0

    def world(self, prim):
        return gf_to_np(self.xcache.GetLocalToWorldTransform(prim))

    def instance_root(self, prim):
        q = prim.GetParent()
        while q and not q.IsPseudoRoot():
            if q.IsInstance():
                return q
            q = q.GetParent()
        return None

    def add_mesh(self, prim, purpose):
        # brushes / patches carry an atvi classname; their visible faces also collide
        is_brush = 'classname' in prim.GetCustomData().get('atvi', {})
        root = self.instance_root(prim) if prim.IsInstanceProxy() else None
        proto_key = str(root.GetPrototype().GetPath()) if root is not None else None
        root_path = str(root.GetPath()) if root is not None else None
        if purpose != 'guide' and not (self.is_terrain or is_brush) and proto_key is not None:
            # prototype geometry is built once, from the first instance seen
            if self._built_from.setdefault(proto_key, root_path) != root_path:
                self.stats['skipped_instanced_reads'] += 1
                return
        data = read_mesh(prim)
        if data is None:
            return
        pts, fvc, fvi, normals, face_mat, left = data
        world = self.world(prim)

        if purpose == 'guide':
            classes = np.array([classify_collision(m) or '' for m in face_mat])
            for cls in set(classes) - {''}:
                self._add_collision(cls, pts, fvc, fvi, face_mat, classes == cls, left, world)
            for m in set(face_mat):
                if m is not None and classify_collision(m) is None:
                    self.unknown_guide[re.sub(r'\d+', '#', m)] += 1
            return

        mats = np.array([classify_render(m, self.is_terrain) for m in face_mat])
        # full-terrain heightfields are plain meshes, not brushes, but they are the ground
        is_height = prim.GetName() == 'height' and prim.GetParent().GetName().startswith('super_terrain')
        if self.is_terrain or is_brush or is_height:
            solid = (mats >= 0) & (mats != MATERIAL_INDEX['water']) & (mats != MATERIAL_INDEX['glass'])
            caulk = np.array([bool(m) and m.lower().startswith('caulk') for m in face_mat])
            self._add_collision('world', pts, fvc, fvi, face_mat, solid | caulk, left, world)

        if proto_key is None and not is_brush and not self.is_terrain:
            # Models placed without USD instancing still repeat - the full terrain's props are
            # referenced over and over without being instanceable. Key them by content (local
            # points + topology) so repeats share one glTF mesh; see resolve_dedupe.
            key = hashlib.blake2b(pts.astype(np.float32).tobytes() + fvi.astype(np.int32).tobytes() +
                                  mats.astype(np.int8).tobytes(), digest_size=16).digest()
            entry = self.dedupe.get(key)
            if entry is None:
                self.dedupe[key] = ((pts, fvc, fvi, normals, mats, left), [world])
            else:
                entry[1].append(world)
            return

        if proto_key is not None:
            # collision above runs for every instance; render geometry only for the first
            if self._built_from.setdefault(proto_key, root_path) != root_path:
                return
            local = CONV @ np.linalg.inv(self.world(root)) @ world
            if is_brush and not self.is_terrain:
                self.deferred[proto_key].append((pts, fvc, fvi, normals, face_mat, mats, left, world, local, None))
                return
            bucket = self.protos.setdefault(proto_key, {})
            for m in np.unique(mats[mats >= 0]):
                tri, _ = triangulate(fvc, mats == m)
                g = build_geometry(pts, fvc, fvi, normals, tri, left, local)
                if g is not None:
                    bucket.setdefault(int(m), GeoBucket()).add(*g)
            return

        # unique geometry -> spatial cell in world space
        c = world[:3, :3] @ pts.mean(axis=0) + world[:3, 3]
        cell = tuple(np.floor(c[:2] / CELL).astype(int)) if not self.is_terrain else ('tile', str(prim.GetPath()))
        full = CONV @ world
        if is_brush and not self.is_terrain:
            self.deferred[None].append((pts, fvc, fvi, normals, face_mat, mats, left, world, full, cell))
            return
        for m in np.unique(mats[mats >= 0]):
            tri, _ = triangulate(fvc, mats == m)
            g = build_geometry(pts, fvc, fvi, normals, tri, left, full)
            if g is not None:
                self.cells[cell][int(m)].add(*g)

    def resolve_dedupe(self):
        """Content-identical meshes become one prototype placed at each world matrix; small
        singletons merge into their spatial cell like brushes do."""
        repeated = placed = 0
        for n, ((pts, fvc, fvi, normals, mats, left), worlds) in enumerate(self.dedupe.values()):
            keep = mats >= 0
            tris = int((fvc[keep] - 2).clip(min=0).sum())
            if tris == 0:
                continue
            if len(worlds) == 1 and tris < DEDUPE_MERGE_TRIS:
                world = worlds[0]
                c = world[:3, :3] @ pts.mean(axis=0) + world[:3, 3]
                cell = tuple(np.floor(c[:2] / CELL).astype(int))
                xf = CONV @ world
                for m in np.unique(mats[keep]):
                    g = build_geometry(pts, fvc, fvi, normals, triangulate(fvc, mats == m)[0], left, xf)
                    if g is not None:
                        self.cells[cell][int(m)].add(*g)
                continue
            key = f'mesh_{n}'
            bucket = self.protos.setdefault(key, {})
            for m in np.unique(mats[keep]):
                g = build_geometry(pts, fvc, fvi, normals, triangulate(fvc, mats == m)[0], left, CONV)
                if g is not None:
                    bucket.setdefault(int(m), GeoBucket()).add(*g)
            self.placements[key].extend(worlds)
            repeated += len(worlds) > 1
            placed += len(worlds)
        print(f'  content dedupe: {len(self.dedupe)} distinct meshes, {repeated} repeated, '
              f'{placed} placements as prototypes', flush=True)
        self.dedupe.clear()

    def resolve_deferred(self):
        """Drop coplanar overlay layers within each deferred group, then add the rest."""
        for key, recs in self.deferred.items():
            tris, owner, rank, per = [], [], [], []
            for oid, (pts, fvc, fvi, normals, face_mat, mats, left, world, xf, cell) in enumerate(recs):
                tri, fid = triangulate(fvc, mats >= 0)
                per.append((tri, fid))
                if not len(tri):
                    continue
                wp = pts @ world[:3, :3].T + world[:3, 3]
                tris.append(wp[fvi[tri]])
                owner.append(np.full(len(tri), oid))
                rank.append(np.array([1 if OVERLAY_NAME.search((face_mat[f] or '').lower()) else 0
                                      for f in fid], np.int8))
            if not tris:
                continue
            tris = np.concatenate(tris)
            drop = overlay_mask(tris, np.concatenate(owner), OVERLAY_OFFSET, np.concatenate(rank))
            self.overlay_dropped += int(drop.sum())
            self.overlay_total += len(drop)
            start = 0
            for (pts, fvc, fvi, normals, face_mat, mats, left, world, xf, cell), (tri, fid) in zip(recs, per):
                if not len(tri):
                    continue
                keep = ~drop[start:start + len(tri)]
                start += len(tri)
                tm = mats[fid]
                for m in np.unique(tm[keep]):
                    sel = tri[keep & (tm == m)]
                    g = build_geometry(pts, fvc, fvi, normals, sel, left, xf)
                    if g is None:
                        continue
                    if key is None:
                        self.cells[cell][int(m)].add(*g)
                    else:
                        self.protos.setdefault(key, {}).setdefault(int(m), GeoBucket()).add(*g)
        self.deferred.clear()
        print(f'  overlay layers dropped: {self.overlay_dropped} of {self.overlay_total} brush triangles', flush=True)

    def _add_collision(self, cls, pts, fvc, fvi, face_mat, keep, left, world):
        if not keep.any():
            return
        tri, _ = triangulate(fvc, keep)
        if len(tri) == 0:
            return
        if left:
            tri = tri[:, [0, 2, 1]]
        vid = fvi[tri.reshape(-1)]
        used, inv = np.unique(vid, return_inverse=True)
        full = CONV @ world
        p = (pts[used] @ full[:3, :3].T + full[:3, 3]).astype(np.float32)
        c = p.mean(axis=0)
        cell = (int(math.floor(c[0] / (CELL * INCH))), int(math.floor(-c[2] / (CELL * INCH))))
        self.coll[str(cls)][cell].add(p, None, inv.astype(np.uint32))

    def run(self, stage, root_path):
        self._built_from = {}
        t = time.time()
        it = iter(Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()))
        n = 0
        for prim in it:
            if prim.IsA(UsdGeom.Imageable):
                img = UsdGeom.Imageable(prim)
                if img.GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible:
                    it.PruneChildren()
                    continue
            if prim.IsA(UsdGeom.PointInstancer):
                # terrain clutter: its prototypes live under it and are placed per point, so
                # traversing them as plain meshes would draw each once at its authored origin
                self.stats['point_instancers_skipped'] += 1
                it.PruneChildren()
                continue
            if prim.IsInstance():
                root = prim
                self.placements[str(root.GetPrototype().GetPath())].append(self.world(root))
            if not prim.IsA(UsdGeom.Mesh):
                continue
            purpose = UsdGeom.Imageable(prim).ComputePurpose()
            if purpose not in ('default', 'render', 'guide'):
                continue
            self.add_mesh(prim, purpose)
            n += 1
            if n % 20000 == 0:
                print(f'  {n} meshes, {len(self.protos)} prototypes, {time.time() - t:.0f}s', flush=True)
        print(f'  traversed {n} meshes in {time.time() - t:.0f}s', flush=True)
        self.resolve_deferred()
        self.resolve_dedupe()
        if self.stats['point_instancers_skipped']:
            print(f"  point instancers skipped (clutter): {self.stats['point_instancers_skipped']}", flush=True)


# ---------------------------------------------------------------------------------------------
# glTF writing

class GlbWriter:
    def __init__(self, path):
        self.path = path
        self.bin_path = path + '.bin.tmp'
        self.bin = open(self.bin_path, 'wb')
        self.offset = 0
        self.gltf = {'asset': {'version': '2.0', 'generator': 'caldera usd2glb'},
                     'buffers': [], 'bufferViews': [], 'accessors': [], 'meshes': [],
                     'nodes': [], 'scenes': [{'nodes': []}], 'scene': 0, 'materials': []}

    def _view(self, data, target):
        pad = (-self.offset) % 4
        if pad:
            self.bin.write(b'\0' * pad)
            self.offset += pad
        b = data.tobytes()
        self.bin.write(b)
        self.gltf['bufferViews'].append({'buffer': 0, 'byteOffset': self.offset, 'byteLength': len(b),
                                         'target': target})
        self.offset += len(b)
        return len(self.gltf['bufferViews']) - 1

    def accessor(self, arr, target, typ, minmax=False):
        comp = {np.dtype(np.float32): 5126, np.dtype(np.uint32): 5125}[arr.dtype]
        acc = {'bufferView': self._view(arr, target), 'componentType': comp,
               'count': int(arr.shape[0]) if arr.ndim > 1 else int(arr.size), 'type': typ}
        if minmax:
            acc['min'] = arr.min(axis=0).tolist()
            acc['max'] = arr.max(axis=0).tolist()
        self.gltf['accessors'].append(acc)
        return len(self.gltf['accessors']) - 1

    def mesh(self, name, prims):
        """prims: list of (material index or None, pos, nor or None, idx)."""
        out = []
        for mat, pos, nor, idx in prims:
            attrs = {'POSITION': self.accessor(pos, 34962, 'VEC3', True)}
            if nor is not None:
                attrs['NORMAL'] = self.accessor(nor, 34962, 'VEC3')
            p = {'attributes': attrs, 'indices': self.accessor(idx, 34963, 'SCALAR'), 'mode': 4}
            if mat is not None:
                p['material'] = mat
            out.append(p)
        self.gltf['meshes'].append({'name': name, 'primitives': out})
        return len(self.gltf['meshes']) - 1

    def node(self, name, mesh=None, matrix=None, children=None, root=False):
        n = {'name': name}
        if mesh is not None:
            n['mesh'] = mesh
        if matrix is not None and not np.allclose(matrix, np.eye(4)):
            n['matrix'] = [float(x) for x in matrix.T.reshape(-1)]  # column-major
        if children:
            n['children'] = children
        self.gltf['nodes'].append(n)
        i = len(self.gltf['nodes']) - 1
        if root:
            self.gltf['scenes'][0]['nodes'].append(i)
        return i

    def close(self):
        self.bin.close()
        self.gltf['buffers'].append({'byteLength': self.offset})
        for k in [k for k, v in self.gltf.items() if isinstance(v, list) and not v]:
            del self.gltf[k]
        js = json.dumps(self.gltf, separators=(',', ':')).encode()
        js += b' ' * ((-len(js)) % 4)
        binpad = (-self.offset) % 4
        total = 12 + 8 + len(js) + 8 + self.offset + binpad
        if total >= 2 ** 32:
            raise RuntimeError(f'{self.path}: GLB would be {total / 2**30:.1f} GiB, over the 4 GiB limit')
        with open(self.path, 'wb') as f:
            f.write(struct.pack('<III', 0x46546C67, 2, total))
            f.write(struct.pack('<II', len(js), 0x4E4F534A))
            f.write(js)
            f.write(struct.pack('<II', self.offset + binpad, 0x004E4942))
            with open(self.bin_path, 'rb') as b:
                while True:
                    chunk = b.read(1 << 26)
                    if not chunk:
                        break
                    f.write(chunk)
            f.write(b'\0' * binpad)
        os.remove(self.bin_path)
        return total


def concat(bucket):
    pos = np.concatenate(bucket.pos)
    nor = np.concatenate(bucket.nor) if bucket.nor else None
    idx = np.concatenate(bucket.idx)
    return pos, nor, idx


def recentre(pos):
    c = ((pos.min(axis=0).astype(np.float64) + pos.max(axis=0)) * 0.5)
    return (pos - c).astype(np.float32), c


def translate(c):
    m = np.eye(4)
    m[:3, 3] = c
    return m


def write_render(ex, path, name):
    w = GlbWriter(path)
    for n in MATERIAL_NAMES:
        entry = next((p for p in PALETTE if p[1] == n), None)
        if entry:
            _, _, rgb, rough, metal, ds = entry
        elif n == 'terrain':
            rgb, rough, metal, ds = (0.42, 0.38, 0.30), 0.95, 0.0, False
        else:
            rgb, rough, metal, ds = (0.60, 0.58, 0.55), 0.8, 0.0, False
        mat = {'name': n, 'pbrMetallicRoughness': {'baseColorFactor': [*rgb, 1.0],
                                                    'metallicFactor': metal, 'roughnessFactor': rough}}
        if ds:
            mat['doubleSided'] = True
        w.gltf['materials'].append(mat)

    children = []
    tris_unique = tris_instanced = 0
    for key, mats in ex.protos.items():
        if not mats:
            continue
        places = ex.placements.get(key)
        if not places:
            continue
        prims = []
        allpos = []
        for m, b in sorted(mats.items()):
            prims.append((m, *concat(b)))
            allpos.append(prims[-1][1])
        lo = np.min([p.min(axis=0) for p in allpos], axis=0).astype(np.float64)
        hi = np.max([p.max(axis=0) for p in allpos], axis=0).astype(np.float64)
        c = (lo + hi) * 0.5
        prims = [(m, (p - c).astype(np.float32), nr, i) for m, p, nr, i in prims]
        t = sum(len(i) for *_, i in prims) // 3
        tris_unique += t
        tris_instanced += t * len(places)
        mi = w.mesh(key.split('/')[-1], prims)
        centre = translate(c)
        for pl in places:
            children.append(w.node(w.gltf['meshes'][mi]['name'], mi, CONV @ pl @ CONV_INV @ centre))
    for cell, mats in ex.cells.items():
        prims = []
        allpos = []
        for m, b in sorted(mats.items()):
            prims.append((m, *concat(b)))
            allpos.append(prims[-1][1])
        lo = np.min([p.min(axis=0) for p in allpos], axis=0).astype(np.float64)
        hi = np.max([p.max(axis=0) for p in allpos], axis=0).astype(np.float64)
        c = (lo + hi) * 0.5
        prims = [(m, (p - c).astype(np.float32), nr, i) for m, p, nr, i in prims]
        t = sum(len(i) for *_, i in prims) // 3
        tris_unique += t
        tris_instanced += t
        cname = 'cell_' + '_'.join(str(x).split('/')[-1] for x in cell)
        mi = w.mesh(cname, prims)
        children.append(w.node(cname, mi, translate(c)))
    w.node(name, children=children, root=True)
    size = w.close()
    print(f'  {path}: {size / 2**20:.0f} MiB, {len(w.gltf["meshes"])} meshes, {len(children)} nodes, '
          f'{tris_unique / 1e6:.2f}M unique tris, {tris_instanced / 1e6:.2f}M instanced tris', flush=True)


def write_collision(ex, path, name):
    if not ex.coll:
        return
    w = GlbWriter(path)
    groups = []
    total = 0
    for cls, cells in sorted(ex.coll.items()):
        kids = []
        for cell, b in sorted(cells.items()):
            pos, _, idx = concat(b)
            pos, c = recentre(pos)
            total += len(idx) // 3
            mi = w.mesh(f'{cls}_{cell[0]}_{cell[1]}', [(None, pos, None, idx)])
            kids.append(w.node(f'{cls}_{cell[0]}_{cell[1]}', mi, translate(c)))
        groups.append(w.node(cls, children=kids))
    w.node(name + '_collision', children=groups, root=True)
    size = w.close()
    print(f'  {path}: {size / 2**20:.0f} MiB, {total / 1e6:.2f}M tris, classes '
          f'{ {k: len(v) for k, v in ex.coll.items()} }', flush=True)


def open_stage(region):
    root_path, districts = REGIONS[region]
    sess = Sdf.Layer.CreateAnonymous('session.usda')
    stage = Usd.Stage.OpenMasked(Sdf.Layer.FindOrOpen(MAP), sess, Usd.StagePopulationMask([root_path]),
                                 Usd.Stage.LoadNone)
    with Usd.EditContext(stage, sess):
        for d in districts:
            stage.GetPrimAtPath(GEO + '/' + d).GetVariantSets().GetVariantSet('districtLod').SetVariantSelection('full')
        if region == 'terrain':
            for c in stage.GetPrimAtPath(root_path).GetChildren():
                vs = c.GetVariantSets()
                if vs.HasVariantSet('districtLod'):
                    vs.GetVariantSet('districtLod').SetVariantSelection('proxy')
    stage.Load(root_path)
    return stage, root_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('regions', nargs='+', choices=list(REGIONS))
    ap.add_argument('--out', default=os.environ.get('CALDERA_RAW', 'caldera-raw'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for r in args.regions:
        t = time.time()
        print(f'== {r}', flush=True)
        stage, root_path = open_stage(r)
        print(f'  stage open+load {time.time() - t:.0f}s', flush=True)
        ex = Exporter(is_terrain=(r == 'terrain'))
        ex.run(stage, root_path)
        write_render(ex, f'{args.out}/caldera_{r}.glb', r)
        write_collision(ex, f'{args.out}/caldera_{r}_collision.glb', r)
        if ex.unknown_guide:
            top = sorted(ex.unknown_guide.items(), key=lambda kv: -kv[1])[:25]
            print('  guide materials not treated as collision:', top, flush=True)
        print(f'  done {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    main()
