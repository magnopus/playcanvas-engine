"""
Convert the JungleRuins USD scene (Houdini / Karma export) to glTF for the streamed-meshlet bake.

One GLB per group, so each bakes on its own and the example can load any subset:
  jungle_terrain.glb   the extended (8 km, E_ tiles) and cinematic (1 km, M_ tiles) terrain
  jungle_static.glb    the pyramid, the banyan forest, river and creek
  jungle_trees.glb     RiverForest, QueenForest, RiverSapling and RiverSeedling scatters
  jungle_plants.glb    grass, moss, shrubs, sorrel, nettle, anthurium scatters

Every PointInstancer prototype becomes one glTF mesh (a primitive per bound material) and one node
carrying an EXT_mesh_gpu_instancing scatter of all its placements, so the whole scatter shares one
set of meshlet pages. Materials are UsdPreviewSurface translated to glTF PBR: opacity maps are
folded into the base colour alpha (alpha MASK at the material's opacityThreshold), roughness maps
become the green channel of a metallicRoughness texture. Images are resized to --tex-size and
cached in <out>/textures.

Coordinates: USD is Z-up centimetres, glTF output is Y-up metres. Meshes are recentred on their
own bounds (offset folded into the node / scatter transforms) because the meshlet bake quantises
positions on a grid sized from the union of mesh-local bounds.

  python3.11 usd2glb.py --out <dir> [--groups trees,plants] [--bounds x0,y0,x1,y1] [--tex-size 2048]

--bounds keeps only scatter instances (and terrain tiles) overlapping that USD-space rectangle,
in metres.
"""
import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade

Image.MAX_IMAGE_PIXELS = None

SCENE = os.environ.get('JUNGLE_USD', os.path.expanduser('~/Downloads/JungleRuins/USD/JungleRuins_Karma.usda'))

# USD (x, y, z) cm, Z-up -> glTF (x, z, -y) m
CONV = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
CONV[:3, :3] *= 0.01
CONV_INV = np.linalg.inv(CONV)

CREASE_COS = math.cos(math.radians(60.0))

TREES = ('RiverForest', 'QueenForest', 'RiverSapling', 'RiverSeedling')
STATIC = ('STT_Fixed_BanyanForest', 'Pyramid_v1a_DT', 'River', 'Creek')
# the big scattered trees reach across the extended terrain; everything else stays in the core
FOREST = ('RiverForest', 'QueenForest')
# centre of the 1 km cinematic core (M_ tiles), USD metres
CORE_CENTRE = (512.0, 512.0)


def group_of_proto(name):
    return 'trees' if name.startswith(TREES) else 'plants'


def group_of_mesh(path):
    top = path.split('/')[2] if path.count('/') >= 2 else path
    if re.match(r'^[EM]_', top):
        return 'terrain'
    # top-level only: the scatter prototypes live under /World, the static meshes under /root
    if path.startswith('/root/') and top.startswith(STATIC) and not top.startswith(FOREST):
        return 'static'
    return None


def gf_to_np(m):
    # Gf matrices are row-vector (v * M); transpose to column-vector convention
    return np.array(m, dtype=np.float64).T


def attr_value(attr):
    if not attr or not attr.HasAuthoredValue():
        return None
    v = attr.Get()
    if v is None and attr.GetNumTimeSamples():
        v = attr.Get(attr.GetTimeSamples()[0])
    return v


def primvar(prim, name):
    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar(name)
    if not pv or not pv.HasAuthoredValue():
        return None
    val = attr_value(pv.GetAttr())
    if val is None or not len(val):
        return None
    val = np.asarray(val, dtype=np.float64)
    if pv.IsIndexed():
        val = val[np.asarray(attr_value(pv.GetIndicesAttr()), dtype=np.int64)]
    return pv.GetInterpolation(), val


def triangulate(fvc, faces):
    """Fan triangulation of the given faces -> corner indices into the face-vertex array (T,3)."""
    starts = np.concatenate([[0], np.cumsum(fvc)[:-1]])
    sel = faces[fvc[faces] >= 3]
    if len(sel) == 0:
        return np.zeros((0, 3), np.int64)
    counts = fvc[sel] - 2
    base = np.repeat(starts[sel], counts)
    k = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    return np.stack([base, base + k + 1, base + k + 2], axis=1)


def per_corner(src, corners, vid, n_fv, n_pts):
    if src is None:
        return None
    interp, val = src
    if interp == 'faceVarying' and len(val) == n_fv:
        return val[corners]
    if interp in ('vertex', 'varying') and len(val) == n_pts:
        return val[vid]
    if interp == 'constant' and len(val) == 1:
        return np.repeat(val, len(corners), axis=0)
    return None


def build_geometry(pts, fvi, n_fv, normals, uv, tri, xform):
    """Per-corner expansion, transform, weld. Returns (pos f32, nor f32, idx u32, uv f32|None)."""
    corners = tri.reshape(-1)
    vid = fvi[corners]
    p = pts[vid] @ xform[:3, :3].T + xform[:3, 3]
    tri_p = p.reshape(-1, 3, 3)
    fn = np.cross(tri_p[:, 1] - tri_p[:, 0], tri_p[:, 2] - tri_p[:, 0])
    area = np.linalg.norm(fn, axis=1)
    good = area > 1e-14
    if not good.all():
        mask = np.repeat(good, 3)
        corners, vid, p = corners[mask], vid[mask], p[mask]
        fn, area = fn[good], area[good]
    if len(p) == 0:
        return None
    fn_unit = fn / area[:, None]
    corner_fn = np.repeat(fn_unit, 3, axis=0)

    nrm = per_corner(normals, corners, vid, n_fv, len(pts))
    if nrm is not None:
        nrm = nrm @ np.linalg.inv(xform[:3, :3])
        ln = np.linalg.norm(nrm, axis=1)
        bad = ln < 1e-8
        nrm = nrm / np.where(bad, 1, ln)[:, None]
        nrm[bad] = corner_fn[bad]
    else:
        acc = np.zeros((len(pts), 3))
        np.add.at(acc, vid, np.repeat(fn, 3, axis=0))
        sm = acc[vid]
        ln = np.linalg.norm(sm, axis=1)
        sm = sm / np.where(ln < 1e-12, 1, ln)[:, None]
        use = (np.einsum('ij,ij->i', sm, corner_fn) >= CREASE_COS) & (ln >= 1e-12)
        nrm = np.where(use[:, None], sm, corner_fn)

    uvc = per_corner(uv, corners, vid, n_fv, len(pts))
    parts = [vid[:, None], np.round(nrm * 1024).astype(np.int64)]
    if uvc is not None:
        parts.append(np.round(uvc * 65536).astype(np.int64))
    key = np.concatenate(parts, axis=1)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    pos = p[first].astype(np.float32)
    nor = nrm[first].astype(np.float32)
    tex = None
    if uvc is not None:
        tex = uvc[first].astype(np.float32)
        tex[:, 1] = 1.0 - tex[:, 1]
    return pos, nor, inverse.reshape(-1).astype(np.uint32), tex


def mesh_primitives(prim, xform, material_of):
    """Reads a UsdGeom.Mesh into [(material key, pos, nor, idx, uv)], one per bound material."""
    mesh = UsdGeom.Mesh(prim)
    pts = attr_value(mesh.GetPointsAttr())
    fvc = attr_value(mesh.GetFaceVertexCountsAttr())
    fvi = attr_value(mesh.GetFaceVertexIndicesAttr())
    if pts is None or fvc is None or fvi is None or not len(fvc):
        return []
    pts = np.asarray(pts, dtype=np.float64)
    fvc = np.asarray(fvc, dtype=np.int64)
    fvi = np.asarray(fvi, dtype=np.int64)
    left = mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
    normals = primvar(prim, 'normals')
    if normals is None:
        val = attr_value(mesh.GetNormalsAttr())
        if val is not None and len(val):
            normals = (mesh.GetNormalsInterpolation(), np.asarray(val, dtype=np.float64))
    uv = primvar(prim, 'st')

    base = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    groups = []
    covered = np.zeros(len(fvc), bool)
    for sub in UsdGeom.Subset.GetGeomSubsets(UsdGeom.Imageable(prim), 'face', 'materialBind'):
        ind = attr_value(sub.GetIndicesAttr())
        if ind is None:
            continue
        ind = np.asarray(ind, dtype=np.int64)
        ind = ind[(ind >= 0) & (ind < len(fvc))]
        m = UsdShade.MaterialBindingAPI(sub.GetPrim()).ComputeBoundMaterial()[0]
        groups.append((m or base, ind))
        covered[ind] = True
    rest = np.nonzero(~covered)[0]
    if len(rest):
        groups.append((base, rest))

    out = []
    for mat, faces in groups:
        tri = triangulate(fvc, faces)
        if left:
            tri = tri[:, [0, 2, 1]]
        if not len(tri):
            continue
        geo = build_geometry(pts, fvi, len(fvi), normals, uv, tri, xform)
        if geo is not None:
            out.append((material_of(mat), *geo))
    return out


# ---------------------------------------------------------------------------------------------
# materials

def connected_texture(shader_input):
    """Follows a UsdPreviewSurface input to its UsdUVTexture: (file path, output channel) or None."""
    if not shader_input or not shader_input.HasConnectedSource():
        return None
    src, out_name, _ = shader_input.GetConnectedSource()
    tex = UsdShade.Shader(src.GetPrim())
    f = tex.GetInput('file')
    if not f:
        return None
    asset = f.Get()
    if asset is None:
        return None
    path = asset.resolvedPath or asset.path
    if not os.path.isabs(path):
        layer = f.GetAttr().GetPropertyStack(Usd.TimeCode.Default())[0].layer
        path = os.path.normpath(os.path.join(os.path.dirname(layer.realPath), asset.path))
    return path, out_name


def surface_shader(material):
    out = material.GetSurfaceOutput()
    if out and out.HasConnectedSource():
        return UsdShade.Shader(out.GetConnectedSource()[0].GetPrim())
    for p in Usd.PrimRange(material.GetPrim()):
        s = UsdShade.Shader(p)
        if s and s.GetIdAttr().Get() == 'UsdPreviewSurface':
            return s
    return None


def input_value(shader, name, default):
    i = shader.GetInput(name)
    v = i.Get() if i else None
    return default if v is None else v


class TextureCache:
    def __init__(self, root, size):
        self.root = root
        self.size = size
        os.makedirs(root, exist_ok=True)

    def _open(self, path):
        if path.lower().endswith(('.tif', '.tiff')):
            im = Image.open(path)
            # Photoshop stores the plant alpha as an 'unspecified' extra sample, which PIL decodes as
            # RGBX and drops - read the uncompressed strip directly to keep it
            t = im.tile[0] if len(im.tile) == 1 else None
            if t and t.codec_name == 'raw' and t.args[0] == 'RGBX':
                a = np.fromfile(path, dtype=np.uint8, count=im.width * im.height * 4, offset=t.offset)
                return Image.fromarray(a.reshape(im.height, im.width, 4), 'RGBA')
            return im
        if not path.lower().endswith('.exr'):
            return Image.open(path)
        # PIL has no EXR reader; ffmpeg decodes most (linear data passed through as 8 bit), but
        # not every DWA-compressed file - those come back black and are treated as unreadable
        png = self._out(os.path.basename(path) + '+decoded', 'png')
        if not os.path.exists(png):
            subprocess.run(['ffmpeg', '-v', 'quiet', '-y', '-i', path, '-pix_fmt', 'rgb24', png], check=False)
        if not os.path.exists(png):
            raise OSError(f'cannot decode {path}')
        im = Image.open(png)
        if not im.getbbox():
            raise OSError(f'cannot decode {path}')
        return im

    def _load(self, path, mode):
        im = self._open(path)
        if max(im.size) > self.size:
            s = self.size / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        return im.convert(mode) if mode else im

    def _out(self, key, ext):
        name = re.sub(r'[^A-Za-z0-9_.-]', '_', key)
        return os.path.join(self.root, f'{name}_{self.size}.{ext}')

    def color(self, path, alpha_path=None, alpha_channel=None):
        out = self._out(os.path.basename(path) + ('+a' if alpha_path else ''), 'png' if alpha_path else 'jpg')
        if not os.path.exists(out):
            im = self._load(path, 'RGB')
            if alpha_path:
                a = self._load(alpha_path, None)
                a = a.getchannel(alpha_channel.upper()) if alpha_channel in ('r', 'g', 'b', 'a') and alpha_channel.upper() in a.getbands() else a.convert('L')
                im.putalpha(a.resize(im.size, Image.LANCZOS))
                im.save(out, optimize=False)
            else:
                im.save(out, quality=92)
        return out

    def normal(self, path):
        out = self._out(os.path.basename(path), 'png')
        if not os.path.exists(out):
            self._load(path, 'RGB').save(out)
        return out

    def roughness(self, path, channel):
        """glTF metallicRoughness: roughness in G, metallic (0) in B."""
        out = self._out(os.path.basename(path) + f'+rough_{channel}', 'png')
        if not os.path.exists(out):
            im = self._load(path, None)
            ch = channel.upper() if channel in ('r', 'g', 'b', 'a') else None
            g = im.getchannel(ch) if ch and ch in im.getbands() else im.convert('L')
            zero = Image.new('L', g.size, 0)
            Image.merge('RGB', (zero, g, zero)).save(out)
        return out


class Materials:
    def __init__(self, cache):
        self.cache = cache
        self.index = {}
        self.defs = []

    def key(self, material):
        if not material:
            return None
        path = str(material.GetPath())
        # the scatter prototypes each carry a copy of the same Looks; dedupe by name
        name = material.GetPrim().GetName()
        if name not in self.index:
            self.index[name] = len(self.defs)
            self.defs.append(self.translate(name, material))
        return self.index[name]

    def translate(self, name, material):
        sh = surface_shader(material)
        mat = {'name': name, 'pbrMetallicRoughness': {'metallicFactor': 0.0, 'roughnessFactor': 1.0}}
        pbr = mat['pbrMetallicRoughness']
        if sh is None:
            return mat
        images = {}
        base = connected_texture(sh.GetInput('diffuseColor'))
        opacity = connected_texture(sh.GetInput('opacity'))
        threshold = float(input_value(sh, 'opacityThreshold', 0.0))
        if base:
            alpha = opacity if opacity and threshold > 0 else None
            images['baseColorTexture'] = self.cache.color(base[0], alpha and alpha[0], alpha and alpha[1])
        else:
            c = input_value(sh, 'diffuseColor', (0.18, 0.18, 0.18))
            pbr['baseColorFactor'] = [float(c[0]), float(c[1]), float(c[2]), 1.0]
        if threshold > 0:
            mat['alphaMode'] = 'MASK'
            mat['alphaCutoff'] = threshold
        # data maps that cannot be decoded (some EXRs) fall back to factors / no normal map
        rough = connected_texture(sh.GetInput('roughness'))
        pbr['roughnessFactor'] = float(input_value(sh, 'roughness', 0.8))
        if rough:
            try:
                images['metallicRoughnessTexture'] = self.cache.roughness(rough[0], rough[1])
                pbr['roughnessFactor'] = 1.0
            except OSError as e:
                print(f'  {name}: {e}, roughness factor used', flush=True)
        pbr['metallicFactor'] = float(input_value(sh, 'metallic', 0.0))
        nrm = connected_texture(sh.GetInput('normal'))
        if nrm:
            try:
                images['normalTexture'] = self.cache.normal(nrm[0])
            except OSError as e:
                print(f'  {name}: {e}, no normal map', flush=True)
        if 'TwoSided' in name or 'Leaves' in name or 'Leaf' in name or threshold > 0:
            mat['doubleSided'] = True
        mat['_images'] = images
        return mat


# ---------------------------------------------------------------------------------------------
# glb writer

class GlbWriter:
    def __init__(self, path):
        self.path = path
        self.bin_path = path + '.bin.tmp'
        self.bin = open(self.bin_path, 'wb')
        self.offset = 0
        self.gltf = {'asset': {'version': '2.0', 'generator': 'jungle usd2glb'},
                     'buffers': [], 'bufferViews': [], 'accessors': [], 'meshes': [],
                     'nodes': [], 'scenes': [{'nodes': []}], 'scene': 0, 'materials': []}
        self.images = {}

    def _view(self, data, target=None):
        pad = (-self.offset) % 4
        if pad:
            self.bin.write(b'\0' * pad)
            self.offset += pad
        b = data.tobytes()
        self.bin.write(b)
        view = {'buffer': 0, 'byteOffset': self.offset, 'byteLength': len(b)}
        if target:
            view['target'] = target
        self.gltf['bufferViews'].append(view)
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

    def texture(self, path):
        if path not in self.images:
            data = np.fromfile(path, dtype=np.uint8)
            view = self._view(data)
            mime = 'image/png' if path.endswith('.png') else 'image/jpeg'
            self.gltf.setdefault('images', []).append({'bufferView': view, 'mimeType': mime,
                                                       'name': os.path.basename(path)})
            self.gltf.setdefault('textures', []).append({'source': len(self.gltf['images']) - 1})
            self.images[path] = len(self.gltf['textures']) - 1
        return self.images[path]

    def materials(self, defs):
        for d in defs:
            m = {k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items() if k != '_images'}
            for slot, path in d.get('_images', {}).items():
                ref = {'index': self.texture(path)}
                if slot == 'normalTexture':
                    m['normalTexture'] = ref
                else:
                    m['pbrMetallicRoughness'][slot] = ref
            self.gltf['materials'].append(m)

    def mesh(self, name, prims):
        out = []
        for mat, pos, nor, idx, uv in prims:
            attrs = {'POSITION': self.accessor(pos, 34962, 'VEC3', True), 'NORMAL': self.accessor(nor, 34962, 'VEC3')}
            if uv is not None:
                attrs['TEXCOORD_0'] = self.accessor(uv, 34962, 'VEC2')
            p = {'attributes': attrs, 'indices': self.accessor(idx, 34963, 'SCALAR'), 'mode': 4}
            if mat is not None:
                p['material'] = mat
            out.append(p)
        self.gltf['meshes'].append({'name': name, 'primitives': out})
        return len(self.gltf['meshes']) - 1

    def node(self, name, mesh=None, matrix=None, children=None, root=False, scatter=None):
        n = {'name': name}
        if mesh is not None:
            n['mesh'] = mesh
        if matrix is not None and not np.allclose(matrix, np.eye(4)):
            n['matrix'] = [float(x) for x in matrix.T.reshape(-1)]
        if children:
            n['children'] = children
        if scatter is not None:
            t, r, s = scatter
            n['extensions'] = {'EXT_mesh_gpu_instancing': {'attributes': {
                'TRANSLATION': self.accessor(t, None, 'VEC3'),
                'ROTATION': self.accessor(r, None, 'VEC4'),
                'SCALE': self.accessor(s, None, 'VEC3')}}}
        self.gltf['nodes'].append(n)
        i = len(self.gltf['nodes']) - 1
        if root:
            self.gltf['scenes'][0]['nodes'].append(i)
        return i

    def close(self):
        self.bin.close()
        self.gltf['buffers'].append({'byteLength': self.offset})
        if any('EXT_mesh_gpu_instancing' in n.get('extensions', {}) for n in self.gltf['nodes']):
            self.gltf['extensionsUsed'] = ['EXT_mesh_gpu_instancing']
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
                while chunk := b.read(1 << 26):
                    f.write(chunk)
            f.write(b'\0' * binpad)
        os.remove(self.bin_path)
        return total


# ---------------------------------------------------------------------------------------------

def decompose(mats):
    """(N,4,4) column-vector affine matrices -> TRS float32 arrays (translation, xyzw quat, scale)."""
    t = mats[:, :3, 3]
    m = mats[:, :3, :3]
    s = np.linalg.norm(m, axis=1)
    flip = np.linalg.det(m) < 0
    s[flip, 0] *= -1
    r = m / s[:, None, :]
    # quaternion from rotation matrix (Shepperd)
    q = np.zeros((len(r), 4))
    tr = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    c0 = tr > 0
    c1 = ~c0 & (r[:, 0, 0] >= r[:, 1, 1]) & (r[:, 0, 0] >= r[:, 2, 2])
    c2 = ~c0 & ~c1 & (r[:, 1, 1] >= r[:, 2, 2])
    c3 = ~c0 & ~c1 & ~c2
    w = np.sqrt(np.maximum(1 + tr[c0], 1e-12)) * 2
    q[c0] = np.stack([(r[c0, 2, 1] - r[c0, 1, 2]) / w, (r[c0, 0, 2] - r[c0, 2, 0]) / w, (r[c0, 1, 0] - r[c0, 0, 1]) / w, w / 4], 1)
    w = np.sqrt(np.maximum(1 + r[c1, 0, 0] - r[c1, 1, 1] - r[c1, 2, 2], 1e-12)) * 2
    q[c1] = np.stack([w / 4, (r[c1, 0, 1] + r[c1, 1, 0]) / w, (r[c1, 0, 2] + r[c1, 2, 0]) / w, (r[c1, 2, 1] - r[c1, 1, 2]) / w], 1)
    w = np.sqrt(np.maximum(1 + r[c2, 1, 1] - r[c2, 0, 0] - r[c2, 2, 2], 1e-12)) * 2
    q[c2] = np.stack([(r[c2, 0, 1] + r[c2, 1, 0]) / w, w / 4, (r[c2, 1, 2] + r[c2, 2, 1]) / w, (r[c2, 0, 2] - r[c2, 2, 0]) / w], 1)
    w = np.sqrt(np.maximum(1 + r[c3, 2, 2] - r[c3, 0, 0] - r[c3, 1, 1], 1e-12)) * 2
    q[c3] = np.stack([(r[c3, 0, 2] + r[c3, 2, 0]) / w, (r[c3, 1, 2] + r[c3, 2, 1]) / w, w / 4, (r[c3, 1, 0] - r[c3, 0, 1]) / w], 1)
    q /= np.linalg.norm(q, axis=1)[:, None]
    return t.astype(np.float32), q.astype(np.float32), s.astype(np.float32)


def recentre(prims):
    lo = np.min([p[1].min(axis=0) for p in prims], axis=0).astype(np.float64)
    hi = np.max([p[1].max(axis=0) for p in prims], axis=0).astype(np.float64)
    c = (lo + hi) * 0.5
    return [(m, (p - c).astype(np.float32), n, i, uv) for m, p, n, i, uv in prims], c


def translate(c):
    m = np.eye(4)
    m[:3, 3] = c
    return m


def in_bounds_xy(lo, hi, bounds):
    if bounds is None:
        return True
    x0, y0, x1, y1 = bounds
    return lo[0] <= x1 and hi[0] >= x0 and lo[1] <= y1 and hi[1] >= y0


def export_group(stage, group, out_path, cache, bounds, only, forest_radius=None):
    t0 = time.time()
    mats = Materials(cache)
    meshes = []  # (name, prims, [column-vector glTF matrices] or None, single matrix)
    stats = defaultdict(int)
    bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])

    if group in ('terrain', 'static'):
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path = str(prim.GetPath())
            if group_of_mesh(path) != group:
                continue
            if only and not any(o in path for o in only):
                continue
            if bounds is not None:
                r = bcache.ComputeWorldBound(prim).ComputeAlignedRange()
                if not in_bounds_xy(np.array(r.GetMin()) * 0.01, np.array(r.GetMax()) * 0.01, bounds):
                    continue
            world = CONV @ gf_to_np(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            prims = mesh_primitives(prim, world, mats.key)
            if not prims:
                continue
            prims, c = recentre(prims)
            meshes.append((prim.GetName(), prims, None, translate(c)))
            stats['meshes'] += 1
            stats['tris'] += sum(len(p[3]) for p in prims) // 3
            print(f'  {path}: {sum(len(p[3]) for p in prims) // 3} tris', flush=True)
    else:
        protos = {}  # prototype name -> (prims, centre, [matrices])
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.PointInstancer):
                continue
            pi = UsdGeom.PointInstancer(prim)
            targets = pi.GetPrototypesRel().GetTargets()
            idx = np.asarray(pi.GetProtoIndicesAttr().Get(), dtype=np.int64)
            if not len(idx):
                continue
            # protoXform * instance TRS, row-vector, PI-local
            inst = np.asarray(pi.ComputeInstanceTransformsAtTime(Usd.TimeCode.Default(), Usd.TimeCode.Default()))
            pi_world = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            inst = inst @ pi_world
            pos = inst[:, 3, :3] * 0.01
            keep = np.ones(len(idx), bool)
            if bounds is not None:
                x0, y0, x1, y1 = bounds
                keep = (pos[:, 0] >= x0) & (pos[:, 0] <= x1) & (pos[:, 1] >= y0) & (pos[:, 1] <= y1)
            for pidx, target in enumerate(targets):
                name = target.name
                if group_of_proto(name) != group or (only and not any(o in name for o in only)):
                    continue
                sel = keep & (idx == pidx)
                if forest_radius is not None and name.startswith(FOREST):
                    sel &= np.hypot(pos[:, 0] - CORE_CENTRE[0], pos[:, 1] - CORE_CENTRE[1]) <= forest_radius
                if not sel.any():
                    continue
                if name not in protos:
                    proto = stage.GetPrimAtPath(target)
                    proto_world = gf_to_np(UsdGeom.Xformable(proto).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                    prims = []
                    for d in Usd.PrimRange(proto, Usd.TraverseInstanceProxies()):
                        if d.IsA(UsdGeom.Mesh):
                            # mesh relative to the prototype root, which the instance transforms
                            # already include; kept in USD units, converted with the instances
                            rel = np.linalg.inv(proto_world) @ gf_to_np(UsdGeom.Xformable(d).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                            prims += mesh_primitives(d, CONV @ rel, mats.key)
                    if not prims:
                        protos[name] = None
                        continue
                    prims, c = recentre(prims)
                    protos[name] = (prims, c, [])
                    print(f'  prototype {name}: {sum(len(p[3]) for p in prims) // 3} tris, {len(prims)} prims', flush=True)
                if protos[name] is None:
                    continue
                # column-vector: CONV * W * CONV^-1 * proto-local recentre (CONV maps the mesh)
                w = np.transpose(inst[sel], (0, 2, 1))
                protos[name][2].append(w)
        for name, entry in protos.items():
            if entry is None:
                continue
            prims, c, ws = entry
            w = np.concatenate(ws)
            g = CONV[None] @ w @ CONV_INV[None] @ translate(c)[None]
            meshes.append((name, prims, g, None))
            t = sum(len(p[3]) for p in prims) // 3
            stats['protos'] += 1
            stats['instances'] += len(g)
            stats['tris'] += t
            stats['instanced_tris'] += t * len(g)
            print(f'  {name}: {len(g)} instances', flush=True)

    if not meshes:
        print(f'{group}: nothing to export')
        return
    w = GlbWriter(out_path)
    w.materials(mats.defs)
    children = []
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for name, prims, scatter, matrix in meshes:
        mi = w.mesh(name, prims)
        r = max(float(np.abs(p[1]).max()) for p in prims)
        origins = scatter[:, :3, 3] if scatter is not None else matrix[None, :3, 3]
        lo = np.minimum(lo, origins.min(axis=0) - r)
        hi = np.maximum(hi, origins.max(axis=0) + r)
        if scatter is not None:
            children.append(w.node(name, mi, scatter=decompose(scatter)))
        else:
            children.append(w.node(name, mi, matrix))
    w.node(f'jungle_{group}', children=children, root=True)
    size = w.close()
    # read by manifest.py: the example frames and labels groups from it
    with open(os.path.splitext(out_path)[0] + '.json', 'w') as f:
        json.dump({'group': group, 'min': lo.round(2).tolist(), 'max': hi.round(2).tolist(),
                   **{k: int(v) for k, v in stats.items()}}, f)
    print(f'{group}: {out_path} {size / 2**20:.0f} MiB, {dict(stats)}, {len(mats.defs)} materials, '
          f'{time.time() - t0:.0f} s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--groups', default='terrain,static,trees,plants')
    ap.add_argument('--bounds', help='x0,y0,x1,y1 in USD metres (x east, y north)')
    ap.add_argument('--forest-radius', type=float,
                    help='keep RiverForest / QueenForest trees within this many metres of the cinematic core '
                         'centre (they cover the whole 8 km terrain, ~3M trees)')
    ap.add_argument('--only', help='comma list of name substrings to keep (debugging)')
    ap.add_argument('--tex-size', type=int, default=2048)
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    bounds = tuple(map(float, args.bounds.split(','))) if args.bounds else None
    only = args.only.split(',') if args.only else None
    t0 = time.time()
    stage = Usd.Stage.Open(SCENE)
    print(f'opened {SCENE} in {time.time() - t0:.0f} s', flush=True)
    cache = TextureCache(os.path.join(args.out, 'textures'), args.tex_size)
    for group in args.groups.split(','):
        export_group(stage, group, os.path.join(args.out, f'jungle_{group}{args.suffix}.glb'), cache, bounds, only, args.forest_radius)


if __name__ == '__main__':
    main()
