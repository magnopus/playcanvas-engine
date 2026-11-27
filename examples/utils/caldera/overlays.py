"""
Find coplanar overlay faces: faces lying on (or just in front of) another prim's surface, as the
source engine's blend layers, puddles and grime patches do. Rendered opaque they z-fight with the
surface beneath. Operates on triangles in one common space (inches).
"""
import numpy as np

GRID = 64.0         # 2D hash cell size, inches
NORMAL_Q = 16.0     # normal quantisation for plane buckets
PLANE_Q = 1.0       # plane-distance bucket size, inches
MIN_DOT = 0.995
# coverage samples on a candidate triangle are at most this far apart (inches), up to MAX_SPLITS
# per edge; corner and edge samples are pulled this fraction toward the centroid
SAMPLE_SPACING = 8.0
MAX_SPLITS = 24
SAMPLE_INSET = 0.01
QUERY_CHUNK = 1 << 20


def _plane_frame(tris):
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    ln = np.linalg.norm(n, axis=1)
    n = n / np.where(ln < 1e-12, 1, ln)[:, None]
    return n, ln > 1e-9


def _point_in_tri(p, a, b, c, tol=0.05):
    """Containment of p in (a, b, c), all (K,3), p assumed near the plane, with up to tol inches of
    slack past each edge. The slack is absolute: a relative one grows with the triangle, and on a
    large floor face it swallowed the thin strips between neighbouring patches."""
    v0, v1, v2 = b - a, c - a, p - a
    d00 = np.einsum('ij,ij->i', v0, v0)
    d01 = np.einsum('ij,ij->i', v0, v1)
    d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0)
    d21 = np.einsum('ij,ij->i', v2, v1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 1e-12
    den = np.where(ok, den, 1)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1 - v - w
    # a barycentric coordinate times its vertex's height is the distance past the opposite edge
    area2 = np.sqrt(np.maximum(den, 0))
    ha = area2 / np.maximum(np.linalg.norm(c - b, axis=1), 1e-12)
    hb = area2 / np.maximum(np.sqrt(d11), 1e-12)
    hc = area2 / np.maximum(np.sqrt(d00), 1e-12)
    return ok & (u * ha >= -tol) & (v * hb >= -tol) & (w * hc >= -tol)


def _key(nqv, dq, gx, gy):
    h = (nqv[:, 0] + 64) * 1_000_003 + (nqv[:, 1] + 64) * 10_007 + (nqv[:, 2] + 64)
    return ((h * 1_000_003 + dq) * 1_000_003 + gx) * 1_000_003 + gy


class _PlaneIndex:
    """Triangles hashed by (plane bucket, 2D cell) for every cell their 2D bounds touch."""

    def __init__(self, tris):
        self.tris = tris
        self.n, self.good = _plane_frame(tris)
        self.d = np.einsum('ij,ij->i', self.n, tris[:, 0])
        self.nq = np.round(self.n * NORMAL_Q).astype(np.int64)
        axis = np.argmax(np.abs(self.n), axis=1)
        # 2D projection dropping the dominant axis (consistent within a normal bucket)
        self.keep_axes = np.array([[1, 2], [0, 2], [0, 1]])[axis]
        rows = np.arange(len(tris))
        t2 = np.stack([tris[rows, :, self.keep_axes[:, 0]], tris[rows, :, self.keep_axes[:, 1]]], axis=2)  # (T,3,2)
        lo = np.floor(t2.min(axis=1) / GRID).astype(np.int64)
        hi = np.floor(t2.max(axis=1) / GRID).astype(np.int64)
        span = (hi - lo + 1)
        cells = span[:, 0] * span[:, 1]
        sel = self.good & (cells <= 4096)
        ti = np.repeat(rows[sel], cells[sel])
        k = np.arange(len(ti)) - np.repeat(np.cumsum(cells[sel]) - cells[sel], cells[sel])
        sx = span[ti, 0]
        gx = lo[ti, 0] + k % sx
        gy = lo[ti, 1] + k // sx
        dq = np.round(self.d[ti] / PLANE_Q).astype(np.int64)
        tkeys = _key(self.nq[ti], dq, gx, gy)
        order = np.argsort(tkeys, kind='stable')
        self.tkeys = tkeys[order]
        self.ti = ti[order]

    def query(self, pts, src, owner, max_offset):
        """pts (S,3) lying on triangles src (S,). Returns (s, j, offset): point s lies on triangle j's
        plane within max_offset (signed along j's normal, positive = in front of j) and inside j,
        where j belongs to another owner than src[s]."""
        n, tris = self.n, self.tris
        ka = self.keep_axes[src]
        rows = np.arange(len(pts))
        cg = np.floor(np.stack([pts[rows, ka[:, 0]], pts[rows, ka[:, 1]]], axis=1) / GRID).astype(np.int64)
        dq0 = np.round(self.d[src] / PLANE_Q).astype(np.int64)
        valid = self.good[src]
        out_s, out_j = [], []
        reach = int(np.ceil(max_offset / PLANE_Q)) + 1
        for off in range(-reach, reach + 1):
            qk = _key(self.nq[src], dq0 + off, cg[:, 0], cg[:, 1])
            a = np.searchsorted(self.tkeys, qk, 'left')
            b = np.searchsorted(self.tkeys, qk, 'right')
            cnt = b - a
            m = valid & (cnt > 0)
            qs = np.repeat(rows[m], cnt[m])
            pos = np.repeat(a[m], cnt[m]) + (np.arange(cnt[m].sum()) - np.repeat(np.cumsum(cnt[m]) - cnt[m], cnt[m]))
            out_s.append(qs)
            out_j.append(self.ti[pos])
        qs = np.concatenate(out_s)
        qj = np.concatenate(out_j)
        m = owner[src[qs]] != owner[qj]
        qs, qj = qs[m], qj[m]
        dot = np.einsum('ij,ij->i', n[src[qs]], n[qj])
        offset = np.einsum('ij,ij->i', pts[qs] - tris[qj, 0], n[qj])
        m = (dot >= MIN_DOT) & (np.abs(offset) <= max_offset)
        qs, qj, offset = qs[m], qj[m], offset[m]
        proj = pts[qs] - offset[:, None] * n[qj]
        m = _point_in_tri(proj, tris[qj, 0], tris[qj, 1], tris[qj, 2])
        return qs[m], qj[m], offset[m]


def overlay_pairs(tris, owner, max_offset):
    """tris (T,3,3) f64, owner (T,) int (prim id; faces of one prim never cover each other).
    Returns (i, j, offset) arrays: centroid of triangle i lies on triangle j's plane within
    max_offset (signed along j's normal, positive = i in front of j) and inside j."""
    return _PlaneIndex(tris).query(tris.mean(axis=1), np.arange(len(tris)), owner, max_offset)


def _drops(i, j, off, rank, owner, max_offset, layer_offset):
    """Pairs (i over j) under which i gives way to j: i sits in front of j (up to max_offset, or
    layer_offset when i is ranked as a layer), or exactly on it with the higher rank, then the higher
    owner id."""
    eps = 1e-2
    same = np.abs(off) <= eps
    worse = (rank[i] > rank[j]) | ((rank[i] == rank[j]) & (owner[i] > owner[j]))
    front = (off > eps) & (off <= np.where(rank[i] > 0, layer_offset, max_offset))
    return front | (same & worse)


def _samples(tris):
    """Barycentric grid samples over each triangle, denser on larger ones. Returns (points (S,3),
    triangle index (S,))."""
    edge = np.linalg.norm(tris - np.roll(tris, 1, axis=1), axis=2).max(axis=1)
    splits = np.clip(np.ceil(edge / SAMPLE_SPACING), 1, MAX_SPLITS).astype(np.int64)
    pts, src = [], []
    for s in np.unique(splits):
        sel = np.nonzero(splits == s)[0]
        a, b = np.meshgrid(np.arange(s + 1), np.arange(s + 1), indexing='ij')
        keep = a + b <= s
        w = np.stack([a[keep], b[keep], s - a[keep] - b[keep]], axis=1) / s
        w = w * (1 - SAMPLE_INSET) + SAMPLE_INSET / 3
        if s > 1:
            w = np.vstack([w, [[1 / 3, 1 / 3, 1 / 3]]])
        pts.append(np.einsum('kc,tcx->tkx', w, tris[sel]).reshape(-1, 3))
        src.append(np.repeat(sel, len(w)))
    return np.concatenate(pts), np.concatenate(src)


def overlay_mask(tris, owner, max_offset, rank=None, layer_offset=None):
    """True for triangles to drop: those covered by other prims' surfaces just behind them (within
    max_offset, or layer_offset for triangles with rank > 0).
    Exactly coplanar duplicates (the common case: a blend layer re-using its base surface) drop the
    higher rank, then the higher owner id, so exactly one copy survives.

    A triangle drops only when the covering surfaces hold all of it: coplanar patches are often
    tessellated differently, so a cover under the centroid can leave most of a large triangle bare.
    Triangles whose centroid is covered are sampled over their area and drop when every sample is."""
    if rank is None:
        rank = np.zeros(len(tris), np.int8)
    if layer_offset is None or not rank.any():
        layer_offset = max_offset
    reach = max(max_offset, layer_offset)
    index = _PlaneIndex(tris)
    i, j, off = index.query(tris.mean(axis=1), np.arange(len(tris)), owner, reach)
    cand = np.unique(i[_drops(i, j, off, rank, owner, max_offset, layer_offset)])
    mask = np.zeros(len(tris), bool)
    if not len(cand):
        return mask
    pts, src = _samples(tris[cand])
    src = cand[src]
    covered = np.zeros(len(pts), bool)
    for c0 in range(0, len(pts), QUERY_CHUNK):
        s, j, off = index.query(pts[c0:c0 + QUERY_CHUNK], src[c0:c0 + QUERY_CHUNK], owner, reach)
        covered[c0 + s[_drops(src[c0 + s], j, off, rank, owner, max_offset, layer_offset)]] = True
    bare = np.zeros(len(tris), bool)
    bare[src[~covered]] = True
    mask[cand] = ~bare[cand]
    return mask
