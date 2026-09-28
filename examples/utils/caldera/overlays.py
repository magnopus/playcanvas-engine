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


def _plane_frame(tris):
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    ln = np.linalg.norm(n, axis=1)
    n = n / np.where(ln < 1e-12, 1, ln)[:, None]
    return n, ln > 1e-9


def _point_in_tri(p, a, b, c, tol=0.02):
    """Barycentric containment of p in (a, b, c), all (K,3), p assumed near the plane."""
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
    return ok & (u >= -tol) & (v >= -tol) & (w >= -tol)


def overlay_pairs(tris, owner, max_offset):
    """tris (T,3,3) f64, owner (T,) int (prim id; faces of one prim never cover each other).
    Returns (i, j, offset) arrays: centroid of triangle i lies on triangle j's plane within
    max_offset (signed along j's normal, positive = i in front of j) and inside j."""
    n, good = _plane_frame(tris)
    cen = tris.mean(axis=1)
    d = np.einsum('ij,ij->i', n, tris[:, 0])
    nq = np.round(n * NORMAL_Q).astype(np.int64)
    axis = np.argmax(np.abs(n), axis=1)
    # 2D projection dropping the dominant axis (consistent within a normal bucket)
    keep_axes = np.array([[1, 2], [0, 2], [0, 1]])[axis]
    rows = np.arange(len(tris))

    def key(nqv, dq, gx, gy):
        h = (nqv[:, 0] + 64) * 1_000_003 + (nqv[:, 1] + 64) * 10_007 + (nqv[:, 2] + 64)
        return ((h * 1_000_003 + dq) * 1_000_003 + gx) * 1_000_003 + gy

    # insert triangles into (plane bucket, 2D cell) for every cell their 2D bounds touch
    t2 = np.stack([tris[rows, :, keep_axes[:, 0]], tris[rows, :, keep_axes[:, 1]]], axis=2)  # (T,3,2)
    lo = np.floor(t2.min(axis=1) / GRID).astype(np.int64)
    hi = np.floor(t2.max(axis=1) / GRID).astype(np.int64)
    span = (hi - lo + 1)
    cells = span[:, 0] * span[:, 1]
    sel = good & (cells <= 4096)
    ti = np.repeat(rows[sel], cells[sel])
    k = np.arange(len(ti)) - np.repeat(np.cumsum(cells[sel]) - cells[sel], cells[sel])
    sx = span[ti, 0]
    gx = lo[ti, 0] + k % sx
    gy = lo[ti, 1] + k // sx
    dq = np.round(d[ti] / PLANE_Q).astype(np.int64)
    tkeys = key(nq[ti], dq, gx, gy)
    order = np.argsort(tkeys, kind='stable')
    tkeys = tkeys[order]
    ti = ti[order]

    # query each centroid against its cell, in the plane buckets within max_offset
    c2 = np.stack([cen[rows, keep_axes[:, 0]], cen[rows, keep_axes[:, 1]]], axis=1)
    cg = np.floor(c2 / GRID).astype(np.int64)
    out_i, out_j = [], []
    reach = int(np.ceil(max_offset / PLANE_Q)) + 1
    for off in range(-reach, reach + 1):
        qk = key(nq, np.round(d / PLANE_Q).astype(np.int64) + off, cg[:, 0], cg[:, 1])
        a = np.searchsorted(tkeys, qk, 'left')
        b = np.searchsorted(tkeys, qk, 'right')
        cnt = b - a
        m = good & (cnt > 0)
        qi = np.repeat(rows[m], cnt[m])
        pos = np.repeat(a[m], cnt[m]) + (np.arange(cnt[m].sum()) - np.repeat(np.cumsum(cnt[m]) - cnt[m], cnt[m]))
        qj = ti[pos]
        out_i.append(qi)
        out_j.append(qj)
    qi = np.concatenate(out_i)
    qj = np.concatenate(out_j)
    m = owner[qi] != owner[qj]
    qi, qj = qi[m], qj[m]
    dot = np.einsum('ij,ij->i', n[qi], n[qj])
    offset = np.einsum('ij,ij->i', cen[qi] - tris[qj, 0], n[qj])
    m = (dot >= MIN_DOT) & (np.abs(offset) <= max_offset)
    qi, qj, offset = qi[m], qj[m], offset[m]
    proj = cen[qi] - offset[:, None] * n[qj]
    m = _point_in_tri(proj, tris[qj, 0], tris[qj, 1], tris[qj, 2])
    return qi[m], qj[m], offset[m]


def overlay_mask(tris, owner, max_offset, rank=None):
    """True for triangles to drop: those covered by another prim's surface just behind them.
    Exactly coplanar duplicates (the common case: a blend layer re-using its base surface) drop the
    higher rank, then the higher owner id, so exactly one copy survives."""
    i, j, off = overlay_pairs(tris, owner, max_offset)
    if rank is None:
        rank = np.zeros(len(tris), np.int8)
    eps = 1e-2
    same = np.abs(off) <= eps
    worse = (rank[i] > rank[j]) | ((rank[i] == rank[j]) & (owner[i] > owner[j]))
    drop_pair = (off > eps) | (same & worse)
    mask = np.zeros(len(tris), bool)
    mask[i[drop_pair]] = True
    return mask
