import { Vec3 } from '../../core/math/vec3.js';

/**
 * @import { Camera } from '../camera.js'
 */

/**
 * Geometry of a directional shadow cascade's caster volume (see
 * MeshletShadowRenderer#tightCasterVolume): the part of the camera's view slice a cascade shades,
 * clipped to the receivers a meshlet caster can reach and extruded towards the light, as cull
 * planes.
 *
 * @ignore
 */

const _tmp = new Vec3();
const _lightDir = new Vec3();
const _axisU = new Vec3();
const _axisV = new Vec3();
const _faceNormal = new Vec3();
const _edgeA = new Vec3();
const _edgeB = new Vec3();
const _polyA = [];
const _polyB = [];
const _hullPoints = [];
const _hullOrder = [];
const _hull = [];
const _facing = [0, 0, 0, 0, 0, 0];
// receiver region planes: up to 6 box faces + 12 silhouette edges, as (nx, ny, nz, d) keeping
// dot(n, p) <= d
const _regionPlanes = new Float32Array(18 * 4);
// slice planes in the same form, for testing world corners against the slice
const _slicePlanes = new Float32Array(6 * 4);

// the six faces of a frustum slice, as corner indices of Camera#getFrustumCorners (near 0-3,
// far 4-7, each in the same winding)
const SLICE_FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];

// the world box: its corners as (x, y, z) picks of min (0) / max (1), and its edges as corner
// index pairs with the two faces (axis * 2 + side) they join
const BOX_CORNERS = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const BOX_EDGES = [
    [0, 1, 2, 4], [3, 2, 3, 4], [4, 5, 2, 5], [7, 6, 3, 5], // along x: faces -y/+y, -z/+z
    [0, 3, 0, 4], [1, 2, 1, 4], [4, 7, 0, 5], [5, 6, 1, 5], // along y: faces -x/+x, -z/+z
    [0, 4, 0, 2], [1, 5, 1, 2], [3, 7, 0, 3], [2, 6, 1, 3] //  along z: faces -x/+x, -y/+y
];

/**
 * A camera's view slice between two distances, in world space, and its bounding sphere about
 * the corners' centroid - the construction the engine fits a cascade to.
 *
 * @param {Camera} camera - The camera.
 * @param {number} near - Slice start distance.
 * @param {number} far - Slice end distance.
 * @param {Vec3[]} corners - Receives the eight world-space corners (near 0-3, far 4-7).
 * @param {Vec3} center - Receives the sphere's center.
 * @returns {number} The sphere's radius.
 */
const sliceSphere = (camera, near, far, corners, center) => {
    const pts = camera.getFrustumCorners(near, far);
    const camWorld = camera.node.getWorldTransform();
    center.set(0, 0, 0);
    for (let i = 0; i < 8; i++) {
        camWorld.transformPoint(pts[i], corners[i]);
        center.add(corners[i]);
    }
    center.mulScalar(1 / 8);
    let radius = 0;
    for (let i = 0; i < 8; i++) {
        radius = Math.max(radius, _tmp.sub2(corners[i], center).length());
    }
    return radius;
};

/**
 * Clips a convex polygon (flat xyz triplets) to the half-space `dot(n, p) <= d`.
 *
 * @param {number[]} src - Input polygon.
 * @param {number[]} dst - Output polygon, overwritten.
 * @param {Float32Array} planes - Packed planes (nx, ny, nz, d).
 * @param {number} o - Offset of the plane in `planes`.
 */
const clipPolygon = (src, dst, planes, o) => {
    dst.length = 0;
    const nx = planes[o], ny = planes[o + 1], nz = planes[o + 2], d = planes[o + 3];
    const n = src.length / 3;
    for (let i = 0; i < n; i++) {
        const a = i * 3;
        const b = ((i + 1) % n) * 3;
        const da = nx * src[a] + ny * src[a + 1] + nz * src[a + 2] - d;
        const db = nx * src[b] + ny * src[b + 1] + nz * src[b + 2] - d;
        if (da <= 0) dst.push(src[a], src[a + 1], src[a + 2]);
        if ((da <= 0) !== (db <= 0)) {
            const t = da / (da - db);
            dst.push(src[a] + (src[b] - src[a]) * t, src[a + 1] + (src[b + 1] - src[a + 1]) * t, src[a + 2] + (src[b + 2] - src[a + 2]) * t);
        }
    }
};

/**
 * Projects a point onto the light-space plane (axes {@link _axisU}, {@link _axisV}), appends it
 * to the hull points and returns its depth along {@link _lightDir}.
 *
 * @param {number} x - X.
 * @param {number} y - Y.
 * @param {number} z - Z.
 * @returns {number} The point's depth along the light.
 */
const addHullPoint = (x, y, z) => {
    _hullPoints.push(
        _axisU.x * x + _axisU.y * y + _axisU.z * z,
        _axisV.x * x + _axisV.y * y + _axisV.z * z
    );
    return _lightDir.x * x + _lightDir.y * y + _lightDir.z * z;
};

// orders projected hull points by x, then y
const compareHullPoints = (a, b) => (_hullPoints[a * 2] - _hullPoints[b * 2]) || (_hullPoints[a * 2 + 1] - _hullPoints[b * 2 + 1]);

// z of the cross product of (a - o) and (b - o): positive for a counter-clockwise turn
const hullCross = (o, a, b) => (_hullPoints[a * 2] - _hullPoints[o * 2]) * (_hullPoints[b * 2 + 1] - _hullPoints[o * 2 + 1]) -
    (_hullPoints[a * 2 + 1] - _hullPoints[o * 2 + 1]) * (_hullPoints[b * 2] - _hullPoints[o * 2]);

/**
 * Builds the convex hull of the projected {@link _hullPoints} into {@link _hull}, counter-clockwise
 * (Andrew's monotone chain).
 */
const buildHull = () => {
    const count = _hullPoints.length / 2;
    _hullOrder.length = count;
    for (let i = 0; i < count; i++) _hullOrder[i] = i;
    _hullOrder.sort(compareHullPoints);
    _hull.length = 0;
    for (let pass = 0; pass < 2; pass++) {
        const start = _hull.length;
        for (let k = 0; k < count; k++) {
            const i = _hullOrder[pass === 0 ? k : count - 1 - k];
            while (_hull.length >= start + 2 && hullCross(_hull[_hull.length - 2], _hull[_hull.length - 1], i) <= 0) _hull.pop();
            _hull.push(i);
        }
        _hull.pop();
    }
};

/**
 * Writes the planes of the region a directional light can carry a caster's shadow into: the
 * box swept to infinity along the light. Its faces are the box faces that do not face away
 * from the light, plus a plane through each silhouette edge (between a lit and an unlit face),
 * parallel to the light.
 *
 * @param {Vec3} min - Box minimum.
 * @param {Vec3} max - Box maximum.
 * @param {Vec3} dir - Normalized light travel direction.
 * @param {Float32Array} out - Receives packed planes (nx, ny, nz, d), `dot(n, p) <= d` inside.
 * @returns {number} The number of planes written.
 */
const sweptBoxPlanes = (min, max, dir, out) => {
    // face f = axis * 2 + side, outward normal +-axis; its facing is the sign of dot(n, dir)
    const facing = _facing;
    facing[0] = -dir.x;
    facing[1] = dir.x;
    facing[2] = -dir.y;
    facing[3] = dir.y;
    facing[4] = -dir.z;
    facing[5] = dir.z;
    let count = 0;
    for (let f = 0; f < 6; f++) {
        if (facing[f] > 0) continue; // faces away from the light: the sweep passes through it
        const axis = f >> 1;
        const sign = (f & 1) ? 1 : -1;
        const o = count++ * 4;
        out[o] = axis === 0 ? sign : 0;
        out[o + 1] = axis === 1 ? sign : 0;
        out[o + 2] = axis === 2 ? sign : 0;
        const bound = (f & 1) ? max : min;
        out[o + 3] = sign * (axis === 0 ? bound.x : axis === 1 ? bound.y : bound.z);
    }
    for (const [ia, ib, fa, fb] of BOX_EDGES) {
        if ((facing[fa] < 0) === (facing[fb] < 0) || facing[fa] === 0 || facing[fb] === 0) continue;
        const ca = BOX_CORNERS[ia];
        const cb = BOX_CORNERS[ib];
        _edgeA.set(ca[0] ? max.x : min.x, ca[1] ? max.y : min.y, ca[2] ? max.z : min.z);
        _edgeB.set(cb[0] ? max.x : min.x, cb[1] ? max.y : min.y, cb[2] ? max.z : min.z);
        _faceNormal.sub2(_edgeB, _edgeA).cross(_faceNormal, dir).normalize();
        // orient outwards: away from the box centre
        const cx = (min.x + max.x) * 0.5 - _edgeA.x;
        const cy = (min.y + max.y) * 0.5 - _edgeA.y;
        const cz = (min.z + max.z) * 0.5 - _edgeA.z;
        if (_faceNormal.x * cx + _faceNormal.y * cy + _faceNormal.z * cz > 0) _faceNormal.mulScalar(-1);
        const o = count++ * 4;
        out[o] = _faceNormal.x;
        out[o + 1] = _faceNormal.y;
        out[o + 2] = _faceNormal.z;
        out[o + 3] = _faceNormal.dot(_edgeA);
    }
    return count;
};

/**
 * Projects onto the light-space plane every vertex of a view slice intersected with the
 * receiver region (the world box swept along the light): the slice faces clipped to the region,
 * plus the box corners inside the slice (a box wholly inside the slice clips every slice face
 * away). Fills {@link _hullPoints}.
 *
 * @param {Vec3[]} corners - The slice's eight world-space corners (see {@link sliceSphere}).
 * @param {Vec3} center - A point inside the slice.
 * @param {number} radius - The slice's bounding radius, scaling the corner test's tolerance.
 * @param {Vec3} worldMin - World box minimum.
 * @param {Vec3} worldMax - World box maximum.
 * @returns {number} The largest depth along the light of the projected points.
 */
const projectSliceRegion = (corners, center, radius, worldMin, worldMax) => {
    const regionCount = sweptBoxPlanes(worldMin, worldMax, _lightDir, _regionPlanes);
    _hullPoints.length = 0;
    let maxDepth = -Infinity;
    for (let f = 0; f < 6; f++) {
        const face = SLICE_FACES[f];
        let src = _polyA;
        let dst = _polyB;
        src.length = 0;
        for (const index of face) {
            const p = corners[index];
            src.push(p.x, p.y, p.z);
        }

        // the face's outward plane, for the corner test below
        const p0 = corners[face[0]];
        _faceNormal.cross(_edgeA.sub2(corners[face[1]], p0), _edgeB.sub2(corners[face[2]], p0)).normalize();
        if (_faceNormal.dot(_tmp.sub2(center, p0)) > 0) _faceNormal.mulScalar(-1);
        _slicePlanes[f * 4] = _faceNormal.x;
        _slicePlanes[f * 4 + 1] = _faceNormal.y;
        _slicePlanes[f * 4 + 2] = _faceNormal.z;
        _slicePlanes[f * 4 + 3] = _faceNormal.dot(p0);

        for (let r = 0; r < regionCount && src.length; r++) {
            clipPolygon(src, dst, _regionPlanes, r * 4);
            [src, dst] = [dst, src];
        }
        for (let i = 0; i < src.length; i += 3) maxDepth = Math.max(maxDepth, addHullPoint(src[i], src[i + 1], src[i + 2]));
    }
    const eps = 1e-4 * (radius + 1);
    for (const [cx, cy, cz] of BOX_CORNERS) {
        const x = cx ? worldMax.x : worldMin.x;
        const y = cy ? worldMax.y : worldMin.y;
        const z = cz ? worldMax.z : worldMin.z;
        let inside = true;
        for (let f = 0; f < 6 && inside; f++) {
            const o = f * 4;
            inside = _slicePlanes[o] * x + _slicePlanes[o + 1] * y + _slicePlanes[o + 2] * z <= _slicePlanes[o + 3] + eps;
        }
        if (inside) maxDepth = Math.max(maxDepth, addHullPoint(x, y, z));
    }
    return maxDepth;
};

/**
 * Writes the cull planes of a directional cascade's caster volume: the convex hull of the view
 * slice clipped to the receiver region, projected along the light - one plane per hull edge,
 * parallel to the light - plus one rejecting casters beyond the slice's farthest point along the
 * light. Planes are packed (nx, ny, nz, w), a sphere inside when `dot(n, c) + w >= -radius`.
 *
 * @param {Vec3[]} corners - The slice's eight world-space corners (see {@link sliceSphere}).
 * @param {Vec3} center - The slice's bounding-sphere center.
 * @param {number} radius - The slice's bounding-sphere radius.
 * @param {Vec3} lightDir - The light's travel direction.
 * @param {Vec3} worldMin - World box minimum.
 * @param {Vec3} worldMax - World box maximum.
 * @param {number} margin - World units every plane is pushed out by.
 * @param {Float32Array} out - Receives the planes.
 * @param {number} maxPlanes - The most planes `out` can take.
 * @returns {number} The planes written: 0 when the volume needs more than `maxPlanes` or is
 * degenerate (the caller then culls without it), and a single plane rejecting everything when
 * the slice sees nothing a caster can shadow.
 */
const casterVolumePlanes = (corners, center, radius, lightDir, worldMin, worldMax, margin, out, maxPlanes) => {
    // light-space axes: the light's travel direction and two perpendiculars
    _lightDir.copy(lightDir).normalize();
    _axisU.cross(Math.abs(_lightDir.y) < 0.99 ? Vec3.UP : Vec3.RIGHT, _lightDir).normalize();
    _axisV.cross(_lightDir, _axisU);

    const maxDepth = projectSliceRegion(corners, center, radius, worldMin, worldMax);
    if (_hullPoints.length < 6) {
        out.fill(0, 0, 4);
        out[3] = -1e30;
        return 1;
    }

    buildHull();
    const edges = _hull.length;
    if (edges < 3 || edges + 1 > maxPlanes) return 0;

    for (let e = 0; e < edges; e++) {
        const a = _hull[e];
        const b = _hull[(e + 1) % edges];
        const ax = _hullPoints[a * 2], ay = _hullPoints[a * 2 + 1];
        // inward (left) normal of a counter-clockwise edge
        let nx = -(_hullPoints[b * 2 + 1] - ay);
        let ny = _hullPoints[b * 2] - ax;
        const len = Math.hypot(nx, ny);
        if (!(len > 0)) return 0;
        nx /= len;
        ny /= len;
        const o = e * 4;
        out[o] = nx * _axisU.x + ny * _axisV.x;
        out[o + 1] = nx * _axisU.y + ny * _axisV.y;
        out[o + 2] = nx * _axisU.z + ny * _axisV.z;
        out[o + 3] = -(nx * ax + ny * ay) + margin;
    }
    const o = edges * 4;
    out[o] = -_lightDir.x;
    out[o + 1] = -_lightDir.y;
    out[o + 2] = -_lightDir.z;
    out[o + 3] = maxDepth + margin;
    return edges + 1;
};

export { casterVolumePlanes, sliceSphere };
