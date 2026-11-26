"""L-shaped pool house for 10 Mancini Drive: the IGES pool house bar along the pavers' north-west side, plus a wing
wrapping the pavers' south-west end for shade over an outdoor kitchen and dining table. Hipped roof (15°), glass doors
facing the pool, windows elsewhere, a galley kitchen and a 6-seat table inside.

Geometry is built in the paver frame (u along the pavers from the south-west end to the north-east wall, v across from
the house side to the pool house; data/paver_frame.json) and returned as triangle lists per material, in MGA + height.
"""
import json, math
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, box
from shapely import constrained_delaunay_triangles

DATA = Path(__file__).resolve().parent.parent / "data"

# ---- design (metres, paver frame)
BAR = (0.0, 15.82, 9.84, 17.95)        # u0, u1, v0, v1: the IGES pool house, widened to butt the pavers
WING_DEPTH = 6.0                      # wing beyond the pavers' south-west end (u < 0)
WING_V0 = 4.9                         # wing reaches across to about the middle of the pool end
SLAB = 0.07                           # slab top above ground
EAVE = 2.55                           # wall plate height above the slab
PITCH = math.radians(15)
OVER = 0.45                           # eave overhang
GABLE_U, GABLE_W = 7.55, 6.0          # through-gable over the bar, centred on the pool (u of the pool centre), 6 m wide
GABLE_PITCH = math.radians(30)        # steeper feature gable
CEIL_PITCH = math.radians(15)         # scissor truss: raked ceiling at half the roof pitch
T = 0.15                              # wall thickness

L_POLY = [(-WING_DEPTH, WING_V0), (0.0, WING_V0), (0.0, BAR[2]), (BAR[1], BAR[2]), (BAR[1], BAR[3]), (-WING_DEPTH, BAR[3])]


def openings():
    """per edge of L_POLY (edge i from vertex i to i+1): list of (s0, s1, z0, z1) along the edge from its start"""
    door, sill = (SLAB + 0.03, 2.4), (0.9, 2.2)
    return [
        [(0.5, 5.5, *door)],                                           # wing south-east end: glass sliders
        [(0.35, 4.6, *door)],                                          # wing facing the pool end: glass
        [(0.6, 4.9, *door), (5.5, 10.3, *door), (10.9, 15.2, *door)],  # bar facing the pool: glass sliders
        [(2.0, 6.1, *sill)],                                           # bar north-east end: window
        [(1.5, 4.5, 1.3, 2.2), (6.0, 9.0, 1.3, 2.2), (10.5, 13.5, 1.3, 2.2), (16.6, 21.2, 1.2, 2.2)],  # outer side, high
        [(3.0, 6.0, *sill), (7.0, 11.6, *sill)],                       # wing south-west end: windows by the table
    ]


class Mesh:
    def __init__(self):
        self.k = {}

    def tri(self, kind, a, b, c):
        self.k.setdefault(kind, []).extend([a, b, c])

    def quad(self, kind, a, b, c, d):
        self.tri(kind, a, b, c); self.tri(kind, a, c, d)

    def box(self, kind, origin, ex, ey, h0, h1):
        """box from a base parallelogram origin + s*ex + t*ey (s, t in 0..1) between heights h0 and h1"""
        o = np.array(origin, float); ex = np.array(ex, float); ey = np.array(ey, float)
        P = [o, o + ex, o + ex + ey, o + ey]
        lo = [(p[0], p[1], h0) for p in P]; hi = [(p[0], p[1], h1) for p in P]
        self.quad(kind, *hi); self.quad(kind, *lo[::-1])
        for i in range(4):
            j = (i + 1) % 4
            self.quad(kind, lo[i], lo[j], hi[j], hi[i])

    def poly(self, kind, ring, z, flip=False):
        g = Polygon(ring)
        for t in constrained_delaunay_triangles(g).geoms:
            c = list(t.exterior.coords)[:3]
            if Polygon(c).area < 1e-9:
                continue
            pts = [(x, y, z if not callable(z) else z(x, y)) for x, y in c]
            self.tri(kind, *(pts[::-1] if flip else pts))


def build():
    m = Mesh()
    Lg = Polygon(L_POLY)
    # slab
    m.poly("slab", L_POLY, SLAB)
    ring = L_POLY + [L_POLY[0]]
    for (a, b), (c, d) in zip(ring[:-1], ring[1:]):
        m.quad("slab", (a, b, 0.0), (c, d, 0.0), (c, d, SLAB), (a, b, SLAB))
    # walls with openings, glass and frames
    for i, ops in enumerate(openings()):
        p, q = np.array(L_POLY[i]), np.array(L_POLY[(i + 1) % len(L_POLY)])
        L = np.linalg.norm(q - p); e = (q - p) / L
        n = np.array([-e[1], e[0]])
        mid = (p + q) / 2
        if Lg.contains(Polygon([tuple(mid + n * 0.05 + e * 0.01), tuple(mid + n * 0.05 - e * 0.01), tuple(mid + n * 0.06)]).centroid):
            n = -n                                                     # n points outside
        inward = -n * T
        segs, s = [], 0.0
        for s0, s1, z0, z1 in sorted(ops):
            if s0 > s:
                segs.append((s, s0, SLAB, EAVE))
            segs.append((s0, s1, SLAB, z0)); segs.append((s0, s1, z1, EAVE))
            s = s1
        if s < L:
            segs.append((s, L, SLAB, EAVE))
        for s0, s1, z0, z1 in segs:
            if s1 - s0 > 1e-3 and z1 - z0 > 1e-3:
                m.box("wall", p + e * s0, e * (s1 - s0), inward, z0, z1)
        for s0, s1, z0, z1 in ops:
            a, b = p + e * s0 + inward / 2, p + e * s1 + inward / 2
            m.quad("glass", (*a, z0), (*b, z0), (*b, z1), (*a, z1))
            k = max(1, round((s1 - s0) / 1.5))
            for j in range(k + 1):                                     # mullions, head and sill
                c0 = p + e * (s0 + (s1 - s0) * j / k - 0.03) + inward * 0.35
                m.box("frame", c0, e * 0.06, inward * 0.3, z0, z1)
            for zz in (z0, z1 - 0.06):
                m.box("frame", p + e * s0 + inward * 0.35, e * (s1 - s0), inward * 0.3, zz, zz + 0.06)
    # roof: hipped bar (run through the corner) and hipped wing, plus a steeper through-gable over the middle of the bar,
    # centred on the pool. Each unit's roof is the lowest of its planes inside its rectangle; the combined roof is the
    # highest unit at each point, which gives the hips, the valleys and the gable meeting the hips properly.
    tp = math.tan(PITCH)
    ze = EAVE - OVER * tp
    gu0, gu1 = GABLE_U - GABLE_W / 2, GABLE_U + GABLE_W / 2
    tg = math.tan(GABLE_PITCH)

    def plane(ax, c, sgn, t):
        return (ax, c, sgn, t)

    def zf(pl):
        ax, c, sgn, t = pl
        z0 = EAVE - OVER * t
        return (lambda x, y: z0 + t * sgn * ((x if ax == "u" else y) - c))

    def hip_unit(r):
        u0, u1, v0, v1 = r
        h = min(u1 - u0, v1 - v0) / 2
        if (u1 - u0) >= (v1 - v0):
            A, B = (u0 + h, (v0 + v1) / 2), (u1 - h, (v0 + v1) / 2)
            F = [([(u0, v0), (u1, v0), B, A], plane("v", v0, 1, tp)), ([(u1, v1), (u0, v1), A, B], plane("v", v1, -1, tp)),
                 ([(u0, v1), (u0, v0), A], plane("u", u0, 1, tp)), ([(u1, v0), (u1, v1), B], plane("u", u1, -1, tp))]
        else:
            A, B = ((u0 + u1) / 2, v0 + h), ((u0 + u1) / 2, v1 - h)
            F = [([(u1, v0), (u1, v1), B, A], plane("u", u1, -1, tp)), ([(u0, v1), (u0, v0), A, B], plane("u", u0, 1, tp)),
                 ([(u0, v0), (u1, v0), A], plane("v", v0, 1, tp)), ([(u1, v1), (u0, v1), B], plane("v", v1, -1, tp))]
        return box(u0, v0, u1, v1), [(Polygon(f), pl) for f, pl in F]

    def gable_unit(r):
        u0, u1, v0, v1 = r
        um = (u0 + u1) / 2
        return box(u0, v0, u1, v1), [(Polygon([(u0, v0), (um, v0), (um, v1), (u0, v1)]), plane("u", u0, 1, tg)),
                                     (Polygon([(um, v0), (u1, v0), (u1, v1), (um, v1)]), plane("u", u1, -1, tg))]

    def higher(pl_other, pl_self, bound):
        """part of a box around `bound` where plane pl_other is above plane pl_self (half-plane clip)"""
        zo, zs = zf(pl_other), zf(pl_self)
        f0 = zo(0, 0) - zs(0, 0); fu = zo(1, 0) - zs(1, 0) - f0; fv = zo(0, 1) - zs(0, 1) - f0
        x0, y0, x1, y1 = bound.bounds
        big = box(x0 - 5, y0 - 5, x1 + 5, y1 + 5)
        if abs(fu) < 1e-12 and abs(fv) < 1e-12:
            return big if f0 > 0 else Polygon()
        pts = list(big.exterior.coords)[:-1]
        out = []
        val = lambda q: f0 + fu * q[0] + fv * q[1]
        for k in range(len(pts)):
            P, Q = pts[k], pts[(k + 1) % len(pts)]
            vp, vq = val(P), val(Q)
            if vp > 0: out.append(P)
            if (vp > 0) != (vq > 0):
                t = vp / (vp - vq); out.append((P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])))
        return Polygon(out) if len(out) >= 3 else Polygon()

    units = [hip_unit((-WING_DEPTH - OVER, BAR[1] + OVER, BAR[2] - OVER, BAR[3] + OVER)),
             hip_unit((-WING_DEPTH - OVER, OVER, WING_V0 - OVER, BAR[3] + OVER)),
             gable_unit((gu0 - OVER, gu1 + OVER, BAR[2] - OVER, BAR[3] + OVER))]
    for i, (Ri, Fi) in enumerate(units):
        for F, pl in Fi:
            vis = F
            for j, (Rj, Fj) in enumerate(units):
                if j == i or not Rj.intersects(F):
                    continue
                cover = Rj
                for G, gl in Fj:
                    cover = cover.intersection(higher(gl, pl, Rj))
                vis = vis.difference(cover)
            vis = vis.buffer(0)
            for part in getattr(vis, "geoms", [vis]):
                if part.geom_type == "Polygon" and part.area > 1e-4:
                    m.poly("roof", list(part.exterior.coords)[:-1], zf(pl))
    outer = units[0][0].union(units[1][0])
    oc = list(outer.exterior.coords)
    for (a_, b_), (c_, d_) in zip(oc[:-1], oc[1:]):                   # fascia along the hip eaves
        m.quad("fascia", (a_, b_, ze - 0.2), (c_, d_, ze - 0.2), (c_, d_, ze), (a_, b_, ze))
    soff = outer.difference(Lg)
    for part in getattr(soff, "geoms", [soff]):                        # eave lining
        m.poly("soffit", list(part.exterior.coords)[:-1], ze - 0.2, flip=True)
    # gable ends: barge boards along the rakes, glazed triangle to the pool (up to the raked ceiling), clad above it
    gm = GABLE_U
    apex = EAVE + tg * GABLE_W / 2                                     # roof apex at the wall line
    capex = EAVE + math.tan(CEIL_PITCH) * GABLE_W / 2                  # scissor-truss ceiling apex
    for vw, glazed in ((BAR[2], True), (BAR[3], False)):
        vb = vw - OVER if vw == BAR[2] else vw + OVER
        zb0 = EAVE - OVER * tg
        for (x0, z0), (x1, z1) in (((gu0 - OVER, zb0), (gm, apex + 0.0)), ((gm, apex), (gu1 + OVER, zb0))):
            m.quad("fascia", (x0, vb, z0 - 0.22), (x1, vb, z1 - 0.22), (x1, vb, z1 + 0.02), (x0, vb, z0 + 0.02))
        if glazed:
            m.tri("glass", (gu0, vw, EAVE), (gu1, vw, EAVE), (gm, vw, capex))
            m.tri("wall", (gu0, vw, EAVE), (gm, vw, apex), (gm, vw, capex))
            m.tri("wall", (gm, vw, capex), (gm, vw, apex), (gu1, vw, EAVE))
            for x in (gu0 + GABLE_W / 4, gm, gu1 - GABLE_W / 4):        # mullions in the gable glass
                ztop = EAVE + math.tan(CEIL_PITCH) * min(x - gu0, gu1 - x)
                m.box("frame", (x - 0.03, vw - 0.06), (0.06, 0), (0, 0.06), EAVE, ztop)
            m.box("frame", (gu0, vw - 0.06), (GABLE_W, 0), (0, 0.06), EAVE - 0.04, EAVE + 0.04)
        else:
            m.tri("wall", (gu0, vw, EAVE), (gu1, vw, EAVE), (gm, vw, apex))
    # ceilings: flat at the plate, except a scissor-truss raked ceiling under the gable
    flat = Lg.difference(box(gu0, BAR[2], gu1, BAR[3]))
    for part in getattr(flat, "geoms", [flat]):
        m.poly("soffit", list(part.exterior.coords)[:-1], EAVE, flip=True)
    tc = math.tan(CEIL_PITCH)
    for x0, x1 in ((gu0, gm), (gm, gu1)):
        z0 = EAVE + tc * min(x0 - gu0, gu1 - x0); z1 = EAVE + tc * min(x1 - gu0, gu1 - x1)
        m.quad("soffit", (x0, BAR[2], z0), (x1, BAR[2], z1), (x1, BAR[3], z1), (x0, BAR[3], z0))
    # galley kitchen along the outer wall in the wing, with an island; 6-seat table
    m.box("cabinet", (-5.7, BAR[3] - T - 0.6), (5.4, 0), (0, 0.6), SLAB, SLAB + 0.86)
    m.box("bench", (-5.72, BAR[3] - T - 0.62), (5.44, 0), (0, 0.62), SLAB + 0.86, SLAB + 0.9)
    m.box("cabinet", (-4.8, 15.4), (3.6, 0), (0, 0.9), SLAB, SLAB + 0.86)
    m.box("bench", (-4.85, 15.35), (3.7, 0), (0, 1.0), SLAB + 0.86, SLAB + 0.9)
    tu, tv = -2.8, 9.5
    m.box("table", (tu - 0.5, tv - 1.2), (1.0, 0), (0, 2.4), SLAB + 0.72, SLAB + 0.76)
    for du in (-0.45, 0.4):
        for dv in (-1.1, 1.05):
            m.box("table", (tu + du, tv + dv), (0.06, 0), (0, 0.06), SLAB, SLAB + 0.72)
    for side in (-1, 1):
        for dv in (-0.8, 0.0, 0.8):
            cu = tu + side * 0.85
            m.box("chair", (cu - 0.22, tv + dv - 0.22), (0.44, 0), (0, 0.44), SLAB + 0.42, SLAB + 0.47)
            bu = cu + side * 0.2
            m.box("chair", (bu - 0.03, tv + dv - 0.22), (0.06, 0), (0, 0.44), SLAB + 0.47, SLAB + 0.92)
            for a in (-0.18, 0.16):
                for b in (-0.18, 0.16):
                    m.box("chair", (cu + a, tv + dv + b), (0.03, 0), (0, 0.03), SLAB, SLAB + 0.42)
    ridge = ze + tp * max(BAR[3] - BAR[2] + 2 * OVER, WING_DEPTH + 2 * OVER) / 2
    ridge = max(ridge, EAVE + tg * GABLE_W / 2)
    info = {"footprint_m2": round(Lg.area, 1), "eave": EAVE, "pitch_deg": math.degrees(PITCH), "ridge": round(ridge, 2),
            "bar": f"{BAR[1] - BAR[0]:.1f} x {BAR[3] - BAR[2]:.1f} m", "ceiling_peak": round(EAVE + math.tan(CEIL_PITCH) * GABLE_W / 2, 2), "wing": f"{WING_DEPTH:.1f} x {BAR[3] - WING_V0:.2f} m"}
    return m.k, Lg, info


def to_world(kinds, ground_z):
    """paver frame (u, v, z above ground) -> MGA easting, northing, AHD"""
    F = json.load(open(DATA / "paver_frame.json"))
    o, u, v = np.array(F["origin"]), np.array(F["u"]), np.array(F["v"])
    out = {}
    for k, pts in kinds.items():
        P = np.array(pts, float)
        E = o[0] + P[:, 0] * u[0] + P[:, 1] * v[0]
        N = o[1] + P[:, 0] * u[1] + P[:, 1] * v[1]
        out[k] = np.column_stack([E, N, P[:, 2] + ground_z])
    return out


def footprint_world():
    F = json.load(open(DATA / "paver_frame.json"))
    o, u, v = np.array(F["origin"]), np.array(F["u"]), np.array(F["v"])
    return Polygon([tuple(o + a * u + b * v) for a, b in L_POLY])


if __name__ == "__main__":
    k, Lg, info = build()
    print(info, {kk: len(vv) // 3 for kk, vv in k.items()})
