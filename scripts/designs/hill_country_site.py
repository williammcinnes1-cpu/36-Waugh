"""Site fit for the "Hill Country Stone" design (hill_country) on Lot 624 DP821594.

  python3 scripts/designs/hill_country_site.py ground  -> designed ground over the WHOLE lot (two terraces), writes
                                                     data/3d/house_massing.json (ground only), the render layout
                                                     data/3d/designs/hill_country_layout.json, and refreshes
                                                     data/3d/render_scene.json via build_3d_render
  python3 scripts/designs/hill_country_site.py plan    -> site plan PNG over the aerial (designs/hill_country_plan.png)

Street frame: u along the frontage chord (SW end = 0), n into the lot, heights in m AHD.
Every part of the lot is designed ground: a planted bank behind the low street wall, the house/courtyard pad
(terrace 157.45) and an upper garden terrace (158.8) behind the pool that runs out to the diagonal rear boundary,
including both rear corners. Retaining walls sit in the 0.3 m strip just inside the boundary.
All dimensions are design assumptions for visualisation, not a design for approval.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import shapely
from PIL import Image
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import house_massing as hm  # noqa: E402  (lot, frame, grid constants)

OUT = ROOT / "data" / "3d"
LOT, A, U, NV = hm.LOT, hm.A, hm.U, hm.NV
E0, N1, NCELL = hm.E0, hm.N1, hm.NCELL
FFL, PAD, GARAGE_FL = hm.FFL, hm.PAD, hm.GARAGE_FL
UPPER = 158.8      # upper garden terrace behind the pool (FFL + 1.2)

# ---- design (street frame, metres) ---------------------------------------------------------------
DESIGN = {
    "west": (0.6, 7.2, 7.2, 29.0),       # stone gable wing, gable to the street; attic storey (main suite) at the front
    "main": (7.2, 16.4, 11.0, 20.0),     # long low standing-seam range: entry, living, dining
    "east": (16.4, 23.0, 7.5, 28.0),     # single-storey stone gable wing over the garage: kitchen / family
    "entry": (10.2, 13.0, 8.3, 11.0),    # small centre gable with the arched entry
    "bay": (9.8, 13.4, 20.0, 21.0),      # north cross-gable with the tall arched window
    "loggia": (7.2, 16.4, 8.3, 11.0),    # timber-post loggia across the middle (behind the 6 m line)
    "bay_e": (18.4, 21.0, 6.9, 7.5),     # projecting stone bay window on the east gable (articulation)
    "pier_w": (0.6, 1.5, 6.9, 7.2),      # stone corner piers on the west gable (articulation)
    "pier_e": (6.3, 7.2, 6.9, 7.2),
    "garage": (16.4, 22.8, 7.5, 15.0),   # street level, under the east wing; door face at n = 7.5
    "apron": (16.4, 22.8, 0.0, 7.5),     # near-flat driveway apron, planted retaining walls either side
    "planter": (14.6, 16.4, 1.6, 7.5),   # stepped planter west of the apron
    "pool": (7.8, 15.4, 24.2, 28.2),
    "stair": (10.6, 12.6, 1.0, 5.0),     # steps from the street gate up to the gravel path
    "upstair": (10.4, 12.8, 32.5, 35.1), # steps from the pool deck to the upper garden
    "pergola": (19.0, 25.0, 36.0, 40.0), # timber pergola on the upper garden
    "terrace_n": 32.5,                   # pad / upper terrace step line
    "bank_top": 5.0,
}
# roof geometry shared with the renderer (heights above FFL)
ROOFS = {"west": {"eave": 3.0, "pitch": 53.0}, "east": {"eave": 3.0, "pitch": 45.0},
         "main": {"ridge_n": 15.5, "ridge_z": 5.6, "front_pitch": 22.0, "rear_pitch": 30.0},
         "entry": {"eave": 2.9, "pitch": 45.0}, "bay": {"eave": 3.3, "pitch": 50.0}}
DESIGN["roofs"] = ROOFS
PLANTER_Z = 156.1
WALL_TOP = 156.4   # low street wall (about 0.9 m above the footpath)
BANK_TOP = DESIGN["bank_top"]
TN = DESIGN["terrace_n"]
INSET = 0.3        # retaining walls occupy the strip between LOT.buffer(-0.3) and the boundary
O = LOT.buffer(-INSET, join_style=2)


def frame_poly(u0, u1, n0, n1):
    return Polygon([tuple(A + U * u + NV * n) for u, n in ((u0, n0), (u1, n0), (u1, n1), (u0, n1))])


def to_frame(e, n):
    d = np.array([e, n]) - A
    return float(d @ U), float(d @ NV)


def frame_geom(g):
    """MGA geometry -> street-frame geometry"""
    return shapely.transform(g, lambda c: np.column_stack([(c - A) @ U, (c - A) @ NV]))


OF = frame_geom(O)            # lot inset, street frame
LOTF = frame_geom(LOT)


def to_en(u, n):
    return A[0] + U[0] * u + NV[0] * n, A[1] + U[1] * u + NV[1] * n


DEM = np.array(Image.open(OUT / "dem_240m.tif"), dtype=np.float64)


def natural(u, n):
    """bilinear natural ground (m AHD) at street-frame points"""
    e, nn = to_en(np.asarray(u, float), np.asarray(n, float))
    x, y = e - E0 - 0.5, N1 - nn - 0.5
    x = np.clip(x, 0, NCELL - 1.001)
    y = np.clip(y, 0, NCELL - 1.001)
    i, j = x.astype(int), y.astype(int)
    fx, fy = x - i, y - j
    return (DEM[j, i] * (1 - fx) + DEM[j, i + 1] * fx) * (1 - fy) + (DEM[j + 1, i] * (1 - fx) + DEM[j + 1, i + 1] * fx) * fy


def band_level(u, n):
    """the terrace level that governs (u, n), ignoring garage/stairs/pool"""
    u, n = np.asarray(u, float), np.asarray(n, float)
    z = np.where(n >= TN, UPPER, PAD)
    return np.where(n < BANK_TOP, np.interp(n, [2.2, BANK_TOP], [WALL_TOP, PAD]), z)


def design_z(u, n, nat):
    """designed ground (m AHD) at street-frame points; nat = natural ground there"""
    u, n = np.asarray(u, float), np.asarray(n, float)
    in_lot = shapely.contains_xy(LOTF, u, n)
    in_o = shapely.contains_xy(OF, u, n)
    z = nat.copy()
    g = DESIGN["garage"]
    s = DESIGN["stair"]
    us = DESIGN["upstair"]
    p = DESIGN["pool"]
    terr = in_o & (n >= 2.2)
    z = np.where(terr, band_level(u, n), z)
    # stair cut from the gate up to the path
    m = in_lot & (s[0] - 0.5 <= u) & (u <= s[1] + 0.5) & (n >= s[2]) & (n <= s[3])
    z = np.where(m, np.interp(n, [s[2], s[3]], [155.6, PAD]), z)
    # upper stair sits on the pad level (the steps are built on it)
    m = (us[0] - 0.4 <= u) & (u <= us[1] + 0.4) & (n >= TN) & (n <= us[3])
    z = np.where(m, PAD, z)
    # garage floor (under the east wing) and the near-flat driveway apron from the street boundary
    m = in_lot & (g[0] <= u) & (u <= g[1] + 0.4) & (n >= g[2]) & (n < g[3])
    z = np.where(m, GARAGE_FL, z)
    m = in_lot & (16.0 <= u) & (u <= g[1] + 0.4) & (n < g[2])
    z = np.where(m, nat + (GARAGE_FL - nat) * np.clip(n / 2.0, 0, 1), z)
    pl = DESIGN["planter"]
    m = in_lot & (pl[0] <= u) & (u < 16.0) & (n >= pl[2]) & (n < pl[3])
    z = np.where(m, PLANTER_Z, z)
    m = in_lot & (u > g[1] + 0.4) & (n < g[2])            # planted strip beside the apron stays natural
    z = np.where(m, nat, z)
    # pool shell
    m = (p[0] - 0.25 <= u) & (u <= p[1] + 0.25) & (p[2] - 0.25 <= n) & (n <= p[3] + 0.25)
    z = np.where(m, PAD - 1.8, z)
    return z


def designed_ground():
    """1 m grid (rows north->south as dem_240m.tif) of designed ground"""
    cc, rr_ = np.meshgrid(np.arange(NCELL), np.arange(NCELL))
    e, nn = E0 + cc + 0.5, N1 - rr_ - 0.5
    d = np.stack([e - A[0], nn - A[1]], -1)
    u, n = d @ U, d @ NV
    z = design_z(u.ravel(), n.ravel(), DEM.ravel()).reshape(DEM.shape)
    return z, u, n


# ---- polygons in the street frame -------------------------------------------------------------------
def rect(k, pad=0.0):
    u0, u1, n0, n1 = DESIGN[k]
    return box(u0 - pad, n0 - pad, u1 + pad, n1 + pad)


def house_poly():
    return unary_union([rect(k) for k in ("west", "main", "east", "entry", "bay", "loggia", "bay_e", "pier_w", "pier_e")])


def zones():
    """ground-cover polygons by material, all inside the lot inset"""
    big = box(-20, 2.2, 45, 60)
    o2 = OF.intersection(big)
    s, us = DESIGN["stair"], DESIGN["upstair"]
    excl = unary_union([house_poly(), rect("garage"), rect("pool", 0.3), box(s[0] - 0.35, 0, s[1] + 0.35, s[3] + 0.05),
                        box(14.6, 0, 30, 7.5),
                        box(us[0] - 0.3, TN - 0.2, us[1] + 0.3, us[3])])
    bank = o2.intersection(box(-20, 2.2, 45, BANK_TOP)).difference(excl)
    pad = o2.intersection(box(-20, BANK_TOP, 45, TN)).difference(excl)
    upper = o2.intersection(box(-20, TN, 45, 60)).difference(excl)
    paving = pad.intersection(unary_union([box(7.2, 20.0, 16.4, TN), box(16.4, 28.0, 27.5, TN), box(-5, 29.0, 6.8, TN)]))
    front = pad.intersection(box(-20, BANK_TOP, 45, 8.6)).difference(paving)
    path = front.intersection(box(10.45, BANK_TOP, 12.75, 8.6))
    beds = front.difference(path)
    side = pad.difference(paving).difference(front)
    ring_ = upper.difference(OF.buffer(-1.6))
    perg = upper.intersection(box(18.4, 35.4, 25.6, 40.6))
    upath = upper.intersection(unary_union([box(10.4, us[3], 12.8, 37.0), box(10.4, 36.0, 18.4, 37.0)]))
    ubeds = ring_.difference(perg).difference(upath)
    lawn = upper.difference(ring_).difference(perg).difference(upath)
    return {"bank": bank, "paving": paving, "gravel": unary_union([path, side, perg, upath]), "beds": beds,
            "ubeds": ubeds, "lawn": lawn}


def tris(geom):
    """triangles (u, n, z AHD) of a polygon, z from the terrace levels"""
    out = []
    for t in shapely.constrained_delaunay_triangles(geom).geoms:
        c = np.array(t.exterior.coords)[:3]
        if Polygon(c).area < 1e-4:
            continue
        z = band_level(c[:, 0], c[:, 1])
        out.extend(np.round(np.column_stack([c, z - FFL]), 3).ravel().tolist())
    return out


# neighbouring houses read off the aerial (street frame quads), for context: walls, roof colour, wall height
NEIGHBOURS = [
    {"quad": [[26.6, 16.6], [38.5, 10.2], [43.0, 21.0], [28.1, 28.6]], "wall": "#f1efe9", "roof": "#a4503a", "h": 2.7, "pitch": 24},
    {"quad": [[-30.0, 7.0], [-12.0, -2.5], [-7.2, 10.2], [-25.6, 19.6]], "wall": "#e9e4da", "roof": "#6d6f70", "h": 2.7, "pitch": 22},
    {"quad": [[1.0, -36.0], [19.0, -36.0], [19.0, -25.0], [1.0, -25.0]], "wall": "#f2f0ea", "roof": "#3c3f42", "h": 5.9, "pitch": 24},
]
FENCES = []   # Colorbond side fences (u0, n0, u1, n1, z0, z1, side) on top of / beside the boundary walls


def walls():
    """retaining walls: lot perimeter (in the 0.3 m boundary strip) and the pad/upper terrace step"""
    res = []
    ring_ = list(OF.exterior.coords)
    sign = 1 if LinearRingArea(ring_) > 0 else -1     # CCW -> outward normal is to the right
    for (ua, na), (ub, nb) in zip(ring_, ring_[1:]):
        L = np.hypot(ub - ua, nb - na)
        k = max(1, int(np.ceil(L / 1.5)))
        tu, tn = (ub - ua) / L, (nb - na) / L
        ou, on = tn * sign * INSET, -tu * sign * INSET
        for i in range(k):
            u0, n0 = ua + (ub - ua) * i / k, na + (nb - na) * i / k
            u1, n1 = ua + (ub - ua) * (i + 1) / k, na + (nb - na) * (i + 1) / k
            um, nm = (u0 + u1) / 2, (n0 + n1) / 2
            if nm < 2.3 or (nm < 7.6 and um > 20):
                continue  # the street wall handles the front; the strip beside the apron stays natural
            lvl = float(band_level(um - ou, nm - on))
            nat = float(natural(um + ou, nm + on))
            if nat > lvl + 0.05:
                bot, top = lvl - 0.35, nat + 0.15
            else:
                bot, top = nat - 0.35, lvl + 0.45
            res.append([round(x, 3) for x in (u0, n0, u1, n1, ou, on, bot - FFL, top - FFL)])
            dsw, dne = SIDE_SW.distance(shapely.Point(um, nm)), SIDE_NE.distance(shapely.Point(um, nm))
            if nm >= 6.0 and min(dsw, dne) < 0.5 and REAR.distance(shapely.Point(um, nm)) > 0.5:
                FENCES.append([round(x, 3) for x in (u0 + ou * 0.9, n0 + on * 0.9, u1 + ou * 0.9, n1 + on * 0.9,
                                                     max(lvl, nat) - 0.05 - FFL, max(lvl, nat) + 1.8 - FFL)] + ["sw" if dsw < dne else "ne"])
    # terrace step wall (with a gap for the steps)
    line = LineString([(-20, TN), (45, TN)]).intersection(OF)
    ua, ub = line.bounds[0], line.bounds[2]
    us = DESIGN["upstair"]
    for a, b in ((ua, us[0] - 0.3), (us[1] + 0.3, ub)):
        k = max(1, int(np.ceil((b - a) / 1.5)))
        for i in range(k):
            u0, u1 = a + (b - a) * i / k, a + (b - a) * (i + 1) / k
            res.append([round(u0, 3), TN - 0.15, round(u1, 3), TN - 0.15, 0, 0.3, round(PAD - 0.35 - FFL, 3), round(UPPER + 0.45 - FFL, 3)])
    # stepped planted walls beside the driveway apron, and the cross wall where the NE side path meets the apron strip
    W = lambda *a: res.append([round(x, 3) for x in a])
    pl = DESIGN["planter"]
    for n0 in np.arange(pl[2], pl[3] - 0.01, 1.475):
        n1 = min(n0 + 1.475, pl[3])
        lv = float(band_level(14.4, (n0 + n1) / 2))
        W(14.6, n0, 14.6, n1, 0.4, 0, PLANTER_Z - 0.35 - FFL, lv + 0.12 - FFL)          # upper planter wall
        W(16.0, n0, 16.0, n1, 0.4, 0, GARAGE_FL - 0.35 - FFL, PLANTER_Z + 0.15 - FFL)   # lower wall to the apron
    W(14.6, pl[3], 16.4, pl[3], 0, 0.4, PLANTER_Z - 0.35 - FFL, PAD + 0.12 - FFL)
    ub = float(23.2 + 0.0)
    une = 22.9 + 6.9 * 7.5 / 44.5
    W(23.2, 7.5, une - 0.05, 7.5, 0, 0.3, float(natural(23.6, 7.0)) - 0.35 - FFL, PAD + 0.45 - FFL)
    return res


def LinearRingArea(c):
    c = np.array(c)
    return 0.5 * np.sum(c[:-1, 0] * c[1:, 1] - c[1:, 0] * c[:-1, 1])


def fine_patch(step=0.25):
    """0.25 m terrain patch over the lot: exact terraces inside, natural outside, sunk under the walls"""
    us = np.arange(-9.0, 34.0 + 1e-6, step)
    ns = np.arange(-3.0, 49.0 + 1e-6, step)
    uu, nn = np.meshgrid(us, ns)
    u, n = uu.ravel(), nn.ravel()
    nat = natural(u, n)
    z = design_z(u, n, nat)
    in_lot = shapely.contains_xy(LOTF, u, n)
    in_o = shapely.contains_xy(OF, u, n)
    strip = in_lot & ~in_o & (n >= 2.0)
    z = np.where(strip, np.minimum(nat, band_level(u, n)) - 0.3, z)
    # outside the lot keep the natural ground, a hair low so the 1 m terrain wins where both exist
    z = np.where(in_lot, z, nat - 0.03)
    return {"u0": -9.0, "n0": -3.0, "step": step, "nu": len(us), "nn": len(ns),
            "z": np.round(z - FFL, 3).tolist()}


def footprint():
    fp = unary_union([rect(k) for k in ("west", "main", "east", "entry", "bay", "garage", "bay_e", "pier_w", "pier_e")])
    return fp.area


# ---- DCP checks: setbacks (perpendicular to each boundary) and ridge height above natural ground ---------
def street_arc():
    pts = [c for c in LOTF.exterior.coords if c[1] < 1.2 and -0.5 < c[0] < 23.4]
    return LineString(sorted(pts))


SIDE_SW = LineString([(0, 0), (-5.1, 32.5)])
SIDE_NE = LineString([(22.9, 0), (29.8, 44.5)])
REAR = LineString([(-5.1, 32.5), (29.8, 44.5)])


def perp(line, geom):
    """distance from geom to the infinite line through `line` (perpendicular to the boundary)"""
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    L = np.hypot(x1 - x0, y1 - y0)
    pts = np.array(geom.exterior.coords) if hasattr(geom, "exterior") else np.vstack([np.array(g.exterior.coords) for g in geom.geoms])
    return float(np.min(np.abs((pts[:, 0] - x0) * (y1 - y0) - (pts[:, 1] - y0) * (x1 - x0)) / L))


def ridges():
    """(name, ridge RL, height above lowest natural ground under the ridge line incl. capping)"""
    out = []
    W, E = DESIGN["west"], DESIGN["east"]
    for k, (u0, u1, n0, n1), og in (("west wing (attic storey)", W, 0.3), ("east wing", E, 0.3)):
        r = ROOFS["west" if k.startswith("west") else "east"]
        um = (u0 + u1) / 2
        z = r["eave"] + (um - u0) * np.tan(np.radians(r["pitch"])) + 0.06 + 0.2
        out.append((k, um, um, n0 - og, n1 + og, z))
    m = ROOFS["main"]
    out.append(("main range", (W[0] + W[1]) / 2 + 1.5, E[0] + 1.0, m["ridge_n"], m["ridge_n"], m["ridge_z"] + 0.26))
    b = DESIGN["bay"]
    out.append(("north bay gable", (b[0] + b[1]) / 2, (b[0] + b[1]) / 2, 13.0, b[3] + 0.3, ROOFS["bay"]["eave"] + 1.8 * np.tan(np.radians(50)) + 0.26))
    e = DESIGN["entry"]
    out.append(("entry gable", (e[0] + e[1]) / 2, (e[0] + e[1]) / 2, e[2] - 0.35, 12.0, ROOFS["entry"]["eave"] + 1.4 + 0.26))
    res = []
    for name, ua, ub, na, nb, z in out:
        t = np.linspace(0, 1, 60)
        nat = natural(ua + (ub - ua) * t, na + (nb - na) * t)
        rl = FFL + z
        res.append({"roof": name, "ridge_RL": round(rl, 2), "above_FFL": round(z, 2), "max_above_natural": round(float(rl - nat.min()), 2)})
    return res


def setbacks():
    ground = unary_union([rect(k) for k in ("west", "main", "east", "entry", "bay", "bay_e", "pier_w", "pier_e")])
    W, E = DESIGN["west"], DESIGN["east"]
    eaves_w = box(W[0] - 0.45, W[2] - 0.3, W[1], W[3] + 0.3)
    eaves_e = box(E[0], E[2] - 0.3, E[1] + 0.45, E[3] + 0.3)
    upper = box(W[0], W[2], W[1], 17.0)             # attic storey in the front of the west wing
    posts = [box(x - 0.24, DESIGN["loggia"][2], x + 0.24, DESIGN["loggia"][2] + 0.48) for x in (8.7, 14.7)]
    arc = street_arc()
    g = DESIGN["garage"]
    door = LineString([(g[0] + 0.7, g[2]), (g[1] - 0.7, g[2])])
    return {"front_dwelling_min": round(unary_union([ground, *posts]).distance(arc), 2),
            "garage_door": round(door.distance(arc), 2),
            "sw_wall": round(perp(SIDE_SW, ground), 2), "sw_eave": round(perp(SIDE_SW, eaves_w), 2),
            "ne_wall": round(perp(SIDE_NE, ground), 2), "ne_eave": round(perp(SIDE_NE, eaves_e), 2),
            "upper_sw_wall": round(perp(SIDE_SW, upper), 2), "upper_sw_eave": round(perp(SIDE_SW, box(W[0] - 0.45, W[2], W[1], 17.0)), 2),
            "rear_wall": round(perp(REAR, ground), 2), "upper_rear": round(perp(REAR, upper), 2)}


def cmd_ground():
    ground, gu, gn = designed_ground()
    data = json.loads((OUT / "house_massing.json").read_text())
    for k in ("positions", "faces", "colors"):
        data.pop(k, None)
    data["ground_cm"] = np.round(ground[::-1] * 100).astype(int).ravel().tolist()
    data["design"] = "hill_country"
    outline = OF.intersection(box(-20, 2.2, 45, 60))
    data["pad_outline"] = [[round(u, 3), round(n, 3)] for u, n in outline.exterior.coords]
    data["layout"] = DESIGN
    data["front"] = {"wall_top": WALL_TOP - FFL, "bank_top": BANK_TOP, "upper": UPPER - FFL, "planter": round(PLANTER_Z - FFL, 3)}
    in_lot = shapely.contains_xy(LOTF, gu, gn)
    p = DESIGN["pool"]
    in_pool = (gu > p[0] - 0.3) & (gu < p[1] + 0.3) & (gn > p[2] - 0.3) & (gn < p[3] + 0.3)
    g = DESIGN["garage"]
    in_gar = (gu > g[0]) & (gu < g[1] + 0.4) & (gn < g[3])
    cut = (ground - DEM)[in_lot & ~in_pool & ~in_gar]
    gcut = (DEM - ground)[in_lot & in_gar]
    fp = footprint()
    a = {"FFL": FFL, "terrace": PAD, "upper_terrace": UPPER, "garage_floor": GARAGE_FL, "ridge": FFL + 7.64,
         "eave": FFL + 3.0, "footprint_m2": round(fp), "coverage": round(fp / LOT.area, 3),
         "max_cut": round(float(max(-cut.min(), gcut.max())), 2), "max_fill": round(float(cut.max()), 2)}
    a["ridges"] = ridges()
    a["setbacks"] = setbacks()
    data["assumptions"] = a
    (OUT / "house_massing.json").write_text(json.dumps(data, separators=(",", ":")))
    z = zones()
    lay = {"design": "hill_country", "layout": DESIGN, "front": data["front"], "pad_outline": data["pad_outline"],
           "lot": [[round(u, 3), round(n, 3)] for u, n in LOTF.exterior.coords],
           "zones": {k: tris(v) for k, v in z.items()}, "walls": walls(), "fences": FENCES,
           "neighbours": NEIGHBOURS, "patch": fine_patch(), "assumptions": a}
    (OUT / "designs").mkdir(exist_ok=True)
    (OUT / "designs" / "hill_country_layout.json").write_text(json.dumps(lay, separators=(",", ":")))
    # coverage check: designed ground reaches the inset boundary everywhere
    covered = unary_union([house_poly(), rect("garage"), rect("pool", 0.3), *z.values(), box(14.6, 0, 30, 7.5),
                           box(DESIGN["stair"][0] - 0.35, 0, DESIGN["stair"][1] + 0.35, 5.05),
                           box(DESIGN["upstair"][0] - 0.3, TN - 0.2, DESIGN["upstair"][1] + 0.3, DESIGN["upstair"][3])])
    left = OF.intersection(box(-20, 2.2, 45, 60)).difference(covered)
    print("pad+terraces %.0f m2 of lot inset %.0f m2 (uncovered %.2f m2)" % (covered.intersection(OF).area, OF.area, left.area))
    print("footprint %.0f m2 (%.1f%%), cut %.2f, fill %.2f, walls %d" % (fp, 100 * fp / LOT.area, a["max_cut"], a["max_fill"], len(lay["walls"])))
    sb = a["setbacks"]
    print("ridges:", json.dumps(a["ridges"], indent=0))
    print("setbacks:", sb)
    assert sb["front_dwelling_min"] >= 6.0 and sb["garage_door"] >= 5.5, sb
    assert sb["sw_wall"] >= 0.9 and sb["ne_wall"] >= 0.9 and sb["sw_eave"] >= 0.45 and sb["ne_eave"] >= 0.45, sb
    assert sb["upper_sw_wall"] >= 1.5 and sb["upper_sw_eave"] >= 0.9 and sb["rear_wall"] >= 2.0 and sb["upper_rear"] >= 3.0, sb
    assert all(r["max_above_natural"] <= 9.0 for r in a["ridges"]), a["ridges"]
    for k in ("west", "main", "east", "entry", "bay", "garage", "pool", "pergola", "upstair"):
        assert LOTF.contains(rect(k)), k
    for w in lay["walls"]:
        for uu, nn_ in ((w[0] + w[4], w[1] + w[5]), (w[2] + w[4], w[3] + w[5])):
            assert LOTF.buffer(0.02).contains(shapely.Point(uu, nn_)), w
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_3d_render.py")], check=True)


def cmd_plan():
    """site plan on the aerial: lot, terraces, house parts, pool, garage"""
    from PIL import ImageDraw, ImageFont
    tex = Image.open(OUT / "ortho_texture.jpg").convert("RGB")
    S = 4
    cx, cy = (A + U * 12 + NV * 22 - [E0, N1 - NCELL])
    box_px = (int((cx - 38) * S), int((NCELL - cy - 38) * S), int((cx + 38) * S), int((NCELL - cy + 38) * S))
    img = tex.crop(box_px).resize((1200, 1200), Image.LANCZOS)
    sc = 1200 / (76 * S)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    F = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)

    def px(u, n):
        g = A + U * u + NV * n - [E0, N1 - NCELL]
        return ((g[0] * S - box_px[0]) * sc, ((NCELL - g[1]) * S - box_px[1]) * sc)

    def shape_(geom, fill, outline=None, width=1):
        for g in getattr(geom, "geoms", [geom]):
            if g.is_empty or g.geom_type != "Polygon":
                continue
            d.polygon([px(*c) for c in g.exterior.coords], fill=fill, outline=outline, width=width)

    def label(u, n, t, s=19):
        d.text(px(u, n), t, fill="white", anchor="mm", font=F(s), stroke_width=3, stroke_fill="black", align="center")

    z = zones()
    shape_(z["bank"], (150, 120, 160, 170))
    shape_(z["beds"], (150, 120, 160, 190))
    shape_(z["ubeds"], (150, 120, 160, 190))
    shape_(z["lawn"], (110, 150, 70, 200))
    shape_(z["gravel"], (205, 190, 160, 210))
    shape_(z["paving"], (226, 218, 200, 220))
    shape_(rect("loggia"), (170, 130, 90, 210), (90, 60, 30, 255), 2)
    for k in ("west", "main", "east", "entry", "bay"):
        shape_(rect(k), (238, 230, 210, 235), (40, 40, 40, 255), 2)
    shape_(rect("garage"), (90, 90, 95, 120), (20, 20, 20, 255), 3)
    shape_(rect("pool"), (60, 170, 215, 235), (20, 90, 140, 255), 2)
    shape_(rect("pergola"), (170, 130, 90, 200), (90, 60, 30, 255), 2)
    for w in walls():
        shape_(Polygon([(w[0], w[1]), (w[2], w[3]), (w[2] + w[4], w[3] + w[5]), (w[0] + w[4], w[1] + w[5])]), (120, 110, 95, 255))
    d.line([px(*c) for c in LOTF.exterior.coords], fill=(230, 40, 40, 255), width=4)
    d.line([px(-20, TN), px(45, TN)], fill=(0, 0, 0, 0), width=1)
    label(3.5, 18, "Bedroom\nwing")
    label(11.6, 15.5, "Entry / living")
    label(19.7, 18, "Kitchen +\nfamily")
    label(11.6, 9.6, "Loggia", 16)
    label(19.7, 11.5, "Garage under\n155.0", 15)
    shape_(rect("apron"), (170, 168, 162, 200), (60, 60, 60, 255), 1)
    label(19.6, 3.8, "Driveway apron", 14)
    label(3.9, 11.0, "+ attic suite", 13)
    label(11.6, 26.2, "Pool", 18)
    label(11.6, 30.6, "North courtyard 157.45", 16)
    label(9.0, 38.5, "Upper garden terrace 158.8", 18)
    label(22.0, 38.0, "Pergola", 16)
    label(5.5, 3.6, "Lavender bank", 15)
    label(9, -2.6, "Waugh Street", 24)
    out = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    dd = ImageDraw.Draw(out)
    a = json.loads((OUT / "house_massing.json").read_text())["assumptions"]
    dd.rectangle((0, 0, 1200, 70), fill=(20, 20, 20))
    dd.text((16, 10), "Hill Country Stone — Lot 624 DP821594, 36 Waugh St Griffith", font=F(24), fill="white")
    dd.text((16, 42), f"Footprint ≈{a['footprint_m2']} m² incl. garage ({a['coverage']:.0%}) · designed ground to every boundary · "
                      f"cut ≤{a['max_cut']:.1f} m, fill ≤{a['max_fill']:.1f} m · indicative only", font=F(16), fill=(220, 220, 220))
    dd.rectangle((0, 1160, 1200, 1200), fill=(20, 20, 20))
    dd.text((16, 1170), "Aerial and cadastre © State of New South Wales (Spatial Services), CC BY 4.0 · north is up · red = lot boundary",
            font=F(15), fill="white")
    dd.polygon([(1160, 90), (1145, 135), (1175, 135)], fill="white", outline="black")
    dd.text((1160, 150), "N", font=F(22), fill="white", anchor="mm", stroke_width=3, stroke_fill="black")
    (ROOT / "designs").mkdir(exist_ok=True)
    out.save(ROOT / "designs" / "hill_country_plan.png")
    print("designs/hill_country_plan.png")


if __name__ == "__main__":
    {"ground": cmd_ground, "plan": cmd_plan}[sys.argv[1]]()
