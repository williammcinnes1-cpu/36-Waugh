"""Site fit for the "Stone and Glass Gables" design on Lot 624 DP821594.

  python stone_glass_site.py ground   -> designed ground over the WHOLE lot (courtyard pad + upper rear garden
                                         terrace + front bank + garage drive); writes data/3d/house_massing.json
                                         (ground_cm, layout, front, pad_outline, assumptions)
  python stone_glass_site.py plan     -> site plan PNG over the aerial (designs/stone_glass_plan.png)

Street frame: u along the frontage chord (SW end = 0), n into the lot, heights in m AHD.
All dimensions are design assumptions for visualisation, not a design for approval.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import house_massing as hm  # noqa: E402  (lot, frame, grid constants)

OUT = ROOT / "data" / "3d"
LOT, A, U, NV = hm.LOT, hm.A, hm.U, hm.NV
E0, N1, NCELL = hm.E0, hm.N1, hm.NCELL
FFL, PAD, GARAGE_FL = hm.FFL, hm.PAD, hm.GARAGE_FL
UPPER_T = 159.2          # upper garden terrace behind the pool (rear triangle, out to the rear boundary)

# ---- design (street frame, metres) ---------------------------------------------------------------
DESIGN = {
    # street-side wings: single storey (+ attic) behind the 6 m building line; tall elements to the rear (uphill)
    "w1": (0.6, 7.6, 7.6, 17.6),         # SW stone gable to the street: single storey + attic, one bay window
    "bay": (2.7, 5.3, 7.0, 7.6),         # projecting stone bay window (the one major street window, living room)
    "w2": (0.1, 7.6, 17.6, 27.6),        # SW rear pavilion, two storeys: tall arched glazed gable to the north
    "e1": (15.8, 22.2, 7.0, 17.0),       # NE single-storey stone wing over the garage; flat roof = upper terrace
    "e2": (15.8, 22.6, 17.0, 25.5),      # NE rear gable (storey + attic): full-height glazed gable to the north
    "link": (7.6, 15.8, 13.0, 21.6),     # two-storey glass link (upper retreat looks south + east)
    "link_up": (7.6, 15.2, 13.0, 21.6),  # its upper storey
    "portico": (10.6, 13.4, 10.8, 13.0), # stone entry portico; its roof is a small balcony to the retreat
    "balcony": (7.6, 15.2, 21.6, 23.2),  # north balcony over the pool courtyard
    "terrace_up": (15.2, 22.2, 7.0, 17.0),  # roof terrace over the garage wing, off the upper retreat
    "garage": (15.8, 22.2, 7.0, 14.5),   # street level (155.0), under the NE wing; door recessed to n = 7.75
    "cabana": (20.6, 25.6, 27.0, 33.0),  # stone-and-timber outdoor room on the NE side of the courtyard
    "pool": (9.4, 19.4, 26.0, 29.8),
    "stair": (1.6, 3.4, 1.0, 5.4),       # front steps from the gate (SW end) up to the entry terrace
    "terrace_line": (-4.75, 28.5, 0.344),  # upper garden terrace starts at n >= 28.5 + 0.344 (u + 4.75)
    # roofs above FFL: gable (u0, u1, n0, n1, eave, pitch_deg) and flat tops (u0, u1, n0, n1, top)
    "roofs": {"w1": [0.6, 7.6, 7.6, 17.6, 3.0, 52], "w2": [0.1, 7.6, 17.6, 27.6, 5.4, 44],
              "e2": [15.8, 22.6, 17.0, 25.5, 3.3, 50], "cabana": [20.6, 25.6, 27.0, 33.0, 2.85, 48]},
    "flat": {"link": [7.6, 15.2, 12.4, 23.4, 5.9], "e1_balustrade": [15.8, 22.2, 7.0, 17.0, 4.35]},
    "garage_door_n": 7.75,
}
WALL_TOP = 156.4   # low stone street wall (about 0.9 m above the footpath at the SW end)
BANK_TOP = 5.0     # planted bank rises from the wall to terrace level by n = 5


def frame_poly(u0, u1, n0, n1):
    return Polygon([tuple(A + U * u + NV * n) for u, n in ((u0, n0), (u1, n0), (u1, n1), (u0, n1))])


def to_frame(e, n):
    d = np.array([e, n]) - A
    return float(d @ U), float(d @ NV)


def terrace_n(u):
    u0, n0, k = DESIGN["terrace_line"]
    return n0 + k * (u - u0)


def upper_poly():
    """the rear garden terrace: lot side of the terrace line"""
    u0, n0, k = DESIGN["terrace_line"]
    pts = [(-30, n0 + k * (-30 - u0)), (60, n0 + k * (60 - u0)), (60, 80), (-30, 80)]
    return Polygon([tuple(A + U * u + NV * n) for u, n in pts])


def pad_polygon():
    """designed ground over the whole lot, inset 0.3 m for boundary walls, from the street wall (n=2.2) back to
    every boundary, less the garage/drive notch at the NE front corner"""
    big = frame_poly(-20, 45, 2.2, 60)
    g = DESIGN["garage"]
    return LOT.buffer(-0.3, join_style=2).intersection(big).difference(frame_poly(g[0], 40, -2, g[3]))


def designed_ground(dem):
    pad = prep(pad_polygon())
    up = prep(upper_poly())
    lot = prep(LOT.buffer(0.5))
    edge = prep(LOT.buffer(0.35))   # grid cells straddling the boundary take the terrace level too
    g = dem.copy()
    gz = DESIGN["garage"]
    s = DESIGN["stair"]
    for r in range(NCELL):
        for c in range(NCELL):
            e, n = E0 + c + 0.5, N1 - r - 0.5
            if not lot.contains(Point(e, n)):
                continue
            uu, vv = to_frame(e, n)
            if gz[0] <= uu <= gz[1] and -1.0 <= vv < gz[3]:                      # garage + near-flat driveway apron
                g[r, c] = GARAGE_FL if vv >= gz[2] else np.interp(vv, [-1.0, gz[2]], [min(g[r, c], 154.7), GARAGE_FL])
            elif uu > gz[1] and -1.0 <= vv < gz[3]:                               # low planted strip NE of the drive
                g[r, c] = 155.4 if vv > 0.8 else min(g[r, c], 155.4)
            elif gz[0] - 1.0 <= uu < gz[0] and 1.6 <= vv < gz[2]:                 # planted tier west of the drive
                g[r, c] = 156.2
            elif s[0] - 0.5 <= uu <= s[1] + 0.5 and s[2] <= vv <= s[3]:          # stair cut
                g[r, c] = np.interp(vv, [s[2], s[3]], [min(g[r, c], 155.7), PAD])
            elif 0.0 <= uu <= gz[0] - 1.0 and 2.2 <= vv < BANK_TOP:                    # planted bank behind the street wall
                g[r, c] = np.interp(vv, [2.2, BANK_TOP], [WALL_TOP, PAD])
            elif up.contains(Point(e, n)) and edge.contains(Point(e, n)):
                g[r, c] = UPPER_T                                               # upper rear garden terrace
            elif pad.contains(Point(e, n)) or edge.contains(Point(e, n)) and vv > 2.2:
                g[r, c] = PAD                                                   # courtyard / house pad
            elif -1.0 <= uu <= gz[0] and -0.5 <= vv < 2.2:                       # front garden bed outside the wall
                g[r, c] = min(g[r, c], 155.6)
    return g


DWELLING = ("w1", "bay", "w2", "e1", "e2", "link", "portico")


def footprint_poly():
    return unary_union([frame_poly(*DESIGN[k]) for k in ("w1", "bay", "w2", "e1", "e2", "link", "garage", "cabana")])


def natural(dem, u, n):
    """natural (pre-works) ground, bilinear from the 1 m DEM, as farmhouse_site reads it"""
    p = A + U * u + NV * n
    x, y = p[0] - E0 - 0.5, N1 - p[1] - 0.5
    c, r = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - c, y - r
    return float(dem[r, c] * (1 - fx) * (1 - fy) + dem[r, c + 1] * fx * (1 - fy) + dem[r + 1, c] * (1 - fx) * fy + dem[r + 1, c + 1] * fx * fy)


ARC_C, ARC_R = (11.45, -74.10), 75.0   # street boundary arc (from build_3d_render)


def street_setback(u, n):
    return float(np.hypot(u - ARC_C[0], n - ARC_C[1]) - ARC_R)


def dcp_checks(dem):
    from shapely.geometry import LineString
    sw = LineString([tuple(A + U * u + NV * n) for u, n in ((0, 0), (-5.1, 32.5))])
    ne = LineString([tuple(A + U * u + NV * n) for u, n in ((22.9, 0), (29.8, 44.5))])
    rear = LineString([tuple(A + U * u + NV * n) for u, n in ((-5.1, 32.5), (29.8, 44.5))])
    rep = {"ridges": {}, "setbacks": {}}
    for k, (u0, u1, n0, n1, eave, pitch) in DESIGN["roofs"].items():
        um = (u0 + u1) / 2
        rz = eave + (u1 - u0) / 2 * np.tan(np.radians(pitch))
        nat = min(natural(dem, um, n) for n in np.linspace(n0, n1, 41))
        if k == "cabana":
            rz += PAD - FFL
        rep["ridges"][k] = {"ridge_rl": round(FFL + rz, 2), "above_natural": round(FFL + rz - nat, 2)}
    for k, (u0, u1, n0, n1, top) in DESIGN["flat"].items():
        nat = min(natural(dem, u, n) for u in np.linspace(u0, u1, 9) for n in np.linspace(n0, n1, 9))
        rep["ridges"][k + "_flat"] = {"ridge_rl": round(FFL + top, 2), "above_natural": round(FFL + top - nat, 2)}
    for k, v in rep["ridges"].items():
        assert v["above_natural"] <= 9.0, (k, v)
    # street setbacks (perpendicular to the arc)
    corners = lambda r: [(r[0], r[2]), (r[1], r[2]), (r[0], r[3]), (r[1], r[3])]
    rep["setbacks"]["street_dwelling_min"] = round(min(street_setback(u, n) for k in DWELLING for u, n in corners(DESIGN[k])), 2)
    g = DESIGN["garage"]
    rep["setbacks"]["garage_door"] = round(min(street_setback(u, DESIGN["garage_door_n"]) for u in np.linspace(g[0] + 0.8, g[1] - 0.8, 9)), 2)
    # side and rear setbacks per storey (walls; eaves 0.45 m beyond)
    for k, storey in (("w1", 2), ("w2", 2), ("e1", 1), ("e2", 2), ("link_up", 2), ("cabana", 1)):
        P = frame_poly(*DESIGN[k])
        d_sw, d_ne, d_r = P.distance(sw), P.distance(ne), P.distance(rear)
        rep["setbacks"][k] = {"sw": round(d_sw, 2), "ne": round(d_ne, 2), "rear": round(d_r, 2), "storeys": storey}
        side, eave_min, rear_min = (1.5, 0.9, 3.0) if storey == 2 else (0.9, 0.45, 2.0)
        assert min(d_sw, d_ne) >= side and min(d_sw, d_ne) - 0.45 >= eave_min and d_r >= rear_min, (k, d_sw, d_ne, d_r)
    assert rep["setbacks"]["street_dwelling_min"] >= 6.0 and rep["setbacks"]["garage_door"] >= 5.5, rep["setbacks"]
    return rep


def cmd_ground():
    dem = np.array(Image.open(OUT / "dem_240m.tif"), dtype=np.float64)
    ground = designed_ground(dem)
    pad = pad_polygon()
    outline = [to_frame(*p) for p in pad.exterior.coords]
    data = json.loads((OUT / "house_massing.json").read_text())
    data["ground_cm"] = np.round(ground[::-1] * 100).astype(int).ravel().tolist()
    data["design"] = "stone_glass"
    data["pad_outline"] = [[round(u, 3), round(n, 3)] for u, n in outline]
    clip = LOT.buffer(-0.3, join_style=2).intersection(frame_poly(-20, 45, 2.2, 60))   # convex: for surface clipping
    data["lot_inner"] = [[round(u, 3), round(n, 3)] for u, n in (to_frame(*p) for p in clip.exterior.coords)]
    data["layout"] = DESIGN
    data["front"] = {"wall_top": WALL_TOP - FFL, "bank_top": BANK_TOP, "upper_terrace": UPPER_T - FFL}
    # earthworks statistics inside the lot only
    inside = np.zeros_like(dem, bool)
    lp = prep(LOT)
    for r in range(NCELL):
        for c in range(NCELL):
            inside[r, c] = lp.contains(Point(E0 + c + 0.5, N1 - r - 0.5))
    cut = (ground - dem)[inside]
    fp = footprint_poly().area
    data["dcp"] = dcp_checks(dem)
    data["assumptions"] = {"FFL": FFL, "terrace": PAD, "upper_terrace": UPPER_T, "garage_floor": GARAGE_FL,
                           "ridge": data["dcp"]["ridges"]["w2"]["ridge_rl"], "eave": FFL + 5.4, "footprint_m2": round(fp),
                           "coverage": round(fp / LOT.area, 3),
                           "max_cut": round(float(-cut.min()), 2), "max_fill": round(float(cut.max()), 2)}
    for k in ("w1", "w2", "e1", "e2", "link", "balcony", "garage", "cabana", "pool"):
        assert LOT.contains(frame_poly(*DESIGN[k])), k
    print(json.dumps(data["dcp"], indent=1))
    # designed ground reaches the boundary all round (pad + terrace + front bed) -- coverage check
    used = pad.union(frame_poly(-5, 30, -2, 2.2).intersection(LOT)).union(frame_poly(DESIGN["garage"][0], 30, -2, 14.6).intersection(LOT))
    left = LOT.difference(used.buffer(0.31, join_style=2))
    (OUT / "house_massing.json").write_text(json.dumps(data, separators=(",", ":")))
    print("pad area %.0f m2 (lot %.0f), unused %.1f m2, footprint %.0f m2 (%.0f%%), cut %.2f, fill %.2f" % (
        pad.area, LOT.area, left.area, fp, 100 * fp / LOT.area, -cut.min(), cut.max()))


def cmd_plan():
    """site plan on the aerial: lot, terraces, house parts, pool, garage, with areas"""
    from PIL import ImageDraw, ImageFont
    tex = Image.open(OUT / "ortho_texture.jpg").convert("RGB")
    S = 4  # px per metre in the texture
    cx, cy = (A + U * 12 + NV * 20 - [E0, N1 - NCELL])
    box_px = (int((cx - 38) * S), int((NCELL - cy - 38) * S), int((cx + 38) * S), int((NCELL - cy + 38) * S))
    img = tex.crop(box_px).resize((1200, 1200), Image.LANCZOS)
    sc = 1200 / (76 * S)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    F = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)

    def px(u, n):
        g = A + U * u + NV * n - [E0, N1 - NCELL]
        return ((g[0] * S - box_px[0]) * sc, ((NCELL - g[1]) * S - box_px[1]) * sc)

    def shp(poly, fill, outline, w=2):
        d.polygon([px(*to_frame(*p)) for p in poly.exterior.coords], fill=fill, outline=outline, width=w)

    def rect(key, fill, outline, label=None, size=20):
        u0, u1, n0, n1 = DESIGN[key]
        d.polygon([px(u0, n0), px(u1, n0), px(u1, n1), px(u0, n1)], fill=fill, outline=outline, width=3)
        if label:
            d.text(px((u0 + u1) / 2, (n0 + n1) / 2), label, fill="white", anchor="mm", font=F(size), stroke_width=3,
                   stroke_fill="black", align="center")

    pad = pad_polygon()
    upper = pad.intersection(upper_poly())
    main = pad.difference(upper_poly())
    shp(main, (214, 200, 170, 110), (214, 200, 170, 255))
    shp(upper, (120, 150, 80, 140), (90, 120, 60, 255))
    front = frame_poly(-5, 30, -2, 2.2).intersection(LOT).difference(frame_poly(DESIGN["garage"][0], 40, -3, 3))
    shp(front, (110, 130, 80, 120), (90, 110, 60, 255))
    g = DESIGN["garage"]
    shp(frame_poly(g[0], 40, -2, g[3]).intersection(LOT), (150, 145, 135, 150), (90, 90, 90, 255))
    rect("garage", (90, 90, 95, 210), (30, 30, 30, 255))
    rect("w1", (232, 214, 178, 235), (40, 40, 40, 255), "Living\n1 storey\n+ attic", 16)
    rect("bay", (232, 214, 178, 235), (40, 40, 40, 255))
    rect("w2", (222, 200, 160, 235), (40, 40, 40, 255), "2-storey\ngable\n(arched\nglass N)", 16)
    rect("e1", (205, 195, 175, 235), (40, 40, 40, 255), "Garage\nbelow,\nroof\nterrace", 15)
    rect("e2", (222, 200, 160, 235), (40, 40, 40, 255), "Kitchen\n+ attic\n(glass\ngable N)", 15)
    rect("link", (170, 200, 215, 235), (30, 60, 80, 255), "2-storey\nglass link\n(upper\nretreat)", 15)
    rect("portico", (190, 175, 150, 220), (40, 40, 40, 255))
    rect("balcony", (120, 150, 170, 200), (30, 60, 80, 255))
    rect("pool", (60, 170, 215, 235), (20, 90, 140, 255), "Pool")
    rect("cabana", (170, 130, 90, 220), (90, 60, 30, 255), "Outdoor\nroom", 16)
    d.text(px(11.5, 8.2), "Entry\ncourt", fill="white", anchor="mm", font=F(17), stroke_width=3, stroke_fill="black", align="center")
    d.text(px(19.0, 3.6), "Drive\n155.0", fill="white", anchor="mm", font=F(17), stroke_width=3, stroke_fill="black", align="center")
    d.text(px(14.0, 33.5), "North courtyard 157.45", fill="white", anchor="mm", font=F(19), stroke_width=3, stroke_fill="black")
    d.text(px(13.0, 40.2), "Upper garden terrace 159.2", fill="white", anchor="mm", font=F(18), stroke_width=3,
           stroke_fill="black")
    d.line([px(*to_frame(*p)) for p in LOT.exterior.coords], fill=(230, 40, 40, 255), width=5)
    d.text(px(9, -2.5), "Waugh Street", fill="white", anchor="mm", font=F(24), stroke_width=3, stroke_fill="black")
    out = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    dd = ImageDraw.Draw(out)
    a = json.loads((OUT / "house_massing.json").read_text())["assumptions"]
    dd.rectangle((0, 0, 1200, 70), fill=(20, 20, 20))
    dd.text((16, 10), "Stone and Glass Gables — Lot 624 DP821594, 36 Waugh St Griffith", font=F(24), fill="white")
    dd.text((16, 42), f"Footprint ≈{a['footprint_m2']} m² incl. garage ({a['coverage']:.0%} of 1,093 m²) · whole lot terraced "
                      f"to the boundary · cut ≤{a['max_cut']:.1f} m, fill ≤{a['max_fill']:.1f} m · indicative only",
            font=F(16), fill=(220, 220, 220))
    dd.rectangle((0, 1160, 1200, 1200), fill=(20, 20, 20))
    dd.text((16, 1170), "Aerial and cadastre © State of New South Wales (Spatial Services), CC BY 4.0 · north is up", font=F(15), fill="white")
    dd.polygon([(1160, 90), (1145, 135), (1175, 135)], fill="white", outline="black")
    dd.text((1160, 150), "N", font=F(22), fill="white", anchor="mm", stroke_width=3, stroke_fill="black")
    dst = ROOT / "designs" / "stone_glass_plan.png"
    dst.parent.mkdir(exist_ok=True)
    out.save(dst)
    print(dst)


if __name__ == "__main__":
    {"ground": cmd_ground, "plan": cmd_plan}[sys.argv[1]]()
