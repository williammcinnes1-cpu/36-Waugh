"""Site fit for the "Wraparound Farmhouse" concept on Lot 624 DP821594 (36 Waugh St, Griffith).

  python3 scripts/designs/wraparound_site.py ground  -> designed ground over the WHOLE lot (two terrace levels,
                                                        garage + drive, front bank) and the landscape surfaces;
                                                        writes data/3d/house_massing.json (ground_cm, layout,
                                                        front, pad_outline, assumptions) and refreshes
                                                        data/3d/render_scene.json via build_3d_render.py
  python3 scripts/designs/wraparound_site.py plan    -> site plan PNG over the aerial (designs/wraparound_plan.png)

Street frame: u along the frontage chord (SW end = 0), n into the lot, heights in m AHD.
Indicative concept only, not a design for approval.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import house_massing as hm  # noqa: E402  (lot, frame, grid constants)

OUT = ROOT / "data" / "3d"
LOT, A, U, NV = hm.LOT, hm.A, hm.U, hm.NV
E0, N1, NCELL = hm.E0, hm.N1, hm.NCELL
FFL, PAD, GARAGE_FL = hm.FFL, hm.PAD, hm.GARAGE_FL
UPPER_Z = PAD + 1.2          # upper garden terrace behind the pool (158.65 m AHD)

# ---- design (street frame, metres) ---------------------------------------------------------------
DESIGN = {
    "west": (2.0, 9.0, 10.0, 21.0),         # two-storey board-and-batten gable wing: main suite upstairs, street + east views
    "main": (9.0, 15.6, 10.0, 21.0),        # whitewashed brick body, three dormers, arched door in a stone portico
    "east": (15.6, 22.5, 7.4, 24.5),        # single-storey vaulted gable wing over the garage, gable to the street
    "rear": (-1.7, 8.6, 21.0, 29.5),        # single-storey family wing (SW side), glazed to the courtyard
    "verandah": (0.0, 14.0, 7.2, 10.0),     # front run of the wraparound verandah (posts at n = 7.5)
    "verandah_side": (0.0, 2.0, 10.0, 21.0),  # ...turning the SW corner and running back to the rear wing
    "garage": (15.6, 22.5, 7.4, 14.9),      # street level (155.0), under the front of the east wing; door at n = 7.4
    "pool": (9.8, 19.8, 24.4, 28.4),
    "outdoor": (20.8, 25.9, 25.5, 31.0),    # covered outdoor room, stone fireplace, timber-lined ceiling
    "stair": (1.4, 3.0, 1.0, 4.6),          # front steps from the gate up to the lawn
    "tier": (14.6, 15.6, 2.2, 7.4),         # planted tier stepping down beside the driveway apron
    "upper_line": (30.4 + 0.344 * 1.4, 0.344),  # upper terrace starts at n = a + b*u (parallel to the rear boundary)
    "upper_z": UPPER_Z,
    "court": (8.0, 27.0, 21.0, 33.5),       # paved courtyard zone (clipped to the lower terrace)
    # roofs: eave height above FFL, pitch (deg); ridge = eave + half-span * tan(pitch)
    "roofs": {"west": (5.4, 37), "main": (3.4, 38), "east": (3.2, 45), "rear": (3.2, 35), "outdoor": (2.9, 30)},
    "upper_floor": 3.1,                     # west wing upper floor above FFL (clears the verandah roof)
    "upper_bay": (6.6, 9.0, 8.8, 10.0),     # cantilevered upper bay over the verandah: corner glass to street + east
    "storeys": {"west": 2, "main": 1.5, "east": 1, "rear": 1},
}
TIER_Z = 156.25    # planted tier between the apron (155.0) and the lawn (157.45)
HOUSE_KEYS = ("west", "main", "east", "rear")
WALL_TOP = 156.4   # low street wall (about 0.9 m above the footpath)
BANK_TOP = 4.6     # planted bank rises from the wall to lawn level by n = 4.6


def frame_poly(u0, u1, n0, n1):
    return Polygon([tuple(A + U * u + NV * n) for u, n in ((u0, n0), (u1, n0), (u1, n1), (u0, n1))])


def to_frame(e, n):
    d = np.array([e, n]) - A
    return float(d @ U), float(d @ NV)


def lot_frame():
    return orient(Polygon([to_frame(*p) for p in LOT.exterior.coords]), 1.0)


def nt(u):
    a, b = DESIGN["upper_line"]
    return a + b * u


def regions():
    """landscape regions in the street frame (shapely)"""
    lot = lot_frame()
    full = lot.buffer(-0.3, join_style=2)
    upper = full.intersection(Polygon([(-20, nt(-20)), (60, nt(60)), (60, 80), (-20, 80)]))
    g = DESIGN["garage"]
    gzone = box(g[0], -5, 40, g[3])
    lower = (full.intersection(box(-20, BANK_TOP, 60, 80)).difference(upper).difference(gzone)
             .difference(box(*_b(DESIGN["tier"]))))
    house = unary_union([box(*_b(DESIGN[k])) for k in HOUSE_KEYS])
    ver = unary_union([box(*_b(DESIGN["verandah"])), box(*_b(DESIGN["verandah_side"]))])
    p = DESIGN["pool"]
    poolc = box(p[0] - 0.3, p[2] - 0.3, p[1] + 0.3, p[3] + 0.3)
    outd = box(*_b(DESIGN["outdoor"]))
    strip = full.difference(full.buffer(-1.3, join_style=2))          # boundary planting strip (hedges)
    court = box(*_b(DESIGN["court"])).intersection(lower)
    s = DESIGN["stair"]
    bed = box(-1, 6.3, 14.6, 7.2).difference(box(11.4, 6.0, 13.2, 7.3))  # hydrangea bed along the verandah
    lower_free = lower.difference(house).difference(ver).difference(poolc).difference(outd)
    paving = court.difference(house).difference(poolc).difference(outd).difference(strip)
    mulch = lower_free.intersection(strip.union(bed)).difference(paving).difference(box(s[0] - 0.3, 0, s[1] + 0.3, BANK_TOP + 0.2))
    lawn = lower_free.difference(paving).difference(mulch)
    return dict(lot=lot, full=full, upper=upper, lower=lower, house=house, ver=ver, poolc=poolc, outd=outd,
                paving=paving, mulch=mulch, lawn=lawn, mulch_up=upper.intersection(strip),
                lawn_up=upper.difference(strip), gzone=gzone)


def _b(r):
    u0, u1, n0, n1 = r
    return u0, n0, u1, n1


def pad_polygon():
    """the whole designed ground inside the boundary walls: lot inset 0.3 m from the street wall back, less the
    garage + drive cut (both terraces). In the street frame."""
    R = regions()
    return R["full"].intersection(box(-20, 2.2, 60, 80)).difference(R["gzone"])


def tiles(geom, cell=1.0):
    """split a region into small polygons (clipped 1 m cells) so the renderer can lay surfaces without shapely"""
    out = []
    if geom.is_empty:
        return out
    x0, y0, x1, y1 = geom.bounds
    pg = prep(geom)
    for u in np.arange(np.floor(x0), x1, cell):
        for n in np.arange(np.floor(y0), y1, cell):
            c = box(u, n, u + cell, n + cell)
            if not pg.intersects(c):
                continue
            g = c.intersection(geom)
            for p in getattr(g, "geoms", [g]):
                if p.geom_type == "Polygon" and p.area > 1e-3:
                    p = orient(p, 1.0)
                    out.append([[round(a, 3), round(b, 3)] for a, b in list(p.exterior.coords)[:-1]])
    return out


def natural(dem, u, n):
    """natural ground (DEM, bilinear) at a street-frame point"""
    p = A + U * u + NV * n
    x, y = p[0] - E0 - 0.5, N1 - p[1] - 0.5
    c, r = int(x), int(y)
    fx, fy = x - c, y - r
    return float(dem[r, c] * (1 - fx) * (1 - fy) + dem[r, c + 1] * fx * (1 - fy) + dem[r + 1, c] * (1 - fx) * fy
                 + dem[r + 1, c + 1] * fx * fy)


def ridges():
    """(name, (u0, n0), (u1, n1), ridge height above FFL) for every roof ridge, including overhangs"""
    import math
    R, out = DESIGN["roofs"], []
    t = lambda k: math.tan(math.radians(R[k][1]))
    w, m, e, r, o = DESIGN["west"], DESIGN["main"], DESIGN["east"], DESIGN["rear"], DESIGN["outdoor"]
    out.append(("west gable wing", ((w[0] + w[1]) / 2, w[2] - 0.45), ((w[0] + w[1]) / 2, w[3]), R["west"][0] + (w[1] - w[0]) / 2 * t("west")))
    out.append(("main body", (m[0], (m[2] + m[3]) / 2), (m[1] + 0.3, (m[2] + m[3]) / 2), R["main"][0] + (m[3] - m[2]) / 2 * t("main")))
    out.append(("east wing (garage)", ((e[0] + e[1]) / 2, e[2] - 0.45), ((e[0] + e[1]) / 2, e[3] + 0.45), R["east"][0] + (e[1] - e[0]) / 2 * t("east")))
    out.append(("rear family wing", ((r[0] + r[1]) / 2, r[2]), ((r[0] + r[1]) / 2, r[3] + 0.45), R["rear"][0] + (r[1] - r[0]) / 2 * t("rear")))
    zo = PAD - FFL + 0.15
    out.append(("outdoor room", (o[0] - 0.35, (o[2] + o[3]) / 2), (o[1] + 0.35, (o[2] + o[3]) / 2), zo + R["outdoor"][0] + (o[3] - o[2]) / 2 * math.tan(math.radians(30)) + 0.05))
    out.append(("upper bay (skillion)", (DESIGN["upper_bay"][0], DESIGN["upper_bay"][2]), (DESIGN["upper_bay"][1], DESIGN["upper_bay"][3]), 5.5))
    zb = R["main"][0] + 1.6 * t("main") - 0.3
    out.append(("dormers", (10.25, 11.6), (10.25, 14.2), zb + 1.7 + 0.75 + 0.05))
    return out


def compliance(dem):
    """DCP checks: ridge heights over natural ground, street / side / rear setbacks (perpendicular distances)"""
    from shapely.geometry import LineString
    lot = [tuple(c) for c in lot_frame().exterior.coords]
    street = LineString([(0.0, 0.0)] + [c for c in lot if 0.5 < c[0] < 22.5 and c[1] < 1.0] + [(22.903, 0.0)])
    sw = LineString([(0.0, 0.0), (-5.096, 32.496)])
    ne = LineString([(22.903, 0.0), (29.767, 44.458)])
    rear = LineString([(-5.096, 32.496), (29.767, 44.458)])
    res = {"ridges": [], "setbacks": {}}
    for name, (u0, n0), (u1, n1), z in ridges():
        hs = [FFL + z - natural(dem, u0 + (u1 - u0) * t, n0 + (n1 - n0) * t) for t in np.linspace(0, 1, 41)]
        res["ridges"].append([name, round(FFL + z, 2), round(max(hs), 2)])
    sb = res["setbacks"]
    for k in HOUSE_KEYS + ("outdoor",):
        wall = box(*_b(DESIGN[k]))
        eave = wall.buffer(0.45 if k != "outdoor" else 0.35, join_style=2)
        sb[k] = {"street": round(wall.distance(street), 2), "sw": round(wall.distance(sw), 2), "ne": round(wall.distance(ne), 2),
                 "rear": round(wall.distance(rear), 2), "eave_sw": round(eave.distance(sw), 2), "eave_ne": round(eave.distance(ne), 2)}
    posts = [(x, 7.5) for x in (0.3, 3.4, 6.5, 9.2, 10.8, 13.75)] + [(0.3, n) for n in (11.0, 14.4, 17.8, 20.7)]
    sb["verandah_posts"] = {"street": round(min(Point(p).distance(street) for p in posts), 2),
                            "sw": round(min(Point(p).distance(sw) for p in posts), 2)}
    sb["garage_door"] = round(LineString([(15.6, 7.4), (22.5, 7.4)]).distance(street), 2)
    sb["east_bay"] = round(box(17.8, 7.0, 20.4, 7.4).distance(street), 2)
    sb["west_bay"] = round(box(4.4, 9.4, 7.0, 10.0).distance(street), 2)
    ub = DESIGN["upper_bay"]
    sb["upper_bay"] = round(box(*_b(ub)).distance(street), 2)
    sb["upper_west"] = {"sw": round(box(*_b(DESIGN["west"])).distance(sw), 2), "ne": round(box(*_b(DESIGN["west"])).union(box(*_b(ub))).distance(ne), 2),
                        "rear": round(box(*_b(DESIGN["west"])).distance(rear), 2)}
    return res


def designed_ground(dem):
    R = regions()
    lotp = prep(LOT.buffer(0.5))
    full, upper = prep(R["full"]), prep(R["upper"])
    padp = prep(pad_polygon())
    g = dem.copy()
    gz = DESIGN["garage"]
    s = DESIGN["stair"]
    p = DESIGN["pool"]
    for r in range(NCELL):
        for c in range(NCELL):
            e, n = E0 + c + 0.5, N1 - r - 0.5
            if not lotp.contains(Point(e, n)):
                continue
            uu, vv = to_frame(e, n)
            pt = Point(uu, vv)
            if gz[0] <= uu and -1.0 <= vv < gz[3]:                                  # garage + driveway
                g[r, c] = GARAGE_FL if vv >= gz[2] else np.interp(vv, [-1.0, gz[2]], [g[r, c], GARAGE_FL])
            elif s[0] - 0.5 <= uu <= s[1] + 0.5 and s[2] <= vv <= s[3]:            # stair cut
                g[r, c] = np.interp(vv, [s[2], s[3]], [min(g[r, c], 155.7), PAD])
            elif DESIGN["tier"][0] <= uu < gz[0] and 2.2 <= vv < DESIGN["tier"][3]:   # planted tier by the apron
                g[r, c] = TIER_Z
            elif uu < gz[0] and 2.2 <= vv < BANK_TOP:                              # planted bank behind the street wall
                g[r, c] = np.interp(vv, [2.2, BANK_TOP], [WALL_TOP, PAD])
            elif p[0] - 0.4 <= uu <= p[1] + 0.4 and p[2] - 0.4 <= vv <= p[3] + 0.4:  # pool shell
                g[r, c] = PAD - 1.8
            elif upper.contains(pt):
                g[r, c] = UPPER_Z
            elif padp.contains(pt):
                g[r, c] = PAD
            elif vv < 2.2:                                                           # front garden bed
                g[r, c] = min(g[r, c], 155.6)
    return g


def cmd_ground():
    dem = np.array(Image.open(OUT / "dem_240m.tif"), dtype=np.float64)
    ground = designed_ground(dem)
    R = regions()
    pad = orient(pad_polygon(), 1.0)
    data = json.loads((OUT / "house_massing.json").read_text())
    data["ground_cm"] = np.round(ground[::-1] * 100).astype(int).ravel().tolist()
    data["design"] = "wraparound"
    data["pad_outline"] = [[round(u, 3), round(n, 3)] for u, n in pad.exterior.coords]
    lay = dict(DESIGN)
    lay["lot_frame"] = [[round(u, 3), round(n, 3)] for u, n in R["lot"].exterior.coords]
    lay["surfaces"] = {k: tiles(R[k]) for k in ("lawn", "paving", "mulch", "lawn_up", "mulch_up")}
    data["layout"] = lay
    data["front"] = {"wall_top": WALL_TOP - FFL, "bank_top": BANK_TOP}
    # earthworks over the lot, excluding the pool shell
    inside = np.zeros_like(dem, bool)
    lotp = prep(LOT)
    p = DESIGN["pool"]
    for r in range(NCELL):
        for c in range(NCELL):
            e, n = E0 + c + 0.5, N1 - r - 0.5
            if lotp.contains(Point(e, n)):
                uu, vv = to_frame(e, n)
                inside[r, c] = not (p[0] - 0.4 <= uu <= p[1] + 0.4 and p[2] - 0.4 <= vv <= p[3] + 0.4)
    d = (ground - dem)[inside]
    house = R["house"]
    footprint = house.area  # garage lies under the east wing
    data["assumptions"] = {"FFL": FFL, "terrace": PAD, "upper_terrace": UPPER_Z, "garage_floor": GARAGE_FL,
                           "eave": FFL + 3.4, "footprint_m2": round(footprint),
                           "coverage": round(footprint / LOT.area, 3), "max_cut": round(float(-d.min()), 2),
                           "max_fill": round(float(d.max()), 2), "designed_ground_m2": round(R["full"].area + 0.0)}
    dcp = compliance(dem)
    data["assumptions"]["dcp"] = dcp
    data["assumptions"]["ridge"] = max(r[1] for r in dcp["ridges"])
    (OUT / "house_massing.json").write_text(json.dumps(data, separators=(",", ":")))
    print(json.dumps(dcp, indent=1))
    for name, rl, h in dcp["ridges"]:
        assert h <= 9.0, (name, rl, h)
    st = DESIGN["storeys"]
    for k, d_ in dcp["setbacks"].items():
        if k in st:
            up = st[k] > 1
            assert d_["street"] >= 6.0 and min(d_["sw"], d_["ne"]) >= (1.5 if up else 0.9), (k, d_)
            assert min(d_["eave_sw"], d_["eave_ne"]) >= (0.9 if up else 0.45), (k, d_)
            assert d_["rear"] >= (3.0 if up else 2.0), (k, d_)
    assert dcp["setbacks"]["garage_door"] >= 5.5 and dcp["setbacks"]["verandah_posts"]["street"] >= 6.0
    assert dcp["setbacks"]["east_bay"] >= 6.0
    print("lot %.0f m2, designed ground (inset 0.3) %.0f m2, pad %.0f, upper %.0f; footprint %.0f m2 (%.1f%%), "
          "cut %.2f, fill %.2f" % (R["lot"].area, R["full"].area, pad.area, R["upper"].area, footprint,
                                   100 * footprint / LOT.area, -d.min(), d.max()))
    for k in HOUSE_KEYS + ("garage", "pool", "outdoor", "verandah", "verandah_side"):
        assert LOT.contains(frame_poly(*DESIGN[k])), k
    lot = R["lot"]
    # rear corners reach designed ground (within 0.3 m of the boundary)
    for corner in list(lot.exterior.coords)[1:3]:     # NE rear and SW rear corners
        assert R["full"].distance(Point(corner)) < 0.6, corner
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_3d_render.py")], check=True)


def cmd_plan():
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

    def shp(geom, fill, outline=None, width=1):
        for p in getattr(geom, "geoms", [geom]):
            if p.geom_type != "Polygon" or p.is_empty:
                continue
            d.polygon([px(*c) for c in p.exterior.coords], fill=fill, outline=outline, width=width)
            for h in p.interiors:
                d.polygon([px(*c) for c in h.coords], fill=(0, 0, 0, 0))

    def label(u, n, t, s=20):
        d.text(px(u, n), t, fill="white", anchor="mm", font=F(s), stroke_width=3, stroke_fill="black", align="center")

    R = regions()
    shp(R["lawn"], (96, 140, 60, 210))
    shp(R["mulch"], (110, 84, 60, 210))
    shp(R["paving"], (222, 212, 190, 225))
    shp(R["lawn_up"], (128, 168, 80, 220))
    shp(R["mulch_up"], (110, 84, 60, 210))
    bank = box(0, 2.2, DESIGN["garage"][0], BANK_TOP).intersection(R["full"])
    shp(bank, (120, 110, 70, 200))
    gz = R["full"].intersection(box(DESIGN["garage"][0], -2, 40, DESIGN["garage"][3]))
    shp(gz, (150, 146, 138, 220))
    # terrace wall + boundary walls
    a, b = DESIGN["upper_line"]
    line = [(u, nt(u)) for u in np.linspace(-6, 31, 30) if R["full"].buffer(0.05).contains(Point(u, nt(u)))]
    d.line([px(*c) for c in line], fill=(250, 250, 245, 255), width=4)
    d.line([px(*c) for c in pad_polygon().exterior.coords], fill=(240, 236, 228, 255), width=2)
    for k in ("verandah", "verandah_side", "outdoor"):
        shp(box(*_b(DESIGN[k])), (170, 128, 86, 220), (90, 60, 30, 255), 2)
    for k in HOUSE_KEYS:
        shp(box(*_b(DESIGN[k])), (246, 245, 240, 235), (40, 40, 40, 255), 3)
    g = DESIGN["garage"]
    gp = [px(g[0], g[2]), px(g[1], g[2]), px(g[1], g[3]), px(g[0], g[3])]
    for i in range(4):
        d.line([gp[i], gp[(i + 1) % 4]], fill=(60, 60, 60, 255), width=2)
    p = DESIGN["pool"]
    shp(box(p[0], p[2], p[1], p[3]), (60, 170, 215, 240), (20, 90, 140, 255), 3)
    label(5.7, 15.0, "2-storey\ngable wing", 17)
    label(12.3, 15.5, "Entry\nliving", 17)
    label(19.1, 15.5, "Gable wing\nover\nGARAGE\n(155.0)", 16)
    label(3.2, 25.2, "Family\nliving", 18)
    label(14.8, 26.4, "Pool", 20)
    label(23.4, 28.2, "Outdoor\nroom", 15)
    label(8.0, 8.4, "wraparound verandah", 15)
    label(14.0, 35.6, "Upper garden terrace 158.65", 16)
    label(14.0, 31.5, "North courtyard 157.45", 16)
    label(8.0, 3.4, "planted bank", 14)
    label(19.1, 1.5, "drive", 15)
    d.line([px(*c) for c in R["lot"].exterior.coords], fill=(230, 40, 40, 255), width=5)
    label(9, -2.6, "Waugh Street", 24)
    out = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    dd = ImageDraw.Draw(out)
    asm = json.loads((OUT / "house_massing.json").read_text())["assumptions"]
    dd.rectangle((0, 0, 1200, 74), fill=(20, 20, 20))
    dd.text((16, 8), "Wraparound Farmhouse — Lot 624 DP821594, 36 Waugh St Griffith", font=F(24), fill="white")
    dd.text((16, 44), f"Footprint ≈{asm['footprint_m2']} m² incl. garage under ({asm['coverage']:.0%}) · whole lot landscaped to the "
                      f"boundary · cut ≤{asm['max_cut']:.1f} m, fill ≤{asm['max_fill']:.1f} m · indicative only",
            font=F(16), fill=(220, 220, 220))
    dd.rectangle((0, 1160, 1200, 1200), fill=(20, 20, 20))
    dd.text((16, 1170), "Aerial and cadastre © State of New South Wales (Spatial Services), CC BY 4.0 · north is up",
            font=F(15), fill="white")
    dd.polygon([(1160, 90), (1145, 135), (1175, 135)], fill="white", outline="black")
    dd.text((1160, 150), "N", font=F(22), fill="white", anchor="mm", stroke_width=3, stroke_fill="black")
    (ROOT / "designs").mkdir(exist_ok=True)
    out.save(ROOT / "designs" / "wraparound_plan.png")
    print("designs/wraparound_plan.png")


if __name__ == "__main__":
    {"ground": cmd_ground, "plan": cmd_plan}[sys.argv[1]]()
