"""Site fit for "Sage and Stone" (two-tone modern farmhouse) on Lot 624 DP821594, 36 Waugh St Griffith.

  python3 scripts/designs/sage_stone_site.py ground  -> designed ground over the WHOLE lot (two terrace levels, garage cut,
                                                       front bank); writes data/3d/house_massing.json and refreshes
                                                       data/3d/render_scene.json via build_3d_render.py
  python3 scripts/designs/sage_stone_site.py plan    -> designs/sage_stone_plan.png (site plan over the aerial)
  python3 scripts/designs/sage_stone_site.py facts   -> data/3d/designs/sage_stone.json (web-viewer meta)

Street frame: u along the frontage chord (SW end = 0), n into the lot, heights in m AHD.
Design assumptions for visualisation only, not a design for approval.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from shapely import constrained_delaunay_triangles
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import house_massing as hm  # noqa: E402

OUT = ROOT / "data" / "3d"
LOT, A, U, NV = hm.LOT, hm.A, hm.U, hm.NV
E0, N1, NCELL = hm.E0, hm.N1, hm.NCELL
FFL, PAD, GARAGE_FL = hm.FFL, hm.PAD, hm.GARAGE_FL
UPPER_T = 158.55       # small raised garden terrace (about 1 m up) along the rear boundary, pergola + seat
INSET = 0.3            # designed ground reaches the lot boundary less the retaining-wall thickness

# ---- design (street frame, metres) ---------------------------------------------------------------
DESIGN = {
    "main": (2.0, 8.8, 9.0, 20.5),         # white two-storey gable to the street (upper retreat/main bedroom, street + east views)
    "gbay": (3.6, 7.2, 8.6, 9.0),           # ground-floor bay window under a metal awning (the one major street window)
    "link": (8.8, 15.6, 11.0, 20.5),        # white single-storey entry/living link, recessed behind the portico
    "portico": (10.0, 13.6, 8.5, 11.0),     # timber entry portico with its own gable
    "east": (15.6, 22.8, 7.5, 23.5),        # sage gabled wing (kitchen/dining) over the garage, gable + bay window to the street
    "west": (2.0, 8.8, 20.5, 29.5),         # single-storey sage living wing on the SW side, glazed north gable
    "outdoor": (21.8, 25.6, 25.0, 31.0),    # covered outdoor room with a stone fireplace
    "garage": (15.6, 22.0, 7.5, 15.0),      # double garage at street level under the east wing (door >= 5.5 m back)
    "oriel": (17.25, 21.15, 6.85, 7.5),     # projecting bay window over the garage door, metal awning roof
    "apron_bed": (14.5, 15.6, 1.6, 7.5),    # planted terrace beside the driveway apron
    "pool": (10.5, 20.0, 25.0, 29.0),
    "court": (8.8, 21.8, 20.5, 31.0),       # paved courtyard between the wings
    "stair": (1.4, 3.2, 1.0, 5.4),          # front steps from the gate (SW end) up to the lawn
    "perg_u": 6.0,
    "upper_line": [[-8.0, 27.62], [33.0, 41.72]],  # pad / upper-terrace retaining wall, 3.6 m in from the rear boundary
}
WALL_TOP = 156.4   # low street wall (about 0.9 m above the footpath)
BANK_TOP = 5.0     # planted bank rises from the wall to lawn level by n = 5
HOUSE_KEYS = ("main", "gbay", "link", "portico", "east", "west", "outdoor", "garage", "oriel")
BED_LEVEL = 156.3
# ridges for the DCP height check: (name, ridge RL above FFL, ridge line incl. overhangs in the street frame)
RIDGES = [("main white gable (2 storey)", 5.0 + 3.4 * 0.8391, (5.4, 8.55), (5.4, 20.95)),
          ("entry link (1 storey)", 3.0 + 4.75 * 0.7002, (8.8, 15.75), (15.6, 15.75)),
          ("portico", 2.6 + 1.8 * 0.7002, (11.8, 8.2), (11.8, 11.0)),
          ("east sage wing", 3.0 + 3.6, (19.2, 7.05), (19.2, 23.95)),
          ("west sage wing", 3.2 + 3.4 * 1.0724, (5.4, 20.5), (5.4, 29.95)),
          ("outdoor room", 0.0 + 2.95 + 1.9 * 0.8391, (23.7, 24.7), (23.7, 31.3))]


def frame_poly(u0, u1, n0, n1):
    return Polygon([tuple(A + U * u + NV * n) for u, n in ((u0, n0), (u1, n0), (u1, n1), (u0, n1))])


def to_frame(e, n):
    d = np.array([e, n]) - A
    return float(d @ U), float(d @ NV)


def F(poly):
    """map-frame polygon -> street-frame polygon"""
    return Polygon([to_frame(*p) for p in poly.exterior.coords], [[to_frame(*p) for p in r.coords] for r in poly.interiors])


def rect(u0, u1, n0, n1):
    return Polygon([(u0, n0), (u1, n0), (u1, n1), (u0, n1)])


def pergola_poly(grow=0.0):
    """pergola footprint on the rear terrace, aligned with the terrace wall (4.6 m along it, 2.6 m deep)"""
    (a, b), (c, d) = DESIGN["upper_line"]
    t = np.array([c - a, d - b]) / np.hypot(c - a, d - b)
    nrm = np.array([-t[1], t[0]])
    o = np.array([DESIGN["perg_u"], upper_n(DESIGN["perg_u"])]) + nrm * 0.45
    pts = [o + t * x + nrm * y for x, y in ((-2.3 - grow, -grow * 0.5), (2.3 + grow, -grow * 0.5), (2.3 + grow, 2.6 + grow), (-2.3 - grow, 2.6 + grow))]
    return Polygon([tuple(p) for p in pts])


def upper_n(u):
    (a, b), (c, d) = DESIGN["upper_line"]
    return b + (d - b) * (u - a) / (c - a)


LOT_F = F(LOT)
INNER = LOT_F.buffer(-INSET, join_style="mitre")
GZONE = rect(DESIGN["garage"][0], 40, -5, DESIGN["garage"][3])        # garage + drive cut (to the NE boundary)
ABOVE = Polygon([(-20, upper_n(-20)), (60, upper_n(60)), (60, 80), (-20, 80)])


def pad_polygon():
    """designed (levelled) ground in the street frame: the lot inset 0.3 m from n = 2.2 back to the rear boundary,
    less the garage/drive cut. Lower level PAD up to the upper line, UPPER_T behind it."""
    ab = DESIGN["apron_bed"]
    return INNER.intersection(rect(-20, 45, 2.2, 60)).difference(GZONE).difference(rect(ab[0], ab[1], -5, ab[3]))


def upper_polygon():
    return pad_polygon().intersection(ABOVE)


def designed_ground(dem):
    pad_lo = prep(pad_polygon().difference(ABOVE))
    pad_up = prep(upper_polygon())
    lot = prep(LOT.buffer(0.5))
    lot_in = prep(LOT)
    g = dem.copy()
    gz = DESIGN["garage"]
    s = DESIGN["stair"]
    ab = DESIGN["apron_bed"]
    for r in range(NCELL):
        for c in range(NCELL):
            e, n = E0 + c + 0.5, N1 - r - 0.5
            if not lot.contains(Point(e, n)):
                continue
            uu, vv = to_frame(e, n)
            p = Point(uu, vv)
            if uu >= gz[0] and -1.5 <= vv < gz[3]:                               # garage, side strip + driveway apron
                g[r, c] = GARAGE_FL if vv >= 0.5 else np.interp(vv, [-1.5, 0.5], [g[r, c], GARAGE_FL])
            elif ab[0] <= uu < ab[1] and ab[2] <= vv < ab[3]:                    # planted terrace beside the apron
                g[r, c] = BED_LEVEL
            elif s[0] - 0.5 <= uu <= s[1] + 0.5 and s[2] <= vv <= s[3]:          # stair cut
                g[r, c] = np.interp(vv, [s[2], s[3]], [min(g[r, c], 155.9), PAD])
            elif uu < ab[0] and 2.2 <= vv < BANK_TOP:                            # planted bank behind the street wall
                g[r, c] = np.interp(vv, [2.2, BANK_TOP], [WALL_TOP, PAD])
            elif pad_up.contains(p):
                g[r, c] = UPPER_T
            elif pad_lo.contains(p):
                g[r, c] = PAD
            elif uu < gz[0] and vv < 1.6:                                       # front garden bed
                g[r, c] = min(g[r, c], 155.9)
            elif vv >= 2.2 and lot_in.contains(Point(e, n)):                     # the 0.3 m wall strip: nearest level
                g[r, c] = UPPER_T if ABOVE.contains(p) else PAD
    return g


def tris(poly, z=None):
    """constrained triangulation of a street-frame (multi)polygon -> [[u,n],[u,n],[u,n]] list"""
    out = []
    geoms = getattr(poly, "geoms", [poly])
    for p in geoms:
        if p.is_empty or p.area < 0.05 or p.geom_type != "Polygon":
            continue
        for t in constrained_delaunay_triangles(p).geoms:
            c = list(t.exterior.coords)[:3]
            out.append([[round(x, 3), round(y, 3)] for x, y in c])
    return out


def surfaces():
    """flat finish surfaces for the renderer (street frame triangles)"""
    D = DESIGN
    houses = unary_union([rect(*D[k]) for k in ("main", "gbay", "link", "portico", "east", "west", "outdoor")])
    lo = pad_polygon().difference(ABOVE).intersection(rect(-20, 45, BANK_TOP, 60))
    court = rect(*D["court"]).difference(houses)
    pool_ring = rect(D["pool"][0] - 0.3, D["pool"][1] + 0.3, D["pool"][2] - 0.3, D["pool"][3] + 0.3)
    stair_top = rect(D["stair"][0] - 0.3, D["stair"][1] + 0.3, 0, D["stair"][3])
    # side path along the NE wing; terrace beside the west wing's glazed gable
    lawn_lo = lo.difference(houses).difference(court).difference(stair_top)
    paving = court.difference(pool_ring)
    up = upper_polygon()
    fire = pergola_poly(0.3)                                                # flagstone floor under the pergola
    drive_side = INNER.intersection(rect(D["garage"][1], 40, 0, D["garage"][3]))
    return {"lawn_lo": tris(lawn_lo), "paving": tris(paving), "lawn_up": tris(up.difference(fire)),
            "fire_pad": tris(up.intersection(fire)), "drive_side": tris(drive_side)}


def footprint():
    return unary_union([rect(*DESIGN[k]) for k in HOUSE_KEYS])


def cmd_ground():
    dem = np.array(Image.open(OUT / "dem_240m.tif"), dtype=np.float64)
    ground = designed_ground(dem)
    pad = pad_polygon()
    data = json.loads((OUT / "house_massing.json").read_text())
    data["ground_cm"] = np.round(ground[::-1] * 100).astype(int).ravel().tolist()
    data["design"] = "sage_stone"
    data["pad_outline"] = [[round(u, 3), round(n, 3)] for u, n in pad.exterior.coords]
    data["upper_outline"] = [[round(u, 3), round(n, 3)] for u, n in upper_polygon().exterior.coords]
    data["surfaces"] = surfaces()
    data["upper_wall"] = [[round(u, 3), round(n, 3)] for u, n in LineString(DESIGN["upper_line"]).intersection(INNER.buffer(0.28)).coords]
    data["lot_outline"] = [[round(u, 3), round(n, 3)] for u, n in LOT_F.exterior.coords]
    data["layout"] = DESIGN
    data["front"] = {"wall_top": WALL_TOP - FFL, "bank_top": BANK_TOP, "upper": UPPER_T - FFL}
    lot_mask = np.zeros_like(dem, bool)
    plot = prep(LOT)
    for r in range(NCELL):
        for c in range(NCELL):
            lot_mask[r, c] = plot.contains(Point(E0 + c + 0.5, N1 - r - 0.5))
    cut = (ground - dem)[lot_mask]
    fp = footprint().area
    data["assumptions"] = {"FFL": FFL, "terrace": PAD, "upper_terrace": UPPER_T, "garage_floor": GARAGE_FL,
                           "ridge": FFL + 9.9, "eave": FFL + 5.4, "footprint_m2": round(fp), "coverage": round(fp / LOT.area, 3),
                           "max_cut": round(float(-cut.min()), 2), "max_fill": round(float(cut.max()), 2),
                           "designed_area_m2": round(pad.area + INNER.intersection(GZONE).area + INNER.intersection(rect(-20, 45, -5, 2.2)).area)}
    (OUT / "house_massing.json").write_text(json.dumps(data, separators=(",", ":")))
    print("pad area %.0f m2 (lot %.0f, lot inset %.0f), footprint %.0f m2 (%.1f%%), cut %.2f, fill %.2f" % (
        pad.area, LOT.area, INNER.area, fp, 100 * fp / LOT.area, -cut.min(), cut.max()))
    # everything inside the lot, house parts >= 0.9 m off the side boundaries
    for k in HOUSE_KEYS + ("pool", "court"):
        assert LOT_F.contains(rect(*DESIGN[k])), k
    sides = [LineString([(0, 0), (-5.1, 32.5)]), LineString([(22.9, 0), (29.8, 44.5)])]
    for k in ("main", "east", "west", "outdoor", "garage"):
        d = min(rect(*DESIGN[k]).distance(s) for s in sides)
        assert d >= 0.9, (k, d)
    # designed ground reaches the boundary all round (rear corners included)
    for u, n in LOT_F.exterior.coords[1:3]:                                  # the two rear corners
        assert pad.distance(Point(u, n)) < 0.65, ("rear corner not levelled", u, n)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_3d_render.py")], check=True)


def cmd_plan():
    """site plan on the aerial: lot, both terrace levels, house parts, pool, garage, with areas"""
    from PIL import ImageDraw, ImageFont
    tex = Image.open(OUT / "ortho_texture.jpg").convert("RGB")
    S = 4  # px per metre in the texture
    cx, cy = (A + U * 12 + NV * 21 - [E0, N1 - NCELL])
    box_px = (int((cx - 38) * S), int((NCELL - cy - 38) * S), int((cx + 38) * S), int((NCELL - cy + 38) * S))
    img = tex.crop(box_px).resize((1200, 1200), Image.LANCZOS)
    sc = 1200 / (76 * S)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    Fo = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)

    def px(u, n):
        g = A + U * u + NV * n - [E0, N1 - NCELL]
        return ((g[0] * S - box_px[0]) * sc, ((NCELL - g[1]) * S - box_px[1]) * sc)

    def shape(poly, fill, outline, width=2):
        for p in getattr(poly, "geoms", [poly]):
            d.polygon([px(*c) for c in p.exterior.coords], fill=fill, outline=outline, width=width)

    def label(u, n, text, s=19):
        d.text(px(u, n), text, fill="white", anchor="mm", font=Fo(s), stroke_width=3, stroke_fill="black", align="center")

    D = DESIGN
    shape(pad_polygon().difference(ABOVE), (160, 196, 110, 110), (214, 230, 170, 255))
    shape(upper_polygon(), (120, 170, 90, 140), (200, 230, 160, 255))
    shape(INNER.intersection(GZONE), (150, 150, 150, 120), (210, 210, 210, 255))
    shape(INNER.intersection(rect(*D["apron_bed"])), (110, 140, 80, 150), None)
    shape(INNER.intersection(rect(-20, 45, -5, 2.2)).difference(GZONE), (110, 90, 70, 110), None)
    shape(rect(*D["court"]).difference(unary_union([rect(*D[k]) for k in ("main", "link", "east", "west", "outdoor")])), (225, 215, 195, 170), None)
    wl = LineString(D["upper_line"]).intersection(INNER)
    d.line([px(*c) for c in wl.coords], fill=(240, 235, 220, 255), width=4)
    for k, fill, lab in (("portico", (170, 130, 90, 220), None),
                         ("main", (245, 245, 240, 225), "White\ngable\n2 storey"), ("gbay", (245, 245, 240, 225), None),
                         ("link", (245, 245, 240, 225), "Entry +\nliving"),
                         ("east", (154, 165, 140, 235), "Sage wing\nkitchen"), ("west", (154, 165, 140, 235), "Sage wing\nliving"),
                         ("outdoor", (170, 130, 90, 210), "Outdoor\nroom"), ("pool", (60, 170, 215, 235), "Pool"),
                         ("oriel", (154, 165, 140, 235), None), ("garage", (60, 60, 65, 150), "Garage\n(below)")):
        shape(rect(*D[k]), fill, (35, 35, 35, 255), 3)
        if lab:
            u0, u1, n0, n1 = D[k]
            label((u0 + u1) / 2, (n0 + n1) / 2 + {"garage": -2.0, "east": 4.5}.get(k, 0), lab)
    label(15.5, 23.0, "North courtyard", 18)
    label(16.0, upper_n(16) + 1.8, "Raised garden terrace %.2f" % UPPER_T, 16)
    label(8.0, 6.3, "Front lawn %.2f" % PAD, 16)
    d.line([px(*c) for c in LOT_F.exterior.coords], fill=(230, 40, 40, 255), width=5)
    label(9, -3.0, "Waugh Street", 24)
    out = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    dd = ImageDraw.Draw(out)
    a = json.loads((OUT / "house_massing.json").read_text())["assumptions"]
    dd.rectangle((0, 0, 1200, 92), fill=(20, 20, 20))
    dd.text((16, 10), "Sage and Stone — Lot 624 DP821594, 36 Waugh St Griffith", font=Fo(24), fill="white")
    dd.text((16, 42), f"Footprint ≈{a['footprint_m2']} m² incl. garage and outdoor room ({a['coverage']:.0%} of 1,093 m²) · "
                      f"FFL {FFL} · court {PAD} · upper {UPPER_T} · garage {GARAGE_FL}", font=Fo(16), fill=(220, 220, 220))
    dd.text((16, 66), f"Whole lot levelled to the boundary (inset {INSET} m for walls) · cut ≤{a['max_cut']:.1f} m · "
                      f"fill ≤{a['max_fill']:.1f} m · indicative only", font=Fo(16), fill=(220, 220, 220))
    dd.rectangle((0, 1160, 1200, 1200), fill=(20, 20, 20))
    dd.text((16, 1170), "Aerial and cadastre © State of New South Wales (Spatial Services), CC BY 4.0 · north is up", font=Fo(15), fill="white")
    dd.polygon([(1160, 110), (1145, 155), (1175, 155)], fill="white", outline="black")
    dd.text((1160, 170), "N", font=Fo(22), fill="white", anchor="mm", stroke_width=3, stroke_fill="black")
    p = ROOT / "designs" / "sage_stone_plan.png"
    p.parent.mkdir(exist_ok=True)
    out.save(p)
    print(p)


def cmd_facts():
    a = json.loads((OUT / "house_massing.json").read_text())["assumptions"]
    meta = {"id": "sage_stone", "name": "Sage and Stone", "file": "designs/sage_stone.txt",
            "note": "Two-tone farmhouse: a white board-and-batten two-storey gable (upper retreat with street and east views) and light sage gabled wings on a pale limestone base, timber portico and brass trim; modest ground-floor street windows, big glass and pool on the north courtyard, terraced out to every boundary.",
            "facts": [["Lot area", "≈1,093 m²"], ["Zoning", "R1 General Residential"],
                      ["House + garage footprint", f"{a['footprint_m2']:,} m² ({a['coverage']:.0%})"],
                      ["Floor level", f"{a['FFL']:.1f} m AHD"],
                      ["Earthworks", f"cut ≤{a['max_cut']:.1f} m · fill ≤{a['max_fill']:.1f} m"],
                      ["Bushfire prone", "rear half (vegetation buffer)"]]}
    p = OUT / "designs" / "sage_stone.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(p, meta["facts"][2:5])


def natural(u, n):
    dem = natural.dem
    p = A + U * u + NV * n
    x, y = p[0] - E0 - 0.5, N1 - p[1] - 0.5
    c, r = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - c, y - r
    return float(dem[r, c] * (1 - fx) * (1 - fy) + dem[r, c + 1] * fx * (1 - fy) + dem[r + 1, c] * (1 - fx) * fy + dem[r + 1, c + 1] * fx * fy)


natural.dem = np.array(Image.open(OUT / "dem_240m.tif"), dtype=np.float64)


def arc_n(u):
    return -74.10 + np.sqrt(75.0 ** 2 - (u - 11.45) ** 2)


def cmd_check():
    """DCP checks: ridge RL and height over natural ground, street / side / rear setbacks"""
    print("ridge heights (RL, max height above natural ground under the ridge):")
    for name, zr, (ua, na), (ub, nb) in RIDGES:
        nat = min(natural(ua + (ub - ua) * t, na + (nb - na) * t) for t in np.linspace(0, 1, 41))
        print(f"  {name:32s} RL {FFL + zr:6.2f}  {FFL + zr - nat:4.2f} m above natural (min {nat:.2f})")
    def street(k, pts):
        return min(n - arc_n(u) for u, n in pts)
    D = DESIGN
    g = D["garage"]
    print("garage door setback  %.2f m" % min(g[2] - arc_n(u) for u in np.linspace(g[0], g[1], 30)))
    for k in ("main", "gbay", "link", "portico", "east", "oriel"):
        u0, u1, n0, n1 = D[k]
        print(f"street setback {k:8s} {min(n0 - arc_n(u) for u in np.linspace(u0, u1, 30)):.2f} m")
    sw, ne = LineString([(0, 0), (-5.1, 32.5)]), LineString([(22.9, 0), (29.8, 44.5)])
    rear = LineString([(-5.1, 32.5), (29.8, 44.5)])
    for k in ("main", "link", "east", "west", "outdoor", "garage"):
        r = rect(*D[k])
        print(f"{k:8s} SW {r.distance(sw):5.2f}  NE {r.distance(ne):5.2f}  rear {r.distance(rear):5.2f}")
    print("upper storey (main gable) eave 0.45 to SW: %.2f" % rect(1.55, 9.25, 8.55, 20.95).distance(sw))
    print("east eave (0.45 m + gutter) to NE: %.2f" % rect(D["east"][0] - 0.6, D["east"][1] + 0.6, D["east"][2] - 0.6, D["east"][3] + 0.6).distance(ne))
    print("outdoor eave (0.3 m + gutter) to NE: %.2f" % rect(21.35, 26.05, 24.55, 31.45).distance(ne))


if __name__ == "__main__":
    {"ground": cmd_ground, "plan": cmd_plan, "facts": cmd_facts, "check": cmd_check}[sys.argv[1]]()
