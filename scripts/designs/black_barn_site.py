"""Site fit for the "Modern Black Barnhouse" concept on Lot 624 DP821594 (36 Waugh St, Griffith).

  python3 scripts/designs/black_barn_site.py ground  -> designed ground over the WHOLE lot (two terraces + front bank,
                                                       garage/drive cut), pad outline, finish surfaces; writes
                                                       data/3d/house_massing.json and refreshes render_scene.json
  python3 scripts/designs/black_barn_site.py finish  -> reads data/3d/black_barn_mesh.json (black_barn_cycles.py --export-mesh)
                                                       and writes the massing mesh / LAS / GLB like farmhouse_site.py
  python3 scripts/designs/black_barn_site.py plan    -> designs/black_barn_plan.png (site plan over the aerial)

Street frame: u along the frontage chord (SW end = 0), n into the lot, heights in m AHD.
Layout: three black barn forms in a U around a north courtyard -- a street barn (living / entry), a gable
wing over the street-level garage (NE) and a family wing down the SW side whose glazed gable looks north.
The ground is levelled right out to every boundary: the main pad at 157.45 from the street bank to a
terrace wall that runs parallel-ish to the diagonal rear boundary, and an upper garden terrace at 158.65
behind it that fills both rear corners. All figures are indicative, not a design for approval.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import shapely
from PIL import Image
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import house_massing as hm  # noqa: E402  (lot, frame, grid constants, LAS/GLB writers)

OUT = ROOT / "data" / "3d"
LOT, A, U, NV = hm.LOT, hm.A, hm.U, hm.NV
E0, N1, NCELL = hm.E0, hm.N1, hm.NCELL
FFL, PAD, GARAGE_FL = hm.FFL, hm.PAD, hm.GARAGE_FL
UPPER_T = 158.65                     # upper garden terrace behind the pool (1.2 m above the pad)

# ---- design (street frame, metres) ---------------------------------------------------------------
DESIGN = {
    "upper": (0.8, 7.6, 6.9, 15.4),       # two-storey barn at the SW front: living below, main bedroom suite above
    "main": (7.6, 15.6, 7.9, 16.4),       # flat-roofed entry/living link, 1 m behind the building line; balcony on top
    "wing": (15.6, 22.3, 6.9, 21.0),      # gable wing to the street, garage under its front half
    "rear": (0.4, 9.0, 15.4, 30.6),       # family wing down the SW side, glazed gable to the north
    "entry": (11.6, 15.6, 7.9, 10.4),     # recessed entry (timber soffit, limewashed brick walls)
    "balcony": (7.6, 10.6, 7.9, 12.6),    # upper-floor balcony on the link roof (east / south-east views)
    "garage": (15.9, 22.3, 6.9, 14.4),    # street level (155.0), under the wing; door face >= 5.5 m from the street
    "pool": (10.2, 19.2, 22.6, 26.6),
    "lawn": (9.4, 19.5, 28.0, 33.0),
    "outdoor": (20.2, 25.2, 24.0, 30.0),  # covered outdoor fireplace room
    "court": (9.0, 27.5, 16.4, 28.0),     # paved courtyard band around the pool
    "stair": (12.2, 14.0, 1.0, 5.4),      # front steps from the footpath up to the entry
    "planter": (14.3, 15.9, 156.35),      # planted terrace between the steps and the driveway (u range, level)
    "tsteps": (20.4, 22.0),               # steps up the terrace wall (u range)
    "edge": [-5.2, 30.6, 0.172],          # terrace wall line: n = n0 + k * (u - u0)
    "upper_z": UPPER_T,
    # elements allowed in the 4.5-6 m articulation zone: chimney breast, two window bays, entry fin wall
    "projections": {"bay_main": (2.9, 5.5, 6.45, 6.9), "bay_wing": (18.15, 19.75, 6.45, 6.9)},
    "fin": (11.25, 11.6, 7.4, 10.4),      # limewashed brick entry wall
    "chimney": (-0.15, 0.8, 10.25, 12.05),  # tall limewashed brick chimney on the SW gable of the two-storey barn
    # roofs: eave and pitch above FFL; ridge extents (street frame) for the DCP height check
    "roofs": {"upper": {"eave": 4.8, "pitch": 36.0, "floor": 3.0}, "dormer": {"eave": 4.8, "pitch": 45.0, "u": [2.8, 5.6]},
              "main": {"eave": 3.2, "flat": True}, "wing": {"eave": 3.2, "pitch": 42.0},
              "rear": {"eave": 3.0, "pitch": 40.0}, "outdoor": {"eave": 2.75, "pitch": 38.0}},
}
WALL_TOP = 156.4   # low street wall (about 0.9 m above the footpath)
BANK_TOP = 5.0     # planted bank rises from the wall to pad level by n = 5


def n_edge(u):
    u0, n0, k = DESIGN["edge"]
    return n0 + k * (u - u0)


def frame_poly(u0, u1, n0, n1):
    return Polygon([tuple(A + U * u + NV * n) for u, n in ((u0, n0), (u1, n0), (u1, n1), (u0, n1))])


def to_frame(e, n):
    d = np.array([e, n]) - A
    return float(d @ U), float(d @ NV)


LOT_F = Polygon([to_frame(*p) for p in LOT.exterior.coords])   # the lot in the street frame


def fbox(u0, u1, n0, n1):
    return box(u0, n0, u1, n1)


def upper_zone():
    return Polygon([(-30, n_edge(-30)), (60, n_edge(60)), (60, 90), (-30, 90)])


def garage_zone():
    g = DESIGN["garage"]
    return fbox(g[0], 40, -3, g[3])


def pad_polygon():
    """whole-lot pad (MGA): the lot inset 0.3 m from n = 2.2 back to the rear boundary, less the garage/drive zone"""
    big = frame_poly(-20, 45, 2.2, 60)
    g = DESIGN["garage"]
    return LOT.buffer(-0.3).intersection(big).difference(frame_poly(g[0], 40, -3, g[3]))


def footprint_f():
    return unary_union([fbox(*DESIGN[k]) for k in ("upper", "main", "wing", "rear")])


def surfaces():
    """finish surfaces (street frame) for the renderer: triangles per material at a fixed AHD level"""
    inner = LOT_F.buffer(-0.3, join_style=2)
    up = upper_zone()
    house = footprint_f()
    p = DESIGN["pool"]
    poolc = fbox(p[0] - 0.3, p[1] + 0.3, p[2] - 0.3, p[3] + 0.3)
    outdoor = fbox(*DESIGN["outdoor"])
    pl = DESIGN["planter"]
    main = inner.intersection(fbox(-20, 60, BANK_TOP, 90)).difference(up).difference(garage_zone())
    main = main.difference(fbox(pl[0], pl[1], 0, 6.3))
    court = main.intersection(fbox(*DESIGN["court"])).difference(house).difference(poolc).difference(outdoor)
    lawn = fbox(*DESIGN["lawn"])
    s = DESIGN["stair"]
    path = fbox(s[0], s[1], BANK_TOP, DESIGN["main"][2])
    gravel = main.difference(house).difference(poolc).difference(outdoor).difference(court).difference(lawn).difference(path)
    upper = inner.intersection(up)
    g = DESIGN["garage"]
    strip = inner.intersection(fbox(g[1], 40, g[2], g[3]))
    out = []
    for mat, geom, z in (("paving", court, PAD), ("lawn", lawn, PAD), ("gravel", gravel, PAD), ("gravel", upper, UPPER_T),
                         ("concrete", strip, GARAGE_FL), ("paving", path, PAD)):
        tris = shapely.constrained_delaunay_triangles(geom)
        pts = [[[round(x, 3), round(y, 3)] for x, y in list(t.exterior.coords)[:3]] for t in tris.geoms]
        out.append({"mat": mat, "z": z, "tris": pts, "area": round(geom.area, 1)})
    return out


def level(uu, vv, existing):
    """designed ground level (AHD) at a street-frame point inside the lot (or within 0.3 m of it)"""
    g = DESIGN["garage"]
    s = DESIGN["stair"]
    if uu >= g[0] and -1.0 <= vv < g[3]:                                     # garage, drive and the side strip
        return GARAGE_FL if vv >= g[2] else float(np.interp(vv, [-1.0, g[2]], [existing, GARAGE_FL]))
    pl = DESIGN["planter"]
    if pl[0] <= uu < pl[1] and 2.2 <= vv < 6.3:                                 # planted terrace beside the drive
        return pl[2]
    if s[0] - 0.5 <= uu <= s[1] + 0.3 and s[2] <= vv <= s[3]:                # stair cut
        return float(np.interp(vv, [s[2], s[3]], [min(existing, 155.4), PAD]))
    if 2.2 <= vv < BANK_TOP:                                                  # planted bank behind the street wall
        return float(np.interp(vv, [2.2, BANK_TOP], [WALL_TOP, PAD]))
    if vv >= 2.2:
        return UPPER_T if vv > n_edge(uu) else PAD
    if 0.0 <= vv < 1.6:                                                       # front garden bed
        return min(existing, 155.6)
    return existing


def designed_ground(dem):
    near = prep(LOT.buffer(0.3))
    g = dem.copy()
    for r in range(NCELL):
        for c in range(NCELL):
            e, n = E0 + c + 0.5, N1 - r - 0.5
            if not near.contains(Point(e, n)):
                continue
            uu, vv = to_frame(e, n)
            g[r, c] = level(uu, vv, g[r, c])
    return g


def ridges():
    """ridge lines (u0, n0, u1, n1, RL of the ridge cap / parapet) from the roof spec"""
    R = DESIGN["roofs"]
    out = {}
    u0, u1, n0, n1 = DESIGN["upper"]
    e, p = R["upper"]["eave"], np.radians(R["upper"]["pitch"])
    nm = (n0 + n1) / 2
    zr_up = e + (n1 - n0) / 2 * np.tan(p)
    out["upper"] = (u0, nm, u1, nm, FFL + zr_up + 0.18)
    du0, du1 = R["dormer"]["u"]
    zd = R["dormer"]["eave"] + (du1 - du0) / 2 * np.tan(np.radians(R["dormer"]["pitch"]))
    out["dormer"] = ((du0 + du1) / 2, n0, (du0 + du1) / 2, n0 + (zd - e) / np.tan(p), FFL + zd + 0.18)
    m = DESIGN["main"]
    out["main"] = (m[0], m[2], m[1], m[2], FFL + R["main"]["eave"] + 0.2)          # flat roof parapet
    for k in ("wing", "rear", "outdoor"):
        u0, u1, n0, n1 = DESIGN[k]
        e, p = R[k]["eave"], np.radians(R[k]["pitch"])
        um = (u0 + u1) / 2
        out[k] = (um, n0, um, n1, FFL + e + (u1 - u0) / 2 * np.tan(p) + 0.18)
    return out


def dcp_checks(dem):
    def nat(u, n):
        e, nn = A + U * u + NV * n
        x, y = e - E0 - 0.5, N1 - nn - 0.5
        i, j = int(x), int(y)
        fx, fy = x - i, y - j
        return (dem[j, i] * (1 - fx) + dem[j, i + 1] * fx) * (1 - fy) + (dem[j + 1, i] * (1 - fx) + dem[j + 1, i + 1] * fx) * fy
    rep = {}
    for k, (u0, n0, u1, n1, rl) in ridges().items():
        hs = [(rl - nat(u0 + (u1 - u0) * t, n0 + (n1 - n0) * t), u0 + (u1 - u0) * t, n0 + (n1 - n0) * t) for t in np.linspace(0, 1, 41)]
        h, hu, hn = max(hs)
        rep[k] = {"ridge_RL": round(rl, 2), "ridge_above_FFL": round(rl - FFL, 2), "max_height_above_natural": round(h, 2),
                  "at": [round(hu, 1), round(hn, 1)]}
        assert h <= 9.0, (k, h)
    c = list(LOT_F.exterior.coords)
    street = shapely.LineString(c[3:])
    sides = {"SW": shapely.LineString([(0, 0), (-5.1, 32.5)]), "NE": shapely.LineString([(22.9, 0), (29.8, 44.5)])}
    rear = shapely.LineString([(-5.1, 32.5), (29.8, 44.5)])
    walls = {k: fbox(*DESIGN[k]) for k in ("upper", "main", "wing", "rear", "outdoor")}
    rep["street_setback_walls"] = round(min(street.distance(w) for k, w in walls.items() if k != "outdoor"), 2)
    proj = {k: fbox(*v) for k, v in DESIGN["projections"].items()}
    rep["street_setback_projections"] = round(min(street.distance(p) for p in proj.values()), 2)
    zone = LOT_F.intersection(street.buffer(6.0)).difference(street.buffer(4.5))
    rep["articulation_zone_m2"] = round(zone.area, 1)
    rep["projections_in_zone_m2"] = round(unary_union(list(proj.values())).intersection(street.buffer(6.0)).area, 2)
    g = DESIGN["garage"]
    door = shapely.LineString([(g[0] + 0.45, g[2] + 0.22), (g[1] - 0.45, g[2] + 0.22)])
    rep["garage_door_setback"] = round(street.distance(door), 2)
    rep["side_setbacks"] = {k: {s: round(l.distance(w), 2) for s, l in sides.items()} for k, w in walls.items()}
    rep["rear_setback"] = round(min(rear.distance(w) for w in walls.values()), 2)
    up = fbox(*DESIGN["upper"])
    rep["upper_storey"] = {"side_SW": round(sides["SW"].distance(up), 2), "side_NE": round(sides["NE"].distance(up), 2),
                           "rear": round(rear.distance(up), 2), "street": round(street.distance(up), 2),
                           "floor_RL": FFL + DESIGN["roofs"]["upper"]["floor"]}
    assert min(rep["upper_storey"]["side_SW"], rep["upper_storey"]["side_NE"]) >= 1.5 and rep["upper_storey"]["rear"] >= 3.0
    assert rep["street_setback_walls"] >= 6.0 and rep["street_setback_projections"] >= 4.5, rep
    assert rep["projections_in_zone_m2"] <= 0.25 * rep["articulation_zone_m2"], rep
    assert rep["garage_door_setback"] >= 5.5 and rep["rear_setback"] >= 2.0, rep
    assert min(v for d in rep["side_setbacks"].values() for v in d.values()) >= 0.9, rep
    return rep


def cmd_ground():
    dem = np.array(Image.open(OUT / "dem_240m.tif"), dtype=np.float64)
    ground = designed_ground(dem)
    pad = pad_polygon()
    outline = [to_frame(*p) for p in pad.exterior.coords]
    data = json.loads((OUT / "house_massing.json").read_text())
    data["ground_cm"] = np.round(ground[::-1] * 100).astype(int).ravel().tolist()
    data["design"] = "black_barn"
    data["pad_outline"] = [[round(u, 3), round(n, 3)] for u, n in outline]
    data["layout"] = DESIGN
    data["front"] = {"wall_top": WALL_TOP - FFL, "bank_top": BANK_TOP}
    data["surfaces"] = surfaces()
    inside = prep(LOT)
    mask = np.zeros_like(dem, bool)
    for r in range(NCELL):
        for c in range(NCELL):
            mask[r, c] = inside.contains(Point(E0 + c + 0.5, N1 - r - 0.5))
    cut = (ground - dem)[mask]
    footprint = footprint_f().union(fbox(*DESIGN["garage"])).area
    data["assumptions"] = {"FFL": FFL, "terrace": PAD, "upper_terrace": UPPER_T, "garage_floor": GARAGE_FL,
                           "ridge": round(max(r[4] for r in ridges().values()), 2), "eave": FFL + 3.0, "footprint_m2": round(footprint),
                           "coverage": round(footprint / LOT.area, 3), "outdoor_room_m2": round(fbox(*DESIGN["outdoor"]).area),
                           "max_cut": round(float(-cut.min()), 2), "max_fill": round(float(cut.max()), 2)}
    data["assumptions"]["dcp"] = dcp_checks(dem)
    (OUT / "house_massing.json").write_text(json.dumps(data, separators=(",", ":")))
    print(json.dumps(data["assumptions"]["dcp"], indent=1))
    print("pad area %.0f m2 (lot %.0f), footprint %.0f m2 (%.1f%%), cut %.2f, fill %.2f" % (
        pad.area, LOT.area, footprint, 100 * footprint / LOT.area, -cut.min(), cut.max()))
    for sf in data["surfaces"]:
        print("  surface %-9s z=%.2f  %6.1f m2  %d tris" % (sf["mat"], sf["z"], sf["area"], len(sf["tris"])))
    for k in ("upper", "main", "wing", "rear", "garage", "pool", "outdoor", "lawn"):
        assert LOT.contains(frame_poly(*DESIGN[k])), k
    sides = [shapely.LineString([(0, 0), (-5.1, 32.5)]), shapely.LineString([(22.9, 0), (29.8, 44.5)])]
    for k in ("upper", "main", "wing", "rear", "outdoor"):          # at least 0.9 m from the side boundaries
        dmin = min(sd.distance(fbox(*DESIGN[k])) for sd in sides)
        print("  %-8s side setback %.2f m" % (k, dmin))
        assert dmin >= 0.9 - 1e-6, k
    # the levelled ground reaches the boundary all round (pad + garage/drive zone covers the inset lot from n = 2.2)
    cover = pad.union(LOT.buffer(-0.3).intersection(frame_poly(DESIGN["garage"][0], 40, -3, DESIGN["garage"][3])))
    gap = LOT.buffer(-0.3).intersection(frame_poly(-20, 45, 2.2, 60)).difference(cover)
    assert gap.area < 0.5, gap.area
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_3d_render.py")], check=True)


def cmd_finish():
    import laspy
    import pyproj
    import trimesh
    mesh = json.loads((OUT / "black_barn_mesh.json").read_text())
    data = json.loads((OUT / "house_massing.json").read_text())
    v = np.array(mesh["positions"], float).reshape(-1, 3)   # u, n, z-above-FFL
    f = np.array(mesh["faces"], int).reshape(-1, 3)
    col = np.array(mesh["colors"], int).reshape(-1, 3)
    grid = hm.to_grid(v[:, 0], v[:, 1], v[:, 2] + FFL)
    data["positions"] = np.round(grid, 3).ravel().tolist()
    data["faces"] = f.ravel().tolist()
    data["colors"] = col.ravel().tolist()
    (OUT / "house_massing.json").write_text(json.dumps(data, separators=(",", ":")))

    class M:
        pass
    m = M()
    m.v = [tuple(p) for p in grid]
    m.f = [tuple(x) for x in f]
    m.c = [tuple(c) for c in col]
    pts, cols = hm.sample(m, spacing=0.25)
    ground = np.array(data["ground_cm"], float).reshape(NCELL, NCELL)[::-1] / 100
    tex = np.array(Image.open(OUT / "ortho_texture.jpg"))
    k = 2
    fine = np.array(Image.fromarray(ground.astype(np.float32)).resize((NCELL * k, NCELL * k), Image.BILINEAR), dtype=float)
    fe = E0 + (np.arange(NCELL * k) + 0.5) / k
    fn = N1 - (np.arange(NCELL * k) + 0.5) / k
    fee, fnn = np.meshgrid(fe, fn)
    ti = ((N1 - fnn) / 0.25).astype(int).clip(0, tex.shape[0] - 1)
    tj = ((fee - E0) / 0.25).astype(int).clip(0, tex.shape[1] - 1)
    g_xyz = np.column_stack([fee.ravel(), fnn.ravel(), fine.ravel()])
    h_xyz = pts + [E0, N1 - NCELL, 0]
    is_pool = np.all(np.abs(cols - np.array(mesh.get("pool_rgb", [64, 172, 212]))) < 3, axis=1)
    cls = np.concatenate([np.full(len(g_xyz), 2), np.where(is_pool, 9, 6)]).astype(np.uint8)
    xyz = np.vstack([g_xyz, h_xyz])
    rgb = np.vstack([tex[ti, tj].reshape(-1, 3), cols]).astype(np.uint16) * 257
    hdr = laspy.LasHeader(point_format=2, version="1.2")
    hdr.offsets = [E0, N1 - NCELL, 0.0]
    hdr.scales = [0.001] * 3
    hdr.add_crs(pyproj.CRS.from_epsg(7855))
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = xyz.T
    las.red, las.green, las.blue = rgb.T
    las.classification = cls
    las.write(OUT / "site_with_house.las")
    tm = trimesh.Trimesh(np.column_stack([grid[:, 0], grid[:, 2], NCELL - grid[:, 1]]), f[:, ::-1], process=False)
    vcol = np.zeros((len(grid), 4), np.uint8)
    for face, c in zip(f, col):
        vcol[face] = (*c, 255)
    tm.visual.vertex_colors = vcol
    tm.export(OUT / "house_massing.glb")
    print(f"house triangles {len(f)}, LAS points {len(xyz)} (house {len(h_xyz)})")


def cmd_plan():
    """site plan on the aerial: lot, the two terraces, house parts, pool, lawn, outdoor room, garage"""
    from PIL import ImageDraw, ImageFont
    tex = Image.open(OUT / "ortho_texture.jpg").convert("RGB")
    S = 4  # px per metre in the texture
    cx, cy = (A + U * 12 + NV * 22 - [E0, N1 - NCELL])
    box_px = (int((cx - 38) * S), int((NCELL - cy - 38) * S), int((cx + 38) * S), int((NCELL - cy + 38) * S))
    img = tex.crop(box_px).resize((1200, 1200), Image.LANCZOS)
    sc = 1200 / (76 * S)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    F = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)  # noqa: E731

    def px(u, n):
        g = A + U * u + NV * n - [E0, N1 - NCELL]
        return ((g[0] * S - box_px[0]) * sc, ((NCELL - g[1]) * S - box_px[1]) * sc)

    def shape_f(geom, fill, outline, width=2):
        for g in getattr(geom, "geoms", [geom]):
            if g.area > 0.05:
                d.polygon([px(*p) for p in g.exterior.coords], fill=fill, outline=outline, width=width)

    def label(u, n, text, size=20):
        d.text(px(u, n), text, fill="white", anchor="mm", font=F(size), stroke_width=3, stroke_fill="black")

    inner = LOT_F.buffer(-0.3, join_style=2)
    up = upper_zone()
    shape_f(inner.difference(up).intersection(fbox(-20, 60, 2.2, 90)).difference(garage_zone()), (205, 196, 170, 120), (205, 196, 170, 255))
    shape_f(inner.intersection(up), (150, 170, 120, 150), (150, 170, 120, 255))
    shape_f(inner.intersection(fbox(-20, 60, 2.2, BANK_TOP)).difference(garage_zone()), (120, 140, 90, 130), None)
    shape_f(fbox(*DESIGN["court"]).intersection(inner).difference(footprint_f()), (226, 220, 206, 170), None)
    shape_f(fbox(*DESIGN["lawn"]), (95, 140, 60, 210), (60, 100, 40, 255))
    shape_f(inner.intersection(garage_zone()), (110, 110, 112, 170), (40, 40, 40, 255))
    for k, lab in (("upper", "2-storey\nliving +\nmain bed"), ("main", "Entry /\nliving"), ("wing", "Garage wing\n(garage below)"), ("rear", "Family\nwing")):
        shape_f(fbox(*DESIGN[k]), (42, 43, 45, 230), (230, 230, 230, 255), 3)
        u0, u1, n0, n1 = DESIGN[k]
        label((u0 + u1) / 2, (n0 + n1) / 2, lab)
    g = DESIGN["garage"]
    d.line([px(g[0], g[2]), px(g[1], g[2]), px(g[1], g[3]), px(g[0], g[3]), px(g[0], g[2])], fill=(240, 200, 120, 255), width=2)
    shape_f(fbox(*DESIGN["pool"]), (60, 170, 215, 235), (20, 90, 140, 255), 3)
    label(14.75, 23.8, "Pool")
    shape_f(fbox(*DESIGN["outdoor"]), (170, 130, 90, 220), (90, 60, 30, 255), 3)
    label(22.7, 26.0, "Outdoor\nfire room", 16)
    label(14.2, 30.0, "Lawn", 18)
    label(13.5, 19.2, "North courtyard", 18)
    label(12.0, 38.0, "Upper garden terrace 158.65", 17)
    label(4.0, 5.0, "Front garden + bank", 15)
    # terrace wall
    e0 = shapely.LineString([(-20, n_edge(-20)), (60, n_edge(60))]).intersection(inner)
    d.line([px(*p) for p in e0.coords], fill=(250, 250, 250, 255), width=4)
    d.line([px(*to_frame(*p)) for p in LOT.exterior.coords], fill=(230, 40, 40, 255), width=5)
    d.text(px(9, -2.5), "Waugh Street", fill="white", anchor="mm", font=F(24), stroke_width=3, stroke_fill="black")
    out = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    dd = ImageDraw.Draw(out)
    a = json.loads((OUT / "house_massing.json").read_text())["assumptions"]
    dd.rectangle((0, 0, 1200, 70), fill=(20, 20, 20))
    dd.text((16, 10), "Modern black barnhouse — Lot 624 DP821594, 36 Waugh St Griffith", font=F(24), fill="white")
    dd.text((16, 42), f"Footprint ≈{a['footprint_m2']} m² incl. garage ({a['coverage']:.0%}) · pad 157.45 + upper terrace 158.65 to every "
                      f"boundary · cut ≤{a['max_cut']:.1f} m, fill ≤{a['max_fill']:.1f} m · indicative", font=F(16), fill=(220, 220, 220))
    dd.rectangle((0, 1160, 1200, 1200), fill=(20, 20, 20))
    dd.text((16, 1170), "Aerial and cadastre © State of New South Wales (Spatial Services), CC BY 4.0 · north is up", font=F(15), fill="white")
    dd.polygon([(1160, 90), (1145, 135), (1175, 135)], fill="white", outline="black")
    dd.text((1160, 150), "N", font=F(22), fill="white", anchor="mm", stroke_width=3, stroke_fill="black")
    dest = ROOT / "designs" / "black_barn_plan.png"
    dest.parent.mkdir(exist_ok=True)
    out.save(dest)
    print(dest)


if __name__ == "__main__":
    {"ground": cmd_ground, "finish": cmd_finish, "plan": cmd_plan}[sys.argv[1]]()
