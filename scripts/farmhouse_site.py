"""Site fit for the modern-farmhouse design (option B) on Lot 624 DP821594.

  python farmhouse_site.py ground   -> designed ground + pad outline; writes data/3d/house_massing.json (ground only)
                                      and refreshes data/3d/render_scene.json via build_3d_render
  python farmhouse_site.py finish   -> reads data/3d/farmhouse_mesh.json (exported by farmhouse_cycles.py),
                                      writes house_massing.json mesh, site_with_house.las, house_massing.glb
  python farmhouse_site.py plan     -> site plan PNG over the aerial (site_plan_farmhouse.png)

Street frame: u along the frontage chord (SW end = 0), n into the lot, heights in m AHD.
All dimensions are design assumptions for visualisation, not a design for approval.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon, box
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import house_massing as hm  # noqa: E402  (lot, frame, grid constants, LAS/GLB writers)

OUT = ROOT / "data" / "3d"
LOT, A, U, NV = hm.LOT, hm.A, hm.U, hm.NV
E0, N1, NCELL = hm.E0, hm.N1, hm.NCELL
FFL, PAD, GARAGE_FL = hm.FFL, hm.PAD, hm.GARAGE_FL

# ---- design (street frame, metres) ---------------------------------------------------------------
DESIGN = {
    "main": (1.0, 15.6, 9.4, 21.4),        # single storey + attic, ridge parallel to the street
    "wing": (15.6, 22.8, 7.5, 22.6),       # gable wing (single storey + attic) over the garage
    "rear": (1.0, 9.0, 21.4, 32.3),        # long single-storey west family wing, glass onto the courtyard (client-approved)
    "upper": (1.0, 22.8, 11.5, 17.5),      # the view storey (street SSE + east windows), own gable roof; ridge ≤ 9 m above natural ground
    "verandah": (0.8, 15.4, 7.0, 9.4),      # deep verandah across the main body
    "outdoor": (22.8, 26.0, 25.0, 31.0),    # covered outdoor room with a stone fireplace
    "garage": (16.4, 22.6, 6.3, 13.8),     # street level, under the wing; door 5.5 m+ from the boundary (DCP 4.7.1c)
    "pool": (12.5, 22.5, 28.5, 32.0),
    "stair": (1.6, 3.4, 1.0, 5.4),         # front steps from the gate (SW end) up to the lawn
    "pad_back": 36.0,                       # rear edge of the courtyard pad
    "terrace": [[15.5, 36.4], [28.8, 36.4], [30.5, 46.0], [15.5, 41.0]],   # small raised rear garden (NE rear corner, ≈1 m up)
    "terrace_z": 1.0,                       # terrace top above the pad
}
WALL_TOP = 156.4   # low street wall (about 0.9 m above the footpath)
BANK_TOP = 5.0     # the planted bank rises from the wall to lawn level by n = 5


def frame_poly(u0, u1, n0, n1):
    return Polygon([tuple(A + U * u + NV * n) for u, n in ((u0, n0), (u1, n0), (u1, n1), (u0, n1))])


def to_frame(e, n):
    d = np.array([e, n]) - A
    return float(d @ U), float(d @ NV)


def pad_polygon():
    """house, courtyard and rear-lawn pad: the whole lot inset 0.4 m behind the street wall, less the garage zone"""
    big = frame_poly(-20, 45, 2.2, 60)
    g = DESIGN["garage"]
    return LOT.buffer(-0.4).intersection(big).difference(frame_poly(g[0], 40, -2, g[3]))


def cut_fill_volumes(ground, dem, pool=None):
    """bulk earthworks from the 1 m grids (m³, in-situ, no bulking): cut where the design is below natural ground,
    fill where it is above, over the lot. Pool excavation is reported separately (shell depth 1.7 m)."""
    lot = prep(LOT.buffer(0.3))
    d = ground - dem
    cut = fill = 0.0
    for r in range(NCELL):
        for c in range(NCELL):
            if d[r, c] != 0 and lot.contains(Point(E0 + c + 0.5, N1 - r - 0.5)):
                cut += max(-d[r, c], 0.0)
                fill += max(d[r, c], 0.0)
    pool_m3 = 0.0 if pool is None else (pool[1] - pool[0]) * (pool[3] - pool[2]) * 1.7
    return round(cut), round(fill), round(pool_m3)


def terrace_polygon():
    """raised rear garden terrace in the NE rear corner, inside the lot"""
    return Polygon([tuple(A + U * u + NV * n) for u, n in DESIGN["terrace"]]).intersection(LOT.buffer(-0.4))


def lawn_triangles(pad, stair):
    """the level part of the pad (behind the front bank, clear of the front steps) as triangles in (u, n),
    so lawn can run right out to the side and rear boundaries; paving and buildings sit just above it.
    The pool (with its coping) and the raised terrace are cut out so the lawn never covers the water."""
    from shapely.ops import triangulate
    p = DESIGN["pool"]
    lawn = pad.intersection(frame_poly(-20, 45, BANK_TOP, 60)).difference(frame_poly(stair[0] - 0.4, stair[1] + 0.4, 0, stair[3]))
    lawn = lawn.difference(frame_poly(p[0] - 0.35, p[1] + 0.35, p[2] - 0.35, p[3] + 0.35)).difference(terrace_polygon())
    from shapely import constrained_delaunay_triangles
    tris = []
    for part in getattr(lawn, "geoms", [lawn]):
        if part.geom_type != "Polygon" or part.area < 0.05:
            continue
        for t in constrained_delaunay_triangles(part.segmentize(1.5)).geoms:
            tris.append([[round(c, 3) for c in to_frame(*q)] for q in list(t.exterior.coords)[:3]])
    return tris


def designed_ground(dem):
    pad = prep(pad_polygon())
    terr = prep(terrace_polygon())
    lot = prep(LOT.buffer(0.5))
    g = dem.copy()
    gz = DESIGN["garage"]
    s = DESIGN["stair"]
    for r in range(NCELL):
        for c in range(NCELL):
            e, n = E0 + c + 0.5, N1 - r - 0.5
            if not lot.contains(Point(e, n)):
                continue
            uu, vv = to_frame(e, n)
            if gz[0] <= uu <= 22.9 and -1.0 <= vv < gz[3]:                       # garage + driveway
                g[r, c] = GARAGE_FL if vv >= gz[2] else np.interp(vv, [-1.0, gz[2]], [g[r, c], GARAGE_FL])
            elif s[0] - 0.5 <= uu <= s[1] + 0.5 and s[2] <= vv <= s[3]:          # stair cut
                g[r, c] = np.interp(vv, [s[2], s[3]], [min(g[r, c], 155.7), PAD])
            elif 0.3 <= uu <= gz[0] and 2.2 <= vv < BANK_TOP:                   # planted bank behind the low street wall
                g[r, c] = np.interp(vv, [2.2, BANK_TOP], [WALL_TOP, PAD])
            elif terr.contains(Point(e, n)):
                g[r, c] = PAD + DESIGN["terrace_z"]
            elif pad.contains(Point(e, n)):
                g[r, c] = PAD
            elif 0.3 <= uu <= 16.2 and 0.0 <= vv < 1.6:                          # front garden bed
                g[r, c] = min(g[r, c], 155.6)
    return g


def cmd_ground():
    dem = np.array(Image.open(OUT / "dem_240m.tif"), dtype=np.float64)
    ground = designed_ground(dem)
    pad = pad_polygon()
    outline = [to_frame(*p) for p in pad.exterior.coords]
    data = json.loads((OUT / "house_massing.json").read_text())
    data["ground_cm"] = np.round(ground[::-1] * 100).astype(int).ravel().tolist()
    data["design"] = "farmhouse"
    data["pad_outline"] = [[round(u, 3), round(n, 3)] for u, n in outline]
    data["lawn_tris"] = lawn_triangles(pad, DESIGN["stair"])
    data["layout"] = dict(DESIGN, terrace_outline=[[round(c, 3) for c in to_frame(*q)] for q in terrace_polygon().exterior.coords])
    data["front"] = {"wall_top": WALL_TOP - FFL, "bank_top": BANK_TOP}
    cut = ground - dem
    footprint = sum(frame_poly(*DESIGN[k]).area for k in ("main", "wing", "rear")) + frame_poly(*DESIGN["garage"]).area
    data["assumptions"] = {"FFL": FFL, "terrace": PAD, "garage_floor": GARAGE_FL, "ridge": FFL + 6.96,
                           "eave": FFL + 3.0, "footprint_m2": round(footprint), "coverage": round(footprint / LOT.area, 3),
                           "max_cut": round(float(-cut.min()), 2), "max_fill": round(float(cut.max()), 2)}
    vc, vf, vp = cut_fill_volumes(ground, dem, DESIGN.get("pool"))
    data["assumptions"].update({"cut_m3": vc, "fill_m3": vf, "pool_m3": vp})
    print(f"earthworks: cut ≈{vc} m³, fill ≈{vf} m³, net {'export' if vc > vf else 'import'} ≈{abs(vc - vf)} m³; pool dig ≈{vp} m³")
    (OUT / "house_massing.json").write_text(json.dumps(data, separators=(",", ":")))
    print("pad area %.0f m2, footprint %.0f m2 (%.0f%%), cut %.2f, fill %.2f" % (
        pad.area, footprint, 100 * footprint / LOT.area, -cut.min(), cut.max()))
    for k in ("main", "wing", "rear", "garage", "pool", "outdoor"):
        assert LOT.contains(frame_poly(*DESIGN[k])), k


def cmd_finish():
    import laspy
    import pyproj
    import trimesh
    mesh = json.loads((OUT / "farmhouse_mesh.json").read_text())
    data = json.loads((OUT / "house_massing.json").read_text())
    v = np.array(mesh["positions"], float).reshape(-1, 3)   # u, n, z-above-FFL
    f = np.array(mesh["faces"], int).reshape(-1, 3)
    col = np.array(mesh["colors"], int).reshape(-1, 3)
    grid = hm.to_grid(v[:, 0], v[:, 1], v[:, 2] + FFL)       # grid-local x, y, AHD
    data["positions"] = np.round(grid, 3).ravel().tolist()
    data["faces"] = f.ravel().tolist()
    data["colors"] = col.ravel().tolist()
    (OUT / "house_massing.json").write_text(json.dumps(data, separators=(",", ":")))

    class M:  # adapter for house_massing.sample()
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
    """site plan on the aerial: lot, pad, house parts, pool, garage, with areas"""
    from PIL import ImageDraw, ImageFont
    tex = Image.open(OUT / "ortho_texture.jpg").convert("RGB")
    S = 4  # px per metre in the texture
    # crop around the lot
    cx, cy = (A + U * 12 + NV * 20 - [E0, N1 - NCELL])
    box_px = (int((cx - 38) * S), int((NCELL - cy - 38) * S), int((cx + 38) * S), int((NCELL - cy + 38) * S))
    img = tex.crop(box_px).resize((1200, 1200), Image.LANCZOS)
    sc = 1200 / (76 * S)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    def px(u, n):
        g = A + U * u + NV * n - [E0, N1 - NCELL]
        return ((g[0] * S - box_px[0]) * sc, ((NCELL - g[1]) * S - box_px[1]) * sc)

    def rect(key, fill, outline, label=None):
        u0, u1, n0, n1 = DESIGN[key]
        pts = [px(u0, n0), px(u1, n0), px(u1, n1), px(u0, n1)]
        d.polygon(pts, fill=fill, outline=outline, width=3)
        if label:
            c = px((u0 + u1) / 2, (n0 + n1) / 2)
            d.text(c, label, fill="white", anchor="mm", font=F(20), stroke_width=3, stroke_fill="black")

    F = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)
    pad = pad_polygon()
    d.polygon([px(*to_frame(*p)) for p in pad.exterior.coords], fill=(214, 200, 170, 90), outline=(214, 200, 170, 255), width=2)
    rect("garage", (90, 90, 95, 200), (30, 30, 30, 255), "Garage")
    rect("verandah", (170, 130, 90, 170), (90, 60, 30, 255))
    rect("main", (245, 245, 240, 220), (40, 40, 40, 255), "Living / entry\n+ attic")
    rect("wing", (245, 245, 240, 235), (40, 40, 40, 255), "2-storey\nwing")
    rect("rear", (245, 245, 240, 220), (40, 40, 40, 255), "Family\nliving")
    rect("pool", (60, 170, 215, 230), (20, 90, 140, 255), "Pool")
    rect("outdoor", (170, 130, 90, 200), (90, 60, 30, 255), "Outdoor\nroom")
    c = px(17.5, 34.0)
    d.text(c, "North courtyard", fill="white", anchor="mm", font=F(22), stroke_width=3, stroke_fill="black")
    d.line([px(*to_frame(*p)) for p in LOT.exterior.coords], fill=(230, 40, 40, 255), width=5)
    d.text(px(9, -2.5), "Waugh Street", fill="white", anchor="mm", font=F(24), stroke_width=3, stroke_fill="black")
    out = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    dd = ImageDraw.Draw(out)
    a = json.loads((OUT / "house_massing.json").read_text())["assumptions"]
    dd.rectangle((0, 0, 1200, 70), fill=(20, 20, 20))
    dd.text((16, 10), "Option B: modern farmhouse — Lot 624 DP821594, 36 Waugh St Griffith", font=F(24), fill="white")
    dd.text((16, 42), f"Ground-floor footprint ≈{a['footprint_m2']} m² incl. garage ({a['coverage']:.0%} of 1,093 m²) · courtyard pad · indicative only",
            font=F(17), fill=(220, 220, 220))
    dd.rectangle((0, 1160, 1200, 1200), fill=(20, 20, 20))
    dd.text((16, 1170), "Aerial and cadastre © State of New South Wales (Spatial Services), CC BY 4.0 · north is up", font=F(15), fill="white")
    # north arrow (texture is north-up)
    dd.polygon([(1160, 90), (1145, 135), (1175, 135)], fill="white", outline="black")
    dd.text((1160, 150), "N", font=F(22), fill="white", anchor="mm", stroke_width=3, stroke_fill="black")
    out.save(ROOT / "site_plan_farmhouse.png")
    print("site_plan_farmhouse.png")


if __name__ == "__main__":
    {"ground": cmd_ground, "finish": cmd_finish, "plan": cmd_plan}[sys.argv[1]]()
