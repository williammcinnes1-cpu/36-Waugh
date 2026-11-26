"""Site plan for 10 Mancini Drive, Lake Wyangan (Lot 5 DP285693): aerial, cadastral boundary, 0.25 m contours from
the NSW 5 m DEM, and the house + pool from House.IGS placed on the house in the aerial.

Inputs (data/): tiles/, lot.geojson, dem_1m.tif, model_plan.json (house/pool/lot outlines in the model's plan frame),
model_to_mga.json (fitted rotation + translation). Outputs: site_plan.png, data/placed_model.geojson, data/levels.json.
"""
import json, math, sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import ImageDraw, ImageFont, Image
from shapely.geometry import shape, mapping, Point
from shapely import affinity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Aerial, lot_mga, DATA, SITE, TO_LL

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
F = lambda s, b=False: ImageFont.truetype(BOLD if b else FONT, s)


def placed():
    M = json.load(open(DATA / "model_plan.json"))
    t = json.load(open(DATA / "model_to_mga.json"))
    put = lambda g: affinity.translate(affinity.rotate(shape(g), t["rotation_deg"], origin=(0, 0)), t["tx"], t["ty"])
    return put(M["house"]), put(M["pool"]), put(M["slab"]), t, {k: put(M[k]) for k in ("pavers", "poolhouse", "wall")}


def main():
    A = Aerial()
    lot = lot_mga()
    house, pool, mlot, t, X = placed()                                          # all from House.IGS
    surround, poolhouse, wall = X["pavers"], X["poolhouse"], X["wall"]
    import poolhouse as PH
    poolhouse = PH.footprint_world()                                       # L-shaped pool house (scripts/poolhouse.py)
    dem = rasterio.open(DATA / "dem_1m.tif")
    Z = dem.read(1)
    tr = dem.transform
    zat = lambda e, n: float(Z[int((tr.f - n) / -tr.e), int((e - tr.c) / tr.a)])

    # --- levels over the lot: range, plane fit (fall and direction)
    rows, cols = np.mgrid[0:Z.shape[0], 0:Z.shape[1]]
    E = tr.c + (cols + 0.5) * tr.a
    N = tr.f + (rows + 0.5) * tr.e
    from shapely import contains_xy
    inside = contains_xy(lot, E, N)
    ze, ne, zz = E[inside], N[inside], Z[inside]
    G = np.column_stack([ze - ze.mean(), ne - ne.mean(), np.ones_like(ze)])
    (gx, gy, _), *_ = np.linalg.lstsq(G, zz, rcond=None)
    slope = math.hypot(gx, gy)
    fall_bearing = (math.degrees(math.atan2(-gx, -gy)) + 360) % 360       # direction the ground falls towards
    levels = {"lot_min": round(float(zz.min()), 2), "lot_max": round(float(zz.max()), 2), "lot_mean": round(float(zz.mean()), 2),
              "plane_slope_pct": round(100 * slope, 2), "falls_towards_bearing_deg": round(fall_bearing),
              "house_ground": round(zat(house.centroid.x, house.centroid.y), 2),
              "pool_ground": round(zat(pool.centroid.x, pool.centroid.y), 2)}
    corners = [c for c in lot.exterior.coords][:-1]
    json.dump(levels, open(DATA / "levels.json", "w"), indent=1)
    print(levels)

    # --- placed model as GeoJSON (WGS84) for reuse
    to_ll = lambda g: affinity.affine_transform(g, [1, 0, 0, 1, 0, 0]) if False else \
        type(g)([TO_LL.transform(*c) for c in g.exterior.coords])
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": n, "area_m2": round(g.area, 1), "source": "House.IGS, fitted to the 2014 aerial"},
         "geometry": mapping(to_ll(g))} for n, g in (("house", house), ("pool", pool), ("pavers", surround), ("planned_pool_house", poolhouse), ("wall", wall))]}
    json.dump(fc, open(DATA / "placed_model.geojson", "w"))

    # --- picture: crop around the lot, 1.6x
    pts = [A.mga2px(*c) for c in lot.exterior.coords]
    pad = 170
    u0, v0 = min(p[0] for p in pts) - pad, min(p[1] for p in pts) - pad
    u1, v1 = max(p[0] for p in pts) + pad, max(p[1] for p in pts) + pad
    K = 1.6
    img = A.img.crop((int(u0), int(v0), int(u1), int(v1))).resize((int((u1 - u0) * K), int((v1 - v0) * K)), Image.LANCZOS)
    img = Image.blend(img, Image.new("RGB", img.size, (255, 255, 255)), 0.18).convert("RGBA")
    P = lambda e, n: ((A.mga2px(e, n)[0] - int(u0)) * K, (A.mga2px(e, n)[1] - int(v0)) * K)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    # contours (0.25 m) from the DEM; 1 m lines heavier and labelled
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lv = np.arange(math.floor(Z.min() * 4) / 4, Z.max() + 0.25, 0.25)
    cs = plt.contour(E, N, Z, levels=lv)
    labelled = []
    for lev, segs in zip(cs.levels, cs.allsegs):
        major = abs(lev - round(lev)) < 1e-6
        for seg in segs:
            if len(seg) < 2:
                continue
            xy = [P(e, n) for e, n in seg]
            d.line(xy, fill=(255, 140, 0, 235) if major else (255, 210, 120, 170), width=3 if major else 1)
            if major and len(seg) > 40:
                for k in range(len(seg) // 3, len(seg), max(1, len(seg) // 2)):
                    e, n = seg[k]
                    if lot.buffer(30).contains(Point(e, n)) and all(math.hypot(e - a, n - b) > 35 for a, b in labelled):
                        labelled.append((e, n))
                        d.text(P(e, n), f"{lev:.0f}", font=F(15, True), fill=(255, 150, 0), anchor="mm", stroke_width=3, stroke_fill=(30, 30, 30))
    plt.close("all")

    # boundary with edge lengths
    d.line([P(*c) for c in lot.exterior.coords], fill=(235, 30, 30, 255), width=5)
    inner = lot.buffer(-10, join_style="mitre")                                  # 10 m inside the boundary
    d.line([P(*c) for c in inner.exterior.coords], fill=(40, 120, 255, 255), width=4)
    cs_ = list(lot.exterior.coords)
    for (a, b), (c, e_) in zip(cs_[:-1], cs_[1:]):
        L = math.hypot(c - a, e_ - b)
        if L > 15:
            mx, my = P((a + c) / 2, (b + e_) / 2)
            d.text((mx, my), f"{L:.1f} m", font=F(16, True), fill="white", anchor="mm", stroke_width=3, stroke_fill=(150, 0, 0))
    # house and pool from the IGES model
    d.polygon([P(*c) for c in house.exterior.coords], fill=(255, 255, 255, 70), outline=(255, 255, 255, 255))
    d.line([P(*c) for c in house.exterior.coords], fill=(255, 255, 255, 255), width=3)
    d.polygon([P(*c) for c in surround.exterior.coords], fill=(236, 236, 228, 235), outline=(90, 90, 90, 255))
    d.polygon([P(*c) for c in pool.exterior.coords], fill=(40, 170, 230, 235), outline=(0, 90, 160, 255))
    d.polygon([P(*c) for c in wall.exterior.coords], fill=(60, 60, 60, 255))
    d.polygon([P(*c) for c in poolhouse.exterior.coords], fill=(255, 220, 120, 110), outline=(255, 200, 60, 255))
    d.line([P(*c) for c in poolhouse.exterior.coords], fill=(255, 190, 40, 255), width=4)
    qx, qy = P(poolhouse.centroid.x, poolhouse.centroid.y)
    d.text((qx, qy), "POOL HOUSE (L)\n≈207 m², hipped roof", font=F(14, True), fill="white", anchor="mm", align="center",
           stroke_width=3, stroke_fill=(120, 80, 0))
    hx, hy = P(house.centroid.x, house.centroid.y)
    d.text((hx, hy), f"HOUSE\n(House.IGS, {house.area:.0f} m²)", font=F(15, True), fill="white", anchor="mm", align="center",
           stroke_width=3, stroke_fill=(20, 20, 20))
    px_, py_ = P(pool.centroid.x, pool.centroid.y)
    d.text((px_, py_), "POOL 12.0 × 4.4 m\npavers 15.8 × 9.844 m", font=F(14, True), fill="white", anchor="mm", align="center",
           stroke_width=3, stroke_fill=(0, 60, 110))
    # spot heights at the lot corners (main corners only) and at the house and pool
    for e, n in corners:
        if min(math.hypot(e - a, n - b) for a, b in corners if (a, b) != (e, n)) > 8:
            x, y = P(e, n)
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
            d.text((x + 8, y - 8), f"{zat(e, n):.2f}", font=F(14, True), fill="white", stroke_width=3, stroke_fill="black")
    img = Image.alpha_composite(img, ov).convert("RGB")

    # frame: title, legend, north arrow, scale
    W = img.width + 460
    sheet = Image.new("RGB", (W, img.height + 70), (250, 250, 248))
    sheet.paste(img, (0, 0))
    s = ImageDraw.Draw(sheet)
    x0, y = img.width + 24, 24
    s.text((x0, y), "10 Mancini Drive", font=F(28, True), fill=(20, 20, 20)); y += 36
    s.text((x0, y), "Lake Wyangan NSW 2680", font=F(18), fill=(60, 60, 60)); y += 40
    rows_ = [("Lot / plan", "Lot 5 DP285693"), ("Lot area", f"≈{lot.area:,.0f} m² ({lot.area / 1e4:.2f} ha)"),
             ("Zoning", "R5 Large Lot Residential"), ("", "(Griffith LEP 2014)"),
             ("Bushfire / flood", "not mapped on the lot"),
             ("Ground levels", f"{levels['lot_min']:.1f}–{levels['lot_max']:.1f} m AHD"),
             ("Fall", f"≈{levels['plane_slope_pct']:.1f}% towards {levels['falls_towards_bearing_deg']}°"),
             ("At house", f"≈{levels['house_ground']:.1f} m AHD"), ("At pool", f"≈{levels['pool_ground']:.1f} m AHD"),
             ("Aerial", "NSW Imagery, Feb 2014, 10 cm")]
    for k, v in rows_:
        s.text((x0, y), k, font=F(15, True), fill=(40, 40, 40))
        s.text((x0 + 150, y), v, font=F(15), fill=(40, 40, 40)); y += 26
    y += 16
    for col, txt, w in (((235, 30, 30), "Lot boundary (cadastre)", 5), ((255, 140, 0), "1 m contour (m AHD)", 3),
                        ((255, 200, 110), "0.25 m contour", 1), ((40, 120, 255), "10 m inside the boundary", 4), ((255, 255, 255), "House from House.IGS", 3)):
        s.rectangle([x0, y + 6, x0 + 40, y + 14], fill=(110, 120, 100))
        s.line([(x0, y + 10), (x0 + 40, y + 10)], fill=col, width=w)
        s.text((x0 + 52, y), txt, font=F(15), fill=(40, 40, 40)); y += 26
    s.rectangle([x0, y + 2, x0 + 40, y + 18], fill=(40, 170, 230)); s.text((x0 + 52, y), "Pool and pavers (House.IGS)", font=F(15), fill=(40, 40, 40)); y += 26
    s.rectangle([x0, y + 2, x0 + 40, y + 18], fill=(255, 220, 120), outline=(255, 190, 40), width=3); s.text((x0 + 52, y), "Pool house, wrapping the pool end", font=F(15), fill=(40, 40, 40)); y += 34
    notes = ["House, pool, pavers and pool house from", "House.IGS, placed by fitting the house", f"footprint to the aerial roof (overlap {t['iou'] * 100:.0f}%,",
             f"rotation {t['rotation_deg']:.1f}°). The model's own lot outline",
             "lands on the cadastre within about a metre.", "",
             "Levels: NSW 5 m DEM (from stereo imagery),", "indicative only (±~1 m). Get a detail and",
             "level survey before siting the pool shed."]
    for n_ in notes:
        s.text((x0, y), n_, font=F(13), fill=(80, 80, 80)); y += 19
    # north arrow (aerial is north-up) and scale bar
    ax, ay = img.width - 60, 70
    s.polygon([(ax, ay - 40), (ax - 14, ay + 8), (ax, ay), (ax + 14, ay + 8)], fill="white", outline="black")
    s.text((ax, ay + 24), "N", font=F(20, True), fill="white", anchor="mm", stroke_width=3, stroke_fill="black")
    m_per_px = math.dist(A.px2mga(1000, 1000), A.px2mga(1100, 1000)) / 100 / K
    bar = 20 / m_per_px
    s.rectangle([30, img.height - 46, 30 + bar, img.height - 36], fill="white", outline="black")
    s.text((40 + bar, img.height - 52), "20 m", font=F(16, True), fill="white", stroke_width=3, stroke_fill="black")
    s.text((20, img.height + 14), "Aerial imagery, cadastre and elevation © State of New South Wales (Spatial Services), CC BY 4.0. "
                                  "GDA2020 / MGA zone 55. Indicative only — not a survey.", font=F(15), fill=(90, 90, 90))
    s.text((20, img.height + 38), "Zoning: NSW Planning Portal spatial layers (mapprod3.environment.nsw.gov.au).",
           font=F(15), fill=(90, 90, 90))
    out = SITE / "site_plan.png"
    sheet.save(out)
    print("wrote", out, sheet.size)


if __name__ == "__main__":
    main()
