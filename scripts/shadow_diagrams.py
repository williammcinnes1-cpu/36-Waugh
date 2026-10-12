"""Winter-solstice shadow diagrams (Griffith Residential DCP 4.6e): 22 June at 9 am, 12 noon and 3 pm AEST.

Projects every triangle of the design's exported mesh (data/3d/farmhouse_mesh.json; u, n, z above FFL)
along the sun direction onto the ground (designed ground on the lot, natural ground elsewhere) and
draws the shadows over the aerial photo with the lot and neighbouring lots.
Existing neighbouring buildings are not modelled, so their own shadows are not shown.

    python3 scripts/shadow_diagrams.py [mesh.json] [out.png]
"""
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyproj
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union
from shapely import transform as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import house_massing as hm  # noqa: E402

D3 = ROOT / "data" / "3d"
LAT, LON = -34.2645, 146.0779
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FENCE_COLOURS = {(217, 208, 184), (158, 169, 141)}


def sun_position(when_utc):
    """NOAA solar position: returns (azimuth from north clockwise, elevation) in degrees"""
    jd = when_utc.timestamp() / 86400 + 2440587.5
    jc = (jd - 2451545) / 36525
    l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    c = (math.sin(math.radians(m)) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + math.sin(math.radians(2 * m)) * (0.019993 - 0.000101 * jc)
         + math.sin(math.radians(3 * m)) * 0.000289)
    lam = l0 + c - 0.00569 - 0.00478 * math.sin(math.radians(125.04 - 1934.136 * jc))
    eps0 = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * math.cos(math.radians(125.04 - 1934.136 * jc))
    decl = math.degrees(math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(lam))))
    y = math.tan(math.radians(eps / 2)) ** 2
    eqt = 4 * math.degrees(y * math.sin(2 * math.radians(l0)) - 2 * e * math.sin(math.radians(m)) + 4 * e * y * math.sin(math.radians(m)) * math.cos(2 * math.radians(l0))
                           - 0.5 * y * y * math.sin(4 * math.radians(l0)) - 1.25 * e * e * math.sin(2 * math.radians(m)))
    minutes = when_utc.hour * 60 + when_utc.minute + when_utc.second / 60
    tst = (minutes + eqt + 4 * LON) % 1440
    ha = tst / 4 - 180
    lat, dec = math.radians(LAT), math.radians(decl)
    cosz = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(math.radians(ha))
    zen = math.degrees(math.acos(max(-1, min(1, cosz))))
    az = math.degrees(math.atan2(math.sin(math.radians(ha)), math.cos(math.radians(ha)) * math.sin(lat) - math.tan(dec) * math.cos(lat))) + 180
    return az % 360, 90 - zen


def main(mesh_path=D3 / "farmhouse_mesh.json", out=ROOT / "shadow_diagrams_22june.png"):
    mesh = json.loads(Path(mesh_path).read_text())
    v = np.array(mesh["positions"], float).reshape(-1, 3)
    f = np.array(mesh["faces"], int).reshape(-1, 3)
    col = [tuple(c) for c in np.array(mesh["colors"], int).reshape(-1, 3)]
    ground = np.array(json.loads((D3 / "house_massing.json").read_text())["ground_cm"], float).reshape(hm.NCELL, hm.NCELL)[::-1] / 100
    to_mga = lambda u, n: hm.A + hm.U * u + hm.NV * n

    def g_at(e, n):  # ground AHD from the designed grid (row 0 = north)
        c, r = e - hm.E0 - 0.5, hm.N1 - n - 0.5
        i, j = int(np.clip(c, 0, hm.NCELL - 2)), int(np.clip(r, 0, hm.NCELL - 2))
        fx, fy = np.clip(c - i, 0, 1), np.clip(r - j, 0, 1)
        return ground[j, i] * (1 - fx) * (1 - fy) + ground[j, i + 1] * fx * (1 - fy) + ground[j + 1, i] * (1 - fx) * fy + ground[j + 1, i + 1] * fx * fy

    pts = np.array([to_mga(u, n) for u, n, _ in v])           # MGA easting, northing
    zs = v[:, 2] + hm.FFL
    keep = [k for k, tri in enumerate(f) if col[k] not in FENCE_COLOURS and v[tri, 2].max() > 0.4]
    footprint = unary_union([Polygon(pts[tri]) for tri in f[keep] if Polygon(pts[tri]).area > 1e-4]).buffer(0)

    times = [("9:00 am", 9), ("12 noon", 12), ("3:00 pm", 15)]
    to_ll = pyproj.Transformer.from_crs(7855, 4326, always_xy=True)
    to_m = pyproj.Transformer.from_crs(4326, 7855, always_xy=True)
    lot = tf(shape(json.loads((ROOT / "data" / "lot.geojson").read_text())["features"][0]["geometry"]), lambda c: np.column_stack(to_m.transform(c[:, 0], c[:, 1])))
    neigh = [(f_["properties"]["lotidstring"], tf(shape(f_["geometry"]), lambda c: np.column_stack(to_m.transform(c[:, 0], c[:, 1]))))
             for f_ in json.loads((ROOT / "data" / "neighbour_lots.geojson").read_text())["features"] if f_["properties"]["lotidstring"] != "624//DP821594"]

    tex = Image.open(D3 / "ortho_texture.jpg").convert("RGB")
    S = 4  # px per metre
    cx, cy = lot.centroid.x, lot.centroid.y
    half = 42
    box_px = (int((cx - half - hm.E0) * S), int((hm.N1 - cy - half) * S), int((cx + half - hm.E0) * S), int((hm.N1 - cy + half) * S))
    W = 760
    sc = W / (2 * half * S)
    px = lambda e, n: (((e - hm.E0) * S - box_px[0]) * sc, ((hm.N1 - n) * S - box_px[1]) * sc)
    F = lambda s: ImageFont.truetype(FONT, s)
    panels, report = [], []
    for label, hour in times:
        when = datetime(2026, 6, 22, hour, 0, tzinfo=timezone(timedelta(hours=10))).astimezone(timezone.utc)
        az, el = sun_position(when)
        s = np.array([math.sin(math.radians(az)) * math.cos(math.radians(el)), math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(el))])
        proj = np.zeros((len(pts), 2))
        for k, ((e, n), z) in enumerate(zip(pts, zs)):
            t = 0.0
            for _ in range(5):  # walk down the sun ray until it meets the ground
                pe, pn = e - s[0] * t, n - s[1] * t
                t = max((z - g_at(pe, pn)) / s[2], 0.0)
            proj[k] = (e - s[0] * t, n - s[1] * t)
        shadow = unary_union([Polygon(proj[tri]).buffer(0) for tri in f[keep] if Polygon(proj[tri]).area > 1e-4]).buffer(0.05).buffer(-0.05)
        cast = shadow.difference(footprint)
        img = tex.crop(box_px).resize((W, W), Image.LANCZOS).convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        for geom in getattr(cast, "geoms", [cast]):
            if geom.geom_type == "Polygon" and not geom.is_empty:
                d.polygon([px(*p) for p in geom.exterior.coords], fill=(20, 30, 80, 150))
                for hole in geom.interiors:
                    d.polygon([px(*p) for p in hole.coords], fill=(0, 0, 0, 0))
        for geom in getattr(footprint, "geoms", [footprint]):
            d.polygon([px(*p) for p in geom.exterior.coords], fill=(245, 243, 238, 235), outline=(30, 30, 30, 255))
        for lid, poly in neigh:
            d.line([px(*p) for p in poly.exterior.coords], fill=(255, 220, 90, 255), width=2)
            c = poly.intersection(lot.buffer(25)).centroid
            if not c.is_empty:
                over = cast.intersection(poly).area
                report.append((label, lid, over))
                d.text(px(c.x, c.y), "Lot " + lid.replace("//", " "), fill="white", anchor="mm", font=F(13), stroke_width=3, stroke_fill="black")
        d.line([px(*p) for p in lot.exterior.coords], fill=(230, 40, 40, 255), width=4)
        img = Image.alpha_composite(img, ov).convert("RGB")
        dd = ImageDraw.Draw(img)
        dd.rectangle((0, 0, W, 54), fill=(18, 18, 20))
        dd.text((14, 8), f"22 June · {label} AEST", font=F(22), fill="white")
        dd.text((14, 33), f"sun azimuth {az:.0f}°, elevation {el:.0f}°", font=F(14), fill=(210, 210, 210))
        sx, sy = W - 40, 90  # north arrow (the aerial is north-up)
        dd.polygon([(sx, sy - 25), (sx - 10, sy + 5), (sx + 10, sy + 5)], fill="white", outline="black")
        dd.text((sx, sy + 18), "N", font=F(16), fill="white", anchor="mm", stroke_width=3, stroke_fill="black")
        bar = 10 * S * sc  # 10 m scale bar
        dd.rectangle((14, W - 34, 14 + bar, W - 28), fill="white", outline="black")
        dd.text((14 + bar + 8, W - 38), "10 m", font=F(14), fill="white", stroke_width=3, stroke_fill="black")
        panels.append(img)
    sheet = Image.new("RGB", (W * 3 + 40, W + 110), (245, 245, 242))
    for i, p in enumerate(panels):
        sheet.paste(p, (10 + i * (W + 10), 10))
    d = ImageDraw.Draw(sheet)
    d.text((14, W + 22), "Shadow diagrams, winter solstice — 36 Waugh Street, Griffith (Lot 624 DP821594), farmhouse option — indicative only",
           font=F(18), fill=(30, 30, 30))
    d.text((14, W + 50), "Blue: shadow cast by the proposed house beyond its own footprint. Red: the lot. Yellow: neighbouring lots. "
                         "Existing neighbouring buildings are not modelled, so their own shadows are not shown.", font=ImageFont.truetype(FONT.replace("-Bold", ""), 14), fill=(60, 60, 60))
    d.text((14, W + 74), "Aerial and cadastre © State of New South Wales (Spatial Services), CC BY 4.0. "
                         "Heights from the NSW 5 m elevation grid; confirm with a detail and level survey.", font=ImageFont.truetype(FONT.replace("-Bold", ""), 14), fill=(60, 60, 60))
    sheet.save(out)
    for label, lid, area in report:
        print(f"{label}: shadow on {lid} ≈ {area:.0f} m²")
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
