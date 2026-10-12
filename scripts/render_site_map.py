"""Render the aerial site map for 36 Waugh Street, Griffith (Lot 624 DP821594).

Inputs (in data/): lot.geojson, address.geojson, bushfire.geojson, zoning.geojson,
contours_200m.geojson, dem_crop.npy and the z20 NSW_Imagery tiles in data/tiles/.
Output: site_aerial.png in the repo root.
"""
import json
import math
from pathlib import Path

import numpy as np
import pyproj
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, box, shape

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

Z, CX, CY = 20, 949770, 630633  # tile containing the lot centre
X0, Y0, N = CX - 2, CY - 2, 2**Z
SIZE = 1280
# 5 m DEM was exported on a 1 m MGA55 grid with this top-left corner
DEM_X0, DEM_Y1 = 415043, 6208213


def px(lon, lat):
    x = (lon + 180) / 360 * N
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * N
    return ((x - X0) * 256, (y - Y0) * 256)


def unpx(u, v):
    x, y = X0 + u / 256, Y0 + v / 256
    lon = x / N * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / N))))
    return lon, lat


def font(size):
    return ImageFont.truetype(FONT, size)


def load(name):
    return json.loads((DATA / name).read_text())


def parts(g):
    return getattr(g, "geoms", [g])


def main():
    img = Image.new("RGB", (SIZE, SIZE))
    for dx in range(5):
        for dy in range(5):
            img.paste(Image.open(DATA / f"tiles/{Z}_{Y0 + dy}_{X0 + dx}.jpg"), (dx * 256, dy * 256))

    view = box(*unpx(0, SIZE), *unpx(SIZE, 0))
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    def draw_lines(g, **kw):
        for p in parts(g.intersection(view)):
            if p.geom_type == "LineString":
                d.line([px(*c) for c in p.coords], **kw)

    # Bushfire prone land (vegetation buffer)
    bf = shape(load("bushfire.geojson")["features"][0]["geometry"]).buffer(0)
    for p in parts(bf.intersection(view)):
        if p.geom_type == "Polygon":
            d.polygon([px(*c) for c in p.exterior.coords], fill=(255, 140, 0, 70))
    draw_lines(bf.boundary, fill=(255, 140, 0, 230), width=3)

    # C2 zone boundary
    for f in load("zoning.geojson")["features"]:
        if f["properties"]["SYM_CODE"] == "C2":
            draw_lines(shape(f["geometry"]).buffer(0).boundary, fill=(60, 220, 90, 255), width=4)

    # 2 m contours, labelled where each line enters the top or left edge of the frame
    placed = []
    for f in load("contours_200m.geojson")["features"]:
        g = shape(f["geometry"])
        draw_lines(g, fill=(255, 255, 120, 200), width=2)
        best = None
        for p in parts(g.intersection(view)):
            if p.geom_type != "LineString" or p.is_empty:
                continue
            for c in (p.coords[0], p.coords[-1]):
                u, v = px(*c)
                score = min(v, u)  # closest to top or left edge
                if best is None or score < best[0]:
                    best = (score, u, v)
        if best:
            _, u, v = best
            u, v = min(max(u, 20), SIZE - 90), min(max(v, 90), SIZE - 60)
            if any(math.hypot(u - pu, v - pv) < 45 for pu, pv in placed):
                continue
            placed.append((u, v))
            d.text((u + 4, v), f"{f['properties']['elevation']:g}", font=font(18),
                   fill=(255, 255, 150, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))

    # Lot boundary
    lot = shape(load("lot.geojson")["features"][0]["geometry"])
    pts = [px(*c) for c in lot.exterior.coords]
    d.line(pts, fill=(255, 40, 40, 255), width=5, joint="curve")

    def side_label(i, j, text):
        u, v = (pts[i][0] + pts[j][0]) / 2, (pts[i][1] + pts[j][1]) / 2
        d.text((u, v), text, font=font(22), fill="white", anchor="mm",
               stroke_width=4, stroke_fill="black")

    side_label(0, 1, "45.0 m")
    side_label(1, 2, "36.9 m")
    side_label(2, 3, "32.9 m")
    side_label(6, 7, "arc ≈23.0 m")

    # Indicative ground levels at the main corners from the 5 m DEM
    to_mga = pyproj.Transformer.from_crs(4326, 7855, always_xy=True)
    dem = np.load(DATA / "dem_crop.npy")
    for i in (0, 1, 2, 3):
        x, y = to_mga.transform(*lot.exterior.coords[i])
        h = dem[int(DEM_Y1 - y), int(x - DEM_X0)]
        u, v = pts[i]
        d.ellipse((u - 6, v - 6, u + 6, v + 6), fill="white", outline="black")
        d.text((u + 10, v + 8), f"≈{h:.1f}", font=font(20), fill=(170, 230, 255),
               stroke_width=3, stroke_fill="black")

    # Address point
    u, v = px(*load("address.geojson")["features"][0]["geometry"]["coordinates"])
    d.ellipse((u - 7, v - 7, u + 7, v + 7), fill=(0, 180, 255, 255), outline="white", width=2)

    # Title
    d.rectangle((0, 0, SIZE, 64), fill=(0, 0, 0, 170))
    d.text((16, 8), "36 Waugh Street, Griffith NSW 2680 — Lot 624 DP821594", font=font(26), fill="white")
    d.text((16, 38), "≈1,093 m² (computed, GDA2020 MGA55)  ·  Zone R1 General Residential  ·  Min. lot size 700 m²",
           font=font(18), fill=(230, 230, 230))

    # Legend
    legend = [
        ((255, 40, 40), "Lot boundary (NSW cadastre)"),
        ((255, 140, 0), "Bushfire prone land – vegetation buffer"),
        ((60, 220, 90), "C2 Environmental Conservation zone boundary"),
        ((255, 255, 120), "2 m contours, m AHD (NSW topographic)"),
        ((170, 230, 255), "Ground level ≈ m AHD (5 m DEM, indicative)"),
        ((0, 180, 255), "Address point"),
    ]
    y = SIZE - 24 * len(legend) - 60
    d.rectangle((10, y - 10, 560, SIZE - 50), fill=(0, 0, 0, 170))
    for col, txt in legend:
        d.rectangle((20, y + 4, 44, y + 18), fill=col)
        d.text((54, y), txt, font=font(17), fill="white")
        y += 24

    # 20 m scale bar and north arrow
    mpp = 156543.033928 / N * math.cos(math.radians(-34.2645))
    sb = 20 / mpp
    sx = SIZE - 40 - sb
    d.rectangle((sx - 10, SIZE - 110, SIZE - 20, SIZE - 56), fill=(0, 0, 0, 170))
    d.line((sx, SIZE - 70, sx + sb, SIZE - 70), fill="white", width=6)
    d.text((sx + sb / 2, SIZE - 95), "20 m", font=font(18), fill="white", anchor="mm")
    d.polygon([(1240, 90), (1225, 135), (1255, 135)], fill="white", outline="black")
    d.text((1240, 150), "N", font=font(22), fill="white", anchor="mm", stroke_width=3, stroke_fill="black")

    # Credit
    d.rectangle((0, SIZE - 40, SIZE, SIZE), fill=(0, 0, 0, 190))
    d.text((16, SIZE - 32),
           "Imagery & cadastre © State of New South Wales (Spatial Services), CC BY 4.0 · imagery Feb 2014",
           font=font(16), fill="white")

    out = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    out.save(ROOT / "site_aerial.png")


if __name__ == "__main__":
    main()
