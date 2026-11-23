"""Shared helpers for 10 Mancini Drive: stitched z20 aerial with web-mercator pixel <-> MGA55 transforms."""
import json, math
from pathlib import Path
import numpy as np
import pyproj
from PIL import Image

SITE = Path(__file__).resolve().parent.parent
DATA = SITE / "data"
TO_LL = pyproj.Transformer.from_crs(7855, 4326, always_xy=True)
TO_MGA = pyproj.Transformer.from_crs(4326, 7855, always_xy=True)


class Aerial:
    def __init__(self):
        ix = json.load(open(DATA / "tiles" / "index.json"))
        self.z, cx, cy, r = ix["z"], ix["cx"], ix["cy"], ix["r"]
        self.X0, self.Y0, n = cx - r, cy - r, 2 * r + 1
        self.img = Image.new("RGB", (256 * n, 256 * n))
        for dx in range(n):
            for dy in range(n):
                p = DATA / "tiles" / f"{self.z}_{self.Y0 + dy}_{self.X0 + dx}.jpg"
                if p.exists():
                    self.img.paste(Image.open(p), (256 * dx, 256 * dy))

    def ll2px(self, lon, lat):
        N = 2 ** self.z
        x = (lon + 180) / 360 * N
        y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * N
        return ((x - self.X0) * 256, (y - self.Y0) * 256)

    def px2ll(self, u, v):
        N = 2 ** self.z
        x, y = self.X0 + u / 256, self.Y0 + v / 256
        return x / N * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / N))))

    def mga2px(self, e, n):
        return self.ll2px(*TO_LL.transform(e, n))

    def px2mga(self, u, v):
        return TO_MGA.transform(*self.px2ll(u, v))


def lot_mga():
    from shapely.geometry import shape
    from shapely import transform
    g = shape(json.load(open(DATA / "lot.geojson"))["features"][0]["geometry"])
    return transform(g, lambda c: np.column_stack(TO_MGA.transform(c[:, 0], c[:, 1])))
