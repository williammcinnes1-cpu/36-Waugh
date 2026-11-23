"""Fetch NSW Spatial Services data for 10 Mancini Drive, Lake Wyangan (Lot 5 DP285693).

Writes into ../data: tiles/ (NSW_Imagery z20), dem_1m.tif (NSW 5 m DEM resampled to 1 m, MGA55 / EPSG:7855),
contours.geojson (2 m contours), imagery_date.json. Address and lot were fetched separately (address.geojson, lot.geojson).
Data © State of New South Wales (Spatial Services), CC BY 4.0.
"""
import json, math, subprocess, sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
LON, LAT = 146.0375482, -34.2277779
SIX = "https://maps.six.nsw.gov.au/arcgis/rest/services/public"
PORTAL = "https://portal.spatial.nsw.gov.au/server/rest/services"
Z, R = 20, 4                    # z20 tiles, (2R+1)^2 block around the lot


def curl(url, out, params=None, timeout=150):
    cmd = ["curl", "-sS", "-f", "-m", str(timeout), "-o", str(out)]
    if params:
        cmd += ["-G"] + [a for k, v in params.items() for a in ("--data-urlencode", f"{k}={v}")]
    for attempt in range(3):
        if subprocess.run(cmd + [url]).returncode == 0:
            return True
    print("FAILED", url, file=sys.stderr)
    return False


def tile_xy(lon, lat, z):
    n = 2 ** z
    return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def main():
    cx, cy = tile_xy(LON, LAT, Z)
    print("tile centre", cx, cy)
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            p = DATA / "tiles" / f"{Z}_{cy + dy}_{cx + dx}.jpg"
            if not p.exists():
                curl(f"{SIX}/NSW_Imagery/MapServer/tile/{Z}/{cy + dy}/{cx + dx}", p)
    json.dump({"z": Z, "cx": cx, "cy": cy, "r": R}, open(DATA / "tiles" / "index.json", "w"))
    # 5 m DEM on a 1 m MGA55 grid, 260 m square around the lot
    import pyproj
    e, n = pyproj.Transformer.from_crs(4326, 7855, always_xy=True).transform(LON, LAT)
    e0, n0 = round(e) - 130, round(n) - 130
    curl(f"{SIX}/NSW_5M_Elevation/ImageServer/exportImage", DATA / "dem_1m.tif",
         {"bbox": f"{e0},{n0},{e0 + 260},{n0 + 260}", "bboxSR": 7855, "imageSR": 7855, "size": "260,260",
          "format": "tiff", "pixelType": "F32", "interpolation": "RSP_BilinearInterpolation", "f": "image"})
    curl(f"{PORTAL}/NSW_Elevation_and_Depth_Theme/MapServer/2/query", DATA / "contours.geojson",
         {"geometry": f"{e0},{n0},{e0 + 260},{n0 + 260}", "geometryType": "esriGeometryEnvelope", "inSR": 7855,
          "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "outSR": 4326, "f": "geojson"})
    curl(f"{SIX}/NSW_Imagery_Dates/MapServer/0/query", DATA / "imagery_date.json",
         {"geometry": f"{LON},{LAT}", "geometryType": "esriGeometryPoint", "inSR": 4326,
          "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "returnGeometry": "false", "f": "json"})


if __name__ == "__main__":
    main()
