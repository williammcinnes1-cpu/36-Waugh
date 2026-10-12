"""Build 3D site products for Lot 624 DP821594 from NSW 5 m DEM + NSW_Imagery.

Outputs (data/3d/): ortho_texture.jpg (imagery resampled to the DEM's MGA55 grid),
site_pointcloud.las / .ply (coloured points, MGA55 + m AHD), site_terrain.glb (textured mesh),
viewer_payload.json (compact data for the web viewer).
"""
import io
import json
import math
import urllib.request
from pathlib import Path

import laspy
import numpy as np
import pyproj
import trimesh
from PIL import Image
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "data" / "3d"
TILE_URL = "https://maps.six.nsw.gov.au/arcgis/rest/services/public/NSW_Imagery/MapServer/tile/{z}/{y}/{x}"
Z = 20
# DEM grid (from data/3d/dem_240m.tif): 1 m cells, top-left corner, 240 x 240
E0, N1, NCELL = 414993.0, 6208260.0, 240
TEX_RES = 0.25  # m per texture pixel

to_ll = pyproj.Transformer.from_crs(7855, 4326, always_xy=True)
to_mga = pyproj.Transformer.from_crs(4326, 7855, always_xy=True)


def tile_xy(lon, lat):
    n = 2**Z
    x = (lon + 180) / 360 * n
    y = (1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2 * n
    return x, y


def build_texture():
    npx = int(NCELL / TEX_RES)
    e = E0 + (np.arange(npx) + 0.5) * TEX_RES
    n = N1 - (np.arange(npx) + 0.5) * TEX_RES
    ee, nn = np.meshgrid(e, n)
    lon, lat = to_ll.transform(ee, nn)
    tx, ty = tile_xy(lon, lat)
    x0, y0 = int(tx.min()), int(ty.min())
    x1, y1 = int(tx.max()), int(ty.max())
    mosaic = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256, 3), np.uint8)
    cache = DATA / "tiles"
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            fn = cache / f"{Z}_{y}_{x}.jpg"
            if not fn.exists():
                fn.write_bytes(urllib.request.urlopen(TILE_URL.format(z=Z, y=y, x=x), timeout=60).read())
            mosaic[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = \
                np.array(Image.open(fn).convert("RGB"))
    u = ((tx - x0) * 256).astype(int).clip(0, mosaic.shape[1] - 1)
    v = ((ty - y0) * 256).astype(int).clip(0, mosaic.shape[0] - 1)
    tex = mosaic[v, u]
    Image.fromarray(tex).save(OUT / "ortho_texture.jpg", quality=88)
    return tex


def main():
    dem = np.array(Image.open(OUT / "dem_240m.tif"), dtype=np.float64)
    assert dem.shape == (NCELL, NCELL) and (dem > 0).all()
    tex = build_texture()

    # Cell-centre coordinates
    e = E0 + np.arange(NCELL) + 0.5
    n = N1 - np.arange(NCELL) - 0.5
    ee, nn = np.meshgrid(e, n)

    # Point cloud at 0.5 m: bilinear-upsample the 1 m grid, colour from the ortho
    step = 0.5
    k = int(1 / step)
    fine = np.array(Image.fromarray(dem.astype(np.float32)).resize((NCELL * k, NCELL * k), Image.BILINEAR), dtype=np.float64)
    fe = E0 + (np.arange(NCELL * k) + 0.5) * step
    fn_ = N1 - (np.arange(NCELL * k) + 0.5) * step
    fee, fnn = np.meshgrid(fe, fn_)
    ti = ((N1 - fnn) / TEX_RES).astype(int).clip(0, tex.shape[0] - 1)
    tj = ((fee - E0) / TEX_RES).astype(int).clip(0, tex.shape[1] - 1)
    rgb = tex[ti, tj].reshape(-1, 3)
    xyz = np.column_stack([fee.ravel(), fnn.ravel(), fine.ravel()])

    hdr = laspy.LasHeader(point_format=2, version="1.2")
    hdr.offsets = [E0, N1 - NCELL, 0.0]
    hdr.scales = [0.001, 0.001, 0.001]
    hdr.add_crs(pyproj.CRS.from_epsg(7855))
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.red, las.green, las.blue = (rgb[:, 0].astype(np.uint16) * 257, rgb[:, 1].astype(np.uint16) * 257,
                                    rgb[:, 2].astype(np.uint16) * 257)
    las.classification = np.full(len(xyz), 2, np.uint8)  # ground (derived surface, not LiDAR returns)
    las.write(OUT / "site_pointcloud.las")
    trimesh.PointCloud(xyz - [E0, N1 - NCELL, 0], colors=np.column_stack([rgb, np.full(len(rgb), 255)])) \
        .export(OUT / "site_pointcloud_local.ply")

    # Textured terrain mesh (local coords: x east, y up = AHD, z south) for GLB
    ny, nx = dem.shape
    verts = np.column_stack([(ee - E0).ravel(), dem.ravel(), (N1 - nn).ravel()])
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.vstack([np.column_stack([a, c, b]), np.column_stack([b, c, d])])
    uv = np.column_stack([(ee - E0).ravel() / NCELL, 1 - (N1 - nn).ravel() / NCELL])
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.open(OUT / "ortho_texture.jpg"),
                                              metallicFactor=0.0, roughnessFactor=1.0)
    mesh = trimesh.Trimesh(verts, faces, visual=trimesh.visual.TextureVisuals(uv=uv, material=mat), process=False)
    mesh.export(OUT / "site_terrain.glb")

    # Overlays in local metres (x east from E0, y north from N1-240)
    def local(geom):
        out = []
        for p in getattr(geom, "geoms", [geom]):
            if p.is_empty:
                continue
            ring = p.exterior.coords if p.geom_type == "Polygon" else p.coords
            out.append([[round(x - E0, 2), round(y - (N1 - NCELL), 2)] for x, y in
                        (to_mga.transform(lon, lat) for lon, lat in ring)])
        return out

    from shapely.geometry import box
    clip = box(*to_ll.transform(E0, N1 - NCELL), *to_ll.transform(E0 + NCELL, N1))
    lot = shape(json.loads((DATA / "lot.geojson").read_text())["features"][0]["geometry"])
    # Zone outlines are clipped as lines, so the clip square's edges are not drawn as boundaries
    bush = shape(json.loads((DATA / "bushfire.geojson").read_text())["features"][0]["geometry"]).buffer(0).boundary.intersection(clip)
    c2 = [shape(f["geometry"]).buffer(0).boundary.intersection(clip) for f in
          json.loads((DATA / "zoning.geojson").read_text())["features"] if f["properties"]["SYM_CODE"] == "C2"]
    payload = {
        "grid": {"e0": E0, "n0": N1 - NCELL, "size": NCELL, "cell": 1.0},
        "heights_cm": np.round(dem[::-1] * 100).astype(int).ravel().tolist(),  # row 0 = south
        "lot": local(lot)[0],
        "bushfire": local(bush),
        "c2": [r for g in c2 for r in local(g)],
    }
    (OUT / "viewer_payload.json").write_text(json.dumps(payload, separators=(",", ":")))
    print("points", len(xyz), "verts", len(verts), "faces", len(faces))


if __name__ == "__main__":
    main()
