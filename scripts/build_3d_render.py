"""Bake the photorealistic dusk render page (site_render.html).

Uses the designed ground and house parameters from house_massing.py, the aerial texture, and
eucalypt positions picked out of the aerial photo inside the C2 bushland behind the lot.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pyproj
from PIL import Image
from shapely.geometry import Point, shape
from shapely import transform as tf
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import house_massing as hm  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "data" / "3d"
E0, N1, NCELL = hm.E0, hm.N1, hm.NCELL
N0 = N1 - NCELL


def find_trees():
    """Dark canopy blobs (2 m blocks) in the aerial, inside the C2 zone."""
    to_mga = pyproj.Transformer.from_crs(4326, 7855, always_xy=True)
    c2 = [tf(shape(f["geometry"]).buffer(0), lambda c: np.column_stack(to_mga.transform(c[:, 0], c[:, 1])))
          for f in json.loads((DATA / "zoning.geojson").read_text())["features"] if f["properties"]["SYM_CODE"] == "C2"]
    zone = prep(c2[0])
    tex = np.array(Image.open(OUT / "ortho_texture.jpg").convert("RGB")).astype(float)
    lum = tex.mean(2)
    b = 8  # 8 px x 0.25 m = 2 m blocks
    nb = lum.shape[0] // b
    blk = lum[:nb * b, :nb * b].reshape(nb, b, nb, b).mean((1, 3))
    dark = np.zeros_like(blk, bool)
    for i in range(nb):
        for j in range(nb):
            if blk[i, j] < 112 and zone.contains(Point(E0 + (j + 0.5) * 2, N1 - (i + 0.5) * 2)):
                dark[i, j] = True
    # Connected components -> one tree per ~3 blocks of canopy, spread through the blob
    seen = np.zeros_like(dark)
    rng = np.random.default_rng(3)
    trees = []
    for i in range(nb):
        for j in range(nb):
            if not dark[i, j] or seen[i, j]:
                continue
            stack, cells = [(i, j)], []
            seen[i, j] = True
            while stack:
                a, c = stack.pop()
                cells.append((a, c))
                for da, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, c + dc
                    if 0 <= x < nb and 0 <= y < nb and dark[x, y] and not seen[x, y]:
                        seen[x, y] = True
                        stack.append((x, y))
            n = max(1, round(len(cells) / 3))
            pick = rng.choice(len(cells), size=min(n, len(cells)), replace=False)
            r = float(np.clip(np.sqrt(len(cells) * 4 / np.pi / n) + 0.8, 1.6, 4.5))
            for k in pick:
                a, c = cells[k]
                gx = (c + 0.5) * 2 + rng.uniform(-0.8, 0.8)
                gy = NCELL - (a + 0.5) * 2 + rng.uniform(-0.8, 0.8)
                trees.append([round(gx, 2), round(gy, 2), round(r, 2)])
    return trees


def main():
    hmj = json.loads((OUT / "house_massing.json").read_text())
    ground = np.array(hmj["ground_cm"], dtype=float).reshape(NCELL, NCELL) / 100  # row 0 = south
    base = float(np.floor(ground.min()))
    trees = find_trees()

    # Street boundary arc in the street frame: centre and radius from three arc vertices
    lot = np.array(hm.LOT.exterior.coords)
    pts = [((v - hm.A) @ hm.U, (v - hm.A) @ hm.NV) for v in (lot[3], lot[7], lot[0])]
    (x1, y1), (x2, y2), (x3, y3) = pts
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    cu = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
    cn = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
    radius = float(np.hypot(x1 - cu, y1 - cn))

    data = {
        "size": NCELL, "base": base,
        "heights": base64.b64encode(np.round((ground - base) * 1000).astype("<u2").tobytes()).decode(),
        "A": [float(hm.A[0] - E0), float(hm.A[1] - N0)], "U": [float(hm.U[0]), float(hm.U[1])],
        "FFL": hm.FFL, "PAD": hm.PAD, "GARAGE_FL": hm.GARAGE_FL,
        "UPPER": hm.UPPER, "EAVE": hm.EAVE, "RIDGE": hm.RIDGE, "LINK_TOP": hm.LINK_TOP,
        "arc": {"cu": round(cu, 3), "cn": round(cn, 3), "r": round(radius, 3)},
        "trees": trees,
    }
    # Plain-JSON copy of the scene for the Cycles renderer (scripts/cycles_render.py)
    scene = {k: v for k, v in data.items() if k != "heights"}
    scene.update({"E0": E0, "N0": N0, "heights_m": np.round(ground, 3).ravel().tolist(),
                  "dem_1km": {"file": "dem_1km_4m.tif", "e0": 414613.0, "n1": 6208640.0, "cell": 4.0, "size": 250}})
    (OUT / "render_scene.json").write_text(json.dumps(scene, separators=(",", ":")))

    tex = base64.b64encode((OUT / "ortho_texture.jpg").read_bytes()).decode()
    html = (ROOT / "scripts" / "render_template.html").read_text()
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__TEXTURE__", tex)
    (ROOT / "site_render.html").write_text(html)
    print(f"trees {len(trees)}, arc r={radius:.1f} centre=({cu:.2f},{cn:.2f}); site_render.html {len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
