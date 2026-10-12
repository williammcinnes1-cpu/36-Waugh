"""Check a design's exported mesh (data/3d/farmhouse_mesh.json or a path given) against Griffith DCP
height (4.5a: ridge ≤ 9 m above natural ground) and report roof heights, plus key setbacks."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import house_massing as hm  # noqa: E402

DEM = np.array(Image.open(ROOT / "data" / "3d" / "dem_240m.tif"), float)  # natural ground, row 0 = north
FFL = hm.FFL


def natural(u, n):
    e, no = hm.A + hm.U * u + hm.NV * n
    c, r = e - hm.E0 - 0.5, hm.N1 - no - 0.5
    i, j = int(c), int(r)
    fx, fy = c - i, r - j
    return DEM[j, i] * (1 - fx) * (1 - fy) + DEM[j, i + 1] * fx * (1 - fy) + DEM[j + 1, i] * (1 - fx) * fy + DEM[j + 1, i + 1] * fx * fy


def main(path=ROOT / "data" / "3d" / "farmhouse_mesh.json"):
    m = json.loads(Path(path).read_text())
    v = np.array(m["positions"], float).reshape(-1, 3)
    roof_pts = v[v[:, 2] > 2.5]
    h = np.array([z + FFL - natural(u, n) for u, n, z in roof_pts])
    worst = int(np.argmax(h))
    u, n, z = roof_pts[worst]
    print(f"highest point above natural ground: {h[worst]:.2f} m at u={u:.1f}, n={n:.1f} (RL {z + FFL:.2f}, ground {natural(u, n):.2f})")
    over = roof_pts[h > 9.0]
    print("points over 9.0 m:", len(over), "" if not len(over) else f"(e.g. u={over[0][0]:.1f}, n={over[0][1]:.1f}, z={over[0][2]:.2f})")
    return h[worst]


if __name__ == "__main__":
    main(*sys.argv[1:])
