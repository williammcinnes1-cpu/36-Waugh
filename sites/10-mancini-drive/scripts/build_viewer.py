"""Build the 3D site viewer for 10 Mancini Drive: terrain from the NSW 5 m DEM draped with the 2014 aerial, the lot
boundary, 0.25 m contours, and the house + pool meshes from House.IGS placed with data/model_to_mga.json.

    python3 scripts/build_viewer.py [path/to/tris.npy]

tris.npy is the gmsh mesh of House.IGS (n x 3 x 3, metres, SolidWorks Y-up). Writes viewer/aerial.jpg and
viewer/site.json; viewer/index.html is the page (kept in the repo, not generated).
"""
import json, math, sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Aerial, lot_mga, DATA, SITE, TO_MGA, TO_LL

OUT = SITE / "viewer"
H0 = 116.0          # height datum for the scene (m AHD)
HOUSE_BASE = 116.55  # model ground level y = 0 sits at the DEM level under the house


def main(tris_path):
    OUT.mkdir(exist_ok=True)
    A = Aerial()
    lot = lot_mga()
    E0, N0 = round(lot.centroid.x), round(lot.centroid.y)
    dem = rasterio.open(DATA / "dem_1m.tif")
    Z = dem.read(1)
    tr = dem.transform
    e_min, n_max = tr.c, tr.f
    size = Z.shape[0]                                         # 260 m square, 1 m cells

    # terrain grid, 2 m spacing (cell centres)
    step = 2
    zz = Z[::step, ::step]
    rows, cols = zz.shape
    xs = [round(e_min + (c * step + 0.5) - E0, 2) for c in range(cols)]
    ns = [round(n_max - (r * step + 0.5) - N0, 2) for r in range(rows)]
    heights = [round(float(v) - H0, 3) for v in zz.ravel()]

    # aerial texture over the DEM extent, north up, 0.2 m/px (bilinear sample of the stitched z20 mosaic)
    W = 1300
    g = (np.arange(W) + 0.5) / W * size
    EE, NN = np.meshgrid(e_min + g, n_max - g)
    lon, lat = TO_LL.transform(EE.ravel(), NN.ravel())
    Nz = 2 ** A.z
    u = ((np.array(lon) + 180) / 360 * Nz - A.X0) * 256
    v = ((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * Nz - A.Y0) * 256
    src = np.asarray(A.img).astype(np.float32)
    u0 = np.clip(np.floor(u).astype(int), 0, src.shape[1] - 2)
    v0 = np.clip(np.floor(v).astype(int), 0, src.shape[0] - 2)
    fu, fv = (u - u0)[:, None], (v - v0)[:, None]
    px = (src[v0, u0] * (1 - fu) * (1 - fv) + src[v0, u0 + 1] * fu * (1 - fv)
          + src[v0 + 1, u0] * (1 - fu) * fv + src[v0 + 1, u0 + 1] * fu * fv)
    Image.fromarray(px.reshape(W, W, 3).clip(0, 255).astype(np.uint8)).save(OUT / "aerial.jpg", quality=82)

    def zat(e, n):
        c = min(max((e - e_min), 0), size - 1.001)
        r = min(max((n_max - n), 0), size - 1.001)
        i, j = int(r), int(c)
        fy, fx = r - i, c - j
        return float(Z[i, j] * (1 - fx) * (1 - fy) + Z[i, j + 1] * fx * (1 - fy) + Z[i + 1, j] * (1 - fx) * fy + Z[i + 1, j + 1] * fx * fy)

    # lot boundary, densified so it drapes
    bnd = []
    cs = list(lot.exterior.coords)
    for (a, b), (c, d) in zip(cs[:-1], cs[1:]):
        k = max(1, int(math.hypot(c - a, d - b) / 2))
        for t in range(k):
            e, n = a + (c - a) * t / k, b + (d - b) * t / k
            bnd.append([round(e - E0, 2), round(zat(e, n) - H0, 3), round(n - N0, 2)])
    bnd.append(bnd[0])
    # line 10 m inside the boundary (mitred corners)
    inner_poly = lot.buffer(-10, join_style="mitre")
    inner = []
    ci = list(inner_poly.exterior.coords)
    for (a, b), (c, d) in zip(ci[:-1], ci[1:]):
        k = max(1, int(math.hypot(c - a, d - b) / 2))
        for t in range(k):
            e, n = a + (c - a) * t / k, b + (d - b) * t / k
            inner.append([round(e - E0, 2), round(zat(e, n) - H0, 3), round(n - N0, 2)])
    inner.append(inner[0])
    edges = []
    for (a, b), (c, d) in zip(cs[:-1], cs[1:]):
        L = math.hypot(c - a, d - b)
        if L > 15:
            e, n = (a + c) / 2, (b + d) / 2
            edges.append({"label": f"{L:.1f} m", "p": [round(e - E0, 2), round(zat(e, n) - H0, 3), round(n - N0, 2)]})

    # contours (0.25 m) inside 30 m of the lot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import LineString
    rr, cc = np.mgrid[0:size, 0:size]
    E = e_min + cc + 0.5
    N = n_max - rr - 0.5
    lv = np.arange(math.floor(Z.min() * 4) / 4, Z.max() + 0.25, 0.25)
    cset = plt.contour(E, N, Z, levels=lv)
    clip = lot.buffer(30)
    contours = []
    for lev, segs in zip(cset.levels, cset.allsegs):
        for seg in segs:
            if len(seg) < 2:
                continue
            ln = LineString(seg).intersection(clip)
            for part in getattr(ln, "geoms", [ln]):
                if part.is_empty or part.geom_type != "LineString":
                    continue
                pts = list(part.simplify(0.3).coords)
                contours.append({"z": round(float(lev), 2), "major": bool(abs(lev - round(lev)) < 1e-6),
                                 "pts": [[round(e - E0, 2), round(float(lev) - H0, 3), round(n - N0, 2)] for e, n in pts]})
    plt.close("all")

    # house + pool meshes from the IGES (drop the model's own flat lot surface)
    T = np.load(tris_path)
    tf = json.load(open(DATA / "model_to_mga.json"))
    th = math.radians(tf["rotation_deg"])
    keep = T[:, :, 1].max(1) > 0.2                           # house only; the IGES pool is replaced by the as-built pool
    T = T[keep]
    px_, py_ = T[..., 0], -T[..., 2]                          # model plan frame (x, -z)
    e = px_ * math.cos(th) - py_ * math.sin(th) + tf["tx"]
    n = px_ * math.sin(th) + py_ * math.cos(th) + tf["ty"]
    y = T[..., 1] + HOUSE_BASE
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    pool = T[:, :, 1].min(1) < -0.2
    roof = (~pool) & (np.abs(nrm[:, 1]) > 0.35) & (T[:, :, 1].min(1) > 2.0)
    kind = np.where(pool, 2, np.where(roof, 1, 0))            # 0 wall, 1 roof, 2 pool shell
    pos = np.stack([e - E0, y - H0, n - N0], axis=-1).reshape(-1, 3)
    # pool water surface from the pool outline
    M = json.load(open(DATA / "model_plan.json"))
    from shapely.geometry import shape
    from shapely import affinity
    from shapely.ops import triangulate
    put = lambda k: affinity.translate(affinity.rotate(shape(M[k]), tf["rotation_deg"], origin=(0, 0)), tf["tx"], tf["ty"])
    pg, sg, phg, wg_ = put("pool"), put("pavers"), put("poolhouse"), put("wall")     # all from House.IGS
    pool_ground = HOUSE_BASE                                    # model ground level (y = 0)
    ring = lambda g: [[round(x - E0, 2), round(yy - N0, 2)] for x, yy in list(g.exterior.coords)[:-1]]
    wpos = []

    site = {
        "origin": {"E": E0, "N": N0, "H": H0, "crs": "GDA2020 / MGA zone 55 (EPSG:7855)"},
        "terrain": {"xs": xs, "ns": ns, "h": heights, "extent": [e_min - E0, e_min + size - E0, n_max - size - N0, n_max - N0]},
        "boundary": bnd, "inner10": inner, "edges": edges, "contours": contours,
        "model": {"pos": [round(float(v), 3) for v in pos.ravel()], "kind": [int(k) for k in kind], "water": wpos, "pool": {"water": ring(pg), "surround": ring(sg), "ground": round(pool_ground - H0, 3),
                           "poolhouse": ring(phg), "wall": ring(wg_)},
                  "house_base": HOUSE_BASE - H0},
        "facts": json.load(open(DATA / "levels.json")),
        "house_centre": [round(float(e.mean()) - E0, 2), round(float(n.mean()) - N0, 2)],
    }
    (OUT / "site.json").write_text(json.dumps(site, separators=(",", ":")))
    print("site.json", round((OUT / "site.json").stat().st_size / 1e6, 2), "MB;", len(T), "model tris;", len(contours), "contours;",
          "aerial.jpg", round((OUT / "aerial.jpg").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/-home-user-36-Waugh/5d310e2a-c88e-5d61-a0d7-d8065c69872b/scratchpad/igs/tris.npy")
