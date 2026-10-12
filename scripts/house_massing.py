"""Indicative massing of the proposed two-gable house on Lot 624 DP821594.

All dimensions are assumptions scaled from the concept image and fitted to the lot; they are not
a design. The house is laid out in a street-aligned frame: u runs along the chord of the street
frontage (SW end -> NE end), n runs into the lot, z is m AHD.

Outputs (data/3d/):
  house_massing.json        triangles + colours (grid-local metres) and the designed ground heights
  site_with_house.las       point cloud: designed ground + house surfaces (MGA55, m AHD)
  house_massing.glb         house mesh in the same local frame as site_terrain.glb
"""
import json
from pathlib import Path

import laspy
import numpy as np
import pyproj
import trimesh
from PIL import Image
from shapely.geometry import Point, Polygon, box, shape
from shapely import transform as tf

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "data" / "3d"
E0, N1, NCELL = 414993.0, 6208260.0, 240
N0 = N1 - NCELL

# ---- Design assumptions (m AHD / m) -------------------------------------------------------------
FFL = 157.6                 # ground floor level: roughly balances cut at the rear and fill at the front
PAD = FFL - 0.15            # terrace / pool deck
GARAGE_FL = 155.0           # garage floor near street level at the low (NE) corner
UPPER = 3.3                 # floor-to-floor
EAVE = 6.4                  # pavilion wall height above FFL
RIDGE = 10.0                # pavilion ridge above FFL
LINK_TOP = 6.6              # flat roof of the central link above FFL

STONE = (226, 213, 186)
ROOF = (52, 55, 60)
GLASS = (78, 96, 108)
DOOR = (62, 50, 42)
POOL = (64, 172, 212)
PAVING = (206, 200, 188)
DRIVE = (170, 168, 162)

to_mga = pyproj.Transformer.from_crs(4326, 7855, always_xy=True)


def frame():
    lot = shape(json.loads((DATA / "lot.geojson").read_text())["features"][0]["geometry"])
    lot = tf(lot, lambda c: np.column_stack(to_mga.transform(c[:, 0], c[:, 1])))
    c = list(lot.exterior.coords)
    a, b = np.array(c[3]), np.array(c[0])  # ends of the street arc
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]])
    if np.dot(np.array(lot.centroid.coords[0]) - a, n) < 0:
        n = -n
    return lot, a, u, n


LOT, A, U, NV = frame()


def to_grid(uu, nn, z):
    """street frame -> grid-local metres (x east from E0, y north from N0, z AHD)"""
    e = A[0] + uu * U[0] + nn * NV[0]
    nrt = A[1] + uu * U[1] + nn * NV[1]
    return np.column_stack([e - E0, nrt - N0, np.broadcast_to(z, np.shape(e))])


class Mesh:
    def __init__(self):
        self.v, self.f, self.c = [], [], []

    def quad(self, p, col):
        """p: 4 points (u, n, z) in order"""
        i = len(self.v)
        self.v += p
        self.f += [(i, i + 1, i + 2), (i, i + 2, i + 3)]
        self.c += [col, col]

    def tri(self, p, col):
        i = len(self.v)
        self.v += p
        self.f += [(i, i + 1, i + 2)]
        self.c += [col]

    def box(self, u0, u1, n0, n1, z0, z1, col, front=None, top=None):
        """front: colour of the street-facing (n0) face; top: colour of the top (None = col)"""
        self.quad([(u0, n0, z0), (u1, n0, z0), (u1, n0, z1), (u0, n0, z1)], front or col)
        self.quad([(u1, n1, z0), (u0, n1, z0), (u0, n1, z1), (u1, n1, z1)], col)
        self.quad([(u0, n1, z0), (u0, n0, z0), (u0, n0, z1), (u0, n1, z1)], col)
        self.quad([(u1, n0, z0), (u1, n1, z0), (u1, n1, z1), (u1, n0, z1)], col)
        self.quad([(u0, n0, z1), (u1, n0, z1), (u1, n1, z1), (u0, n1, z1)], top or col)

    def pavilion(self, u0, u1, n0, n1):
        """two-storey gabled wing, gable end (mostly glass) facing the street"""
        zf, ze, zr = FFL, FFL + EAVE, FFL + RIDGE
        um = (u0 + u1) / 2
        # stone walls with a glazed street front, set in from the corners by 0.6 m
        self.box(u0, u1, n0, n1, zf, ze, STONE, front=STONE)
        self.quad([(u0 + 0.6, n0 - 0.02, zf), (u1 - 0.6, n0 - 0.02, zf), (u1 - 0.6, n0 - 0.02, ze), (u0 + 0.6, n0 - 0.02, ze)], GLASS)
        # gable triangles (front glazed, rear stone)
        self.tri([(u0, n0, ze), (u1, n0, ze), (um, n0, zr)], STONE)
        self.tri([(u0 + 0.9, n0 - 0.02, ze), (u1 - 0.9, n0 - 0.02, ze), (um, n0 - 0.02, zr - 0.9)], GLASS)
        self.tri([(u1, n1, ze), (u0, n1, ze), (um, n1, zr)], STONE)
        # roof planes with a 0.4 m overhang
        o = 0.4
        self.quad([(u0 - o, n0 - o, ze - 0.2), (u0 - o, n1 + o, ze - 0.2), (um, n1 + o, zr + 0.05), (um, n0 - o, zr + 0.05)], ROOF)
        self.quad([(um, n0 - o, zr + 0.05), (um, n1 + o, zr + 0.05), (u1 + o, n1 + o, ze - 0.2), (u1 + o, n0 - o, ze - 0.2)], ROOF)


def build_house():
    m = Mesh()
    # Wings and link (u along the street, n back from it)
    m.pavilion(0.8, 7.8, 10.0, 24.0)
    m.pavilion(15.6, 22.6, 10.0, 24.0)
    m.box(7.8, 15.6, 11.0, 23.0, FFL, FFL + LINK_TOP, STONE, front=GLASS, top=ROOF)
    m.box(7.8, 15.6, 10.2, 11.0, FFL + UPPER - 0.3, FFL + UPPER, STONE)  # first-floor slab / balcony edge
    # Chimney on the NE wing
    m.box(22.6, 23.6, 18.0, 20.0, FFL, FFL + RIDGE + 1.6, STONE)
    # Garage at street level in front of the NE wing; its roof is part of the terrace
    m.box(16.2, 22.6, 2.2, 9.6, GARAGE_FL, PAD, STONE, top=PAVING)
    m.quad([(16.9, 2.18, GARAGE_FL), (21.9, 2.18, GARAGE_FL), (21.9, 2.18, GARAGE_FL + 2.4), (16.9, 2.18, GARAGE_FL + 2.4)], DOOR)
    # Front retaining / garden wall, terrace, pool, steps
    m.box(0.3, 16.2, 1.6, 2.2, 154.5, PAD + 1.0, STONE)
    m.quad([(0.3, 2.2, PAD), (16.2, 2.2, PAD), (16.2, 10.0, PAD), (0.3, 10.0, PAD)], PAVING)
    m.quad([(3.0, 4.0, PAD + 0.01), (13.0, 4.0, PAD + 0.01), (13.0, 7.6, PAD + 0.01), (3.0, 7.6, PAD + 0.01)], POOL)
    # Rear retaining wall where the pad cuts into the slope
    m.box(0.3, 22.9, 27.0, 27.4, PAD, 160.0, STONE)
    return m


def designed_ground(dem):
    """Existing DEM (row 0 = north) with the building pad, garage and driveway cut/filled in."""
    g = dem.copy()
    for r in range(NCELL):
        for c in range(NCELL):
            e, nn = E0 + c + 0.5, N1 - r - 0.5
            d = np.array([e, nn]) - A
            uu, vv = float(d @ U), float(d @ NV)
            if not LOT.buffer(0.5).contains(Point(e, nn)):
                continue
            if 16.2 <= uu <= 22.9 and -1.0 <= vv < 9.6:          # garage + driveway
                g[r, c] = GARAGE_FL if vv >= 2.2 else np.interp(vv, [-1.0, 2.2], [g[r, c], GARAGE_FL])
            elif 0.3 <= uu <= 22.9 and 2.2 <= vv <= 27.0:          # house + terrace pad
                g[r, c] = PAD
            elif 0.3 <= uu <= 16.2 and 0.0 <= vv < 1.6:            # front garden bed below the wall
                g[r, c] = min(g[r, c], 155.6)
    return g


def sample(m, spacing=0.25, seed=7):
    rng = np.random.default_rng(seed)
    v = np.array(m.v, float)
    pts, cols = [], []
    for (i, j, k), col in zip(m.f, m.c):
        p0, p1, p2 = v[i], v[j], v[k]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        cnt = max(1, int(area / spacing**2))
        r1, r2 = rng.random(cnt), rng.random(cnt)
        flip = r1 + r2 > 1
        r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
        pts.append(p0 + np.outer(r1, p1 - p0) + np.outer(r2, p2 - p0))
        cols.append(np.tile(col, (cnt, 1)))
    return np.vstack(pts), np.vstack(cols)


def main():
    dem = np.array(Image.open(OUT / "dem_240m.tif"), dtype=np.float64)
    ground = designed_ground(dem)
    house = build_house()

    # Check the footprint sits inside the lot
    fp = Polygon([(0.8, 10), (22.6, 10), (22.6, 24), (0.8, 24)])
    corners = to_grid(np.array(fp.exterior.coords)[:, 0], np.array(fp.exterior.coords)[:, 1], 0)[:, :2] + [E0, N0]
    assert LOT.contains(Polygon(corners)), "house footprint leaves the lot"

    # Point cloud: designed ground (0.5 m) + house surfaces (0.25 m)
    tex = np.array(Image.open(OUT / "ortho_texture.jpg"))
    k = 2
    fine = np.array(Image.fromarray(ground.astype(np.float32)).resize((NCELL * k, NCELL * k), Image.BILINEAR), dtype=float)
    fe = E0 + (np.arange(NCELL * k) + 0.5) / k
    fn = N1 - (np.arange(NCELL * k) + 0.5) / k
    fee, fnn = np.meshgrid(fe, fn)
    ti = ((N1 - fnn) / 0.25).astype(int).clip(0, tex.shape[0] - 1)
    tj = ((fee - E0) / 0.25).astype(int).clip(0, tex.shape[1] - 1)
    g_rgb = tex[ti, tj].reshape(-1, 3)
    g_xyz = np.column_stack([fee.ravel(), fnn.ravel(), fine.ravel()])

    hp, hc = sample(house)
    h_xyz = to_grid(hp[:, 0], hp[:, 1], hp[:, 2]) + [E0, N0, 0]
    is_pool = np.all(hc == POOL, axis=1)
    cls = np.concatenate([np.full(len(g_xyz), 2), np.where(is_pool, 9, 6)]).astype(np.uint8)

    xyz = np.vstack([g_xyz, h_xyz])
    rgb = np.vstack([g_rgb, hc]).astype(np.uint16) * 257
    hdr = laspy.LasHeader(point_format=2, version="1.2")
    hdr.offsets = [E0, N0, 0.0]
    hdr.scales = [0.001, 0.001, 0.001]
    hdr.add_crs(pyproj.CRS.from_epsg(7855))
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = xyz.T
    las.red, las.green, las.blue = rgb.T
    las.classification = cls
    las.write(OUT / "site_with_house.las")

    # House mesh for the viewer and GLB (grid-local metres)
    hv = to_grid(*np.array(house.v, float).T)
    (OUT / "house_massing.json").write_text(json.dumps({
        "positions": np.round(hv, 3).ravel().tolist(),
        "faces": np.array(house.f).ravel().tolist(),
        "colors": np.array(house.c).ravel().tolist(),
        "ground_cm": np.round(ground[::-1] * 100).astype(int).ravel().tolist(),  # row 0 = south
        "assumptions": {"FFL": FFL, "terrace": PAD, "garage_floor": GARAGE_FL, "ridge": FFL + RIDGE,
                        "eave": FFL + EAVE, "width_m": 21.8, "depth_m": 14.0},
    }, separators=(",", ":")))
    tm = trimesh.Trimesh(np.column_stack([hv[:, 0], hv[:, 2], NCELL - hv[:, 1]]), np.array(house.f)[:, ::-1], process=False)
    vcol = np.zeros((len(hv), 4), np.uint8)
    for f, col in zip(house.f, house.c):  # vertices are never shared between faces of different colours
        vcol[list(f)] = (*col, 255)
    tm.visual.vertex_colors = vcol
    tm.export(OUT / "house_massing.glb")

    fl = LOT.area
    cover = Polygon(corners).area + 6.4 * 7.4
    print(f"points: ground {len(g_xyz)}, house {len(h_xyz)} (pool {is_pool.sum()})")
    print(f"footprint incl. garage ≈{cover:.0f} m² of {fl:.0f} m² lot ({cover / fl:.0%})")
    cut = ground - dem
    print(f"earthworks within 1 m cells: max fill {cut.max():.2f} m, max cut {-cut.min():.2f} m")


if __name__ == "__main__":
    main()
