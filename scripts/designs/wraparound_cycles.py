"""Cycles render of the "Wraparound Farmhouse": whitewashed brick and white board-and-batten, a deep verandah that
turns the SW corner, dark standing-seam roofs with gables and three dormers, a tall fieldstone chimney, an arched
timber front door, carriage garage doors at street level, and a north courtyard with pool and outdoor fire room.

    python scripts/designs/wraparound_cycles.py --view street|courtyard|aerial --scale 50 --samples 32 --out x.png
    python scripts/designs/wraparound_cycles.py --export-glb data/3d/designs/wraparound.glb

Run scripts/designs/wraparound_site.py ground first (it writes house_massing.json and render_scene.json).
Scene axes: X = u (along the frontage), Y = n (into the lot), Z = metres above FFL 157.6.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

import cycles_render as cr  # noqa: E402
from cycles_render import (ARC, GZ, PADZ, B, MATS, TEX, arc_n, box, cyl, drape, faces, poly, prism, slab,  # noqa: E402
                           zg, light, bush, tree, WARM, srgb, img_mat, flat_mat, noisy_mat, seam_roof, thin_glass,
                           leaf_mat, terrain_mat)
# shared wall / window / roof helpers from the current farmhouse design (reused, not edited)
from farmhouse_cycles import (frame_rect, window_u, window_n, gable_u, gable_n,  # noqa: E402
                              roof_ridge_u, roof_ridge_n, lantern, room_light, cr_pendant)

D3 = ROOT / "data" / "3d"
LAY = json.loads((D3 / "house_massing.json").read_text())
assert LAY.get("design") == "wraparound", "run scripts/designs/wraparound_site.py ground first"
L = LAY["layout"]
FRONT = LAY["front"]
PAD_OUTLINE = LAY["pad_outline"]
LOTF = L["lot_frame"]
UPZ = L["upper_z"] - cr.FFL
TIERZ = 156.25 - cr.FFL
rng = random.Random(9)
rr = rng.uniform
T = 0.25
RF = L["roofs"]
EAVE_W, PITCH_W = RF["west"][0], math.radians(RF["west"][1])
EAVE_M, PITCH_M = RF["main"][0], math.radians(RF["main"][1])
EAVE_E, PITCH_E = RF["east"][0], math.radians(RF["east"][1])
EAVE_R, PITCH_R = RF["rear"][0], math.radians(RF["rear"][1])
UPW = L["upper_floor"]  # west wing upper floor
UB = L["upper_bay"]


def nt(u):
    a, b = L["upper_line"]
    return a + b * u


def inside_lot(u, n, poly_=None):
    pts = poly_ or LOTF
    c = False
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        if (y0 > n) != (y1 > n) and u < x0 + (n - y0) * (x1 - x0) / (y1 - y0):
            c = not c
    return c


def textures():
    R = random.Random(21)
    # white board-and-batten: 1024 px = 4 m, battens every 0.4 m
    im = Image.new("RGB", (1024, 1024), (236, 234, 228))
    d = ImageDraw.Draw(im)
    for x in range(0, 1024, 102):
        d.rectangle([x, 0, x + 12, 1024], fill=(246, 245, 241))
        d.rectangle([x + 13, 0, x + 16, 1024], fill=(196, 194, 188))
    arr = np.array(im).astype(np.int16) + np.random.default_rng(3).integers(-4, 4, (1024, 1024))[..., None]
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(TEX / "cladding.png")
    # whitewashed brick (1024 px = 2 m): limewash over red-brown brick, the brick showing through in places
    im = Image.new("RGB", (1024, 1024), (206, 202, 194))
    d = ImageDraw.Draw(im)
    for row, y in enumerate(range(0, 1024, 39)):
        off = 0 if row % 2 == 0 else -59
        for x in range(off, 1024, 118):
            f = R.uniform(0.0, 0.16) if R.random() < 0.8 else R.uniform(0.25, 0.5)
            wash, brick = (243, 240, 233), (178, 132, 104)
            c = tuple(int(wash[i] * (1 - f) + brick[i] * f + R.uniform(-4, 4)) for i in range(3))
            d.rectangle([x + 3, y + 3, x + 115, y + 36], fill=c)
    arr = np.array(im).astype(np.float32)
    wash = np.array(Image.fromarray((np.random.default_rng(5).random((16, 16)) * 255).astype(np.uint8)).resize((1024, 1024), Image.BICUBIC), np.float32) / 255
    arr = arr * (1 - 0.25 * wash[..., None]) + 245 * 0.25 * wash[..., None]
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)).save(TEX / "wbrick.png")
    # coursed fieldstone (1024 px = 2 m): warm grey and tan stones of mixed size, recessed dark mortar
    im = Image.new("RGB", (1024, 1024), (96, 90, 82))
    d = ImageDraw.Draw(im)
    y = 0
    while y < 1024:
        h = int(R.uniform(48, 120))
        x = -R.uniform(0, 120)
        while x < 1024:
            w = R.uniform(80, 230)
            tone = R.choice([(196, 186, 168), (176, 168, 156), (210, 198, 176), (160, 150, 136), (188, 172, 150)])
            l = R.uniform(-14, 14)
            c = tuple(int(t + l) for t in tone)
            j = [R.uniform(4, 12) for _ in range(8)]
            d.polygon([(x + j[0], y + j[1]), (x + w * 0.5, y + j[2] * 0.5), (x + w - j[3], y + j[4]), (x + w - j[5] * 0.5, y + h * 0.5),
                       (x + w - j[6], y + h - j[7]), (x + w * 0.5, y + h - j[0] * 0.6), (x + j[2], y + h - j[3]), (x + j[4] * 0.5, y + h * 0.5)], fill=c)
            x += w
        y += h
    arr = np.array(im).astype(np.int16) + np.random.default_rng(7).integers(-12, 12, (1024, 1024))[..., None]
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.9)).save(TEX / "fstone.png")
    # timber lining boards (1024 px = 4 m, 0.1 m boards), honey-toned
    im = Image.new("RGB", (1024, 1024), (120, 80, 48))
    d = ImageDraw.Draw(im)
    for y in range(0, 1024, 26):
        l = R.uniform(-18, 14)
        d.rectangle([0, y, 1024, y + 24], fill=(int(190 + l), int(142 + l), int(96 + l * 0.8)))
    arr = np.array(im).astype(np.int16) + np.random.default_rng(8).integers(-6, 6, (1024, 1024))[..., None]
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(TEX / "lining.png")
    Image.fromarray(np.ascontiguousarray(arr.clip(0, 255).astype(np.uint8).transpose(1, 0, 2))).save(TEX / "lining_v.png")
    # carriage doors: vertical cedar boards (512 px = 2 m, 0.12 m boards)
    im = Image.new("RGB", (512, 512), (52, 32, 20))
    d = ImageDraw.Draw(im)
    for x in range(0, 512, 31):
        l = R.uniform(-12, 12)
        d.rectangle([x, 0, x + 28, 512], fill=(int(150 + l), int(96 + l), int(58 + l * 0.6)))
    arr = np.array(im).astype(np.int16) + np.random.default_rng(9).integers(-8, 8, (512, 512))[..., None]
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(TEX / "carriage.png")
    # hydrangeas: dark leaves with creamy-white mophead blooms, and a lime-green variant
    im = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for _ in range(160):
        cx, cy, a = R.uniform(40, 472), R.uniform(40, 472), R.uniform(0, 6.28)
        if math.hypot(cx - 256, cy - 256) > 225:
            continue
        L_, W_ = R.uniform(22, 34), R.uniform(12, 18)
        pts = [(cx + math.cos(a + t) * (L_ if k % 2 == 0 else W_), cy + math.sin(a + t) * (L_ if k % 2 == 0 else W_))
               for k, t in enumerate((0, math.pi / 2, math.pi, 3 * math.pi / 2))]
        l = R.uniform(-15, 15)
        d.polygon(pts, fill=(int(48 + l), int(86 + l), int(42 + l), 255))
    for _ in range(10):
        cx, cy, r = R.uniform(110, 400), R.uniform(90, 330), R.uniform(36, 58)
        for _ in range(140):
            a, rd = R.uniform(0, 6.28), R.uniform(0, r)
            x, y = cx + math.cos(a) * rd, cy + math.sin(a) * rd
            l = R.uniform(-12, 8)
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(int(246 + l), int(244 + l), int(232 + l), 255))
    im.save(TEX / "leaf_hydrangea.png")
    arr = np.array(im).astype(np.int16)
    bloom = (arr[..., 0] > 200) & (arr[..., 3] > 0)
    arr[bloom, 0] -= 40
    arr[bloom, 2] -= 95
    Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA").save(TEX / "leaf_hydrangea_lime.png")
    # box: small dense dark leaves
    im = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for _ in range(1400):
        a, rd = R.uniform(0, 6.28), R.uniform(0, 230) ** 0.9 * 1.15
        cx, cy = 256 + math.cos(a) * rd, 256 + math.sin(a) * rd
        if math.hypot(cx - 256, cy - 256) > 236:
            continue
        l = R.uniform(-14, 16)
        d.ellipse([cx - 7, cy - 4, cx + 7, cy + 4], fill=(int(46 + l), int(78 + l), int(34 + l), 255))
    im.save(TEX / "leaf_box.png")
    # Colorbond corrugated fence sheet (256 px = 1 m, 76 mm corrugation): Paperbark beige (NE), Pale Eucalypt (SW)
    x = np.arange(256)
    shade = 0.86 + 0.14 * np.sin(x / 256 * 2 * math.pi * 13.0)
    for name, col in (("fence_beige", (203, 191, 160)), ("fence_sage", (124, 135, 110))):
        arr = (np.array(col)[None, None, :] * shade[None, :, None]).repeat(256, 0)
        Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(TEX / f"{name}.png")


# ------------------------------------------------------------------ walls with several (stacked) openings
def _wall(a0, a1, z0, z1, openings, put):
    cuts = sorted({a0, a1, *[x for o in openings for x in o[:2] if a0 < x < a1]})
    for a, b in zip(cuts, cuts[1:]):
        hits = sorted((o for o in openings if o[0] <= a and b <= o[1]), key=lambda o: o[2])
        z = z0
        for o in hits:
            if o[2] > z:
                put(a, b, z, o[2])
            z = max(z, o[3])
        if z < z1:
            put(a, b, z, z1)


def wall_u(u0, u1, nf, nb, z0, z1, openings, mat):
    """wall spanning u0..u1 between n=nf and n=nb; openings (a, b, zb, zt) along u, may share u ranges"""
    _wall(u0, u1, z0, z1, openings, lambda a, b, za, zb: box(a, b, nf, nb, za, zb, mat))


def wall_n(n0, n1, uf, ub, z0, z1, openings, mat):
    _wall(n0, n1, z0, z1, openings, lambda a, b, za, zb: box(uf, ub, a, b, za, zb, mat))


# ------------------------------------------------------------------ small helpers
def wall_lantern(u, n_face, outward, z, power=18):
    """black box lantern on a wall facing -n (outward=-1) or +n"""
    n0, n1 = sorted((n_face, n_face + outward * 0.16))
    box(u - 0.11, u + 0.11, n0, n1, z, z + 0.46, "black")
    nb0, nb1 = sorted((n_face + outward * 0.02, n_face + outward * 0.17))
    box(u - 0.08, u + 0.08, nb0, nb1, z + 0.06, z + 0.38, "bulb")
    box(u - 0.14, u + 0.14, n0 - 0.02, n1 + 0.02, z + 0.46, z + 0.52, "black")
    light("POINT", (u, n_face + outward * 0.3, z + 0.25), power, WARM, size=0.05, name="door_lantern")


def arch_pts(c, zs, r, k=14):
    return [(c + r * math.cos(math.pi * i / k), zs + r * math.sin(math.pi * i / k)) for i in range(k + 1)]


def spandrels(a, b, zs, top, n0, n1, mat):
    """fill between a semicircular arch (springing at zs) and the flat head at `top`"""
    pts = arch_pts((a + b) / 2, zs, (b - a) / 2)
    for (u0, z0), (u1, z1) in zip(pts, pts[1:]):
        prism([(u0, z0), (u0, top), (u1, top), (u1, z1)], n0, n1, mat)


def arched_door(a, b, zs, n_face):
    """arched double timber door set in a wall whose street face is n_face (facing -n)"""
    c, r = (a + b) / 2, (b - a) / 2
    pts = arch_pts(c, zs, r)
    prism([(a, 0.0), (b, 0.0)] + pts, n_face + 0.12, n_face + 0.18, "door")
    box(c - 0.015, c + 0.015, n_face + 0.1, n_face + 0.12, 0.0, zs + r - 0.03, "black")
    for x0, x1 in ((a + 0.14, c - 0.12), (c + 0.12, b - 0.14)):
        poly([(x0, n_face + 0.11, 1.15), (x1, n_face + 0.11, 1.15), (x1, n_face + 0.11, zs - 0.05), (x0, n_face + 0.11, zs - 0.05)], "glass")
        frame_rect(x0, x1, 1.15, zs - 0.05, n_face + 0.105, "u", 1, 2, w=0.04, depth=0.03)
        box(x0, x1, n_face + 0.1, n_face + 0.12, 0.25, 0.95, "timber")           # raised lower panel
        box(x1 - 0.1 if x0 < c else x0 + 0.04, x1 - 0.04 if x0 < c else x0 + 0.1, n_face + 0.06, n_face + 0.1, 1.0, 1.25, "black")  # handles
    poly([(a, n_face + 0.11, zs), (b, n_face + 0.11, zs)] + [(u, n_face + 0.11, z) for u, z in pts[1:-1]], "glass") if False else None


# ------------------------------------------------------------------ the house
def build_house():
    wu0, wu1, wn0, wn1 = L["west"]
    mu0, mu1, mn0, mn1 = L["main"]
    eu0, eu1, en0, en1 = L["east"]
    ru0, ru1, rn0, rn1 = L["rear"]
    vu0, vu1, vn0, vn1 = L["verandah"]
    gu0, gu1, gn0, gn1 = L["garage"]
    for u0, u1, n0, n1 in (L["west"], L["main"], L["east"], L["rear"]):
        box(u0, u1, n0, n1, -0.25, 0.0, "floor")

    # ---- west gable wing: two storeys of board-and-batten; the main suite upstairs looks over the street and east
    bay = (4.4, 7.0)
    ju = (3.0, 4.8)                       # Juliet balcony doors (upper floor, street)
    wall_u(wu0, wu1, wn0, wn0 + T, 0, EAVE_W, [(bay[0], bay[1], 0.0, 2.7), (ju[0], ju[1], UPW, UPW + 2.15),
                                                (UB[0], wu1, UPW, UPW + 2.3)], "clad")
    window_u(ju[0], ju[1], UPW, UPW + 2.15, wn0, wn0 + T, cols=2, rows=3)
    box(ju[0] - 0.15, ju[1] + 0.15, wn0 - 0.35, wn0, UPW - 0.12, UPW, "black")          # Juliet ledge + railing
    for k in range(13):
        x = ju[0] - 0.1 + k * (ju[1] - ju[0] + 0.2) / 12
        box(x - 0.012, x + 0.012, wn0 - 0.32, wn0 - 0.3, UPW, UPW + 1.0, "black")
    box(ju[0] - 0.15, ju[1] + 0.15, wn0 - 0.34, wn0 - 0.28, UPW + 1.0, UPW + 1.05, "black")
    zr_w = gable_u(wu0, wu1, wn0, wn0 + T, EAVE_W, PITCH_W, "clad", opening=(5.15, 5.85, EAVE_W + 0.45, EAVE_W + 1.35))
    window_u(5.15, 5.85, EAVE_W + 0.45, EAVE_W + 1.35, wn0, wn0 + T, cols=1, rows=2)
    # cantilevered upper bay over the verandah roof: corner glazing to the street (SSE) and east; the NE face
    # (towards the neighbour's yard) has a high clear sill with fixed timber louvres below
    bu0, bu1, bn0, bn1 = UB
    zf, zt = UPW + 0.1, UPW + 2.3
    box(bu0, bu1, bn0, bn1, zf - 0.11, zf, "clad")
    box(bu0, bu0 + 0.15, bn0, bn1, zf, zt, "clad")
    box(bu0, bu1, bn0, bn1, zt, zt + 0.3, "clad")
    poly([(bu0 + 0.15, bn0 + 0.05, UPW + 0.35), (bu1 - 0.05, bn0 + 0.05, UPW + 0.35), (bu1 - 0.05, bn0 + 0.05, zt), (bu0 + 0.15, bn0 + 0.05, zt)], "glass")
    frame_rect(bu0 + 0.15, bu1 - 0.05, UPW + 0.35, zt, bn0 + 0.03, "u", 3, 1)
    box(bu0 + 0.15, bu1, bn0, bn0 + 0.1, zf, UPW + 0.35, "clad")
    poly([(bu1 - 0.05, bn0 + 0.1, UPW + 0.2), (bu1 - 0.05, bn1 - 0.05, UPW + 0.2), (bu1 - 0.05, bn1 - 0.05, zt), (bu1 - 0.05, bn0 + 0.1, zt)], "glass")
    frame_rect(bn0 + 0.05, bn1, UPW + 0.2, zt, bu1 + 0.0, "n", 1, 2)
    for k in range(9):                                                         # fixed louvres, lower part
        z = UPW + 0.25 + k * 0.13
        box(bu1 - 0.02, bu1 + 0.1, bn0 + 0.1, bn1 - 0.05, z, z + 0.035, "timber")
    slab((bu0 - 0.15, bn0 - 0.2, zt + 0.3), (bu1 + 0.15, bn0 - 0.2, zt + 0.3), (bu1 + 0.15, bn1, zt + 0.48), (bu0 - 0.15, bn1, zt + 0.48), 0.06, "roof_x")
    box(wu0 + T, wu1 - T, wn0 + T, wn1 - T, UPW - 0.25, UPW, "slab")
    room_light(wu0 + 0.5, wu1 - 0.5, wn0 + 0.5, wn1 - 0.5, UPW - 0.25, 1100)
    room_light(wu0 + 0.5, wu1 - 0.5, wn0 + 0.5, wn1 - 0.5, EAVE_W, 900)
    room_light(bu0 + 0.3, bu1 - 0.3, bn0 + 0.2, bn1 + 0.6, zt, 250)
    box(3.2, 5.2, wn0 + 1.2, wn0 + 3.4, UPW, UPW + 0.55, "linen")             # bed
    box(bu0 + 0.5, bu0 + 1.3, bn0 + 0.3, bn1 + 0.3, UPW, UPW + 0.45, "fabric")  # chair in the bay
    # SW side wall facing the verandah leg, split by the external fieldstone chimney
    sw_w = [(11.4, 12.8, 0.8, 2.5), (17.2, 18.6, 0.8, 2.5), (18.0, 19.0, UPW + 1.0, UPW + 2.0)]
    wall_n(wn0, wn1, wu0, wu0 + T, 0, EAVE_W, sw_w, "clad")
    for o in sw_w:
        window_n(*o, wu0, wu0 + T, cols=2, rows=2, outward=-1)
    wall_n(wn0, wn1, wu1 - T, wu1, 0, EAVE_W, [], "clad")
    gable_u(wu0, wu1, wn1 - T, wn1, EAVE_W, PITCH_W, "clad")
    roof_ridge_n(wn0, wn1, wu0, wu1, EAVE_W, PITCH_W)
    box(bay[0] + 0.2, bay[1] - 0.2, wn0 + 0.6, wn0 + 1.4, 0, 0.45, "fabric")
    cr_pendant(5.7, wn0 + 2.4, 2.4, 50)
    # tall fieldstone chimney on the SW wall, rising past the ridge
    cn0, cn1 = 14.6, 16.4
    box(wu0 - 0.8, wu0 + 0.05, cn0, cn1, PADZ - 0.3, EAVE_W - 0.3, "fstone")
    faces("fstone", [(wu0 - 0.8, cn0, EAVE_W - 0.3), (wu0 + 0.05, cn0, EAVE_W - 0.3), (wu0 + 0.05, cn1, EAVE_W - 0.3), (wu0 - 0.8, cn1, EAVE_W - 0.3),
                     (wu0 - 0.65, cn0 + 0.35, EAVE_W + 0.5), (wu0 + 0.05, cn0 + 0.35, EAVE_W + 0.5), (wu0 + 0.05, cn1 - 0.35, EAVE_W + 0.5), (wu0 - 0.65, cn1 - 0.35, EAVE_W + 0.5)],
          [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    ztop = zr_w + 1.3
    box(wu0 - 0.65, wu0 + 0.05, cn0 + 0.35, cn1 - 0.35, EAVE_W + 0.5, ztop, "fstone")
    box(wu0 - 0.73, wu0 + 0.13, cn0 + 0.27, cn1 - 0.27, ztop, ztop + 0.12, "limestone")
    for dn in (-0.25, 0.25):
        cyl((wu0 - 0.3, (cn0 + cn1) / 2 + dn, ztop + 0.12), (wu0 - 0.3, (cn0 + cn1) / 2 + dn, ztop + 0.45), 0.1, 0.09, "black", 8)

    # ---- main body (whitewashed brick): arched door in a limestone portico, two small windows, three dormers
    dr = (11.4, 13.2)
    zs = 2.1
    w1, w2 = (9.55, 10.75, 0.9, 2.5), (14.25, 15.25, 0.9, 2.5)
    wall_u(mu0, mu1, mn0, mn0 + T, 0, EAVE_M, [(dr[0], dr[1], 0.0, zs + 0.9), w1, w2], "wbrick")
    spandrels(dr[0], dr[1], zs, zs + 0.9, mn0, mn0 + T, "wbrick")
    window_u(*w1, mn0, mn0 + T, cols=2, rows=2)
    window_u(*w2, mn0, mn0 + T, cols=2, rows=2)
    arched_door(dr[0], dr[1], zs, mn0)
    # projecting limestone portico (0.4 m) with an arched opening, under the verandah roof
    pn = mn0 - 0.4
    pa, pb = 11.0, 13.6
    box(pa, dr[0], pn, mn0, 0, zs, "limestone")
    box(dr[1], pb, pn, mn0, 0, zs, "limestone")
    box(pa, pb, pn, mn0, zs + 0.9, 2.82, "limestone")
    box(pa, dr[0], pn, mn0, zs, zs + 0.9, "limestone")
    box(dr[1], pb, pn, mn0, zs, zs + 0.9, "limestone")
    spandrels(dr[0], dr[1], zs, zs + 0.9, pn, mn0, "limestone")
    box(pa - 0.06, pb + 0.06, pn - 0.06, mn0, 2.82, 2.9, "limestone")
    box(12.3 - 0.13, 12.3 + 0.13, pn - 0.04, pn, zs + 0.78, zs + 1.05, "limestone")  # keystone
    for x in (pa - 0.25, pb + 0.25):
        wall_lantern(x, mn0, -1, 1.65)
    # rear (north) wall: big black-framed glazing onto the courtyard
    big = (mu0 + 0.4, mu1 - 0.4, 0.0, 2.9)
    wall_u(mu0, mu1, mn1 - T, mn1, 0, EAVE_M, [big], "wbrick")
    window_u(*big, mn1 - T, mn1, cols=4, rows=1, outward=1)
    gable_n(mn0, mn1, mu1 - T, mu1, EAVE_M, PITCH_M, "wbrick")
    roof_ridge_u(mu0 + 0.3, mu1, mn0, mn1, EAVE_M, PITCH_M, o=0.3)   # starts at the west wing wall
    for du in (10.25, 12.3, 14.35):
        n_f = mn0 + 1.6
        z_b = EAVE_M + 1.6 * math.tan(PITCH_M) - 0.3
        wall_u(du - 0.75, du + 0.75, n_f, n_f + 0.12, z_b, z_b + 1.7, [(du - 0.4, du + 0.4, z_b + 0.45, z_b + 1.4)], "clad")
        window_u(du - 0.4, du + 0.4, z_b + 0.45, z_b + 1.4, n_f, n_f + 0.12, cols=2, rows=2)
        box(du - 0.75, du - 0.63, n_f, n_f + 2.4, z_b, z_b + 1.7, "clad")
        box(du + 0.63, du + 0.75, n_f, n_f + 2.4, z_b, z_b + 1.7, "clad")
        gable_u(du - 0.75, du + 0.75, n_f, n_f + 0.12, z_b + 1.7, math.radians(45), "clad")
        roof_ridge_n(n_f, n_f + 2.4, du - 0.75, du + 0.75, z_b + 1.7, math.radians(45), o=0.18)
        light("POINT", (du, n_f + 1.2, z_b + 1.1), 35, WARM, size=0.1, name="attic")
    room_light(mu0 + 0.5, mu1 - 0.5, mn0 + 0.5, mn1 - 0.5, EAVE_M, 1500)
    box(9.8, 12.6, mn1 - 3.6, mn1 - 2.6, 0.72, 0.77, "walnut")
    cr_pendant(11.2, mn1 - 3.1, 2.0)
    cr_pendant(13.6, mn0 + 2.0, 2.3, 50)
    box(13.6, 15.2, mn1 - 1.8, mn1 - 0.9, 0, 0.45, "fabric")

    # ---- wraparound verandah: front run + SW leg, hipped at the corner, chunky posts on limestone piers
    su0, su1, sn0, sn1 = L["verandah_side"]
    for a, b, c_, d_ in ((vu0, vu1, vn0, vn1), (su0, su1, sn0, sn1)):
        box(a, b, c_, d_, PADZ - 0.3, -0.03, "limestone")
        box(a + 0.03, b - (0.03 if b < 10 else 0), c_ + 0.03, d_ - (0.03 if d_ > 20 else 0), -0.03, 0.0, "deck")
    zl, zh, ZB = 2.78, 3.0, 2.48
    eo = 0.25
    p0 = (vu0 - eo, vn0 - eo)
    front = [(p0[0], p0[1], zl), (vu1 + 0.2, p0[1], zl), (vu1 + 0.2, vn1, zh), (su1, vn1, zh)]
    side = [(p0[0], p0[1], zl), (su1, vn1, zh), (su1, sn1, zh), (p0[0], sn1, zl)]
    slab(*front, 0.08, "roof_x")
    slab(*side, 0.08, "roof_y")
    poly([(x, y, z - 0.015) for x, y, z in front], "lining_u")
    poly([(x, y, z - 0.015) for x, y, z in side], "lining_n")
    box(p0[0] - 0.05, vu1 + 0.2, p0[1] - 0.06, p0[1] + 0.06, zl - 0.2, zl + 0.03, "black")
    box(p0[0] - 0.06, p0[0] + 0.06, p0[1], sn1, zl - 0.2, zl + 0.03, "black")
    box(vu1 + 0.14, vu1 + 0.26, p0[1], vn1, zl - 0.2, zh + 0.03, "black")
    fposts = [0.3, 3.4, 6.5, 9.2, 10.8, 13.75]
    sposts = [11.0, 14.4, 17.8, 20.7]
    pts_ = [(x, vn0 + 0.3) for x in fposts] + [(su0 + 0.3, n) for n in sposts]
    for x, n in pts_:
        box(x - 0.3, x + 0.3, n - 0.3, n + 0.3, 0, 0.8, "limestone")
        box(x - 0.34, x + 0.34, n - 0.34, n + 0.34, 0.8, 0.88, "limestone")
        box(x - 0.16, x + 0.16, n - 0.16, n + 0.16, 0.88, ZB, "timber")
    box(vu0, vu1 + 0.2, vn0 + 0.14, vn0 + 0.46, ZB, ZB + 0.3, "timber")
    box(su0 + 0.14, su0 + 0.46, vn0, sn1, ZB, ZB + 0.3, "timber")
    for x in fposts:
        for sg in (-1, 1):
            if vu0 + 0.4 < x + sg * 0.6 < vu1 - 0.1:
                cyl((x, vn0 + 0.3, ZB - 0.6), (x + sg * 0.6, vn0 + 0.3, ZB - 0.02), 0.06, 0.06, "timber")
    for n in sposts + [vn0 + 0.3]:
        for sg in (-1, 1):
            if vn0 + 0.5 < n + sg * 0.6 < sn1 - 0.2:
                cyl((su0 + 0.3, n, ZB - 0.6), (su0 + 0.3, n + sg * 0.6, ZB - 0.02), 0.06, 0.06, "timber")
    for x in (1.8, 5.0, 8.0, 12.3):
        box(x - 0.08, x + 0.08, vn0 + 1.5, vn0 + 1.66, 2.88, 2.9, "bulb")
        light("POINT", (x, vn0 + 1.58, 2.7), 12, WARM, size=0.05, name="verandah")
    for n in (13.0, 17.0):
        box(1.0, 1.16, n - 0.08, n + 0.08, 2.86, 2.88, "bulb")
        light("POINT", (1.08, n, 2.68), 10, WARM, size=0.05, name="verandah")
    box(10.9, 13.7, vn0 - 0.45, vn0 + 0.02, PADZ - 0.2, -0.06, "limestone")   # stone step at the door
    for x in (2.0, 3.2, 7.6, 8.6):                                              # rocking chairs and linen cushions
        box(x - 0.32, x + 0.32, vn0 + 0.9, vn0 + 1.55, 0.0, 0.42, "timber")
        box(x - 0.32, x + 0.32, vn0 + 1.45, vn0 + 1.55, 0.42, 1.05, "timber")
        box(x - 0.28, x + 0.28, vn0 + 0.95, vn0 + 1.42, 0.42, 0.5, "linen")
    box(su0 + 0.6, su0 + 1.3, 12.6, 14.2, 0.0, 0.45, "timber")                 # daybed on the SW leg
    box(su0 + 0.65, su0 + 1.25, 12.65, 14.15, 0.45, 0.56, "linen")
    for x in (pa - 0.55, pb + 0.55):                                            # potted box balls at the portico
        cyl((x, vn1 - 0.75, 0.0), (x, vn1 - 0.75, 0.5), 0.22, 0.27, "fstone", 10)
        bush((x, vn1 - 0.75, 0.85), 0.36, "box", 20)

    # ---- east wing (single storey, vaulted) over the street-level garage
    # garage: whitewashed brick base with two carriage doors; a pent awning on timber brackets
    d1, d2 = (16.15, 18.85), (19.25, 21.95)
    wall_u(eu0, eu1, en0, en0 + 0.35, GZ - 0.3, 0.0, [(d1[0], d1[1], GZ, GZ + 2.3), (d2[0], d2[1], GZ, GZ + 2.3)], "wbrick")
    box(eu0, eu0 + 0.3, en0, gn1, GZ - 0.3, 0.0, "wbrick")
    box(eu1 - 0.3, eu1, en0, gn1, GZ - 0.3, 0.0, "wbrick")
    box(eu0, eu1, gn1 - 0.3, gn1, GZ - 0.3, 0.0, "wbrick")
    for a, b in (d1, d2):
        box(a, b, en0 + 0.22, en0 + 0.28, GZ, GZ + 2.3, "carriage")
        c = (a + b) / 2
        box(c - 0.02, c + 0.02, en0 + 0.19, en0 + 0.22, GZ, GZ + 2.3, "black")
        for k in range(4):                                                     # glazed top lites
            x0 = a + 0.15 + k * (b - a - 0.3) / 4
            x1 = x0 + (b - a - 0.3) / 4 - 0.08
            poly([(x0, en0 + 0.215, GZ + 1.75), (x1, en0 + 0.215, GZ + 1.75), (x1, en0 + 0.215, GZ + 2.12), (x0, en0 + 0.215, GZ + 2.12)], "glass")
            frame_rect(x0, x1, GZ + 1.75, GZ + 2.12, en0 + 0.21, "u", 1, 1, w=0.04, depth=0.02)
        for z in (GZ + 0.45, GZ + 1.45):                                       # strap hinges
            box(a + 0.05, a + 0.75, en0 + 0.18, en0 + 0.22, z, z + 0.06, "black")
            box(b - 0.75, b - 0.05, en0 + 0.18, en0 + 0.22, z, z + 0.06, "black")
        box(c - 0.18, c - 0.05, en0 + 0.15, en0 + 0.19, GZ + 1.0, GZ + 1.3, "black")
        box(c + 0.05, c + 0.18, en0 + 0.15, en0 + 0.19, GZ + 1.0, GZ + 1.3, "black")
        box(a - 0.1, b + 0.1, en0 - 0.05, en0 + 0.2, GZ + 2.3, GZ + 2.5, "timber")  # timber head beam
    slab((eu0 + 0.2, en0, -0.12), (eu1 - 0.2, en0, -0.12), (eu1 - 0.2, en0 - 0.85, -0.38), (eu0 + 0.2, en0 - 0.85, -0.38), 0.06, "roof_x")
    for x in (eu0 + 0.35, 19.05, eu1 - 0.35):
        cyl((x, en0, -0.9), (x, en0 - 0.75, -0.36), 0.05, 0.05, "timber")
        if x != 19.05:
            wall_lantern(x, en0, -1, GZ + 1.6, 15)
    box(eu0, eu1, en0 - 0.06, en0, -0.08, 0.02, "black")                     # flashing between brick and cladding
    # upper front: board-and-batten gable with a small projecting bay window (articulation) and small windows
    fw = [(16.1, 17.3, 0.9, 2.3), (20.7, 21.9, 0.9, 2.3)]
    wall_u(eu0, eu1, en0, en0 + T, 0, EAVE_E, fw + [(17.8, 20.4, 0.7, 2.5)], "clad")
    for o in fw:
        window_u(*o, en0, en0 + T, cols=2, rows=2, awning=True)
    ebn = en0 - 0.4
    box(17.8, 20.4, ebn, en0, 0.0, 0.7, "clad")
    box(17.75, 20.45, ebn - 0.04, en0, 0.7, 0.76, "limestone")
    poly([(17.9, ebn + 0.04, 0.76), (20.3, ebn + 0.04, 0.76), (20.3, ebn + 0.04, 2.35), (17.9, ebn + 0.04, 2.35)], "glass")
    frame_rect(17.9, 20.3, 0.76, 2.35, ebn + 0.02, "u", 3, 2)
    for x in (17.8, 20.34):
        box(x, x + 0.06, ebn, en0, 0.76, 2.35, "black")
    box(17.75, 20.45, ebn - 0.06, en0, 2.35, 2.5, "black")
    slab((17.65, ebn - 0.15, 2.5), (20.55, ebn - 0.15, 2.5), (20.55, en0, 2.75), (17.65, en0, 2.75), 0.05, "roof_x")
    zr_e = gable_u(eu0, eu1, en0, en0 + T, EAVE_E, PITCH_E, "clad", opening=(18.6, 19.5, EAVE_E + 0.45, EAVE_E + 1.5))
    window_u(18.6, 19.5, EAVE_E + 0.45, EAVE_E + 1.5, en0, en0 + T, cols=1, rows=2)
    # SW face: small window by the apron, glazing to the courtyard beyond the main body
    wall_n(en0, en1, eu0, eu0 + T, 0, EAVE_E, [(8.0, 9.2, 1.0, 2.3), (21.5, 24.0, 0.0, 2.7)], "clad")
    window_n(8.0, 9.2, 1.0, 2.3, eu0, eu0 + T, cols=1, rows=2, outward=-1)
    window_n(21.5, 24.0, 0.0, 2.7, eu0, eu0 + T, cols=2, rows=1, outward=-1)
    # NE face to the neighbour, stepped in 0.4 m at n = 15.4 (articulation), small windows
    nstep = 15.4
    wall_n(en0, nstep, eu1 - T, eu1, 0, EAVE_E, [(9.6, 10.6, 1.2, 2.2), (12.6, 13.6, 1.2, 2.2)], "clad")
    window_n(9.6, 10.6, 1.2, 2.2, eu1 - T, eu1, cols=1, rows=2)
    window_n(12.6, 13.6, 1.2, 2.2, eu1 - T, eu1, cols=1, rows=2)
    wall_u(eu1 - 0.4 - T, eu1, nstep - T, nstep, 0, EAVE_E, [], "clad")
    wall_n(nstep, en1, eu1 - 0.4 - T, eu1 - 0.4, 0, EAVE_E, [(18.0, 19.2, 1.2, 2.2)], "clad")
    window_n(18.0, 19.2, 1.2, 2.2, eu1 - 0.4 - T, eu1 - 0.4, cols=1, rows=2)
    # north gable: full-width glazing to the pool, clerestory in the gable
    eg = (eu0 + 0.6, eu1 - 0.9, 0.0, 2.7)
    wall_u(eu0, eu1 - 0.4, en1 - T, en1, 0, EAVE_E, [eg], "clad")
    window_u(*eg, en1 - T, en1, cols=4, rows=1, outward=1)
    gable_u(eu0, eu1, en1 - T, en1, EAVE_E, PITCH_E, "clad", opening=(18.5, 19.6, EAVE_E + 0.4, EAVE_E + 1.9))
    window_u(18.5, 19.6, EAVE_E + 0.4, EAVE_E + 1.9, en1 - T, en1, cols=1, rows=2, outward=1)
    roof_ridge_n(en0, en1, eu0, eu1, EAVE_E, PITCH_E)
    room_light(eu0 + 0.5, eu1 - 0.5, gn1 + 0.5, en1 - 0.5, EAVE_E, 1300)
    room_light(eu0 + 0.5, eu1 - 0.5, en0 + 0.5, gn1, EAVE_E, 700)
    box(16.6, 18.6, en1 - 3.0, en1 - 1.0, 0, 0.55, "linen")
    cr_pendant(19.0, en1 - 4.5, 2.4, 50)

    # ---- rear two-storey family wing (whitewashed brick), glazed east to the pool and north to the garden
    sl = (rn0 + 0.8, rn1 - 0.6, 0.0, 2.8)
    wall_n(rn0, rn1, ru1 - T, ru1, 0, EAVE_R, [sl], "wbrick")
    window_n(*sl, ru1 - T, ru1, cols=5, rows=1, outward=1)
    sw_r = [(23.0, 24.2, 1.2, 2.2), (26.6, 27.8, 1.2, 2.2)]
    wall_n(rn0, rn1, ru0, ru0 + T, 0, EAVE_R, sw_r, "wbrick")
    for o in sw_r:
        window_n(*o, ru0, ru0 + T, cols=1, rows=2, outward=-1)
    gw = (ru0 + 0.8, ru1 - 0.8, 0.0, 2.8)
    wall_u(ru0, ru1, rn1 - T, rn1, 0, EAVE_R, [gw], "wbrick")
    window_u(*gw, rn1 - T, rn1, cols=5, rows=1, outward=1)
    rc = (ru0 + ru1) / 2
    gable_u(ru0, ru1, rn1 - T, rn1, EAVE_R, PITCH_R, "wbrick", opening=(rc - 1.6, rc + 1.6, EAVE_R + 0.25, EAVE_R + 2.0))
    window_u(rc - 1.6, rc + 1.6, EAVE_R + 0.25, EAVE_R + 2.0, rn1 - T, rn1, cols=1, rows=2, outward=1)
    wall_u(ru0, ru1, rn0, rn0 + T, 0, EAVE_R, [(ru0 + 0.6, ru0 + 1.8, 0.9, 2.4)], "wbrick")
    window_u(ru0 + 0.6, ru0 + 1.8, 0.9, 2.4, rn0, rn0 + T, cols=1, rows=2)
    gable_u(ru0, ru1, rn0, rn0 + T, EAVE_R, PITCH_R, "wbrick")
    roof_ridge_n(rn0, rn1, ru0, ru1, EAVE_R, PITCH_R)
    room_light(ru0 + 0.5, ru1 - 0.5, rn0 + 0.5, rn1 - 0.5, EAVE_R + 1.5, 2200)
    box(1.0, 4.6, 24.0, 25.0, 0, 0.45, "fabric")
    box(1.0, 1.9, 25.0, 27.4, 0, 0.45, "fabric")
    box(2.2, 4.0, 25.6, 26.8, 0, 0.02, "rug")
    cr_pendant(5.6, 26.0, 2.0)

    # ---- covered outdoor room by the pool: fieldstone fireplace, timber posts, timber-lined vaulted ceiling
    ou0, ou1, on0, on1 = L["outdoor"]
    zo = PADZ + 0.15
    ze = zo + RF["outdoor"][0]
    po = math.radians(RF["outdoor"][1])
    box(ou0, ou1, on0, on1, PADZ - 0.3, zo, "flagstone")
    for x, nn in ((ou0 + 0.2, on0 + 0.2), (ou0 + 0.2, on1 - 0.2), (ou1 - 0.25, on0 + 0.2), (ou1 - 0.25, on1 - 0.2)):
        box(x - 0.28, x + 0.28, nn - 0.28, nn + 0.28, zo, zo + 0.7, "limestone")
        box(x - 0.15, x + 0.15, nn - 0.15, nn + 0.15, zo + 0.7, ze, "timber")
    box(ou0, ou1, on0 + 0.05, on0 + 0.35, ze - 0.3, ze, "timber")
    box(ou0, ou1, on1 - 0.35, on1 - 0.05, ze - 0.3, ze, "timber")
    zr_o = roof_ridge_u(ou0, ou1, on0, on1, ze, po, o=0.35)
    nm = (on0 + on1) / 2
    for nn0, nn1 in ((on0, nm), (on1, nm)):                                   # timber-lined ceiling under the roof
        poly([(ou0 - 0.3, nn0, ze - 0.02), (ou1 + 0.3, nn0, ze - 0.02), (ou1 + 0.3, nn1, zr_o + 0.02), (ou0 - 0.3, nn1, zr_o + 0.02)], "lining_u")
    for x in (ou0 + 0.1, (ou0 + ou1) / 2):                                     # exposed tie beams and king posts
        box(x - 0.1, x + 0.1, on0, on1, ze - 0.25, ze, "timber")
        box(x - 0.07, x + 0.07, nm - 0.07, nm + 0.07, ze, zr_o - 0.05, "timber")
    box(ou0 - 0.05, ou1, nm - 0.1, nm + 0.1, zr_o - 0.25, zr_o, "timber")
    # fireplace on the NE side facing the pool, chimney above the ridge
    fn0, fn1 = on0 + 1.2, on1 - 1.2
    box(ou1 - 0.8, ou1, fn0, fn1, zo, ze + 0.2, "fstone")
    gable_n(on0, on1, ou1 - 0.15, ou1, ze, po, "fstone")
    box(ou1 - 0.7, ou1, fn0 + 0.5, fn1 - 0.5, ze + 0.2, zr_o + 1.4, "fstone")
    box(ou1 - 0.78, ou1 + 0.08, fn0 + 0.42, fn1 - 0.42, zr_o + 1.4, zr_o + 1.52, "limestone")
    box(ou1 - 0.86, ou1 - 0.78, fn0 + 0.6, fn1 - 0.6, zo + 0.45, zo + 1.4, "black")
    box(ou1 - 0.78, ou1 - 0.5, fn0 + 0.75, fn1 - 0.75, zo + 0.5, zo + 0.7, "fire")
    box(ou1 - 0.95, ou1 - 0.78, fn0 + 0.2, fn1 - 0.2, zo + 1.55, zo + 1.75, "timber")   # mantel
    box(ou1 - 1.3, ou1 - 0.8, fn0, fn1, zo, zo + 0.35, "limestone")                    # raised hearth
    light("POINT", (ou1 - 1.1, (fn0 + fn1) / 2, zo + 0.8), 140, (1.0, 0.45, 0.15), size=0.3, name="pool_fire")
    box(ou0 + 0.5, ou0 + 1.35, on0 + 1.3, on1 - 1.3, zo, zo + 0.45, "linen")
    box(ou0 + 0.5, ou0 + 0.78, on0 + 1.3, on1 - 1.3, zo + 0.45, zo + 0.85, "linen")
    box(ou0 + 1.8, ou0 + 2.7, on0 + 2.2, on1 - 2.2, zo, zo + 0.4, "walnut")
    box(ou0 + 1.4, ou1 - 1.4, on0 + 1.2, on1 - 1.2, zo, zo + 0.01, "rug")
    cr_pendant((ou0 + ou1) / 2 - 0.4, nm, zo + 2.3, 45)

    # ---- north courtyard: pool with stone coping, sun loungers, dining table
    pu0, pu1, pn0, pn1 = L["pool"]
    for a, c, n0_, n1_ in ((pu0 - 0.3, pu1 + 0.3, pn0 - 0.3, pn0), (pu0 - 0.3, pu1 + 0.3, pn1, pn1 + 0.3), (pu0 - 0.3, pu0, pn0, pn1), (pu1, pu1 + 0.3, pn0, pn1)):
        box(a, c, n0_, n1_, PADZ - 0.2, PADZ + 0.07, "coping")
    box(pu0, pu1, pn0, pn1, PADZ - 1.7, PADZ - 1.6, "pooltile")
    for a, c, n0_, n1_ in ((pu0, pu1, pn0, pn0 + 0.02), (pu0, pu1, pn1 - 0.02, pn1), (pu0, pu0 + 0.02, pn0, pn1), (pu1 - 0.02, pu1, pn0, pn1)):
        box(a, c, n0_, n1_, PADZ - 1.7, PADZ - 0.1, "pooltile")
    poly([(pu0, pn0, PADZ - 0.12), (pu1, pn0, PADZ - 0.12), (pu1, pn1, PADZ - 0.12), (pu0, pn1, PADZ - 0.12)], "water")
    for x in (pu0 + 2.5, (pu0 + pu1) / 2, pu1 - 2.5):
        light("POINT", (x, pn0 + 0.15, PADZ - 0.9), 55, (0.55, 0.85, 1.0), size=0.05, name="pool")
    # the shared terrain sinks a fixed patch for another design's pool; fill it under this design's paving
    for a, c, n0_, n1_ in ((3.0, pu0 - 0.3, 26.5, 32.0), (pu0 - 0.3, 15.0, pn1 + 0.3, 32.0)):
        box(a, c, n0_, n1_, PADZ - 2.0, PADZ - 0.02, "slab")
    for k, x in enumerate((11.2, 12.6, 14.0, 15.4)):
        box(x - 0.35, x + 0.35, pn1 + 0.7, pn1 + 2.5, PADZ, PADZ + 0.3, "timber")
        box(x - 0.32, x + 0.32, pn1 + 0.72, pn1 + 2.48, PADZ + 0.3, PADZ + 0.38, "linen")
    box(10.4, 13.0, 21.9, 22.9, PADZ + 0.72, PADZ + 0.77, "walnut")
    for x in (10.8, 11.7, 12.6):
        for nn in (21.5, 23.3):
            box(x - 0.2, x + 0.2, nn - 0.2, nn + 0.2, PADZ, PADZ + 0.45, "black")

    # ---- landscape surfaces from the site script (whole lot, two terrace levels)
    lift = {"lawn": PADZ + 0.03, "paving": PADZ + 0.035, "mulch": PADZ + 0.04, "lawn_up": UPZ + 0.03, "mulch_up": UPZ + 0.04}
    mat = {"lawn": "lawn", "paving": "paving", "mulch": "mulch", "lawn_up": "lawn", "mulch_up": "mulch"}
    for k, tiles in L["surfaces"].items():
        for t in tiles:
            poly([(u, n, lift[k]) for u, n in t], mat[k])
    tu0, tu1, tn0, tn1 = L["tier"]
    box(tu0, tu1, tn0, tn1, TIERZ - 0.3, TIERZ + 0.02, "mulch")

    # ---- upper garden terrace: limestone-capped retaining wall following the rear boundary, steps behind the pool
    st0, st1 = 13.4, 15.4
    a_, b_ = L["upper_line"]
    ln = math.hypot(1, b_)
    tx, tn_ = 1 / ln, b_ / ln
    nx, nn_ = -b_ / ln, 1 / ln     # towards the upper terrace
    us = [u for u in np.arange(-6, 31, 0.25) if inside_lot(u, nt(u))]
    ua, ub = us[0] + 0.3, us[-1] - 0.35

    def obox(u0, u1, s0, s1, z0, z1, m):
        """box aligned with the terrace line: u0..u1 along it (by u), s0..s1 across it (metres, + = uphill)"""
        P = lambda u, s: (u + nx * s, nt(u) + nn_ * s)
        q = [P(u0, s0), P(u1, s0), P(u1, s1), P(u0, s1)]
        faces(m, [(x, y, z0) for x, y in q] + [(x, y, z1) for x, y in q],
              [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    for u0, u1 in ((ua, st0), (st1, ub)):
        obox(u0, u1, -0.3, 0.0, PADZ - 0.4, UPZ + 0.12, "limestone")
        obox(u0, u1, -0.36, 0.04, UPZ + 0.12, UPZ + 0.2, "limestone")
    nst = 6
    for k in range(nst):
        z = PADZ + (UPZ - PADZ) * (k + 1) / nst
        obox(st0, st1, -(nst - k) * 0.34, 0.05, PADZ - 0.3, z, "limestone")
    for u in (st0 - 0.15, st1 + 0.15):
        obox(u - 0.15, u + 0.15, -nst * 0.34, 0.1, PADZ - 0.3, UPZ + 0.25, "limestone")
    for u in (st0 - 0.4, st1 + 0.4):
        light("POINT", (u + nx * -0.5, nt(u) + nn_ * -0.5, PADZ + 0.4), 8, WARM, size=0.05, name="sconce")

    # ---- boundary retaining walls along the whole pad outline (painted brick, on the lot side of the boundary)
    pts = PAD_OUTLINE
    for (ua_, na), (ub_, nb) in zip(pts, pts[1:]):
        if max(na, nb) < 2.6:
            continue                                  # the street wall handles the front edge
        if min(ua_, ub_) > gu0 - 0.5 and max(na, nb) < gn1 + 0.6:
            continue                                  # garage / apron edges, built below
        Lg = math.hypot(ub_ - ua_, nb - na)
        steps = max(1, int(Lg / 1.5))
        dx, dn = (ub_ - ua_) / Lg, (nb - na) / Lg
        ox, on = dn * 0.3, -dx * 0.3                  # outward (CCW outline: interior on the left)
        for k in range(steps):
            t0, t1 = k / steps, (k + 1) / steps
            u0_, n0_ = ua_ + (ub_ - ua_) * t0, na + (nb - na) * t0
            u1_, n1_ = ua_ + (ub_ - ua_) * t1, na + (nb - na) * t1
            um, nmid = (u0_ + u1_) / 2, (n0_ + n1_) / 2
            lvl = UPZ if nmid >= nt(um) - 0.05 else (zg(um - ox * 2, nmid - on * 2) if nmid < 4.8 else PADZ)
            g = zg(um + ox * 3.5, nmid + on * 3.5)
            if g > lvl:
                top = max(g + 0.15, lvl + 0.45)
            else:
                top = lvl + (0.95 if lvl - g > 0.6 else 0.45)
            bot = min(g, lvl) - 0.4
            faces("whitebrick", [(u0_, n0_, bot), (u1_, n1_, bot), (u1_ + ox, n1_ + on, bot), (u0_ + ox, n0_ + on, bot),
                                 (u0_, n0_, top), (u1_, n1_, top), (u1_ + ox, n1_ + on, top), (u0_ + ox, n0_ + on, top)],
                  [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    # wall behind the garage between the east wing and the NE boundary (pad above, side path below)
    ne_u = lambda n: 22.903 + 6.864 * n / 44.458 - 0.3
    box(eu1, ne_u(gn1), gn1 - 0.3, gn1, GZ - 0.3, PADZ + 0.95, "whitebrick")

    # ---- street: low painted brick wall, pillars with lanterns, X gate, steps; bank; tiers by the apron
    s0, s1, sn0, sn1 = L["stair"]
    topw = FRONT["wall_top"]
    for a, b in ((0.25, s0 - 0.45), (s1 + 0.45, gu0 - 0.3)):
        box(a, b, 1.6, 2.2, -3.4, topw, "whitebrick")
        box(a - 0.03, b + 0.03, 1.55, 2.25, topw, topw + 0.08, "limestone")
    gg = zg((s0 + s1) / 2, 1.2)
    for x in (s0 - 0.45, s1 + 0.45):
        box(x - 0.45, x + 0.45, 1.45, 2.35, gg - 0.3, topw + 0.5, "whitebrick")
        box(x - 0.52, x + 0.52, 1.38, 2.42, topw + 0.5, topw + 0.62, "limestone")
        lantern(x, 1.9, topw + 0.62, 0.36, 30)
    gz0, gz1 = gg + 0.05, gg + 1.25
    for x0, x1 in ((s0, s0 + 0.08), (s1 - 0.08, s1)):
        box(x0, x1, 1.86, 1.94, gz0, gz1, "black")
    box(s0, s1, 1.86, 1.94, gz1 - 0.08, gz1, "black")
    box(s0, s1, 1.86, 1.94, gz0, gz0 + 0.08, "black")
    for (xa, za), (xb, zb_) in (((s0, gz0), (s1, gz1)), ((s0, gz1), (s1, gz0))):
        L_ = math.hypot(xb - xa, zb_ - za)
        tx_, tz_ = (xb - xa) / L_, (zb_ - za) / L_
        slab((xa - tz_ * 0.05, 1.86, za + tx_ * 0.05), (xb - tz_ * 0.05, 1.86, zb_ + tx_ * 0.05), (xb + tz_ * 0.05, 1.86, zb_ - tx_ * 0.05), (xa + tz_ * 0.05, 1.86, za - tx_ * 0.05), 0.08, "black")
    n_steps = 10
    for k in range(n_steps):
        z = gg + (PADZ - gg) * (k + 1) / n_steps
        box(s0, s1, 2.0 + k * (sn1 - 2.0) / n_steps, sn1, gg - 0.4, z, "flagstone")
    for x0, x1 in ((s0 - 0.3, s0), (s1, s1 + 0.3)):
        box(x0, x1, 2.2, sn1, gg - 0.4, PADZ + 0.35, "whitebrick")
    bt = FRONT["bank_top"]
    drape(0.05, s0 - 0.3, lambda u: 2.2, lambda u: bt, 0.05, "mulch", 12, 6)
    drape(s1 + 0.3, tu0 - 0.3, lambda u: 2.2, lambda u: bt, 0.05, "mulch", 24, 6)
    # tiered planters stepping down to the driveway apron (155.0 -> 156.25 -> 157.45)
    box(tu1 - 0.3, tu1, 1.6, gn0, GZ - 0.3, TIERZ + 0.2, "whitebrick")
    box(tu1 - 0.33, tu1 + 0.03, 1.57, gn0, TIERZ + 0.2, TIERZ + 0.27, "limestone")
    box(tu0 - 0.3, tu0, 2.2, gn0, TIERZ - 0.3, PADZ + 0.2, "whitebrick")
    box(tu0 - 0.33, tu0 + 0.03, 2.2, gn0, PADZ + 0.2, PADZ + 0.27, "limestone")
    # low planted wall on the NE side of the apron
    box(eu1 + 0.05, eu1 + 0.35, 0.4, gn0, GZ - 0.3, GZ + 0.55, "whitebrick")
    box(eu1 + 0.02, eu1 + 0.38, 0.4, gn0, GZ + 0.55, GZ + 0.62, "limestone")
    # curving flagstone path (as in the client's photo) from the top of the steps to the portico
    P = [((s0 + s1) / 2, sn1), ((s0 + s1) / 2 + 0.6, sn1 + 1.9), (7.5, 4.3), (12.3, vn0 - 0.5)]

    def bez(t):
        return tuple((1 - t) ** 3 * P[0][i] + 3 * (1 - t) ** 2 * t * P[1][i] + 3 * (1 - t) * t * t * P[2][i] + t ** 3 * P[3][i] for i in (0, 1))
    PATH.clear()
    for k in range(26):
        t = k / 25
        (cu, cn), (du_, dn_) = bez(t), bez(min(t + 0.01, 1))
        if k == 25:
            (cu, cn), (du_, dn_) = bez(0.99), bez(1.0)
        ll = math.hypot(du_ - cu, dn_ - cn) or 1
        px, pn_ = -(dn_ - cn) / ll, (du_ - cu) / ll
        PATH.append((cu, cn, px, pn_))
        for off in (-0.38, 0.0, 0.38):
            a_r = rng.uniform(0, 3.14)
            w_, h_ = rr(0.16, 0.24), rr(0.14, 0.22)
            c_, s_ = math.cos(a_r), math.sin(a_r)
            x0, n0_ = cu + px * off + rr(-0.05, 0.05), cn + pn_ * off + rr(-0.05, 0.05)
            corners = [(x0 + c_ * dx - s_ * dn, n0_ + s_ * dx + c_ * dn, PADZ + 0.065) for dx, dn in
                       ((-w_, -h_), (w_ * 0.2, -h_ * 1.2), (w_, -h_ * 0.4), (w_ * 0.8, h_), (-w_ * 0.5, h_ * 1.1), (-w_ * 1.1, h_ * 0.2))]
            faces("flagstone", corners + [(x, y, z - 0.07) for x, y, z in corners],
                  [list(range(6))] + [(i, (i + 1) % 6, 6 + (i + 1) % 6, 6 + i) for i in range(6)])
    # garden path lights
    for k in (5, 12, 19):
        cu, cn, px, pn_ = PATH[k]
        x, n = cu + px * 0.8, cn + pn_ * 0.8
        cyl((x, n, PADZ), (x, n, PADZ + 0.45), 0.04, 0.04, "black", 6)
        box(x - 0.05, x + 0.05, n - 0.05, n + 0.05, PADZ + 0.35, PADZ + 0.43, "bulb")
        light("POINT", (x, n, PADZ + 0.3), 4, WARM, size=0.03, name="sconce")

    # ---- street: road, kerbs, footpaths, nature strip, driveway apron and side path
    R = ARC["r"]
    drape(-45, 68, lambda u: arc_n(u, R - 15.0), lambda u: arc_n(u, R - 4.0), 0.03, "asphalt", 150, 10)
    for r0, r1 in ((R - 4.0, R - 3.75), (R - 15.25, R - 15.0)):
        drape(-45, 68, lambda u, r0=r0: arc_n(u, r0), lambda u, r1=r1: arc_n(u, r1), 0.17, "kerb", 150, 1)
    drape(-45, 68, lambda u: arc_n(u, R - 16.7), lambda u: arc_n(u, R - 15.25), 0.1, "concrete", 150, 2)
    drape(-45, 68, lambda u: arc_n(u, R - 24.0), lambda u: arc_n(u, R - 16.7), 0.05, "lawn", 150, 4)
    drape(-45, gu0 + 0.2, lambda u: arc_n(u, R - 3.75), lambda u: arc_n(u, R - 2.3), 0.1, "concrete", 80, 2)
    drape(eu1 + 0.4, 68, lambda u: arc_n(u, R - 3.75), lambda u: arc_n(u, R - 2.3), 0.1, "concrete", 60, 2)
    drape(gu0 + 0.05, eu1 + 0.4, lambda u: arc_n(u, R - 3.75), lambda u: gn0, 0.08, "aggregate", 14, 14)
    faces("aggregate", [(eu1, gn0, GZ + 0.04), (ne_u(gn0), gn0, GZ + 0.04), (ne_u(gn1 - 0.3), gn1 - 0.3, GZ + 0.04), (eu1, gn1 - 0.3, GZ + 0.04)], [(0, 1, 2, 3)])
    drape(0.05, tu0 - 0.3, lambda u: arc_n(u, R - 2.3), lambda u: 1.58, 0.06, "mulch", 40, 4)
    drape(-45, -0.3, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 0.8, 0.04, "lawn", 30, 2)
    drape(26.0, 68, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 1.0, 0.04, "lawn", 30, 2)


PATH = []
EUC = []


def hip_house(c, a, L_, W_, z0, wall, pitch, wall_mat, roof_mat, o=0.5):
    """simple hip-roofed neighbour massing: centre c (u, n), long axis a, half-length L_, half-width W_"""
    ax, ay = a
    bx, by = -ay, ax
    P = lambda l, w, z: (c[0] + ax * l + bx * w, c[1] + ay * l + by * w, z)
    q = [(-L_, -W_), (L_, -W_), (L_, W_), (-L_, W_)]
    faces(wall_mat, [P(l, w, z0 - 0.5) for l, w in q] + [P(l, w, z0 + wall) for l, w in q],
          [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    ze = z0 + wall - o * math.tan(pitch)
    zr = z0 + wall + W_ * math.tan(pitch)
    E = [P(l + (o if l > 0 else -o), w + (o if w > 0 else -o), ze) for l, w in q]
    r0, r1 = P(-(L_ - W_), 0, zr), P(L_ - W_, 0, zr)
    faces(roof_mat, E + [r0, r1], [(0, 1, 5, 4), (2, 3, 4, 5), (1, 2, 5), (3, 0, 4)])


def build_context():
    """the two side fences (Colorbond, 1.8 m, stepped with the ground) and the neighbouring houses"""
    for (ua, na), (ub, nb), mat in (((0.0, 0.0), (-5.096, 32.496), "fence_sage"), ((22.903, 0.0), (29.767, 44.458), "fence_beige")):
        Lg = math.hypot(ub - ua, nb - na)
        tx, tn_ = (ub - ua) / Lg, (nb - na) / Lg
        ix, iy = (-tn_, tx) if inside_lot(ua + (ub - ua) / 2 - tn_ * 2, na + (nb - na) / 2 + tx * 2) else (tn_, -tx)
        d = 7.0 / tn_
        while d < Lg - 0.3:
            d1 = min(d + 2.35, Lg - 0.25)
            p0 = (ua + tx * d, na + tn_ * d)
            p1 = (ua + tx * d1, na + tn_ * d1)
            base = max(max(zg(p[0] + ix * 0.5, p[1] + iy * 0.5), zg(p[0] - ix * 0.6, p[1] - iy * 0.6)) for p in (p0, p1))
            faces(mat, [(p0[0], p0[1], base - 0.1), (p1[0], p1[1], base - 0.1), (p1[0], p1[1], base + 1.8), (p0[0], p0[1], base + 1.8)], [(0, 1, 2, 3)])
            faces(mat, [(p0[0] + ix * 0.04, p0[1] + iy * 0.04, base + 1.78), (p1[0] + ix * 0.04, p1[1] + iy * 0.04, base + 1.78),
                        (p1[0] - ix * 0.04, p1[1] - iy * 0.04, base + 1.78), (p0[0] - ix * 0.04, p0[1] - iy * 0.04, base + 1.78),
                        (p0[0] + ix * 0.04, p0[1] + iy * 0.04, base + 1.84), (p1[0] + ix * 0.04, p1[1] + iy * 0.04, base + 1.84),
                        (p1[0] - ix * 0.04, p1[1] - iy * 0.04, base + 1.84), (p0[0] - ix * 0.04, p0[1] - iy * 0.04, base + 1.84)],
                  [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)])
            box(p0[0] - 0.03, p0[0] + 0.03, p0[1] - 0.03, p0[1] + 0.03, base - 0.2, base + 1.84, mat)
            d = d1
    # NE neighbour: single-storey white render, terracotta hip roof, lower than our lot
    no = (0.9882, -0.1532)
    c = (22.903 + 0.1544 * 23 + no[0] * 8.1, 23 + no[1] * 8.1)
    hip_house(c, (0.1532, 0.9882), 9.0, 6.5, zg(*c), 2.7, math.radians(22), "render", "terracotta")
    c2 = (c[0] + 3.5, c[1] - 11.5)
    hip_house(c2, (0.1532, 0.9882), 3.2, 3.0, zg(*c), 2.7, math.radians(22), "render", "terracotta")
    # across the street: two-storey white render, dark grey hip roof
    hip_house((10.0, -31.0), (1.0, 0.0), 8.5, 6.0, zg(10.0, -31.0), 5.9, math.radians(22), "render", "roof_grey")


def near_house(u, n, pad=0.7):
    for k in ("west", "main", "east", "rear", "verandah", "verandah_side", "outdoor"):
        u0, u1, n0, n1 = L[k]
        if u0 - pad < u < u1 + pad and n0 - pad < n < n1 + pad:
            return True
    p = L["pool"]
    return p[0] - pad < u < p[1] + pad and p[2] - pad < n < p[3] + pad


def build_garden(eye):
    S = cr.S
    ex, ey = eye
    for gx, gy, r in S["trees"]:
        u, n = cr.to_frame(gx, gy)
        if rng.random() < 0.3 or inside_lot(u, n) or math.hypot(u - ex, n - ey) < 9:
            continue
        h = min(max(r * 2.8 + rr(2, 5), 7), 17)
        tree((u, n, cr.h_grid(gx, gy) - cr.FFL - 0.2), h, r, "euc", "bark", cards_per=14)
        EUC.append((u, n, h, r))
    dem = np.array(Image.open(D3 / S["dem_1km"]["file"]), dtype=float)
    e0, n1, cell = S["dem_1km"]["e0"], S["dem_1km"]["n1"], S["dem_1km"]["cell"]
    for r_ in range(0, dem.shape[0], 2):
        for c_ in range(0, dem.shape[1], 2):
            gx, gy = e0 + (c_ + 0.5) * cell - S["E0"] + rr(-3, 3), n1 - (r_ + 0.5) * cell - S["N0"] + rr(-3, 3)
            if 0 <= gx <= cr.NG and 0 <= gy <= cr.NG:
                continue
            u, n = cr.to_frame(gx, gy)
            if n < 30 or dem[r_, c_] < 165 or rng.random() > 0.35:
                continue
            tree((u, n, dem[r_, c_] - cr.FFL - 0.3), rr(8, 14), rr(2.0, 3.5), "euc", "bark", cards_per=10)
    # street tree in the SW nature strip framing the view
    tree((-6.5, -2.4, zg(-6.5, -2.4)), 10.5, 4.6, "hedge", "darkbark", stems=1, cards_per=30, crown_flat=0.45)
    s0, s1, sn0, sn1 = L["stair"]
    vu0, vu1, vn0, vn1 = L["verandah"]
    hyd = lambda: "hydrangea" if rng.random() < 0.55 else "hydrangea_lime"
    # hydrangea bed along the verandah, box balls at the front edge
    for x in np.arange(0.4, 14.4, 0.7):
        if 10.9 < x < 13.7:
            continue
        bush((x + rr(-0.1, 0.1), 6.8, PADZ + 0.42), rr(0.4, 0.5), hyd(), 20)
    for x in np.arange(0.6, 14.4, 1.4):
        if 10.6 < x < 14.0:
            continue
        bush((x, 6.45, PADZ + 0.3), 0.3, "box", 18)
    for x in (10.7, 13.9):
        bush((x, 6.5, PADZ + 0.4), 0.42, "box", 22)
    # box balls and hydrangeas edging the curving path
    for k in range(2, 24, 3):
        cu, cn, px, pn_ = PATH[k]
        for sg in (-1, 1):
            if k % 2 and sg > 0:
                continue
            x, n = cu + px * 0.95 * sg, cn + pn_ * 0.95 * sg
            if 4.9 < n < 6.2:
                bush((x, n, PADZ + 0.28), 0.27, "box" if k % 6 else "hydrangea", 16)
    # SW verandah leg: white hydrangeas along its outer edge
    for n in np.arange(8.0, 20.6, 0.8):
        u = -0.45
        if cr.to_frame and inside_lot(u - 0.3, n):
            bush((u, n, PADZ + 0.42), rr(0.38, 0.46), "hydrangea", 18)
    # the bank: two rows of hydrangea and box
    tu0 = L["tier"][0]
    for x in np.arange(0.6, tu0 - 0.3, 0.85):
        if s0 - 0.8 < x < s1 + 0.8:
            continue
        bush((x, 2.85, zg(x, 2.85) + 0.38), rr(0.34, 0.42), hyd(), 15)
        bush((x + 0.4, 3.85, zg(x + 0.4, 3.85) + 0.32), 0.3, "box", 14)
    # tiered planters by the apron
    for n in np.arange(2.6, 7.2, 0.7):
        bush((15.1, n, TIERZ + 0.35), 0.33, hyd(), 15)
        bush((14.0, n, PADZ + 0.4), 0.33, "box" if int(n * 10) % 2 else "hydrangea", 15)
    for n in np.arange(0.9, 7.2, 0.7):
        bush((22.95, n, GZ + 0.9), 0.3, "box", 14)
    # verge planting
    for _ in range(50):
        u = rr(0.2, 14.4)
        if s0 - 0.8 < u < s1 + 0.8:
            continue
        t = rr(0.2, 0.9)
        nn = arc_n(u, ARC["r"] - 2.3) * (1 - t) + 1.5 * t
        r = rr(0.25, 0.42)
        bush((u, nn, zg(u, nn) + r * 0.6), r, "shrub" if rng.random() < 0.6 else "box", 14)
    # boundary hedges (privacy, kept clear of the house); lower shrubs along the rear (bushfire buffer)
    lot = L["lot_frame"]
    edges = [((0.0, 0.0), (-5.096, 32.496), 21.5, 1.0, 1.0), ((22.903, 0.0), (29.767, 44.458), 15.5, 1.0, 1.0),
             ((-5.096, 32.496), (29.767, 44.458), -1, 0.9, 0.55)]
    for (ua, na), (ub, nb), nmin, inset, size in edges:
        Lg = math.hypot(ub - ua, nb - na)
        tx, tn_ = (ub - ua) / Lg, (nb - na) / Lg
        cands = [(-tn_, tx), (tn_, -tx)]
        mu_, mn_ = (ua + ub) / 2, (na + nb) / 2
        ix, iy = next(c for c in cands if inside_lot(mu_ + c[0] * 2, mn_ + c[1] * 2))
        for k in range(int(Lg / 0.75)):
            d = (k + 0.5) * 0.75
            u, n = ua + tx * d + ix * inset, na + tn_ * d + iy * inset
            if n < nmin or not inside_lot(u, n) or near_house(u, n, 0.5):
                continue
            g = zg(u, n)
            layers = (0.7, 1.5, 2.2) if size > 0.9 else (0.5, 1.0)
            for zz in layers:
                bush((u, n, g + zz), rr(0.6, 0.8) * size, "hedge", 14)
    # upper terrace: olives, hydrangeas along the wall top; olives and box in the courtyard
    for u in (2.0, 9.5, 17.5, 24.0):
        n = nt(u) + 1.7
        if inside_lot(u, n + 0.8):
            tree((u, n, UPZ), rr(3.6, 4.3), rr(1.6, 2.0), "olive", "darkbark", stems=3, cards_per=22, crown_flat=0.6)
    for u in np.arange(-2.5, 27.0, 0.9):
        if 13.0 < u < 15.8:
            continue
        n = nt(u) + 0.45
        if inside_lot(u, n + 0.4):
            bush((u, n, UPZ + 0.4), rr(0.36, 0.44), hyd(), 15)
    for u, n in ((8.9, 22.2), (9.0, 29.2), (20.5, 30.6), (20.4, 24.0)):
        cyl((u, n, PADZ), (u, n, PADZ + 0.55), 0.3, 0.36, "fstone", 10)
        bush((u, n, PADZ + 0.95), 0.42, "box", 22)
    # NE side lawn by the east wing and SW side garden
    for n in np.arange(26.5, 31.0, 1.0):
        bush((26.4 + (n - 26) * 0.15, n, PADZ + 0.4), 0.38, hyd(), 15)


GLB_TEX = {"clad": ("cladding.png", 0.25), "wbrick": ("wbrick.png", 0.5), "whitebrick": ("wbrick.png", 0.5),
           "fstone": ("fstone.png", 0.5), "paving": ("paving.png", 0.25), "coping": ("paving.png", 0.25),
           "lining_u": ("lining.png", 0.25), "fence_beige": ("fence_beige.png", 1.0), "fence_sage": ("fence_sage.png", 1.0), "lining_n": ("lining_v.png", 0.25), "carriage": ("carriage.png", 0.5),
           "limestone": ("stone.png", 0.35), "flagstone": ("stone.png", 0.6)}
GLB_COL = {"roof_x": ("#2b2d30", 0.42, 0.55), "roof_y": ("#2b2d30", 0.42, 0.55), "black": ("#141516", 0.4, 0.3),
           "timber": ("#7a5636", 0.6, 0), "deck": ("#8a6644", 0.55, 0), "door": ("#6b4a2c", 0.45, 0),
           "lawn": ("#5f7d38", 0.95, 0), "mulch": ("#4f3f31", 1.0, 0), "concrete": ("#bbb5aa", 0.9, 0),
           "kerb": ("#b6b0a5", 0.9, 0), "asphalt": ("#38393b", 0.9, 0), "aggregate": ("#a39b8e", 0.9, 0),
           "pooltile": ("#8fd0e3", 0.3, 0), "water": ("#3fa9cf", 0.05, 0), "bark": ("#a09684", 0.85, 0),
           "darkbark": ("#4d443b", 0.9, 0), "grass": ("#a8874a", 1.0, 0), "fire": ("#ff7a2a", 0.5, 0), "floor": ("#8a6646", 0.4, 0),
           "slab": ("#e8e2d6", 0.7, 0), "render": ("#ebe7e0", 0.85, 0), "terracotta": ("#a04c2e", 0.7, 0),
           "roof_grey": ("#46494c", 0.6, 0), "rug": ("#b9ab94", 1, 0), "glass": ("#0e1416", 0.05, 0.0), "bulb": ("#fff2d8", 0.5, 0)}
INTERIOR = {"fabric", "linen", "walnut", "rug", "art", "lamp"}
EXTERIOR_LIGHTS = ("lantern", "door_lantern", "garage_lamp", "verandah", "pool", "sconce")  # pool_fire included
LEAVES = ("euc", "olive", "shrub", "hedge", "hydrangea", "hydrangea_lime", "box", "tussock")


def export_glb(path):
    """the scene without interiors as GLB: textures on world-scale UVs, flat PBR colours elsewhere"""
    for ob in list(bpy.data.objects):
        mat = ob.data.materials[0].name if ob.type == "MESH" and ob.data.materials else ""
        if ob.type == "MESH" and (mat in INTERIOR or mat in ("euc", "bark") or (mat == "bulb" and len(ob.data.polygons) > 100)):
            bpy.data.objects.remove(ob)
        elif ob.type == "LIGHT" and not ob.name.startswith(EXTERIOR_LIGHTS):
            bpy.data.objects.remove(ob)
    for ob in bpy.data.objects:
        if ob.type != "MESH" or not ob.data.materials:
            continue
        name = ob.data.materials[0].name.split(".")[0]
        me = ob.data
        if not me.uv_layers:
            sc_ = GLB_TEX.get(name, (None, 0.25))[1]
            lay = me.uv_layers.new(name="UVMap")
            uv = [0.0] * (len(me.loops) * 2)
            for p in me.polygons:
                nx, ny, nz = (abs(c) for c in p.normal)
                for li in p.loop_indices:
                    co = me.vertices[me.loops[li].vertex_index].co
                    a_, b_ = (co.x, co.y) if nz >= nx and nz >= ny else ((co.y, co.z) if nx >= ny else (co.x, co.z))
                    uv[2 * li], uv[2 * li + 1] = a_ * sc_, b_ * sc_
            lay.data.foreach_set("uv", uv)
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        bsdf = m.node_tree.nodes["Principled BSDF"]
        nt_ = m.node_tree
        if name in GLB_TEX or name in LEAVES or name == "terrain":
            tex = nt_.nodes.new("ShaderNodeTexImage")
            file = GLB_TEX[name][0] if name in GLB_TEX else ("leaf_%s.png" % name if name != "terrain" else None)
            tex.image = bpy.data.images.load(str(TEX / file)) if file else bpy.data.images.load(str(D3 / "ortho_texture.jpg"))
            nt_.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
            if file and file.startswith("leaf_"):
                nt_.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
            bsdf.inputs["Roughness"].default_value = 0.8
        else:
            hexc, rough, metal = GLB_COL.get(name, ("#c8c4bc", 0.8, 0))
            bsdf.inputs["Base Color"].default_value = (*srgb(hexc), 1)
            bsdf.inputs["Roughness"].default_value = rough
            bsdf.inputs["Metallic"].default_value = metal
            if name in ("bulb", "fire"):
                bsdf.inputs["Emission Color"].default_value = (1.0, 0.75, 0.45, 1)
                bsdf.inputs["Emission Strength"].default_value = 4.0
        ob.data.materials[0] = m
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", export_lights=True, export_cameras=False,
                              export_yup=True, export_apply=True, export_image_format="JPEG", export_jpeg_quality=85)


VIEWS = {
    "street": dict(eye=(4.5, -12.5, 2.0), at=(12.0, 12.0, 2.9), lens=22),
    "courtyard": dict(eye=(18.0, 50.0, 9.0), at=(10.5, 24.5, 0.0), lens=24),
    "aerial": dict(eye=(48.0, -24.0, 42.0), at=(12.0, 22.0, 0.0), lens=30),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="street")
    ap.add_argument("--scale", type=int, default=50)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "designs" / "wraparound_street.png"))
    ap.add_argument("--exposure", type=float, default=0.9)
    ap.add_argument("--export-glb", default="")
    a = ap.parse_args()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    cr.make_textures()
    textures()
    MATS.update({
        "clad": img_mat("clad", "cladding.png", 0.25, 0.6, bump=0.25),
        "wbrick": img_mat("wbrick", "wbrick.png", 0.5, 0.82, bump=0.35),
        "whitebrick": img_mat("whitebrick", "wbrick.png", 0.5, 0.82, bump=0.35),
        "fstone": img_mat("fstone", "fstone.png", 0.5, 0.9, bump=0.9),
        "lining_u": img_mat("lining_u", "lining.png", 0.25, 0.55, bump=0.1),
        "lining_n": img_mat("lining_n", "lining_v.png", 0.25, 0.55, bump=0.1),
        "carriage": img_mat("carriage", "carriage.png", 0.5, 0.5, bump=0.3),
        "paving": img_mat("paving", "paving.png", 0.25, 0.8, bump=0.1),
        "coping": img_mat("coping", "paving.png", 0.25, 0.75),
        "roof_x": seam_roof("roof_x", "X"), "roof_y": seam_roof("roof_y", "Y"),
        "black": flat_mat("black", srgb("#141516"), 0.4, 0.3),
        "glass": thin_glass("glass"),
        "slab": flat_mat("slab", srgb("#d8d2c6"), 0.8),
        "floor": flat_mat("floor", srgb("#8a6646"), 0.35),
        "fabric": flat_mat("fabric", srgb("#d9cfbf"), 0.95),
        "linen": flat_mat("linen", srgb("#f1ece3"), 0.95),
        "walnut": flat_mat("walnut", srgb("#4a3424"), 0.45),
        "rug": flat_mat("rug", srgb("#b9ab94"), 1.0),
        "fence_beige": img_mat("fence_beige", "fence_beige.png", 1.0, 0.45, bump=0.3, metallic=0.2),
        "fence_sage": img_mat("fence_sage", "fence_sage.png", 1.0, 0.45, bump=0.3, metallic=0.2),
        "render": noisy_mat("render", "#e6e2da", "#f2efe9", 6, 0.85, 0.05),
        "terracotta": noisy_mat("terracotta", "#8e3f25", "#b25a36", 9, 0.7, 0.2),
        "roof_grey": noisy_mat("roof_grey", "#3e4144", "#55585b", 9, 0.6, 0.1),
        "lamp": flat_mat("lamp", (1, 1, 1), 0.5, emit=WARM, strength=14),
        "bulb": flat_mat("bulb", (1, 1, 1), 0.5, emit=WARM, strength=25),
        "lawn": noisy_mat("lawn", "#4f6e2f", "#7a9446", 30, 0.95, 0.3),
        "mulch": noisy_mat("mulch", "#3d2f24", "#6b5644", 40, 1.0, 0.4),
        "concrete": noisy_mat("concrete", "#aaa499", "#c9c3b8", 8, 0.9, 0.05),
        "kerb": flat_mat("kerb", srgb("#b6b0a5"), 0.9),
        "asphalt": noisy_mat("asphalt", "#2a2b2d", "#4a4a4b", 90, 0.9, 0.3),
        "aggregate": noisy_mat("aggregate", "#8f877a", "#c2b9aa", 70, 0.9, 0.25),
        "timber": noisy_mat("timber", "#6b4a30", "#8c6644", 3, 0.6, 0.1),
        "deck": noisy_mat("deck", "#7a5636", "#9a7350", 2, 0.55, 0.1),
        "door": noisy_mat("door", "#6a4526", "#8a5e36", 4, 0.45, 0.05),
        "pooltile": flat_mat("pooltile", srgb("#8fd0e3"), 0.3),
        "bark": noisy_mat("bark", "#8a8070", "#b0a796", 6, 0.85, 0.2),
        "darkbark": flat_mat("darkbark", srgb("#4d443b"), 0.9),
        "euc": leaf_mat("euc"), "olive": leaf_mat("olive"), "shrub": leaf_mat("shrub"), "hedge": leaf_mat("hedge"),
        "box": leaf_mat("box"), "hydrangea": leaf_mat("hydrangea"), "hydrangea_lime": leaf_mat("hydrangea_lime"),
        "limestone": img_mat("limestone", "stone.png", 0.35, 0.92, bump=0.7, tint=(1.07, 1.05, 1.0)),
        "flagstone": img_mat("flagstone", "stone.png", 0.6, 0.85, bump=0.4, tint=(0.92, 0.93, 0.95)),
        "fire": flat_mat("fire", (1, 0.4, 0.1), 0.5, emit=(1.0, 0.42, 0.12), strength=40),
        "terrain": terrain_mat("terrain", True), "grass": terrain_mat("grass", False),
    })
    w = MATS["water"] = bpy.data.materials.new("water")
    w.use_nodes = True
    bw = w.node_tree.nodes["Principled BSDF"]
    bw.inputs["Base Color"].default_value = (0.75, 0.95, 1.0, 1)
    bw.inputs["Transmission Weight"].default_value = 1.0
    bw.inputs["IOR"].default_value = 1.33
    bw.inputs["Roughness"].default_value = 0.03

    cr.build_terrain()
    build_house()
    view = VIEWS.get(a.view, VIEWS["street"])
    build_context()
    build_garden(view["eye"][:2] if not a.export_glb else (1e3, 1e3))
    cr.flush()
    if a.export_glb:
        export_glb(a.export_glb)
        print("exported", a.export_glb)
        return
    cr.VIEWS.update(VIEWS)
    cr.setup_world_and_camera(a.view, 6.0, 232.0, a.exposure)

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = a.samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.03
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 5
    sc.cycles.transparent_max_bounces = 16
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.resolution_x, sc.render.resolution_y = 1414, 913
    sc.render.resolution_percentage = a.scale
    sc.render.filepath = a.out
    sc.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
