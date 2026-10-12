"""Cycles render of option B, the modern farmhouse (privacy to the street, living to a north courtyard).

    python farmhouse_cycles.py --view street|courtyard|aerial --scale 50 --samples 48 --out x.png [--export-mesh]

Reuses terrain, street, vegetation helpers and materials from cycles_render.py. Run farmhouse_site.py ground
first so render_scene.json carries this design's earthworks. With --export-mesh it also writes
data/3d/farmhouse_mesh.json (house triangles + colours) for farmhouse_site.py finish.
"""
import argparse
import json
import math
import random
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFilter

import cycles_render as cr
from cycles_render import (ARC, GZ, PADZ, B, MATS, TEX, arc_n, box, card, cyl, drape, faces, poly, prism, ring, slab,
                           zg, light, bush, tree, WARM, srgb, img_mat, flat_mat, noisy_mat, seam_roof, thin_glass,
                           leaf_mat, terrain_mat)

ROOT = Path(__file__).resolve().parent.parent
D3 = ROOT / "data" / "3d"
LAY = json.loads((D3 / "house_massing.json").read_text())
L = LAY["layout"]
FRONT = LAY["front"]
PAD_OUTLINE = LAY["pad_outline"]
rng = random.Random(4)
rr = rng.uniform

EAVE_MAIN, PITCH_MAIN = 3.0, math.radians(40)
EAVE_WING, PITCH_WING = 5.6, math.radians(45)
EAVE_REAR, PITCH_REAR = 3.0, math.radians(35)
UPW = 3.2  # wing upper floor level
T = 0.25   # wall thickness


def textures():
    R = random.Random(11)
    # white board-and-batten: 1024 px = 4 m, battens every 0.4 m
    im = Image.new("RGB", (1024, 1024), (236, 234, 228))
    d = ImageDraw.Draw(im)
    for x in range(0, 1024, 102):
        d.rectangle([x, 0, x + 12, 1024], fill=(246, 245, 241))
        d.rectangle([x + 13, 0, x + 16, 1024], fill=(196, 194, 188))
    arr = np.array(im).astype(np.int16) + np.random.default_rng(3).integers(-4, 4, (1024, 1024))[..., None]
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(TEX / "cladding.png")
    # painted white brick, 1024 = 2 m
    im = Image.new("RGB", (1024, 1024), (200, 198, 192))
    d = ImageDraw.Draw(im)
    for row, y in enumerate(range(0, 1024, 39)):
        off = 0 if row % 2 == 0 else -59
        for x in range(off, 1024, 118):
            l = R.uniform(-8, 6)
            d.rectangle([x + 3, y + 3, x + 115, y + 36], fill=(int(240 + l), int(239 + l), int(234 + l)))
    Image.fromarray(np.array(im)).filter(ImageFilter.GaussianBlur(0.8)).save(TEX / "whitebrick.png")
    # hydrangea: dark green leaves with creamy-white mophead blooms
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
    for _ in range(9):
        cx, cy, r = R.uniform(110, 400), R.uniform(90, 330), R.uniform(36, 58)
        for _ in range(140):
            a, rd = R.uniform(0, 6.28), R.uniform(0, r)
            x, y = cx + math.cos(a) * rd, cy + math.sin(a) * rd
            l = R.uniform(-12, 8)
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(int(244 + l), int(242 + l), int(230 + l), 255))
    im.save(TEX / "leaf_hydrangea.png")


# ------------------------------------------------------------------ wall helpers with rectangular openings
def wall_u(u0, u1, nf, nb, z0, z1, openings, mat):
    """wall spanning u0..u1 between n=nf and n=nb; openings: (a, b, zb, zt) along u"""
    cuts = sorted({u0, u1, *[x for o in openings for x in o[:2]]})
    for a, b in zip(cuts, cuts[1:]):
        hits = [o for o in openings if o[0] <= a and b <= o[1]]
        if not hits:
            box(a, b, nf, nb, z0, z1, mat)
            continue
        o = hits[0]
        box(a, b, nf, nb, z0, o[2], mat)
        box(a, b, nf, nb, o[3], z1, mat)


def wall_n(n0, n1, uf, ub, z0, z1, openings, mat):
    """wall spanning n0..n1 between u=uf and u=ub; openings: (a, b, zb, zt) along n"""
    cuts = sorted({n0, n1, *[x for o in openings for x in o[:2]]})
    for a, b in zip(cuts, cuts[1:]):
        hits = [o for o in openings if o[0] <= a and b <= o[1]]
        if not hits:
            box(uf, ub, a, b, z0, z1, mat)
            continue
        o = hits[0]
        box(uf, ub, a, b, z0, o[2], mat)
        box(uf, ub, a, b, o[3], z1, mat)


def frame_rect(a, b, zb, zt, plane, axis, cols, rows, mat="black", w=0.06, depth=0.07):
    """black window frame + glazing bars in the plane n=plane (axis 'u') or u=plane (axis 'n')"""
    def bar(p0, p1, z0, z1):
        if axis == "u":
            box(min(p0, p1) - w / 2 if p0 == p1 else p0, max(p0, p1) + w / 2 if p0 == p1 else p1, plane - depth / 2, plane + depth / 2,
                z0 - (w / 2 if z0 == z1 else 0), z1 + (w / 2 if z0 == z1 else 0), mat)
        else:
            box(plane - depth / 2, plane + depth / 2, min(p0, p1) - w / 2 if p0 == p1 else p0, max(p0, p1) + w / 2 if p0 == p1 else p1,
                z0 - (w / 2 if z0 == z1 else 0), z1 + (w / 2 if z0 == z1 else 0), mat)
    for k in range(cols + 1):
        x = a + (b - a) * k / cols
        bar(x, x, zb, zt)
    for k in range(rows + 1):
        z = zb + (zt - zb) * k / rows
        bar(a, b, z, z)


def window_u(a, b, zb, zt, nf, nb, cols=2, rows=3, outward=-1, awning=False):
    """window in a wall facing -n (outward=-1, street side) or +n (outward=+1)"""
    face = nf if outward < 0 else nb
    glass_n = (nf + nb) / 2
    poly([(a, glass_n, zb), (b, glass_n, zb), (b, glass_n, zt), (a, glass_n, zt)], "glass")
    frame_rect(a, b, zb, zt, face + outward * 0.02, "u", cols, rows)
    box(a - 0.05, b + 0.05, face + outward * 0.06, face, zb - 0.06, zb, "black")  # sill
    if awning:
        out = face + outward * 1.0
        slab((a - 0.35, face, zt + 0.55), (b + 0.35, face, zt + 0.55), (b + 0.35, out, zt + 0.3), (a - 0.35, out, zt + 0.3), 0.06, "roof_x")
        for x in (a - 0.2, b + 0.2):
            cyl((x, face, zt + 0.1), (x, out + 0.1 * -outward, zt + 0.3), 0.04, 0.04, "timber")


def window_n(a, b, zb, zt, uf, ub, cols=2, rows=3, outward=1):
    face = ub if outward > 0 else uf
    glass_u = (uf + ub) / 2
    poly([(glass_u, a, zb), (glass_u, b, zb), (glass_u, b, zt), (glass_u, a, zt)], "glass")
    frame_rect(a, b, zb, zt, face + outward * 0.02, "n", cols, rows)


def gable_u(u0, u1, nf, nb, ze, pitch, mat, opening=None):
    """triangular gable wall in the u-z plane at n between nf..nb, ridge over the middle"""
    um = (u0 + u1) / 2
    zr = ze + (u1 - u0) / 2 * math.tan(pitch)
    if opening is None:
        prism([(u0, ze), (u1, ze), (um, zr)], nf, nb, mat)
    else:
        a, b, zb, zt = opening
        ring([(u0, ze), (u1, ze), (um, zr), (um, zr)], [(a, zb), (b, zb), (b, zt), (a, zt)], nf, nb, mat)
    return zr


def gable_n(n0, n1, uf, ub, ze, pitch, mat):
    nm = (n0 + n1) / 2
    zr = ze + (n1 - n0) / 2 * math.tan(pitch)
    faces(mat, [(uf, n0, ze), (uf, n1, ze), (uf, nm, zr), (ub, n0, ze), (ub, n1, ze), (ub, nm, zr)],
          [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)])
    return zr


def roof_ridge_u(u0, u1, n0, n1, ze, pitch, o=0.45, mat="roof_y"):
    """two planes, ridge parallel to u at the middle of n0..n1; seams run down-slope (vary along u)"""
    nm = (n0 + n1) / 2
    zr = ze + (n1 - n0) / 2 * math.tan(pitch)
    dz = o * math.tan(pitch)
    slab((u0 - o, n0 - o, ze - dz), (u1 + o, n0 - o, ze - dz), (u1 + o, nm, zr + 0.05), (u0 - o, nm, zr + 0.05), 0.1, "roof_x")
    slab((u1 + o, n1 + o, ze - dz), (u0 - o, n1 + o, ze - dz), (u0 - o, nm, zr + 0.05), (u1 + o, nm, zr + 0.05), 0.1, "roof_x")
    for u in (u0 - o, u1 + o):  # barge fascia
        slab((u, n0 - o, ze - dz - 0.12), (u, nm, zr - 0.07), (u, nm, zr + 0.2), (u, n0 - o, ze - dz + 0.15), 0.05, "black")
        slab((u, nm, zr - 0.07), (u, n1 + o, ze - dz - 0.12), (u, n1 + o, ze - dz + 0.15), (u, nm, zr + 0.2), 0.05, "black")
    for n in (n0 - o, n1 + o):  # gutters
        box(u0 - o, u1 + o, n - 0.06, n + 0.06, ze - dz - 0.15, ze - dz + 0.02, "black")
    return zr


def roof_ridge_n(n0, n1, u0, u1, ze, pitch, o=0.45):
    """two planes, ridge parallel to n; seams vary along n"""
    um = (u0 + u1) / 2
    zr = ze + (u1 - u0) / 2 * math.tan(pitch)
    dz = o * math.tan(pitch)
    slab((u0 - o, n1 + o, ze - dz), (u0 - o, n0 - o, ze - dz), (um, n0 - o, zr + 0.05), (um, n1 + o, zr + 0.05), 0.1, "roof_y")
    slab((u1 + o, n0 - o, ze - dz), (u1 + o, n1 + o, ze - dz), (um, n1 + o, zr + 0.05), (um, n0 - o, zr + 0.05), 0.1, "roof_y")
    for n in (n0 - o, n1 + o):
        slab((u0 - o, n, ze - dz - 0.12), (um, n, zr - 0.07), (um, n, zr + 0.2), (u0 - o, n, ze - dz + 0.15), 0.05, "black")
        slab((um, n, zr - 0.07), (u1 + o, n, ze - dz - 0.12), (u1 + o, n, ze - dz + 0.15), (um, n, zr + 0.2), 0.05, "black")
    for u in (u0 - o, u1 + o):
        box(u - 0.06, u + 0.06, n0 - o, n1 + o, ze - dz - 0.15, ze - dz + 0.02, "black")
    return zr


def lantern(x, n, z, size=0.32, power=25):
    s = size / 2
    box(x - s, x + s, n - s, n + s, z, z + 0.05, "black")
    box(x - s * 0.9, x + s * 0.9, n - s * 0.9, n + s * 0.9, z + size * 1.55, z + size * 1.62, "black")
    faces("black", [(x - s * 1.1, n - s * 1.1, z + size * 1.62), (x + s * 1.1, n - s * 1.1, z + size * 1.62), (x + s * 1.1, n + s * 1.1, z + size * 1.62),
                    (x - s * 1.1, n + s * 1.1, z + size * 1.62), (x, n, z + size * 2.1)], [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)])
    for dx, dn in ((-s, -s), (s, -s), (s, s), (-s, s)):
        box(x + dx - 0.015, x + dx + 0.015, n + dn - 0.015, n + dn + 0.015, z, z + size * 1.6, "black")
    poly([(x - s, n - s, z + 0.05), (x + s, n - s, z + 0.05), (x + s, n - s, z + size * 1.55), (x - s, n - s, z + size * 1.55)], "glass")
    cyl((x, n, z + 0.05), (x, n, z + size * 0.9), 0.03, 0.03, "lamp")
    light("POINT", (x, n, z + size * 0.75), power, WARM, size=0.04, name="lantern")


def room_light(u0, u1, n0, n1, z, power):
    light("AREA", ((u0 + u1) / 2, (n0 + n1) / 2, z - 0.05), power, WARM, aim=((u0 + u1) / 2, (n0 + n1) / 2, -5),
          size=min(u1 - u0, n1 - n0) * 0.6, name="room")


# ------------------------------------------------------------------ the farmhouse
def build_house():
    mu0, mu1, mn0, mn1 = L["main"]
    wu0, wu1, wn0, wn1 = L["wing"]
    ru0, ru1, rn0, rn1 = L["rear"]
    vu0, vu1, vn0, vn1 = L["verandah"]

    # floors (warm timber) under every body
    for u0, u1, n0, n1 in (L["main"], L["wing"], L["rear"]):
        box(u0, u1, n0, n1, -0.25, 0.0, "floor")

    # --- main body: street wall with a front door and two windows behind the verandah, small attic dormers
    door = (5.95, 7.65, 0.0, 2.4)
    w1, w2 = (2.1, 4.5, 0.75, 2.4), (8.8, 11.2, 0.75, 2.4)
    wall_u(mu0, mu1, mn0, mn0 + T, 0, EAVE_MAIN, [door, w1, w2], "clad")
    window_u(*w1, mn0, mn0 + T, cols=3, rows=2)
    window_u(*w2, mn0, mn0 + T, cols=3, rows=2)
    box(door[0], door[1], mn0 + 0.08, mn0 + 0.13, 0, 2.4, "door")
    poly([(door[0] + 0.45, mn0 + 0.06, 1.0), (door[1] - 0.45, mn0 + 0.06, 1.0), (door[1] - 0.45, mn0 + 0.06, 2.1), (door[0] + 0.45, mn0 + 0.06, 2.1)], "glass")
    for x in (door[0] - 0.35, door[1] + 0.35):  # wall lanterns either side of the door
        box(x - 0.09, x + 0.09, mn0 - 0.14, mn0, 1.75, 2.2, "black")
        box(x - 0.06, x + 0.06, mn0 - 0.13, mn0 - 0.02, 1.8, 2.12, "bulb")
        light("POINT", (x, mn0 - 0.25, 1.95), 18, WARM, size=0.05, name="door_lantern")
    # rear (north) wall onto the courtyard: full-height glazing between the rear wing and the gable wing
    big = (ru1 + 0.3, wu0 - 0.3, 0.0, 2.7)
    wall_u(mu0, mu1, mn1 - T, mn1, 0, EAVE_MAIN, [big], "clad")
    window_u(*big, mn1 - T, mn1, cols=4, rows=1, outward=1)
    wall_n(mn0, mn1, mu0, mu0 + T, 0, EAVE_MAIN, [], "clad")
    gable_u(mu0, mu1, mn0, mn0, EAVE_MAIN, 0, "clad") if False else None
    # gable end at the SW side (wall in the n-z plane) and roof
    gable_n(mn0, mn1, mu0, mu0 + T, EAVE_MAIN, PITCH_MAIN, "clad")
    zr_main = roof_ridge_u(mu0, wu0 - 0.1, mn0, mn1, EAVE_MAIN, PITCH_MAIN)
    # dormers with small windows (privacy) on the street slope
    for du in (3.3, 9.2):
        n_f = mn0 + 1.6
        z_b = EAVE_MAIN + 1.6 * math.tan(PITCH_MAIN) - 0.3
        wall_u(du - 1.0, du + 1.0, n_f, n_f + 0.15, z_b, z_b + 1.9, [(du - 0.5, du + 0.5, z_b + 0.5, z_b + 1.5)], "clad")
        window_u(du - 0.5, du + 0.5, z_b + 0.5, z_b + 1.5, n_f, n_f + 0.15, cols=2, rows=2)
        box(du - 1.0, du - 0.88, n_f, n_f + 2.6, z_b, z_b + 1.9, "clad")
        box(du + 0.88, du + 1.0, n_f, n_f + 2.6, z_b, z_b + 1.9, "clad")
        gable_u(du - 1.0, du + 1.0, n_f, n_f + 0.15, z_b + 1.9, math.radians(45), "clad")
        roof_ridge_n(n_f, n_f + 2.6, du - 1.0, du + 1.0, z_b + 1.9, math.radians(45), o=0.2)
        light("POINT", (du, n_f + 1.2, z_b + 1.3), 40, WARM, size=0.1, name="attic")
    room_light(mu0 + 0.5, wu0 - 0.5, mn0 + 0.5, mn1 - 0.5, EAVE_MAIN, 1200)
    # simple furniture glimpsed through the front windows
    box(2.3, 4.3, mn0 + 1.0, mn0 + 1.9, 0, 0.45, "fabric")
    box(9.0, 11.0, mn0 + 1.4, mn0 + 2.4, 0.72, 0.77, "walnut")
    cr_pendant(10.0, mn0 + 1.9, 2.0)
    cr_pendant(3.3, mn0 + 2.4, 2.3)

    # --- verandah: timber deck, posts with knee braces, metal skillion roof
    box(vu0, vu1, vn0, mn0, -0.25, 0.12, "deck")
    zr_v, zl_v = EAVE_MAIN - 0.15, 2.55
    for x in (vu0 + 0.15, 4.0, 9.6, vu1 - 0.15):
        box(x - 0.1, x + 0.1, vn0 + 0.1, vn0 + 0.3, 0.12, zl_v, "timber")
        for s in (-1, 1):
            if vu0 + 0.3 < x + s * 0.5 < vu1 - 0.3:
                cyl((x, vn0 + 0.2, zl_v - 0.55), (x + s * 0.55, vn0 + 0.2, zl_v - 0.02), 0.04, 0.04, "timber")
    box(vu0, vu1, vn0 + 0.1, vn0 + 0.3, zl_v, zl_v + 0.2, "timber")
    slab((vu0 - 0.2, vn0 - 0.25, zl_v + 0.2), (vu1 + 0.2, vn0 - 0.25, zl_v + 0.2), (vu1 + 0.2, mn0, zr_v + 0.05), (vu0 - 0.2, mn0, zr_v + 0.05), 0.08, "roof_x")
    box(vu0 - 0.2, vu1 + 0.2, vn0 - 0.31, vn0 - 0.19, zl_v + 0.05, zl_v + 0.22, "black")
    light("POINT", (6.8, vn0 + 1.4, zl_v - 0.3), 20, WARM, size=0.1, name="verandah")
    for x in (2.6, 4.1):  # outdoor chairs
        box(x - 0.35, x + 0.35, vn0 + 1.0, vn0 + 1.7, 0.12, 0.55, "black")
        box(x - 0.3, x + 0.3, vn0 + 1.05, vn0 + 1.65, 0.45, 0.6, "linen")
    for x in (door[0] - 0.75, door[1] + 0.75):  # potted plants by the door
        cyl((x, mn0 - 0.45, 0.12), (x, mn0 - 0.45, 0.6), 0.22, 0.26, "black")
        bush((x, mn0 - 0.45, 1.0), 0.38, "hedge", 16)

    # --- two-storey gable wing, gable to the street: one window per floor with an awning on the lower one
    wl = (17.4, 21.2, 0.9, 2.5)
    wu_ = (18.7, 19.9, UPW + 2.9, UPW + 4.3)
    wall_u(wu0, wu1, wn0, wn0 + T, 0, EAVE_WING, [wl], "clad")
    window_u(*wl, wn0, wn0 + T, cols=3, rows=2, awning=True)
    zr_w = gable_u(wu0, wu1, wn0, wn0 + T, EAVE_WING, PITCH_WING, "clad", opening=wu_)
    window_u(*wu_, wn0, wn0 + T, cols=2, rows=2)
    rl = (16.6, 22.0, 0.0, 2.7)
    wall_u(wu0, wu1, wn1 - T, wn1, 0, EAVE_WING, [rl, (17.9, 20.7, UPW + 0.9, UPW + 2.1)], "clad")
    window_u(*rl, wn1 - T, wn1, cols=3, rows=1, outward=1)
    window_u(17.9, 20.7, UPW + 0.9, UPW + 2.1, wn1 - T, wn1, cols=3, rows=1, outward=1)
    gable_u(wu0, wu1, wn1 - T, wn1, EAVE_WING, PITCH_WING, "clad")
    wall_n(wn0, wn1, wu1 - T, wu1, 0, EAVE_WING, [], "clad")
    wall_n(mn1, wn1, wu0, wu0 + T, 0, EAVE_WING, [], "clad")
    box(wu0 + T, wu1 - T, wn0 + T, wn1 - T, UPW - 0.25, UPW, "slab")
    roof_ridge_n(wn0, wn1, wu0, wu1, EAVE_WING, PITCH_WING)
    room_light(wu0 + 0.5, wu1 - 0.5, wn0 + 0.5, wn1 - 0.5, UPW - 0.25, 900)
    room_light(wu0 + 0.5, wu1 - 0.5, wn0 + 0.5, wn1 - 0.5, EAVE_WING + 1.0, 900)
    box(18.0, 20.6, wn0 + 1.2, wn0 + 2.2, 0.72, 0.77, "walnut")
    cr_pendant(19.3, wn0 + 1.7, 2.0)

    # --- rear living wing: big sliding doors east onto the courtyard, glazed gable north
    sl = (rn0 + 0.6, rn1 - 0.6, 0.0, 2.7)
    wall_n(rn0, rn1, ru1 - T, ru1, 0, EAVE_REAR, [sl], "clad")
    window_n(*sl, ru1 - T, ru1, cols=4, rows=1, outward=1)
    wall_n(rn0, rn1, ru0, ru0 + T, 0, EAVE_REAR, [(rn0 + 3, rn0 + 4.2, 1.2, 2.2)], "clad")
    window_n(rn0 + 3, rn0 + 4.2, 1.2, 2.2, ru0, ru0 + T, cols=1, rows=1, outward=-1)
    gw = (ru0 + 1.2, ru1 - 1.2, 0.0, 2.7)
    wall_u(ru0, ru1, rn1 - T, rn1, 0, EAVE_REAR, [gw], "clad")
    window_u(*gw, rn1 - T, rn1, cols=3, rows=1, outward=1)
    gable_u(ru0, ru1, rn1 - T, rn1, EAVE_REAR, PITCH_REAR, "clad", opening=(ru0 + 2.6, ru1 - 2.6, EAVE_REAR + 0.2, EAVE_REAR + 2.0))
    window_u(ru0 + 2.6, ru1 - 2.6, EAVE_REAR + 0.2, EAVE_REAR + 2.0, rn1 - T, rn1, cols=2, rows=1, outward=1)
    roof_ridge_n(rn0 - 0.2, rn1, ru0, ru1, EAVE_REAR, PITCH_REAR)
    room_light(ru0 + 0.5, ru1 - 0.5, rn0 + 0.5, rn1 - 0.5, EAVE_REAR, 1500)
    box(2.4, 6.0, rn0 + 4.0, rn0 + 5.0, 0, 0.45, "fabric")
    cr_pendant(5.0, rn0 + 7.5, 2.0)

    # --- chimney on the main roof
    box(12.8, 13.9, mn0 + 6.5, mn0 + 7.6, 0, zr_main + 1.2, "whitebrick")
    box(12.7, 14.0, mn0 + 6.4, mn0 + 7.7, zr_main + 1.2, zr_main + 1.32, "black")

    # --- garage (street level) in painted brick, timber-look door; lawn on its roof
    gu0, gu1, gn0, gn1 = L["garage"]
    ring([(gu0, GZ - 0.3), (gu1, GZ - 0.3), (gu1, PADZ + 0.05), (gu0, PADZ + 0.05)],
         [(gu0 + 0.7, GZ), (gu1 - 0.7, GZ), (gu1 - 0.7, GZ + 2.4), (gu0 + 0.7, GZ + 2.4)], gn0, gn0 + 0.35, "whitebrick")
    box(gu0 + 0.7, gu1 - 0.7, gn0 + 0.45, gn0 + 0.5, GZ, GZ + 2.4, "slats")
    box(gu0, gu0 + 0.3, gn0, gn1, GZ - 0.3, PADZ + 0.05, "whitebrick")
    box(gu1 - 0.3, gu1 + 0.3, gn0, gn1, GZ - 0.3, PADZ + 0.05, "whitebrick")
    box(gu0, gu1, gn0, gn1, PADZ - 0.3, PADZ + 0.06, "lawn")
    box(gu0 - 0.03, gu1 + 0.03, gn0 - 0.03, gn0 + 0.38, PADZ + 0.05, PADZ + 0.13, "whitebrick")
    for x in (gu0 + 0.35, gu1 - 0.35):
        box(x - 0.08, x + 0.08, gn0 - 0.14, gn0, GZ + 1.9, GZ + 2.3, "black")
        light("POINT", (x, gn0 - 0.25, GZ + 2.1), 15, WARM, size=0.05, name="garage_lamp")

    # --- street wall: low painted brick retaining wall, brick pillars with lanterns, black X gate, steps
    s0, s1, sn0, sn1 = L["stair"]
    topw = FRONT["wall_top"]
    for a, b in ((0.3, s0 - 0.45), (s1 + 0.45, gu0)):
        box(a, b, 1.6, 2.2, -3.4, topw, "whitebrick")
        box(a - 0.03, b + 0.03, 1.55, 2.25, topw, topw + 0.08, "whitebrick")
    gg = zg((s0 + s1) / 2, 1.2)
    for x in (s0 - 0.45, s1 + 0.45):  # pillars
        box(x - 0.45, x + 0.45, 1.45, 2.35, gg - 0.3, topw + 0.5, "whitebrick")
        box(x - 0.52, x + 0.52, 1.38, 2.42, topw + 0.5, topw + 0.62, "whitebrick")
        lantern(x, 1.9, topw + 0.62, 0.36, 30)
    # X-braced timber gate (painted black) across the bottom of the steps
    gz0, gz1 = gg + 0.05, gg + 1.25
    for x0, x1 in ((s0, s0 + 0.08), (s1 - 0.08, s1)):
        box(x0, x1, 1.86, 1.94, gz0, gz1, "black")
    box(s0, s1, 1.86, 1.94, gz1 - 0.08, gz1, "black")
    box(s0, s1, 1.86, 1.94, gz0, gz0 + 0.08, "black")
    for (xa, za), (xb, zb_) in (((s0, gz0), (s1, gz1)), ((s0, gz1), (s1, gz0))):
        L_ = math.hypot(xb - xa, zb_ - za)
        tx, tz = (xb - xa) / L_, (zb_ - za) / L_
        slab((xa - tz * 0.05, 1.86, za + tx * 0.05), (xb - tz * 0.05, 1.86, zb_ + tx * 0.05), (xb + tz * 0.05, 1.86, zb_ - tx * 0.05), (xa + tz * 0.05, 1.86, za - tx * 0.05), 0.08, "black")
    # steps from the gate up to lawn level, with brick cheek walls
    n_steps = 10
    z_lo, z_hi = gg, PADZ
    for k in range(n_steps):
        z = z_lo + (z_hi - z_lo) * (k + 1) / n_steps
        n_a = 2.0 + k * (sn1 - 2.0) / n_steps
        box(s0, s1, n_a, sn1, z_lo - 0.4, z, "paving")
    for x0, x1 in ((s0 - 0.3, s0), (s1, s1 + 0.3)):
        box(x0, x1, 2.2, sn1, z_lo - 0.4, PADZ + 0.35, "whitebrick")
    # stepping-stone path across the lawn to the verandah
    n = sn1 + 0.2
    while n < vn0 - 0.2:
        box(s0 + 0.1, s1 - 0.1, n, n + 0.55, PADZ + 0.02, PADZ + 0.07, "paving")
        n += 0.75
    # planted bank from the low wall up to the lawn, then the front lawn
    bt = FRONT["bank_top"]
    drape(0.3, s0 - 0.3, lambda u: 2.2, lambda u: bt, 0.05, "mulch", 24, 6)
    drape(s1 + 0.3, gu0, lambda u: 2.2, lambda u: bt, 0.05, "mulch", 24, 6)
    box(0.3, gu0, bt, vn0, PADZ - 0.3, PADZ + 0.03, "lawn")
    box(vu1, gu0, vn0, mn0, PADZ - 0.3, PADZ + 0.03, "lawn")

    # --- north courtyard: paving, pool, outdoor dining; lawn on the rest of the pad
    pu0, pu1, pn0, pn1 = L["pool"]
    cu0, cu1, cn0, cn1 = ru1, 27.5, mn1, L["pad_back"]
    for a, c, n0_, n1_ in ((cu0, cu1, cn0, pn0 - 0.3), (cu0, cu1, pn1 + 0.3, cn1), (cu0, pu0 - 0.3, pn0 - 0.3, pn1 + 0.3), (pu1 + 0.3, cu1, pn0 - 0.3, pn1 + 0.3)):
        box(a, c, n0_, n1_, PADZ - 0.3, PADZ + 0.03, "paving")
    for a, c, n0_, n1_ in ((pu0 - 0.3, pu1 + 0.3, pn0 - 0.3, pn0), (pu0 - 0.3, pu1 + 0.3, pn1, pn1 + 0.3), (pu0 - 0.3, pu0, pn0, pn1), (pu1, pu1 + 0.3, pn0, pn1)):
        box(a, c, n0_, n1_, PADZ - 0.2, PADZ + 0.06, "coping")
    box(pu0, pu1, pn0, pn1, PADZ - 1.7, PADZ - 1.6, "pooltile")
    for a, c, n0_, n1_ in ((pu0, pu1, pn0, pn0 + 0.02), (pu0, pu1, pn1 - 0.02, pn1), (pu0, pu0 + 0.02, pn0, pn1), (pu1 - 0.02, pu1, pn0, pn1)):
        box(a, c, n0_, n1_, PADZ - 1.7, PADZ - 0.1, "pooltile")
    poly([(pu0, pn0, PADZ - 0.12), (pu1, pn0, PADZ - 0.12), (pu1, pn1, PADZ - 0.12), (pu0, pn1, PADZ - 0.12)], "water")
    for x in (pu0 + 2.5, pu1 - 2.5):
        light("POINT", (x, pn0 + 0.15, PADZ - 0.9), 60, (0.55, 0.85, 1.0), size=0.05, name="pool")
    box(11.0, 13.4, 23.0, 24.0, PADZ + 0.72, PADZ + 0.77, "walnut")
    for x in (11.4, 12.2, 13.0):
        for nn in (22.6, 24.4):
            box(x - 0.2, x + 0.2, nn - 0.2, nn + 0.2, PADZ, PADZ + 0.45, "black")
    box(0.4, ru0 - 0.1, rn1, cn1, PADZ - 0.3, PADZ + 0.03, "lawn")
    box(0.4, ru1, rn1, cn1, PADZ - 0.3, PADZ + 0.03, "lawn")

    # --- retaining walls along the pad edge (cut at the rear, fill at the sides), stepping to the ground
    pts = PAD_OUTLINE
    for (ua, na), (ub, nb) in zip(pts, pts[1:]):
        if max(na, nb) < 2.6:
            continue  # the street wall handles the front edge
        if min(ua, ub) > L["garage"][0] - 0.5 and max(na, nb) < L["garage"][3] + 0.6:
            continue  # garage side walls already retain this edge
        Lg = math.hypot(ub - ua, nb - na)
        steps = max(1, int(Lg / 2))
        for k in range(steps):
            t0, t1 = k / steps, (k + 1) / steps
            u0_, n0_ = ua + (ub - ua) * t0, na + (nb - na) * t0
            u1_, n1_ = ua + (ub - ua) * t1, na + (nb - na) * t1
            g = zg((u0_ + u1_) / 2, (n0_ + n1_) / 2)
            top = max(PADZ + 0.9, g + 0.25) if g > PADZ else PADZ + 0.9
            bot = min(g, PADZ) - 0.4
            dx, dn = u1_ - u0_, n1_ - n0_
            ln = math.hypot(dx, dn) or 1
            ox, on = -dn / ln * 0.3, dx / ln * 0.3
            faces("whitebrick", [(u0_, n0_, bot), (u1_, n1_, bot), (u1_ + ox, n1_ + on, bot), (u0_ + ox, n0_ + on, bot),
                                 (u0_, n0_, top), (u1_, n1_, top), (u1_ + ox, n1_ + on, top), (u0_ + ox, n0_ + on, top)],
                  [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])

    # --- street: road, kerbs, footpaths, nature strips, verge garden, driveway
    R = ARC["r"]
    drape(-45, 68, lambda u: arc_n(u, R - 15.0), lambda u: arc_n(u, R - 4.0), 0.03, "asphalt", 150, 10)
    for r0, r1 in ((R - 4.0, R - 3.75), (R - 15.25, R - 15.0)):
        drape(-45, 68, lambda u, r0=r0: arc_n(u, r0), lambda u, r1=r1: arc_n(u, r1), 0.17, "kerb", 150, 1)
    drape(-45, 68, lambda u: arc_n(u, R - 16.7), lambda u: arc_n(u, R - 15.25), 0.1, "concrete", 150, 2)
    drape(-45, 68, lambda u: arc_n(u, R - 24.0), lambda u: arc_n(u, R - 16.7), 0.05, "lawn", 150, 4)
    drape(-45, gu0 + 0.2, lambda u: arc_n(u, R - 3.75), lambda u: arc_n(u, R - 2.3), 0.1, "concrete", 80, 2)
    drape(gu1 - 0.2, 68, lambda u: arc_n(u, R - 3.75), lambda u: arc_n(u, R - 2.3), 0.1, "concrete", 60, 2)
    drape(gu0 + 0.2, gu1 - 0.2, lambda u: arc_n(u, R - 3.75), lambda u: gn0, 0.08, "aggregate", 12, 12)
    drape(-1.5, gu0, lambda u: arc_n(u, R - 2.3), lambda u: 1.58, 0.06, "mulch", 40, 4)
    drape(-45, -1.5, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 1.0, 0.04, "lawn", 30, 2)
    drape(gu1 + 0.3, 68, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 1.0, 0.04, "lawn", 30, 2)


def cr_pendant(x, n, z, strength=60):
    cyl((x, n, z + 0.15), (x, n, z + 1.2), 0.006, 0.006, "black")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.14, location=(x, n, z))
    bpy.context.active_object.data.materials.append(MATS["bulb"])
    light("POINT", (x, n, z - 0.2), strength, WARM, size=0.1, name="pendant")


EUC = []  # (u, n, z, height, radius) for the web viewer, which grows these itself


def build_garden():
    """eucalypts on the hill, a big shade tree, hydrangea and box borders, hedges on the side boundaries"""
    S = cr.S
    for gx, gy, r in S["trees"]:
        if rng.random() < 0.3:
            continue
        u, n = cr.to_frame(gx, gy)
        h = min(max(r * 2.8 + rr(2, 5), 7), 17)
        tree((u, n, cr.h_grid(gx, gy) - cr.FFL - 0.2), h, r, "euc", "bark", cards_per=14)
        EUC.append((u, n, cr.h_grid(gx, gy) - cr.FFL - 0.2, h, r))
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
            h_, r__ = rr(8, 14), rr(2.0, 3.5)
            tree((u, n, dem[r_, c_] - cr.FFL - 0.3), h_, r__, "euc", "bark", cards_per=10)
            EUC.append((u, n, dem[r_, c_] - cr.FFL - 0.3, h_, r__))
    # big shade tree in the front SW corner, as in the reference
    tree((-0.6, 4.0, PADZ), 11.5, 5.5, "hedge", "darkbark", stems=1, cards_per=34, crown_flat=0.45)
    # borders: box balls and white hydrangeas along the path and the verandah, hydrangeas behind the street wall
    s0, s1, sn0, sn1 = L["stair"]
    vn0 = L["verandah"][2]
    n = sn1 + 0.3
    while n < vn0 - 0.4:
        for x in (s0 - 0.55, s1 + 0.55):
            bush((x + rr(-0.1, 0.1), n, PADZ + 0.35), rr(0.32, 0.42), "hydrangea", 16)
        for x in (s0 - 1.25, s1 + 1.25):
            bush((x, n + 0.3, PADZ + 0.35), 0.36, "hedge", 18)
        n += 0.8
    for x in np.arange(1.4, 11.2, 0.9):
        if not (s0 - 0.6 < x < s1 + 0.6):
            bush((x, vn0 - 0.45, PADZ + 0.4), rr(0.35, 0.45), "hydrangea", 16)
    for x in np.arange(0.9, 15.6, 0.8):
        for nn in (2.8, 3.8):
            if not (s0 - 0.8 < x < s1 + 0.8):
                xx = x + (0.4 if nn > 3 else 0)
                bush((xx, nn, zg(xx, nn) + 0.38), rr(0.34, 0.44), "hydrangea" if nn < 3 or rng.random() < 0.5 else "hedge", 14)
    # verge garden: low grasses and box balls along the street wall
    for _ in range(60):
        u = rr(-1.2, 15.8)
        if s0 - 0.8 < u < s1 + 0.8:
            continue
        t = rr(0.2, 0.9)
        nn = arc_n(u, ARC["r"] - 2.3) * (1 - t) + 1.5 * t
        r = rr(0.25, 0.45)
        bush((u, nn, zg(u, nn) + r * 0.6), r, "shrub" if rng.random() < 0.6 else "hedge", 16)
    for u in (s0 - 1.4, s1 + 1.4):  # standards either side of the gate
        bush((u, 1.0, zg(u, 1.0) + 0.6), 0.45, "hedge", 20)
    # tall hedges on both side boundaries for privacy
    for (ua, na), (ub, nb), inset in (((0.0, 0.0), (-5.1, 32.5), 0.9), ((22.9, 0.0), (29.8, 44.5), -0.9)):
        for k in range(46):
            t = 0.08 + k / 46 * 0.8
            u, nn = ua + (ub - ua) * t + inset, na + (nb - na) * t
            for zz in (0.8, 1.7, 2.5):
                bush((u, nn, zg(u, nn) + zz), rr(0.75, 0.95), "hedge", 18)
    # courtyard: olives and planting
    for u, nn in ((24.5, 26.0), (10.5, 34.5), (25.5, 34.0)):
        tree((u, nn, PADZ), rr(3.4, 4.0), rr(1.6, 1.9), "olive", "darkbark", stems=3, cards_per=22, crown_flat=0.6)


GLB_TEX = {"clad": ("cladding.png", 0.25), "whitebrick": ("whitebrick.png", 0.5), "paving": ("paving.png", 0.25),
           "coping": ("paving.png", 0.25), "slats": ("slats.png", 0.25)}
GLB_COL = {"roof_x": ("#2b2d30", 0.42, 0.55), "roof_y": ("#2b2d30", 0.42, 0.55), "black": ("#141516", 0.4, 0.3),
           "timber": ("#7a5636", 0.6, 0), "deck": ("#8a6644", 0.55, 0), "door": ("#6b4a2c", 0.45, 0),
           "lawn": ("#5f7d38", 0.95, 0), "mulch": ("#4f3f31", 1.0, 0), "concrete": ("#bbb5aa", 0.9, 0),
           "kerb": ("#b6b0a5", 0.9, 0), "asphalt": ("#38393b", 0.9, 0), "aggregate": ("#a39b8e", 0.9, 0),
           "pooltile": ("#8fd0e3", 0.3, 0), "water": ("#3fa9cf", 0.05, 0), "bark": ("#a09684", 0.85, 0),
           "darkbark": ("#4d443b", 0.9, 0), "grass": ("#a8874a", 1.0, 0), "floor": ("#8a6646", 0.4, 0),
           "slab": ("#e8e2d6", 0.7, 0), "glass": ("#0e1416", 0.05, 0.0), "bulb": ("#fff2d8", 0.5, 0)}
INTERIOR = {"fabric", "linen", "walnut", "rug", "art", "lamp"}
EXTERIOR_LIGHTS = ("lantern", "door_lantern", "garage_lamp", "verandah", "pool")


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
        if not me.uv_layers:  # world-space box projection, 1 UV unit = 4 m unless the texture says otherwise
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
        nt = m.node_tree
        if name in GLB_TEX or name in ("euc", "olive", "shrub", "hedge", "hydrangea", "tussock") or name == "terrain":
            tex = nt.nodes.new("ShaderNodeTexImage")
            file = GLB_TEX[name][0] if name in GLB_TEX else ("leaf_%s.png" % name if name != "terrain" else None)
            tex.image = bpy.data.images.load(str(TEX / file)) if file else bpy.data.images.load(str(D3 / "ortho_texture.jpg"))
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
            if file and file.startswith("leaf_"):
                nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
            bsdf.inputs["Roughness"].default_value = 0.8
        else:
            hexc, rough, metal = GLB_COL.get(name, ("#c8c4bc", 0.8, 0))
            bsdf.inputs["Base Color"].default_value = (*srgb(hexc), 1)
            bsdf.inputs["Roughness"].default_value = rough
            bsdf.inputs["Metallic"].default_value = metal
            if name == "bulb":
                bsdf.inputs["Emission Color"].default_value = (1.0, 0.75, 0.45, 1)
                bsdf.inputs["Emission Strength"].default_value = 4.0
        ob.data.materials[0] = m
    (D3 / "farmhouse_trees.json").write_text(json.dumps([[round(x, 2) for x in t] for t in EUC]))
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", export_lights=True, export_cameras=False,
                              export_yup=True, export_apply=True, export_image_format="JPEG", export_jpeg_quality=85)


def export_mesh(path):
    """house geometry (everything except terrain, street, plants, trees) as triangles with base colours"""
    skip = {"terrain", "backdrop", "asphalt", "kerb", "concrete", "aggregate", "mulch", "euc", "olive", "shrub",
            "hedge", "hydrangea", "tussock", "bark", "darkbark", "glass", "bulb", "lamp"}
    colours = {"clad": (238, 236, 230), "whitebrick": (236, 234, 228), "roof_x": (45, 47, 50), "roof_y": (45, 47, 50),
               "black": (25, 25, 26), "timber": (122, 90, 62), "deck": (130, 98, 70), "door": (110, 78, 50),
               "paving": (212, 205, 190), "coping": (225, 220, 208), "lawn": (95, 122, 58), "water": (64, 172, 212),
               "pooltile": (143, 208, 227), "slats": (70, 52, 38), "floor": (138, 102, 70), "slab": (232, 226, 214)}
    pos, fcs, col = [], [], []
    for mat, b in B.items():
        if mat in skip or not b.f:
            continue
        c = colours.get(mat, (200, 196, 188))
        o = len(pos)
        pos.extend(b.v)
        for f in b.f:
            for k in range(1, len(f) - 1):
                fcs.append((o + f[0], o + f[k], o + f[k + 1]))
                col.append(c)
    Path(path).write_text(json.dumps({"positions": np.round(np.array(pos), 3).ravel().tolist(), "faces": np.array(fcs).ravel().tolist(),
                                      "colors": np.array(col).ravel().tolist(), "pool_rgb": [64, 172, 212]}, separators=(",", ":")))


VIEWS = {
    "street": dict(eye=(6.8, -12.0, 1.8), at=(8.6, 14.0, 3.0), lens=26),
    "street_wide": dict(eye=(9.5, -26.0, 3.2), at=(11.5, 16.0, 4.0), lens=30),
    "courtyard": dict(eye=(22.0, 41.0, 6.5), at=(12.0, 24.0, 1.5), lens=28),
    "aerial": dict(eye=(48.0, -24.0, 42.0), at=(12.0, 20.0, 0.0), lens=30),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="street")
    ap.add_argument("--scale", type=int, default=50)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--out", default=str(ROOT / "farmhouse_street.png"))
    ap.add_argument("--exposure", type=float, default=0.9)
    ap.add_argument("--export-mesh", action="store_true")
    ap.add_argument("--export-glb", default="")
    a = ap.parse_args()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    cr.make_textures()
    textures()
    MATS.update({
        "clad": img_mat("clad", "cladding.png", 0.25, 0.6, bump=0.25),
        "whitebrick": img_mat("whitebrick", "whitebrick.png", 0.5, 0.75, bump=0.25),
        "paving": img_mat("paving", "paving.png", 0.25, 0.8, bump=0.1),
        "coping": img_mat("coping", "paving.png", 0.25, 0.75),
        "slats": img_mat("slats", "slats.png", 0.25, 0.65, bump=0.2),
        "roof_x": seam_roof("roof_x", "X"), "roof_y": seam_roof("roof_y", "Y"),
        "black": flat_mat("black", srgb("#141516"), 0.4, 0.3),
        "glass": thin_glass("glass"),
        "slab": flat_mat("slab", srgb("#e8e2d6"), 0.7),
        "floor": flat_mat("floor", srgb("#8a6646"), 0.35),
        "fabric": flat_mat("fabric", srgb("#d9cfbf"), 0.95),
        "linen": flat_mat("linen", srgb("#f1ece3"), 0.95),
        "walnut": flat_mat("walnut", srgb("#4a3424"), 0.45),
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
        "door": noisy_mat("door", "#5e3f25", "#7d5634", 4, 0.45, 0.05),
        "pooltile": flat_mat("pooltile", srgb("#8fd0e3"), 0.3),
        "bark": noisy_mat("bark", "#8a8070", "#b0a796", 6, 0.85, 0.2),
        "darkbark": flat_mat("darkbark", srgb("#4d443b"), 0.9),
        "euc": leaf_mat("euc"), "olive": leaf_mat("olive"), "shrub": leaf_mat("shrub"), "hedge": leaf_mat("hedge"),
        "hydrangea": leaf_mat("hydrangea"),
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
    if a.export_mesh:
        export_mesh(D3 / "farmhouse_mesh.json")
    build_garden()
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
