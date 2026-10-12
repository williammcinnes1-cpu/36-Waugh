"""Cycles render of "Stone and Glass Gables": two steep sawn-sandstone gables joined by a two-storey glass link.

    python stone_glass_cycles.py --view street|courtyard|aerial --scale 50 --samples 32 --out x.png
    python stone_glass_cycles.py --export-glb data/3d/designs/stone_glass.glb

Street side: solid coursed-ashlar gables with slot windows and one modest arched feature window, a black glass
sectional garage door at street level. North side: full-height black-framed glazing (one tall arched gable), a
glass link with a balcony, the pool courtyard and an upper garden terrace out to the rear boundary.

Run scripts/designs/stone_glass_site.py ground and scripts/build_3d_render.py first.
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
from cycles_render import (ARC, GZ, PADZ, B, MATS, TEX, arc_n, box, cyl, drape, faces, poly, prism, ring, slab,  # noqa: E402
                           zg, light, bush, tree, WARM, srgb, img_mat, flat_mat, noisy_mat, seam_roof, thin_glass,
                           leaf_mat, terrain_mat)
import farmhouse_cycles as fc  # noqa: E402  (shared wall/window/roof helpers and the GLB export)
from farmhouse_cycles import wall_n, wall_u, window_u, window_n, frame_rect, lantern, cr_pendant  # noqa: E402

D3 = ROOT / "data" / "3d"
LAY = json.loads((D3 / "house_massing.json").read_text())
assert LAY.get("design") == "stone_glass", "run scripts/designs/stone_glass_site.py ground first"
L = LAY["layout"]
FRONT = LAY["front"]
PAD_OUTLINE = LAY["pad_outline"]
LOT_INNER = LAY["lot_inner"]
UPZ = FRONT["upper_terrace"]
rng = random.Random(624)
rr = rng.uniform

EAVE, PITCH = 6.6, math.radians(48)
UP = 3.2       # upper floor level
T = 0.35       # stone wall thickness
LINK_TOP = 5.9


def terrace_n(u):
    u0, n0, k = L["terrace_line"]
    return n0 + k * (u - u0)


# ------------------------------------------------------------------ textures
def textures():
    R = random.Random(624)
    # sawn sandstone in coursed ashlar: 2048 px = 4 m; courses alternate 0.30 / 0.20 m, blocks 0.5-1.1 m
    im = Image.new("RGB", (2048, 2048), (150, 132, 104))
    d = ImageDraw.Draw(im)
    y, k = 0, 0
    while y < 2048:
        ch = int((0.30 if k % 3 != 2 else 0.20) * 512)
        x = -R.uniform(0, 400)
        while x < 2048:
            bw = R.uniform(0.5, 1.1) * 512
            l, w = R.uniform(-18, 12), R.uniform(-9, 9)
            base = (int(222 + l + w), int(196 + l), int(150 + l - w * 1.4))
            d.rectangle([x + 3, y + 3, x + bw - 3, y + ch - 3], fill=base)
            for _ in range(int(bw / 12)):  # sawn bedding streaks
                yy = y + R.uniform(6, ch - 6)
                t = R.uniform(-10, 8)
                d.line([(x + 4, yy), (x + bw - 4, yy + R.uniform(-3, 3))], fill=(base[0] + int(t), base[1] + int(t), base[2] + int(t)), width=R.choice((1, 2, 3)))
            for _ in range(8):
                cx, cy = x + R.uniform(0, bw), y + R.uniform(0, ch)
                rw, rh = R.uniform(10, 60), R.uniform(4, 14)
                t = R.uniform(-16, 10)
                d.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=(base[0] + int(t), base[1] + int(t * 0.9), base[2] + int(t * 0.8)))
            x += bw
        y += ch
        k += 1
    arr = np.array(im).astype(np.int16)
    arr += np.random.default_rng(5).integers(-9, 9, arr.shape[:2])[..., None]
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)).save(TEX / "ashlar.png")
    # warm spotted-gum lining boards: 1024 px = 4 m, 0.135 m boards running along the image x axis
    im = Image.new("RGB", (1024, 1024), (90, 58, 34))
    d = ImageDraw.Draw(im)
    for yy in range(0, 1024, 35):
        l = R.uniform(-14, 14)
        base = (int(168 + l), int(112 + l * 0.8), int(66 + l * 0.6))
        d.rectangle([0, yy + 2, 1024, yy + 33], fill=base)
        for _ in range(14):
            x0 = R.uniform(0, 1024)
            t = R.uniform(-18, 10)
            d.line([(x0, yy + R.uniform(4, 30)), (x0 + R.uniform(60, 260), yy + R.uniform(4, 30))],
                   fill=(base[0] + int(t), base[1] + int(t), base[2] + int(t)), width=2)
    Image.fromarray(np.array(im)).filter(ImageFilter.GaussianBlur(0.6)).save(TEX / "boards.png")
    # Colorbond corrugated fence sheet: 512 px = 4 m, 76 mm pitch, shaded as a sine profile
    x = np.arange(512)
    shade = 0.84 + 0.16 * np.sin(2 * np.pi * x / (512 * 0.076 / 4))
    for name, rgb in (("fence_ne", (217, 207, 182)), ("fence_sw", (126, 136, 112))):   # beige / Pale Eucalypt
        arr = np.clip(np.array(rgb)[None, None, :] * shade[None, :, None] * np.ones((512, 1, 1)), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(TEX / f"{name}.png")


# ------------------------------------------------------------------ geometry helpers
def frame_bar(p0, p1, n, w=0.07, d=0.1, mat="black"):
    """steel bar in the u-z plane at n (for arched / raked frames)"""
    (x0, z0), (x1, z1) = p0, p1
    Lb = math.hypot(x1 - x0, z1 - z0)
    if Lb < 1e-4:
        return
    tx, tz = (x1 - x0) / Lb, (z1 - z0) / Lb
    ox, oz = -tz * w / 2, tx * w / 2
    slab((x0 + ox, n, z0 + oz), (x1 + ox, n, z1 + oz), (x1 - ox, n, z1 - oz), (x0 - ox, n, z0 - oz), d, mat)


def arch_top(o, x):
    """top of opening o at u=x: flat, or a semicircular arch springing at o['z1']"""
    if not o.get("arch"):
        return o["z1"]
    r, c = (o["b"] - o["a"]) / 2, (o["a"] + o["b"]) / 2
    return o["z1"] + math.sqrt(max(r * r - (x - c) ** 2, 0.0))


def gable_wall(u0, u1, nf, nb, zb, eave, pitch, openings, mat="ashlar"):
    """full gable end wall (rectangle + triangle) in the u-z plane between nf..nb with any number of
    rectangular or arched openings: dicts a, b, z0, z1 (+ arch=True: z1 is the springline)"""
    um, half = (u0 + u1) / 2, (u1 - u0) / 2
    tp = math.tan(pitch)
    roof = lambda x: eave + (half - abs(x - um)) * tp
    cuts = {u0, u1, um}
    for o in openings:
        cuts |= {o["a"], o["b"]}
        if o.get("arch"):
            cuts |= {o["a"] + (o["b"] - o["a"]) * k / 24 for k in range(1, 24)}
    cuts = sorted(cuts)
    for x0, x1 in zip(cuts, cuts[1:]):
        if x1 - x0 < 1e-6:
            continue
        hits = sorted([o for o in openings if o["a"] <= x0 + 1e-6 and x1 <= o["b"] + 1e-6], key=lambda o: o["z0"])
        cur = (zb, zb)
        for o in hits:
            if o["z0"] > cur[0] + 1e-4:
                prism([(x0, cur[0]), (x1, cur[1]), (x1, o["z0"]), (x0, o["z0"])], nf, nb, mat)
            cur = (arch_top(o, x0), arch_top(o, x1))
        if roof(x0) > cur[0] + 1e-4 or roof(x1) > cur[1] + 1e-4:
            prism([(x0, cur[0]), (x1, cur[1]), (x1, roof(x1)), (x0, roof(x0))], nf, nb, mat)
    return eave + half * tp


def glaze(o, nf, nb, outward, cols=1, rows=1, transoms=()):
    """glass + black frame for an opening dict in a wall between nf..nb (outward -1 = street, +1 = north)"""
    face = (nf if outward < 0 else nb) + outward * 0.03
    gn = (nf + nb) / 2
    if not o.get("arch"):
        poly([(o["a"], gn, o["z0"]), (o["b"], gn, o["z0"]), (o["b"], gn, o["z1"]), (o["a"], gn, o["z1"])], "glass")
        frame_rect(o["a"], o["b"], o["z0"], o["z1"], face, "u", cols, rows)
        for z in transoms:
            frame_bar((o["a"], z), (o["b"], z), face - 0.05, w=0.09)
        return
    a, b, z0, z1 = o["a"], o["b"], o["z0"], o["z1"]
    r, c = (b - a) / 2, (a + b) / 2
    arc = [(c + r * math.cos(math.pi * k / 24), z1 + r * math.sin(math.pi * k / 24)) for k in range(25)]  # b -> a
    poly([(a, gn, z0), (b, gn, z0)] + [(x, gn, z) for x, z in arc], "glass")
    fb = face - 0.05
    for k in range(cols + 1):
        x = a + (b - a) * k / cols
        frame_bar((x, z0), (x, arch_top(o, x) - (0.02 if 0 < k < cols else 0)), fb)
    for z in (z0 + 0.03, z1, *transoms):
        frame_bar((a, z), (b, z), fb, w=0.09 if z in transoms else 0.07)
    for (xa, za), (xb, zb_) in zip(arc, arc[1:]):
        frame_bar((xa, za), (xb, zb_), fb, w=0.09)
    # a fan of radial bars in the arch head, like a fanlight
    for k in (1, 2, 3):
        t = math.pi * k / 4
        frame_bar((c, z1), (c + r * math.cos(t), z1 + r * math.sin(t)), fb)


def sconce(x, n, z, outward):
    """slim black up/down wall light on a wall face at n facing +-n"""
    n1 = n + outward * 0.13
    box(x - 0.06, x + 0.06, min(n, n1), max(n, n1), z - 0.16, z + 0.16, "black")
    light("SPOT", (x, n + outward * 0.07, z + 0.2), 14, WARM, aim=(x, n + outward * 0.07, z + 3), spot=0.6, blend=0.9, size=0.02, name="sconce")
    light("SPOT", (x, n + outward * 0.07, z - 0.2), 14, WARM, aim=(x, n + outward * 0.07, z - 3), spot=0.6, blend=0.9, size=0.02, name="sconce")


def wall_up(x, n, z, aim_n, power=90, name="wall_up"):
    """in-ground warm uplight grazing a stone face"""
    box(x - 0.07, x + 0.07, n - 0.07, n + 0.07, z, z + 0.04, "black")
    light("SPOT", (x, n, z + 0.08), power, WARM, aim=(x, aim_n, z + 4.0), spot=0.75, blend=1.0, size=0.03, name=name)


def room_light(u0, u1, n0, n1, z, power):
    light("AREA", ((u0 + u1) / 2, (n0 + n1) / 2, z - 0.05), power, WARM, aim=((u0 + u1) / 2, (n0 + n1) / 2, -5),
          size=min(u1 - u0, n1 - n0) * 0.6, name="room")


def sofa(u0, u1, n0, n1, z=0.0):
    box(u0, u1, n0, n1, z, z + 0.42, "fabric")
    box(u0, u1, n1 - 0.22, n1, z, z + 0.8, "fabric")


# ------------------------------------------------------------------ the house
def build_house():
    R_ = L["roofs"]
    w1, w2, e1, e2 = L["w1"], L["w2"], L["e1"], L["e2"]
    lu0, lu1, ln0, ln1 = L["link"]
    luu1 = L["link_up"][1]
    gu0, gu1, gn0, gn1 = L["garage"]

    # ================= W1: SW street gable, single storey + attic; solid ashlar, one bay window, an attic slot
    u0, u1, n0, n1 = w1
    eave, pitch = R_["w1"][4], math.radians(R_["w1"][5])
    um = (u0 + u1) / 2
    box(u0, u1, n0, n1, -0.25, 0.0, "floor")
    bu0, bu1, bn0, _ = L["bay"]
    bay_o = dict(a=bu0, b=bu1, z0=0.45, z1=2.65)
    slot_a = dict(a=um - 0.22, b=um + 0.22, z0=3.7, z1=5.3)
    gable_wall(u0, u1, n0, n0 + T, -0.45, eave, pitch, [bay_o, slot_a])
    glaze(slot_a, n0, n0 + T, -1)
    # projecting bay: stone base and cheeks, black steel frame, steel hood lined in timber
    box(bu0, bu1, bn0, n0, -0.45, 0.45, "ashlar")
    box(bu0 - 0.05, bu1 + 0.05, bn0 - 0.06, n0, 0.4, 0.47, "coping")
    for a, b in ((bu0, bu0 + 0.22), (bu1 - 0.22, bu1)):
        box(a, b, bn0, n0, 0.47, 2.65, "ashlar")
    poly([(bu0 + 0.22, bn0 + 0.1, 0.47), (bu1 - 0.22, bn0 + 0.1, 0.47), (bu1 - 0.22, bn0 + 0.1, 2.65), (bu0 + 0.22, bn0 + 0.1, 2.65)], "glass")
    frame_rect(bu0 + 0.22, bu1 - 0.22, 0.47, 2.65, bn0 + 0.04, "u", 3, 1, w=0.07)
    frame_bar((bu0 + 0.22, 2.0), (bu1 - 0.22, 2.0), bn0 + 0.0, w=0.07)
    box(bu0 - 0.1, bu1 + 0.1, bn0 - 0.12, n0, 2.65, 2.85, "black")
    poly([(bu0 - 0.1, bn0 - 0.12, 2.645), (bu1 + 0.1, bn0 - 0.12, 2.645), (bu1 + 0.1, n0, 2.645), (bu0 - 0.1, n0, 2.645)], "soffit")
    light("SPOT", ((bu0 + bu1) / 2, bn0 - 0.05, 2.6), 20, WARM, aim=((bu0 + bu1) / 2, bn0 - 0.1, -2), spot=1.2, size=0.03, name="soffit")
    wo = [(11.5, 11.95, 0.45, 2.65), (15.0, 15.45, 0.45, 2.65)]
    wall_n(n0, n1, u0, u0 + T, -0.45, eave, wo, "ashlar")
    for o in wo:
        window_n(*o, u0, u0 + T, cols=1, rows=1, outward=-1)
    we = [(9.6, 10.05, 0.45, 2.65)]
    wall_n(n0, n1, u1 - T, u1, -0.45, eave, we, "ashlar")
    window_n(*we[0], u1 - T, u1, cols=1, rows=1, outward=1)
    box(u0 + T, u1 - T, n0 + T, n1, eave - 0.25, eave, "slab")
    gable_roof(n0, n1, u0, u1, eave, pitch, ow=0.45, oe=0.0, os=0.45, on=0.0)
    room_light(u0 + 0.5, u1 - 0.5, n0 + 0.5, n1 - 0.5, eave - 0.25, 1300)
    sofa(bu0, bu1, n0 + 2.6, n0 + 3.5)
    box(bu0 + 0.3, bu1 - 0.3, n0 + 1.2, n0 + 2.0, 0, 0.38, "walnut")
    box(u0 + 1.0, u1 - 1.0, n0 + 0.8, n0 + 4.0, 0, 0.01, "rug")
    cr_pendant(um, n0 + 1.6, 1.9, 40)

    # ================= W2: SW rear pavilion, two storeys, the tall arched glazed gable onto the pool courtyard
    u0, u1, n0, n1 = w2
    eave, pitch = R_["w2"][4], math.radians(R_["w2"][5])
    um = (u0 + u1) / 2
    box(u0, u1, n0, n1, -0.25, 0.0, "floor")
    zr = gable_wall(u0, u1, n0, n0 + T, -0.45, eave, pitch, [])
    arch = dict(a=um - 2.3, b=um + 2.3, z0=0.0, z1=eave - 0.6, arch=True)
    gable_wall(u0, u1, n1 - T, n1, -0.45, eave, pitch, [arch])
    glaze(arch, n1 - T, n1, 1, cols=4, transoms=(2.75,))
    sconce(arch["a"] - 0.45, n1, 2.3, 1)
    sconce(arch["b"] + 0.45, n1, 2.3, 1)
    wo = [(20.5, 20.95, 0.45, 2.65), (24.5, 24.95, 0.45, 2.65), (22.0, 22.45, UP + 0.4, UP + 1.9)]
    wall_n(n0, n1, u0, u0 + T, -0.45, eave, wo, "ashlar")
    for o in wo:
        window_n(*o, u0, u0 + T, cols=1, rows=1, outward=-1)
    sg = (ln1 + 0.6, n1 - 0.6, 0.0, eave - 0.5)
    wall_n(n0, n1, u1 - T, u1, -0.45, eave, [sg], "ashlar")
    window_n(*sg, u1 - T, u1, cols=4, rows=2, outward=1)
    box(u0 + T, u1 - T, n0 + T, ln1 + 1.2, UP - 0.25, UP, "slab")     # upper floor; double-height living to the north
    gable_roof(n0, n1, u0, u1, eave, pitch, ow=0.45, oe=0.0, os=0.45, on=0.45)
    for s in (-1, 1):
        xe = um + s * ((u1 - u0) / 2 - T)
        poly([(xe, n0 + T, eave - 0.05), (xe, n1 - T, eave - 0.05), (um, n1 - T, zr - 0.6), (um, n0 + T, zr - 0.6)], "plaster")
    room_light(u0 + 0.5, u1 - 0.5, ln1 + 1.2, n1 - 0.5, eave + 1.0, 1600)
    room_light(u0 + 0.5, u1 - 0.5, n0 + 0.5, ln1 + 1.0, UP - 0.25, 600)
    room_light(u0 + 0.5, u1 - 0.5, n0 + 0.5, ln1 + 1.0, eave + 0.3, 500)
    sofa(u0 + 1.4, u1 - 1.6, n1 - 4.6, n1 - 3.7)
    box(um - 0.7, um + 0.7, n1 - 3.3, n1 - 2.5, 0, 0.38, "walnut")
    box(u0 + 1.0, u1 - 1.0, n1 - 5.2, n1 - 1.6, 0, 0.01, "rug")
    box(u0 + T, u0 + T + 0.03, n1 - 5.0, n1 - 3.0, 1.4, 2.6, "art")
    cr_pendant(um, n1 - 3.0, 4.2, 110)

    # ================= E1: NE single-storey ashlar wing over the garage; flat roof = terrace off the upper retreat
    u0, u1, n0, n1 = e1
    top = 3.25          # terrace deck level
    par = 3.65          # parapet top
    box(u0, u1, n0, gn1, -0.3, 0.0, "slab")
    box(u0, u1, gn1, n1, -0.25, 0.0, "floor")
    # street face: two projecting ashlar piers, recessed ashlar band with one slot, recessed black glass garage door
    for a, b in ((u0, gu0 + 0.8), (gu1 - 0.8, u1)):
        box(a, b, n0, n0 + T, GZ - 0.3, par, "ashlar")
        box(a - 0.03, b + 0.03, n0 - 0.04, n0 + T + 0.03, par, par + 0.06, "coping")
    bnd = n0 + T
    eum = (u0 + u1) / 2
    slot_e = (eum - 0.22, eum + 0.22)
    gd_top = GZ + 2.35
    wall_u(gu0 + 0.8, gu1 - 0.8, bnd, bnd + T, gd_top, par, [(slot_e[0], slot_e[1], 0.45, 2.65)], "ashlar")
    window_u(slot_e[0], slot_e[1], 0.45, 2.65, bnd, bnd + T, cols=1, rows=1)
    box(gu0 + 0.77, gu1 - 0.77, bnd - 0.04, bnd + T + 0.03, par, par + 0.06, "coping")
    poly([(gu0 + 0.8, n0, gd_top - 0.005), (gu1 - 0.8, n0, gd_top - 0.005), (gu1 - 0.8, bnd + T, gd_top - 0.005), (gu0 + 0.8, bnd + T, gd_top - 0.005)], "soffit")
    gd_n = L["garage_door_n"]
    poly([(gu0 + 0.8, gd_n, GZ), (gu1 - 0.8, gd_n, GZ), (gu1 - 0.8, gd_n, gd_top), (gu0 + 0.8, gd_n, gd_top)], "darkglass")
    frame_rect(gu0 + 0.8, gu1 - 0.8, GZ, gd_top, gd_n - 0.04, "u", 4, 4, w=0.08)
    for a, b in ((gu0 + 0.8, gu0 + 0.85), (gu1 - 0.85, gu1 - 0.8)):   # reveal returns to the door
        box(a, b, n0, gd_n, GZ, gd_top, "ashlar")
    box(gu0 + 0.3, gu1 - 0.3, gd_n + 0.5, gn1 - 0.3, GZ, GZ + 0.01, "slab")
    light("AREA", ((gu0 + gu1) / 2, (gn0 + gn1) / 2, -0.4), 160, (1.0, 0.85, 0.7), aim=((gu0 + gu1) / 2, (gn0 + gn1) / 2, -5), size=3, name="garage_in")
    for x in (gu0 + 0.4, gu1 - 0.4):
        sconce(x, n0, GZ + 2.0, -1)
    ws = [(9.6, 10.05, 0.45, 2.65)]
    wall_n(n0, n1, u0, u0 + T, GZ - 0.3, par, ws, "ashlar")
    window_n(*ws[0], u0, u0 + T, cols=1, rows=1, outward=-1)
    es = [(9.5, 9.95, 0.45, 2.65), (13.5, 13.95, 0.45, 2.65)]
    wall_n(n0, n1, u1 - T, u1, GZ - 0.3, par, es, "ashlar")
    for o in es:
        window_n(*o, u1 - T, u1, cols=1, rows=1, outward=1)
    for a, b, c, d in ((u0 - 0.03, u0 + T + 0.03, n0, L["link"][2]), (u1 - T - 0.03, u1 + 0.03, n0, n1)):
        box(a, b, c, d, par, par + 0.06, "coping")
    box(u0 + T, u1 - T, gn1, gn1 + 0.3, GZ - 0.3, -0.3, "ashlar")       # garage back (retaining) wall
    box(u0 + T, u1 - T, n0 + T, n1, 2.9, top - 0.02, "slab")             # roof slab
    box(luu1, u0 + T, ln0, n1, 2.9, top - 0.02, "slab")
    poly([(luu1, ln0 + 0.1, top), (u1 - T, ln0 + 0.1, top), (u1 - T, n1, top), (luu1, n1, top)], "paving")
    poly([(u0 + T, n0 + 2 * T, top), (u1 - T, n0 + 2 * T, top), (u1 - T, ln0 + 0.1, top), (u0 + T, ln0 + 0.1, top)], "paving")
    # frameless glass balustrade on the parapets; fixed louvre screen on the NE side (privacy to the neighbour's
    # yard, blades aligned with the long view east / south-east)
    for (a, b, c, d) in ((u0, u1, n0 + 0.1, n0 + 0.1), (u0 + 0.1, u0 + 0.1, n0, ln0), (luu1, u0, ln0 + 0.05, ln0 + 0.05)):
        poly([(a, c, par + 0.06), (b, d, par + 0.06), (b, d, par + 1.05), (a, c, par + 1.05)], "glass")
        box(min(a, b) - 0.02, max(a, b) + 0.02, min(c, d) - 0.03, max(c, d) + 0.03, par + 1.02, par + 1.07, "black")
    ex, en_ = 0.89, -0.45
    for nn in np.arange(n0 + 0.6, n1 - 0.2, 0.32):
        uc = u1 - 0.2
        slab((uc - ex * 0.14, nn - en_ * 0.14, top), (uc + ex * 0.14, nn + en_ * 0.14, top),
             (uc + ex * 0.14, nn + en_ * 0.14, top + 1.9), (uc - ex * 0.14, nn - en_ * 0.14, top + 1.9), 0.03, "black")
    box(u1 - 0.38, u1 - 0.02, n0 + 0.4, n1 - 0.1, top + 1.9, top + 1.96, "black")
    for x, nn in ((17.6, 9.2), (19.2, 9.2)):                            # terrace loungers and planters
        box(x - 0.35, x + 0.35, nn, nn + 1.9, top, top + 0.32, "linen")
        box(x - 0.35, x + 0.35, nn, nn + 0.4, top + 0.32, top + 0.7, "linen")
    for nn in (8.0, 12.5, 15.8):
        box(u1 - 1.1, u1 - 0.45, nn - 0.35, nn + 0.35, top, top + 0.55, "ashlar")
        bush((u1 - 0.78, nn, top + 0.9), 0.38, "hedge", 14)
    for x in (16.6, 21.4):
        light("POINT", (x, n0 + 0.5, top + 0.3), 12, WARM, size=0.05, name="sconce")
    room_light(u0 + 0.5, u1 - 0.5, n0 + 0.5, n1 - 0.5, 2.9, 900)
    box(u0 + 1.2, u1 - 1.2, n1 - 3.4, n1 - 1.4, 0, 0.55, "linen")       # guest bedroom over the garage

    # ================= E2: NE rear gable (storey + attic), full-height glazed gable to the north courtyard
    u0, u1, n0, n1 = e2
    eave, pitch = R_["e2"][4], math.radians(R_["e2"][5])
    um = (u0 + u1) / 2
    box(u0, u1, n0, n1, -0.25, 0.0, "floor")
    slot_s = dict(a=um - 0.22, b=um + 0.22, z0=3.95, z1=5.35)
    zr = gable_wall(u0, u1, n0, n0 + T, -0.45, eave, pitch, [slot_s])
    glaze(slot_s, n0, n0 + T, -1)
    b_ = 0.45
    outer = [(u0, -0.45), (u1, -0.45), (u1, eave), (um, zr), (u0, eave)]
    apex = zr - 0.7
    inner = [(u0 + b_, 0), (u1 - b_, 0), (u1 - b_, eave), (um, apex), (u0 + b_, eave)]
    ring(outer, inner, n1 - T, n1, "ashlar")
    poly([(u, n1 - T / 2, z) for u, z in inner], "glass")
    fb = n1 + 0.04
    for k in range(5):
        x = u0 + b_ + k * (u1 - u0 - 2 * b_) / 4
        tp_ = eave + (apex - eave) * (1 - abs(x - um) / ((u1 - u0) / 2 - b_))
        frame_bar((x, 0), (x, max(tp_, eave)), fb)
    for z in (0.03, eave, 2.75):
        frame_bar((u0 + b_, z), (u1 - b_, z), fb)
    frame_bar((u0 + b_, eave), (um, apex), fb)
    frame_bar((um, apex), (u1 - b_, eave), fb)
    sconce(u0 + 0.22, n1, 2.3, 1)
    sconce(u1 - 0.22, n1, 2.3, 1)
    cg = (ln1 + 0.4, n1 - 0.5, 0.0, 2.75)
    wall_n(n0, n1, u0, u0 + T, -0.45, eave, [cg], "ashlar")
    window_n(*cg, u0, u0 + T, cols=3, rows=1, outward=-1)
    es = [(18.5, 18.95, 0.45, 2.65), (22.5, 22.95, 0.45, 2.65)]
    wall_n(n0, n1, u1 - T, u1, -0.45, eave, es, "ashlar")
    for o in es:
        window_n(*o, u1 - T, u1, cols=1, rows=1, outward=1)
    gable_roof(n0, n1, u0, u1, eave, pitch, ow=0.0, oe=0.45, os=0.45, on=0.45)
    for s in (-1, 1):
        xe = um + s * ((u1 - u0) / 2 - T)
        poly([(xe, n0 + T, eave - 0.05), (xe, n1 - T, eave - 0.05), (um, n1 - T, zr - 0.6), (um, n0 + T, zr - 0.6)], "plaster")
    room_light(u0 + 0.5, u1 - 0.5, n0 + 0.5, n1 - 0.5, eave + 1.2, 1500)
    box(u0 + 1.6, u1 - 1.6, n1 - 3.4, n1 - 2.4, 0.72, 0.77, "walnut")   # dining behind the glass gable
    for x in (u0 + 1.9, um, u1 - 1.9):
        for nn in (n1 - 3.75, n1 - 2.05):
            box(x - 0.22, x + 0.22, nn - 0.22, nn + 0.22, 0, 0.46, "fabric")
    cr_pendant(um, n1 - 2.9, 1.9, 70)
    box(u0 + 1.0, u1 - 1.0, n0 + 1.0, n0 + 1.9, 0, 0.9, "walnut")        # kitchen island

    # ================= the two-storey link: timber-clad entry + portico; upper retreat glazed to the street and east
    box(lu0, lu1, ln0, ln1, -0.25, 0.0, "floor")
    door = (11.3, 12.7, 0.0, 2.75)
    ls = [door, (8.6, 9.0, 0.45, 2.65), (14.4, 14.8, 0.45, 2.65)]
    wall_u(lu0, lu1, ln0, ln0 + 0.25, -0.45, 2.9, ls, "soffit")
    for o in ls[1:]:
        window_u(*o, ln0, ln0 + 0.25, cols=1, rows=1)
    box(door[0], door[1], ln0 + 0.1, ln0 + 0.16, 0, door[3], "door")
    box(door[1] - 0.25, door[1] - 0.2, ln0 + 0.02, ln0 + 0.1, 0.9, 1.9, "black")
    box(lu0, lu1, ln0, ln1, 2.9, UP, "slab")
    box(luu1, lu1, ln0 + 4.0, ln1, UP, UP + 0.08, "black")                # ground-floor-only strip roof by E2
    # stone entry portico, its roof a small balcony off the retreat
    pu0, pu1, pn0, pn1 = L["portico"]
    for a in (pu0, pu1 - 0.45):
        box(a, a + 0.45, pn0, pn0 + 0.45, PADZ, UP - 0.3, "ashlar")
    box(pu0 - 0.1, pu1 + 0.1, pn0 - 0.1, ln0, UP - 0.3, UP, "coping")
    poly([(pu0 - 0.1, pn0 - 0.1, UP - 0.305), (pu1 + 0.1, pn0 - 0.1, UP - 0.305), (pu1 + 0.1, ln0, UP - 0.305), (pu0 - 0.1, ln0, UP - 0.305)], "soffit")
    light("SPOT", ((pu0 + pu1) / 2, (pn0 + ln0) / 2, UP - 0.35), 35, WARM, aim=((pu0 + pu1) / 2, (pn0 + ln0) / 2, -2), spot=1.3, size=0.03, name="soffit")
    box(pu0 - 0.3, pu1 + 0.3, pn0 - 0.6, ln0, PADZ - 0.2, 0.0, "coping")
    for (a, b, c, d) in ((pu0 - 0.05, pu1 + 0.05, pn0 - 0.05, pn0 - 0.05), (pu0 - 0.05, pu0 - 0.05, pn0 - 0.05, ln0), (pu1 + 0.05, pu1 + 0.05, pn0 - 0.05, ln0)):
        poly([(a, c, UP), (b, d, UP), (b, d, UP + 1.05), (a, c, UP + 1.05)], "glass")
        box(min(a, b) - 0.02, max(a, b) + 0.02, min(c, d) - 0.02, max(c, d) + 0.02, UP + 1.02, UP + 1.07, "black")
    # upper storey: generous glass south (street view) and east (over the terrace), stone elsewhere
    up_s = (lu0 + T, luu1 - T, UP, 5.5)
    wall_u(lu0, luu1, ln0, ln0 + 0.25, 2.9, 5.6, [up_s], "soffit")
    poly([(up_s[0], ln0 + 0.12, UP), (up_s[1], ln0 + 0.12, UP), (up_s[1], ln0 + 0.12, 5.5), (up_s[0], ln0 + 0.12, 5.5)], "glass")
    frame_rect(up_s[0], up_s[1], UP, 5.5, ln0 - 0.02, "u", 6, 1, w=0.07)
    frame_bar((up_s[0], UP + 2.4), (up_s[1], UP + 2.4), ln0 - 0.04, w=0.07)
    up_e = (ln0 + 0.3, ln0 + 3.7, UP, 5.5)
    wall_n(ln0, ln1, luu1 - T, luu1, 2.9, 5.6, [up_e], "ashlar")
    window_n(*up_e, luu1 - T, luu1, cols=3, rows=1, outward=1)
    wall_n(ln0, ln1, lu0, lu0 + T, 2.9, 5.6, [], "ashlar")
    up_n = (lu0 + T, luu1 - T, UP, 5.5)
    wall_u(lu0, luu1, ln1 - 0.25, ln1, 2.9, 5.6, [up_n], "ashlar")
    window_u(*up_n, ln1 - 0.25, ln1, cols=6, rows=1, outward=1)
    gn_l = ln1 - 0.1                                                    # ground floor north: full glass
    poly([(lu0, gn_l, 0.0), (lu1, gn_l, 0.0), (lu1, gn_l, 2.9), (lu0, gn_l, 2.9)], "glass")
    frame_rect(lu0 + 0.03, lu1 - 0.03, 0.0, 2.9, gn_l + 0.05, "u", 6, 1, w=0.07)
    # north balcony with timber soffit and glass balustrade
    bu0, bu1, bn0, bn1 = L["balcony"]
    box(bu0, bu1, ln1 - 0.1, bn1, UP - 0.3, UP, "coping")
    poly([(bu0, bn1 - 0.02, UP - 0.32), (bu1, bn1 - 0.02, UP - 0.32), (bu1, ln1, UP - 0.32), (bu0, ln1, UP - 0.32)], "soffit")
    poly([(bu0 + 0.05, bn1 - 0.1, UP), (bu1 - 0.05, bn1 - 0.1, UP), (bu1 - 0.05, bn1 - 0.1, UP + 1.05), (bu0 + 0.05, bn1 - 0.1, UP + 1.05)], "glass")
    box(bu0 + 0.05, bu1 - 0.05, bn1 - 0.14, bn1 - 0.06, UP + 1.02, UP + 1.08, "black")
    for f in (0.2, 0.4, 0.6, 0.8):
        x = bu0 + (bu1 - bu0) * f
        box(x - 0.05, x + 0.05, bn1 - 0.85, bn1 - 0.75, UP - 0.33, UP - 0.32, "bulb")
        light("SPOT", (x, bn1 - 0.8, UP - 0.38), 25, WARM, aim=(x, bn1 - 0.8, -1), spot=1.0, blend=0.8, size=0.03, name="soffit")
    for x in (9.4, 10.6):
        box(x - 0.3, x + 0.3, bn0 + 0.3, bn0 + 0.9, UP, UP + 0.42, "linen")
    # flat roof between the gables: dark membrane, black fascia, timber soffits
    rt = L["flat"]["link"]
    box(lu0, luu1, rt[2], w2[2], 5.6, rt[4], "roof_x")
    box(lu0 - 0.3, luu1, w2[2], rt[3], 5.6, rt[4], "roof_x")
    box(lu0, luu1, rt[2] - 0.1, rt[2], 5.55, rt[4] + 0.02, "black")
    box(lu0 - 0.3, luu1, rt[3], rt[3] + 0.1, 5.55, rt[4] + 0.02, "black")
    box(luu1, luu1 + 0.1, rt[2] - 0.1, rt[3] + 0.1, 5.55, rt[4] + 0.02, "black")
    poly([(lu0, rt[2], 5.595), (luu1, rt[2], 5.595), (luu1, ln0, 5.595), (lu0, ln0, 5.595)], "soffit")
    poly([(lu0, ln1, 5.595), (luu1, ln1, 5.595), (luu1, rt[3], 5.595), (lu0, rt[3], 5.595)], "soffit")
    box(lu0, luu1, ln0, ln1, 5.59, 5.6, "plaster")
    # interiors: entry/dining below; upper retreat lounge (south + east views) and main bedroom (north balcony)
    box(lu0 + 1.6, lu1 - 1.6, ln1 - 4.0, ln1 - 3.0, 0.72, 0.77, "walnut")
    for x in np.linspace(lu0 + 2.0, lu1 - 2.0, 4):
        for nn in (ln1 - 4.35, ln1 - 2.65):
            box(x - 0.22, x + 0.22, nn - 0.22, nn + 0.22, 0, 0.46, "fabric")
    for x in (lu0 + 2.6, (lu0 + lu1) / 2, lu1 - 2.6):
        cr_pendant(x, ln1 - 3.5, 1.95, 50)
    sofa(lu0 + 1.5, luu1 - 2.0, ln0 + 2.4, ln0 + 3.3, UP)
    box(lu0 + 1.8, luu1 - 2.2, ln0 + 0.8, ln0 + 2.2, UP, UP + 0.01, "rug")
    box(lu0 + 2.0, luu1 - 2.0, ln1 - 2.6, ln1 - 0.6, UP, UP + 0.55, "linen")
    box((lu0 + luu1) / 2 - 1.0, (lu0 + luu1) / 2 + 1.0, ln1 - 4.5, ln1 - 4.47, UP + 1.0, UP + 2.0, "art")
    room_light(lu0 + 0.5, lu1 - 0.5, ln0 + 0.5, ln1 - 0.5, 2.9, 1100)
    room_light(lu0 + 0.5, luu1 - 0.5, ln0 + 0.5, ln1 - 0.5, 5.6, 900)

    # ================= outdoor room (cabana): stone back walls and fireplace, timber posts, steep roof
    ou0, ou1, on0, on1 = L["cabana"]
    ze_c, pc = R_["cabana"][4], math.radians(R_["cabana"][5])
    zo = PADZ + 0.12
    box(ou0, ou1, on0, on1, PADZ - 0.3, zo, "coping")
    box(ou1 - T, ou1, on0, on1, PADZ - 0.3, PADZ + ze_c, "ashlar")
    box(ou0, ou1 - T, on1 - T, on1, PADZ - 0.3, PADZ + ze_c, "ashlar")
    for x, nn in ((ou0 + 0.2, on0 + 0.2), (ou1 - 0.6, on0 + 0.2), (ou0 + 0.2, on1 - 0.6)):
        box(x - 0.14, x + 0.14, nn - 0.14, nn + 0.14, zo, PADZ + ze_c, "timber")
    box(ou0, ou1, on0 + 0.06, on0 + 0.34, PADZ + ze_c - 0.3, PADZ + ze_c, "timber")
    box(ou0 + 0.06, ou0 + 0.34, on0, on1, PADZ + ze_c - 0.3, PADZ + ze_c, "timber")
    gable_wall(ou0, ou1, on1 - T, on1, PADZ + ze_c - 0.1, PADZ + ze_c, pc, [])
    zr_c = gable_roof(on0, on1, ou0, ou1, PADZ + ze_c, pc, ow=0.35, oe=0.35, os=0.35, on=0.35)
    poly([(ou0, on0, PADZ + ze_c - 0.31), (ou1, on0, PADZ + ze_c - 0.31), (ou1, on1, PADZ + ze_c - 0.31), (ou0, on1, PADZ + ze_c - 0.31)], "soffit")
    fn0, fn1 = on0 + 1.6, on1 - 1.6
    box(ou1 - 0.9, ou1 - T, fn0, fn1, zo, zo + 2.4, "ashlar")
    box(ou1 - 0.8, ou1 - T + 0.1, fn0 + 0.45, fn1 - 0.45, zo + 2.4, zr_c + 0.9, "ashlar")
    box(ou1 - 0.86, ou1 - T + 0.16, fn0 + 0.39, fn1 - 0.39, zr_c + 0.9, zr_c + 1.02, "black")
    box(ou1 - 0.95, ou1 - 0.9, fn0 + 0.5, fn1 - 0.5, zo + 0.45, zo + 1.3, "black")
    box(ou1 - 0.9, ou1 - 0.6, fn0 + 0.65, fn1 - 0.65, zo + 0.5, zo + 0.72, "fire")
    light("POINT", (ou1 - 1.2, (fn0 + fn1) / 2, zo + 0.8), 140, (1.0, 0.45, 0.15), size=0.3, name="pool_fire")
    box(ou0 + 0.6, ou0 + 1.5, on0 + 1.3, on1 - 1.3, zo, zo + 0.45, "linen")
    box(ou0 + 0.6, ou0 + 0.85, on0 + 1.3, on1 - 1.3, zo + 0.45, zo + 0.85, "linen")
    box(ou0 + 2.0, ou0 + 2.8, on0 + 2.4, on1 - 2.4, zo, zo + 0.4, "walnut")
    cr_pendant((ou0 + ou1) / 2, (on0 + on1) / 2, zo + 2.0, 40)


def gable_roof(n0, n1, u0, u1, ze, pitch, ow=0.45, oe=0.45, os=0.45, on=0.45):
    """steep standing-seam gable roof (ridge along n) with per-side overhangs (0 where it abuts the link),
    black cappings, gutters and warm timber eave and rake soffits"""
    tp = math.tan(pitch)
    um = (u0 + u1) / 2
    zr = ze + (u1 - u0) / 2 * tp
    zw, zE = ze - ow * tp, ze - oe * tp
    na, nb = n0 - os, n1 + on
    slab((u0 - ow, nb, zw), (u0 - ow, na, zw), (um, na, zr + 0.05), (um, nb, zr + 0.05), 0.1, "roof_y")
    slab((u1 + oe, na, zE), (u1 + oe, nb, zE), (um, nb, zr + 0.05), (um, na, zr + 0.05), 0.1, "roof_y")
    for n in (na, nb):
        slab((u0 - ow, n, zw - 0.12), (um, n, zr - 0.07), (um, n, zr + 0.2), (u0 - ow, n, zw + 0.15), 0.05, "black")
        slab((um, n, zr - 0.07), (u1 + oe, n, zE - 0.12), (u1 + oe, n, zE + 0.15), (um, n, zr + 0.2), 0.05, "black")
    box(um - 0.09, um + 0.09, na, nb, zr + 0.1, zr + 0.22, "black")
    e = 0.06
    for o, ue, ui, zz in ((ow, u0 - ow, u0, zw), (oe, u1 + oe, u1, zE)):
        if o > 0:
            box(ue - 0.06, ue + 0.06, na, nb, zz - 0.15, zz + 0.02, "black")
            poly([(ue, na, zz - e), (ui, na, ze - e), (ui, nb, ze - e), (ue, nb, zz - e)], "soffit")
    for o, n_a, n_b in ((os, na, n0), (on, n1, nb)):
        if o > 0:
            poly([(u0 - ow, n_a, zw - e), (um, n_a, zr - e + 0.02), (um, n_b, zr - e + 0.02), (u0 - ow, n_b, zw - e)], "soffit")
            poly([(u1 + oe, n_a, zE - e), (um, n_a, zr - e + 0.02), (um, n_b, zr - e + 0.02), (u1 + oe, n_b, zE - e)], "soffit")
    return zr


# ------------------------------------------------------------------ landscape: whole-lot terraces
def clip_convex(pts, clip):
    """Sutherland-Hodgman: polygon pts clipped by the convex polygon clip (both lists of (u, n))"""
    def inside(p, a, b):
        return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) * SIGN >= 0

    def inter(p, q, a, b):
        x1, y1, x2, y2, x3, y3, x4, y4 = *p, *q, *a, *b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    out = pts
    for a, b in zip(clip, clip[1:] + clip[:1]):
        if not out:
            break
        inp, out = out, []
        for i, p in enumerate(inp):
            q = inp[i - 1]
            if inside(p, a, b):
                if not inside(q, a, b):
                    out.append(inter(q, p, a, b))
                out.append(p)
            elif inside(q, a, b):
                out.append(inter(q, p, a, b))
    return out


CLIP = [tuple(p) for p in LOT_INNER[:-1]]
SIGN = 1 if sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(CLIP, CLIP[1:] + CLIP[:1])) > 0 else -1
PO = [tuple(p) for p in PAD_OUTLINE[:-1]]
PO_SIGN = 1 if sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(PO, PO[1:] + PO[:1])) > 0 else -1


def inr(u, n, r):
    return r[0] <= u <= r[1] and r[2] <= n <= r[3]


def surface(u, n):
    """finish material and level for a point of the designed ground (None = something else covers it)"""
    g = L["garage"]
    if n < L["stair"][3] and L["stair"][0] - 0.35 <= u <= L["stair"][1] + 0.35:
        return None
    if n < g[3] and u >= g[0]:
        return ("mulch", 155.4 - cr.FFL) if u > g[1] + 0.3 else None    # planted strip NE of the drive / garage
    if n < g[2] and u >= g[0] - 1.0:
        return ("mulch", 156.2 - cr.FFL) if u < g[0] - 0.3 else None    # planted tier west of the drive
    if n < 5.0:
        return None                               # planted bank (draped separately)
    for k in ("w1", "w2", "e1", "e2", "link", "cabana", "bay"):
        if inr(u, n, L[k]):
            return None
    pu0, pu1, pn0, pn1 = L["pool"]
    if inr(u, n, (pu0 - 0.3, pu1 + 0.3, pn0 - 0.3, pn1 + 0.3)):
        return None
    tn = terrace_n(u)
    if n >= tn:                                   # upper garden terrace: lawn, gravel walk, olive grove
        return ("gravel", UPZ) if tn + 2.2 <= n <= tn + 3.2 or 16.0 <= u <= 18.8 else ("lawn", UPZ)
    if n < L["link"][2]:                          # entry forecourt: gravel parterre, paved path to the portico
        if 5.4 <= n <= 6.5 and u <= 12.7 or 11.3 <= u <= 12.7 and n >= 5.4:
            return ("paving", PADZ)
        return ("gravel", PADZ)
    if 7.6 <= u <= 21.0 or n >= 26.5 and u <= 26.0:
        return ("paving", PADZ)                   # courtyard terrace and pool deck
    return ("lawn", PADZ)


def build_surfaces():
    us = sorted(set(np.round(np.arange(-6.0, 31.0, 0.5), 3)) | {L["garage"][0], L["garage"][1], 7.6, 21.0, 16.0, 18.8, 11.0, 12.8})
    ns = sorted(set(np.round(np.arange(2.0, 46.0, 0.5), 3)) | {5.0, 5.6, 6.9, L["garage"][3], L["link"][2], 26.5})
    for u0, u1 in zip(us, us[1:]):
        for n0, n1 in zip(ns, ns[1:]):
            # split each cell along the terrace line so the two levels meet the wall exactly
            for lo, hi in ((-1e9, 0), (0, 1e9)):
                cell = [(u0, n0), (u1, n0), (u1, n1), (u0, n1)]
                pts = clip_convex(cell, CLIP)
                if len(pts) < 3:
                    continue
                # half-plane split by the terrace line n - terrace_n(u) in [lo, hi]
                tl = L["terrace_line"]
                a_, b_ = (tl[0] - 50, tl[1] - 50 * tl[2]), (tl[0] + 50, tl[1] + 50 * tl[2])
                if lo < 0:
                    half = [b_, a_, (a_[0], -100), (b_[0], -100)]
                else:
                    half = [a_, b_, (b_[0], 200), (a_[0], 200)]
                half = half if sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(half, half[1:] + half[:1])) * SIGN > 0 else half[::-1]
                pts = clip_convex(pts, half)
                if len(pts) < 3:
                    continue
                cu = sum(p[0] for p in pts) / len(pts)
                cn = sum(p[1] for p in pts) / len(pts)
                s = surface(cu, cn)
                if s is None:
                    continue
                mat, z = s
                vs = [(p[0], p[1], z + 0.03) for p in pts]
                if SIGN < 0:
                    vs = vs[::-1]
                poly(vs, mat)


def wall_seg(ua, na, ub, nb, bot, top, side, mat="ashlar", t=0.3):
    """wall on the line a->b, thickness t to the given side (+1 left, -1 right), with a stone coping"""
    dx, dn = ub - ua, nb - na
    ln = math.hypot(dx, dn) or 1
    ox, on = -dn / ln * t * side, dx / ln * t * side
    faces(mat, [(ua, na, bot), (ub, nb, bot), (ub + ox, nb + on, bot), (ua + ox, na + on, bot),
                (ua, na, top), (ub, nb, top), (ub + ox, nb + on, top), (ua + ox, na + on, top)],
          [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    cx, cn_ = ox * 0.15 / t, on * 0.15 / t
    faces("coping", [(ua - cx, na - cn_, top), (ub - cx, nb - cn_, top), (ub + ox + cx, nb + on + cn_, top), (ua + ox + cx, na + on + cn_, top),
                     (ua - cx, na - cn_, top + 0.06), (ub - cx, nb - cn_, top + 0.06), (ub + ox + cx, nb + on + cn_, top + 0.06), (ua + ox + cx, na + on + cn_, top + 0.06)],
          [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])


def build_landscape():
    g = L["garage"]
    gu0, gu1, gn0, gn1 = g
    build_surfaces()

    # --- boundary retaining walls following the side and rear boundaries (0-0.3 m inside the lot line)
    for (ua, na), (ub, nb) in zip(PO, PO[1:] + PO[:1]):
        if max(na, nb) < 2.6:
            continue  # the street wall handles the front edge
        if min(ua, ub) > gu0 - 0.5 and max(na, nb) < gn1 + 0.6:
            continue  # garage notch: retained by the garage walls
        Lg = math.hypot(ub - ua, nb - na)
        steps = max(1, int(Lg / 1.5))
        for k in range(steps):
            t0, t1 = k / steps, (k + 1) / steps
            u0_, n0_ = ua + (ub - ua) * t0, na + (nb - na) * t0
            u1_, n1_ = ua + (ub - ua) * t1, na + (nb - na) * t1
            mu, mn = (u0_ + u1_) / 2, (n0_ + n1_) / 2
            dx, dn = u1_ - u0_, n1_ - n0_
            ln = math.hypot(dx, dn) or 1
            out_u, out_n = dn / ln * PO_SIGN, -dx / ln * PO_SIGN          # outward normal
            level = UPZ if mn >= terrace_n(mu) + 0.1 else PADZ
            gout = zg(mu + out_u * 0.9, mn + out_n * 0.9)
            top = max(level + 0.9, gout + 0.25)
            bot = min(gout, level) - 0.4
            wall_seg(u0_, n0_, u1_, n1_, bot, top, -PO_SIGN)               # thickness outward, toward the boundary
            if gout > level + 0.6 and k % 2 == 0:
                wall_up(mu - out_u * 0.35, mn - out_n * 0.35, level + 0.03, mn, power=60)
    # --- terrace wall: courtyard (157.45) up to the upper garden terrace (159.2), stair gap east of the pool
    tl = L["terrace_line"]
    su0, su1 = 16.4, 18.4
    u_sw, u_ne = -4.25, 28.75
    for a, b in ((u_sw, su0), (su1, u_ne)):
        steps = max(1, int((b - a) / 2.0))
        for k in range(steps):
            x0, x1 = a + (b - a) * k / steps, a + (b - a) * (k + 1) / steps
            wall_seg(x0, terrace_n(x0), x1, terrace_n(x1), PADZ - 0.3, UPZ + 0.45, 1)
    for x in np.arange(u_sw + 1.5, u_ne - 0.5, 3.0):
        if su0 - 1 < x < su1 + 1 or L["cabana"][0] - 0.5 < x < L["cabana"][1] + 0.5:
            continue
        wall_up(x, terrace_n(x) - 0.35, PADZ + 0.03, terrace_n(x), power=70)
    # stair up to the terrace with stone cheeks
    n_top = terrace_n(su0)
    n_bot = n_top - 3.4
    nsteps = 11
    for k in range(nsteps):
        z = PADZ + (UPZ - PADZ) * (k + 1) / nsteps
        box(su0, su1, n_bot + k * 3.4 / nsteps, terrace_n(su1) + 0.35, PADZ - 0.3, z, "coping")
    for x0, x1 in ((su0 - 0.3, su0), (su1, su1 + 0.3)):
        box(x0, x1, n_bot, terrace_n(x1) + 0.3, PADZ - 0.3, UPZ + 0.45, "ashlar")

    # --- pool (black-framed courtyard pool, stone coping)
    pu0, pu1, pn0, pn1 = L["pool"]
    for a, c, n0_, n1_ in ((pu0 - 0.3, pu1 + 0.3, pn0 - 0.3, pn0), (pu0 - 0.3, pu1 + 0.3, pn1, pn1 + 0.3), (pu0 - 0.3, pu0, pn0, pn1), (pu1, pu1 + 0.3, pn0, pn1)):
        box(a, c, n0_, n1_, PADZ - 0.2, PADZ + 0.07, "coping")
    box(pu0, pu1, pn0, pn1, PADZ - 1.7, PADZ - 1.6, "pooltile")
    for a, c, n0_, n1_ in ((pu0, pu1, pn0, pn0 + 0.02), (pu0, pu1, pn1 - 0.02, pn1), (pu0, pu0 + 0.02, pn0, pn1), (pu1 - 0.02, pu1, pn0, pn1)):
        box(a, c, n0_, n1_, PADZ - 1.7, PADZ - 0.1, "pooltile")
    poly([(pu0, pn0, PADZ - 0.12), (pu1, pn0, PADZ - 0.12), (pu1, pn1, PADZ - 0.12), (pu0, pn1, PADZ - 0.12)], "water")
    for x in (pu0 + 2.5, (pu0 + pu1) / 2, pu1 - 2.5):
        light("POINT", (x, pn0 + 0.15, PADZ - 0.9), 55, (0.55, 0.85, 1.0), size=0.05, name="pool")
    for x in (pu0 + 1.5, pu0 + 3.0):  # sun loungers on the deck west of the link
        box(x - 0.35, x + 0.35, pn1 + 0.8, pn1 + 2.7, PADZ, PADZ + 0.3, "linen")
        box(x - 0.35, x + 0.35, pn1 + 2.3, pn1 + 2.7, PADZ + 0.3, PADZ + 0.7, "linen")

    # --- street: low ashlar wall with the gate and steps at the SW end, planted bank, driveway apron
    s0, s1, sn0, sn1 = L["stair"]
    topw = FRONT["wall_top"]
    tier = 156.2 - cr.FFL
    strip = 155.4 - cr.FFL
    for a, b in ((0.3, s0 - 0.45), (s1 + 0.45, gu0 - 1.3)):
        box(a, b, 1.6, 2.2, -3.4, topw, "ashlar")
        box(a - 0.03, b + 0.03, 1.55, 2.25, topw, topw + 0.07, "coping")
    gg = zg((s0 + s1) / 2, 1.2)
    for x in (s0 - 0.45, s1 + 0.45):  # ashlar piers with lanterns
        box(x - 0.42, x + 0.42, 1.48, 2.32, gg - 0.3, topw + 0.55, "ashlar")
        box(x - 0.48, x + 0.48, 1.42, 2.38, topw + 0.55, topw + 0.65, "coping")
        lantern(x, 1.9, topw + 0.65, 0.34, 28)
    for k in range(9):  # black steel slatted gate
        x = s0 + 0.05 + k * (s1 - s0 - 0.1) / 8
        box(x - 0.025, x + 0.025, 1.87, 1.93, gg + 0.05, gg + 1.4, "black")
    box(s0, s1, 1.86, 1.94, gg + 1.36, gg + 1.42, "black")
    for k in range(10):
        z = gg + (PADZ - gg) * (k + 1) / 10
        box(s0, s1, 2.0 + k * (sn1 - 2.0) / 10, sn1, gg - 0.4, z, "coping")
    for x0, x1 in ((s0 - 0.3, s0), (s1, s1 + 0.3)):
        box(x0, x1, 2.2, sn1, gg - 0.4, PADZ + 0.35, "ashlar")
    # low planted retaining walls either side of the near-flat driveway apron (two tiers on the high SW side)
    box(gu0 - 0.3, gu0, 0.95, gn0, GZ - 0.4, tier + 0.3, "ashlar")
    box(gu0 - 0.33, gu0 + 0.03, 0.92, gn0, tier + 0.3, tier + 0.36, "coping")
    box(gu0 - 1.3, gu0 - 1.0, 1.6, gn0, tier - 0.4, PADZ + 0.3, "ashlar")
    box(gu0 - 1.33, gu0 - 0.97, 1.57, gn0, PADZ + 0.3, PADZ + 0.36, "coping")
    box(gu0 - 1.3, gu0 - 0.3, 0.95, 1.6, tier - 0.6, tier + 0.3, "ashlar")
    box(gu1, gu1 + 0.3, 0.6, gn1, GZ - 0.4, strip + 0.3, "ashlar")
    box(gu1 - 0.03, gu1 + 0.33, 0.57, gn1, strip + 0.3, strip + 0.36, "coping")
    nb_u = 22.9 + 6.9 * (gn1 + 0.3) / 44.5 - 0.05
    box(gu1, nb_u, gn1, gn1 + 0.3, strip - 0.4, PADZ + 0.9, "ashlar")      # back of the NE strip, up to the courtyard
    box(gu1 - 0.03, nb_u, gn1 - 0.03, gn1 + 0.33, PADZ + 0.9, PADZ + 0.96, "coping")
    drape(gu0 - 1.0, gu0 - 0.3, lambda u: 1.6, lambda u: gn0, 0.0, "mulch", 3, 8, zfix=tier + 0.05)
    drape(gu1 + 0.3, 22.9, lambda u: arc_n(u, ARC["r"]) + 0.05, lambda u: 2.3, 0.05, "mulch", 4, 4)
    bt = FRONT["bank_top"]
    drape(0.3, s0 - 0.3, lambda u: 2.2, lambda u: bt, 0.05, "mulch", 24, 6)
    drape(s1 + 0.3, gu0 - 1.3, lambda u: 2.2, lambda u: bt, 0.05, "mulch", 36, 6)
    for x in (2.0, 5.4, 8.4, 11.6, 13.8):  # uplights grazing the street wall
        g0 = zg(x, 1.2)
        wall_up(x, 1.25, g0, 1.6, power=110)
    w1 = L["w1"]
    for x in (w1[0] + 0.9, w1[1] - 0.9):  # uplights on the SW street gable
        wall_up(x, w1[2] - 0.45, PADZ + 0.03, w1[2], power=200)
    e1 = L["e1"]
    for x in (e1[0] + 0.4, e1[1] - 0.4):  # and on the NE wing's ashlar piers from the drive
        wall_up(x, e1[2] - 0.4, GZ + 0.03, e1[2], power=160)
    e2 = L["e2"]
    wall_up((e2[0] + e2[1]) / 2, e2[2] - 0.6, 3.25, e2[2], power=160)   # E2 street gable, from the roof terrace

    R = ARC["r"]
    drape(-45, 68, lambda u: arc_n(u, R - 15.0), lambda u: arc_n(u, R - 4.0), 0.03, "asphalt", 150, 10)
    for r0, r1 in ((R - 4.0, R - 3.75), (R - 15.25, R - 15.0)):
        drape(-45, 68, lambda u, r0=r0: arc_n(u, r0), lambda u, r1=r1: arc_n(u, r1), 0.17, "kerb", 150, 1)
    drape(-45, 68, lambda u: arc_n(u, R - 16.7), lambda u: arc_n(u, R - 15.25), 0.1, "concrete", 150, 2)
    drape(-45, 68, lambda u: arc_n(u, R - 24.0), lambda u: arc_n(u, R - 16.7), 0.05, "lawn", 150, 4)
    drape(-45, gu0, lambda u: arc_n(u, R - 3.75), lambda u: arc_n(u, R - 2.3), 0.1, "concrete", 80, 2)
    drape(gu1, 68, lambda u: arc_n(u, R - 3.75), lambda u: arc_n(u, R - 2.3), 0.1, "concrete", 60, 2)
    drape(gu0, gu1, lambda u: arc_n(u, R - 3.75), lambda u: gn0, 0.06, "aggregate", 16, 16)
    drape(0.0, gu0 - 1.3, lambda u: arc_n(u, R) + 0.03, lambda u: 1.58, 0.06, "gravel", 40, 4)   # front bed inside the lot
    drape(gu0 - 1.3, gu0 - 0.3, lambda u: arc_n(u, R) + 0.03, lambda u: 0.95, 0.06, "gravel", 4, 2)
    drape(-45, gu0, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 0.03, 0.04, "lawn", 60, 2)  # nature strip
    drape(gu1, 68, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 0.03, 0.04, "lawn", 60, 2)
    drape(-45, 0.0, lambda u: arc_n(u, R) + 0.03, lambda u: arc_n(u, R) + 1.0, 0.04, "lawn", 30, 2)
    drape(22.9, 68, lambda u: arc_n(u, R) + 0.03, lambda u: arc_n(u, R) + 1.0, 0.04, "lawn", 30, 2)

    # --- Colorbond side fences (1.8 m, corrugated with capping) behind the front setback: beige NE, sage SW
    for (ua, na), (ub, nb), mat, inward in (((0.0, 0.0), (-5.1, 32.5), "fence_sw", 1), ((22.9, 0.0), (29.8, 44.5), "fence_ne", -1)):
        Lb = math.hypot(ub - ua, nb - na)
        du, dn = (ub - ua) / Lb, (nb - na) / Lb
        iu, in_ = dn * inward, -du * inward          # unit normal into the lot
        t = 6.0 / dn
        while t < Lb - 0.05:
            t1 = min(t + 2.4, Lb - 0.02)
            pa = (ua + du * t + iu * 0.04, na + dn * t + in_ * 0.04)
            pb = (ua + du * t1 + iu * 0.04, na + dn * t1 + in_ * 0.04)
            zs = []
            for p in (pa, pb):
                zi = zg(p[0] + iu * 0.5, p[1] + in_ * 0.5)
                zo = zg(p[0] - iu * 0.8, p[1] - in_ * 0.8)
                zs.append(max(zi, zo))
            z0 = min(zs) - 0.05
            z1 = max(zs) + 1.8
            faces(mat, [(pa[0], pa[1], z0), (pb[0], pb[1], z0), (pb[0], pb[1], z1), (pa[0], pa[1], z1),
                        (pa[0] - iu * 0.03, pa[1] - in_ * 0.03, z0), (pb[0] - iu * 0.03, pb[1] - in_ * 0.03, z0),
                        (pb[0] - iu * 0.03, pb[1] - in_ * 0.03, z1), (pa[0] - iu * 0.03, pa[1] - in_ * 0.03, z1)],
                  [(0, 1, 2, 3), (5, 4, 7, 6), (3, 2, 6, 7)])
            faces(mat, [(pa[0] + iu * 0.02, pa[1] + in_ * 0.02, z1), (pb[0] + iu * 0.02, pb[1] + in_ * 0.02, z1),
                        (pb[0] - iu * 0.05, pb[1] - in_ * 0.05, z1), (pa[0] - iu * 0.05, pa[1] - in_ * 0.05, z1),
                        (pa[0] + iu * 0.02, pa[1] + in_ * 0.02, z1 + 0.05), (pb[0] + iu * 0.02, pb[1] + in_ * 0.02, z1 + 0.05),
                        (pb[0] - iu * 0.05, pb[1] - in_ * 0.05, z1 + 0.05), (pa[0] - iu * 0.05, pa[1] - in_ * 0.05, z1 + 0.05)],
                  [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6)])
            cyl((pa[0] - iu * 0.015, pa[1] - in_ * 0.015, z0), (pa[0] - iu * 0.015, pa[1] - in_ * 0.015, z1 + 0.05), 0.035, 0.035, mat, 6)
            t = t1


# ------------------------------------------------------------------ terrain (pool sink at this design's pool)
def build_terrain():
    pu0, pu1, pn0, pn1 = L["pool"]
    v, uv, fcs = [], [], []
    NG = cr.NG
    for j in range(NG):
        for i in range(NG):
            gx, gy = i + 0.5, j + 0.5
            u, n = cr.to_frame(gx, gy)
            z = cr.H[j, i] - cr.FFL
            if pu0 - 0.25 < u < pu1 + 0.25 and pn0 - 0.25 < n < pn1 + 0.25:
                z = PADZ - 1.8
            v.append((u, n, z))
            uv.append((gx / NG, gy / NG))
    for j in range(NG - 1):
        for i in range(NG - 1):
            a = j * NG + i
            fcs.append((a, a + 1, a + NG + 1, a + NG))
    me = bpy.data.meshes.new("terrain")
    me.from_pydata(v, [], fcs)
    lay = me.uv_layers.new(name="UVMap")
    flat = []
    for f in fcs:
        for k in f:
            flat.extend(uv[k])
    lay.data.foreach_set("uv", flat)
    me.update()
    ob = bpy.data.objects.new("terrain", me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(MATS["terrain"])
    for p in me.polygons:
        p.use_smooth = True
    dem = np.array(Image.open(D3 / cr.S["dem_1km"]["file"]), dtype=float)
    S = cr.S
    e0, n1, cell, nd = S["dem_1km"]["e0"], S["dem_1km"]["n1"], S["dem_1km"]["cell"], dem.shape[0]
    v, fcs, inside = [], [], []
    for r_ in range(nd):
        for c_ in range(nd):
            gx, gy = e0 + (c_ + 0.5) * cell - S["E0"], n1 - (r_ + 0.5) * cell - S["N0"]
            u, n = cr.to_frame(gx, gy)
            v.append((u, n, dem[r_, c_] - cr.FFL - 0.15))
            inside.append(3 < gx < NG - 3 and 3 < gy < NG - 3)
    for r_ in range(nd - 1):
        for c_ in range(nd - 1):
            a = r_ * nd + c_
            q = (a, a + nd, a + nd + 1, a + 1)
            if all(inside[k] for k in q):
                continue
            fcs.append(q)
    me = bpy.data.meshes.new("backdrop")
    me.from_pydata(v, [], fcs)
    me.update()
    ob = bpy.data.objects.new("backdrop", me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(MATS["grass"])
    for p in me.polygons:
        p.use_smooth = True


# ------------------------------------------------------------------ garden
EUC = []


def olive(u, n, z, lit=False, h=None):
    tree((u, n, z), h or rr(3.3, 4.1), rr(1.5, 1.9), "olive", "darkbark", stems=3, cards_per=22, crown_flat=0.6)
    if lit:
        light("SPOT", (u + 0.6, n - 0.6, z + 0.1), 150, WARM, aim=(u, n, z + 3.2), spot=1.1, blend=0.9, size=0.03, name="olive_up")


def box_ball(u, n, z, r=None):
    r = r or rr(0.3, 0.42)
    bush((u, n, z + r * 0.75), r, "hedge", 16)


def hedge_row(ua, na, ub, nb, z, h=0.55, w=0.45):
    """clipped box hedge as a row of tight bush clusters"""
    Lh = math.hypot(ub - ua, nb - na)
    k = max(2, int(Lh / 0.45))
    for i in range(k + 1):
        t = i / k
        u, n = ua + (ub - ua) * t, na + (nb - na) * t
        bush((u, n, z + h * 0.55), w * 0.6, "hedge", 9)


def build_garden():
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
    s0, s1, sn0, sn1 = L["stair"]
    g = L["garage"]
    # front: uplit olives in the gravel verge bed, grey shrubs and boulders, box balls on the bank
    for u in (0.9, 6.8, 12.6):
        nn = (arc_n(u, ARC["r"]) + 1.55) / 2
        olive(u, nn, zg(u, nn), lit=True)
    for _ in range(46):
        u = rr(0.3, g[0] - 1.6)
        if s0 - 0.8 < u < s1 + 0.8:
            continue
        t = rr(0.25, 0.9)
        nn = arc_n(u, ARC["r"]) * (1 - t) + 1.5 * t
        r = rr(0.25, 0.42)
        bush((u, nn, zg(u, nn) + r * 0.55), r, "shrub" if rng.random() < 0.65 else "hedge", 14)
    for x in np.arange(0.9, g[0] - 1.8, 0.85):
        for nn in (2.75, 3.7):
            if not (s0 - 0.8 < x < s1 + 0.8):
                xx = x + (0.42 if nn > 3 else 0)
                if nn > 3:
                    bush((xx, nn, zg(xx, nn) + 0.35), rr(0.32, 0.42), "shrub", 14)
                else:
                    box_ball(xx, nn, zg(xx, nn))
    for nn in np.arange(sn1 - 3.2, sn1 + 0.1, 0.7):  # box balls along the entry stair
        for x in (s0 - 0.55, s1 + 0.55):
            box_ball(x, nn, zg(x, nn), 0.28)
    # entry forecourt: box parterre beds, two olives either side of the path
    ln0 = L["link"][2]
    for (a, b, c, d) in ((7.95, 10.0, 7.0, 12.4), (14.0, 15.45, 7.0, 12.4)):
        for (ua, na, ub, nb) in ((a, c, b, c), (b, c, b, d), (b, d, a, d), (a, d, a, c)):
            hedge_row(ua, na, ub, nb, PADZ, 0.5, 0.42)
    olive(9.0, 9.4, PADZ, lit=True, h=3.6)
    olive(14.7, 9.4, PADZ, lit=True, h=3.4)
    w1 = L["w1"]
    for x in list(np.arange(w1[0] + 0.3, L["bay"][0] - 0.2, 0.55)) + list(np.arange(L["bay"][1] + 0.3, w1[1], 0.55)):
        box_ball(x, w1[2] - 0.4, PADZ, 0.3)                                # box hedge screening the bay window
    olive(5.9, 4.4, zg(5.9, 4.4), lit=True, h=3.8)                          # olive on the bank in front of the bay
    # courtyard: olives in the lawn strips, box balls along the glass, hedges on the terrace wall top
    olive(-1.9, 23.0, PADZ, lit=True, h=3.0)
    olive(24.4, 19.5, PADZ, lit=True, h=3.0)
    olive(-1.8, 28.9, PADZ, lit=True, h=3.2)
    for a, b in ((-3.9, 16.0), (18.8, 28.4)):
        hedge_row(a, terrace_n(a) + 0.65, b, terrace_n(b) + 0.65, UPZ, 0.6, 0.5)
    # upper garden terrace: olive grove in lawn, gravel walk, a bench
    for x in np.arange(-2.0, 28.0, 4.2):
        if 15.0 < x < 19.5:
            continue
        nn = terrace_n(x) + 1.4                    # olive grove in the lawn between the terrace wall and the walk
        if x > -3.0 and nn < 32.5 + (x + 5.1) * 0.344 - 2.0:
            olive(x, nn, UPZ, lit=True, h=rr(3.0, 3.6))
    # tall hedges inside the side boundaries for privacy (behind the boundary walls)
    for (ua, na), (ub, nb), inset in (((0.0, 0.0), (-5.1, 32.5), 0.75), ((22.9, 0.0), (29.8, 44.5), -0.75)):
        for k in range(40):
            t = 0.2 + k / 40 * 0.62
            u, nn = ua + (ub - ua) * t + inset, na + (nb - na) * t
            if inset < 0 and nn < L["garage"][3] + 1.0:
                continue
            if inset < 0 and L["cabana"][2] - 0.5 < nn < L["cabana"][3] + 0.5:
                continue
            z0 = UPZ if nn >= terrace_n(u) + 0.3 else PADZ
            for zz in (0.7, 1.5):
                bush((u, nn, z0 + zz), rr(0.6, 0.8), "hedge", 14)


GLB_EXTRA_TEX = {"ashlar": ("ashlar.png", 0.25), "soffit": ("boards.png", 0.25), "fence_ne": ("fence_ne.png", 0.25),
                 "fence_sw": ("fence_sw.png", 0.25)}
GLB_EXTRA_COL = {"darkglass": ("#16191b", 0.08, 0.3), "gravel": ("#a89a85", 1.0, 0), "plaster": ("#ece3d4", 0.85, 0)}

VIEWS = {
    "street": dict(eye=(10.8, -14.6, 1.7), at=(11.8, 12.0, 3.4), lens=24),
    "courtyard": dict(eye=(4.0, 46.0, 6.3), at=(14.0, 21.0, 1.8), lens=22),
    "aerial": dict(eye=(48.0, -24.0, 42.0), at=(12.0, 20.0, 0.0), lens=30),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="street")
    ap.add_argument("--scale", type=int, default=50)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "designs" / "stone_glass_street.png"))
    ap.add_argument("--exposure", type=float, default=0.9)
    ap.add_argument("--export-glb", default="")
    a = ap.parse_args()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    cr.make_textures()
    textures()
    MATS.update({
        "ashlar": img_mat("ashlar", "ashlar.png", 0.25, 0.88, bump=0.35),
        "coping": img_mat("coping", "stone.png", 0.3, 0.8, bump=0.2, tint=(1.1, 1.08, 1.04)),
        "soffit": img_mat("soffit", "boards.png", 0.25, 0.55, bump=0.1),
        "fence_ne": img_mat("fence_ne", "fence_ne.png", 0.25, 0.45, bump=0.3, metallic=0.2),
        "fence_sw": img_mat("fence_sw", "fence_sw.png", 0.25, 0.45, bump=0.3, metallic=0.2),
        "paving": img_mat("paving", "paving.png", 0.25, 0.8, bump=0.1, tint=(1.02, 0.98, 0.92)),
        "roof_x": seam_roof("roof_x", "X"), "roof_y": seam_roof("roof_y", "Y"),
        "black": flat_mat("black", srgb("#141516"), 0.35, 0.4),
        "glass": thin_glass("glass"),
        "darkglass": thin_glass("darkglass", tint=(0.1, 0.1, 0.1)),
        "slab": flat_mat("slab", srgb("#e8e2d6"), 0.7),
        "plaster": flat_mat("plaster", srgb("#ece3d4"), 0.85),
        "floor": flat_mat("floor", srgb("#8a6646"), 0.35),
        "fabric": flat_mat("fabric", srgb("#d9cfbf"), 0.95),
        "linen": flat_mat("linen", srgb("#f1ece3"), 0.95),
        "walnut": flat_mat("walnut", srgb("#4a3424"), 0.45),
        "rug": flat_mat("rug", srgb("#b8aa95"), 1.0),
        "art": flat_mat("art", srgb("#3d4a4f"), 0.6),
        "lamp": flat_mat("lamp", (1, 1, 1), 0.5, emit=WARM, strength=14),
        "bulb": flat_mat("bulb", (1, 1, 1), 0.5, emit=WARM, strength=25),
        "lawn": noisy_mat("lawn", "#4f6e2f", "#7a9446", 30, 0.95, 0.3),
        "mulch": noisy_mat("mulch", "#3d2f24", "#6b5644", 40, 1.0, 0.4),
        "gravel": noisy_mat("gravel", "#8a7d6a", "#c9bba2", 70, 1.0, 0.5),
        "concrete": noisy_mat("concrete", "#aaa499", "#c9c3b8", 8, 0.9, 0.05),
        "kerb": flat_mat("kerb", srgb("#b6b0a5"), 0.9),
        "asphalt": noisy_mat("asphalt", "#2a2b2d", "#4a4a4b", 90, 0.9, 0.3),
        "aggregate": noisy_mat("aggregate", "#8f877a", "#c2b9aa", 70, 0.9, 0.25),
        "timber": noisy_mat("timber", "#7a5232", "#9c6e44", 3, 0.6, 0.1),
        "door": noisy_mat("door", "#6e4626", "#8f6038", 4, 0.45, 0.05),
        "pooltile": flat_mat("pooltile", srgb("#8fd0e3"), 0.3),
        "bark": noisy_mat("bark", "#8a8070", "#b0a796", 6, 0.85, 0.2),
        "darkbark": flat_mat("darkbark", srgb("#4d443b"), 0.9),
        "euc": leaf_mat("euc"), "olive": leaf_mat("olive"), "shrub": leaf_mat("shrub"), "hedge": leaf_mat("hedge"),
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

    build_terrain()
    build_house()
    build_landscape()
    build_garden()
    cr.flush()
    if a.export_glb:
        fc.GLB_TEX.update(GLB_EXTRA_TEX)
        fc.GLB_COL.update(GLB_EXTRA_COL)
        fc.export_glb(a.export_glb)
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
