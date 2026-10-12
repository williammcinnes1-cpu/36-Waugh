"""Cycles render of the "Modern Black Barnhouse" concept (ZEB-style black board-and-batten barns).

    python black_barn_cycles.py --view street|courtyard|aerial --scale 50 --samples 32 --out x.png
                                [--export-mesh] [--export-glb data/3d/designs/black_barn.glb]

Run scripts/designs/black_barn_site.py ground first: it writes this design's earthworks, pad outline and
finish surfaces into data/3d/house_massing.json and refreshes render_scene.json, which cycles_render.py
reads at import. Reuses the shared terrain, street, vegetation helpers and materials from cycles_render.py.
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

D3 = ROOT / "data" / "3d"
LAY = json.loads((D3 / "house_massing.json").read_text())
L = LAY.get("layout", {})
FRONT = LAY.get("front", {})
PAD_OUTLINE = LAY.get("pad_outline", [])
SURF = LAY.get("surfaces", [])
FFL = cr.FFL
UPZ = L.get("upper_z", 158.65) - FFL      # upper garden terrace above floor level
rng = random.Random(29)
rr = rng.uniform

T = 0.3            # external wall thickness (cladding + lining)
RF = L.get("roofs", {})
EAVE_B = RF.get("main", {}).get("eave", 3.2)                                                                 # flat link
EAVE_A, PITCH_A = RF.get("wing", {}).get("eave", 3.2), math.radians(RF.get("wing", {}).get("pitch", 42))    # garage wing
EAVE_C, PITCH_C = RF.get("rear", {}).get("eave", 3.0), math.radians(RF.get("rear", {}).get("pitch", 40))    # family wing
EAVE_O, PITCH_O = RF.get("outdoor", {}).get("eave", 2.75), math.radians(RF.get("outdoor", {}).get("pitch", 38))
O = 0.1            # crisp barn edges: almost no eaves or rakes
COOL = (0.55, 0.85, 1.0)


def n_edge(u):
    u0, n0, k = L["edge"]
    return n0 + k * (u - u0)


# ------------------------------------------------------------------ textures (PIL)
def textures():
    R = random.Random(5)
    g = np.random.default_rng(5)
    # matte black board-and-batten, 1024 px = 4 m, battens every 0.4 m
    im = Image.new("RGB", (1024, 1024), (40, 41, 43))
    d = ImageDraw.Draw(im)
    for x in range(0, 1024, 102):
        d.rectangle([x, 0, x + 12, 1024], fill=(52, 53, 55))
        d.rectangle([x + 13, 0, x + 16, 1024], fill=(22, 22, 24))
        d.rectangle([x - 1, 0, x, 1024], fill=(58, 59, 61))
    arr = np.array(im).astype(np.int16) + g.integers(-3, 4, (1024, 1024))[..., None]
    arr += (g.normal(0, 1.5, (1, 1024)).repeat(1024, 0))[..., None].astype(np.int16)
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(TEX / "bb_clad.png")
    # rough limewashed brick, 1024 px = 2 m
    im = Image.new("RGB", (1024, 1024), (196, 192, 184))
    d = ImageDraw.Draw(im)
    for row, y in enumerate(range(0, 1024, 39)):
        off = R.randint(0, 59) if row % 2 else 0
        for x in range(-off, 1024, 118):
            l = R.uniform(-10, 6)
            worn = R.random() < 0.12
            base = (176, 160, 146) if worn else (238, 236, 230)
            d.rectangle([x + R.randint(2, 5), y + R.randint(2, 5), x + 113 + R.randint(0, 3), y + 35 + R.randint(0, 2)],
                        fill=(int(base[0] + l), int(base[1] + l), int(base[2] + l)))
    arr = np.array(im).astype(np.int16)
    blot = np.array(Image.fromarray(g.integers(0, 255, (64, 64)).astype(np.uint8)).resize((1024, 1024), Image.BICUBIC)).astype(np.int16)
    arr += ((blot - 128) * 0.12).astype(np.int16)[..., None] + g.integers(-9, 9, (1024, 1024))[..., None]
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.1)).save(TEX / "limewash.png")
    # warm cedar lining boards, 1024 px = 4 m, 90 mm boards
    im = Image.new("RGB", (1024, 1024), (70, 46, 30))
    d = ImageDraw.Draw(im)
    for y in range(0, 1024, 23):
        l = R.uniform(-22, 18)
        d.rectangle([0, y + 2, 1024, y + 22], fill=(int(178 + l), int(124 + l * 0.8), int(80 + l * 0.6)))
        for _ in range(10):
            yy = y + R.randint(3, 20)
            x0 = R.randint(0, 1024)
            d.line([x0, yy, x0 + R.randint(80, 400), yy], fill=(int(150 + l), int(100 + l * 0.8), int(62 + l * 0.6)), width=1)
    arr = np.array(im).astype(np.int16) + g.integers(-5, 5, (1024, 1024))[..., None]
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(TEX / "cedar.png")
    # ornamental grass clump (fountain of blades with plumes), RGBA, base at the bottom centre
    im = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for _ in range(150):
        a = R.gauss(0, 0.5)
        ln = R.uniform(260, 470)
        bend = R.uniform(0.2, 0.6) * (1 if a >= 0 else -1)
        x, y = 256 + R.uniform(-30, 30), 511
        l = R.uniform(-18, 18)
        straw = R.random() < 0.35
        col = (int(170 + l), int(152 + l), int(98 + l)) if straw else (int(110 + l), int(122 + l), int(70 + l * 0.6))
        pts = []
        for k in range(12):
            t = k / 11
            ang = a + bend * t * t
            pts.append((x + math.sin(ang) * ln * t, y - math.cos(ang) * ln * t))
        for k in range(11):
            w = max(1, int(5 * (1 - k / 11)))
            d.line([pts[k], pts[k + 1]], fill=(*col, 255), width=w)
        if R.random() < 0.45:
            tx, ty = pts[-1]
            for _ in range(26):
                px_, py_ = tx + R.uniform(-9, 9), ty + R.uniform(-4, 40)
                c = R.uniform(-12, 10)
                d.ellipse([px_ - 3, py_ - 5, px_ + 3, py_ + 5], fill=(int(226 + c), int(208 + c), int(170 + c), 255))
    im.save(TEX / "leaf_plume.png")
    # rough grey flagstone (from the shared sandstone, desaturated)
    st = np.array(Image.open(TEX / "stone.png").convert("L")).astype(np.int16)
    st = (st * 0.62 + 30).clip(0, 255).astype(np.uint8)
    Image.merge("RGB", [Image.fromarray(st)] * 3).save(TEX / "bluestone.png")


# ------------------------------------------------------------------ wall helpers
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
    cuts = sorted({n0, n1, *[x for o in openings for x in o[:2]]})
    for a, b in zip(cuts, cuts[1:]):
        hits = [o for o in openings if o[0] <= a and b <= o[1]]
        if not hits:
            box(uf, ub, a, b, z0, z1, mat)
            continue
        o = hits[0]
        box(uf, ub, a, b, z0, o[2], mat)
        box(uf, ub, a, b, o[3], z1, mat)


def wu(u0, u1, n_out, inward, z0, z1, ops, ext="clad", lining="plaster"):
    """external wall in the u direction: cladding outside, a thin lining inside (inward = +1 or -1 in n)"""
    a, b, c = n_out, n_out + inward * (T - 0.03), n_out + inward * T
    wall_u(u0, u1, min(a, b), max(a, b), z0, z1, ops, ext)
    wall_u(u0, u1, min(b, c), max(b, c), max(z0, -0.05), z1, ops, lining)


def wn(n0, n1, u_out, inward, z0, z1, ops, ext="clad", lining="plaster"):
    a, b, c = u_out, u_out + inward * (T - 0.03), u_out + inward * T
    wall_n(n0, n1, min(a, b), max(a, b), z0, z1, ops, ext)
    wall_n(n0, n1, min(b, c), max(b, c), max(z0, -0.05), z1, ops, lining)


def frame_rect(a, b, zb, zt, plane, axis, cols, rows, w=0.06, depth=0.07, mat="black"):
    """black window frame + glazing bars in the plane n=plane (axis 'u') or u=plane (axis 'n')"""
    def bar(p0, p1, z0, z1):
        lo0, hi0 = (p0 - w / 2, p0 + w / 2) if p0 == p1 else (p0, p1)
        zl, zh = (z0 - w / 2, z0 + w / 2) if z0 == z1 else (z0, z1)
        if axis == "u":
            box(lo0, hi0, plane - depth / 2, plane + depth / 2, zl, zh, mat)
        else:
            box(plane - depth / 2, plane + depth / 2, lo0, hi0, zl, zh, mat)
    for k in range(cols + 1):
        x = a + (b - a) * k / cols
        bar(x, x, zb, zt)
    for k in range(rows + 1):
        z = zb + (zt - zb) * k / rows
        bar(a, b, z, z)


def glaze_u(a, b, zb, zt, n_glass, cols, rows=1, w=0.06):
    poly([(a, n_glass, zb), (b, n_glass, zb), (b, n_glass, zt), (a, n_glass, zt)], "glass")
    frame_rect(a, b, zb, zt, n_glass, "u", cols, rows, w=w, depth=0.09)


def glaze_n(a, b, zb, zt, u_glass, cols, rows=1, w=0.06):
    poly([(u_glass, a, zb), (u_glass, b, zb), (u_glass, b, zt), (u_glass, a, zt)], "glass")
    frame_rect(a, b, zb, zt, u_glass, "n", cols, rows, w=w, depth=0.09)


def deep_window_u(a, b, zb, zt, n_face, outward, cols=1, rows=1):
    """modest window set deep in the wall, with a projecting black steel surround (street side)"""
    n_glass = n_face - outward * (T - 0.06)
    glaze_u(a, b, zb, zt, n_glass, cols, rows, w=0.05)
    n0, n1 = sorted((n_face - outward * 0.04, n_face + outward * 0.32))
    box(a - 0.06, b + 0.06, n0, n1, zt, zt + 0.06, "black")
    box(a - 0.06, b + 0.06, n0, n1, zb - 0.06, zb, "black")
    box(a - 0.06, a, n0, n1, zb, zt, "black")
    box(b, b + 0.06, n0, n1, zb, zt, "black")
    # warm cedar reveal inside the steel box
    m0, m1 = sorted((n_glass, n_face))
    box(a, b, m0, m1, zb - 0.01, zb, "cedar")


def deep_window_n(a, b, zb, zt, u_face, outward, cols=1, rows=1):
    u_glass = u_face - outward * (T - 0.06)
    glaze_n(a, b, zb, zt, u_glass, cols, rows, w=0.05)
    u0, u1 = sorted((u_face - outward * 0.04, u_face + outward * 0.32))
    box(u0, u1, a - 0.06, b + 0.06, zt, zt + 0.06, "black")
    box(u0, u1, a - 0.06, b + 0.06, zb - 0.06, zb, "black")
    box(u0, u1, a - 0.06, a, zb, zt, "black")
    box(u0, u1, b, b + 0.06, zb, zt, "black")


def bay_u(a, b, zb, zt, n_face, depth, cols=3):
    """projecting black steel window bay on a street-facing (-n) wall, cedar-lined, glass at the front"""
    nf = n_face - depth
    glaze_u(a + 0.06, b - 0.06, zb, zt, nf + 0.05, cols, 1, w=0.05)
    box(a, b, nf, n_face, zt, zt + 0.12, "black")
    box(a, b, nf, n_face, zb - 0.12, zb, "black")
    box(a, a + 0.06, nf, n_face, zb, zt, "black")
    box(b - 0.06, b, nf, n_face, zb, zt, "black")
    box(a + 0.06, b - 0.06, nf + 0.1, n_face, zb - 0.01, zb, "cedar")
    box(a + 0.06, b - 0.06, nf + 0.1, n_face, zt, zt + 0.01, "cedar")


# ------------------------------------------------------------------ crisp barn roofs (no eaves) with cedar-lined vaults
def roof_u(u0, u1, n0, n1, ze, pitch, lining=None, o=O):
    """ridge parallel to u over the middle of n0..n1; seams run down-slope"""
    nm = (n0 + n1) / 2
    tn = math.tan(pitch)
    zr = ze + (n1 - n0) / 2 * tn
    dz = o * tn
    slab((u0 - o, n0 - o, ze - dz), (u1 + o, n0 - o, ze - dz), (u1 + o, nm, zr + 0.02), (u0 - o, nm, zr + 0.02), 0.09, "roof_x")
    slab((u1 + o, n1 + o, ze - dz), (u0 - o, n1 + o, ze - dz), (u0 - o, nm, zr + 0.02), (u1 + o, nm, zr + 0.02), 0.09, "roof_x")
    box(u0 - o, u1 + o, nm - 0.07, nm + 0.07, zr + 0.06, zr + 0.16, "black")                      # ridge cap
    for u in (u0 - o, u1 + o):                                                                    # thin black barge flashing
        for na, nb_ in ((n0 - o, nm), (n1 + o, nm)):
            za = ze - dz
            slab((u, na, za - 0.06), (u, nb_, zr - 0.04), (u, nb_, zr + 0.17), (u, na, za + 0.14), 0.025 if u > u0 else -0.025, "black")
    if lining:
        la, lb = lining
        for nf, sgn in ((n0 + T, 1), (n1 - T, -1)):
            zf = ze + T * tn - 0.22
            slab((la, nf, zf), (lb, nf, zf), (lb, nm, zr - 0.22), (la, nm, zr - 0.22), -0.03 if sgn > 0 else 0.03, "cedar")
    return zr


def roof_n(n0, n1, u0, u1, ze, pitch, lining=None, o=O, rake0=True, rake1=True):
    """ridge parallel to n; seams vary along n"""
    um = (u0 + u1) / 2
    tn = math.tan(pitch)
    zr = ze + (u1 - u0) / 2 * tn
    dz = o * tn
    a0, a1 = n0 - (o if rake0 else 0), n1 + (o if rake1 else 0)
    slab((u0 - o, a1, ze - dz), (u0 - o, a0, ze - dz), (um, a0, zr + 0.02), (um, a1, zr + 0.02), 0.09, "roof_y")
    slab((u1 + o, a0, ze - dz), (u1 + o, a1, ze - dz), (um, a1, zr + 0.02), (um, a0, zr + 0.02), 0.09, "roof_y")
    box(um - 0.07, um + 0.07, a0, a1, zr + 0.06, zr + 0.16, "black")
    for n, on in ((a0, rake0), (a1, rake1)):
        if not on:
            continue
        for ua, ub_ in ((u0 - o, um), (u1 + o, um)):
            za = ze - dz
            slab((ua, n, za - 0.06), (ub_, n, zr - 0.04), (ub_, n, zr + 0.17), (ua, n, za + 0.14), 0.025 if n > n0 else -0.025, "black")
    if lining:
        la, lb = lining
        for uf, sgn in ((u0 + T, 1), (u1 - T, -1)):
            zf = ze + T * tn - 0.22
            slab((uf, lb, zf), (uf, la, zf), (um, la, zr - 0.22), (um, lb, zr - 0.22), -0.03 if sgn > 0 else 0.03, "cedar")
    return zr


def ring_n(outer, inner, u0, u1, mat):
    """wall in the n-z plane (between u0 and u1) with an opening: outer/inner outlines of (n, z) with equal counts"""
    k = len(outer)
    v = [(u0, n, z) for n, z in outer] + [(u0, n, z) for n, z in inner] + [(u1, n, z) for n, z in outer] + [(u1, n, z) for n, z in inner]
    fcs = []
    for i in range(k):
        j = (i + 1) % k
        fcs += [(i, j, k + j, k + i), (2 * k + j, 2 * k + i, 3 * k + i, 3 * k + j),
                (i, 2 * k + i, 2 * k + j, j), (k + j, 3 * k + j, 3 * k + i, k + i)]
    faces(mat, v, fcs)


def gable_n_wall(n0, n1, uf, ub, ze, pitch, mat):
    """triangular gable (ridge along u) in the n-z plane between u=uf..ub"""
    nm = (n0 + n1) / 2
    zr = ze + (n1 - n0) / 2 * math.tan(pitch)
    faces(mat, [(uf, n0, ze), (uf, n1, ze), (uf, nm, zr), (ub, n0, ze), (ub, n1, ze), (ub, nm, zr)],
          [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)])


def lantern_sconce(x, n, z, outward_n=None, outward_u=None, name="sconce"):
    """slim black cylinder wall light washing up and down"""
    if outward_n is not None:
        c = (x, n + outward_n * 0.09)
        box(x - 0.05, x + 0.05, *sorted((n, n + outward_n * 0.14)), z - 0.18, z + 0.18, "black")
        light("SPOT", (c[0], c[1] + outward_n * 0.05, z - 0.2), 22, WARM, aim=(c[0], c[1] + outward_n * 0.3, z - 3), spot=1.2, blend=0.8, size=0.02, name=name)
        light("SPOT", (c[0], c[1] + outward_n * 0.05, z + 0.2), 14, WARM, aim=(c[0], c[1] + outward_n * 0.3, z + 3), spot=1.0, blend=0.8, size=0.02, name=name)
    else:
        c = (x + outward_u * 0.09, n)
        box(*sorted((x, x + outward_u * 0.14)), n - 0.05, n + 0.05, z - 0.18, z + 0.18, "black")
        light("SPOT", (c[0] + outward_u * 0.05, c[1], z - 0.2), 22, WARM, aim=(c[0] + outward_u * 0.3, c[1], z - 3), spot=1.2, blend=0.8, size=0.02, name=name)
        light("SPOT", (c[0] + outward_u * 0.05, c[1], z + 0.2), 14, WARM, aim=(c[0] + outward_u * 0.3, c[1], z + 3), spot=1.0, blend=0.8, size=0.02, name=name)


def room_light(u0, u1, n0, n1, z, power):
    light("AREA", ((u0 + u1) / 2, (n0 + n1) / 2, z - 0.05), power, WARM, aim=((u0 + u1) / 2, (n0 + n1) / 2, -5),
          size=min(u1 - u0, n1 - n0) * 0.6, name="room")


def vault_light(u0, u1, n0, n1, z, power):
    """soft fill under a vaulted ceiling (a point light, so nothing pokes through the sloping lining)"""
    light("POINT", ((u0 + u1) / 2, (n0 + n1) / 2, z), power, WARM, size=0.6, name="room")


def pendant(x, n, z, strength=50, r=0.22):
    cyl((x, n, z + 0.15), (x, n, z + 2.4), 0.006, 0.006, "black")
    cyl((x, n, z), (x, n, z + 0.18), r, 0.05, "black", sides=12)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, location=(x, n, z + 0.02))
    bpy.context.active_object.data.materials.append(MATS["bulb"])
    light("POINT", (x, n, z - 0.1), strength, WARM, size=0.08, name="pendant")


def downlight(x, n, z, power=30, name="soffit"):
    cyl((x, n, z - 0.01), (x, n, z), 0.05, 0.05, "bulb", sides=10)
    light("SPOT", (x, n, z - 0.03), power, WARM, aim=(x, n, z - 5), spot=1.6, blend=0.9, size=0.02, name=name)


def vcard(c, w, h, ang, mat):
    """vertical textured card standing on c (u, n, z)"""
    dx, dn = math.cos(ang) * w / 2, math.sin(ang) * w / 2
    u, n, z = c
    faces(mat, [(u - dx, n - dn, z), (u + dx, n + dn, z), (u + dx, n + dn, z + h), (u - dx, n - dn, z + h)], [(0, 1, 2, 3)],
          [[(0, 0), (1, 0), (1, 1), (0, 1)]])


def grass(u, n, z, h=None, w=None):
    h = h or rr(0.8, 1.25)
    w = w or h * rr(1.0, 1.3)
    a0 = rr(0, 3.14)
    for k in range(3):
        vcard((u, n, z - 0.05), w, h, a0 + k * math.pi / 3, "plume")


# ------------------------------------------------------------------ the house
def build_house():
    mu0, mu1, mn0, mn1 = L["main"]
    wu0_, wu1_, wn0, wn1 = L["wing"]
    ru0, ru1, rn0, rn1 = L["rear"]
    eu0, eu1, en0, en1 = L["entry"]
    gu0, gu1, gn0, gn1 = L["garage"]
    zb = -0.45   # wall base below the gravel

    uu0, uu1, un0, un1 = L["upper"]
    for u0, u1, n0, n1 in ((mu0, mu1, mn0, mn1), (wu0_, wu1_, gn1, wn1), (ru0, ru1, rn0, rn1), (uu0, uu1, un0, un1)):
        box(u0 + 0.1, u1 - 0.1, n0 + 0.1, n1 - 0.1, -0.25, 0.0, "floor")
    box(wu0_ + 0.1, wu1_ - 0.1, wn0 + 0.1, gn1, -0.3, 0.0, "floor")     # suspended slab over the garage
    P = L["projections"]

    # ---------------- two-storey barn (SW front): living below, main bedroom + retreat above
    RU = L["roofs"]["upper"]
    EU, PU, FU = RU["eave"], math.radians(RU["pitch"]), RU["floor"]
    tnu = math.tan(PU)
    unm = (un0 + un1) / 2
    zr_u = EU + (un1 - un0) / 2 * tnu
    ba = P["bay_main"]
    w2 = (ba[0], ba[1], 0.55, 2.45)                                    # the one major ground-floor street window: a bay
    wu(uu0, uu1, un0, 1, zb, FU, [w2])
    bay_u(*w2, un0, un0 - ba[2], cols=3)
    du0, du1 = L["roofs"]["dormer"]["u"]
    wu(uu0, du0, un0, 1, FU, EU, [])
    wu(du1, uu1, un0, 1, FU, EU, [])
    # street wall dormer: generous upper window with a Juliet balcony (main bedroom looks over the street, SSE)
    pd = math.radians(L["roofs"]["dormer"]["pitch"])
    dm = (du0 + du1) / 2
    zd = EU + (du1 - du0) / 2 * math.tan(pd)
    dwin = [(du0 + 0.3, FU + 0.1), (du1 - 0.3, FU + 0.1), (du1 - 0.3, EU + 0.55), (dm, EU + 0.95), (du0 + 0.3, EU + 0.55)]
    ring([(du0, FU), (du1, FU), (du1, EU), (dm, zd), (du0, EU)], dwin, un0, un0 + T, "clad")
    poly([(x, un0 + 0.15, z) for x, z in dwin], "glass")
    for x in (du0 + 0.3, dm - 0.45, dm + 0.45, du1 - 0.3):
        frac = 1 - abs(x - dm) / (dm - du0 - 0.3)
        box(x - 0.04, x + 0.04, un0 + 0.1, un0 + 0.2, FU + 0.1, EU + 0.55 + 0.4 * frac, "black")
    box(du0 + 0.3, du1 - 0.3, un0 + 0.1, un0 + 0.2, FU + 0.1, FU + 0.16, "black")
    box(du0 + 0.3, du1 - 0.3, un0 + 0.1, un0 + 0.2, EU - 0.02, EU + 0.04, "black")
    for k in range(12):                                                 # Juliet balcony rail
        x = du0 + 0.35 + (du1 - du0 - 0.7) * k / 11
        box(x - 0.012, x + 0.012, un0 - 0.14, un0 - 0.11, FU + 0.1, FU + 1.05, "black")
    box(du0 + 0.25, du1 - 0.25, un0 - 0.17, un0 - 0.08, FU + 1.05, FU + 1.1, "black")
    roof_n(un0, un0 + (zd - EU) / tnu + 0.3, du0, du1, EU, pd, rake1=False)
    # SW gable end with the chimney, small windows
    wn(un0, un1, uu0, 1, zb, EU, [(un0 + 1.5, un0 + 2.5, 1.0, 2.2), (un1 - 2.5, un1 - 1.5, FU + 0.9, FU + 1.9)])
    deep_window_n(un0 + 1.5, un0 + 2.5, 1.0, 2.2, uu0, -1)
    deep_window_n(un1 - 2.5, un1 - 1.5, FU + 0.9, FU + 1.9, uu0, -1)
    gable_n_wall(un0, un1, uu0, uu0 + T, EU, PU, "clad")
    wu(uu0, uu1, un1, -1, zb, EU, [])
    # NE gable: the upper floor looks east / south-east over the link roof; glazed, with fixed cedar louvres on the
    # north half (towards the NE neighbour's yard) and clear glass to the south-east

    def zg_u(n):
        return EU + min(n - un0, un1 - n) * tnu
    gin = [(un0 + 0.45, FU + 0.2), (un1 - 0.45, FU + 0.2), (un1 - 0.45, zg_u(un1 - 0.45) - 0.35), (unm, zr_u - 0.55),
           (un0 + 0.45, zg_u(un0 + 0.45) - 0.35)]
    gout = [(un0, FU - 0.05), (un1, FU - 0.05), (un1, EU), (unm, zr_u), (un0, EU)]
    ring_n(gout, gin, uu1 - T, uu1, "clad")
    ug = uu1 - 0.12
    poly([(ug, n, z) for n, z in gin], "glass")
    for k in range(7):
        n = gin[0][0] + (gin[1][0] - gin[0][0]) * k / 6
        box(ug - 0.05, ug + 0.05, n - 0.04, n + 0.04, FU + 0.2, zg_u(n) - 0.35 - (0.2 if abs(n - unm) < 0.1 else 0), "black")
    box(ug - 0.05, ug + 0.05, gin[0][0], gin[1][0], FU + 0.2, FU + 0.27, "black")
    box(ug - 0.05, ug + 0.05, gin[0][0], gin[1][0], EU - 0.04, EU + 0.03, "black")
    for nn in np.arange(unm + 0.15, un1 - 0.45, 0.16):                  # fixed vertical louvres (privacy to the NE)
        top = min(EU + 1.2, zg_u(nn) - 0.4)
        slab((uu1 + 0.05, nn, FU + 1.5), (uu1 + 0.05, nn + 0.02, FU + 1.5), (uu1 + 0.05, nn + 0.02, top), (uu1 + 0.05, nn, top), 0.18, "cedar")
    box(uu1 + 0.03, uu1 + 0.25, unm + 0.1, un1 - 0.45, FU + 1.45, FU + 1.5, "black")
    zr_u = roof_u(uu0, uu1, un0, un1, EU, PU, lining=(uu0 + T, uu1 - T))
    # upper floor slab, rooms
    box(uu0 + T, uu1 - T, un0 + T, un1 - T, FU - 0.25, FU, "floor")
    room_light(uu0 + 0.6, uu1 - 0.6, un0 + 0.6, un1 - 0.6, FU - 0.25, 900)
    vault_light(uu0 + 1.2, uu1 - 1.2, un0 + 1.2, un1 - 1.2, zr_u - 1.4, 400)
    box(2.4, 4.6, un1 - 3.6, un1 - 1.4, FU, FU + 0.55, "linen")             # bed
    box(2.3, 4.7, un1 - 1.45, un1 - 1.3, FU, FU + 1.2, "walnut")
    box(5.6, 6.9, un0 + 0.8, un0 + 1.6, FU, FU + 0.42, "fabric")             # retreat chair at the east glass
    pendant(3.5, un1 - 2.5, FU + 2.2, 30, r=0.2)
    box(1.6, 3.2, unm - 1.0, unm + 0.6, 0.0, 0.01, "rug")
    box(1.8, 3.0, un0 + 2.4, un0 + 4.4, 0, 0.42, "fabric")                   # ground-floor living
    pendant(4.0, unm, 2.4, 35, r=0.3)
    # tall limewashed brick chimney on the SW gable
    cu0, cu1, cn0, cn1 = L["chimney"]
    box(cu0, cu1, cn0, cn1, -0.6, zr_u + 1.0, "limewash")
    box(cu0 - 0.05, cu1 + 0.05, cn0 - 0.05, cn1 + 0.05, zr_u + 1.0, zr_u + 1.08, "black")
    box(cu0 + 0.3, cu1 - 0.3, cn0 + 0.5, cn1 - 0.5, zr_u + 1.08, zr_u + 1.35, "black")
    box(uu0 + T, uu0 + T + 0.05, cn0 + 0.35, cn1 - 0.35, 0.4, 1.1, "black")
    box(uu0 + T + 0.02, uu0 + T + 0.06, cn0 + 0.45, cn1 - 0.45, 0.45, 0.65, "fire")
    light("POINT", (uu0 + T + 0.4, (cn0 + cn1) / 2, 0.6), 60, (1.0, 0.45, 0.15), size=0.2, name="fireplace")

    # ---------------- flat-roofed link: entry + living, north glazing; upper balcony on its roof
    EB = EAVE_B
    w1 = (mu0 + 0.8, mu0 + 1.8, 0.9, 2.3)
    wu(mu0, eu0, mn0, 1, zb, EB, [w1])
    deep_window_u(*w1, mn0, -1)
    box(eu0, mu1, mn0, mn0 + 0.3, 2.75, EB, "clad")                   # band over the entry recess
    big = (ru1 + 0.25, mu1 - 0.25, 0.0, 2.75)
    wu(ru1, mu1, mn1, -1, zb, EB, [big])
    glaze_u(big[0], big[1], 0, 2.75, mn1 - 0.12, 4, 1)
    box(big[0] - 0.05, big[1] + 0.05, mn1 - 0.05, mn1 + 0.04, 2.75, 2.82, "black")
    wn(un0, mn0 + 0.01, uu1, -1, zb, FU, [])
    box(mu0 - 0.3, mu1, mn0, mn1, EB - 0.25, EB - 0.05, "roofflat")       # flat roof deck
    box(mu0, mu1, mn0 + 0.3, mn1 - 0.3, EB - 0.3, EB - 0.25, "cedar")     # cedar ceiling inside
    for a_, b_, c_, d_ in ((mu0, mu1, mn0, mn0 + 0.3), (mu0, mu1, mn1 - 0.3, mn1), (wu0_ - 0.3, wu0_, mn0, mn1)):
        box(a_, b_, c_, d_, EB - 0.05, EB + 0.2, "black")                   # parapet capping
    bu0, bu1, bn0, bn1 = L["balcony"]
    box(bu0, bu1, bn0 + 0.3, bn1, EB - 0.05, EB + 0.02, "cedar")            # timber balcony deck
    for (x0, x1, y0, y1) in ((bu1 - 0.04, bu1, bn0 + 0.3, bn1), (bu0, bu1, bn0 + 0.3, bn0 + 0.34), (bu0, bu1, bn1 - 0.04, bn1)):
        box(x0, x1, y0, y1, EB + 0.95, EB + 1.0, "black")
        L_ = max(x1 - x0, y1 - y0)
        for k in range(int(L_ / 0.12) + 1):
            if x1 - x0 > y1 - y0:
                box(x0 + k * 0.12, x0 + k * 0.12 + 0.02, y0, y1, EB, EB + 0.95, "black")
            else:
                box(x0, x1, y0 + k * 0.12, y0 + k * 0.12 + 0.02, EB, EB + 0.95, "black")
    for x in (bu0 + 1.0, bu0 + 1.8):
        box(x - 0.3, x + 0.3, bn0 + 1.6, bn0 + 2.2, EB, EB + 0.42, "linen")
    room_light(mu0 + 0.6, mu1 - 0.6, mn0 + 0.8, mn1 - 0.8, EB - 0.3, 1300)
    box(9.6, 12.6, mn1 - 4.2, mn1 - 3.3, 0, 0.42, "fabric")
    box(9.6, 12.6, mn1 - 4.2, mn1 - 3.95, 0.42, 0.8, "fabric")
    box(10.2, 12.0, mn1 - 2.6, mn1 - 1.8, 0, 0.36, "walnut")
    box(8.9, 13.6, mn1 - 4.6, mn1 - 1.0, 0.0, 0.01, "rug")
    pendant(11.1, mn1 - 2.2, 2.0, 35, r=0.3)
    box(mu0 + 2.5, mu0 + 4.5, mn0 + T + 0.01, mn0 + T + 0.04, 1.0, 2.2, "art")

    # ---------------- recessed entry: limewashed brick walls, timber-lined soffit, oak pivot door
    fu0, fu1, fn0, fn1 = L["fin"]
    box(fu0, fu1, fn0, fn1, zb, EB + 0.25, "limewash")                    # brick fin wall beside the entry
    box(fu1 - 0.01, eu0 + 0.01, en0, en1, zb, 2.75, "limewash")
    door = (12.9, 14.2, 0.0, 2.7)
    wall_u(eu0, wu0_, en1, en1 + 0.35, zb, 2.75, [door, (14.55, 15.2, 0.0, 2.7)], "limewash")
    box(door[0], door[1], en1 + 0.15, en1 + 0.21, 0, 2.7, "cedar")
    box(door[1] - 0.25, door[1] - 0.2, en1 + 0.1, en1 + 0.15, 0.9, 1.9, "black")   # pull handle
    poly([(14.55, en1 + 0.18, 0), (15.2, en1 + 0.18, 0), (15.2, en1 + 0.18, 2.7), (14.55, en1 + 0.18, 2.7)], "glass")
    frame_rect(14.55, 15.2, 0, 2.7, en1 + 0.18, "u", 1, 1, w=0.05)
    box(eu0, wu0_, en0 - 0.1, en1 + 0.1, 2.75, 2.8, "cedar")                 # soffit
    box(eu0, wu0_, en0 - 1.0, en1, -0.05, 0.0, "paving")                       # entry floor (one step up from the path)
    for x, nn in ((eu0 + 0.9, en0 + 0.7), (wu0_ - 0.9, en0 + 0.7), (eu0 + 0.9, en1 - 0.6), (wu0_ - 0.9, en1 - 0.6)):
        downlight(x, nn, 2.75, 28)
    light("AREA", ((eu0 + wu0_) / 2, en1 - 0.4, 0.3), 30, WARM, aim=((eu0 + wu0_) / 2, en1 + 2, 0.3), size=1.5, name="soffit")
    light("SPOT", (eu0 + 0.15, (en0 + en1) / 2, 0.35), 25, WARM, aim=(eu0 + 0.1, (en0 + en1) / 2, 4), spot=0.8, blend=0.7, size=0.02, name="wall_up")
    room_light(eu0, wu0_, en1 + 0.4, en1 + 3.0, 2.7, 250)
    zr_b = zr_u

    # ---------------- garage wing (ridge along n): gable to the street over the street-level garage
    gd = (gu0 + 0.6, gu1 - 0.6, GZ, GZ + 2.15)
    bw = P["bay_wing"]
    ww = (bw[0], bw[1], 0.8, 2.5)
    # street gable: garage storey + main storey + gable triangle
    wall_u(wu0_, wu1_, wn0, wn0 + T, GZ - 0.3, 0.0, [gd], "clad")
    wu(wu0_, wu1_, wn0, 1, 0.0, EAVE_A, [ww])
    bay_u(*ww, wn0, wn0 - bw[2], cols=2)
    prism([(wu0_, EAVE_A), (wu1_, EAVE_A), ((wu0_ + wu1_) / 2, EAVE_A + (wu1_ - wu0_) / 2 * math.tan(PITCH_A))], wn0, wn0 + T, "clad")
    # timber-slat garage door, set back under a slim black steel hood
    box(gd[0], gd[1], wn0 + 0.22, wn0 + 0.27, GZ, gd[3], "slats")
    box(gd[0] - 0.15, gd[1] + 0.15, wn0 - 0.45, wn0 + 0.02, gd[3] + 0.05, gd[3] + 0.18, "black")
    for x in (gd[0] - 0.4, gd[1] + 0.4):
        box(x - 0.05, x + 0.05, wn0 - 0.12, wn0, gd[3] - 0.25, gd[3] + 0.0, "black")
        light("SPOT", (x, wn0 - 0.15, gd[3] - 0.2), 25, WARM, aim=(x, wn0 - 0.3, GZ), spot=1.2, blend=0.8, size=0.02, name="garage_lamp")
    for x in (gd[0] + 0.6, (gd[0] + gd[1]) / 2, gd[1] - 0.6):
        downlight(x, wn0 - 0.25, gd[3] + 0.05, 14, name="garage_up")
    # garage box below floor level
    wn(wn0, gn1, wu0_, 1, GZ - 0.3, 0.0, [], lining="clad")
    wn(wn0, gn1, wu1_, -1, GZ - 0.3, 0.0, [], lining="clad")
    box(wu0_, wu1_, gn1 - 0.3, gn1, GZ - 0.3, 0.0, "clad")
    box(gu0, gu1, wn0, gn1, GZ - 0.3, GZ, "concrete")
    # side walls of the wing above floor
    wn(wn0, wn1, wu1_, -1, zb, EAVE_A, [(8.6, 9.8, 1.0, 2.3), (16.4, 17.6, 1.0, 2.3)])
    for a_ in (8.6, 16.4):
        deep_window_n(a_, a_ + 1.2, 1.0, 2.3, wu1_, 1)
    box(wu1_, wu1_ + 0.04, 12.0, 14.0, GZ - 0.3, EAVE_A, "cedar")      # cedar panel breaks the long NE side wall
    wn(wn0, mn0, wu0_, 1, zb, EAVE_A, [])
    wn(mn0, en1, wu0_, 1, zb, 2.75, [], ext="limewash", lining="limewash")
    wn(mn0, mn1, wu0_, 1, 2.75, EAVE_A, [])
    wn(mn1, wn1, wu0_, 1, zb, EAVE_A, [(mn1 + 0.6, wn1 - 0.6, 0.0, 2.7)])
    glaze_n(mn1 + 0.6, wn1 - 0.6, 0, 2.7, wu0_ + 0.12, 2, 1)
    # north gable: tall black-framed glazing
    ng = (wu0_ + 0.5, wu1_ - 0.5, 0.0, 2.9)
    wu(wu0_, wu1_, wn1, -1, zb, EAVE_A, [ng])
    glaze_u(ng[0], ng[1], 0, 2.9, wn1 - 0.12, 3, 1)
    zr_a_ = EAVE_A + (wu1_ - wu0_) / 2 * math.tan(PITCH_A)
    um_a = (wu0_ + wu1_) / 2
    ring([(wu0_, EAVE_A), (wu1_, EAVE_A), (um_a, zr_a_), (um_a, zr_a_)],
         [(um_a - 0.7, EAVE_A + 0.3), (um_a + 0.7, EAVE_A + 0.3), (um_a + 0.7, EAVE_A + 1.6), (um_a - 0.7, EAVE_A + 1.6)], wn1 - T, wn1, "clad")
    glaze_u(um_a - 0.7, um_a + 0.7, EAVE_A + 0.3, EAVE_A + 1.6, wn1 - 0.12, 2, 1)
    zr_a = roof_n(wn0, wn1, wu0_, wu1_, EAVE_A, PITCH_A, lining=(gn1 - 3, wn1 - T))
    room_light(wu0_ + 0.5, wu1_ - 0.5, wn0 + 1, wn1 - 1, EAVE_A, 900)
    vault_light(wu0_ + 1, wu1_ - 1, mn1, wn1 - 0.6, zr_a - 1.4, 300)
    box(17.0, 20.9, wn1 - 3.0, wn1 - 2.0, 0, 0.9, "walnut")                 # kitchen island
    for x in (17.8, 19.0, 20.2):
        pendant(x, wn1 - 2.5, 1.95, 30, r=0.16)
    box(wu1_ - T - 0.65, wu1_ - T, mn1 + 0.5, wn1 - 0.8, 0, 0.9, "walnut")

    # ---------------- family wing (ridge along n): sliders east, fully glazed gable north
    sl = (mn1 + 0.6, rn1 - 1.0, 0.0, 2.7)
    wn(mn1, rn1, ru1, -1, zb, EAVE_C, [sl])
    glaze_n(sl[0], sl[1], 0, 2.7, ru1 - 0.12, 5, 1)
    box(ru1 - 0.05, ru1 + 0.05, sl[0] - 0.05, sl[1] + 0.05, 2.7, 2.77, "black")
    wn(rn0, rn1, ru0, 1, zb, EAVE_C, [(rn0 + 3.0, rn0 + 4.0, 1.0, 2.3), (rn0 + 9.5, rn0 + 10.5, 1.0, 2.3)])
    for a_ in (rn0 + 3.0, rn0 + 9.5):
        deep_window_n(a_, a_ + 1.0, 1.0, 2.3, ru0, -1)
    box(ru0 - 0.04, ru0, rn0 + 5.8, rn0 + 7.8, zb, EAVE_C, "cedar")     # cedar panel breaks the long SW side wall
    um_c = (ru0 + ru1) / 2
    tn_c = math.tan(PITCH_C)
    zr_c_ = EAVE_C + (ru1 - ru0) / 2 * tn_c
    pier = 0.45

    def ztop(u):  # underside of the glazing frame below the gable roof line
        return EAVE_C + (min(u - ru0, ru1 - u)) * tn_c - 0.32

    ring([(ru0, zb), (ru1, zb), (ru1, EAVE_C), (um_c, zr_c_), (ru0, EAVE_C)],
         [(ru0 + pier, 0.0), (ru1 - pier, 0.0), (ru1 - pier, ztop(ru1 - pier)), (um_c, ztop(um_c)), (ru0 + pier, ztop(ru0 + pier))],
         rn1 - T, rn1, "clad")
    ng_c = rn1 - 0.12
    poly([(ru0 + pier, ng_c, 0), (ru1 - pier, ng_c, 0), (ru1 - pier, ng_c, ztop(ru1 - pier)), (um_c, ng_c, ztop(um_c)),
          (ru0 + pier, ng_c, ztop(ru0 + pier))], "glass")
    cols = 5
    for k in range(cols + 1):
        x = ru0 + pier + (ru1 - ru0 - 2 * pier) * k / cols
        box(x - 0.04, x + 0.04, ng_c - 0.05, ng_c + 0.05, 0, ztop(x), "black")
    box(ru0 + pier, ru1 - pier, ng_c - 0.05, ng_c + 0.05, 0, 0.06, "black")
    box(ru0 + pier, ru1 - pier, ng_c - 0.05, ng_c + 0.05, 2.7, 2.78, "black")
    for (xa, xb) in ((ru0 + pier, um_c), (um_c, ru1 - pier)):
        za, zb_ = ztop(xa), ztop(xb)
        slab((xa, ng_c - 0.05, za - 0.08), (xb, ng_c - 0.05, zb_ - 0.08), (xb, ng_c - 0.05, zb_), (xa, ng_c - 0.05, za), 0.1, "black")
    zr_c = roof_n(un1 - (zr_c_ - EU) / tnu - 0.2, rn1, ru0, ru1, EAVE_C, PITCH_C, lining=(rn0, rn1 - T), rake0=False)
    room_light(ru0 + 0.6, ru1 - 0.6, rn0 + 0.6, rn1 - 0.6, EAVE_C, 1400)
    vault_light(ru0 + 2, ru1 - 2, rn0 + 2, rn1 - 1.5, zr_c - 1.4, 350)
    box(2.0, 6.6, rn1 - 5.2, rn1 - 4.2, 0, 0.42, "fabric")
    box(2.0, 6.6, rn1 - 5.2, rn1 - 4.9, 0.42, 0.8, "fabric")
    box(2.6, 5.6, rn1 - 3.4, rn1 - 2.4, 0, 0.35, "walnut")
    box(1.6, 7.2, rn1 - 6.0, rn1 - 1.5, 0, 0.01, "rug")
    pendant(4.5, rn1 - 2.9, 2.4, 45, r=0.35)
    box(ru0 + T + 0.01, ru0 + T + 0.04, rn0 + 5.5, rn0 + 7.5, 1.0, 2.2, "art")
    for nn in (rn0 + 2.0, rn1 - 1.2):
        cyl((ru0 + 1.2, nn, 0), (ru0 + 1.2, nn, 1.5), 0.02, 0.02, "black")
        cyl((ru0 + 1.2, nn, 1.5), (ru0 + 1.2, nn, 1.75), 0.18, 0.12, "lamp", sides=10)

    # wall lights by the courtyard glazing
    lantern_sconce(big[0] - 0.1, mn1, 2.3, outward_n=1)
    lantern_sconce(big[1] + 0.1, mn1, 2.3, outward_n=1)
    lantern_sconce(ru1, sl[0] - 0.25, 2.3, outward_u=1)
    lantern_sconce(ru1, sl[1] + 0.25, 2.3, outward_u=1)
    lantern_sconce(ng[0] - 0.2, wn1, 2.3, outward_n=1)
    return zr_b, zr_a, zr_c


def build_courtyard():
    pu0, pu1, pn0, pn1 = L["pool"]
    # finish surfaces from the site script (paving, lawn, gravel at both terrace levels, garage-side strip)
    for sf in SURF:
        z = sf["z"] - FFL + 0.03
        for t in sf["tris"]:
            poly([(t[0][0], t[0][1], z), (t[1][0], t[1][1], z), (t[2][0], t[2][1], z)], sf["mat"])
    # pool: coping, tiles, water, lights
    for a, c, n0_, n1_ in ((pu0 - 0.3, pu1 + 0.3, pn0 - 0.3, pn0), (pu0 - 0.3, pu1 + 0.3, pn1, pn1 + 0.3), (pu0 - 0.3, pu0, pn0, pn1), (pu1, pu1 + 0.3, pn0, pn1)):
        box(a, c, n0_, n1_, PADZ - 0.25, PADZ + 0.07, "coping")
    box(pu0, pu1, pn0, pn1, PADZ - 1.75, PADZ - 1.6, "pooltile")
    for a, c, n0_, n1_ in ((pu0, pu1, pn0, pn0 + 0.02), (pu0, pu1, pn1 - 0.02, pn1), (pu0, pu0 + 0.02, pn0, pn1), (pu1 - 0.02, pu1, pn0, pn1)):
        box(a, c, n0_, n1_, PADZ - 1.75, PADZ - 0.05, "pooltile")
    poly([(pu0, pn0, PADZ - 0.12), (pu1, pn0, PADZ - 0.12), (pu1, pn1, PADZ - 0.12), (pu0, pn1, PADZ - 0.12)], "water")
    for x in (pu0 + 2.4, (pu0 + pu1) / 2, pu1 - 2.4):
        light("POINT", (x, pn0 + 0.2, PADZ - 0.8), 45, COOL, size=0.05, name="pool")
    # loungers on the paving north-east of the pool
    for x in (pu1 - 4.2, pu1 - 2.9):
        slab((x, pn1 + 0.5, PADZ + 0.3), (x + 0.75, pn1 + 0.5, PADZ + 0.3), (x + 0.75, pn1 + 0.5, PADZ + 0.3), (x, pn1 + 0.5, PADZ + 0.3), 0, "black") if False else None
    # outdoor dining by the family wing sliders
    box(9.4, 11.6, 18.2, 19.2, PADZ + 0.72, PADZ + 0.77, "walnut")
    for x in (9.8, 10.5, 11.2):
        for nn in (17.8, 19.6):
            box(x - 0.2, x + 0.2, nn - 0.2, nn + 0.2, PADZ, PADZ + 0.45, "black")

    # ---------------- covered outdoor fireplace room (gable open to the north, timber-lined vault)
    ou0, ou1, on0, on1 = L["outdoor"]
    zo = PADZ + 0.12
    box(ou0, ou1, on0, on1, PADZ - 0.3, zo, "bluestone")
    eave_o, pitch_o = EAVE_O, PITCH_O
    um = (ou0 + ou1) / 2
    zr_o_ = eave_o + (ou1 - ou0) / 2 * math.tan(pitch_o)
    # back wall: limewashed brick fireplace with a chimney rising above the ridge
    fo = (um - 0.8, um + 0.8, zo + 0.45, zo + 1.45)
    wall_u(ou0, ou1, on0, on0 + 0.45, PADZ - 0.3, eave_o, [(fo[0], fo[1], fo[2], fo[3])], "limewash")
    prism([(ou0, eave_o), (ou1, eave_o), (um, zr_o_)], on0, on0 + 0.45, "limewash")
    box(um - 1.1, um + 1.1, on0 - 0.1, on0 + 0.75, PADZ - 0.3, zr_o_ + 1.6, "limewash")
    box(um - 1.15, um + 1.15, on0 - 0.15, on0 + 0.8, zr_o_ + 1.6, zr_o_ + 1.68, "black")
    box(fo[0], fo[1], on0 + 0.2, on0 + 0.3, fo[2], fo[3], "black")
    box(fo[0] + 0.15, fo[1] - 0.15, on0 + 0.3, on0 + 0.6, fo[2], fo[2] + 0.2, "fire")
    box(um - 1.4, um + 1.4, on0 + 0.75, on0 + 1.15, zo, zo + 0.42, "bluestone")              # raised hearth
    light("POINT", (um, on0 + 0.9, fo[2] + 0.35), 160, (1.0, 0.45, 0.15), size=0.3, name="pool_fire")
    # firewood niches either side
    for x0 in (ou0 + 0.35, ou1 - 1.15):
        box(x0, x0 + 0.8, on0 + 0.45, on0 + 0.6, zo + 0.42, zo + 2.1, "darkbark")
    # black steel posts and eave beams
    for x in (ou0 + 0.1, ou1 - 0.1):
        for nn in ((on0 + on1) / 2, on1 - 0.1):
            box(x - 0.08, x + 0.08, nn - 0.08, nn + 0.08, zo, eave_o, "black")
        box(x - 0.1, x + 0.1, on0, on1, eave_o - 0.3, eave_o, "black")
    box(ou0, ou1, on1 - 0.2, on1, eave_o - 0.3, eave_o, "black")
    roof_n(on0, on1, ou0, ou1, eave_o, pitch_o, lining=(on0 + 0.45, on1), rake0=False)
    # open north gable: thin black truss line, cedar-lined triangle behind
    slab((ou0, on1 - 0.15, eave_o), (um, on1 - 0.15, zr_o_), (um, on1 - 0.15, zr_o_ - 0.1), (ou0, on1 - 0.15, eave_o - 0.1), 0.12, "black")
    slab((um, on1 - 0.15, zr_o_), (ou1, on1 - 0.15, eave_o), (ou1, on1 - 0.15, eave_o - 0.1), (um, on1 - 0.15, zr_o_ - 0.1), 0.12, "black")
    # lounge facing the fire
    box(ou0 + 1.0, ou1 - 1.0, on1 - 1.6, on1 - 0.8, zo, zo + 0.42, "linen")
    box(ou0 + 1.0, ou1 - 1.0, on1 - 1.0, on1 - 0.7, zo + 0.42, zo + 0.85, "linen")
    for x in (ou0 + 0.6, ou1 - 0.6):
        box(x - 0.4, x + 0.4, on0 + 2.2, on0 + 3.0, zo, zo + 0.42, "linen")
    box(um - 0.7, um + 0.7, on0 + 2.4, on0 + 3.4, zo, zo + 0.38, "walnut")
    box(ou0 + 0.6, ou1 - 0.6, on0 + 1.6, on1 - 0.5, zo, zo + 0.01, "rug")
    pendant(um, (on0 + on1) / 2 + 0.6, zo + 2.6, 40, r=0.3)
    # festoon lights along both eaves
    for x in (ou0 + 0.1, ou1 - 0.1):
        for k in range(10):
            t = (k + 0.5) / 10
            nn = on0 + 0.6 + (on1 - on0 - 0.7) * t
            zz = eave_o - 0.35 - 0.18 * math.sin(math.pi * ((t * 2) % 1))
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.035, location=(x, nn, zz), segments=8, ring_count=6)
            bpy.context.active_object.data.materials.append(MATS["bulb"])
            if k % 3 == 1:
                light("POINT", (x, nn, zz - 0.05), 6, WARM, size=0.04, name="cabana")


def build_site_walls():
    """retaining walls along the full-lot pad edge, the terrace wall and steps, the street wall, front steps"""
    gu0, gu1, gn0, gn1 = L["garage"]
    pts = PAD_OUTLINE
    poly_ = [tuple(p) for p in pts]

    def inside(u, n):
        c = False
        for (ua, na), (ub, nb) in zip(poly_, poly_[1:] + poly_[:1]):
            if (na > n) != (nb > n) and u < ua + (n - na) * (ub - ua) / (nb - na):
                c = not c
        return c

    def lvl(u, n):
        if n < L["stair"][3] and n < FRONT["bank_top"]:
            return np.interp(n, [2.2, FRONT["bank_top"]], [FRONT["wall_top"], PADZ])
        return UPZ if n > n_edge(u) else PADZ

    for (ua, na), (ub, nb) in zip(pts, pts[1:]):
        Lg = math.hypot(ub - ua, nb - na)
        steps = max(1, int(Lg / 1.6))
        for k in range(steps):
            t0, t1 = k / steps, (k + 1) / steps
            u0_, n0_ = ua + (ub - ua) * t0, na + (nb - na) * t0
            u1_, n1_ = ua + (ub - ua) * t1, na + (nb - na) * t1
            um_, nm_ = (u0_ + u1_) / 2, (n0_ + n1_) / 2
            if max(n0_, n1_) < 2.6:
                continue                                   # street wall handles the front edge
            if gu0 - 0.5 < um_ < gu1 + 0.05 and nm_ < gn1 + 0.6:
                continue                                   # garage walls retain this edge
            dx, dn = u1_ - u0_, n1_ - n0_
            ln = math.hypot(dx, dn) or 1
            ox, on = -dn / ln, dx / ln
            if inside(um_ + ox * 0.15, nm_ + on * 0.15):
                ox, on = -ox, -on                           # make (ox, on) point out of the pad
            nat = zg(um_ + ox * 1.6, nm_ + on * 1.6)
            lv = lvl(um_ - ox * 0.5, nm_ - on * 0.5)
            top = max(nat + 0.15, lv + 0.5)
            bot = min(nat, lv) - 0.35
            w = 0.3
            faces("wallrender", [(u0_, n0_, bot), (u1_, n1_, bot), (u1_ + ox * w, n1_ + on * w, bot), (u0_ + ox * w, n0_ + on * w, bot),
                                 (u0_, n0_, top), (u1_, n1_, top), (u1_ + ox * w, n1_ + on * w, top), (u0_ + ox * w, n0_ + on * w, top)],
                  [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
            box(min(u0_, u1_) - 0.02, max(u0_, u1_) + 0.02, min(n0_, n1_) - 0.02, max(n0_, n1_) + 0.02, top, top + 0.001, "wallrender") if False else None

    # terrace wall (limewashed brick, 1.2 m) along the edge line, with steps up from the lawn
    ts0, ts1 = L["tsteps"]
    us = [p[0] for p in pts]
    u_lo, u_hi = min(us) - 1, max(us) + 1
    k_ = L["edge"][2]
    seg = []
    u = u_lo
    while u < u_hi:
        seg.append(u)
        u += 1.0
    seg.append(u_hi)
    for a, b in zip(seg, seg[1:]):
        for c0, c1 in ((a, min(b, ts0)), (max(a, ts1), b)):
            if c1 <= c0:
                continue
            p = [(c0, n_edge(c0)), (c1, n_edge(c1))]
            if not (inside(c0 + 0.01, n_edge(c0) - 0.05) or inside(c1 - 0.01, n_edge(c1) - 0.05)):
                continue
            n0a, n1a = n_edge(c0), n_edge(c1)
            faces("limewash", [(c0, n0a, PADZ - 0.3), (c1, n1a, PADZ - 0.3), (c1, n1a + 0.3, PADZ - 0.3), (c0, n0a + 0.3, PADZ - 0.3),
                               (c0, n0a, UPZ + 0.12), (c1, n1a, UPZ + 0.12), (c1, n1a + 0.3, UPZ + 0.12), (c0, n0a + 0.3, UPZ + 0.12)],
                  [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    # clip the terrace wall at the side boundaries: the boundary walls overlap its ends
    n_top = n_edge(ts1) + 0.35
    rise = (UPZ - PADZ) / 4
    for k in range(4):
        z = PADZ + rise * (k + 1)
        box(ts0, ts1, n_top - 1.6 + 0.4 * k, n_top, PADZ - 0.3, z + 0.03, "bluestone")
    for x in (ts0 - 0.25, ts1):
        box(x, x + 0.25, n_top - 1.7, n_top, PADZ - 0.3, UPZ + 0.12, "limewash")
    for x in (ts0 + 0.2, ts1 - 0.2):
        light("SPOT", (x, n_top - 1.5, UPZ + 0.6), 10, WARM, aim=(x, n_top - 2.4, PADZ), spot=1.4, blend=0.9, size=0.02, name="soffit_step")
    # uplights washing the terrace wall
    for u in np.arange(u_lo + 2.0, u_hi - 1.0, 3.2):
        if ts0 - 0.8 < u < ts1 + 0.8:
            continue
        if inside(u, n_edge(u) - 0.6):
            light("SPOT", (u, n_edge(u) - 0.35, PADZ + 0.1), 18, WARM, aim=(u, n_edge(u) + 0.1, UPZ + 1.5), spot=1.2, blend=0.9, size=0.02, name="wall_up")

    # street wall: low charcoal rendered wall, front steps between rendered cheek walls
    s0, s1, sn0, sn1 = L["stair"]
    topw = FRONT["wall_top"]
    box(0.3, s0 - 0.3, 1.6, 2.2, -3.4, topw, "wallrender")
    box(0.25, s0 - 0.25, 1.55, 2.25, topw, topw + 0.06, "black")
    gg = zg((s0 + s1) / 2, 1.0)
    n_steps = 9
    z_lo, z_hi = gg, PADZ
    for k in range(n_steps):
        z = z_lo + (z_hi - z_lo) * (k + 1) / n_steps
        n_a = 1.2 + k * (sn1 - 1.2) / n_steps
        box(s0, s1, n_a, sn1, z_lo - 0.4, z, "bluestone")
    for x0, x1 in ((s0 - 0.3, s0), (s1, s1 + 0.3)):
        box(x0, x1, 1.6, sn1, z_lo - 0.4, PADZ + 0.3, "wallrender")
    box(s1, s1 + 0.3, sn1, 6.6, PADZ - 0.6, PADZ + 0.3, "wallrender")
    # planted terrace beside the driveway: low rendered walls, grasses on top
    gu0_ = L["garage"][0]
    pz = L["planter"][2] - FFL
    box(s1 + 0.3, gu0_, 1.6, 2.2, GZ - 0.5, pz + 0.15, "wallrender")
    box(gu0_ - 0.3, gu0_, 2.2, 6.3, GZ - 0.5, pz + 0.15, "wallrender")
    box(s1 + 0.3, gu0_, 6.3, 6.6, GZ - 0.5, PADZ + 0.05, "wallrender")
    box(gu0_ - 0.3, gu0_, 6.6, L["garage"][2], GZ - 0.5, PADZ + 0.05, "wallrender")
    box(s1 + 0.3, gu0_ - 0.3, 2.2, 6.3, pz - 0.3, pz + 0.02, "mulch")
    for k in range(5):
        grass(s1 + 0.75 + (k % 2) * 0.5, 2.7 + k * 0.8, pz, rr(0.7, 0.95))
    box(s1 + 0.3, wu_left(), 6.6, L["main"][2] + 0.05, PADZ - 0.3, -0.05, "paving")
    for k in range(3):
        nn = 1.8 + k * (sn1 - 1.8) / 3
        zz = z_lo + (z_hi - z_lo) * ((nn - 1.2) / (sn1 - 1.2)) + 0.35
        light("SPOT", (s0 + 0.08, nn, zz), 8, WARM, aim=(s0 + 0.8, nn, zz - 2), spot=1.4, blend=0.9, size=0.02, name="soffit_step")
    # black steel letterbox / number post at the gate
    box(s0 - 0.75, s0 - 0.35, 1.3, 1.5, gg - 0.1, gg + 1.2, "black")
    # planted bank (mulch) from the wall to the front garden
    bt = FRONT["bank_top"]
    drape(-0.6, s0 - 0.3, lambda u: 2.2 if u > 0.3 else max(2.2, -u * 32.5 / 5.1 + 0.4), lambda u: bt, 0.05, "mulch", 30, 6)
    # flagstone steppers from the top of the steps to the entry
    for k, nn in enumerate((5.6, 6.2)):
        slab((s0 + 0.05, nn, PADZ + 0.05), (s1 + 0.4, nn, PADZ + 0.05), (s1 + 0.4, nn + 0.55, PADZ + 0.05), (s0 + 0.05, nn + 0.55, PADZ + 0.05), -0.08, "bluestone")


def wu_left():
    return L["wing"][0]


def fence_mat(name, hexc):
    """Colorbond corrugated steel: colour with a sine bump every 76 mm along the panel"""
    m, nt, b = cr.node_mat(name)
    b.inputs["Base Color"].default_value = (*srgb(hexc), 1)
    b.inputs["Roughness"].default_value = 0.5
    b.inputs["Metallic"].default_value = 0.3
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wv = nt.nodes.new("ShaderNodeTexWave")
    wv.inputs["Scale"].default_value = 1.0 / 0.076
    wv.bands_direction = "Z"
    wv.wave_profile = "SIN"
    nt.links.new(tc.outputs["Object"], wv.inputs["Vector"])
    bn = nt.nodes.new("ShaderNodeBump"); bn.inputs["Strength"].default_value = 0.5; bn.inputs["Distance"].default_value = 0.01
    nt.links.new(wv.outputs["Fac"], bn.inputs["Height"])
    nt.links.new(bn.outputs["Normal"], b.inputs["Normal"])
    return m


def build_fences():
    """1.8 m Colorbond fences on the side boundaries (beige NE, Pale Eucalypt SW) from the 6 m line to the rear,
    on the boundary line, following the higher of the two ground levels; the retaining walls sit just inside"""
    for (ua, na), (ub, nb), mat in (((0.0, 0.0), (-5.1, 32.5), "fence_sw"), ((22.9, 0.0), (29.8, 44.5), "fence_ne")):
        Lg = math.hypot(ub - ua, nb - na)
        tu, tn = (ub - ua) / Lg, (nb - na) / Lg
        ou, on = (tn, -tu) if mat == "fence_sw" else (-tn, tu)      # outward normal (away from the lot)
        s = 6.6
        while s < Lg - 0.05:
            e = min(s + 2.35, Lg)
            u0, n0 = ua + tu * s, na + tn * s
            u1, n1 = ua + tu * e, na + tn * e
            um, nm = (u0 + u1) / 2, (n0 + n1) / 2
            gm = max(zg(um - ou * 0.7, nm - on * 0.7), zg(um + ou * 1.2, nm + on * 1.2))
            z0, z1 = gm - 0.05, gm + 1.8
            faces(mat, [(u0, n0, z0), (u1, n1, z0), (u1, n1, z1), (u0, n0, z1)], [(0, 1, 2, 3)])
            faces(mat, [(u1 + ou * 0.01, n1 + on * 0.01, z0), (u0 + ou * 0.01, n0 + on * 0.01, z0), (u0 + ou * 0.01, n0 + on * 0.01, z1),
                        (u1 + ou * 0.01, n1 + on * 0.01, z1)], [(0, 1, 2, 3)])
            slab((u0 - ou * 0.03, n0 - on * 0.03, z1), (u1 - ou * 0.03, n1 - on * 0.03, z1), (u1 + ou * 0.04, n1 + on * 0.04, z1),
                 (u0 + ou * 0.04, n0 + on * 0.04, z1), 0.05, mat)                               # capping
            cyl((u0, n0, z0 - 0.3), (u0, n0, z1), 0.035, 0.035, mat, sides=4)                   # post
            s = e


def build_street():
    gu0, gu1, gn0, gn1 = L["garage"]
    R = ARC["r"]
    drape(-45, 68, lambda u: arc_n(u, R - 15.0), lambda u: arc_n(u, R - 4.0), 0.03, "asphalt", 150, 10)
    for r0, r1 in ((R - 4.0, R - 3.75), (R - 15.25, R - 15.0)):
        drape(-45, 68, lambda u, r0=r0: arc_n(u, r0), lambda u, r1=r1: arc_n(u, r1), 0.17, "kerb", 150, 1)
    drape(-45, 68, lambda u: arc_n(u, R - 16.7), lambda u: arc_n(u, R - 15.25), 0.1, "concrete", 150, 2)
    drape(-45, 68, lambda u: arc_n(u, R - 24.0), lambda u: arc_n(u, R - 16.7), 0.05, "lawn", 150, 4)
    drape(-45, gu0 + 0.2, lambda u: arc_n(u, R - 3.75), lambda u: arc_n(u, R - 2.3), 0.1, "concrete", 80, 2)
    drape(23.4, 68, lambda u: arc_n(u, R - 3.75), lambda u: arc_n(u, R - 2.3), 0.1, "concrete", 60, 2)
    drape(gu0 + 0.2, 23.4, lambda u: arc_n(u, R - 3.75), lambda u: gn0, 0.08, "aggregate", 14, 12)
    drape(-1.5, L["stair"][0] - 0.2, lambda u: arc_n(u, R - 2.3), lambda u: 1.58, 0.06, "gravel", 40, 4)
    drape(L["stair"][1] + 0.2, gu0 + 0.2, lambda u: arc_n(u, R - 2.3), lambda u: 1.58, 0.06, "gravel", 6, 4)
    drape(-45, -1.5, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 1.0, 0.04, "lawn", 30, 2)
    drape(23.4, 68, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 1.0, 0.04, "lawn", 30, 2)


EUC = []


def build_garden():
    """eucalypts on the hill (shared), olives, sculptural ornamental grasses, gravel with bluestone steppers"""
    S = cr.S
    for gx, gy, r in S["trees"]:
        if rng.random() < 0.3:
            continue
        u, n = cr.to_frame(gx, gy)
        if -6 < u < 31 and -1 < n < 45 and cr.to_frame(gx, gy)[1] < 32.5 + 0.3438 * (u + 5.1):
            continue   # nothing self-seeded inside the lot
        h = min(max(r * 2.8 + rr(2, 5), 7), 17)
        tree((u, n, cr.h_grid(gx, gy) - FFL - 0.2), h, r, "euc", "bark", cards_per=14)
        EUC.append((u, n, cr.h_grid(gx, gy) - FFL - 0.2, h, r))
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
            tree((u, n, dem[r_, c_] - FFL - 0.3), h_, r__, "euc", "bark", cards_per=10)
            EUC.append((u, n, dem[r_, c_] - FFL - 0.3, h_, r__))

    def olive(u, n, z, lit=True):
        tree((u, n, z), rr(3.6, 4.4), rr(1.6, 2.0), "olive", "darkbark", stems=3, cards_per=24, crown_flat=0.55)
        if lit:
            light("SPOT", (u + 0.5, n - 0.6, z + 0.1), 120, WARM, aim=(u, n, z + 3.2), spot=1.1, blend=0.9, size=0.03, name="olive_up")

    # front garden: two olives and drifts of grasses in gravel, grasses on the planted bank
    olive(2.6, 5.7, PADZ)
    olive(9.3, 5.6, PADZ)                       # screens the street bay window
    for _ in range(46):
        u, n = rr(-0.3, 12.6), rr(2.5, 4.8)
        grass(u, n, zg(u, n), rr(0.7, 1.05))
    for u in np.arange(0.6, 12.5, 0.95):
        if not (1.6 < u < 3.6 or 8.3 < u < 10.3 or 4.9 < u < 7.3 or u > 11.2):
            grass(u + rr(-0.15, 0.15), 6.0 + rr(-0.2, 0.2), PADZ, rr(0.9, 1.2))
    for _ in range(28):   # verge garden
        u = rr(-1.2, 12.6)
        t = rr(0.25, 0.85)
        nn = arc_n(u, ARC["r"] - 2.3) * (1 - t) + 1.5 * t
        grass(u, nn, zg(u, nn), rr(0.6, 0.9))
    # SW side garden along the family wing, NE side path
    for nn in np.arange(17.0, 30.5, 2.2):
        uu = -3.9 * nn / 30 + rr(-0.2, 0.2) - 0.2
        grass(uu, nn, PADZ, rr(0.9, 1.25))
    olive(-2.1, 22.5, PADZ)
    olive(-1.3, 30.2, PADZ)
    for nn in np.arange(4.5, 21.0, 1.9):
        uu = 22.9 + 6.9 * nn / 44.5 - 0.65
        if nn > 10.4:
            grass(uu, nn, PADZ, rr(0.8, 1.1))
    for nn in np.arange(11.0, 22.5, 0.75):
        uu = 23.3 + 6.9 * nn / 44.5 * 0.5
        slab((uu - 0.35, nn, PADZ + 0.05), (uu + 0.35, nn, PADZ + 0.05), (uu + 0.35, nn + 0.5, PADZ + 0.05), (uu - 0.35, nn + 0.5, PADZ + 0.05), -0.08, "bluestone") if False else None
    # courtyard: grasses framing the lawn, olives in the lawn corners and on the upper terrace
    lu0, lu1, ln0, ln1 = L["lawn"]
    for u in np.arange(lu0 + 0.3, lu1, 1.05):
        grass(u, ln1 + 0.35, PADZ, rr(0.9, 1.2))
    olive(lu1 + 1.3, ln1 - 0.2, PADZ)
    olive(lu0 - 1.4, ln0 + 1.5, PADZ)
    for k in range(9):    # grasses between the lawn and the outdoor room
        grass(lu1 + rr(0.5, 2.2), rr(29.3, 32.5), PADZ, rr(0.9, 1.25))
    # upper terrace: olive grove + grasses in gravel with bluestone steppers to both rear corners
    for u, n_off in ((-2.6, 1.0), (3.0, 1.9), (9.0, 2.5), (15.0, 3.0), (25.0, 5.5), (27.6, 7.0)):
        olive(u, n_edge(u) + n_off, UPZ, lit=u > 0)
    for _ in range(60):
        u = rr(-4.4, 28.8)
        lo, hi = n_edge(u) + 0.6, 32.5 + 0.3438 * (u + 5.1) - 0.6
        if hi <= lo:
            continue
        grass(u, rr(lo, hi), UPZ, rr(0.8, 1.2))
    ts0, ts1 = L["tsteps"]
    for k in range(16):   # stepper path across the upper terrace from the steps to each rear corner
        for side in (-1, 1):
            u = (ts0 + ts1) / 2 + side * (0.6 + k * 1.55)
            if not (-3.8 < u < 28.3):
                continue
            n = n_edge(u) + 1.2 + (0.25 if k % 2 else 0) + (k * 0.1 if side > 0 else 0)
            if n > 32.5 + 0.3438 * (u + 5.1) - 0.8:
                continue
            a_ = rr(-0.15, 0.15)
            c_, s_ = math.cos(a_), math.sin(a_)
            w_, h_ = 0.45, 0.32
            corners = [(u + c_ * dx - s_ * dn, n + s_ * dx + c_ * dn, UPZ + 0.06) for dx, dn in ((-w_, -h_), (w_, -h_), (w_, h_), (-w_, h_))]
            slab(*corners, -0.08, "bluestone")
    # lawn-to-steps steppers
    for k in range(3):
        nn = L["lawn"][3] + 0.4 + k * 0.55
        slab((ts0 + 0.05, nn, PADZ + 0.05), (ts1 - 0.05, nn, PADZ + 0.05), (ts1 - 0.05, nn + 0.4, PADZ + 0.05), (ts0 + 0.05, nn + 0.4, PADZ + 0.05), -0.08, "bluestone")
    # sparse screening shrubs along the side boundaries (low fuel, kept clear of the house)
    for (ua, na), (ub, nb), inset in (((0.0, 0.0), (-5.1, 32.5), 0.75), ((22.9, 0.0), (29.8, 44.5), -0.75)):
        for k in range(14):
            t = 0.3 + k / 14 * 0.66
            u, nn = ua + (ub - ua) * t + inset, na + (nb - na) * t
            z = UPZ if nn > n_edge(u) + 0.3 else PADZ
            bush((u, nn, z + 0.9), rr(0.7, 0.85), "hedge", 16)
            bush((u, nn, z + 1.8), rr(0.6, 0.75), "hedge", 14)


def adjust_terrain():
    """sink the terrain under the pool and keep it just below every finished level inside the lot"""
    ob = bpy.data.objects["terrain"]
    pu0, pu1, pn0, pn1 = L["pool"]
    poly_ = [tuple(p) for p in PAD_OUTLINE]

    def inside(u, n):
        c = False
        for (ua, na), (ub, nb) in zip(poly_, poly_[1:] + poly_[:1]):
            if (na > n) != (nb > n) and u < ua + (n - na) * (ub - ua) / (nb - na):
                c = not c
        return c

    for idx, v in enumerate(ob.data.vertices):
        j, i = divmod(idx, cr.NG)
        u, n = v.co.x, v.co.y
        if -8 < u < 34 and -2 < n < 48:
            v.co.z = cr.H[j, i] - FFL                      # undo the shared script's fixed pool sink
            if pu0 - 1.0 < u < pu1 + 1.0 and pn0 - 1.0 < n < pn1 + 1.0:
                v.co.z = PADZ - 1.85
            elif n >= FRONT["bank_top"] and inside(u, n):
                v.co.z = (UPZ if n > n_edge(u) + 0.3 else PADZ) - 0.08
    ob.data.update()


GLB_TEX = {"clad": ("bb_clad.png", 0.25), "limewash": ("limewash.png", 0.5), "paving": ("paving.png", 0.25),
           "coping": ("paving.png", 0.25), "slats": ("cedar.png", 0.25), "cedar": ("cedar.png", 0.25),
           "bluestone": ("bluestone.png", 0.6)}
GLB_COL = {"roof_x": ("#232427", 0.42, 0.55), "roof_y": ("#232427", 0.42, 0.55), "black": ("#121314", 0.4, 0.3),
           "timber": ("#7a5636", 0.6, 0), "lawn": ("#5f7d38", 0.95, 0), "mulch": ("#4f3f31", 1.0, 0),
           "concrete": ("#bbb5aa", 0.9, 0), "kerb": ("#b6b0a5", 0.9, 0), "asphalt": ("#38393b", 0.9, 0),
           "aggregate": ("#a39b8e", 0.9, 0), "gravel": ("#b9b2a6", 0.95, 0), "wallrender": ("#46464a", 0.9, 0),
           "pooltile": ("#7fc4d8", 0.3, 0), "water": ("#3fa9cf", 0.05, 0), "bark": ("#a09684", 0.85, 0),
           "darkbark": ("#4d443b", 0.9, 0), "grass": ("#a8874a", 1.0, 0), "fire": ("#ff7a2a", 0.5, 0),
           "floor": ("#a07a52", 0.4, 0), "plaster": ("#ece6dc", 0.9, 0), "glass": ("#0e1416", 0.05, 0.0),
           "bulb": ("#fff2d8", 0.5, 0), "roofflat": ("#2a2b2d", 0.8, 0), "fence_ne": ("#d8cdb0", 0.5, 0.3),
           "fence_sw": ("#9aa58f", 0.5, 0.3)}
INTERIOR = {"fabric", "linen", "walnut", "rug", "art", "lamp"}
EXTERIOR_LIGHTS = ("lantern", "door_lantern", "garage_lamp", "verandah", "pool", "wall_up", "olive_up", "garage_up",
                   "sconce", "soffit", "cabana")
LEAVES = ("euc", "olive", "shrub", "hedge", "plume")


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
        nt = m.node_tree
        if name in GLB_TEX or name in LEAVES or name == "terrain":
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
            if name in ("bulb", "fire"):
                bsdf.inputs["Emission Color"].default_value = (1.0, 0.75, 0.45, 1)
                bsdf.inputs["Emission Strength"].default_value = 4.0
        ob.data.materials[0] = m
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", export_lights=True, export_cameras=False,
                              export_yup=True, export_apply=True, export_image_format="JPEG", export_jpeg_quality=85)


def export_mesh(path):
    """house geometry (no terrain, street, plants) as triangles with base colours, for black_barn_site.py finish"""
    skip = {"terrain", "backdrop", "asphalt", "kerb", "concrete", "aggregate", "mulch", "euc", "olive", "shrub", "plume",
            "hedge", "bark", "darkbark", "glass", "bulb", "lamp", "fire", "fabric", "linen", "walnut", "rug", "art"}
    colours = {"clad": (42, 43, 45), "limewash": (236, 234, 228), "bluestone": (120, 122, 124), "roof_x": (35, 36, 39),
               "roof_y": (35, 36, 39), "black": (20, 20, 21), "cedar": (176, 122, 80), "slats": (176, 122, 80),
               "paving": (212, 205, 190), "coping": (225, 220, 208), "lawn": (95, 122, 58), "water": (64, 172, 212),
               "pooltile": (127, 196, 216), "floor": (160, 122, 82), "plaster": (236, 230, 220), "gravel": (185, 178, 166),
               "wallrender": (70, 70, 74)}
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
    "street": dict(eye=(9.5, -14.0, 2.2), at=(11.0, 10.0, 3.9), lens=24),
    "courtyard": dict(eye=(5.0, 43.5, 7.2), at=(14.8, 21.5, -0.5), lens=22),
    "aerial": dict(eye=(48.0, -24.0, 42.0), at=(12.0, 22.0, 0.0), lens=30),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="street")
    ap.add_argument("--scale", type=int, default=50)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "designs" / "black_barn_street.png"))
    ap.add_argument("--exposure", type=float, default=0.85)
    ap.add_argument("--sun", type=float, default=3.0)
    ap.add_argument("--export-mesh", action="store_true")
    ap.add_argument("--export-glb", default="")
    a = ap.parse_args()
    assert L.get("upper_z"), "run scripts/designs/black_barn_site.py ground first"

    bpy.ops.wm.read_factory_settings(use_empty=True)
    cr.make_textures()
    textures()
    MATS.update({
        "clad": img_mat("clad", "bb_clad.png", 0.25, 0.82, bump=0.3),
        "limewash": img_mat("limewash", "limewash.png", 0.5, 0.92, bump=0.8),
        "cedar": img_mat("cedar", "cedar.png", 0.25, 0.55, bump=0.1),
        "slats": img_mat("slats", "cedar.png", 0.25, 0.55, bump=0.3),
        "plaster": flat_mat("plaster", srgb("#ece6dc"), 0.9),
        "paving": img_mat("paving", "paving.png", 0.25, 0.8, bump=0.1, tint=(0.93, 0.93, 0.93)),
        "coping": img_mat("coping", "paving.png", 0.25, 0.75),
        "bluestone": img_mat("bluestone", "bluestone.png", 0.6, 0.85, bump=0.5),
        "roof_x": seam_roof("roof_x", "X"), "roof_y": seam_roof("roof_y", "Y"),
        "black": flat_mat("black", srgb("#121314"), 0.45, 0.3),
        "roofflat": noisy_mat("roofflat", "#262729", "#333436", 20, 0.8, 0.05),
        "fence_ne": fence_mat("fence_ne", "#d8cdb0"), "fence_sw": fence_mat("fence_sw", "#9aa58f"),
        "glass": thin_glass("glass"),
        "floor": flat_mat("floor", srgb("#a07a52"), 0.35),
        "fabric": flat_mat("fabric", srgb("#d6ccbc"), 0.95),
        "linen": flat_mat("linen", srgb("#eee8dd"), 0.95),
        "walnut": flat_mat("walnut", srgb("#4a3424"), 0.45),
        "rug": flat_mat("rug", srgb("#b8ab96"), 1.0),
        "art": flat_mat("art", srgb("#8a6a52"), 0.7),
        "lamp": flat_mat("lamp", (1, 1, 1), 0.5, emit=WARM, strength=14),
        "bulb": flat_mat("bulb", (1, 1, 1), 0.5, emit=WARM, strength=25),
        "lawn": noisy_mat("lawn", "#4f6e2f", "#7a9446", 30, 0.95, 0.3),
        "mulch": noisy_mat("mulch", "#3d2f24", "#6b5644", 40, 1.0, 0.4),
        "gravel": noisy_mat("gravel", "#9d978c", "#d2ccc1", 140, 0.95, 0.45),
        "wallrender": noisy_mat("wallrender", "#3c3c3e", "#4c4c4f", 6, 0.9, 0.08),
        "concrete": noisy_mat("concrete", "#aaa499", "#c9c3b8", 8, 0.9, 0.05),
        "kerb": flat_mat("kerb", srgb("#b6b0a5"), 0.9),
        "asphalt": noisy_mat("asphalt", "#2a2b2d", "#4a4a4b", 90, 0.9, 0.3),
        "aggregate": noisy_mat("aggregate", "#8f877a", "#c2b9aa", 70, 0.9, 0.25),
        "timber": noisy_mat("timber", "#6b4a30", "#8c6644", 3, 0.6, 0.1),
        "pooltile": flat_mat("pooltile", srgb("#7fc4d8"), 0.3),
        "bark": noisy_mat("bark", "#8a8070", "#b0a796", 6, 0.85, 0.2),
        "darkbark": flat_mat("darkbark", srgb("#4d443b"), 0.9),
        "euc": leaf_mat("euc"), "olive": leaf_mat("olive"), "shrub": leaf_mat("shrub"), "hedge": leaf_mat("hedge"),
        "plume": leaf_mat("plume"),
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
    adjust_terrain()
    build_house()
    build_courtyard()
    build_site_walls()
    build_street()
    build_fences()
    if a.export_mesh:
        export_mesh(D3 / "black_barn_mesh.json")
    build_garden()
    cr.flush()
    if a.export_glb:
        export_glb(a.export_glb)
        print("exported", a.export_glb)
        return
    cr.VIEWS.update(VIEWS)
    cr.setup_world_and_camera(a.view, a.sun, 232.0, a.exposure)

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
