"""Cycles render of "Hill Country Stone" (hill_country): pale rough limestone, black steel arched windows, a long low
standing-seam roof between steep stone gables, a timber loggia, and a U-shaped plan around a north pool courtyard.

    python hill_country_cycles.py --view street|courtyard|aerial --scale 50 --samples 32 --out x.png [--export-glb f.glb]

Run scripts/designs/hill_country_site.py ground first: it writes render_scene.json (designed ground) and
data/3d/designs/hill_country_layout.json (layout, ground-cover zones, retaining walls, fine terrain patch).
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
from mathutils import Vector  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

import cycles_render as cr  # noqa: E402
from cycles_render import (ARC, GZ, PADZ, B, MATS, TEX, arc_n, box, card, cyl, drape, faces, poly, slab,  # noqa: E402
                           zg, light, bush, tree, WARM, srgb, img_mat, flat_mat, noisy_mat, thin_glass, leaf_mat,
                           terrain_mat, node_mat)

D3 = ROOT / "data" / "3d"
LAYF = json.loads((D3 / "designs" / "hill_country_layout.json").read_text())
L = LAYF["layout"]
FRONT = LAYF["front"]
PAD_OUTLINE = LAYF["pad_outline"]
LOTP = LAYF["lot"]
UPZ = FRONT["upper"]
TN = L["terrace_n"]
rng = random.Random(9)
rr = rng.uniform

T = 0.4                        # stone wall thickness
EAVE = 3.0   # wing eaves (pitches come from the layout: west 53 deg attic storey, east 45 deg)


def inside_lot(u, n, pts=LOTP):
    c = False
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 > n) != (y1 > n) and u < x0 + (n - y0) * (x1 - x0) / (y1 - y0):
            c = not c
    return c


def zd(u, n):
    """designed ground above FFL"""
    if n < 2.2:
        return zg(u, n)
    if n < L["bank_top"]:
        return np.interp(n, [2.2, L["bank_top"]], [FRONT["wall_top"], PADZ])
    return PADZ if n < TN else UPZ


# ------------------------------------------------------------------ textures
def textures():
    R = random.Random(21)
    # rough-faced pale limestone in random courses: 2048 px = 4 m
    S = 2048
    arr = np.zeros((S, S, 3), np.float32)
    arr[:] = (196, 188, 170)                       # lime mortar
    yy, xx = np.mgrid[0:S, 0:S]
    y = 0
    while y < S:
        ch = int(R.choice((0.16, 0.2, 0.24, 0.3, 0.36)) * 512)
        ch = min(ch, S - y)
        x = -int(R.uniform(0, 300))
        while x < S:
            bw = int(R.uniform(0.35, 0.95) * 512)
            x0, x1, y0, y1 = max(x + 6, 0), min(x + bw - 6, S), y + 6, y + ch - 6
            if x1 > x0 and y1 > y0:
                tone = R.uniform(-14, 10)
                warm = R.uniform(-6, 8)
                base = np.array((226 + tone + warm, 216 + tone, 192 + tone - warm * 1.4))
                h, w = y1 - y0, x1 - x0
                sy, sx = np.mgrid[0:h, 0:w]
                # pillowed rough face: darker towards the arrises
                ey = np.minimum(sy, h - 1 - sy) / max(h, 1)
                ex = np.minimum(sx, w - 1 - sx) / max(w, 1)
                pill = np.clip(np.minimum(ey * 6, 1) * np.minimum(ex * 9, 1), 0, 1) ** 0.6
                arr[y0:y1, x0:x1] = base[None, None, :] * (0.82 + 0.18 * pill[..., None])
            x += bw
        y += ch
    noise = np.random.default_rng(5)
    for s_, amp in ((4, 10), (16, 12), (64, 8)):
        n_ = noise.normal(0, 1, (S // s_, S // s_)).astype(np.float32)
        n_ = np.array(Image.fromarray(n_).resize((S, S), Image.BILINEAR))
        arr += n_[..., None] * amp
    arr += noise.integers(-8, 8, (S, S))[..., None]
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)).save(TEX / "hcstone.png")
    # weathered timber boards for shutters and doors: 512 = 2 m
    im = Image.new("RGB", (512, 512), (60, 48, 38))
    d = ImageDraw.Draw(im)
    for x in range(0, 512, 40):
        l = R.uniform(-10, 10)
        d.rectangle([x + 2, 0, x + 37, 512], fill=(int(116 + l), int(98 + l), int(80 + l)))
        for _ in range(20):
            gx = x + R.uniform(4, 34)
            d.line([gx, 0, gx + R.uniform(-3, 3), 512], fill=(int(100 + l), int(84 + l), int(68 + l)), width=1)
    im.filter(ImageFilter.GaussianBlur(0.6)).save(TEX / "boards.png")
    # lavender: grey-green needles with purple spikes
    im = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for _ in range(420):
        cx, cy = R.uniform(30, 482), R.uniform(30, 482)
        if math.hypot(cx - 256, cy - 256) > 230:
            continue
        a = R.uniform(0, 6.28)
        Lr = R.uniform(10, 22)
        l = R.uniform(-15, 15)
        d.line([cx, cy, cx + math.cos(a) * Lr, cy + math.sin(a) * Lr], fill=(int(120 + l), int(134 + l), int(112 + l), 255), width=3)
    for _ in range(70):
        cx, cy = R.uniform(60, 452), R.uniform(40, 330)
        l = R.uniform(-20, 20)
        for k in range(7):
            d.ellipse([cx - 5, cy + k * 8 - 5, cx + 5, cy + k * 8 + 5], fill=(int(128 + l), int(96 + l), int(178 + l), 255))
    im.save(TEX / "leaf_lavender.png")
    # ornamental grass: straw blades fanning up from the base, with buff plumes (card UV: v=0 at the base)
    im = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for _ in range(150):
        x0 = 256 + R.uniform(-40, 40)
        lean = R.uniform(-1, 1)
        h = R.uniform(300, 500)
        pts = [(x0 + lean * 230 * (t / 8) ** 1.6, 511 - h * t / 8) for t in range(9)]
        l = R.uniform(-20, 20)
        g = R.random() < 0.35
        col = (int(150 + l), int(152 + l), int(96 + l), 255) if g else (int(204 + l), int(178 + l), int(122 + l), 255)
        d.line(pts, fill=col, width=3)
        if R.random() < 0.5:
            tx, ty = pts[-1]
            d.ellipse([tx - 9, ty - 4, tx + 9, ty + 34], fill=(int(232 + l / 2), int(214 + l / 2), int(170 + l / 2), 255))
    im.save(TEX / "leaf_grass.png")


def fence_mat(name, hexc):
    """Colorbond corrugated panel: vertical corrugations every 76 mm along the fence (object Y ~ n)"""
    m, nt, b = node_mat(name)
    b.inputs["Base Color"].default_value = (*srgb(hexc), 1)
    b.inputs["Metallic"].default_value = 0.3
    b.inputs["Roughness"].default_value = 0.5
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = 2 * math.pi / 0.076
    nt.links.new(sep.outputs["Y"], mul.inputs[0])
    sn = nt.nodes.new("ShaderNodeMath"); sn.operation = "SINE"
    nt.links.new(mul.outputs[0], sn.inputs[0])
    bn = nt.nodes.new("ShaderNodeBump"); bn.inputs["Strength"].default_value = 0.5; bn.inputs["Distance"].default_value = 0.01
    nt.links.new(sn.outputs[0], bn.inputs["Height"])
    nt.links.new(bn.outputs["Normal"], b.inputs["Normal"])
    return m


def roof_mat(name, axis):
    m, nt, b = node_mat(name)
    b.inputs["Base Color"].default_value = (*srgb("#5a5e61"), 1)
    b.inputs["Metallic"].default_value = 0.55
    b.inputs["Roughness"].default_value = 0.45
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    div = nt.nodes.new("ShaderNodeMath"); div.operation = "DIVIDE"; div.inputs[1].default_value = 0.42
    nt.links.new(sep.outputs[axis], div.inputs[0])
    fr = nt.nodes.new("ShaderNodeMath"); fr.operation = "FRACT"
    nt.links.new(div.outputs[0], fr.inputs[0])
    lt = nt.nodes.new("ShaderNodeMath"); lt.operation = "LESS_THAN"; lt.inputs[1].default_value = 0.07
    nt.links.new(fr.outputs[0], lt.inputs[0])
    bn = nt.nodes.new("ShaderNodeBump"); bn.inputs["Strength"].default_value = 0.7; bn.inputs["Distance"].default_value = 0.02
    nt.links.new(lt.outputs[0], bn.inputs["Height"])
    nt.links.new(bn.outputs["Normal"], b.inputs["Normal"])
    return m


# ------------------------------------------------------------------ axis-aware geometry
def P3(axis, x, t, z):
    return (x, t, z) if axis == "u" else (t, x, z)


def bx(axis, x0, x1, t0, t1, z0, z1, mat):
    if axis == "u":
        box(x0, x1, t0, t1, z0, z1, mat)
    else:
        box(t0, t1, x0, x1, z0, z1, mat)


def ring_ax(axis, outer, inner, t0, t1, mat):
    k = len(outer)
    v = [P3(axis, x, t0, z) for x, z in outer] + [P3(axis, x, t0, z) for x, z in inner] + \
        [P3(axis, x, t1, z) for x, z in outer] + [P3(axis, x, t1, z) for x, z in inner]
    fcs = []
    for i in range(k):
        j = (i + 1) % k
        fcs += [(i, j, k + j, k + i), (2 * k + j, 2 * k + i, 3 * k + i, 3 * k + j),
                (i, 2 * k + i, 2 * k + j, j), (k + j, 3 * k + j, 3 * k + i, k + i)]
    faces(mat, v, fcs)


def solid(axis, pts, t0, t1, mat):
    k = len(pts)
    v = [P3(axis, x, t0, z) for x, z in pts] + [P3(axis, x, t1, z) for x, z in pts]
    fcs = [list(range(k)), [k + i for i in reversed(range(k))]] + [(i, (i + 1) % k, k + (i + 1) % k, k + i) for i in range(k)]
    faces(mat, v, fcs)


def ray_hit(pts, c, ang):
    dx, dz = math.cos(ang), math.sin(ang)
    best = None
    for i in range(len(pts)):
        p, q = pts[i], pts[(i + 1) % len(pts)]
        ex, ez = q[0] - p[0], q[1] - p[1]
        det = -dx * ez + ex * dz
        if abs(det) < 1e-12:
            continue
        rx, rz = p[0] - c[0], p[1] - c[1]
        t = (rx * -ez + ex * rz) / det
        s = (dx * rz - dz * rx) / det
        if t > 1e-9 and -1e-9 <= s <= 1 + 1e-9 and (best is None or t < best):
            best = t
    return (c[0] + dx * best, c[1] + dz * best)


def panel(axis, outer, inner, c, t0, t1, mat):
    """convex wall panel `outer` with one convex hole `inner` (both CCW in x-z) around centre c"""
    angs = sorted({round(math.atan2(z - c[1], x - c[0]), 9) for x, z in list(outer) + list(inner)})
    o = [ray_hit(outer, c, a) for a in angs]
    i = [ray_hit(inner, c, a) for a in angs]
    ring_ax(axis, o, i, t0, t1, mat)


class Op:
    """opening in a wall plane: kind 'arch' (semicircular head), 'rect' or 'round'"""
    def __init__(self, a, b, zb, zt, kind="arch", cols=2, transoms=(), door=False):
        self.a, self.b, self.zb, self.zt, self.kind = a, b, zb, zt, kind
        self.cols, self.transoms, self.door = cols, transoms, door
        self.r = (b - a) / 2
        self.cx = (a + b) / 2
        self.zs = zt - self.r if kind == "arch" else zt

    def poly(self, d=0.0, seg=16):
        a, b, zb, zt = self.a + d, self.b - d, self.zb + d, self.zt - d
        if self.kind == "rect":
            return [(a, zb), (b, zb), (b, zt), (a, zt)]
        if self.kind == "round":
            cz = (self.zb + self.zt) / 2
            return [(self.cx + (self.r - d) * math.cos(2 * math.pi * k / 20), cz + (self.r - d) * math.sin(2 * math.pi * k / 20)) for k in range(20)]
        r = self.r - d
        return [(a, zb), (b, zb)] + [(self.cx + r * math.cos(math.pi * k / seg), self.zs + r * math.sin(math.pi * k / seg)) for k in range(seg + 1)]

    def centre(self):
        return (self.cx, (self.zb + self.zt) / 2 if self.kind != "arch" else (self.zb + self.zs) / 2 + 0.1)

    def top(self, x):
        if self.kind == "arch":
            return self.zs + math.sqrt(max(self.r ** 2 - (x - self.cx) ** 2, 0))
        return self.zt


def wall(axis, x0, x1, t0, t1, z0, z1, ops, mat="stone", m=0.3):
    cur = x0
    for op in sorted(ops, key=lambda o: o.a):
        pa, pb = max(cur, op.a - m), min(x1, op.b + m)
        if pa > cur + 1e-6:
            bx(axis, cur, pa, t0, t1, z0, z1, mat)
        panel(axis, [(pa, z0), (pb, z0), (pb, z1), (pa, z1)], op.poly(), op.centre(), t0, t1, mat)
        cur = pb
    if x1 > cur + 1e-6:
        bx(axis, cur, x1, t0, t1, z0, z1, mat)


def gable(axis, x0, x1, t0, t1, ze, pitch, z0=None, op=None, mat="stone"):
    """gable end wall: triangle above the eave (or a full pentagon from z0), optional single opening"""
    xm = (x0 + x1) / 2
    zr = ze + (xm - x0) * math.tan(pitch)
    pts = ([(x0, z0), (x1, z0)] if z0 is not None else [(x0, ze), (x1, ze)]) + ([(x1, ze)] if z0 is not None else []) + [(xm, zr)] + \
          ([(x0, ze)] if z0 is not None else [])
    if op is None:
        solid(axis, pts, t0, t1, mat)
    else:
        panel(axis, pts, op.poly(), op.centre(), t0, t1, mat)
    return zr


def glaze(axis, face, outward, op, glass=True):
    """black steel frame + glazing bars set 0.12 m back from the wall face"""
    tp = face - outward * 0.12
    pts = op.poly()
    if glass:
        poly([P3(axis, x, tp, z) for x, z in pts], "glass" if not op.door else "glass")
    ring_ax(axis, pts, op.poly(0.06), tp - 0.035, tp + 0.035, "black")
    w = 0.045
    for k in range(1, op.cols):
        x = op.a + (op.b - op.a) * k / op.cols
        bx(axis, x - w / 2, x + w / 2, tp - 0.03, tp + 0.03, op.zb, op.top(x), "black")
    for z in op.transoms:
        if op.kind == "arch" and z > op.zs:
            h = math.sqrt(max(op.r ** 2 - (z - op.zs) ** 2, 0))
            xa, xb = op.cx - h, op.cx + h
        else:
            xa, xb = op.a, op.b
        bx(axis, xa, xb, tp - 0.03, tp + 0.03, z - w / 2, z + w / 2, "black")
    if op.kind == "arch" and op.cols >= 2:   # radial bars in the head, as in the reference
        for k in range(1, op.cols):
            ang = math.pi * k / op.cols
            x1, z1_ = op.cx + op.r * math.cos(ang) * 0.98, op.zs + op.r * math.sin(ang) * 0.98
            if abs(math.cos(ang)) < 1e-6:
                continue
    # stone sill
    if not op.door:
        bx(axis, op.a - 0.08, op.b + 0.08, *sorted((face, face + outward * 0.06)), op.zb - 0.08, op.zb, "limestone")


def shutters(axis, face, outward, op):
    w = (op.b - op.a) / 2
    t0, t1 = sorted((face + outward * 0.02, face + outward * 0.07))
    for a, b in ((op.a - w - 0.05, op.a - 0.05), (op.b + 0.05, op.b + w + 0.05)):
        bx(axis, a, b, t0, t1, op.zb, op.zt, "shutter")
        for z in (op.zb + 0.35, op.zt - 0.35):      # black strap hinges
            bx(axis, a + 0.03, b - 0.03, t0 - 0.01 * outward, t1 + 0.01 * outward, z - 0.02, z + 0.02, "black")


def plane(p0, p1, p2, p3, mat, t=0.1, lining=True):
    """roof plane (upward slab) with a pale lining underneath"""
    slab(p0, p1, p2, p3, t, mat)
    if lining:
        q = [(x, y, z - 0.03) for x, y, z in (p0, p1, p2, p3)]
        slab(*q, -0.02, "ceiling")


def roof_n(n0, n1, u0, u1, ze, pitch, oe=0.45, og=0.3, mat="roof_y", fasc=True, gut=True, gap=None):
    """two planes, ridge parallel to n (gables at n0/n1); gap=(na, nb, u_stop) cuts the east plane for a wall dormer"""
    um = (u0 + u1) / 2
    tp = math.tan(pitch)
    zr = ze + (um - u0) * tp
    dz = oe * tp
    zl = ze - dz
    plane((u0 - oe, n1 + og, zl), (u0 - oe, n0 - og, zl), (um, n0 - og, zr + 0.06), (um, n1 + og, zr + 0.06), mat)
    spans = [(n0 - og, n1 + og)] if gap is None else [(n0 - og, gap[0]), (gap[1], n1 + og)]
    for a, b in spans:
        plane((u1 + oe, a, zl), (u1 + oe, b, zl), (um, b, zr + 0.06), (um, a, zr + 0.06), mat)
    if gap is not None:
        zs = zr - (gap[2] - um) * tp
        plane((gap[2], gap[0], zs), (gap[2], gap[1], zs), (um, gap[1], zr + 0.06), (um, gap[0], zr + 0.06), mat)
    box(um - 0.08, um + 0.08, n0 - og, n1 + og, zr + 0.1, zr + 0.2, "roof_x")          # ridge capping
    if fasc:
        for n in (n0 - og, n1 + og):
            slab((u0 - oe, n, zl - 0.12), (um, n, zr - 0.06), (um, n, zr + 0.18), (u0 - oe, n, zl + 0.16), 0.05, "black")
            slab((um, n, zr - 0.06), (u1 + oe, n, zl - 0.12), (u1 + oe, n, zl + 0.16), (um, n, zr + 0.18), 0.05, "black")
    if gut:
        box(u0 - oe - 0.07, u0 - oe + 0.07, n0 - og, n1 + og, zl - 0.16, zl + 0.02, "black")
        for a, b in spans:
            box(u1 + oe - 0.07, u1 + oe + 0.07, a, b, zl - 0.16, zl + 0.02, "black")
    return zr


def roof_u(u0, u1, n0, n1, ze, pitch, oe=0.2, og=0.3, mat="roof_x"):
    """two planes, ridge parallel to u (gable at u1)"""
    nm = (n0 + n1) / 2
    tp = math.tan(pitch)
    zr = ze + (nm - n0) * tp
    zl = ze - oe * tp
    plane((u0, n0 - oe, zl), (u1 + og, n0 - oe, zl), (u1 + og, nm, zr + 0.05), (u0, nm, zr + 0.05), mat)
    plane((u1 + og, n1 + oe, zl), (u0, n1 + oe, zl), (u0, nm, zr + 0.05), (u1 + og, nm, zr + 0.05), mat)
    u = u1 + og
    slab((u, n0 - oe, zl - 0.12), (u, nm, zr - 0.07), (u, nm, zr + 0.18), (u, n0 - oe, zl + 0.15), 0.05, "black")
    slab((u, nm, zr - 0.07), (u, n1 + oe, zl - 0.12), (u, n1 + oe, zl + 0.15), (u, nm, zr + 0.18), 0.05, "black")
    return zr


def juliet(axis, face, outward, a, b, z0, z1=None):
    """black steel Juliet balcony rail on a wall face"""
    z1 = z1 if z1 is not None else z0 + 1.0
    t = face + outward * 0.14
    t0, t1 = sorted((t - 0.02, t + 0.02))
    for z in (z0 + 0.08, z1 - 0.04):
        bx(axis, a - 0.05, b + 0.05, t0, t1, z - 0.025, z + 0.025, "black")
    for x in np.arange(a, b + 1e-6, 0.11):
        bx(axis, x - 0.01, x + 0.01, t0, t1, z0 + 0.08, z1 - 0.04, "black")
    for x in (a - 0.05, b + 0.05):
        bx(axis, x - 0.025, x + 0.025, *sorted((face, t)), z1 - 0.07, z1 - 0.01, "black")
    bx(axis, a - 0.15, b + 0.15, *sorted((face, face + outward * 0.28)), z0 - 0.12, z0, "limestone")


def wall_pier(axis, x0, x1, face, outward, z0, z1, d=0.35):
    """projecting stone pier / buttress with a sloped limestone weathering"""
    t0, t1 = sorted((face, face + outward * d))
    bx(axis, x0, x1, t0, t1, z0, z1, "stone")
    tf = face + outward * (d + 0.04)
    p2 = [P3(axis, x0, face, z1 + 0.2), P3(axis, x1, face, z1 + 0.2), P3(axis, x1, tf, z1 - 0.05), P3(axis, x0, tf, z1 - 0.05)]
    faces("limestone", p2 + [(x, y, z + 0.08) for x, y, z in p2],
          [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    bx(axis, x0, x1, t0, t1, z1, z1 + 0.15, "stone")


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


def wall_lantern(x, n, z, outward):
    """bracket lantern on a wall face at n (street side outward=-1)"""
    box(x - 0.06, x + 0.06, n + outward * 0.08, n, z + 0.1, z + 0.5, "black")
    lantern(x, n + outward * 0.24, z - 0.05, 0.22, 16)


def cr_pendant(x, n, z, strength=60):
    cyl((x, n, z + 0.15), (x, n, z + 1.4), 0.006, 0.006, "black")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.16, location=(x, n, z))
    bpy.context.active_object.data.materials.append(MATS["bulb"])
    light("POINT", (x, n, z - 0.2), strength, WARM, size=0.1, name="pendant")


def room_light(u0, u1, n0, n1, z, power):
    light("AREA", ((u0 + u1) / 2, (n0 + n1) / 2, z), power, WARM, aim=((u0 + u1) / 2, (n0 + n1) / 2, -5),
          size=min(u1 - u0, n1 - n0) * 0.6, name="room")


def rock(c, r, flat=0.6):
    """low-poly boulder"""
    cu, cn, cz = c
    lat, lon = 5, 9
    v = [(cu, cn, cz + r * flat * rr(0.8, 1.05))]
    for i in range(1, lat):
        th = math.pi * i / lat
        for j in range(lon):
            ph = 2 * math.pi * j / lon + rr(-0.15, 0.15)
            k = rr(0.78, 1.12)
            v.append((cu + r * k * math.sin(th) * math.cos(ph) * 1.2, cn + r * k * math.sin(th) * math.sin(ph),
                      cz + r * flat * k * math.cos(th)))
    v.append((cu, cn, cz - r * flat))
    fcs = [(0, 1 + (j + 1) % lon, 1 + j) for j in range(lon)]
    for i in range(lat - 2):
        for j in range(lon):
            a, b = 1 + i * lon + j, 1 + i * lon + (j + 1) % lon
            fcs.append((a, b, b + lon, a + lon))
    last = len(v) - 1
    fcs += [(last, 1 + (lat - 2) * lon + j, 1 + (lat - 2) * lon + (j + 1) % lon) for j in range(lon)]
    faces("boulder", v, fcs)


def grass(u, n, z, h):
    """ornamental grass clump: three crossed vertical cards"""
    for k in range(3):
        a = k * math.pi / 3 + rr(0, 1)
        dx, dn = math.cos(a) * h * 0.55, math.sin(a) * h * 0.55
        faces("grass", [(u - dx, n - dn, z - 0.05), (u + dx, n + dn, z - 0.05), (u + dx, n + dn, z + h), (u - dx, n - dn, z + h)],
              [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]])


def lavender(u, n, z, r=0.36):
    bush((u, n, z + r * 0.7), r, "lavender", 12)


def cypress(u, n, z, h=5.0):
    for k in range(9):
        t = k / 8
        rad = 0.55 * (1 - t) ** 0.8 + 0.08
        bush((u, n, z + 0.4 + t * (h - 0.6)), rad, "hedge", 10)


# ------------------------------------------------------------------ the house
def hip_house(q, wall_hex, roof_hex, h, pitch, k):
    """simple rendered neighbour house with a hip roof, quad corners in (u, n)"""
    P = [Vector((u, n, 0)) for u, n in q]
    if (P[1] - P[0]).length < (P[2] - P[1]).length:
        P = P[1:] + P[:1]
    base = min(zg(p.x, p.y) for p in P) - 0.1
    wm, rm = f"nwall{k}", f"nroof{k}"
    MATS[wm] = flat_mat(wm, srgb(wall_hex), 0.8)
    MATS[rm] = flat_mat(rm, srgb(roof_hex), 0.7)
    faces(wm, [(p.x, p.y, base - 1.0) for p in P] + [(p.x, p.y, base + h) for p in P],
          [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    cen = sum(P, Vector()) / 4
    E = [cen + (p - cen) + (p - cen).normalized() * 0.6 for p in P]
    dvec = (P[1] - P[0]).normalized()
    hs = (P[3] - P[0]).length / 2 + 0.42
    hl = (P[1] - P[0]).length / 2 + 0.42
    zr = base + h + hs * math.tan(math.radians(pitch))
    R0, R1 = cen - dvec * (hl - hs), cen + dvec * (hl - hs)
    v = [(e.x, e.y, base + h) for e in E] + [(R0.x, R0.y, zr), (R1.x, R1.y, zr)]
    faces(rm, v, [(0, 1, 5, 4), (1, 2, 5), (2, 3, 4, 5), (3, 0, 4)])


def build_house():
    RF = L["roofs"]
    wu0, wu1, wn0, wn1 = L["west"]
    eu0, eu1, en0, en1 = L["east"]
    mu0, mu1, mn0, mn1 = L["main"]
    xu0, xu1, xn0, xn1 = L["entry"]
    bu0, bu1, bn0, bn1 = L["bay"]
    PW, PE = math.radians(RF["west"]["pitch"]), math.radians(RF["east"]["pitch"])
    for r_ in (L["west"], L["east"], L["main"], L["entry"], L["bay"], L["bay_e"]):
        box(r_[0], r_[1], r_[2], r_[3], -0.25, 0.0, "floor")

    # ---- the two stone gable wings: tall arched steel glass gables to the north, long side walls with piers
    zr_w = zr_e = 0
    for (u0, u1, n0, n1), side, pitch in ((L["west"], -1, PW), (L["east"], 1, PE)):
        c = (u0 + u1) / 2
        z0 = PADZ - 0.35
        big = Op(c - 1.2, c + 1.2, 0.0, 5.2, "arch", cols=3, transoms=(2.3, 4.0), door=True)
        gable("u", u0, u1, n1 - T, n1, EAVE, pitch, z0=z0, op=big)
        glaze("u", n1, 1, big)
        outer_u, inner_u = (u0, u1) if side < 0 else (u1, u0)
        small = [Op(n - 0.45, n + 0.45, 1.0, 2.45, "arch", cols=1, transoms=(1.6,)) for n in (10.0, 16.8, 19.2, 25.3)]
        ws = (outer_u, outer_u + T) if side < 0 else (outer_u - T, outer_u)
        wall("n", n0 + T, n1 - T, ws[0], ws[1], z0, EAVE, small)
        for o in small:
            glaze("n", outer_u, side, o)
        piers = [(21.6, 22.6)] + ([(13.0, 14.0)] if side > 0 else [])
        for a, b in piers:                                       # articulation: stone buttresses every < 10 m
            wall_pier("n", a, b, outer_u, side, z0, 2.4)
        doors = [Op(21.6, 23.2, 0.0, 2.75, "arch", cols=2, transoms=(2.0,), door=True),
                 Op(24.6, 26.2, 0.0, 2.75, "arch", cols=2, transoms=(2.0,), door=True)]
        wi = (inner_u - T, inner_u) if side < 0 else (inner_u, inner_u + T)
        wall("n", n0 + T, n1 - T, wi[0], wi[1], z0, EAVE, doors)
        for o in doors:
            glaze("n", inner_u, -side, o)
        gap = (7.9, 9.9, 4.34) if side < 0 else None
        zr = roof_n(n0, n1, u0, u1, EAVE, pitch, gap=gap)
        if side < 0:
            zr_w = zr
        else:
            zr_e = zr
        room_light(u0 + 0.6, u1 - 0.6, n1 - 6.0, n1 - 0.8, 2.9, 900)
        room_light(u0 + 0.6, u1 - 0.6, 18.5, 24.0, 2.9, 700)
        room_light(u0 + 0.6, u1 - 0.6, n0 + 1.0, 14.0, 2.6, 500)
        cr_pendant(c, n1 - 3.0, 2.6, 50)
        box(c - 1.3, c + 1.3, n1 - 4.6, n1 - 3.7, 0, 0.42, "fabric")
        box(c - 1.3, c + 1.3, n1 - 4.6, n1 - 4.35, 0.42, 0.8, "fabric")
        box(c - 1.6, c + 1.6, n1 - 4.0, n1 - 1.6, 0.0, 0.01, "rug")
        box(c - 0.5, c + 0.5, n1 - 2.6, n1 - 2.0, 0.0, 0.4, "walnut")

    # ---- west wing street gable: corner piers, small shuttered arched window below, the upper suite's
    #      arched French window with a Juliet balcony in the gable (street / SSE view)
    c = (wu0 + wu1) / 2
    w = Op(c - 0.6, c + 0.6, 0.85, 2.6, "arch", cols=2, transoms=(1.75,))
    wall("u", wu0, wu1, wn0, wn0 + T, PADZ - 0.35, EAVE, [w])
    glaze("u", wn0, -1, w)
    shutters("u", wn0, -1, w)
    for a, b in ((L["pier_w"][0], L["pier_w"][1]), (L["pier_e"][0], L["pier_e"][1])):
        wall_pier("u", a, b, wn0, -1, PADZ - 0.35, 2.6, d=wn0 - L["pier_w"][2])
    up = Op(c - 1.0, c + 1.0, 3.15, 5.6, "arch", cols=3, transoms=(4.55,), door=True)
    gable("u", wu0, wu1, wn0, wn0 + T, EAVE, PW, op=up)
    glaze("u", wn0, -1, up)
    juliet("u", wn0, -1, up.a, up.b, 3.15)
    box(wu0 + T, wu1 - T, wn0 + T, 17.0, 2.85, 3.1, "slab")                     # attic-storey floor
    gable("u", wu0 + T, wu1 - T, 17.0, 17.2, 3.45, PW, z0=3.1, mat="ceiling")    # suite back wall under the roof
    light("AREA", (c, 10.5, 5.6), 380, WARM, aim=(c, 10.5, 0), size=1.8, name="room")
    box(c - 0.9, c + 0.9, 11.5, 13.6, 3.1, 3.6, "linen")                         # bed
    box(c - 0.9, c + 0.9, 13.5, 13.65, 3.1, 4.3, "walnut")
    # east wall dormer of the suite: arched French window + Juliet balcony looking east / ESE over the loggia
    # roof; obscure lower glazing for privacy towards the NE neighbour's yard, clear above for the long view
    dn0, dn1, dzt = 7.9, 9.9, 5.6
    dop = Op(8.3, 9.5, 3.15, 5.35, "arch", cols=2, transoms=(4.75,), door=True)
    wall("n", dn0, dn1, wu1 - T, wu1, EAVE, dzt, [dop], m=0.2)
    glaze("n", wu1, 1, dop)
    poly([P3("n", x, wu1 - 0.15, z) for x, z in [(dop.a, dop.zb), (dop.b, dop.zb), (dop.b, 4.75), (dop.a, 4.75)]], "obscure")
    juliet("n", wu1, 1, dop.a, dop.b, 3.15)
    gable("n", dn0, dn1, wu1 - T, wu1, dzt, math.radians(50))
    for a, b in ((dn0, dn0 + 0.15), (dn1 - 0.15, dn1)):
        box(4.3, wu1 - T, a, b, EAVE, dzt, "stone")
    roof_u(4.3, wu1, dn0, dn1, dzt, math.radians(50), oe=0.2, og=0.3)
    # limestone chimney on the SW wall
    nmid = 13.5
    box(wu0 - 0.5, wu0, nmid - 0.9, nmid + 0.9, PADZ - 0.35, EAVE + 0.2, "stone")
    faces("stone", [(wu0 - 0.5, nmid - 0.9, EAVE + 0.2), (wu0, nmid - 0.9, EAVE + 0.2), (wu0, nmid + 0.9, EAVE + 0.2), (wu0 - 0.5, nmid + 0.9, EAVE + 0.2),
                    (wu0 - 0.5, nmid - 0.55, EAVE + 0.8), (wu0, nmid - 0.55, EAVE + 0.8), (wu0, nmid + 0.55, EAVE + 0.8), (wu0 - 0.5, nmid + 0.55, EAVE + 0.8)],
          [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    box(wu0 - 0.5, wu0 + 0.25, nmid - 0.55, nmid + 0.55, EAVE + 0.8, 7.75, "stone")
    box(wu0 - 0.58, wu0 + 0.33, nmid - 0.63, nmid + 0.63, 7.75, 7.87, "limestone")
    box(wu0 - 0.3, wu0 + 0.05, nmid - 0.18, nmid + 0.18, 7.87, 8.1, "black")

    # ---- east wing street gable over the garage: projecting stone bay window (the one major street window,
    #      screened by the lavender planting below), oculus in the gable
    c = (eu0 + eu1) / 2
    ba0, ba1, bn0_, bn1_ = L["bay_e"]
    wall("u", eu0, eu1, en0, en0 + T, -0.35, EAVE, [Op(ba0 + 0.3, ba1 - 0.3, 0.0, 2.6, "rect", door=True)])
    v = Op(c - 0.36, c + 0.36, 4.25, 4.97, "round", cols=1)
    gable("u", eu0, eu1, en0, en0 + T, EAVE, PE, op=v)
    glaze("u", en0, -1, v)
    ring_ax("u", Op(c - 0.46, c + 0.46, 4.15, 5.07, "round").poly(), v.poly(), en0 - 0.04, en0 + 0.01, "limestone")
    lights = [Op(x - 0.3, x + 0.3, 0.75, 2.45, "arch", cols=1, transoms=(1.7,)) for x in (ba0 + 0.6, (ba0 + ba1) / 2, ba1 - 0.6)]
    wall("u", ba0, ba1, bn0_, bn0_ + 0.3, -0.35, 2.95, lights, m=0.12)
    for o in lights:
        glaze("u", bn0_, -1, o)
    for a, b in ((ba0, ba0 + 0.3), (ba1 - 0.3, ba1)):
        box(a, b, bn0_ + 0.3, bn1_, -0.35, 2.95, "stone")
    box(ba0 - 0.05, ba1 + 0.05, bn0_ - 0.05, bn1_, -0.5, -0.35, "limestone")      # corbelled base
    plane((ba0 - 0.15, bn0_ - 0.2, 2.95), (ba1 + 0.15, bn0_ - 0.2, 2.95), (ba1 + 0.15, bn1_, 3.4), (ba0 - 0.15, bn1_, 3.4), "roof_x")
    room_light(eu0 + 0.8, eu1 - 0.8, en0 + 0.5, en0 + 4.0, 2.6, 400)

    # ---- main range: long low standing-seam roof between the gables, catslide over the loggia; the planes stop
    #      on the valley lines where they meet the wing roofs
    m_ = RF["main"]
    RN, RZ = m_["ridge_n"], m_["ridge_z"]
    PF, PR = math.radians(m_["front_pitch"]), math.radians(m_["rear_pitch"])
    um_w, um_e = (wu0 + wu1) / 2, (eu0 + eu1) / 2
    uvw = lambda z: um_w + (zr_w + 0.06 - z) / math.tan(PW)
    uve = lambda z: um_e - (zr_e + 0.06 - z) / math.tan(PE)
    zf = RZ - (RN - 8.0) * math.tan(PF)
    zrr = RZ - (mn1 + 0.45 - RN) * math.tan(PR)
    R6 = RZ + 0.06
    plane((uvw(zf), 8.0, zf), (uve(zf), 8.0, zf), (uve(R6), RN, R6), (uvw(R6), RN, R6), "roof_x")
    plane((uve(zrr), mn1 + 0.45, zrr), (uvw(zrr), mn1 + 0.45, zrr), (uvw(R6), RN, R6), (uve(R6), RN, R6), "roof_x")
    box(uvw(R6), uve(R6), RN - 0.08, RN + 0.08, RZ + 0.1, RZ + 0.2, "roof_y")
    box(uvw(zf), uve(zf), 7.93, 8.07, zf - 0.16, zf + 0.02, "black")
    for a, b in ((uvw(zrr), bu0), (bu1, uve(zrr))):
        box(a, b, mn1 + 0.38, mn1 + 0.52, zrr - 0.16, zrr + 0.02, "black")
    # street wall behind the loggia: one generous arched living-room window (screened by the loggia), two small ones
    lw = [Op(7.7, 9.7, 0.5, 2.65, "arch", cols=3, transoms=(1.65,)),
          Op(13.45, 14.35, 0.7, 2.45, "arch", cols=1, transoms=(1.55,)), Op(15.05, 15.95, 0.7, 2.45, "arch", cols=1, transoms=(1.55,))]
    wall("u", mu0, mu1, mn0, mn0 + T, -0.4, 3.6, lw + [Op(10.95, 12.25, 0.0, 2.6, "arch", door=True)], m=0.25)
    for o in lw:
        glaze("u", mn0, -1, o)
    rd = [Op(7.7, 9.3, 0.0, 2.8, "arch", cols=2, transoms=(2.0,), door=True), Op(14.3, 15.9, 0.0, 2.8, "arch", cols=2, transoms=(2.0,), door=True),
          Op(bu0 + T, bu1 - T, 0.0, 2.95, "rect", door=True)]
    wall("u", mu0, mu1, mn1 - T, mn1, -0.4, EAVE, rd, m=0.2)
    for o in rd[:2]:
        glaze("u", mn1, 1, o)
    room_light(mu0 + 0.6, mu1 - 0.6, mn0 + 0.8, mn1 - 0.8, 3.2, 1600)

    # ---- north bay: cross-gable with the tall arched window (the 8.webp reference)
    tall = Op(bu0 + 0.7, bu1 - 0.7, 0.0, 4.3, "arch", cols=3, transoms=(2.3, 3.2), door=True)
    gable("u", bu0, bu1, bn1 - T, bn1, RF["bay"]["eave"], math.radians(RF["bay"]["pitch"]), z0=-0.4, op=tall)
    glaze("u", bn1, 1, tall)
    bx("n", bn0, bn1 - T, bu0, bu0 + T, -0.4, 3.3, "stone")
    bx("n", bn0, bn1 - T, bu1 - T, bu1, -0.4, 3.3, "stone")
    roof_n(RN, bn1, bu0, bu1, RF["bay"]["eave"], math.radians(RF["bay"]["pitch"]), oe=0.3, og=0.3, gut=False)
    cr_pendant((bu0 + bu1) / 2, bn0 - 1.5, 2.7, 70)
    box(bu0 + 0.3, bu1 - 0.3, mn1 - 4.0, mn1 - 3.1, 0, 0.42, "linen")
    box(bu0 + 0.3, bu1 - 0.3, mn1 - 4.0, mn1 - 3.75, 0.42, 0.85, "linen")
    box(bu0 - 0.6, bu1 + 0.6, mn1 - 3.5, mn1 - 0.5, 0, 0.01, "rug")
    box(7.9, 10.0, 14.0, 15.0, 0.72, 0.77, "walnut")
    cr_pendant(8.95, 14.5, 2.0, 50)
    box(mu0 + 0.05, mu0 + 0.1, 13.0, 15.5, 1.2, 2.4, "art")

    # ---- entry gable with the arched front door
    EZ, EP = RF["entry"]["eave"], math.radians(RF["entry"]["pitch"])
    door = Op(10.85, 12.35, 0.0, 2.75, "arch", door=True)
    gable("u", xu0, xu1, xn0, xn0 + T, EZ, EP, z0=-0.4, op=door)
    poly([P3("u", x, xn0 + 0.22, z) for x, z in door.poly(0.04)], "door")
    ring_ax("u", door.poly(), door.poly(0.07), xn0 + 0.16, xn0 + 0.24, "door")
    box(door.cx - 0.02, door.cx + 0.02, xn0 + 0.15, xn0 + 0.22, 0.0, 2.6, "black")
    for x in (door.cx - 0.12, door.cx + 0.12):
        box(x - 0.02, x + 0.02, xn0 + 0.12, xn0 + 0.16, 0.95, 1.25, "black")
    for k in range(13):                                      # dressed limestone voussoirs
        a0, a1 = math.pi * k / 13, math.pi * (k + 1) / 13
        r0, r1 = door.r + 0.02, door.r + 0.28
        pts = [(door.cx + r0 * math.cos(a0), door.zs + r0 * math.sin(a0)), (door.cx + r1 * math.cos(a0), door.zs + r1 * math.sin(a0)),
               (door.cx + r1 * math.cos(a1), door.zs + r1 * math.sin(a1)), (door.cx + r0 * math.cos(a1), door.zs + r0 * math.sin(a1))]
        solid("u", pts[::-1], xn0 - 0.03, xn0 + 0.02, "limestone")
    for x in (xu0 + 0.28, xu1 - 0.28):
        wall_lantern(x, xn0, 1.8, -1)
    bx("n", xn0 + T, xn1, xu0, xu0 + T, -0.4, EZ, "stone")
    bx("n", xn0 + T, xn1, xu1 - T, xu1, -0.4, EZ, "stone")
    roof_n(xn0, RN, xu0, xu1, EZ, EP, oe=0.3, og=0.35, gut=False)
    box(xu0 - 0.3, xu1 + 0.3, xn0 - 0.7, xn0, PADZ - 0.3, -0.02, "limestone")
    cr_pendant((xu0 + xu1) / 2, xn0 + 1.3, 2.3, 40)
    # the small dormer on the main roof above the entry
    du = (xu0 + xu1) / 2
    nf = 12.0
    zb = RZ - (RN - nf) * math.tan(PF) - 0.15
    dw = Op(du - 0.22, du + 0.22, zb + 0.2, zb + 0.72, "arch", cols=1)
    wall("u", du - 0.45, du + 0.45, nf, nf + 0.18, zb, zb + 0.8, [dw], m=0.1)
    glaze("u", nf, -1, dw)
    gable("u", du - 0.45, du + 0.45, nf, nf + 0.18, zb + 0.8, math.radians(48))
    for x0, x1 in ((du - 0.45, du - 0.38), (du + 0.38, du + 0.45)):
        box(x0, x1, nf, nf + 2.4, zb, zb + 0.8, "roof_y")
    roof_n(nf, nf + 3.0, du - 0.45, du + 0.45, zb + 0.8, math.radians(48), oe=0.1, og=0.1, gut=False)
    light("POINT", (du, nf + 0.7, zb + 0.4), 20, WARM, size=0.1, name="attic")
    # louvred roof vent on the ridge
    vx = 9.0
    box(vx - 0.35, vx + 0.35, RN - 0.35, RN + 0.35, RZ - 0.3, RZ + 0.75, "roof_y")
    for z in np.arange(RZ + 0.25, RZ + 0.7, 0.09):
        box(vx - 0.37, vx + 0.37, RN - 0.37, RN + 0.37, z, z + 0.035, "black")
    faces("roof_y", [(vx - 0.5, RN - 0.5, RZ + 0.75), (vx + 0.5, RN - 0.5, RZ + 0.75), (vx + 0.5, RN + 0.5, RZ + 0.75), (vx - 0.5, RN + 0.5, RZ + 0.75),
                     (vx, RN, RZ + 1.2)], [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (3, 2, 1, 0)])

    # ---- loggia: timber posts on stone bases, curved brackets, boarded ceiling
    lu0, lu1, ln0, ln1 = L["loggia"]
    for a, b in ((lu0, xu0), (xu1, lu1)):
        box(a, b, ln0, ln1, PADZ - 0.3, 0.0, "limestone")
        box(a, b, ln0 + 0.3, ln1, 2.62, 2.66, "deck")
        box(a, b, ln0, ln0 + 0.32, 2.3, 2.62, "timber")
        for k in range(int((b - a) / 0.6) + 1):
            x = a + 0.15 + k * 0.6
            if x < b - 0.1:
                box(x - 0.05, x + 0.05, ln0 - 0.2, ln1, 2.62, 2.72, "timber")
        x = (a + b) / 2
        box(x - 0.24, x + 0.24, ln0, ln0 + 0.48, 0.0, 0.55, "limestone")
        box(x - 0.13, x + 0.13, ln0 + 0.11, ln0 + 0.37, 0.55, 2.3, "timber")
        for s in (-1, 1):
            pts = [(x + s * 0.13 + s * 0.7 * (1 - math.cos(t * math.pi / 2)), 2.3 - 0.7 * (1 - math.sin(t * math.pi / 2))) for t in np.linspace(0, 1, 7)]
            for (x0, z0_), (x1, z1_) in zip(pts, pts[1:]):
                cyl((x0, ln0 + 0.24, z0_), (x1, ln0 + 0.24, z1_), 0.06, 0.06, "timber")
        for xx in (a + 0.9, b - 0.9):
            light("POINT", (xx, ln0 + 1.3, 2.5), 18, WARM, size=0.05, name="soffit")
        for xx in (a + 0.55, b - 0.55):
            box(xx - 0.3, xx + 0.3, ln1 - 0.9, ln1 - 0.35, 0.0, 0.45, "timber")
            box(xx - 0.26, xx + 0.26, ln1 - 0.86, ln1 - 0.39, 0.45, 0.52, "linen")
    for x in (xu0 - 0.45, xu1 + 0.45):
        cyl((x, ln0 + 0.35, 0.0), (x, ln0 + 0.35, 0.55), 0.22, 0.3, "black", sides=10)
        bush((x, ln0 + 0.35, 0.85), 0.3, "hedge", 10)

    # ---- garage at street level under the east wing, door face 7.5 m back; near-flat apron in front
    gu0, gu1, gn0, gn1 = L["garage"]
    gd = Op(gu0 + 0.7, gu1 - 0.7, GZ, -0.4, "rect", door=True)
    wall("u", gu0 - T, gu1 + T, gn0, gn0 + T, GZ - 0.3, -0.35, [gd], m=0.35)
    box(gd.a, gd.b, gn0 + 0.25, gn0 + 0.3, GZ, -0.4, "shutter")
    for z in np.arange(GZ + 0.55, -0.4, 0.55):
        box(gd.a, gd.b, gn0 + 0.23, gn0 + 0.25, z - 0.015, z + 0.015, "black")
    box(gu0 - T, gu0, gn0, gn1, GZ - 0.3, -0.35, "stone")
    box(gu1, gu1 + T, gn0, gn1, GZ - 0.3, -0.35, "stone")
    box(gu0, gu1, gn0, gn1, GZ - 0.25, GZ, "concrete")
    for x in (gu0 + 0.3, gu1 - 0.3):
        wall_lantern(x, gn0, GZ + 1.6, -1)
    # planter west of the apron (between the stepped walls), low wall + natural planted strip to the NE
    pl = L["planter"]
    pz = FRONT["planter"]
    box(15.0, 16.0, pl[2], pl[3], pz - 0.3, pz + 0.03, "mulch")
    for n in np.arange(pl[2] + 0.5, pl[3] - 0.2, 0.6):
        lavender(15.5 + rr(-0.15, 0.15), n, pz, rr(0.3, 0.38))
    for n0 in np.arange(0.8, 7.5, 1.35):
        n1 = min(n0 + 1.35, 7.5)
        g_ = zg(23.4, (n0 + n1) / 2)
        box(gu1 + T, gu1 + T + 0.3, n0, n1, GZ - 0.3, max(g_ + 0.3, GZ + 0.45), "stone")

    # ---- street wall, gate pillars and steps
    s0, s1, sn0, sn1 = L["stair"]
    topw = FRONT["wall_top"]
    for a, b in ((-0.2, s0 - 0.45), (s1 + 0.45, gu0)):
        box(a, b, 1.6, 2.2, -3.4, topw, "stone")
        box(a - 0.03, b + 0.03, 1.55, 2.25, topw, topw + 0.08, "limestone")
    gg = zg((s0 + s1) / 2, 1.2)
    for x in (s0 - 0.45, s1 + 0.45):
        box(x - 0.45, x + 0.45, 1.45, 2.35, gg - 0.3, topw + 0.45, "stone")
        box(x - 0.52, x + 0.52, 1.38, 2.42, topw + 0.45, topw + 0.57, "limestone")
        lantern(x, 1.9, topw + 0.57, 0.34, 28)
    n_steps = 11
    for k in range(n_steps):
        z = gg + (PADZ - gg) * (k + 1) / n_steps
        n_a = 1.5 + k * (sn1 - 1.5) / n_steps
        box(s0, s1, n_a, sn1, gg - 0.4, z, "limestone")
    for x0, x1 in ((s0 - 0.3, s0), (s1, s1 + 0.3)):
        box(x0, x1, 2.2, sn1, gg - 0.4, PADZ + 0.3, "stone")

    # ---- Colorbond side fences (sage green SW, beige NE) on the boundary, behind the 6 m front setback
    for u0, n0, u1, n1, z0, z1, sd in LAYF["fences"]:
        mat = "fence_sw" if sd == "sw" else "fence_ne"
        faces(mat, [(u0, n0, z0), (u1, n1, z0), (u1, n1, z1), (u0, n0, z1)], [(0, 1, 2, 3)])
        dx, dn = u1 - u0, n1 - n0
        ln = math.hypot(dx, dn)
        ox, on = -dn / ln * 0.03, dx / ln * 0.03
        faces(mat, [(u0 - ox, n0 - on, z1), (u1 - ox, n1 - on, z1), (u1 + ox, n1 + on, z1), (u0 + ox, n0 + on, z1),
                    (u0 - ox, n0 - on, z1 + 0.05), (u1 - ox, n1 - on, z1 + 0.05), (u1 + ox, n1 + on, z1 + 0.05), (u0 + ox, n0 + on, z1 + 0.05)],
              [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
        box(u0 - 0.03, u0 + 0.03, n0 - 0.03, n0 + 0.03, z0 - 0.2, z1, mat)    # posts

    # ---- neighbouring houses (context, from the aerial)
    for k, nb in enumerate(LAYF["neighbours"]):
        hip_house(nb["quad"], nb["wall"], nb["roof"], nb["h"], nb["pitch"], k)

    # ---- pool and the north courtyard
    pu0, pu1, pn0, pn1 = L["pool"]
    for a, c, n0_, n1_ in ((pu0 - 0.3, pu1 + 0.3, pn0 - 0.3, pn0), (pu0 - 0.3, pu1 + 0.3, pn1, pn1 + 0.3), (pu0 - 0.3, pu0, pn0, pn1), (pu1, pu1 + 0.3, pn0, pn1)):
        box(a, c, n0_, n1_, PADZ - 0.3, PADZ + 0.06, "coping")
    box(pu0, pu1, pn0, pn1, PADZ - 1.7, PADZ - 1.6, "pooltile")
    for a, c, n0_, n1_ in ((pu0, pu1, pn0, pn0 + 0.02), (pu0, pu1, pn1 - 0.02, pn1), (pu0, pu0 + 0.02, pn0, pn1), (pu1 - 0.02, pu1, pn0, pn1)):
        box(a, c, n0_, n1_, PADZ - 1.7, PADZ - 0.1, "pooltile")
    poly([(pu0, pn0, PADZ - 0.1), (pu1, pn0, PADZ - 0.1), (pu1, pn1, PADZ - 0.1), (pu0, pn1, PADZ - 0.1)], "water")
    for x in (pu0 + 2.0, (pu0 + pu1) / 2, pu1 - 2.0):
        light("POINT", (x, pn0 + 0.15, PADZ - 0.9), 50, (0.55, 0.85, 1.0), size=0.05, name="pool")
    for x in (pu0 + 1.2, pu0 + 2.4, pu1 - 2.4, pu1 - 1.2):        # loungers on the north side of the pool
        box(x - 0.35, x + 0.35, pn1 + 0.6, pn1 + 2.5, PADZ + 0.06, PADZ + 0.32, "timber")
        box(x - 0.32, x + 0.32, pn1 + 0.62, pn1 + 2.48, PADZ + 0.32, PADZ + 0.4, "linen")
    box(17.6, 20.4, 29.4, 30.4, PADZ + 0.72, PADZ + 0.77, "walnut")       # outdoor dining
    for x in (18.0, 19.0, 20.0):
        for nn in (29.0, 30.8):
            box(x - 0.2, x + 0.2, nn - 0.2, nn + 0.2, PADZ, PADZ + 0.45, "black")
    for x in (17.9, 20.1):
        cyl((x, 29.9, PADZ), (x, 29.9, PADZ + 0.72), 0.04, 0.04, "black")

    # ---- upper garden: stone steps, pergola with dining, fire bowl
    a0, a1, un0, un1 = L["upstair"]
    nst = 8
    for k in range(nst):
        z = PADZ + (UPZ - PADZ) * (k + 1) / nst
        box(a0, a1, un0 - 0.2 + k * (un1 - un0 + 0.2) / nst, un1, PADZ - 0.3, z, "limestone")
    for x0, x1 in ((a0 - 0.3, a0), (a1, a1 + 0.3)):
        box(x0, x1, un0 - 0.2, un1, PADZ - 0.3, UPZ + 0.3, "stone")
        box(x0 - 0.02, x1 + 0.02, un0 - 0.22, un1, UPZ + 0.3, UPZ + 0.36, "limestone")
    for x in (a0 - 0.15, a1 + 0.15):
        lantern(x, un0 - 0.05, UPZ + 0.45, 0.26, 16)
    pu0_, pu1_, pn0_, pn1_ = L["pergola"]
    for x in (pu0_ + 0.2, (pu0_ + pu1_) / 2, pu1_ - 0.2):
        for nn in (pn0_ + 0.2, pn1_ - 0.2):
            box(x - 0.12, x + 0.12, nn - 0.12, nn + 0.12, UPZ, UPZ + 2.7, "timber")
    for nn in (pn0_ + 0.2, pn1_ - 0.2):
        box(pu0_ - 0.3, pu1_ + 0.3, nn - 0.1, nn + 0.1, UPZ + 2.7, UPZ + 2.95, "timber")
    for k in range(13):
        x = pu0_ + k * (pu1_ - pu0_) / 12
        box(x - 0.04, x + 0.04, pn0_ - 0.3, pn1_ + 0.3, UPZ + 2.95, UPZ + 3.1, "timber")
    box(20.6, 23.4, 37.6, 38.5, UPZ + 0.72, UPZ + 0.77, "walnut")
    for x in (21.0, 22.0, 23.0):
        for nn in (37.2, 38.9):
            box(x - 0.2, x + 0.2, nn - 0.2, nn + 0.2, UPZ, UPZ + 0.45, "black")
    for x in (20.8, 23.2):
        cyl((x, 38.05, UPZ), (x, 38.05, UPZ + 0.72), 0.04, 0.04, "black")
    for k in range(5):
        light("POINT", (pu0_ + 0.8 + k * 1.1, (pn0_ + pn1_) / 2, UPZ + 2.6), 8, WARM, size=0.04, name="cabana")
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(pu0_ + 0.8 + k * 1.1, (pn0_ + pn1_) / 2, UPZ + 2.65))
        bpy.context.active_object.data.materials.append(MATS["bulb"])
    cyl((15.5, 38.0, UPZ), (15.5, 38.0, UPZ + 0.45), 0.55, 0.45, "stone", sides=14)
    box(15.1, 15.9, 37.6, 38.4, UPZ + 0.3, UPZ + 0.46, "fire")
    light("POINT", (15.5, 38.0, UPZ + 0.8), 80, (1.0, 0.45, 0.15), size=0.3, name="pool_fire")

    # ---- retaining walls: lot perimeter (in the boundary strip) and the terrace step
    for u0, n0, u1, n1, ou, on, bot, top in LAYF["walls"]:
        q = [(u0, n0), (u1, n1), (u1 + ou, n1 + on), (u0 + ou, n0 + on)]
        faces("stone", [(u, n, bot) for u, n in q] + [(u, n, top) for u, n in q],
              [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
        faces("limestone", [(u, n, top) for u, n in q] + [(u, n, top + 0.06) for u, n in q],
              [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])

    # ---- designed ground cover over the whole lot
    cover = {"bank": "mulch", "beds": "mulch", "ubeds": "mulch", "lawn": "lawn", "gravel": "gravel", "paving": "paving"}
    for zone, mat in cover.items():
        t = LAYF["zones"][zone]
        v = [(t[i], t[i + 1], t[i + 2] + 0.025) for i in range(0, len(t), 3)]
        faces(mat, v, [(k, k + 1, k + 2) for k in range(0, len(v), 3)])

    # ---- street: road, kerbs, footpaths, nature strip, driveway crossing
    R = ARC["r"]
    drape(-45, 68, lambda u: arc_n(u, R - 15.0), lambda u: arc_n(u, R - 4.0), 0.03, "asphalt", 150, 10)
    for r0, r1 in ((R - 4.0, R - 3.75), (R - 15.25, R - 15.0)):
        drape(-45, 68, lambda u, r0=r0: arc_n(u, r0), lambda u, r1=r1: arc_n(u, r1), 0.17, "kerb", 150, 1)
    drape(-45, 68, lambda u: arc_n(u, R - 16.7), lambda u: arc_n(u, R - 15.25), 0.1, "concrete", 150, 2)
    drape(-45, 68, lambda u: arc_n(u, R - 24.0), lambda u: arc_n(u, R - 16.7), 0.05, "lawn", 150, 4)
    drape(-45, gu0 + 0.2, lambda u: arc_n(u, R - 3.75), lambda u: arc_n(u, R - 2.3), 0.1, "concrete", 80, 2)
    drape(gu1 - 0.2, 68, lambda u: arc_n(u, R - 3.75), lambda u: arc_n(u, R - 2.3), 0.1, "concrete", 60, 2)
    drape(gu0 + 0.2, gu1 - 0.2, lambda u: arc_n(u, R - 3.75), lambda u: gn0, 0.06, "aggregate", 12, 12)
    drape(-1.5, s0 - 0.9, lambda u: arc_n(u, R - 2.3), lambda u: 1.58, 0.05, "gravel", 30, 4)
    drape(s1 + 0.9, gu0 - T, lambda u: arc_n(u, R - 2.3), lambda u: 1.58, 0.05, "gravel", 30, 4)
    drape(s0 - 0.9, s1 + 0.9, lambda u: arc_n(u, R - 2.3), lambda u: 1.5, 0.05, "gravel", 6, 4)
    drape(-45, -1.5, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 1.0, 0.04, "lawn", 30, 2)
    drape(gu1 + T, 68, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 1.0, 0.04, "lawn", 30, 2)


EUC = []


def build_garden(view):
    S = cr.S
    eye = VIEWS[view]["eye"] if view in VIEWS else (0, -100, 0)
    at = VIEWS[view]["at"] if view in VIEWS else (0, 0, 0)

    def near_sightline(u, n, d=7.0):
        ax, an = at[0] - eye[0], at[1] - eye[1]
        L2 = ax * ax + an * an
        t = max(0, min(1, ((u - eye[0]) * ax + (n - eye[1]) * an) / L2))
        return math.hypot(u - eye[0] - t * ax, n - eye[1] - t * an) < d and t < 0.6

    for gx, gy, r in S["trees"]:
        if rng.random() < 0.3:
            continue
        u, n = cr.to_frame(gx, gy)
        if near_sightline(u, n) or inside_lot(u, n):
            continue
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
    # dry tussocks on the slope behind the lot
    for _ in range(1400):
        u, n = rr(-40, 60), rr(46, 120)
        if inside_lot(u, n) or near_sightline(u, n, 4):
            continue
        card((u, n, zg(u, n) + 0.25), rr(0.5, 0.9), "tussock")

    s0, s1, sn0, sn1 = L["stair"]
    gu0 = L["garage"][0]
    # spreading shade trees framing the front (oak-like), Italian cypress beside the gables
    tree((0.7, 3.4, zd(0.7, 3.4)), 9.5, 4.6, "oak", "darkbark", stems=1, cards_per=30, crown_flat=0.45)
    cypress(0.0, 6.3, PADZ)
    cypress(23.6, 6.4, zg(23.6, 6.4), 4.6)
    # lavender and ornamental grasses on the bank and front beds (dry, waterwise), boulder edging along the path
    for u in np.arange(0.0, 14.3, 0.7):
        for n in np.arange(2.7, 8.4, 0.7):
            uu, nn = u + rr(-0.2, 0.2), n + rr(-0.2, 0.2)
            if s0 - 0.6 < uu < s1 + 0.6 or not inside_lot(uu, nn):
                continue
            if n > 6.6 and not (7.4 < uu < 14.3):
                continue  # in front of the wing walls only a narrow strip
            z = zd(uu, nn)
            k = (int(u / 0.7) * 7 + int(n / 0.7) * 3) % 5
            if k in (0, 3):
                grass(uu, nn, z, rr(0.7, 1.0))
            elif k == 4 and rng.random() < 0.5:
                bush((uu, nn, z + 0.3), rr(0.3, 0.4), "shrub", 10)
            else:
                lavender(uu, nn, z, rr(0.3, 0.4))
    for n in np.arange(5.2, 8.2, 0.55):
        for x in (10.3, 12.9):
            rock((x + rr(-0.05, 0.05), n, PADZ + 0.05), rr(0.2, 0.3), 0.55)
    for n in np.arange(2.4, 5.0, 0.6):
        for x in (s0 - 0.55, s1 + 0.55):
            rock((x, n, zd(x, n)), rr(0.22, 0.32), 0.6)
    for u, n in ((2.0, 3.0), (5.2, 4.2), (8.4, 3.2), (14.4, 3.5), (15.4, 4.4), (3.8, 6.6)):
        rock((u, n, zd(u, n) - 0.05), rr(0.4, 0.6), 0.55)
    # verge garden inside the street boundary: grasses, boulders
    for _ in range(46):
        u = rr(-0.8, 14.2)
        if s0 - 1.0 < u < s1 + 1.0:
            continue
        t = rr(0.2, 0.85)
        nn = arc_n(u, ARC["r"] - 2.3) * (1 - t) + 1.4 * t
        if rng.random() < 0.6:
            grass(u, nn, zg(u, nn), rr(0.6, 0.9))
        else:
            lavender(u, nn, zg(u, nn), rr(0.3, 0.38))
    for u in (1.6, 6.0, 9.4, 14.0):
        nn = (arc_n(u, ARC["r"] - 2.3) + 1.4) / 2
        rock((u, nn, zg(u, nn)), rr(0.45, 0.6), 0.55)
    # side yards: low grasses and rosemary along the boundaries (fire-wise, low fuel)
    for n in np.arange(9.0, 31.5, 1.6):
        for u in (-5.1 * n / 32.5 + 0.9, 22.9 + 6.9 * n / 44.5 - 0.9):
            if inside_lot(u, n):
                z = zd(u, n)
                (grass if rng.random() < 0.5 else lambda a, b, c, d: bush((a, b, c + 0.35), 0.4, "shrub", 12))(u, n, z, 0.8)
    # courtyard olives in the corners near the terrace wall
    for u, n in ((-1.5, 31.2), (25.0, 31.2), (3.0, 31.0)):
        tree((u, n, PADZ), rr(3.2, 3.8), rr(1.5, 1.8), "olive", "darkbark", stems=3, cards_per=20, crown_flat=0.6)
    # upper garden: olives, lavender + grass beds along the rear and side boundaries
    for u, n in ((2.0, 36.0), (8.0, 38.4), (27.0, 41.2), (15.0, 41.5)):
        if inside_lot(u, n):
            tree((u, n, UPZ), rr(3.0, 3.6), rr(1.4, 1.7), "olive", "darkbark", stems=3, cards_per=18, crown_flat=0.6)
    t = LAYF["zones"]["ubeds"]
    pts = [(t[i], t[i + 1]) for i in range(0, len(t), 3)]
    for k in range(0, len(pts), 3):
        (ua, na), (ub, nb), (uc, nc) = pts[k:k + 3]
        area = abs((ub - ua) * (nc - na) - (uc - ua) * (nb - na)) / 2
        for _ in range(int(area / 0.55)):
            a_, b_ = rng.random(), rng.random()
            if a_ + b_ > 1:
                a_, b_ = 1 - a_, 1 - b_
            u = ua + a_ * (ub - ua) + b_ * (uc - ua)
            n = na + a_ * (nb - na) + b_ * (nc - na)
            if rng.random() < 0.55:
                lavender(u, n, UPZ, rr(0.3, 0.4))
            else:
                grass(u, n, UPZ, rr(0.7, 1.0))
    # lavender pots by the pool
    for u, n in ((7.7, 22.6), (15.9, 22.6)):
        cyl((u, n, PADZ), (u, n, PADZ + 0.55), 0.3, 0.38, "limestone", sides=10)
        lavender(u, n, PADZ + 0.5, 0.35)


# ------------------------------------------------------------------ terrain (designed lot patch + surroundings)
def build_terrain(coarse=1):
    P = LAYF["patch"]
    pu0, pn0, st, nu, nn = P["u0"], P["n0"], P["step"], P["nu"], P["nn"]
    core = (pu0 + 1.0, pu0 + (nu - 1) * st - 1.0, pn0 + 1.0, pn0 + (nn - 1) * st - 1.0)
    H, NG = cr.H, cr.NG
    v, uv, fcs = [], [], []
    for j in range(NG):
        for i in range(NG):
            gx, gy = i + 0.5, j + 0.5
            u, n = cr.to_frame(gx, gy)
            v.append((u, n, H[j, i] - cr.FFL))
            uv.append((gx / NG, gy / NG))
    for j in range(NG - 1):
        for i in range(NG - 1):
            a = j * NG + i
            q = (a, a + 1, a + NG + 1, a + NG)
            if all(core[0] < v[k][0] < core[1] and core[2] < v[k][1] < core[3] for k in q):
                continue
            fcs.append(q)
    o = len(v)
    z = P["z"]
    I, J = list(range(0, nu, coarse)), list(range(0, nn, coarse))   # the web-viewer GLB uses a 0.5 m patch
    for j in J:
        for i in I:
            u, n = pu0 + i * st, pn0 + j * st
            v.append((u, n, z[j * nu + i]))
            gx, gy = cr.to_grid(u, n)
            uv.append((gx / NG, gy / NG))
    ni = len(I)
    for j in range(len(J) - 1):
        for i in range(ni - 1):
            a = o + j * ni + i
            fcs.append((a, a + 1, a + ni + 1, a + ni))
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
    S = cr.S
    dem = np.array(Image.open(D3 / S["dem_1km"]["file"]), dtype=float)
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


GLB_TEX = {"stone": ("hcstone.png", 0.25), "paving": ("paving.png", 0.25), "coping": ("paving.png", 0.25),
           "limestone": ("stone.png", 0.35), "shutter": ("boards.png", 0.5), "door": ("boards.png", 0.5)}
GLB_COL = {"roof_x": ("#5a5e61", 0.45, 0.55), "roof_y": ("#5a5e61", 0.45, 0.55), "black": ("#141516", 0.4, 0.3),
           "timber": ("#7a5a3c", 0.6, 0), "deck": ("#8a6644", 0.55, 0), "ceiling": ("#efe9de", 0.8, 0),
           "lawn": ("#5f7d38", 0.95, 0), "mulch": ("#5a4636", 1.0, 0), "concrete": ("#bbb5aa", 0.9, 0), "gravel": ("#cbbd9f", 0.95, 0),
           "kerb": ("#b6b0a5", 0.9, 0), "asphalt": ("#38393b", 0.9, 0), "aggregate": ("#a39b8e", 0.9, 0), "boulder": ("#b3a58a", 0.9, 0),
           "pooltile": ("#8fd0e3", 0.3, 0), "water": ("#3fa9cf", 0.05, 0), "bark": ("#a09684", 0.85, 0),
           "darkbark": ("#4d443b", 0.9, 0), "fire": ("#ff7a2a", 0.5, 0), "floor": ("#9a7a56", 0.4, 0),
           "fence_sw": ("#7d876c", 0.5, 0.3), "fence_ne": ("#cbbf9f", 0.5, 0.3), "obscure": ("#d6dcdc", 0.35, 0),
           "slab": ("#e8e2d6", 0.7, 0), "glass": ("#0e1416", 0.05, 0.0), "bulb": ("#fff2d8", 0.5, 0)}
INTERIOR = {"fabric", "linen", "walnut", "rug", "art", "lamp"}
LEAVES = ("euc", "olive", "shrub", "hedge", "tussock", "lavender", "grass", "oak")
EXTERIOR_LIGHTS = ("lantern", "garage_lamp", "soffit", "pool", "cabana", "attic")  # pool_fire included


def export_glb(path):
    """the scene without interiors as GLB: textures on world-scale UVs, flat PBR colours elsewhere"""
    for ob in list(bpy.data.objects):
        mat = ob.data.materials[0].name if ob.type == "MESH" and ob.data.materials else ""
        if ob.type == "MESH" and (mat in INTERIOR or mat in ("euc", "bark", "tussock") or (mat == "bulb" and len(ob.data.polygons) > 100)):
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
            file = GLB_TEX[name][0] if name in GLB_TEX else ("leaf_%s.png" % ("hedge" if name == "oak" else name) if name != "terrain" else None)
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


VIEWS = {
    "street": dict(eye=(11.6, -12.8, 1.7), at=(11.8, 14.0, 3.0), lens=24),
    "courtyard": dict(eye=(-6.0, 52.0, 8.5), at=(14.0, 27.0, 1.2), lens=22),
    "aerial": dict(eye=(46.0, -22.0, 40.0), at=(12.0, 22.0, 0.0), lens=28),
}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="street")
    ap.add_argument("--scale", type=int, default=50)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "designs" / "hill_country_street.png"))
    ap.add_argument("--exposure", type=float, default=0.9)
    ap.add_argument("--export-glb", default="")
    a = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    cr.make_textures()
    textures()
    MATS.update({
        "stone": img_mat("stone", "hcstone.png", 0.25, 0.92, bump=0.9),
        "limestone": img_mat("limestone", "stone.png", 0.35, 0.9, bump=0.4, tint=(1.08, 1.07, 1.04)),
        "paving": img_mat("paving", "paving.png", 0.25, 0.8, bump=0.1),
        "coping": img_mat("coping", "paving.png", 0.25, 0.75),
        "shutter": img_mat("shutter", "boards.png", 0.5, 0.7, bump=0.3),
        "door": img_mat("door", "boards.png", 0.5, 0.5, bump=0.3, tint=(0.8, 0.72, 0.62)),
        "roof_x": roof_mat("roof_x", "X"), "roof_y": roof_mat("roof_y", "Y"),
        "black": flat_mat("black", srgb("#141516"), 0.4, 0.3),
        "glass": thin_glass("glass"),
        "obscure": flat_mat("obscure", srgb("#d6dcdc"), 0.35),
        "fence_sw": fence_mat("fence_sw", "#7d876c"), "fence_ne": fence_mat("fence_ne", "#cbbf9f"),
        "ceiling": flat_mat("ceiling", srgb("#efe9de"), 0.8),
        "slab": flat_mat("slab", srgb("#e8e2d6"), 0.7),
        "floor": flat_mat("floor", srgb("#9a7a56"), 0.35),
        "fabric": flat_mat("fabric", srgb("#d9cfbf"), 0.95),
        "linen": flat_mat("linen", srgb("#f1ece3"), 0.95),
        "walnut": flat_mat("walnut", srgb("#4a3424"), 0.45),
        "rug": flat_mat("rug", srgb("#b9a58a"), 1.0),
        "art": flat_mat("art", srgb("#6f7f86"), 0.6),
        "lamp": flat_mat("lamp", (1, 1, 1), 0.5, emit=WARM, strength=14),
        "bulb": flat_mat("bulb", (1, 1, 1), 0.5, emit=WARM, strength=25),
        "lawn": noisy_mat("lawn", "#566f31", "#7d9446", 30, 0.95, 0.3),
        "mulch": noisy_mat("mulch", "#5a4636", "#806a54", 40, 1.0, 0.4),
        "gravel": noisy_mat("gravel", "#b8a888", "#ddd1b6", 90, 0.95, 0.5),
        "concrete": noisy_mat("concrete", "#aaa499", "#c9c3b8", 8, 0.9, 0.05),
        "kerb": flat_mat("kerb", srgb("#b6b0a5"), 0.9),
        "asphalt": noisy_mat("asphalt", "#2a2b2d", "#4a4a4b", 90, 0.9, 0.3),
        "aggregate": noisy_mat("aggregate", "#9a9285", "#c8bfb0", 70, 0.9, 0.25),
        "boulder": noisy_mat("boulder", "#9c8f78", "#cfc2a6", 3, 0.9, 0.8),
        "timber": noisy_mat("timber", "#6e4f33", "#8f6a46", 3, 0.6, 0.1),
        "deck": noisy_mat("deck", "#7a5636", "#9a7350", 2, 0.55, 0.1),
        "pooltile": flat_mat("pooltile", srgb("#86c9dc"), 0.3),
        "bark": noisy_mat("bark", "#8a8070", "#b0a796", 6, 0.85, 0.2),
        "darkbark": flat_mat("darkbark", srgb("#4d443b"), 0.9),
        "euc": leaf_mat("euc"), "olive": leaf_mat("olive"), "shrub": leaf_mat("shrub"), "hedge": leaf_mat("hedge"),
        "tussock": leaf_mat("tussock"), "lavender": leaf_mat("lavender"), "grass": leaf_mat("grass"),
        "fire": flat_mat("fire", (1, 0.4, 0.1), 0.5, emit=(1.0, 0.42, 0.12), strength=40),
        "terrain": terrain_mat("terrain", True), "grass_far": terrain_mat("grass", False),
    })
    MATS["grass_bg"] = MATS["grass_far"]
    MATS["oak"] = leaf_mat("hedge")
    MATS["oak"].name = "oak"
    w = MATS["water"] = bpy.data.materials.new("water")
    w.use_nodes = True
    bw = w.node_tree.nodes["Principled BSDF"]
    bw.inputs["Base Color"].default_value = (0.75, 0.95, 1.0, 1)
    bw.inputs["Transmission Weight"].default_value = 1.0
    bw.inputs["IOR"].default_value = 1.33
    bw.inputs["Roughness"].default_value = 0.03

    # the backdrop mesh uses MATS["grass"]; keep the grass-card material separate under "grass"
    grass_card = MATS["grass"]
    MATS["grass"] = MATS["grass_far"]
    build_terrain(2 if a.export_glb else 1)
    MATS["grass"] = grass_card
    build_house()
    build_garden(a.view)
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
