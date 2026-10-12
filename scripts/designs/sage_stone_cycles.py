"""Cycles render of "Sage and Stone": a two-tone modern farmhouse (client photo images/7.webp).

    python sage_stone_cycles.py --view street|courtyard|aerial --scale 50 --samples 32 --out x.png [--export-glb x.glb]

White board-and-batten main house, light sage (#9aa58c) gabled wings on a pale limestone base, charcoal shingle roofs,
brass half-round gutters and lanterns, black-framed windows, a timber entry portico with knee braces, stone chimney.
Run scripts/designs/sage_stone_site.py ground first so house_massing.json / render_scene.json carry this design.
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
from cycles_render import (ARC, GZ, PADZ, B, MATS, TEX, arc_n, box, card, cyl, drape, faces, poly, prism, ring, slab,  # noqa: E402
                           zg, light, bush, tree, WARM, srgb, img_mat, flat_mat, noisy_mat, seam_roof, thin_glass,
                           leaf_mat, terrain_mat, node_mat, obj_coords)

D3 = ROOT / "data" / "3d"
LAY = json.loads((D3 / "house_massing.json").read_text())
L = LAY.get("layout", {})
FRONT = LAY.get("front", {})
PAD_OUTLINE = LAY.get("pad_outline", [])
SURF = LAY.get("surfaces", {})
UPZ = FRONT.get("upper", 1.0)
rng = random.Random(624)
rr = rng.uniform

T = 0.25                                   # wall thickness
EAVE_M, PITCH_M, UPM = 5.0, math.radians(40), 2.9   # white main gable: two storeys, ridge along n (gable to the street)
EAVE_L, PITCH_L = 3.0, math.radians(35)             # white entry/living link, ridge along u
EAVE_E, PITCH_E = 3.0, math.radians(45)             # sage east wing over the garage
EAVE_W, PITCH_W = 3.2, math.radians(47)             # sage west living wing
PITCH_P = math.radians(35)                          # portico
BED = 156.3                                         # planted terrace beside the driveway apron (m AHD)
STONE_TOP = 1.2                            # limestone base on the wings


def bnd_ne(n):
    return 22.9 + 6.9 / 44.5 * n


def bnd_sw(n):
    return -5.1 / 32.5 * n


def upper_n(u):
    (a, b), (c, d) = L["upper_line"]
    return b + (d - b) * (u - a) / (c - a)


# ------------------------------------------------------------------ textures
def textures():
    R = random.Random(11)

    def batten(fname, base, hi, lo):
        im = Image.new("RGB", (1024, 1024), base)
        d = ImageDraw.Draw(im)
        for x in range(0, 1024, 102):                     # battens every 0.4 m (1024 px = 4 m)
            d.rectangle([x, 0, x + 12, 1024], fill=hi)
            d.rectangle([x + 13, 0, x + 16, 1024], fill=lo)
        arr = np.array(im).astype(np.int16) + np.random.default_rng(3).integers(-4, 4, (1024, 1024))[..., None]
        Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(TEX / fname)
    batten("cladding.png", (236, 234, 228), (246, 245, 241), (196, 194, 188))
    batten("sage.png", (154, 165, 140), (166, 177, 152), (118, 128, 106))
    # pale limestone / light tumbled brick in random courses, tileable; 1024 px = 2 m
    im = Image.new("RGB", (1024, 1024), (196, 191, 180))
    d = ImageDraw.Draw(im)
    y = 0
    while y < 1024:
        ch = R.choice((44, 52, 60, 76, 92))
        ch = min(ch, 1024 - y) if 1024 - y >= 40 else 1024 - y
        x = R.uniform(0, 200)
        x0 = x
        while x < x0 + 1024:
            bw = R.uniform(110, 300)
            l = R.uniform(-14, 10)
            warm = R.uniform(-6, 8)
            col = (int(232 + l + warm), int(226 + l), int(212 + l - warm))
            for off in (0, -1024):
                d.rectangle([x + off + 3, y + 3, x + off + bw - 3, y + ch - 3], fill=col)
                for _ in range(10):
                    cx, cy = x + off + R.uniform(0, bw), y + R.uniform(0, ch)
                    rw, rh = R.uniform(4, 22), R.uniform(2, 7)
                    t = R.uniform(-12, 8)
                    d.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=tuple(int(c + t) for c in col))
            x += bw
        y += ch
    arr = np.array(im).astype(np.int16) + np.random.default_rng(5).integers(-7, 7, (1024, 1024))[..., None]
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7)).save(TEX / "limestone.png")
    # hydrangea: dark green leaves with mophead blooms (white, lime, pink variants)
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
    arr = np.array(im).astype(np.int16)
    bloom = (arr[..., 0] > 200) & (arr[..., 3] > 0)
    lime = arr.copy()
    lime[bloom, 0] -= 34; lime[bloom, 2] -= 80
    Image.fromarray(lime.clip(0, 255).astype(np.uint8), "RGBA").save(TEX / "leaf_hydrangea_lime.png")
    pink = arr.copy()
    pink[bloom, 1] -= 70; pink[bloom, 2] -= 38; pink[bloom, 0] -= 4
    Image.fromarray(pink.clip(0, 255).astype(np.uint8), "RGBA").save(TEX / "leaf_hydrangea_pink.png")


def shingle_roof(name):
    """dark charcoal architectural shingles: horizontal courses every 0.18 m with mottled tabs"""
    m, nt, b = node_mat(name)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    div = nt.nodes.new("ShaderNodeMath"); div.operation = "DIVIDE"; div.inputs[1].default_value = 0.17
    nt.links.new(sep.outputs["Z"], div.inputs[0])
    fr = nt.nodes.new("ShaderNodeMath"); fr.operation = "FRACT"
    nt.links.new(div.outputs[0], fr.inputs[0])
    co = obj_coords(nt, 1.0)
    vor = nt.nodes.new("ShaderNodeTexVoronoi"); vor.inputs["Scale"].default_value = 3.2
    nt.links.new(co, vor.inputs["Vector"])
    nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 30; nz.inputs["Detail"].default_value = 8
    nt.links.new(co, nz.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*srgb("#232427"), 1)
    ramp.color_ramp.elements[1].color = (*srgb("#46484c"), 1)
    mix = nt.nodes.new("ShaderNodeMath"); mix.operation = "MULTIPLY_ADD"; mix.inputs[1].default_value = 0.6
    nt.links.new(vor.outputs["Distance"], mix.inputs[0])
    nt.links.new(nz.outputs["Fac"], mix.inputs[2])
    sub = nt.nodes.new("ShaderNodeMath"); sub.operation = "SUBTRACT"; sub.inputs[1].default_value = 0.25
    nt.links.new(mix.outputs[0], sub.inputs[0])
    nt.links.new(sub.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.85
    bn = nt.nodes.new("ShaderNodeBump"); bn.inputs["Strength"].default_value = 0.8; bn.inputs["Distance"].default_value = 0.03
    nt.links.new(fr.outputs[0], bn.inputs["Height"])
    bn2 = nt.nodes.new("ShaderNodeBump"); bn2.inputs["Strength"].default_value = 0.3
    nt.links.new(nz.outputs["Fac"], bn2.inputs["Height"])
    nt.links.new(bn2.outputs["Normal"], bn.inputs["Normal"])
    nt.links.new(bn.outputs["Normal"], b.inputs["Normal"])
    return m


# ------------------------------------------------------------------ walls with openings, in bands of material
def _pieces(a0, a1, z0, z1, openings):
    """rectangles (a, b, zlo, zhi) covering a0..a1 x z0..z1 less openings (a, b, zb, zt); openings may share columns"""
    cuts = sorted({a0, a1, *[min(max(x, a0), a1) for o in openings for x in o[:2]]})
    out = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a < 1e-6:
            continue
        z = z0
        for o in sorted((o for o in openings if o[0] <= a + 1e-6 and b <= o[1] + 1e-6), key=lambda o: o[2]):
            if o[2] > z:
                out.append((a, b, z, min(o[2], z1)))
            z = max(z, o[3])
            if z >= z1:
                break
        if z < z1:
            out.append((a, b, z, z1))
    return [p for p in out if p[3] - p[2] > 1e-6]


def wall_u(u0, u1, nf, nb, bands, openings=()):
    """wall along u between n = nf..nb; bands: [(z0, z1, mat)]"""
    for z0, z1, mat in bands:
        for a, b, zl, zh in _pieces(u0, u1, z0, z1, openings):
            box(a, b, nf, nb, zl, zh, mat)


def wall_n(n0, n1, uf, ub, bands, openings=()):
    for z0, z1, mat in bands:
        for a, b, zl, zh in _pieces(n0, n1, z0, z1, openings):
            box(uf, ub, a, b, zl, zh, mat)


def wing_bands(top, base=-0.35):
    return [(base, STONE_TOP, "limestone"), (STONE_TOP, top, "sage")]


def ledge_u(u0, u1, n_face, outward, gaps=()):
    """projecting stone sill course on top of the limestone base"""
    n0, n1 = sorted((n_face, n_face + outward * 0.06))
    for a, b, _, _ in _pieces(u0, u1, 0, 1, [(g[0], g[1], -1, 2) for g in gaps]):
        box(a, b, n0, n1, STONE_TOP - 0.02, STONE_TOP + 0.06, "limestone")


def ledge_n(n0_, n1_, u_face, outward, gaps=()):
    u0, u1 = sorted((u_face, u_face + outward * 0.06))
    for a, b, _, _ in _pieces(n0_, n1_, 0, 1, [(g[0], g[1], -1, 2) for g in gaps]):
        box(u0, u1, a, b, STONE_TOP - 0.02, STONE_TOP + 0.06, "limestone")


def frame_rect(a, b, zb, zt, plane, axis, cols, rows, mat="black", w=0.06, depth=0.07):
    def bar(p0, p1, z0, z1):
        lo_p, hi_p = (p0 - w / 2, p1 + w / 2) if p0 == p1 else (p0, p1)
        lo_z, hi_z = (z0 - w / 2, z1 + w / 2) if z0 == z1 else (z0, z1)
        if axis == "u":
            box(lo_p, hi_p, plane - depth / 2, plane + depth / 2, lo_z, hi_z, mat)
        else:
            box(plane - depth / 2, plane + depth / 2, lo_p, hi_p, lo_z, hi_z, mat)
    for k in range(cols + 1):
        x = a + (b - a) * k / cols
        bar(x, x, zb, zt)
    for k in range(rows + 1):
        z = zb + (zt - zb) * k / rows
        bar(a, b, z, z)


def window_u(a, b, zb, zt, nf, nb, cols=2, rows=3, outward=-1):
    face = nf if outward < 0 else nb
    poly([(a, (nf + nb) / 2, zb), (b, (nf + nb) / 2, zb), (b, (nf + nb) / 2, zt), (a, (nf + nb) / 2, zt)], "glass")
    frame_rect(a, b, zb, zt, face + outward * 0.02, "u", cols, rows)
    if zb > 0.2:
        box(a - 0.06, b + 0.06, min(face, face + outward * 0.07), max(face, face + outward * 0.07), zb - 0.07, zb, "black")


def window_n(a, b, zb, zt, uf, ub, cols=2, rows=3, outward=1):
    face = ub if outward > 0 else uf
    poly([((uf + ub) / 2, a, zb), ((uf + ub) / 2, b, zb), ((uf + ub) / 2, b, zt), ((uf + ub) / 2, a, zt)], "glass")
    frame_rect(a, b, zb, zt, face + outward * 0.02, "n", cols, rows)
    if zb > 0.2:
        box(min(face, face + outward * 0.07), max(face, face + outward * 0.07), a - 0.06, b + 0.06, zb - 0.07, zb, "black")


def gable_u(u0, u1, nf, nb, ze, pitch, mat, opening=None):
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


def gutter(p0, p1):
    """brass half-round gutter (drawn as a round tube with a fascia behind)"""
    cyl(p0, p1, 0.075, 0.075, "brass", sides=10)


def downpipe(gx, gn, z_top, px, pn, z_bot):
    cyl((gx, gn, z_top), (px, pn, z_top - 0.35), 0.042, 0.042, "brass", sides=8)
    cyl((px, pn, z_top - 0.35), (px, pn, z_bot), 0.042, 0.042, "brass", sides=8)
    cyl((px, pn, z_bot + 0.25), (px, pn + 0.0, z_bot), 0.05, 0.05, "brass", sides=8)


def roof_u(u0, u1, n0, n1, ze, pitch, o=0.45, gut=(True, True), mat="roof"):
    """two shingle planes, ridge parallel to u; charcoal barge boards; brass gutters on the eaves"""
    nm = (n0 + n1) / 2
    zr = ze + (n1 - n0) / 2 * math.tan(pitch)
    dz = o * math.tan(pitch)
    slab((u0 - o, n0 - o, ze - dz), (u1 + o, n0 - o, ze - dz), (u1 + o, nm, zr + 0.05), (u0 - o, nm, zr + 0.05), 0.1, mat)
    slab((u1 + o, n1 + o, ze - dz), (u0 - o, n1 + o, ze - dz), (u0 - o, nm, zr + 0.05), (u1 + o, nm, zr + 0.05), 0.1, mat)
    for u in (u0 - o, u1 + o):
        slab((u, n0 - o, ze - dz - 0.12), (u, nm, zr - 0.07), (u, nm, zr + 0.2), (u, n0 - o, ze - dz + 0.15), 0.06, "black")
        slab((u, nm, zr - 0.07), (u, n1 + o, ze - dz - 0.12), (u, n1 + o, ze - dz + 0.15), (u, nm, zr + 0.2), 0.06, "black")
    box(u0 - o, u1 + o, nm - 0.09, nm + 0.09, zr + 0.08, zr + 0.2, "black")      # ridge cap
    for n, g, s in ((n0 - o, gut[0], -1), (n1 + o, gut[1], 1)):
        box(u0 - o, u1 + o, n - 0.03 if s < 0 else n, n if s < 0 else n + 0.03, ze - dz - 0.2, ze - dz + 0.04, "black")
        if g:
            gutter((u0 - o, n + s * 0.1, ze - dz - 0.1), (u1 + o, n + s * 0.1, ze - dz - 0.1))
    return zr


def roof_n(n0, n1, u0, u1, ze, pitch, o=0.45, gut=(True, True), mat="roof"):
    um = (u0 + u1) / 2
    zr = ze + (u1 - u0) / 2 * math.tan(pitch)
    dz = o * math.tan(pitch)
    slab((u0 - o, n1 + o, ze - dz), (u0 - o, n0 - o, ze - dz), (um, n0 - o, zr + 0.05), (um, n1 + o, zr + 0.05), 0.1, mat)
    slab((u1 + o, n0 - o, ze - dz), (u1 + o, n1 + o, ze - dz), (um, n1 + o, zr + 0.05), (um, n0 - o, zr + 0.05), 0.1, mat)
    for n in (n0 - o, n1 + o):
        slab((u0 - o, n, ze - dz - 0.12), (um, n, zr - 0.07), (um, n, zr + 0.2), (u0 - o, n, ze - dz + 0.15), 0.06, "black")
        slab((um, n, zr - 0.07), (u1 + o, n, ze - dz - 0.12), (u1 + o, n, ze - dz + 0.15), (um, n, zr + 0.2), 0.06, "black")
    box(um - 0.09, um + 0.09, n0 - o, n1 + o, zr + 0.08, zr + 0.2, "black")
    for u, g, s in ((u0 - o, gut[0], -1), (u1 + o, gut[1], 1)):
        box(u - 0.03 if s < 0 else u, u if s < 0 else u + 0.03, n0 - o, n1 + o, ze - dz - 0.2, ze - dz + 0.04, "black")
        if g:
            gutter((u + s * 0.1, n0 - o, ze - dz - 0.1), (u + s * 0.1, n1 + o, ze - dz - 0.1))
    return zr


def wall_lantern(x, n, z, outward_n=None, outward_u=None, power=22, size=0.24):
    """brass carriage lantern on a wall bracket; outward_n = +-1 for walls facing +-n, outward_u for +-u"""
    s = size / 2
    if outward_n is not None:
        cx, cn = x, n + outward_n * (s + 0.12)
        box(x - 0.06, x + 0.06, min(n, n + outward_n * 0.03), max(n, n + outward_n * 0.03), z + 0.1, z + 0.45, "brass")
        cyl((x, n, z + 0.38), (cx, cn, z + 0.38), 0.015, 0.015, "brass", 6)
    else:
        cx, cn = x + outward_u * (s + 0.12), n
        box(min(x, x + outward_u * 0.03), max(x, x + outward_u * 0.03), n - 0.06, n + 0.06, z + 0.1, z + 0.45, "brass")
        cyl((x, n, z + 0.38), (cx, cn, z + 0.38), 0.015, 0.015, "brass", 6)
    h = size * 1.6
    box(cx - s, cx + s, cn - s, cn + s, z, z + 0.04, "brass")
    for dx, dn in ((-s, -s), (s, -s), (s, s), (-s, s)):
        box(cx + dx - 0.012, cx + dx + 0.012, cn + dn - 0.012, cn + dn + 0.012, z, z + h, "brass")
    faces("brass", [(cx - s * 1.15, cn - s * 1.15, z + h), (cx + s * 1.15, cn - s * 1.15, z + h), (cx + s * 1.15, cn + s * 1.15, z + h),
                    (cx - s * 1.15, cn + s * 1.15, z + h), (cx, cn, z + h + size * 0.5)], [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)])
    for p in ([(cx - s, cn - s), (cx + s, cn - s)], [(cx + s, cn - s), (cx + s, cn + s)], [(cx + s, cn + s), (cx - s, cn + s)], [(cx - s, cn + s), (cx - s, cn - s)]):
        poly([(p[0][0], p[0][1], z + 0.04), (p[1][0], p[1][1], z + 0.04), (p[1][0], p[1][1], z + h), (p[0][0], p[0][1], z + h)], "glass")
    cyl((cx, cn, z + 0.04), (cx, cn, z + h * 0.55), 0.035, 0.03, "bulb", 6)
    light("POINT", (cx, cn, z + h * 0.6), power, WARM, size=0.04, name="lantern")


def room_light(u0, u1, n0, n1, z, power):
    light("AREA", ((u0 + u1) / 2, (n0 + n1) / 2, z - 0.05), power, WARM, aim=((u0 + u1) / 2, (n0 + n1) / 2, -5),
          size=min(u1 - u0, n1 - n0) * 0.6, name="room")


def pendant(x, n, z, strength=50):
    cyl((x, n, z + 0.15), (x, n, z + 1.2), 0.006, 0.006, "black")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.14, location=(x, n, z))
    bpy.context.active_object.data.materials.append(MATS["bulb"])
    light("POINT", (x, n, z - 0.2), strength, WARM, size=0.1, name="pendant")


def disc(x, n, z, r, mat, k=16):
    poly([(x + r * math.cos(2 * math.pi * i / k), n + r * math.sin(2 * math.pi * i / k), z) for i in range(k)], mat)


def topiary(x, n, z, h=1.35, r=0.34, pot_h=0.7, pot_r=0.36):
    """spiral topiary in a large stone-grey pot"""
    cyl((x, n, z), (x, n, z + pot_h), pot_r * 0.82, pot_r, "pot", 16)
    cyl((x, n, z + pot_h - 0.02), (x, n, z + pot_h + 0.04), pot_r + 0.03, pot_r + 0.03, "pot", 16)
    disc(x, n, z + pot_h - 0.05, pot_r - 0.02, "mulch")
    cyl((x, n, z + pot_h - 0.1), (x, n, z + pot_h + h + 0.1), 0.025, 0.012, "darkbark", 5)
    turns, steps = 3.2, 64
    z0 = z + pot_h + 0.12
    for i in range(steps):
        t = i / steps
        ang = t * turns * 2 * math.pi
        rad = r * (1 - 0.72 * t)
        c = (x + math.cos(ang) * rad * 0.55, n + math.sin(ang) * rad * 0.55, z0 + t * h)
        for _ in range(3):
            card((c[0] + rr(-1, 1) * rad * 0.25, c[1] + rr(-1, 1) * rad * 0.25, c[2] + rr(-1, 1) * rad * 0.2), rad * rr(0.6, 0.85), "hedge")
    for _ in range(6):
        card((x + rr(-0.04, 0.04), n + rr(-0.04, 0.04), z0 + h + 0.05), 0.12, "hedge")


def path_light(x, n, z):
    cyl((x, n, z), (x, n, z + 0.5), 0.025, 0.025, "black", 6)
    box(x - 0.07, x + 0.07, n - 0.07, n + 0.07, z + 0.5, z + 0.54, "black")
    box(x - 0.045, x + 0.045, n - 0.045, n + 0.045, z + 0.44, z + 0.5, "bulb")
    light("POINT", (x, n, z + 0.42), 4, WARM, size=0.03, name="path")


def flagstone(cu, cn, z, w=0.45, h=0.32):
    a_ = rng.uniform(0, 3.14)
    w_, h_ = w * rr(0.85, 1.15), h * rr(0.85, 1.15)
    c_, s_ = math.cos(a_), math.sin(a_)
    k = 7
    pts = []
    for i in range(k):
        t = 2 * math.pi * i / k
        rx, rn = math.cos(t) * w_ * rr(0.85, 1.05), math.sin(t) * h_ * rr(0.85, 1.05)
        pts.append((cu + c_ * rx - s_ * rn, cn + s_ * rx + c_ * rn))
    v = [(u, n, z) for u, n in pts] + [(u, n, z - 0.06) for u, n in pts]
    faces("flagstone", v, [list(range(k))] + [(i, (i + 1) % k, k + (i + 1) % k, k + i) for i in range(k)])


def surface(key, z, mat, follow=False):
    for t in SURF.get(key, []):
        faces(mat, [(u, n, (zg(u, n) + z) if follow else z) for u, n in t], [(0, 1, 2)])


def pip(u, n, pts):
    inside = False
    for (ua, na), (ub, nb) in zip(pts, pts[1:] + pts[:1]):
        if (na > n) != (nb > n) and u < (ub - ua) * (n - na) / (nb - na) + ua:
            inside = not inside
    return inside


def bezier(p0, p1, p2, p3, t):
    return tuple((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t * t * c + t ** 3 * d for a, b, c, d in zip(p0, p1, p2, p3))


def arch_wall(u0, u1, n0, t, dc, rad, zs, ztop, mat):
    """wall piece u0..u1 x 0..ztop with an arched opening (springing zs, radius rad) centred on dc"""
    k = 12
    arc = [(dc + rad * math.cos(math.pi * i / k), zs + rad * math.sin(math.pi * i / k)) for i in range(k + 1)]
    box(u0, dc - rad, n0, n0 + t, -0.35, zs, mat)
    box(dc + rad, u1, n0, n0 + t, -0.35, zs, mat)
    outer = []
    for x, z in arc:  # radial projection of the arch onto the rectangle above the springing line
        dx, dz = x - dc, z - zs
        ts = [(u1 - dc) / dx] if dx > 1e-6 else ([(u0 - dc) / dx] if dx < -1e-6 else [])
        if dz > 1e-6:
            ts.append((ztop - zs) / dz)
        tt = min(ts)
        outer.append((dc + dx * tt, zs + dz * tt))
    for i in range(k):
        (x0, z0), (x1, z1) = arc[i], arc[i + 1]
        (X0, Z0), (X1, Z1) = outer[i], outer[i + 1]
        loop = [(x0, z0), (x1, z1), (X1, Z1)]
        if abs(Z1 - ztop) < 1e-6 and abs(Z0 - ztop) > 1e-6:      # crossing a top corner
            loop.append((u1 if X0 > dc else u0, ztop))
        if abs(Z0 - ztop) < 1e-6 and abs(Z1 - ztop) > 1e-6:
            loop.append((u0 if X1 < dc else u1, ztop))
        loop.append((X0, Z0))
        m = len(loop)
        v = [(x, n0, z) for x, z in loop] + [(x, n0 + t, z) for x, z in loop]
        faces(mat, v, [list(range(m)), [m + i for i in reversed(range(m))], (0, 1, m + 1, m)])
    return arc


def arched_door(dc, rad, zs, nf):
    arc = [(dc + rad * math.cos(math.pi * i / 12), zs + rad * math.sin(math.pi * i / 12)) for i in range(13)]
    box(dc - rad, dc + rad, nf + 0.1, nf + 0.15, 0, zs, "door")
    fan = [(dc, nf + 0.12, zs)] + [(x, nf + 0.12, z) for x, z in arc]
    for a_, b_ in zip(fan[1:], fan[2:]):
        faces("glass", [fan[0], a_, b_], [(0, 1, 2)])
    for (x0, z0_), (x1, z1_) in zip(arc, arc[1:]):
        slab((x0, nf - 0.02, z0_), (x1, nf - 0.02, z1_), (x1, nf - 0.02, z1_ - 0.07), (x0, nf - 0.02, z0_ - 0.07), 0.06, "black")
    for x in (dc - rad, dc + rad):
        box(x - 0.04, x + 0.04, nf - 0.04, nf + 0.02, 0, zs, "black")
    box(dc - rad, dc + rad, nf + 0.09, nf + 0.1, zs - 0.05, zs, "black")
    for x in (dc - 0.16, dc + 0.16):
        box(x - 0.015, x + 0.015, nf + 0.05, nf + 0.1, 0.9, 1.4, "brass")
    light("POINT", (dc, nf + 1.2, 2.1), 60, WARM, size=0.2, name="hall")


def colorbond(name, hexc):
    """corrugated Colorbond steel: ribs every 76 mm along the panel (object Y, the side boundaries run along n)"""
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


def fences():
    """1.8 m Colorbond fences on the side boundaries behind the front setback: beige (NE), Pale Eucalypt (SW)"""
    for (ua, na), (ub, nb), mat in (((0.0, 0.0), (-5.1, 32.5), "fence_sw"), ((22.9, 0.0), (29.8, 44.5), "fence_ne")):
        Lb = math.hypot(ub - ua, nb - na)
        tu, tn = (ub - ua) / Lb, (nb - na) / Lb
        iu, in_ = (tn, -tu) if mat == "fence_sw" else (-tn, tu)       # unit normal into the lot
        s = 6.5 / tn
        while s < Lb - 0.05:
            s1 = min(s + 2.4, Lb - 0.02)
            u0, n0 = ua + tu * s, na + tn * s
            u1, n1 = ua + tu * s1, na + tn * s1
            um, nm = (u0 + u1) / 2, (n0 + n1) / 2
            zi = zg(um + iu * 0.6, nm + in_ * 0.6)
            zo = zg(um - iu * 1.2, nm - in_ * 1.2)
            bot, top = min(zi, zo) - 0.15, max(zi, zo) + 1.8
            off = -0.01
            p = [(u0 + iu * off, n0 + in_ * off), (u1 + iu * off, n1 + in_ * off)]
            slab((p[0][0], p[0][1], bot), (p[1][0], p[1][1], bot), (p[1][0], p[1][1], top), (p[0][0], p[0][1], top), 0.03, mat)
            slab((p[0][0], p[0][1], top), (p[1][0], p[1][1], top), (p[1][0], p[1][1], top + 0.05), (p[0][0], p[0][1], top + 0.05), 0.07, mat)
            cyl((u0, n0, bot), (u0, n0, top + 0.05), 0.04, 0.04, mat, 4)
            s = s1


# ------------------------------------------------------------------ the house
def build_house():
    mu0, mu1, mn0, mn1 = L["main"]
    lu0, lu1, ln0, ln1 = L["link"]
    yu0, yu1, yn0, yn1 = L["gbay"]
    pu0_, pu1_, pn0_, pn1_ = L["portico"]
    eu0, eu1, en0, en1 = L["east"]
    wu0, wu1, wn0, wn1 = L["west"]
    gu0, gu1, gn0, gn1 = L["garage"]
    ou_, ouu, on_, onn = L["oriel"]

    for u0, u1, n0, n1 in (L["main"], L["link"], L["east"], L["west"], L["gbay"]):
        box(u0, u1, n0, n1, -0.25, 0.0, "floor")
    white = [(-0.35, 0.45, "limestone"), (0.45, EAVE_M, "clad")]
    o = 0.45

    # --- MAIN white two-storey gable to the street: upper retreat with a Juliet balcony (street view) and a
    #     louvred corner window (east view); ground floor: one major window in a bay under a metal awning
    jw = (4.0, 6.8, UPM + 0.1, 5.6)
    cw = (7.25, 8.4, UPM + 0.55, UPM + 2.0)
    wall_u(mu0, mu1, mn0, mn0 + T, white, [(yu0 + 0.2, yu1 - 0.2, 0.5, 2.6), (jw[0], jw[1], jw[2], EAVE_M + 1), cw])
    window_u(jw[0], jw[1], jw[2], jw[3], mn0, mn0 + T, cols=3, rows=2)
    window_u(*cw, mn0, mn0 + T, cols=1, rows=2)
    gable_u(mu0, mu1, mn0, mn0 + T, EAVE_M, PITCH_M, "clad", opening=(jw[0], jw[1], EAVE_M + 0.002, jw[3]))
    for x in np.arange(jw[0] - 0.05, jw[1] + 0.06, 0.13):                           # Juliet balcony
        box(x - 0.012, x + 0.012, mn0 - 0.12, mn0 - 0.096, UPM + 0.1, UPM + 1.05, "black")
    box(jw[0] - 0.1, jw[1] + 0.1, mn0 - 0.14, mn0 - 0.08, UPM + 1.02, UPM + 1.08, "black")
    box(jw[0] - 0.1, jw[1] + 0.1, mn0 - 0.16, mn0, UPM + 0.02, UPM + 0.1, "black")
    # ground-floor bay window (major street window, screened by the hydrangea bed)
    box(yu0, yu1, yn0, yn1, -0.35, 0.5, "limestone")
    box(yu0 - 0.03, yu1 + 0.03, yn0 - 0.03, yn1, 0.45, 0.52, "limestone")
    box(yu0, yu0 + 0.2, yn0, yn1, 0.5, 2.6, "clad")
    box(yu1 - 0.2, yu1, yn0, yn1, 0.5, 2.6, "clad")
    bw = (yu0 + 0.3, yu1 - 0.3, 0.75, 2.4)
    wall_u(yu0 + 0.2, yu1 - 0.2, yn0, yn0 + 0.2, [(0.5, 2.62, "clad")], [bw])
    window_u(*bw, yn0, yn0 + 0.2, cols=3, rows=2)
    slab((yu0 - 0.15, mn0, 3.0), (yu1 + 0.15, mn0, 3.0), (yu1 + 0.15, yn0 - 0.15, 2.66), (yu0 - 0.15, yn0 - 0.15, 2.66), 0.05, "awning")
    box(yu0 - 0.15, yu1 + 0.15, yn0 - 0.2, yn0 - 0.14, 2.56, 2.7, "black")
    # SW side (chimney) and NE side (louvred upper window above the link roof)
    sw_win = [(12.0, 13.0, 1.0, 2.2), (17.6, 18.6, 1.0, 2.2), (10.2, 11.2, UPM + 0.8, UPM + 1.9), (19.0, 19.9, UPM + 0.8, UPM + 1.9)]
    wall_n(mn0, mn1, mu0, mu0 + T, white, sw_win)
    for w in sw_win:
        window_n(*w, mu0, mu0 + T, 1, 2, outward=-1)
    ew_ = (mn0 + 0.3, ln0 - 0.35, UPM + 0.6, UPM + 2.0)      # east view, fixed louvres turn it SE, away from the NE yard
    wall_n(mn0, mn1, mu1 - T, mu1, white, [ew_])
    window_n(*ew_, mu1 - T, mu1, 2, 2, outward=1)
    for nn in np.arange(ew_[0] + 0.08, ew_[1], 0.16):
        slab((mu1 + 0.03, nn, ew_[2]), (mu1 + 0.03, nn, ew_[3]), (mu1 + 0.25, nn - 0.13, ew_[3]), (mu1 + 0.25, nn - 0.13, ew_[2]), 0.02, "timber")
    wall_u(mu0, mu1, mn1 - T, mn1, [(-0.35, EAVE_M, "clad")])
    gable_u(mu0, mu1, mn1 - T, mn1, EAVE_M, PITCH_M, "clad")
    box(mu0 + T, mu1 - T, mn0 + T, mn1 - T, UPM - 0.25, UPM, "slab")
    zr_main = roof_n(mn0, mn1, mu0, mu1, EAVE_M, PITCH_M)
    room_light(mu0 + 0.5, mu1 - 0.5, mn0 + 0.5, mn1 - 0.5, UPM - 0.25, 1100)
    room_light(mu0 + 0.5, mu1 - 0.5, mn0 + 0.5, mn1 - 0.5, EAVE_M + 0.8, 700)
    box(3.0, 7.6, 12.6, 13.6, 0, 0.42, "fabric")
    box(3.0, 7.6, 13.45, 13.7, 0, 0.8, "fabric")
    box(3.4, 7.2, 10.6, 12.4, 0, 0.01, "rug")
    box(3.4, 7.4, 13.0, 15.2, UPM, UPM + 0.55, "linen")                             # bed in the upper retreat
    pendant(5.4, 11.5, 2.1, 35)
    dzm = o * math.tan(PITCH_M)
    downpipe(mu0 - o - 0.1, mn0 - o + 0.1, EAVE_M - dzm - 0.1, mu0 - 0.1, mn0 + 0.15, PADZ)
    downpipe(mu1 + o + 0.1, mn0 - o + 0.1, EAVE_M - dzm - 0.1, mu1 + 0.1, mn0 + 0.15, PADZ)

    # --- rough limestone chimney on the main gable's SW side
    nmid = 15.5
    cu0 = mu0 - 0.8
    box(cu0, mu0 + 0.05, nmid - 1.0, nmid + 1.0, -0.4, EAVE_M - 0.6, "limestone")
    faces("limestone", [(cu0, nmid - 1.0, EAVE_M - 0.6), (mu0 + 0.05, nmid - 1.0, EAVE_M - 0.6), (mu0 + 0.05, nmid + 1.0, EAVE_M - 0.6), (cu0, nmid + 1.0, EAVE_M - 0.6),
                        (cu0 + 0.15, nmid - 0.6, EAVE_M + 0.1), (mu0 + 0.05, nmid - 0.6, EAVE_M + 0.1), (mu0 + 0.05, nmid + 0.6, EAVE_M + 0.1), (cu0 + 0.15, nmid + 0.6, EAVE_M + 0.1)],
          [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    box(cu0 + 0.15, mu0 + 0.05, nmid - 0.6, nmid + 0.6, EAVE_M + 0.1, zr_main + 0.5, "limestone")
    box(cu0 + 0.07, mu0 + 0.13, nmid - 0.68, nmid + 0.68, zr_main + 0.5, zr_main + 0.62, "limestone")
    for dn in (-0.25, 0.25):
        cyl((cu0 + 0.55, nmid + dn, zr_main + 0.6), (cu0 + 0.55, nmid + dn, zr_main + 0.9), 0.1, 0.09, "black", 8)

    # --- LINK: white single-storey entry + living, recessed behind the portico; big glass to the courtyard
    dc, rad, zs = (pu0_ + pu1_) / 2, 0.65, 2.05
    arch_wall(dc - 1.0, dc + 1.0, ln0, T, dc, rad, zs, EAVE_L, "clad")
    arched_door(dc, rad, zs, ln0)
    lw = (14.2, 15.0, 1.0, 2.2)
    wall_u(lu0, dc - 1.0, ln0, ln0 + T, [(-0.35, EAVE_L, "clad")])
    wall_u(dc + 1.0, lu1, ln0, ln0 + T, [(-0.35, EAVE_L, "clad")], [lw])
    window_u(*lw, ln0, ln0 + T, cols=1, rows=2)
    lbig = (lu0 + 0.3, lu1 - 0.3, 0.0, 2.7)
    wall_u(lu0, lu1, ln1 - T, ln1, [(-0.35, EAVE_L, "clad")], [lbig])
    window_u(*lbig, ln1 - T, ln1, cols=4, rows=1, outward=1)
    wall_n(ln0, ln1, lu1 - T, lu1, [(-0.35, EAVE_L, "clad")])
    gable_n(ln0, ln1, lu1 - T, lu1, EAVE_L, PITCH_L, "clad")
    roof_u(7.5, lu1, ln0, ln1, EAVE_L, PITCH_L)
    room_light(lu0 + 0.5, lu1 - 0.5, ln0 + 0.5, ln1 - 0.5, EAVE_L, 1300)
    box(10.0, 14.2, ln1 - 2.6, ln1 - 1.6, 0, 0.42, "fabric")
    box(10.0, 14.2, ln1 - 1.75, ln1 - 1.55, 0, 0.8, "fabric")
    box(10.4, 13.8, ln1 - 4.6, ln1 - 2.9, 0, 0.01, "rug")
    box(10.2, 14.2, 13.6, 14.6, 0.72, 0.77, "walnut")
    for x in (11.2, 12.2, 13.2):
        pendant(x, 14.1, 2.0, 30)
    dzl = o * math.tan(PITCH_L)
    downpipe(lu1 - 0.4, ln0 - o - 0.1, EAVE_L - dzl - 0.1, lu1 - 0.4, ln0 - 0.1, PADZ)
    for x in (dc - 0.95, dc + 0.95):
        wall_lantern(x, ln0, 1.75, outward_n=-1, power=18)

    # --- PORTICO: chunky timber posts on stone footings, knee braces, its own gable
    zb = 2.6
    box(pu0_ - 0.3, pu1_ + 0.3, pn0_ - 0.3, ln0, -0.45, 0.0, "limestone")
    box(pu0_ - 0.1, pu1_ + 0.1, pn0_ - 0.75, pn0_ - 0.3, -0.45, -0.08, "limestone")
    posts = [(pu0_ + 0.18, pn0_ + 0.18), (pu1_ - 0.18, pn0_ + 0.18)]
    for x, nn in posts:
        box(x - 0.26, x + 0.26, nn - 0.26, nn + 0.26, 0.0, 0.5, "limestone")
        box(x - 0.29, x + 0.29, nn - 0.29, nn + 0.29, 0.5, 0.56, "limestone")
        box(x - 0.17, x + 0.17, nn - 0.17, nn + 0.17, 0.56, zb, "timber")
    for x in (pu0_ + 0.18, pu1_ - 0.18):
        box(x - 0.15, x + 0.15, ln0 - 0.14, ln0, 0.0, zb, "timber")
    box(pu0_, pu1_, pn0_ + 0.02, pn0_ + 0.34, zb - 0.32, zb, "timber")
    for x in (pu0_ + 0.18, pu1_ - 0.18):
        box(x - 0.15, x + 0.15, pn0_, ln0, zb - 0.32, zb, "timber")
    for x, nn in posts:
        s = 1 if x < (pu0_ + pu1_) / 2 else -1
        cyl((x + s * 0.1, nn, zb - 0.95), (x + s * 0.75, nn, zb - 0.3), 0.07, 0.07, "timber", 4)
        cyl((x, nn + 0.1, zb - 0.95), (x, nn + 0.75, zb - 0.3), 0.07, 0.07, "timber", 4)
    box(pu0_ + 0.2, pu1_ - 0.2, pn0_ + 0.3, ln0, zb - 0.05, zb, "timber")
    gp = PITCH_P
    zr_p = gable_u(pu0_, pu1_, pn0_, pn0_ + 0.12, zb, gp, "timber")
    box(pu0_, pu1_, pn0_ - 0.02, pn0_ + 0.14, zb, zb + 0.22, "timber")
    box(dc - 0.09, dc + 0.09, pn0_ - 0.03, pn0_ + 0.14, zb + 0.22, zr_p - 0.1, "timber")
    roof_n(pn0_, ln0 + 0.1, pu0_, pu1_, zb, gp, o=0.32)
    box(pu0_ + 0.2, pu1_ - 0.2, pn0_ + 0.12, ln0, 0.0, 0.01, "deck")
    hx, hn, hz = dc, (pn0_ + ln0) / 2, zb - 1.0
    cyl((hx, hn, zb - 0.05), (hx, hn, hz + 0.5), 0.01, 0.01, "black", 4)
    box(hx - 0.16, hx + 0.16, hn - 0.16, hn + 0.16, hz, hz + 0.03, "brass")
    box(hx - 0.18, hx + 0.18, hn - 0.18, hn + 0.18, hz + 0.47, hz + 0.5, "brass")
    for dx, dn in ((-0.16, -0.16), (0.16, -0.16), (0.16, 0.16), (-0.16, 0.16)):
        box(hx + dx - 0.012, hx + dx + 0.012, hn + dn - 0.012, hn + dn + 0.012, hz, hz + 0.5, "brass")
    box(hx - 0.04, hx + 0.04, hn - 0.04, hn + 0.04, hz + 0.05, hz + 0.3, "bulb")
    light("POINT", (hx, hn, hz + 0.25), 35, WARM, size=0.05, name="lantern_portico")
    topiary(dc - 1.15, ln0 - 0.55, 0.0, h=1.15, r=0.3, pot_h=0.6, pot_r=0.3)
    topiary(dc + 1.15, ln0 - 0.55, 0.0, h=1.15, r=0.3, pot_h=0.6, pot_r=0.3)
    topiary(pu0_ - 0.75, pn0_ - 0.2, PADZ, h=1.5, r=0.36)
    topiary(pu1_ + 0.75, pn0_ - 0.2, PADZ, h=1.5, r=0.36)

    # --- EAST sage wing over the garage: gable to the street with an oriel bay window under a metal awning
    wall_u(eu0, eu1, en0, en0 + T, wing_bands(EAVE_E, base=0.0), [(ou_ + 0.2, ouu - 0.2, 0.5, 2.75)])
    ledge_u(eu0, eu1, en0, -1, gaps=[(ou_, ouu)])
    box(ou_, ouu, on_, onn, 0.3, 0.5, "limestone")
    for x in (ou_ + 0.4, ouu - 0.4):
        cyl((x, en0 - 0.02, -0.25), (x, on_ + 0.15, 0.3), 0.06, 0.06, "timber", 4)
    box(ou_, ou_ + 0.2, on_, onn, 0.5, 2.75, "sage")
    box(ouu - 0.2, ouu, on_, onn, 0.5, 2.75, "sage")
    ow = (ou_ + 0.5, ouu - 0.5, 0.85, 2.3)
    wall_u(ou_ + 0.2, ouu - 0.2, on_, on_ + 0.2, [(0.5, 2.8, "sage")], [ow])
    window_u(*ow, on_, on_ + 0.2, cols=2, rows=2)
    slab((ou_ - 0.15, en0, 3.08), (ouu + 0.15, en0, 3.08), (ouu + 0.15, on_ - 0.18, 2.78), (ou_ - 0.15, on_ - 0.18, 2.78), 0.05, "awning")
    box(ou_ - 0.15, ouu + 0.15, on_ - 0.23, on_ - 0.17, 2.68, 2.82, "black")
    ga = (18.7, 19.7, EAVE_E + 0.8, EAVE_E + 1.9)
    gable_u(eu0, eu1, en0, en0 + T, EAVE_E, PITCH_E, "sage", opening=ga)
    window_u(*ga, en0, en0 + T, cols=1, rows=2)
    for x in (ou_ - 0.45, ouu + 0.45):
        wall_lantern(x, en0, 1.55, outward_n=-1, power=16)
    wg = (ln1 + 0.25, en1 - 0.35, 0.0, 2.7)
    wall_n(en0, en1, eu0, eu0 + T, wing_bands(EAVE_E, base=0.0), [wg])
    window_n(*wg, eu0, eu0 + T, cols=2, rows=1, outward=-1)
    ng = (eu0 + 0.5, eu1 - 0.5, 0.0, 2.7)
    wall_u(eu0, eu1, en1 - T, en1, wing_bands(EAVE_E), [ng])
    window_u(*ng, en1 - T, en1, cols=4, rows=1, outward=1)
    gw = (18.0, 20.4, EAVE_E + 0.25, EAVE_E + 2.0)
    gable_u(eu0, eu1, en1 - T, en1, EAVE_E, PITCH_E, "sage", opening=gw)
    window_u(*gw, en1 - T, en1, cols=3, rows=2, outward=1)
    e1, e2 = (11.4, 12.8, 1.0, 2.2), (18.0, 19.4, 1.0, 2.2)
    wall_n(en0, gn1, eu1 - T, eu1, [(GZ - 0.3, STONE_TOP, "limestone"), (STONE_TOP, EAVE_E, "sage")], [e1])
    wall_n(gn1, en1, eu1 - T, eu1, wing_bands(EAVE_E), [e2])
    for w in (e1, e2):
        window_n(*w, eu1 - T, eu1, cols=1, rows=2, outward=1)
    ledge_n(en0, en1, eu1, 1, gaps=[e1[:2], e2[:2], (14.0, 16.5)])
    box(eu1, eu1 + 0.3, 14.0, 16.5, GZ - 0.3, EAVE_E - 0.05, "limestone")   # stone panel breaks the long side wall
    roof_n(en0, en1, eu0, eu1, EAVE_E, PITCH_E)
    dz = o * math.tan(PITCH_E)
    downpipe(eu1 + o + 0.1, en0 - o + 0.1, EAVE_E - dz - 0.1, eu1 + 0.1, en0 + 0.15, GZ)
    downpipe(eu1 + o + 0.1, en1 + o - 0.1, EAVE_E - dz - 0.1, eu1 + 0.1, en1 - 0.15, PADZ)
    room_light(eu0 + 0.5, eu1 - 0.5, en0 + 0.5, en1 - 0.5, EAVE_E, 1100)
    box(17.0, 21.4, en1 - 3.6, en1 - 2.6, 0.0, 0.92, "walnut")
    for x in (17.9, 19.2, 20.5):
        pendant(x, en1 - 3.1, 1.95, 30)
    box(17.4, 21.0, en0 + 1.0, en0 + 2.0, 0.72, 0.77, "walnut")

    # --- WEST sage living wing: sliders onto the courtyard, glazed north gable; its roof dies into the main gable
    sl = (wn0 + 0.6, wn1 - 0.6, 0.0, 2.7)
    wall_n(wn0, wn1, wu1 - T, wu1, wing_bands(EAVE_W), [sl])
    window_n(*sl, wu1 - T, wu1, cols=5, rows=1, outward=1)
    sw_ = [(23.6, 24.8, 1.1, 2.3), (27.4, 28.6, 1.1, 2.3)]
    wall_n(wn0, wn1, wu0, wu0 + T, wing_bands(EAVE_W), sw_)
    for w in sw_:
        window_n(*w, wu0, wu0 + T, cols=1, rows=2, outward=-1)
    ledge_n(wn0, wn1, wu0, -1, gaps=[w[:2] for w in sw_])
    nw = (wu0 + 0.6, wu1 - 0.6, 0.0, 2.7)
    wall_u(wu0, wu1, wn1 - T, wn1, wing_bands(EAVE_W), [nw])
    window_u(*nw, wn1 - T, wn1, cols=4, rows=1, outward=1)
    ngw = (wu0 + 2.0, wu1 - 2.0, EAVE_W + 0.25, EAVE_W + 1.95)
    gable_u(wu0, wu1, wn1 - T, wn1, EAVE_W, PITCH_W, "sage", opening=ngw)
    window_u(*ngw, wn1 - T, wn1, cols=3, rows=2, outward=1)
    ledge_u(wu0, wu1, wn1, 1, gaps=[nw[:2]])
    roof_n(wn0 + o, wn1, wu0, wu1, EAVE_W, PITCH_W)
    dzw = o * math.tan(PITCH_W)
    downpipe(wu1 + o + 0.1, wn1 + o - 0.1, EAVE_W - dzw - 0.1, wu1 + 0.1, wn1 - 0.15, PADZ)
    downpipe(wu0 - o - 0.1, wn1 + o - 0.1, EAVE_W - dzw - 0.1, wu0 - 0.1, wn1 - 0.15, PADZ)
    room_light(wu0 + 0.5, wu1 - 0.5, wn0 + 0.5, wn1 - 0.5, EAVE_W, 1300)
    box(3.0, 7.6, 25.0, 26.0, 0, 0.42, "fabric")
    box(3.0, 7.6, 24.8, 25.05, 0, 0.8, "fabric")
    box(3.4, 7.2, 26.4, 28.6, 0, 0.01, "rug")
    box(4.6, 5.8, 27.0, 27.8, 0, 0.38, "walnut")
    pendant(5.4, 28.6, 2.2, 40)
    for nn in (22.6, 29.4):
        wall_lantern(wu1, nn, 1.8, outward_u=1, power=14)

    # --- GARAGE at street level under the east wing (door set back > 5.5 m); timber carriage door
    ring([(gu0, GZ - 0.3), (gu1, GZ - 0.3), (gu1, 0.0), (gu0, 0.0)],
         [(gu0 + 0.6, GZ), (gu1 - 0.6, GZ), (gu1 - 0.6, GZ + 2.3), (gu0 + 0.6, GZ + 2.3)], gn0, gn0 + 0.35, "limestone")
    box(gu1, eu1, gn0, gn0 + 0.35, GZ - 0.3, 0.0, "limestone")
    box(gu0 + 0.6, gu1 - 0.6, gn0 + 0.42, gn0 + 0.47, GZ, GZ + 2.3, "slats")
    for x in np.linspace(gu0 + 0.6, gu1 - 0.6, 7):
        box(x - 0.03, x + 0.03, gn0 + 0.38, gn0 + 0.43, GZ, GZ + 2.3, "black")
    box(gu0 + 0.6, gu1 - 0.6, gn0 + 0.38, gn0 + 0.43, GZ + 1.6, GZ + 1.66, "black")
    box(gu0 + 0.55, gu1 - 0.55, gn0 - 0.02, gn0 + 0.36, GZ + 2.3, GZ + 2.42, "black")
    box(gu0, gu0 + 0.3, gn0, gn1, GZ - 0.3, PADZ + 0.05, "limestone")
    box(gu1 - 0.3, gu1, gn0, gn1, GZ - 0.3, -0.25, "limestone")
    box(gu0, gu1, gn0, gn1, -0.4, -0.25, "slab")
    for x in (gu0 + 0.3, gu1 - 0.3):
        wall_lantern(x, gn0, GZ + 1.8, outward_n=-1, power=14)
    light("AREA", ((gu0 + gu1) / 2, (gn0 + gn1) / 2, GZ + 2.2), 120, WARM, aim=((gu0 + gu1) / 2, (gn0 + gn1) / 2, GZ - 5), size=3, name="garage_in")
    box(eu1, bnd_ne(gn1) - 0.05, gn1 - 0.3, gn1, GZ - 0.3, PADZ + 0.9, "limestone")     # side strip ends at the pad
    # driveway apron: low planted retaining walls either side
    ab0, ab1, abn0, abn1 = L["apron_bed"]
    bedz = BED - cr.FFL
    box(ab1 - 0.3, ab1, 0.9, abn1, GZ - 0.3, bedz + 0.35, "limestone")
    box(ab1 - 0.33, ab1 + 0.03, 0.87, abn1, bedz + 0.35, bedz + 0.42, "limestone")
    box(ab0, ab1 - 0.3, 2.2, abn1, bedz - 0.3, bedz + 0.05, "mulch")
    box(ab0, ab1, abn1 - 0.3, abn1, bedz - 0.3, PADZ + 0.05, "limestone")
    box(gu1, gu1 + 0.3, 0.8, gn0, GZ - 0.3, GZ + 0.45, "limestone")
    box(gu1 - 0.03, gu1 + 0.33, 0.77, gn0, GZ + 0.45, GZ + 0.52, "limestone")
    eb = [(gu1 + 0.3, 0.8), (bnd_ne(0.8) - 0.3, 0.8), (bnd_ne(gn0) - 0.3, gn0), (gu1 + 0.3, gn0)]
    faces("mulch", [(u, n, GZ + 0.4) for u, n in eb] + [(u, n, GZ - 0.1) for u, n in eb], [(0, 1, 2, 3), (4, 5, 1, 0)])

    # --- street wall: low limestone wall with pillars and brass lanterns, steps up at the SW end
    s0, s1, sn0, sn1 = L["stair"]
    topw = FRONT["wall_top"]
    for a, b in ((0.3, s0 - 0.45), (s1 + 0.45, ab1 - 0.3)):
        box(a, b, 1.6, 2.2, -3.4, topw, "limestone")
        box(a - 0.03, b + 0.03, 1.55, 2.25, topw, topw + 0.08, "limestone")
    gg = zg((s0 + s1) / 2, 1.2)
    for x in (s0 - 0.45, s1 + 0.45):
        box(x - 0.42, x + 0.42, 1.48, 2.32, gg - 0.3, topw + 0.45, "limestone")
        box(x - 0.48, x + 0.48, 1.42, 2.38, topw + 0.45, topw + 0.55, "limestone")
        wall_lantern(x, 1.48, topw - 0.2, outward_n=-1, power=14)
    n_steps = 11
    for k in range(n_steps):
        z = gg + (PADZ - gg) * (k + 1) / n_steps
        n_a = 1.6 + k * (sn1 - 1.6) / n_steps
        box(s0, s1, n_a, sn1, gg - 0.4, z, "paving")
    for x0, x1 in ((s0 - 0.3, s0), (s1, s1 + 0.3)):
        box(x0, x1, 2.2, sn1, gg - 0.4, PADZ + 0.3, "limestone")
    # flagstone steppers across the lawn to the portico, with path lights
    p0, p1, p2, p3 = ((s0 + s1) / 2, sn1 + 0.2), ((s0 + s1) / 2 + 0.6, sn1 + 2.0), (pu0_ - 2.5, pn0_ - 2.6), (dc, pn0_ - 0.95)
    for k in range(15):
        cu, cn = bezier(p0, p1, p2, p3, k / 14)
        flagstone(cu, cn, PADZ + 0.06, 0.42, 0.3)
        if k % 3 == 1:
            path_light(cu + 0.2, cn - 0.75, PADZ)
    bt = FRONT["bank_top"]
    drape(-0.3, s0 - 0.3, lambda u: 2.2, lambda u: bt, 0.05, "mulch", 8, 6)
    drape(s1 + 0.3, ab0 - 0.3, lambda u: 2.2, lambda u: bt, 0.05, "mulch", 30, 6)
    # garden beds along the white facade
    box(mu0 + 0.1, mu1 - 0.1, yn0 - 1.1, mn0, PADZ - 0.1, PADZ + 0.06, "mulch")
    box(lu0 + 0.1, pu0_ - 0.4, ln0 - 1.1, ln0, PADZ - 0.1, PADZ + 0.06, "mulch")
    box(pu1_ + 0.4, lu1 - 0.1, ln0 - 1.1, ln0, PADZ - 0.1, PADZ + 0.06, "mulch")
    fences()

    # --- surfaces from the site fit: lawns, courtyard paving, upper terrace, garage roof terrace, drive strip
    surface("lawn_lo", PADZ + 0.03, "lawn")
    surface("paving", PADZ + 0.05, "paving")
    surface("lawn_up", UPZ + 0.03, "lawn")
    surface("fire_pad", UPZ + 0.05, "flagstone2")
    surface("drive_side", 0.06, "aggregate", follow=True)

    # --- POOL in the courtyard, pale coping, outdoor dining at the big glass
    pu0, pu1, pn0, pn1 = L["pool"]
    for a, c, n0_, n1_ in ((pu0 - 0.35, pu1 + 0.35, pn0 - 0.35, pn0), (pu0 - 0.35, pu1 + 0.35, pn1, pn1 + 0.35),
                           (pu0 - 0.35, pu0, pn0, pn1), (pu1, pu1 + 0.35, pn0, pn1)):
        box(a, c, n0_, n1_, PADZ - 0.2, PADZ + 0.08, "coping")
    box(pu0, pu1, pn0, pn1, PADZ - 1.7, PADZ - 1.6, "pooltile")
    for a, c, n0_, n1_ in ((pu0, pu1, pn0, pn0 + 0.02), (pu0, pu1, pn1 - 0.02, pn1), (pu0, pu0 + 0.02, pn0, pn1), (pu1 - 0.02, pu1, pn0, pn1)):
        box(a, c, n0_, n1_, PADZ - 1.7, PADZ - 0.1, "pooltile")
    poly([(pu0, pn0, PADZ - 0.12), (pu1, pn0, PADZ - 0.12), (pu1, pn1, PADZ - 0.12), (pu0, pn1, PADZ - 0.12)], "water")
    for x in (pu0 + 2.4, pu1 - 2.4):
        light("POINT", (x, pn0 + 0.15, PADZ - 0.9), 55, (0.55, 0.85, 1.0), size=0.05, name="pool")
    for x in (pu0 + 1.5, pu0 + 3.0, pu1 - 3.0, pu1 - 1.5):           # sun loungers on the north deck
        box(x - 0.35, x + 0.35, pn1 + 0.6, pn1 + 1.6, PADZ + 0.05, PADZ + 0.3, "black")
        box(x - 0.32, x + 0.32, pn1 + 0.62, pn1 + 1.58, PADZ + 0.3, PADZ + 0.38, "linen")
    box(12.0, 14.8, 22.0, 23.0, PADZ + 0.72, PADZ + 0.77, "walnut")
    for x in (12.3, 13.4, 14.5):
        for nn in (21.6, 23.4):
            box(x - 0.2, x + 0.2, nn - 0.2, nn + 0.2, PADZ, PADZ + 0.45, "black")
    for x, nn in ((10.3, 22.0), (20.9, 24.3)):
        topiary(x, nn, PADZ, h=1.3, r=0.32)

    # --- covered outdoor room with a stone fireplace (NE side of the courtyard)
    ou0, ou1, on0, on1 = L["outdoor"]
    zo = PADZ + 0.15
    box(ou0, ou1, on0, on1, PADZ - 0.3, zo, "flagstone")
    for x, nn in ((ou0 + 0.2, on0 + 0.2), (ou0 + 0.2, on1 - 0.2), (ou1 - 0.2, on0 + 0.2), (ou1 - 0.2, on1 - 0.2)):
        box(x - 0.16, x + 0.16, nn - 0.16, nn + 0.16, zo, zo + 2.7, "timber")
    for nn, sgn in ((on0 + 0.2, 1), (on1 - 0.2, -1)):
        cyl((ou0 + 0.2, nn + sgn * 0.1, zo + 1.9), (ou0 + 0.2, nn + sgn * 0.7, zo + 2.55), 0.06, 0.06, "timber", 4)
    box(ou0, ou1, on0 + 0.05, on0 + 0.35, zo + 2.7, zo + 2.95, "timber")
    box(ou0, ou1, on1 - 0.35, on1 - 0.05, zo + 2.7, zo + 2.95, "timber")
    box(ou0 + 0.05, ou0 + 0.35, on0, on1, zo + 2.7, zo + 2.95, "timber")
    roof_n(on0, on1, ou0, ou1, zo + 2.95, math.radians(40), o=0.3)
    fn0, fn1 = on0 + 1.4, on1 - 1.4
    box(ou1 - 0.6, ou1, fn0, fn1, zo, zo + 2.6, "limestone")
    box(ou1 - 0.5, ou1, fn0 + 0.6, fn1 - 0.6, zo + 2.6, zo + 5.9, "limestone")
    box(ou1 - 0.55, ou1 + 0.05, fn0 + 0.55, fn1 - 0.55, zo + 5.9, zo + 6.02, "black")
    box(ou1 - 0.65, ou1 - 0.55, fn0 + 0.6, fn1 - 0.6, zo + 0.45, zo + 1.35, "black")
    box(ou1 - 0.55, ou1 - 0.3, fn0 + 0.75, fn1 - 0.75, zo + 0.5, zo + 0.75, "fire")
    light("POINT", (ou1 - 0.9, (fn0 + fn1) / 2, zo + 0.8), 110, (1.0, 0.45, 0.15), size=0.3, name="pool_fire")
    box(ou1 - 1.0, ou1 - 0.6, fn0 - 0.2, fn1 + 0.2, zo, zo + 0.4, "limestone")
    box(ou0 + 0.5, ou0 + 1.4, on0 + 1.2, on1 - 1.2, zo, zo + 0.45, "linen")
    box(ou0 + 0.5, ou0 + 0.8, on0 + 1.2, on1 - 1.2, zo + 0.45, zo + 0.85, "linen")
    box(ou0 + 1.9, ou0 + 2.6, on0 + 2.4, on1 - 2.4, zo, zo + 0.4, "walnut")
    pendant((ou0 + ou1) / 2, (on0 + on1) / 2, zo + 2.3, 35)

    # --- upper garden terrace: limestone retaining wall with a stair, fire-pit circle, out to the rear boundary
    (ua, na), (ub, nb) = LAY["upper_wall"][0], LAY["upper_wall"][-1]
    st0, st1 = 13.6, 15.8
    Lw = math.hypot(ub - ua, nb - na)
    tu, tn = (ub - ua) / Lw, (nb - na) / Lw
    nu, nn_ = -tn, tu            # normal pointing to the upper side (+n)

    def wseg(s0_, s1_, z0, z1, off0=0.0, off1=0.3, mat="limestone"):
        p = [(ua + tu * s + nu * off, na + tn * s + nn_ * off) for s, off in ((s0_, off0), (s1_, off0), (s1_, off1), (s0_, off1))]
        faces(mat, [(u, n, z0) for u, n in p] + [(u, n, z1) for u, n in p],
              [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    s_st0 = (st0 - ua) / tu
    s_st1 = (st1 - ua) / tu
    for a, b in ((0, s_st0), (s_st1, Lw)):
        wseg(a, b, PADZ - 0.4, UPZ + 0.35)
        wseg(a, b, UPZ + 0.35, UPZ + 0.42, -0.04, 0.34)
    nst = 5
    for k in range(nst):
        z = PADZ + (UPZ - PADZ) * (k + 1) / nst
        wseg(s_st0, s_st1, PADZ - 0.3, z, -(nst - k) * 0.34, 0.3 if k == nst - 1 else -(nst - k - 1) * 0.34, "paving")
    for s in (s_st0 - 0.15, s_st1 + 0.15):
        path_light(ua + tu * s - nu * 0.3, na + tn * s - nn_ * 0.3, PADZ)
    # fire pit + timber chairs on the flagstone circle
    fx, fn = 12.0, upper_n(12.0) + 3.6
    cyl((fx, fn, UPZ), (fx, fn, UPZ + 0.42), 0.75, 0.75, "limestone", 16)
    disc(fx, fn, UPZ + 0.42, 0.75, "limestone")
    disc(fx, fn, UPZ + 0.43, 0.5, "fire")
    light("POINT", (fx, fn, UPZ + 0.75), 90, (1.0, 0.45, 0.15), size=0.3, name="pool_firepit")
    for a in (0.6, 2.0, 3.6, 5.0):
        cx, cn = fx + math.cos(a) * 1.65, fn + math.sin(a) * 1.65
        dx_, dn_ = math.cos(a), math.sin(a)
        tx_, tn_ = -dn_, dx_
        box(cx - 0.3, cx + 0.3, cn - 0.3, cn + 0.3, UPZ + 0.05, UPZ + 0.4, "timber")
        bx, bn_ = cx + dx_ * 0.32, cn + dn_ * 0.32
        slab((bx + tx_ * 0.3, bn_ + tn_ * 0.3, UPZ + 0.4), (bx - tx_ * 0.3, bn_ - tn_ * 0.3, UPZ + 0.4),
             (bx - tx_ * 0.3 + dx_ * 0.18, bn_ - tn_ * 0.3 + dn_ * 0.18, UPZ + 1.0), (bx + tx_ * 0.3 + dx_ * 0.18, bn_ + tn_ * 0.3 + dn_ * 0.18, UPZ + 1.0),
             0.05, "timber")

    # --- boundary retaining walls along the levelled ground (cut: retain the neighbour's higher ground; fill: 0.9 m wall)
    pts = [tuple(p) for p in PAD_OUTLINE[:-1]]
    for (ua_, na_), (ub_, nb_) in zip(pts, pts[1:] + pts[:1]):
        if max(na_, nb_) < 2.6:
            continue
        if min(ua_, ub_) > gu0 - 0.5 and max(na_, nb_) < gn1 + 0.6:
            continue
        Lg = math.hypot(ub_ - ua_, nb_ - na_)
        if Lg < 0.05:
            continue
        dx, dn = (ub_ - ua_) / Lg, (nb_ - na_) / Lg
        ox, on = -dn, dx
        mu, mn = (ua_ + ub_) / 2, (na_ + nb_) / 2
        if pip(mu + ox * 0.2, mn + on * 0.2, pts):
            ox, on = -ox, -on           # (ox, on) points out of the pad
        steps = max(1, int(Lg / 1.5))
        for k in range(steps):
            t0, t1 = k / steps, (k + 1) / steps
            u0_, n0_ = ua_ + (ub_ - ua_) * t0, na_ + (nb_ - na_) * t0
            u1_, n1_ = ua_ + (ub_ - ua_) * t1, na_ + (nb_ - na_) * t1
            cu_, cn_ = (u0_ + u1_) / 2, (n0_ + n1_) / 2
            zi = UPZ if cn_ > upper_n(cu_) + 0.2 else (PADZ if cn_ > FRONT["bank_top"] else zg(cu_ - ox * 0.6, cn_ - on * 0.6))
            zo_ = zg(cu_ + ox * 1.4, cn_ + on * 1.4)
            top = zo_ + 0.15 if zo_ > zi + 0.05 else zi + 0.9
            bot = min(zi, zo_) - 0.45
            q = [(u0_, n0_), (u1_, n1_), (u1_ + ox * 0.27, n1_ + on * 0.27), (u0_ + ox * 0.27, n0_ + on * 0.27)]
            faces("limestone", [(u, n, bot) for u, n in q] + [(u, n, top) for u, n in q],
                  [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
            qc = [(u0_ - ox * 0.03, n0_ - on * 0.03), (u1_ - ox * 0.03, n1_ - on * 0.03), (u1_ + ox * 0.29, n1_ + on * 0.29), (u0_ + ox * 0.29, n0_ + on * 0.29)]
            faces("limestone", [(u, n, top) for u, n in qc] + [(u, n, top + 0.07) for u, n in qc],
                  [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])

    # --- street: road, kerbs, footpaths, nature strip, verge garden, driveway
    R = ARC["r"]
    drape(-45, 68, lambda u: arc_n(u, R - 15.0), lambda u: arc_n(u, R - 4.0), 0.03, "asphalt", 150, 10)
    for r0, r1 in ((R - 4.0, R - 3.75), (R - 15.25, R - 15.0)):
        drape(-45, 68, lambda u, r0=r0: arc_n(u, r0), lambda u, r1=r1: arc_n(u, r1), 0.17, "kerb", 150, 1)
    drape(-45, 68, lambda u: arc_n(u, R - 16.7), lambda u: arc_n(u, R - 15.25), 0.1, "concrete", 150, 2)
    drape(-45, 68, lambda u: arc_n(u, R - 24.0), lambda u: arc_n(u, R - 16.7), 0.05, "lawn", 150, 4)
    drape(-45, gu0 + 0.2, lambda u: arc_n(u, R - 3.75), lambda u: arc_n(u, R - 2.3), 0.1, "concrete", 80, 2)
    drape(gu1 + 0.6, 68, lambda u: arc_n(u, R - 3.75), lambda u: arc_n(u, R - 2.3), 0.1, "concrete", 60, 2)
    drape(gu0 + 0.2, gu1 + 0.6, lambda u: arc_n(u, R - 3.75), lambda u: gn0, 0.08, "aggregate", 12, 12)
    drape(-1.5, gu0, lambda u: arc_n(u, R - 2.3), lambda u: 1.58, 0.06, "mulch", 40, 4)
    drape(-45, -1.5, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 1.0, 0.04, "lawn", 30, 2)
    drape(bnd_ne(0) + 0.6, 68, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 1.0, 0.04, "lawn", 30, 2)


EUC = []


def hydr(x, n, z, r=None):
    kind = rng.choice(("hydrangea", "hydrangea_pink", "hydrangea_lime", "hydrangea_pink", "hydrangea"))
    bush((x, n, z), r or rr(0.36, 0.48), kind, 18)


def build_garden():
    """eucalypts on the hill; pink, white and lime hydrangeas, box balls, hedges on the side boundaries"""
    S = cr.S
    for gx, gy, r in S["trees"]:
        if rng.random() < 0.3:
            continue
        u, n = cr.to_frame(gx, gy)
        if -8 < u < 34 and n < 47 and pip(u, n, [tuple(p) for p in LAY["lot_outline"][:-1]]):
            continue                                    # nothing tall inside the levelled lot (bushfire buffer)
        h = min(max(r * 2.8 + rr(2, 5), 7), 17)
        tree((u, n, cr.h_grid(gx, gy) - cr.FFL - 0.2), h, r, "euc", "bark", cards_per=14)
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
    s0, s1, sn0, sn1 = L["stair"]
    mu0, mu1, mn0, mn1 = L["main"]
    lu0, lu1, ln0, ln1 = L["link"]
    yu0, yu1, yn0, yn1 = L["gbay"]
    pu0_, pu1_, pn0_, pn1_ = L["portico"]
    ab0, ab1, abn0, abn1 = L["apron_bed"]
    gu0, gu1, gn0, gn1 = L["garage"]
    # ornamental tree in the front SW corner
    tree((0.4, 6.9, PADZ), 5.5, 2.2, "olive", "darkbark", stems=3, cards_per=26, crown_flat=0.55)
    # hydrangea beds along the white facades (mixed pink / white / lime): screen the bay window; box balls at the ends
    for x in np.arange(mu0 + 0.45, mu1 - 0.2, 0.7):
        nb = yn0 - 0.5 if yu0 - 0.3 < x < yu1 + 0.3 else mn0 - 0.55
        hydr(x, nb, PADZ + 0.45, rr(0.42, 0.52))
    for x in list(np.arange(lu0 + 0.45, pu0_ - 0.5, 0.7)) + list(np.arange(pu1_ + 0.6, lu1 - 0.2, 0.7)):
        hydr(x, ln0 - 0.55, PADZ + 0.42)
    for x in (mu0 + 0.2, mu1 - 0.15, lu1 - 0.25):
        bush((x, (mn0 if x < mu1 else ln0) - 1.0, PADZ + 0.3), 0.3, "hedge", 18)
    # bank behind the street wall: hydrangeas and box
    for x in np.arange(-0.1, ab0 - 0.4, 0.75):
        for nn in (2.75, 3.75):
            if s0 - 0.8 < x < s1 + 0.8:
                continue
            xx = x + (0.35 if nn > 3 else 0)
            if nn > 3 and rng.random() < 0.35:
                bush((xx, nn, zg(xx, nn) + 0.3), 0.32, "hedge", 16)
            else:
                hydr(xx, nn, zg(xx, nn) + 0.38, rr(0.34, 0.44))
    # the planted terraces either side of the driveway apron
    for nn in np.arange(2.0, abn1 - 0.3, 0.7):
        hydr(ab0 + 0.4, nn, BED - cr.FFL + 0.35, rr(0.32, 0.4))
    for nn in np.arange(1.3, gn0 - 0.2, 0.7):
        bush((gu1 + 0.75, nn, GZ + 0.75), rr(0.32, 0.4), "hedge" if int(nn * 10) % 3 == 0 else "hydrangea_pink", 16)
        bush((bnd_ne(nn) - 0.7, nn, GZ + 0.8), rr(0.4, 0.5), "shrub", 16)
    # verge garden: low grasses and box balls
    for _ in range(55):
        u = rr(-1.2, ab1 - 0.4)
        if s0 - 0.8 < u < s1 + 0.8:
            continue
        t = rr(0.2, 0.9)
        nn = arc_n(u, ARC["r"] - 2.3) * (1 - t) + 1.5 * t
        r = rr(0.25, 0.42)
        bush((u, nn, zg(u, nn) + r * 0.6), r, "shrub" if rng.random() < 0.6 else "hedge", 16)
    # privacy hedges along both side boundaries (clear of the outdoor room and the garage strip)
    for k in range(40):
        n = 5.5 + k * 0.7
        u = bnd_sw(n) + 0.75
        for zz in (0.6, 1.4, 2.1):
            bush((u, n, zg(u, n) + zz), rr(0.5, 0.62), "hedge", 14)
    for k in range(48):
        n = 10.8 + k * 0.7
        if 24.3 < n < 31.8 or n > 42.5:
            continue
        u = bnd_ne(n) - 0.75
        for zz in (0.6, 1.4, 2.1):
            bush((u, n, zg(u, n) + zz), rr(0.5, 0.62), "hedge", 14)
    # upper terrace: hydrangea border along the wall, low shrubs along the rear boundary, three small trees
    (ua, na), (ub, nb) = LAY["upper_wall"][0], LAY["upper_wall"][-1]
    for k in range(int(math.hypot(ub - ua, nb - na) / 0.8)):
        u = ua + (ub - ua) * (k + 0.5) / int(math.hypot(ub - ua, nb - na) / 0.8)
        if 13.0 < u < 16.4:
            continue
        hydr(u, upper_n(u) + 0.75, UPZ + 0.4)
    lot = LAY["lot_outline"]
    (ra, rna), (rb, rnb) = lot[1], lot[2]          # rear boundary (NE corner -> SW corner)
    for k in range(34):
        t = (k + 0.5) / 34
        u, n = ra + (rb - ra) * t, rna + (rnb - rna) * t - 0.85
        if not pip(u, n, [tuple(p) for p in PAD_OUTLINE[:-1]]):
            continue
        bush((u, n, UPZ + 0.35), rr(0.32, 0.42), "shrub" if k % 3 else "hydrangea_lime", 14)
    for u, nn in ((1.5, 33.9), (24.5, 40.4), (19.0, 39.6)):
        tree((u, nn, UPZ), rr(3.4, 4.0), rr(1.4, 1.7), "olive", "darkbark", stems=3, cards_per=20, crown_flat=0.6)
    # courtyard olives either side of the outdoor room
    tree((21.6, 32.6, PADZ), 3.8, 1.6, "olive", "darkbark", stems=3, cards_per=22, crown_flat=0.6)


GLB_TEX = {"clad": ("cladding.png", 0.25), "sage": ("sage.png", 0.25), "limestone": ("limestone.png", 0.5),
           "paving": ("paving.png", 0.25), "coping": ("paving.png", 0.25), "slats": ("slats.png", 0.25),
           "flagstone": ("stone.png", 0.6), "flagstone2": ("stone.png", 0.6)}
GLB_COL = {"roof": ("#303235", 0.85, 0.0), "awning": ("#2b2d30", 0.42, 0.55), "black": ("#141516", 0.4, 0.3),
           "brass": ("#b08d57", 0.3, 1.0), "fence_ne": ("#cdbf9f", 0.5, 0.3), "fence_sw": ("#7d866b", 0.5, 0.3), "pot": ("#9a978f", 0.8, 0), "glassdark": ("#0e1416", 0.1, 0),
           "timber": ("#8a5f3a", 0.6, 0), "deck": ("#8a6644", 0.55, 0), "door": ("#6b4a2c", 0.45, 0),
           "lawn": ("#5f7d38", 0.95, 0), "mulch": ("#4f3f31", 1.0, 0), "concrete": ("#bbb5aa", 0.9, 0),
           "kerb": ("#b6b0a5", 0.9, 0), "asphalt": ("#38393b", 0.9, 0), "aggregate": ("#a39b8e", 0.9, 0),
           "pooltile": ("#8fd0e3", 0.3, 0), "water": ("#3fa9cf", 0.05, 0), "bark": ("#a09684", 0.85, 0),
           "darkbark": ("#4d443b", 0.9, 0), "grass": ("#a8874a", 1.0, 0), "fire": ("#ff7a2a", 0.5, 0), "floor": ("#8a6646", 0.4, 0),
           "slab": ("#e8e2d6", 0.7, 0), "glass": ("#0e1416", 0.05, 0.0), "bulb": ("#fff2d8", 0.5, 0)}
INTERIOR = {"fabric", "linen", "walnut", "rug", "art", "lamp"}
LEAVES = ("euc", "olive", "shrub", "hedge", "hydrangea", "hydrangea_lime", "hydrangea_pink", "tussock")
EXTERIOR_LIGHTS = ("lantern", "path", "garage_in", "pool", "hall")


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


def fix_terrain_under_pool():
    """the shared terrain sinks a fixed pool rectangle; sink ours instead and restore the rest"""
    ob = bpy.data.objects["terrain"]
    pu0, pu1, pn0, pn1 = L["pool"]
    for v in ob.data.vertices:
        u, n, z = v.co
        if pu0 - 0.3 < u < pu1 + 0.3 and pn0 - 0.3 < n < pn1 + 0.3:
            v.co.z = PADZ - 1.8
        elif 3.7 < u < 14.3 and 27.2 < n < 31.3:
            v.co.z = zg(u, n)
    ob.data.update()


def golden_sky():
    """warm golden-hour gradient with soft lit clouds for camera rays (lighting still from the physical sky + sun)"""
    nt = bpy.context.scene.world.node_tree
    ramp = [n for n in nt.nodes if n.type == "VALTORGB"][0]
    els = sorted(ramp.color_ramp.elements, key=lambda e: e.position)
    els[0].color = (*srgb("#f3bf7e"), 1)
    els[1].position = 0.2
    els[1].color = (*srgb("#f2d2a8"), 1)
    els[2].position = 0.75
    els[2].color = (*srgb("#8fa9cc"), 1)
    for n_ in nt.nodes:                 # world Generated z is sin(elevation): spread the gradient over 0..30 degrees
        if n_.type == "MAP_RANGE":
            n_.inputs["From Min"].default_value = 0.0
            n_.inputs["From Max"].default_value = 0.5
    tc = [n for n in nt.nodes if n.type == "TEX_COORD"][0]
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (2.2, 2.2, 7.0)
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 3.0; nz.inputs["Detail"].default_value = 7
    nz.inputs["Roughness"].default_value = 0.6
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    cm = nt.nodes.new("ShaderNodeMapRange")
    cm.inputs["From Min"].default_value = 0.52; cm.inputs["From Max"].default_value = 0.72
    nt.links.new(nz.outputs["Fac"], cm.inputs["Value"])
    sep = [n for n in nt.nodes if n.type == "SEPXYZ"][0]
    hm_ = nt.nodes.new("ShaderNodeMapRange")
    hm_.inputs["From Min"].default_value = 0.03; hm_.inputs["From Max"].default_value = 0.15
    hm_.inputs["To Min"].default_value = 0.0; hm_.inputs["To Max"].default_value = 0.75
    nt.links.new(sep.outputs["Z"], hm_.inputs["Value"])
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
    nt.links.new(cm.outputs["Result"], mul.inputs[0]); nt.links.new(hm_.outputs["Result"], mul.inputs[1])
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
    nt.links.new(mul.outputs[0], mix.inputs["Factor"])
    nt.links.new(ramp.outputs["Color"], mix.inputs["A"])
    mix.inputs["B"].default_value = (*srgb("#ffd6a3"), 1)
    cam_bg = [n for n in nt.nodes if n.type == "BACKGROUND" and n.name != "Background"][0]
    nt.links.new(mix.outputs["Result"], cam_bg.inputs["Color"])
    bg = nt.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.55


VIEWS = {
    "street": dict(eye=(10.8, -16.0, 1.7), at=(11.6, 14.0, 3.9), lens=24),
    "courtyard": dict(eye=(11.5, 45.5, 7.0), at=(14.0, 25.5, 0.6), lens=21),
    "aerial": dict(eye=(46.0, -22.0, 40.0), at=(12.0, 22.0, 0.0), lens=28),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="street")
    ap.add_argument("--scale", type=int, default=50)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "designs" / "sage_stone_street.png"))
    ap.add_argument("--exposure", type=float, default=0.95)
    ap.add_argument("--export-glb", default="")
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    cr.make_textures()
    textures()
    MATS.update({
        "clad": img_mat("clad", "cladding.png", 0.25, 0.6, bump=0.25),
        "sage": img_mat("sage", "sage.png", 0.25, 0.65, bump=0.25),
        "limestone": img_mat("limestone", "limestone.png", 0.5, 0.88, bump=0.6),
        "paving": img_mat("paving", "paving.png", 0.25, 0.8, bump=0.1, tint=(1.04, 1.03, 1.0)),
        "coping": img_mat("coping", "paving.png", 0.25, 0.75, tint=(1.07, 1.06, 1.03)),
        "slats": img_mat("slats", "slats.png", 0.25, 0.6, bump=0.2, tint=(1.6, 1.4, 1.2)),
        "roof": shingle_roof("roof"),
        "awning": seam_roof("awning", "X"),
        "fence_ne": colorbond("fence_ne", "#cdbf9f"), "fence_sw": colorbond("fence_sw", "#7d866b"),
        "black": flat_mat("black", srgb("#18191a"), 0.4, 0.3),
        "brass": flat_mat("brass", srgb("#b8925a"), 0.28, 1.0),
        "pot": noisy_mat("pot", "#8d8a83", "#b3afa6", 6, 0.8, 0.15),
        "glass": thin_glass("glass"),
        "glassdark": flat_mat("glassdark", srgb("#1a2226"), 0.08),
        "slab": flat_mat("slab", srgb("#e8e2d6"), 0.7),
        "floor": flat_mat("floor", srgb("#9a7550"), 0.35),
        "fabric": flat_mat("fabric", srgb("#d9cfbf"), 0.95),
        "linen": flat_mat("linen", srgb("#f1ece3"), 0.95),
        "walnut": flat_mat("walnut", srgb("#4a3424"), 0.45),
        "rug": flat_mat("rug", srgb("#b9ab94"), 0.95),
        "lamp": flat_mat("lamp", (1, 1, 1), 0.5, emit=WARM, strength=14),
        "bulb": flat_mat("bulb", (1, 1, 1), 0.5, emit=WARM, strength=25),
        "lawn": noisy_mat("lawn", "#4d6f2c", "#7d9a46", 30, 0.95, 0.3),
        "mulch": noisy_mat("mulch", "#3d2f24", "#6b5644", 40, 1.0, 0.4),
        "concrete": noisy_mat("concrete", "#aaa499", "#c9c3b8", 8, 0.9, 0.05),
        "kerb": flat_mat("kerb", srgb("#b6b0a5"), 0.9),
        "asphalt": noisy_mat("asphalt", "#2a2b2d", "#4a4a4b", 90, 0.9, 0.3),
        "aggregate": noisy_mat("aggregate", "#8f877a", "#c2b9aa", 70, 0.9, 0.25),
        "timber": noisy_mat("timber", "#7a5130", "#9c6c43", 3, 0.6, 0.1),
        "deck": noisy_mat("deck", "#7a5636", "#9a7350", 2, 0.55, 0.1),
        "door": noisy_mat("door", "#6a4426", "#8a5c36", 4, 0.45, 0.05),
        "pooltile": flat_mat("pooltile", srgb("#8fd0e3"), 0.3),
        "bark": noisy_mat("bark", "#8a8070", "#b0a796", 6, 0.85, 0.2),
        "darkbark": flat_mat("darkbark", srgb("#4d443b"), 0.9),
        "euc": leaf_mat("euc"), "olive": leaf_mat("olive"), "shrub": leaf_mat("shrub"), "hedge": leaf_mat("hedge"),
        "hydrangea": leaf_mat("hydrangea"), "hydrangea_lime": leaf_mat("hydrangea_lime"), "hydrangea_pink": leaf_mat("hydrangea_pink"),
        "flagstone": img_mat("flagstone", "stone.png", 0.6, 0.85, bump=0.4, tint=(0.92, 0.93, 0.94)),
        "flagstone2": img_mat("flagstone2", "stone.png", 0.6, 0.85, bump=0.4, tint=(0.95, 0.95, 0.95)),
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
    fix_terrain_under_pool()
    build_house()
    build_garden()
    cr.flush()
    if a.export_glb:
        export_glb(a.export_glb)
        print("exported", a.export_glb)
        return
    cr.VIEWS.update(VIEWS)
    sun_az = {"street": 232.0, "courtyard": 290.0}.get(a.view, 260.0)       # low golden-hour sun from the W/SW
    cr.setup_world_and_camera(a.view, 10.0, sun_az, a.exposure)
    golden_sky()

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
