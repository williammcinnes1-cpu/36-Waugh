"""Modern Mediterranean (rough concept), built on the farmhouse site fit. Derived from farmhouse_cycles.py:
only build_house() differs.
Cycles render of option B, the modern farmhouse (privacy to the street, living to a north courtyard).

    python farmhouse_cycles.py --view street|courtyard|aerial --scale 50 --samples 48 --out x.png [--export-mesh]

Reuses terrain, street, vegetation helpers and materials from cycles_render.py. Run farmhouse_site.py ground
first so render_scene.json carries this design's earthworks. With --export-mesh it also writes
data/3d/farmhouse_mesh.json (house triangles + colours) for farmhouse_site.py finish.
"""
import argparse
import json
import os
import math
import random
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFilter

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cycles_render as cr
from cycles_render import (ARC, GZ, PADZ, B, MATS, TEX, arc_n, box, card, cyl, drape, faces, poly, prism, ring, slab,
                           zg, light, bush, tree, WARM, srgb, img_mat, flat_mat, noisy_mat, seam_roof, thin_glass,
                           leaf_mat, terrain_mat)

ROOT = Path(__file__).resolve().parent.parent.parent
D3 = ROOT / "data" / "3d"
LAY = json.loads((D3 / "house_massing.json").read_text())
L = LAY.get("layout", {})          # empty when another design's massing is current (export_glb still works)
FRONT = LAY.get("front", {})
PAD_OUTLINE = LAY.get("pad_outline", [])
rng = random.Random(4)
rr = rng.uniform

UF, UE, PITCH_UP = 3.0, 4.72, math.radians(36)  # view storey: floor, eave, pitch (ridge ≤ 9 m above natural ground, DCP 4.5a)
EAVE_REAR, PITCH_REAR = 2.7, math.radians(19)   # long west wing: its ridge stays below the rear lean-to's top
NF, FE, PF = 6.7, 2.65, math.radians(12)        # front lean-to: front edge n, underside height there, pitch
ZH, PL = 4.2, math.radians(14)                  # rear lean-to: underside height at the view-storey wall, pitch
ROOF = {}
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
    # lime-green variant (Annabelle / Limelight blooms)
    arr = np.array(im).astype(np.int16)
    bloom = (arr[..., 0] > 200) & (arr[..., 3] > 0)
    arr[bloom, 0] -= 34; arr[bloom, 2] -= 80
    Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA").save(TEX / "leaf_hydrangea_lime.png")
    # lavender: grey-green needles with purple spikes
    im = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for _ in range(420):
        cx, cy = R.uniform(30, 482), R.uniform(30, 482)
        if math.hypot(cx - 256, cy - 256) > 230:
            continue
        a, Lr, l = R.uniform(0, 6.28), R.uniform(10, 22), R.uniform(-15, 15)
        d.line([cx, cy, cx + math.cos(a) * Lr, cy + math.sin(a) * Lr], fill=(int(120 + l), int(134 + l), int(112 + l), 255), width=3)
    for _ in range(70):
        cx, cy, l = R.uniform(60, 452), R.uniform(40, 330), R.uniform(-20, 20)
        for k in range(7):
            d.ellipse([cx - 5, cy + k * 8 - 5, cx + 5, cy + k * 8 + 5], fill=(int(128 + l), int(96 + l), int(178 + l), 255))
    im.save(TEX / "leaf_lavender.png")
    # ornamental grass: blades fanning up from the base with buff plumes (card UV: v=0 at the base)
    im = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for _ in range(150):
        x0, lean, h = 256 + R.uniform(-40, 40), R.uniform(-1, 1), R.uniform(300, 500)
        pts = [(x0 + lean * 230 * (t / 8) ** 1.6, 511 - h * t / 8) for t in range(9)]
        l = R.uniform(-20, 20)
        col = (int(150 + l), int(152 + l), int(96 + l), 255) if R.random() < 0.35 else (int(204 + l), int(178 + l), int(122 + l), 255)
        d.line(pts, fill=col, width=3)
        if R.random() < 0.5:
            tx, ty = pts[-1]
            d.ellipse([tx - 9, ty - 4, tx + 9, ty + 34], fill=(int(232 + l / 2), int(214 + l / 2), int(170 + l / 2), 255))
    im.save(TEX / "leaf_ograss.png")
    # bottlebrush: narrow dark leaves with crimson brushes
    im = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for _ in range(260):
        cx, cy = R.uniform(30, 482), R.uniform(30, 482)
        if math.hypot(cx - 256, cy - 256) > 225:
            continue
        a, Lr, l = R.uniform(0, 6.28), R.uniform(14, 26), R.uniform(-12, 12)
        d.line([cx, cy, cx + math.cos(a) * Lr, cy + math.sin(a) * Lr], fill=(int(58 + l), int(82 + l), int(46 + l), 255), width=5)
    for _ in range(26):
        cx, cy, l = R.uniform(70, 440), R.uniform(70, 440), R.uniform(-20, 20)
        if math.hypot(cx - 256, cy - 256) > 200:
            continue
        for k in range(30):
            y = cy + R.uniform(-22, 22)
            d.line([cx - 11, y + R.uniform(-3, 3), cx + 11, y + R.uniform(-3, 3)], fill=(int(196 + l), int(30 + l / 2), int(38 + l / 2), 255), width=2)
    im.save(TEX / "leaf_bottlebrush.png")


def fence_mat(name, hexc):
    """Colorbond corrugated panel (colours from the site photos): fine vertical corrugations"""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*srgb(hexc), 1)
    b.inputs["Metallic"].default_value = 0.3
    b.inputs["Roughness"].default_value = 0.5
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    ad = nt.nodes.new("ShaderNodeMath"); ad.operation = "ADD"
    nt.links.new(sep.outputs["X"], ad.inputs[0]); nt.links.new(sep.outputs["Y"], ad.inputs[1])
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = 2 * math.pi / 0.076
    nt.links.new(ad.outputs[0], mul.inputs[0])
    sn = nt.nodes.new("ShaderNodeMath"); sn.operation = "SINE"
    nt.links.new(mul.outputs[0], sn.inputs[0])
    bn = nt.nodes.new("ShaderNodeBump"); bn.inputs["Strength"].default_value = 0.45; bn.inputs["Distance"].default_value = 0.01
    nt.links.new(sn.outputs[0], bn.inputs["Height"])
    nt.links.new(bn.outputs["Normal"], b.inputs["Normal"])
    return m


def tile_mat(name, c1, c2):
    """roof tiles: horizontal courses (object z) with per-tile colour noise"""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    dv = nt.nodes.new("ShaderNodeMath"); dv.operation = "DIVIDE"; dv.inputs[1].default_value = 0.135
    nt.links.new(sep.outputs["Z"], dv.inputs[0])
    fr = nt.nodes.new("ShaderNodeMath"); fr.operation = "FRACT"
    nt.links.new(dv.outputs[0], fr.inputs[0])
    nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 5.0; nz.inputs["Detail"].default_value = 3
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*srgb(c1), 1); ramp.color_ramp.elements[1].color = (*srgb(c2), 1)
    nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.7
    bn = nt.nodes.new("ShaderNodeBump"); bn.inputs["Strength"].default_value = 0.5; bn.inputs["Distance"].default_value = 0.03
    nt.links.new(fr.outputs[0], bn.inputs["Height"])
    nt.links.new(bn.outputs["Normal"], b.inputs["Normal"])
    return m


# ------------------------------------------------------------------ wall helpers with rectangular openings
def solid_spans(openings, a, b, z0, z1):
    """solid z-intervals of a wall strip a..b between z0 and z1, leaving every opening that covers the strip"""
    hits = sorted((max(o[2], z0), min(o[3], z1)) for o in openings if o[0] <= a + 1e-9 and b <= o[1] + 1e-9)
    out, z = [], z0
    for zb, zt in hits:
        if zb > z + 1e-6:
            out.append((z, zb))
        z = max(z, zt)
    if z1 > z + 1e-6:
        out.append((z, z1))
    return out


def wall_u(u0, u1, nf, nb, z0, z1, openings, mat):
    """wall spanning u0..u1 between n=nf and n=nb; openings: (a, b, zb, zt) along u"""
    cuts = sorted({u0, u1, *[x for o in openings for x in o[:2]]})
    for a, b in zip(cuts, cuts[1:]):
        for za, zb in solid_spans(openings, a, b, z0, z1):
            box(a, b, nf, nb, za, zb, mat)


def wall_n(n0, n1, uf, ub, z0, z1, openings, mat):
    """wall spanning n0..n1 between u=uf and u=ub; openings: (a, b, zb, zt) along n"""
    cuts = sorted({n0, n1, *[x for o in openings for x in o[:2]]})
    for a, b in zip(cuts, cuts[1:]):
        for za, zb in solid_spans(openings, a, b, z0, z1):
            box(uf, ub, a, b, za, zb, mat)


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


def plinth_u(u0, u1, n_face, outward, z1=0.9, gaps=()):
    """limestone base course on a wall facing -n (outward=-1) or +n (+1), skipping door/window gaps"""
    n0, n1 = sorted((n_face, n_face + outward * 0.07))
    cuts = sorted({u0, u1, *[x for g in gaps for x in g]})
    for a, b in zip(cuts, cuts[1:]):
        if not any(g[0] <= a and b <= g[1] for g in gaps):
            box(a, b, n0, n1, -0.4, z1, "limestone")
            box(a - 0.02, b + 0.02, min(n0, n1) - 0.03 * (outward < 0), max(n0, n1) + 0.03 * (outward > 0), z1, z1 + 0.06, "limestone")


def plinth_n(n0_, n1_, u_face, outward, z1=0.9, gaps=()):
    u0, u1 = sorted((u_face, u_face + outward * 0.07))
    cuts = sorted({n0_, n1_, *[x for g in gaps for x in g]})
    for a, b in zip(cuts, cuts[1:]):
        if not any(g[0] <= a and b <= g[1] for g in gaps):
            box(u0, u1, a, b, -0.4, z1, "limestone")
            box(u0 - 0.03 * (outward < 0), u1 + 0.03 * (outward > 0), a - 0.02, b + 0.02, z1, z1 + 0.06, "limestone")


def gable_u(u0, u1, nf, nb, ze, pitch, mat, opening=None):
    """triangular gable wall in the u-z plane at n between nf..nb, ridge over the middle"""
    um = (u0 + u1) / 2
    tp = math.tan(pitch)
    zr = ze + (u1 - u0) / 2 * tp
    if opening is None:
        prism([(u0, ze), (u1, ze), (um, zr)], nf, nb, mat)
    elif opening[2] <= ze + 1e-6:          # opening rising from the eave line: build the triangle in pieces
        a, b, _, zt = opening
        rake = lambda u: ze + min(u - u0, u1 - u) * tp
        prism([(u0, ze), (a, ze), (a, rake(a))], nf, nb, mat)
        prism([(b, ze), (u1, ze), (b, rake(b))], nf, nb, mat)
        prism([(a, zt), (b, zt), (b, rake(b)), (um, zr), (a, rake(a))], nf, nb, mat)
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


def wall_n_poly(nz, uf, ub, mat):
    """wall in the n-z plane (polygon of (n, z)) extruded from u=uf to u=ub (sloping wall tops under lean-tos)"""
    k = len(nz)
    v = [(uf, n, z) for n, z in nz] + [(ub, n, z) for n, z in nz]
    fcs = [list(reversed(range(k))), [k + i for i in range(k)]]
    fcs += [(i, (i + 1) % k, k + (i + 1) % k, k + i) for i in range(k)]
    faces(mat, v, fcs)


def gable_n_open(n0, n1, uf, ub, ze, pitch, op):
    """triangular gable wall in the n-z plane (ridge over the middle) with a rectangular opening (a, b, z_top)
    rising from the eave line"""
    a, b, zt = op
    tp = math.tan(pitch)
    nm = (n0 + n1) / 2
    rake = lambda n: ze + (min(n - n0, n1 - n)) * tp
    wall_n_poly([(n0, ze), (a, ze), (a, rake(a))], uf, ub, "clad")
    wall_n_poly([(b, ze), (n1, ze), (b, rake(b))], uf, ub, "clad")
    wall_n_poly([(a, zt), (b, zt), (b, rake(b)), (nm, ze + (nm - n0) * tp), (a, rake(a))], uf, ub, "clad")


def ear_clip(pts):
    """triangulate a simple plan polygon (u, n) -> index triples (counter-clockwise input)"""
    idx = list(range(len(pts)))
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= -1e-9 and cross(b, c, p) >= -1e-9 and cross(c, a, p) >= -1e-9
    guard = 0
    while len(idx) > 3 and guard < 1000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross(a, b, c) <= 1e-9:
                continue
            if any(inside(pts[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def roof_poly(pts, zfun, mat, t=0.1):
    """roof sheet over a simple plan polygon (u, n): underside at zfun(u, n), top t above. Concave outlines are
    ear-clipped so every exported face is a proper triangle."""
    area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    if area < 0:
        pts = list(reversed(pts))
    k = len(pts)
    v = [(u, n, zfun(u, n)) for u, n in pts] + [(u, n, zfun(u, n) + t) for u, n in pts]
    tris = ear_clip(pts)
    fcs = [(k + a, k + b, k + c) for a, b, c in tris] + [(c, b, a) for a, b, c in tris]
    fcs += [(i, (i + 1) % k, k + (i + 1) % k, k + i) for i in range(k)]
    faces(mat, v, fcs)



def gutter(p0, p1):
    """quad-profile Colorbond gutter (drawn as a tube) with a fascia behind it"""
    cyl(p0, p1, 0.07, 0.07, "gutter", sides=10)


def downpipe(u, n, z_top, z_bot, du=0.0, dn=0.0):
    """round downpipe from the gutter outlet, offset back to the wall, down to the ground"""
    cyl((u, n, z_top), (u + du, n + dn, z_top - 0.3), 0.045, 0.045, "gutter", sides=8)
    cyl((u + du, n + dn, z_top - 0.3), (u + du, n + dn, z_bot), 0.045, 0.045, "gutter", sides=8)
    for zz in (z_bot + 0.6, (z_top + z_bot) / 2, z_top - 0.5):  # brackets
        cyl((u + du, n + dn, zz), (u + du, n + dn, zz + 0.04), 0.055, 0.055, "gutter", sides=8)


def eave_trim(u_a, u_b, n, z, outward, t=0.1):
    """fascia board and gutter along an eave running in u at n (outward = +-1 in n), roof underside z"""
    n0, n1 = sorted((n, n + outward * 0.04))
    box(u_a, u_b, n0, n1, z - 0.12, z + t + 0.02, "black")
    gutter((u_a, n + outward * 0.11, z - 0.03), (u_b, n + outward * 0.11, z - 0.03))


def eave_trim_n(n_a, n_b, u, z, outward, t=0.1):
    u0, u1 = sorted((u, u + outward * 0.04))
    box(u0, u1, n_a, n_b, z - 0.12, z + t + 0.02, "black")
    gutter((u + outward * 0.11, n_a, z - 0.03), (u + outward * 0.11, n_b, z - 0.03))


def lean_roof(pts, zfun_n, mat, eave=None, barges=(), barge_n=None, soffit=True, t=0.1):
    """mono-pitch roof sheet (slope along n) over a plan polygon, with fascia + gutter on the low eave, barge boards
    on the sloping ends and a painted lining underneath"""
    roof_poly(pts, lambda u, n: zfun_n(n), mat, t)
    if soffit:
        roof_poly(pts, lambda u, n: zfun_n(n) - 0.03, "soffit", 0.02)
    if eave:
        _, ne, ua, ub = eave
        outward = 1 if zfun_n(ne + 0.1) < zfun_n(ne) else -1
        eave_trim(ua, ub, ne, zfun_n(ne), outward, t)
    ns = [n for _, n in pts]
    na, nb = barge_n if barge_n else (min(ns), max(ns))
    for u in barges:
        slab((u, na, zfun_n(na) - 0.12), (u, nb, zfun_n(nb) - 0.12), (u, nb, zfun_n(nb) + t + 0.03), (u, na, zfun_n(na) + t + 0.03), 0.05, "black")


def soffit_strip(pts_u, n_wall, n_eave, z):
    """flat painted soffit lining under an eave running in u (from the wall line out to the fascia)"""
    a, b = pts_u
    n0, n1 = sorted((n_wall, n_eave))
    box(a, b, n0, n1, z - 0.04, z - 0.02, "soffit")


def soffit_strip_n(pts_n, u_wall, u_eave, z):
    a, b = pts_n
    u0, u1 = sorted((u_wall, u_eave))
    box(u0, u1, a, b, z - 0.04, z - 0.02, "soffit")


def main_roof(u0, u1, n0, n1, ze, pitch, front=(), rear=(), o=0.45, oc=0.3, cap_skip=()):
    """the view storey's gable roof (ridge along u) with equal-pitch cross gables (c0, c1, cn) on its street (front)
    and courtyard (rear) faces. Each main plane is cut along the valley lines, so every edge is a single deliberate
    line: barge boards on the gable ends, fascia + gutter on every eave, ridge caps, painted soffits."""
    tp = math.tan(pitch)
    nm = (n0 + n1) / 2
    zr = ze + (n1 - n0) / 2 * tp
    dz = o * tp
    for side, cross in ((-1, front), (1, rear)):
        R = (lambda n: n) if side < 0 else (lambda n: n0 + n1 - n)      # local (front-like) n -> world n
        cr_ = sorted([(c0, c1, R(cn)) for c0, c1, cn in cross])
        poly_ = [(u0 - o, n0 - o)]
        for c0, c1, _ in cr_:
            cm = (c0 + c1) / 2
            poly_ += [(c0 - o, n0 - o), (cm, n0 + (cm - c0)), (c1 + o, n0 - o)]
        poly_ += [(u1 + o, n0 - o), (u1 + o, nm), (u0 - o, nm)]
        roof_poly([(u, R(n)) for u, n in poly_], lambda u, n, R=R: ze + (R(n) - n0) * tp, "roof_x")
        for u in (u0 - o, u1 + o):  # barge boards on the two main gable ends
            slab((u, R(n0 - o), ze - dz - 0.12), (u, R(nm), zr - 0.07), (u, R(nm), zr + 0.2), (u, R(n0 - o), ze - dz + 0.15), 0.05, "black")
        xs = [u0 - o] + [x for c0, c1, _ in cr_ for x in (c0 - o, c1 + o)] + [u1 + o]
        for a, b in zip(xs[::2], xs[1::2]):
            eave_trim(a, b, R(n0 - o), ze - dz, side)
            soffit_strip((a, b), R(n0), R(n0 - o), ze - dz)
        for c0, c1, cn in cr_:
            cm = (c0 + c1) / 2
            zrc = ze + (cm - c0) * tp
            nf, nv = cn - oc, n0 + (cm - c0)
            roof_poly([(c0 - o, R(nf)), (cm, R(nf)), (cm, R(nv)), (c0 - o, R(n0 - o))], lambda u, n, c0=c0: ze + (u - c0) * tp, "roof_y")
            roof_poly([(cm, R(nf)), (c1 + o, R(nf)), (c1 + o, R(n0 - o)), (cm, R(nv))], lambda u, n, c1=c1: ze + (c1 - u) * tp, "roof_y")
            slab((c0 - o, R(nf), ze - dz - 0.12), (cm, R(nf), zrc - 0.07), (cm, R(nf), zrc + 0.2), (c0 - o, R(nf), ze - dz + 0.15), 0.05, "black")
            slab((cm, R(nf), zrc - 0.07), (c1 + o, R(nf), ze - dz - 0.12), (c1 + o, R(nf), ze - dz + 0.15), (cm, R(nf), zrc + 0.2), 0.05, "black")
            na, nb = sorted((R(nf), R(nv)))
            box(cm - 0.09, cm + 0.09, na, nb, zrc + 0.08, zrc + 0.2, "black")
            na, nb = sorted((R(nf), R(n0 - o)))
            eave_trim_n(na, nb, c0 - o, ze - dz, -1)
            eave_trim_n(na, nb, c1 + o, ze - dz, 1)
            soffit_strip_n((na, nb), c0, c0 - o, ze - dz)
            soffit_strip_n((na, nb), c1, c1 + o, ze - dz)
    box(u0 - o, u1 + o, nm - 0.09, nm + 0.09, zr + 0.08, zr + 0.2, "black")   # ridge cap
    return zr


def rear_wing_roof(u0, u1, ns, n1, ze, pitch, zh, tl, o=0.45):
    """the long west wing's gable roof (ridge along n). Its ridge sits below the rear lean-to's top, so towards the
    house the two wing planes die into the lean-to along two proper valleys that meet the ridge at n_r; the lean-to
    eave and the wing's side eaves are at the same height and join at the valley feet."""
    tp = math.tan(pitch)
    um = (u0 + u1) / 2
    zr = ze + (u1 - u0) / 2 * tp
    dz = o * tp
    n_e = ns + (zh - (ze - dz)) / tl          # valley foot (lean-to eave line)
    n_r = ns + (zh - zr) / tl                 # valley head on the ridge
    roof_poly([(u0 - o, n1 + o), (u0 - o, n_e), (um, n_r), (um, n1 + o)], lambda u, n: ze + (u - u0) * tp, "roof_y")
    roof_poly([(um, n1 + o), (um, n_r), (u1 + o, n_e), (u1 + o, n1 + o)], lambda u, n: ze + (u1 - u) * tp, "roof_y")
    n = n1 + o
    slab((u0 - o, n, ze - dz - 0.12), (um, n, zr - 0.07), (um, n, zr + 0.2), (u0 - o, n, ze - dz + 0.15), 0.05, "black")
    slab((um, n, zr - 0.07), (u1 + o, n, ze - dz - 0.12), (u1 + o, n, ze - dz + 0.15), (um, n, zr + 0.2), 0.05, "black")
    box(um - 0.09, um + 0.09, n_r + 0.25, n1 + o, zr + 0.08, zr + 0.2, "black")
    eave_trim_n(n_e, n1 + o, u0 - o, ze - dz, -1)
    eave_trim_n(n_e, n1 + o, u1 + o, ze - dz, 1)
    soffit_strip_n((n_e, n1 + o), u0, u0 - o, ze - dz)
    soffit_strip_n((n_e, n1 + o), u1, u1 + o, ze - dz)
    return zr, n_e, n_r



# ------------------------------------------------------------------ site context: lot, terrace, neighbours
LOT_Q = [(0.0, 0.0), (22.9, 0.0), (29.8, 44.5), (-5.1, 32.5)]
LOT625 = [(22.9, 0.0), (40.2, -4.8), (61.5, 30.5), (29.8, 44.5)]      # NE neighbour (cadastre, street frame)
LOT623 = [(0.0, 0.0), (-5.1, 32.5), (-40.0, 20.5), (-24.5, -8.3)]     # SW neighbour
DT = L.get("terrace_z", 1.0)                                           # raised rear garden above the pad
TERR = [tuple(p) for p in L.get("terrace_outline", [])]
NEIGHBOURS = [   # simple massing read off the 2014 aerial: 625 white render + terracotta hip, 623 brick + tile hip
    dict(q=[(26.6, 16.6), (38.5, 10.2), (43.0, 21.0), (28.1, 28.6)], wall="nrender", roof="tile_terra", h=2.7, pitch=24),
    dict(q=[(-29.0, 10.0), (-12.5, 1.5), (-8.0, 13.0), (-24.5, 21.5)], wall="nbrick", roof="tile_grey", h=2.7, pitch=22),
    dict(q=[(1.0, -36.0), (19.0, -36.0), (19.0, -25.0), (1.0, -25.0)], wall="nrender", roof="tile_grey", h=2.9, pitch=22),
]


def pip(u, n, poly_):
    inside = False
    for (a, b), (c, d) in zip(poly_, poly_[1:] + poly_[:1]):
        if (b > n) != (d > n) and u < a + (n - b) * (c - a) / (d - b):
            inside = not inside
    return inside


def lot_inside(u, n):
    return pip(u, n, LOT_Q) or (0 < u < 22.9 and -1 < n < 0.9)


def terrace_inside(u, n):
    return bool(TERR) and pip(u, n, TERR)


def flat_poly(pts, z, mat):
    """flat horizontal polygon (any simple outline) at height z"""
    area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    if area < 0:
        pts = list(reversed(pts))
    faces(mat, [(u, n, z) for u, n in pts], ear_clip(pts))


def ground_cover(poly_, mat, step=1.5, lift=0.04, skip=None):
    """draped quads over the terrain inside a plan polygon (neighbour lawns, beds)"""
    us, ns = [p[0] for p in poly_], [p[1] for p in poly_]
    u = min(us)
    while u < max(us):
        n = min(ns)
        while n < max(ns):
            c = (u + step / 2, n + step / 2)
            if pip(*c, poly_) and not (skip and skip(*c)):
                q = [(u, n), (u + step, n), (u + step, n + step), (u, n + step)]
                faces(mat, [(a, b, zg(a, b) + lift) for a, b in q], [(0, 1, 2, 3)])
            n += step
        u += step


def hip_house(q, wall, roof, h, pitch, k):
    """single-storey neighbour house: walls, tiled hip roof with eaves, fascia + gutter, a few lit windows"""
    P = [Vector((u, n, 0)) for u, n in q]
    if (P[1] - P[0]).length < (P[2] - P[1]).length:
        P = P[1:] + P[:1]
    base = min(zg(p.x, p.y) for p in P) + 0.15
    faces(wall, [(p.x, p.y, base - 1.5) for p in P] + [(p.x, p.y, base + h) for p in P],
          [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    cen = sum(P, Vector()) / 4
    dl = (P[1] - P[0]).normalized()
    ds = (P[3] - P[0]).normalized()
    hl, hs = (P[1] - P[0]).length / 2 + 0.6, (P[3] - P[0]).length / 2 + 0.6
    zr = base + h + hs * math.tan(math.radians(pitch)) - 0.15
    ze = base + h - 0.15
    E = [cen - dl * hl - ds * hs, cen + dl * hl - ds * hs, cen + dl * hl + ds * hs, cen - dl * hl + ds * hs]
    R0, R1 = cen - dl * (hl - hs), cen + dl * (hl - hs)
    v = [(e.x, e.y, ze) for e in E] + [(R0.x, R0.y, zr), (R1.x, R1.y, zr)]
    v += [(x, y, z + 0.08) for x, y, z in v]
    faces(roof, v, [(0, 1, 5, 4), (1, 2, 5), (2, 3, 4, 5), (3, 0, 4),
                    (6, 10, 11, 7), (7, 11, 8), (8, 11, 10, 9), (9, 10, 6)])
    for i in range(4):                                     # fascia + gutter, painted soffit
        a, b = E[i], E[(i + 1) % 4]
        cyl((a.x, a.y, ze - 0.05), (b.x, b.y, ze - 0.05), 0.08, 0.08, "gutter", 8)
        a2, b2 = P[i] + (P[i] - cen).normalized() * 0.05, P[(i + 1) % 4] + (P[(i + 1) % 4] - cen).normalized() * 0.05
        faces("soffit", [(a.x, a.y, ze - 0.02), (b.x, b.y, ze - 0.02), (b2.x, b2.y, ze - 0.02), (a2.x, a2.y, ze - 0.02)], [(0, 1, 2, 3)])
    for i in range(4):                                     # windows along each wall
        a, b = P[i], P[(i + 1) % 4]
        L_ = (b - a).length
        d = (b - a).normalized()
        out = (a + b) / 2 - cen
        nrm = Vector((d.y, -d.x, 0))
        if nrm.dot(out) < 0:
            nrm = -nrm
        m = int(L_ / 3.6)
        for j in range(m):
            c = a + d * (L_ * (j + 0.5) / m)
            w = 0.75 if j % 3 else 1.0
            p0, p1 = c - d * w + nrm * 0.03, c + d * w + nrm * 0.03
            lit = rng.random() < 0.45
            faces("nglass_lit" if lit else "nglass", [(p0.x, p0.y, base + 0.9), (p1.x, p1.y, base + 0.9), (p1.x, p1.y, base + 2.2), (p0.x, p0.y, base + 2.2)], [(0, 1, 2, 3)])
            for zz in (0.85, 2.2):
                q0, q1 = p0 + nrm * 0.02 - d * 0.05, p1 + nrm * 0.02 + d * 0.05
                faces("nframe", [(q0.x, q0.y, base + zz), (q1.x, q1.y, base + zz), (q1.x, q1.y, base + zz + 0.06), (q0.x, q0.y, base + zz + 0.06)], [(0, 1, 2, 3)])
    return base


def build_context():
    """the two neighbour houses (and one across the street) on their lots, with lawns, beds and fences"""
    houses = []
    for k, nb in enumerate(NEIGHBOURS):
        hip_house(nb["q"], nb["wall"], nb["roof"], nb["h"], nb["pitch"], k)
        houses.append(nb["q"])
    in_house = lambda u, n: any(pip(u, n, q) for q in houses)
    # neighbours' lawns and garden beds (not bare paddock)
    ground_cover(LOT625, "lawn_dry", 1.5, 0.05, skip=lambda u, n: in_house(u, n) or n > 40 or (u < 31 and n < 3))
    ground_cover(LOT623, "lawn_dry", 1.5, 0.05, skip=lambda u, n: in_house(u, n) or n > 28)
    ground_cover([(-2.0, -25.0), (24.0, -25.0), (24.0, -45.0), (-2.0, -45.0)], "lawn_dry", 1.5, 0.05, skip=in_house)
    for q in houses:      # mulch beds and shrubs along the house fronts
        a, b = Vector(q[0] + (0,)), Vector(q[1] + (0,))
        d = (b - a)
        for t in np.linspace(0.05, 0.95, 9):
            p = a + d * t - Vector((d.y, -d.x, 0)).normalized() * 1.1
            if not lot_inside(p.x, p.y):
                bush((p.x, p.y, zg(p.x, p.y) + 0.45), rr(0.45, 0.7), "shrub" if rng.random() < 0.5 else "hedge", 14)
    # the big red bottlebrush behind the SW fence, gums and ornamental trees in the neighbours' yards
    tree((-3.2, 11.5, zg(-3.2, 11.5)), 6.0, 2.8, "bottlebrush", "darkbark", stems=3, cards_per=26, crown_flat=0.7)
    tree((-4.6, 21.0, zg(-4.6, 21.0)), 5.0, 2.4, "bottlebrush", "darkbark", stems=2, cards_per=20, crown_flat=0.7)
    for u, n, h, r, leaf in ((33.0, 6.5, 7.5, 3.2, "hedge"), (45.0, 27.0, 9.0, 3.5, "euc"), (31.5, 36.0, 8.0, 3.0, "euc"),
                             (-16.0, 25.0, 9.0, 3.5, "euc"), (-30.0, 0.5, 7.0, 3.0, "hedge"), (-19.0, 4.0, 5.5, 2.4, "olive"),
                             (5.0, -30.0, 6.0, 2.8, "hedge"), (22.0, -29.0, 7.0, 3.0, "olive"), (54.0, 16.0, 9.0, 3.4, "hedge")):
        tree((u, n, zg(u, n)), h, r, leaf, "darkbark" if leaf != "euc" else "bark", stems=2 if leaf != "euc" else 1,
             cards_per=24 if leaf != "euc" else 14, crown_flat=0.6)
    # street trees: a Chinese elm on the verge at the NE end, one across the road
    R = ARC["r"]
    for u, rr_ in ((26.0, R - 1.0),):
        n = arc_n(u, rr_)
        tree((u, n, zg(u, n)), 8.5, 3.6, "hedge", "darkbark", stems=1, cards_per=34, crown_flat=0.55)
    # 623's front: low hedge and letterbox; 625's front lawn edge
    for k in range(14):
        u = -22.0 + k * 1.4
        n = arc_n(u, R) + 1.2
        bush((u, n, zg(u, n) + 0.4), 0.5, "hedge", 12)


def build_terrace():
    """small raised rear garden in the NE rear corner (≈1 m above the courtyard): low stone retaining wall with
    steps down to the lawn, a timber pergola over a built-in seat and table, lavender, grasses and an olive"""
    if not TERR:
        return
    zt = PADZ + DT
    flat_poly(TERR, zt + 0.02, "lawn")
    pu0, pn0, pu1, pn1 = 16.1, 36.75, 19.7, 38.95                        # pergola at the west end, seen from the courtyard
    pv = [(pu0 - 0.3, pn0 - 0.1), (pu1 + 0.3, pn0 - 0.1), (pu1 + 0.3, pn1 + 0.1), (pu0 - 0.3, pn1 + 0.1)]   # paved floor
    flat_poly(pv, zt + 0.05, "flagstone")
    # retaining wall along the terrace's front (n = 37.5) and west (u = 16) edges, with a coping, gap for the steps
    n_f = min(p[1] for p in TERR)
    u_w = min(p[0] for p in TERR)
    s0, s1 = 21.0, 23.0
    u_e = max(p[0] for p in TERR if abs(p[1] - n_f) < 0.3)
    for a, b in ((u_w, s0), (s1, u_e)):      # retaining wall rising 0.45 m above the terrace as a low seat wall
        box(a, b, n_f - 0.3, n_f, PADZ - 0.3, zt + 0.45, "limestone")
        box(a - 0.03, b + 0.03, n_f - 0.34, n_f + 0.04, zt + 0.45, zt + 0.52, "coping")
    n_rw = max(p[1] for p in TERR if abs(p[0] - u_w) < 0.3)
    box(u_w - 0.3, u_w, n_f - 0.3, n_rw, PADZ - 0.3, zt + 0.45, "limestone")
    box(u_w - 0.34, u_w + 0.04, n_f - 0.34, n_rw, zt + 0.45, zt + 0.52, "coping")
    nst = 5
    for k in range(nst):                                      # stone steps from the lawn up to the terrace
        z = PADZ + DT * (k + 1) / nst
        box(s0, s1, n_f - 0.3 - (nst - 1 - k) * 0.32, n_f + 0.3, PADZ - 0.3, z, "flagstone")
    for x in (s0 - 0.25, s1 + 0.25):
        box(x - 0.25, x + 0.25, n_f - 0.3 - nst * 0.32, n_f - 0.3, PADZ - 0.3, PADZ + 0.35, "limestone")
        lantern(x, n_f - 0.3 - nst * 0.32 + 0.25, PADZ + 0.35, 0.22, 12)
    # pergola: four timber posts, beams and rafters; built-in limestone seat with cushions and a timber table
    for x, n in ((pu0, pn0), (pu1, pn0), (pu0, pn1), (pu1, pn1)):
        box(x - 0.1, x + 0.1, n - 0.1, n + 0.1, zt, zt + 2.6, "timber")
    for n in (pn0, pn1):
        box(pu0 - 0.4, pu1 + 0.4, n - 0.09, n + 0.09, zt + 2.6, zt + 2.85, "timber")
    for k in range(10):
        x = pu0 - 0.2 + k * (pu1 - pu0 + 0.4) / 9
        box(x - 0.04, x + 0.04, pn0 - 0.4, pn1 + 0.4, zt + 2.85, zt + 3.0, "timber")
    box(pu0 + 0.2, pu1 - 0.2, pn1 - 0.55, pn1 - 0.05, zt, zt + 0.45, "limestone")
    box(pu0 + 0.25, pu1 - 0.25, pn1 - 0.5, pn1 - 0.1, zt + 0.45, zt + 0.55, "linen")
    box(pu0 + 0.25, pu1 - 0.25, pn1 - 0.15, pn1 - 0.02, zt + 0.55, zt + 0.95, "linen")
    box(pu0 + 0.9, pu1 - 0.9, pn0 + 0.9, pn1 - 1.0, zt + 0.7, zt + 0.76, "walnut")
    for x in (pu0 + 1.0, pu1 - 1.0):
        box(x - 0.05, x + 0.05, pn0 + 1.1, pn0 + 1.2, zt, zt + 0.7, "black")
    cr_pendant((pu0 + pu1) / 2, (pn0 + pn1) / 2, zt + 2.2, 25)
    # festoon lights between the posts
    for (xa, na), (xb, nb) in (((pu0, pn0), (pu1, pn0)), ((pu0, pn1), (pu1, pn1))):
        for k in range(1, 7):
            t = k / 7
            x, n = xa + (xb - xa) * t, na + (nb - na) * t
            z = zt + 2.5 - 0.25 * math.sin(math.pi * t)
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04, segments=8, ring_count=6, location=(x, n, z))
            bpy.context.active_object.data.materials.append(MATS["bulb"])
            light("POINT", (x, n, z - 0.05), 2.5, WARM, size=0.03, name="festoon")


def terrace_planting():
    if not TERR:
        return
    zt = PADZ + DT
    n_f = min(p[1] for p in TERR)
    for k in range(26):                                        # lavender and grasses along the wall top
        u = 16.4 + k * 0.48
        if 20.7 < u < 23.3 or u < 20.2 or not terrace_inside(u, n_f + 0.4):
            continue
        if k % 3 == 2:
            ograss(u, n_f + 0.45, zt, rr(0.6, 0.8))
        else:
            bush((u, n_f + 0.45, zt + 0.28), rr(0.3, 0.38), "lavender", 12)
    for u in np.arange(16.6, 30.0, 0.9):                       # hedge along the rear fence
        for n in (32.5 + (u + 5.1) * 0.3438 - 0.9,):
            if terrace_inside(u, n):
                bush((u, n, zt + 0.8), rr(0.6, 0.75), "hedge", 16)
    tree((25.8, 40.2, zt), 3.6, 1.7, "olive", "darkbark", stems=3, cards_per=22, crown_flat=0.6)
    tree((28.4, 42.8, zt), 3.2, 1.5, "olive", "darkbark", stems=3, cards_per=20, crown_flat=0.6)
    for u, n in ((20.4, 37.2), (24.0, 37.3), (27.0, 37.4)):
        bush((u, n, zt + 0.45), 0.45, "hydrangea", 16)


def ograss(u, n, z, h):
    """ornamental grass clump: crossed vertical cards"""
    for k in range(3):
        a = k * math.pi / 3 + rr(0, 1)
        dx, dn = math.cos(a) * h * 0.55, math.sin(a) * h * 0.55
        faces("ograss", [(u - dx, n - dn, z - 0.05), (u + dx, n + dn, z - 0.05), (u + dx, n + dn, z + h), (u - dx, n - dn, z + h)],
              [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]])



# ------------------------------------------------------------------ the farmhouse
def build_house():
    mu0, mu1, mn0, mn1 = L["main"]
    wu0, wu1, wn0, wn1 = L["wing"]
    ru0, ru1, rn0, rn1 = L["rear"]
    vu0, vu1, vn0, vn1 = L["verandah"]
    uu0, uu1, un0, un1 = L["upper"]

    # floors (warm timber) under every body
    for u0, u1, n0, n1 in (L["main"], L["wing"], L["rear"]):
        box(u0, u1, n0, n1, -0.25, 0.0, "floor")

    # ===== MODERN MEDITERRANEAN (rough concept): white rendered cubic volumes, flat roofs behind parapets,
    # one arched entry portal, deep-set rectangular windows, a view terrace upstairs, a north loggia over the pool.
    R = "slab"                     # white render
    CAP = "coping"                 # pale stone parapet capping
    def block(u0, u1, n0, n1, z_top, par=0.45):
        """flat roof slab + parapet capping around a rendered block"""
        box(u0, u1, n0, n1, z_top - 0.3, z_top, R)
        for (x0, x1, y0, y1) in ((u0, u1, n0, n0 + 0.25), (u0, u1, n1 - 0.25, n1), (u0, u0 + 0.25, n0, n1), (u1 - 0.25, u1, n0, n1)):
            box(x0, x1, y0, y1, z_top, z_top + par, R)
            box(x0 - 0.03, x1 + 0.03, y0 - 0.03, y1 + 0.03, z_top + par, z_top + par + 0.06, CAP)

    GH = 3.4                        # ground-floor parapet roof level (above FFL)
    # ground floor: main block, wing over the garage, west family wing
    eb0, eb1, ebn = 5.0, 9.0, mn0 - 0.6               # projecting entry bay with the arched portal
    door = (6.2, 7.8, 0.0, 2.15)
    wall_u(mu0, eb0, mn0, mn0 + T, 0, GH, [(2.0, 3.6, 0.9, 2.3)], R)
    window_u(2.0, 3.6, 0.9, 2.3, mn0, mn0 + T, cols=2, rows=2)
    wall_u(eb0, eb1, ebn, ebn + T, 0, GH + 0.6, [(door[0], door[1], 0.0, 2.95)], R)
    # arch head: render infill between the semicircle and the top of the opening, timber door below, fanlight above
    cx, rad, spring = (door[0] + door[1]) / 2, (door[1] - door[0]) / 2, 2.15
    K = 12
    for k in range(K):
        a0, a1 = math.pi * k / K, math.pi * (k + 1) / K
        x0, z0 = cx - rad * math.cos(a0), spring + rad * math.sin(a0)
        x1, z1 = cx - rad * math.cos(a1), spring + rad * math.sin(a1)
        slab((x0, ebn, z0), (x1, ebn, z1), (x1, ebn, 2.95), (x0, ebn, 2.95), T, R)
        poly([(cx, ebn + 0.1, spring), (x0, ebn + 0.1, z0), (x1, ebn + 0.1, z1)], "glass")
    box(door[0], door[1], ebn + 0.1, ebn + 0.16, 0, spring, "door")
    box(door[0] - 0.25, door[1] + 0.25, ebn - 0.6, ebn, -0.3, 0.0, "coping")   # stone threshold
    wall_n(ebn, mn0 + T, eb0 - T, eb0, 0, GH + 0.6, [], R)
    wall_n(ebn, mn0 + T, eb1, eb1 + T, 0, GH + 0.6, [], R)
    block(eb0 - T, eb1 + T, ebn, mn0 + T, GH + 0.6, par=0.0)
    for x in (door[0] - 0.4, door[1] + 0.4):
        box(x - 0.09, x + 0.09, ebn - 0.14, ebn, 1.7, 2.15, "black")
        light("POINT", (x, ebn - 0.25, 1.95), 18, WARM, size=0.05, name="door_lantern")
    wall_u(eb1, mu1, mn0 + 0.4, mn0 + 0.4 + T, 0, GH, [(10.2, 11.8, 0.9, 2.3), (12.8, 14.4, 0.9, 2.3)], R)
    window_u(10.2, 11.8, 0.9, 2.3, mn0 + 0.4, mn0 + 0.4 + T, cols=2, rows=2)
    window_u(12.8, 14.4, 0.9, 2.3, mn0 + 0.4, mn0 + 0.4 + T, cols=2, rows=2)
    wall_n(mn0, mn1, mu0, mu0 + T, 0, GH, [(13.0, 14.4, 1.0, 2.3)], R)
    window_n(13.0, 14.4, 1.0, 2.3, mu0, mu0 + T, cols=1, rows=2, outward=-1)
    big = (ru1 + 0.3, mu1 - 0.3, 0.0, 2.8)
    wall_u(ru1, mu1, mn1 - T, mn1, 0, GH, [big], R)
    window_u(*big, mn1 - T, mn1, cols=5, rows=1, outward=1)
    block(mu0, mu1, mn0, mn1, GH)
    # wing over the garage: one modest street window with a black awning, roof terrace behind a rendered balustrade
    wall_u(wu0, wu1, wn0, wn0 + T, 0, GH, [(17.6, 20.8, 0.9, 2.4)], R)
    window_u(17.6, 20.8, 0.9, 2.4, wn0, wn0 + T, cols=3, rows=2, awning=True)
    wall_n(wn0, wn1, wu1 - T, wu1, 0, GH, [], R)
    wall_u(wu0, wu1, wn1 - T, wn1, 0, GH, [(16.4, 22.0, 0.0, 2.8)], R)
    window_u(16.4, 22.0, 0.0, 2.8, wn1 - T, wn1, cols=3, rows=1, outward=1)
    block(wu0, wu1, wn0, wn1, GH, par=1.0)
    # west family wing (long, low): big rectangular glass east onto the courtyard
    sl = (rn0 + 0.6, rn1 - 0.6, 0.0, 2.8)
    wall_n(rn0, rn1, ru1 - T, ru1, 0, GH, [sl], R)
    window_n(*sl, ru1 - T, ru1, cols=5, rows=1, outward=1)
    wall_n(rn0, rn1, ru0, ru0 + T, 0, GH, [(rn0 + 3, rn0 + 4.2, 1.2, 2.2)], R)
    window_n(rn0 + 3, rn0 + 4.2, 1.2, 2.2, ru0, ru0 + T, cols=1, rows=1, outward=-1)
    wall_u(ru0, ru1, rn1 - T, rn1, 0, GH, [(ru0 + 1.5, ru1 - 1.5, 0.0, 2.8)], R)
    window_u(ru0 + 1.5, ru1 - 1.5, 0.0, 2.8, rn1 - T, rn1, cols=3, rows=1, outward=1)
    block(ru0, ru1, rn0 - 0.01, rn1, GH)

    # upper floor: a white box set back from the street over the middle, big rectangular glass to the street
    # and east onto two roof terraces (rendered balustrades on the ground-floor and wing roofs)
    xu0, xu1, xn0, xn1 = 3.5, 15.6, 11.8, 18.2
    UH = GH + 3.1
    wall_u(xu0, xu1, xn0, xn0 + T, GH, UH, [(4.6, 8.2, GH + 0.1, GH + 2.6), (9.4, 14.6, GH + 0.1, GH + 2.6)], R)
    window_u(4.6, 8.2, GH + 0.1, GH + 2.6, xn0, xn0 + T, cols=2, rows=1)
    window_u(9.4, 14.6, GH + 0.1, GH + 2.6, xn0, xn0 + T, cols=4, rows=1)
    wall_n(xn0, xn1, xu1 - T, xu1, GH, UH, [(xn0 + 0.8, xn1 - 0.8, GH + 0.1, GH + 2.6)], R)
    window_n(xn0 + 0.8, xn1 - 0.8, GH + 0.1, GH + 2.6, xu1 - T, xu1, cols=3, rows=1, outward=1)
    wall_n(xn0, xn1, xu0, xu0 + T, GH, UH, [(14.4, 15.6, GH + 1.0, GH + 2.2)], R)
    window_n(14.4, 15.6, GH + 1.0, GH + 2.2, xu0, xu0 + T, cols=1, rows=1, outward=-1)
    wall_u(xu0, xu1, xn1 - T, xn1, GH, UH, [(5.0, 7.0, GH + 1.0, GH + 2.3), (11.0, 13.0, GH + 1.0, GH + 2.3)], R)
    window_u(5.0, 7.0, GH + 1.0, GH + 2.3, xn1 - T, xn1, cols=1, rows=2, outward=1)
    window_u(11.0, 13.0, GH + 1.0, GH + 2.3, xn1 - T, xn1, cols=1, rows=2, outward=1)
    block(xu0, xu1, xn0, xn1, UH, par=0.35)
    box(xu0 + 0.3, xu1 - 0.3, xn0 + 0.3, xn1 - 0.3, GH - 0.05, GH + 0.05, "floor")
    # street-side view terrace on the ground-floor roof: raise the front parapet to a 1.0 m balustrade
    for (x0, x1, y0, y1) in ((mu0, mu1, mn0, mn0 + 0.25),):
        box(x0, x1, y0, y1, GH + 0.45, GH + 1.0, R)
        box(x0 - 0.03, x1 + 0.03, y0 - 0.03, y1 + 0.03, GH + 1.0, GH + 1.06, CAP)
    room_light(xu0 + 0.5, xu1 - 0.5, xn0 + 0.5, xn1 - 0.5, UH - 0.3, 1400)
    room_light(mu0 + 0.5, mu1 - 0.5, mn0 + 0.5, mn1 - 0.5, GH - 0.3, 1400)
    room_light(wu0 + 0.5, wu1 - 0.5, wn0 + 0.5, wn1 - 0.5, GH - 0.3, 900)
    room_light(ru0 + 0.5, ru1 - 0.5, rn0 + 0.5, rn1 - 0.5, GH - 0.3, 1200)
    for x, n in ((7.0, 10.5), (13.0, 10.5), (18.0, 12.0)):   # potted olives on the terraces
        cyl((x, n, GH), (x, n, GH + 0.6), 0.3, 0.36, "coping")
        tree((x, n, GH + 0.6), 2.2, 0.9, "olive", "darkbark", stems=2, cards_per=14, crown_flat=0.6)

    # north loggia over the pool side: rendered piers and a flat slab, rectangular openings
    lu0, lu1, ln0, ln1 = ru1, mu1, mn1, mn1 + 3.2
    for x in (lu0 + 0.15, (lu0 + lu1) / 2, lu1 - 0.15):
        box(x - 0.3, x + 0.3, ln1 - 0.6, ln1, 0, GH - 0.3, R)
    block(lu0, lu1, ln0, ln1, GH, par=0.25)
    for x in (lu0 + 1.6, lu1 - 1.6):
        cr_pendant(x, (ln0 + ln1) / 2, GH - 1.0, 40)

    # --- garage (street level) in painted brick, timber-look door; lawn on its roof
    gu0, gu1, gn0, gn1 = L["garage"]
    ring([(gu0, GZ - 0.3), (gu1, GZ - 0.3), (gu1, PADZ + 0.05), (gu0, PADZ + 0.05)],
         [(gu0 + 0.7, GZ), (gu1 - 0.7, GZ), (gu1 - 0.7, GZ + 2.4), (gu0 + 0.7, GZ + 2.4)], gn0, gn0 + 0.35, "whitebrick")
    box(gu0 + 0.7, gu1 - 0.7, gn0 + 0.45, gn0 + 0.5, GZ, GZ + 2.4, "slats")
    box(gu0, gu0 + 0.3, gn0, gn1, GZ - 0.3, PADZ + 0.05, "whitebrick")
    box(gu1 - 0.3, gu1 + 0.3, gn0, gn1, GZ - 0.3, PADZ + 0.05, "whitebrick")
    box(gu0, gu1, gn0, wn0 - 0.5, PADZ - 0.3, PADZ + 0.06, "lawn")  # planter roof in front of the wing
    box(gu0 - 0.3, gu0, 1.6, gn0, GZ - 0.3, PADZ + 0.05, "whitebrick")  # apron wall to the front lawn
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
    # curving flagstone path from the top of the steps to the verandah steps
    p0, p1, p2, p3 = ((s0 + s1) / 2, sn1), ((s0 + s1) / 2, sn1 + 1.4), (door[0] - 1.6, vn0 - 2.2), ((door[0] + door[1]) / 2, vn0 - 0.7)
    for k in range(13):
        t = k / 12
        cu = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t * t * p2[0] + t ** 3 * p3[0]
        cn = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t * t * p2[1] + t ** 3 * p3[1]
        for off in (-0.45, 0.35):
            a_ = rng.uniform(0, 3.14)
            w_, h_ = rr(0.35, 0.5), rr(0.25, 0.4)
            c_, s_ = math.cos(a_), math.sin(a_)
            x0, n0_ = cu + off * 0.9, cn + rr(-0.1, 0.1)
            corners = [(x0 + c_ * dx - s_ * dn, n0_ + s_ * dx + c_ * dn, PADZ + 0.06) for dx, dn in ((-w_, -h_), (w_, -h_), (w_, h_), (-w_, h_))]
            slab(*corners, -0.08, "flagstone")
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
    for x in (pu0 + 2.0, pu0 + 3.2, pu1 - 3.2, pu1 - 2.0):      # timber loungers with linen cushions north of the pool
        box(x - 0.35, x + 0.35, pn1 + 0.7, pn1 + 2.6, PADZ + 0.03, PADZ + 0.3, "timber")
        box(x - 0.32, x + 0.32, pn1 + 0.72, pn1 + 2.58, PADZ + 0.3, PADZ + 0.38, "linen")
        slab((x - 0.32, pn1 + 2.1, PADZ + 0.38), (x + 0.32, pn1 + 2.1, PADZ + 0.38), (x + 0.32, pn1 + 2.6, PADZ + 0.8), (x - 0.32, pn1 + 2.6, PADZ + 0.8), 0.07, "linen")
    for x in ((pu0 + pu1) / 2 - 0.8, (pu0 + pu1) / 2 + 0.8):     # big pots with clipped balls
        cyl((x, pn1 + 1.6, PADZ + 0.03), (x, pn1 + 1.6, PADZ + 0.6), 0.32, 0.4, "limestone", 12)
        bush((x, pn1 + 1.6, PADZ + 1.0), 0.42, "hedge", 18)
    box(11.0, 13.4, 23.0, 24.0, PADZ + 0.72, PADZ + 0.77, "walnut")
    for x in (11.4, 12.2, 13.0):
        for nn in (22.6, 24.4):
            box(x - 0.2, x + 0.2, nn - 0.2, nn + 0.2, PADZ, PADZ + 0.45, "black")
    box(0.4, ru0 - 0.1, rn1, cn1, PADZ - 0.3, PADZ + 0.03, "lawn")
    box(0.4, ru1, rn1, cn1, PADZ - 0.3, PADZ + 0.03, "lawn")

    # --- covered outdoor room with a stone fireplace on the NE side of the courtyard
    ou0, ou1, on0, on1 = L["outdoor"]
    zo = PADZ + 0.15
    box(ou0, ou1, on0, on1, PADZ - 0.3, zo, "flagstone")
    for x, nn in ((ou0 + 0.2, on0 + 0.2), (ou0 + 0.2, on1 - 0.2), (ou1 - 0.2, on0 + 0.2), (ou1 - 0.2, on1 - 0.2)):
        box(x - 0.15, x + 0.15, nn - 0.15, nn + 0.15, zo, zo + 2.8, "timber")
    box(ou0, ou1, on0 + 0.05, on0 + 0.35, zo + 2.8, zo + 3.05, "timber")
    box(ou0, ou1, on1 - 0.35, on1 - 0.05, zo + 2.8, zo + 3.05, "timber")
    box(ou0 + 0.05, ou0 + 0.35, on0, on1, zo + 2.8, zo + 3.05, "timber")
    roof_ridge_n(on0, on1, ou0, ou1, zo + 3.05, math.radians(35), o=0.35)
    fn0, fn1 = on0 + 1.4, on1 - 1.4
    box(ou1 - 0.6, ou1, fn0, fn1, zo, zo + 2.6, "limestone")
    box(ou1 - 0.5, ou1, fn0 + 0.6, fn1 - 0.6, zo + 2.6, zo + 6.2, "limestone")
    box(ou1 - 0.55, ou1 + 0.05, fn0 + 0.55, fn1 - 0.55, zo + 6.2, zo + 6.32, "black")
    box(ou1 - 0.65, ou1 - 0.55, fn0 + 0.6, fn1 - 0.6, zo + 0.45, zo + 1.35, "black")
    box(ou1 - 0.55, ou1 - 0.3, fn0 + 0.75, fn1 - 0.75, zo + 0.5, zo + 0.75, "fire")
    light("POINT", (ou1 - 0.9, (fn0 + fn1) / 2, zo + 0.8), 120, (1.0, 0.45, 0.15), size=0.3, name="pool_fire")
    box(ou1 - 1.0, ou1 - 0.6, fn0 - 0.2, fn1 + 0.2, zo, zo + 0.4, "limestone")  # hearth
    box(ou0 + 0.5, ou0 + 1.4, on0 + 1.2, on1 - 1.2, zo, zo + 0.45, "linen")       # sofa facing the fire
    box(ou0 + 0.5, ou0 + 0.8, on0 + 1.2, on1 - 1.2, zo + 0.45, zo + 0.85, "linen")
    box(ou0 + 1.9, ou0 + 2.6, on0 + 2.4, on1 - 2.4, zo, zo + 0.4, "walnut")
    cr_pendant((ou0 + ou1) / 2, (on0 + on1) / 2, zo + 2.4, 40)

    build_terrace()
    # --- lawn over the whole level pad, out to the side and rear boundaries (paving and floors sit above it)
    for tri in LAY.get("lawn_tris", []):
        poly([(u, n, PADZ + 0.012) for u, n in tri], "lawn")

    # --- retaining walls along the pad edge, just inside the boundary: in cut they hold the neighbours' (natural)
    #     ground and finish 0.15 m above it; in fill they stand 0.45 m above the pad. Tops follow the ground smoothly.
    #     Colorbond fences on the side walls (beige NE, sage-green SW, as in the site photos); an open black steel
    #     palisade on the rear boundary to the bushland (bushfire buffer, keeps the hill view)
    pts = PAD_OUTLINE
    for (ua, na), (ub, nb) in zip(pts, pts[1:]):
        if max(na, nb) < 2.6:
            continue  # the street wall handles the front edge
        if min(ua, ub) > L["garage"][0] - 0.5 and max(na, nb) < L["garage"][3] + 0.6:
            continue  # garage side walls already retain this edge
        Lg = math.hypot(ub - ua, nb - na)
        if Lg < 0.05:
            continue
        dx, dn = (ub - ua) / Lg, (nb - na) / Lg
        ou, on = -dn, dx                                   # unit normal, pointing out of the lot
        if lot_inside((ua + ub) / 2 + ou * 0.6, (na + nb) / 2 + on * 0.6):
            ou, on = -ou, -on

        def top_at(u, n):
            inner = PADZ + (DT if terrace_inside(u - ou * 0.5, n - on * 0.5) else 0.0)
            return max(inner + 0.45, zg(u + ou * 1.2, n + on * 1.2) + 0.15)
        d_sw = abs(32.5 * (ua + ub) / 2 + 5.1 * (na + nb) / 2) / math.hypot(32.5, 5.1)
        rear = abs(dn) < 0.6 and (na + nb) / 2 > 30
        steps = max(1, int(Lg / 1.5))
        for k in range(steps):
            t0, t1 = k / steps, (k + 1) / steps
            u0_, n0_ = ua + (ub - ua) * t0, na + (nb - na) * t0
            u1_, n1_ = ua + (ub - ua) * t1, na + (nb - na) * t1
            ta, tb = top_at(u0_, n0_), top_at(u1_, n1_)
            bot = min(zg(u0_ + ou * 1.2, n0_ + on * 1.2), PADZ) - 0.6
            q = [(u0_, n0_), (u1_, n1_), (u1_ + ou * 0.3, n1_ + on * 0.3), (u0_ + ou * 0.3, n0_ + on * 0.3)]
            tz = [ta, tb, tb, ta]
            faces("limestone", [(a, b, bot) for a, b in q] + [(a, b, z) for (a, b), z in zip(q, tz)],
                  [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
            qc = [(u0_ - ou * 0.04, n0_ - on * 0.04), (u1_ - ou * 0.04, n1_ - on * 0.04), (u1_ + ou * 0.34, n1_ + on * 0.34), (u0_ + ou * 0.34, n0_ + on * 0.34)]
            faces("coping", [(a, b, z) for (a, b), z in zip(qc, tz)] + [(a, b, z + 0.06) for (a, b), z in zip(qc, tz)],
                  [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
            if max(n0_, n1_) < 6.5:
                continue
            fu0, fn0, fu1, fn1 = u0_ + ou * 0.2, n0_ + on * 0.2, u1_ + ou * 0.2, n1_ + on * 0.2
            if rear:      # open palisade: posts, two rails, pickets every 0.12 m
                for z in (0.12, 1.12):
                    cyl((fu0, fn0, ta + 0.06 + z), (fu1, fn1, tb + 0.06 + z), 0.02, 0.02, "black", 4)
                m_ = max(1, int(math.hypot(fu1 - fu0, fn1 - fn0) / 0.12))
                for j in range(m_):
                    t = (j + 0.5) / m_
                    x, y, z = fu0 + (fu1 - fu0) * t, fn0 + (fn1 - fn0) * t, ta + (tb - ta) * t + 0.06
                    box(x - 0.01, x + 0.01, y - 0.01, y + 0.01, z, z + 1.2, "black")
                box(fu0 - 0.03, fu0 + 0.03, fn0 - 0.03, fn0 + 0.03, ta, ta + 1.3, "black")
                continue
            mat = "fence_sage" if d_sw < 1.0 else "fence_beige"
            za, zb2 = ta + 0.06, tb + 0.06
            fa, fb = max(ta, zg(u0_ + ou, n0_ + on)) + 1.8, max(tb, zg(u1_ + ou, n1_ + on)) + 1.8
            fa = fb = max(fa, fb)
            f4 = [(fu0 - ou * 0.02, fn0 - on * 0.02), (fu1 - ou * 0.02, fn1 - on * 0.02), (fu1 + ou * 0.02, fn1 + on * 0.02), (fu0 + ou * 0.02, fn0 + on * 0.02)]
            zb4, zt4 = [za, zb2, zb2, za], [fa, fb, fb, fa]
            faces(mat, [(a, b, z) for (a, b), z in zip(f4, zb4)] + [(a, b, z) for (a, b), z in zip(f4, zt4)],
                  [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
            box(fu0 - 0.035, fu0 + 0.035, fn0 - 0.035, fn0 + 0.035, za, fa + 0.03, mat)

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
    drape(-1.5, gu0, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R - 2.3) + 0.35, 0.06, "lawn", 40, 1)
    drape(-1.5, gu0, lambda u: arc_n(u, R - 2.3) + 0.35, lambda u: 1.58, 0.07, "mulch", 40, 4)
    drape(-45, -1.5, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 1.0, 0.04, "lawn", 30, 2)
    drape(gu1 + 0.3, 68, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 1.0, 0.04, "lawn", 30, 2)


def cr_pendant(x, n, z, strength=60, top=None):
    cyl((x, n, z + 0.15), (x, n, min(z + 1.2, top - 0.02) if top else z + 1.2), 0.006, 0.006, "black")
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
    tree((-9.5, 5.5, zg(-9.5, 5.5)), 12.0, 5.8, "hedge", "darkbark", stems=1, cards_per=36, crown_flat=0.45)
    # borders: box balls and white hydrangeas along the path and the verandah, hydrangeas behind the street wall
    s0, s1, sn0, sn1 = L["stair"]
    vn0 = L["verandah"][2]
    n = sn1 + 0.3
    while n < vn0 - 0.4:
        for x in (s0 - 0.55, s1 + 0.55):
            bush((x + rr(-0.1, 0.1), n, PADZ + 0.35), rr(0.32, 0.42), "hydrangea" if rng.random() < 0.6 else "hydrangea_lime", 16)
        for x in (s0 - 1.25, s1 + 1.25):
            bush((x, n + 0.3, PADZ + 0.35), 0.36, "hedge", 18)
        n += 0.8
    for x in np.arange(1.4, 11.2, 0.9):
        if not (s0 - 0.6 < x < s1 + 0.6):
            bush((x, vn0 - 0.45, PADZ + 0.4), rr(0.38, 0.5), "hydrangea" if rng.random() < 0.55 else "hydrangea_lime", 18)
    for x in np.arange(0.9, 15.6, 0.8):
        for nn in (2.8, 3.8):
            if not (s0 - 0.8 < x < s1 + 0.8):
                xx = x + (0.4 if nn > 3 else 0)
                bush((xx, nn, zg(xx, nn) + 0.38), rr(0.34, 0.44), "hydrangea" if nn < 3 or rng.random() < 0.5 else "hedge", 14)
    # verge garden, layered like the reference: star jasmine hedge against the wall, white hydrangeas and lavender
    # in the middle, box balls and ornamental grasses at the front, a lawn strip along the footpath
    gu0 = L["garage"][0]
    for u in np.arange(-1.0, gu0 - 0.2, 0.55):
        if s0 - 0.8 < u < s1 + 0.8:
            continue
        nw = 1.25
        bush((u + rr(-0.1, 0.1), nw, zg(u, nw) + 0.75), rr(0.45, 0.55), "hedge", 14)          # back hedge
        nm_ = (arc_n(u, ARC["r"] - 2.3) + 1.58) / 2 + 0.25
        if int(u / 0.55) % 2:
            bush((u, nm_, zg(u, nm_) + 0.42), rr(0.38, 0.46), "hydrangea" if rng.random() < 0.7 else "hydrangea_lime", 16)
        else:
            bush((u, nm_, zg(u, nm_) + 0.3), rr(0.32, 0.38), "lavender", 12)
        nf_ = arc_n(u, ARC["r"] - 2.3) + 0.75
        if int(u / 0.55) % 7 == 3:
            ograss(u + 0.2, nf_, zg(u, nf_), rr(0.5, 0.6))
        else:
            bush((u, nf_, zg(u, nf_) + 0.24), rr(0.22, 0.28), "hedge", 14)                    # box balls
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
    for u, nn in ((10.4, 35.2), (25.6, 33.2)):
        tree((u, nn, PADZ), rr(3.4, 4.0), rr(1.6, 1.9), "olive", "darkbark", stems=3, cards_per=22, crown_flat=0.6)
    # distant trees on the plains and along the streets below (the view is over a leafy town edge, not bare paddock)
    for _ in range(140):
        u, n = rr(-120, 160), rr(-260, -30)
        if -10 < u < 40 and n > -50:
            continue
        h = rr(6, 12)
        tree((u, n, zg(u, n)), h, h * rr(0.3, 0.42), rng.choice(["hedge", "euc", "olive", "hedge"]), "darkbark", stems=1, cards_per=10, crown_flat=0.6)
    for _ in range(40):
        u, n = rr(45, 140), rr(-30, 60)
        tree((u, n, zg(u, n)), rr(7, 12), rr(2.5, 4), rng.choice(["hedge", "euc"]), "darkbark", stems=1, cards_per=10, crown_flat=0.6)


GLB_TEX = {"clad": ("cladding.png", 0.25), "whitebrick": ("whitebrick.png", 0.5), "paving": ("paving.png", 0.25),
           "coping": ("paving.png", 0.25), "slats": ("slats.png", 0.25), "limestone": ("stone.png", 0.35),
           "flagstone": ("stone.png", 0.6), "stone": ("stone.png", 0.25)}
GLB_COL = {"roof_x": ("#2b2d30", 0.42, 0.55), "roof_y": ("#2b2d30", 0.42, 0.55), "black": ("#141516", 0.4, 0.3),
           "timber": ("#7a5636", 0.6, 0), "deck": ("#8a6644", 0.55, 0), "door": ("#6b4a2c", 0.45, 0),
           "lawn": ("#5f7d38", 0.95, 0), "mulch": ("#4f3f31", 1.0, 0), "concrete": ("#bbb5aa", 0.9, 0),
           "kerb": ("#b6b0a5", 0.9, 0), "asphalt": ("#38393b", 0.9, 0), "aggregate": ("#a39b8e", 0.9, 0),
           "pooltile": ("#8fd0e3", 0.3, 0), "water": ("#3fa9cf", 0.05, 0), "bark": ("#a09684", 0.85, 0),
           "darkbark": ("#4d443b", 0.9, 0), "grass": ("#a8874a", 1.0, 0), "fire": ("#ff7a2a", 0.5, 0), "floor": ("#8a6646", 0.4, 0),
           "slab": ("#e8e2d6", 0.7, 0), "glass": ("#0e1416", 0.05, 0.0), "bulb": ("#fff2d8", 0.5, 0)}
INTERIOR = {"fabric", "linen", "walnut", "rug", "art", "lamp"}
EXTERIOR_LIGHTS = ("lantern", "door_lantern", "garage_lamp", "verandah", "pool", "wall_up", "olive_up", "garage_up",
                   "sconce", "soffit", "cabana")  # pool_fire included


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
        if name in GLB_TEX or name in ("euc", "olive", "shrub", "hedge", "hydrangea", "hydrangea_lime", "tussock") or name == "terrain":
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
    if EUC:  # the eucalypt list is shared by every design
        (D3 / "farmhouse_trees.json").write_text(json.dumps([[round(x, 2) for x in t] for t in EUC]))
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", export_lights=True, export_cameras=False,
                              export_yup=True, export_apply=True, export_image_format="JPEG", export_jpeg_quality=85)


def export_mesh(path):
    """house geometry (everything except terrain, street, plants, trees) as triangles with base colours"""
    skip = {"terrain", "backdrop", "asphalt", "kerb", "concrete", "aggregate", "mulch", "euc", "olive", "shrub",
            "hedge", "hydrangea", "hydrangea_lime", "tussock", "bark", "darkbark", "glass", "bulb", "lamp", "fire"}
    colours = {"fence_beige": (217, 208, 184), "fence_sage": (158, 169, 141), "clad": (238, 236, 230), "limestone": (214, 202, 178), "flagstone": (190, 182, 168), "whitebrick": (236, 234, 228), "roof_x": (45, 47, 50), "roof_y": (45, 47, 50),
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


def fix_terrain_under_pool():
    """the shared terrain sinks a fixed pool rectangle; sink ours instead and restore the rest"""
    ob = bpy.data.objects["terrain"]
    pu0, pu1, pn0, pn1 = L["pool"]
    for v in ob.data.vertices:
        u, n, z = v.co
        if pu0 - 1.4 < u < pu1 + 1.4 and pn0 - 1.4 < n < pn1 + 1.4:   # wide, so no slanted cell reaches into the water
            v.co.z = PADZ - 1.8
        elif 3.7 < u < 14.3 and 27.2 < n < 31.3:
            v.co.z = zg(u, n)
    ob.data.update()


BEVEL = {"clad": 0.012, "whitebrick": 0.015, "limestone": 0.02, "black": 0.006, "coping": 0.012, "timber": 0.012,
         "paving": 0.006, "render": 0.015, "fence_beige": 0.004, "fence_sage": 0.004, "kerb": 0.02, "slab": 0.01,
         "soffit": 0.004, "deck": 0.006, "flagstone": 0.01}


def add_bevels():
    """small bevels on walls, trims and copings so edges catch the low sun (render only, not the web model)"""
    for ob in bpy.data.objects:
        if ob.type == "MESH" and ob.name.split(".")[0] in BEVEL:
            m = ob.modifiers.new("bevel", "BEVEL")
            m.width = BEVEL[ob.name.split(".")[0]]
            m.segments = 2
            m.limit_method = "ANGLE"
            m.angle_limit = math.radians(40)
            m.use_clamp_overlap = True
            for p in ob.data.polygons:
                p.use_smooth = False


def golden_sky():
    """warm golden-hour gradient with soft lit clouds for camera rays (lighting still from the physical sky + sun)
    (from scripts/designs/sage_stone_cycles.py)"""
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
    "street": dict(eye=(11.6, -14.5, 1.7), at=(12.2, 14.0, 4.3), lens=25),
    "roofcheck": dict(eye=(-5.0, -2.0, 8.0), at=(9.0, 14.0, 4.0), lens=26),
    "roofcheck_st": dict(eye=(11.6, -14.5, 1.7), at=(12.2, 14.0, 4.3), lens=25),
    "roofcheck5": dict(eye=(17.0, 36.5, 3.5), at=(17.0, 30.0, -1.0), lens=24),
    "roofcheck6": dict(eye=(10.0, 26.0, 7.5), at=(12.5, 19.0, 4.0), lens=30),
    "roofcheck4": dict(eye=(25.0, 3.0, 7.5), at=(17.0, 13.0, 5.0), lens=30),
    "roofcheck3": dict(eye=(-6.0, 44.0, 9.0), at=(12.0, 20.0, 2.0), lens=26),
    "roofcheck2": dict(eye=(30.0, 2.0, 9.0), at=(14.0, 16.0, 4.0), lens=28),
    "street_wide": dict(eye=(9.5, -26.0, 3.2), at=(11.5, 16.0, 4.0), lens=30),
    "courtyard": dict(eye=(9.0, 42.0, 6.0), at=(17.6, 29.4, 0.5), lens=24),
    "aerial": dict(eye=(48.0, -24.0, 42.0), at=(12.0, 20.0, 0.0), lens=30),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="street")
    ap.add_argument("--scale", type=int, default=50)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--out", default=str(ROOT / "farmhouse_street.png"))
    ap.add_argument("--exposure", type=float, default=1.02)
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
        "gutter": flat_mat("gutter", srgb("#3a3d40"), 0.4, 0.5),
        "soffit": flat_mat("soffit", srgb("#eee9df"), 0.8),
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
        "hydrangea": leaf_mat("hydrangea"), "hydrangea_lime": leaf_mat("hydrangea_lime"),
        "limestone": img_mat("limestone", "stone.png", 0.35, 0.92, bump=0.7, tint=(1.07, 1.05, 1.0)),
        "flagstone": img_mat("flagstone", "stone.png", 0.6, 0.85, bump=0.4, tint=(0.95, 0.95, 0.95)),
        "fire": flat_mat("fire", (1, 0.4, 0.1), 0.5, emit=(1.0, 0.42, 0.12), strength=40),
        "fence_beige": fence_mat("fence_beige", "#d9d0b8"), "fence_sage": fence_mat("fence_sage", "#9ea98d"),
        "lavender": leaf_mat("lavender"), "ograss": leaf_mat("ograss"), "bottlebrush": leaf_mat("bottlebrush"),
        "lawn_dry": noisy_mat("lawn_dry", "#5d6b2e", "#8e8c4a", 25, 0.95, 0.3),
        "tile_terra": tile_mat("tile_terra", "#9c4a33", "#b8664a"), "tile_grey": tile_mat("tile_grey", "#4f4c49", "#6b6865"),
        "nrender": flat_mat("nrender", srgb("#efece4"), 0.85), "nbrick": img_mat("nbrick", "whitebrick.png", 0.5, 0.8, bump=0.2, tint=(0.86, 0.74, 0.6)),
        "nglass": flat_mat("nglass", srgb("#1d2326"), 0.08, 0.2), "nframe": flat_mat("nframe", srgb("#efefeb"), 0.5),
        "nglass_lit": flat_mat("nglass_lit", srgb("#3a3026"), 0.1, emit=WARM, strength=2.2),
        "terrain": terrain_mat("terrain", True), "grass": terrain_mat("grass", False),
    })
    for rn in ("roof_x", "roof_y"):   # charcoal standing seam, not pure black
        rb = MATS[rn].node_tree.nodes["Principled BSDF"]
        rb.inputs["Base Color"].default_value = (*srgb("#3a3d40"), 1)
        rb.inputs["Roughness"].default_value = 0.45
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
    if a.export_mesh:
        export_mesh(D3 / "farmhouse_mesh.json")
    if not a.view.startswith("roofcheck") and not os.environ.get("NOGARDEN"):
        build_context()
        build_garden()
        terrace_planting()
    cr.flush()
    if a.export_glb:
        export_glb(a.export_glb)
        print("exported", a.export_glb)
        return
    add_bevels()
    for nm in ("terrain", "backdrop"):   # the hill must not put the whole street in shade at a 10° sun
        if nm in bpy.data.objects:
            bpy.data.objects[nm].visible_shadow = False
    for ob in bpy.data.objects:          # light emitters never show as shapes in the picture
        if ob.type == "LIGHT":
            ob.visible_camera = False
            ob.visible_glossy = False
    cr.VIEWS.update(VIEWS)
    sun_az = {"street": 212.0, "roofcheck_st": 212.0, "courtyard": 288.0}.get(a.view, 260.0)    # low golden-hour sun from the W/SW
    cr.setup_world_and_camera(a.view, 10.0, sun_az, a.exposure)
    golden_sky()
    bpy.data.lights["sun"].energy = float(os.environ.get("SUN_E", "8.0"))      # stronger low sun: warm raking light
    bpy.context.scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = float(os.environ.get("SKY_E", "0.4"))

    sc = bpy.context.scene
    if os.environ.get("RAYDEBUG"):
        bpy.context.view_layer.update()
        sun = bpy.data.objects["sun"]
        to_sun = sun.matrix_world.to_quaternion() @ Vector((0, 0, 1))
        print("to_sun", to_sun)
        dg = bpy.context.evaluated_depsgraph_get()
        for p0 in ((12.0, 7.4, 5.0), (17.9, 7.3, 4.5), (8.0, 9.0, 2.0), (0.9, 14.0, 4.0)):
            hit, loc, nrm, idx, obj, mtx = sc.ray_cast(dg, Vector(p0), to_sun)
            print("RAY", p0, hit, obj.name if obj else None, tuple(round(c, 1) for c in loc))
        return
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
