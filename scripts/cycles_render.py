"""Path-traced (Blender Cycles) concept render of the proposed house at 36 Waugh Street.

Run with Blender's Python module (bpy 4.2, Python 3.11):
    python cycles_render.py --view street --scale 50 --samples 64 --out render.png

Inputs come from data/3d/render_scene.json (written by build_3d_render.py): designed ground,
house parameters, the street-frame transform, eucalypt positions from the aerial photo.
Scene axes: X = u (along the street frontage), Y = n (into the lot), Z = metres above floor level.
"""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
D3 = ROOT / "data" / "3d"
TEX = D3 / "tex"
TEX.mkdir(exist_ok=True)
S = json.loads((D3 / "render_scene.json").read_text())
FFL, PADZ, GZ = S["FFL"], S["PAD"] - S["FFL"], S["GARAGE_FL"] - S["FFL"]
UP, EAVE, RIDGE = S["UPPER"], S["EAVE"], S["RIDGE"]
NG = S["size"]
H = np.array(S["heights_m"]).reshape(NG, NG)  # row 0 = south, metres AHD
AX, AY = S["A"]
UX, UY = S["U"]
NX, NY = -UY, UX
ARC = S["arc"]
rng = random.Random(20261012)
rr = rng.uniform


# ---------------------------------------------------------------- coordinates
def to_frame(gx, gy):
    dx, dy = gx - AX, gy - AY
    return dx * UX + dy * UY, dx * NX + dy * NY


def to_grid(u, n):
    return AX + u * UX + n * NX, AY + u * UY + n * NY


def h_grid(gx, gy):
    x = min(max(gx - 0.5, 0), NG - 1.001)
    y = min(max(gy - 0.5, 0), NG - 1.001)
    i, j = int(x), int(y)
    fx, fy = x - i, y - j
    a, b, c, d = H[j, i], H[j, i + 1], H[j + 1, i], H[j + 1, i + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def zg(u, n):
    """ground height above floor level at (u, n)"""
    return h_grid(*to_grid(u, n)) - FFL


def arc_n(u, r):
    return ARC["cn"] + math.sqrt(max(r * r - (u - ARC["cu"]) ** 2, 0))


# ---------------------------------------------------------------- textures (PIL)
def make_textures():
    R = random.Random(7)
    # sawn sandstone ashlar, 2048 px = 4 m
    im = Image.new("RGB", (2048, 2048), (125, 111, 90))
    d = ImageDraw.Draw(im)
    y = 0
    while y < 2048:
        ch = int(R.uniform(0.27, 0.38) * 512)
        x = -R.uniform(0, 300)
        while x < 2048:
            bw = R.uniform(0.5, 1.15) * 512
            l, w = R.uniform(-16, 10), R.uniform(-8, 8)
            d.rectangle([x + 5, y + 5, x + bw - 5, y + ch - 5], fill=(int(214 + l + w), int(199 + l), int(168 + l - w)))
            for _ in range(60):
                cx, cy = x + R.uniform(0, bw), y + R.uniform(0, ch)
                rw, rh = R.uniform(6, 40), R.uniform(2, 10)
                t = R.uniform(-14, 12)
                d.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=(int(212 + l + t), int(197 + l + t), int(166 + l + t)))
            x += bw
        y += ch
    arr = np.array(im).astype(np.int16)
    arr += np.random.default_rng(1).integers(-10, 10, arr.shape[:2])[..., None]
    im = Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
    im.save(TEX / "stone.png")
    # large-format pale pavers 1024 = 4 m
    im = Image.new("RGB", (1024, 1024), (150, 144, 132))
    d = ImageDraw.Draw(im)
    for yy in range(0, 1024, 154):
        off = 0 if (yy // 154) % 2 == 0 else -154
        for xx in range(off, 1024, 307):
            l = R.uniform(-8, 8)
            d.rectangle([xx + 3, yy + 3, xx + 304, yy + 151], fill=(int(212 + l), int(204 + l), int(188 + l)))
    arr = np.array(im).astype(np.int16) + np.random.default_rng(2).integers(-7, 7, (1024, 1024))[..., None]
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(TEX / "paving.png")
    # vertical timber slats 512 = 4 m (16 px slats)
    im = Image.new("RGB", (512, 512), (30, 23, 18))
    d = ImageDraw.Draw(im)
    for xx in range(0, 512, 16):
        l = R.uniform(-8, 10)
        d.rectangle([xx, 0, xx + 12, 512], fill=(int(70 + l), int(52 + l), int(38 + l)))
    im.save(TEX / "slats.png")
    # leaf cards: RGBA twig clusters
    for name, base, n_leaves, size in (("euc", (78, 92, 56), 140, (12, 4)), ("olive", (138, 150, 118), 220, (10, 3)),
                                       ("shrub", (96, 110, 84), 320, (7, 3)), ("hedge", (52, 82, 40), 420, (8, 4)), ("tussock", (176, 150, 96), 260, (14, 1.5))):
        im = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        twigs = [(R.uniform(150, 362), R.uniform(150, 362)) for _ in range(5)]
        for _ in range(n_leaves):
            tx, ty = R.choice(twigs)
            a = R.uniform(0, 6.283)
            dist = R.uniform(0, 190)
            cx, cy = 256 + (tx - 256) * 0.4 + math.cos(a) * dist, 256 + (ty - 256) * 0.4 + math.sin(a) * dist
            if math.hypot(cx - 256, cy - 256) > 235:
                continue
            ang = R.uniform(0, 6.283)
            L, W = size[0] * R.uniform(1.6, 2.6), size[1] * R.uniform(1.5, 2.4)
            pts = [(cx + math.cos(ang + t) * (L if k % 2 == 0 else W) * (1 if k < 2 else 1), cy + math.sin(ang + t) * (L if k % 2 == 0 else W))
                   for k, t in enumerate((0, math.pi / 2, math.pi, 3 * math.pi / 2))]
            l = R.uniform(-22, 22)
            d.polygon(pts, fill=(int(base[0] + l), int(base[1] + l), int(base[2] + l * 0.8), 255))
        im.save(TEX / f"leaf_{name}.png")


# ---------------------------------------------------------------- materials
def node_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    return m, nt, bsdf


def obj_coords(nt, scale):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    return mp.outputs["Vector"]


def img_mat(name, file, scale, rough, bump=0.0, tint=None, metallic=0.0):
    m, nt, b = node_mat(name)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(TEX / file))
    tex.projection = "BOX"
    tex.projection_blend = 0.25
    nt.links.new(obj_coords(nt, scale), tex.inputs["Vector"])
    col = tex.outputs["Color"]
    if tint:
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        mix.inputs["B"].default_value = (*tint, 1)
        nt.links.new(col, mix.inputs["A"])
        col = mix.outputs["Result"]
    nt.links.new(col, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    if bump:
        bn = nt.nodes.new("ShaderNodeBump")
        bn.inputs["Strength"].default_value = bump
        bn.inputs["Distance"].default_value = 0.02
        nt.links.new(tex.outputs["Color"], bn.inputs["Height"])
        nt.links.new(bn.outputs["Normal"], b.inputs["Normal"])
    return m


def flat_mat(name, rgb, rough=0.6, metallic=0.0, emit=None, strength=0.0):
    m, nt, b = node_mat(name)
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = strength
    return m


def noisy_mat(name, c1, c2, scale, rough, bump=0.2):
    """two-tone noise surface (asphalt, gravel, concrete) in object space"""
    m, nt, b = node_mat(name)
    co = obj_coords(nt, 1.0)
    nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = scale; nz.inputs["Detail"].default_value = 12
    nt.links.new(co, nz.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35; ramp.color_ramp.elements[0].color = (*srgb(c1), 1)
    ramp.color_ramp.elements[1].position = 0.65; ramp.color_ramp.elements[1].color = (*srgb(c2), 1)
    nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    bn = nt.nodes.new("ShaderNodeBump"); bn.inputs["Strength"].default_value = bump; bn.inputs["Distance"].default_value = 0.01
    nt.links.new(nz.outputs["Fac"], bn.inputs["Height"]); nt.links.new(bn.outputs["Normal"], b.inputs["Normal"])
    return m


def srgb(hexs):
    c = [int(hexs[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


def seam_roof(name, axis):
    """dark standing-seam steel; ribs every 0.45 m along the given object axis"""
    m, nt, b = node_mat(name)
    b.inputs["Base Color"].default_value = (*srgb("#2b2d30"), 1)
    b.inputs["Metallic"].default_value = 0.55
    b.inputs["Roughness"].default_value = 0.42
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    div = nt.nodes.new("ShaderNodeMath"); div.operation = "DIVIDE"; div.inputs[1].default_value = 0.45
    nt.links.new(sep.outputs[axis], div.inputs[0])
    fr = nt.nodes.new("ShaderNodeMath"); fr.operation = "FRACT"
    nt.links.new(div.outputs[0], fr.inputs[0])
    lt = nt.nodes.new("ShaderNodeMath"); lt.operation = "LESS_THAN"; lt.inputs[1].default_value = 0.06
    nt.links.new(fr.outputs[0], lt.inputs[0])
    bn = nt.nodes.new("ShaderNodeBump"); bn.inputs["Strength"].default_value = 0.6; bn.inputs["Distance"].default_value = 0.02
    nt.links.new(lt.outputs[0], bn.inputs["Height"])
    nt.links.new(bn.outputs["Normal"], b.inputs["Normal"])
    return m


def thin_glass(name, tint=(0.9, 0.94, 0.94)):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent"); tr.inputs["Color"].default_value = (*tint, 1)
    gl = nt.nodes.new("ShaderNodeBsdfGlossy"); gl.inputs["Roughness"].default_value = 0.0
    fr = nt.nodes.new("ShaderNodeFresnel"); fr.inputs["IOR"].default_value = 1.52
    mx = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fr.outputs[0], mx.inputs[0])
    nt.links.new(tr.outputs[0], mx.inputs[1])
    nt.links.new(gl.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs["Surface"])
    return m


def leaf_mat(name, translucent=(0.25, 0.32, 0.12)):
    m, nt, b = node_mat(name)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(TEX / f"leaf_{name}.png"))
    uv = nt.nodes.new("ShaderNodeUVMap")
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(tex.outputs["Alpha"], b.inputs["Alpha"])
    b.inputs["Roughness"].default_value = 0.65
    b.inputs["Subsurface Weight"].default_value = 0.0
    b.inputs["Transmission Weight"].default_value = 0.0
    return m


def terrain_mat(name, use_aerial):
    """dry golden pasture grass, with the aerial photo blended in for real tracks, trees and shading"""
    m, nt, b = node_mat(name)
    co = obj_coords(nt, 1.0)
    n1 = nt.nodes.new("ShaderNodeTexNoise"); n1.inputs["Scale"].default_value = 0.08; n1.inputs["Detail"].default_value = 6
    n2 = nt.nodes.new("ShaderNodeTexNoise"); n2.inputs["Scale"].default_value = 3.0; n2.inputs["Detail"].default_value = 8
    nt.links.new(co, n1.inputs["Vector"]); nt.links.new(co, n2.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*srgb("#7d6a3c"), 1)
    ramp.color_ramp.elements[1].color = (*srgb("#c99f55"), 1)
    e = ramp.color_ramp.elements.new(0.55); e.color = (*srgb("#a8874a"), 1)
    mixn = nt.nodes.new("ShaderNodeMath"); mixn.operation = "MULTIPLY_ADD"
    mixn.inputs[1].default_value = 0.6
    nt.links.new(n2.outputs["Fac"], mixn.inputs[0])
    nt.links.new(n1.outputs["Fac"], mixn.inputs[2])
    nt.links.new(mixn.outputs[0], ramp.inputs["Fac"])
    col = ramp.outputs["Color"]
    if use_aerial:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(D3 / "ortho_texture.jpg"))
        uv = nt.nodes.new("ShaderNodeUVMap")
        nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
        bw = nt.nodes.new("ShaderNodeRGBToBW")
        nt.links.new(tex.outputs["Color"], bw.inputs["Color"])
        # shade the grass by the aerial's brightness (trees, tracks), keep the grass colour
        sc = nt.nodes.new("ShaderNodeMath"); sc.operation = "MULTIPLY_ADD"
        sc.inputs[1].default_value = 1.4; sc.inputs[2].default_value = 0.45
        nt.links.new(bw.outputs["Val"], sc.inputs[0])
        mul = nt.nodes.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"; mul.inputs["Factor"].default_value = 1.0
        nt.links.new(col, mul.inputs["A"])
        cmb = nt.nodes.new("ShaderNodeCombineColor")
        for k in range(3):
            nt.links.new(sc.outputs[0], cmb.inputs[k])
        nt.links.new(cmb.outputs[0], mul.inputs["B"])
        col = mul.outputs["Result"]
    nt.links.new(col, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.95
    n3 = nt.nodes.new("ShaderNodeTexNoise"); n3.inputs["Scale"].default_value = 40.0; n3.inputs["Detail"].default_value = 10
    nt.links.new(co, n3.inputs["Vector"])
    bn = nt.nodes.new("ShaderNodeBump"); bn.inputs["Strength"].default_value = 0.4
    nt.links.new(n3.outputs["Fac"], bn.inputs["Height"])
    nt.links.new(bn.outputs["Normal"], b.inputs["Normal"])
    return m


# ---------------------------------------------------------------- geometry batching
class Batch:
    def __init__(self):
        self.v, self.f, self.uv = [], [], []


B = defaultdict(Batch)
MATS = {}


def faces(mat, verts, fcs, uvs=None):
    b = B[mat]
    o = len(b.v)
    b.v.extend(verts)
    for k, f in enumerate(fcs):
        b.f.append([i + o for i in f])
        b.uv.append(uvs[k] if uvs else None)


def box(u0, u1, n0, n1, z0, z1, mat):
    v = [(u0, n0, z0), (u1, n0, z0), (u1, n1, z0), (u0, n1, z0), (u0, n0, z1), (u1, n0, z1), (u1, n1, z1), (u0, n1, z1)]
    faces(mat, v, [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])


def prism(poly, n0, n1, mat):
    """polygon in the u-z plane extruded from n0 to n1"""
    k = len(poly)
    v = [(u, n0, z) for u, z in poly] + [(u, n1, z) for u, z in poly]
    fcs = [list(range(k)), [k + i for i in reversed(range(k))]]
    fcs += [(i, (i + 1) % k, k + (i + 1) % k, k + i) for i in range(k)]
    faces(mat, v, fcs)


def ring(outer, inner, n0, n1, mat):
    """wall with an opening: outer and inner outlines have the same vertex count"""
    k = len(outer)
    v = [(u, n0, z) for u, z in outer] + [(u, n0, z) for u, z in inner] + [(u, n1, z) for u, z in outer] + [(u, n1, z) for u, z in inner]
    fcs = []
    for i in range(k):
        j = (i + 1) % k
        fcs += [(i, j, k + j, k + i), (2 * k + j, 2 * k + i, 3 * k + i, 3 * k + j),
                (i, 2 * k + i, 2 * k + j, j), (k + j, 3 * k + j, 3 * k + i, k + i)]
    faces(mat, v, fcs)


def poly(pts, mat):
    faces(mat, pts, [list(range(len(pts)))])


def slab(p0, p1, p2, p3, t, mat):
    p = [Vector(x) for x in (p0, p1, p2, p3)]
    nrm = (p[1] - p[0]).cross(p[3] - p[0]).normalized() * t
    v = [tuple(x) for x in p] + [tuple(x + nrm) for x in p]
    faces(mat, v, [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])


def cyl(p0, p1, r0, r1, mat, sides=7):
    a, b = Vector(p0), Vector(p1)
    ax = (b - a).normalized()
    t = ax.orthogonal().normalized()
    s = ax.cross(t)
    v = []
    for p, r in ((a, r0), (b, r1)):
        for i in range(sides):
            ang = 2 * math.pi * i / sides
            v.append(tuple(p + (t * math.cos(ang) + s * math.sin(ang)) * r))
    faces(mat, v, [(i, (i + 1) % sides, sides + (i + 1) % sides, sides + i) for i in range(sides)])


def card(c, size, mat):
    a = Vector((rr(-1, 1), rr(-1, 1), rr(-1, 1))).normalized()
    t = a.orthogonal().normalized()
    s = a.cross(t)
    h = size / 2
    c = Vector(c)
    v = [tuple(c + (-t - s) * h), tuple(c + (t - s) * h), tuple(c + (t + s) * h), tuple(c + (-t + s) * h)]
    faces(mat, v, [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]])


def drape(u0, u1, nlo, nhi, lift, mat, su=60, sn=4, zfix=None):
    v, fcs = [], []
    for i in range(su + 1):
        u = u0 + (u1 - u0) * i / su
        a, b = nlo(u), nhi(u)
        for j in range(sn + 1):
            n = a + (b - a) * j / sn
            v.append((u, n, zfix if zfix is not None else zg(u, n) + lift))
    for i in range(su):
        for j in range(sn):
            p = i * (sn + 1) + j
            fcs.append((p, p + sn + 1, p + sn + 2, p + 1))
    faces(mat, v, fcs)


def flush():
    for mat, b in B.items():
        if not b.f:
            continue
        me = bpy.data.meshes.new(mat)
        me.from_pydata(b.v, [], b.f)
        if any(u is not None for u in b.uv):
            layer = me.uv_layers.new(name="UVMap")
            flat = []
            for f, u in zip(b.f, b.uv):
                uvs = u if u is not None else [(0, 0)] * len(f)
                for c in uvs:
                    flat.extend(c)
            layer.data.foreach_set("uv", flat)
        me.update()
        ob = bpy.data.objects.new(mat, me)
        bpy.context.collection.objects.link(ob)
        ob.data.materials.append(MATS[mat])


# ---------------------------------------------------------------- lights
def light(kind, loc, power, color=(1.0, 0.68, 0.42), aim=None, size=0.1, spot=1.0, blend=0.6, name="L"):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = power
    ld.color = color
    if kind == "AREA":
        ld.size = size
    elif kind in ("POINT", "SPOT"):
        ld.shadow_soft_size = size
    if kind == "SPOT":
        ld.spot_size = spot
        ld.spot_blend = blend
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    if aim is not None:
        d = Vector(aim) - Vector(loc)
        ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(ob)
    return ob


WARM = (1.0, 0.64, 0.36)


# ---------------------------------------------------------------- the house
def build_house():
    # interiors: plaster walls, timber floors, furniture, lights
    def room(u0, u1, n0, n1, z0, z1, kind, back_n=None):
        box(u0, u1, n0, n1, z0 - 0.02, z0, "floor")
        poly([(u0, n1, z0), (u1, n1, z0), (u1, n1, z1), (u0, n1, z1)], "plaster")
        poly([(u0, n0, z0), (u0, n1, z0), (u0, n1, z1), (u0, n0, z1)], "plaster")
        poly([(u1, n1, z0), (u1, n0, z0), (u1, n0, z1), (u1, n1, z1)], "plaster")
        um, nm = (u0 + u1) / 2, (n0 + n1) / 2
        w = u1 - u0
        if kind == "living":
            box(um - w * 0.3, um + w * 0.25, n1 - 1.4, n1 - 0.4, z0, z0 + 0.42, "fabric")
            box(um - w * 0.3, um + w * 0.25, n1 - 0.55, n1 - 0.35, z0, z0 + 0.8, "fabric")
            box(um - 0.6, um + 0.6, nm - 0.4, nm + 0.4, z0, z0 + 0.38, "walnut")
            box(um - w * 0.35, um + w * 0.35, nm - 1.5, nm + 1.0, z0, z0 + 0.01, "rug")
            box(um - 0.9, um + 0.9, n1 - 0.03, n1, z0 + 1.4, z0 + 2.4, "art")
        elif kind == "bed":
            box(um - 1.0, um + 1.0, n1 - 2.4, n1 - 0.3, z0, z0 + 0.55, "linen")
            box(um - 1.15, um + 1.15, n1 - 0.32, n1 - 0.2, z0, z0 + 1.3, "fabric")
            for s in (-1, 1):
                box(um + s * 1.5 - 0.25, um + s * 1.5 + 0.25, n1 - 0.7, n1 - 0.2, z0, z0 + 0.5, "walnut")
                box(um + s * 1.5 - 0.1, um + s * 1.5 + 0.1, n1 - 0.55, n1 - 0.35, z0 + 0.5, z0 + 0.85, "lamp")
            box(um - 0.8, um + 0.8, n1 - 0.03, n1, z0 + 1.6, z0 + 2.5, "art")
        elif kind == "dining":
            box(um - 1.4, um + 1.4, nm - 0.5, nm + 0.5, z0 + 0.72, z0 + 0.77, "walnut")
            box(um - 1.3, um - 1.2, nm - 0.4, nm + 0.4, z0, z0 + 0.72, "walnut")
            box(um + 1.2, um + 1.3, nm - 0.4, nm + 0.4, z0, z0 + 0.72, "walnut")
            for k in range(3):
                for s in (-1, 1):
                    cx = um - 0.9 + k * 0.9
                    box(cx - 0.22, cx + 0.22, nm + s * 0.75 - 0.22, nm + s * 0.75 + 0.22, z0, z0 + 0.46, "fabric")
                    box(cx - 0.22, cx + 0.22, nm + s * 0.95 - 0.04, nm + s * 0.95 + 0.04, z0 + 0.46, z0 + 0.95, "fabric")
            box(um - 1.2, um + 1.2, n1 - 0.03, n1, z0 + 1.3, z0 + 2.3, "art")
        light("AREA", (um, nm, z1 - 0.05), 4200 if z1 - z0 < 3.5 else 7000, WARM, aim=(um, nm, z0), size=min(w, n1 - n0) * 0.6, name="room")

    def pendant(x, n, z, strength=30):
        cyl((x, n, z + 0.05), (x, n, z + 1.2), 0.006, 0.006, "black")
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.14, location=(x, n, z))
        bpy.context.active_object.data.materials.append(MATS["bulb"])
        light("POINT", (x, n, z - 0.2), strength, WARM, size=0.1, name="pendant")

    def sconce(x, n, z):
        box(x - 0.06, x + 0.06, n - 0.13, n, z - 0.15, z + 0.15, "black")
        box(x - 0.04, x + 0.04, n - 0.12, n - 0.02, z + 0.15, z + 0.16, "bulb")
        box(x - 0.04, x + 0.04, n - 0.12, n - 0.02, z - 0.16, z - 0.15, "bulb")
        light("SPOT", (x, n - 0.07, z + 0.2), 12, WARM, aim=(x, n - 0.07, z + 3), spot=0.7, blend=0.9, size=0.02, name="sconce")
        light("SPOT", (x, n - 0.07, z - 0.2), 12, WARM, aim=(x, n - 0.07, z - 3), spot=0.7, blend=0.9, size=0.02, name="sconce")

    def frame_bar(p0, p1, n, w=0.07, d=0.1):
        (x0, z0), (x1, z1) = p0, p1
        L = math.hypot(x1 - x0, z1 - z0)
        tx, tz = (x1 - x0) / L, (z1 - z0) / L
        ox, oz = -tz * w / 2, tx * w / 2
        slab((x0 + ox, n, z0 + oz), (x1 + ox, n, z1 + oz), (x1 - ox, n, z1 - oz), (x0 - ox, n, z0 - oz), d, "black")

    def pavilion(u0, u1, n0, n1, low, high):
        um, hw, b = (u0 + u1) / 2, (u1 - u0) / 2, 0.6
        apex = RIDGE - 0.85
        outer = [(u0, -0.6), (u1, -0.6), (u1, EAVE), (um, RIDGE), (u0, EAVE)]
        inner = [(u0 + b, 0), (u1 - b, 0), (u1 - b, EAVE), (um, apex), (u0 + b, EAVE)]
        ring(outer, inner, n0, n0 + 0.35, "stone")
        box(u0, u0 + 0.35, n0 + 0.35, n1 - 0.35, -0.6, EAVE, "stone")
        box(u1 - 0.35, u1, n0 + 0.35, n1 - 0.35, -0.6, EAVE, "stone")
        rw_o = [(u0, -0.6), (u1, -0.6), (u1, EAVE), (um, RIDGE), (u0, EAVE)]
        rw_i = [(um - 1.0, 3.9), (um + 1.0, 3.9), (um + 1.0, 5.9), (um, 5.9), (um - 1.0, 5.9)]
        ring(rw_o, rw_i, n1 - 0.35, n1, "stone")
        poly([(um - 1.0, n1 - 0.15, 3.9), (um + 1.0, n1 - 0.15, 3.9), (um + 1.0, n1 - 0.15, 5.9), (um - 1.0, n1 - 0.15, 5.9)], "glass")
        # glazing + frames
        gn = n0 + 0.22
        poly([(u, gn, z) for u, z in inner], "glass")
        panes = 4
        for k in range(panes + 1):
            x = u0 + b + k * (u1 - u0 - 2 * b) / panes
            top = EAVE + (apex - EAVE) * (1 - abs(x - um) / (hw - b))
            frame_bar((x, 0), (x, max(top, EAVE)), gn - 0.05)
        for zline in (0.03, EAVE):
            frame_bar((u0 + b, zline), (u1 - b, zline), gn - 0.05)
        frame_bar((u0 + b, UP - 0.16), (u1 - b, UP - 0.16), gn - 0.05, w=0.3)
        frame_bar((u0 + b, EAVE), (um, apex), gn - 0.05)
        frame_bar((um, apex), (u1 - b, EAVE), gn - 0.05)
        # floors and rooms
        box(u0 + 0.35, u1 - 0.35, n0 + 0.35, n1 - 0.35, UP - 0.32, UP, "slab")
        room(u0 + 0.35, u1 - 0.35, n0 + 0.35, n1 - 0.35, 0.0, UP - 0.32, low)
        room(u0 + 0.35, u1 - 0.35, n0 + 0.35, n1 - 0.35, UP, EAVE, high)
        # cathedral ceiling under the roof
        for s in (-1, 1):
            xe = um + s * (hw - 0.35)
            poly([(xe, n0 + 0.35, EAVE - 0.05), (xe, n1 - 0.35, EAVE - 0.05), (um, n1 - 0.35, RIDGE - 0.55), (um, n0 + 0.35, RIDGE - 0.55)], "plaster")
        pendant(um, n0 + 2.6, apex - 1.9, 120)
        pendant(um, n0 + 3.0, UP - 1.0, 80)
        # roof planes with overhang and dark barge cappings
        o = 0.4
        ey, ry = EAVE - 0.3, RIDGE + 0.12
        for s in (-1, 1):
            xe = um + s * (hw + o)
            slab((xe, n0 - o, ey), (xe, n1 + o, ey), (um, n1 + o, ry), (um, n0 - o, ry), 0.12, "roof_y")
            for nb in (n0 - o, n1 + o):
                slab((xe, nb, ey - 0.12), (um, nb, ry - 0.12), (um, nb, ry + 0.16), (xe, nb, ey + 0.16), 0.06, "black")
        for x in (u0 + 0.3, u1 - 0.3):
            sconce(x, n0, 2.2)
            sconce(x, n0, UP + 2.0)

    def link(u0, u1, n0, n1):
        gl, gu = n0 + 0.05, n0 + 1.0
        poly([(u0, gl, 0), (u1, gl, 0), (u1, gl, UP - 0.4), (u0, gl, UP - 0.4)], "glass")
        poly([(u0, gu, UP + 0.05), (u1, gu, UP + 0.05), (u1, gu, UP + 2.85), (u0, gu, UP + 2.85)], "glass")
        poly([(u0, n1, 0), (u1, n1, 0), (u1, n1, UP + 2.85), (u0, n1, UP + 2.85)], "glass")
        box(u0, u1, n0 - 0.6, n1, UP - 0.4, UP + 0.05, "coping")
        box(u0, u1, n0 - 0.4, n1 + 0.3, UP + 2.85, UP + 3.15, "coping")
        poly([(u0, n0 - 0.55, UP + 0.05), (u1, n0 - 0.55, UP + 0.05), (u1, n0 - 0.55, UP + 1.05), (u0, n0 - 0.55, UP + 1.05)], "glass")
        for k in range(7):
            x = u0 + k * (u1 - u0) / 6
            frame_bar((x, 0), (x, UP - 0.4), gl - 0.05)
            frame_bar((x, UP + 0.05), (x, UP + 2.85), gu - 0.05)
            frame_bar((x, 0), (x, UP + 2.85), n1 + 0.06)
        room(u0, u1, gl + 0.1, n1 - 0.1, 0.0, UP - 0.4, "dining")
        room(u0, u1, gu + 0.1, n1 - 0.1, UP + 0.05, UP + 2.85, "living")
        for f in (0.25, 0.5, 0.75):
            pendant(u0 + (u1 - u0) * f, n0 + 3.0, UP - 1.2, 80)
        for f in (0.2, 0.4, 0.6, 0.8):  # soffit downlights under the balcony
            x = u0 + (u1 - u0) * f
            box(x - 0.05, x + 0.05, n0 - 0.35, n0 - 0.25, UP - 0.41, UP - 0.4, "bulb")
            light("SPOT", (x, n0 - 0.3, UP - 0.45), 25, WARM, aim=(x, n0 - 0.4, -1), spot=1.0, blend=0.8, size=0.03, name="soffit")
        nm, ey, ry = (n0 + n1) / 2, UP + 3.15, UP + 4.2
        for ne in (n0 - 0.4, n1 + 0.4):
            slab((u0 - 0.1, ne, ey), (u1 + 0.1, ne, ey), (u1 + 0.1, nm, ry), (u0 - 0.1, nm, ry), 0.12, "roof_x")

    pavilion(0.8, 7.8, 10.0, 24.0, "living", "bed")
    pavilion(15.6, 22.6, 10.0, 24.0, "dining", "bed")
    link(7.8, 15.6, 11.0, 23.0)
    box(22.6, 23.8, 17.4, 19.4, -0.6, RIDGE + 1.9, "stone")
    box(22.5, 23.9, 17.3, 19.5, RIDGE + 1.9, RIDGE + 2.05, "black")
    box(23.0, 23.4, 18.2, 18.6, RIDGE + 2.05, RIDGE + 2.4, "black")

    # garage, parapet, hedge planter
    ring([(16.2, GZ - 0.3), (22.6, GZ - 0.3), (22.6, PADZ + 1.0), (16.2, PADZ + 1.0)],
         [(16.9, GZ), (21.9, GZ), (21.9, GZ + 2.45), (16.9, GZ + 2.45)], 2.2, 2.6, "stone")
    box(16.9, 21.9, 2.75, 2.85, GZ, GZ + 2.45, "slats")
    box(16.15, 22.65, 2.15, 2.65, PADZ + 1.0, PADZ + 1.07, "coping")
    box(16.2, 16.55, 2.6, 9.6, GZ - 0.3, PADZ + 1.0, "stone")
    box(22.25, 22.6, 2.6, 9.6, GZ - 0.3, PADZ + 1.0, "stone")
    box(16.2, 22.6, 2.6, 9.6, PADZ - 0.3, PADZ + 0.03, "paving")
    for x in (16.7, 21.8):
        light("SPOT", (x, 2.0, GZ + 0.1), 110, WARM, aim=(x, 2.3, GZ + 3), spot=0.9, blend=0.9, size=0.03, name="garage_up")

    # wall return between the gate and the garage, and side retaining walls along the pad
    box(15.9, 16.6, 0.6, 2.6, GZ - 0.4, top_ := PADZ + 0.95, "stone")
    box(22.6, 22.95, 9.6, 33.0, -3.0, PADZ + 0.9, "stone")
    box(0.0, 0.3, 2.2, 33.0, -3.0, PADZ + 0.9, "stone")
    # street wall with timber gate
    top = PADZ + 0.95
    for a, c in ((0.3, 13.9), (15.9, 16.2)):
        box(a, c, 1.6, 2.2, -3.4, top, "stone")
        box(a - 0.03, c + 0.03, 1.55, 2.25, top, top + 0.08, "coping")
    gg = zg(14.9, 1.4)
    box(13.9, 15.9, 1.6, 2.2, gg + 2.25, top, "stone")
    box(14.0, 15.8, 1.85, 1.95, gg, gg + 2.2, "slats")
    for x in (2.2, 4.6, 7.0, 9.4, 11.8, 13.4):
        g0 = zg(x, 1.1)
        box(x - 0.06, x + 0.06, 1.25, 1.4, g0, g0 + 0.08, "black")
        light("SPOT", (x, 1.3, g0 + 0.12), 150, WARM, aim=(x, 1.62, g0 + 3.2), spot=0.6, blend=1.0, size=0.03, name="wall_up")

    # street-side garden terrace
    box(0.3, 16.2, 2.2, 10.0, PADZ - 0.3, PADZ + 0.03, "paving")
    box(1.2, 7.2, 4.0, 9.0, PADZ + 0.03, PADZ + 0.08, "lawn")
    box(8.3, 15.2, 4.0, 9.0, PADZ + 0.03, PADZ + 0.08, "lawn")
    box(0.6, 13.6, 2.25, 3.1, PADZ + 0.03, PADZ + 0.1, "gravel")

    # north courtyard, pool and cabana
    pu0, pu1, pn0, pn1 = 4.0, 14.0, 27.5, 31.0
    for a, c, n0_, n1_ in ((0.3, 22.9, 24.0, pn0 - 0.3), (0.3, 22.9, pn1 + 0.3, 33.0), (0.3, pu0 - 0.3, pn0 - 0.3, pn1 + 0.3), (pu1 + 0.3, 22.9, pn0 - 0.3, pn1 + 0.3)):
        box(a, c, n0_, n1_, PADZ - 0.3, PADZ + 0.03, "paving")
    for a, c, n0_, n1_ in ((pu0 - 0.3, pu1 + 0.3, pn0 - 0.3, pn0), (pu0 - 0.3, pu1 + 0.3, pn1, pn1 + 0.3), (pu0 - 0.3, pu0, pn0, pn1), (pu1, pu1 + 0.3, pn0, pn1)):
        box(a, c, n0_, n1_, PADZ - 0.2, PADZ + 0.06, "coping")
    box(pu0, pu1, pn0, pn1, PADZ - 1.7, PADZ - 1.6, "pooltile")
    for a, c, n0_, n1_ in ((pu0, pu1, pn0, pn0 + 0.02), (pu0, pu1, pn1 - 0.02, pn1), (pu0, pu0 + 0.02, pn0, pn1), (pu1 - 0.02, pu1, pn0, pn1)):
        box(a, c, n0_, n1_, PADZ - 1.7, PADZ - 0.1, "pooltile")
    poly([(pu0, pn0, PADZ - 0.12), (pu1, pn0, PADZ - 0.12), (pu1, pn1, PADZ - 0.12), (pu0, pn1, PADZ - 0.12)], "water")
    for x in (6.5, 11.5):
        light("POINT", (x, pn0 + 0.15, PADZ - 0.9), 60, (0.55, 0.85, 1.0), size=0.05, name="pool")
    box(0.4, 3.6, 32.6, 32.9, PADZ, PADZ + 2.9, "slats")
    box(0.4, 0.7, 26.0, 32.9, PADZ, PADZ + 2.9, "slats")
    box(0.2, 3.9, 25.6, 33.0, PADZ + 2.9, PADZ + 3.12, "timber")
    box(0.15, 3.95, 25.55, 33.05, PADZ + 3.12, PADZ + 3.2, "black")
    box(0.8, 3.4, 31.6, 32.5, PADZ, PADZ + 0.95, "stone")
    light("AREA", (2.1, 29.3, PADZ + 2.85), 120, WARM, aim=(2.1, 29.3, 0), size=1.5, name="cabana")
    for x in (16.2, 18.4):
        box(x, x + 0.7, 25.3, 27.2, PADZ, PADZ + 0.32, "linen")
    cyl((20.6, 27.5, PADZ), (20.6, 27.5, PADZ + 2.5), 0.03, 0.03, "black")
    rear_top = max([PADZ + 0.9] + [zg(x, 33.8) + 0.2 for x in np.arange(0.3, 22.9, 1.0)])
    box(0.3, 22.9, 33.0, 33.4, PADZ - 0.3, rear_top, "stone")
    box(0.27, 22.93, 32.95, 33.45, rear_top, rear_top + 0.06, "coping")

    # street: road, kerb, footpath, verge garden, driveway (concentric with the real boundary arc)
    R = ARC["r"]
    drape(-45, 68, lambda u: arc_n(u, R - 15.0), lambda u: arc_n(u, R - 4.0), 0.03, "asphalt", 150, 10)
    drape(-45, 68, lambda u: arc_n(u, R - 4.0), lambda u: arc_n(u, R - 3.75), 0.17, "kerb", 150, 1)
    drape(-45, 68, lambda u: arc_n(u, R - 15.25), lambda u: arc_n(u, R - 15.0), 0.17, "kerb", 150, 1)
    drape(-45, 68, lambda u: arc_n(u, R - 16.7), lambda u: arc_n(u, R - 15.25), 0.1, "concrete", 150, 2)
    drape(-45, 68, lambda u: arc_n(u, R - 24.0), lambda u: arc_n(u, R - 16.7), 0.05, "lawn", 150, 4)
    drape(-45, 16.4, lambda u: arc_n(u, R - 3.75), lambda u: arc_n(u, R - 2.3), 0.1, "concrete", 80, 2)
    drape(22.4, 68, lambda u: arc_n(u, R - 3.75), lambda u: arc_n(u, R - 2.3), 0.1, "concrete", 60, 2)
    drape(16.4, 22.4, lambda u: arc_n(u, R - 3.75), lambda u: 2.2, 0.08, "aggregate", 12, 12)
    drape(-1.5, 16.2, lambda u: arc_n(u, R - 2.3), lambda u: 1.58, 0.06, "gravel", 40, 4)
    drape(22.6, 27.0, lambda u: arc_n(u, R - 2.3), lambda u: arc_n(u, R) + 0.4, 0.06, "gravel", 8, 2)


# ---------------------------------------------------------------- vegetation
def tree(base, h, r, leaf, bark, stems=1, cards_per=16, crown_flat=0.5):
    base = Vector(base)
    for s in range(stems):
        lean = Vector((rr(-0.22, 0.22), rr(-0.22, 0.22), 1)).normalized()
        cyl(base, base + lean * h * 0.55, r * 0.07, r * 0.045, bark)
    for _ in range(rng.randint(3, 5)):
        a, d = rr(0, 6.283), rr(r * 0.2, r * 0.9)
        c = base + Vector((math.cos(a) * d, math.sin(a) * d, h * rr(0.5, 0.9)))
        cyl(base + Vector((0, 0, h * 0.5)), c, r * 0.04, r * 0.015, bark, 5)
        sr = r * rr(0.5, 0.75)
        for _ in range(cards_per):
            card(c + Vector((rr(-sr, sr), rr(-sr, sr), rr(-sr * crown_flat, sr * crown_flat))), sr * rr(1.0, 1.5), leaf)


def bush(c, r, leaf, n=26):
    c = Vector(c)
    for _ in range(n):
        d = Vector((rr(-1, 1), rr(-1, 1), rr(-0.7, 0.8))) * r
        card(c + d * 0.7, r * rr(0.7, 1.1), leaf)


def build_vegetation():
    # eucalypts at positions picked from the aerial photo (C2 bushland)
    for gx, gy, r in S["trees"]:
        if rng.random() < 0.3:
            continue
        u, n = to_frame(gx, gy)
        h = min(max(r * 2.8 + rr(2, 5), 7), 17)
        tree((u, n, h_grid(gx, gy) - FFL - 0.2), h, r, "euc", "bark", cards_per=14)
    # sparser eucalypts on the hill beyond the 240 m block (from the 1 km DEM, above 165 m AHD)
    dem = np.array(Image.open(D3 / S["dem_1km"]["file"]), dtype=float)
    e0, n1, cell = S["dem_1km"]["e0"], S["dem_1km"]["n1"], S["dem_1km"]["cell"]
    for r_ in range(0, dem.shape[0], 2):
        for c_ in range(0, dem.shape[1], 2):
            gx, gy = e0 + (c_ + 0.5) * cell - S["E0"] + rr(-3, 3), n1 - (r_ + 0.5) * cell - S["N0"] + rr(-3, 3)
            if 0 <= gx <= NG and 0 <= gy <= NG:
                continue
            u, n = to_frame(gx, gy)
            if n < 30 or dem[r_, c_] < 165 or rng.random() > 0.35:
                continue
            tree((u, n, dem[r_, c_] - FFL - 0.3), rr(8, 14), rr(2.0, 3.5), "euc", "bark", cards_per=10)
    # dry tussocks on the slope behind the lot
    for _ in range(2600):
        u, n = rr(-40, 60), rr(34, 120)
        z = zg(u, n)
        card((u, n, z + 0.25), rr(0.5, 0.9), "tussock")
    # street olives (uplit), terrace olives, courtyard olives
    for u, n, lit in ((3.4, None, True), (10.6, None, True), (12.2, 7.0, False), (4.6, 7.2, False), (21.5, 25.6, False), (1.6, 24.6, False)):
        if n is None:
            n = (arc_n(u, ARC["r"] - 2.3) + 1.58) / 2
            z = zg(u, n)
        else:
            z = PADZ
        tree((u, n, z), rr(3.4, 4.2), rr(1.6, 2.0), "olive", "darkbark", stems=3, cards_per=22, crown_flat=0.6)
        if lit:
            light("SPOT", (u, n - 0.8, z + 0.1), 160, WARM, aim=(u, n, z + 3.5), spot=1.1, blend=0.9, size=0.03, name="olive_up")
    # grey-green shrubs in the verge garden and the planter
    for _ in range(96):
        u = rr(-1.2, 16.0)
        t = rr(0.1, 0.9)
        n = arc_n(u, ARC["r"] - 2.3) * (1 - t) + 1.5 * t
        r = rr(0.3, 0.55)
        bush((u, n, zg(u, n) + r * 0.5), r, "shrub")
    for _ in range(24):
        u, n, r = rr(0.8, 13.4), rr(2.4, 2.95), rr(0.3, 0.45)
        bush((u, n, PADZ + 0.1 + r * 0.5), r, "shrub")
    # boulders
    for u, t in ((1.2, 0.2), (5.6, -0.1), (8.2, -0.2), (12.4, 0.15), (14.6, 0.1)):
        n = (arc_n(u, ARC["r"] - 2.3) + 1.58) / 2 + t
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.55, location=(u, n, zg(u, n) + 0.12))
        ob = bpy.context.active_object
        ob.scale = (rr(1.0, 1.5), rr(0.8, 1.1), rr(0.55, 0.8))
        ob.rotation_euler[2] = rr(0, 3)
        ob.data.materials.append(MATS["boulder"])
        disp = ob.modifiers.new("d", "DISPLACE")
        disp.texture = bpy.data.textures.new("rock", "CLOUDS")
        disp.strength = 0.18
    # hedges along both side boundaries and behind the garage parapet frame the house as in the concept
    lot_sw = [(0.0, 0.0), (-5.1, 32.5)]
    lot_ne = [(22.9, 0.0), (29.8, 44.5)]
    for (ua, na), (ub, nb), inset in ((lot_sw[0], lot_sw[1], 0.9), (lot_ne[0], lot_ne[1], -0.9)):
        for k in range(46):
            t = 0.08 + k / 46 * 0.75
            u, n = ua + (ub - ua) * t + inset, na + (nb - na) * t
            for zz in (0.7, 1.6, 2.4):
                bush((u, n, zg(u, n) + zz), rr(0.75, 0.95), "hedge", 18)


# ---------------------------------------------------------------- terrain
def build_terrain():
    v, uv, fcs = [], [], []
    for j in range(NG):
        for i in range(NG):
            gx, gy = i + 0.5, j + 0.5
            u, n = to_frame(gx, gy)
            z = H[j, i] - FFL
            if 4.0 - 0.25 < u < 14.0 + 0.25 and 27.5 - 0.25 < n < 31.0 + 0.25:
                z = PADZ - 1.8  # sink under the pool
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
    # backdrop from the 1 km DEM, with the near block cut out
    dem = np.array(Image.open(D3 / S["dem_1km"]["file"]), dtype=float)
    e0, n1, cell, nd = S["dem_1km"]["e0"], S["dem_1km"]["n1"], S["dem_1km"]["cell"], dem.shape[0]
    v, fcs, inside = [], [], []
    for r_ in range(nd):
        for c_ in range(nd):
            gx, gy = e0 + (c_ + 0.5) * cell - S["E0"], n1 - (r_ + 0.5) * cell - S["N0"]
            u, n = to_frame(gx, gy)
            v.append((u, n, dem[r_, c_] - FFL - 0.15))
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


# ---------------------------------------------------------------- world, camera, render
VIEWS = {
    "street": dict(eye=(11.7, -32.0, 3.8), at=(11.7, 18.0, 5.6), lens=30),
    "courtyard": dict(eye=(15.0, 47.0, 9.0), at=(11.0, 26.0, 1.0), lens=30),
    "aerial": dict(eye=(48.0, -30.0, 40.0), at=(11.5, 17.0, 0.0), lens=32),
}


def setup_world_and_camera(view, sun_el, sun_az, exposure):
    sc = bpy.context.scene
    world = bpy.data.worlds.new("sky")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_disc = False
    sky.sun_elevation = math.radians(sun_el)
    # direction to the sun in scene axes (north and east expressed in u/n)
    north = Vector((UY, UX, 0))
    east = Vector((UX, NX, 0))
    a = math.radians(sun_az)
    d = (north * math.cos(a) + east * math.sin(a)).normalized()
    sky.sun_rotation = math.atan2(d.x, d.y)
    sky.air_density = 1.0
    sky.dust_density = 0.6
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.35
    # the camera sees a clear dusk gradient (deep blue zenith to warm pale horizon); lighting still uses the physical sky
    out = nt.nodes["World Output"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (*srgb("#e9dccb"), 1)
    ramp.color_ramp.elements[1].position = 0.6
    ramp.color_ramp.elements[1].color = (*srgb("#2c63ae"), 1)
    mid = ramp.color_ramp.elements.new(0.12); mid.color = (*srgb("#9dbcdc"), 1)
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.5; mr.inputs["From Max"].default_value = 0.85
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
    cam_bg = nt.nodes.new("ShaderNodeBackground")
    cam_bg.inputs["Strength"].default_value = 1.0
    nt.links.new(ramp.outputs["Color"], cam_bg.inputs["Color"])
    lp = nt.nodes.new("ShaderNodeLightPath")
    mx = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mx.inputs[0])
    nt.links.new(bg.outputs[0], mx.inputs[1])
    nt.links.new(cam_bg.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs["Surface"])
    sun_dir = Vector((d.x * math.cos(math.radians(sun_el)), d.y * math.cos(math.radians(sun_el)), math.sin(math.radians(sun_el))))
    ld = bpy.data.lights.new("sun", "SUN")
    ld.energy = 4.5
    ld.color = (1.0, 0.55, 0.28)
    ld.angle = math.radians(1.5)
    ob = bpy.data.objects.new("sun", ld)
    ob.rotation_euler = (-sun_dir).to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(ob)

    v = VIEWS[view]
    cam = bpy.data.cameras.new("cam")
    cam.lens = v["lens"]
    cam.sensor_width = 36
    cam.clip_end = 3000
    co = bpy.data.objects.new("cam", cam)
    ex, ey, eh = v["eye"]
    co.location = (ex, ey, zg(ex, ey) + eh)
    co.rotation_euler = (Vector(v["at"]) - co.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(co)
    sc.camera = co
    sc.view_settings.view_transform = "AgX"
    for look in ("AgX - Punchy", "AgX - Medium High Contrast", "None"):
        try:
            sc.view_settings.look = look
            break
        except TypeError:
            continue
    sc.view_settings.exposure = exposure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="street")
    ap.add_argument("--scale", type=int, default=100)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--out", default=str(ROOT / "site_render_street.png"))
    ap.add_argument("--sun-el", type=float, default=6.0)
    ap.add_argument("--sun-az", type=float, default=232.0)
    ap.add_argument("--exposure", type=float, default=0.45)
    ap.add_argument("--blend", default="")
    ap.add_argument("--export-glb", default="")
    a = ap.parse_args()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    make_textures()
    MATS.update({
        "stone": img_mat("stone", "stone.png", 0.25, 0.9, bump=0.6),
        "coping": img_mat("coping", "stone.png", 0.25, 0.75, tint=(1.08, 1.06, 1.02)),
        "paving": img_mat("paving", "paving.png", 0.25, 0.8, bump=0.1),
        "slats": img_mat("slats", "slats.png", 0.25, 0.65, bump=0.2),
        "roof_y": seam_roof("roof_y", "Y"),
        "roof_x": seam_roof("roof_x", "X"),
        "black": flat_mat("black", srgb("#141516"), 0.4, 0.5),
        "glass": thin_glass("glass"),
        "slab": flat_mat("slab", srgb("#e8e2d6"), 0.7),
        "floor": flat_mat("floor", srgb("#8a6646"), 0.35),
        "plaster": flat_mat("plaster", srgb("#ece3d4"), 0.85),
        "fabric": flat_mat("fabric", srgb("#d9cfbf"), 0.95),
        "linen": flat_mat("linen", srgb("#f1ece3"), 0.95),
        "walnut": flat_mat("walnut", srgb("#4a3424"), 0.45),
        "rug": flat_mat("rug", srgb("#b8aa95"), 1.0),
        "art": flat_mat("art", srgb("#3d4a4f"), 0.6),
        "lamp": flat_mat("lamp", (1, 1, 1), 0.5, emit=WARM, strength=12),
        "bulb": flat_mat("bulb", (1, 1, 1), 0.5, emit=WARM, strength=25),
        "lawn": flat_mat("lawn", srgb("#5f7a3a"), 0.95),
        "gravel": noisy_mat("gravel", "#6f6355", "#a89a85", 60, 1.0, 0.5),
        "concrete": noisy_mat("concrete", "#aaa499", "#c9c3b8", 8, 0.9, 0.05),
        "kerb": flat_mat("kerb", srgb("#b6b0a5"), 0.9),
        "asphalt": noisy_mat("asphalt", "#2a2b2d", "#4a4a4b", 90, 0.9, 0.3),
        "aggregate": noisy_mat("aggregate", "#8f877a", "#c2b9aa", 70, 0.9, 0.25),
        "timber": flat_mat("timber", srgb("#7a5a3e"), 0.6),
        "pooltile": flat_mat("pooltile", srgb("#8fd0e3"), 0.3),
        "boulder": flat_mat("boulder", srgb("#b8a68a"), 0.95),
        "bark": noisy_mat("bark", "#8a8070", "#b0a796", 6, 0.85, 0.2),
        "darkbark": flat_mat("darkbark", srgb("#4d443b"), 0.9),
        "euc": leaf_mat("euc"), "olive": leaf_mat("olive"), "shrub": leaf_mat("shrub"), "hedge": leaf_mat("hedge"), "tussock": leaf_mat("tussock"),
        "terrain": terrain_mat("terrain", True), "grass": terrain_mat("grass", False),
    })
    w = MATS["water"] = bpy.data.materials.new("water")
    w.use_nodes = True
    b = w.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.75, 0.95, 1.0, 1)
    b.inputs["Transmission Weight"].default_value = 1.0
    b.inputs["IOR"].default_value = 1.33
    b.inputs["Roughness"].default_value = 0.03

    build_terrain()
    build_house()
    build_vegetation()
    flush()
    if a.export_glb:
        import farmhouse_cycles  # shares the GLB export used by the web viewer
        farmhouse_cycles.export_glb(a.export_glb)
        print("exported", a.export_glb)
        return
    setup_world_and_camera(a.view, a.sun_el, a.sun_az, a.exposure)

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = a.samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.03
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    sc.cycles.transparent_max_bounces = 12
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.resolution_x, sc.render.resolution_y = 1414, 913
    sc.render.resolution_percentage = a.scale
    sc.render.filepath = a.out
    sc.render.image_settings.file_format = "PNG"
    if a.blend:
        bpy.ops.wm.save_as_mainfile(filepath=a.blend)
    bpy.ops.render.render(write_still=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
