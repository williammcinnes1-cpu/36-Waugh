"""Shared photographic polish for the design renders (sage_stone, wraparound, black_barn).

Golden-hour sky, shadow-free fill light, shader bevels on hard edges, neighbouring houses from the cadastre,
a mature street tree, proper outdoor chairs and final render settings. Import after cycles_render.
"""
import math

import bpy
from mathutils import Vector

import cycles_render as cr
from cycles_render import MATS, box, cyl, faces, flat_mat, light, noisy_mat, slab, srgb, tree, bush, zg, WARM

# neighbour footprints in the street frame (u, n), traced from the 2014 aerial and data/neighbour_lots.geojson
NB_NE = [(26.6, 16.6), (38.5, 10.2), (43.0, 21.0), (28.1, 28.6)]        # lot 625: white render, terracotta hip
NB_NE2 = [(29.0, 5.2), (34.6, 2.6), (36.6, 7.4), (31.0, 10.0)]          # its front wing / carport
NB_SW = [(-28.0, 12.0), (-11.0, 3.5), (-7.5, 13.0), (-24.0, 22.0)]     # lot 623: cream render, grey tile hip
NB_ACROSS = [(-2.0, -36.0), (19.0, -36.0), (19.0, -25.0), (-2.0, -25.0)]


def golden_sky(strength=0.6, horizon="#f3bf7e", mid="#f2d2a8", zenith="#8fa9cc"):
    """warm golden-hour gradient with soft lit clouds for camera rays (lighting still from the physical sky + sun)"""
    nt = bpy.context.scene.world.node_tree
    ramp = [n for n in nt.nodes if n.type == "VALTORGB"][0]
    els = sorted(ramp.color_ramp.elements, key=lambda e: e.position)
    els[0].color = (*srgb(horizon), 1)
    els[1].position = 0.2
    els[1].color = (*srgb(mid), 1)
    els[2].position = 0.75
    els[2].color = (*srgb(zenith), 1)
    ramp.color_ramp.interpolation = "EASE"
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
    bg.inputs["Strength"].default_value = strength


def fill_light(view, energy=0.9, color=(0.86, 0.9, 1.0)):
    """shadowless sun from behind the camera: lifts the shaded walls like bounce light off the street and sky"""
    v = cr.VIEWS[view]
    ex, ey, _ = v["eye"]
    ax, ay, _ = v["at"]
    d = Vector((ex - ax, ey - ay, 0)).normalized()
    d = Vector((d.x, d.y, 0.7)).normalized()
    ld = bpy.data.lights.new("fill", "SUN")
    ld.energy = energy
    ld.color = color
    ld.angle = math.radians(20)
    ld.use_shadow = False
    ob = bpy.data.objects.new("fill", ld)
    ob.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(ob)


def bevel(names, radius=0.018):
    """rounded hard edges (walls, trims, copings) via the Cycles bevel shader"""
    for name in names:
        m = MATS.get(name)
        if m is None or not m.use_nodes:
            continue
        nt = m.node_tree
        bs = [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"]
        if not bs:
            continue
        b = bs[0]
        bv = nt.nodes.new("ShaderNodeBevel")
        bv.samples = 6
        bv.inputs["Radius"].default_value = radius
        bumps = [n for n in nt.nodes if n.type == "BUMP"]
        if bumps:
            last = [n for n in bumps if any(l.to_node == b for l in n.outputs["Normal"].links)]
            first = [n for n in bumps if not n.inputs["Normal"].is_linked]
            tgt = first[0] if first else (last[0] if last else bumps[0])
            nt.links.new(bv.outputs["Normal"], tgt.inputs["Normal"])
        else:
            nt.links.new(bv.outputs["Normal"], b.inputs["Normal"])


def _hip(q, wall, roof, h, pitch, base=None, o=0.55):
    P = [Vector((u, n, 0)) for u, n in q]
    if (P[1] - P[0]).length < (P[2] - P[1]).length:
        P = P[1:] + P[:1]
    if base is None:
        base = min(zg(p.x, p.y) for p in P) + 0.15
    faces(wall, [(p.x, p.y, base - 1.5) for p in P] + [(p.x, p.y, base + h) for p in P],
          [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    cen = sum(P, Vector()) / 4
    ax_ = (P[1] - P[0]).normalized()
    bx_ = (P[3] - P[0]).normalized()
    hl = (P[1] - P[0]).length / 2 + o
    hs = (P[3] - P[0]).length / 2 + o
    ze = base + h - o * math.tan(math.radians(pitch))
    zr = ze + hs * math.tan(math.radians(pitch))
    E = [cen + ax_ * sl * hl + bx_ * sw * hs for sl, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    R0, R1 = cen - ax_ * (hl - hs), cen + ax_ * (hl - hs)
    v = [(e.x, e.y, ze) for e in E] + [(R0.x, R0.y, zr), (R1.x, R1.y, zr)]
    faces(roof, v, [(0, 1, 5, 4), (1, 2, 5), (2, 3, 4, 5), (3, 0, 4)])
    for i in range(4):                       # fascia + gutter line
        a, b = E[i], E[(i + 1) % 4]
        cyl((a.x, a.y, ze - 0.06), (b.x, b.y, ze - 0.06), 0.07, 0.07, "nb_trim", 6)
    # windows: dark glass with white frames on each wall, spaced along the walls
    for i in range(4):
        a, b = P[i], P[(i + 1) % 4]
        L = (b - a).length
        t = (b - a).normalized()
        out = Vector((t.y, -t.x, 0))
        if (a + t * L / 2 + out - cen).length < (a + t * L / 2 - cen).length:
            out = -out
        k = max(1, int(L / 4.5))
        for j in range(k):
            c = a + t * (L * (j + 0.5) / k) + out * 0.03
            w = 0.75
            p0, p1 = c - t * w, c + t * w
            faces("nb_glass", [(p0.x, p0.y, base + 0.9), (p1.x, p1.y, base + 0.9), (p1.x, p1.y, base + 2.1), (p0.x, p0.y, base + 2.1)],
                  [(0, 1, 2, 3)])
            for zz in (0.85, 2.1):
                cyl((p0.x, p0.y, base + zz), (p1.x, p1.y, base + zz), 0.035, 0.035, "nb_trim", 4)
    return base, zr


LOT_625 = [(22.9, 0.0), (40.2, -4.8), (61.5, 30.5), (29.8, 44.5)]
LOT_623 = [(0.0, 0.0), (-24.5, -8.3), (-40.0, 20.5), (-5.1, 32.5)]


def _inside(u, n, pts):
    ins = False
    for (ua, na), (ub, nb) in zip(pts, pts[1:] + pts[:1]):
        if (na > n) != (nb > n) and u < (ub - ua) * (n - na) / (nb - na) + ua:
            ins = not ins
    return ins


def lawn_patch(pts, mat="lawn", step=1.0, lift=0.05, shrink=0.35):
    """mown lawn draped on the terrain over a polygon (u, n), cell by cell"""
    us = [p[0] for p in pts]
    ns = [p[1] for p in pts]
    cen = (sum(us) / len(us), sum(ns) / len(ns))
    q = [(cen[0] + (u - cen[0]) * (1 - shrink / 20), cen[1] + (n - cen[1]) * (1 - shrink / 20)) for u, n in pts]
    v, f, idx = [], [], {}
    u0, n0 = math.floor(min(us)), math.floor(min(ns))
    for i in range(int((max(us) - u0) / step) + 2):
        for j in range(int((max(ns) - n0) / step) + 2):
            cs = [(u0 + (i + a) * step, n0 + (j + b) * step) for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))]
            if not all(_inside(cu, cn, q) for cu, cn in cs):
                continue
            ids = []
            for c in cs:
                if c not in idx:
                    idx[c] = len(v)
                    v.append((c[0], c[1], zg(*c) + lift))
                ids.append(idx[c])
            f.append(ids)
    if f:
        faces(mat, v, f)


def neighbours(across=True):
    """lot 625 (NE): single-storey white render, terracotta hip roof; lot 623 (SW): cream render, grey tile roof"""
    MATS.setdefault("nb_render", noisy_mat("nb_render", "#e9e5dd", "#f4f1eb", 6, 0.85, 0.05))
    MATS.setdefault("nb_cream", noisy_mat("nb_cream", "#ddd3c2", "#e9e1d3", 6, 0.85, 0.05))
    MATS.setdefault("nb_terracotta", noisy_mat("nb_terracotta", "#8e3f25", "#b45c38", 14, 0.65, 0.25))
    MATS.setdefault("nb_tile", noisy_mat("nb_tile", "#4d4f51", "#66686a", 14, 0.6, 0.2))
    MATS.setdefault("nb_trim", flat_mat("nb_trim", srgb("#f2f0ea"), 0.6))
    MATS.setdefault("nb_glass", flat_mat("nb_glass", srgb("#2a3236"), 0.08, emit=WARM, strength=0.6))
    _hip(NB_NE, "nb_render", "nb_terracotta", 2.8, 23)
    _hip(NB_NE2, "nb_render", "nb_terracotta", 2.7, 23)
    _hip(NB_SW, "nb_cream", "nb_tile", 2.8, 22)
    if across:
        _hip(NB_ACROSS, "nb_render", "nb_tile", 5.9, 24)
    # their gardens: mown lawn, a hedge on the street side, and trees
    lawn_patch(LOT_625)
    lawn_patch(LOT_623)
    if across:
        R = cr.ARC["r"]
        cr.drape(-45, 68, lambda u: cr.arc_n(u, R - 50.0), lambda u: cr.arc_n(u, R - 24.0), 0.05, "lawn", 60, 12)
    for k in range(18):
        u, n = 30.5 + k * 0.85, 1.6 - k * 0.24
        bush((u, n, zg(u, n) + 0.45), 0.5, "hedge", 14)
    for k in range(16):
        u, n = -2.5 - k * 1.1, 1.2 - k * 0.37
        bush((u, n, zg(u, n) + 0.45), 0.5, "hedge", 14)
    for u, n, h, r in ((33.0, 24.0, 5.5, 2.4), (40.0, 6.0, 6.5, 2.8), (-14.0, 14.0, 6.0, 2.6), (-20.0, 1.0, 4.5, 2.0)):
        tree((u, n, zg(u, n) - 0.1), h, r, "olive", "darkbark", stems=2, cards_per=26, crown_flat=0.6)


def street_tree(u, n, h=9.0, r=3.4, leaf="euc"):
    """a mature street tree on the nature strip"""
    z = zg(u, n)
    tree((u, n, z - 0.1), h, r, leaf, "bark", stems=1, cards_per=34, crown_flat=0.55)
    cyl((u, n, z - 0.1), (u, n, z + h * 0.45), r * 0.11, r * 0.07, "bark", 9)


def outdoor_chair(x, n, z, ang, frame="teak", cushion="oatmeal"):
    """slatted timber lounge chair with a seat and back cushion; ang = direction the sitter faces (radians)"""
    f = Vector((math.cos(ang), math.sin(ang), 0))
    s = Vector((-f.y, f.x, 0))
    c = Vector((x, n, z))

    def P(a, b, h):            # a forward, b sideways, h up
        p = c + f * a + s * b
        return (p.x, p.y, z + h)

    W, D = 0.34, 0.36
    for a in (D, -D):         # legs
        for b in (W, -W):
            cyl(P(a, b, 0), P(a, b, 0.42), 0.028, 0.028, frame, 6)
    for b in (W, -W):         # arms
        slab(P(D + 0.04, b - 0.04, 0.6), P(-D - 0.08, b - 0.04, 0.6), P(-D - 0.08, b + 0.04, 0.6), P(D + 0.04, b + 0.04, 0.6), 0.035, frame)
        cyl(P(D, b, 0.42), P(D, b, 0.6), 0.028, 0.028, frame, 6)
        cyl(P(-D, b, 0.42), P(-D, b, 0.6), 0.028, 0.028, frame, 6)
    for k in range(6):        # seat slats
        a = -D + 0.06 + k * (2 * D - 0.12) / 5
        slab(P(a - 0.035, -W, 0.38), P(a - 0.035, W, 0.38), P(a + 0.035, W, 0.38), P(a + 0.035, -W, 0.38), 0.025, frame)
    for k in range(5):        # reclined back slats
        b = -W + 0.05 + k * (2 * W - 0.1) / 4
        slab(P(-D, b - 0.035, 0.4), P(-D - 0.2, b - 0.035, 0.95), P(-D - 0.2, b + 0.035, 0.95), P(-D, b + 0.035, 0.4), 0.025, frame)
    slab(P(-D + 0.02, -W + 0.03, 0.41), P(-D + 0.02, W - 0.03, 0.41), P(D - 0.02, W - 0.03, 0.41), P(D - 0.02, -W + 0.03, 0.41), 0.09, cushion)
    slab(P(-D + 0.06, -W + 0.04, 0.5), P(-D - 0.12, -W + 0.04, 0.92), P(-D - 0.12, W - 0.04, 0.92), P(-D + 0.06, W - 0.04, 0.5), 0.08, cushion)


def lounger(x, n, z, ang, frame="teak", cushion="oatmeal"):
    """timber chaise lounger: slatted frame on short legs, thick cushion, raised back; ang = direction from foot to head"""
    f = Vector((math.cos(ang), math.sin(ang), 0))
    s = Vector((-f.y, f.x, 0))
    c = Vector((x, n, z))

    def P(a, b, h):
        p = c + f * a + s * b
        return (p.x, p.y, z + h)

    W, Lh = 0.33, 1.0
    for a in (0.08, 1.85):
        for b in (W - 0.04, -W + 0.04):
            cyl(P(a, b, 0), P(a, b, 0.26), 0.025, 0.025, frame, 6)
    for b in (W, -W):
        slab(P(0, b - 0.025, 0.24), P(2.0, b - 0.025, 0.24), P(2.0, b + 0.025, 0.24), P(0, b + 0.025, 0.24), 0.08, frame)
    for a in (0.0, 2.0):
        slab(P(a - 0.025, -W, 0.24), P(a - 0.025, W, 0.24), P(a + 0.025, W, 0.24), P(a + 0.025, -W, 0.24), 0.08, frame)
    slab(P(0.04, -W + 0.03, 0.32), P(0.04, W - 0.03, 0.32), P(1.35, W - 0.03, 0.32), P(1.35, -W + 0.03, 0.32), 0.08, cushion)
    slab(P(1.35, -W + 0.03, 0.32), P(1.35, W - 0.03, 0.32), P(1.95, W - 0.03, 0.72), P(1.95, -W + 0.03, 0.72), 0.08, cushion)
    cyl(P(1.8, -W + 0.06, 0.3), P(1.8, -W + 0.06, 0.62), 0.02, 0.02, frame, 5)
    cyl(P(1.8, W - 0.06, 0.3), P(1.8, W - 0.06, 0.62), 0.02, 0.02, frame, 5)


def dining_chair(x, n, z, ang, frame="teak", cushion="oatmeal"):
    """timber outdoor dining chair with slatted back and seat pad; ang = direction the sitter faces"""
    f = Vector((math.cos(ang), math.sin(ang), 0))
    s = Vector((-f.y, f.x, 0))
    c = Vector((x, n, z))

    def P(a, b, h):
        p = c + f * a + s * b
        return (p.x, p.y, z + h)

    W, D = 0.23, 0.22
    for a in (D, -D):
        for b in (W, -W):
            cyl(P(a, b, 0), P(a, b, 0.45 if a > 0 else 0.9), 0.02, 0.02, frame, 6)
    slab(P(-D, -W, 0.43), P(-D, W, 0.43), P(D, W, 0.43), P(D, -W, 0.43), 0.04, frame)
    slab(P(-D + 0.02, -W + 0.02, 0.47), P(-D + 0.02, W - 0.02, 0.47), P(D - 0.02, W - 0.02, 0.47), P(D - 0.02, -W + 0.02, 0.47), 0.05, cushion)
    for h in (0.62, 0.74, 0.86):
        slab(P(-D - 0.01, -W, h), P(-D - 0.01, W, h), P(-D - 0.01, W, h + 0.06), P(-D - 0.01, -W, h + 0.06), -0.025, frame)


def chair_mats():
    MATS.setdefault("teak", noisy_mat("teak", "#8a7a64", "#a8977d", 5, 0.7, 0.1))
    MATS.setdefault("oatmeal", noisy_mat("oatmeal", "#d9d0bf", "#e6dfd1", 40, 0.95, 0.15))


def render(a, adaptive=0.012):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = a.samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = adaptive
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    sc.cycles.denoising_prefilter = "ACCURATE"
    sc.cycles.max_bounces = 6
    sc.cycles.diffuse_bounces = 3
    sc.cycles.transparent_max_bounces = 16
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.blur_glossy = 1.0
    sc.cycles.sample_clamp_indirect = 8.0
    sc.render.resolution_x, sc.render.resolution_y = 1414, 913
    sc.render.resolution_percentage = a.scale
    sc.render.filepath = a.out
    sc.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    print("wrote", a.out)


def backdrop_trees(seed=7, count=30):
    """a loose belt of mature trees behind and beside the three lots so no view ends in bare paddock"""
    import random
    R = random.Random(seed)
    placed = 0
    tries = 0
    while placed < count and tries < 2000:
        tries += 1
        u, n = R.uniform(-45, 65), R.uniform(8, 75)
        rear = 32.5 + 0.3438 * (u + 5.1)                 # our rear boundary line extended
        if n < rear + 2.5 and -6 < u < 31:
            continue                                      # inside our lot
        if _inside(u, n, NB_NE) or _inside(u, n, NB_SW) or _inside(u, n, NB_NE2):
            continue
        if min(math.hypot(u - p[0], n - p[1]) for p in NB_NE + NB_SW) < 3.0:
            continue
        h = R.uniform(6.0, 11.0)
        leaf = R.choice(("euc", "olive", "euc"))
        tree((u, n, zg(u, n) - 0.1), h, h * R.uniform(0.32, 0.4), leaf, "bark" if leaf == "euc" else "darkbark",
             stems=1 if leaf == "euc" else 2, cards_per=22, crown_flat=0.6)
        placed += 1


def sun(energy=6.0, color=(1.0, 0.6, 0.32), angle=1.0):
    """stronger, warmer golden-hour key (the physical sky still lights the shadows)"""
    for ld in bpy.data.lights:
        if ld.name.startswith("sun"):
            ld.energy = energy
            ld.color = color
            ld.angle = math.radians(angle)
