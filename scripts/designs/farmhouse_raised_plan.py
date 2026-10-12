"""Indicative floor plans for the farmhouse raised-pool variant (scripts/designs/farmhouse_raised_cycles.py).

Three levels side by side in the street frame (street at the bottom): garage, ground floor (with the lower courtyard
and the raised pool deck), upper floor (with the pool house). Walls and openings follow the 3D model; the room layout
inside them is a concept only.

    python3 scripts/designs/farmhouse_raised_plan.py [out.png]
"""
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
F = lambda s, b=False: ImageFont.truetype(BOLD if b else FONT, s)

S = 20                       # px per metre
U0, N1 = -6.5, 46.5          # panel window: u from U0, n down from N1
PW, PH = int(38 * S), int(48 * S)
LOT = [(0.0, 0.0), (22.9, 0.0), (29.8, 44.5), (-5.1, 32.5)]
FFL, PADZ, DECK, UPPER, GARAGE = 157.6, 157.45, 159.25, 160.6, 155.0

ROOM, WET, OUT, DECKC, POOL, ROOFC = (250, 249, 245), (226, 236, 242), (236, 233, 226), (214, 232, 200), (120, 190, 225), (205, 205, 205)


def arc_front():
    cu, cn, r = 11.45, -74.10, 75.0
    a0, a1 = math.atan2(0 - cn, 0 - cu), math.atan2(0 - cn, 22.9 - cu)
    return [(cu + r * math.cos(a0 + (a1 - a0) * k / 20), cn + r * math.sin(a0 + (a1 - a0) * k / 20)) for k in range(21)]


class Panel:
    def __init__(self, sheet, ox, oy, title, rl):
        self.d = ImageDraw.Draw(sheet)
        self.ox, self.oy = ox, oy
        self.d.text((ox + 10, oy - 52), title, font=F(26, True), fill=(25, 25, 25))
        self.d.text((ox + 10, oy - 20), rl, font=F(16), fill=(90, 90, 90))
        lot = [LOT[0]] + arc_front()[1:] + [LOT[2], LOT[3]]
        self.d.line([self.p(u, n) for u, n in lot + [lot[0]]], fill=(200, 40, 40), width=2)

    def p(self, u, n):
        return (self.ox + (u - U0) * S, self.oy + (N1 - n) * S)

    def rect(self, u0, u1, n0, n1, fill, outline=None, w=0):
        a, b = self.p(u0, n1), self.p(u1, n0)
        self.d.rectangle([a, b], fill=fill, outline=outline, width=w)

    def room(self, name, u0, u1, n0, n1, fill=ROOM, area=True, size=14, at=None):
        self.rect(u0, u1, n0, n1, fill, (40, 40, 40), 3)
        cx, cy = self.p(*(at or ((u0 + u1) / 2, (n0 + n1) / 2)))
        sep = "\n" if n1 - n0 > 2.2 else " "
        txt = name + (f"{sep}{(u1 - u0) * (n1 - n0):.0f} m²" if area else "")
        self.d.multiline_text((cx, cy), txt, font=F(size), fill=(30, 30, 30), anchor="mm", align="center")

    def envelope(self, u0, u1, n0, n1):
        self.rect(u0, u1, n0, n1, None, (15, 15, 15), 6)

    def outdoor(self, name, u0, u1, n0, n1, fill=OUT, size=13, dashed=False):
        self.rect(u0, u1, n0, n1, fill, (150, 150, 140), 1)
        if name:
            cx, cy = self.p((u0 + u1) / 2, (n0 + n1) / 2)
            self.d.multiline_text((cx, cy), name, font=F(size), fill=(70, 70, 60), anchor="mm", align="center")

    def poly(self, pts, fill, outline=(150, 150, 140)):
        self.d.polygon([self.p(*q) for q in pts], fill=fill, outline=outline)

    def stair(self, u0, u1, n0, n1, along="n", label="", up=True):
        self.rect(u0, u1, n0, n1, (245, 245, 240), (60, 60, 60), 2)
        steps = int(((n1 - n0) if along == "n" else (u1 - u0)) / 0.28)
        for k in range(1, steps):
            if along == "n":
                n = n0 + k * (n1 - n0) / steps
                self.d.line([self.p(u0, n), self.p(u1, n)], fill=(120, 120, 120), width=1)
            else:
                u = u0 + k * (u1 - u0) / steps
                self.d.line([self.p(u, n0), self.p(u, n1)], fill=(120, 120, 120), width=1)
        if label:
            cx, cy = self.p((u0 + u1) / 2, (n0 + n1) / 2)
            self.d.text((cx, cy), label, font=F(11, True), fill=(40, 40, 40), anchor="mm")

    def glass_n(self, u0, u1, n, w=5):     # glazing along a wall of constant n
        self.d.line([self.p(u0, n), self.p(u1, n)], fill=(40, 130, 200), width=w)

    def glass_u(self, n0, n1, u, w=5):
        self.d.line([self.p(u, n0), self.p(u, n1)], fill=(40, 130, 200), width=w)

    def door_n(self, u0, u1, n):
        self.d.line([self.p(u0, n), self.p(u1, n)], fill=(150, 95, 50), width=6)

    def note(self, u, n, txt, size=12, fill=(90, 90, 90)):
        self.d.text(self.p(u, n), txt, font=F(size), fill=fill, anchor="mm")


def main(out=ROOT / "designs" / "farmhouse_raised_floorplan.png"):
    W, H = 3 * PW + 4 * 30 + 330, PH + 200
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    oy = 110
    xs = [30, 60 + PW, 90 + 2 * PW]
    d = ImageDraw.Draw(sheet)
    d.text((30, 18), "36 Waugh Street, Griffith — Modern Farmhouse, raised-pool variant: indicative floor plans",
           font=F(30, True), fill=(20, 20, 20))

    # ---------------- garage level
    g = Panel(sheet, xs[0], oy, "Garage level", f"RL {GARAGE:.1f} m AHD (street level)")
    for r in ((1, 15.6, 9.4, 21.4), (15.6, 22.8, 7.5, 22.6), (1, 9, 17.5, 32.3)):
        g.rect(*r, None, (180, 180, 180), 1)
    g.note(8.0, 16.0, "ground floor above\n(on fill / slab)", 13)
    g.outdoor("Driveway\n(new crossover)", 16.4, 22.6, 0.6, 6.3, (225, 225, 222))
    g.outdoor("Front lawn & planted bank", 3.6, 16.0, 1.5, 6.6, DECKC, 12)
    g.stair(1.6, 3.4, 1.0, 5.4, "n", "UP")
    g.note(2.5, 0.3, "gate", 11)
    g.room("Double garage", 16.4, 22.6, 6.3, 11.3, size=15)
    g.stair(16.4, 18.4, 11.3, 13.8, "n", "UP")
    g.room("Store", 18.4, 22.6, 11.3, 13.8, size=12)
    g.envelope(16.4, 22.6, 6.3, 13.8)
    g.door_n(17.0, 22.0, 6.3)
    g.note(19.5, 5.7, "garage door 5.5 m+ from boundary", 10)

    # ---------------- ground floor
    p = Panel(sheet, xs[1], oy, "Ground floor", f"RL {FFL:.1f} · lower courtyard {PADZ:.2f} · pool deck & rear lawn {DECK:.2f}")
    p.outdoor("Front lawn", 3.6, 16.0, 1.5, 6.6, DECKC, 12)
    p.stair(1.6, 3.4, 1.0, 5.4, "n")
    p.outdoor("Driveway", 16.4, 22.6, 0.6, 6.3, (225, 225, 222), 11)
    # raised rear level (pool deck, lawn) first, then the lower courtyard over it
    p.poly([LOT[0]] + arc_front()[1:] + [LOT[2], LOT[3]], DECKC, None)              # lawn to the boundaries
    p.outdoor("", 11.0, 23.5, 26.8, 33.2, (232, 230, 222))
    p.rect(12.5, 22.5, 28.5, 32.0, POOL, (30, 90, 140), 2)
    p.note(17.5, 30.25, "Pool  10.0 × 3.5 m", 13, (20, 50, 80))
    p.note(17.5, 34.6, f"Pool deck & lawn  RL {DECK:.2f}", 13, (50, 80, 40))
    p.outdoor("Outdoor\nroom", 22.8, 26.0, 25.0, 31.0, (222, 214, 200), 11)
    p.outdoor("Pergola", 16.1, 19.7, 36.75, 38.95, (222, 214, 200), 11)
    p.note(4.0, 34.0, "raised lawn", 11, (50, 80, 40))
    p.outdoor("", 9.0, 27.5, 21.4, 26.8, (232, 230, 224))
    p.d.multiline_text(p.p(24.6, 23.6), f"Lower\ncourtyard\nRL {PADZ:.2f}", font=F(12), fill=(70, 70, 60), anchor="mm", align="center")
    p.rect(11.0, 13.4, 23.0, 24.0, (190, 160, 120), (90, 70, 50), 1)
    p.stair(14.4, 17.6, 24.4, 26.8, "n", "UP 1.8 m")
    p.rect(18.4, 22.4, 26.1, 26.8, (170, 165, 155), (60, 60, 60), 2)
    p.note(20.4, 25.6, "BBQ bench", 11)
    p.outdoor("Verandah", 0.8, 15.4, 7.0, 9.4, (230, 222, 205), 12)
    rooms = [("Study", 1, 5.2, 9.4, 14.0), ("Entry", 5.2, 10.0, 9.4, 14.0, ROOM, True, 14, (6.6, 11.2)), ("Media", 10.0, 15.6, 9.4, 14.0),
             ("Living", 1, 8.0, 14.0, 21.4), ("Kitchen / dining", 8.0, 15.6, 14.0, 21.4),
             ("Bed 5 / guest", 15.6, 22.8, 7.5, 11.3), ("Mud", 15.6, 18.5, 11.3, 15.5, ROOM, True, 11, (17.05, 14.65)),
             ("Rumpus", 15.6, 22.8, 18.0, 22.6), ("Gym\n(under pool house)", 1, 9, 21.4, 26.8),
             ("Cellar / store", 1, 9, 26.8, 32.3)]
    for r in rooms:
        p.room(*r)
    for name, *r in (("Ensuite", 18.5, 22.8, 11.3, 14.0), ("Linen", 18.5, 22.8, 14.0, 15.5), ("Pantry", 15.6, 18.5, 15.5, 18.0),
                     ("WC", 18.5, 20.5, 15.5, 18.0), ("Store", 20.5, 22.8, 15.5, 18.0)):
        p.room(name, *r, fill=WET, size=11)
    p.room("Pool\nplant", 9.0, 10.6, 23.2, 26.8, fill=WET, area=False, size=10)
    p.stair(8.0, 10.0, 11.5, 14.0, "n", "UP")
    p.stair(16.4, 18.4, 11.3, 13.8, "n", "DN")
    p.rect(10.0, 13.6, 15.4, 16.3, (215, 210, 200), (90, 90, 90), 1)   # island
    p.rect(0.3, 1.0, 14.0, 15.4, (200, 190, 170), (90, 90, 90), 1)    # fireplace / chimney
    for r in ((1, 15.6, 9.4, 21.4), (15.6, 22.8, 7.5, 22.6), (1, 9, 21.4, 32.3)):
        p.envelope(*r)
    p.glass_n(9.3, 15.3, 21.4, 7)
    p.glass_n(16.6, 22.0, 22.6)
    p.glass_n(2.0, 4.2, 9.4)
    p.glass_n(11.4, 14.4, 9.4)
    p.glass_n(17.0, 21.4, 7.5)
    p.door_n(5.95, 7.65, 9.4)
    p.note(6.8, 8.6, "front door", 10)

    # ---------------- upper floor
    q = Panel(sheet, xs[2], oy, "Upper floor & pool house", f"RL {UPPER:.1f} m AHD · pool deck below at {DECK:.2f}")
    for r in ((1, 15.6, 9.4, 21.4), (15.6, 22.8, 7.5, 22.6)):
        q.rect(*r, None, (190, 190, 190), 1)
    q.poly([(9, 17.5), (22.8, 17.5), (22.8, 23.05), (9, 23.05)], ROOFC)
    q.note(16.0, 20.3, "roof below (lowered rear roof)", 12, (80, 80, 80))
    q.rect(12.5, 22.5, 28.5, 32.0, (190, 220, 235), (120, 160, 190), 1)
    q.note(17.5, 30.25, f"pool (deck RL {DECK:.2f})", 12, (60, 90, 120))
    rooms = [("Bed 3", 1, 4.6, 11.5, 16.0), ("Bed 4", 4.6, 8.0, 11.5, 16.0), ("Bed 2", 10.0, 13.2, 11.5, 16.0),
             ("Master bed\n(views E + street)", 15.6, 22.8, 11.5, 15.0), ("Hall", 1, 15.6, 16.0, 17.5),
             ("Bar /\nkitchenette", 1, 5.5, 17.5, 21.4), ("Pool house  8.0 × 10.9 m", 1, 9, 21.4, 32.3, ROOM, True, 12, (5.0, 27.2))]
    for r in rooms:
        q.room(*r)
    for name, *r in (("Bath", 13.2, 15.6, 11.5, 16.0), ("WIR", 15.6, 19.2, 15.0, 17.5),
                     ("Ensuite", 19.2, 22.8, 15.0, 17.5), ("Shower\n/ WC", 5.5, 9.0, 17.5, 21.4)):
        q.room(name, *r, fill=WET, size=11)
    q.stair(8.0, 10.0, 11.5, 14.0, "n", "DN")
    # pool house furniture at true size: 10-seat timber table, couch facing a TV on the west wall
    TIM, FAB, LN = (196, 160, 115), (175, 180, 190), (70, 70, 70)
    q.rect(3.4, 6.6, 28.9, 30.0, TIM, LN, 2)                                     # table 3.2 × 1.1 m
    for k in range(4):
        u = 3.75 + k * 0.83
        q.rect(u - 0.23, u + 0.23, 28.25, 28.7, (230, 225, 215), LN, 1)
        q.rect(u - 0.23, u + 0.23, 30.2, 30.65, (230, 225, 215), LN, 1)
    for u in (2.75, 6.8):
        q.rect(u, u + 0.45, 29.22, 29.68, (230, 225, 215), LN, 1)
    q.note(5.0, 31.3, "10-seat table 3.2 × 1.1 m", 10, (90, 70, 40))
    q.rect(4.9, 5.85, 22.4, 25.2, FAB, LN, 2)                                    # 3-seat couch 2.8 m
    q.rect(5.55, 5.85, 22.4, 25.2, (150, 155, 165), LN, 1)                       # back
    q.rect(2.6, 3.5, 23.2, 24.4, TIM, LN, 1)                                     # coffee table
    q.rect(1.0, 1.3, 23.0, 24.6, (30, 30, 30), None)                            # 75" TV on the west wall
    q.note(2.0, 25.6, "TV", 10)
    q.note(5.4, 26.2, "couch 2.8 m", 10)
    q.rect(8.0, 10.0, 14.0, 16.0, ROOM, (40, 40, 40), 3)
    for r in ((1, 22.8, 11.5, 17.5), (1, 9, 17.5, 32.3)):
        q.envelope(*r)
    q.rect(9.0, 9.7, 25.1, 32.0, (225, 215, 195), (90, 90, 90), 1)
    q.stair(9.7, 11.62, 28.3, 32.0, "u", "DN")
    q.note(12.6, 27.4, "stairs down to pool", 10)
    q.glass_u(25.4, 31.7, 9.0, 7)
    q.note(7.6, 28.6, "bifolds", 10, (40, 100, 160))
    q.glass_u(22.6, 24.8, 9.0)
    q.glass_u(18.6, 20.8, 9.0)
    q.glass_u(11.8, 15.0, 22.8, 7)
    q.glass_n(1.8, 3.8, 11.5, 6)
    q.glass_n(5.3, 7.3, 11.5, 6)
    q.glass_n(16.4, 22.0, 11.5, 6)
    q.glass_n(10.6, 12.6, 11.5)
    q.glass_n(3.3, 6.7, 32.3)

    # north arrow and scale on each panel
    for x in xs:
        cx, cy = x + 60, oy + 70
        nu, nn = math.sin(math.radians(27)), math.cos(math.radians(27))   # +n faces 333°, so north is 27° clockwise of it
        tip = (cx + nu * 32, cy - nn * 32)
        tail = (cx - nu * 22, cy + nn * 22)
        left = (cx - nn * 9, cy - nu * 9)
        right = (cx + nn * 9, cy + nu * 9)
        d.polygon([tip, left, tail, right], fill=(30, 30, 30))
        d.text((tip[0] + nu * 14, tip[1] - nn * 14), "N", font=F(16, True), fill=(30, 30, 30), anchor="mm")
        d.rectangle([x + 20, oy + PH - 30, x + 20 + 10 * S, oy + PH - 24], fill=(30, 30, 30))
        d.text((x + 30 + 10 * S, oy + PH - 36), "10 m", font=F(14), fill=(30, 30, 30))
        d.text((x + PW / 2, oy + PH + 4), "STREET (Waugh St)", font=F(13, True), fill=(120, 120, 120), anchor="mm")

    # side panel: levels and areas
    sx = 3 * PW + 120
    d.text((sx, oy - 52), "5 bedrooms", font=F(24, True), fill=(25, 25, 25))
    d.multiline_text((sx, oy - 24), "Master + beds 2–4 upstairs,\nbed 5 / guest on the ground floor", font=F(13), fill=(70, 70, 70))
    gf = 14.6 * 12.0 + 7.2 * 15.1 + 8 * 10.9
    up = 21.8 * 6.0 + 8 * 14.8
    y0 = oy + 14
    lines = [("Levels (m AHD)", True), (f"Garage floor  {GARAGE:.2f}", False), (f"Lower courtyard  {PADZ:.2f}", False),
             (f"Ground floor  {FFL:.2f}", False), (f"Pool deck / rear lawn  {DECK:.2f}", False),
             (f"Upper floor / pool house  {UPPER:.2f}", False), ("", False),
             ("Approx. floor areas", True), (f"Ground floor  ≈{gf:.0f} m²", False), ("  incl. gym & cellar under", False),
             ("  the pool house", False), (f"Upper + pool house  ≈{up:.0f} m²", False), ("Garage  ≈31 m² + store 11 m²", False),
             (f"Total (excl. garage)  ≈{gf + up:.0f} m²", False), ("", False),
             ("Key", True), ("Blue: glazing", False), ("Brown: doors", False), ("Hatched: stairs", False),
             ("Red: lot boundary", False), ("", False), ("Notes", True),
             ("Walls, openings and levels follow", False), ("the 3D model; the room layout", False),
             ("inside them is a concept only.", False), ("Gross areas to external walls.", False),
             ("Not for construction.", False)]
    y = y0
    for txt, bold in lines:
        d.text((sx, y), txt, font=F(17 if bold else 15, bold), fill=(25, 25, 25) if bold else (60, 60, 60))
        y += 26 if bold else 22
    for k, (fill, name) in enumerate(((ROOM, "Room"), (WET, "Wet area / service"), (DECKC, "Raised lawn (deck level)"),
                                      (OUT, "Paving / outdoor"))):
        d.rectangle([sx, y + 10 + k * 26, sx + 22, y + 28 + k * 26], fill=fill, outline=(90, 90, 90))
        d.text((sx + 32, y + 10 + k * 26), name, font=F(14), fill=(60, 60, 60))
    d.text((30, H - 40), "Lot boundary © State of New South Wales (Spatial Services), CC BY 4.0. Levels from the NSW 5 m "
           "elevation grid; confirm with a detail and level survey. Indicative concept only.", font=F(14), fill=(100, 100, 100))
    sheet.save(out)
    print("wrote", out, f"ground ≈{gf:.0f} m², upper ≈{up:.0f} m²")


if __name__ == "__main__":
    main(*sys.argv[1:])
