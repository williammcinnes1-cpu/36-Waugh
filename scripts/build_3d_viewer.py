"""Bake the 3D terrain viewer (site_3d.html) from data/3d outputs of build_3d_site.py."""
import base64
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "3d"


def main():
    p = json.loads((OUT / "viewer_payload.json").read_text())
    n = p["grid"]["size"]
    h = np.array(p["heights_cm"], dtype=float).reshape(n, n) / 100  # row 0 = south
    base = float(np.floor(h.min()))
    enc = np.round((h - base) * 1000).astype("<u2")
    lot = p["lot"]

    def h_at(x, y):
        return float(h[int(min(max(y - 0.5, 0), n - 1)), int(min(max(x - 0.5, 0), n - 1))])

    corners = [{"x": x, "y": y, "h": h_at(x, y)} for x, y in lot[:4]]
    centre = Polygon(lot).centroid
    arc_mid = lot[7]
    dx, dy = arc_mid[0] - centre.x, arc_mid[1] - centre.y
    k = 14 / np.hypot(dx, dy)
    data = {
        "size": n, "base": base, "datum": base - 6,
        "heights": base64.b64encode(enc.tobytes()).decode(),
        "lot": lot, "bushfire": p["bushfire"], "c2": p["c2"],
        "corners": corners, "lotCentre": [centre.x, centre.y],
        "street": [arc_mid[0] + dx * k, arc_mid[1] + dy * k],
    }
    tex = base64.b64encode((OUT / "ortho_texture.jpg").read_bytes()).decode()
    html = (ROOT / "scripts" / "viewer_template.html").read_text()
    html = (html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
                .replace("__TEXTURE__", tex)
                .replace("__RANGE__", f"{h.min():.0f}–{h.max():.0f} m AHD"))
    (ROOT / "site_3d.html").write_text(html)
    print("site_3d.html", round(len(html) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
