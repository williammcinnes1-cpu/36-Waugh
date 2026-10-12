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
    # Proposed house (scripts/house_massing.py): mesh, designed ground and point samples
    hm = json.loads((OUT / "house_massing.json").read_text())
    g2 = np.array(hm["ground_cm"], dtype=float).reshape(n, n) / 100
    data["heightsHouse"] = base64.b64encode(np.round((g2 - base) * 1000).astype("<u2").tobytes()).decode()
    data["house"] = {k: hm[k] for k in ("positions", "faces", "colors")}
    data["houseInfo"] = hm["assumptions"]

    # Point colours: ground colour per 1 m cell from the ortho (row 0 = south), house points from the LAS
    from PIL import Image
    import laspy
    tex_img = np.array(Image.open(OUT / "ortho_texture.jpg").convert("RGB"))
    gcol = tex_img[2::4, 2::4][::-1][:n, :n]
    data["groundRGB"] = base64.b64encode(gcol.astype(np.uint8).tobytes()).decode()
    las = laspy.read(OUT / "site_with_house.las")
    sel = np.isin(np.array(las.classification), [6, 9])
    hx = np.round((np.array(las.x)[sel] - p["grid"]["e0"]) * 100).astype("<u2")
    hy = np.round((np.array(las.y)[sel] - p["grid"]["n0"]) * 100).astype("<u2")
    hz = np.round((np.array(las.z)[sel] - base) * 100).astype("<u2")
    hrgb = (np.column_stack([las.red[sel], las.green[sel], las.blue[sel]]) // 257).astype(np.uint8)
    data["housePts"] = base64.b64encode(np.column_stack([hx, hy, hz]).tobytes()).decode()
    data["housePtsRGB"] = base64.b64encode(hrgb.tobytes()).decode()

    tex = base64.b64encode((OUT / "ortho_texture.jpg").read_bytes()).decode()
    html = (ROOT / "scripts" / "viewer_template.html").read_text()
    html = (html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
                .replace("__TEXTURE__", tex)
                .replace("__RANGE__", f"{h.min():.0f}–{h.max():.0f} m AHD")
                .replace("__FFL__", f"{hm['assumptions']['FFL']:.1f}")
                .replace("__RIDGE__", f"{hm['assumptions']['ridge']:.1f}"))
    (ROOT / "site_3d.html").write_text(html)
    print("site_3d.html", round(len(html) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
