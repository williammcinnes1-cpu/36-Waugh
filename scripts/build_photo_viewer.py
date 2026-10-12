"""Bake the photoreal interactive viewer (site_3d.html + data/3d/farmhouse_scene.glb).

Needs farmhouse_cycles.py --export-glb to have written farmhouse_scene.glb and farmhouse_trees.json.
The GLB is published next to the page as farmhouse_scene.glb.
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D3 = ROOT / "data" / "3d"


def main():
    scene = json.loads((D3 / "render_scene.json").read_text())
    hm = json.loads((D3 / "house_massing.json").read_text())
    payload = json.loads((D3 / "viewer_payload.json").read_text())
    ax, ay = scene["A"]
    ux, uy = scene["U"]
    nx, ny = -uy, ux
    # viewer_payload rings are in metres from the SW corner of the 240 m block (same origin as A)
    to_un = lambda x, y: [round((x - ax) * ux + (y - ay) * uy, 2), round((x - ax) * nx + (y - ay) * ny, 2)]
    a = hm["assumptions"]
    data = {
        "trees": json.loads((D3 / "farmhouse_trees.json").read_text()),
        "lot": [to_un(*p) for p in payload["lot"]],
        "bushfire": [[to_un(*p) for p in ring] for ring in payload["bushfire"]],
        "north": [uy, ux], "east": [ux, -uy],
        "views": {
            "street": {"eye": [6.8, -12.0, 1.8], "at": [8.6, 14.0, 3.0], "fov": 52},
            "street_wide": {"eye": [9.5, -26.0, 3.2], "at": [11.5, 16.0, 4.0], "fov": 42},
            "courtyard": {"eye": [22.0, 41.0, 6.5], "at": [12.0, 24.0, 1.5], "fov": 52},
            "aerial": {"eye": [48.0, -24.0, 42.0], "at": [12.0, 20.0, 0.0], "fov": 45},
        },
    }
    html = (ROOT / "scripts" / "photo_viewer_template.html").read_text()
    html = (html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
                .replace("__LEAF__", base64.b64encode((D3 / "tex" / "leaf_euc.png").read_bytes()).decode())
                .replace("__FOOT__", f"{a['footprint_m2']:,}")
                .replace("__COVER__", f"{a['coverage']:.0%}")
                .replace("__FFL__", f"{a['FFL']:.1f}")
                .replace("__CUT__", f"{a['max_cut']:.1f}")
                .replace("__FILL__", f"{a['max_fill']:.1f}"))
    (ROOT / "site_3d.html").write_text(html)
    (D3 / "farmhouse_scene.txt").write_text(base64.b64encode((D3 / "farmhouse_scene.glb").read_bytes()).decode())
    print("site_3d.html", round(len(html) / 1e6, 2), "MB; trees", len(data["trees"]))


if __name__ == "__main__":
    main()
