"""Export every design for the web viewer's design drop-down.

Each design has its own earthworks, so for each one this regenerates the designed ground,
render_scene.json, and a GLB (base64 in data/3d/designs/<id>.txt) plus a meta file with its facts.
The current design (farmhouse v2) is built last so the repo's LAS, mesh and render inputs match it.

    python3 scripts/build_designs.py     (needs the Blender venv path in BPY below)
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SC = ROOT / "scripts"
D3 = ROOT / "data" / "3d"
OUT = D3 / "designs"
OUT.mkdir(exist_ok=True)
BPY = sys.argv[1] if len(sys.argv) > 1 else "python3.11"
V1 = "9f1c2f6"  # commit holding the painted-brick farmhouse (v1)


def run(*cmd, cwd=ROOT):
    print("$", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run(list(map(str, cmd)), cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"failed: {cmd}")
    return r.stdout


def meta(id_, name, note):
    a = json.loads((D3 / "house_massing.json").read_text())["assumptions"]
    foot = a.get("footprint_m2") or 353
    cover = a.get("coverage") or foot / 1093
    cut = a.get("max_cut") or 2.39
    fill = a.get("max_fill") or 2.27
    glb = OUT / f"{id_}.glb"
    (OUT / f"{id_}.txt").write_text(base64.b64encode(glb.read_bytes()).decode())
    glb.unlink()
    return {"id": id_, "name": name, "file": f"designs/{id_}.txt", "note": note, "facts": [
        ["Lot area", "≈1,093 m²"], ["Zoning", "R1 General Residential"],
        ["House + garage footprint", f"{foot:,} m² ({cover:.0%})"], ["Floor level", f"{a['FFL']:.1f} m AHD"],
        ["Earthworks", f"cut ≤{cut:.1f} m · fill ≤{fill:.1f} m"], ["Bushfire prone", "rear half (vegetation buffer)"]]}


def main():
    designs = []
    # option A: glass gables (hybrid layout)
    run(sys.executable, SC / "house_massing.py")
    run(sys.executable, SC / "build_3d_render.py")
    run(BPY, "cycles_render.py", "--export-glb", OUT / "glass_gables.glb", cwd=SC)
    a_meta = meta("glass_gables", "Glass gables",
                  "Two glazed gable wings to the street; living rooms and pool face a north courtyard.")
    # option B v1: painted-brick farmhouse, from git history
    for f in ("farmhouse_site.py", "farmhouse_cycles.py"):
        (SC / f"_v1_{f}").write_text(run("git", "show", f"{V1}:scripts/{f}"))
    try:
        run(sys.executable, SC / "_v1_farmhouse_site.py", "ground")
        run(sys.executable, SC / "build_3d_render.py")
        run(BPY, "_v1_farmhouse_cycles.py", "--export-glb", OUT / "farmhouse_v1.glb", cwd=SC)
        v1_meta = meta("farmhouse_v1", "Farmhouse: painted brick",
                       "First farmhouse: painted brick street wall, small verandah, two dormers.")
    finally:
        for f in ("farmhouse_site.py", "farmhouse_cycles.py"):
            (SC / f"_v1_{f}").unlink(missing_ok=True)
    # option B v2 (current): stone base, deep verandah, outdoor room
    run(sys.executable, SC / "farmhouse_site.py", "ground")
    run(sys.executable, SC / "build_3d_render.py")
    run(BPY, "farmhouse_cycles.py", "--export-mesh", "--export-glb", OUT / "farmhouse_v2.glb", cwd=SC)
    run(sys.executable, SC / "farmhouse_site.py", "finish")
    v2_meta = meta("farmhouse_v2", "Farmhouse: stone and verandah",
                   "White board-and-batten over a limestone base, deep verandah, three dormers, stone chimney; "
                   "small windows to the street, living rooms, pool and an outdoor room on the north courtyard.")
    designs = [v2_meta, v1_meta, a_meta]
    (OUT / "designs.json").write_text(json.dumps(designs, indent=1, ensure_ascii=False))
    print("designs:", [d["id"] for d in designs])


if __name__ == "__main__":
    main()
