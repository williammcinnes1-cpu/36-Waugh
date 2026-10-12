"""Rebuild data/3d/designs/designs.json from every <id>.json + <id>.txt pair, in a fixed order."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data" / "3d" / "designs"
ORDER = ["farmhouse_v2", "farmhouse_raised", "farmhouse_fp", "hill_country", "wraparound", "sage_stone", "stone_glass", "black_barn", "farmhouse_v1"]
found = {p.stem: json.loads(p.read_text()) for p in D.glob("*.json") if p.name != "designs.json" and (D / f"{p.stem}.txt").exists()}
designs = [found[i] for i in ORDER if i in found] + [v for k, v in found.items() if k not in ORDER]
(D / "designs.json").write_text(json.dumps(designs, indent=1, ensure_ascii=False))
print([d["id"] for d in designs])
