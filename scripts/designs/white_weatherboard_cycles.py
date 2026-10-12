"""Quick concept (rough, about 2/10): White Modern Weatherboard.

Reuses the farmhouse layout, levels and DCP fit (farmhouse_site.py), re-skinned:
horizontal white weatherboards in place of board-and-batten, and white render in place of the limestone.
    python white_weatherboard_cycles.py [farmhouse_cycles flags...]
"""
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import farmhouse_cycles as fc  # noqa: E402

_orig_textures = fc.textures


def textures():
    _orig_textures()
    R = random.Random(5)
    # horizontal weatherboards: 1024 px = 4 m, 180 mm boards with a dark shadow line under each lap
    im = Image.new("RGB", (1024, 1024), (240, 239, 235))
    d = ImageDraw.Draw(im)
    step = 46
    for y in range(0, 1024, step):
        l = R.uniform(-3, 3)
        d.rectangle([0, y, 1024, y + step - 6], fill=(int(242 + l), int(241 + l), int(237 + l)))
        d.rectangle([0, y + step - 6, 1024, y + step - 3], fill=(170, 170, 166))
        d.rectangle([0, y + step - 3, 1024, y + step], fill=(212, 212, 208))
    arr = np.array(im).astype(np.int16) + np.random.default_rng(4).integers(-3, 3, (1024, 1024))[..., None]
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(fc.TEX / "cladding.png")
    # smooth white render replaces the limestone base course, piers and chimney
    im = Image.new("RGB", (2048, 2048), (232, 230, 224))
    arr = np.array(im).astype(np.int16) + np.random.default_rng(6).integers(-6, 6, (2048, 2048))[..., None]
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)).save(fc.TEX / "stone.png")


fc.textures = textures
if __name__ == "__main__":
    fc.main()
