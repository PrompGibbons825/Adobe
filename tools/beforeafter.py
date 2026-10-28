"""Contact sheet: original | A | B for chosen Crello pairs.

usage (repo root):
  python tools/beforeafter.py OUT.png DIR_A DIR_B KEY [KEY ...]
DIR_A / DIR_B hold tools/h2h.py outputs; a directory named v_* or sprint is a
partner engine and is rendered in the brand-kit fonts it measured with.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import legible, saliency, textfit  # noqa: E402
from renderer.converter import convert  # noqa: E402


def fonts_for(d):
    return Path("brand_kit/fonts") if d.name == "sprint" or d.name.startswith("v_") else None


def main(out, a, b, keys):
    a, b = Path(a), Path(b)
    try:
        F = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    except Exception:
        F = ImageFont.load_default()
    rows, cur = [], None
    for key in keys:
        root = max((r for r in ("crello", "crello_holdout", "crello_fresh", "crello_val") if key.startswith(r + "_p")), key=len)
        if root != cur:
            saliency.ASSETS_DIR = Path(f"{root}/assets")
            textfit.FONTS_PICKLE, textfit._FM = f"{root}/fonts_subset.pickle", None
            legible._RENDERERS.clear()
            convert._renderer_cache.clear()
            cur = root
        p = int(key[len(root) + 2:].split("_")[0])
        wh = key.rsplit("_", 1)[1]
        idx = json.load(open(f"{root}/index.json"))
        ims = [legible.render(json.load(open(f"{root}/designs/{idx[p][0]['file']}")))]   # the original
        for d in (a, b):
            f = d / f"{key}.json"
            ims.append(legible.render(json.load(open(f)), fonts_for(d)) if f.is_file() else None)
        fit = []
        for im in ims:
            if im is None:
                im = Image.new("RGB", (300, 100), (255, 230, 230))
            sc = min(300 / im.height, 620 / im.width)
            im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))))
            fr = Image.new("RGB", (im.width + 4, im.height + 4), (40, 40, 40))
            fr.paste(im, (2, 2))
            fit.append(fr)
        r = Image.new("RGB", (3 * 640 + 20, max(f.height for f in fit) + 46), "white")
        ImageDraw.Draw(r).text((10, 6), key, fill=(20, 20, 20), font=F)
        for k, f in enumerate(fit):
            r.paste(f, (10 + k * 640, 40))
        rows.append(r)
    head = Image.new("RGB", (rows[0].width, 40), "white")
    d = ImageDraw.Draw(head)
    for k, label in enumerate(["Original", a.name, b.name]):
        d.text((10 + k * 640, 6), label, fill=(90, 90, 90), font=F)
    rows = [head] + rows
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
