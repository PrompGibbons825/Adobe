"""Comparison sheets: source | designer | engine A | engine B | ... for every
pair in a Crello pair folder, a few pairs per page.

usage (repo root):
  python tools/sheet.py ROOT OUT_PREFIX DIR[,DIR...] [--labels A,B,...] [--per 8] [--scores FILE] [--no-designer]
Writes OUT_PREFIX_1.png, OUT_PREFIX_2.png, ... An engine folder named v_* or
sprint is a partner engine and is rendered in the brand-kit fonts it measured
with. --scores takes a scoreboard -v dump to print each engine's distance and
words-read under its picture.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import legible, saliency, textfit  # noqa: E402

CELL_W, CELL_H = 430, 290


def font(n):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", n)
    except Exception:
        return ImageFont.load_default()


def fit(im):
    if im is None:
        im = Image.new("RGB", (CELL_W, 60), (255, 225, 225))
    s = min(CELL_H / im.height, CELL_W / im.width)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    fr = Image.new("RGB", (im.width + 4, im.height + 4), (40, 40, 40))
    fr.paste(im, (2, 2))
    return fr


def main():
    args = sys.argv[1:]
    root, prefix, dirs = args[0], args[1], [Path(d) for d in args[2].split(",")]
    labels = args[args.index("--labels") + 1].split(",") if "--labels" in args else [d.name for d in dirs]
    per = int(args[args.index("--per") + 1]) if "--per" in args else 8
    scores = json.load(open(args[args.index("--scores") + 1])) if "--scores" in args else {}
    saliency.ASSETS_DIR = Path(f"{root}/assets")
    textfit.FONTS_PICKLE, textfit._FM = f"{root}/fonts_subset.pickle", None
    idx = json.load(open(f"{root}/index.json"))
    F, Fs = font(22), font(18)
    designer = "--no-designer" not in args
    cols = ["Original"] + (["Designer"] if designer else []) + labels
    lead = len(cols) - len(labels)
    rows = []
    for p, vs in enumerate(idx):
        src = json.load(open(f"{root}/designs/{vs[0]['file']}"))
        for v in vs[1:]:
            key = f"{root}_p{p}_{v['w']}x{v['h']}"
            ims = [legible.render(src)]
            if designer:
                ims.append(legible.render(json.load(open(f"{root}/designs/{v['file']}"))))
            for d in dirs:
                f = d / f"{key}.json"
                fd = Path("brand_kit/fonts") if d.name == "sprint" or d.name.startswith("v_") else None
                ims.append(legible.render(json.load(open(f)), fd) if f.is_file() else None)
            cells = [fit(im) for im in ims]
            h = max(c.height for c in cells) + 70
            r = Image.new("RGB", (len(cols) * (CELL_W + 14) + 14, h), "white")
            d = ImageDraw.Draw(r)
            d.text((14, 6), f"{key}   {vs[0]['title'][:60]}   {vs[0]['w']}x{vs[0]['h']} -> {v['w']}x{v['h']}",
                   fill=(20, 20, 20), font=F)
            for k, c in enumerate(cells):
                x = 14 + k * (CELL_W + 14)
                r.paste(c, (x, 36))
                name = dirs[k - lead].name if k >= lead else None
                sc = scores.get(name, {}).get(key) if name else None
                if sc:
                    d.text((x, 40 + c.height), f"dist {sc['dist']:.2f}   read {sc.get('read', float('nan')):.0%}",
                           fill=(90, 90, 90), font=Fs)
            rows.append(r)
    for page in range(0, len(rows), per):
        chunk = rows[page:page + per]
        head = Image.new("RGB", (chunk[0].width, 40), "white")
        d = ImageDraw.Draw(head)
        for k, c in enumerate(cols):
            d.text((14 + k * (CELL_W + 14), 8), c, fill=(80, 80, 80), font=F)
        chunk = [head] + chunk
        out = Image.new("RGB", (chunk[0].width, sum(c.height for c in chunk)), "white")
        y = 0
        for c in chunk:
            out.paste(c, (0, y))
            y += c.height
        out.save(f"{prefix}_{page // per + 1}.png")
        print(f"{prefix}_{page // per + 1}.png")


if __name__ == "__main__":
    main()
