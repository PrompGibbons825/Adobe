"""Mine vs designer on Crello tasks (training families only). python -m lab.viz -o /tmp/v.png --k 8 --seed 1"""
import argparse, json, random
from pathlib import Path
from PIL import Image, ImageDraw
from engine.compose import compose
from renderer.converter.convert import render_design_to_png
from lab.learn import tasks
ap = argparse.ArgumentParser(); ap.add_argument("-o", default="/tmp/v.png"); ap.add_argument("--k", type=int, default=8); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--fmt", default="")
a = ap.parse_args()
ts = tasks(); random.Random(a.seed).shuffle(ts)
A = "/data/agents/work/crello-wide/cases/assets"; C = 300
rows = []
for name, src, tgt, ref in ts[:a.k]:
    if a.fmt: tgt = dict(zip(('width','height'), map(int, a.fmt.split('x'))))
    o = compose(src, tgt); lay = o.pop("_layout")
    ims = []
    for lab, d in (("src", src), ("mine " + lay, o), ("designer", ref if not a.fmt else src)):
        p = f"/tmp/viz_{lab[:3]}.png"; render_design_to_png(d, A, p, fonts_dir="brand_kit/fonts", max_size=C * 2)
        im = Image.open(p).convert("RGB"); im.thumbnail((C, C)); ims.append((lab, im))
    row = Image.new("RGB", (C * 3, C + 14), (50, 50, 50))
    for i, (lab, im) in enumerate(ims):
        row.paste(im, (i * C, 14)); ImageDraw.Draw(row).text((i * C + 2, 1), f"{lab} {name[:8]} {tgt['width']}x{tgt['height']}", fill=(255, 255, 0))
    rows.append(row)
sh = Image.new("RGB", (C * 6, (C + 14) * ((len(rows) + 1) // 2)))
for i, r in enumerate(rows): sh.paste(r, ((i % 2) * C * 3, (i // 2) * (C + 14)))
sh.save(a.o)
