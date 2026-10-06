"""Dev tool: contact sheet of the designs where the learned ranker changes the pick.

Runs the engine twice per (family, target), with the ranker off and on, and
renders source | designer | hand-score pick | ranked pick for every target
where the two outputs differ, with each pick's designer similarity.

    .venv/bin/python -m tools.ranker_sheet --set old  --split held  -o output/ranker_held.png
    .venv/bin/python -m tools.ranker_sheet --set train --split held2 -o output/ranker_held2.png
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRELLO = ROOT / "tools" / "crello"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["old", "train"], required=True)
    ap.add_argument("--split", default="held")
    ap.add_argument("--cell", type=int, default=260)
    ap.add_argument("--max-rows", type=int, default=40)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    kit = CRELLO / ("cases" if a.set == "old" else "train")
    os.environ["CHAMELEON_BRAND_KIT"] = str(kit)
    sys.path.insert(0, str(ROOT))
    from PIL import Image, ImageDraw

    from engine.resize import resize
    from engine.smart import ranker
    from engine.smart.textfit import drawn_box
    from renderer.converter.convert import render_design_to_png
    from tools.designer_sim import similarity

    def render(design):
        p = Path(f"/tmp/_ranker_sheet_{os.getpid()}.png")
        with contextlib.redirect_stdout(io.StringIO()):
            render_design_to_png(design, kit / "assets", p, fonts_dir=ROOT / "brand_kit" / "fonts")
        im = Image.open(p).convert("RGB")
        f = min(a.cell / im.width, a.cell * 0.8 / im.height)
        return im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))))

    def run(src, target, on: bool):
        os.environ["CHAMELEON_RANKER"] = "1" if on else "0"
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            return resize(json.loads(json.dumps(src)), target)

    if a.set == "old":
        jobs = [(f, 0) for f in sorted((CRELLO / "cases").glob("fam??.json"))
                if (int(f.stem[3:]) % 2 == 1) == (a.split == "held")]
    else:
        jobs = []
        for f in sorted((CRELLO / "train" / "cases").glob("t???.json")):
            fam = json.loads(f.read_text())
            if fam["split"] == a.split:
                jobs += [(f, i) for i in range(len(fam["targets"]))]
    assert ranker.model() is not None, "no ranker weights"
    rows, n = [], 0
    for f, ti in jobs:
        fam = json.loads(f.read_text())
        src, des = fam["source"], fam["targets"][ti]
        target = {"width": des["canvas"]["width"], "height": des["canvas"]["height"]}
        off, on = run(src, target, False), run(src, target, True)
        n += 1
        if json.dumps(off) == json.dumps(on):
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            s0 = similarity(off["elements"], des, src, text_box=drawn_box)["sim"]
            s1 = similarity(on["elements"], des, src, text_box=drawn_box)["sim"]
        rows.append((f"{f.stem}:{ti} {src['canvas']['width']}x{src['canvas']['height']} -> "
                     f"{target['width']}x{target['height']}   designer-sim hand {s0:.2f} | ranked {s1:.2f}",
                     [render(src), render(des), render(off), render(on)]))
        print(rows[-1][0], flush=True)
        if len(rows) >= a.max_rows:
            break
    print(f"{len(rows)} changed of {n} targets")
    if not rows:
        return
    cw = a.cell + 6
    heights = [max(im.height for im in ims) + 22 for _, ims in rows]
    sheet = Image.new("RGB", (4 * cw, sum(heights)), (210, 210, 210))
    y = 0
    for (label, ims), h in zip(rows, heights):
        ImageDraw.Draw(sheet).text((4, y + 4), label + "    [source | designer | hand | ranked]", fill=(0, 0, 0))
        for j, im in enumerate(ims):
            sheet.paste(im, (j * cw, y + 20))
        y += h
    sheet.save(a.out)
    print(a.out, sheet.size)


if __name__ == "__main__":
    main()
