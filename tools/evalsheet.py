"""Dev tool: side-by-side sheet of saved eval outputs, to compare engine versions.

    .venv/bin/python -m tools.evalsheet --tags base,ws --only 3,9,14 -o output/cmp.png
    .venv/bin/python -m tools.evalsheet --tags base,ws --changed -o output/cmp.png

One row per family: source | designer's resize | each tag's output (from
tools/crello/eval/<tag>/, written by tools.evalset run). --changed keeps only
families whose output differs between the first two tags. Not part of the
engine or the Docker image.
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
CASES = ROOT / "tools" / "crello" / "cases"
EVAL = ROOT / "tools" / "crello" / "eval"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--changed", action="store_true")
    ap.add_argument("--cell", type=int, default=260)
    ap.add_argument("-o", "--out", default="output/evalsheet.png")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from PIL import Image, ImageDraw

    from renderer.converter.convert import render_design_to_png

    tags = a.tags.split(",")
    tmp = Path(f"/tmp/_evalsheet_{os.getpid()}.png")

    def render(design):
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            render_design_to_png(design, CASES / "assets", tmp, fonts_dir=ROOT / "brand_kit" / "fonts")
        im = Image.open(tmp).convert("RGB")
        tmp.unlink(missing_ok=True)
        f = min(a.cell / im.width, a.cell * 0.8 / im.height)
        return im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))))

    fams = sorted(p.stem for p in (EVAL / tags[0]).glob("fam??.json"))
    if a.only:
        fams = [f for f in fams if int(f[3:]) in {int(x) for x in a.only.split(",")}]
    if a.changed and len(tags) > 1:
        fams = [f for f in fams if (EVAL / tags[0] / f"{f}.json").read_bytes() != (EVAL / tags[1] / f"{f}.json").read_bytes()]
    rows = []
    for fam in fams:
        case = json.loads((CASES / f"{fam}.json").read_text())
        ims = [render(case["source"]), render(case["targets"][0])]
        ims += [render(json.loads((EVAL / t / f"{fam}.json").read_text())) for t in tags]
        tc = case["targets"][0]["canvas"]
        rows.append((f"{fam} -> {tc['width']}x{tc['height']}   [source | designer | {' | '.join(tags)}]", ims))
    if not rows:
        print("nothing to show")
        return 0
    cw = a.cell + 6
    ncol = 2 + len(tags)
    heights = [max(im.height for im in ims) + 20 for _, ims in rows]
    sheet = Image.new("RGB", (ncol * cw, sum(heights)), (210, 210, 210))
    y = 0
    for (label, ims), h in zip(rows, heights):
        ImageDraw.Draw(sheet).text((4, y + 3), label, fill=(0, 0, 0))
        for j, im in enumerate(ims):
            sheet.paste(im, (j * cw, y + 18))
        y += h
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print(a.out, sheet.size, len(rows), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
