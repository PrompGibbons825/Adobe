"""Dev tool: what the strategy-portfolio candidates (engine/smart/strategies.py) change.

    .venv/bin/python -m tools.strategies_ab wins [--crello] [--compare]   # who wins, by aspect class
    .venv/bin/python -m tools.strategies_ab sheet -o /tmp/ab.png          # off | on, where the winner changed
    .venv/bin/python -m tools.strategies_ab sheet --crello --only 1,3 -o /tmp/ab.png

`wins` plans every sample/case design at the catalogue formats (or, with
--crello, every Crello family at all its targets) and counts the winning
candidate family per aspect class. One data set per run: the asset folders are
resolved once per process. `sheet` renders the winner with the new generators off and on, side by
side, only where the winner changed. Not part of the engine or the image.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = ROOT / "tools" / "crello" / "cases"


def aspect_class(W: float, H: float, W0: float, H0: float) -> str:
    r = W / H
    shape = ("strip" if r >= 5 else "wide" if r >= 2.4 else "landscape" if r >= 1.25 else
             "square" if r >= 0.8 else "tall")
    q = r / (W0 / H0)
    move = "wider" if q >= 1.25 else ("taller" if q <= 0.8 else "same")
    return f"{shape}/{move}"


def family(name: str) -> str:
    for k in ("pin", "springs", "propfit", "flow-keep", "flow", "adapt", "stack", "split", "row"):
        if name.startswith(k):
            return k
    return name


def tasks(crello: bool, only: str = ""):
    from tools.sheet import CATALOGUE

    if crello:
        for f in sorted(CASES.glob("fam*.json")):
            k = int(f.stem[3:5])
            if only in ("tune", "held"):
                if (k % 2 == 0) != (only == "tune"):
                    continue
            elif only and k not in {int(x) for x in only.split(",") if x}:
                continue
            fam = json.loads(f.read_text())
            for i, t in enumerate(fam["targets"]):
                yield f"{f.stem}.{i}", fam["source"], t["canvas"], "crello"
    else:
        files = sorted((ROOT / "input").glob("*.json")) + sorted((ROOT / "tools" / "cases").glob("*.json"))
        for f in files:
            src = json.loads(f.read_text())["source"]
            for k, (w, h) in CATALOGUE.items():
                yield f"{f.stem[-14:]}.{k}", src, {"width": w, "height": h}, "kit"


def _winner(src, tgt):
    from engine.smart import plan_design

    with contextlib.redirect_stderr(io.StringIO()):
        try:
            _, _, plans = plan_design(json.loads(json.dumps(src)), tgt)
            return plans[0].name if plans else "FALLBACK"
        except Exception:
            return "FALLBACK"


def cmd_wins(a):
    from engine.smart import strategies

    by_cls = defaultdict(Counter)
    tot = Counter()
    changed = 0
    n = 0
    for name, src, tgt, kind in tasks(a.crello):
        strategies.ENABLED = True
        w = _winner(src, tgt)
        if a.compare:
            strategies.ENABLED = False
            w0 = _winner(src, tgt)
            changed += family(w0) != family(w) or w0 != w
        cls = aspect_class(tgt["width"], tgt["height"], src["canvas"]["width"], src["canvas"]["height"])
        f = w.split(":")[0] if a.detail else family(w)
        by_cls[cls][f] += 1
        tot[f] += 1
        n += 1
    fams = [f for f, _ in tot.most_common()]
    print("winning candidate family by aspect class (target shape / move vs source)")
    wd = max(10, max(len(f) for f in fams) + 1) if a.detail else 10
    print("class".ljust(18) + "n".rjust(5) + "".join(f[:wd - 1].rjust(wd) for f in fams))
    for cls in sorted(by_cls):
        c = by_cls[cls]
        print(cls.ljust(18) + f"{sum(c.values()):5d}" + "".join(f"{c[f]:{wd}d}" for f in fams))
    print("all".ljust(18) + f"{n:5d}" + "".join(f"{tot[f]:{wd}d}" for f in fams))
    if a.compare:
        print(f"winner changed vs strategies off: {changed}/{n}")


def cmd_sheet(a):
    from PIL import Image, ImageDraw

    from engine.resize import resize
    from engine.smart import plan_design, strategies
    from renderer.converter.convert import render_design_to_png

    def render(design, assets, cell):
        p = Path(a.out).with_suffix(".tmp.png")
        render_design_to_png(design, assets, p, fonts_dir=ROOT / "brand_kit" / "fonts")
        im = Image.open(p).convert("RGB")
        p.unlink(missing_ok=True)
        f = min(cell / im.width, cell * 0.8 / im.height)
        return im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))))

    rows = []
    assets = CASES / "assets" if a.crello else ROOT / "brand_kit" / "assets"
    for name, src, tgt, kind in tasks(a.crello, a.only):
        if a.targets and f"{tgt['width']}x{tgt['height']}" not in a.targets.split(","):
            continue
        outs, names = [], []
        for on in (False, True):
            strategies.ENABLED = on
            with contextlib.redirect_stderr(io.StringIO()):
                outs.append(resize(json.loads(json.dumps(src)), tgt))
            names.append(_winner(src, tgt))
        if names[0] == names[1] and not a.all:
            continue
        ims = [render(src, assets, a.cell)] + [render(o, assets, a.cell) for o in outs]
        rows.append((f"{name} {tgt['width']}x{tgt['height']}   [source | off: {names[0]} | on: {names[1]}]", ims))
        if len(rows) >= a.max_rows:
            break
    strategies.ENABLED = True
    if not rows:
        print("no changed winners")
        return
    cw = a.cell + 6
    heights = [max(im.height for im in ims) + 20 for _, ims in rows]
    sheet = Image.new("RGB", (3 * cw, sum(heights)), (210, 210, 210))
    y = 0
    for (label, ims), h in zip(rows, heights):
        ImageDraw.Draw(sheet).text((4, y + 3), label, fill=(0, 0, 0))
        for j, im in enumerate(ims):
            sheet.paste(im, (j * cw, y + 18))
        y += h
    sheet.save(a.out)
    Path(a.out).with_suffix(".tmp.png").unlink(missing_ok=True)
    print(a.out, sheet.size, len(rows), "rows")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("wins")
    p.add_argument("--compare", action="store_true", help="also count how often the winner changed")
    p.add_argument("--detail", action="store_true", help="count each flow operation separately")
    p.add_argument("--crello", action="store_true", help="Crello families instead of the kit designs")
    p = sub.add_parser("sheet")
    p.add_argument("-o", "--out", default="/tmp/strategies_ab.png")
    p.add_argument("--crello", action="store_true", help="Crello families instead of the kit designs")
    p.add_argument("--only", default="", help="Crello family numbers (1,3,19), or tune / held")
    p.add_argument("--targets", default="", help="only these sizes, e.g. 728x90,1200x628")
    p.add_argument("--all", action="store_true", help="also rows where the winner did not change")
    p.add_argument("--cell", type=int, default=300)
    p.add_argument("--max-rows", type=int, default=40)
    a = ap.parse_args()
    if a.crello:
        os.environ["CHAMELEON_BRAND_KIT"] = str(CASES)
    {"wins": cmd_wins, "sheet": cmd_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
