"""Dev tool: render candidate layouts / many targets into one contact sheet.

    .venv/bin/python -m tools.sheet input/x.json --targets 1080x1920,1600x400 --top 3 -o /tmp/sheet.png
    .venv/bin/python -m tools.sheet input/*.json --catalogue -o /tmp/all.png

Not part of the engine or the Docker image.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.resize import resize  # noqa: E402
from engine.smart import emit, finalize, plan_design  # noqa: E402
from tools.rules import render_image  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "brand_kit" / "assets"

CATALOGUE = {
    "ig-square": (1080, 1080), "ig-portrait": (1080, 1350), "story": (1080, 1920),
    "ig-landscape": (1080, 566), "fb-link": (1200, 628), "yt-thumb": (1280, 720),
    "x-header": (1500, 500), "li-banner": (1584, 396), "fb-cover": (820, 312),
    "pinterest": (1000, 1500), "leaderboard": (728, 90), "billboard": (970, 250),
    "med-rect": (300, 250), "skyscraper": (160, 600), "half-page": (300, 600),
    "mobile-banner": (320, 50), "large-leader": (970, 90), "a4": (2480, 3508),
    "email-header": (600, 200), "pres-16x9": (1920, 1080),
}


def render(design: dict, max_side: int = 480) -> Image.Image:
    im = render_image(design, ASSETS)
    w, h = im.size
    f = min(max_side / w, (max_side * 1.4) / h)
    return im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS)


def resize_explained(src: dict, tgt: dict):
    """resize(), plus the name of the plan that won (or FALLBACK)."""
    try:
        d, ctx, plans = plan_design(src, tgt)
        return finalize(src, {"canvas": tgt, "elements": emit(plans[0], d, ctx)}, tgt), plans[0].name
    except Exception:  # noqa: BLE001 - resize() falls back the same way
        return resize(src, tgt), "FALLBACK"


def tile(im: Image.Image, label: str, cell: int) -> Image.Image:
    t = Image.new("RGB", (cell, im.height + 22), (245, 245, 245))
    t.paste(im, ((cell - im.width) // 2, 20))
    ImageDraw.Draw(t).text((4, 4), label[:70], fill=(20, 20, 20))
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--targets", default="")
    ap.add_argument("--catalogue", action="store_true")
    ap.add_argument("--top", type=int, default=1)
    ap.add_argument("--source", action="store_true", help="first tile per input = the source design")
    ap.add_argument("--cell", type=int, default=360)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("-o", "--out", default="/tmp/sheet.png")
    a = ap.parse_args()

    tiles = []
    for inp in a.inputs:
        task = json.loads(Path(inp).read_text())
        src = task["source"]
        if a.source:
            tiles.append(tile(render(src, a.cell), f"SOURCE {Path(inp).stem[-20:]} {src['canvas']['width']}x{src['canvas']['height']}", a.cell))
        targets = []
        if a.targets:
            for t in a.targets.split(","):
                w, h = t.lower().split("x")
                targets.append((t, {"width": int(w), "height": int(h)}))
        elif a.catalogue:
            targets = [(k, {"width": w, "height": h}) for k, (w, h) in CATALOGUE.items()]
        else:
            targets = [("given", task["target_canvas"])]
        for name, tgt in targets:
            if a.top <= 1:
                out, how = resize_explained(src, tgt)
                lab = f"{Path(inp).stem[-12:]} {name} {tgt['width']}x{tgt['height']} [{how}]"
                tiles.append(tile(render(out, a.cell), lab, a.cell))
            else:
                d, ctx, plans = plan_design(src, tgt)
                for p in plans[: a.top]:
                    out = finalize(src, {"canvas": tgt, "elements": emit(p, d, ctx)}, tgt)
                    lab = f"{name} {p.name} {p.score:.0f} hid={len([h for h in p.hidden if h.role != 'missing'])}"
                    tiles.append(tile(render(out, a.cell), lab, a.cell))
    cols = min(a.cols, len(tiles))
    grid = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    heights = [max(t.height for t in r) + 4 for r in grid]
    sheet = Image.new("RGB", (cols * (a.cell + 4), sum(heights)), (200, 200, 200))
    y = 0
    for r, h in zip(grid, heights):
        for i, t in enumerate(r):
            sheet.paste(t, (i * (a.cell + 4), y))
        y += h
    sheet.save(a.out)
    print(a.out, sheet.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
