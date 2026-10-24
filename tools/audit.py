"""Dev tool: a layout-quality audit over designs the engine has not seen.

    .venv/bin/python -m tools.audit --n 40 --seed 1 --sheet /tmp/audit.png

Generates designs with tools.designs, resizes each to every catalogue format and
measures what a reviewer would notice, each relative to the source design:

  empty    the largest band along the long axis with no content (fraction)
  lopsided content pushed to one side: |top - bottom| margins over the height
           (and left/right on strips)
  cramped  content within 1.5% of a canvas edge that the source kept clear
  sparse   the content's bounding box covers far less of the canvas than in the source
  hidden   a headline, CTA, subhead, product or logo dropped
  flat     the headline no larger than its subhead, or the product a speck
  overlap  copy over copy the source did not have (tools.rules.promise_errors)

It prints how often each flag fires and a contact sheet of the worst runs, so a
change is judged on many designs at once rather than one.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.smart.assets import ink_fraction  # noqa: E402
from engine.smart.geom import Box, el_box  # noqa: E402
from engine.smart.textfit import drawn_box  # noqa: E402
from tools.designs import designs  # noqa: E402
from tools.rules import promise_errors, render_image  # noqa: E402
from tools.sheet import ASSETS, CATALOGUE, resize_explained  # noqa: E402

FRAMES = ("background", "decor", "card", "band")
CORE = ("headline", "cta_label", "subhead", "product", "logo")


def _shown(e: dict) -> bool:
    if float(e.get("opacity", 1) or 0) <= 0:
        return False
    return e.get("type") != "text" or float(e.get("font_size", 0) or 0) > 1


def _boxes(design: dict) -> Dict[str, Box]:
    """Where each visible content element paints, clipped to the canvas."""
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    out = {}
    for e in design["elements"]:
        if not _shown(e) or str(e["id"]).startswith(FRAMES):
            continue
        if e["type"] == "text":
            b = drawn_box(e)
        else:  # where the visible pixels are, not the padded box
            b = el_box(e)
            if e["type"] == "image":
                f = ink_fraction(e.get("asset_id"), b.w, b.h, e.get("fit", "contain"))
                b = Box(b.x + f[0] * b.w, b.y + f[1] * b.h, (f[2] - f[0]) * b.w, (f[3] - f[1]) * b.h)
        x0, y0 = max(0.0, b.x), max(0.0, b.y)
        x1, y1 = min(float(W), b.x + b.w), min(float(H), b.y + b.h)
        if x1 > x0 and y1 > y0:
            out[e["id"]] = Box(x0, y0, x1 - x0, y1 - y0)
    return out


def _empty(boxes: List[Box], W: float, H: float, vertical: bool) -> float:
    """The largest band across the canvas that no content touches, as a fraction."""
    L = H if vertical else W
    spans = sorted((b.y, b.y + b.h) if vertical else (b.x, b.x + b.w) for b in boxes)
    gap, reach = 0.0, 0.0
    for a, z in spans:
        gap = max(gap, a - reach)
        reach = max(reach, z)
    return max(gap, L - reach) / L


def measures(design: dict) -> Dict[str, float]:
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    bx = _boxes(design)
    if not bx:
        return {}
    bs = list(bx.values())
    # balance is judged on the main group: something small pinned to a corner is a pattern, not a lean
    near = 0.12 * min(W, H)

    def cornered(b: Box) -> bool:
        return b.w * b.h < 0.05 * W * H and min(b.x, W - b.x - b.w) < near and min(b.y, H - b.y - b.h) < near

    main = [b for b in bs if not cornered(b)] or bs
    x0, y0 = min(b.x for b in main), min(b.y for b in main)
    x1, y1 = max(b.x + b.w for b in main), max(b.y + b.h for b in main)
    vertical = H >= W * 0.8
    m = {
        "empty": _empty(bs, W, H, vertical),
        "lopsided": abs(y0 - (H - y1)) / H if vertical else abs(x0 - (W - x1)) / W,
        "edge": min(x0, y0, W - x1, H - y1) / min(W, H),
        "fill": (x1 - x0) * (y1 - y0) / (W * H),
    }
    els = {e["id"]: e for e in design["elements"]}
    if "headline" in bx and "subhead" in bx:
        m["contrast"] = float(els["headline"]["font_size"]) / float(els["subhead"]["font_size"])
    if "product" in bx:
        m["product"] = bx["product"].w * bx["product"].h / (W * H)
    m["shown"] = {k for k in CORE if k in bx}
    return m


def flags(src: dict, out: dict, tgt: dict) -> List[str]:
    s, o = measures(src), measures(out)
    if not o:
        return ["nothing shown"]
    f = []
    lost = s["shown"] - o["shown"]
    roomy = min(tgt["width"], tgt["height"]) >= 200  # a strip keeps the core message (Adobe's 728x90 drops the product)
    if lost & {"headline", "cta_label"} or (lost and roomy):
        f.append("hidden " + ",".join(sorted(lost)))
    if o["empty"] > 0.28 and o["empty"] > s["empty"] + 0.1:
        f.append(f"empty {o['empty']:.2f} (src {s['empty']:.2f})")
    if o["lopsided"] > 0.18 and o["lopsided"] > s["lopsided"] + 0.1:
        f.append(f"lopsided {o['lopsided']:.2f} (src {s['lopsided']:.2f})")
    if o["edge"] < 0.015 and s["edge"] >= 0.03:
        f.append(f"cramped {o['edge']:.3f}")
    if o["fill"] < 0.5 * s["fill"] and o["fill"] < 0.35:
        f.append(f"sparse {o['fill']:.2f} (src {s['fill']:.2f})")
    if "contrast" in o and o["contrast"] < 1.15 and s.get("contrast", 0) >= 1.5:
        f.append(f"flat {o['contrast']:.2f}")
    if roomy and "product" in o and "product" in s and o["product"] < 0.25 * s["product"]:
        f.append(f"speck {o['product'] / s['product']:.2f}")
    errs = promise_errors(src, out, tgt)
    if errs:
        f.append("overlap " + "; ".join(errs)[:60])
    return f


def run(n: int, seed: int, formats: List[str]) -> List[Tuple[str, str, str, List[str], dict, float]]:
    rows = []
    for t in designs(n, seed):
        src = t["source"]
        for fmt in formats:
            w, h = CATALOGUE[fmt]
            tgt = {"width": w, "height": h}
            t0 = time.time()
            with redirect_stderr(io.StringIO()):
                out, plan = resize_explained(src, tgt)
            rows.append((t["task_id"] + " " + t["notes"], fmt, plan, flags(src, out, tgt), out, time.time() - t0))
    return rows


def sheet(rows, path: str, limit: int = 48, cell: int = 300) -> None:
    from PIL import Image, ImageDraw

    rows = [r for r in rows if r[3]][:limit]
    if not rows:
        return
    tiles = []
    for name, fmt, plan, f, out, _ in rows:
        im = render_image(out, ASSETS)
        k = min(cell / im.width, cell * 1.3 / im.height)
        im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))))
        t = Image.new("RGB", (cell + 8, int(cell * 1.3) + 48), (235, 235, 235))
        t.paste(im, (4, 44))
        d = ImageDraw.Draw(t)
        d.text((4, 2), f"{name[:44]}", fill=(0, 0, 0))
        d.text((4, 15), f"-> {fmt} [{plan}]", fill=(0, 0, 0))
        d.text((4, 28), "; ".join(f)[:60], fill=(170, 30, 30))
        tiles.append(t)
    cols = 6
    W = cols * (cell + 8)
    Hs = (len(tiles) + cols - 1) // cols * tiles[0].height
    s = Image.new("RGB", (W, Hs), (200, 200, 200))
    for i, t in enumerate(tiles):
        s.paste(t, ((i % cols) * (cell + 8), (i // cols) * t.height))
    s.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--formats", default=",".join(CATALOGUE))
    ap.add_argument("--sheet", default="")
    a = ap.parse_args()
    rows = run(a.n, a.seed, a.formats.split(","))
    counts: Dict[str, int] = {}
    for r in rows:
        for f in r[3]:
            k = f.split()[0]
            counts[k] = counts.get(k, 0) + 1
    flagged = sum(1 for r in rows if r[3])
    times = sorted(r[5] for r in rows)
    print(f"{len(rows)} runs, {flagged} flagged ({100 * flagged / len(rows):.0f}%); "
          f"median {times[len(times) // 2]:.2f}s, p95 {times[int(len(times) * 0.95)]:.2f}s")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s} {v}")
    for name, fmt, plan, f, _, _ in rows:
        if f:
            print(f"{name:44s} {fmt:13s} {plan:24s} {'; '.join(f)}")
    if a.sheet:
        sheet(rows, a.sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
