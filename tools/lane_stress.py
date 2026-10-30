"""Random and degenerate designs x real formats; every hard rule checked on every output.

    .venv/bin/python -m tools.lane_stress --n 300 --seed 1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
from engine.learn import compose  # noqa: E402
from engine.resize import resize  # noqa: E402
from tools.lane_check import FORMATS, faults  # noqa: E402

ASSETS = ["solstice_bg_gradient.png", "solstice_bottle.png", "solstice_logo_lockup.png", "missing_asset.png"]
WORDS = "chase the light golden hour botanical sparkling infusions limited edition shop now summer line " \
        "supercalifragilisticexpialidocious 50% OFF — crafted exclusively".split()


def rnd_design(r: random.Random):
    W, H = r.choice(list(FORMATS.values()))
    if r.random() < 0.3:
        W, H = r.randint(50, 3000), r.randint(50, 3000)
    els = []
    n = r.choice([0, 1, 2, 3, 5, 8, 12, 17, 25, 40]) if r.random() < 0.3 else r.randint(1, 15)
    if r.random() < 0.7:
        els.append(dict(id="bg", type=r.choice(["image", "shape"]), x=0, y=0, width=W, height=H, z_index=0,
                        asset_id=ASSETS[0], fit=r.choice(["cover", "fill", "contain"]), shape_type="rectangle",
                        fill_color="rgba(40,30,50,1)"))
    for k in range(n):
        t = r.choice(["text", "text", "image", "shape"])
        w = r.uniform(1, W * 1.2) if r.random() < 0.9 else r.choice([0, 0.5, W * 3])
        h = r.uniform(1, H * 0.6) if r.random() < 0.9 else r.choice([0, 0.5, H * 3])
        e = dict(id=f"e{k}", type=t, x=r.uniform(-0.2 * W, W), y=r.uniform(-0.2 * H, H), width=w, height=h,
                 z_index=r.randint(0, 6))
        if r.random() < 0.15:
            e["rotation"] = r.choice([-90, -15, 8, 45, 180, 270])
        if r.random() < 0.2:
            e["group_id"] = f"g{r.randint(0, 3)}"
        if r.random() < 0.2:
            e["opacity"] = r.uniform(0.05, 1)
        if t == "text":
            nw = r.choice([1, 2, 4, 8, 20, 40])
            e.update(content=" ".join(r.choice(WORDS) for _ in range(nw)) if r.random() > 0.03 else "",
                     font_family=r.choice(["Solstice Sans", "Solstice Serif", "Montserrat"]),
                     font_size=r.choice([1, 6, 12, 18, 30, 60, 120, 300]),
                     color="rgba(255,255,255,1)", text_align=r.choice(["left", "center", "right"]))
            if r.random() < 0.3:
                e["font_weight"] = r.choice(["bold", "light", "semibold"])
            if r.random() < 0.15:
                e["letter_spacing"] = r.choice([1, 4, 10])
            if r.random() < 0.1:
                e["capitalize"] = True
            if r.random() < 0.1:
                e["line_height"] = r.choice([0.8, 1.0, 1.6])
        elif t == "image":
            e.update(asset_id=r.choice(ASSETS), fit=r.choice(["contain", "cover", "fill"]))
        else:
            e.update(shape_type=r.choice(["rectangle", "ellipse", "line"]), fill_color="rgba(232,106,58,0.8)")
            if r.random() < 0.3:
                e["corner_radius"] = r.uniform(0, 50)
        els.append(e)
    return dict(canvas=dict(width=W, height=H), elements=els)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    r = random.Random(a.seed)
    crashes = [0]
    orig = compose.layout

    def wrapped(s, t):
        try:
            return orig(s, t)
        except Exception:
            crashes[0] += 1
            raise
    compose.layout = wrapped
    bad, times, n = 0, [], 0
    soft = [0]
    names = list(FORMATS)
    for d in range(a.n):
        src = rnd_design(r)
        for f in r.sample(names, 3):
            tgt = dict(width=FORMATS[f][0], height=FORMATS[f][1])
            t0 = time.time()
            out = resize(src, tgt)
            times.append(time.time() - t0)
            n += 1
            fl = [x for x in faults(src, out, tgt) if not x.startswith(("tiny", "soft"))]
            soft[0] += any(x.startswith("soft") for x in faults(src, out, tgt))
            json.dumps(out)
            if fl:
                bad += 1
                if bad <= 15:
                    print(d, f, fl[:5])
    times.sort()
    print(json.dumps(dict(designs=a.n, outputs=n, faulty=bad, soft_text_overflow=soft[0], smart_path_crashes=crashes[0],
                          p50=round(times[n // 2], 3), p95=round(times[int(n * 0.95)], 3), max=round(times[-1], 3))))


if __name__ == "__main__":
    main()
