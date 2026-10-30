"""Random / degenerate designs x formats through resize(); every output must pass tools.lane.check.

    python -m tools.stress_l3 --n 300 --seed 1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.lane import FORMATS, check  # noqa: E402

WORDS = ("summer sale light golden botanical sparkling new limited edition shop now free entry "
         "festival december resort tickets supercalifragilisticexpialidocious 2026 % — & join us").split()
ASSETS = ["solstice_bg_gradient.png", "solstice_bottle.png", "solstice_logo_lockup.png", "missing_asset.png"]


def rand_design(rng: random.Random) -> dict:
    W, H = rng.choice(list(FORMATS.values()))
    if rng.random() < 0.2:
        W, H = rng.randint(50, 4000), rng.randint(40, 4000)
    els = []
    z = 0
    if rng.random() < 0.7:
        els.append(dict(id="bg", type=rng.choice(["image", "shape"]), x=0, y=0, width=W, height=H, z_index=z,
                        asset_id=ASSETS[0], fit=rng.choice(["cover", "fill", "contain"]), shape_type="rectangle",
                        fill_color="rgba(30,20,40,1)"))
        z += 1
    for k in range(rng.randint(1, 16)):
        t = rng.choice(["text", "text", "image", "shape"])
        w = rng.uniform(0.02, 1.1) * W
        h = rng.uniform(0.02, 0.6) * H
        x = rng.uniform(-0.15, 1.0) * W
        y = rng.uniform(-0.15, 1.0) * H
        e = dict(id=f"e{k}", type=t, x=x, y=y, width=max(0.5, w), height=max(0.5, h), z_index=z)
        z += 1
        if rng.random() < 0.15:
            e["rotation"] = rng.uniform(-30, 30)
        if rng.random() < 0.2:
            e["group_id"] = f"g{rng.randint(0, 3)}"
        if t == "text":
            n = rng.choice([1, 2, 3, 6, 15, 40])
            e.update(content=" ".join(rng.choice(WORDS) for _ in range(n)),
                     font_family=rng.choice(["Solstice Sans", "Solstice Serif", "Unknown Font"]),
                     font_size=rng.choice([6, 12, 18, 24, 40, 72, 160]), color="rgba(255,255,255,1)",
                     text_align=rng.choice(["left", "center", "right"]))
            if rng.random() < 0.3:
                e["font_weight"] = "bold"
            if rng.random() < 0.1:
                e["letter_spacing"] = rng.uniform(0, 8)
        elif t == "image":
            e.update(asset_id=rng.choice(ASSETS), fit=rng.choice(["contain", "cover", "fill"]))
        else:
            e.update(shape_type=rng.choice(["rectangle", "ellipse", "line"]),
                     fill_color=f"rgba(200,100,50,{rng.choice([1, 0.2])})", corner_radius=rng.choice([0, 20]))
        els.append(e)
    return {"canvas": {"width": W, "height": H}, "elements": els}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from engine.resize import resize
    rng = random.Random(a.seed)
    fmts = list(FORMATS.items())
    bad = 0
    early = []
    times = []
    for i in range(a.n):
        src = rand_design(rng)
        name, (W, H) = fmts[i % len(fmts)]
        t0 = time.time()
        out = resize(json.loads(json.dumps(src)), {"width": W, "height": H})
        times.append(time.time() - t0)
        errs = check(src, out, {"width": W, "height": H})
        out2 = resize(json.loads(json.dumps(src)), {"width": W, "height": H})
        if json.dumps(out2, sort_keys=True) != json.dumps(out, sort_keys=True):
            errs.append("nondeterministic")
        if i < 25:
            early.append((src, (W, H), json.dumps(out, sort_keys=True)))
        if errs:
            bad += 1
            print(i, name, errs[:6], flush=True)
    for k, (src, (W, H), ref) in enumerate(early):   # order independence: recompute after warm caches
        if json.dumps(resize(json.loads(json.dumps(src)), {"width": W, "height": H}), sort_keys=True) != ref:
            bad += 1
            print("order-dependent", k, flush=True)
    times.sort()
    print(f"stress n={a.n} faulty={bad} p50={times[len(times)//2]:.2f}s p95={times[int(.95*len(times))]:.2f}s max={times[-1]:.2f}s")


if __name__ == "__main__":
    main()
