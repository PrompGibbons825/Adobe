"""Dev tool: random + degenerate designs through resize(), checking every hard rule.

    .venv/bin/python -m tools.stress --n 300 --seed 1

Reports crashes, fallbacks (smart engine raised), rule violations and timing.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.resize import resize  # noqa: E402
from tools.rules import hard_rule_errors, promise_errors, render_image  # noqa: E402
from tools.sheet import CATALOGUE  # noqa: E402

ASSETS = ["solstice_bottle.png", "solstice_logo_lockup.png", "solstice_bg_gradient.png", "does_not_exist.png"]
WORDS = ("golden hour light sparkling botanical summer sale new limited batch crafted orange honey rosemary "
         "taste sunset evening order now shop collection discover the a of for with and in").split()
FONTS = ["Solstice Serif", "Solstice Sans", "Comic Sans", ""]
COLORS = ["rgba(255,255,255,1)", "rgba(232,106,58,1)", "rgba(43,27,46,1)", "rgba(240,178,66,0.8)", "#ffffff",
          "#E86A3A80", "white", "hsl(10, 50%, 50%)", None]


def words(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize()


def random_design(rng: random.Random) -> dict:
    W, H = rng.choice(list(CATALOGUE.values()))
    els = []
    z = 0
    if rng.random() < 0.85:
        els.append({"id": "bg", "type": "image", "x": 0, "y": 0, "width": W, "height": H, "z_index": z,
                    "asset_id": "solstice_bg_gradient.png", "fit": "cover"})
    elif rng.random() < 0.5:
        els.append({"id": "bg", "type": "shape", "x": 0, "y": 0, "width": W, "height": H, "z_index": z,
                    "shape_type": "rectangle", "fill_color": "rgba(250,240,228,1)"})
    for i in range(rng.randint(0, 3)):
        z += 1
        s = rng.uniform(0.1, 0.6) * min(W, H)
        els.append({"id": f"deco{i}", "type": "shape", "x": rng.uniform(-0.2, 1.0) * W, "y": rng.uniform(-0.2, 1.0) * H,
                    "width": s, "height": s, "z_index": z, "shape_type": rng.choice(["ellipse", "rectangle"]),
                    "fill_color": "rgba(255,255,255,1)", "opacity": rng.uniform(0.05, 0.3)})
    n_img = rng.randint(0, 3)
    for i in range(n_img):
        z += 1
        w = rng.uniform(0.1, 0.6) * W
        h = w * rng.choice([1.0, 0.33, 0.5, 1.5])
        els.append({"id": f"img{i}", "type": "image", "x": rng.uniform(0, 0.9) * W, "y": rng.uniform(0, 0.9) * H,
                    "width": w, "height": h, "z_index": z, "asset_id": rng.choice(ASSETS),
                    "fit": rng.choice(["contain", "contain", "cover", "fill"]),
                    "rotation": rng.choice([0, 0, 0, 8, -15])})
    for i in range(rng.randint(0, 5)):
        z += 1
        fs = rng.uniform(0.012, 0.08) * max(W, H)
        n = rng.choice([2, 4, 8, 20, 40])
        e = {"id": f"txt{i}", "type": "text", "x": rng.uniform(0, 0.8) * W, "y": rng.uniform(0, 0.9) * H,
             "width": rng.uniform(0.2, 0.9) * W, "height": fs * 2, "z_index": z, "content": words(rng, n),
             "font_family": rng.choice(FONTS), "font_size": fs, "color": rng.choice(COLORS),
             "text_align": rng.choice(["left", "center", "right"])}
        if rng.random() < 0.3:
            e["font_weight"] = rng.choice(["bold", "bold", 700, "semibold"])
        if rng.random() < 0.05:
            e["text_align"] = rng.choice(["justify", "middle"])
        if rng.random() < 0.1:
            e["style_runs"] = [{"start": 0, "end": 3, "color": "rgba(240,178,66,1)"}]
        if rng.random() < 0.1:
            e["line_height"] = rng.choice([0.9, 1.5, 2.0])
        if rng.random() < 0.1:
            e["letter_spacing"] = rng.choice([2, 5])
        if rng.random() < 0.05:
            e["content"] = ""
        if rng.random() < 0.05:
            e["content"] = "Line one\nLine two"
        els.append(e)
    for i in range(rng.randint(0, 3)):
        z += 1
        w, h = rng.uniform(0.15, 0.45) * W, rng.uniform(0.04, 0.1) * H
        x, y = rng.uniform(0, 0.8) * W, rng.uniform(0, 0.9) * H
        g = f"btn{i}" if rng.random() < 0.7 else None
        rot = rng.choice([0, 0, 0, -12])
        els.append({"id": f"btn{i}_bg", "type": "shape", "x": x, "y": y, "width": w, "height": h, "z_index": z,
                    "shape_type": "rectangle", "fill_color": "rgba(232,106,58,1)", "corner_radius": h / 2,
                    "group_id": g, "rotation": rot})
        z += 1
        fs = h * 0.4
        els.append({"id": f"btn{i}_label", "type": "text", "x": x, "y": y + (h - fs * 1.4) / 2, "width": w,
                    "height": fs * 1.4, "z_index": z, "content": words(rng, rng.randint(1, 4)),
                    "font_family": "Solstice Sans", "font_size": fs, "font_weight": "bold",
                    "color": "rgba(255,255,255,1)", "text_align": "center", "group_id": g, "rotation": rot})
    if rng.random() < 0.05:
        els.append({"id": "zero", "type": "image", "x": 10, "y": 10, "width": 0, "height": 0, "z_index": 99,
                    "asset_id": "solstice_logo_lockup.png"})
    if rng.random() < 0.05:
        els.append({"id": "offcanvas", "type": "text", "x": W * 2, "y": H * 2, "width": 100, "height": 40,
                    "z_index": 99, "content": "Far away", "font_family": "Solstice Sans", "font_size": 20,
                    "color": "rgba(255,255,255,1)"})
    if not els:
        els.append({"id": "only", "type": "text", "x": 10, "y": 10, "width": 200, "height": 40, "z_index": 0,
                    "content": "Alone", "font_family": "Solstice Sans", "font_size": 24, "color": "rgba(0,0,0,1)"})
    return {"canvas": {"width": W, "height": H}, "elements": els}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true", help="also push every output through the real renderer")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    targets = list(CATALOGUE.values())
    fails, fallbacks, worst = 0, 0, (0.0, None)
    times = []
    for i in range(a.n):
        src = random_design(rng)
        W, H = rng.choice(targets)
        tgt = {"width": W, "height": H}
        buf = io.StringIO()
        t0 = time.time()
        try:
            with redirect_stderr(buf):
                out = resize(json.loads(json.dumps(src)), tgt)
        except Exception as ex:  # noqa: BLE001
            print(f"[{i}] CRASH {ex!r}")
            fails += 1
            continue
        dt = time.time() - t0
        times.append(dt)
        if dt > worst[0]:
            worst = (dt, i)
        fell_back = "Traceback" in buf.getvalue()
        if fell_back:
            fallbacks += 1
            print(f"[{i}] FALLBACK {src['canvas']} -> {tgt}\n" + buf.getvalue().strip().splitlines()[-1])
        # the safe fallback (impossible content) only promises the hard rules
        errs = hard_rule_errors(src, out, tgt) + ([] if fell_back else promise_errors(src, out, tgt))
        if a.render:
            try:
                with redirect_stderr(io.StringIO()):
                    render_image(out)
            except Exception as ex:  # noqa: BLE001
                errs.append(f"RENDER CRASH {type(ex).__name__}: {str(ex)[:120]}")
        if errs:
            fails += 1
            print(f"[{i}] {src['canvas']} -> {tgt}: {errs[:4]}")
    times.sort()
    print(f"\n{a.n} designs: {fails} rule failures, {fallbacks} fallbacks; "
          f"median {times[len(times) // 2]:.2f}s, p95 {times[int(len(times) * 0.95)]:.2f}s, worst {worst[0]:.2f}s (#{worst[1]})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
