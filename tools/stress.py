"""Dev tool: random + degenerate designs through resize(), checking every hard rule.

    .venv/bin/python -m tools.stress --n 300 --seed 1

Reports crashes, fallbacks (smart engine raised), rule violations and timing.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.resize import resize  # noqa: E402
from engine.smart.geom import Box, rotated_aabb  # noqa: E402
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


def check(src: dict, out: dict, tgt: dict, smart: bool = True) -> list:
    errs = []
    if smart:  # what the layout engine promises (the safe fallback for impossible content does not)
        from engine.smart.items import Ctx
        from engine.smart.textfit import deliberate_overlaps, text_overlaps

        floor = Ctx(W=tgt["width"], H=tgt["height"], W0=1, H0=1).judge_floor
        for e in out["elements"]:
            if e["type"] == "text" and float(e.get("opacity", 1) or 0) > 0 and str(e.get("content", "")).strip() \
                    and e.get("font_size", 99) < floor:
                errs.append(f"{e['id']}: small print {e.get('font_size')}px < {floor}px")
        for a, b, area, _ in text_overlaps(out["elements"], exempt=deliberate_overlaps(src["elements"])):
            errs.append(f"text over text: {a} x {b} ({area:.0f} px2)")
    W, H = tgt["width"], tgt["height"]
    if out.get("canvas") != {"width": W, "height": H}:
        errs.append("canvas")
    sid = [e["id"] for e in src["elements"]]
    oid = [e.get("id") for e in out["elements"]]
    if sorted(sid) != sorted(oid) or len(set(oid)) != len(oid):
        errs.append("ids")
    so = {e["id"]: e for e in src["elements"]}
    for e in out["elements"]:
        s = so.get(e["id"], {})
        for k in ("x", "y", "width", "height"):
            v = e.get(k)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                errs.append(f"{e['id']}: bad {k}")
        b = Box(e["x"], e["y"], e["width"], e["height"])
        vb = b.union(rotated_aabb(b, float(e.get("rotation", 0) or 0)))
        if vb.x < -0.6 or vb.y < -0.6 or vb.r > W + 0.6 or vb.b > H + 0.6:
            errs.append(f"{e['id']}: off-canvas {vb}")
        if e["type"] == "text":
            if e.get("content") != s.get("content") or e.get("font_family") != s.get("font_family"):
                errs.append(f"{e['id']}: content/font changed")
            if not isinstance(e.get("font_size"), int) or e["font_size"] < 1:
                errs.append(f"{e['id']}: font_size {e.get('font_size')}")
        if e["type"] == "image" and s.get("width", 0) > 0 and s.get("height", 0) > 0 and \
                (e["width"] <= 0 or e["height"] <= 0):
            errs.append(f"{e['id']}: image collapsed to {e['width']}x{e['height']}")
        if e["type"] == "image" and s.get("width", 0) > 0 and s.get("height", 0) > 0 and e["height"] > 0:
            full = e["x"] <= 0.5 and e["y"] <= 0.5 and e.get("fit") == "cover" and e["width"] >= W - 1
            ra = s["width"] / s["height"]
            rb = e["width"] / e["height"]
            if not full and abs(rb / ra - 1) > 0.05:
                errs.append(f"{e['id']}: ratio {ra:.3f}->{rb:.3f}")
    return errs


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
        errs = check(src, out, tgt, smart=not fell_back)
        if a.render:
            try:
                from renderer.converter.convert import render_design_to_png
                with redirect_stderr(io.StringIO()):
                    render_design_to_png(out, Path(__file__).resolve().parents[1] / "brand_kit" / "assets",
                                         "/tmp/_stress_render.png")
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
