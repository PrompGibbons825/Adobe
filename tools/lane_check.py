"""Lane-local hard-rule check (until L0's fundamentals.check is vendored): every input/ task and
tools/cases at ~20 real formats, plus optional Crello dev cases. Prints faults and timing.

    .venv/bin/python -m tools.lane_check [--crello N] [--render DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.learn import textfit  # noqa: E402
from engine.resize import resize  # noqa: E402

FORMATS = {
    "ig_post": (1080, 1080), "ig_story": (1080, 1920), "ig_portrait": (1080, 1350), "fb_post": (1200, 630),
    "linkedin": (1200, 627), "x_post": (1600, 900), "yt_thumb": (1280, 720), "pinterest": (1000, 1500),
    "mrec": (300, 250), "leader": (728, 90), "skyscraper": (160, 600), "mobile": (320, 50),
    "billboard": (970, 250), "hd": (1920, 1080), "og": (1200, 628), "a4": (2480, 3508),
    "half_page": (300, 600), "fb_cover": (820, 312), "story_ad": (1080, 1920), "wide_banner": (1600, 400),
}


def faults(src, out, tgt):
    f = []
    TW, TH = tgt["width"], tgt["height"]
    ids = [e["id"] for e in src["elements"]]
    oids = [e.get("id") for e in out["elements"]]
    if sorted(ids) != sorted(oids):
        f.append("ids")
    so = {e["id"]: e for e in src["elements"]}
    for e in out["elements"]:
        x, y, w, h = e["x"], e["y"], e["width"], e["height"]
        if x < -0.5 or y < -0.5 or x + w > TW + 0.5 or y + h > TH + 0.5:
            f.append(f"off:{e['id']}")
        s = so[e["id"]]
        if e["type"] == "image" and e.get("fit") != "cover":
            a0 = s["width"] / max(1e-6, s["height"])
            a1 = w / max(1e-6, h)
            if abs(a1 / a0 - 1) > 0.05:
                f.append(f"aspect:{e['id']}")
        if e["type"] == "text":
            for k in ("content", "font_family"):
                if e.get(k) != s.get(k):
                    f.append(f"{k}:{e['id']}")
            mw, bh, _ = textfit.layout(e, e["font_size"], w)
            if mw > w + 1:
                f.append(f"text_w:{e['id']}")
            if y + bh > TH + 1:
                f.append(f"text_h:{e['id']}")
            if int(e["font_size"]) < 6:
                f.append(f"tiny:{e['id']}")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crello", type=int, default=0)
    ap.add_argument("--render", default=None)
    a = ap.parse_args()
    tasks = []
    for p in sorted(glob.glob(str(ROOT / "input" / "*.json"))):
        src = json.load(open(p))["source"]
        for k, (w, h) in FORMATS.items():
            tasks.append((Path(p).stem.replace("chameleon_sample_", "") + "@" + k, src, dict(width=w, height=h)))
    if a.crello:
        for p in sorted(glob.glob("/data/agents/work/crello-wide/cases/*.json"))[: a.crello]:
            d = json.load(open(p))
            for t in d["targets"]:
                tasks.append((Path(p).stem + "@" + f"{t['canvas']['width']}x{t['canvas']['height']}", d["source"], t["canvas"]))
    nf, times = 0, []
    for name, src, tgt in tasks:
        t0 = time.time()
        out = resize(src, tgt)
        times.append(time.time() - t0)
        f = faults(src, out, tgt)
        if f:
            nf += 1
            print(name, f[:6])
        if a.render and "@" in name and not name[0].isdigit():
            from renderer.converter.convert import render_design_to_png
            Path(a.render).mkdir(parents=True, exist_ok=True)
            render_design_to_png(out, ROOT / "brand_kit" / "assets", str(Path(a.render) / f"{name}.png"))
    times.sort()
    print(json.dumps(dict(tasks=len(tasks), faulty=nf, p50=round(times[len(times) // 2], 3),
                          p95=round(times[int(len(times) * 0.95)], 3), max=round(times[-1], 3))))


if __name__ == "__main__":
    main()
