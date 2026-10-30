"""Offline dev eval on Crello dev cases outside the tournament set: mean per-element box IoU with
the designer's own resize (ids correspond), by aspect class. Not a quality score, a closeness one.

    .venv/bin/python -m tools.learn_eval --mode retrieve --beta 0.5 [--repair 0] [--n 200]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import logging  # noqa: E402
logging.disable(logging.WARNING)
from engine.learn import compose  # noqa: E402
from engine.resize import resize  # noqa: E402

WIDE = Path("/data/agents/work/crello-wide")


def cls(W, H):
    ar = W / H
    return "extreme" if ar >= 6 or ar <= 0.35 else "strip" if ar >= 3.5 else "wide" if ar >= 1.8 else "tall" if ar < 0.75 else "square"


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def fr(e, W, H):
    return (e["x"] / W, e["y"] / H, (e["x"] + e["width"]) / W, (e["y"] + e["height"]) / H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="retrieve")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--repair", type=int, default=1)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--set", default="dev")
    ap.add_argument("--reflow", type=float, default=None)  # dev = not in tournament; tour = tournament set
    a = ap.parse_args()
    compose.MODE.update(transfer=a.mode, beta=a.beta, repair=bool(a.repair))
    if a.reflow is not None:
        compose.REFLOW_DAR = a.reflow
    tour = set(json.load(open(WIDE / "tournament_set.json"))["tasks"])
    tasks = []
    for p in sorted((WIDE / "cases").glob("*.json")):
        d = json.load(open(p))
        for k, t in enumerate(d["targets"]):
            name = f"{d['family']}_{k}"
            if (name in tour) == (a.set == "tour"):
                tasks.append((name, d["source"], t))
    step = max(1, len(tasks) // a.n)
    tasks = tasks[::step][: a.n]
    res, by = [], {}
    t0 = time.time()
    for name, src, ref in tasks:
        tgt = ref["canvas"]
        out = resize(src, tgt)
        W, H = tgt["width"], tgt["height"]
        ro = {e["id"]: e for e in ref["elements"]}
        v = [iou(fr(e, W, H), fr(ro[e["id"]], W, H)) for e in out["elements"] if e["id"] in ro]
        if v:
            s = sum(v) / len(v)
            res.append(s)
            by.setdefault(cls(W, H), []).append(s)
    print(json.dumps(dict(reflow=compose.REFLOW_DAR, mode=a.mode, beta=a.beta, repair=a.repair, n=len(res), iou=round(sum(res) / len(res), 4),
                          by={k: (len(v), round(sum(v) / len(v), 4)) for k, v in sorted(by.items())},
                          secs=round(time.time() - t0))))


if __name__ == "__main__":
    main()
