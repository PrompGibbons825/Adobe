"""Fit the energy weights on Crello designer resizes (NIO with a convex surrogate).

Data: /data/agents/work/crello-wide/families.jsonl (CDLA-Permissive-2.0), geometry only.
Families in the dev cases/ sample (which holds the tournament set) are excluded.
For each (source member -> target member) pair, the designer's target layout must have lower
energy than (a) every candidate our search generates and (b) random perturbations of the
designer layout. Pairwise logistic loss on feature differences, weights >= 0, L2 to the prior.

    python -m tools.fit --pairs 600 --out engine/opt/weights.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.opt.analyse import analyse  # noqa: E402
from engine.opt.energy import DEFAULT, NAMES, features  # noqa: E402
from engine.opt.layout import Placed  # noqa: E402
from engine.opt.measure import fast_block  # noqa: E402
from engine.opt.solve import search  # noqa: E402

WIDE = Path("/data/agents/work/crello-wide")


def to_design(m: dict) -> dict:
    els = []
    for k, e in enumerate(m["els"]):
        base = dict(id=f"e{k}", x=e["x"], y=e["y"], width=max(1.0, e["w"]), height=max(1.0, e["h"]),
                    z_index=k, rotation=e.get("angle", 0.0) or 0.0, opacity=e.get("opacity", 1.0))
        kind = e["kind"]
        if kind == "TextElement":
            base.update(type="text", content=e["text"], font_family=e.get("font") or "Montserrat",
                        font_size=max(1.0, e["fs"] or 12), text_align=e.get("align") or "left",
                        color=e.get("color") or "rgba(0,0,0,1)", line_height=1.0)
            if e.get("bold"):
                base["font_weight"] = "bold"
        elif kind == "ColoredBackground":
            base.update(type="shape", shape_type="rectangle", x=0, y=0, width=m["W"], height=m["H"],
                        fill_color=e.get("color") or "rgba(255,255,255,1)")
        elif kind == "ImageElement" or kind == "SvgMaskElement":
            base.update(type="image", asset_id=f"x{k}.png", fit="fill")
        else:
            base.update(type="shape", shape_type="rectangle", fill_color="rgba(128,128,128,1)")
        els.append(base)
    return {"canvas": {"width": m["W"], "height": m["H"]}, "elements": els}


def match(src: dict, tgt: dict):
    """source element index -> target element index (texts by content, others by type order)."""
    out = {}
    used = set()
    st = [(i, e) for i, e in enumerate(src["elements"])]
    tt = [(i, e) for i, e in enumerate(tgt["elements"])]
    for i, e in st:
        for j, f in tt:
            if j in used or f["type"] != e["type"]:
                continue
            if e["type"] == "text" and f.get("content") != e.get("content"):
                continue
            out[i] = j
            used.add(j)
            break
    return out


def designer_placed(S, tgt: dict, mp: dict):
    placed = {}
    for b in S.blocks:
        if any(i not in mp for i in b.members):
            return None
        bx = [tgt["elements"][mp[i]] for i in b.members]
        x0 = min(e["x"] for e in bx)
        y0 = min(e["y"] for e in bx)
        x1 = max(e["x"] + e["width"] for e in bx)
        y1 = max(e["y"] + e["height"] for e in bx)
        s = math.sqrt(max(1.0, (x1 - x0) * (y1 - y0)) / max(1.0, b.w * b.h))
        pl = Placed(x0, y0, x1 - x0, y1 - y0, s)
        if b.kind == "text":
            e = bx[0]
            pl.fs = max(1, int(round(e["font_size"])))
            pl.lines = fast_block(e, pl.fs, e["width"])[0]
        placed[b.bid] = pl
    return placed


def perturb(placed, W, H, rng):
    import copy
    p = copy.deepcopy(placed)
    keys = list(p)
    mode = rng.choice(["shift", "scale", "swap", "shrink", "squash"])
    if mode == "shift":
        k = rng.choice(keys)
        p[k].x += rng.uniform(-0.25, 0.25) * W
        p[k].y += rng.uniform(-0.25, 0.25) * H
    elif mode == "scale":
        k = rng.choice(keys)
        f = rng.choice([0.5, 0.7, 1.4, 1.8])
        cx, cy = p[k].x + p[k].w / 2, p[k].y + p[k].h / 2
        p[k].w *= f
        p[k].h *= f
        p[k].x, p[k].y = cx - p[k].w / 2, cy - p[k].h / 2
        if p[k].fs:
            p[k].fs = max(1, int(p[k].fs * f))
    elif mode == "swap" and len(keys) > 1:
        a, b = rng.sample(keys, 2)
        ca = (p[a].x + p[a].w / 2, p[a].y + p[a].h / 2)
        cb = (p[b].x + p[b].w / 2, p[b].y + p[b].h / 2)
        p[a].x, p[a].y = cb[0] - p[a].w / 2, cb[1] - p[a].h / 2
        p[b].x, p[b].y = ca[0] - p[b].w / 2, ca[1] - p[b].h / 2
    elif mode == "shrink":
        f = rng.choice([0.6, 0.75])
        for k in keys:
            q = p[k]
            q.x = W / 2 + (q.x - W / 2) * f
            q.y = H / 2 + (q.y - H / 2) * f
            q.w *= f
            q.h *= f
            if q.fs:
                q.fs = max(1, int(q.fs * f))
    else:
        f = rng.choice([0.5, 0.7])
        ax = rng.choice([0, 1])
        for k in keys:
            q = p[k]
            if ax == 0:
                q.y = H / 2 + (q.y - H / 2) * f
            else:
                q.x = W / 2 + (q.x - W / 2) * f
    return p


def iou(a, b):
    iw = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    ih = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    i = max(0, iw) * max(0, ih)
    u = a.w * a.h + b.w * b.h - i
    return i / u if u > 0 else 0.0


def build(n_pairs: int, seed: int):
    rng = random.Random(seed)
    cases = {p.stem for p in (WIDE / "cases").glob("*.json")}
    fams = [json.loads(l) for l in open(WIDE / "families.jsonl")]
    fams = [f for f in fams if f["fam"] not in cases]
    rng.shuffle(fams)
    rows = []
    t0 = time.time()
    for f in fams:
        if len(rows) >= n_pairs:
            break
        ms = f["members"]
        if len(ms) < 2:
            continue
        a, b = rng.sample(ms, 2)
        if a["n_el"] > 25:
            continue
        src, tgt = to_design(a), to_design(b)
        S = analyse(src)
        if not S.blocks:
            continue
        mp = match(src, tgt)
        des = designer_placed(S, tgt, mp)
        if des is None:
            continue
        W, H = float(b["W"]), float(b["H"])
        try:
            scored = search(S, W, H)
        except Exception:  # noqa: BLE001
            continue
        fd = features(S, des, W, H)
        cands = [(name, pl) for _, _, name, pl in scored]
        cf = [features(S, pl, W, H) for _, pl in cands]
        sims = [np.mean([iou(pl[k], des[k]) for k in des]) for _, pl in cands]
        pf = [features(S, perturb(des, W, H, rng), W, H) for _ in range(4)]
        rows.append(dict(fam=f["fam"], fd=fd, cf=cf, sims=sims, pf=pf, ar=(a["W"] / a["H"], W / H)))
        if len(rows) % 50 == 0:
            print(len(rows), f"{time.time() - t0:.0f}s", flush=True)
    return rows


def fit(rows, prior, lam=0.05, iters=3000, lr=0.05):
    X = []
    for r in rows:
        fd = np.array(r["fd"])
        for c in r["cf"] + r["pf"]:
            d = np.array(c) - fd           # want w.d > 0
            if np.abs(d).sum() > 1e-9:
                X.append(d)
        # listwise: a candidate closer to the designer (IoU) should have lower energy
        cf, sims = r["cf"], r["sims"]
        order = sorted(range(len(cf)), key=lambda i: -sims[i])
        for i in order[:3]:
            for j in order[3:]:
                if sims[i] > sims[j] + 0.05:
                    d = np.array(cf[j]) - np.array(cf[i])
                    if np.abs(d).sum() > 1e-9:
                        X.append(d * 0.5)
    X = np.array(X)
    scale = X.std(0) + 1e-6
    Xs = X / scale
    w0 = np.array([prior[k] for k in NAMES]) * scale
    w0 = w0 / (np.linalg.norm(w0) + 1e-9) * 3
    w = w0.copy()
    for _ in range(iters):
        m = Xs @ w
        g = -(Xs * (1 / (1 + np.exp(np.clip(m, -30, 30))))[:, None]).mean(0) + lam * (w - w0)
        w = np.maximum(0.0, w - lr * g)
    return w / scale


def evaluate(rows, w):
    w = np.array(w)
    acc, n, sel, orc, top = 0, 0, [], [], []
    for r in rows:
        ed = np.dot(w, r["fd"])
        for c in r["cf"] + r["pf"]:
            n += 1
            acc += np.dot(w, c) > ed
        es = [np.dot(w, c) for c in r["cf"]]
        k = int(np.argmin(es))
        sel.append(r["sims"][k])
        orc.append(max(r["sims"]))
    return dict(pair_acc=acc / max(1, n), sel_iou=float(np.mean(sel)), oracle_iou=float(np.mean(orc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="/data/agents/work/night/lane-optimize/fitrows.json")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    cp = Path(a.cache)
    if cp.exists() and not a.rebuild:
        rows = json.loads(cp.read_text())
    else:
        rows = build(a.pairs, a.seed)
        cp.write_text(json.dumps(rows))
    fams = sorted({r["fam"] for r in rows})
    random.Random(1).shuffle(fams)
    test = set(fams[: len(fams) // 5])
    tr = [r for r in rows if r["fam"] not in test]
    te = [r for r in rows if r["fam"] in test]
    d = np.array([DEFAULT[k] for k in NAMES])
    print("pairs", len(rows), "train", len(tr), "test", len(te))
    print("prior   test", evaluate(te, d), "train", evaluate(tr, d))
    best = None
    for lam in (0.3, 0.1, 0.03, 0.01):
        w = fit(tr, DEFAULT, lam=lam)
        ev = evaluate(te, w)
        print(f"lam {lam} test", ev, "train", evaluate(tr, w))
        if best is None or ev["sel_iou"] + ev["pair_acc"] > best[0]:
            best = (ev["sel_iou"] + ev["pair_acc"], lam, w)
    _, lam, _ = best
    wf = fit(rows, DEFAULT, lam=lam)      # refit on all with the chosen lambda
    print("chosen lam", lam, {k: round(float(v), 3) for k, v in zip(NAMES, wf)})
    if a.out:
        Path(a.out).write_text(json.dumps({"w": {k: float(v) for k, v in zip(NAMES, wf)}, "lam": lam,
                                           "n_pairs": len(rows), "data": "crello-wide families.jsonl minus cases/",
                                           "test": evaluate(te, best[2])}, indent=1))


if __name__ == "__main__":
    main()
