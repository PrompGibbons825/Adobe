"""A tiny learned re-ranker for candidate layouts, trained on designer resizes.

The hand-weighted score (score.py) encodes what a layout must not do. What it
cannot know is which of several valid arrangements a designer would pick for
this design at this size: keep the source arrangement or reflow, a column or
hero-beside-copy, how far to scale type on a strip. Crello holds designers'
own answers to exactly that question (the same template re-made at other
sizes), so a linear pairwise ranker was fitted on several hundred of those
families: for every candidate the engine generates, how close is it to the
designer's version (tools/designer_sim.py)? See tools/ranker_train.py.

At run time it only *re-orders*. The ranker's value is blended into the hand
score (score + BLEND * r, BLEND tuned on training families); the pixel
re-rank and the hard checks (no copy over copy, legibility floor) still run
afterwards on the blended order, so a candidate that fails them is never
shipped because the ranker liked it. Without the weights file, or with
CHAMELEON_RANKER=0, the hand score is used unchanged.

Inference is pure Python (a dot product over ~100 features).
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from .geom import Box

WEIGHTS = Path(__file__).with_name("ranker_weights.json")
ARCHS = ("adapt", "stack", "stack-side", "split", "split-fold", "row", "row-stacked")
TERMS = ("visible", "hidden", "small_text", "lines", "overlap", "coverage", "balance", "prior")


def enabled() -> bool:
    return os.environ.get("CHAMELEON_RANKER", "1") != "0" and model() is not None


@lru_cache(maxsize=1)
def model() -> Optional[dict]:
    try:
        m = json.loads(WEIGHTS.read_text())
        if m.get("kind") != "linear-pairwise" or not m.get("features"):
            return None
        return m
    except Exception:
        return None


def arch_of(name: str) -> Optional[str]:
    if name.startswith("adapt"):
        return "adapt"
    if name.startswith("stack-"):
        return "stack-side"
    if name.startswith("stack"):
        return "stack"
    if name.startswith("split"):
        return "split-fold" if "-fold" in name else "split"
    if name.startswith("row"):
        return "row-stacked" if "stacked" in name else "row"
    return None


# ----------------------------------------------------------------------------
# features


def context(d, ctx) -> Dict[str, float]:
    """Per (design, target) features: constant across candidates, so they only
    matter through their products with the archetype."""
    lt = math.log(ctx.W / ctx.H)
    ls = math.log(d.W / d.H)
    texts = [it for it in d.items if it.kind == "text" and it.role not in ("missing", "void")]
    chars = sum(len(str(e.get("content", ""))) for it in d.items for e in it.els if e.get("type") == "text")
    tarea = sum(it.src_box.area for it in texts) / max(1.0, d.W * d.H)
    return dict(lt=lt, dl=lt - ls, adl=abs(lt - ls), la=0.5 * math.log(ctx.W * ctx.H / max(1.0, d.W * d.H)),
                chars=math.log1p(chars), tarea=tarea, ntext=float(len(texts)),
                wide=max(0.0, lt - math.log(2.4)), tall=max(0.0, math.log(1 / 1.6) - lt))


def _nbox(b: Box, W: float, H: float):
    return (b.x / W, b.y / H, b.w / W, b.h / H)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 1e-12 else 0.0


def _rel(a, b):
    h = -1 if a[0] + a[2] <= b[0] + 0.01 else (1 if b[0] + b[2] <= a[0] + 0.01 else 0)
    v = -1 if a[1] + a[3] <= b[1] + 0.01 else (1 if b[1] + b[3] <= a[1] + 0.01 else 0)
    return h, v


def plan_features(p, d, ctx, cx: Dict[str, float]) -> Dict[str, float]:
    W, H = ctx.W, ctx.H
    f: Dict[str, float] = {}
    a = arch_of(p.name)
    for k in ARCHS:
        f["a:" + k] = 1.0 if a == k else 0.0
    f["split_frac"] = 0.48 if "0.48" in p.name else (0.34 if "0.34" in p.name else 0.0)
    for t in TERMS:
        f["t:" + t] = p.breakdown.get(t, 0.0) / 100.0
    tot_imp = sum(it.importance for it in d.items if it.role not in ("missing", "void")) or 1.0
    hid = [h for h in p.hidden if h.role not in ("missing", "void")]
    f["n_hidden"] = float(len(hid))
    f["hidden_imp"] = sum(h.importance for h in hid) / tot_imp
    f["relax"] = 1.0 - p.relax
    f["log_s"] = math.log(max(1e-4, p.s)) - cx["la"]
    placed = p.placed + p.floaters
    src, out, fsr, imp = [], [], [], []
    text_area = 0.0
    hero_rel = 0.0
    for r, x, y in placed:
        it = r.item
        b = Box(x, y, r.w, r.h)
        src.append(_nbox(it.src_box, d.W, d.H))
        out.append(_nbox(b, W, H))
        imp.append(it.importance)
        if it.kind == "text":
            text_area += b.area
        if it.role == "hero" and it.src_box.area > 0:
            hero_rel = math.log(max(1e-4, (b.area / (W * H)) / (it.src_box.area / (d.W * d.H))))
        for eid, fs in r.fonts.items():
            fs0 = float(d.by_id.get(eid, {}).get("font_size", 0) or 0)
            if fs0 > 0:
                fsr.append((it.role, math.log(fs / fs0) - cx["la"], fs0, fs))
    f["hero_rel"] = hero_rel
    f["text_cov"] = text_area / (W * H)
    n = len(out)
    if n:
        f["pos_iou"] = sum(_iou(s, o) for s, o in zip(src, out)) / n
        f["cx_shift"] = sum(abs((s[0] + s[2] / 2) - (o[0] + o[2] / 2)) for s, o in zip(src, out)) / n
        f["cy_shift"] = sum(abs((s[1] + s[3] / 2) - (o[1] + o[3] / 2)) for s, o in zip(src, out)) / n
        area = sum(o[2] * o[3] for o in out)
        f["cov"] = area
        f["margin"] = min(min(o[0], o[1], 1 - o[0] - o[2], 1 - o[1] - o[3]) for o in out)
    keep_h = keep_v = pairs = 0
    order = 0
    for i in range(n):
        for j in range(i + 1, n):
            rs, ro = _rel(src[i], src[j]), _rel(out[i], out[j])
            pairs += 1
            keep_h += rs[0] == ro[0]
            keep_v += rs[1] == ro[1]
            # reading order: top-to-bottom, then left-to-right, by centres
            ks = (round(src[i][1] + src[i][3] / 2, 2), src[i][0]) < (round(src[j][1] + src[j][3] / 2, 2), src[j][0])
            ko = (round(out[i][1] + out[i][3] / 2, 2), out[i][0]) < (round(out[j][1] + out[j][3] / 2, 2), out[j][0])
            order += ks == ko
    f["keep_h"] = keep_h / pairs if pairs else 1.0
    f["keep_v"] = keep_v / pairs if pairs else 1.0
    f["keep_order"] = order / pairs if pairs else 1.0
    if fsr:
        vals = sorted(v for _, v, _, _ in fsr)
        f["font_med"] = vals[len(vals) // 2]
        f["font_min"] = vals[0]
        f["font_max"] = vals[-1]
        head = [v for role, v, _, _ in fsr if role == "headline"]
        f["font_head"] = head[0] if head else f["font_med"]
        agree = tot = 0
        for i in range(len(fsr)):
            for j in range(i + 1, len(fsr)):
                if abs(math.log(fsr[i][2] / fsr[j][2])) < 0.05:
                    continue
                tot += 1
                agree += (fsr[i][2] > fsr[j][2]) == (fsr[i][3] > fsr[j][3])
        f["hier_keep"] = agree / tot if tot else 1.0
        f["min_fs_floor"] = math.log(min(fs for _, _, _, fs in fsr) / ctx.judge_floor)
    # archetype x context: when each arrangement suits the shape and the load
    for k in ARCHS:
        if f["a:" + k]:
            for c in ("lt", "dl", "adl", "chars", "tarea", "ntext", "wide", "tall"):
                f[f"a:{k}*{c}"] = cx[c]
    # keeping the arrangement / scaling type pays off differently by shape change
    for c in ("adl", "wide", "tall"):
        f[f"pos_iou*{c}"] = f.get("pos_iou", 0.0) * cx[c]
        f[f"keep_v*{c}"] = f["keep_v"] * cx[c]
        f[f"keep_h*{c}"] = f["keep_h"] * cx[c]
        f[f"font_med*{c}"] = f.get("font_med", 0.0) * cx[c]
        f[f"log_s*{c}"] = f["log_s"] * cx[c]
    return f


def all_features(plans, d, ctx) -> List[Dict[str, float]]:
    cx = context(d, ctx)
    return [plan_features(p, d, ctx, cx) for p in plans]


def predict(feats: Dict[str, float], m: dict) -> float:
    s = 0.0
    for name, w, mu, sd in zip(m["features"], m["weights"], m["mean"], m["std"]):
        s += w * (feats.get(name, 0.0) - mu) / sd
    return s


def apply(plans, d, ctx) -> None:
    """Blend the learned preference into each plan's score (in place)."""
    if not plans or not enabled():
        return
    m = model()
    feats = all_features(plans, d, ctx)
    rs = [predict(f, m) for f in feats]
    top = max(rs)
    lam = float(m.get("blend", 0.0))
    for p, r in zip(plans, rs):
        bonus = lam * (r - top)
        p.breakdown["ranker"] = bonus
        p.score += bonus
