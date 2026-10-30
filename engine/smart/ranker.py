"""Tiny learned re-ranker for candidate layouts (trained on designer Crello resizes).

Re-orders candidates the engine already generates; does not change pixels, copy or ids.
Hand score (score.py) still supplies hard constraints via prune_relaxed, pixel re-rank
and the no-copy-over-copy check. Without ranker_weights.json, or with CHAMELEON_RANKER=0,
the hand score alone decides. Inference is a dot product (linear) or a small stump ensemble.
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
ARCHS = (
    "scaled", "adapt", "regions", "stack", "stack-side", "stack-v", "split", "split-fold",
    "row", "row-stacked", "anchor", "flex", "retrieve", "quiet",
)
TERMS = (
    "visible", "hidden", "small_text", "lines", "overlap", "coverage", "balance", "prior",
    "density", "hierarchy", "inflation", "fill", "shrunk", "trivial",
)


def enabled() -> bool:
    return os.environ.get("CHAMELEON_RANKER", "1") != "0" and model() is not None


@lru_cache(maxsize=1)
def model() -> Optional[dict]:
    try:
        m = json.loads(WEIGHTS.read_text())
        if not m.get("features"):
            return None
        return m
    except Exception:
        return None


def arch_of(name: str) -> Optional[str]:
    base = name.split("@")[0].split("~")[0]
    for k in ("anchor", "flex", "retrieve", "quiet"):
        if base.startswith(k):
            return k
    if base.startswith("scaled"):
        return "scaled"
    if base.startswith("adapt"):
        return "adapt"
    if base.startswith("regions"):
        return "regions"
    if base.startswith("stack-top") or base.startswith("stack-bottom"):
        return "stack-v"
    if base.startswith("stack-start") or base.startswith("stack-end"):
        return "stack-side"
    if base.startswith("stack"):
        return "stack"
    if base.startswith("split"):
        return "split-fold" if "-fold" in base else "split"
    if base.startswith("row"):
        return "row-stacked" if "stacked" in base else "row"
    return None


def context(d, ctx) -> Dict[str, float]:
    lt = math.log(ctx.W / ctx.H)
    ls = math.log(d.W / d.H)
    texts = [it for it in d.items if it.kind == "text" and it.role not in ("missing", "void")]
    chars = sum(len(str(e.get("content", ""))) for it in d.items for e in it.els if e.get("type") == "text")
    tarea = sum(it.src_box.area for it in texts) / max(1.0, d.W * d.H)
    return dict(
        lt=lt, dl=lt - ls, adl=abs(lt - ls),
        la=0.5 * math.log(ctx.W * ctx.H / max(1.0, d.W * d.H)),
        chars=math.log1p(chars), tarea=tarea, ntext=float(len(texts)),
        wide=max(0.0, lt - math.log(2.4)), tall=max(0.0, math.log(1 / 1.6) - lt),
    )


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
    f["core_lost"] = float(p.core_lost())
    f["lost"] = float(p.lost())
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
        f["min_fs_floor"] = math.log(min(fs for _, _, _, fs in fsr) / max(1.0, ctx.judge_floor))
    for k in ARCHS:
        if f["a:" + k]:
            for c in ("lt", "dl", "adl", "chars", "tarea", "ntext", "wide", "tall"):
                f[f"a:{k}*{c}"] = cx[c]
    for c in ("adl", "wide", "tall"):
        f[f"pos_iou*{c}"] = f.get("pos_iou", 0.0) * cx[c]
        f[f"keep_v*{c}"] = f["keep_v"] * cx[c]
        f[f"keep_h*{c}"] = f["keep_h"] * cx[c]
        f[f"font_med*{c}"] = f.get("font_med", 0.0) * cx[c]
        f[f"log_s*{c}"] = f["log_s"] * cx[c]
    f["hand_norm"] = p.score / 100.0
    return f


def all_features(plans, d, ctx) -> List[Dict[str, float]]:
    cx = context(d, ctx)
    return [plan_features(p, d, ctx, cx) for p in plans]


def predict(feats: Dict[str, float], m: dict) -> float:
    kind = m.get("kind", "linear-pairwise")
    if kind == "stumps":
        s = float(m.get("bias", 0.0))
        names = m["features"]
        for stump in m["stumps"]:
            j = names.index(stump["feat"])
            x = feats.get(stump["feat"], 0.0)
            s += stump["w"] * (1.0 if x >= stump["thr"] else -1.0)
        return s
    s = 0.0
    for name, w, mu, sd in zip(m["features"], m["weights"], m["mean"], m["std"]):
        s += w * (feats.get(name, 0.0) - mu) / sd
    return s


def blended(hand: List[float], learned: List[Optional[float]], lam: float, margin: float) -> List[float]:
    """Conservative blend: only the candidates the critic (hand score) keeps within ``margin`` of
    its best are re-ordered by the learned preference; they stay above every other candidate, so the
    ranker can choose among near-ties but never promote a plan the critic ranks clearly worse."""
    if not hand:
        return []
    best = max(hand)
    floor = best - margin
    allowed = [i for i, h in enumerate(hand) if h >= floor and learned[i] is not None]
    if len(allowed) < 2 or lam <= 0:
        return list(hand)
    top = max(learned[i] for i in allowed)
    out = list(hand)
    for i in allowed:
        out[i] = max(floor + 1e-6, hand[i] + lam * (learned[i] - top))
    return out


def apply(plans, d, ctx) -> None:
    """Re-order the critic's near-ties by what designers did (in place); see ``blended``."""
    if not plans or not enabled():
        return
    m = model()
    lam = float(os.environ.get("CHAMELEON_RANKER_BLEND") or m.get("blend", 0.0))
    if lam <= 0:
        return
    margin = float(os.environ.get("CHAMELEON_RANKER_MARGIN") or m.get("margin", 4.0))
    hand = [p.score for p in plans]
    best = max(hand)
    near = [i for i, h in enumerate(hand) if h >= best - margin]
    if len(near) < 2:
        return
    cx = context(d, ctx)
    learned: List[Optional[float]] = [None] * len(plans)
    for i in near:
        learned[i] = predict(plan_features(plans[i], d, ctx, cx), m)
    new = blended(hand, learned, lam, margin)
    for p, h, n in zip(plans, hand, new):
        if n != h:
            p.breakdown["ranker"] = n - h
            p.score = n
