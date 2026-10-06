"""Judge terms: how much worse a resize is than its source, rubric row by rubric row.

The engine's candidates are built and scored by geometry; the official scorer
is a VLM looking at the rendered PNG. tools/judge/ calibrated a proxy of that
judge on ~240 blind pairwise labels: render-measured terms (tools/judge/terms.py)
and the same terms measured here from boxes plus the pixel stats the engine
already has (pixels.measure). This module holds the shared term definitions and
the box measurement, so the engine can re-rank its top candidates with the
calibrated weights. Nothing here calls a model: the weights are numbers fitted
offline (tools/judge/fit.py).

Off by default (ENABLED); set CHAMELEON_JUDGE_RERANK=1 to try it.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional

TERMS = [
    "dropped_text", "dropped_art", "contrast_loss", "busy_gain", "abs_low_contrast", "abs_busy",
    "coll_text_art", "coll_text_text", "coll_art", "spill", "cut_text", "cut_art",
    "tiny512", "tiny1024", "text_shrink", "art_shrink", "hierarchy", "align_lost", "near_miss",
    "tight_margin", "imbalance", "underfill", "overfill", "aspect", "order_flip", "bare_canvas",
]


# ----------------------------------------------------------------------------- terms


def _pair(rep: dict, a: str, b: str) -> Optional[list]:
    v = rep["pairs"].get(f"{a}|{b}")
    if v is not None:
        return v
    v = rep["pairs"].get(f"{b}|{a}")
    if v is not None:
        return [v[0], v[2], v[1]]
    return None


def _need(rec: dict) -> float:
    fs = rec.get("fs", 12)
    return 3.0 if fs >= 24 or (rec.get("bold") and fs >= 18.66) else 4.5


def _stacked(b1, b2, axis: int) -> bool:
    """Boxes separated along `axis` (1 = one above the other, 0 = side by side)."""
    lo, hi = (1, 3) if axis == 1 else (0, 2)
    return b1[hi] <= b2[lo] or b2[hi] <= b1[lo]


def _relations(rep: dict, ids: Iterable[str]) -> Dict[tuple, float]:
    """Alignment relations in normalised coordinates: (a, b, kind) -> |offset|."""
    out = {}
    ids = [i for i in ids if "bbox" in rep["els"][i]]
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            a, b = ids[x], ids[y]
            ba, bb = rep["els"][a]["bbox"], rep["els"][b]["bbox"]
            if _stacked(ba, bb, 1):  # vertical stack: left / centre / right lines
                for kind, f in (("l", lambda q: q[0]), ("r", lambda q: q[2]), ("cx", lambda q: (q[0] + q[2]) / 2)):
                    out[(a, b, kind)] = abs(f(ba) - f(bb))
            if _stacked(ba, bb, 0):  # side by side: top / middle / bottom lines
                for kind, f in (("t", lambda q: q[1]), ("b", lambda q: q[3]), ("cy", lambda q: (q[1] + q[3]) / 2)):
                    out[(a, b, kind)] = abs(f(ba) - f(bb))
    return out


def _near_miss_rate(rep: dict, ids: List[str]) -> float:
    """Share of stacked pairs whose best alignment is almost, but not quite, exact
    (O'Donovan: a near miss reads as a mistake, a clear offset as a choice)."""
    W, H = rep["W"], rep["H"]
    best: Dict[tuple, float] = {}
    for (a, b, kind), d in _relations(rep, ids).items():
        span = W if kind in ("l", "r", "cx") else H
        px = d * span
        key = (a, b, kind in ("l", "r", "cx"))
        best[key] = min(best.get(key, 1e9), px)
    if not best:
        return 0.0
    L = max(W, H)
    tol = max(2.0, 0.004 * L)
    near = sum(1 for px in best.values() if tol < px <= 0.03 * L)
    return near / len(best)


def terms(src: dict, out: dict) -> Dict[str, float]:
    t = {k: 0.0 for k in TERMS}
    S, O = src["els"], out["els"]
    ids = [i for i in S if i in O]

    # what the viewer loses
    texts_s = [i for i in ids if S[i]["type"] == "text" and S[i]["live"]]
    max_fs = max([S[i]["fs"] for i in texts_s] or [1.0])
    wsum = sum(S[i]["fs"] / max_fs for i in texts_s)
    if wsum:
        t["dropped_text"] = sum(S[i]["fs"] / max_fs for i in texts_s if not O[i]["live"]) / wsum
    art_s = [i for i in ids if S[i]["type"] != "text" and S[i]["live"] and not S[i]["bg"]]
    aw = {i: math.sqrt(S[i]["area"]) * max(0.1, S[i].get("solid", 1.0)) for i in art_s}  # faint decor counts little
    asum = sum(aw.values())
    if asum:
        t["dropped_art"] = sum(aw[i] for i in art_s if not O[i]["live"]) / asum

    # contrast and texture under the copy, relative to the source
    both = [i for i in texts_s if O[i]["live"] and i in src["texts"] and i in out["texts"]]
    live_o = [i for i in ids if O[i]["type"] == "text" and O[i]["live"] and i in out["texts"]]
    if texts_s:
        cl = bg = 0.0
        for i in both:
            need = _need(O[i])
            c0, c1 = src["texts"][i]["wcag"], out["texts"][i]["wcag"]
            cl += max(0.0, min(need, c0) - c1) / need
            b0, b1 = src["texts"][i]["busy"], out["texts"][i]["busy"]
            bg += min(1.0, max(0.0, b1 - max(b0, 0.03)) / 0.08)
        t["contrast_loss"] = cl / len(texts_s)
        t["busy_gain"] = bg / len(texts_s)
    if live_o:
        t["abs_low_contrast"] = sum(max(0.0, _need(O[i]) - out["texts"][i]["wcag"]) / _need(O[i]) for i in live_o) / len(live_o)
        t["abs_busy"] = sum(min(1.0, out["texts"][i]["busy"] / 0.08) for i in live_o) / len(live_o)

    # collisions the source did not have; overlaps it had must not grow or spill
    for key, (fmin, fa, fb) in out["pairs"].items():
        a, b = key.split("|")
        if a not in S or b not in S or S[a]["bg"] or S[b]["bg"]:
            continue  # anything over what was a full-canvas background in the source is intended
        sp = (_pair(src, a, b) if (S[a]["live"] and S[b]["live"]) else None) or [0.0, 0.0, 0.0]
        ta, tb = S[a]["type"] == "text", S[b]["type"] == "text"
        on_surface = False
        for txt, cin0, cin1 in ((a, sp[1], fa), (b, sp[2], fb)):
            if S[txt]["type"] == "text" and cin0 >= 0.6:  # copy on its panel / photo / button
                t["spill"] += max(0.0, cin0 - cin1)
                on_surface = True
        if on_surface:
            continue
        grown = fmin - sp[0]
        if grown <= 0.05:  # what the source already had (resolution noise included)
            continue
        if ta and tb:
            t["coll_text_text"] += grown
        elif ta or tb:
            t["coll_text_art"] += grown
        else:
            t["coll_art"] += grown
    for k in ("coll_text_art", "coll_text_text", "coll_art", "spill"):
        t[k] = min(3.0, t[k])

    # cut off at the canvas edge (decoration that bled in the source may keep bleeding)
    for i in ids:
        if not O[i]["live"]:
            continue
        d = max(0.0, O[i]["off"] - S[i]["off"])
        if d > 0.002:
            t["cut_text" if O[i]["type"] == "text" else "cut_art"] += d
    t["cut_text"] = min(2.0, t["cut_text"])
    t["cut_art"] = min(2.0, t["cut_art"])

    # size: legibility at the judge's view, and prominence relative to the canvas
    def cap_pen(fs, W, H, view):
        cap = 0.7 * fs * min(1.0, view / max(W, H))
        return min(1.0, max(0.0, (7.0 - cap) / 7.0))

    texts_o = [i for i in ids if O[i]["type"] == "text" and O[i]["live"] and S[i]["live"]]
    if texts_o:
        for view, key in ((512, "tiny512"), (1024, "tiny1024")):
            t[key] = sum(max(0.0, cap_pen(O[i]["fs"], out["W"], out["H"], view) - cap_pen(S[i]["fs"], src["W"], src["H"], view))
                         for i in texts_o) / len(texts_o)
        ns, no = math.sqrt(src["W"] * src["H"]), math.sqrt(out["W"] * out["H"])
        acc = wt = 0.0
        for i in texts_o:
            r = (O[i]["fs"] / no) / max(1e-9, S[i]["fs"] / ns)
            w = S[i]["fs"] / max_fs
            acc += w * min(1.0, max(0.0, -math.log(max(1e-6, r)) / math.log(2)))
            wt += w
        t["text_shrink"] = acc / max(1e-9, wt)
        inv = pairs = 0
        for x in range(len(texts_o)):
            for y in range(x + 1, len(texts_o)):
                a, b = texts_o[x], texts_o[y]
                d0 = S[a]["fs"] - S[b]["fs"]
                if abs(d0) < 1.0:
                    continue
                pairs += 1
                inv += d0 * (O[a]["fs"] - O[b]["fs"]) < 0
        t["hierarchy"] = inv / max(1, pairs)
    arts = [i for i in art_s if O[i]["live"]]
    if arts:
        acc = wt = 0.0
        for i in arts:
            r = math.sqrt(max(1e-9, O[i]["area"]) / max(1e-9, S[i]["area"]))
            w = aw[i]
            acc += w * min(1.0, max(0.0, -math.log(r) / math.log(2)))
            wt += w
        t["art_shrink"] = acc / max(1e-9, wt)

    # alignment: source relations that survive where the arrangement survived
    keep = [i for i in ids if S[i]["live"] and O[i]["live"] and not S[i]["bg"] and not O[i]["bg"]]
    rs = _relations(src, keep)
    ro = _relations(out, keep)
    Ls, Lo = max(src["W"], src["H"]), max(out["W"], out["H"])
    lost = n = 0.0
    for key, d0 in rs.items():
        span0 = src["W"] if key[2] in ("l", "r", "cx") else src["H"]
        if d0 * span0 > max(1.5, 0.004 * Ls) or key not in ro:
            continue
        span1 = out["W"] if key[2] in ("l", "r", "cx") else out["H"]
        px = ro[key] * span1
        tol = max(1.5, 0.002 * Lo)
        n += 1
        lost += min(1.0, max(0.0, px - tol) / (0.02 * Lo))
    t["align_lost"] = lost / n if n else 0.0
    t["near_miss"] = max(0.0, _near_miss_rate(out, keep) - _near_miss_rate(src, keep))

    # the source's reading order: what sat left of / above what still does (where the
    # arrangement survived); a mirrored layout reads as a different design
    flips = n = 0
    for x in range(len(keep)):
        for y in range(x + 1, len(keep)):
            a, b = keep[x], keep[y]
            if "bbox" not in S[a] or "bbox" not in S[b] or "bbox" not in O[a] or "bbox" not in O[b]:
                continue
            sa, sb, oa, ob = S[a]["bbox"], S[b]["bbox"], O[a]["bbox"], O[b]["bbox"]
            for axis, lo, hi in ((0, 0, 2), (1, 1, 3)):
                if _stacked(sa, sb, axis) and _stacked(oa, ob, axis):
                    n += 1
                    flips += (sa[hi] <= sb[lo]) != (oa[hi] <= ob[lo])
    t["order_flip"] = flips / n if n else 0.0
    t["bare_canvas"] = max(0.0, out.get("bare", 0.0) - src.get("bare", 0.0))

    # whitespace and balance, relative to the source
    t["tight_margin"] = sum(1 for m0, m1 in zip(src["margins"], out["margins"]) if m1 < 0.02 and m0 >= 0.02) / 4.0
    imb = lambda r: 2 * math.hypot(r["centroid"][0] - 0.5, r["centroid"][1] - 0.5)  # noqa: E731
    t["imbalance"] = max(0.0, imb(out) - imb(src))
    c0, c1 = src["coverage"] + 0.02, out["coverage"] + 0.02
    t["underfill"] = min(2.0, max(0.0, math.log(c0 / c1)))
    t["overfill"] = min(2.0, max(0.0, math.log(c1 / c0)))

    # pictures keep their proportions (a full-bleed cover background may crop)
    imgs = [i for i in ids if O[i]["type"] == "image" and O[i].get("fit") != "cover" and O[i]["live"]]
    if imgs:
        t["aspect"] = sum(abs(O[i]["box_aspect"] / max(1e-6, S[i]["box_aspect"]) - 1) > 0.05 for i in imgs) / len(imgs)
    return {k: round(v, 4) for k, v in t.items()}


def penalty(t: Dict[str, float], weights: Dict[str, float]) -> float:
    return float(sum(weights.get(k, 0.0) * v for k, v in t.items()))


# ----------------------------------------------------------------------------
# box-level measurement (what the engine can afford for every top candidate)

ENABLED = os.environ.get("CHAMELEON_JUDGE_RERANK", "") not in ("", "0")

# Fitted offline by tools/judge/fit.py on the box-level terms: Bradley-Terry on 239
# blind pairwise labels (478 asks), L2 towards the scaled hand weights, lambda
# 0.1 by cross-validation over held-out designs (tools/judge/data/weights_box.json).
WEIGHTS: Dict[str, float] = {
    "dropped_text": 1.155, "dropped_art": 0.846, "contrast_loss": 1.412, "busy_gain": 1.110,
    "abs_low_contrast": 0.051, "coll_text_art": 1.619, "coll_text_text": 1.666, "coll_art": 0.484,
    "spill": 0.806, "cut_text": 1.156, "cut_art": 0.558, "tiny512": 0.567, "tiny1024": 1.109,
    "text_shrink": 0.602, "art_shrink": 0.301, "hierarchy": 0.548, "align_lost": 0.314,
    "near_miss": 0.276, "tight_margin": 0.288, "imbalance": 0.291, "underfill": 0.327,
    "overfill": 0.271, "aspect": 1.673, "order_flip": 0.258, "bare_canvas": 0.571,
}

# Only plans whose engine score is within BAND of the best are re-ordered by the
# judge. On labelled pairs of the engine's own top-k, the engine score agreed with
# the judge 73% of the time when it preferred one plan by >= 8 points, and was a
# coin flip below that (49%, n=41), where these weights agreed 65%.
BAND = 8.0

_GRID = 64  # raster for coverage, centroid, margins and bare canvas


def _ink_box(e: dict):
    """Where an element paints: a text's drawn glyph box (renderer-exact), an
    image's non-transparent pixels inside its fit, a shape's box; rotation as
    a bounding box."""
    from . import assets
    from .geom import Box, el_box, rotated_aabb

    t = e.get("type")
    if t == "text":
        from .textfit import drawn_box

        if not str(e.get("content", "")).strip():
            return Box(float(e.get("x", 0) or 0), float(e.get("y", 0) or 0), 0.0, 0.0)
        return drawn_box(e)
    b = el_box(e)
    if t == "image" and b.w > 0 and b.h > 0:
        fx0, fy0, fx1, fy1 = assets.ink_fraction(e.get("asset_id"), b.w, b.h, e.get("fit", "contain") or "contain")
        b = Box(b.x + fx0 * b.w, b.y + fy0 * b.h, (fx1 - fx0) * b.w, (fy1 - fy0) * b.h)
    rot = float(e.get("rotation", 0) or 0)
    return rotated_aabb(b, rot) if rot else b


def _solid(e: dict) -> float:
    from .color import parse_rgba

    op = e.get("opacity")
    op = 1.0 if op is None else float(op)
    if e.get("type") == "shape":
        op *= parse_rgba(e.get("fill_color"), default=(0, 0, 0, 1.0))[3]
    return 1.0 if op >= 0.25 else 0.0


def box_report(elements: List[dict], W: float, H: float, stats: Dict[str, tuple]) -> dict:
    """The report terms() reads (same schema as tools/judge/terms.measure), from
    element geometry and the pixel stats of pixels.measure: {id: (contrast p10, busy)}.
    Occlusion is not resolved: an element counts as shown unless its opacity is 0
    (which is how the engine drops an element)."""
    from .geom import Box

    canvas = Box(0.0, 0.0, float(W), float(H))
    rep: dict = {"W": float(W), "H": float(H), "s": 1.0, "els": {}, "texts": {}, "pairs": {}}
    boxes = {}
    for e in elements:
        i = e["id"]
        t = e.get("type")
        op = e.get("opacity")
        op = 1.0 if op is None else float(op)
        b = _ink_box(e)
        area = b.w * b.h
        clip = b.inter_box(canvas) if area > 0 else None
        carea = clip.area if clip else 0.0
        live = op > 0 and carea > 0
        rec = {"type": t, "area": carea / (W * H), "off": round(1.0 - carea / area, 4) if area > 0 else 0.0,
               "vis": 1.0 if live else 0.0, "solid": _solid(e), "bg": bool(carea >= 0.85 * W * H), "live": bool(live)}
        if clip:
            rec["bbox"] = [clip.x / W, clip.y / H, clip.r / W, clip.b / H]
            boxes[i] = clip
        if t == "text":
            rec["fs"] = float(e.get("font_size", 12) or 12)
            rec["bold"] = str(e.get("font_weight", "regular")) == "bold"
        if t == "image":
            rec["box_aspect"] = float(e.get("width", 0) or 0) / max(1e-6, float(e.get("height", 0) or 0))
            rec["fit"] = e.get("fit", "contain")
            rec["opacity"] = op
        rep["els"][i] = rec
        if t == "text" and live and i in stats:
            c, busy = stats[i]
            rep["texts"][i] = {"wcag": round(float(c), 3), "busy": round(float(busy), 4)}

    live = [e["id"] for e in elements if rep["els"][e["id"]]["live"] and not rep["els"][e["id"]]["bg"]]
    for x in range(len(live)):
        for y in range(x + 1, len(live)):
            a, b = boxes[live[x]], boxes[live[y]]
            inter = a.inter(b)
            if inter <= 0:
                continue
            rep["pairs"][f"{live[x]}|{live[y]}"] = [round(inter / max(1e-9, min(a.area, b.area)), 4),
                                                   round(inter / max(1e-9, a.area), 4), round(inter / max(1e-9, b.area), 4)]

    def raster(ids):
        g = [[False] * _GRID for _ in range(_GRID)]
        for i in ids:
            bx = boxes[i]
            for gy in range(int(bx.y / H * _GRID), min(_GRID, int(math.ceil(bx.b / H * _GRID)))):
                row = g[gy]
                for gx in range(int(bx.x / W * _GRID), min(_GRID, int(math.ceil(bx.r / W * _GRID)))):
                    row[gx] = True
        return g

    painted = raster([e["id"] for e in elements if rep["els"][e["id"]]["live"] and rep["els"][e["id"]]["solid"]])
    rep["bare"] = round(1.0 - sum(map(sum, painted)) / float(_GRID * _GRID), 4)
    occ = raster(live)
    cells = [(gx, gy) for gy in range(_GRID) for gx in range(_GRID) if occ[gy][gx]]
    if cells:
        S = min(W, H)
        rep["centroid"] = [(sum(c[0] for c in cells) / len(cells) + 0.5) / _GRID,
                           (sum(c[1] for c in cells) / len(cells) + 0.5) / _GRID]
        rep["coverage"] = len(cells) / float(_GRID * _GRID)
        x0 = min(boxes[i].x for i in live)
        y0 = min(boxes[i].y for i in live)
        x1 = max(boxes[i].r for i in live)
        y1 = max(boxes[i].b for i in live)
        rep["margins"] = [x0 / S, y0 / S, (W - x1) / S, (H - y1) / S]
    else:
        rep["centroid"], rep["coverage"], rep["margins"] = [0.5, 0.5], 0.0, [0.5] * 4
    return rep


def judge_penalty(src_rep: dict, elements: List[dict], W: float, H: float, stats: Dict[str, tuple]) -> float:
    return penalty(terms(src_rep, box_report(elements, W, H, stats)), WEIGHTS)
