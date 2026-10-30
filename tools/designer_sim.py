"""Dev tool: how close a layout is to what the designer did at that size.

Crello families give, for one source design, the designer's own version at
another canvas size. Element ids differ between the two (each Crello design
numbers its elements e0..en), so elements are matched first:

  * text by its content (the families are grouped by identical copy),
    duplicates in reading order;
  * everything else by an order-preserving alignment of the non-text
    sequences, costed by aspect-ratio difference.

Then, over matched elements that are not full-canvas backgrounds:

  iou    per-element generalised IoU of the boxes normalised to the canvas
         (0..1 each axis), mapped to 0..1 so that near misses still rank
         above far ones; weighted towards large elements; a hidden element
         scores 0. Text is compared by where its glyphs land (drawn box),
         not by its often loose element box.
  rel    agreement of pairwise relative positions (left-of / right-of /
         overlapping horizontally, and the same vertically)
  font   font-size agreement, 1 - |log(ours / designer)| / log 2, floored at 0

  sim = 0.5 iou + 0.3 rel + 0.2 font

Dev-only; the engine never imports this.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

W_IOU, W_REL, W_FONT = 0.5, 0.3, 0.2


def _box(e: dict, W: float, H: float) -> Tuple[float, float, float, float]:
    x, y = float(e.get("x", 0) or 0) / W, float(e.get("y", 0) or 0) / H
    w, h = float(e.get("width", 0) or 0) / W, float(e.get("height", 0) or 0) / H
    return x, y, max(0.0, w), max(0.0, h)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 1e-12 else 0.0


def _giou01(a, b) -> float:
    """Generalised IoU (Rezatofighi et al. 2019) mapped from [-1, 1] to [0, 1]."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    cw = max(a[0] + a[2], b[0] + b[2]) - min(a[0], b[0])
    ch = max(a[1] + a[3], b[1] + b[3]) - min(a[1], b[1])
    c = cw * ch
    if u <= 1e-12 or c <= 1e-12:
        return 0.0
    return 0.5 * (inter / u - (c - u) / c + 1.0)


def _norm_text(e: dict) -> str:
    return " ".join(str(e.get("content", "")).strip().lower().split())


def _ar(e: dict) -> float:
    return max(1e-3, float(e.get("width", 1) or 1)) / max(1e-3, float(e.get("height", 1) or 1))


def match(src: List[dict], tgt: List[dict]) -> Dict[str, dict]:
    """Source element id -> the designer's element at the target size."""
    out: Dict[str, dict] = {}
    by_text: Dict[str, List[dict]] = {}
    for e in tgt:
        if e.get("type") == "text":
            by_text.setdefault(_norm_text(e), []).append(e)
    for e in src:
        if e.get("type") == "text":
            lst = by_text.get(_norm_text(e))
            if lst:
                out[e["id"]] = lst.pop(0)
    a = [e for e in src if e.get("type") != "text"]
    b = [e for e in tgt if e.get("type") != "text"]
    if not a or not b:
        return out
    n, m = len(a), len(b)
    gap = 0.7
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i * gap
    for j in range(1, m + 1):
        cost[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = min(1.2, abs(math.log(_ar(a[i - 1]) / _ar(b[j - 1]))) / math.log(2.0))
            cost[i][j] = min(cost[i - 1][j - 1] + sub, cost[i - 1][j] + gap, cost[i][j - 1] + gap)
    i, j = n, m
    while i > 0 and j > 0:
        sub = min(1.2, abs(math.log(_ar(a[i - 1]) / _ar(b[j - 1]))) / math.log(2.0))
        if abs(cost[i][j] - (cost[i - 1][j - 1] + sub)) < 1e-9:
            if sub < 1.0:
                out[a[i - 1]["id"]] = b[j - 1]
            i, j = i - 1, j - 1
        elif abs(cost[i][j] - (cost[i - 1][j] + gap)) < 1e-9:
            i -= 1
        else:
            j -= 1
    return out


def _is_bg(b) -> bool:
    x, y, w, h = b
    return w * h >= 0.85 and x <= 0.05 and y <= 0.05


def _rel(a, b, tol: float = 0.01) -> Tuple[int, int]:
    """(horizontal, vertical) relation of box a to box b: -1 before, 1 after, 0 overlapping."""
    if a[0] + a[2] <= b[0] + tol:
        h = -1
    elif b[0] + b[2] <= a[0] + tol:
        h = 1
    else:
        h = 0
    if a[1] + a[3] <= b[1] + tol:
        v = -1
    elif b[1] + b[3] <= a[1] + tol:
        v = 1
    else:
        v = 0
    return h, v


def _visible(e: dict) -> bool:
    return float(e.get("opacity", 1) if e.get("opacity") is not None else 1) > 0.0


def similarity(ours: List[dict], designer: dict, source: dict, mapping: Optional[Dict[str, dict]] = None,
               text_box: Optional[Callable] = None) -> dict:
    """Similarity of `ours` (elements on the target canvas, source ids) to the
    designer's version `designer` of `source`. `text_box(el)` -> Box gives
    where a text's glyphs land (engine.smart.textfit.drawn_box); without it
    the element box is used."""
    W, H = float(designer["canvas"]["width"]), float(designer["canvas"]["height"])

    def box(e):
        if text_box is not None and e.get("type") == "text":
            try:
                b = text_box(e)
                return (b.x / W, b.y / H, max(0.0, b.w) / W, max(0.0, b.h) / H)
            except Exception:
                pass
        return _box(e, W, H)
    if mapping is None:
        mapping = match(source["elements"], designer["elements"])
    by_id = {e["id"]: e for e in ours}
    pairs = []  # (src id, our el, designer el)
    for sid, de in mapping.items():
        oe = by_id.get(sid)
        if oe is None:
            continue
        db = _box(de, W, H)
        if _is_bg(db):
            continue
        pairs.append((sid, oe, de))
    if not pairs:
        return dict(iou=0.0, rel=0.0, font=0.0, sim=0.0, n=0)
    num = den = 0.0
    boxes = []
    for sid, oe, de in pairs:
        db, ob = box(de), box(oe)
        w = min(1.0, max(0.08, math.sqrt(db[2] * db[3])))
        if oe.get("type") == "text":
            w *= 1.5
        vis = _visible(oe)
        num += w * (_giou01(ob, db) if vis else 0.0)
        den += w
        boxes.append((ob, db, vis))
    iou = num / den
    agree = tot = 0.0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            oi, di, vi = boxes[i]
            oj, dj, vj = boxes[j]
            tot += 1.0
            if not (vi and vj):
                continue
            ro, rd = _rel(oi, oj), _rel(di, dj)
            agree += 0.5 * (ro[0] == rd[0]) + 0.5 * (ro[1] == rd[1])
    rel = agree / tot if tot else 1.0
    fs = []
    for sid, oe, de in pairs:
        if oe.get("type") != "text" or de.get("type") != "text":
            continue
        if not _visible(oe):
            fs.append(0.0)
            continue
        a = max(1.0, float(oe.get("font_size", 12) or 12))
        b = max(1.0, float(de.get("font_size", 12) or 12))
        fs.append(max(0.0, 1.0 - abs(math.log(a / b)) / math.log(2.0)))
    font = sum(fs) / len(fs) if fs else 1.0
    return dict(iou=iou, rel=rel, font=font, sim=W_IOU * iou + W_REL * rel + W_FONT * font, n=len(pairs))
