"""Stages, one owner per decision:

1. units      which elements move together (group_id, or text sitting inside a small box)  [structure]
2. transfer   where each element goes and how big: nearest designer moves (knn.py)          [learned]
3. bleed      which layers cover the canvas: source bleeds, or the designers' bleed vote     [learned]
4. type       font size: designers' size change, then the largest size that fits the box   [learned + measured]
5. repair     inside the canvas, no new collisions, legibility floor                        [constraints]
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from . import knn, textfit
from .features import element_features, is_bleed, num


def _box(e):
    return [num(e.get("x")), num(e.get("y")), max(1e-3, num(e.get("width"), 1.0)), max(1e-3, num(e.get("height"), 1.0))]


def _inter(a, b):
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, iw) * max(0.0, ih)


def _ioa(a, b):
    """overlap as a share of the smaller box"""
    return _inter(a, b) / max(1e-6, min(a[2] * a[3], b[2] * b[3]))


def find_units(els: List[Dict[str, Any]], bleed: List[bool], W: float, H: float) -> List[List[int]]:
    n = len(els)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    by_group: Dict[str, int] = {}
    for i, e in enumerate(els):
        g = e.get("group_id")
        if g and not bleed[i]:
            if g in by_group:
                union(i, by_group[g])
            else:
                by_group[g] = i
    boxes = [_box(e) for e in els]
    for i, e in enumerate(els):
        if e.get("type") != "text" or bleed[i]:
            continue
        best, ba = None, None
        for j, f in enumerate(els):
            if j == i or bleed[j] or f.get("type") == "text":
                continue
            bj = boxes[j]
            if bj[2] * bj[3] > 0.35 * W * H:
                continue
            if _inter(boxes[i], bj) >= 0.8 * boxes[i][2] * boxes[i][3]:
                a = bj[2] * bj[3]
                if ba is None or a < ba:
                    best, ba = j, a
        if best is not None:
            union(i, best)
    units: Dict[int, List[int]] = {}
    for i in range(n):
        units.setdefault(find(i), []).append(i)
    return list(units.values())


def layout(source: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    W = max(1.0, num(source["canvas"].get("width"), 1.0))
    H = max(1.0, num(source["canvas"].get("height"), 1.0))
    TW = max(1.0, num(target.get("width"), 1.0))
    TH = max(1.0, num(target.get("height"), 1.0))
    els = [dict(e) for e in source["elements"]]
    n = len(els)
    out = [dict(e) for e in els]
    if n == 0:
        return dict(canvas=dict(width=int(TW), height=int(TH)), elements=out)
    s_geo = math.sqrt((TW * TH) / (W * H))
    X = element_features(dict(canvas=dict(width=W, height=H), elements=els), dict(width=TW, height=TH))
    P = knn.predict_design(X)
    P[:, 0] += X[:, 2]
    P[:, 1] += X[:, 3]

    # 3. bleed: layers that cover the source canvas stay covering it
    bleed = [is_bleed(e, W, H) for e in els]
    src = [_box(e) for e in els]
    for i, e in enumerate(els):
        if bleed[i]:
            out[i].update(x=0.0, y=0.0, width=float(TW), height=float(TH))
            if e.get("type") == "image":
                out[i]["fit"] = "cover"

    # 1. units, 2. transfer
    units = find_units(els, bleed, W, H)
    boxes: Dict[int, List[float]] = {}
    unit_scale: Dict[int, float] = {}
    for u in units:
        if all(bleed[i] for i in u):
            continue
        u = [i for i in u if not bleed[i]]
        if len(u) == 1 and els[u[0]].get("type") == "text":
            i = u[0]
            w0, h0 = src[i][2], src[i][3]
            side = math.sqrt(w0 * h0) * s_geo * math.exp(float(P[i, 2]))
            ar = (w0 / h0) * math.exp(float(P[i, 3]))
            w, h = side * math.sqrt(ar), side / math.sqrt(ar)
            cx, cy = P[i, 0] * TW, P[i, 1] * TH
            boxes[i] = [cx - w / 2, cy - h / 2, w, h]
            unit_scale[i] = s_geo * math.exp(float(P[i, 2]))
            continue
        x0 = min(src[i][0] for i in u)
        y0 = min(src[i][1] for i in u)
        x1 = max(src[i][0] + src[i][2] for i in u)
        y1 = max(src[i][1] + src[i][3] for i in u)
        wt = np.array([src[i][2] * src[i][3] for i in u])
        wt = wt / wt.sum()
        ls = float(sum(wt[k] * P[i, 2] for k, i in enumerate(u)))
        sc = s_geo * math.exp(ls)
        # centre: where the members' moves put the unit's centre
        ucx = float(sum(wt[k] * (P[i, 0] * TW - ((src[i][0] + src[i][2] / 2) - (x0 + x1) / 2) * sc) for k, i in enumerate(u)))
        ucy = float(sum(wt[k] * (P[i, 1] * TH - ((src[i][1] + src[i][3] / 2) - (y0 + y1) / 2) * sc) for k, i in enumerate(u)))
        for i in u:
            boxes[i] = [ucx + (src[i][0] - (x0 + x1) / 2) * sc, ucy + (src[i][1] - (y0 + y1) / 2) * sc,
                        src[i][2] * sc, src[i][3] * sc]
            unit_scale[i] = sc
    fg_units = [[i for i in u if not bleed[i]] for u in units]
    fg_units = [u for u in fg_units if u]

    # 4. type
    floor = max(8.0, 0.028 * min(TW, TH))
    fsz: Dict[int, int] = {}
    for u in fg_units:
        for i in u:
            if els[i].get("type") != "text":
                continue
            fs0 = num(els[i].get("font_size"), 16.0)
            want = fs0 * (unit_scale[i] if len(u) > 1 else s_geo * math.exp(float(P[i, 4])))
            want = max(want, min(fs0, floor))
            fsz[i] = int(max(1, want))

    _resolve(els, out, boxes, fsz, fg_units, src, bleed, TW, TH, floor)
    return dict(canvas=dict(width=int(TW), height=int(TH)), elements=out)


def _fit_text(e, b, fs_want, TW, TH, floor):
    """Settle a text's font and box: keep the box centre, keep the width unless the words need
    more, grow the height to the block. Returns (fs, box)."""
    x, y, w, h = b
    cx, cy = x + w / 2, y + h / 2
    w = min(w, TW)
    lw = textfit.longest_word(e, fs_want)
    if lw > w:
        w = min(TW, lw * 1.02 + 1)
    fs = textfit.max_font(e, w, max(h, TH), fs_want, 1)
    _, bh, _ = textfit.layout(e, fs, w)
    if bh > TH * 0.6 and fs > floor:  # a tall block: widen it before shrinking type
        w2 = min(TW * 0.92, max(w, w * 1.6))
        fs = textfit.max_font(e, w2, max(h, TH), fs_want, 1)
        w = w2
        _, bh, _ = textfit.layout(e, fs, w)
    if bh > TH:
        fs = textfit.max_font(e, w, TH, fs, 1)
        _, bh, _ = textfit.layout(e, fs, w)
    mw, _, _ = textfit.layout(e, fs, w)
    if mw > w:  # a single word wider than the canvas: shrink until it fits
        while fs > 1 and textfit.layout(e, fs, w)[0] > w:
            fs -= 1
        _, bh, _ = textfit.layout(e, fs, w)
    h = min(TH, bh + 1)
    return fs, [cx - w / 2, cy - h / 2, w, h]


def _unit_bbox(u, boxes):
    x0 = min(boxes[i][0] for i in u)
    y0 = min(boxes[i][1] for i in u)
    x1 = max(boxes[i][0] + boxes[i][2] for i in u)
    y1 = max(boxes[i][1] + boxes[i][3] for i in u)
    return [x0, y0, x1 - x0, y1 - y0]


def _shift(u, boxes, dx, dy):
    for i in u:
        boxes[i][0] += dx
        boxes[i][1] += dy


def _scale_unit(u, boxes, s, fsz):
    b = _unit_bbox(u, boxes)
    cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
    for i in u:
        x, y, w, h = boxes[i]
        boxes[i] = [cx + (x - cx) * s, cy + (y - cy) * s, w * s, h * s]
        if i in fsz:
            fsz[i] = max(1, int(fsz[i] * s))


def _resolve(els, out, boxes, fsz, units, src, bleed, TW, TH, floor):
    margin = 0.03 * min(TW, TH)
    # text: settle font and box (singletons may reflow; texts in units keep the unit's box width)
    for u in units:
        for i in u:
            if i in fsz:
                fs, b = _fit_text(els[i], boxes[i], fsz[i], TW, TH, floor)
                if len(u) > 1:  # keep the label centred where the unit put it
                    b[0] = boxes[i][0] + boxes[i][2] / 2 - b[2] / 2
                    b[1] = boxes[i][1] + boxes[i][3] / 2 - b[3] / 2
                fsz[i], boxes[i] = fs, b
    # units that came out bigger than the canvas shrink uniformly
    for u in units:
        b = _unit_bbox(u, boxes)
        s = min(1.0, (TW - 2 * margin) / max(1e-3, b[2]), (TH - 2 * margin) / max(1e-3, b[3]))
        if s < 1.0:
            _scale_unit(u, boxes, s, fsz)
    src_bb = {k: _unit_bbox(u, {i: src[i] for i in u}) for k, u in enumerate(units)}

    def clamp(k):
        u = units[k]
        b = _unit_bbox(u, boxes)
        m = margin
        dx = (m - b[0]) if b[0] < m else ((TW - m) - (b[0] + b[2])) if b[0] + b[2] > TW - m else 0.0
        dy = (m - b[1]) if b[1] < m else ((TH - m) - (b[1] + b[3])) if b[1] + b[3] > TH - m else 0.0
        if b[2] > TW - 2 * m:
            dx = TW / 2 - (b[0] + b[2] / 2)
        if b[3] > TH - 2 * m:
            dy = TH / 2 - (b[1] + b[3] / 2)
        _shift(u, boxes, dx, dy)

    for k in range(len(units)):
        clamp(k)
    # collisions that the source did not have
    allowed = set()
    for a in range(len(units)):
        for b in range(a + 1, len(units)):
            if _ioa(src_bb[a], src_bb[b]) > 0.05:
                allowed.add((a, b))
    area = [src_bb[k][2] * src_bb[k][3] for k in range(len(units))]
    for it in range(60):
        worst, wv = None, 0.0
        bbs = [_unit_bbox(u, boxes) for u in units]
        for a in range(len(units)):
            for b in range(a + 1, len(units)):
                if (a, b) in allowed:
                    continue
                v = _inter(bbs[a], bbs[b])
                if v > wv:
                    worst, wv = (a, b), v
        if worst is None:
            break
        a, b = worst
        mv = b if area[b] <= area[a] else a
        st = a if mv == b else b
        A, B = bbs[st], bbs[mv]
        # minimal translation of the mover out of the static box, inside the canvas
        opts = []
        for dx, dy in ((A[0] - (B[0] + B[2]), 0), (A[0] + A[2] - B[0], 0), (0, A[1] - (B[1] + B[3])), (0, A[1] + A[3] - B[1])):
            nx, ny = B[0] + dx, B[1] + dy
            inside = nx >= -0.5 and ny >= -0.5 and nx + B[2] <= TW + 0.5 and ny + B[3] <= TH + 0.5
            opts.append((0 if inside else 1, abs(dx) + abs(dy), dx, dy))
        opts.sort()
        if opts[0][0] == 0:
            _shift(units[mv], boxes, opts[0][2] * 1.02, opts[0][3] * 1.02)
            clamp(mv)
        else:
            # no room: shrink both a little about their centres
            _scale_unit(units[mv], boxes, 0.9, fsz)
            if it % 2:
                _scale_unit(units[st], boxes, 0.95, fsz)
            clamp(mv)
            clamp(st)
    # write back, hard-clamped to the canvas
    for u in units:
        for i in u:
            x, y, w, h = boxes[i]
            w, h = min(w, TW), min(h, TH)
            x = min(max(0.0, x), TW - w)
            y = min(max(0.0, y), TH - h)
            out[i].update(x=float(x), y=float(y), width=float(w), height=float(h))
            if i in fsz:
                e = els[i]
                fs = fsz[i]
                # after any shrink, make sure the text still fits its box
                while fs > 1 and not textfit.fits(e, fs, w, h):
                    fs -= 1
                out[i]["font_size"] = int(fs)
            if els[i].get("corner_radius") and src[i][2] > 0:
                out[i]["corner_radius"] = num(els[i]["corner_radius"]) * (w / src[i][2])
