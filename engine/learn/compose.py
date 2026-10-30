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


def _backdrop(e, i, els, W, H):
    """A picture covering most of the canvas beneath all the copy is the background: it covers
    the target canvas (cover-cropped) rather than leaving a strip."""
    if e.get("type") == "text" or (e.get("type") == "shape" and e.get("shape_type", "rectangle") != "rectangle"):
        return False
    x, y, w, h = _box(e)
    ix = max(0.0, min(x + w, W) - max(x, 0.0))
    iy = max(0.0, min(y + h, H) - max(y, 0.0))
    if ix * iy < 0.8 * W * H:
        return False
    z = num(e.get("z_index"), i)
    tz = [num(t.get("z_index"), j) for j, t in enumerate(els) if t.get("type") == "text"]
    return not tz or z <= min(tz)


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


REFLOW_DAR = 0.9


def floor_px(TW, TH):
    """Legibility floor: 7 px of cap height when the canvas is viewed at 1024 px on its long side
    (cap height ~0.70 em), as L0's fundamentals.check defines small print."""
    return float(max(6, math.ceil(7.0 * max(TW, TH) / 1024.0 / 0.70)))
MODE = {"transfer": "median", "beta": 0.5, "repair": True}
_policy = None


def _median_policy(X):
    """The designers' median move per signed aspect-change bin (tools/learn_policy.py)."""
    global _policy
    if _policy is None:
        import json
        from pathlib import Path
        _policy = json.loads(Path(__file__).with_name("policy.json").read_text())
    c = _policy["centers"]
    d = float(X[0, 14])
    f = lambda key: float(np.interp(d, c, _policy[key]))  # noqa: E731
    P = np.zeros((len(X), 6), np.float32)
    P[:, 0], P[:, 1], P[:, 5] = X[:, 2], X[:, 3], X[:, 1]
    t = X[:, 0] > 0.5
    P[:, 2] = np.where(t, f("text_scale"), f("other_scale"))
    P[:, 3] = np.where(t, f("text_ar"), 0.0)
    P[:, 4] = np.where(t, f("fs"), 0.0)
    return P


def layout(source: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    W = max(1.0, num(source["canvas"].get("width"), 1.0))
    H = max(1.0, num(source["canvas"].get("height"), 1.0))
    TW = max(1.0, num(target.get("width"), 1.0))
    TH = max(1.0, num(target.get("height"), 1.0))
    els = [dict(e) for e in source["elements"]]
    n = len(els)
    out = [dict(e) for e in els]
    _ROT.clear()
    for i, e in enumerate(els):
        if num(e.get("rotation"), 0.0) % 360.0:
            _ROT[i] = num(e.get("rotation"), 0.0)
    for e in els:
        if e.get("type") == "text" and num(e.get("letter_spacing"), 0.0):
            e["_lsem"] = num(e.get("letter_spacing"), 0.0) / max(1.0, num(e.get("font_size"), 16.0))
    if n == 0:
        return dict(canvas=dict(width=int(TW), height=int(TH)), elements=out)
    s_geo = math.sqrt((TW * TH) / (W * H))
    X = element_features(dict(canvas=dict(width=W, height=H), elements=els), dict(width=TW, height=TH))
    if MODE["transfer"] == "median":
        P = _median_policy(X)
    elif MODE["transfer"] == "retrieve":
        from . import retrieve
        P, _ = retrieve.predict(X, MODE["beta"])
    else:
        P = knn.predict_design(X)
        P[:, 0] += X[:, 2]
        P[:, 1] += X[:, 3]
    if MODE["transfer"] == "prop":
        P[:, :5] = 0.0
        P[:, 0], P[:, 1], P[:, 5] = X[:, 2], X[:, 3], X[:, 1]

    # 3. bleed: layers that cover the source canvas stay covering it
    bleed = [is_bleed(e, W, H) or _backdrop(e, i, els, W, H) for i, e in enumerate(els)]
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
    floor = floor_px(TW, TH)
    fsz: Dict[int, int] = {}
    for u in fg_units:
        for i in u:
            if els[i].get("type") != "text":
                continue
            fs0 = num(els[i].get("font_size"), 16.0)
            want = fs0 * (unit_scale[i] if len(u) > 1 else s_geo * math.exp(float(P[i, 4])))
            want = max(want, floor)
            fsz[i] = int(max(1, want))

    if not MODE["repair"]:
        for i, b in boxes.items():
            out[i].update(x=b[0], y=b[1], width=b[2], height=b[3])
        return dict(canvas=dict(width=int(TW), height=int(TH)), elements=out)
    _resolve(els, out, boxes, fsz, fg_units, src, bleed, TW, TH, floor, abs(math.log((TW / TH) / (W / H))))
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


_ROT: Dict[int, float] = {}  # element index -> rotation (degrees) of the design being laid out


def _ext(i, b):
    """box of element i as drawn: rotated about its centre"""
    a = _ROT.get(i, 0.0)
    if not a:
        return b[0], b[1], b[0] + b[2], b[1] + b[3]
    r = math.radians(a)
    bw = abs(b[2] * math.cos(r)) + abs(b[3] * math.sin(r))
    bh = abs(b[2] * math.sin(r)) + abs(b[3] * math.cos(r))
    cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def _unit_bbox(u, boxes):
    ex = [_ext(i, boxes[i]) for i in u]
    x0 = min(e[0] for e in ex)
    y0 = min(e[1] for e in ex)
    x1 = max(e[2] for e in ex)
    y1 = max(e[3] for e in ex)
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


def _alpha(e):
    a = num(e.get("opacity"), 1.0)
    c = str(e.get("fill_color") or "")
    if c.startswith("rgba("):
        try:
            a *= float(c[5:-1].split(",")[3])
        except (ValueError, IndexError):
            pass
    return a


def _decor(u, els):
    """faint non-text layers (glows, bubbles): free to overlap, never pushed or packed"""
    return all(els[i].get("type") != "text" and _alpha(els[i]) <= 0.5 for i in u)


def _resolve(els, out, boxes, fsz, units, src, bleed, TW, TH, floor, dar=0.0):
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
    # source extents as drawn: a text's box is padding, its ink is what may not collide
    drawn = {}
    for u in units:
        for i in u:
            drawn[i] = _drawn_box(els[i], src[i]) if els[i].get("type") == "text" else src[i]
    # layered compositions (a photo in a frame with a title over it) move as one cluster
    units = _clusters(units, drawn, els)
    src_bb = {k: _unit_bbox(u, drawn) for k, u in enumerate(units)}
    # units that came out bigger than the canvas shrink uniformly
    for u in units:
        b = _unit_bbox(u, boxes)
        s = min(1.0, (TW - 2 * margin) / max(1e-3, b[2]), (TH - 2 * margin) / max(1e-3, b[3]))
        if s < 1.0:
            _scale_unit(u, boxes, s, fsz)

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

    if dar > REFLOW_DAR:
        dec0 = [_decor(u, els) for u in units]
        _pack(units, boxes, fsz, src_bb, set(), els, TW, TH, margin, grow=1.6)
    for k in range(len(units)):
        clamp(k)
    # collisions that the source did not have
    allowed = set()
    dec = [_decor(u, els) for u in units]
    for a in range(len(units)):
        for b in range(a + 1, len(units)):
            if dec[a] or dec[b] or _ioa(src_bb[a], src_bb[b]) > 0.05:
                allowed.add((a, b))
    area = [src_bb[k][2] * src_bb[k][3] for k in range(len(units))]
    for it in range(60):
        worst, wv = None, 0.0
        bbs = [_unit_bbox(u, boxes) for u in units]
        for a in range(len(units)):
            for b in range(a + 1, len(units)):
                if dec[a] or dec[b]:
                    continue
                if (a, b) in allowed and _ioa(bbs[a], bbs[b]) <= _ioa(src_bb[a], src_bb[b]) + 0.1:
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
    # still colliding: pack the units in reading order (the designer-transfer order), the
    # largest uniform scale that fits, rows or columns (one owner: this only re-places)
    bbs = [_unit_bbox(u, boxes) for u in units]
    left = sum(_inter(bbs[a], bbs[b]) for a in range(len(units)) for b in range(a + 1, len(units))
               if not (dec[a] or dec[b]) and ((a, b) not in allowed or _ioa(bbs[a], bbs[b]) > _ioa(src_bb[a], src_bb[b]) + 0.1))
    tot = sum(b[2] * b[3] for b in bbs) or 1.0
    if left > 0.02 * tot:
        _pack(units, boxes, fsz, src_bb, allowed, els, TW, TH, margin)
    _separate_copy(els, boxes, fsz, units, drawn, TH)
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
                if not textfit.fits(e, fs, w, h):
                    fs, x, y, w, h = _rescue_text(e, fsz[i], x, y, w, h, TW, TH)
                    out[i].update(x=float(x), y=float(y), width=float(w), height=float(h))
                out[i]["font_size"] = int(fs)
                if e.get("_lsem") is not None:
                    out[i]["letter_spacing"] = round(float(e["_lsem"]) * int(fs), 3)
            if els[i].get("corner_radius") and src[i][2] > 0:
                out[i]["corner_radius"] = num(els[i]["corner_radius"]) * (w / src[i][2])


def _pack(units, boxes, fsz, src_bb, allowed, els, TW, TH, margin, grow=1.0):
    n = len(units)
    items = [k for k in range(n) if not _decor(units[k], els)]
    if len(items) == 1:  # one composition: as large as fits, centred
        k = items[0]
        b = _unit_bbox(units[k], boxes)
        sc = min(grow, (TW - 2 * margin) / max(1e-6, b[2]), (TH - 2 * margin) / max(1e-6, b[3]))
        _scale_unit(units[k], boxes, sc, fsz)
        b = _unit_bbox(units[k], boxes)
        _shift(units[k], boxes, (TW - b[2]) / 2 - b[0], (TH - b[3]) / 2 - b[1])
        return
    if len(items) < 2:
        return
    # reading order of the source
    items = _reading_order(items, src_bb)
    gap = margin * 0.6
    floor = floor_px(TW, TH)
    base = {k: _unit_bbox(units[k], boxes)[2:] for k in items}
    single_text = [k for k in items if len(units[k]) == 1 and units[k][0] in fsz]
    min_fs = min([fsz[i] for k in items for i in units[k] if i in fsz] or [floor])
    best = None
    # text re-wrap variants: keep, or 1..4 lines (the box takes the shape its text needs)
    for TL in (0, 1, 2, 3, 4):
        sz = dict(base)
        shapes = {}
        if TL:
            if not single_text:
                continue
            for k in single_text:
                i = units[k][0]
                e, fs = els[i], fsz[i]
                w = max(textfit.longest_word(e, fs) * 1.02 + 1, textfit.natural_width(e, fs) / TL * 1.08 + 1)
                _, h, _ = textfit.layout(e, fs, w)
                shapes[k] = (w, h + 1)
                sz[k] = shapes[k]
        for rows in (True, False):
            for L in range(1, min(4, len(items)) + 1):
                main = [sz[k][0] if rows else sz[k][1] for k in items]
                total = sum(main)
                lines, cur, acc = [], [], 0.0
                for k, m in zip(items, main):
                    if cur and acc + m > total / L * 1.15 and len(lines) < L - 1:
                        lines.append(cur)
                        cur, acc = [], 0.0
                    cur.append(k)
                    acc += m
                lines.append(cur)
                for tf in (False, True):
                    res = _pack_scales(items, lines, sz, rows, units, fsz, floor, gap, margin, TW, TH, grow, tf)
                    if res is None:
                        continue
                    val, scales = res
                    if best is None or val > best[0] + 1e-6:
                        best = (val, scales, rows, lines, shapes)
    if best is None:
        return
    _, scales, rows, lines, shapes = best
    for k, (w, h) in shapes.items():
        i = units[k][0]
        x, y, w0, h0 = boxes[i]
        boxes[i] = [x + w0 / 2 - w / 2, y + h0 / 2 - h / 2, w, h]
    sz = {k: _unit_bbox(units[k], boxes)[2:] for k in items}
    for k in items:
        if abs(scales[k] - 1) > 1e-3:
            _scale_unit(units[k], boxes, scales[k], fsz)
        sz[k] = _unit_bbox(units[k], boxes)[2:]
    if rows:
        block_h = sum(max(sz[k][1] for k in ln) for ln in lines) + (len(lines) - 1) * gap
        y = (TH - block_h) / 2
        for ln in lines:
            lh = max(sz[k][1] for k in ln)
            lw = sum(sz[k][0] for k in ln) + (len(ln) - 1) * gap
            x = (TW - lw) / 2
            for k in ln:
                b = _unit_bbox(units[k], boxes)
                _shift(units[k], boxes, x - b[0], y + (lh - b[3]) / 2 - b[1])
                x += sz[k][0] + gap
            y += lh + gap
    else:
        block_w = sum(max(sz[k][0] for k in ln) for ln in lines) + (len(lines) - 1) * gap
        x = (TW - block_w) / 2
        for ln in lines:
            lw = max(sz[k][0] for k in ln)
            lh = sum(sz[k][1] for k in ln) + (len(ln) - 1) * gap
            y = (TH - lh) / 2
            for k in ln:
                b = _unit_bbox(units[k], boxes)
                _shift(units[k], boxes, x + (lw - b[2]) / 2 - b[0], y - b[1])
                y += sz[k][1] + gap
            x += lw + gap


def _pack_scales(items, lines, sz, rows, units, fsz, floor, gap, margin, TW, TH, grow, text_first):
    """Per-item scales for one arrangement: everything at the largest common scale s, or (text
    first) copy held at the legibility floor while pictures and shapes take the largest s that
    still fits. Returns (value, scales) or None."""
    tmin = {k: min([fsz[i] for i in units[k] if i in fsz] or [0]) for k in items}

    def scales_for(s):
        out = {}
        for k in items:
            if text_first and tmin[k] > 0:
                out[k] = min(grow, max(s, floor / tmin[k]))
            else:
                out[k] = s
        return out

    def fits(sc):
        if rows:
            need_w = max(sum(sz[k][0] * sc[k] for k in ln) + (len(ln) - 1) * gap for ln in lines)
            need_h = sum(max(sz[k][1] * sc[k] for k in ln) for ln in lines) + (len(lines) - 1) * gap
        else:
            need_h = max(sum(sz[k][1] * sc[k] for k in ln) + (len(ln) - 1) * gap for ln in lines)
            need_w = sum(max(sz[k][0] * sc[k] for k in ln) for ln in lines) + (len(lines) - 1) * gap
        return need_w <= TW - 2 * margin + 1e-6 and need_h <= TH - 2 * margin + 1e-6

    lo, hi = 0.0, grow
    if fits(scales_for(hi)):
        lo = hi
    else:
        if not fits(scales_for(1e-4)):
            return None
        lo = 1e-4
        for _ in range(30):
            mid = (lo + hi) / 2
            if fits(scales_for(mid)):
                lo = mid
            else:
                hi = mid
    sc = scales_for(lo)
    texts = [tmin[k] * sc[k] for k in items if tmin[k] > 0]
    leg = min(1.0, min(texts) / floor) if texts else 1.0
    # pictures (logo, product) must stay recognisable too
    pics = [min(sz[k]) * sc[k] for k in items if tmin[k] == 0]
    vis = min(1.0, min(pics) / (0.3 * min(TW, TH))) if pics else 1.0
    mean = sum(sc.values()) / len(sc)
    return mean * leg ** 3 * vis ** 2, sc


def _rescue_text(e, fs0, x, y, w, h, TW, TH):
    """Text that fits at no size in its box (long words, wide letter spacing): widen the box to
    the canvas, then take the largest size <= the wanted one that fits, else the smallest
    larger size that does (more wrapping means fewer spaced glyphs per line)."""
    w = TW
    x = 0.0
    fs0 = max(1, int(fs0))
    pick = None
    for fs in range(fs0, 0, -1):
        if textfit.fits(e, fs, w, TH):
            pick = fs
            break
    if pick is None:
        f = float(fs0 + 1)
        while f <= max(fs0 + 1, TH):
            if textfit.fits(e, int(f), w, TH):
                pick = int(f)
                break
            f = max(f + 1, f * 1.12)
    if pick is None:
        pick = 1
    _, bh, _ = textfit.layout(e, pick, w)
    h = min(TH, bh + 1)
    y = min(max(0.0, y + 0.0), TH - h)
    return pick, x, y, w, h


def _drawn_box(e, b):
    try:
        fs = num(e.get("font_size"), 16.0)
        mw, bh, _ = textfit.layout(e, fs, b[2])
    except Exception:  # noqa: BLE001
        return b
    al = e.get("text_align", "left")
    x = b[0] + ((b[2] - mw) / 2 if al == "center" else (b[2] - mw) if al == "right" else 0.0)
    return [x, b[1], max(1.0, mw), max(1.0, bh)]


def _reading_order(items, bb):
    """Top to bottom; items sharing a line (vertical spans overlapping by half the shorter) left
    to right."""
    items = sorted(items, key=lambda k: (bb[k][1] + bb[k][3] / 2, bb[k][0]))
    lines = []
    for k in items:
        if lines:
            ln = lines[-1]
            y0 = min(bb[j][1] for j in ln)
            y1 = max(bb[j][1] + bb[j][3] for j in ln)
            ov = min(y1, bb[k][1] + bb[k][3]) - max(y0, bb[k][1])
            if ov > 0.5 * min(bb[k][3], y1 - y0):
                ln.append(k)
                continue
        lines.append([k])
    return [k for ln in lines for k in sorted(ln, key=lambda j: bb[j][0])]


def _clusters(units, drawn, els, thr=0.2):
    """Merge units whose drawn source extents overlap (share of the smaller >= thr); faint decor
    stays apart."""
    n = len(units)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    bbs = [_unit_bbox(u, drawn) for u in units]
    dec = [_decor(u, els) for u in units]
    pic = [any(els[i].get("type") != "text" for i in u) for u in units]
    for a in range(n):
        for b in range(a + 1, n):
            # copy over copy is not a composition (padded boxes touch); copy on a picture is
            if not dec[a] and not dec[b] and (pic[a] or pic[b]) and _ioa(bbs[a], bbs[b]) >= thr:
                parent[find(a)] = find(b)
    out = {}
    for k in range(n):
        out.setdefault(find(k), []).extend(units[k])
    return list(out.values())


def _separate_copy(els, boxes, fsz, units, drawn, TH):
    """Inside a cluster, copy that did not touch in the source must not touch now: the lower
    text moves down by the overlap (or the upper one up when there is no room)."""
    for u in units:
        txt = [i for i in u if i in fsz]
        if len(txt) < 2:
            continue
        for _ in range(3 * len(txt)):
            moved = False
            ink = {i: _drawn_box(dict(els[i], font_size=fsz[i]), boxes[i]) for i in txt}
            for a in range(len(txt)):
                for b in range(a + 1, len(txt)):
                    i, j = txt[a], txt[b]
                    if _ioa(drawn[i], drawn[j]) > 0.05 or _inter(ink[i], ink[j]) <= 0:
                        continue
                    up, lo = (i, j) if ink[i][1] <= ink[j][1] else (j, i)
                    dy = ink[up][1] + ink[up][3] - ink[lo][1] + 1
                    if ink[lo][1] + ink[lo][3] + dy <= TH:
                        boxes[lo][1] += dy
                    else:
                        boxes[up][1] -= min(dy, ink[up][1])
                    moved = True
                    break
                if moved:
                    break
            if not moved:
                break
