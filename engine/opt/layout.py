"""Realise a topology (columns -> rows -> blocks) at the largest scale that fits (owner: geometry
of a candidate). A placed layout is a dict: bid -> Placed."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .analyse import Block, Scene
from .measure import fast_block, fast_longest, fast_natural


@dataclass
class Placed:
    x: float
    y: float
    w: float
    h: float
    s: float                 # geometric scale of the block
    fs: Optional[int] = None  # single-text font size
    lines: int = 0


@dataclass
class Params:
    margin_frac: float = 0.05      # margin, fraction of the short side
    gap_frac: float = 1.0          # multiplier on the source gap
    text_widen: float = 1.0        # exponent on aspect change for text widening
    min_fs: float = 8.0


def text_floor(S, fs0: float, W: float, H: float) -> float:
    """Per-text floor: the target floor, relaxed in proportion for copy the designer already set
    below the source's own floor (small print stays small print, but never shrinks further in
    judge-view terms)."""
    ft, fsrc = fs_floor(W, H), fs_floor(S.W, S.H)
    return max(1.0, min(ft, ft * fs0 / fsrc) * getattr(S, "floor_k", 1.0))


def fs_floor(W: float, H: float) -> float:
    """Legibility floor = L0's small-print floor: 7 px cap height when the judge views the canvas
    at 1024 px on the long side (cap = 0.70 em), never under 10 px."""
    return float(max(10, math.ceil(7.0 * max(W, H) / 1024.0 / 0.70)))


def margins(W, H, p: Params) -> Tuple[float, float]:
    md = min(W, H)
    mx = max(p.margin_frac * md, 0.02 * W)
    my = max(p.margin_frac * md, 0.02 * H)
    if H < 140:
        my = max(2.0, 0.06 * H)
    if W < 200:
        mx = max(3.0, 0.05 * W)
    return mx, my


def size_block(S: Scene, b: Block, s: float, slot_w: float, kw: float, W: float, H: float, hug: bool = True) -> Placed:
    if b.kind == "text":
        el = S.els[b.text_idx]
        fs0 = float(el.get("font_size", 16) or 16)
        fs = max(1, int(round(fs0 * s)))
        fs = max(fs, min(int(math.ceil(text_floor(S, fs0, W, H))), int(round(fs0 * 3))))
        fs = min(fs, 4000)
        # shrink until the longest word fits the slot
        lw = fast_longest(el, fs)
        if lw > slot_w and lw > 0:
            fs = max(1, int(fs * slot_w / lw))
            while fs > 1 and fast_longest(el, fs) > slot_w:
                fs -= 1
        lw = fast_longest(el, fs)
        want = b.w * s * kw
        nat = fast_natural(el, fs)
        width = max(lw + 1, min(slot_w, max(want, 1.0)))
        if hug:
            width = min(width, nat * 1.05 + 4)
        n, maxw, h = fast_block(el, fs, width)
        width = max(width, maxw + 1)
        return Placed(0, 0, width, h, s=fs / fs0, fs=fs, lines=n)
    k = s
    tfs = [float(S.els[i].get("font_size", 16) or 16) for i in b.members if S.els[i].get("type") == "text"]
    if tfs:
        # legibility floor for text inside a rigid group (badge, button), bounded by the slot
        need = max(text_floor(S, f, W, H) / f for f in tfs)
        k = max(k, min(need, 1.8 * s, slot_w / b.w, 0.9 * H / b.h))   # boost, but stay in proportion
    k = max(k, min(20.0, 0.4 * min(W, H)) / max(1e-6, min(b.w, b.h)))   # visible; >= 20 px keeps bitmap rounding under 5%
    return Placed(0, 0, b.w * k, b.h * k, s=k)


Topo = List[List[List[int]]]   # columns -> rows -> block ids


def realise(S: Scene, topo: Topo, W: float, H: float, p: Params, s_cap: float, gap0: float,
            valign: str = "spread") -> Tuple[Dict[int, Placed], float]:
    mx, my = margins(W, H, p)
    iw, ih = W - 2 * mx, H - 2 * my
    ncol = len(topo)
    kw = 1.0
    ar_t = W / H
    ar_s = S.W / S.H
    kw = min(3.0, max(1.0, (ar_t / ar_s) ** (0.5 * p.text_widen)))
    byid = {b.bid: b for b in S.blocks}

    def build(s):
        g = max(0.006 * min(W, H), gap0 * s * p.gap_frac)
        placed = {}
        cols = []
        for col in topo:
            slot_c = iw / ncol if ncol > 1 else iw
            rows = []
            for row in col:
                slot = (slot_c - g * (len(row) - 1)) / max(1, len(row))
                ps = [size_block(S, byid[i], s, max(4.0, slot), kw, W, H) for i in row]
                for i, pl in zip(row, ps):
                    placed[i] = pl
                rw = sum(pl.w for pl in ps) + g * (len(ps) - 1)
                rh = max(pl.h for pl in ps)
                rows.append((row, rw, rh))
            cw = max(r[1] for r in rows)
            chh = sum(r[2] for r in rows) + g * (len(rows) - 1)
            cols.append((rows, cw, chh))
        tw = sum(c[1] for c in cols) + g * (len(cols) - 1)
        th = max(c[2] for c in cols)
        return placed, cols, tw, th, g

    lo, hi = 0.005, s_cap
    best = None
    pl = build(hi)
    if pl[2] <= iw and pl[3] <= ih:
        best = (hi, pl)
    else:
        for _ in range(14):
            mid = math.sqrt(lo * hi)
            pl = build(mid)
            if pl[2] <= iw and pl[3] <= ih:
                lo = mid
                best = (mid, pl)
            else:
                hi = mid
        if best is None:
            best = (lo, build(lo))
    s, (placed, cols, tw, th, g) = best

    # horizontal distribution of columns
    free_x = max(0.0, iw - tw)
    if len(cols) > 1:
        gx = g + min(free_x / (len(cols) + 1), 4 * g) if True else g
        used = sum(c[1] for c in cols) + gx * (len(cols) - 1)
        x = mx + (iw - used) / 2
    else:
        gx = g
        used = cols[0][1]
        x = mx + {"left": 0.0, "center": (iw - used) / 2, "right": iw - used}.get(S.halign, (iw - used) / 2)
    for rows, cw, chh in cols:
        free_y = max(0.0, ih - chh)
        nr = len(rows)
        if valign == "spread" and nr > 1:
            gy = g + min(free_y / (nr + 1), 3 * g)
        else:
            gy = g
        used_h = sum(r[2] for r in rows) + gy * (nr - 1)
        y = my + (ih - used_h) / 2
        for row, rw, rh in rows:
            if len(cols) > 1 or S.halign == "center":
                rx = x + (cw - rw) / 2
            elif S.halign == "left":
                rx = x
            else:
                rx = x + cw - rw
            for i in row:
                pp = placed[i]
                pp.x = rx
                pp.y = y + (rh - pp.h) / 2
                rx += pp.w + g
            y += rh + gy
        x += cw + gx
    return placed, s


def keep_layout(S: Scene, W: float, H: float, p: Params, spread) -> Tuple[Dict[int, Placed], float]:
    """Uniform scale of the whole composition (spread=False), or each block at its normalised
    centre with the uniform scale (spread=True)."""
    mx, my = margins(W, H, p)
    if not S.blocks:
        return {}, 1.0
    x0 = min(b.x for b in S.blocks)
    y0 = min(b.y for b in S.blocks)
    x1 = max(b.x + b.w for b in S.blocks)
    y1 = max(b.y + b.h for b in S.blocks)
    cw, chh = max(1, x1 - x0), max(1, y1 - y0)
    s = min((W - 2 * mx) / cw, (H - 2 * my) / chh, max(W / S.W, H / S.H))
    if spread == "canvas":
        s = min(W / S.W, H / S.H)
    placed = {}
    kw = 1.0
    for b in S.blocks:
        pl = size_block(S, b, s, max(4.0, W - 2 * mx), kw, W, H, hug=False)
        if spread == "canvas":
            cx = W / 2 + ((b.x + b.w / 2) - S.W / 2) * s
            cy = H / 2 + ((b.y + b.h / 2) - S.H / 2) * s
        elif spread:
            cx = (b.x + b.w / 2) / S.W * W
            cy = (b.y + b.h / 2) / S.H * H
        else:
            cx = W / 2 + ((b.x + b.w / 2) - (x0 + x1) / 2) * s
            cy = H / 2 + ((b.y + b.h / 2) - (y0 + y1) / 2) * s
        pl.x = min(max(mx, cx - pl.w / 2), W - mx - pl.w)
        pl.y = min(max(my, cy - pl.h / 2), H - my - pl.h)
        placed[b.bid] = pl
    return placed, s
