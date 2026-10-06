"""A small flexbox-like layout engine.

Nodes are measured at a global scale ``s`` (1.0 = source size) inside an
available width/height and either return a concrete size or ``None`` when
they cannot fit. The caller searches for the largest ``s`` that fits.

* Leaf    - one item (text / image / group)
* VStack  - children top to bottom
* HStack  - children left to right; text children share the free width, each
            getting at least the narrowest width that still fits the height
* Flow    - like HStack but wraps onto more lines (badge rows)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .analyze import Item
from .items import INF, Ctx, Real, realize

Placement = Tuple[Real, float, float]


@dataclass
class Laid:
    w: float
    h: float
    place: Callable[[float, float], List[Placement]]
    min_scale: float = 1.0  # smallest effective scale of any text inside (for scoring)
    oh: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # painted overhang l, t, r, b


def _oh_stack(parts, offsets, w: float, h: float):
    """Overhang of a container from its parts' overhangs and (x, y) offsets."""
    l = t = r = b = 0.0
    for p, (ox, oy) in zip(parts, offsets):
        pl, pt, pr, pb = p.oh
        l = max(l, pl - ox)
        t = max(t, pt - oy)
        r = max(r, pr - (w - ox - p.w))
        b = max(b, pb - (h - oy - p.h))
    return (l, t, r, b)


class Node:
    flex = False  # can trade width for height

    def measure(self, s: float, ctx: Ctx, max_w: float, max_h: float, relax: float) -> Optional[Laid]:
        raise NotImplementedError

    def items(self) -> List[Item]:
        raise NotImplementedError


class Leaf(Node):
    def __init__(self, item: Item, mode: str = "stack", k: float = 1.0, fill: bool = False,
                 max_w_frac: float = 1.0, cap_rel: float = 1.6, cap_floor: float = 0.0, min_rel: float = 0.0,
                 nowrap: bool = False):
        self.item = item
        self.nowrap = nowrap  # buttons: keep the label on one line
        self.mode = mode
        self.k = k
        self.fill = fill and item.kind in ("image", "composite")  # grow into leftover space (heroes)
        self.max_w_frac = max_w_frac
        self.cap_rel = cap_rel  # fill images never exceed max(cap_rel * s, cap_floor) x source size
        self.cap_floor = cap_floor
        self.min_rel = min_rel  # ... and must reach at least min_rel * s
        # text can trade width for lines; a button can trade padding for width
        self.flex = item.kind == "text" or (item.kind == "group" and item.container is not None
                                            and item.centered_label)

    def items(self) -> List[Item]:
        return [self.item]

    def measure(self, s, ctx, max_w, max_h, relax):
        it = self.item
        if self.fill:
            cap = max(self.cap_rel * s, self.cap_floor)
            r = realize(it, cap, ctx, max_w * self.max_w_frac, max_h)
            if r is None or (r.scale < self.min_rel * s * 0.999 and it.key not in ctx.pinned):  # pinned: at its floor
                return None
        else:
            r = realize(it, s, ctx, max_w, max_h, relax=relax, k=self.k, mode=self.mode, nowrap=self.nowrap)
        if r is None:
            return None
        return Laid(r.w, r.h, lambda x, y, r=r: [(r, x, y)],
                    min_scale=r.scale if it.kind not in ("image", "composite") else 9.0, oh=r.oh)


def _gap(base: float, s: float, ctx: Ctx, lo: float = 0.35) -> float:
    return max(base * s, lo * ctx.base_fs)


class VStack(Node):
    def __init__(self, children: Sequence[Node], gaps: Optional[Sequence[float]] = None,
                 align: Optional[Sequence[str]] = None, default_align: str = "center",
                 gap_min: float = 0.35):
        self.children = list(children)
        n = len(self.children)
        self.gaps = list(gaps) if gaps is not None else [None] * max(0, n - 1)
        self.align = list(align) if align is not None else [default_align] * n
        self.gap_min = gap_min
        self.flex = any(c.flex for c in self.children)

    def items(self):
        return [i for c in self.children for i in c.items()]

    def gap_px(self, i: int, s: float, ctx: Ctx) -> float:
        g = self.gaps[i]
        if g is None:
            g = 1.2 * ctx.base_fs / max(s, 1e-6)
        return _gap(g, s, ctx, self.gap_min)

    def measure(self, s, ctx, max_w, max_h, relax):
        if not self.children:
            return Laid(0, 0, lambda x, y: [])
        n = len(self.children)
        gaps = [self.gap_px(i, s, ctx) for i in range(n - 1)]
        total = sum(gaps)
        if total > max_h + 0.5:
            return None
        laids: List[Optional[Laid]] = [None] * n
        fill = [i for i, c in enumerate(self.children) if isinstance(c, Leaf) and c.fill]
        for i, c in enumerate(self.children):
            if i in fill:
                continue
            l = c.measure(s, ctx, max_w, max_h - total, relax)
            if l is None:
                return None
            laids[i] = l
            total += l.h
            if total > max_h + 0.5:
                return None
        for i in fill:  # heroes take what is left, shared evenly
            share = (max_h - total) / max(1, len([j for j in fill if laids[j] is None]))
            l = self.children[i].measure(s, ctx, max_w, share, relax)
            if l is None:
                return None
            laids[i] = l
            total += l.h
        w = max(l.w for l in laids)
        align = self.align

        def place(x, y, laids=laids, gaps=gaps, w=w):
            out = []
            cy = y
            for i, l in enumerate(laids):
                a = align[i] if i < len(align) else "center"
                lx = x + (w - l.w) / 2.0 if a == "center" else (x if a == "start" else x + w - l.w)
                out += l.place(lx, cy)
                cy += l.h + (gaps[i] if i < len(gaps) else 0.0)
            return out

        offs, cy = [], 0.0
        for i, l in enumerate(laids):
            a = align[i] if i < len(align) else "center"
            offs.append(((w - l.w) / 2.0 if a == "center" else (0.0 if a == "start" else w - l.w), cy))
            cy += l.h + (gaps[i] if i < len(gaps) else 0.0)
        laid = Laid(w, total, place, min_scale=min(l.min_scale for l in laids), oh=_oh_stack(laids, offs, w, total))
        laid.parts = laids  # type: ignore[attr-defined]
        laid.gap_list = gaps  # type: ignore[attr-defined]
        return laid


def _min_width(node: Node, s, ctx, max_w, max_h, relax) -> Optional[float]:
    """Narrowest width at which ``node`` still fits ``max_h``."""
    if node.measure(s, ctx, max_w, max_h, relax) is None:
        return None
    lo, hi = 1.0, max_w
    for _ in range(14):
        mid = (lo + hi) / 2.0
        if node.measure(s, ctx, mid, max_h, relax) is None:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0:
            break
    return hi


class HStack(Node):
    def __init__(self, children: Sequence[Node], gaps: Optional[Sequence[float]] = None,
                 valign: str = "center", gap_min: float = 0.8):
        self.children = list(children)
        n = len(self.children)
        self.gaps = list(gaps) if gaps is not None else [None] * max(0, n - 1)
        self.valign = valign
        self.gap_min = gap_min
        self.flex = any(c.flex for c in self.children)

    def items(self):
        return [i for c in self.children for i in c.items()]

    def gap_px(self, i, s, ctx):
        g = self.gaps[i]
        if g is None:
            return max(1.6 * ctx.base_fs, self.gap_min * ctx.base_fs)
        return _gap(g, s, ctx, self.gap_min)

    def measure(self, s, ctx, max_w, max_h, relax):
        n = len(self.children)
        if n == 0:
            return Laid(0, 0, lambda x, y: [])
        gaps = [self.gap_px(i, s, ctx) for i in range(n - 1)]
        avail = max_w - sum(gaps)
        if avail <= 0:
            return None
        fixed: dict = {}
        flex_idx = []
        for i, c in enumerate(self.children):
            if c.flex:
                flex_idx.append(i)
            else:
                l = c.measure(s, ctx, avail, max_h, relax)
                if l is None:
                    return None
                fixed[i] = l
        used = sum(l.w for l in fixed.values())
        free = avail - used
        if free < 0:
            return None
        laids: dict = dict(fixed)
        if flex_idx:
            mins, nats = {}, {}
            for i in flex_idx:
                mw = _min_width(self.children[i], s, ctx, free, max_h, relax)
                if mw is None:
                    return None
                mins[i] = mw
                nat = self.children[i].measure(s, ctx, free, max_h, relax)
                nats[i] = nat.w if nat is not None else mw
            if sum(mins.values()) > free + 0.5:
                return None
            extra = free - sum(mins.values())
            want = {i: max(0.0, nats[i] - mins[i]) for i in flex_idx}
            tot_want = sum(want.values())
            for i in flex_idx:
                give = mins[i] + (extra * want[i] / tot_want if tot_want > 0 else 0.0)
                give = min(give, max(mins[i], nats[i])) if tot_want <= extra else give
                l = self.children[i].measure(s, ctx, give + 0.01, max_h, relax)
                if l is None:
                    l = self.children[i].measure(s, ctx, mins[i] + 0.01, max_h, relax)
                if l is None:
                    return None
                laids[i] = l
        ordered = [laids[i] for i in range(n)]
        w = sum(l.w for l in ordered) + sum(gaps)
        h = max(l.h for l in ordered)
        if w > max_w + 0.5 or h > max_h + 0.5:
            return None
        valign = self.valign

        def place(x, y, ordered=ordered, gaps=gaps, h=h, extra_gap=0.0):
            out = []
            cx = x
            for i, l in enumerate(ordered):
                ly = y + (h - l.h) / 2.0 if valign == "center" else (y if valign == "start" else y + h - l.h)
                out += l.place(cx, ly)
                cx += l.w + (gaps[i] + extra_gap if i < len(gaps) else 0.0)
            return out

        offs, cx = [], 0.0
        for i, l in enumerate(ordered):
            oy = (h - l.h) / 2.0 if valign == "center" else (0.0 if valign == "start" else h - l.h)
            offs.append((cx, oy))
            cx += l.w + (gaps[i] if i < len(gaps) else 0.0)
        laid = Laid(w, h, place, min_scale=min(l.min_scale for l in ordered), oh=_oh_stack(ordered, offs, w, h))
        laid.parts = ordered  # type: ignore[attr-defined]
        laid.gap_list = gaps  # type: ignore[attr-defined]
        return laid


class Flow(Node):
    """Children in rows, wrapping when the width runs out (rows centred)."""

    def __init__(self, children: Sequence[Node], gap: Optional[float] = None, align: str = "center"):
        self.children = list(children)
        self.gap = gap
        self.align = align

    def items(self):
        return [i for c in self.children for i in c.items()]

    def measure(self, s, ctx, max_w, max_h, relax):
        g = _gap(self.gap, s, ctx, 0.5) if self.gap is not None else 0.8 * ctx.base_fs
        laids = []
        for c in self.children:
            l = c.measure(s, ctx, max_w, max_h, relax)
            if l is None:
                return None
            laids.append(l)
        rows: List[List[Laid]] = [[]]
        rw = 0.0
        for l in laids:
            need = l.w if not rows[-1] else rw + g + l.w
            if rows[-1] and need > max_w + 0.5:
                rows.append([l])
                rw = l.w
            else:
                rows[-1].append(l)
                rw = need
        row_w = [sum(l.w for l in r) + g * (len(r) - 1) for r in rows]
        row_h = [max(l.h for l in r) for r in rows]
        w = max(row_w)
        h = sum(row_h) + g * 0.8 * (len(rows) - 1)
        if w > max_w + 0.5 or h > max_h + 0.5:
            return None
        align = self.align

        def place(x, y):
            out = []
            cy = y
            for r, rw_, rh in zip(rows, row_w, row_h):
                cx = x + (w - rw_) / 2.0 if align == "center" else (x if align == "start" else x + w - rw_)
                for l in r:
                    out += l.place(cx, cy + (rh - l.h) / 2.0)
                    cx += l.w + g
                cy += rh + g * 0.8
            return out

        oh = tuple(max(l.oh[k] for l in laids) for k in range(4))
        return Laid(w, h, place, min_scale=min(l.min_scale for l in laids), oh=oh)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# global scale search


def search_scale(fits: Callable[[float], Optional[Laid]], lo: float, hi: float, iters: int = 18):
    """Largest s in [lo, hi] for which ``fits(s)`` succeeds (assumes monotone)."""
    best = fits(hi)
    if best is not None:
        return hi, best
    best = fits(lo)
    if best is None:
        # feasibility is not always monotone at the very bottom (floors kick in):
        # walk up a ladder before giving up
        for k in range(1, 7):
            cand = lo * (hi / lo) ** (k / 7.0)
            best = fits(cand)
            if best is not None:
                lo = cand
                break
        if best is None:
            return None, None
    best_s = lo
    a, b = lo, hi
    for _ in range(iters):
        mid = (a * b) ** 0.5 if a > 0 else (a + b) / 2.0
        r = fits(mid)
        if r is None:
            b = mid
        else:
            a, best_s, best = mid, mid, r
        if b / max(a, 1e-9) < 1.004:
            break
    return best_s, best


def isotonic(t: List[float], w: List[float]) -> List[float]:
    """Weighted pool-adjacent-violators: closest non-decreasing sequence to t."""
    blocks = []  # [value, weight, count]
    for ti, wi in zip(t, w):
        blocks.append([ti, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            ww = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / ww, ww, c1 + c2])
    out = []
    for v, _, c in blocks:
        out += [v] * c
    return out


def chain_positions(targets: List[float], weights: List[float], sizes: List[float], gaps: List[float],
                    lo: float, hi: float) -> List[float]:
    """Positions closest to ``targets`` keeping order, sizes and minimum gaps, inside [lo, hi]."""
    n = len(sizes)
    off = [0.0] * n
    for i in range(1, n):
        off[i] = off[i - 1] + sizes[i - 1] + gaps[i - 1]
    t = [targets[i] - off[i] for i in range(n)]
    z = isotonic(t, weights)
    upper = hi - sizes[-1] - off[-1]
    if upper < lo:
        z = [lo] * n
    else:
        z = [min(max(v, lo), upper) for v in z]
    return [z[i] + off[i] for i in range(n)]
