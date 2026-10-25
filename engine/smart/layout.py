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

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .analyze import Item
from .items import BUDGET, INF, Budget, BudgetExceeded, Ctx, Real, realize  # noqa: F401 (re-exported)

Placement = Tuple[Real, float, float]
RHYTHM = 1.0  # a gap in a column is at most this times the smaller of the rows it separates
SEPARATION = 0.5  # ... and never gives way below this times it (unless the source set them closer)
CLEAR_SPACE = 0.5  # a logo keeps this times its height clear above and below


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




def _step(ctx: Ctx) -> float:
    return max(0.5, Budget.GRID * ctx.base_fs)


def _grid(v: float, ctx: Ctx) -> float:
    """``v`` rounded down to the economy grid (a quarter of the readable size)."""
    step = _step(ctx)
    return math.floor(v / step) * step if v < INF else v


def _snap(s: float) -> float:
    """A scale rounded to the economy's geometric grid, so the many searches of
    one design (relax steps, zones, insets) probe the same scales and share
    their measurements."""
    g = math.log(Budget.TOLERANCE)
    return math.exp(math.floor(math.log(s) / g) * g)


def _ctx_key(ctx: Ctx) -> tuple:
    return (ctx.W, ctx.H, ctx.W0, ctx.H0, ctx.emphasis, ctx.measure, ctx.display_room, ctx.cta_min, ctx.keep_px)


class Node:
    flex = False  # can trade width for height

    def measure(self, s: float, ctx: Ctx, max_w: float, max_h: float, relax: float) -> Optional[Laid]:
        """The node's size at scale ``s`` within max_w x max_h, or None if it
        does not fit. In economy mode (see Budget) the room is rounded down to the economy grid
        and the answer kept on the node, so nested searches stop re-measuring
        the same subtree at widths a hair apart. Rounding down only ever offers
        less room, so what fits still fits."""
        if not BUDGET.economy:
            return self._measure(s, ctx, max_w, max_h, relax)
        max_w, max_h = _grid(max_w, ctx), _grid(max_h, ctx)
        if isinstance(self, Leaf) and not self.fill:
            return self._economy_measure(s, ctx, max_w, max_h, relax)
        key = (s, max_w, max_h, relax, _ctx_key(ctx))
        memo = self.__dict__.setdefault("_memo", {})
        if key not in memo:
            memo[key] = self._measure(s, ctx, max_w, max_h, relax)
        return memo[key]

    def _measure(self, s: float, ctx: Ctx, max_w: float, max_h: float, relax: float) -> Optional[Laid]:
        raise NotImplementedError

    def items(self) -> List[Item]:
        raise NotImplementedError


class Leaf(Node):
    def __init__(self, item: Item, mode: str = "stack", k: float = 1.0, fill: bool = False,
                 max_w_frac: float = 1.0, cap_rel: float = 1.6, cap_floor: float = 0.0, min_rel: float = 0.0,
                 nowrap: bool = False, min_abs: float = 0.0):
        self.item = item
        self.nowrap = nowrap  # buttons: keep the label on one line
        self.mode = mode
        self.k = k
        self.fill = fill and item.kind in ("image", "composite")  # grow into leftover space (heroes)
        self.max_w_frac = max_w_frac
        self.cap_rel = cap_rel  # fill images never exceed max(cap_rel * s, cap_floor) x source size
        self.cap_floor = cap_floor
        self.min_rel = min_rel  # ... and must reach at least min_rel * s (or as wide as max_w allows)
        self.min_abs = min_abs  # ... and at least this times its source size (a picture is no thumbnail)
        self.cap_floor = max(cap_floor, min_abs)
        # text can trade width for lines; a button can trade padding for width
        self.flex = item.kind == "text" or (item.kind == "group" and item.container is not None
                                            and item.centered_label)

    def items(self) -> List[Item]:
        return [self.item]

    def _economy_measure(self, s, ctx, max_w, max_h, relax):
        """Economy (see Budget): an item set in a room keeps that size in any
        smaller room it still fits, and one that did not fit a room fits no
        smaller one, so most rooms are answered from those already tried."""
        seen = self.__dict__.setdefault("_sizes", {}).setdefault((s, relax, _ctx_key(ctx)), [])
        for w, h, laid in seen:
            if laid is None:
                if max_w <= w and max_h <= h:
                    return None
            elif laid.w <= max_w + 0.01 and max_w <= w and laid.h <= max_h + 0.01 and max_h <= h:
                return laid
        laid = self._measure(s, ctx, max_w, max_h, relax)
        seen.append((max_w, max_h, laid))
        return laid

    def natural_h(self, s: float, ctx: Ctx, max_w: float) -> float:
        """Height at the layout scale itself, room permitting (what a hero
        would like before it starts to give way)."""
        r = realize(self.item, s, ctx, max_w * self.max_w_frac, INF)
        return r.h if r is not None else 0.0

    def _measure(self, s, ctx, max_w, max_h, relax):
        it = self.item
        if self.fill:
            grow = max(self.cap_rel * s, self.cap_floor) / max(s, 1e-9)  # as k, so the layout scale stays s
            r = realize(it, s, ctx, max_w * self.max_w_frac, max_h, k=grow)
            if r is None:
                return None
            need = max(self.min_rel * s, self.min_abs)
            if r.scale < need * 0.999:  # too small, unless the width cannot hold more
                widest = realize(it, s, ctx, max_w * self.max_w_frac, INF, k=grow)
                if widest is None or r.scale < min(need, widest.scale) * 0.999:
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

    def _measure(self, s, ctx, max_w, max_h, relax):
        """Rows top to bottom. A gap is the source's gap at this scale, but never
        more than RHYTHM times the smaller of the two rows it separates: spacing
        follows the content, so a large empty stretch in the source does not
        take the room a hero needs. Heroes (fill leaves) take what is left;
        where they leave room, place_column() spreads it as the source did."""
        if not self.children:
            return Laid(0, 0, lambda x, y: [])
        n = len(self.children)
        base = [self.gap_px(i, s, ctx) for i in range(n - 1)]
        least = [min(b, self.gap_min * ctx.base_fs) for b in base]
        total = sum(least)
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
        for i in range(n - 1):  # what a gap never gives up, unless the source set the rows closer
            hs = [laids[j].h for j in (i, i + 1) if laids[j] is not None]
            want = SEPARATION * min(hs) if hs else 0.0
            for j in (i, i + 1):
                c = self.children[j]
                if laids[j] is not None and isinstance(c, Leaf) and c.item.role == "logo":
                    want = max(want, CLEAR_SPACE * laids[j].h)
            lift = max(least[i], min(base[i], want))
            total += lift - least[i]
            least[i] = lift
        if total > max_h + 0.5:
            return None
        gaps = []
        for i, b in enumerate(base):
            sides = [laids[j].h for j in (i, i + 1) if laids[j] is not None]
            # a hairline between two rows does not set the rhythm: its taller neighbour does
            room = max(min(sides), 0.5 * max(sides)) if sides else b
            gaps.append(max(least[i], min(b, RHYTHM * room)))
        if fill:  # spacing gives way before a hero shrinks below its size at this scale
            want = sum(self.children[i].natural_h(s, ctx, max_w) for i in fill)
            short = want - (max_h - total - sum(gaps) + sum(least))
            give = sum(g - l for g, l in zip(gaps, least))
            if short > 0 and give > 0:
                k = min(1.0, short / give)
                gaps = [g - k * (g - l) for g, l in zip(gaps, least)]
        total += sum(gaps) - sum(least)
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


PANEL_PAD = 0.6  # a panel's padding never drops below this times the readable size
PANEL_PAD_MAX = 0.06  # ... nor grows past this share of the canvas a side
PANEL_PAD_LEAST = 0.4  # where room is short it may halve, but never below this times the readable size


def panel_pads(pad: Tuple[float, ...], edges: Tuple[bool, ...], s: float, ctx: Ctx) -> Tuple[float, ...]:
    """A panel's padding on a new canvas (left, top, right, bottom): the
    source's at the layout scale, never thinner than a readable sliver nor
    wider than PANEL_PAD_MAX of the canvas. A side where the panel met the
    canvas edge (a band) has no padding of its own: the canvas margin is it."""
    floor = PANEL_PAD * ctx.base_fs
    out = []
    for i, (p, edge) in enumerate(zip(pad, edges)):
        span = ctx.W if i in (0, 2) else ctx.H
        out.append(floor if edge else max(floor, min(p * s, PANEL_PAD_MAX * span)))
    return tuple(out)


class Padded(Node):
    """Rows framed by a panel (a card, a band). The panel is redrawn around
    them afterwards, so its padding has to be part of the layout: without it
    a card either hugs its copy or runs off the canvas."""

    def __init__(self, child: Node, pad: Tuple[float, ...], edges: Tuple[bool, ...]):
        self.child = child
        self.pad = pad
        self.edges = edges
        self.flex = child.flex

    def items(self):
        return self.child.items()

    def _measure(self, s, ctx, max_w, max_h, relax):
        full = panel_pads(self.pad, self.edges, s, ctx)
        least = PANEL_PAD_LEAST * ctx.base_fs
        tight = tuple(min(p, max(least, p / 2.0)) for p in full)
        for l, t, r, b in (full, tight):  # padding gives way (to a visible sliver) before the content
            if max_w - l - r <= 0 or max_h - t - b <= 0:
                continue
            inner = self.child.measure(s, ctx, max_w - l - r, max_h - t - b, relax)
            if inner is not None:
                break
        else:
            return None
        w, h = inner.w + l + r, inner.h + t + b
        il, it_, ir, ib = inner.oh
        return Laid(w, h, lambda x, y: inner.place(x + l, y + t), min_scale=inner.min_scale,
                    oh=(max(0.0, il - l), max(0.0, it_ - t), max(0.0, ir - r), max(0.0, ib - b)))


def _min_width(node: Node, s, ctx, max_w, max_h, relax) -> Optional[float]:
    """Narrowest width at which ``node`` still fits ``max_h``. In economy mode
    (see Budget) it is searched once per subtree, scale and height over the
    whole canvas width, and only compared with ``max_w``: the narrowest width
    of a subtree does not depend on how much room its parent has."""
    if BUDGET.economy and not isinstance(node, Leaf):
        # short of budget a nested block is not squeezed: it takes the width it has
        laid = node.measure(s, ctx, max_w, max_h, relax)
        return laid.w if laid is not None else None
    if BUDGET.economy:
        key = (s, _grid(max_h, ctx), relax, _ctx_key(ctx))
        memo = node.__dict__.setdefault("_minw", {})
        if key not in memo:
            memo[key] = _economy_width(node, s, ctx, max_h, relax)
        w = memo[key]
        return w if w is not None and w <= max_w else None
    return _bisect_width(node, s, ctx, max_w, max_h, relax)


def _economy_width(node: Node, s, ctx, max_h, relax) -> Optional[float]:
    """Narrowest width to the economy grid, searched below the node's own
    width at the full canvas width (it never needs more)."""
    step = _step(ctx)
    widest = node.measure(s, ctx, ctx.W, max_h, relax)
    if widest is None:
        return None
    hi = widest.w + step
    if hi >= ctx.W or node.measure(s, ctx, hi, max_h, relax) is None:
        hi = ctx.W
    lo = 1.0
    while hi - lo >= step:
        mid = (lo + hi) / 2.0
        if node.measure(s, ctx, mid, max_h, relax) is None:
            lo = mid
        else:
            hi = mid
    return hi


def _bisect_width(node: Node, s, ctx, max_w, max_h, relax) -> Optional[float]:
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

    def _measure(self, s, ctx, max_w, max_h, relax):
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

    def _measure(self, s, ctx, max_w, max_h, relax):
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
        if BUDGET.economy and a > 0:
            mid = _snap(mid)
            if not a < mid < b:
                break
        r = fits(mid)
        if r is None:
            b = mid
        else:
            a, best_s, best = mid, mid, r
        if b / max(a, 1e-9) < (Budget.TOLERANCE if BUDGET.economy else 1.004):
            break
    return best_s, best
