"""Adobe's resize strategies as candidate generators, plus a flow flip.

Adobe's Express-era resize (US12266076B2) picks ONE of four fixed strategies
(springs-and-struts, pin, proportional scale, fit), checks the result with a
critic and falls back to the next one. None of the four can turn a column into
a row. Adobe's verticalize/reflect operator (US11233920B1) can, but it was
never merged into that orchestrator. Here each strategy is one more candidate
generator, and the engine's own critic (score.py, the pixel re-rank and the
hard checks) picks among all of them and the archetypes:

  pin      XD-style edge pins (US11320975B2: an offset under 15% of the canvas
           pins that edge, nearly equal offsets under 25% pin both sides, copy
           spanning 66% pins both sides, within 20% pins the near side), or a
           pin to the neighbour an element hugs (subhead under headline). Each
           element is placed by its pins on the target, then grows into free
           space without overlap, keeping its proportions.
  springs  springs-and-struts: stiffness from size, importance, distance to the
           boundary and centrality. The artboard is resized in 10% steps from
           source to target; at every step an area-preserving force pulls each
           element towards its share of the canvas, struts hold elements near an
           edge to it, and collisions push the softer element away (or shrink
           it). Source alignments are re-applied at the end.
  propfit  proportional fit per block: each top-level XY-cut block gets its own
           uniform scale (the cell it maps to, not the whole canvas), anchored
           to its nearest edge or quadrant, then grows into free space.
  flow     flow flip on the XY-cut tree: a node's direction is transposed
           (column <-> row) at the top one or two levels, or a column is wrapped
           into two columns (a row into two rows), and child sizes are re-solved
           by the flexbox engine. A column becomes a row for wide targets, a row
           becomes a column for tall ones; this generalises verticalize/reflect.

Everything here is wired in by two marked hooks (compose.generate and
score._prior) and switched off by ENABLED = False.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .analyze import Design, Item
from .geom import Box, union_all
from .items import Ctx, Real, realize
from .layout import Flow, HStack, Leaf, Node, VStack
from .structure import dominant_align, row_align, text_edges, xy_tree

ENABLED = os.environ.get("CHAMELEON_STRATEGIES", "1") != "0"  # A/B switch for the dev tools
GENERATORS = ("pin", "springs", "propfit", "flow")  # drop a name to switch one generator off
STEPS = 10  # springs-and-struts: the artboard moves in 10% increments

Placement = Tuple[Real, float, float]
Parts = Tuple[List[Placement], float, float]  # placements relative to the unit's top-left, w, h


# ----------------------------------------------------------------------------
# hooks


def variants(d: Design, ctx: Ctx, force_all: bool = False):
    """(name, planner, kwargs) for compose.generate, like its own archetypes."""
    if not ENABLED:
        return
    if "pin" in GENERATORS:
        yield "pin", plan_pin, {}
    if "springs" in GENERATORS:
        yield "springs", plan_springs, {}
    if "propfit" in GENERATORS:
        yield "propfit", plan_propfit, {}
    if "flow" in GENERATORS:
        for op in _flow_ops(d, ctx):
            yield "flow-" + "-".join(str(x) for x in op), plan_flow, dict(op=op)


def prior(name: str, r: float, r0: float) -> Optional[float]:
    """Format prior for the new candidates (score._prior asks first).

    pin / springs / propfit keep the source arrangement, which is what a
    designer does when the shape barely changes, and less so the further it
    moves. A flow candidate ends as a column or a row (its name says which) and
    competes on the same prior as the archetype of that shape."""
    from .score import _prior

    if name.startswith(("pin", "springs", "propfit")):
        q = abs(math.log(r / r0))
        return 14.0 * max(0.0, 1.0 - q / math.log(2.0)) - 4.0
    if name.startswith("flow"):
        kind = name.rsplit(":", 1)[-1]
        if kind == "h":
            base = max(_prior("split", r, r0), _prior("row", r, r0))
        else:
            base = _prior("stack", r, r0)
        return base - (1.0 if "keep" in name else 2.0)
    return None


# ----------------------------------------------------------------------------
# units: one item, or a block of items that keeps its internal arrangement


@dataclass
class Unit:
    items: List[Item]
    src: Box

    @property
    def importance(self) -> float:
        return max(i.importance for i in self.items)

    @property
    def art(self) -> bool:
        return all(i.kind in ("image", "composite") for i in self.items)

    @property
    def text(self) -> bool:
        return all(i.kind == "text" for i in self.items)


def _unit(items: Sequence[Item]) -> Unit:
    return Unit(list(items), union_all(i.src_box for i in items))


def _h_anchor(it: Item) -> float:
    """Which point of an item keeps its place inside a block: copy keeps its
    alignment edge, everything else its centre."""
    if it.kind == "text":
        return {"left": 0.0, "center": 0.5, "right": 1.0}.get(it.els[0].get("text_align", "left"), 0.0)
    return 0.5


def _src_overlap(a: Sequence[Item], b: Sequence[Item]) -> bool:
    """Deliberate overlap in the source (copy on art), not a sliver between
    neighbours whose loose boxes touch."""
    return any(i.src_box.inter(j.src_box) > 0.1 * min(i.src_box.area, j.src_box.area) for i in a for j in b)


def realize_unit(u: Unit, s: float, ctx: Ctx, max_w: float, max_h: float) -> Optional[Parts]:
    """A unit at scale s. A block keeps its arrangement; copy that grew past the
    block's scale (legibility floors) pushes later members down instead of
    running into them."""
    if len(u.items) == 1:
        r = realize(u.items[0], s, ctx, max_w=max_w, max_h=max_h)
        return None if r is None else ([(r, 0.0, 0.0)], r.w, r.h)
    parts: List[List] = []
    for it in sorted(u.items, key=lambda i: (i.src_box.y, i.src_box.x)):
        r = realize(it, s, ctx, max_w=max_w, max_h=max_h)
        if r is None:
            return None
        b = it.src_box
        f = _h_anchor(it)
        parts.append([r, (b.x + f * b.w - u.src.x) * s - f * r.w, (b.y - u.src.y) * s])
    gap = 0.3 * ctx.base_fs
    for i, p in enumerate(parts):
        for _ in range(i + 1):
            pb = Box(p[1], p[2], p[0].w, p[0].h)
            hit = [q for q in parts[:i] if pb.inter(Box(q[1], q[2], q[0].w, q[0].h)) > 0
                   and not _src_overlap([p[0].item], [q[0].item])]
            if not hit:
                break
            p[2] = max(q[2] + q[0].h for q in hit) + gap
    x0 = min(p[1] for p in parts)
    y0 = min(p[2] for p in parts)
    w = max(p[1] + p[0].w for p in parts) - x0
    h = max(p[2] + p[0].h for p in parts) - y0
    if w > max_w + 0.5 or h > max_h + 0.5:
        return None
    return [(r, x - x0, y - y0) for r, x, y in parts], w, h


# ----------------------------------------------------------------------------
# anchored layout + growth into free space (pin and propfit)
#
# An axis spec is ("pt", coordinate, f): the point at fraction f of the unit's
# extent sits at that coordinate; or ("nb", j, ef, gap, f): it sits at fraction
# ef of unit j's box plus a gap that scales with the two units.


Spec = Tuple


class Anchored:
    def __init__(self, units: List[Unit], specs: List[Tuple[Spec, Spec]], ctx: Ctx, zone: Box,
                 d: Optional[Design] = None):
        self.units, self.specs, self.ctx, self.zone = units, specs, ctx, zone
        self.cache: Dict[Tuple[int, float], Optional[Parts]] = {}
        n = len(units)
        self.src_ov = [[_src_overlap(units[i].items, units[j].items) for j in range(n)] for i in range(n)]
        self.pad = 0.4 * ctx.base_fs
        self.panels = _panels(d, units) if d is not None else []
        # the axis along which unit i comes before unit j in the source (1: above, 0: left of)
        self.before: List[List[Optional[int]]] = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                a, b = units[i].src, units[j].src
                if i == j or self.src_ov[i][j]:
                    continue
                # a sliver of overlap is still an order (as in the XY-cut)
                gy = (b.y - a.b + 0.15 * min(a.h, b.h)) / max(a.h, b.h, 1.0)
                gx = (b.x - a.r + 0.15 * min(a.w, b.w)) / max(a.w, b.w, 1.0)
                if max(gx, gy) > 0:
                    self.before[i][j] = 1 if gy >= gx else 0

    def parts(self, i: int, s: float) -> Optional[Parts]:
        key = (i, round(s, 5))
        if key not in self.cache:
            self.cache[key] = realize_unit(self.units[i], s, self.ctx, self.zone.w, self.zone.h)
        return self.cache[key]

    def layout(self, scales: Sequence[float]):
        n = len(self.units)
        boxes: List[Optional[Box]] = [None] * n
        reals: List[Optional[List[Placement]]] = [None] * n
        state = [0] * n
        z = self.zone
        floor = 1.2 * self.pad

        def go(i: int) -> bool:
            if state[i] == 2:
                return True
            if state[i] == 1:
                return False
            state[i] = 1
            got = self.parts(i, scales[i])
            if got is None:
                return False
            ps, w, h = got
            xy = []
            for axis, spec in enumerate(self.specs[i]):
                size = w if axis == 0 else h
                if spec[0] == "pt":
                    a, f = spec[1], spec[2]
                else:
                    _, j, ef, gap0, f = spec
                    if not go(j):
                        return False
                    bj = boxes[j]
                    a = (bj.x + ef * bj.w) if axis == 0 else (bj.y + ef * bj.h)
                    if gap0 is not None:  # beside / under the neighbour, not aligned with it
                        a += max(floor, gap0 * 0.5 * (scales[i] + scales[j]))
                v = a - f * size
                lo_, hi_ = (z.x, z.r - w) if axis == 0 else (z.y, z.b - h)
                xy.append(lo_ if hi_ < lo_ else min(max(v, lo_), hi_))
            boxes[i] = Box(xy[0], xy[1], w, h)
            reals[i] = ps
            state[i] = 2
            return True

        for i in range(n):
            if not go(i):
                return None
        self._separate(boxes)
        return boxes, reals

    def _separate(self, boxes: List[Box]) -> None:
        """Copy that grew to its legibility floor (or art that grew) can reach
        into a neighbour. Push such pairs apart along the axis they were apart
        on in the source, the later one first, the earlier one if the later
        has no room left in the safe area."""
        z, pad, n = self.zone, self.pad, len(boxes)
        for _ in range(3 * n):
            moved = False
            for i in range(n):
                for j in range(n):
                    ax = self.before[i][j]
                    if ax is None:
                        continue
                    a, b = boxes[i], boxes[j]
                    if a.expanded(pad / 2.0).inter(b.expanded(pad / 2.0)) <= 0:
                        continue
                    if ax == 1:
                        need = a.b + pad - b.y
                        mv = max(0.0, min(need, z.b - b.b))
                        b.y += mv
                        a.y -= max(0.0, min(need - mv, a.y - z.y))
                    else:
                        need = a.r + pad - b.x
                        mv = max(0.0, min(need, z.r - b.r))
                        b.x += mv
                        a.x -= max(0.0, min(need - mv, a.x - z.x))
                    moved = True
            if not moved:
                return

    def ok(self, scales: Sequence[float]):
        got = self.layout(scales)
        if got is None:
            return None
        boxes, _ = got
        z = self.zone
        for b in boxes:
            if b.w > z.w + 0.5 or b.h > z.h + 0.5:
                return None
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                pad = 0.0 if self.src_ov[i][j] else self.pad
                if boxes[i].expanded(pad / 2.0).inter(boxes[j].expanded(pad / 2.0)) > 0:
                    return None
        for members, pad, edges, exempt in self.panels:  # nothing grows into a panel's reach
            region = _panel_region([boxes[i] for i in members], [scales[i] for i in members], pad, edges,
                                   self.ctx.W, self.ctx.H)
            if any(boxes[j].inter(region) > 0 for j in range(len(boxes)) if j not in members and j not in exempt):
                return None
        return got


def _panel_region(boxes: Sequence[Box], scales: Sequence[float], pad, edges, W: float, H: float) -> Box:
    """Where emit()'s _backdrop will redraw a panel: around its members with the
    source padding, out to the canvas edges it reached."""
    u = union_all(boxes)
    sc = sorted(scales)[len(scales) // 2]
    pl, pt, pr, pb = (p * sc for p in pad)
    x0, y0, x1, y1 = u.x - pl, u.y - pt, u.r + pr, u.b + pb
    el, et, er, eb = edges
    x0, y0 = (0.0 if el else x0), (0.0 if et else y0)
    x1, y1 = (W if er else x1), (H if eb else y1)
    return Box(x0, y0, x1 - x0, y1 - y0)


def _panels(d: Design, units: Sequence[Unit]):
    """Panels whose members are whole units: (member unit indices, padding,
    edges, units already inside its reach in the source)."""
    out = []
    owner = {id(it): i for i, u in enumerate(units) for it in u.items}
    for bd in d.backdrops:
        idx = {owner[id(m)] for m in bd.members if id(m) in owner}
        if not idx or any(id(it) not in {id(m) for m in bd.members} for i in idx for it in units[i].items):
            continue
        src = _panel_region([units[i].src for i in idx], [1.0], bd.pad, bd.edges, d.W, d.H)
        exempt = {j for j, u in enumerate(units) if j not in idx and u.src.inter(src) > 0}
        out.append((idx, bd.pad, bd.edges, exempt))
    return out


def _largest(ok: Callable[[float], bool], lo: float, hi: float, iters: int = 9) -> float:
    """Largest value in [lo, hi] that is ok (lo is known to be ok)."""
    if hi <= lo * 1.002:
        return lo
    if ok(hi):
        return hi
    a, b = lo, hi
    for _ in range(iters):
        mid = math.sqrt(a * b)
        if ok(mid):
            a = mid
        else:
            b = mid
        if b / a < 1.01:
            break
    return a


def settle_and_grow(lay: Anchored, scales: List[float], caps: List[float], lo: float) -> Optional[List[float]]:
    """Shrink everything together until it fits, then grow: first all together
    (keeps the hierarchy), then each unit on its own, most important first, into
    whatever free space is left around it."""
    n = len(scales)
    if lay.ok(scales) is None:
        g_lo = max(lo / max(scales), 0.05)
        if lay.ok([s * g_lo for s in scales]) is None:
            return None
        g = _largest(lambda g: lay.ok([s * g for s in scales]) is not None, g_lo, 1.0)
        scales = [s * g for s in scales]
        return scales  # it had to shrink: no room to grow
    g_hi = min(c / s for c, s in zip(caps, scales))
    if g_hi > 1.002:
        g = _largest(lambda g: lay.ok([s * g for s in scales]) is not None, 1.0, g_hi)
        scales = [s * g for s in scales]
    for i in sorted(range(n), key=lambda i: -lay.units[i].importance):
        if caps[i] <= scales[i] * 1.002:
            continue

        def ok_i(v, i=i):
            t = list(scales)
            t[i] = v
            return lay.ok(t) is not None

        scales[i] = _largest(ok_i, scales[i], caps[i], iters=7)
    return scales


def _plan_from(lay: Anchored, scales: List[float], name: str, hidden, zone: Box, caps: List[float]):
    from .compose import Plan

    got = lay.ok(scales)
    if got is None:
        return None
    boxes, reals = got
    placed: List[Placement] = []
    for b, ps in zip(boxes, reals):
        placed += [(r, b.x + px, b.y + py) for r, px, py in ps]
    ss = sorted(r.scale for r, _, _ in placed)
    return Plan(name=name, placed=placed, hidden=list(hidden), s=ss[len(ss) // 2] if ss else 1.0, zone=zone,
                capped=all(s >= c * 0.999 for s, c in zip(scales, caps)))


def _caps(units: Sequence[Unit], hi: float) -> List[float]:
    """Art may grow past the type scale into free space; a logo or a corner
    item stays a signature, not a feature."""
    out = []
    for u in units:
        small = any(i.role == "logo" or i.corner for i in u.items)
        out.append(hi * (1.6 if u.art and not small else 1.0))
    return out


# ----------------------------------------------------------------------------
# 1. pin


def xd_pins(b: Box, W0: float, H0: float, text: bool) -> Tuple[List[bool], List[bool]]:
    """Edge pins from offsets to the parent, after Adobe XD's responsive resize
    (US11320975B2, rules 1, 3, 4 and 6). Returns ([left, right], [top, bottom])."""
    l, r = b.x / W0, (W0 - b.r) / W0
    t, bo = b.y / H0, (H0 - b.b) / H0
    hx = [l < 0.15, r < 0.15]
    vy = [t < 0.15, bo < 0.15]
    if abs(l - r) <= 0.03 and max(l, r) < 0.25:
        hx = [True, True]
    if abs(t - bo) <= 0.03 and max(t, bo) < 0.25:
        vy = [True, True]
    if text and b.w >= 0.66 * W0:
        hx = [True, True]
    if text and b.h >= 0.66 * H0:
        vy = [True, True]
    if not any(hx) and min(l, r) < 0.2:
        hx = [l <= r, r < l]
    return hx, vy


def _canvas_spec(pins: List[bool], lo_off: float, hi_off: float, c_frac: float, L: float, k: float) -> Spec:
    if pins[0] and pins[1]:
        return ("pt", (lo_off * k + L - hi_off * k) / 2.0, 0.5)
    if pins[0]:
        return ("pt", lo_off * k, 0.0)
    if pins[1]:
        return ("pt", L - hi_off * k, 1.0)
    return ("pt", c_frac * L, 0.5)


def _aligned(a: Box, b: Box, W0: float) -> Optional[float]:
    """The edge two boxes share in x (0 left, 0.5 centre, 1 right), if any."""
    tol = 0.015 * W0
    best = None
    for f in (0.0, 0.5, 1.0):
        dv = abs((a.x + f * a.w) - (b.x + f * b.w))
        if dv <= tol and (best is None or dv < best[1]):
            best = (f, dv)
    return None if best is None else best[0]


def plan_pin(d: Design, ctx: Ctx, hidden: List[Item], name: str = "pin", insets=(0, 0, 0, 0)):
    from .compose import _visible, _zone, scale_bounds

    vis = _visible(d, hidden)
    if not vis:
        return None
    zone, _, _ = _zone(d, ctx, insets)
    if zone.w < 20 or zone.h < 12:
        return None
    lo, hi = scale_bounds(d, ctx, "split")
    W0, H0 = d.W, d.H
    k = min(max(min(ctx.W / W0, ctx.H / H0), lo), hi)
    units = [_unit([it]) for it in vis]
    order = sorted(range(len(units)), key=lambda i: (units[i].src.y, units[i].src.x))
    rank = {i: n for n, i in enumerate(order)}
    specs: List[Tuple[Spec, Spec]] = []
    for i, u in enumerate(units):
        b = u.src
        hx, vy = xd_pins(b, W0, H0, u.text)
        xs = _canvas_spec(hx, b.x, W0 - b.r, b.cx / W0, ctx.W, k)
        ys = _canvas_spec(vy, b.y, H0 - b.b, b.cy / H0, ctx.H, k)
        if not any(vy):  # pinned to the neighbour it hangs under
            above = [j for j in range(len(units)) if rank[j] < rank[i] and units[j].src.b <= b.y + 0.1 * b.h
                     and min(b.r, units[j].src.r) - max(b.x, units[j].src.x) >= 0.5 * min(b.w, units[j].src.w)]
            if above:
                j = max(above, key=lambda j: units[j].src.b)
                gap = b.y - units[j].src.b
                if gap <= max(0.6 * min(b.h, units[j].src.h), 0.04 * H0):
                    ys = ("nb", j, 1.0, max(0.0, gap), 0.0)
                    f = _aligned(b, units[j].src, W0)
                    if not any(hx) and f is not None:
                        xs = ("nb", j, f, None, f)
        if not any(hx) and xs[0] == "pt":  # ... or beside
            left = [j for j in range(len(units)) if rank[j] < rank[i] and units[j].src.r <= b.x + 0.1 * b.w
                    and min(b.b, units[j].src.b) - max(b.y, units[j].src.y) >= 0.5 * min(b.h, units[j].src.h)]
            if left:
                j = max(left, key=lambda j: units[j].src.r)
                gap = b.x - units[j].src.r
                if gap <= max(0.6 * min(b.w, units[j].src.w), 0.04 * W0):
                    xs = ("nb", j, 1.0, max(0.0, gap), 0.0)
        specs.append((xs, ys))
    lay = Anchored(units, specs, ctx, zone, d)
    caps = _caps(units, hi)
    scales = settle_and_grow(lay, [k] * len(units), caps, lo)
    if scales is None:
        return None
    return _plan_from(lay, scales, name, hidden, zone, caps)


# ----------------------------------------------------------------------------
# 2. proportional fit per block


def _blocks(d: Design, vis: Sequence[Item]) -> List[List[Item]]:
    body = [it for it in vis if not it.corner]
    out = [[it] for it in vis if it.corner]
    if body:
        t = xy_tree(body)
        if isinstance(t, Item):
            out.append([t])
        else:
            out += [_leaves(c) for c in t[1]]
    return out


def _leaves(t) -> List[Item]:
    if isinstance(t, Item):
        return [t]
    return [i for c in t[1] for i in _leaves(c)]


def _quadrant(lo_off: float, hi_off: float, c: float) -> float:
    """0 = anchored to the low edge, 1 = the high edge, 0.5 = centred."""
    if min(lo_off, hi_off) < 0.15:
        return 0.0 if lo_off <= hi_off else 1.0
    if abs(c - 0.5) <= 0.1:
        return 0.5
    return 0.0 if c < 0.5 else 1.0


def plan_propfit(d: Design, ctx: Ctx, hidden: List[Item], name: str = "propfit", insets=(0, 0, 0, 0)):
    from .compose import _visible, _zone, scale_bounds

    vis = _visible(d, hidden)
    if not vis:
        return None
    zone, _, _ = _zone(d, ctx, insets)
    if zone.w < 20 or zone.h < 12:
        return None
    lo, hi = scale_bounds(d, ctx, "split")
    W0, H0, W, H = d.W, d.H, ctx.W, ctx.H
    units = [_unit(b) for b in _blocks(d, vis)]
    specs, scales = [], []
    for u in units:
        b = u.src
        cell = Box(b.x / W0 * W, b.y / H0 * H, b.w / W0 * W, b.h / H0 * H)
        s = min(cell.w / max(1.0, b.w), cell.h / max(1.0, b.h))
        scales.append(min(max(s, lo), hi))
        fx = _quadrant(b.x / W0, (W0 - b.r) / W0, b.cx / W0)
        fy = _quadrant(b.y / H0, (H0 - b.b) / H0, b.cy / H0)
        specs.append((("pt", cell.x + fx * cell.w, fx), ("pt", cell.y + fy * cell.h, fy)))
    lay = Anchored(units, specs, ctx, zone, d)
    caps = _caps(units, hi)
    caps = [max(c, s) for c, s in zip(caps, scales)]
    got = settle_and_grow(lay, scales, caps, lo)
    if got is None:
        return None
    return _plan_from(lay, got, name, hidden, zone, caps)


# ----------------------------------------------------------------------------
# 3. springs and struts


def plan_springs(d: Design, ctx: Ctx, hidden: List[Item], name: str = "springs", insets=(0, 0, 0, 0)):
    from .compose import Plan, _visible, _zone, scale_bounds

    vis = _visible(d, hidden)
    if not vis:
        return None
    zone, _, _ = _zone(d, ctx, insets)
    if zone.w < 20 or zone.h < 12:
        return None
    lo, hi = scale_bounds(d, ctx, "split")
    W0, H0, W, H = d.W, d.H, ctx.W, ctx.H
    n = len(vis)
    src = [it.src_box for it in vis]
    zf = (zone.x / W, zone.y / H, zone.r / W, zone.b / H)  # the safe area, as canvas fractions
    # stiffness: big, important and central elements keep their share of the
    # canvas (size spring); elements near an edge keep their distance to it (strut)
    kap, strut, side, soft = [], [], [], []
    for it, b in zip(vis, src):
        size = min(1.0, math.sqrt(b.area / (W0 * H0)) / 0.5)
        cen = 1.0 - 2.0 * max(abs(b.cx / W0 - 0.5), abs(b.cy / H0 - 0.5))
        k_ = min(1.0, max(0.0, 0.4 * it.importance / 100.0 + 0.35 * size + 0.25 * cen))
        kap.append(k_)
        soft.append(1.1 - k_)
        ex, ey = min(b.x, W0 - b.r) / W0, min(b.y, H0 - b.b) / H0
        strut.append((max(0.0, 1.0 - ex / 0.15), max(0.0, 1.0 - ey / 0.15)))
        side.append((b.x <= W0 - b.r, b.y <= H0 - b.b))
    ov = [[src[i].inter(src[j]) > 0 for j in range(n)] for i in range(n)]
    caps = _caps([_unit([it]) for it in vis], hi)

    def want(i, Wj, Hj):
        fit = min(Wj / W0, Hj / H0)
        area = math.sqrt(Wj * Hj / (W0 * H0))
        return fit ** (1.0 - kap[i]) * area ** kap[i]

    def desired_c(i, w, h, Wj, Hj):
        b = src[i]
        fit = min(Wj / W0, Hj / H0)
        out = []
        for axis in (0, 1):
            L, L0 = (Wj, W0) if axis == 0 else (Hj, H0)
            lo_e, hi_e, c, sz = (b.x, W0 - b.r, b.cx, w) if axis == 0 else (b.y, H0 - b.b, b.cy, h)
            spring = c / L0 * L
            strut_c = (lo_e * fit + sz / 2.0) if side[i][axis] else (L - hi_e * fit - sz / 2.0)
            a = strut[i][axis]
            out.append(a * strut_c + (1.0 - a) * spring)
        return out

    def est(i, s, cj):  # cheap size at intermediate steps: text never below its floor
        it, b = vis[i], src[i]
        g = s
        if it.kind == "text" and it.fs0 > 0:
            g = max(s, cj.floor(it.role) / it.fs0)
        return b.w * g, b.h * g

    s = [1.0] * n
    cx = [b.cx for b in src]
    cy = [b.cy for b in src]
    Wp, Hp = W0, H0
    for j in range(1, STEPS + 1):
        t = j / STEPS
        Wj, Hj = W0 * (W / W0) ** t, H0 * (H / H0) ** t
        cj = Ctx(W=Wj, H=Hj, W0=W0, H0=H0)
        zj = Box(zf[0] * Wj, zf[1] * Hj, (zf[2] - zf[0]) * Wj, (zf[3] - zf[1]) * Hj)
        sizes = []
        for i in range(n):
            follow = s[i] * want(i, Wj, Hj) / want(i, Wp, Hp)
            s[i] = math.sqrt(follow * want(i, Wj, Hj))  # area-preserving pull, half way per step
            if j == STEPS:
                s[i] = min(max(s[i], lo), caps[i])
            sizes.append(est(i, s[i], cj))
        for i in range(n):
            dx, dy = desired_c(i, sizes[i][0], sizes[i][1], Wj, Hj)
            cx[i] = 0.5 * (cx[i] / Wp * Wj + dx)
            cy[i] = 0.5 * (cy[i] / Hp * Hj + dy)
        # intermediate artboards are estimates: what does not settle here is
        # settled on the real target below
        _resolve(cx, cy, s, sizes, soft, ov, zj, 0.4 * cj.base_fs, lambda i, v: est(i, v, cj), lo)
        Wp, Hp = Wj, Hj

    # the target: real sizes, measured text
    def real(i, v):
        return realize(vis[i], v, ctx, max_w=zone.w, max_h=zone.h)

    reals: List[Optional[Real]] = [real(i, s[i]) for i in range(n)]
    if any(r is None for r in reals):
        return None
    sizes = [(r.w, r.h) for r in reals]

    def resize_real(i, v):
        r = real(i, v)
        if r is None:
            return None
        reals[i] = r
        return r.w, r.h

    if not _resolve(cx, cy, s, sizes, soft, ov, zone, 0.4 * ctx.base_fs, resize_real, lo):
        return None
    _snap_alignments(vis, src, cx, cy, sizes, zone, ov, 0.4 * ctx.base_fs, W0, H0)
    placed = [(reals[i], cx[i] - sizes[i][0] / 2.0, cy[i] - sizes[i][1] / 2.0) for i in range(n)]
    ss = sorted(r.scale for r in reals)
    return Plan(name=name, placed=placed, hidden=list(hidden), s=ss[len(ss) // 2], zone=zone)


def _overlaps(cx, cy, sizes, ov, pad) -> List[Tuple[int, int, float, float]]:
    out = []
    n = len(cx)
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.0 if ov[i][j] else pad
            dx = (sizes[i][0] + sizes[j][0]) / 2.0 + p - abs(cx[i] - cx[j])
            dy = (sizes[i][1] + sizes[j][1]) / 2.0 + p - abs(cy[i] - cy[j])
            if dx > 0.01 and dy > 0.01:
                out.append((i, j, dx, dy))
    return out


def _clamp(cx, cy, sizes, zone: Box) -> bool:
    fits = True
    for i in range(len(cx)):
        w, h = sizes[i]
        if w > zone.w + 0.5 or h > zone.h + 0.5:
            fits = False
        cx[i] = min(max(cx[i], zone.x + w / 2.0), max(zone.x + w / 2.0, zone.r - w / 2.0))
        cy[i] = min(max(cy[i], zone.y + h / 2.0), max(zone.y + h / 2.0, zone.b - h / 2.0))
    return fits


def _resolve(cx, cy, s, sizes, soft, ov, zone: Box, pad: float, resize, lo: float) -> bool:
    """Push overlapping elements apart (the softer one moves more), keep them in
    the safe area, and shrink the softer one when pushing is not enough."""
    for _ in range(40):
        for _ in range(25):
            hits = _overlaps(cx, cy, sizes, ov, pad)
            if not hits:
                break
            for i, j, dx, dy in hits:
                wi = soft[i] / (soft[i] + soft[j])
                if dx <= dy:
                    sg = 1.0 if (cx[i], i) >= (cx[j], j) else -1.0
                    cx[i] += sg * dx * wi
                    cx[j] -= sg * dx * (1.0 - wi)
                else:
                    sg = 1.0 if (cy[i], i) >= (cy[j], j) else -1.0
                    cy[i] += sg * dy * wi
                    cy[j] -= sg * dy * (1.0 - wi)
            _clamp(cx, cy, sizes, zone)
        fits = _clamp(cx, cy, sizes, zone)
        hits = _overlaps(cx, cy, sizes, ov, pad)
        if not hits and fits:
            return True
        victims = set()
        for i, j, _, _ in hits:
            victims.add(i if soft[i] >= soft[j] else j)
        for i, (w, h) in enumerate(sizes):
            if w > zone.w + 0.5 or h > zone.h + 0.5:
                victims.add(i)
        shrunk = False
        for i in victims:
            v = s[i] * 0.93
            if v < lo:
                continue
            got = resize(i, v)
            if got is None:
                continue
            if got[0] * got[1] >= sizes[i][0] * sizes[i][1] * 0.999:  # floors: it cannot get smaller
                continue
            s[i], sizes[i] = v, got
            shrunk = True
        if not shrunk:
            return False
    return False


def _snap_alignments(vis, src, cx, cy, sizes, zone: Box, ov, pad: float, W0: float, H0: float) -> None:
    """Re-apply the source's alignments: elements that shared a left edge, a
    centre line or a right edge (or a top / middle / bottom, side by side) share
    it again, wherever that does not cause a collision."""
    n = len(vis)
    for axis in (0, 1):
        L0 = W0 if axis == 0 else H0
        done = set()
        for f in (0.0, 0.5, 1.0):
            def edge0(i):
                b = src[i]
                return (b.x + f * b.w) if axis == 0 else (b.y + f * b.h)

            for i in range(n):
                if (i, f) in done:
                    continue
                grp = [j for j in range(n) if abs(edge0(j) - edge0(i)) <= 0.012 * L0]
                if len(grp) < 2:
                    continue
                for j in grp:
                    done.add((j, f))
                c, sz = (cx, 0) if axis == 0 else (cy, 1)
                cur = [c[j] - (0.5 - f) * sizes[j][sz] for j in grp]  # the aligned edge now
                wts = [vis[j].importance for j in grp]
                target = sum(a * w for a, w in zip(cur, wts)) / sum(wts)
                old = [c[j] for j in grp]
                for j in grp:
                    c[j] = target + (0.5 - f) * sizes[j][sz]
                inside = all(
                    (zone.x - 0.5 <= cx[j] - sizes[j][0] / 2.0 and cx[j] + sizes[j][0] / 2.0 <= zone.r + 0.5) and
                    (zone.y - 0.5 <= cy[j] - sizes[j][1] / 2.0 and cy[j] + sizes[j][1] / 2.0 <= zone.b + 0.5)
                    for j in grp)
                if not inside or _overlaps(cx, cy, sizes, ov, pad):
                    for j, v in zip(grp, old):
                        c[j] = v


# ----------------------------------------------------------------------------
# 4. flow flip on the XY-cut tree


@dataclass
class FNode:
    kind: str  # "v" (a column) or "h" (a row)
    kids: list
    flipped: bool = False


def _fnode(t) -> object:
    if isinstance(t, Item):
        return t
    return FNode(t[0], [_fnode(c) for c in t[1]])


def _fleaves(t) -> List[Item]:
    if isinstance(t, Item):
        return [t]
    return [i for c in t.kids for i in _fleaves(c)]


def _fbox(t) -> Box:
    return union_all(i.src_box for i in _fleaves(t))


def _sig(t) -> str:
    if isinstance(t, Item):
        return t.key
    return t.kind + "(" + ",".join(_sig(c) for c in t.kids) + ")"


def _direction(d: Design, ctx: Ctx) -> Optional[str]:
    """Which way the target pulls: towards rows ("h") or columns ("v")."""
    q = (ctx.W / ctx.H) / (d.W / d.H)
    if q >= 1.25:
        return "h"
    if q <= 0.8:
        return "v"
    return None


def _flow_ops(d: Design, ctx: Ctx) -> List[tuple]:
    """The identity when the shape barely changes (keeping the structure under a
    big change is Adobe's "fit" failure). Flips only below strip proportions: a
    5:1 strip has no height for a nested column, and flattening everything into
    one line is the row archetype's job (the stack's on a 1:5 tower)."""
    r = ctx.W / ctx.H
    if _direction(d, ctx) is None:  # same shape: the source structure, re-solved
        return [("keep",)]
    if r >= 5.0 or r <= 0.2:
        return []
    return [("root",), ("both",), ("l2", 0), ("l2", 1), ("wrap", 0), ("wrap", 1)]


def _flip(n: FNode) -> None:
    n.kind = "h" if n.kind == "v" else "v"
    n.flipped = not n.flipped


def _apply_op(t: FNode, op: tuple, want: Optional[str]):
    """The tree after one flow operation, or None when it does not apply."""
    if op[0] == "keep":
        return t
    if want is None:
        return None
    src_kind = "v" if want == "h" else "h"  # flipping this kind moves towards the target
    inner = [c for c in t.kids if isinstance(c, FNode)]
    if op[0] == "root":
        if t.kind != src_kind:
            return None
        _flip(t)
        return t
    if op[0] == "both":
        if t.kind != src_kind or not inner:
            return None
        _flip(t)
        for c in inner:
            _flip(c)
        return t
    if op[0] == "l2":
        cand = sorted([c for c in inner if c.kind == src_kind], key=lambda c: -_fbox(c).area)
        if op[1] >= len(cand):
            return None
        _flip(cand[op[1]])
        return t
    if op[0] == "wrap":  # a column wrapped into two columns side by side (or a row into two rows)
        if t.kind != src_kind or len(t.kids) < 3:
            return None
        areas = [_fbox(c).area for c in t.kids]
        tot = sum(areas)
        cuts = sorted(range(1, len(t.kids)), key=lambda c: abs(sum(areas[:c]) - tot / 2.0))
        if op[1] >= len(cuts):
            return None
        c = cuts[op[1]]
        a, b = t.kids[:c], t.kids[c:]
        ga = a[0] if len(a) == 1 else FNode(t.kind, a)
        gb = b[0] if len(b) == 1 else FNode(t.kind, b)
        return FNode(want, [ga, gb], flipped=True)
    return None


def _build(t, d: Design, ctx: Ctx, parent: Optional[str], dom: str, tl: float, tr: float):
    """FNode -> layout Node, with gaps and alignment from the source where the
    direction is the source's own."""
    if isinstance(t, Item):
        if t.role == "hero":
            if parent == "h":
                return Leaf(t, fill=True, max_w_frac=0.5, cap_rel=2.2, min_rel=0.5)
            return Leaf(t, fill=True, cap_rel=1.9, min_rel=0.45)
        return Leaf(t)
    kids = [_build(c, d, ctx, t.kind, dom, tl, tr) for c in t.kids]
    boxes = [_fbox(c) for c in t.kids]
    if t.kind == "v":
        gaps: List[Optional[float]] = []
        for a, b in zip(boxes, boxes[1:]):
            if t.flipped:
                gaps.append(None)
            else:
                gaps.append(max(b.y - a.b, 0.18 * min(a.h, b.h)))
        if t.flipped:
            al = {"center": "center", "left": "start", "right": "end"}[dom]
            aligns = [al] * len(kids)
        else:
            aligns = [row_align(_fleaves(c), d.W, dom, tl, tr) for c in t.kids]
        return VStack(kids, gaps=gaps, align=aligns)
    leaves = all(isinstance(c, Item) for c in t.kids)
    if leaves and not t.flipped and all(c.kind != "text" for c in t.kids):
        gs = [max(0.0, b.x - a.r) for a, b in zip(boxes, boxes[1:])]
        return Flow(kids, gap=sum(gs) / len(gs) if gs else None)
    gaps = [None if t.flipped else max(0.0, b.x - a.r) for a, b in zip(boxes, boxes[1:])]
    return HStack(kids, gaps=gaps)


def _solve_full(root: Node, ctx: Ctx, zone: Box, lo: float, hi: float):
    """compose.solve at full legibility only: a flip is an option, not a rescue
    (relaxed type is pruned whenever any candidate keeps full legibility)."""
    from .compose import effective_zone
    from .layout import search_scale

    def fits(s):
        w, h = zone.w, zone.h
        for _ in range(3):
            laid = root.measure(s, ctx, w, h, 1.0)
            if laid is None:
                return None
            ez = effective_zone(laid, ctx, zone)
            if laid.w <= ez.w + 0.5 and laid.h <= ez.h + 0.5:
                laid.zone = ez  # type: ignore[attr-defined]
                return laid
            w, h = min(w, ez.w), min(h, ez.h)
        return None

    s, laid = search_scale(fits, lo, hi)
    return s, laid, 1.0


def plan_flow(d: Design, ctx: Ctx, hidden: List[Item], name: str = "flow", insets=(0, 0, 0, 0),
              op: tuple = ("keep",)):
    from .compose import Plan, _settle_floaters, _visible, _zone, place_column, place_row, scale_bounds

    vis = _visible(d, hidden)
    floaters = [it for it in vis if it.corner]
    body = [it for it in vis if not it.corner]
    if len(body) < 2:
        return None
    t0 = xy_tree(body)
    if isinstance(t0, Item):
        return None
    tree = _apply_op(_fnode(t0), op, _direction(d, ctx))
    if tree is None or isinstance(tree, Item):
        return None
    # one candidate per distinct structure and hide-set
    seen = d.__dict__.setdefault("_flow_seen", {})
    key = (frozenset(id(h) for h in hidden), _sig(tree))
    if seen.setdefault(key, name) != name:
        return None
    dom = dominant_align(d.items)
    tl, tr = text_edges(d.items)
    root = _build(tree, d, ctx, None, dom, tl, tr)
    kind = tree.kind
    lo, hi = scale_bounds(d, ctx, "split" if kind == "h" else "stack")
    zone, mx, my = _zone(d, ctx, insets)
    if zone.w < 20 or zone.h < 12:
        return None
    boxes = [_fbox(c) for c in tree.kids]
    weights = [3.0 if any(i.role == "headline" for i in _fleaves(c)) else
               (1.5 if any(i.role == "hero" for i in _fleaves(c)) else 1.0) for c in tree.kids]
    for _ in range(4):
        s, laid, relax = _solve_full(root, ctx, zone, lo, hi)
        if laid is None:
            return None
        if isinstance(root, VStack):
            if tree.flipped:  # no source positions along the new axis: keep the column together
                boxes = [Box(0, d.H / 2.0, 1, 1)] * len(tree.kids)
            placed = place_column(laid, boxes, root.align, d, ctx, laid.zone, weights)
        elif isinstance(root, HStack):
            placed = place_row(laid, ctx, laid.zone)
        else:
            placed = laid.place(laid.zone.cx - laid.w / 2.0, laid.zone.cy - laid.h / 2.0)
        fl, bad, clash, new_zone = _settle_floaters(placed, floaters, s, ctx, zone, mx, my, relax)
        if not clash:
            return Plan(name=f"{name}:{kind}", placed=placed, hidden=list(hidden) + bad, s=s, zone=zone,
                        floaters=fl, relax=relax, capped=s >= hi * 0.999)
        if new_zone.h < 0.45 * (ctx.H - 2 * my):
            return None
        zone = new_zone
    return None
