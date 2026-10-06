"""Candidate layouts ("plans") for a target canvas.

Each archetype is a different way a designer might rebuild the same design:

  stack  - one column in the source's reading order (two-column sources are
           read column by column), each row kept near its source-relative
           height. The natural choice for portrait/square targets.
  split  - hero image on one side, everything else stacked on the other.
           For landscape targets (16:9, 1.91:1, 3:1 ...).
  row    - a single line: logo | hero | copy | CTA. For banners and
           leaderboards.

For every archetype we also try hiding the least important items, a unit at
a time (legal line, watermark, ribbon, a set of badges...), because on a
728x90 a designer drops those rather than shrinking everything into
illegibility. Scoring (score.py) picks the winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .analyze import Design, Item
from .geom import Box, el_visual_box, union_all
from .items import Ctx, Real, realize
from .layout import Flow, HStack, Laid, Leaf, Node, VStack, chain_positions, search_scale
from .structure import dominant_align, row_align, rows_for, text_edges

Placement = Tuple[Real, float, float]

CORE_ROLES = ("headline", "cta", "logo", "hero")


@dataclass
class Plan:
    name: str
    placed: List[Placement]
    hidden: List[Item]
    s: float
    zone: Box
    floaters: List[Placement] = field(default_factory=list)
    relax: float = 1.0
    capped: bool = False  # reached its scale cap: hiding more cannot make anything bigger
    score: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)

    def footprints(self) -> List[Tuple[Item, Box]]:
        return [(r.item, Box(x, y, r.w, r.h)) for r, x, y in self.placed + self.floaters]


# ----------------------------------------------------------------------------
# helpers


def margins(d: Design, ctx: Ctx) -> Tuple[float, float]:
    """Safe-area margins, following the source's own breathing room."""
    body = [it for it in d.items if it.corner is None and it.role != "missing"]
    fr = 0.06
    if body:
        u = union_all(it.src_box for it in body)
        edge = min(u.x, u.y, d.W - u.r, d.H - u.b)
        fr = edge / max(1.0, min(d.W, d.H))
    fr = min(0.075, max(0.045, fr))
    short = min(ctx.W, ctx.H)
    floor = min(6.0, max(3.0, 0.08 * short))  # tiny banners cannot spare 6px a side
    my = max(floor, fr * short)
    mx = max(floor, fr * short, 0.012 * ctx.W)
    return mx, my


def scale_bounds(d: Design, ctx: Ctx, kind: str = "stack") -> Tuple[float, float]:
    """Search range for the global scale.

    Designers re-making a template at another size scale type by about
    sqrt(target area / source area), and ~1.4x more on strips 6:1 and wider
    (measured on 1,630 Crello multi-size families). Reflowed layouts may also
    bring a short headline up to the display size typical for the format
    (~0.28 H on strips, ~0.16 H on wide banners, ~0.13 H landscape), and a
    design moved to a much roomier canvas may grow towards a display headline,
    so it does not look like a small design lost in space."""
    import math

    rw, rh = ctx.W / d.W, ctx.H / d.H
    lo_r, hi_r = min(rw, rh), max(rw, rh)
    r = ctx.W / ctx.H
    cap = math.sqrt(rw * rh)
    if kind == "stack":
        cap = min(cap, lo_r ** 0.6 * hi_r ** 0.4)  # one column shares the short side
    if r >= 6.0 or r <= 1 / 6.0:
        cap *= 1.4
    head = next((i for i in d.items if i.role == "headline"), None)
    if head is not None and head.fs0 > 0:
        n = max(1, len(str(head.els[0].get("content", ""))))
        short = min(1.0, max(0.45, 28.0 / n))
        target = 0.075 * min(ctx.W, ctx.H) * short
        room = 1.0 + 0.5 * math.log(max(1.0, hi_r / max(lo_r, 1e-6)))
        cap = max(cap, min(target / head.fs0, cap * room))
        if kind in ("row", "split") and r >= 1.4:
            frac = 0.28 if r >= 5.0 else (0.16 if r >= 2.4 else 0.13)
            cap = max(cap, min(frac * ctx.H * short / head.fs0, 2.0 * cap))
    return 0.01, max(0.02, cap)


def _row_box(row: Sequence[Item]) -> Box:
    return union_all(i.src_box for i in row)


def _leaf(it: Item, hero_fill: bool, cap_rel: float = 1.5, min_rel: float = 0.5) -> Leaf:
    if it.role == "hero" and hero_fill:
        return Leaf(it, fill=True, cap_rel=cap_rel, min_rel=min_rel)
    return Leaf(it)


def _row_node(row: Sequence[Item], hero_fill: bool, **kw) -> Node:
    if len(row) == 1:
        return _leaf(row[0], hero_fill, **kw)
    gaps = [max(0.0, b.src_box.x - a.src_box.r) for a, b in zip(row, row[1:])]
    g = sum(gaps) / len(gaps) if gaps else None
    if all(i.kind != "text" for i in row):
        return Flow([_leaf(i, False) for i in row], gap=g)
    return HStack([_leaf(i, hero_fill, **kw) for i in row], gaps=gaps)


def column(d: Design, rows: List[List[Item]], hero_fill: bool = False, force_align: Optional[str] = None,
           gap_scale: float = 1.0, **kw) -> Tuple[VStack, List[Box]]:
    """A column of source rows. Gaps come from the source, except where something
    that is not in this column used to sit between two rows (its space is gone)."""
    nodes, gaps, aligns, boxes = [], [], [], []
    inside = {id(i) for r in rows for i in r}
    outside = [o for o in d.items if id(o) not in inside]
    dom = dominant_align(d.items)
    tl, tr = text_edges(d.items)
    raw: List[Optional[float]] = []
    for i, row in enumerate(rows):
        rb = _row_box(row)
        boxes.append(rb)
        if i:
            prev = boxes[i - 1]
            g = rb.y - prev.b
            between = any(prev.b - 1 <= o.src_box.cy <= rb.y + 1 for o in outside)
            # rows that were side by side in the source have no vertical gap to copy
            raw.append(None if between or g < -0.3 * min(prev.h, rb.h) else max(g, 0.18 * min(prev.h, rb.h)))
    known = sorted(g for g in raw if g is not None)
    typical = known[len(known) // 2] if known else None
    for i, row in enumerate(rows):
        nodes.append(_row_node(row, hero_fill, **kw))
        aligns.append(force_align or row_align(row, d.W, dom, tl, tr))
        if i:
            g = raw[i - 1]
            if g is None:
                prev, rb = boxes[i - 1], boxes[i]
                ref = min(prev.h, rb.h)
                g = typical if typical is not None else 0.5 * ref
                g = min(max(g, 0.3 * ref), 1.0 * ref)
            gaps.append(g * gap_scale)
    return VStack(nodes, gaps=gaps, align=aligns), boxes


def effective_zone(laid: Laid, ctx: Ctx, zone: Box) -> Box:
    """The zone, narrowed where painted overhang (transparent image margins)
    would otherwise run off the canvas."""
    l, t, r, b = laid.oh
    x0 = max(zone.x, l)
    y0 = max(zone.y, t)
    x1 = min(zone.r, ctx.W - r)
    y1 = min(zone.b, ctx.H - b)
    return Box(x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


def solve(root: Node, ctx: Ctx, zone: Box, lo: float, hi: float) -> Tuple[Optional[float], Optional[Laid], float]:
    def fits(s, relax):
        w, h = zone.w, zone.h
        laid = None
        for _ in range(3):
            laid = root.measure(s, ctx, w, h, relax)
            if laid is None:
                return None
            ez = effective_zone(laid, ctx, zone)
            if laid.w <= ez.w + 0.5 and laid.h <= ez.h + 0.5:
                laid.zone = ez  # type: ignore[attr-defined]
                return laid
            w, h = min(w, ez.w), min(h, ez.h)
        return None

    for relax in (1.0, 0.85, 0.72, 0.6):
        s, laid = search_scale(lambda s: fits(s, relax), lo, hi)
        if laid is not None:
            return s, laid, relax
    return None, None, 1.0


# ----------------------------------------------------------------------------
# placing a solved tree inside the zone


def place_column(laid: Laid, boxes: List[Box], aligns: Sequence[str], d: Design, ctx: Ctx, zone: Box,
                 weights: Sequence[float]) -> List[Placement]:
    """Vertical positions as close as possible to where each row sat in the source."""
    parts: List[Laid] = laid.parts  # type: ignore[attr-defined]
    gaps: List[float] = laid.gap_list  # type: ignore[attr-defined]
    sizes = [p.h for p in parts]
    targets = [b.cy / d.H * ctx.H - h / 2.0 for b, h in zip(boxes, sizes)]
    ys = chain_positions(targets, list(weights), sizes, gaps, zone.y, zone.b)
    out = []
    for p, y, a in zip(parts, ys, aligns):
        x = zone.x if a == "start" else (zone.r - p.w if a == "end" else zone.cx - p.w / 2.0)
        out += p.place(x, y)
    return out


def place_row(laid: Laid, ctx: Ctx, zone: Box, grow: float = 2.0) -> List[Placement]:
    """Centre a row in the zone, letting the gaps breathe before the margins do."""
    parts: List[Laid] = laid.parts  # type: ignore[attr-defined]
    gaps: List[float] = laid.gap_list  # type: ignore[attr-defined]
    extra = max(0.0, zone.w - laid.w)
    n = len(gaps)
    add = min(extra / (n + 1.0), max(gaps) * grow) if n else 0.0
    x = zone.x + (zone.w - (laid.w + add * n)) / 2.0
    y = zone.y + (zone.h - laid.h) / 2.0
    out = []
    for i, p in enumerate(parts):
        out += p.place(x, y + (laid.h - p.h) / 2.0)
        x += p.w + (gaps[i] + add if i < n else 0.0)
    return out


def _place_split(laid: Laid, ctx: Ctx, zone: Box) -> List[Placement]:
    parts: List[Laid] = laid.parts  # type: ignore[attr-defined]
    gaps: List[float] = laid.gap_list  # type: ignore[attr-defined]
    extra = max(0.0, zone.w - laid.w)
    add = min(extra * 0.34, gaps[0] * 1.5)  # some spare width into the gutter, the rest around
    x = zone.x + (zone.w - (laid.w + add)) / 2.0
    out = []
    for i, p in enumerate(parts):
        out += p.place(x, zone.y + (zone.h - p.h) / 2.0)
        x += p.w + (gaps[i] + add if i < len(gaps) else 0.0)
    return out


# ----------------------------------------------------------------------------
# corners and edges


def place_floater(it: Item, s: float, ctx: Ctx, mx: float, my: float, relax: float) -> Optional[Placement]:
    r = realize(it, s, ctx, max_w=0.45 * ctx.W, max_h=0.3 * ctx.H, relax=relax)
    if r is None:
        return None
    c = it.corner or "tr"
    # keep the painted elements (not just the footprint) inside the margins
    u = union_all(el_visual_box(e) for e in r.emit(0.0, 0.0)) or Box(0, 0, r.w, r.h)
    x = mx - min(0.0, u.x) if c[1] == "l" else ctx.W - mx - max(r.w, u.r)
    y = my - min(0.0, u.y) if c[0] == "t" else ctx.H - my - max(r.h, u.b)
    return (r, x, y)


def overflow(placed: Sequence[Placement], ctx: Ctx) -> Tuple[float, float, float, float]:
    """How far painted element boxes poke out of the canvas on each side."""
    l = t = r_ = b = 0.0
    for r, x, y in placed:
        for e in r.emit(x, y):
            vb = el_visual_box(e)
            l, t = max(l, -vb.x), max(t, -vb.y)
            r_, b = max(r_, vb.r - ctx.W), max(b, vb.b - ctx.H)
    return l, t, r_, b


def collides(a: Box, b: Box, pad: float = 0.0) -> bool:
    return a.expanded(pad).inter(b) > 0


def _settle_floaters(placed, floaters, s, ctx, zone, mx, my, relax):
    """Place corner items; report a smaller zone if they clash with the content."""
    fl, bad = [], []
    for f in floaters:
        p = place_floater(f, s, ctx, mx, my, relax)
        (bad if p is None else fl).append(f if p is None else p)
    gap = 0.6 * ctx.base_fs
    new_zone = zone.copy()
    clash = False
    for r, x, y in fl:
        fb = Box(x, y, r.w, r.h)
        if any(collides(fb, Box(xx, yy, rr.w, rr.h), gap) for rr, xx, yy in placed):
            clash = True
            if r.item.corner[0] == "t":
                top = max(new_zone.y, fb.b + gap)
                new_zone = Box(new_zone.x, top, new_zone.w, new_zone.b - top)
            else:
                bot = min(new_zone.b, fb.y - gap)
                new_zone = Box(new_zone.x, new_zone.y, new_zone.w, bot - new_zone.y)
    return fl, bad, clash, new_zone


# ----------------------------------------------------------------------------
# the archetypes


def _visible(d: Design, hidden: Sequence[Item]) -> List[Item]:
    hid = {id(h) for h in hidden}
    return [it for it in d.items if id(it) not in hid and it.role not in ("missing", "void")]


def _zone(d: Design, ctx: Ctx, insets) -> Tuple[Box, float, float]:
    mx, my = margins(d, ctx)
    il, it_, ir, ib = insets
    return Box(mx + il, my + it_, ctx.W - 2 * mx - il - ir, ctx.H - 2 * my - it_ - ib), mx, my


def plan_stack(d: Design, ctx: Ctx, hidden: List[Item], name: str = "stack", insets=(0, 0, 0, 0),
               force_align: Optional[str] = None) -> Optional[Plan]:
    vis = _visible(d, hidden)
    floaters = [it for it in vis if it.corner]
    body = [it for it in vis if not it.corner]
    if not body:
        return None
    rows = rows_for(body)
    lo, hi = scale_bounds(d, ctx, "stack")
    zone, mx, my = _zone(d, ctx, insets)
    if zone.w < 20 or zone.h < 20:
        return None
    root, boxes = column(d, rows, hero_fill=True, cap_rel=1.9, min_rel=0.45, force_align=force_align)
    weights = [3.0 if any(i.role == "headline" for i in r) else (1.5 if any(i.role == "hero" for i in r) else 1.0)
               for r in rows]
    for _ in range(4):
        s, laid, relax = solve(root, ctx, zone, lo, hi)
        if laid is None:
            return None
        placed = place_column(laid, boxes, root.align, d, ctx, laid.zone, weights)
        fl, bad, clash, new_zone = _settle_floaters(placed, floaters, s, ctx, zone, mx, my, relax)
        if not clash:
            return Plan(name=name, placed=placed, hidden=list(hidden) + bad, s=s, zone=zone, floaters=fl,
                        relax=relax, capped=s >= hi * 0.999)
        if new_zone.h < 0.4 * (ctx.H - 2 * my):
            return None
        zone = new_zone
    return None


def plan_split(d: Design, ctx: Ctx, hidden: List[Item], hero_side: str = "left", hero_frac: float = 0.45,
               fold_corners: bool = False, name: str = "split", insets=(0, 0, 0, 0)) -> Optional[Plan]:
    vis = _visible(d, hidden)
    heroes = [it for it in vis if it.role == "hero"]
    if not heroes:
        imgs = [it for it in vis if it.kind in ("image", "composite") and it.role == "image"]
        if not imgs:
            return None
        heroes = [max(imgs, key=lambda i: i.src_box.area)]
    hero = heroes[0]
    floaters = [it for it in vis if it.corner and not fold_corners and it is not hero]
    rest = [it for it in vis if it is not hero and it not in floaters]
    if not rest:
        return None
    rows = rows_for(rest)
    col, _ = column(d, rows)
    lo, hi = scale_bounds(d, ctx, "split")
    zone, mx, my = _zone(d, ctx, insets)
    if zone.w < 20 or zone.h < 20:
        return None
    gutter = max(2.0 * ctx.base_fs, 0.04 * ctx.W)
    hero_leaf = Leaf(hero, fill=True, max_w_frac=hero_frac, cap_rel=2.2, min_rel=0.5)
    root = HStack([hero_leaf, col] if hero_side == "left" else [col, hero_leaf])
    root.gap_px = lambda i, s, ctx: gutter  # type: ignore[assignment]
    for _ in range(4):
        s, laid, relax = solve(root, ctx, zone, lo, hi)
        if laid is None:
            return None
        placed = _place_split(laid, ctx, laid.zone)
        fl, bad, clash, new_zone = _settle_floaters(placed, floaters, s, ctx, zone, mx, my, relax)
        if not clash:
            return Plan(name=name, placed=placed, hidden=list(hidden) + bad, s=s, zone=zone, floaters=fl,
                        relax=relax, capped=s >= hi * 0.999)
        if new_zone.h < 0.55 * (ctx.H - 2 * my):
            return None
        zone = new_zone
    return None


def plan_row(d: Design, ctx: Ctx, hidden: List[Item], name: str = "row", stack_logo: bool = False,
             insets=(0, 0, 0, 0)) -> Optional[Plan]:
    vis = _visible(d, hidden)
    hid = list(hidden)
    logo = next((i for i in vis if i.role == "logo"), None)
    hero = next((i for i in vis if i.role == "hero"), None)
    ctas = [i for i in vis if i.role == "cta"]
    texts = [i for i in vis if i.kind == "text"]
    badges = [i for i in vis if i.role in ("badge", "panel", "group") and not i.corner]
    floaters = [i for i in vis if i.corner and i.role != "logo"]
    others = [i for i in vis if i not in texts and i not in ctas and i not in badges and i not in floaters
              and i is not logo and i is not hero]
    hid += others  # secondary art does not fit a one-line banner
    copy_rows = rows_for(texts + badges)
    lo, hi = scale_bounds(d, ctx, "row")
    zone, mx, my = _zone(d, ctx, insets)
    if zone.w < 20 or zone.h < 12 or (not copy_rows and not ctas and logo is None and hero is None):
        return None
    centred = dominant_align(texts) == "center" if texts else True
    kids: List[Node] = []
    if logo is not None and not stack_logo:
        kids.append(Leaf(logo, fill=True, cap_rel=1.7, min_rel=1.0, max_w_frac=0.2))
    if hero is not None:
        kids.append(Leaf(hero, fill=True, cap_rel=1.6, min_rel=0.4, max_w_frac=0.3))
    if copy_rows or (stack_logo and logo is not None):
        rows = ([[logo]] if stack_logo and logo is not None else []) + copy_rows
        col, _ = column(d, rows, force_align="center" if centred else "start", gap_scale=0.6)
        kids.append(col)
    gap = max(1.2 * ctx.base_fs, 0.022 * ctx.W)
    s = laid = None
    relax = 1.0
    for nowrap in (True, False):  # a one-line CTA reads as a button; wrap only if forced
        root = HStack(kids + [Leaf(c, nowrap=nowrap) for c in ctas])
        root.gap_px = lambda i, s, ctx: gap  # type: ignore[assignment]
        s, laid, relax = solve(root, ctx, zone, lo, hi)
        if laid is not None:
            break
    if laid is None:
        return None
    placed = place_row(laid, ctx, laid.zone)
    fl = []
    for f in floaters:
        p = place_floater(f, s, ctx, mx, my, relax)
        if p is None or any(collides(Box(p[1], p[2], p[0].w, p[0].h), Box(xx, yy, rr.w, rr.h), 0.4 * ctx.base_fs)
                            for rr, xx, yy in placed):
            hid.append(f)
        else:
            fl.append(p)
    return Plan(name=name, placed=placed, hidden=hid, s=s, zone=zone, floaters=fl, relax=relax,
                capped=s >= hi * 0.999)


def plan_adapt(d: Design, ctx: Ctx, hidden: List[Item], name: str = "adapt", insets=(0, 0, 0, 0)) -> Optional[Plan]:
    """Keep the source arrangement (for targets of nearly the same shape):
    everything scales uniformly, positions follow the canvas, edges stay edges."""
    import math

    if abs(math.log((ctx.W / ctx.H) / (d.W / d.H))) > math.log(1.2):
        return None
    vis = _visible(d, hidden)
    if not vis:
        return None
    sx, sy = ctx.W / d.W, ctx.H / d.H
    s = min(sx, sy)
    zone, mx, my = _zone(d, ctx, insets)
    placed: List[Placement] = []
    for it in vis:
        b = it.src_box
        r = realize(it, s, ctx, max_w=max(b.w * s * 1.08, 1.0) if it.kind == "text" else zone.w, max_h=zone.h)
        if r is None:
            return None
        if b.cx < 0.4 * d.W:
            x = b.x * sx
        elif b.cx > 0.6 * d.W:
            x = ctx.W - (d.W - b.r) * sx - r.w
        else:
            x = b.cx * sx - r.w / 2.0
        if b.cy < 0.3 * d.H:
            y = b.y * sy
        elif b.cy > 0.7 * d.H:
            y = ctx.H - (d.H - b.b) * sy - r.h
        else:
            y = b.cy * sy - r.h / 2.0
        if it.kind == "text":  # copy keeps its alignment edge
            a = it.els[0].get("text_align", "left")
            if a == "left":
                x = b.x * sx
            elif a == "right":
                x = b.r * sx - r.w
        x = min(max(x, mx if it.corner is None else 0.0), ctx.W - r.w - (mx if it.corner is None else 0.0))
        y = min(max(y, my if it.corner is None else 0.0), ctx.H - r.h - (my if it.corner is None else 0.0))
        # art that hung off the source canvas is nudged back on
        l, t, rr, bb = overflow([(r, x, y)], ctx)
        x += l - rr
        y += t - bb
        placed.append((r, x, y))
    # nothing may touch that did not touch in the source
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            ri, xi, yi = placed[i]
            rj, xj, yj = placed[j]
            if Box(xi, yi, ri.w, ri.h).inter(Box(xj, yj, rj.w, rj.h)) > 0 and \
                    ri.item.src_box.inter(rj.item.src_box) <= 0:
                return None
    return Plan(name=name, placed=placed, hidden=list(hidden), s=s, zone=zone, capped=True)


# ----------------------------------------------------------------------------


def drop_orders(d: Design) -> List[List[Item]]:
    """Cumulative hide-sets, least important first. Items of one role that sit
    in one row (a set of badges) are dropped together. Headline and CTA are
    never hidden; hero and logo only as a last resort."""
    opt = [it for it in d.items if it.role not in CORE_ROLES and it.role not in ("missing", "void") and not it.core]
    units: List[List[Item]] = []
    for row in rows_for(opt):
        by_role: Dict[str, List[Item]] = {}
        for it in row:
            by_role.setdefault(it.role, []).append(it)
        units += list(by_role.values())
    units.sort(key=lambda u: (max(i.importance for i in u), -min(i.src_box.y for i in u)))
    out: List[List[Item]] = [[]]
    acc: List[Item] = []
    for u in units:
        acc = acc + u
        out.append(list(acc))
    for role in ("hero", "logo", "cta", "panel"):
        extra = [it for it in d.items if it.role == role and it not in acc]
        if extra:
            acc = acc + extra
            out.append(list(acc))
    return out


def _shift(p: Plan, dx: float, dy: float) -> Plan:
    p.placed = [(r, x + dx, y + dy) for r, x, y in p.placed]
    return p


def _with_insets(fn, d: Design, ctx: Ctx, *args, **kw) -> Optional[Plan]:
    """Run a planner and make sure every painted box stays on the canvas.

    Images carry transparent margins, so content that sits right at the safe
    edge can still poke out. First try sliding the content (there is often
    spare room on the other side); otherwise give that side more room and plan
    again."""
    ins = [0.0, 0.0, 0.0, 0.0]
    for _ in range(5):
        p = fn(d, ctx, *args, insets=tuple(ins), **kw)
        if p is None:
            return None
        l, t, r, b = overflow(p.placed, ctx)
        fl = overflow(p.floaters, ctx)
        if max(l, t, r, b) <= 0.5 and max(fl) <= 0.5:
            return p
        dx = (l + 0.5 if l > 0.5 and r <= 0.5 else 0.0) - (r + 0.5 if r > 0.5 and l <= 0.5 else 0.0)
        dy = (t + 0.5 if t > 0.5 and b <= 0.5 else 0.0) - (b + 0.5 if b > 0.5 and t <= 0.5 else 0.0)
        if dx or dy:
            q = _shift(Plan(p.name, list(p.placed), p.hidden, p.s, p.zone, p.floaters, p.relax, p.capped), dx, dy)
            if max(overflow(q.placed, ctx)) <= 0.5 and not any(
                    collides(Box(x, y, rr.w, rr.h), Box(fx, fy, fr.w, fr.h))
                    for rr, x, y in q.placed for fr, fx, fy in q.floaters):
                return q
        # centred content moves by half an inset, so ask for twice the overflow
        ins = [a + (2.0 * o + 1.0 if o > 0.5 else 0.0) for a, o in zip(ins, (l, t, r, b))]
    return None


def generate(d: Design, ctx: Ctx) -> List[Plan]:
    r = ctx.W / ctx.H
    plans: List[Plan] = []
    missing = [it for it in d.items if it.role in ("missing", "void")]
    has_corners = any(it.corner for it in d.items if it.role not in ("missing", "void"))
    from . import assets as _assets

    busy_bg = any(b.get("type") == "image" and _assets.info(b.get("asset_id")).busy > 0.3 for b in d.backgrounds)

    def variants(force_all: bool = False):
        yield "adapt", plan_adapt, {}
        if r <= 2.2 or force_all:
            yield "stack", plan_stack, {}
            if busy_bg:  # copy over a photo: also try it against either side
                yield "stack-start", plan_stack, dict(force_align="start")
                yield "stack-end", plan_stack, dict(force_align="end")
        if r >= 1.15:
            for side in ("left", "right"):
                for frac in (0.34, 0.48):
                    for fold in ((False, True) if has_corners else (False,)):
                        yield (f"split-{side}-{frac}{'-fold' if fold else ''}", plan_split,
                               dict(hero_side=side, hero_frac=frac, fold_corners=fold))
        if r >= 1.7 or force_all:
            for stack_logo in (False, True):
                yield "row" + ("-logo-stacked" if stack_logo else ""), plan_row, dict(stack_logo=stack_logo)

    if not _visible(d, missing):  # nothing to lay out: every element is void or missing
        return [Plan(name="empty", placed=[], hidden=list(missing), s=1.0, zone=Box(0, 0, ctx.W, ctx.H))]

    done = set()  # archetypes that already reached their cap without relaxing type
    for force_all in (False, True):
        for drop in drop_orders(d):
            hidden = missing + drop
            if any(h.role in CORE_ROLES for h in drop) and plans:
                break  # core items are only given up when nothing else worked
            for name, fn, kw in variants(force_all):
                if name in done:
                    continue
                p = _with_insets(fn, d, ctx, hidden, name=name, **kw)
                if p:
                    plans.append(p)
                    if p.relax >= 1.0 and p.capped:
                        done.add(name)
        if plans:
            break
    return plans
