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

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from .analyze import Design, Item
from .geom import Box, el_box, el_visual_box, union_all
from .items import Ctx, Real, display_size, realize, realize_scaled
from .layout import Flow, HStack, Laid, Leaf, Node, Padded, VStack, search_scale
from .structure import dominant_align, lead_text, row_align, rows_for, text_edges

Placement = Tuple[Real, float, float]

CORE_ROLES = ("headline", "cta", "logo", "hero")
SAME_SHAPE = 1.03  # aspect ratios this close are the same format at another size
EMPHASIS = (1.35, 1.7)  # display emphasis tried for a short headline when the design reflows
DISPLAY_WORDS = 8  # a headline this short can be set as display type; a longer one only gets the first step
BALANCE = 0.5  # how far a reflowed column's top and bottom margins move towards even
BOND = 0.3  # rows closer than this times the smaller one's height read as one unit and spread as one
MARGIN_LONG = 0.05  # the least margin along a canvas's length, as a share of it
STRIP = 3.0  # a canvas this many times wider than tall is a strip
STRIP_CTA = 0.45  # on a strip a CTA label is at least this share of the strip's display size ...
TALL_AD = 1.6  # a canvas this many times taller than wide ...
SMALL_AD = 400  # ... with a short side under this (px) is a tall ad unit (half-page, skyscraper)
TALL_CTA = 0.3  # ... whose CTA label is at least this share of its display size ...
BOLD_CTA_KEEP = 0.95  # ... when that costs the rest of the layout no more than 5% of its scale
HERO_SHARE = 0.2  # in a column, a hero may grow into leftover room until it covers this share of the zone
COPY_POSTER = 1.4  # a design of copy alone on a tall canvas may set its headline this much past display size
COLUMN_ROOM = 1.35  # how much further than sqrt(area) a column may grow on a taller canvas
ROW_GAP = 0.25  # a banner row's gaps widen to fill it, up to this share of the canvas width each
WIDE_MEASURE = 1.8  # on a much wider canvas, copy is also tried this much wider than its source measure
WIDER = 1.5  # ... when the target's aspect ratio is this many times the source's
TOUCH = 0.01  # px: boxes this much closer than their gap only touch it


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
    """Safe-area margins, following the source's own breathing room, and never
    thinner than a banner's: about 5% of the long side (a 728x90 keeps ~27 px
    a side), though no more than 0.3 of the short side."""
    body = [it for it in d.items if it.corner is None and it.role != "missing"]
    fr = 0.06
    if body:
        u = union_all(it.src_box for it in body)
        edge = min(u.x, u.y, d.W - u.r, d.H - u.b)
        fr = edge / max(1.0, min(d.W, d.H))
    fr = min(0.075, max(0.045, fr))
    short = min(ctx.W, ctx.H)
    floor = min(6.0, max(3.0, 0.08 * short))  # tiny banners cannot spare 6px a side
    my = max(floor, fr * short, min(MARGIN_LONG * 0.875 * ctx.H, 0.3 * ctx.W))
    mx = max(floor, fr * short, min(MARGIN_LONG * ctx.W, 0.3 * ctx.H))
    return mx, my


def display_room(d: Design, W: float, H: float) -> float:
    """How far past the canvas's display size this design's headline may go.
    A design of copy alone (no art beyond its logo) on a canvas taller than
    wide has only its type to carry the height, so it is set larger, as a
    typographic poster is (COPY_POSTER)."""
    art = any(i.kind in ("image", "composite") and i.role not in ("logo", "missing", "void", "watermark")
              for i in d.items)
    return COPY_POSTER if H > W and not art else 1.0


def scale_bounds(d: Design, ctx: Ctx, kind: str = "stack") -> Tuple[float, float]:
    """Search range for the global scale.

    Designers re-making a template at another size scale type by about
    sqrt(target area / source area), and ~1.4x more on strips 6:1 and wider
    (measured on 1,630 Crello multi-size families). Reflowed layouts may also
    bring a short headline up to the display size typical for the format
    (~0.28 H on strips, ~0.16 H on wide banners, ~0.13 H landscape), and a
    design moved to a much roomier canvas may grow towards a display headline,
    so it does not look like a small design lost in space.

    A column moved to a canvas taller than its source gains height to grow
    into, as Adobe's reference story does (the post set about 1.6x larger, its
    headline broken in two): up to COLUMN_ROOM times further, the more the
    shape changed. On a wider canvas one column shares the short side."""
    import math

    rw, rh = ctx.W / d.W, ctx.H / d.H
    lo_r, hi_r = min(rw, rh), max(rw, rh)
    r = ctx.W / ctx.H
    cap = math.sqrt(rw * rh)
    if kind == "stack":
        if rh > rw:
            cap *= min(COLUMN_ROOM, 1.0 + 0.5 * math.log(rh / rw))
        else:
            cap = min(cap, lo_r ** 0.6 * hi_r ** 0.4)
    if r >= 6.0 or r <= 1 / 6.0:
        cap *= 1.4
    head = next((i for i in d.items if i.role == "headline"), None)
    if head is not None and head.fs0 > 0:
        n = max(1, len(str(lead_text(head).get("content", ""))))
        short = min(1.0, max(0.45, 28.0 / n))
        target = 0.075 * min(ctx.W, ctx.H) * short
        room = 1.0 + 0.5 * math.log(max(1.0, hi_r / max(lo_r, 1e-6)))
        cap = max(cap, min(target / head.fs0, cap * room))
        if kind in ("row", "split") and r >= 1.4:
            frac = 0.28 if r >= 5.0 else (0.16 if r >= 2.4 else 0.13)
            cap = max(cap, min(frac * ctx.H * short / head.fs0, 2.0 * cap))
    if head is not None and head.main_fs() > 0:
        # the design grows only until its headline reaches the canvas's display size (or past it,
        # as far past as the source set it): nothing outgrows the headline, the hierarchy holds
        rank = max(1.0, head.main_fs() / display_size(d.W, d.H))
        cap = min(cap, rank * ctx.display_max / head.main_fs())
    return 0.01, max(0.02, cap)


def _row_box(row: Sequence[Item]) -> Box:
    return union_all(i.src_box for i in row)


def _leaf(it: Item, hero_fill: bool, cap_rel: float = 1.5, min_rel: float = 0.5, hero_share: float = 0.0) -> Leaf:
    """A row's item; a hero that fills takes leftover room, up to ``cap_rel``
    times the layout's scale, or (``hero_share``) until it covers that share
    of the zone, whichever is larger: a small product in the source still
    reaches a size that carries a roomy canvas, from space no copy needed."""
    if it.role == "hero" and hero_fill:
        floor = math.sqrt(hero_share / max(1e-9, it.src_box.area)) if hero_share else 0.0
        return Leaf(it, fill=True, cap_rel=cap_rel, min_rel=min_rel, cap_floor=floor)
    return Leaf(it)


def _row_node(row: Sequence[Item], hero_fill: bool, **kw) -> Node:
    if len(row) == 1:
        return _leaf(row[0], hero_fill, **kw)
    gaps = [max(0.0, b.src_box.x - a.src_box.r) for a, b in zip(row, row[1:])]
    g = sum(gaps) / len(gaps) if gaps else None
    if not any(i.is_copy for i in row):
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
    # proximity: lines of one text block (two copy rows in a row) sit tight; groups stand apart.
    # A brand set in type and a call to action are groups of their own, not lines of the block
    copy = [all(i.is_copy and i.role not in ("cta", "logo") for i in row) for row in rows]
    tight = [a and b for a, b in zip(copy, copy[1:])]
    for a, b, bd in reversed(_panel_runs(d, rows)):  # a panel's rows become one padded block
        inner = VStack(nodes[a:b + 1], gaps=gaps[a:b], align=aligns[a:b + 1], tight=tight[a:b])
        nodes[a:b + 1] = [Padded(inner, bd.pad, bd.edges)]
        gaps[a:b] = []
        tight[a:b] = []
        aligns[a:b + 1] = [aligns[a]]
        boxes[a:b + 1] = [el_box(bd.el)]
    return VStack(nodes, gaps=gaps, align=aligns, tight=tight), boxes


def _padded_runs(d: Design, parts: List[Node], join) -> Optional[List[Node]]:
    """Side by side parts that hold exactly one panel's shown members become
    one padded block (``join`` sets them in a row), as a column's do. None if
    a panel's members end up in separate parts (the logo stacked onto its
    headline, its button beside): a panel drawn around pieces that do not
    sit together reads as a mistake."""
    runs = _panel_runs(d, [p.items() for p in parts])
    for a, b, bd in reversed(runs):
        parts[a:b + 1] = [Padded(join(parts[a:b + 1]), bd.pad, bd.edges)]
    for bd in d.backdrops:
        mine = {id(m) for m in bd.members}
        if sum(1 for p in parts if any(id(it) in mine for it in p.items())) > 1:
            return None
    return parts


def _panel_runs(d: Design, rows: List[List[Item]]) -> List[Tuple[int, int, object]]:
    """Runs of consecutive rows that hold exactly one panel's shown members
    (all of them, nothing else), as (first, last, backdrop)."""
    out = []
    shown = {id(it) for row in rows for it in row}
    for bd in d.backdrops:
        mine = {id(m) for m in bd.members} & shown
        idx = [i for i, row in enumerate(rows) if all(id(it) in mine for it in row)]
        if not idx or idx != list(range(idx[0], idx[-1] + 1)):
            continue
        if {id(it) for i in idx for it in rows[i]} != mine:
            continue
        if any(a <= idx[-1] and idx[0] <= b for a, b, _ in out):
            continue
        out.append((idx[0], idx[-1], bd))
    return sorted(out, key=lambda r: r[0])


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


def _room(a: Box, b: Box) -> float:
    """The space the source put between two consecutive rows of a column: the
    gap between them, whichever way it ran (a banner's gutters become a
    column's breathing room, though, as with any gap in the rhythm, a gutter
    counts for no more than the smaller of its two rows: past that it is the
    strip's spare width, not spacing). Rows set tight (a subhead tucked under
    its headline) are bonded and get none."""
    side_by_side = b.y < a.b - 0.5 and a.y < b.b - 0.5
    g = min(max(0.0, b.x - a.r, a.x - b.r), min(a.h, b.h)) if side_by_side else b.y - a.b
    return g if g >= BOND * min(a.h, b.h) else 0.0


def place_column(laid: Laid, boxes: List[Box], aligns: Sequence[str], d: Design, ctx: Ctx,
                 zone: Box, side: Optional[str] = None, tight: Sequence[bool] = ()) -> List[Placement]:
    """Rows top to bottom with their measured gaps; the height left over is
    shared between the top margin, each gap and the bottom margin in
    proportion to the room the source left there (``_room``; the margins are
    the content's own, moved half way to even, BALANCE). The source's spacing
    pattern survives (a design that breathes at the bottom still does, a
    little less; rows that sat close stay close) while gaps follow the
    content, not the canvas.

    Across, the column is one block whose centre sits where the source's
    content was centred, relative to the canvas width (a balanced design is
    centred whatever its text alignment; one set off to a side stays there),
    unless ``side`` pins it. Rows align inside the block."""
    parts: List[Laid] = laid.parts  # type: ignore[attr-defined]
    gaps: List[float] = laid.gap_list  # type: ignore[attr-defined]
    top, bottom = min(b.y for b in boxes), d.H - max(b.b for b in boxes)
    mid = (top + bottom) / 2.0  # a reflowed column leans less than its source: half way to centred
    top, bottom = top + BALANCE * (mid - top), bottom + BALANCE * (mid - bottom)
    inner = [0.0 if i < len(tight) and tight[i] else _room(a, b) for i, (a, b) in enumerate(zip(boxes, boxes[1:]))]
    room = [top] + inner + [bottom]  # a text block's own gaps take none of the spare height
    slack = max(0.0, zone.h - sum(p.h for p in parts) - sum(gaps))
    if sum(room) > 0:
        share = [slack * r / sum(room) for r in room]
    else:  # nothing to go by: centre the column
        share = [slack / 2.0] + [0.0] * (len(parts) - 1) + [slack / 2.0]
    bw = max(p.w for p in parts)
    if side in ("start", "end"):
        bx = zone.x if side == "start" else zone.r - bw
    else:
        cx0 = (max(0.0, min(b.x for b in boxes)) + min(d.W, max(b.r for b in boxes))) / 2.0
        bx = zone.x + zone.w * cx0 / max(1.0, d.W) - bw / 2.0
        bx = min(max(bx, zone.x), zone.r - bw) if zone.w > bw else zone.cx - bw / 2.0
    out, y = [], zone.y + share[0]
    for i, (p, a) in enumerate(zip(parts, aligns)):
        x = bx if a == "start" else (bx + bw - p.w if a == "end" else bx + (bw - p.w) / 2.0)
        out += p.place(x, y)
        y += p.h + (gaps[i] + share[i + 1] if i < len(gaps) else 0.0)
    return out


def place_row(laid: Laid, ctx: Ctx, zone: Box, ends_in_cta: bool = False) -> List[Placement]:
    """A row with ends is set the way a banner is: spare width goes into the
    gaps, so the first part holds the left margin and the last the right (the
    brand at one end, the call to action at the other), no gap opening past a
    quarter of the canvas. A row has ends when it has three or more parts, or
    when its last part is the call to action. Otherwise two parts read as one
    group: their gap breathes a little (up to three times its size) and the
    pair is centred."""
    parts: List[Laid] = laid.parts  # type: ignore[attr-defined]
    gaps: List[float] = laid.gap_list  # type: ignore[attr-defined]
    extra = max(0.0, zone.w - laid.w)
    n = len(gaps)
    if n >= 2 or (n == 1 and ends_in_cta):
        add = min(extra / n, max(0.0, ROW_GAP * ctx.W - max(gaps)))
    else:
        add = min(extra / (n + 1.0), 2.0 * max(gaps)) if n else 0.0
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


def place_floaters(floaters: Sequence[Item], s: float, ctx: Ctx, mx: float, my: float,
                   relax: float) -> Tuple[List[Placement], List[Item]]:
    """Corner items, most important first. One that would sit on another in
    its corner queues beside it along the edge; if it cannot, it is dropped."""
    gap = 0.6 * ctx.base_fs
    placed: List[Placement] = []
    failed: List[Item] = []
    for f in sorted(floaters, key=lambda f: -f.importance):
        p = place_floater(f, s, ctx, mx, my, relax)
        if p is None:
            failed.append(f)
            continue
        r, x, y = p
        right = (f.corner or "tr")[1] == "r"
        for _ in range(len(placed) + 1):
            hit = next((Box(xx, yy, rr.w, rr.h) for rr, xx, yy in placed
                        if collides(Box(x, y, r.w, r.h), Box(xx, yy, rr.w, rr.h), gap)), None)
            if hit is None:
                break
            x = hit.x - gap - r.w if right else hit.r + gap
        if hit is not None or x < mx - 0.5 or x + r.w > ctx.W - mx + 0.5:
            failed.append(f)
        else:
            placed.append((r, x, y))
    return placed, failed


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
    """Closer than pad. Boxes that sit exactly pad apart (a zone cut to clear a
    corner item and then filled) do not collide: a hair of float error is not a clash."""
    return a.expanded(pad - TOUCH).inter(b) > 0


def _settle_floaters(placed, floaters, s, ctx, zone, mx, my, relax):
    """Place corner items; report a smaller zone if they clash with the content."""
    fl, bad = place_floaters(floaters, s, ctx, mx, my, relax)
    gap = 0.6 * ctx.base_fs
    new_zone = zone.copy()
    clash = False
    for r, x, y in fl:
        fb = Box(x, y, r.w, r.h)
        if any(collides(fb, Box(xx, yy, rr.w, rr.h), gap) for rr, xx, yy in placed):
            clash = True
            new_zone = _clear_of(new_zone, fb, r.item.corner, gap)
    return fl, bad, clash, new_zone


def _clear_of(zone: Box, fb: Box, corner: str, gap: float) -> Box:
    """The zone with a corner item's box (plus gap) taken off: from the top or
    bottom, or from the side, whichever leaves the zone more room."""
    if corner[0] == "t":
        top = max(zone.y, fb.b + gap)
        vertical = Box(zone.x, top, zone.w, zone.b - top)
    else:
        bot = min(zone.b, fb.y - gap)
        vertical = Box(zone.x, zone.y, zone.w, bot - zone.y)
    if corner[1] == "l":
        left = max(zone.x, fb.r + gap)
        side = Box(left, zone.y, zone.r - left, zone.h)
    else:
        right = min(zone.r, fb.x - gap)
        side = Box(zone.x, zone.y, right - zone.x, zone.h)
    return vertical if vertical.area >= side.area else side


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
    root, boxes = column(d, rows, hero_fill=True, cap_rel=1.9, min_rel=0.7, force_align=force_align,
                         hero_share=HERO_SHARE * zone.w * zone.h)
    tall_ad = ctx.H >= TALL_AD * ctx.W and min(ctx.W, ctx.H) < SMALL_AD and any(i.role == "cta" for i in body)
    for _ in range(4):
        s, laid, relax = solve(root, ctx, zone, lo, hi)
        if laid is None:
            return None
        if tall_ad:  # a tall ad unit read at a glance: the button is the column's second focal point
            s, laid, relax = _with_bold_cta(root, d, ctx, zone, lo, hi, (s, laid, relax), TALL_CTA)
        placed = place_column(laid, boxes, root.align, d, ctx, laid.zone, side=force_align, tight=root.tight)
        fl, bad, clash, new_zone = _settle_floaters(placed, floaters, s, ctx, zone, mx, my, relax)
        if not clash:
            return Plan(name=name, placed=placed, hidden=list(hidden) + bad, s=s, zone=zone, floaters=fl,
                        relax=relax, capped=s >= hi * 0.999)
        if new_zone.h < 0.4 * (ctx.H - 2 * my) or new_zone.w < 0.5 * (ctx.W - 2 * mx):
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
        if new_zone.h < 0.55 * (ctx.H - 2 * my) or new_zone.w < 0.5 * (ctx.W - 2 * mx):
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
    texts = [i for i in vis if i.is_copy and i is not logo]
    badges = [i for i in vis if i.role in ("badge", "panel", "group") and not i.corner and not i.is_copy]
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
    if hero is not None:  # on a strip the type scale is tiny; the picture may still fill the strip's height
        kids.append(Leaf(hero, fill=True, cap_rel=1.6, min_rel=0.4, max_w_frac=0.3,
                         cap_floor=0.9 * zone.h / max(1.0, hero.src_box.h)))
    if copy_rows or (stack_logo and logo is not None):
        rows = ([[logo]] if stack_logo and logo is not None else []) + copy_rows
        col, _ = column(d, rows, force_align="center" if centred else "start", gap_scale=0.6)
        kids.append(col)
    gap = max(1.2 * ctx.base_fs, 0.022 * ctx.W)

    def hstack(parts: List[Node]) -> HStack:
        h = HStack(parts)
        h.gap_px = lambda i, s, ctx: gap  # type: ignore[assignment]
        return h

    s = laid = None
    relax = 1.0
    # on a strip the call to action is the banner's second focal point, in a column of its own:
    # its label keeps pace with the strip's display type instead of sitting at the legibility floor
    root = None
    for nowrap in (True, False):  # a one-line CTA reads as a button; wrap only if forced
        parts = _padded_runs(d, kids + [Leaf(c, nowrap=nowrap) for c in ctas], hstack)
        if parts is None:
            return None
        root = hstack(parts)
        s, laid, relax = solve(root, ctx, zone, lo, hi)
        if laid is not None:
            break
    if laid is None:
        return None
    if ctas and ctx.W >= STRIP * ctx.H:  # on a strip the CTA sits in a column of its own
        s, laid, relax = _with_bold_cta(root, d, ctx, zone, lo, hi, (s, laid, relax), STRIP_CTA)
    
    placed = place_row(laid, ctx, laid.zone, ends_in_cta=bool(ctas))
    fl = []
    corner, failed = place_floaters(floaters, s, ctx, mx, my, relax)
    hid += failed
    for p in corner:  # a one-line banner has no room to give: a corner item that clashes is dropped
        if any(collides(Box(p[1], p[2], p[0].w, p[0].h), Box(xx, yy, rr.w, rr.h), 0.4 * ctx.base_fs)
               for rr, xx, yy in placed):
            hid.append(p[0].item)
        else:
            fl.append(p)
    return Plan(name=name, placed=placed, hidden=hid, s=s, zone=zone, floaters=fl, relax=relax,
                capped=s >= hi * 0.999)


def _with_bold_cta(root: Node, d: Design, ctx: Ctx, zone: Box, lo: float, hi: float, plain, share: float):
    """The layout solved again with its call-to-action label set at no less
    than ``share`` of the canvas's display size, kept only when that is free:
    when the label outgrows no copy that outranked it and the rest of the
    layout keeps at least BOLD_CTA_KEEP of its scale. Otherwise ``plain``."""
    s, laid, relax = plain
    bs, blaid, brelax = solve(root, replace(ctx, cta_min=share * ctx.display_max), zone, lo, hi)
    if blaid is not None and bs >= BOLD_CTA_KEEP * s and _cta_in_rank(blaid.place(0.0, 0.0), d):
        return bs, blaid, brelax
    return plain


def _cta_in_rank(placed: List[Placement], d: Design) -> bool:
    """No call-to-action label is set larger than copy that was at least 25%
    larger than it in the source (the hierarchy the scorer and the checks
    hold every layout to), and no button is wider than the headline."""
    out, cta = {}, set()
    for r, _, _ in placed:
        for tid, fs in r.fonts.items():
            out[tid] = fs
            if r.item.role == "cta":
                cta.add(tid)
    src = {tid: float(d.by_id[tid].get("font_size") or 0) for tid in out if tid in d.by_id}
    if any(src[a] >= 1.25 * src[c] and out[a] < out[c] - 1 for c in cta for a in out if a not in cta):
        return False
    head = max((r.w for r, _, _ in placed if r.item.role == "headline"), default=0.0)
    return not head or all(r.w <= head + 0.5 for r, _, _ in placed if r.item.role == "cta")


def same_shape(d: Design, ctx: Ctx) -> bool:
    import math

    return abs(math.log((ctx.W / ctx.H) / (d.W / d.H))) <= math.log(SAME_SHAPE)


def plan_scaled(d: Design, ctx: Ctx, hidden: List[Item], name: str = "scaled") -> Optional[Plan]:
    """A target of the source's own shape gets the design itself, uniformly
    scaled: nothing to rearrange, so nothing is. Copy that has to be set
    larger to stay legible may not run into anything it did not touch."""
    from .textfit import drawn_box, new_text_overlaps

    k = min(ctx.W / d.W, ctx.H / d.H)
    ox, oy = (ctx.W - d.W * k) / 2.0, (ctx.H - d.H * k) / 2.0
    placed: List[Placement] = []
    for it in _visible(d, hidden):
        placed.append((realize_scaled(it, k, ctx), ox + it.src_box.x * k, oy + it.src_box.y * k))
    els = [e for r, x, y in placed for e in r.emit(x, y)]
    if new_text_overlaps(els, d.elements):
        return None
    canvas = Box(0, 0, ctx.W, ctx.H)
    for r, x, y in placed:
        if not r.fonts:
            continue
        for e in r.emit(x, y):
            if e["id"] not in r.fonts:
                continue
            db = drawn_box(e)
            if db.inter(canvas) < db.area - 1.0:
                return None
            for o, ox_, oy_ in placed:
                ob = Box(ox_, oy_, o.w, o.h)
                if o is not r and db.inter(ob) > 0 and r.item.src_box.inter(o.item.src_box) <= 0:
                    return None
    return Plan(name=name, placed=placed, hidden=list(hidden), s=k, zone=canvas, capped=True)


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


SET_ROLES = ("badge", "ribbon", "legal", "watermark")  # dropped as a whole set, never one by one


def drop_orders(d: Design) -> List[List[Item]]:
    """Cumulative hide-sets, least important first. Small items of one role
    that sit in one row (a set of badges) are dropped together; a row of
    content (product cards) gives way one item at a time from its end, so the
    ones that fit stay. Headline and CTA are never hidden; hero and logo only
    as a last resort."""
    opt = [it for it in d.items if it.role not in CORE_ROLES and it.role not in ("missing", "void") and not it.core]
    units: List[List[Item]] = []
    for row in rows_for(opt):
        by_role: Dict[str, List[Item]] = {}
        for it in row:
            by_role.setdefault(it.role, []).append(it)
        for role, items in by_role.items():
            units += [items] if role in SET_ROLES or role in ("subhead", "body") else [[i] for i in items]
    units.sort(key=lambda u: (max(i.importance for i in u), -min(i.src_box.y for i in u),
                              -max(i.src_box.x for i in u)))
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
    head = next((it for it in d.items if it.role == "headline" and it.is_copy), None)
    words = str(lead_text(head).get("content", "")).split() if head is not None else []
    emphasis = () if not words else EMPHASIS if len(words) <= DISPLAY_WORDS else EMPHASIS[:1]

    def archetypes(force_all: bool):
        if r <= 2.2 or force_all:
            yield "stack", plan_stack, {}
            if busy_bg:  # copy over a photo: also try it against either side
                yield "stack-start", plan_stack, dict(force_align="start")
                yield "stack-end", plan_stack, dict(force_align="end")
        if r >= 0.95:
            for side in ("left", "right"):
                for frac in (0.34, 0.48):
                    for fold in ((False, True) if has_corners else (False,)):
                        yield (f"split-{side}-{frac}{'-fold' if fold else ''}", plan_split,
                               dict(hero_side=side, hero_frac=frac, fold_corners=fold))
        if r >= 1.7 or force_all:
            for stack_logo in (False, True):
                yield "row" + ("-logo-stacked" if stack_logo else ""), plan_row, dict(stack_logo=stack_logo)

    measures = (1.0, WIDE_MEASURE) if r >= WIDER * d.W / d.H else (1.0,)

    def variants(force_all: bool = False):
        """(name, planner, options, ctx). A reflowed design is also offered with
        its headline set as display type, larger than the rest (EMPHASIS; a
        long headline only the first step, see items._display), and, on a much
        wider canvas, with its copy set to a longer line, so a column of copy
        can trade lines for size (WIDE_MEASURE)."""
        yield "adapt", plan_adapt, {}, ctx
        for name, fn, kw in archetypes(force_all):
            for m in measures:
                mname, mctx = (name, ctx) if m == 1.0 else (f"{name}~{m:g}", replace(ctx, measure=m))
                yield mname, fn, kw, mctx
                for e in emphasis:
                    yield f"{mname}@{e:g}", fn, kw, replace(mctx, emphasis=e)

    if not _visible(d, missing):  # nothing to lay out: every element is void or missing
        return [Plan(name="empty", placed=[], hidden=list(missing), s=1.0, zone=Box(0, 0, ctx.W, ctx.H))]
    if same_shape(d, ctx):
        p = plan_scaled(d, ctx, missing)
        if p is not None:
            return [p]

    done = set()  # archetypes that already reached their cap without relaxing type
    for force_all in (False, True):
        for drop in drop_orders(d):
            hidden = missing + drop
            if any(h.role in CORE_ROLES for h in drop) and plans:
                break  # core items are only given up when nothing else worked
            fitted = set()
            for name, fn, kw, c in variants(force_all):
                base = name.split("@")[0]
                # emphasis only makes the headline larger: if the plain plan did not fit, neither will it
                if name in done or (base != name and base not in fitted):
                    continue
                p = _with_insets(fn, d, c, hidden, name=name, **kw)
                if p:
                    plans.append(p)
                    fitted.add(name)
                    if p.relax >= 1.0 and p.capped:
                        done.add(name)
        if plans:
            break
    return plans
