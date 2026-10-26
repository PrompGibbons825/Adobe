"""Rank candidate plans the way the rubric would.

  * everything important is there         sum of importance of visible items,
                                          minus that of hidden ones
  * text is big / readable                each item's scale relative to the best
                                          scale any candidate achieved for it,
                                          plus a penalty below comfortable sizes
  * no collisions                         any overlap of footprints is heavily
                                          penalised (layouts are built not to)
  * looks intentional / balanced          canvas coverage close to the source's,
                                          centred visual mass
  * suits the format                      a small prior per archetype by aspect
"""

from __future__ import annotations

import math
from typing import Dict, List

from . import textfit
from .analyze import Design
from .compose import KEEP_HERO, ROOMY, Plan, art_box, keep_share, story_band
from .geom import Box
from .items import Ctx, display_lines

ALPHA = 1.1
LINE_CAP = {"headline": 4, "subhead": 5, "body": 6, "legal": 3}
INVERSION = 25.0  # for each text now smaller than one it was 25% larger than in the source
DENSITY = 120.0  # per unit of crowding past a comfortable copy density (see _crowding)
SMALL_UNIT = 400  # px: canvases with a shorter short side are ad units, read at a glance
RUNT = 0.12  # times its importance, for a headline or subhead ending on a lone word after a fuller line
COLLAPSE = 1.5  # times its importance, for copy set level with copy it was 1.5x smaller than
HIERARCHY = 12.0  # for a headline that stands out 1.6x more than in the source (or less, negative)
INFLATION = 2.0  # per unit of importance and of log-excess share (see _inflation)
SHARE_SLACK = math.log(2.5)  # a secondary item may grow to 2.5x its share of the canvas freely
COLLAGE = 15.0  # prior for re-flowing a collage of pictures into a grid for the new shape
REGIONS = 65.0  # prior for keeping a photo / panel split as two regions that tile the canvas
FILL = 15.0  # on a roomy canvas, for using it up to the larger of 1.1x the source's coverage, at most 55%
SHRINK = 0.6  # per unit of importance and of missing share, for a headline or picture shrunk past keep (_shrunk)
TRIVIAL = 30.0  # for a plan that is the source uniformly scaled into letterbox bands (_trivial)
TRIVIAL_ASPECT = math.log(1.5)  # ... once the aspect ratio has changed this much


def _prior(name: str, r: float, r0: float = 1.0) -> float:
    lr = math.log(r)
    if name.startswith("anchor"):
        return 38.0 * max(0.0, 1.0 - abs(math.log(r / r0)) / math.log(1.45))
    if name.startswith("adapt"):
        # the designer's own arrangement, when the shape barely changes
        return 40.0 * max(0.0, 1.0 - abs(math.log(r / r0)) / math.log(1.2))
    if name.startswith("retrieve"):
        return 14.0 - 6.0 * min(1.0, abs(math.log(r / r0)) / math.log(2.0))
    if name.startswith("flex-grid"):
        return 9.0 - 4.0 * max(0.0, lr - math.log(1.1))
    if name.startswith("flex-tile"):
        return REGIONS * 0.85
    if name.startswith("flex"):
        return 7.0 - 5.0 * max(0.0, abs(lr - math.log(1.0)))
    if name.startswith("stack") or name.startswith("quiet"):  # a quiet plan is a stack (quiet.py)
        return 10.0 - 14.0 * max(0.0, lr - math.log(1.25))
    if name.startswith("split"):
        p = 17.0 - 12.0 * abs(lr - math.log(2.0))
        if "-fold" in name:
            p -= 3.0
        return p
    if name.startswith("regions"):
        # a photo region beside a flat panel carrying the copy (regions.py): the designer's own
        # structure, rebuilt for the new shape, rather than a thumbnail floating on blank canvas
        return REGIONS
    if name.startswith("collage"):
        # a collage or grid of pictures re-flowed for the shape (collage.py): the designer's
        # own structure kept, as a split or a regions plan keeps theirs
        return COLLAGE
    if name.startswith("row"):
        p = 10.0 - 18.0 * max(0.0, math.log(3.6) - lr)
        if "stacked" in name:
            p -= 2.0
        return p
    return 0.0


def _core_hidden(p: Plan) -> int:
    return p.core_lost()


def _loss(p: Plan):
    return (_core_hidden(p), p.lost())


def prune_relaxed(plans: List[Plan]) -> List[Plan]:
    """Type below the legibility floor is a last resort: drop such a plan when a
    plan at full legibility keeps at least as much on the canvas (core items
    first, then every element: S1, keeping an element beats setting it larger)."""
    full = [_loss(p) for p in plans if p.relax >= 1.0]
    if not full:
        return plans
    best = min(full)
    return [p for p in plans if p.relax >= 1.0 or _loss(p) < best]


def keep_everything_first(plans: List[Plan]) -> None:
    """S1 (loop/STEERING.md): a plan that leaves an element out ranks after every plan
    that shows them all, whatever its score; among the rest, fewer core items lost first
    (a product shrunk to a speck is worse than a row of badges left out), then fewer lost."""
    plans.sort(key=_loss)  # stable: the score order holds within each class


def score_plans(plans: List[Plan], d: Design, ctx: Ctx) -> None:
    best: Dict[str, float] = {}
    for p in plans:
        for r, _, _ in p.placed + p.floaters:
            best[r.item.key] = max(best.get(r.item.key, 0.0), r.scale)
    src_cov = sum(it.src_box.area for it in d.items if it.role != "missing") / max(1.0, d.W * d.H)
    for p in plans:
        p.score, p.breakdown = _score(p, best, d, ctx, src_cov)


def _score(p: Plan, best: Dict[str, float], d: Design, ctx: Ctx, src_cov: float):
    b: Dict[str, float] = {}
    placed = p.placed + p.floaters
    vis = 0.0
    small = 0.0
    lines_pen = 0.0
    roomy = min(ctx.W, ctx.H) >= ROOMY
    for r, _, _ in placed:
        it = r.item
        rel = r.scale / max(1e-9, best.get(it.key, r.scale))
        if it.kind in ("image", "composite"):
            # art only needs to be big enough; past 60% of the best size it is fine (a hero on a
            # roomy canvas, within 20% of it: it carries the design there)
            enough = 0.8 if it.role == "hero" and roomy else 0.6
            vis += it.importance * (1.0 + 0.6 * math.log(max(1e-3, min(1.0, rel / enough))))
        else:
            # copy should be as large as the best layout manages, within ~15%
            vis += it.importance * (1.0 + ALPHA * math.log(max(1e-3, min(1.0, rel / 0.85))))
        for fs in r.fonts.values():
            role = it.role if it.role in ("legal", "badge", "ribbon", "headline") else "body"
            floor = ctx.floor(role)
            comfy = 1.25 * ctx.floor("legal" if role == "legal" else "body")
            if fs < comfy:
                small += it.importance * 0.6 * (1.0 - fs / comfy)
            if fs < floor - 0.5:  # below the legibility floor: only if nothing else fits
                small += it.importance * 1.5 * (1.0 - fs / floor)
        for blk in r.blocks.values():
            cap = LINE_CAP.get(it.role, 1)
            if blk.n_lines > cap:
                lines_pen += it.importance * 0.05 * (blk.n_lines - cap)
            words = len(" ".join(blk.lines).split())
            display = it.role == "headline" and blk.n_lines <= max(it.src_lines, display_lines(words))
            if it.role in ("headline", "subhead") and textfit.has_runt(blk):
                lines_pen += it.importance * RUNT  # a lone word left on the last line of a block
            if blk.n_lines > 1 and not display:
                chars = sum(len(l) for l in blk.lines) / blk.n_lines
                if chars < 14:
                    lines_pen += it.importance * 0.25 * (14 - chars) / 14
    b["visible"] = vis
    b["hidden"] = -sum(h.importance for h in p.hidden if h.role != "missing")
    b["inflation"] = -INFLATION * _inflation(placed, d, ctx)
    b["small_text"] = -small
    b["lines"] = -lines_pen

    boxes = [(r.item, Box(x, y, r.w, r.h)) for r, x, y in placed]
    ov = 0.0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, bb = boxes[i][1], boxes[j][1]
            inter = a.inter(bb)
            if inter > 0:
                ov += inter / max(1.0, min(a.area, bb.area))
    b["overlap"] = -150.0 * ov

    area = sum(bx.area for _, bx in boxes)
    cov = area / max(1.0, ctx.W * ctx.H)
    want = min(0.55, src_cov)
    b["coverage"] = -40.0 * max(0.0, want * 0.7 - cov) / max(1e-6, want)
    if roomy:  # room to grow into (small ad units keep their density logic)
        b["fill"] = FILL * min(1.0, cov / max(1e-6, min(0.55, 1.1 * src_cov)))
        b["trivial"] = -TRIVIAL * _trivial(boxes, d, ctx)
    b["shrunk"] = -SHRINK * _shrunk(placed, d, ctx)  # ad units too, at KEEP_UNIT (keep_share)
    b["density"] = -DENSITY * _crowding(boxes, d, ctx)
    if area > 0:
        mx = sum(bx.cx * bx.area for _, bx in boxes) / area
        my = sum(bx.cy * bx.area for _, bx in boxes) / area
        b["balance"] = -25.0 * abs(mx - ctx.W / 2.0) / ctx.W - 12.0 * abs(my - ctx.H / 2.0) / ctx.H
    b["hierarchy"] = HIERARCHY * _hierarchy(placed, d) - INVERSION * _inversions(placed, d) \
        - COLLAPSE * _collapsed(placed, d)
    b["prior"] = _prior(p.name.split("@")[0], ctx.W / ctx.H, d.W / d.H)
    return sum(b.values()), b


def _inversions(placed, d: Design) -> int:
    """Texts now set (more than a pixel) smaller than a text they were at least
    25% larger than."""
    out = {tid: fs for r, _, _ in placed for tid, fs in r.fonts.items()}
    src = {tid: float(d.by_id[tid].get("font_size") or 0) for tid in out if tid in d.by_id}
    return sum(1 for a in out for b in out if src[a] >= 1.25 * src[b] and out[a] < out[b] - 1)


def _crowding(boxes, d: Design, ctx: Ctx) -> float:
    """How far past a comfortable density the copy is packed: the share of
    the canvas it covers over the larger of a quarter and 1.5x the source's
    own share, less one. A small ad unit (read at a glance) that keeps every
    line of a long design reads as a wall of text; a designer drops the
    details. Large canvases have the room for big type, and strips are dense
    by nature: both are left alone."""
    r = ctx.W / max(1.0, ctx.H)
    if r >= 3.0 or r <= 1 / 3.0 or min(ctx.W, ctx.H) >= SMALL_UNIT:
        return 0.0
    src = sum(it.src_box.area for it in d.items if it.is_copy) / max(1.0, d.W * d.H)
    copy = sum(bx.area for it, bx in boxes if it.is_copy) / max(1.0, ctx.W * ctx.H)
    return max(0.0, copy / max(0.25, 1.5 * src) - 1.0)


def _collapsed(placed, d: Design) -> float:
    """Importance of the copy set level with (within 10% of) copy it was at
    least 1.5x smaller than, as the legibility floor does on tiny formats: a
    subhead the size of its headline no longer reads as a subhead, only as
    clutter, and a designer leaves it out."""
    out, item = {}, {}
    for r, _, _ in placed:
        for tid, fs in r.fonts.items():
            out[tid], item[tid] = fs, r.item
    src = {tid: float(d.by_id[tid].get("font_size") or 0) for tid in out if tid in d.by_id}
    level = {id(item[b]): item[b].importance for a in src for b in src
             if item[a] is not item[b] and src[a] >= 1.5 * src[b] and out[b] * 1.1 >= out[a]}
    return float(sum(level.values()))


def _hierarchy(placed, d: Design) -> float:
    """Whether the headline stands out from the next largest copy as a
    designer would set it on a new format, in [-1, 1]. Reflowing a design is
    when the headline becomes display type, so contrast up to 1.6x the
    source's earns up to 1; past that it gives the credit back (at 2.56x it
    is worth nothing: the rest of the copy has become an afterthought). Less
    contrast than the source (the legibility floor raising small copy) costs
    the same way down to -1."""
    def size(it, fonts=None):
        return max(fonts.values()) if fonts else it.main_fs()

    head = [r for r, _, _ in placed if r.item.role == "headline" and r.fonts]
    rest = [r for r, _, _ in placed if r.item.role != "headline" and r.fonts]
    if not head or not rest:
        return 0.0
    out = size(head[0].item, head[0].fonts) / max(size(r.item, r.fonts) for r in rest)
    src = head[0].item.main_fs() / max(1e-6, max(r.item.main_fs() for r in rest))
    x = math.log(out / max(1e-6, src)) / math.log(1.6)
    return max(-1.0, x if x <= 1.0 else 2.0 - x)


def _inflation(placed, d: Design, ctx: Ctx) -> float:
    """Secondary items blown up past their place in the hierarchy.

    On a small canvas the legibility floor can make a badge or a caption take
    many times its share of the canvas, until it rivals the headline; a
    designer would drop it instead. Each secondary item pays for the excess
    of its canvas share over 2.5x its share in the source (log scale)."""
    from .compose import CORE_ROLES

    total = 0.0
    for r, _, _ in placed:
        it = r.item
        if it.core or it.role in CORE_ROLES or it.src_box.area <= 0:
            continue
        share0 = it.src_box.area / (d.W * d.H)
        share = r.w * r.h / (ctx.W * ctx.H)
        if share > 0:
            total += it.importance * max(0.0, math.log(share / share0) - SHARE_SLACK)
    return total


SPLIT = 0.6  # x importance, for copy that now straddles two grounds (pixels.split_share) ...
SPLIT_SLACK = 0.15  # ... past the source's share plus this
DISPLAY_BUSY = 1.5  # x the busy charge, for display type (the headline): it must read at a glance


def pixel_penalty(stats, src_stats, importance, display=frozenset()) -> float:
    """Copy that lost contrast against the pixels under it, or landed on texture
    the source kept it off. Display type (``display``: the headline's text ids)
    pays DISPLAY_BUSY times as much for texture: on a photo whose quiet part is
    small, the headline is the copy that must keep to it, while a button carries
    its own fill and secondary copy may run on past it."""
    pen = 0.0
    for tid, st in stats.items():
        c, busy = st[0], st[1]
        imp = importance.get(tid, 40.0)
        s0 = src_stats.get(tid, (4.5, 0.0, 0.0))
        c0, b0 = s0[0], s0[1]
        target = max(1.5, min(c0, 4.5))
        if c < 0.85 * target:
            pen += imp * 1.5 * (1.0 - c / target)
        calm = max(1.5 * b0, 0.06)
        if busy > calm:
            pen += imp * 0.6 * min(1.0, (busy - calm) / 0.1) * (DISPLAY_BUSY if tid in display else 1.0)
        if len(st) > 2:  # straddling two grounds the source kept it on one of (split_share)
            sp = st[2] - (s0[2] if len(s0) > 2 else 0.0) - SPLIT_SLACK
            if sp > 0:
                pen += imp * SPLIT * min(1.0, sp / 0.3) * (DISPLAY_BUSY if tid in display else 1.0)
    return pen


def _shrunk(placed, d: Design, ctx: Ctx) -> float:
    """Growth into the room the design left, the other way round: the
    headline and the hero keep at least 70% of their source share of the
    canvas, and any picture that covered 8% of the source keeps 60% of its
    share (compose.keep_share), so a product never ends up a thumbnail. Each
    pays its importance times the share it is missing, as a fraction."""
    total = 0.0
    roomy = min(ctx.W, ctx.H) >= ROOMY
    for r, _, _ in placed:
        it = r.item
        if it.src_box.area <= 0:
            continue
        if it.role == "headline":
            keep = KEEP_HERO if roomy else 0.0
        else:
            keep = keep_share(it, d, ctx, unit=True)
        if not keep:
            continue
        if it.kind == "composite" and art_box(it) is not it.src_box:  # the host image, not its stickers
            a0 = art_box(it).area
            ratio = (r.scale * r.scale * a0 / (ctx.W * ctx.H)) / (a0 / (d.W * d.H))
        else:
            ratio = (r.w * r.h / (ctx.W * ctx.H)) / (it.src_box.area / (d.W * d.H))
        total += it.importance * max(0.0, keep - ratio) / keep
    return total


def _trivial(boxes, d: Design, ctx: Ctx) -> float:
    """1 for a plan that is just the source uniformly scaled into the middle
    of a canvas of another shape, with empty bands either side: every item at
    one scale (within 15%), in the source's arrangement, spanning less than
    three quarters of the axis that grew. VLM rubrics mark "trivial scaling"
    down, and the coverage term barely notices it."""
    if abs(math.log((ctx.W / ctx.H) / (d.W / d.H))) < TRIVIAL_ASPECT:
        return 0.0
    pairs = [(it, bx) for it, bx in boxes if it.src_box.w > 0 and it.src_box.h > 0]
    if len(pairs) < 2:
        return 0.0
    ks = [math.sqrt(bx.area / it.src_box.area) for it, bx in pairs]
    if max(ks) > 1.15 * min(ks):
        return 0.0
    u0 = pairs[0][0].src_box
    u = pairs[0][1]
    for it, bx in pairs[1:]:
        u0, u = u0.union(it.src_box), u.union(bx)
    if u0.w <= 0 or u0.h <= 0 or u.w <= 0 or u.h <= 0:
        return 0.0
    dev = sum(abs((bx.cx - u.x) / u.w - (it.src_box.cx - u0.x) / u0.w)
              + abs((bx.cy - u.y) / u.h - (it.src_box.cy - u0.y) / u0.h) for it, bx in pairs) / len(pairs)
    if dev > 0.08:
        return 0.0
    top, bottom = story_band(ctx)  # a story's platform UI is not room a layout could have used
    span = u.w / ctx.W if ctx.W / ctx.H > d.W / d.H else u.h / max(1.0, ctx.H - top - bottom)
    return 1.0 if span < 0.75 else 0.0

