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

from .analyze import Design
from .compose import Plan
from .geom import Box
from .items import Ctx

ALPHA = 1.1
LINE_CAP = {"headline": 4, "subhead": 5, "body": 6, "legal": 3}


def _prior(name: str, r: float, r0: float = 1.0) -> float:
    lr = math.log(r)
    if name.startswith("adapt"):
        # the designer's own arrangement, when the shape barely changes
        return 40.0 * max(0.0, 1.0 - abs(math.log(r / r0)) / math.log(1.2))
    if name.startswith("stack"):
        return 10.0 - 14.0 * max(0.0, lr - math.log(1.25))
    if name.startswith("split"):
        p = 17.0 - 12.0 * abs(lr - math.log(2.0))
        if "-fold" in name:
            p -= 3.0
        return p
    if name.startswith("row"):
        p = 10.0 - 18.0 * max(0.0, math.log(3.6) - lr)
        if "stacked" in name:
            p -= 2.0
        return p
    return 0.0


def _core_hidden(p: Plan) -> int:
    return sum(1 for h in p.hidden if h.role in ("headline", "cta", "logo", "hero"))


def prune_relaxed(plans: List[Plan]) -> List[Plan]:
    """Type below the legibility floor is a last resort: drop such a plan when a
    plan at full legibility keeps at least as many core items on the canvas."""
    full = [_core_hidden(p) for p in plans if p.relax >= 1.0]
    if not full:
        return plans
    best = min(full)
    return [p for p in plans if p.relax >= 1.0 or _core_hidden(p) < best]


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
    from .demote import RUNG_COST

    for r, _, _ in placed:
        it = r.item
        rel = r.scale / max(1e-9, best.get(it.key, r.scale))
        if it.kind in ("image", "composite"):
            # art only needs to be big enough; past 60% of the best size it is fine
            term = 1.0 + 0.6 * math.log(max(1e-3, min(1.0, rel / 0.6)))
        else:
            # copy should be as large as the best layout manages, within ~15%
            term = 1.0 + ALPHA * math.log(max(1e-3, min(1.0, rel / 0.85)))
        rung = p.demoted.get(it.key)
        if rung in RUNG_COST:  # demoted on purpose (demote.py): worth a fixed share, never less
            term = max(term, 1.0 - RUNG_COST[rung])
        vis += it.importance * term
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
            if blk.n_lines > 1:
                chars = sum(len(l) for l in blk.lines) / blk.n_lines
                if chars < 14:
                    lines_pen += it.importance * 0.25 * (14 - chars) / 14
    b["visible"] = vis
    b["hidden"] = -sum(h.importance for h in p.hidden if h.role != "missing")
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
    if area > 0:
        mx = sum(bx.cx * bx.area for _, bx in boxes) / area
        my = sum(bx.cy * bx.area for _, bx in boxes) / area
        b["balance"] = -25.0 * abs(mx - ctx.W / 2.0) / ctx.W - 12.0 * abs(my - ctx.H / 2.0) / ctx.H
    b["prior"] = _prior(p.name, ctx.W / ctx.H, d.W / d.H)
    return sum(b.values()), b


def pixel_penalty(stats, src_stats, importance) -> float:
    """Copy that lost contrast against the pixels under it, or landed on texture
    the source kept it off."""
    pen = 0.0
    for tid, (c, busy) in stats.items():
        imp = importance.get(tid, 40.0)
        c0, b0 = src_stats.get(tid, (4.5, 0.0))
        target = max(1.5, min(c0, 4.5))
        if c < 0.85 * target:
            pen += imp * 1.5 * (1.0 - c / target)
        calm = max(1.5 * b0, 0.06)
        if busy > calm:
            pen += imp * 0.6 * min(1.0, (busy - calm) / 0.1)
    return pen
