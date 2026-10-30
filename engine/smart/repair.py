"""Hard-constraint repair on the chosen layout (switch: ``repair``).

Post-hoc and deterministic, and only for a hard constraint: copy drawn past the canvas edge.
The scorer's choice is never re-ranked here; a trial replaces the leader only when it removes the
overflow. Aesthetic preferences (copy off a busy photo, quiet zones) live in the candidate
generator and the scorer, not here.
"""

from __future__ import annotations

import copy
from typing import Callable, List, Optional

from .analyze import Design
from .compose import Plan
from .geom import Box
from .items import Ctx, realize_text
from .textfit import drawn_box


def repair_overflow(plans: List[Plan], d: Design, ctx: Ctx, emit_fn: Callable) -> None:
    """Shrink the leading plan's overflowing copy until it is drawn inside the canvas."""
    if not plans or _overflow(plans[0], d, ctx, emit_fn) <= 0.0:
        return
    best, worst = plans[0], _overflow(plans[0], d, ctx, emit_fn)
    for _ in range(4):  # a step at a time, each a font size or so smaller
        trial = _shrink_to_fit(best, d, ctx)
        if trial is None:
            break
        o = _overflow(trial, d, ctx, emit_fn)
        if o >= worst:
            break
        best, worst = trial, o
        if o <= 0.0:
            break
    if best is not plans[0]:
        plans[0] = best


def _overflow(plan: Plan, d: Design, ctx: Ctx, emit_fn: Callable) -> float:
    """Area of visible copy drawn past the canvas (0 when none)."""
    canvas = Box(-0.5, -0.5, ctx.W + 1.0, ctx.H + 1.0)
    out = 0.0
    for e in emit_fn(plan, d, ctx):
        if e.get("type") != "text" or float(e.get("opacity", 1) or 0) <= 0:
            continue
        b = drawn_box(e)
        out += max(0.0, b.area - b.inter(canvas))
    return out


def _shrink_to_fit(plan: Plan, d: Design, ctx: Ctx) -> Optional[Plan]:
    p = copy.deepcopy(plan)
    canvas = Box(0, 0, ctx.W, ctx.H)
    changed = False
    for i, (r, x, y) in enumerate(p.placed):
        if r.item.kind != "text":
            continue
        if Box(x, y, r.w, r.h).r <= canvas.r + 0.5 and Box(x, y, r.w, r.h).b <= canvas.b + 0.5:
            continue
        fs = min(r.fonts.values()) if r.fonts else None
        if fs is None or fs <= 7:
            continue
        nr = realize_text(r.item, r.scale * 0.9, ctx, max_w=max(1.0, r.w - 2), max_h=r.h, fs=max(7, fs - 1))
        if nr:
            p.placed[i] = (nr, x, y)
            changed = True
    return p if changed else None
