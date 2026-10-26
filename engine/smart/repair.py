"""Deterministic fault repair on the best layout candidate before emit."""

from __future__ import annotations

import copy
from typing import Callable, List, Optional, Tuple

from .analyze import Design
from .compose import Plan
from .geom import Box
from .items import Ctx, realize_text
from .textfit import drawn_box


def maybe_repair(plans: List[Plan], d: Design, ctx: Ctx, source: dict, emit_fn: Callable,
                 rescore: Callable[[Plan], None]) -> None:
    """Try one repair per fault on the leading plan; keep it when score improves."""
    if not plans:
        return
    p = plans[0]
    faults = _faults(p, d, ctx, source, emit_fn)
    if not faults:
        return
    base = p.score
    best = p
    for kind in faults:
        trial = _repair(kind, best, d, ctx)
        if trial is None:
            continue
        rescore(trial)
        if trial.score > best.score + 0.5:
            best = trial
    if best is not p and best.score > base + 0.5:
        plans[0] = best


def _faults(plan: Plan, d: Design, ctx: Ctx, source: dict, emit_fn: Callable) -> Tuple[str, ...]:
    els = emit_fn(plan, d, ctx)
    kinds: List[str] = []
    canvas = Box(-0.5, -0.5, ctx.W + 1.0, ctx.H + 1.0)
    for e in els:
        if e.get("type") != "text" or float(e.get("opacity", 1) or 0) <= 0:
            continue
        if drawn_box(e).inter(canvas) < drawn_box(e).area - 1.0:
            kinds.append("overflow")
            break
    from . import pixels
    from .score import pixel_penalty

    if pixels.available():
        src = pixels.measure(d.elements, d.W, d.H, skip=set())
        imp = {}
        for r, _, _ in plan.placed + plan.floaters:
            for e in r.item.els:
                imp[e["id"]] = r.item.importance
        stats = pixels.measure(els, ctx.W, ctx.H, skip=set())
        if pixel_penalty(stats, src, imp) > 14.0:
            kinds.append("busy_photo")
    try:
        from tools.audit import flags

        tgt = {"width": ctx.W, "height": ctx.H}
        fl = flags(source, {"canvas": tgt, "elements": els}, tgt)
        if any(f.startswith("empty ") for f in fl):
            kinds.append("blank_band")
    except Exception:
        pass
    return tuple(dict.fromkeys(kinds))


def _repair(kind: str, plan: Plan, d: Design, ctx: Ctx) -> Optional[Plan]:
    if kind == "overflow":
        return _shrink_to_fit(plan, d, ctx)
    if kind == "busy_photo":
        return _nudge_copy(plan, d, ctx)
    if kind == "blank_band":
        return _nudge_copy(plan, d, ctx, dy=-8.0)
    return None


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


def _nudge_copy(plan: Plan, d: Design, ctx: Ctx, dy: float = 0.0) -> Optional[Plan]:
    from . import pixels
    from . import vision

    if not pixels.available() or not vision.photo_subject(d):
        return None
    p = copy.deepcopy(plan)
    emit_mod = __import__("engine.smart.__init__", fromlist=["emit"])
    els = emit_mod.emit(p, d, ctx)
    arr, f = pixels.backdrop(els, ctx.W, ctx.H)
    if arr is None:
        return None
    if dy == 0.0:
        best_dy, best_pen = 0.0, 1e9
        for try_dy in range(-15, 16, 3):
            pen = 0.0
            for r, x, y in p.placed:
                if r.item.role not in ("headline", "subhead"):
                    continue
                for e in r.item.els:
                    e2 = dict(e)
                    e2["y"] = y + try_dy
                    st = pixels.text_stats(arr, f, e2)
                    if st:
                        pen += st[1] * r.item.importance
            if pen < best_pen:
                best_pen, best_dy = pen, float(try_dy)
        dy = best_dy
    if abs(dy) < 1.0:
        return None
    new_placed = []
    for r, x, y in p.placed:
        if r.item.role in ("headline", "subhead"):
            new_placed.append((r, x, min(max(y + dy, plan.zone.y), plan.zone.b - r.h)))
        else:
            new_placed.append((r, x, y))
    p.placed = new_placed
    return p
