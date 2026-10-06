"""Optional user constraints for the finale API (Express add-on).

    pinned         id -> fixed box {x, y, width, height[, font_size]} in target px,
                   or "keep" / "position" / "size" / True (derived from the source)
    priority       id -> "must" | "nice" | "optional"
    allow_drop     False (default): every id is returned, hidden ones at opacity 0
                   (the stage-one rule). True: hidden elements are removed from
                   the output and listed as dropped (a finale-only behaviour).
    min_font_px    no visible text below this size
    safe_insets    [left, top, right, bottom] (or {left: ...} or one number); values
                   <= 1 are fractions of the canvas, larger ones px. Content stays
                   inside; backgrounds and decoration may still bleed.
    measured_text  id -> {font_size, width[, height]}: the element's natural one-line
                   width (and line height) as the host measured it at font_size.
                   Folded into per-font correction factors for the text measurer,
                   so wrapping follows the host's fonts (the settle loop).

Nothing here runs unless a caller passes constraints: `active()` is None by
default and every hook in the engine returns early, so `resize()` without
constraints is unchanged. The engine reads the active set through a context
variable, which keeps the wiring to a few marked lines in compose/items/score.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .geom import Box

PRIORITIES = ("must", "nice", "optional")

_ACTIVE: ContextVar[Optional["Constraints"]] = ContextVar("chameleon_constraints", default=None)


@dataclass
class PinSpec:
    mode: str  # box | keep | position | size
    box: Optional[Box] = None  # target px (mode == "box")
    font_size: Optional[int] = None


@dataclass
class Constraints:
    pinned: Dict[str, PinSpec] = field(default_factory=dict)
    priority: Dict[str, str] = field(default_factory=dict)
    allow_drop: bool = False
    min_font_px: float = 0.0
    safe: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # px: left, top, right, bottom
    measured_text: Dict[str, dict] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_safe(self) -> bool:
        return any(v > 0 for v in self.safe)


def active() -> Optional[Constraints]:
    return _ACTIVE.get()


@contextmanager
def using(c: Optional[Constraints]):
    token = _ACTIVE.set(c)
    try:
        yield c
    finally:
        _ACTIVE.reset(token)


# ----------------------------------------------------------------------------
# parsing


def _num(v, default=None):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _insets(raw, W: float, H: float) -> Tuple[float, float, float, float]:
    if raw is None:
        return (0.0, 0.0, 0.0, 0.0)
    if isinstance(raw, (int, float)):
        vals = [raw] * 4
    elif isinstance(raw, dict):
        vals = [raw.get(k, 0) for k in ("left", "top", "right", "bottom")]
    else:
        vals = list(raw)[:4] + [0] * max(0, 4 - len(list(raw)))
    out = []
    for i, v in enumerate(vals):
        f = max(0.0, _num(v, 0.0) or 0.0)
        ref = W if i in (0, 2) else H
        out.append(f * ref if f <= 1.0 else f)
    l, t, r, b = out
    # never leave less than a third of the canvas
    if l + r > 0.66 * W:
        k = 0.66 * W / (l + r)
        l, r = l * k, r * k
    if t + b > 0.66 * H:
        k = 0.66 * H / (t + b)
        t, b = t * k, b * k
    return (l, t, r, b)


def _pin(v, warnings: List[str], key: str) -> Optional[PinSpec]:
    if v is True or v in ("keep", "both", "lock", "locked"):
        return PinSpec("keep")
    if v in ("position", "size"):
        return PinSpec(v)
    if isinstance(v, dict):
        x, y, w, h = (_num(v.get(k)) for k in ("x", "y", "width", "height"))
        fs = _num(v.get("font_size"))
        if None in (x, y, w, h) or w <= 0 or h <= 0:
            warnings.append(f"pin {key}: needs x, y, width, height > 0; ignored")
            return None
        return PinSpec("box", Box(x, y, w, h), int(round(fs)) if fs and fs > 0 else None)
    if v in (None, False):
        return None
    warnings.append(f"pin {key}: unknown value {v!r}; ignored")
    return None


def parse(raw: Optional[dict], target: dict) -> Optional[Constraints]:
    """A Constraints object, or None when nothing in `raw` would change the result."""
    if not raw:
        return None
    if isinstance(raw, Constraints):
        return raw
    W, H = float(target["width"]), float(target["height"])
    c = Constraints()
    for k, v in sorted((raw.get("pinned") or {}).items()):
        p = _pin(v, c.warnings, k)
        if p is not None:
            c.pinned[str(k)] = p
    for k, v in sorted((raw.get("priority") or {}).items()):
        v = str(v).lower()
        if v in PRIORITIES:
            c.priority[str(k)] = v
        else:
            c.warnings.append(f"priority {k}: {v!r} is not one of {PRIORITIES}; ignored")
    c.allow_drop = bool(raw.get("allow_drop", False))
    c.min_font_px = max(0.0, _num(raw.get("min_font_px"), 0.0) or 0.0)
    c.safe = _insets(raw.get("safe_insets"), W, H)
    for k, v in sorted((raw.get("measured_text") or {}).items()):
        if isinstance(v, dict) and _num(v.get("font_size")) and _num(v.get("width")):
            c.measured_text[str(k)] = dict(v)
        else:
            c.warnings.append(f"measured_text {k}: needs font_size and width; ignored")
    return c


# ----------------------------------------------------------------------------
# hooks the engine calls (each returns the default when nothing is active)


def has_safe() -> bool:
    c = _ACTIVE.get()
    return c is not None and c.has_safe


def min_font() -> float:
    c = _ACTIVE.get()
    return c.min_font_px if c is not None else 0.0


def safe_margins(mx: float, my: float) -> Tuple[float, float, float, float]:
    """Content margins (left, top, right, bottom): the engine's own breathing room,
    or the safe area plus half of it, whichever is larger."""
    c = _ACTIVE.get()
    if c is None or not c.has_safe:
        return mx, my, mx, my
    l, t, r, b = c.safe
    return max(mx, l + 0.5 * mx), max(my, t + 0.5 * my), max(mx, r + 0.5 * mx), max(my, b + 0.5 * my)


def clamp_safe(x: float, y: float, w: float, h: float, W: float, H: float) -> Tuple[float, float]:
    c = _ACTIVE.get()
    if c is None or not c.has_safe:
        return x, y
    l, t, r, b = c.safe
    x = min(max(x, l), max(l, W - r - w))
    y = min(max(y, t), max(t, H - b - h))
    return x, y


def apply_priorities(d) -> None:
    """Priorities and pins change how much an item matters: a must (or pinned)
    item is never hidden, an optional one goes first."""
    c = _ACTIVE.get()
    if c is None or not (c.priority or c.pinned):
        return
    for it in d.items:
        ps = [c.priority[i] for i in it.ids if i in c.priority]
        if any(i in c.pinned for i in it.ids):
            ps.append("must")
        if "must" in ps:
            it.core = True
            it.importance = max(it.importance, 95.0)
            it.priority = "must"  # type: ignore[attr-defined]
        elif "optional" in ps:
            it.core = False
            it.importance = min(it.importance, 8.0)
            it.priority = "optional"  # type: ignore[attr-defined]
        elif "nice" in ps:
            it.priority = "nice"  # type: ignore[attr-defined]


def is_core(it, core_roles: Sequence[str]) -> bool:
    """Core roles are only given up when nothing else worked, unless the user said optional."""
    return it.role in core_roles and getattr(it, "priority", None) != "optional"


def adjust_drop_orders(d, orders: List[list]) -> List[list]:
    c = _ACTIVE.get()
    if c is None or not (c.priority or c.pinned):
        return orders
    must = {id(it) for it in d.items if getattr(it, "priority", None) == "must"}
    opt = sorted((it for it in d.items if getattr(it, "priority", None) == "optional"
                  and it.role not in ("missing", "void")), key=lambda it: (it.importance, it.key))
    seq: List[list] = [[]]
    acc: list = []
    for it in opt:
        acc = acc + [it]
        seq.append(list(acc))
    seen_acc = {id(i) for i in acc}
    for s in orders[1:]:
        seq.append(acc + [i for i in s if id(i) not in seen_acc])
    out: List[list] = []
    seen = set()
    for s in seq:
        s = [i for i in s if id(i) not in must]
        key = tuple(sorted(i.key for i in s))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# ----------------------------------------------------------------------------
# pins


def has_pins() -> bool:
    c = _ACTIVE.get()
    return c is not None and bool(c.pinned)


def _pin_for(it, c: Constraints) -> Optional[PinSpec]:
    for i in it.ids:
        if i in c.pinned:
            return c.pinned[i]
    return None


def _derived_box(it, spec: PinSpec, d, ctx) -> Box:
    """keep / position / size: the source box carried over, anchored to the
    nearest canvas edges (springs and struts)."""
    from .geom import el_box, union_all

    b = union_all(el_box(e) for e in it.els)  # the element boxes, as a host shows them
    sx, sy = ctx.W / d.W, ctx.H / d.H
    s = min(sx, sy) if spec.mode != "size" else 1.0
    w, h = b.w * s, b.h * s
    if b.cx < 0.4 * d.W:
        x = b.x * sx
    elif b.cx > 0.6 * d.W:
        x = ctx.W - (d.W - b.r) * sx - w
    else:
        x = b.cx * sx - w / 2.0
    if b.cy < 0.3 * d.H:
        y = b.y * sy
    elif b.cy > 0.7 * d.H:
        y = ctx.H - (d.H - b.b) * sy - h
    else:
        y = b.cy * sy - h / 2.0
    x = min(max(0.0, x), max(0.0, ctx.W - w))
    y = min(max(0.0, y), max(0.0, ctx.H - h))
    return Box(x, y, w, h)


def realize_pin(it, spec: PinSpec, d, ctx):
    """(Real, x, y) for a pinned item: the largest rendition that fits its box."""
    from . import textfit
    from .items import Real, realize
    from .layout import search_scale

    box = spec.box if spec.mode == "box" else _derived_box(it, spec, d, ctx)
    if it.kind == "text" and spec.font_size:
        el = it.els[0]
        style = it.style or textfit.TextStyle.of(el)
        blk = textfit.measure(str(el.get("content", "")), style, spec.font_size, box.w, exact=True)
        w, h = float(box.w), float(blk.height)

        def emit(x, y, el=el, w=w, h=h, fs=blk.font_size):
            e = {k: v for k, v in el.items() if k != "_idx"}
            e.update(x=x, y=y, width=w, height=h, font_size=fs)
            return [e]

        r = Real(item=it, w=w, h=h, scale=blk.font_size / max(1.0, it.fs0), fonts={el["id"]: blk.font_size},
                 blocks={el["id"]: blk}, emit_fn=emit)
        return r, box.x, box.y
    from .geom import el_box, union_all

    def fits(s):  # the pin box holds the element boxes (transparent margins included)
        r = realize(it, s, ctx)
        if r is None:
            return None
        u = union_all(el_box(e) for e in r.emit(0.0, 0.0)) or Box(0, 0, r.w, r.h)
        return (r, u) if u.w <= box.w + 0.5 and u.h <= box.h + 0.5 else None

    s, res = search_scale(fits, 0.01, 8.0)
    if res is None:
        return None
    r, u = res
    return r, box.x + (box.w - u.w) / 2.0 - u.x, box.y + (box.h - u.h) / 2.0 - u.y


def pins_for(d, ctx) -> List[tuple]:
    """Realized pins for this design and canvas, computed once per run."""
    c = _ACTIVE.get()
    cache = getattr(d, "_pins", None)
    if cache is not None and cache[0] == (ctx.W, ctx.H):
        return cache[1]
    out = []
    if c is not None:
        for it in d.items:
            spec = _pin_for(it, c)
            if spec is None or it.role in ("missing", "void"):
                continue
            p = realize_pin(it, spec, d, ctx)
            if p is None:
                c.warnings.append(f"pin {it.key}: nothing fits the pinned box; laid out freely")
                continue
            out.append(p)
    d._pins = ((ctx.W, ctx.H), out)
    return out


def free_rects(M: Box, obstacles: List[Box], gap: float, k: int = 3) -> List[Box]:
    """The k largest rectangles inside M clear of every obstacle (maximal, distinct)."""
    obs = [o.expanded(gap) for o in obstacles if o.inter(M.expanded(gap)) > 0]
    if not obs:
        return [M]
    xs = sorted({M.x, M.r} | {min(max(v, M.x), M.r) for o in obs for v in (o.x, o.r)})
    ys = sorted({M.y, M.b} | {min(max(v, M.y), M.b) for o in obs for v in (o.y, o.b)})
    cands = []
    for i, x0 in enumerate(xs):
        for x1 in xs[i + 1:]:
            for j, y0 in enumerate(ys):
                for y1 in ys[j + 1:]:
                    R = Box(x0, y0, x1 - x0, y1 - y0)
                    if R.w < 20 or R.h < 20 or any(R.inter(o) > 0 for o in obs):
                        continue
                    cands.append(R)
    cands.sort(key=lambda R: (-R.area, R.x, R.y))
    out: List[Box] = []
    for R in cands:
        if any(R.x >= P.x - 0.5 and R.y >= P.y - 0.5 and R.r <= P.r + 0.5 and R.b <= P.b + 0.5 for P in out):
            continue
        out.append(R)
        if len(out) >= k:
            break
    return out


def plan_pinned(fn, d, ctx, hidden, name: str, kw: dict, with_insets, zone_of):
    """Plan around the pinned items: they keep their boxes, everything else is
    laid out in the largest free regions next to them (best scale wins)."""
    pins = pins_for(d, ctx)
    if not pins:
        return with_insets(fn, d, ctx, hidden, name=name, **kw)
    pinned_items = [r.item for r, _, _ in pins]
    pid = {id(i) for i in pinned_items}
    hidden2 = list(hidden) + [i for i in pinned_items if id(i) not in {id(h) for h in hidden}]
    M = zone_of(d, ctx, (0.0, 0.0, 0.0, 0.0))[0]
    gap = 0.8 * ctx.base_fs
    best = None
    for R in free_rects(M, [Box(x, y, r.w, r.h) for r, x, y in pins], gap):
        base = (R.x - M.x, R.y - M.y, M.r - R.r, M.b - R.b)
        p = with_insets(fn, d, ctx, hidden2, name=name, base=base, **kw)
        if p is None:
            continue
        if best is None or p.s > best.s * 1.0001:
            best = p
    if best is None:
        return None
    best.hidden = [h for h in best.hidden if id(h) not in pid]
    best.floaters = _clear_corners(best, pins, d, ctx) + list(pins)
    best.pinned = [i.key for i in pinned_items]  # type: ignore[attr-defined]
    return best


def _clear_corners(plan, pins, d, ctx) -> list:
    """Corner items that landed on a pinned box move to the nearest free corner."""
    from .compose import collides, place_floater, margins

    pboxes = [Box(x, y, r.w, r.h) for r, x, y in pins]
    taken = [Box(x, y, r.w, r.h) for r, x, y in plan.placed]
    gap = 0.4 * ctx.base_fs
    mx, my = margins(d, ctx)
    out = []
    for f in plan.floaters:
        r, x, y = f
        fb = Box(x, y, r.w, r.h)
        if not any(collides(fb, pb, gap) for pb in pboxes):
            out.append(f)
            taken.append(fb)
            continue
        it = r.item
        c0 = it.corner or "tr"
        moved = None
        for c in (c0[0] + ("r" if c0[1] == "l" else "l"), ("b" if c0[0] == "t" else "t") + c0[1],
                  ("b" if c0[0] == "t" else "t") + ("r" if c0[1] == "l" else "l")):
            it.corner = c
            try:
                p = place_floater(it, plan.s, ctx, mx, my, plan.relax)
            finally:
                it.corner = c0
            if p is None:
                continue
            nb = Box(p[1], p[2], p[0].w, p[0].h)
            if not any(collides(nb, o, gap) for o in pboxes + taken):
                moved = p
                break
        out.append(moved or f)
        taken.append(Box(*(moved or f)[1:], (moved or f)[0].w, (moved or f)[0].h))
    return out


# ----------------------------------------------------------------------------
# scoring


def score_terms(p, d, ctx, boxes) -> Dict[str, float]:
    """Extra score terms under constraints: content outside the safe area."""
    c = _ACTIVE.get()
    if c is None or not c.has_safe:
        return {}
    l, t, r, b = c.safe
    safe = Box(l, t, ctx.W - l - r, ctx.H - t - b)
    out = 0.0
    for it, bx in boxes:
        if bx.area <= 0:
            continue
        out += (bx.area - bx.inter(safe)) / bx.area * it.importance / 50.0
    return {"safe_area": -60.0 * out} if out > 0 else {}


# ----------------------------------------------------------------------------
# measured text -> per-font correction


def text_corrections(source: dict, c: Optional[Constraints]) -> Dict[tuple, Tuple[float, float]]:
    """(family, weight, style) -> (width factor, height factor), the median over
    the measured elements of that font."""
    if c is None or not c.measured_text:
        return {}
    from . import textfit
    from .analyze import normalize

    by_id = {e.get("id"): normalize(e, i) for i, e in enumerate(source.get("elements") or [])}
    acc: Dict[tuple, List[Tuple[float, float]]] = {}
    for eid, m in sorted(c.measured_text.items()):
        el = by_id.get(eid)
        if el is None or el.get("type") != "text":
            c.warnings.append(f"measured_text {eid}: not a text element; ignored")
            continue
        st = textfit.TextStyle.of(el)
        fs = max(1, int(_num(m.get("font_size"), 12)))
        content = str(el.get("content", ""))
        nat = textfit.measure(content, st, fs, 1e9, exact=True)
        first = content.split("\n")[0]
        if nat.ink_width <= 0:
            continue
        kw = _num(m.get("width"), 0.0) / max(1e-6, max(nat.line_widths) if "\n" in content else nat.ink_width)
        kh = 1.0
        if _num(m.get("height")):
            one = textfit.measure(first, st, fs, 1e9, exact=True)
            kh = _num(m.get("height")) / max(1e-6, one.height)
        if 0.3 <= kw <= 3.0 and 0.3 <= kh <= 3.0:
            acc.setdefault((st.family, st.weight, st.style), []).append((kw, kh))
        else:
            c.warnings.append(f"measured_text {eid}: factor {kw:.2f} implausible; ignored")
    out = {}
    for k, v in acc.items():
        ws = sorted(a for a, _ in v)
        hs = sorted(b for _, b in v)
        out[k] = (ws[len(ws) // 2], hs[len(hs) // 2])
    return out
