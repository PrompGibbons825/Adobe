"""Extra candidate layout generators (anchor, flex/grid, retrieval, parameter variants).

Structure priors and flex arrangements only: every element id, copy, font and asset
stay the source's. Plans must pass the same overflow and collision checks as the
core archetypes in compose.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .analyze import Design, Item
from .compose import (
    HERO_SHARE,
    Placement,
    Plan,
    _fold_all,
    _folded,
    _settle_floaters,
    _visible,
    _zone,
    column,
    collides,
    margins,
    overflow,
    place_column,
    place_row,
    scale_bounds,
    solve,
)
from .geom import Box, el_box, union_all
from .items import Ctx, Real, realize
from .layout import Flow, HStack, Leaf, Node, VStack
from .structure import rows_for

ANCHOR_AR = math.log(1.45)  # mild aspect change: keep edge anchors
_INDEX_PATH = Path(__file__).resolve().parent / "data" / "layout_index.json"

Placement = Tuple[Real, float, float]


def _anchor_xy(b: Box, r: Real, d: Design, ctx: Ctx) -> Tuple[float, float]:
    """Penpot-style: keep distance ratio to the nearer canvas edge on each axis."""
    dl, dr = b.x / max(1.0, d.W), (d.W - b.r) / max(1.0, d.W)
    if dl <= dr:
        x = dl * ctx.W
    else:
        x = ctx.W - dr * ctx.W - r.w
    dt, db = b.y / max(1.0, d.H), (d.H - b.b) / max(1.0, d.H)
    if dt <= db:
        y = dt * ctx.H
    else:
        y = ctx.H - db * ctx.H - r.h
    return x, y


def plan_anchor(d: Design, ctx: Ctx, hidden: List[Item], name: str = "anchor", insets=(0, 0, 0, 0)) -> Optional[Plan]:
    """Each element keeps its edge-distance ratios; type is re-fit in the new box."""
    if abs(math.log((ctx.W / ctx.H) / (d.W / d.H))) > ANCHOR_AR:
        return None
    vis = _visible(d, hidden)
    if not vis:
        return None
    s = math.sqrt(ctx.W * ctx.H / max(1.0, d.W * d.H))
    zone, mx, my = _zone(d, ctx, insets)
    placed: List[Placement] = []
    for it in vis:
        b = it.src_box
        mw = max(b.w * s * 1.1, 1.0) if it.kind == "text" else zone.w
        r = realize(it, s, ctx, max_w=mw, max_h=zone.h)
        if r is None:
            return None
        x, y = _anchor_xy(b, r, d, ctx)
        if it.kind == "text":
            a = it.els[0].get("text_align", "left")
            if a == "left":
                x = b.x / d.W * ctx.W
            elif a == "right":
                x = b.r / d.W * ctx.W - r.w
        x = min(max(x, mx if it.corner is None else 0.0), ctx.W - r.w - (mx if it.corner is None else 0.0))
        y = min(max(y, my if it.corner is None else 0.0), ctx.H - r.h - (my if it.corner is None else 0.0))
        placed.append((r, x, y))
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            ri, xi, yi = placed[i]
            rj, xj, yj = placed[j]
            if Box(xi, yi, ri.w, ri.h).inter(Box(xj, yj, rj.w, rj.h)) > 0 and \
                    ri.item.src_box.inter(rj.item.src_box) <= 0:
                return None
    l, t, rr, bb = overflow(placed, ctx)
    if max(l, t, rr, bb) > 0.5:
        return None
    return Plan(name=name, placed=placed, hidden=list(hidden), s=s, zone=zone, capped=True)


def plan_flex_col(d: Design, ctx: Ctx, hidden: List[Item], name: str = "flex-col", insets=(0, 0, 0, 0),
                  gap_scale: float = 1.0, force_align: Optional[str] = None) -> Optional[Plan]:
    """Column stack with an explicit gap scale (tighter or looser than the source rhythm)."""
    vis = _visible(d, hidden)
    floaters = [it for it in vis if it.corner]
    body = [it for it in vis if not it.corner]
    if not body:
        return None
    rows = rows_for(body)
    lo, hi = scale_bounds(d, ctx, "stack")
    zone, mx, my = _zone(d, ctx, insets, rows)
    if zone.w < 20 or zone.h < 20:
        return None
    root, boxes = column(d, rows, hero_fill=True, cap_rel=1.7, min_rel=0.65, force_align=force_align,
                         gap_scale=gap_scale, hero_share=HERO_SHARE * zone.w * max(zone.h, ctx.H - 2 * my),
                         keep=(d, ctx))
    s, laid, relax = solve(root, ctx, zone, lo, hi)
    if laid is None:
        return None
    placed = place_column(laid, boxes, root.align, d, ctx, laid.zone, side=force_align)
    fl, bad, clash, new_zone = _settle_floaters(placed, floaters, s, ctx, zone, mx, my, relax)
    if bad or clash:
        return None
    return Plan(name=name, placed=placed, hidden=list(hidden), s=s, zone=zone, floaters=fl,
                relax=relax, capped=s >= hi * 0.999)


def _grid_parts(rows: List[List[Item]], ncol: int) -> List[List[Item]]:
    flat = [it for row in rows for it in row]
    if ncol <= 1 or len(flat) < ncol:
        return flat
    n = len(flat)
    ncol = min(ncol, n)
    out: List[List[Item]] = [[] for _ in range(ncol)]
    for i, it in enumerate(flat):
        out[i % ncol].append(it)
    return [c for c in out if c]


def plan_flex_grid(d: Design, ctx: Ctx, hidden: List[Item], name: str = "flex-grid", insets=(0, 0, 0, 0),
                   ncol: int = 2) -> Optional[Plan]:
    """Reading-order items in ``ncol`` columns side by side."""
    vis = _visible(d, hidden)
    floaters = [it for it in vis if it.corner]
    body = [it for it in vis if not it.corner]
    if len(body) < ncol + 1:
        return None
    cols = _grid_parts(rows_for(body), ncol)
    if len(cols) < 2:
        return None
    lo, hi = scale_bounds(d, ctx, "stack")
    zone, mx, my = _zone(d, ctx, insets)
    if zone.w < 24 or zone.h < 24:
        return None
    parts = [column(d, rows_for(c), hero_fill=False, gap_scale=0.85)[0] for c in cols]
    root = HStack(parts)
    gap = max(0.8 * ctx.base_fs, 0.03 * ctx.W)
    root.gap_px = lambda i, s, c: gap  # type: ignore[assignment]
    s, laid, relax = solve(root, ctx, zone, lo, hi)
    if laid is None:
        return None
    placed = _place_solved(laid, ctx, laid.zone if hasattr(laid, "zone") else zone)
    fl, bad, clash, _ = _settle_floaters(placed, floaters, s, ctx, zone, mx, my, relax)
    if bad or clash:
        return None
    return Plan(name=name, placed=placed, hidden=list(hidden), s=s, zone=zone, floaters=fl,
                relax=relax, capped=s >= hi * 0.999)


def plan_flex_tile(d: Design, ctx: Ctx, hidden: List[Item], name: str = "flex-tile", insets=(0, 0, 0, 0),
                   photo_frac: float = 0.5, stack_tall: bool = False) -> Optional[Plan]:
    """Photo region and copy region as two tiles (side by side or stacked)."""
    vis = _visible(d, hidden)
    imgs = [it for it in vis if it.kind in ("image", "composite") and it.role in ("hero", "image")]
    if not imgs:
        return None
    photo = max(imgs, key=lambda i: i.src_box.area)
    rest = [it for it in vis if it is not photo and not it.corner]
    if not rest:
        return None
    lo, hi = scale_bounds(d, ctx, "split")
    zone, mx, my = _zone(d, ctx, insets)
    if zone.w < 24 or zone.h < 20:
        return None
    rows = rows_for(rest)
    col, _ = column(d, rows, hero_fill=False, gap_scale=0.9)
    photo_leaf = Leaf(photo, fill=True, max_w_frac=photo_frac, cap_rel=2.0, min_rel=0.45)
    gutter = max(1.5 * ctx.base_fs, 0.03 * min(ctx.W, ctx.H))
    if stack_tall or ctx.H > ctx.W * 1.05:
        root: Node = VStack([photo_leaf, col])
        root.gap_px = lambda i, s, c: gutter  # type: ignore[assignment]
        place_fn = lambda laid, z: _place_vtile(laid, ctx, z)
    else:
        root = HStack([photo_leaf, col])
        root.gap_px = lambda i, s, c: gutter  # type: ignore[assignment]
        place_fn = lambda laid, z: place_row(laid, ctx, z)
    s, laid, relax = solve(root, ctx, zone, lo, hi)
    if laid is None:
        return None
    placed = place_fn(laid, laid.zone)
    fl, bad, clash, _ = _settle_floaters(placed, [it for it in vis if it.corner], s, ctx, zone, mx, my, relax)
    if bad or clash:
        return None
    return Plan(name=name, placed=placed, hidden=list(hidden), s=s, zone=zone, floaters=fl,
                relax=relax, capped=s >= hi * 0.999)


def _place_vtile(laid, ctx: Ctx, zone: Box) -> List[Placement]:
    parts = laid.parts  # type: ignore[attr-defined]
    gaps = laid.gap_list  # type: ignore[attr-defined]
    extra = max(0.0, zone.h - laid.h)
    add = extra / (len(gaps) + 1) if gaps else extra / 2.0
    y = zone.y + add
    x0 = zone.x + (zone.w - max(p.w for p in parts)) / 2.0
    out: List[Placement] = []
    for i, p in enumerate(parts):
        out += p.place(x0 + (zone.w - p.w) / 2.0, y)
        y += p.h + (gaps[i] + add if i < len(gaps) else 0.0)
    return out


def plan_flex_strip(d: Design, ctx: Ctx, hidden: List[Item], name: str = "flex-strip", insets=(0, 0, 0, 0),
                    wrap: bool = True) -> Optional[Plan]:
    """Banner: items in one row, optionally wrapping badge-like pieces."""
    vis = _visible(d, hidden)
    body = [it for it in vis if not it.corner]
    if not body:
        return None
    lo, hi = scale_bounds(d, ctx, "row")
    zone, mx, my = _zone(d, ctx, insets)
    if zone.w < 30 or zone.h < 10:
        return None
    leaves = [Leaf(it, fill=it.role == "hero", cap_rel=1.5, min_rel=0.35, max_w_frac=0.35) for it in body]
    root: Node = Flow(leaves, gap=max(0.6 * ctx.base_fs, 0.02 * ctx.W)) if wrap else HStack(leaves)
    s, laid, relax = solve(root, ctx, zone, lo, hi)
    if laid is None:
        return None
    z = laid.zone if hasattr(laid, "zone") else zone
    placed = laid.place(z.x + (z.w - laid.w) / 2.0, z.y + (z.h - laid.h) / 2.0)
    fl, bad, clash, _ = _settle_floaters(placed, [it for it in vis if it.corner], s, ctx, zone, mx, my, relax)
    if bad or clash:
        return None
    return Plan(name=name, placed=placed, hidden=list(hidden), s=s, zone=zone, floaters=fl,
                relax=relax, capped=s >= hi * 0.999)


def _place_solved(laid, ctx: Ctx, zone: Box) -> List[Placement]:
    z = getattr(laid, "zone", zone)
    if hasattr(laid, "parts"):
        return place_row(laid, ctx, z)
    return laid.place(z.x + (z.w - laid.w) / 2.0, z.y + (z.h - laid.h) / 2.0)


# ----------------------------------------------------------------------------
# retrieval index: designers' own resizes from Crello v5 (CDLA-Permissive-2.0), baked at dev time by
# tools/build_layout_index.py. Each family is one design at several canvas sizes with its elements
# aligned across the sizes, so a family whose source-size member looks like this design says where a
# designer moved and how much they scaled each element for this target shape.

EXCLUDE: set = set()  # dev only: family ids to leave out (the family being evaluated)
RETRIEVE_K = 3  # nearest families tried per target
SRC_AR = math.log(1.3)  # a family member counts as "the source" within this aspect ratio
TGT_AR = math.log(1.2)  # ... and as "the target" within this one


@lru_cache(maxsize=1)
def _layout_index() -> List[dict]:
    if not _INDEX_PATH.is_file():
        return []
    data = json.loads(_INDEX_PATH.read_text())
    if not isinstance(data, dict) or data.get("v") != 2:
        return []
    return data["families"]


def _slot_cost(nb: Tuple[float, float, float, float], s: Sequence[float]) -> float:
    """Distance between a normalized item box (x, y, w, h) and an index slot [x, y, w, h]."""
    cx, cy = nb[0] + nb[2] / 2, nb[1] + nb[3] / 2
    sx, sy = s[0] + s[2] / 2, s[1] + s[3] / 2
    return (abs(cx - sx) + abs(cy - sy)
            + 0.25 * abs(math.log(max(1e-3, nb[2]) / max(1e-3, s[2])))
            + 0.25 * abs(math.log(max(1e-3, nb[3]) / max(1e-3, s[3]))))


def _match_family(items: List[Tuple[Item, str, tuple]], src: dict) -> Optional[Tuple[float, Dict[int, int]]]:
    """Greedy kind-respecting match of the design's items onto a member's slots: (cost, item -> slot)."""
    slots = src["s"]
    pairs = []
    for ii, (_, kind, nb) in enumerate(items):
        for si, sl in enumerate(slots):
            if sl[4] == kind:
                pairs.append((_slot_cost(nb, sl), ii, si))
    pairs.sort()
    got: Dict[int, int] = {}
    used = set()
    cost = 0.0
    for c, ii, si in pairs:
        if ii in got or si in used:
            continue
        got[ii] = si
        used.add(si)
        cost += c
    missing = len(items) - len(got)
    if missing > max(1, len(items) // 4):
        return None
    left = sum(min(1.0, slots[si][2] * slots[si][3]) for si in range(len(slots)) if si not in used)
    return (cost + 1.0 * missing) / max(1, len(items)) + 0.5 * left, got


_MEMO: Dict[tuple, List[tuple]] = {}


def _retrieved(d: Design, ctx: Ctx, body: List[Item]) -> List[tuple]:
    """The nearest (family, source member, target member, item -> slot) for this design and target."""
    key = (id(d), ctx.W, ctx.H, tuple(id(it) for it in body), frozenset(EXCLUDE))
    if key not in _MEMO:
        if len(_MEMO) > 64:
            _MEMO.clear()
        _MEMO[key] = _retrieve(d, ctx, body)
    return _MEMO[key]


def _retrieve(d: Design, ctx: Ctx, body: List[Item]) -> List[tuple]:
    index = _layout_index()
    if not index or not body:
        return []
    ls, lt = math.log(d.W / d.H), math.log(ctx.W / ctx.H)
    items = [(it, "t" if it.is_copy else "i",
              (it.src_box.x / d.W, it.src_box.y / d.H, it.src_box.w / d.W, it.src_box.h / d.H)) for it in body]
    found = []
    for fam in index:
        if fam["id"] in EXCLUDE:
            continue
        mem = fam["m"]
        srcs = [m for m in mem if abs(math.log(m["ar"]) - ls) <= SRC_AR]
        tgts = [m for m in mem if abs(math.log(m["ar"]) - lt) <= TGT_AR]
        if not srcs or not tgts:
            continue
        best = None
        for sm in srcs:
            for tm in tgts:
                if tm is sm:
                    continue
                r = _match_family(items, sm)
                if r is None:
                    continue
                c = r[0] + 0.5 * abs(math.log(sm["ar"]) - ls) + 0.5 * abs(math.log(tm["ar"]) - lt)
                if best is None or c < best[0]:
                    best = (c, sm, tm, r[1])
        if best is not None:
            found.append((best[0], fam["id"], best[1], best[2], best[3]))
    found.sort(key=lambda f: f[0])
    return found[:RETRIEVE_K]


def plan_retrieve(d: Design, ctx: Ctx, hidden: List[Item], name: str = "retrieve", insets=(0, 0, 0, 0),
                  rank: int = 0) -> Optional[Plan]:
    """Transfer the layout of the ``rank``-th nearest designer resize: each item goes where the
    designer put its counterpart at this shape, scaled as the designer scaled it."""
    vis = _visible(d, hidden)
    body = [it for it in vis if not it.corner]
    if not body:
        return None
    found = _retrieved(d, ctx, body)
    if rank >= len(found):
        return None
    _, fid, sm, tm, got = found[rank]
    s = math.sqrt(ctx.W * ctx.H / max(1.0, d.W * d.H))
    zone, mx, my = _zone(d, ctx, insets)
    placed: List[Placement] = []
    for ii, it in enumerate(body):
        si = got.get(ii)
        if si is None:
            return None
        a, b = sm["s"][si], tm["s"][si]
        tw, th = b[2] * ctx.W, b[3] * ctx.H
        # the designer's own scale for this element, measured as the slot's size change
        k = math.sqrt(max(1e-6, tw * th) / max(1e-6, a[2] * d.W * a[3] * d.H))
        sk = it.src_box.w * it.src_box.h
        si_scale = k if sk > 0 else s
        r = realize(it, si_scale, ctx, max_w=max(tw * 1.1, 1.0), max_h=max(th * 1.3, 1.0))
        if r is None:
            r = realize(it, s, ctx, max_w=max(tw * 1.1, 1.0), max_h=max(th * 1.3, 1.0))
        if r is None:
            return None
        x = (b[0] + b[2] / 2) * ctx.W - r.w / 2.0
        y = (b[1] + b[3] / 2) * ctx.H - r.h / 2.0
        x = min(max(x, mx), ctx.W - r.w - mx)
        y = min(max(y, my), ctx.H - r.h - my)
        placed.append((r, x, y))
    if max(overflow(placed, ctx)) > 0.5:  # type: ignore[arg-type]
        return None
    return Plan(name=name, placed=placed, hidden=list(hidden), s=s, zone=zone, capped=True)


def param_stack_variants() -> List[Tuple[str, dict]]:
    return [(f"stack-pad{g:g}", dict(gap_scale=g)) for g in (0.55, 0.85, 1.15)]


def param_split_variants(has_corners: bool) -> List[Tuple[str, dict]]:
    """Extra split fractions beyond the core 0.34 / 0.48 grid."""
    out = []
    for side in ("left", "right"):
        for frac in (0.4, 0.55):
            out.append((f"split-{side}-{frac:g}", dict(hero_side=side, hero_frac=frac, fold_corners=False)))
            if has_corners:
                out.append((f"split-{side}-{frac:g}-fold",
                            dict(hero_side=side, hero_frac=frac, fold_corners=True)))
    return out


def extra_variants(d: Design, ctx: Ctx, force_all: bool) -> Iterator[Tuple[str, object, dict, Ctx]]:
    """(name, planner, kwargs, ctx) tuples wired into compose.generate."""
    from .items import BUDGET

    if BUDGET.economy:
        return
    r = ctx.W / ctx.H
    yield "anchor", plan_anchor, {}, ctx
    yield "flex-col-tight", plan_flex_col, dict(gap_scale=0.65), ctx
    if r <= 1.8 or force_all:
        yield "flex-col-start", plan_flex_col, dict(force_align="start"), ctx
    if len(_visible(d, [])) >= 5:
        yield "flex-grid-2", plan_flex_grid, dict(ncol=2), ctx
    imgs = [it for it in d.items if it.kind in ("image", "composite") and it.role in ("hero", "image")]
    if imgs and (0.4 <= r <= 2.8 or force_all):
        yield "flex-tile-0.5", plan_flex_tile, dict(photo_frac=0.5, stack_tall=r < 0.9), ctx
    if r >= 2.2 or force_all:
        yield "flex-strip", plan_flex_strip, dict(wrap=True), ctx
    if _layout_index():
        for k in range(RETRIEVE_K):
            yield f"retrieve-{k}", plan_retrieve, dict(rank=k), ctx
