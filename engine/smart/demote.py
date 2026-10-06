"""Feasibility first, a demotion ladder instead of dropping, repair instead of fit.

Cramped formats (strips, skyscrapers, small ads) used to be handled in two ways: drop
low-priority items with opacity 0, and, when no candidate passed the hard checks, fall
back to fit-and-centre, which only scales the source (exactly Adobe's "fit" failure).
This module replaces both.

1. Feasibility first (`prepare`, `budget`). Before any archetype runs, the design's load
   is measured against the target: every text's em advance at its legibility floor, and
   every rigid unit's floor footprint, against the content box and its line capacity.
   - A rigid unit that cannot fit even at its floor (a panel of copy on a block, a photo
     with captions) is dissolved into its parts: the copy joins the flow, the block is
     redrawn behind wherever the copy lands.
   - A strip whose copy cannot stack in one column at the floor gets rows with two or
     three copy columns side by side, and a hero thumbnail when the hero leaves no room.
   - The ladder starts at the first state whose load can physically fit.

2. A demotion ladder (`units`, `ladder`), after Google's arbitrary-size patent
   (US11687707B2): anchors, priority growth, and conflict resolution by removing the
   lowest-priority element. Chameleon may not remove ids, and a hidden element can read
   as lost brand content, so the lowest-priority element is demoted one rung at a time:
       floor   shrink it to its legibility floor (art: a thumbnail)
       zone    move it out of the main flow into a secondary zone: a bottom legal strip,
               an end column on strips, a free corner
       chips   (a set of badges) compact it into one chip row or column at the floor
       hide    only then: opacity 0, never the headline, logo, CTA or hero
   Each element goes floor -> zone/chips before the next one is touched; hiding starts
   only when no state short of it gives a clean plan (legible type, every core item
   shown). Every step is recorded in Plan.demoted and Plan.log.

3. Repair instead of fit-and-centre (`repair`, `generate`). When every candidate draws
   copy over copy, the best candidate is repaired: the lower-priority text of each clash
   goes to its floor, then to a free spot, and only a non-core one is hidden. When no
   candidate exists at all, every rigid unit carrying copy is dissolved and the ladder
   climbed again; core art (hero, then logo) is given up only after that. Fit-and-centre
   is left for content that cannot fit at all; STATS counts what happened.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import textfit
from .analyze import Backdrop, Design, Item
from .geom import Box, el_box, union_all
from .items import Ctx, Real, font_for, pinned_scale, realize
from .textfit import TextStyle

ENABLED = True
STATS: Dict[str, object] = {}  # per plan_design() call: log, dissolved, repair, core_hidden

CORE = ("headline", "cta", "logo", "hero")
# share of an item's value a demotion costs in score.py (hiding costs all of it)
RUNG_COST = {"floor": 0.25, "zone": 0.35, "chips": 0.3}
RHO = 0.9  # packing efficiency of the area budget: optimistic, so only what clearly cannot fit is skipped
SLACK = 1.12  # word-wrap slack on a text's area
EXTRA_STATES = 3  # demotion states tried past the first that gives a clean plan
REPAIR_TRIES = 4  # candidates the copy-over-copy repair works on
# implicitly hidden by an archetype (no slot for it): may still go to a free spot at its floor
RESCUE = ("legal", "body", "subhead", "badge", "ribbon", "group", "watermark", "panel")
WHY = {
    "floor": "shrunk to its legibility floor",
    "zone": "moved out of the main flow to a secondary zone at its floor",
    "chips": "compacted into a chip row/column at the floor",
    "hide": "hidden (opacity 0): no state short of it fits",
}


def _log(msg: str) -> None:
    STATS.setdefault("log", []).append(msg)  # type: ignore[union-attr]


def _bump(key: str, n: int = 1) -> None:
    STATS[key] = int(STATS.get(key, 0) or 0) + n  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# text load


def em_advance(el: dict) -> float:
    """Sum of a text's glyph advances in ems (renderer-exact, letter spacing included)."""
    blk = textfit.measure(str(el.get("content", "")), TextStyle.of(el), 100, 1e9)
    return sum(blk.line_widths) / 100.0


def floor_font(it: Item, ctx: Ctx) -> int:
    return font_for(it.fs0, 0.01, it.role, ctx)


def content_box(d: Design, ctx: Ctx) -> Box:
    from .compose import margins

    mx, my = margins(d, ctx)
    return Box(mx, my, ctx.W - 2 * mx, ctx.H - 2 * my)


def _floor_real(it: Item, ctx: Ctx, max_w: float = math.inf, max_h: float = math.inf,
                mode: str = "stack", nowrap: bool = False) -> Optional[Real]:
    pc = ctx if it.key in ctx.pinned else dataclasses.replace(ctx, pinned=ctx.pinned | {it.key})
    return realize(it, 1.0, pc, max_w, max_h, mode=mode, nowrap=nowrap)


def chip_real(it: Item, ctx: Ctx, max_w: float, max_h: float) -> Optional[Real]:
    """A badge compacted to a chip: its label at the floor with tight padding (a button
    at its floor otherwise keeps the source's generous padding)."""
    r = _floor_real(it, ctx, max_w, max_h, nowrap=True) or _floor_real(it, ctx, max_w, max_h)
    if r is None or it.kind != "group" or not it.labels or not r.blocks:
        return r
    ink = max(b.ink_width for b in r.blocks.values())
    f = max(r.fonts.values())
    tight = _floor_real(it, ctx, min(max_w, ink + 1.2 * f), min(max_h, 1.9 * f), nowrap=True)
    return tight if tight is not None and tight.w * tight.h < r.w * r.h else r


def floor_area(it: Item, ctx: Ctx, zone: Box, chip: bool = False) -> Optional[float]:
    """Area an item needs at its floor; None when it cannot fit the content box at all."""
    if it.kind == "text":
        f = floor_font(it, ctx)
        content = str(it.els[0].get("content", ""))
        if textfit.longest_word(content, it.style or TextStyle.of(it.els[0]), f) > zone.w + 0.5:
            return None
        lh = (it.style.line_height if it.style else 1.2)
        return em_advance(it.els[0]) * f * f * lh * SLACK
    r = chip_real(it, ctx, zone.w, zone.h) if chip else _floor_real(it, ctx, zone.w, zone.h)
    return None if r is None else r.w * r.h


def _carries_copy(it: Item) -> bool:
    if it.kind == "group" and it.role == "panel" and len(it.parts) > 1:
        return any(p.kind == "text" for p in it.parts)
    return it.kind == "composite" and any(s.kind == "text" for s in it.sats)


# ----------------------------------------------------------------------------
# feasibility first


@dataclass
class Budget:
    zone: Box
    need: List[float] = field(default_factory=list)  # px2 at the floors, per ladder state
    avail: float = 0.0  # px2 the content box offers
    start: int = 0  # first ladder state that can physically fit
    strip: bool = False
    cols: int = 1  # strips: copy columns needed at the floor (0: none of 1-3 fits)
    thumb: bool = False  # strips: the hero should shrink to a thumbnail
    thumb_cols: int = 1


def prepare(d: Design, ctx: Ctx) -> None:
    """Before any archetype: dissolve every rigid unit that cannot fit the content box
    even at its floor. Its copy goes back into the flow; its block becomes a backdrop."""
    STATS.clear()
    STATS["log"] = []
    if not ENABLED:
        return
    zone = content_box(d, ctx)
    for it in list(d.items):
        if it.role in ("missing", "void") or not _carries_copy(it):
            continue
        if _floor_real(it, ctx, zone.w, zone.h) is None and dissolve(d, it):
            _log(f"feasibility: {it.key} ({it.role}) cannot fit the {zone.w:.0f}x{zone.h:.0f} content box "
                 f"even at its floor: dissolved into its parts")


def dissolve(d: Design, it: Item) -> bool:
    """Replace a rigid unit by its parts. A panel's block becomes a backdrop redrawn behind
    its copy; a photo with captions becomes the photo plus free captions."""
    if it not in d.items:
        return False
    if it.kind == "group" and len(it.parts) > 1:
        host, rest = it.parts[0], list(it.parts[1:])
        hb = el_box(host.els[0])
        u = union_all(m.src_box for m in rest) or hb
        d.backdrops.append(Backdrop(
            el=host.els[0], members=rest,
            edges=(hb.x <= 2, hb.y <= 2, hb.r >= d.W - 2, hb.b >= d.H - 2),
            pad=(max(0.0, u.x - hb.x), max(0.0, u.y - hb.y), max(0.0, hb.r - u.r), max(0.0, hb.b - u.b))))
        parts = rest
    elif it.kind == "composite" and it.host is not None:
        parts = [it.host] + list(it.sats)
    else:
        return False
    i = d.items.index(it)
    d.items[i:i + 1] = parts
    for bd in d.backdrops:
        if any(m is it for m in bd.members):
            bd.members = [m for m in bd.members if m is not it] + parts
    STATS.setdefault("dissolved", []).append(it.key)  # type: ignore[union-attr]
    return True


def ratio_box(box: Box, src: Box, ctx: Ctx) -> Optional[Box]:
    """`box` grown to the proportions of `src` (an image block keeps its ratio), kept on
    the canvas; None if it no longer fits."""
    r = src.w / max(1e-6, src.h)
    w, h = box.w, box.h
    if w / max(1e-6, h) < r:
        w = h * r
    else:
        h = w / r
    if w > ctx.W + 0.5 or h > ctx.H + 0.5:
        return None
    x = min(max(0.0, box.cx - w / 2.0), ctx.W - w)
    y = min(max(0.0, box.cy - h / 2.0), ctx.H - h)
    return Box(x, y, w, h)


def budget(d: Design, ctx: Ctx, us: List["Unit"], seq: List[Dict[int, str]], missing: Sequence[Item]) -> Budget:
    zone = content_box(d, ctx)
    b = Budget(zone=zone, avail=zone.area * RHO)
    gone = {id(m) for m in missing}
    area, chip = {}, {}
    chips = {id(it) for u in us if "chips" in u.rungs for it in u.items}
    for it in d.items:
        if id(it) in gone or it.role in ("missing", "void"):
            continue
        a = floor_area(it, ctx, zone)
        if a is None:  # cannot fit at all: the archetypes leave it out, it is no load
            continue
        area[id(it)] = a
        if id(it) in chips:
            chip[id(it)] = floor_area(it, ctx, zone, chip=True) or a
    for rungs in seq:
        hid = {id(it) for i, r in rungs.items() if r == "hide" for it in us[i].items}
        chipped = {id(it) for i, r in rungs.items() if r == "chips" for it in us[i].items}
        b.need.append(sum(chip[k] if k in chipped else a for k, a in area.items() if k not in hid))
    b.start = next((i for i, n in enumerate(b.need) if n <= b.avail), 0)
    r = ctx.W / ctx.H
    b.strip = r >= 3.5
    if b.strip:
        vis = [it for it in d.items if id(it) not in gone and it.role not in ("missing", "void")]
        b.cols, b.thumb, b.thumb_cols = _strip_columns(d, ctx, zone, vis)
    return b


def _strip_columns(d: Design, ctx: Ctx, zone: Box, vis: List[Item]) -> Tuple[int, bool, int]:
    """Copy columns a strip needs so every row fits the zone height at its floor, with the
    hero at its usual width and as a thumbnail: (cols, thumb helps, cols with thumb)."""
    from .structure import rows_for

    logo = next((i for i in vis if i.role == "logo"), None)
    hero = next((i for i in vis if i.role == "hero"), None)
    ctas = [i for i in vis if i.role == "cta"]
    copy = [i for i in vis if i.kind == "text" or (i.role in ("badge", "panel", "group") and not i.corner)]
    rows = rows_for(copy)
    if not rows:
        return 1, False, 1
    gap = max(1.2 * ctx.base_fs, 0.022 * ctx.W)
    fixed, n_fixed = 0.0, 0
    for it in ([logo] if logo else []) + ctas:
        rr = _floor_real(it, ctx, zone.w, zone.h, nowrap=it.role == "cta") or _floor_real(it, ctx, zone.w, zone.h)
        if rr is not None:
            fixed += rr.w
            n_fixed += 1

    def row_h(row: List[Item], w: float) -> float:
        hs = []
        for it in row:
            if it.kind == "text":
                f = floor_font(it, ctx)
                content = str(it.els[0].get("content", ""))
                style = it.style or TextStyle.of(it.els[0])
                ww = w / len(row)
                if textfit.longest_word(content, style, f) > ww:
                    return math.inf
                hs.append(textfit.measure(content, style, f, ww).height)
            else:
                rr = _floor_real(it, ctx, w, zone.h)
                if rr is None:
                    return math.inf
                hs.append(rr.h)
        if all(it.kind != "text" for it in row) and len(row) > 1:  # a wrapping chip row
            ws = [(_floor_real(it, ctx, w, zone.h) or Real(it, w, 0, 0)).w for it in row]
            lines = max(1, math.ceil((sum(ws) + 0.8 * ctx.base_fs * (len(ws) - 1)) / max(1.0, w)))
            return lines * max(hs) + 0.8 * ctx.base_fs * (lines - 1)
        return max(hs)

    def cols_for(hero_w: float) -> int:
        for k in (1, 2, 3):
            if k > len(rows):
                break
            n = n_fixed + (1 if hero is not None else 0) + k
            avail = zone.w - fixed - hero_w - gap * (n - 1)
            if avail <= 0:
                break
            w = avail / k
            hs = [row_h(r, w) for r in rows]
            g = 0.35 * ctx.base_fs
            best = math.inf
            for cuts in itertools.combinations(range(1, len(rows)), k - 1):
                b, prev = 0.0, 0
                for c in list(cuts) + [len(rows)]:
                    b = max(b, sum(hs[prev:c]) + g * (c - prev - 1))
                    prev = c
                best = min(best, b)
            if best <= zone.h + 0.5:
                return k
        return 0

    full = 0.0
    if hero is not None:
        full = min(0.3 * zone.w, zone.h * hero.src_box.w / max(1e-6, hero.src_box.h))
    k = cols_for(full)
    if hero is None:
        return k, False, k
    tk = cols_for(min(0.14 * zone.w, 0.6 * zone.h * hero.src_box.w / max(1e-6, hero.src_box.h)))
    better = tk != 0 and (k == 0 or tk < k)
    return k, better, tk


def split_rows(rows: List[List[Item]], k: int) -> Optional[List[List[List[Item]]]]:
    """Copy rows in reading order, cut into k contiguous columns of balanced load."""
    n = len(rows)
    if n < k:
        return None

    def load(it: Item) -> float:
        if it.kind == "text":
            return em_advance(it.els[0]) * max(1.0, it.fs0) ** 2
        return it.src_box.area

    w = [sum(load(it) for it in r) for r in rows]
    best = None
    for cuts in itertools.combinations(range(1, n), k - 1):
        b, prev = 0.0, 0
        for c in list(cuts) + [n]:
            b = max(b, sum(w[prev:c]))
            prev = c
        if best is None or b < best[0]:
            best = (b, cuts)
    out, prev = [], 0
    for c in list(best[1]) + [n]:
        out.append(rows[prev:c])
        prev = c
    return out


# ----------------------------------------------------------------------------
# the ladder


@dataclass
class Unit:
    items: List[Item]
    role: str
    rungs: Tuple[str, ...]

    @property
    def importance(self) -> float:
        return max(i.importance for i in self.items)

    @property
    def label(self) -> str:
        ids = "+".join(it.key for it in self.items)
        return f"{self.role} {ids}"


def _rungs(role: str, n: int) -> Tuple[str, ...]:
    if role in ("legal", "body"):
        return ("floor", "zone", "hide")
    if role == "subhead":
        return ("floor", "hide")
    if role in ("badge", "group", "ribbon"):
        return ("floor", "chips", "hide") if n >= 2 else ("floor", "zone", "hide")
    if role in ("image", "shape", "watermark"):
        return ("floor", "zone", "hide")
    if role == "panel":
        return ("floor", "hide")
    return ("hide",)


def units(d: Design, core: bool = False) -> List[Unit]:
    """Optional content as demotion units, least important first. Items of one role in one
    row (a set of badges) move together. With `core`, the last resort: hero and logo may
    shrink and, only then, go; the headline and CTA never do."""
    from .structure import rows_for

    opt = [it for it in d.items if it.role not in CORE and it.role not in ("missing", "void") and not it.core]
    out: List[Unit] = []
    for row in rows_for(opt):
        by_role: Dict[str, List[Item]] = {}
        for it in row:
            by_role.setdefault(it.role, []).append(it)
        out += [Unit(its, role, _rungs(role, len(its))) for role, its in by_role.items()]
    out.sort(key=lambda u: (u.importance, -min(i.src_box.y for i in u.items)))
    if core:
        for role in ("hero", "logo"):
            out += [Unit([it], role, ("floor", "hide")) for it in d.items if it.role == role and not it.core]
    return out


def ladder(us: List[Unit]) -> List[Dict[int, str]]:
    """Cumulative demotion states. The least important unit goes down its rungs short of
    hiding (floor, then zone or chips) before the next one is touched; hiding comes last,
    least important first."""
    seq: List[Dict[int, str]] = [{}]
    cur: Dict[int, str] = {}
    for i, u in enumerate(us):
        for r in u.rungs:
            if r != "hide":
                cur = dict(cur)
                cur[i] = r
                seq.append(cur)
    for i, u in enumerate(us):
        if "hide" in u.rungs:
            cur = dict(cur)
            cur[i] = "hide"
            seq.append(cur)
    return seq


@dataclass
class State:
    units: List[Unit]
    rungs: Dict[int, str]

    def items(self, *rungs: str) -> List[Item]:
        return [it for i, r in self.rungs.items() if r in rungs for it in self.units[i].items]

    @property
    def hidden(self) -> List[Item]:
        return self.items("hide")

    @property
    def moved(self) -> List[Item]:
        return self.items("zone", "chips")

    @property
    def pinned(self) -> frozenset:
        return frozenset(it.key for it in self.items("floor", "zone", "chips"))

    @property
    def hides(self) -> bool:
        return any(r == "hide" for r in self.rungs.values())

    def zoned(self) -> List[Tuple[Unit, str]]:
        return [(self.units[i], r) for i, r in self.rungs.items() if r in ("zone", "chips")]

    def stamp(self, p) -> None:
        for i, r in self.rungs.items():
            for it in self.units[i].items:
                p.demoted[it.key] = r
        p.log = [f"{self.units[i].label}: {WHY[r]}" for i, r in self.rungs.items()] + list(p.log)


def clean(p) -> bool:
    """Legible type and every core item on the canvas: no reason to demote further."""
    return p.relax >= 1.0 and not any(h.role in CORE or h.core for h in p.hidden)


# ----------------------------------------------------------------------------
# candidate generation


def generate(d: Design, ctx: Ctx) -> list:
    from . import compose as C

    missing = [it for it in d.items if it.role in ("missing", "void")]
    if not C._visible(d, missing):  # nothing to lay out: every element is void or missing
        return [C.Plan(name="empty", placed=[], hidden=list(missing), s=1.0, zone=Box(0, 0, ctx.W, ctx.H))]
    plans = _climb(d, ctx, missing)
    if not plans:  # repair, not fit: dissolve every rigid unit carrying copy, then climb again
        rigid = [it for it in d.items if _carries_copy(it)]
        for it in rigid:
            if dissolve(d, it):
                _log(f"repair: no candidate fits; {it.key} ({it.role}) dissolved into its parts")
        if rigid:
            plans = _climb(d, ctx, missing)
    if not plans:  # the last resort before fit-and-centre: core art may go, copy never
        plans = _climb(d, ctx, missing, core=True)
        if plans:
            _bump("core_hidden")
            _log("repair: core art demoted (hero, then logo): nothing else fits")
    return plans


def _variants(d: Design, ctx: Ctx, force_all: bool, b: Budget):
    """The archetypes, plus what the budget asks for on strips: copy in two or three
    columns, and a hero thumbnail."""
    from . import compose as C

    yield from C.variants(d, ctx, force_all)
    if not b.strip:
        return
    ks = [b.cols] if b.cols >= 2 else ([2, 3] if b.cols == 0 else [])
    for k in ks:
        yield f"row-{k}col", C.plan_row, dict(cols=k)
    if b.thumb:
        k = max(1, b.thumb_cols)
        yield "row-thumb" + (f"-{k}col" if k > 1 else ""), C.plan_row, dict(cols=k, hero_max=0.14)


def _climb(d: Design, ctx: Ctx, missing: List[Item], core: bool = False) -> list:
    from . import compose as C

    us = units(d, core=core)
    seq = ladder(us)
    b = budget(d, ctx, us, seq, missing)
    if b.start:
        _log(f"feasibility: load {b.need[0]:.0f} px2 at the floors vs {b.avail:.0f} px2: "
             f"start at demotion state {b.start}/{len(seq) - 1}")
    if b.strip and (b.cols != 1 or b.thumb):
        _log(f"feasibility: strip copy needs {b.cols or '>3'} column(s) at the floor"
             + (f"; {b.thumb_cols} with a hero thumbnail" if b.thumb else ""))
    plans: list = []
    done = set()  # archetypes that already reached their cap without relaxing type
    found = None  # index of the first state that gave a clean plan
    for force_all in (False, True):
        for i in range(b.start, len(seq)):
            st = State(us, seq[i])
            if found is not None and (st.hides or i > found + EXTRA_STATES):
                break
            sctx = dataclasses.replace(ctx, pinned=st.pinned)
            hidden = list(missing) + st.hidden
            flow_hidden = hidden + st.moved
            for name, fn, kw in _variants(d, ctx, force_all, b):
                if name in done:
                    continue
                p = C._with_insets(fn, d, sctx, flow_hidden, name=name, **kw)
                if p is None:
                    continue

                def replan(base, fn=fn, name=name, kw=kw):
                    return C._with_insets(fn, d, sctx, flow_hidden, name=name, base=base, **kw)

                p = place_secondary(p, st, d, sctx, replan)
                if p is None:
                    continue
                st.stamp(p)
                plans.append(p)
                if p.relax >= 1.0 and p.capped:
                    done.add(name)
                if found is None and clean(p):
                    found = i
        if plans:
            break
    return plans


# ----------------------------------------------------------------------------
# secondary zones


def _slots(kind: str, ctx: Ctx, d: Design, near: Optional[str]) -> List[str]:
    strip = ctx.W / ctx.H >= 3.5
    corners = ["br", "bl", "tr", "tl"]
    if near in corners:
        corners.remove(near)
        corners.insert(0, near)
    if kind == "band":
        return ["bottom", "top"]
    if kind == "column":  # an end column, or tucked into a corner
        return (["right", "left"] + corners) if strip else (corners + ["right", "left"])
    return corners + ["bottom", "top", "right", "left"]


def _near_corner(items: Sequence[Item], d: Design) -> Optional[str]:
    if items[0].corner:
        return items[0].corner
    u = union_all(it.src_box for it in items)
    return ("t" if u.cy < d.H / 2 else "b") + ("l" if u.cx < d.W / 2 else "r")


def _arrangements(items: List[Item], rung: str, ctx: Ctx, d: Design, zone: Box):
    """Ways to pack a unit at its floor: (kind, [(Real, dx, dy)], w, h)."""
    gap = 0.6 * ctx.base_fs
    out = []
    if len(items) == 1 and items[0].kind == "text":
        it = items[0]
        f = floor_font(it, ctx)
        step = f * (it.style.line_height if it.style else 1.2)
        for lines in (1, 2):  # a legal strip: one line across, two at most
            r = _floor_real(it, ctx, zone.w, step * lines + 0.6 * f, mode="fit")
            if r is not None:
                out.append(("band", [(r, 0.0, 0.0)], r.w, r.h))
                break
        r = _floor_real(it, ctx, min(zone.w, max(0.24 * ctx.W, 8 * f)), zone.h)
        if r is not None:
            out.append(("column", [(r, 0.0, 0.0)], r.w, r.h))
        return out
    reals = [chip_real(it, ctx, zone.w, zone.h) if rung == "chips" else
             (_floor_real(it, ctx, zone.w, zone.h, nowrap=True) or _floor_real(it, ctx, zone.w, zone.h))
             for it in items]
    if any(r is None for r in reals):
        return out
    if len(reals) == 1:
        r = reals[0]
        out.append(("single", [(r, 0.0, 0.0)], r.w, r.h))
        return out
    # chips: a row (wrapping onto a second line if it must), or one column, at the floor
    lines: List[List[Real]] = [[]]
    lw = 0.0
    for r in reals:
        need = r.w if not lines[-1] else lw + gap + r.w
        if lines[-1] and need > zone.w + 0.5:
            lines.append([r])
            lw = r.w
        else:
            lines[-1].append(r)
            lw = need
    widths = [sum(r.w for r in ln) + gap * (len(ln) - 1) for ln in lines]
    heights = [max(r.h for r in ln) for ln in lines]
    w = max(widths)
    h = sum(heights) + 0.5 * gap * (len(lines) - 1)
    row, y = [], 0.0
    for ln, lw_, lh in zip(lines, widths, heights):
        x = (w - lw_) / 2.0
        for r in ln:
            row.append((r, x, y + (lh - r.h) / 2.0))
            x += r.w + gap
        y += lh + 0.5 * gap
    col_w = max(r.w for r in reals)
    col_h = sum(r.h for r in reals) + 0.5 * gap * (len(reals) - 1)
    y, col = 0.0, []
    for r in reals:
        col.append((r, (col_w - r.w) / 2.0, y))
        y += r.h + 0.5 * gap
    strip = ctx.W / ctx.H >= 3.5
    if strip:
        out.append(("column", col, col_w, col_h))
        out.append(("band", row, w, h))
    else:
        out.append(("band", row, w, h))
        out.append(("column", col, col_w, col_h))
    return out


def _slot_xy(slot: str, w: float, h: float, ctx: Ctx, mx: float, my: float, align: str) -> Tuple[float, float]:
    W, H = ctx.W, ctx.H
    cx = mx if align == "start" else ((W - mx - w) if align == "end" else (W - w) / 2.0)
    return {
        "bottom": (cx, H - my - h), "top": (cx, my),
        "right": (W - mx - w, (H - h) / 2.0), "left": (mx, (H - h) / 2.0),
        "tl": (mx, my), "tr": (W - mx - w, my), "bl": (mx, H - my - h), "br": (W - mx - w, H - my - h),
    }[slot]


def _place_unit(items: List[Item], rung: str, d: Design, ctx: Ctx, obst: List[Box]):
    """Put a unit at its floor into free space: a band, an end column or a corner.
    Returns (placements, reservation) where reservation is the band to ask for if no
    spot is free: (side, size) or None."""
    from .compose import margins
    from .structure import dominant_align

    mx, my = margins(d, ctx)
    zone = Box(mx, my, ctx.W - 2 * mx, ctx.H - 2 * my)
    gap = 0.6 * ctx.base_fs
    align = {"left": "start", "right": "end"}.get(dominant_align(d.items), "center")
    arr = _arrangements(items, rung, ctx, d, zone)
    near = _near_corner(items, d)
    reserve = None
    for kind, parts, w, h in arr:
        if w > zone.w + 0.5 or h > zone.h + 0.5:
            continue
        for slot in _slots(kind, ctx, d, near):
            x, y = _slot_xy(slot, w, h, ctx, mx, my, align if slot in ("bottom", "top") else "center")
            box = Box(x, y, w, h)
            if any(box.expanded(gap).inter(o) > 0 for o in obst):
                continue
            return [(r, x + dx, y + dy) for r, dx, dy in parts], None
        if reserve is None and kind in ("band", "column"):
            reserve = ("bottom", h + gap) if kind == "band" else ("right", w + gap)
    return None, reserve


def place_secondary(p, st: State, d: Design, ctx: Ctx, replan: Callable):
    """Put the units moved out of the flow (zone, chips) into free space around the laid-out
    content. If one finds no free spot, reserve a band for it (bottom strip or end column)
    and plan once more. Items the archetype itself had no slot for get the same chance
    before they count as hidden."""
    todo = st.zoned()
    hid = {it.key for it in st.hidden}
    moved = {it.key for it in st.moved}
    rescue = [it for it in p.hidden if it.key not in hid and it.key not in moved and it.role in RESCUE]
    if not todo and not rescue:
        return p
    for attempt in (0, 1):
        obst = [Box(x, y, r.w, r.h) for r, x, y in list(p.placed) + list(p.floaters)]
        added: List[Tuple[Real, float, float]] = []
        failed, need = False, None
        for u, rung in todo:
            pl, reserve = _place_unit(u.items, rung, d, ctx, obst)
            if pl is None:
                failed, need = True, reserve
                break
            added += pl
            obst += [Box(x, y, r.w, r.h) for r, x, y in pl]
        if failed:
            if attempt or need is None:
                return None
            side, size = need
            base = (0.0, 0.0, size, 0.0) if side == "right" else (0.0, 0.0, 0.0, size)
            q = replan(base)
            if q is None:
                return None
            p = q
            rescue = [it for it in p.hidden if it.key not in hid and it.key not in moved and it.role in RESCUE]
            continue
        saved = []
        for it in rescue:
            pl, _ = _place_unit([it], "zone", d, ctx, obst)
            if pl is not None:
                added += pl
                obst += [Box(x, y, r.w, r.h) for r, x, y in pl]
                saved.append(it)
        p.floaters = list(p.floaters) + added
        gone = moved | {it.key for it in saved}
        p.hidden = [h for h in p.hidden if h.key not in gone]
        for it in saved:
            p.demoted[it.key] = "zone"
            p.log.append(f"{it.role} {it.key}: no slot in the {p.name} archetype: moved to a free spot at its floor")
        for it in rescue:
            if it not in saved:
                p.log.append(f"{it.role} {it.key}: no slot in the {p.name} archetype and no free spot: hidden")
        return p
    return None


# ----------------------------------------------------------------------------
# repair instead of fit-and-centre


def repair(plans: list, d: Design, ctx: Ctx) -> bool:
    """Every candidate draws copy over copy: repair the best ones instead of falling back.
    The lower-priority text of each clash goes to its floor, then to a free spot; only a
    non-core one is hidden. The repaired candidate goes first."""
    from . import emit
    from .textfit import deliberate_overlaps

    ex = deliberate_overlaps(d.elements)
    for p in list(plans[:REPAIR_TRIES]):
        fixed, steps = _repair_overlaps(emit(p, d, ctx), p, d, ctx, ex)
        if fixed is None:
            continue
        p.override = fixed
        p.log = list(p.log) + steps
        plans.remove(p)
        plans.insert(0, p)
        _bump("repair")
        _log(f"repair: {p.name}: " + "; ".join(steps))
        return True
    return False


def _el_importance(eid: str, unit_of: Dict[str, Item], by_id: Dict[str, dict]) -> Tuple[float, float]:
    it = unit_of.get(eid)
    return (it.importance if it is not None else 0.0, float(by_id[eid].get("font_size", 0) or 0))


def _el_role(e: dict, it: Optional[Item]) -> str:
    if it is None:
        return "body"
    if it.kind == "text":
        return it.role
    labels = [x for x in it.els if x.get("type") == "text"]
    biggest = max(labels, key=lambda x: float(x.get("font_size", 0) or 0)) if labels else None
    if biggest is not None and biggest["id"] == e["id"] and (it.core or it.role in CORE):
        return "headline" if it.role != "cta" else "cta"
    return "legal" if it.role == "legal" else "body"


def _repair_overlaps(els: List[dict], p, d: Design, ctx: Ctx, ex) -> Tuple[Optional[List[dict]], List[str]]:
    from .textfit import text_overlaps

    els = [dict(e) for e in els]
    by_id = {e["id"]: e for e in els}
    unit_of: Dict[str, Item] = {}
    for r, _, _ in list(p.placed) + list(p.floaters):
        for e in r.item.els:
            unit_of[e["id"]] = r.item
    steps: List[str] = []
    n = sum(1 for e in els if e.get("type") == "text")
    for _ in range(3 * n + 3):
        ov = text_overlaps(els, exempt=ex)
        if not ov:
            return els, steps
        a, b = ov[0][0], ov[0][1]
        pair = sorted((a, b), key=lambda k: _el_importance(k, unit_of, by_id))
        done = None
        for vid, oid in (pair, pair[::-1]):
            done = _demote_text(vid, oid, els, by_id, unit_of, p, d, ctx, ex)
            if done:
                steps.append(done)
                break
        if not done:
            return None, steps
    return None, steps


def _drawn(e: dict) -> Box:
    from .textfit import drawn_box

    return drawn_box(e)


def _clash(e: dict, o: dict) -> bool:
    a, b = _drawn(e), _drawn(o)
    return min(a.r, b.r) - max(a.x, b.x) > 1.0 and min(a.b, b.b) - max(a.y, b.y) > 1.0


def _demote_text(vid: str, oid: str, els, by_id, unit_of, p, d: Design, ctx: Ctx, ex) -> Optional[str]:
    from . import _hidden

    e, o = by_id[vid], by_id[oid]
    it = unit_of.get(vid)
    role = _el_role(e, it)
    src = d.by_id.get(vid, e)
    fs0 = float(src.get("font_size", e.get("font_size", 12)) or 12)
    floor = font_for(fs0, 0.01, role, ctx)
    fs = int(e.get("font_size", floor) or floor)
    old = dict(e)
    note = []
    if fs > floor:  # 1. its floor
        w = max(1.0, float(e["width"]) * floor / max(1, fs))
        blk = textfit.measure(str(e.get("content", "")), TextStyle.of(e), floor, w, exact=True)
        w = max(w, blk.ink_width + 1.0)
        a = e.get("text_align", "left")
        x0, w0 = float(e["x"]), float(e["width"])
        x = x0 if a == "left" else (x0 + w0 - w if a == "right" else x0 + (w0 - w) / 2.0)
        e.update(x=x, width=w, height=blk.height, font_size=floor)
        if not _clash(e, o):
            return f"{vid} clashed with {oid}: shrunk {fs}->{floor}px (its floor)"
        note.append(f"shrunk {fs}->{floor}px")
    spot = _free_spot(e, els, unit_of, p, ctx, d)  # 2. a free spot
    if spot is not None:
        return f"{vid} clashed with {oid}: " + ", ".join(note + [f"moved {spot}"])
    if role not in ("headline", "cta") and not (it is not None and it.kind == "text" and it.role in CORE):
        h = _hidden(src, d, ctx)  # 3. only then hidden
        e.clear()
        e.update({k: v for k, v in h.items() if k != "_idx"})
        return f"{vid} clashed with {oid}: no free spot, hidden"
    e.clear()
    e.update(old)
    return None


def _free_spot(e: dict, els, unit_of, p, ctx: Ctx, d: Design) -> Optional[str]:
    """Move a text to the nearest spot where its drawn box touches no other copy and no
    other unit's footprint (inside the margins)."""
    from .compose import margins

    mx, my = margins(d, ctx)
    me = unit_of.get(e["id"])
    db = _drawn(e)
    gap = 0.4 * ctx.base_fs
    obst = [_drawn(o) for o in els if o is not e and o.get("type") == "text"
            and float(o.get("opacity", 1) or 0) > 0 and str(o.get("content", "")).strip()]
    obst += [b for it, b in p.footprints() if it is not me]
    x0, x1 = mx, ctx.W - mx - db.w
    y0, y1 = my, ctx.H - my - db.h
    if x1 < x0 or y1 < y0:
        return None
    step = max(3.0, min(ctx.W, ctx.H) / 60.0)
    best = None
    nx, ny = int((x1 - x0) / step) + 1, int((y1 - y0) / step) + 1
    for i in range(nx + 1):
        x = min(x1, x0 + i * step)
        for j in range(ny + 1):
            y = min(y1, y0 + j * step)
            box = Box(x, y, db.w, db.h).expanded(gap)
            if any(box.inter(ob) > 0 for ob in obst):
                continue
            dist = math.hypot(x - db.x, y - db.y)
            if best is None or dist < best[0]:
                best = (dist, x, y)
    if best is None:
        return None
    dx, dy = best[1] - db.x, best[2] - db.y
    e["x"] = float(e["x"]) + dx
    e["y"] = float(e["y"]) + dy
    return f"by ({dx:+.0f},{dy:+.0f})px to a free spot"
