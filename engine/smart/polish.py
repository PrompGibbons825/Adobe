"""Final polish of the best candidates (after LayoutRectifier: snap, equalize,
separate).  [relations-polish]

Only positions move. Sizes, fonts and wraps stay exactly as realized, so the
polish cannot change an aspect ratio or push text below the legibility floor.
It works on what is painted (glyph boxes, art ink boxes, via relations.py),
moving each layout unit rigidly:

  1. snap      a unit that sits slightly off the left / centre / right line of
               a unit stacked with it is moved onto that line; on the other
               axis only lines the source itself shared are restored. The more
               important unit holds the line; a unit already exactly on a line
               with a settled unit stays.
  2. gaps      runs of near-equal gaps between consecutive units of a column
               (or row) become equal; the run's ends stay put, and a unit that
               shares its row with another (a button beside a logo) stays
  3. separate  units that overlap now but did not in the source are pushed
               apart along the shallower axis (the less important one moves)
  4. clamp     painted boxes stay on the canvas and no closer to an edge than
               the margin (or than they already were)

A whole-pixel grid step was tried and dropped: units have fractional painted
offsets (glyph boxes), so rounding positions turned equal 4 px gaps into
3.4 / 4.1 px ones, and the renderer anti-aliases sub-pixel positions anyway.

Corner-pinned units (floaters) never move; their lines can still be snapped
to. The polished plan is kept only if it passes the hard checks again (on
canvas, no new copy over copy, no new unit overlap) and its relation and look
terms are no worse.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .geom import Box

POLISH = True
TOP_K = 8


def _ov(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def _anchors(b: Box, axis: str) -> Dict[str, float]:
    if axis == "x":
        return {"L": b.x, "C": b.cx, "R": b.r}
    return {"T": b.y, "M": b.cy, "B": b.b}


def _stacked(a: Box, b: Box, axis: str) -> bool:
    """Units that read as one column (axis x) or one row (axis y)."""
    if axis == "x":
        return _ov(a.x, a.r, b.x, b.r) > 0 and _ov(a.y, a.b, b.y, b.b) <= 0.15 * min(a.h, b.h)
    return _ov(a.y, a.b, b.y, b.b) > 0 and _ov(a.x, a.r, b.x, b.r) <= 0.15 * min(a.w, b.w)


def _snap(boxes: Dict[str, Box], imp: Dict[str, float], movable: Set[str], axis: str, tol: float,
          src_lines: Dict[Tuple[str, str], str]) -> Dict[str, float]:
    """Shifts along ``axis`` that put near-aligned units on shared lines."""
    shift = {k: 0.0 for k in boxes}
    order = sorted(boxes, key=lambda k: (k in movable, -imp[k], k))  # fixed units settle first
    settled: List[str] = []

    def box(k):
        b = boxes[k]
        return b.moved(shift[k], 0.0) if axis == "x" else b.moved(0.0, shift[k])

    for k in order:
        if k not in movable:
            settled.append(k)
            continue
        b = box(k)
        pa = _anchors(b, axis)
        best = None
        exact = False
        for j in settled:
            bj = box(j)
            want = src_lines.get((k, j), "") + src_lines.get((j, k), "")
            if axis == "x":
                if not want and not _stacked(b, bj, "x"):
                    continue
            elif not want:
                continue  # rows: only lines the source shared
            pj = _anchors(bj, axis)
            for t in pa:
                dlt = pj[t] - pa[t]
                if abs(dlt) <= 0.5:
                    exact = True
                elif abs(dlt) <= tol and (axis == "x" or t in want):
                    rank = (t not in want, abs(dlt))
                    if best is None or rank < best[0]:
                        best = (rank, dlt)
        if best is not None and not exact:
            shift[k] += best[1]
        settled.append(k)
    return shift


def _before(a: Box, b: Box, axis: str) -> bool:
    if axis == "y":
        return a.cy < b.cy and a.b <= b.y + 0.5
    return a.cx < b.cx and a.r <= b.x + 0.5


def _chains(boxes: Dict[str, Box], axis: str) -> List[List[str]]:
    """Maximal chains of consecutive units along the axis within one column / row."""
    nxt: Dict[str, str] = {}
    for k, a in boxes.items():
        best, bd = None, 1e18
        for k2, b in boxes.items():
            if k2 == k or not _before(a, b, axis):
                continue
            share = _ov(a.x, a.r, b.x, b.r) if axis == "y" else _ov(a.y, a.b, b.y, b.b)
            if share <= 0:
                continue
            gap = (b.y - a.b) if axis == "y" else (b.x - a.r)
            if gap < bd:
                best, bd = k2, gap
        if best is not None:
            nxt[k] = best
    prev: Dict[str, str] = {}
    for a in sorted(nxt):
        prev.setdefault(nxt[a], a)
    chains = []
    for k in sorted(boxes):
        if k in prev and nxt.get(prev[k]) == k:
            continue  # not the head of a chain
        ch = [k]
        while ch[-1] in nxt and nxt[ch[-1]] not in ch and prev.get(nxt[ch[-1]]) == ch[-1]:
            ch.append(nxt[ch[-1]])
        if len(ch) >= 3:
            chains.append(ch)
    return chains


def _partnered(boxes: Dict[str, Box], axis: str) -> Set[str]:
    """Units that share their row (axis y) or column (axis x) with another unit:
    moving one of them alone along that axis would break the row."""
    out = set()
    ks = sorted(boxes)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if _stacked(boxes[a], boxes[b], axis):
                ba, bb = boxes[a], boxes[b]
                share = _ov(ba.y, ba.b, bb.y, bb.b) if axis == "y" else _ov(ba.x, ba.r, bb.x, bb.r)
                size = min(ba.h, bb.h) if axis == "y" else min(ba.w, bb.w)
                if share >= 0.5 * size:
                    out.update((a, b))
    return out


def _equalize(boxes: Dict[str, Box], movable: Set[str], axis: str, cap: float) -> Dict[str, float]:
    shift = {k: 0.0 for k in boxes}
    free = movable - _partnered(boxes, axis)
    for ch in _chains(boxes, axis):
        bs = [boxes[k] for k in ch]
        gaps = [(b.y - a.b) if axis == "y" else (b.x - a.r) for a, b in zip(bs, bs[1:])]
        i = 0
        while i < len(gaps):
            j = i
            while j + 1 < len(gaps) and min(gaps[i:j + 2]) > 0 and max(gaps[i:j + 2]) <= 1.25 * min(gaps[i:j + 2]):
                j += 1
            if j > i:  # gaps i..j (units i..j+1): the ends stay, the inside moves
                mean = sum(gaps[i:j + 1]) / (j - i + 1)
                pos = 0.0
                moves = []
                for t in range(i, j):
                    pos += mean - gaps[t]
                    moves.append((ch[t + 1], pos))
                if all(k in free and abs(m) <= cap for k, m in moves):
                    for k, m in moves:
                        shift[k] += m
            i = j + 1
    return shift


def _separate(boxes: Dict[str, Box], imp: Dict[str, float], movable: Set[str], intended: Set[Tuple[str, str]],
              lim: Dict[str, Tuple[float, float, float, float]]) -> Dict[str, Tuple[float, float]]:
    """1-D pushes that remove overlaps the source did not have."""
    shift = {k: [0.0, 0.0] for k in boxes}

    def box(k):
        return boxes[k].moved(*shift[k])

    ks = sorted(boxes)
    for _ in range(3):
        moved = False
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                if (a, b) in intended or (b, a) in intended:
                    continue
                ba, bb = box(a), box(b)
                ox, oy = _ov(ba.x, ba.r, bb.x, bb.r), _ov(ba.y, ba.b, bb.y, bb.b)
                if ox <= 0.5 or oy <= 0.5:
                    continue
                cand = [k for k in (a, b) if k in movable]
                if not cand:
                    continue
                k = min(cand, key=lambda k: (imp[k], k))
                o = b if k == a else a
                bk, bo = box(k), box(o)
                if ox <= oy:
                    opts = [(bo.x - bk.r - 0.5, 0.0), (bo.r - bk.x + 0.5, 0.0)]
                else:
                    opts = [(0.0, bo.y - bk.b - 0.5), (0.0, bo.b - bk.y + 0.5)]
                lo_x, hi_x, lo_y, hi_y = lim[k]
                opts = [(dx, dy) for dx, dy in opts
                        if lo_x <= shift[k][0] + dx <= hi_x and lo_y <= shift[k][1] + dy <= hi_y]
                if not opts:
                    continue
                dx, dy = min(opts, key=lambda v: abs(v[0]) + abs(v[1]))
                shift[k][0] += dx
                shift[k][1] += dy
                moved = True
        if not moved:
            break
    return {k: (v[0], v[1]) for k, v in shift.items()}


def polish(p, d, ctx, g):
    """A polished copy of plan ``p``, or None when nothing moves."""
    from . import emit
    from .compose import Plan, margins, overflow
    from .relations import units_of

    hidden = {e["id"] for it in p.hidden for e in it.els}
    units = units_of(d, emit(p, d, ctx), hidden)
    keys_placed = [r.item.key for r, _, _ in p.placed]
    keys_float = [r.item.key for r, _, _ in p.floaters]
    boxes = {k: units[k].box for k in keys_placed + keys_float if k in units}
    if len(boxes) < 2:
        return None
    imp = {k: units[k].imp for k in boxes}
    movable = {k for k in keys_placed if k in boxes}
    mx, my = margins(d, ctx)

    # how far each unit may travel: its painted box stays on the canvas, and no
    # closer to an edge than the margin unless it already was
    reals = {r.item.key: (r, x, y) for r, x, y in p.placed}
    lim: Dict[str, Tuple[float, float, float, float]] = {}
    for k in movable:
        r, x, y = reals[k]
        ol, ot, orr, ob = r.oh
        px0, py0, px1, py1 = x - ol, y - ot, x + r.w + orr, y + r.h + ob
        el, et = min(mx, max(0.0, px0)), min(my, max(0.0, py0))
        er, eb = min(mx, max(0.0, ctx.W - px1)), min(my, max(0.0, ctx.H - py1))
        lim[k] = (min(0.0, el - px0), max(0.0, ctx.W - er - px1), min(0.0, et - py0), max(0.0, ctx.H - eb - py1))

    src_lines: Dict[Tuple[str, str], str] = {}
    for rel in g.rels:
        if rel.kind == "align":
            src_lines[(rel.a, rel.b)] = src_lines.get((rel.a, rel.b), "") + rel.sub
    intended: Set[Tuple[str, str]] = set()
    ks = list(g.units)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if g.units[a].box.inter(g.units[b].box) > 0:
                intended.add((a, b))

    total = {k: [0.0, 0.0] for k in boxes}

    def cur() -> Dict[str, Box]:
        return {k: boxes[k].moved(*total[k]) for k in boxes}

    def apply(sh: Dict[str, float], axis: str) -> None:
        i = 0 if axis == "x" else 1
        for k, v in sh.items():
            if k in movable and v:
                lo, hi = lim[k][2 * i], lim[k][2 * i + 1]
                total[k][i] = min(max(total[k][i] + v, lo), hi)

    apply(_snap(cur(), imp, movable, "x", max(2.0, 0.012 * ctx.W), src_lines), "x")
    apply(_snap(cur(), imp, movable, "y", max(2.0, 0.012 * ctx.H), src_lines), "y")
    apply(_equalize(cur(), movable, "y", 0.04 * ctx.H), "y")
    apply(_equalize(cur(), movable, "x", 0.04 * ctx.W), "x")
    rest = {k: (lim[k][0] - total[k][0], lim[k][1] - total[k][0], lim[k][2] - total[k][1], lim[k][3] - total[k][1])
            for k in movable}
    for k, (dx, dy) in _separate(cur(), imp, movable, intended, rest).items():
        total[k][0] += dx
        total[k][1] += dy

    placed = []
    changed = False
    for r, x, y in p.placed:
        dx, dy = total.get(r.item.key, (0.0, 0.0))
        changed |= abs(dx) > 0.01 or abs(dy) > 0.01
        placed.append((r, x + dx, y + dy))
    if not changed:
        return None
    q = Plan(name=p.name, placed=placed, hidden=p.hidden, s=p.s, zone=p.zone, floaters=p.floaters,
             relax=p.relax, capped=p.capped, score=p.score, breakdown=dict(p.breakdown))
    if max(overflow(q.placed, ctx)) > 0.5:
        return None
    return q


def _unit_overlap(p) -> float:
    fp = [Box(x, y, r.w, r.h) for r, x, y in p.placed + p.floaters]
    ov = 0.0
    for i in range(len(fp)):
        for j in range(i + 1, len(fp)):
            inter = fp[i].inter(fp[j])
            if inter > 0:
                ov += inter / max(1.0, min(fp[i].area, fp[j].area))
    return ov


def polish_plans(plans: list, d, ctx) -> int:
    """Polish the best ``TOP_K`` candidates in place; returns how many changed."""
    if not POLISH or not plans:
        return 0
    from . import emit
    from .relations import evaluate, extract
    from .score import REL_W, look_terms
    from .textfit import deliberate_overlaps, text_overlaps

    g = extract(d)
    exempt = frozenset(deliberate_overlaps(d.elements))
    n = 0
    for i, p in enumerate(plans[:TOP_K]):
        try:
            q = polish(p, d, ctx, g)
        except Exception:  # the polish is optional: never lose a good plan to it
            q = None
        if q is None:
            continue
        hidden = {e["id"] for it in p.hidden for e in it.els}
        before, after = emit(p, d, ctx), emit(q, d, ctx)
        if len(text_overlaps(after, exempt=exempt)) > len(text_overlaps(before, exempt=exempt)):
            continue
        if _unit_overlap(q) > _unit_overlap(p) + 1e-6:
            continue
        ev0 = getattr(p, "relations", None) or evaluate(g, d, before, ctx.W, ctx.H, hidden)
        ev1 = evaluate(g, d, after, ctx.W, ctx.H, hidden)
        t0, t1 = look_terms(ev0, g.look), look_terms(ev1, g.look)
        t0["relations"] = -REL_W * (1.0 - ev0["survival"])
        t1["relations"] = -REL_W * (1.0 - ev1["survival"])
        gain = sum(t1.values()) - sum(t0.values())
        if gain < -1e-6:
            continue
        q.breakdown.update(t1)
        q.breakdown["polish"] = 0.0
        q.score = p.score + gain
        q.relations = ev1  # type: ignore[attr-defined]
        plans[i] = q
        n += 1
    return n
