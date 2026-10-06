"""Repeating fragments: sets of chips, badges or button pairs built alike.

After Canva's US12056802B2 (grouping by alignment, which never merges
"repeating fragments"): three badges, a pair of buttons or a row of tags are
one *set* of separate units, not one block and not unrelated items. A
designer re-making the design at another size keeps them uniform:

* one shared font size for every label (the largest any of them needs, so no
  label drops below its legibility floor);
* equal chip heights (and, stacked, equal widths), equal gaps;
* a uniform arrangement: a single row, a balanced grid (3+3, 4+3, never an
  orphan), or a column. Wide targets prefer a row, tall targets fall back to a
  column, the source arrangement goes first where the class allows it.

``detect`` (end of analyze) marks the sets; ``merge_rows`` (in rows_for) keeps a
set that the reading order split over several rows together; ``node`` (in
compose's row builder) lays a set out as a ``FragmentNode`` instead of the
greedy ``Flow`` that wrapped Level 3's badges 2+1.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

from .geom import Box, el_box
from .items import Ctx, Real, realize
from .layout import Laid, Node, _gap

REPEATS_ON = os.environ.get("CHAMELEON_REPEATS", "1") != "0"


# ----------------------------------------------------------------------------
# detection


def _radius_class(sh: dict) -> str:
    r = float(sh.get("corner_radius", 0) or 0)
    h = max(1e-6, el_box(sh).h)
    return "pill" if r >= 0.45 * h else ("round" if r > 0.5 else "square")


def _signature(it) -> Optional[tuple]:
    if it.kind != "group" or not it.labels or abs(it.rotation) >= 1 or it.corner:
        return None
    if it.role not in ("badge", "cta", "group"):
        return None
    lab = max(it.labels, key=lambda l: float(l.get("font_size", 0) or 0))
    font = (str(lab.get("font_family", "")), str(lab.get("font_weight", "")), str(lab.get("font_style", "")))
    if it.container is not None:
        sh = it.container
        return ("chip", sh.get("type"), sh.get("shape_type"), _radius_class(sh), len(it.labels), len(it.els)) + font
    return ("card", tuple(sorted(str(e.get("type")) for e in it.els))) + font


def _alike(a, b) -> bool:
    fa, fb = a.main_fs(), b.main_fs()
    if fa <= 0 or fb <= 0 or max(fa, fb) / min(fa, fb) > 1.15:
        return False
    if a.container is not None:
        ha, hb = el_box(a.container).h, el_box(b.container).h
        return ha > 0 and hb > 0 and max(ha, hb) / min(ha, hb) <= 1.25
    wa, wb, ha, hb = a.src_box.w, b.src_box.w, a.src_box.h, b.src_box.h
    return min(wa, wb, ha, hb) > 0 and max(wa, wb) / min(wa, wb) <= 1.25 and max(ha, hb) / min(ha, hb) <= 1.25


def detect(d) -> List[list]:
    """Mark sets of repeated fragments (``item.repeat_key``); returns the sets."""
    sets: List[list] = []
    if not REPEATS_ON:
        return sets
    by_sig: Dict[tuple, list] = {}
    for it in d.items:
        sig = _signature(it)
        if sig is not None:
            by_sig.setdefault(sig, []).append(it)
    for members in by_sig.values():
        if len(members) < 2:
            continue
        parent = list(range(len(members)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if _alike(members[i], members[j]):
                    parent[find(i)] = find(j)
        groups: Dict[int, list] = {}
        for i, m in enumerate(members):
            groups.setdefault(find(i), []).append(m)
        for g in groups.values():
            if len(g) >= 2:
                sets.append(g)
    for k, g in enumerate(sets):
        for it in g:
            it.repeat_key = f"rep{k}"  # type: ignore[attr-defined]
    d.repeats = sets  # type: ignore[attr-defined]
    return sets


def _key(row: Sequence) -> Optional[str]:
    keys = {getattr(it, "repeat_key", None) for it in row}
    return keys.pop() if len(keys) == 1 else None


def merge_rows(rows: List[list]) -> List[list]:
    """Consecutive reading-order rows made only of one set's members are one row."""
    if not REPEATS_ON:
        return rows
    out: List[list] = []
    for row in rows:
        k = _key(row)
        if out and k is not None and _key(out[-1]) == k:
            out[-1] = out[-1] + list(row)
        else:
            out.append(list(row))
    return out


def applies(row: Sequence) -> bool:
    return REPEATS_ON and len(row) >= 2 and _key(row) is not None


# ----------------------------------------------------------------------------
# layout


def _resized(r: Real, it, Hc: Optional[float], Wc: Optional[float]) -> Real:
    """The same chip with its shape grown to Hc x Wc around its centre (label
    stays centred). Only for unrotated chips with a container."""
    sh = it.container
    if sh is None or abs(it.rotation) >= 1:
        return r
    els = r.emit(0.0, 0.0)
    me = next((e for e in els if e.get("id") == sh.get("id")), None)
    if me is None:
        return r
    dh = max(0.0, (Hc or 0.0) - float(me["height"]))
    dw = max(0.0, (Wc or 0.0) - float(me["width"]))
    if dh <= 0.5 and dw <= 0.5:
        return r
    pill = _radius_class(sh) == "pill"

    def emit(x: float, y: float, r=r, dw=dw, dh=dh) -> List[dict]:
        out = r.emit(x + dw / 2.0, y + dh / 2.0)
        for e in out:
            if e.get("id") == sh.get("id"):
                e.update(x=e["x"] - dw / 2.0, y=e["y"] - dh / 2.0, width=e["width"] + dw, height=e["height"] + dh)
                if pill:
                    e["corner_radius"] = e["height"] / 2.0
        return out

    return Real(item=it, w=r.w + dw, h=r.h + dh, scale=r.scale, fonts=r.fonts, blocks=r.blocks, emit_fn=emit)


def _balanced(n: int, rows: int) -> List[int]:
    base, extra = divmod(n, rows)
    return [base + 1] * extra + [base] * (rows - extra)


class FragmentNode(Node):
    """A set of repeated fragments as one uniform row / grid / column."""

    def __init__(self, items: Sequence, gap: Optional[float] = None):
        self.items_ = list(items)
        self.gap = gap
        bs = [it.src_box for it in self.items_]
        sx = max(b.cx for b in bs) - min(b.cx for b in bs)
        sy = max(b.cy for b in bs) - min(b.cy for b in bs)
        self.src_arr = "row" if sx >= sy else "column"

    def items(self):
        return list(self.items_)

    # -- realizing the chips uniformly
    def _reals(self, s: float, ctx: Ctx, max_w: float, max_h: float, relax: float) -> Optional[List[Real]]:
        reals = []
        for it in self.items_:
            r = realize(it, s, ctx, max_w, max_h, relax=relax, nowrap=True)
            if r is None:
                r = realize(it, s, ctx, max_w, max_h, relax=relax)
            if r is None:
                return None
            reals.append(r)
        fonts = [max(r.fonts.values()) if r.fonts else 0 for r in reals]
        F = max(fonts)
        if F and len(set(f for f in fonts if f)) > 1:  # one shared size: the largest anyone needs
            out = []
            for it, r, f in zip(self.items_, reals, fonts):
                fs0 = it.main_fs()
                if f == F or not f or fs0 <= 0:
                    out.append(r)
                    continue
                r2 = realize(it, F / fs0, ctx, max_w, max_h, relax=relax, nowrap=True) or \
                    realize(it, F / fs0, ctx, max_w, max_h, relax=relax)
                if r2 is None:
                    return None
                out.append(r2)
            reals = out
        return reals

    def _arrangements(self, ctx: Ctx) -> List[List[int]]:
        from .formats import TALL_CLASSES, WIDE_CLASSES, aspect_class

        n = len(self.items_)
        row, col = [n], [1] * n
        grids = [_balanced(n, k) for k in range(2, n) if min(_balanced(n, k)) >= 2]
        cls = aspect_class(ctx.W / max(ctx.H, 1e-6))
        if cls in WIDE_CLASSES:
            order = [row] + grids + [col]
        elif cls in TALL_CLASSES:
            order = [row, col] + grids if self.src_arr == "row" else [col] + grids + [row]
        else:
            order = ([row] + grids + [col]) if self.src_arr == "row" else ([col] + grids + [row])
        seen, out = set(), []
        for a in order:
            if tuple(a) not in seen:
                seen.add(tuple(a))
                out.append(a)
        return out

    def measure(self, s, ctx, max_w, max_h, relax):
        reals = self._reals(s, ctx, max_w, max_h, relax)
        if reals is None:
            return None
        g = _gap(self.gap, s, ctx, 0.5) if self.gap is not None else 0.8 * ctx.base_fs
        chips = [it.container is not None and abs(it.rotation) < 1 for it in self.items_]
        Hc = max((float(el_box(e).h) for r, it, c in zip(reals, self.items_, chips) if c
                  for e in r.emit(0.0, 0.0) if e.get("id") == it.container.get("id")), default=None)
        widths = [r.w for r in reals]
        for rows in self._arrangements(ctx):
            stacked = len(rows) > 1
            Wc = None
            if stacked and all(chips) and max(widths) / max(1e-6, min(widths)) <= 1.8:
                Wc = max(float(el_box(e).w) for r, it in zip(reals, self.items_)
                         for e in r.emit(0.0, 0.0) if e.get("id") == it.container.get("id"))
            rs = [_resized(r, it, Hc, Wc) if c else r for r, it, c in zip(reals, self.items_, chips)]
            lines, i = [], 0
            for k in rows:
                lines.append(rs[i:i + k])
                i += k
            row_w = [sum(r.w for r in ln) + g * (len(ln) - 1) for ln in lines]
            row_h = [max(r.h for r in ln) for ln in lines]
            w, h = max(row_w), sum(row_h) + g * (len(lines) - 1)
            if w > max_w + 0.5 or h > max_h + 0.5:
                continue

            def place(x, y, lines=lines, row_w=row_w, row_h=row_h, w=w, g=g):
                out = []
                cy = y
                for ln, rw, rh in zip(lines, row_w, row_h):
                    cx = x + (w - rw) / 2.0
                    for r in ln:
                        out.append((r, cx, cy + (rh - r.h) / 2.0))
                        cx += r.w + g
                    cy += rh + g
                return out

            oh = tuple(max(r.oh[k] for r in rs) for k in range(4))
            texts = [r.scale for r in rs if r.item.kind not in ("image", "composite")]
            laid = Laid(w, h, place, min_scale=min(texts) if texts else 9.0, oh=oh)  # type: ignore[arg-type]
            laid.arrangement = rows  # type: ignore[attr-defined]
            return laid
        return None


def node(row: Sequence) -> FragmentNode:
    gaps = []
    bs = sorted((it.src_box for it in row), key=lambda b: (b.cy, b.x))
    xs = sorted(bs, key=lambda b: b.x)
    for a, b in zip(xs, xs[1:]):
        if b.x - a.r > 0:
            gaps.append(b.x - a.r)
    if not gaps:
        ys = sorted(bs, key=lambda b: b.y)
        gaps = [b.y - a.b for a, b in zip(ys, ys[1:]) if b.y - a.b > 0]
    return FragmentNode(row, gap=(sum(gaps) / len(gaps)) if gaps else None)
