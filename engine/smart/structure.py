"""Reading structure of the source, by recursive XY-cut.

A design is cut into horizontal bands wherever nothing straddles a horizontal
line, each band into columns wherever nothing straddles a vertical line, and
so on. The result says what sits beside what. From it we derive:

* stack rows - the order a single column should present things in. Columns
  are read left to right (a logo at the top of a column comes first); a band
  of *like* things (badges, product shots) stays a row.
* per-row alignment - from the copy's own text_align and edges.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

from .analyze import Item
from .geom import Box, union_all

Tree = Union[Item, Tuple[str, list]]  # leaf, or ("v"|"h", children)


def _segments(items: Sequence[Item], axis: str) -> List[List[Item]]:
    def span(it: Item):
        b = it.src_box
        return (b.y, b.b) if axis == "y" else (b.x, b.r)

    order = sorted(items, key=lambda i: span(i)[0])
    segs: List[List[Item]] = []
    end = 0.0
    seg_min = 0.0
    for it in order:
        a, b = span(it)
        size = max(1.0, b - a)
        if segs:
            tol = 0.15 * min(size, seg_min)  # a sliver of overlap is still a cut
            if a < end - tol:
                segs[-1].append(it)
                end = max(end, b)
                seg_min = min(seg_min, size)
                continue
        segs.append([it])
        end, seg_min = b, size
    return segs


def _widest_gap(segs: List[List[Item]], axis: str) -> float:
    def lo(seg):
        return min(i.src_box.y if axis == "y" else i.src_box.x for i in seg)

    def hi(seg):
        return max(i.src_box.b if axis == "y" else i.src_box.r for i in seg)
    return max((lo(b) - hi(a) for a, b in zip(segs, segs[1:])), default=0.0)


def xy_tree(items: Sequence[Item], depth: int = 0) -> Tree:
    """Cut where the widest whitespace runs: across (into bands) or down (into
    columns), whichever gap is wider, so a sliver of space under a headline does
    not split it from its subhead when gutters clearly set the columns apart."""
    items = list(items)
    if len(items) == 1:
        return items[0]
    if depth > 12:
        return ("v", sorted(items, key=lambda i: (i.src_box.cy, i.src_box.x)))
    bands = _segments(items, "y")
    cols = _segments(items, "x")
    if len(bands) > 1 and (len(cols) < 2 or _widest_gap(bands, "y") >= _widest_gap(cols, "x")):
        return ("v", [xy_tree(b, depth + 1) for b in bands])
    if len(cols) > 1:
        return ("h", [xy_tree(c, depth + 1) for c in cols])
    return ("v", sorted(items, key=lambda i: (i.src_box.cy, i.src_box.x)))


def _leaves(t: Tree) -> List[Item]:
    if isinstance(t, Item):
        return [t]
    return [i for c in t[1] for i in _leaves(c)]


def _alike(items: Sequence[Item]) -> bool:
    """Siblings that read as one row: same kind, similar size (badges, products, icons)."""
    if len(items) < 2:
        return False
    kinds = {("copy" if i.is_copy else ("button" if i.kind == "group" else i.kind)) for i in items}
    if len(kinds) != 1:
        return False
    if any(i.role in ("headline", "logo") for i in items):
        return False
    areas = [max(1.0, i.src_box.area) for i in items]
    if max(areas) / min(areas) > 3.5:
        return False
    hs = [i.src_box.h for i in items]
    return max(hs) / max(1.0, min(hs)) < 2.2


def stack_rows(t: Tree) -> List[List[Item]]:
    if isinstance(t, Item):
        return [[t]]
    kind, kids = t
    if kind == "v":
        return [r for c in kids for r in stack_rows(c)]
    if all(isinstance(c, Item) for c in kids) and _alike(kids):
        return [sorted(kids, key=lambda i: i.src_box.x)]
    cols = [stack_rows(c) for c in kids]
    out: List[List[Item]] = []
    # a logo that heads any column leads the whole band
    for rows in cols:
        if rows and len(rows[0]) == 1 and rows[0][0].role == "logo":
            out.append(rows.pop(0))
    for rows in cols:
        out += rows
    return out


def rows_for(items: Sequence[Item]) -> List[List[Item]]:
    if not items:
        return []
    return stack_rows(xy_tree(items))


def lead_text(it: Item) -> dict:
    """The biggest text of a copy item (a lockup's main line)."""
    texts = [e for e in it.els if e.get("type") == "text"]
    return max(texts, key=lambda e: float(e.get("font_size", 0) or 0))


def dominant_align(items: Sequence[Item]) -> str:
    votes = {"left": 0.0, "center": 0.0, "right": 0.0}
    for it in items:
        els = [e for e in it.els if e.get("type") == "text"] if it.is_copy else []
        for e in els:
            a = e.get("text_align", "left")
            votes[a if a in votes else "left"] += it.importance
    if sum(votes.values()) == 0:
        return "center"
    return max(votes, key=votes.get)


def row_align(row: Sequence[Item], W0: float, dom: str, text_left: float, text_right: float) -> str:
    """start / center / end for a row in a single column."""
    b = union_all(i.src_box for i in row)
    if len(row) == 1 and row[0].is_copy:
        a = lead_text(row[0]).get("text_align", "left")
        return {"left": "start", "center": "center", "right": "end"}.get(a, "start")
    if abs(b.cx - W0 / 2.0) <= 0.06 * W0 and dom == "center":
        return "center"
    if dom == "left" and (abs(b.x - text_left) <= 0.06 * W0 or b.x <= 0.25 * W0):
        return "start"
    if dom == "right" and (abs(b.r - text_right) <= 0.06 * W0 or b.r >= 0.75 * W0):
        return "end"
    if dom == "center":
        return "center"
    return "start" if dom == "left" else "end"


def text_edges(items: Sequence[Item]) -> Tuple[float, float]:
    texts = [i for i in items if i.is_copy]
    if not texts:
        return 0.0, 0.0
    return min(t.src_box.x for t in texts), max(t.src_box.r for t in texts)
