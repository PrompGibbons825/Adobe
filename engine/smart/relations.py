"""What makes a resize "the same design": the source's relation graph, how much
of it a candidate keeps, and a few look metrics (alignment, spacing, margins).

Units are the layout items the engine moves (a text, a button, a panel, a photo
with its stickers). Their boxes are measured from *elements*, the same way for
the source and for any output: drawn glyph boxes for copy, ink boxes for art.
So the same code scores candidates inside the engine and finished outputs in
the dev tools.

Relation types (after ReLayout's 9 position + 3 size relations and VLT's
inferred alignments):

  align    two units share a left / centre / right line (x) or a top / middle /
           bottom line (y)
  order    A above B, A left of B
  band     same row (shared y-band, side by side) or same column
  gap      equal gaps between consecutive units of a column or row
  read     consecutive units in the source's reading order (XY-cut)
  eqsize   equal font size (one level of the type scale), equal art size
  hier     A's type (or art) is clearly bigger than B's
  contain  element level: copy inside its panel, label in its button, caption
           on its photo, content on its backdrop or slab

Survival is the importance-weighted share of relations that still hold, per
type, averaged with type weights. Relations that live on one axis (align,
order, band, gap) may flip with the flow when the target's shape differs a lot
from the source's (a column becoming a row): the transposed relation earns
partial credit, and a break costs less in proportion to how much the aspect
ratio changed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .geom import Box, el_box, el_visual_box, union_all

# weight of each relation type in the survival average
TYPE_W = {"contain": 3.0, "read": 2.0, "hier": 1.5, "align": 1.5, "eqsize": 1.0, "order": 1.0,
          "band": 1.0, "gap": 1.0}
AXIAL = ("align", "order", "band", "gap")
DROPPED = 0.5  # a relation lost because a unit was dropped (dropping is penalised on its own)
_T = {"L": "T", "C": "M", "R": "B", "T": "L", "M": "C", "B": "R"}


@dataclass
class Rel:
    kind: str
    a: str
    b: str
    sub: str = ""
    c: Optional[str] = None  # third unit (gap)
    w: float = 1.0
    axis: str = ""  # "x" / "y" for relations that live on one axis
    val: float = 0.0  # source measurement where a check needs it (containment share)


@dataclass
class Unit:
    key: str
    box: Box
    imp: float
    fs: float  # largest visible font size, 0 when the unit has no copy
    text: bool  # a plain text unit (eligible for type-scale relations)
    art: bool  # an image / composite unit (eligible for art-size relations)


@dataclass
class Graph:
    W: float
    H: float
    rels: List[Rel]
    units: Dict[str, Unit]
    look: Dict[str, float] = field(default_factory=dict)  # the source's own look metrics


# ----------------------------------------------------------------------------
# measuring units from elements


def _inks(it, out: Dict[str, tuple]) -> None:
    if it.kind == "image" and it.els:
        out[it.els[0]["id"]] = it.ink
    if it.kind == "composite":
        if it.host is not None:
            _inks(it.host, out)
        for s in it.sats:
            _inks(s, out)


def _visible(e: dict) -> bool:
    try:
        return float(e.get("opacity", 1.0) if e.get("opacity") is not None else 1.0) > 0.0
    except (TypeError, ValueError):
        return True


def elem_box(e: dict, ink: Optional[tuple] = None) -> Box:
    """What an element paints: glyph box for text, ink box for (unrotated) art."""
    if e.get("type") == "text":
        from .textfit import drawn_box

        return drawn_box(e)
    if e.get("type") == "image" and ink is not None and not float(e.get("rotation", 0) or 0):
        b = el_box(e)
        f0, f1, f2, f3 = ink
        return Box(b.x + f0 * b.w, b.y + f1 * b.h, (f2 - f0) * b.w, (f3 - f1) * b.h)
    return el_visual_box(e)


def units_of(d, elements: Iterable[dict], hidden: Set[str] = frozenset()) -> Dict[str, Unit]:
    """Unit boxes of the design's items as laid out in ``elements`` (by id)."""
    by_id = {e["id"]: e for e in elements}
    inks: Dict[str, tuple] = {}
    for it in d.items:
        _inks(it, inks)
    out: Dict[str, Unit] = {}
    for it in d.items:
        if it.role in ("missing", "void"):
            continue
        els = [by_id[e["id"]] for e in it.els if e["id"] in by_id and e["id"] not in hidden]
        els = [e for e in els if _visible(e)]
        if not els:
            continue
        boxes = [elem_box(e, inks.get(e["id"])) for e in els]
        boxes = [b for b in boxes if b.w > 0 and b.h > 0]
        if not boxes:
            continue
        fs = max((float(e.get("font_size", 0) or 0) for e in els if e.get("type") == "text"), default=0.0)
        out[it.key] = Unit(key=it.key, box=union_all(boxes), imp=float(it.importance), fs=fs,
                           text=it.kind == "text", art=it.kind in ("image", "composite"))
    return out


# ----------------------------------------------------------------------------
# pairwise geometry


def _anchors(b: Box, axis: str) -> Dict[str, float]:
    if axis == "x":
        return {"L": b.x, "C": b.cx, "R": b.r}
    return {"T": b.y, "M": b.cy, "B": b.b}


def _aligned(a: Box, b: Box, axis: str, tol: float) -> str:
    pa, pb = _anchors(a, axis), _anchors(b, axis)
    return "".join(k for k in pa if abs(pa[k] - pb[k]) <= tol)


def _ov(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def _before(a: Box, b: Box, axis: str) -> bool:
    """a entirely before b along the axis (a sliver of overlap is allowed)."""
    if axis == "y":
        return a.cy < b.cy and a.b <= b.y + 0.15 * min(a.h, b.h)
    return a.cx < b.cx and a.r <= b.x + 0.15 * min(a.w, b.w)


def _band(a: Box, b: Box, sub: str) -> bool:
    if sub == "row":  # side by side, sharing a y-band
        return _ov(a.y, a.b, b.y, b.b) >= 0.5 * min(a.h, b.h) and _ov(a.x, a.r, b.x, b.r) <= 0.15 * min(a.w, b.w)
    return _ov(a.x, a.r, b.x, b.r) >= 0.5 * min(a.w, b.w) and _ov(a.y, a.b, b.y, b.b) <= 0.15 * min(a.h, b.h)


def _contains(outer: Box, inner: Box) -> bool:
    return inner.area > 0 and outer.inter(inner) >= 0.9 * inner.area


def _neighbours(units: Dict[str, Unit], axis: str) -> Dict[str, str]:
    """Each unit's next unit along the axis within its own column (or row)."""
    nxt = {}
    for k, u in units.items():
        best, bd = None, 1e18
        for k2, v in units.items():
            if k2 == k or not _before(u.box, v.box, axis):
                continue
            if axis == "y":
                share = _ov(u.box.x, u.box.r, v.box.x, v.box.r)
                gap = v.box.y - u.box.b
            else:
                share = _ov(u.box.y, u.box.b, v.box.y, v.box.b)
                gap = v.box.x - u.box.r
            if share <= 0:
                continue
            if gap < bd:
                best, bd = k2, gap
        if best is not None:
            nxt[k] = best
    return nxt


def _gap(a: Box, b: Box, axis: str) -> float:
    return (b.y - a.b) if axis == "y" else (b.x - a.r)


def _segments(keys: List[str], boxes: Dict[str, Box], axis: str) -> List[List[str]]:
    def span(k):
        b = boxes[k]
        return (b.y, b.b) if axis == "y" else (b.x, b.r)

    segs: List[List[str]] = []
    end = seg_min = 0.0
    for k in sorted(keys, key=lambda k: span(k)[0]):
        a, b = span(k)
        size = max(1.0, b - a)
        if segs and a < end - 0.15 * min(size, seg_min):
            segs[-1].append(k)
            end = max(end, b)
            seg_min = min(seg_min, size)
            continue
        segs.append([k])
        end, seg_min = b, size
    return segs


def reading_order(boxes: Dict[str, Box]) -> List[str]:
    """XY-cut reading order: bands top to bottom, columns left to right."""

    def rec(keys: List[str], depth: int) -> List[str]:
        if len(keys) <= 1:
            return list(keys)
        if depth > 12:
            return sorted(keys, key=lambda k: (boxes[k].cy, boxes[k].x))
        bands = _segments(keys, boxes, "y")
        if len(bands) > 1:
            return [k for b in bands for k in rec(b, depth + 1)]
        cols = _segments(keys, boxes, "x")
        if len(cols) > 1:
            return [k for c in cols for k in rec(c, depth + 1)]
        return sorted(keys, key=lambda k: (boxes[k].cy, boxes[k].x))

    return rec(sorted(boxes), 0)


# ----------------------------------------------------------------------------
# the source graph


def _pair_w(a: Unit, b: Unit) -> float:
    return math.sqrt(max(1.0, a.imp) * max(1.0, b.imp)) / 100.0


def _containers(d) -> Tuple[Dict[str, dict], Dict[str, float]]:
    """Elements that can hold others (not the background, not free decoration),
    and the importance each element carries (its unit's)."""
    imp: Dict[str, float] = {}
    for it in d.items:
        for e in it.els:
            imp[e["id"]] = float(it.importance)
    for bd in d.backdrops:
        imp[bd.el["id"]] = max((m.importance for m in bd.members), default=40.0)
    bg = {e["id"] for e in d.backgrounds}
    plain_decor = {e["id"] for e in d.decor if not e.get("_slab")}
    holders = {e["id"]: e for e in d.elements
               if e.get("type") in ("shape", "image") and e["id"] not in bg and e["id"] not in plain_decor}
    return holders, imp


def extract(d) -> Graph:
    W, H = d.W, d.H
    units = units_of(d, d.elements)
    keys = list(units)
    tx, ty = max(1.5, 0.006 * W), max(1.5, 0.006 * H)
    rels: List[Rel] = []
    for i, ka in enumerate(keys):
        a = units[ka]
        for kb in keys[i + 1:]:
            b = units[kb]
            w = _pair_w(a, b)
            if _contains(a.box, b.box) or _contains(b.box, a.box):
                continue  # nested units: containment is judged at the element level
            for axis, tol in (("x", tx), ("y", ty)):
                sub = _aligned(a.box, b.box, axis, tol)
                if sub:
                    rels.append(Rel("align", ka, kb, sub=sub, w=w, axis=axis))
            for axis, sub in (("y", "above"), ("x", "left")):
                if _before(a.box, b.box, axis):
                    rels.append(Rel("order", ka, kb, sub=sub, w=w, axis=axis))
                elif _before(b.box, a.box, axis):
                    rels.append(Rel("order", kb, ka, sub=sub, w=w, axis=axis))
            for sub, axis in (("row", "y"), ("col", "x")):
                if _band(a.box, b.box, sub):
                    rels.append(Rel("band", ka, kb, sub=sub, w=w, axis=axis))
            if a.text and b.text and a.fs > 0 and b.fs > 0:
                if abs(a.fs - b.fs) <= 0.5:
                    rels.append(Rel("eqsize", ka, kb, sub="fs", w=w))
                elif a.fs >= 1.15 * b.fs:
                    rels.append(Rel("hier", ka, kb, sub="fs", w=w))
                elif b.fs >= 1.15 * a.fs:
                    rels.append(Rel("hier", kb, ka, sub="fs", w=w))
            elif a.art and b.art:
                if abs(a.box.h / max(1e-6, b.box.h) - 1.0) <= 0.04:
                    rels.append(Rel("eqsize", ka, kb, sub="h", w=w))
                elif abs(a.box.w / max(1e-6, b.box.w) - 1.0) <= 0.04:
                    rels.append(Rel("eqsize", ka, kb, sub="w", w=w))
                elif a.box.area >= 1.8 * b.box.area:
                    rels.append(Rel("hier", ka, kb, sub="area", w=w))
                elif b.box.area >= 1.8 * a.box.area:
                    rels.append(Rel("hier", kb, ka, sub="area", w=w))
    # equal gaps along columns and rows
    for axis in ("y", "x"):
        nxt = _neighbours(units, axis)
        for ka, kb in nxt.items():
            kc = nxt.get(kb)
            if kc is None:
                continue
            g1 = _gap(units[ka].box, units[kb].box, axis)
            g2 = _gap(units[kb].box, units[kc].box, axis)
            if g1 > 0 and g2 > 0 and abs(g1 - g2) <= max(2.0, 0.12 * max(g1, g2)):
                w = (units[ka].imp * units[kb].imp * units[kc].imp) ** (1 / 3) / 100.0
                rels.append(Rel("gap", ka, kb, c=kc, sub=axis, w=w, axis=axis))
    # reading order
    order = reading_order({k: u.box for k, u in units.items()})
    for ka, kb in zip(order, order[1:]):
        rels.append(Rel("read", ka, kb, w=_pair_w(units[ka], units[kb])))
    # element-level containment
    holders, imp = _containers(d)
    owner = {e["id"]: it.key for it in d.items for e in it.els}
    for bd in d.backdrops:
        for m in bd.members:
            for e in m.els:
                owner.setdefault(e["id"], m.key)
    inks: Dict[str, tuple] = {}
    for it in d.items:
        _inks(it, inks)
    bg = {e["id"] for e in d.backgrounds}
    boxes = {e["id"]: elem_box(e, inks.get(e["id"])) for e in d.elements if e["id"] not in bg}
    for pid, p in holders.items():
        pb = boxes.get(pid)
        if pb is None or pb.area <= 0:
            continue
        for e in d.elements:
            cid = e["id"]
            if cid == pid or cid in bg or cid not in boxes or e["z_index"] <= p["z_index"]:
                continue
            cb = boxes[cid]
            if cb.area <= 0 or cb.area > 0.8 * pb.area:
                continue
            share = pb.inter(cb) / cb.area
            if share < 0.9:
                continue
            same = owner.get(cid) is not None and owner.get(cid) == owner.get(pid)
            framed = pid not in owner or p.get("_slab")  # a backdrop or slab
            w = imp.get(cid, 40.0) / 100.0 * (1.0 if same or framed else 0.5)
            rels.append(Rel("contain", pid, cid, w=w, val=share))
    g = Graph(W=W, H=H, rels=rels, units=units)
    g.look = look(units, W, H)
    return g


# ----------------------------------------------------------------------------
# evaluating a layout


def necessity(g: Graph, W: float, H: float) -> float:
    """How much the target's shape forces relations along one axis to change."""
    return min(1.0, abs(math.log((W / H) / (g.W / g.H))) / math.log(4.0))


def _fs_eq(a: float, b: float) -> bool:
    return abs(a - b) <= max(1.0, 0.03 * max(a, b))


def evaluate(g: Graph, d, elements: Sequence[dict], W: float, H: float,
             hidden: Set[str] = frozenset(), trace: Optional[list] = None) -> Dict[str, float]:
    """Survival of the source's relations in a layout, plus raw per-type shares.
    ``trace`` (a list) collects (relation, credit) pairs for debugging."""
    units = units_of(d, elements, hidden)
    nec = necessity(g, W, H)
    tx, ty = max(1.5, 0.006 * W), max(1.5, 0.006 * H)
    tol = {"x": tx, "y": ty}
    other = {"x": "y", "y": "x"}
    order = reading_order({k: u.box for k, u in units.items()})
    rank = {k: i for i, k in enumerate(order)}
    by_id = {e["id"]: e for e in elements}
    inks: Dict[str, tuple] = {}
    for it in d.items:
        _inks(it, inks)
    tr = 0.4 + 0.6 * nec  # credit for a relation that flipped with the flow
    soft = 1.0 - 0.6 * nec  # what an axis break costs, relative to a non-axis break
    acc: Dict[str, List[float]] = {}  # kind -> [sum w, sum w*loss, sum w*held]
    ebox: Dict[str, Box] = {}

    def _add(r: Rel, c: float, m: float) -> None:
        if trace is not None:
            trace.append((r, c))
        s = acc.setdefault(r.kind, [0.0, 0.0, 0.0])
        s[0] += r.w
        s[1] += r.w * (1.0 - c) * m
        s[2] += r.w * (1.0 if c >= 1.0 else 0.0)

    def box_of(eid: str) -> Optional[Box]:
        if eid not in ebox:
            e = by_id.get(eid)
            ebox[eid] = elem_box(e, inks.get(eid)) if e is not None and eid not in hidden and _visible(e) else None
        return ebox[eid]

    for r in g.rels:
        if r.kind == "contain":
            pb, cb = box_of(r.a), box_of(r.b)
            if pb is None or cb is None or cb.area <= 0:
                c, m = 0.0, DROPPED  # dropped for this format: the relation is gone
                _add(r, c, m)
                continue
            need = min(0.9, r.val) - 0.02
            share = pb.inter(cb) / cb.area
            c = 1.0 if share >= need else max(0.0, (share - 0.5) / max(1e-6, need - 0.5))
            m = 1.0
        else:
            ua, ub = units.get(r.a), units.get(r.b)
            uc = units.get(r.c) if r.c else None
            if ua is None or ub is None or (r.c and uc is None):
                _add(r, 0.0, DROPPED)
                continue
            a, b = ua.box, ub.box
            m = soft if r.kind in AXIAL else 1.0
            if r.kind == "align":
                now = _aligned(a, b, r.axis, tol[r.axis])
                if any(s in now for s in r.sub):
                    c = 1.0
                elif now:
                    c = 0.5  # still on a shared line, a different one
                else:
                    flip = _aligned(a, b, other[r.axis], tol[other[r.axis]])
                    c = tr if any(_T[s] in flip for s in r.sub) else 0.0
            elif r.kind == "order":
                if _before(a, b, r.axis):
                    c = 1.0
                elif _before(a, b, other[r.axis]):
                    c = tr
                else:
                    c = 0.0
            elif r.kind == "band":
                if _band(a, b, r.sub):
                    c = 1.0
                elif _band(a, b, "col" if r.sub == "row" else "row"):
                    c = tr
                else:
                    c = 0.0
            elif r.kind == "gap":
                cbox = uc.box
                c = 0.0
                for ax, credit in ((r.axis, 1.0), (other[r.axis], tr)):
                    if _before(a, b, ax) and _before(b, cbox, ax):
                        g1, g2 = _gap(a, b, ax), _gap(b, cbox, ax)
                        if abs(g1 - g2) <= max(2.0, 0.15 * max(g1, g2, 0.0)):
                            c = credit
                            break
            elif r.kind == "read":
                c = 1.0 if rank.get(r.a, 0) < rank.get(r.b, 0) else 0.0
            elif r.kind == "eqsize":
                if r.sub == "fs":
                    c = 1.0 if _fs_eq(ua.fs, ub.fs) else 0.0
                else:
                    va, vb = (a.h, b.h) if r.sub == "h" else (a.w, b.w)
                    c = 1.0 if abs(va / max(1e-6, vb) - 1.0) <= 0.06 else 0.0
            elif r.kind == "hier":
                if r.sub == "fs":
                    c = 1.0 if ua.fs > ub.fs + 0.5 else (0.5 if abs(ua.fs - ub.fs) <= 0.5 else 0.0)
                else:
                    c = 1.0 if a.area >= b.area else 0.0
            else:
                continue
        _add(r, c, m)
    out: Dict[str, float] = {"necessity": nec}
    out.update({"look_" + k: v for k, v in look(units, W, H).items()})
    out["n"] = len(g.rels)
    tw = sum(TYPE_W[k] for k, s in acc.items() if s[0] > 0)
    if tw <= 0:
        out.update(survival=1.0, raw=1.0)
        return out
    out["survival"] = 1.0 - sum(TYPE_W[k] * s[1] / s[0] for k, s in acc.items() if s[0] > 0) / tw
    out["raw"] = sum(TYPE_W[k] * s[2] / s[0] for k, s in acc.items() if s[0] > 0) / tw
    for k, s in acc.items():
        if s[0] > 0:
            out["held_" + k] = s[2] / s[0]
    return out


# ----------------------------------------------------------------------------
# look metrics


def look(units: Dict[str, Unit], W: float, H: float) -> Dict[str, float]:
    """Alignment crispness and simplicity, spacing regularity, margins, density.

    near_miss   share of units that sit slightly off the line of a unit in
                the same column (or row): more than ~0.2% and less than 1.2%
                of the canvas away, with no exact line shared on that axis.
                The "almost aligned" look
    x_lines     distinct left/centre/right lines used, per unit (simplicity)
    gap_cv      coefficient of variation of the gaps between consecutive units
                along the main axis (columns, or rows on strips)
    margin_x/y  imbalance of the outer margins, |l - r| / W (and |t - b| / H)
    density     summed unit area / canvas area
    """
    ks = list(units)
    n = len(ks)
    out = {"near_miss": 0.0, "x_lines": 0.0, "gap_cv": 0.0, "margin_x": 0.0, "margin_y": 0.0, "density": 0.0}
    if n == 0:
        return out
    near = set()
    for i, ka in enumerate(ks):
        for kb in ks[i + 1:]:
            a, b = units[ka].box, units[kb].box
            for axis, dim in (("x", W), ("y", H)):
                share = _ov(a.x, a.r, b.x, b.r) if axis == "x" else _ov(a.y, a.b, b.y, b.b)
                if share <= 0:  # only units in one column (row) read as aligned or not
                    continue
                pa, pb = _anchors(a, axis), _anchors(b, axis)
                ds = [abs(pa[k] - pb[k]) for k in pa]
                if min(ds) <= max(1.0, 0.002 * dim):
                    continue
                if min(ds) <= 0.012 * dim:
                    near.update((ka, kb))
    out["near_miss"] = len(near) / n
    # simplicity: each unit picks the x-line it shares with most others
    lines: List[float] = []
    for ka in ks:
        a = units[ka].box
        pa = _anchors(a, "x")
        best = max(pa, key=lambda k: (sum(1 for kb in ks if kb != ka and
                                          abs(_anchors(units[kb].box, "x")[k] - pa[k]) <= 1.0), k == "C"))
        v = pa[best]
        if not any(abs(v - l) <= 1.0 for l in lines):
            lines.append(v)
    out["x_lines"] = len(lines) / n
    best_cv, best_n = 0.0, 0
    for axis in ("y", "x"):
        nxt = _neighbours(units, axis)
        gaps = [_gap(units[a].box, units[b].box, axis) for a, b in nxt.items()]
        gaps = [g for g in gaps if g > 0]
        if len(gaps) >= 2 and len(gaps) > best_n:
            mu = sum(gaps) / len(gaps)
            sd = math.sqrt(sum((g - mu) ** 2 for g in gaps) / len(gaps))
            best_cv, best_n = sd / max(1e-6, mu), len(gaps)
    out["gap_cv"] = best_cv
    u = union_all(units[k].box for k in ks)
    l, t, r, b = max(0.0, u.x), max(0.0, u.y), max(0.0, W - u.r), max(0.0, H - u.b)
    out["margin_x"] = abs(l - r) / max(1.0, W)
    out["margin_y"] = abs(t - b) / max(1.0, H)
    out["density"] = sum(units[k].box.area for k in ks) / max(1.0, W * H)
    return out
