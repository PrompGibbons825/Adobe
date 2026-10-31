"""Keep and re-space: a collage or grid of pictures, re-flowed for the new shape.

A poster that is a set of pictures packed edge to edge or with even gutters (a photo collage,
a grid of products, a mosaic of one large and two small shots) reads as one block. Scaled as a
block onto a very different shape it comes out small, with the canvas around it empty; laid
out one picture at a time it falls apart. A designer re-flows it instead: the same pictures,
in the same reading order and each at its own proportions, packed into a new grid whose row
or column count suits the new shape, with the gutters the source had, filling the area the
copy leaves free. The copy goes in a band of its own beside or below the pictures.

``find`` recognises such a set from geometry alone (3 to 9 pictures that do not overlap and
together cover most of their joint box, no copy on them). ``plan_collage`` builds the layout:
the copy column is solved first in a band of the canvas, the band is shrunk to what the copy
uses, and the pictures are packed into what is left by the best of

  * a justified layout (every contiguous split of the reading order into rows, each row one
    height and as wide as the area; or the same with columns), or
  * for pictures of one shape, a uniform grid of cols x rows, the last row centred.

Every picture keeps its own aspect ratio and stays inside the canvas: nothing is cropped or
redrawn. The ranking (score.py and the pixel critic) decides whether the re-flow beats the
other candidates.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from .geom import Box, el_box, union_all

MIN_TILES, MAX_TILES = 3, 9
PACKED = 0.68  # the pictures cover at least this share of their joint box
OVERLAP = 0.08  # no two pictures overlap by more than this share of the smaller
SPREAD = 9.0  # the largest picture is at most this many times the area of the smallest
PROMINENT = 0.18  # the set takes at least this share of the source canvas
ON_COPY = 0.2  # no copy has more than this share of its box on the set
GUTTER = 0.06  # gutters at most this share of the set's short side


@dataclass
class Collage:
    items: list  # analyze.Item, each an image or a composite of images
    tiles: List[dict]  # every picture element, in reading order
    owner: Dict[str, object]  # element id -> its item
    box: Box  # the set's joint box in the source
    gutter: float  # median gutter / short side of the set's box
    bleeds: bool  # the set touched the canvas edges in the source (a full-bleed collage)
    first: bool  # the set came before the copy (above it / left of it) in the source


def _pictures(it) -> bool:
    return it.kind in ("image", "composite") and it.role in ("hero", "image") and \
        all(e.get("type") == "image" for e in it.els) and not it.rotation and not it.corner


def find(d) -> Optional[Collage]:
    memo = getattr(d, "_collage", False)
    if memo is not False:
        return memo
    try:
        c = _find(d)
    except Exception:  # noqa: BLE001  a detector that fails adds no candidates
        c = None
    d._collage = c
    return c


def _find(d) -> Optional[Collage]:
    items = [it for it in d.items if _pictures(it)]
    tiles = [e for it in items for e in it.els]
    if not (MIN_TILES <= len(tiles) <= MAX_TILES):
        return None
    if any(abs(float(e.get("rotation", 0) or 0)) > 1e-6 for e in tiles):
        return None
    boxes = [el_box(e) for e in tiles]
    if any(b.w <= 1 or b.h <= 1 for b in boxes):
        return None
    for a, b in itertools.combinations(boxes, 2):
        if a.inter(b) > OVERLAP * min(a.area, b.area):
            return None
    areas = [b.area for b in boxes]
    if max(areas) > SPREAD * min(areas):
        return None
    u = union_all(boxes)
    if sum(areas) < PACKED * u.area or u.area < PROMINENT * d.W * d.H:
        return None
    copy = [it for it in d.items if not _pictures(it) and it.role not in ("missing", "void")]
    if any(it.src_box.inter(u) > ON_COPY * max(1.0, it.src_box.area) for it in copy):
        return None
    # gutters: for each picture, the nearest clear gap to a neighbour that faces it
    gaps = []
    for i, a in enumerate(boxes):
        best = None
        for j, b in enumerate(boxes):
            if i == j:
                continue
            if min(a.b, b.b) - max(a.y, b.y) > 0.3 * min(a.h, b.h):  # side by side
                g = max(b.x - a.r, a.x - b.r)
            elif min(a.r, b.r) - max(a.x, b.x) > 0.3 * min(a.w, b.w):  # one above the other
                g = max(b.y - a.b, a.y - b.b)
            else:
                continue
            if g > -1.0 and (best is None or g < best):
                best = max(0.0, g)
        if best is not None:
            gaps.append(best)
    gaps.sort()
    short = max(1.0, min(u.w, u.h))
    gutter = min(GUTTER, (gaps[len(gaps) // 2] / short) if gaps else 0.02)
    t = 0.02 * min(d.W, d.H)
    bleeds = sum((u.x <= t, u.y <= t, u.r >= d.W - t, u.b >= d.H - t)) >= 2
    owner = {e["id"]: it for it in items for e in it.els}
    order = sorted(range(len(tiles)), key=lambda k: (round(boxes[k].cy / (0.35 * min(b.h for b in boxes))), boxes[k].cx))
    tiles = [tiles[k] for k in order]
    cu = union_all([it.src_box for it in copy]) if copy else None
    first = True
    if cu is not None:
        first = (cu.cy - u.cy) / d.H + (cu.cx - u.cx) / d.W >= 0
    return Collage(items=items, tiles=tiles, owner=owner, box=u, gutter=gutter, bleeds=bleeds, first=first)


# ----------------------------------------------------------------------------
# packing


def _justified(asp: List[float], W: float, H: float, g: float) -> Tuple[float, List[Tuple[float, float, float, float]]]:
    """The best split of the reading order into rows (each row one height, filling the width,
    the whole scaled down to the height if it is taller). Returns (area, boxes)."""
    n = len(asp)
    best = (0.0, [])
    for cuts in range(1 << (n - 1)):
        rows, cur = [], [0]
        for k in range(1, n):
            if cuts >> (k - 1) & 1:
                rows.append(cur)
                cur = []
            cur.append(k)
        rows.append(cur)
        hs = [(W - (len(r) - 1) * g) / sum(asp[k] for k in r) for r in rows]
        if min(hs) <= 0:
            continue
        # rows much taller than their neighbours read as a mistake, not a mosaic
        if max(hs) > 2.6 * min(hs):
            continue
        tot = sum(hs) + (len(rows) - 1) * g
        t = min(1.0, (H - (len(rows) - 1) * g) / max(1e-6, sum(hs)))
        if t <= 0:
            continue
        hs = [h * t for h in hs]
        tot = sum(hs) + (len(rows) - 1) * g
        y = (H - tot) / 2.0
        out: List[Tuple[float, float, float, float]] = [None] * n  # type: ignore[list-item]
        area = 0.0
        for r, h in zip(rows, hs):
            rw = sum(asp[k] * h for k in r) + (len(r) - 1) * g
            x = (W - rw) / 2.0
            for k in r:
                w = asp[k] * h
                out[k] = (x, y, w, h)
                area += w * h
                x += w + g
            y += h + g
        if area > best[0] * 1.0001:
            best = (area, out)
    return best


def _uniform(asp: List[float], W: float, H: float, g: float) -> Tuple[float, list]:
    """Pictures of one shape in a cols x rows grid, the last row centred."""
    n = len(asp)
    a = sum(asp) / n
    best = (0.0, [])
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        if rows > 1 and n - (rows - 1) * cols < 1:
            continue
        w = min((W - (cols - 1) * g) / cols, ((H - (rows - 1) * g) / rows) * a)
        if w <= 0:
            continue
        h = w / a
        gw, gh = cols * w + (cols - 1) * g, rows * h + (rows - 1) * g
        y0 = (H - gh) / 2.0
        out = []
        for k in range(n):
            ri, ci = divmod(k, cols)
            in_row = min(cols, n - ri * cols)
            rw = in_row * w + (in_row - 1) * g
            x0 = (W - rw) / 2.0
            out.append((x0 + ci * (w + g), y0 + ri * (h + g), w, h))
        area = n * w * h
        if area > best[0]:
            best = (area, out)
    return best


def pack(asp: List[float], W: float, H: float, g: float, alike: bool = False) -> list:
    """Boxes (x, y, w, h) in a W x H area for pictures of aspect ``asp`` (reading order);
    ``alike``: pictures of one shape and size, laid out as a uniform grid."""
    cands = []
    ja, jb = _justified(asp, W, H, g)
    if jb:
        cands.append((ja, jb))
    # the same on its side: columns of one width, each as tall as the area
    ta, tb = _justified([1.0 / a for a in asp], H, W, g)
    if tb:
        cands.append((ta, [(y, x, h, w) for x, y, w, h in tb]))
    if alike:  # alike pictures stay alike: one size, in a true grid (never a mosaic of them)
        ua, ub = _uniform(asp, W, H, g)
        return ub
    if not cands:
        return []
    return max(cands, key=lambda c: c[0])[1]


# ----------------------------------------------------------------------------
# the layout


def plan_collage(d, ctx, hidden, name: str = "collage", insets=(0, 0, 0, 0), side: str = "bottom",
                 frac: float = 0.36):
    """The pictures re-flowed into the canvas, the copy in a band at ``side`` (top, bottom,
    left, right) taking at most ``frac`` of the canvas. None if the design is no collage or the
    copy does not fit."""
    from .compose import Plan, _settle_floaters, _visible, collides, column, margins, scale_bounds, solve
    from .items import Real
    from .structure import dominant_align, rows_for

    c = find(d)
    if c is None:
        return None
    vis = _visible(d, hidden)
    ids = {id(it) for it in c.items}
    if any(id(it) not in {id(v) for v in vis} for it in c.items):
        return None
    body = [it for it in vis if id(it) not in ids and not it.corner]
    floaters = [it for it in vis if it.corner and id(it) not in ids]
    W, H = ctx.W, ctx.H
    mx, my = margins(d, ctx)
    il, it_, ir, ib = insets
    vert = side in ("top", "bottom")
    placed = []
    s, relax, capped = 1.0, 1.0, False
    used = 0.0  # the band's extent along the split axis, margins included
    if body:
        if vert:
            band = Box(mx + il, my + it_, W - 2 * mx - il - ir, frac * H - 2 * my)
        else:
            band = Box(mx + il, my + it_, frac * W - 2 * mx, H - 2 * my - it_ - ib)
        if band.w < 20 or band.h < 16:
            return None
        dom = dominant_align(body)
        align = {"left": "start", "center": "center", "right": "end"}.get(dom, "start")
        rows = rows_for(body)
        root, _ = column(d, rows, force_align=align)
        lo, hi = scale_bounds(d, ctx, "split" if not vert else "stack")
        s, laid, relax = solve(root, ctx, band, lo, hi)
        if laid is None:
            return None
        capped = s >= hi * 0.999
        parts, gaps = laid.parts, laid.gap_list  # type: ignore[attr-defined]
        ch = sum(p.h for p in parts) + sum(gaps)
        cw = max(p.w for p in parts)
        if vert:
            used = ch + 2 * my
            by = my + it_ if side == "top" else H - my - ib - ch
            bx0 = band.x
        else:
            used = cw + 2 * mx
            by = band.y + (band.h - ch) / 2.0
            bx0 = band.x if side == "left" else W - mx - ir - cw
            band = Box(bx0, band.y, cw, band.h)
        bw = cw
        bx = bx0 if (align == "start" or not vert) else (band.r - bw if align == "end" else band.cx - bw / 2.0)
        y = by
        for i, p in enumerate(parts):
            a = root.align[i] if i < len(root.align) else align
            x = bx if a == "start" else (bx + bw - p.w if a == "end" else bx + (bw - p.w) / 2.0)
            placed += p.place(x, y)
            y += p.h + (gaps[i] if i < len(gaps) else 0.0)

    # the pictures' area: what the band leaves, inset by the margins unless the source bled
    ex, ey = (0.0, 0.0) if c.bleeds else (mx, my)
    gx, gy, gw, gh = ex, ey, W - 2 * ex, H - 2 * ey
    if body:
        sep = 0.4 * (my if vert else mx)  # the band's inner margin already parts it from the pictures
        if side == "top":
            gy, gh = used - sep, H - ey - (used - sep)
        elif side == "bottom":
            gh = H - ey - used + sep - gy
        elif side == "left":
            gx, gw = used - sep, W - ex - (used - sep)
        else:
            gw = W - ex - used + sep - gx
    gx, gy = gx + il, gy + it_
    gw, gh = gw - il - ir, gh - it_ - ib
    if gw < 20 or gh < 20:
        return None
    asp = [max(1e-3, el_box(e).w) / max(1e-3, el_box(e).h) for e in c.tiles]
    g = c.gutter * min(gw, gh)
    sizes = [el_box(e).area for e in c.tiles]
    alike = max(asp) <= 1.1 * min(asp) and max(sizes) <= 1.3 * min(sizes)
    cells = pack(asp, gw, gh, g, alike)
    if not cells:
        return None
    cells = [(gx + x, gy + y, w, h) for x, y, w, h in cells]
    # snap the whole block toward the band (keeps one clean channel between copy and pictures)
    ux0, uy0 = min(b[0] for b in cells), min(b[1] for b in cells)
    ux1, uy1 = max(b[0] + b[2] for b in cells), max(b[1] + b[3] for b in cells)
    dx = dy = 0.0
    if body and vert:
        dy = (gy - uy0) if side == "top" else (gy + gh - uy1)
        dy *= 0.5
    elif body:
        dx = (gx - ux0) if side == "left" else (gx + gw - ux1)
        dx *= 0.5
    cells = [(x + dx, y + dy, w, h) for x, y, w, h in cells]
    at = {e["id"]: cell for e, cell in zip(c.tiles, cells)}

    pics = []
    for item in c.items:
        host = item.els[0]
        hx, hy, hw, hh = at[host["id"]]
        mine = [(e, at[e["id"]]) for e in item.els]

        def emit(x: float, y: float, mine=mine, hx=hx, hy=hy) -> list:
            out = []
            for e, (cx, cy, cw_, ch_) in mine:
                o = {k: v for k, v in e.items() if k != "_idx"}
                o.update(x=cx + x - hx, y=cy + y - hy, width=cw_, height=ch_)
                out.append(o)
            return out

        sb = el_box(host)
        r = Real(item=item, w=hw, h=hh, scale=hw / max(1.0, sb.w), emit_fn=emit)
        r._oh = (0.0, 0.0, 0.0, 0.0)
        pics.append((r, hx, hy))
    gap = 0.6 * ctx.base_fs
    for pr, px, py in pics:
        for e, (cx, cy, cw_, ch_) in [(e, at[e["id"]]) for e in pr.item.els]:
            if any(collides(Box(cx, cy, cw_, ch_), Box(x, y, r.w, r.h), 0.5 * gap) for r, x, y in placed):
                return None
    fl = []
    if floaters:
        allp = placed + pics
        fl, bad, _, _ = _settle_floaters(allp, floaters, s, ctx, Box(0, 0, W, H), mx, my, relax)
        if bad or any(collides(Box(fx, fy, fr.w, fr.h), Box(x, y, r.w, r.h), gap)
                      for fr, fx, fy in fl for r, x, y in allp):
            return None
    zone = Box(gx, gy, gw, gh)
    return Plan(name=name, placed=pics + placed, hidden=list(hidden), s=s, zone=zone, floaters=fl,
                relax=relax, capped=capped)


def variants(d, ctx):
    """(name, planner, options) for the collage candidates on this canvas: the copy band on the
    sides the shape allows, at three band sizes (the copy is set as large as its band allows)."""
    if find(d) is None:
        return []
    r = ctx.W / ctx.H
    c = d._collage
    sides = []
    # the copy keeps its place in the reading order: after the pictures (below or right of
    # them) if it came after them in the source, before them otherwise
    if r <= 1.35:
        sides.append("bottom" if c.first else "top")
    if r >= 0.8:
        sides.append("right" if c.first else "left")
    out = []
    for sd in sides:
        vert = sd in ("top", "bottom")
        for f in ((0.24, 0.32, 0.42) if vert else (0.3, 0.4, 0.5)):
            out.append((f"collage-{sd}-{f:g}", plan_collage, dict(side=sd, frac=f)))
    return out
