"""Edge-anchored regions: a design split into a photo and a flat colour panel.

A very common layout fills one side of the canvas with a photo, edge to edge,
and the other side with a flat colour panel that carries the copy. The two
together tile the canvas. A designer resizing it keeps that structure: the
photo region and the panel region still tile the new canvas (side by side on a
wide one, stacked on a tall one), each edge to edge, the copy laid out in its
panel with comfortable padding, and the photo filling its region by cropping.

``find_split`` recognises the structure from geometry alone (an opaque picture
and a flat panel that each touch at least three canvas edges, on opposite
sides, covering the canvas between them, every piece of copy on the panel).
``plan_regions`` (compose.py) builds the layout; ``photo_box`` picks how the
photo fills its region, legally:

  * at the source element's own proportions, scaled to cover the region and
    slid along the free axis; what is left over runs past the canvas edges or
    under the opaque panel (stacked above it), keeping at least 60% of the box
    on the canvas and less than half of it under the panel; or
  * at the picture's own proportions, covering the whole canvas (a cover crop,
    as a background photo is drawn), with the panel over the part it hides.

Of the legal boxes, the one that shows most of the picture with its subject in
view wins: no head cut (vision.py's face map), the subject near the middle of
the region, as much as possible of what the source showed. Nothing here
changes a pixel: the crop is made by positioning only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .geom import Box, el_box

TOL = 0.02  # share of the canvas's short side within which a box touches an edge
TILE = 0.92  # photo and panel together cover at least this share of the canvas
FRAC = (0.4, 0.6)  # the photo's share of the canvas along the split, clamped
ON_CANVAS = 0.62  # a bleeding photo box keeps at least this share on the canvas (bleed policy: 60%)
UNDER = 0.46  # ... and less than this share of it under the panel (never "hidden")
OPAQUE = 0.95  # a panel this opaque hides what runs under it
STEPS = 24  # crop positions tried along each free axis


@dataclass
class Split:
    photo: object  # analyze.Item: the picture region
    panel: object  # analyze.Backdrop: the flat panel that carries the copy
    axis: str  # "x": side by side in the source, "y": stacked
    photo_first: bool  # the photo is at the left (x) / top (y)
    frac: float  # the photo's share of the canvas along the axis
    hides: bool  # the panel is opaque and stacked above the photo: overflow may run under it


def _edges(b: Box, W: float, H: float) -> Tuple[bool, bool, bool, bool]:
    t = TOL * min(W, H)
    return (b.x <= t, b.y <= t, b.r >= W - t, b.b >= H - t)


def _covered(boxes: List[Box], W: float, H: float, n: int = 24) -> float:
    hits = 0
    for i in range(n):
        for j in range(n):
            x, y = (i + 0.5) * W / n, (j + 0.5) * H / n
            hits += any(b.x <= x <= b.r and b.y <= y <= b.b for b in boxes)
    return hits / (n * n)


@lru_cache(maxsize=128)
def solid(asset_id) -> bool:
    """An opaque picture (a photo, not a cut-out or a logo): under 5% of its
    pixels let anything through (tools/bleed.py's line)."""
    from . import assets, paths

    a = assets.info(asset_id)
    if not a.found:
        return False
    if a.opaque:
        return True
    try:
        from PIL import Image

        with Image.open(paths.find_asset(asset_id)) as im:
            im = im.convert("RGBA")
            im.thumbnail((256, 256))
            al = im.getchannel("A").histogram()
        return sum(al[:250]) / max(1, sum(al)) < 0.04
    except Exception:  # noqa: BLE001
        return False


def panel_alpha(el: dict) -> float:
    from .color import parse_rgba

    try:
        a = float(el.get("opacity", 1.0) if el.get("opacity") is not None else 1.0)
    except (TypeError, ValueError):
        a = 1.0
    return a * parse_rgba(el.get("fill_color"))[3]


def find_split(d) -> Optional[Split]:
    """The photo / panel split of this design, or None."""
    W, H = d.W, d.H
    canvas = Box(0, 0, W, H)
    for bd in d.backdrops:
        sh = bd.el
        if sh.get("type") != "shape" or abs(float(sh.get("rotation", 0) or 0)) > 0.5 or \
                sh.get("shape_type", "rectangle") != "rectangle":
            continue
        pb = el_box(sh).inter_box(canvas)
        if pb is None or pb.area < 0.2 * canvas.area:
            continue
        pe = _edges(pb, W, H)
        if sum(pe) < 3 or not any(m.core or m.role == "headline" for m in bd.members):
            continue
        members = {id(m) for m in bd.members}
        for it in d.items:
            if it.kind != "image" or it.role not in ("hero", "image") or it.rotation or id(it) in members:
                continue
            e = it.els[0]
            if e.get("type") != "image":
                continue
            if not isinstance(e.get("asset_id"), str) or not solid(e.get("asset_id")):
                continue
            ib = el_box(e).inter_box(canvas)
            if ib is None or ib.area < 0.2 * canvas.area:
                continue
            ie = _edges(ib, W, H)
            if sum(ie) < 3:
                continue
            if ie[0] and not ie[2] and pe[2] and not pe[0]:
                axis, first, frac = "x", True, ib.r / W
            elif ie[2] and not ie[0] and pe[0] and not pe[2]:
                axis, first, frac = "x", False, (W - ib.x) / W
            elif ie[1] and not ie[3] and pe[3] and not pe[1]:
                axis, first, frac = "y", True, ib.b / H
            elif ie[3] and not ie[1] and pe[1] and not pe[3]:
                axis, first, frac = "y", False, (H - ib.y) / H
            else:
                continue
            if ib.inter(pb) > 0.15 * min(ib.area, pb.area) or _covered([ib, pb], W, H) < TILE:
                continue
            others = [o for o in d.items if o is not it and id(o) not in members and o.corner is None
                      and o.role not in ("missing", "void")]
            if others:  # copy or art off the panel: not this structure
                continue
            z_photo = min(int(x.get("z_index", 0) or 0) for x in it.els)
            hides = panel_alpha(sh) >= OPAQUE and int(sh.get("z_index", 0) or 0) > z_photo
            return Split(photo=it, panel=bd, axis=axis, photo_first=first, frac=frac, hides=hides)
    return None


def regions(sp: Split, W: float, H: float, axis: str) -> Tuple[Box, Box]:
    """(photo region, panel region) on a W x H canvas split along ``axis``.
    Turned from side by side to stacked, the photo goes on top."""
    # the share that would show the photo at its source proportions, uncropped; kept along the
    # same axis, half way to the source's own share
    pb = el_box(sp.photo.els[0])
    r_s = pb.w / max(1e-6, pb.h)
    ideal = (H * r_s / W) if axis == "x" else (W / r_s / H)
    f = ideal if axis != sp.axis else 0.5 * (ideal + sp.frac)
    # turned side by side, the copy column is short of width: the photo takes at most half
    f = min(FRAC[1] if axis == sp.axis or axis == "y" else 0.5, max(FRAC[0], f))
    first = sp.photo_first if axis == sp.axis else (True if axis == "y" else sp.photo_first)
    if axis == "x":
        pw = round(W * f)
        return (Box(0, 0, pw, H), Box(pw, 0, W - pw, H)) if first else (Box(W - pw, 0, pw, H), Box(0, 0, W - pw, H))
    ph = round(H * f)
    return (Box(0, 0, W, ph), Box(0, ph, W, H - ph)) if first else (Box(0, H - ph, W, ph), Box(0, 0, W, H - ph))


# ----------------------------------------------------------------------------
# how the photo fills its region

_CACHE: Dict[tuple, Optional[Tuple[float, float, float, float]]] = {}


def photo_box(e: dict, P: Box, Q: Box, W: float, H: float, W0: float, H0: float,
              hides: bool) -> Optional[Tuple[float, float, float, float]]:
    """(x, y, w, h) of the photo element filling region ``P`` (fit: cover),
    the panel over region ``Q``; None if no legal box fills it."""
    key = (e.get("asset_id"), round(float(e.get("width", 0) or 0), 2), round(float(e.get("height", 0) or 0), 2),
           round(float(e.get("x", 0) or 0), 2), round(float(e.get("y", 0) or 0), 2),
           P.x, P.y, P.w, P.h, Q.x, Q.y, Q.w, Q.h, W, H, W0, H0, hides)
    if key not in _CACHE:
        if len(_CACHE) > 512:
            _CACHE.clear()
        try:
            _CACHE[key] = _photo_box(e, P, Q, W, H, W0, H0, hides)
        except Exception:  # noqa: BLE001  never let a crop break a layout
            _CACHE[key] = None
    return _CACHE[key]


def _inter(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _photo_box(e, P: Box, Q: Box, W, H, W0, H0, hides):
    from . import assets, vision

    a = assets.info(e.get("asset_id"))
    if not a.found or P.w < 2 or P.h < 2:
        return None
    iw, ih = a.size
    sb = el_box(e)
    if sb.w <= 0 or sb.h <= 0 or iw <= 0 or ih <= 0:
        return None
    v = vision.analyze(e.get("asset_id"))
    shown = vision.source_window(e, W0, H0, (iw, ih))

    options = []  # (box w, box h, covers canvas)
    r_s = sb.w / sb.h  # the source element's proportions (kept within the rules' 5%)
    bw = max(P.w, P.h * r_s)
    options.append((bw, bw / r_s, False))
    if hides:  # the picture's own proportions, as a canvas cover crop under the panel
        s = max(W / iw, H / ih)
        cw, ch = iw * s, ih * s
        under = Q.area / (cw * ch)
        if under > UNDER:  # zoom until the panel hides less than half of it
            k = math.sqrt(under / UNDER) * 1.01
            cw, ch = cw * k, ch * k
        options.append((cw, ch, True))

    best, best_sc = None, -1e9
    for bw, bh, full in options:
        if full:
            xs = _span(W - bw, 0.0)
            ys = _span(H - bh, 0.0)
        else:
            xs = _span(P.r - bw, P.x)
            ys = _span(P.b - bh, P.y)
        # where the renderer draws the asset inside the box (cover: centred)
        sA = max(bw / iw, bh / ih)
        dw, dh = iw * sA, ih * sA
        for x in xs:
            for y in ys:
                box = (x, y, x + bw, y + bh)
                if not full:
                    on = _inter(box, (0.0, 0.0, W, H)) / (bw * bh)
                    if on < ON_CANVAS:
                        continue
                if _inter(box, (Q.x, Q.y, Q.r, Q.b)) > (UNDER * bw * bh if hides else 1.0):
                    continue
                ox, oy = x + (bw - dw) / 2.0, y + (bh - dh) / 2.0
                win = ((P.x - ox) / dw, (P.y - oy) / dh, (P.r - ox) / dw, (P.b - oy) / dh)
                sc = _window_score(v, win, shown)
                if sc > best_sc + 1e-9:
                    best, best_sc = (x, y, bw, bh), sc
    return best


def _span(lo: float, hi: float) -> List[float]:
    if hi - lo < 1.0:
        return [min(lo, hi) if hi < lo else lo]
    n = STEPS
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def _window_score(v, win, shown) -> float:
    """How good a crop window (image fractions) is: as much of the picture as
    possible, what the source showed, the subject's weight, no head cut, the
    subject near the middle of its region."""
    wa = max(1e-6, (win[2] - win[0]) * (win[3] - win[1]))
    sc = 1.0 * min(1.0, wa)  # less zoom
    sa = max(1e-6, (shown[2] - shown[0]) * (shown[3] - shown[1]))
    sc += 0.6 * _inter(win, shown) / min(wa, sa)
    if v is None:
        return sc
    total = v.mass((0.0, 0.0, 1.0, 1.0)) + 1e-6
    clip = (max(0.0, win[0]), max(0.0, win[1]), min(1.0, win[2]), min(1.0, win[3]))
    if clip[2] > clip[0] and clip[3] > clip[1]:
        sc += 0.8 * v.mass(clip) * (clip[2] - clip[0]) * (clip[3] - clip[1]) / total
    for hd in v.heads:  # never cut a head
        ha = max(1e-6, (hd[2] - hd[0]) * (hd[3] - hd[1]))
        cut = 1.0 - _inter(win, hd) / ha
        if cut > 1e-3:
            sc -= 4.0 + 20.0 * cut
    if v.subject:
        fx, fy = v.focal
        cx, cy = (win[0] + win[2]) / 2.0, (win[1] + win[3]) / 2.0
        sc -= 0.8 * (abs(fx - cx) / max(1e-6, win[2] - win[0]) + abs(fy - cy) / max(1e-6, win[3] - win[1]))
    return sc


# ----------------------------------------------------------------------------
# finalize: the one bleed this layout makes


def legal_bleed(e: dict, se: dict, W: float, H: float, W0: float, H0: float) -> bool:
    """A photo region's box may run past the canvas edges: an opaque picture
    that took at least a tenth of its source canvas, drawn with fit cover,
    unrotated, at least 60% of its box on the canvas (tools/bleed.py's rule
    for photos)."""
    try:
        return _legal_bleed(e, se, W, H, W0, H0)
    except Exception:  # noqa: BLE001  finalize never raises: when in doubt, pull it inside
        return False


def _legal_bleed(e: dict, se: dict, W: float, H: float, W0: float, H0: float) -> bool:
    if e.get("type") != "image" or e.get("fit") != "cover" or abs(float(e.get("rotation", 0) or 0)) > 1e-6:
        return False
    if not solid(e.get("asset_id")):
        return False
    try:
        sw, sh = float(se.get("width", 0) or 0), float(se.get("height", 0) or 0)
        x, y, w, h = (float(e[k]) for k in ("x", "y", "width", "height"))
    except (TypeError, ValueError, KeyError):
        return False
    if sw * sh < 0.1 * W0 * H0 or w <= 0 or h <= 0 or w * h > 64.0 * W * H:
        return False
    on = max(0.0, min(x + w, W) - max(x, 0.0)) * max(0.0, min(y + h, H) - max(y, 0.0)) / (w * h)
    return on >= 0.6


# ----------------------------------------------------------------------------
# the layout


def plan_regions(d, ctx, hidden, name: str = "regions", insets=(0, 0, 0, 0), axis: str = "x"):
    """The photo region and the panel region tiling the canvas along ``axis``;
    the panel's copy laid out inside the panel as one column. None if the
    design has no such split, or the copy does not fit."""
    from dataclasses import replace

    from .compose import (BALANCE, Plan, _settle_floaters, _visible, collides, column, legible_cta, margins,
                          scale_bounds, solve)
    from .items import Real
    from .structure import dominant_align, rows_for

    sp = getattr(d, "split", None)
    if sp is None:
        return None
    vis = _visible(d, hidden)
    if not any(it is sp.photo for it in vis):
        return None
    mine = {id(m) for m in sp.panel.members}
    body = [it for it in vis if id(it) in mine and not it.corner]
    floaters = [it for it in vis if it.corner and it is not sp.photo]
    if not body or any(it is not sp.photo and id(it) not in mine and not it.corner for it in vis):
        return None
    W, H = ctx.W, ctx.H
    P, Q = regions(sp, W, H, axis)
    box = photo_box(sp.photo.els[0], P, Q, W, H, d.W, d.H, sp.hides)
    if box is None:
        return None

    # the copy column, padded inside the panel as the source padded it
    sb = el_box(sp.panel.el)
    pad = [p for p in sp.panel.pad if p > 0.5]
    rel = (min(pad) / max(1.0, min(sb.w, sb.h))) if pad else 0.08
    mx, my = margins(d, ctx)
    m = max(min(mx, my), min(0.14, max(0.06, rel)) * min(Q.w, Q.h))
    il, it_, ir, ib = insets
    zone = Box(Q.x + m + il, Q.y + m + it_, Q.w - 2 * m - il - ir, Q.h - 2 * m - it_ - ib)
    if zone.w < 20 or zone.h < 20:
        return None
    dom = dominant_align(body)
    align = {"left": "start", "center": "center", "right": "end"}.get(dom, "start")
    rows = rows_for(body)
    root, _ = column(replace(d, backdrops=[]), rows, force_align=align)
    lo, hi = scale_bounds(d, ctx, "split" if axis == "x" else "stack")
    s, laid, relax = solve(root, ctx, zone, lo, hi)
    if laid is None:
        return None
    s, laid, relax = legible_cta(root, d, ctx, zone, lo, hi, (s, laid, relax))
    z = laid.zone  # type: ignore[attr-defined]
    parts, gaps = laid.parts, laid.gap_list  # type: ignore[attr-defined]
    u = None
    for it in body:
        u = it.src_box if u is None else u.union(it.src_box)
    top0, bot0 = max(0.0, u.y - sb.y), max(0.0, sb.b - u.b)
    mid = (top0 + bot0) / 2.0
    top0, bot0 = top0 + BALANCE * (mid - top0), bot0 + BALANCE * (mid - bot0)
    slack = max(0.0, z.h - sum(p.h for p in parts) - sum(gaps))
    y = z.y + (slack * top0 / (top0 + bot0) if top0 + bot0 > 0 else slack / 2.0)
    bw = max(p.w for p in parts)
    bx = z.x if align == "start" else (z.r - bw if align == "end" else z.cx - bw / 2.0)
    placed = []
    for i, p in enumerate(parts):
        a = root.align[i] if i < len(root.align) else align
        x = bx if a == "start" else (bx + bw - p.w if a == "end" else bx + (bw - p.w) / 2.0)
        placed += p.place(x, y)
        y += p.h + (gaps[i] if i < len(gaps) else 0.0)

    e0 = sp.photo.els[0]
    bx0, by0, bw0, bh0 = box

    def emit(x: float, y: float) -> list:
        e = {k: v for k, v in e0.items() if k != "_idx"}
        e.update(x=bx0 + x - P.x, y=by0 + y - P.y, width=bw0, height=bh0, fit="cover")
        e["_region_bleed"] = True  # finalize leaves it past the canvas edge (legal_bleed, or a cover crop)
        return [e]

    src = sp.photo.src_box
    photo = Real(item=sp.photo, w=P.w, h=P.h, scale=max(P.w / max(1.0, src.w), P.h / max(1.0, src.h)),
                 emit_fn=emit)
    photo._oh = (0.0, 0.0, 0.0, 0.0)  # its footprint is its region: what runs past is cropped or under the panel
    fl = []
    if floaters:  # corner items keep their corners, clear of the copy; none is left out (S1)
        fl, bad, _, _ = _settle_floaters(placed, floaters, s, ctx, Box(0, 0, W, H), mx, my, relax)
        gap = 0.6 * ctx.base_fs
        if bad or any(collides(Box(fx, fy, fr.w, fr.h), Box(x, y, r.w, r.h), gap)
                      for fr, fx, fy in fl for r, x, y in placed):
            return None
    return Plan(name=name, placed=[(photo, P.x, P.y)] + placed, hidden=list(hidden), s=s, zone=zone,
                floaters=fl, relax=relax, capped=s >= hi * 0.999,
                fixed={sp.panel.el["id"]: (Q.x, Q.y, Q.w, Q.h)})
