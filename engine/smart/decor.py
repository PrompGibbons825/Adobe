"""Decorations: what they are and where they go on the new canvas.

After Canva's US12307078B2 ("identifying and transferring decorative design
elements") and the print-bleed filing US20230244421A1:

* **What.** A decoration is art that carries no copy: a translucent shape or
  ornament, or a shape/sparse image that touches or runs past a canvas edge.
  ``classify`` moves such images and edge-touching shapes out of the layout
  (where they would be stacked like content) into ``Design.decor``.
* **Anchor.** Each decoration is pinned to where it sat: per axis the edge it
  touches (or the nearer third), giving one of the four corners, the four
  edge centres or the centre. Its **coverage class** says whether it spans the
  width, the height, both or neither.
* **Transfer.** It is re-placed at the same anchor on the target, scaled by the
  min factor ``min(W/W0, H/H0)``. A spanning shape is stretched along the
  axis it spans (shapes may stretch, images never do), keeping its thickness
  at the min factor. Edge contact survives the resize.
* **Bleed.** A decoration that ran off the source canvas (the Level 2 circle)
  can keep its bleed fraction at the same anchor, but RULES.md's structural
  check says nothing may be off-canvas, so this is behind ``DECOR_BLEED``,
  **off by default** until the organisers answer (info session 2026-10-09).
  Off, the decoration is shrunk to its visible area and tucked into the same
  corner or edge, touching it.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

from . import assets
from .geom import Box, el_box, el_visual_box

DECOR_ANCHORS = os.environ.get("CHAMELEON_DECOR_ANCHORS", "1") != "0"  # anchor + coverage placement
DECOR_IMAGES = os.environ.get("CHAMELEON_DECOR_IMAGES", "1") != "0"  # sparse/edge images count as decoration
DECOR_BLEED = os.environ.get("CHAMELEON_DECOR_BLEED", "0") == "1"  # keep off-canvas bleed (default OFF)


# ----------------------------------------------------------------------------
# classification


def _ink_box(e: dict) -> Box:
    """What is actually painted: the visible pixels of an image, the rotated
    footprint of anything rotated."""
    b = el_box(e)
    if e.get("type") == "image" and not float(e.get("rotation", 0) or 0):
        fx0, fy0, fx1, fy1 = assets.ink_fraction(e.get("asset_id"), b.w, b.h, e.get("fit", "contain"))
        return Box(b.x + fx0 * b.w, b.y + fy0 * b.h, (fx1 - fx0) * b.w, (fy1 - fy0) * b.h)
    return el_visual_box(e)


def _edges(v: Box, W: float, H: float, tol: float) -> Tuple[bool, bool, bool, bool]:
    return v.x <= tol, v.y <= tol, v.r >= W - tol, v.b >= H - tol


def _encloses_text(e: dict, d) -> bool:
    from .analyze import text_tight_box
    from .textfit import TextStyle

    eb = _ink_box(e)
    for t in d.elements:
        if t.get("type") != "text" or not str(t.get("content", "")).strip():
            continue
        tb, _ = text_tight_box(t, TextStyle.of(t))
        if tb.area > 0 and tb.inter(eb) >= 0.2 * tb.area:
            return True
    return False


def classify(d) -> None:
    """Move decorative items out of the layout. Runs at the end of analyze().

    * a faint shape (alpha < 0.3) that was taken for a panel behind copy;
    * a standalone shape (no copy on it) that touches a canvas edge;
    * an image that is not the hero, the logo or a missing asset, carries no
      copy, and is faint (opacity < 0.6) or sparse line art (coverage < 0.3)
      running past an edge."""
    if not DECOR_ANCHORS:
        return
    from .analyze import _fill_alpha

    for bd in list(d.backdrops):  # a faint tint behind a chip is ornament, not the panel the copy needs
        sh = bd.el
        if float(sh.get("opacity", 1.0)) * _fill_alpha(sh) < 0.3:
            d.backdrops.remove(bd)
            sh["_decor_kind"] = "faint"
            d.decor.append(sh)
    if not DECOR_IMAGES:
        return
    canvas = Box(0, 0, d.W, d.H)
    tol = 0.005 * min(d.W, d.H) + 1.0
    keep = []
    for it in d.items:
        if it.kind != "image" or len(it.els) != 1 or it.role in ("hero", "logo", "missing", "void"):
            keep.append(it)
            continue
        e = it.els[0]
        v = _ink_box(e)
        if v.area <= 0 or v.inter(canvas) <= 0:
            keep.append(it)
            continue
        touch = any(_edges(v, d.W, d.H, tol))
        if e.get("type") == "shape":
            deco = touch
        else:
            a = assets.info(e.get("asset_id"))
            faint = float(e.get("opacity", 1.0)) < 0.6
            # sparse line art running off the canvas (a wordmark that only touches an edge stays content)
            bleeds = v.inter(canvas) < 0.97 * v.area
            sparse = a.found and a.coverage < 0.3 and bleeds
            deco = faint or sparse
        if deco and not _encloses_text(e, d):
            e["_decor_kind"] = "edge" if touch else "faint"
            d.decor.append(e)
        else:
            keep.append(it)
    d.items = keep


# ----------------------------------------------------------------------------
# anchors


def anchor(v: Box, W: float, H: float, tol: Optional[float] = None) -> Tuple[str, str, Tuple[bool, ...]]:
    """(horizontal, vertical, edge contacts) for a painted box on a W x H canvas.
    Horizontal is l / c / r / span, vertical t / c / b / span. A box that
    touches an edge is anchored to it; a free one to the nearer third."""
    if tol is None:
        tol = 0.005 * min(W, H) + 1.0
    vis = v.inter_box(Box(0, 0, W, H)) or v
    tl, tt, tr, tb = _edges(vis, W, H, tol)
    if tl and tr:
        hx = "span"
    elif tl or tr:
        hx = "l" if tl else "r"
    else:
        hx = "l" if vis.cx < W / 3.0 else ("r" if vis.cx > 2.0 * W / 3.0 else "c")
    if tt and tb:
        hy = "span"
    elif tt or tb:
        hy = "t" if tt else "b"
    else:
        hy = "t" if vis.cy < H / 3.0 else ("b" if vis.cy > 2.0 * H / 3.0 else "c")
    return hx, hy, (tl, tt, tr, tb)


def coverage_class(hx: str, hy: str) -> str:
    if hx == "span" and hy == "span":
        return "both"
    if hx == "span":
        return "horizontal"
    if hy == "span":
        return "vertical"
    return "neither"


def _axis(a: str, touch0: bool, touch1: bool, v0: float, v1: float, vis0: float, vis1: float, S0: float,
          S: float, size: float, s: float, bleed: bool) -> float:
    """Start coordinate, along one axis, of a painted extent of ``size``."""
    if a == "span":
        return v0 / S0 * S if bleed else 0.0
    if a in ("l", "t"):
        if touch0:  # touching (or bled past) the start edge: stays on it
            return v0 / max(1e-6, v1 - v0) * size if (bleed and v0 < 0) else 0.0
        return vis0 * s  # free: keep the gap to the edge, scaled
    if a in ("r", "b"):
        if touch1:
            over = (v1 - S0) / max(1e-6, v1 - v0) * size if (bleed and v1 > S0) else 0.0
            return S - size + over
        return S - size - (S0 - vis1) * s
    return (vis0 + vis1) / 2.0 / S0 * S - size / 2.0  # centre: keep the relative position


def place(e: dict, d, ctx, bleed: Optional[bool] = None) -> Optional[dict]:
    """The decoration's element on the target canvas, or None to leave it to
    the old placement (unknown types, empty or off-canvas boxes)."""
    if bleed is None:
        bleed = DECOR_BLEED
    t = e.get("type")
    if t not in ("shape", "image"):
        return None
    eb = el_box(e)
    v = _ink_box(e)
    if eb.w <= 0 or eb.h <= 0 or v.w <= 0 or v.h <= 0:
        return None
    W0, H0, W, H = d.W, d.H, ctx.W, ctx.H
    vis = v.inter_box(Box(0, 0, W0, H0))
    if vis is None:
        return None
    hx, hy, (tl, tt, tr, tb) = anchor(v, W0, H0)
    rot = float(e.get("rotation", 0) or 0)
    s = min(W / W0, H / H0)
    if t != "shape" or rot:  # images and rotated shapes keep their proportions: no spanning stretch
        hx = "c" if hx == "span" else hx
        hy = "c" if hy == "span" else hy
    # with bleed off, the whole decoration comes back on canvas at the painted area it showed
    kf = s if bleed else s * math.sqrt(vis.area / max(1e-6, v.area))
    if hx == "span":
        pw = v.w * W / W0 if bleed else W
    elif hy == "span":
        pw = (v.w if bleed else vis.w) * s  # a vertical band keeps its (visible) thickness
    else:
        pw = v.w * kf
    if hy == "span":
        ph = v.h * H / H0 if bleed else H
    elif hx == "span":
        ph = (v.h if bleed else vis.h) * s
    else:
        ph = v.h * kf
    if not bleed:
        pw, ph = min(pw, W), min(ph, H)
    px = _axis(hx, tl, tr, v.x, v.r, vis.x, vis.r, W0, W, pw, s, bleed)
    py = _axis(hy, tt, tb, v.y, v.b, vis.y, vis.b, H0, H, ph, s, bleed)
    o = {k: val for k, val in e.items() if k not in ("_idx", "_decor_kind")}
    if t == "shape" and not rot:  # element box == painted box
        o.update(x=px, y=py, width=pw, height=ph)
        kk = min(pw / v.w, ph / v.h)
    elif rot:  # rotated: same centre, element box scaled like its footprint
        kk = min(pw / v.w, ph / v.h)
        cx, cy = px + pw / 2.0, py + ph / 2.0
        o.update(x=cx - eb.w * kk / 2.0, y=cy - eb.h * kk / 2.0, width=eb.w * kk, height=eb.h * kk)
    else:  # an image: from its visible pixels back to its (padded) element box
        kk = min(pw / v.w, ph / v.h)
        ew, eh = eb.w * kk, eb.h * kk
        o.update(x=px - (v.x - eb.x) * kk, y=py - (v.y - eb.y) * kk, width=ew, height=eh)
    if o.get("corner_radius"):
        o["corner_radius"] = float(o["corner_radius"]) * kk
    if o.get("stroke_width"):
        o["stroke_width"] = float(o["stroke_width"]) * kk
    if bleed:
        pb = el_visual_box(o)
        if pb.x < -0.5 or pb.y < -0.5 or pb.r > W + 0.5 or pb.b > H + 0.5:
            o["_bleed"] = True  # finalize() leaves it hanging off the edge
    return o
