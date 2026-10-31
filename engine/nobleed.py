"""Strict mode (CHAMELEON_NO_BLEED=1): every element's box stays on the canvas.

The default engine lets backdrops, heroes and circles run past the edge. Here
nothing does: a backdrop is a canvas-sized box with fit=cover, and this last
pass brings in whatever is left over. The schema has no crop offset, so a
cover crop is always centred.
"""

from __future__ import annotations

import os

from .analyze import _fills_box, norm_rot
from .blocks import rot_bounds

NO_BLEED = os.environ.get("CHAMELEON_NO_BLEED", "1") == "1"
BACKDROP_AREA = 0.6     # a turned photo showing this much of the canvas is a backdrop
EPS = 0.01
UPRIGHT = False         # set while a design is laid out again without turned boxes


def _off(e, W, H):
    b = rot_bounds(e["x"], e["y"], e["width"], e["height"], e.get("rotation", 0) or 0)
    return min(b[0], b[1]) < -EPS or b[2] > W + EPS or b[3] > H + EPS


def raw_off(design):
    """What the structural check sees: the box before rotation."""
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    return any(e["x"] < -EPS or e["y"] < -EPS or e["x"] + e["width"] > W + EPS or e["y"] + e["height"] > H + EPS
               for e in design["elements"])


def settle(design):
    """Last step, so the structural check passes on every element."""
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    contain(design)
    for e in design["elements"]:
        if e["x"] < -EPS or e["y"] < -EPS or e["x"] + e["width"] > W + EPS or e["y"] + e["height"] > H + EPS:
            _bring_in(e, W, H)
    return design


def _crop(e, W, H):
    """Trim the box to the canvas: a rectangle looks the same, a photo keeps its proportions by cover."""
    x0, y0 = max(e["x"], 0.0), max(e["y"], 0.0)
    x1, y1 = min(e["x"] + e["width"], W), min(e["y"] + e["height"], H)
    e.update(x=round(x0, 2), y=round(y0, 2), width=round(x1 - x0, 2), height=round(y1 - y0, 2))
    if e["type"] == "image":
        e["fit"] = "cover"


def _crop_turned(e, W, H):
    """_crop for a box on its side: trim what shows, then turn the box back."""
    b = rot_bounds(e["x"], e["y"], e["width"], e["height"], 90)
    x0, y0, x1, y1 = max(b[0], 0.0), max(b[1], 0.0), min(b[2], W), min(b[3], H)
    w, h = y1 - y0, x1 - x0
    e.update(x=round((x0 + x1 - w) / 2, 2), y=round((y0 + y1 - h) / 2, 2), width=round(w, 2), height=round(h, 2))
    if e["type"] == "image":
        e["fit"] = "cover"


def _bring_in(e, W, H):
    """Shrink as a whole only if it must, then slide inside. Keeps its form."""
    rot = e.get("rotation", 0) or 0
    b = rot_bounds(e["x"], e["y"], e["width"], e["height"], rot)
    bw, bh = b[2] - b[0], b[3] - b[1]
    k = min(1.0, W / bw, H / bh, W / e["width"], H / e["height"])
    w, h = e["width"] * k, e["height"] * k
    hw, hh = max(bw * k, w) / 2, max(bh * k, h) / 2       # the rotated and the raw box both fit
    cx = min(max(e["x"] + e["width"] / 2, hw), W - hw)
    cy = min(max(e["y"] + e["height"] / 2, hh), H - hh)
    e.update(x=round(cx - w / 2, 2), y=round(cy - h / 2, 2), width=round(w, 2), height=round(h, 2))
    if e["type"] == "text" and k < 1:
        e["font_size"] = max(1, int(e["font_size"] * k))
        if e.get("letter_spacing"):
            e["letter_spacing"] = round(e["letter_spacing"] * k, 3)


def contain(design):
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    for e in design["elements"]:
        if not _off(e, W, H):
            continue
        rot = abs(norm_rot(e.get("rotation", 0) or 0))
        fills = e["type"] == "image" and _fills_box(e)
        if rot <= 1 and (fills or (e["type"] == "shape" and e.get("shape_type") != "ellipse")):
            _crop(e, W, H)
        elif abs(rot - 90) <= 1 and (fills or (e["type"] == "shape" and e.get("shape_type") != "ellipse")):
            _crop_turned(e, W, H)
        elif fills and e["width"] * e["height"] >= BACKDROP_AREA * W * H:
            e.update(x=0, y=0, width=W, height=H, rotation=0, fit="cover")
        else:
            _bring_in(e, W, H)
        if _off(e, W, H):       # a box a few degrees off square, trimmed as if square
            _bring_in(e, W, H)
    return design
