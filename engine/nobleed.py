"""Strict mode (CHAMELEON_NO_BLEED=1): every element's box stays on the canvas.

The default engine lets backdrops, heroes and circles run past the edge. Here
nothing does: a backdrop is a canvas-sized box with fit=cover, and this last
pass brings in whatever is left over. The schema has no crop offset, so a
cover crop is always centred.
"""

from __future__ import annotations

import os

from . import saliency, textfit as tf
from .analyze import _fills_box, norm_rot
from .blocks import rot_bounds

NO_BLEED = os.environ.get("CHAMELEON_NO_BLEED", "1") == "1"
BACKDROP_AREA = 0.6     # a turned photo showing this much of the canvas is a backdrop
EPS = 0.01


def _off(e, W, H):
    b = rot_bounds(e["x"], e["y"], e["width"], e["height"], e.get("rotation", 0) or 0)
    return min(b[0], b[1]) < -EPS or b[2] > W + EPS or b[3] > H + EPS


def settle(design):
    """Last step, so the structural check passes on every element."""
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    contain(design)
    for e in design["elements"]:
        if e["x"] < -EPS or e["y"] < -EPS or e["x"] + e["width"] > W + EPS or e["y"] + e["height"] > H + EPS:
            if abs(abs(norm_rot(e.get("rotation", 0) or 0)) - 90) <= 30 and e.get("shape_type") != "ellipse":
                _stand_up(e, design, W, H)
            else:
                _bring_in(e, W, H)
    return design


def _square_turn(e, W, H):
    """A cut-out on its side keeps its turn in a square box: a square's box is
    the same before and after rotation, and contain shows the whole picture."""
    info = saliency.analyze(e.get("asset_id") or "")
    side = min(W, H, max(e["width"], e["height"]))
    b = rot_bounds(e["x"], e["y"], e["width"], e["height"], e.get("rotation", 0) or 0)
    u = info["centroid"][0] if info else 0.5
    sign = -1 if norm_rot(e["rotation"]) > 0 else 1         # which way the picture's left end points
    cx = (b[0] + b[2]) / 2
    cy = (b[1] + b[3]) / 2 + sign * (0.5 - u) * e["width"]
    e.update(x=round(min(max(cx - side / 2, 0), W - side), 2), y=round(min(max(cy - side / 2, 0), H - side), 2),
             width=side, height=side, fit="contain")


def _free_spot(e, design, W, H, b):
    """Nearest clear place for text set horizontal, at its own size."""
    others = [o for o in design["elements"] if o is not e and o["width"] * o["height"] < 0.5 * W * H
              and o["width"] > 2 and o["height"] > 2]
    boxes = [rot_bounds(o["x"], o["y"], o["width"], o["height"], o.get("rotation", 0) or 0) for o in others]
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    pad = e["font_size"] * 0.3
    best = None
    fs0 = e["font_size"]
    for fs in (fs0, int(fs0 * 0.85), int(fs0 * 0.7)):
        m1 = tf.measure(e, fs, 10 * W)
        for w in (m1["width"] + 2, m1["width"] / 2 + fs, m1["width"] / 3 + fs):
            w = min(w, W)
            m = tf.measure(e, fs, w)
            h = m["height"] + 2
            if m["word_w"] > w or h > H:
                continue
            for i in range(25):
                for j in range(25):
                    x, y = (W - w) * i / 24, (H - h) * j / 24
                    if any(x - pad < q[2] and x + w + pad > q[0] and y - pad < q[3] and y + h + pad > q[1] for q in boxes):
                        continue
                    d = abs(x + w / 2 - cx) + abs(y + h / 2 - cy) + (fs0 - fs) * 40
                    if best is None or d < best[0]:
                        best = (d, x, y, w, h, fs)
    return best


def _stand_up(e, design, W, H):
    """A box on its side overruns the canvas before rotation, which the
    structural check counts. It stands upright: text moves to free space, a
    photo takes the space it had on screen."""
    b = rot_bounds(e["x"], e["y"], e["width"], e["height"], e.get("rotation", 0) or 0)
    info = saliency.analyze(e.get("asset_id") or "") if e["type"] == "image" else None
    # the square only for a subject the centre crop would lose; on a strip it would be a speck
    if info and min(W, H) >= 0.4 * max(W, H) and abs(info["centroid"][0] - 0.5) > 0.15:
        _square_turn(e, W, H)
        return
    x0, y0, x1, y1 = max(b[0], 0.0), max(b[1], 0.0), min(b[2], W), min(b[3], H)
    e.update(x=round(x0, 2), y=round(y0, 2), width=round(x1 - x0, 2), height=round(y1 - y0, 2), rotation=0)
    if e["type"] == "image":
        e["fit"] = "cover"
    elif e["type"] == "text":
        fs0 = e["font_size"]
        spot = _free_spot(e, design, W, H, b)
        if spot:
            _, x, y, w, h, fs = spot
            e.update(x=round(x, 2), y=round(y, 2), width=round(w, 2), height=round(h, 2))
        else:
            fs = fs0
            while fs > 1:
                m = tf.measure(e, fs, e["width"])
                if m["height"] <= e["height"] and m["word_w"] <= e["width"]:
                    break
                fs -= 1
        e["font_size"] = fs
        if e.get("letter_spacing"):
            e["letter_spacing"] = round(e["letter_spacing"] * fs / fs0, 3)


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
