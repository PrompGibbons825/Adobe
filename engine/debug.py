"""Diagnostics helpers (used by tools/, not by resize()).

source_placed(model) builds the same view the scorer has of a candidate,
but for the untouched source design: visual boxes, the photo subject, and
the visual-weight items, so source and result can be compared like for like.
"""

from __future__ import annotations

from . import saliency
from . import textfit as tf
from .blocks import Placed, img_vis_frac, rot_bounds
from .score import weight_items


def source_placed(model):
    W, H = model["canvas"]
    P = Placed()
    P.bg_focal, P.bg_info = None, None
    units = [u for u in model["units"] if u.role not in ("bg", "deco", "border", "rule")]
    for u in units:
        for eid in u.ids:
            e = model["els"][eid]
            if u.role == "missing":
                continue
            if e["type"] == "text":
                fs = max(1, int(e["font_size"]))
                m = tf.measure(e, fs, e["width"])
                b = tf.visual_box(e, e["x"], e["y"], e["width"], fs, m)
                P.vis[eid] = rot_bounds(b[0], b[1], b[2] - b[0], b[3] - b[1], e.get("rotation", 0))
            elif e["type"] == "image":
                v = img_vis_frac(e)
                P.vis[eid] = rot_bounds(e["x"] + v[0] * e["width"], e["y"] + v[1] * e["height"],
                                        (v[2] - v[0]) * e["width"], (v[3] - v[1]) * e["height"], e.get("rotation", 0))
            else:
                P.vis[eid] = rot_bounds(e["x"], e["y"], e["width"], e["height"], e.get("rotation", 0))
    def _detail(u):
        info = saliency.analyze(u.els[0].get("asset_id") or "") if u.els[0]["type"] == "image" else None
        return info["busyness"] if info else 0.0
    for u in sorted((u for u in model["units"] if u.role == "bg"), key=_detail):
        e = u.els[0]
        info = saliency.analyze(e.get("asset_id") or "") if e["type"] == "image" else None
        if info is None:
            continue
        bx, by, bw, bh = e["x"], e["y"], e["width"], e["height"]
        sb = info["sal_box"]
        fx, fy = info["centroid"]
        conc = (sb[2] - sb[0]) * (sb[3] - sb[1]) < 0.45
        P.bg_info = {"id": e["id"], "asset": e.get("asset_id"), "box": (bx, by, bw, bh), "concentrated": conc,
                     "focal": (bx + fx * bw, by + fy * bh), "focal_centered": (bx + fx * bw, by + fy * bh),
                     "sal_box": (bx + sb[0] * bw, by + sb[1] * bh, bx + sb[2] * bw, by + sb[3] * bh)}
        if conc:
            sa = ((sb[2] - sb[0]) * bw * (sb[3] - sb[1]) * bh) / (W * H)
            strength = float(info["sal"].max() - info["sal"].mean())
            P.bg_focal = (bx + fx * bw, by + fy * bh, min(1.0, sa) * (0.5 + strength))
    ids = list(P.vis)
    P.weight_items, P.cm = weight_items(P, model, W, H, ids, units)
    return P
