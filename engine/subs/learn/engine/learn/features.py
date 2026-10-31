"""Per-element features of a design, shared by training (Crello families) and run time.

A design here is the Chameleon shape: {"canvas": {width, height}, "elements": [...]}.
Nothing in here knows about roles: every number is geometry, type, z order or type size.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

FEATURES = [
    "is_text",      # 0
    "is_bleed",     # 1  covers (nearly) the whole canvas
    "cx", "cy",     # 2,3 centre in canvas fractions
    "w", "h",       # 4,5 size in canvas fractions
    "log_ar",       # 6  log aspect of the element box
    "sqrt_area",    # 7  sqrt of the canvas share
    "z",            # 8  z rank in [0,1]
    "fs",           # 9  font size / sqrt(canvas area)  (0 for non-text)
    "fs_rank",      # 10 1 = largest type in the design, 0 = smallest; -0.5 for non-text
    "chars",        # 11 log(1+len(content)) / 5
    "src_ar",       # 12 log source canvas aspect
    "tgt_ar",       # 13 log target canvas aspect
    "d_ar",         # 14 tgt_ar - src_ar
    "n_el",         # 15 log(1+n elements) / 3
    "edge",         # 16 distance to the nearest canvas edge, canvas fractions
    "n_text",       # 17 share of text elements in the design
]
NF = len(FEATURES)


def num(v: Any, d: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return d
    return f if math.isfinite(f) else d


def is_bleed(e: Dict[str, Any], W: float, H: float) -> bool:
    x, y, w, h = num(e.get("x")), num(e.get("y")), num(e.get("width")), num(e.get("height"))
    if e.get("type") == "text":
        return False
    ix = max(0.0, min(x + w, W) - max(x, 0.0))
    iy = max(0.0, min(y + h, H) - max(y, 0.0))
    return ix >= 0.95 * W and iy >= 0.95 * H


def element_features(design: Dict[str, Any], target: Dict[str, Any]) -> np.ndarray:
    W = max(1.0, num(design["canvas"].get("width"), 1.0))
    H = max(1.0, num(design["canvas"].get("height"), 1.0))
    TW = max(1.0, num(target.get("width"), 1.0))
    TH = max(1.0, num(target.get("height"), 1.0))
    els: List[Dict[str, Any]] = design["elements"]
    n = len(els)
    X = np.zeros((n, NF), dtype=np.float32)
    zs = [num(e.get("z_index"), i) for i, e in enumerate(els)]
    zorder = sorted(range(n), key=lambda i: (zs[i], i))
    zrank = {j: (r / max(1, n - 1)) for r, j in enumerate(zorder)}
    texts = [i for i, e in enumerate(els) if e.get("type") == "text"]
    fss = sorted({num(els[i].get("font_size"), 12.0) for i in texts})
    src_ar, tgt_ar = math.log(W / H), math.log(TW / TH)
    for i, e in enumerate(els):
        x, y = num(e.get("x")), num(e.get("y"))
        w, h = max(1e-3, num(e.get("width"), 1.0)), max(1e-3, num(e.get("height"), 1.0))
        t = e.get("type") == "text"
        X[i, 0] = 1.0 if t else 0.0
        X[i, 1] = 1.0 if is_bleed(e, W, H) else 0.0
        X[i, 2] = (x + w / 2) / W
        X[i, 3] = (y + h / 2) / H
        X[i, 4] = w / W
        X[i, 5] = h / H
        X[i, 6] = math.log(w / h)
        X[i, 7] = math.sqrt(max(0.0, (w * h) / (W * H)))
        X[i, 8] = zrank[i]
        if t:
            fs = num(e.get("font_size"), 12.0)
            X[i, 9] = fs / math.sqrt(W * H)
            X[i, 10] = (fss.index(fs) / (len(fss) - 1)) if len(fss) > 1 else 1.0
            X[i, 11] = math.log1p(len(str(e.get("content", "")))) / 5.0
        else:
            X[i, 10] = -0.5
        X[i, 12] = src_ar
        X[i, 13] = tgt_ar
        X[i, 14] = tgt_ar - src_ar
        X[i, 15] = math.log1p(n) / 3.0
        X[i, 16] = max(0.0, min(x / W, y / H, 1 - (x + w) / W, 1 - (y + h) / H))
        X[i, 17] = len(texts) / max(1, n)
    return X


# Targets learned per element (what the designer did), in a frame that is size-free:
TARGETS = ["tcx", "tcy", "log_scale", "d_log_ar", "log_fs", "t_bleed"]


def element_targets(src: Dict[str, Any], dst: Dict[str, Any], pairs: List[tuple]) -> np.ndarray:
    """pairs: (i_src, j_dst). Row k describes how the designer moved src element i."""
    W, H = num(src["canvas"]["width"]), num(src["canvas"]["height"])
    TW, TH = num(dst["canvas"]["width"]), num(dst["canvas"]["height"])
    s_geo = math.sqrt((TW * TH) / (W * H))
    Y = np.zeros((len(pairs), len(TARGETS)), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        a, b = src["elements"][i], dst["elements"][j]
        aw, ah = max(1e-3, num(a["width"])), max(1e-3, num(a["height"]))
        bw, bh = max(1e-3, num(b["width"])), max(1e-3, num(b["height"]))
        Y[k, 0] = (num(b["x"]) + bw / 2) / TW
        Y[k, 1] = (num(b["y"]) + bh / 2) / TH
        Y[k, 2] = math.log(math.sqrt(bw * bh) / math.sqrt(aw * ah)) - math.log(s_geo)
        Y[k, 3] = math.log(bw / bh) - math.log(aw / ah)
        if a.get("type") == "text":
            Y[k, 4] = math.log(max(0.5, num(b.get("font_size"), 12)) / max(0.5, num(a.get("font_size"), 12))) - math.log(s_geo)
        Y[k, 5] = 1.0 if is_bleed(b, TW, TH) else 0.0
    return Y
