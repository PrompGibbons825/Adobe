"""Nearest designer moves: for each query element, the k most similar (element, target-aspect)
rows in the Crello training table, and what their designers did with them.

The feature weights and k are fitted on held-out families (tools/learn_fit.py), not set by hand.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .features import NF

MODEL = Path(__file__).with_name("model.npz")
_cache: Dict[str, np.ndarray] = {}


def load(path: Optional[Path] = None):
    p = str(path or MODEL)
    if p not in _cache:
        z = np.load(p)
        _cache[p] = {k: (z[k].astype(np.float32) if z[k].dtype == np.float16 else z[k]) for k in z.files}
    return _cache[p]


def predict(Xq: np.ndarray, Xt: np.ndarray, Yt: np.ndarray, w: np.ndarray, k: int,
            chunk: int = 2048) -> np.ndarray:
    """Distance-weighted mean of the k nearest rows' targets. Text rows only match text rows."""
    out = np.zeros((len(Xq), Yt.shape[1]), dtype=np.float32)
    sw = np.sqrt(np.asarray(w, dtype=np.float32))
    Tt = Xt * sw
    tn = (Tt ** 2).sum(1)
    txt_t = Xt[:, 0] > 0.5
    for s in range(0, len(Xq), chunk):
        Q = Xq[s:s + chunk] * sw
        d2 = (Q ** 2).sum(1)[:, None] + tn[None, :] - 2.0 * Q @ Tt.T
        txt_q = Xq[s:s + chunk, 0] > 0.5
        d2[txt_q[:, None] != txt_t[None, :]] = np.inf
        kk = min(k, Xt.shape[0])
        idx = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        dd = np.take_along_axis(d2, idx, 1)
        wt = 1.0 / (np.sqrt(np.maximum(dd, 0)) + 0.05)
        wt /= wt.sum(1, keepdims=True)
        out[s:s + chunk] = np.einsum("qk,qkt->qt", wt, Yt[idx])
    return out


def predict_design(Xq: np.ndarray) -> np.ndarray:
    m = load()
    return predict(Xq, m["X"], m["Y"], m["w"], int(m["k"]))


def boxes_from_pred(Xq: np.ndarray, P: np.ndarray, src_px: np.ndarray, S, T, keep_aspect: np.ndarray) -> np.ndarray:
    """Turn predictions into target boxes (x, y, w, h) in target pixels.
    src_px: (n,4) source boxes in pixels. S, T: (W,H) of source and target."""
    W, H = S
    TW, TH = T
    s_geo = math.sqrt((TW * TH) / (W * H))
    out = np.zeros((len(Xq), 4), dtype=np.float64)
    for i in range(len(Xq)):
        if P[i, 5] > 0.5:
            out[i] = (0, 0, TW, TH)
            continue
        w0, h0 = max(1e-3, src_px[i, 2]), max(1e-3, src_px[i, 3])
        side = math.sqrt(w0 * h0) * s_geo * math.exp(float(P[i, 2]))
        ar = w0 / h0 * (1.0 if keep_aspect[i] else math.exp(float(P[i, 3])))
        w, h = side * math.sqrt(ar), side / math.sqrt(ar)
        cx, cy = P[i, 0] * TW, P[i, 1] * TH
        out[i] = (cx - w / 2, cy - h / 2, w, h)
    return out
