"""Whole-layout transfer: find the Crello resize pairs whose source composition best matches this
design at a similar aspect change, match elements one to one, and take each matched element's
designer move. Positions come from one designer's layout, so they are coherent with each other.

Elements left unmatched take the per-element kNN move.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from . import knn

# features used to match an element of the query to an element of a candidate source
MATCH_F = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_idx: Dict[str, object] = {}


def _index():
    if "pairs" not in _idx:
        m = knn.load()
        pair = m["pair"].astype(np.int64)
        order = np.argsort(pair, kind="stable")
        pair_s = pair[order]
        starts = np.searchsorted(pair_s, np.arange(pair_s.max() + 2))
        X = m["X"]
        first = order[starts[:-1][starts[:-1] < len(order)]]
        _idx.update(order=order, starts=starts, src_ar=X[first, 12], tgt_ar=X[first, 13],
                    n_el=X[first, 15], n_text=X[first, 17], npairs=len(first))
        _idx["pairs"] = True
    return _idx


def _greedy(D: np.ndarray):
    D = D.copy()
    out = []
    n = min(D.shape)
    for _ in range(n):
        k = int(np.argmin(D))
        i, j = divmod(k, D.shape[1])
        if not np.isfinite(D[i, j]):
            break
        out.append((i, j, float(D[i, j])))
        D[i, :] = np.inf
        D[:, j] = np.inf
    return out


def candidates(Xq: np.ndarray, n_short: int = 150, top: int = 8) -> list:
    """[(cost, rows_of_candidate, matches[(q_i, cand_row)])] best first."""
    m = knn.load()
    ix = _index()
    w = np.asarray(m["w"], np.float32)
    sw = np.sqrt(np.maximum(w[MATCH_F], 1e-6) * (w[MATCH_F] > 0) + 1e-3)
    sa, ta = float(Xq[0, 12]), float(Xq[0, 13])
    da = (ix["tgt_ar"] - ta) ** 2 * 4.0 + (ix["src_ar"] - sa) ** 2 + (ix["n_el"] - Xq[0, 15]) ** 2 * 0.5
    short = np.argsort(da)[:n_short]
    Q = Xq[:, MATCH_F] * sw
    qt = Xq[:, 0] > 0.5
    out = []
    for p in short:
        rows = ix["order"][ix["starts"][p]:ix["starts"][p + 1]]
        if len(rows) == 0:
            continue
        C = m["X"][rows][:, MATCH_F] * sw
        ct = m["X"][rows, 0] > 0.5
        D = np.sqrt(((Q[:, None, :] - C[None, :, :]) ** 2).sum(-1))
        D[qt[:, None] != ct[None, :]] = np.inf
        mt = _greedy(D)
        if not mt:
            continue
        unmatched = len(Xq) - len(mt)
        cost = (sum(c for _, _, c in mt) + 1.5 * unmatched + 0.5 * (len(rows) - len(mt))) / len(Xq) + float(da[p])
        out.append((cost, rows, [(i, int(rows[j])) for i, j, _ in mt]))
    out.sort(key=lambda t: t[0])
    return out[:top]


def predict(Xq: np.ndarray, beta: float = 0.5, P0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """P in absolute form (tcx, tcy, log_scale, d_log_ar, log_fs, bleed) for every query element."""
    m = knn.load()
    cands = candidates(Xq)
    matched = {i for i, _ in cands[0][2]} if cands else set()
    rest = [i for i in range(len(Xq)) if i not in matched]
    P = np.zeros((len(Xq), 6), np.float32)
    if rest:
        P0 = knn.predict(Xq[rest], m["X"], m["Y"], m["w"], int(m["k"]))
        P0[:, 0] += Xq[rest, 2]
        P0[:, 1] += Xq[rest, 3]
        P[rest] = P0
    if not cands:
        return P, float("inf")
    cost, rows, mt = cands[0]
    for i, r in mt:
        y = m["Y"][r]
        xc = m["X"][r]
        P[i, 0] = y[0] + xc[2]
        P[i, 1] = y[1] + xc[3]
        # relative move (beta=0) vs the designer's absolute slot size (beta=1)
        P[i, 2] = y[2] + beta * float(np.log(max(1e-4, xc[7]) / max(1e-4, Xq[i, 7])))
        P[i, 3] = y[3]
        P[i, 4] = y[4]
        P[i, 5] = y[5]
    return P, cost
