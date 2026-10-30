"""Global search over topologies + continuous refinement, minimising the energy (owner: topology
and final geometry)."""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Tuple

from .analyse import Scene, analyse
from .energy import energy
from .layout import Params, Placed, keep_layout, realise


def _partitions(n: int, k: int):
    """Contiguous partitions of range(n) into k non-empty parts."""
    for cuts in itertools.combinations(range(1, n), k - 1):
        b = (0,) + cuts + (n,)
        yield [list(range(b[i], b[i + 1])) for i in range(k)]


def topologies(S: Scene, W: float, H: float) -> List[Tuple[str, list]]:
    rows = S.rows
    out = []
    if not rows:
        return out
    out.append(("stack", [rows]))
    flat = [i for r in rows for i in r]
    if len(flat) > 1:
        out.append(("column", [[[i] for i in flat]]))
        out.append(("line", [[flat]]))
    m = len(rows)
    for k in range(2, min(4, m) + 1):
        parts = list(_partitions(m, k))
        if len(parts) > 6:
            parts = parts[:: max(1, len(parts) // 6)]
        for part in parts:
            out.append((f"cols{k}", [[rows[r] for r in p] for p in part]))
    # hero split: the largest non-text block alone on one side
    nt = [b for b in S.blocks if b.kind != "text"]
    if nt and len(S.blocks) > 1:
        hero = max(nt, key=lambda b: b.w * b.h)
        restrows = [[i for i in r if i != hero.bid] for r in rows]
        restrows = [r for r in restrows if r]
        if restrows:
            out.append(("hero-left", [[[hero.bid]], restrows]))
            out.append(("hero-right", [restrows, [[hero.bid]]]))
            out.append(("hero-top", [[[hero.bid]] + restrows]))
    # rows of two: pair consecutive single rows (grids for wide-ish targets)
    if m >= 4:
        paired = [sum(rows[i:i + 2], []) for i in range(0, m, 2)]
        out.append(("pairs", [paired]))
    # dedupe
    seen = set()
    res = []
    for name, t in out:
        key = repr(t)
        if key not in seen:
            seen.add(key)
            res.append((name, t))
    return res


def _gap0(S: Scene) -> float:
    gaps = []
    for r0, r1 in zip(S.rows, S.rows[1:]):
        b0 = max(S.blocks[i].y + S.blocks[i].h for i in r0)
        b1 = min(S.blocks[i].y for i in r1)
        if b1 - b0 > 0:
            gaps.append(b1 - b0)
    md = min(S.W, S.H)
    if not gaps:
        return 0.03 * md
    gaps.sort()
    return min(max(gaps[len(gaps) // 2], 0.015 * md), 0.08 * md)


def search(S: Scene, W: float, H: float, p: Params = None):
    p = p or Params()
    s_cap = max(W / S.W, H / S.H) * 1.0
    s_cap = min(s_cap, 1.6 * math.sqrt(W * H / (S.W * S.H)))
    g0 = _gap0(S)
    cands = []
    for spread in (False, True):
        pl, s = keep_layout(S, W, H, p, spread)
        cands.append((("keep-spread" if spread else "keep"), pl))
    for name, topo in topologies(S, W, H):
        for va in ("spread", "center"):
            pl, s = realise(S, topo, W, H, p, s_cap, g0, valign=va)
            cands.append((f"{name}/{va}", pl))
    scored = [(energy(S, pl, W, H), k, name, pl) for k, (name, pl) in enumerate(cands)]
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored


def _clamp_box(x, y, w, h, W, H):
    if w > W:
        w = W
    if h > H:
        h = H
    x = min(max(0.0, x), W - w)
    y = min(max(0.0, y), H - h)
    return x, y, w, h
