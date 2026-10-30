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


def search(S: Scene, W: float, H: float, p: Params = None, refine: bool = True):
    p = p or Params()
    s_cap = max(W / S.W, H / S.H) * 1.0
    s_cap = min(s_cap, 1.6 * math.sqrt(W * H / (S.W * S.H)))
    g0 = _gap0(S)
    cands = []
    for spread, nm in ((False, "keep"), (True, "keep-spread"), ("canvas", "keep-canvas")):
        pl, s = keep_layout(S, W, H, p, spread)
        cands.append((nm, pl, None))
    for name, topo in topologies(S, W, H):
        for va in (("spread", "center") if not name.startswith("cols") else ("spread",)):
            pl, s = realise(S, topo, W, H, p, s_cap, g0, valign=va)
            cands.append((f"{name}/{va}", pl, (topo, va)))
    scored = [(energy(S, pl, W, H), k, name, pl) for k, (name, pl, _) in enumerate(cands)]
    scored.sort(key=lambda t: (t[0], t[1]))
    if refine:
        # coordinate descent over the global geometry knobs for the 2 best topologies
        extra = []
        # knob refinement only for the winning topology (refining all would bias selection
        # away from what the weights were fit on)
        tops = [cands[k] for _, k, _, _ in scored[:1] if cands[k][2] is not None]
        for name, _, (topo, va) in tops:
            cur = Params(p.margin_frac, p.gap_frac, p.text_widen)
            pl, _ = realise(S, topo, W, H, cur, s_cap, g0, valign=va)
            e_cur = energy(S, pl, W, H)
            for _sweep in range(2):
                for attr, vals in (("margin_frac", (0.025, 0.05, 0.08)), ("gap_frac", (0.6, 1.0, 1.6)),
                                   ("text_widen", (0.5, 1.0, 1.6))):
                    for v in vals:
                        if getattr(cur, attr) == v:
                            continue
                        trial = Params(cur.margin_frac, cur.gap_frac, cur.text_widen)
                        setattr(trial, attr, v)
                        pl2, _ = realise(S, topo, W, H, trial, s_cap, g0, valign=va)
                        e2 = energy(S, pl2, W, H)
                        if e2 < e_cur - 1e-9:
                            cur, pl, e_cur = trial, pl2, e2
            extra.append((e_cur, len(cands) + len(extra), name + "/refined", pl))
        scored = sorted(scored + extra, key=lambda t: (t[0], t[1]))
        # polish (collision-resolving coordinate descent) the 2 best plus the position-keeping
        # candidates, whose only defect is usually a local collision
        pick = scored[:1] + [t for t in scored if t[2] in ("keep-spread", "keep-canvas") and t not in scored[:1]]
        pol = []
        for e0, k0, n0, pl0 in pick:
            pl1, e1 = polish(S, pl0, W, H, e0, budget=80)
            if e1 < e0:
                pol.append((e1, -1 - len(pol), n0 + "/polished", pl1))
        scored = sorted(scored + pol, key=lambda t: (t[0], t[1]))
    return scored


def polish(S: Scene, placed, W: float, H: float, e_cur: float, budget: int = 250):
    """Per-block coordinate descent on the energy: move / shrink blocks that collide."""
    import copy
    evals = 0
    pl = copy.deepcopy(placed)

    def colliding():
        ids = list(pl)
        bad = set()
        for a in range(len(ids)):
            for c in range(a + 1, len(ids)):
                p, q = pl[ids[a]], pl[ids[c]]
                if min(p.x + p.w, q.x + q.w) - max(p.x, q.x) > 1 and min(p.y + p.h, q.y + q.h) - max(p.y, q.y) > 1:
                    bad.add(ids[a])
                    bad.add(ids[c])
        return sorted(bad)

    for step in (0.02, 0.05, 0.12):
        improved = True
        while improved and evals < budget:
            improved = False
            for bid in colliding():
                p = pl[bid]
                best = None
                for dx, dy, f in ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (0, 0, 0.85)):
                    q = copy.copy(p)
                    q.x = min(max(0.0, p.x + dx * step * W), W - p.w)
                    q.y = min(max(0.0, p.y + dy * step * H), H - p.h)
                    if f != 1:
                        if p.fs and p.fs * f < 6:
                            continue
                        q.w, q.h = p.w * f, p.h * f
                        q.x, q.y = p.x + (p.w - q.w) / 2, p.y + (p.h - q.h) / 2
                        q.s = p.s * f
                        if p.fs:
                            q.fs = max(1, int(p.fs * f))
                    pl[bid] = q
                    e = energy(S, pl, W, H)
                    evals += 1
                    if e < e_cur - 1e-6 and (best is None or e < best[0]):
                        best = (e, q)
                    pl[bid] = p
                if best:
                    e_cur, pl[bid] = best
                    improved = True
                if evals >= budget:
                    break
    return pl, e_cur


def _clamp_box(x, y, w, h, W, H):
    if w > W:
        w = W
    if h > H:
        h = H
    x = min(max(0.0, x), W - w)
    y = min(max(0.0, y), H - h)
    return x, y, w, h
