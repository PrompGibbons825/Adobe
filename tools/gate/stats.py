"""Gate tool: the decision statistics (pre-registered in prereg.json, applied by run.py).

The unit is the **design**: a design's tiles are averaged first, and the bootstrap resamples
designs (a cluster bootstrap), so one design resized to 12 formats does not count 12 times.

A judged tile is a record

    {"tile": "i007", "set": "novel", "design": "<id>", "tier": 1,
     "score": {"overall": 0.75, "collisions": 1.0, ...},   # share of votes for the candidate
     "delta": {"overall": 0.5, ...}}                       # mean 1-10 score, candidate - base

Tests (one-sided, percentile cluster bootstrap with ``boot`` resamples):

* **win**: on the primary set's tier-1 tiles (loop/STEERING.md S2), criterion overall: at least
  ``min_win_designs`` designs, the lower bound at the look's alpha > 0.5 and the point estimate
  >= ``point_min``.
* **harm**: every other (set, tier) cell on overall, and every criterion on all tiles pooled: a
  cell whose upper bound at ``alpha_harm`` is below 0.5 is significant harm (cells with fewer
  than ``min_designs`` designs are reported, never significant).
* **futility** (early looks only): the win cell's upper bound at ``alpha_harm`` is below
  ``futility_upper``, so it cannot plausibly reach the bar; stopping then costs power, never
  false wins.

**Two families** (loop/STEERING.md S10): a tile record may carry ``family`` (each judge
family's own score, from tools.gate.panel.score_tile); ``score`` is then the pooled score.

* **family harm**: per family, the win cell on overall; an upper bound at ``alpha_family_harm``
  below 0.5 is significant harm and blocks an ACCEPT (no change only one family likes).
* **families_required**: a run whose tiles carry fewer families (one was dropped on its
  sentinels) can only REJECT or REPORT; where it would accept, the decision is REPORT.
* **family_coverage_min**: when a family voted on fewer than this share of the judged tiles (its
  asks failed), the pooled score is mostly the other family's, so an ACCEPT becomes REPORT.
* ``family_agreement``: the share of tiles where the families' overall votes agree (report).

Sequential looks: the panel judges designs in a pre-registered random order and looks after
``looks`` designs (40/80/120). The win test's alpha is spent with an O'Brien-Fleming-type
function, alpha(t) = 2 - 2 Phi(z_{1-alpha/2} / sqrt t), and each look gets the increment, so
the false-win rate over all looks stays at most alpha (Bonferroni over the increments).
Ties are rejections: anything short of a passing win test is REJECT.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

CRITERIA = ["overall", "collisions", "readable", "brand", "intentional"]

DEFAULT_RULE = {
    "primary": "novel",       # the set whose tier-1 tiles carry the win test
    "win_tier": 1,
    "alpha_win": 0.05,        # one-sided, spent over the looks
    "point_min": 0.58,
    "alpha_harm": 0.10,       # one-sided, at every look
    "min_designs": 5,         # a harm cell needs this many designs to be significant
    "min_win_designs": 20,    # the win cell needs this many (a percentile bootstrap is too narrow below)
    "futility_upper": 0.55,
    "looks": [40, 80, 120],   # designs judged before each look; the last is the cap
    "spending": "obf",        # "obf" or "pocock-bonferroni" (alpha / number of looks)
    "boot": 10000,
    "sentinel_min": 0.85,     # a family whose cumulative sentinel accuracy is below this is dropped
    "sentinel_min_n": 20,     # ... at an early look only once it has this many sentinels (always at the stop)
    "sentinel_share": 0.10,
    "alpha_family_harm": 0.10,  # one-sided, per family on the win cell (S10)
    "families_required": 2,   # fewer families judging: REJECT or REPORT, never ACCEPT
    "family_coverage_min": 0.8,  # a family that voted on fewer of the judged tiles: REPORT, never ACCEPT
    "proxy_reject_upper": 0.35,  # proxy may reject when its win-rate upper bound is below this
    "seed": 0,
}


def rule_with_defaults(rule: Optional[dict]) -> dict:
    r = dict(DEFAULT_RULE)
    r.update(rule or {})
    return r


# ---- bootstrap -------------------------------------------------------------------------------

def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _z(p: float) -> float:
    """Inverse normal CDF by bisection (no scipy dependency)."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def spend(alpha: float, fractions: Sequence[float], how: str = "obf") -> List[float]:
    """The nominal one-sided alpha of each look (increments of the spending function)."""
    if how == "pocock-bonferroni":
        return [alpha / len(fractions)] * len(fractions)
    z = _z(1 - alpha / 2)
    cum = [0.0] + [min(alpha, 2 - 2 * _phi(z / math.sqrt(max(t, 1e-9)))) for t in fractions]
    cum[-1] = alpha if fractions and fractions[-1] >= 1 else cum[-1]
    return [max(1e-6, cum[i + 1] - cum[i]) for i in range(len(fractions))]


def look_sizes(n_designs: int, looks: Sequence[int]) -> List[int]:
    """Look points for ``n_designs`` available designs: the planned looks below the total, then
    the total (capped at the last planned look)."""
    cap = min(n_designs, looks[-1])
    return [k for k in looks if k < cap] + ([cap] if cap > 0 else [])


def boot_means(values: Sequence[float], n: int, seed: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.zeros(0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    return v[idx].mean(axis=1)


def bounds(values: Sequence[float], alpha_lower: float, alpha_upper: float, n: int, seed: int) -> dict:
    """Point estimate (mean of design scores) and one-sided lower/upper bounds."""
    v = list(values)
    if not v:
        return {"n": 0, "point": None, "lower": None, "upper": None}
    b = boot_means(v, n, seed)
    return {"n": len(v), "point": float(np.mean(v)), "lower": float(np.quantile(b, alpha_lower)),
            "upper": float(np.quantile(b, 1 - alpha_upper))}


def design_scores(tiles: Iterable[dict], crit: str = "overall", sets: Optional[Iterable[str]] = None,
                  tier: Optional[int] = None) -> Dict[str, float]:
    """Mean score per design over the tiles that pass the filters."""
    sets = set(sets) if sets is not None else None
    acc: Dict[str, List[float]] = {}
    for t in tiles:
        if sets is not None and t["set"] not in sets:
            continue
        if tier is not None and t["tier"] != tier:
            continue
        s = t["score"].get(crit)
        if s is None:
            continue
        acc.setdefault(f"{t['set']}/{t['design']}", []).append(float(s))
    return {d: sum(v) / len(v) for d, v in acc.items()}


# ---- the decision ----------------------------------------------------------------------------

def evaluate(tiles: List[dict], rule: dict, alpha_win: float, final: bool, seed: int = 0) -> dict:
    """All tests on ``tiles`` at one look. ``alpha_win`` is this look's spent alpha."""
    r = rule_with_defaults(rule)
    n, ah = r["boot"], r["alpha_harm"]
    sets = sorted({t["set"] for t in tiles} | {r["primary"]})
    cells: Dict[str, dict] = {}
    for k, s in enumerate(sets):
        for tier in (1, 2):
            v = list(design_scores(tiles, "overall", [s], tier).values())
            al = alpha_win if (s == r["primary"] and tier == r["win_tier"]) else ah
            cells[f"{s} t{tier}"] = bounds(v, al, ah, n, seed + 17 * k + tier)
    crits = {}
    for j, c in enumerate(CRITERIA):
        v = list(design_scores(tiles, c).values())
        crits[c] = bounds(v, ah, ah, n, seed + 1000 + j)
    win_key = f"{r['primary']} t{r['win_tier']}"
    win = cells[win_key]
    win_pass = bool(win["n"] >= r["min_win_designs"] and win["lower"] > 0.5 and win["point"] >= r["point_min"])
    harm = []
    for key, c in list(cells.items()) + [(f"criterion {k}", v) for k, v in crits.items()]:
        if key == win_key or not c["n"] or c["n"] < r["min_designs"]:
            continue
        if c["upper"] < 0.5:
            harm.append(key)
    fams = sorted({f for t in tiles for f in (t.get("family") or {})})
    family = {}
    for j, f in enumerate(fams):
        ft = [dict(t, score=t["family"][f]) for t in tiles if f in (t.get("family") or {})]
        v = list(design_scores(ft, "overall", [r["primary"]], r["win_tier"]).values())
        family[f] = b = bounds(v, r["alpha_family_harm"], r["alpha_family_harm"], n, seed + 2000 + j)
        if b["n"] and b["n"] >= r["min_designs"] and b["upper"] < 0.5:
            harm.append(f"family {f}")
    single = bool(fams) and len(fams) < int(r["families_required"])
    coverage = {f: sum(1 for t in tiles if f in (t.get("family") or {})) / len(tiles) for f in fams} if tiles else {}
    thin = [f for f, c in coverage.items() if c < float(r["family_coverage_min"])]
    reasons = []
    if harm:
        reasons.append("significant harm on " + ", ".join(harm))
    if final:
        if win["n"] < r["min_win_designs"]:
            reasons.append(f"{win['n']} changed designs in {win_key} (< {r['min_win_designs']})")
        elif not win_pass:
            reasons.append(f"no significant win on {win_key}")
        decision = "ACCEPT" if win_pass and not harm else "REJECT"
    else:
        futile = bool(win["n"] and win["upper"] < r["futility_upper"])
        if harm:
            decision = "REJECT"
        elif win_pass:
            decision = "ACCEPT"
        elif futile:
            decision = "REJECT"
            reasons.append(f"futile: {win_key} upper {win['upper']:.2f} < {r['futility_upper']}")
        else:
            decision = "CONTINUE"
    if single and decision == "ACCEPT":
        decision = "REPORT"
        reasons.append(f"single-family run ({', '.join(fams)}): would accept, but can only reject or report")
    if thin and decision == "ACCEPT":
        decision = "REPORT"
        reasons.append("would accept, but " + ", ".join(f"{f} voted on {coverage[f]:.0%}" for f in thin)
                       + f" of tiles (< {float(r['family_coverage_min']):.0%})")
    return {"decision": decision, "reasons": reasons, "win_cell": win_key, "win_pass": win_pass,
            "alpha_win": alpha_win, "cells": cells, "criteria": crits, "harm": harm, "family": family,
            "families": fams, "coverage": coverage, "agreement": family_agreement(tiles)}


def family_agreement(tiles: Iterable[dict]) -> Optional[dict]:
    """Share of tiles judged by two or more families whose overall votes all agree."""
    n = agree = 0
    for t in tiles:
        fv = [v.get("overall") for v in (t.get("family") or {}).values() if v.get("overall") is not None]
        if len(fv) >= 2:
            n += 1
            agree += len(set(fv)) == 1
    return {"n": n, "rate": agree / n} if n else None


def sequential(tiles: List[dict], order: List[str], rule: dict, seed: int = 0) -> dict:
    """Run the looks over designs in ``order`` (ids ``<set>/<design>``) and stop at the first
    decision. Used to replay a finished panel (null permutations, tests)."""
    r = rule_with_defaults(rule)
    sizes = look_sizes(len(order), r["looks"])
    if not sizes:
        return {"decision": "REJECT", "reasons": ["no changed designs"], "look": 0, "looks": []}
    alphas = spend(r["alpha_win"], [k / sizes[-1] for k in sizes], r["spending"])
    by = {}
    for t in tiles:
        by.setdefault(f"{t['set']}/{t['design']}", []).append(t)
    res = None
    for i, (k, a) in enumerate(zip(sizes, alphas)):
        sub = [t for d in order[:k] for t in by.get(d, [])]
        res = evaluate(sub, r, a, final=i == len(sizes) - 1, seed=seed + i)
        res["look"], res["designs"] = i + 1, k
        if res["decision"] != "CONTINUE":
            break
    res["looks"] = sizes
    return res


def flip(tiles: List[dict], designs: set) -> List[dict]:
    """The tiles with candidate and base swapped on the given designs."""
    out = []
    for t in tiles:
        if f"{t['set']}/{t['design']}" in designs:
            t = dict(t, score={k: 1 - v for k, v in t["score"].items()},
                     delta={k: -v for k, v in (t.get("delta") or {}).items()})
            if t.get("family"):
                t["family"] = {f: {k: 1 - v for k, v in sc.items()} for f, sc in t["family"].items()}
        out.append(t)
    return out


def family_view(tiles: Iterable[dict], fam: str) -> List[dict]:
    """The tiles as family ``fam`` alone scored them (for its own null run)."""
    return [dict(t, score=dict(t["family"][fam]), family={fam: dict(t["family"][fam])})
            for t in tiles if fam in (t.get("family") or {})]


def null_false_win_rate(tiles: List[dict], order: List[str], rule: dict, n_perm: int = 200,
                        seed: int = 0, boot: int = 2000) -> dict:
    """For a null run (A vs A'), the share of random relabellings the rule would ACCEPT. Under the
    null which side is "candidate" is exchangeable per design, so each permutation flips a random
    half of the designs and replays the sequential rule."""
    r = rule_with_defaults(rule)
    r["boot"] = min(r["boot"], boot)
    rng = np.random.default_rng(seed)
    wins = 0
    for p in range(n_perm):
        fl = {d for d in order if rng.random() < 0.5}
        if sequential(flip(tiles, fl), order, r, seed=seed + 7919 * (p + 1))["decision"] == "ACCEPT":
            wins += 1
    observed = sequential(tiles, order, r, seed=seed)["decision"]
    return {"n_perm": n_perm, "false_wins": wins, "rate": wins / n_perm if n_perm else None,
            "observed": observed}


def score_deltas(tiles: List[dict]) -> Dict[str, Optional[float]]:
    """Mean over designs of the 1-10 score difference (candidate - base), per criterion."""
    out = {}
    for c in CRITERIA:
        acc: Dict[str, List[float]] = {}
        for t in tiles:
            v = (t.get("delta") or {}).get(c)
            if v is not None:
                acc.setdefault(f"{t['set']}/{t['design']}", []).append(float(v))
        out[c] = round(float(np.mean([np.mean(v) for v in acc.values()])), 3) if acc else None
    return out
