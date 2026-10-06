"""Dev tool: replay the engine's choice among dumped candidates.

tools/ranker_data.py dumps every candidate the engine generates with its gates
wide open. Given a threshold setting, a ranker model and a blend weight, this
replays exactly what plan_design() would ship:

  gate filter -> prune_relaxed -> hand score (the "visible" term and the
  archetype prior recomputed for that candidate set) -> + blend * ranker ->
  sort -> pixel re-rank of the top 8 -> first candidate without copy over copy
  (none: the fit-and-centre fallback)

and returns the designer similarity of the pick. Dev-only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.smart import ranker as R  # noqa: E402
from engine.smart import thresholds as T  # noqa: E402
from engine.smart.score import ALPHA, _prior  # noqa: E402

DATA = ROOT / "tools" / "crello" / "ranker"
DEFAULT = T.snapshot()


def load(*sets: str) -> List[dict]:
    rows = []
    for s in sets:
        p = DATA / f"{s}.jsonl"
        for line in p.read_text().splitlines():
            r = json.loads(line)
            r["key"] = f"{r['fam']}:{r['ti']}"
            rows.append(r)
    return rows


def family(row: dict) -> str:
    return row["fam"]


def _allowed(c: dict, r: float, r0: float, t: dict, ignore_shape: bool = False) -> bool:
    a = R.arch_of(c["name"])
    if a == "adapt":
        return abs(math.log(r / r0)) <= math.log(t["ADAPT_TOL"]) + 1e-9
    if ignore_shape:
        return True
    if a in ("stack", "stack-side"):
        return r <= t["STACK_MAX_R"]
    if a in ("split", "split-fold"):
        return r >= t["SPLIT_MIN_R"]
    if a in ("row", "row-stacked"):
        return r >= t["ROW_MIN_R"]
    return True


def candidates(row: dict, t: Optional[dict] = None) -> List[dict]:
    """The candidate set the engine would score under thresholds `t`, with the
    hand score recomputed ("hand") and the ranker's feature dict updated."""
    t = dict(DEFAULT, **(t or {}))
    r, r0 = row["W"] / row["H"], row["W0"] / row["H0"]
    cs = [c for c in row["cands"] if _allowed(c, r, r0, t)]
    if not cs:  # the engine's second pass forces stack and row for any shape
        cs = [c for c in row["cands"] if _allowed(c, r, r0, t, ignore_shape=True)]
    full = [c["core_hidden"] for c in cs if c["relax"] >= 1.0]
    if full:
        b = min(full)
        cs = [c for c in cs if c["relax"] >= 1.0 or c["core_hidden"] < b]
    best: Dict[str, float] = {}
    for c in cs:
        for key, _, _, s in c["items"]:
            best[key] = max(best.get(key, 0.0), s)
    saved = {k: getattr(T, k) for k in t}
    for k, v in t.items():
        setattr(T, k, v)
    try:
        out = []
        for c in cs:
            vis = 0.0
            for key, kind, imp, s in c["items"]:
                rel = s / max(1e-9, best.get(key, s))
                if kind in ("image", "composite"):
                    vis += imp * (1.0 + 0.6 * math.log(max(1e-3, min(1.0, rel / 0.6))))
                else:
                    vis += imp * (1.0 + ALPHA * math.log(max(1e-3, min(1.0, rel / 0.85))))
            pr = _prior(c["name"], r, r0)
            f = dict(c["feats"])
            f["t:visible"] = vis / 100.0
            f["t:prior"] = pr / 100.0
            out.append(dict(c, hand=vis + c["rest"] + pr, f=f))
        return out
    finally:
        for k, v in saved.items():
            setattr(T, k, v)


def pick(row: dict, cs: List[dict], model: Optional[dict] = None, blend: float = 0.0, top: int = 8):
    """(chosen candidate or None for the fallback, its label dict)."""
    if not cs:
        return None, row["fallback"]
    if model is not None and blend:
        rs = [R.predict(c["f"], model) for c in cs]
        mx = max(rs)
        keys = [c["hand"] + blend * (x - mx) for c, x in zip(cs, rs)]
    else:
        keys = [c["hand"] for c in cs]
    order = sorted(range(len(cs)), key=lambda i: -keys[i])
    head = sorted(order[:top], key=lambda i: -(keys[i] - cs[i]["pix"]))
    order = head + order[top:]
    for i in order:
        if cs[i]["ok"]:
            return cs[i], cs[i]["label"]
    return None, row["fallback"]


def evaluate(rows: List[dict], t: Optional[dict] = None, model: Optional[dict] = None, blend: float = 0.0,
             cache: Optional[dict] = None) -> dict:
    sims, names, oracle = [], [], []
    for row in rows:
        k = (row["key"], json.dumps(t, sort_keys=True))
        if cache is not None and k in cache:
            cs = cache[k]
        else:
            cs = candidates(row, t)
            if cache is not None:
                cache[k] = cs
        c, lab = pick(row, cs, model, blend)
        sims.append(lab["sim"])
        names.append(c["name"] if c else "fallback")
        ok = [x["label"]["sim"] for x in cs if x["ok"]]
        oracle.append(max(ok) if ok else row["fallback"]["sim"])
    n = max(1, len(sims))
    return dict(sim=sum(sims) / n, oracle=sum(oracle) / n, n=len(sims), sims=sims, names=names)
