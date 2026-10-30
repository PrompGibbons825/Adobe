"""Designer-decision term: re-order near-tied candidates toward what real designers did.

The top K plans (by the hand score) are emitted, the decisions visible in each output are read
off with ``decisions.decisions_of`` and compared with what the designer tree predicts for this
source and target shape (``decisions.predict``). A plan earns a confidence-weighted agreement;
the term is ``clip(blend * (agree - best agree of the K), -cap, 0)``. It only demotes, is
bounded by ``cap``, and never lets a plan that hides more core items than the hand score's
best plan move up: such plans (and every plan below the K) take ``-cap``, so their order
relative to the eligible plans can only stay or fall. The hard checks
(``_promote_passing``), the pixel critic and ``finalize`` still run afterwards, unchanged.

Off without ``designer_weights.json`` or with ``CHAMELEON_OFF=designer`` (comma-separated).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

WEIGHTS = Path(__file__).with_name("designer_weights.json")
CORE = ("headline", "cta", "logo", "hero")


def off(name: str) -> bool:
    return name in {s.strip() for s in os.environ.get("CHAMELEON_OFF", "").split(",")}


@lru_cache(maxsize=1)
def model() -> Optional[dict]:
    try:
        m = json.loads(WEIGHTS.read_text())
    except Exception:  # noqa: BLE001
        return None
    if not m.get("weights") or float(m.get("blend", 0)) <= 0 or float(m.get("cap", 0)) <= 0:
        return None
    return m


def _decisions():
    try:
        from . import decisions
        return decisions
    except Exception:  # noqa: BLE001
        return None


def enabled() -> bool:
    return not off("designer") and model() is not None and _decisions() is not None


def weight(m: dict, key: str) -> float:
    w = m["weights"]
    if key in w:
        return float(w[key])
    return float(w.get(key.split(":")[0], 0.0))


def agreement(pred: Dict[str, tuple], dec: Dict[str, str], m: dict) -> float:
    a = 0.0
    for k, (v, conf) in pred.items():
        if k in dec:
            w = weight(m, k)
            if w and dec[k] == v:
                a += w * float(conf)
    return a


def core_hidden(p) -> int:
    return sum(1 for h in p.hidden if h.role in CORE)


def output_of(p, source: dict, target: dict, d, ctx) -> dict:
    from . import emit, finalize

    return finalize(source, {"canvas": target, "elements": emit(p, d, ctx)}, target)


def apply(plans: List, source: dict, target: dict, d, ctx) -> None:
    """Add the bounded designer term to ``p.score`` (in place) and re-sort ``plans``."""
    if not plans or not enabled():
        return
    m = model()
    dm = _decisions()
    try:
        pred = dm.predict(dm.source_features(source, target))
    except Exception:  # noqa: BLE001
        return
    if not pred:
        return
    k_top = int(m.get("K", 12))
    blend, cap = float(m["blend"]), float(m["cap"])
    tgt = {"width": ctx.W, "height": ctx.H}
    floor = core_hidden(plans[0])
    head = plans[:k_top]
    agree = {}
    for p in head:
        if core_hidden(p) > floor:
            continue
        try:
            dec = dm.decisions_of(source, output_of(p, source, tgt, d, ctx))
        except Exception:  # noqa: BLE001
            continue
        agree[id(p)] = agreement(pred, dec, m)
    if not agree:
        return
    best = max(agree.values())
    for p in plans:
        a = agree.get(id(p))
        term = -cap if a is None else max(-cap, min(0.0, blend * (a - best)))
        p.breakdown["designer"] = term
        p.score += term
    plans.sort(key=lambda p: -p.score)
