"""Replay engine pick among dumped candidates (dev only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.smart import ranker as R  # noqa: E402

DATA = ROOT / "tools" / "crello" / "ranker"


def load(*sets: str) -> List[dict]:
    rows = []
    for s in sets:
        p = DATA / f"{s}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            r["key"] = f"{r['fam']}:{r['ti']}"
            rows.append(r)
    return rows


def candidates(row: dict) -> List[dict]:
    """Prune like prune_relaxed, attach updated features."""
    cs = list(row["cands"])
    full = [c["core_hidden"] for c in cs if c["relax"] >= 1.0]
    if full:
        b = min(full)
        cs = [c for c in cs if c["relax"] >= 1.0 or c["core_hidden"] < b]
    return cs


def pick(row: dict, cs: List[dict], model: Optional[dict] = None, blend: float = 0.0, top: int = 8,
         margin: float = 10.0):
    if not cs:
        return None, row["fallback"]
    if model is not None and blend:
        rs = [R.predict(c["feats"], model) for c in cs]
        keys = R.blended([c["hand"] for c in cs], rs, blend, margin)
    else:
        keys = [c["hand"] for c in cs]
    order = sorted(range(len(cs)), key=lambda i: -keys[i])
    head = sorted(order[:top], key=lambda i: -(keys[i] - cs[i]["pix"]))
    order = head + order[top:]
    for i in order:
        if cs[i]["ok"]:
            return cs[i], cs[i]["label"]
    return None, row["fallback"]
