"""Build the training table: one row per (source element, designer's resize of it).

Data: B0's wide Crello family index (CDLA-Permissive-2.0), $CRELLO_WIDE/families.jsonl.
Families that are in the dev case sample ($CRELLO_WIDE/cases, the bench + tournament pool) are
held out entirely, so the bench measures transfer to unseen families.

    .venv/bin/python -m tools.learn_prep  -> /data/agents/work/night/lane-learn/train.npz
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.learn.features import element_features, element_targets  # noqa: E402

WIDE = Path(os.environ.get("CRELLO_WIDE", "/data/agents/work/crello-wide"))
OUT = Path("/data/agents/work/night/lane-learn/train.npz")


def member_design(m):
    els = []
    for k, e in enumerate(m["els"]):
        t = "text" if e["kind"] == "TextElement" else "image"
        d = dict(id=f"e{k}", type=t, x=e["x"], y=e["y"], width=e["w"], height=e["h"], z_index=k,
                 rotation=e.get("angle", 0.0), kind=e["kind"])
        if t == "text":
            d.update(content=e.get("text", ""), font_size=e.get("fs") or 12.0)
        els.append(d)
    return dict(canvas=dict(width=m["W"], height=m["H"]), elements=els)


def match(a, b):
    """Element correspondence between two sizes of one family: texts by content, the rest by
    kind in z order."""
    def keys(d):
        seen = defaultdict(int)
        out = {}
        for i, e in enumerate(d["elements"]):
            k = ("t", " ".join(str(e.get("content", "")).lower().split())) if e["type"] == "text" else ("o", e["kind"])
            out[(k, seen[k])] = i
            seen[k] += 1
        return out
    ka, kb = keys(a), keys(b)
    return [(ka[k], kb[k]) for k in ka if k in kb]


def main():
    held = {p.stem for p in (WIDE / "cases").glob("*.json")}
    Xs, Ys, fam_ids, pair_ids, src_idx = [], [], [], [], []
    nf = npairs = 0
    for line in open(WIDE / "families.jsonl"):
        f = json.loads(line)
        if f["fam"] in held:
            continue
        designs = [member_design(m) for m in f["members"]]
        for a in range(len(designs)):
            for b in range(len(designs)):
                if a == b:
                    continue
                src, dst = designs[a], designs[b]
                pairs = match(src, dst)
                if len(pairs) < 2:
                    continue
                X = element_features(src, dst["canvas"])
                Y = element_targets(src, dst, pairs)
                Xs.append(X[[i for i, _ in pairs]])
                Ys.append(Y)
                fam_ids += [nf] * len(pairs)
                pair_ids += [npairs] * len(pairs)
                src_idx += [i for i, _ in pairs]
                npairs += 1
        nf += 1
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=X, Y=Y, fam=np.array(fam_ids, np.int32), pair=np.array(pair_ids, np.int32),
                        src=np.array(src_idx, np.int32))
    print(f"families {nf}, pairs {npairs}, rows {len(X)} -> {OUT}")


if __name__ == "__main__":
    main()
