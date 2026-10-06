"""Dev tool: train the candidate ranker on designer resizes and ship its weights.

Data: tools/crello/ranker/{old,train}.jsonl (tools/ranker_data.py). Training
groups are the "train" families of tools/crello_train.py plus the even
("tune") half of the 64 regression families; the odd half and the "held2"
families are only ever evaluated.

Model: pairwise logistic regression (RankNet with a linear scorer). Within a
(family, target) group every pair of candidates whose designer similarity
differs by more than MARGIN is a training pair, weighted by that difference;
each group carries the same total weight. Features are standardised; L2
penalty. The blend weight (how many hand-score points one logit is worth) and
the L2 strength are chosen by 5-fold cross-validation over families, scoring
the *pick* the full replayed pipeline makes (tools/ranker_sim.py), not the
pairwise loss. The blend is constrained: the candidate it picks may not, on
average over the validation folds, carry more pixel penalty (copy that lost
contrast or landed on texture, engine/smart/pixels.py) than the hand score's
pick plus PIX_TOL. Unconstrained, a large blend buys more designer similarity
by overruling the pixel critic, and the proxy judge's contrast and busy-ness
counts get worse (measured, see the workstream note). The final model is
refit on all training groups and written to engine/smart/ranker_weights.json.

    .venv/bin/python -m tools.ranker_train            # CV, fit, write weights, report
    .venv/bin/python -m tools.ranker_train --report   # evaluate the shipped weights only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.smart import ranker as R  # noqa: E402
from engine.smart import thresholds as T  # noqa: E402
from tools import ranker_sim as S  # noqa: E402

OUT = ROOT / "engine" / "smart" / "ranker_weights.json"
MARGIN = 0.02
MAX_PAIRS = 200  # per group, sampled
BLENDS = [0.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 80.0, 160.0, 400.0]
PIX_TOL = 0.02  # relative: mean pixel penalty of the pick may rise by at most 2%
L2S = [0.3, 3.0, 30.0]


def fold_of(fam: str, k: int = 5) -> int:
    return int(hashlib.md5(fam.encode()).hexdigest()[:6], 16) % k


def prepared(rows, t):
    """[(row, candidates under thresholds t)] for rows that have candidates."""
    out = []
    for r in rows:
        cs = S.candidates(r, t)
        if cs:
            out.append((r, cs))
    return out


def feature_names(prep):
    names = set()
    for _, cs in prep:
        for c in cs:
            names.update(c["f"].keys())
    return sorted(names)


def matrix(cs, names):
    return np.array([[c["f"].get(n, 0.0) for n in names] for c in cs], dtype=np.float64)


def fit(prep, names, l2: float, seed: int = 0):
    rng = random.Random(seed)
    X_all = np.concatenate([matrix(cs, names) for _, cs in prep])
    mu = X_all.mean(0)
    sd = X_all.std(0)
    sd[sd < 1e-9] = 1.0
    D, Wt = [], []
    for _, cs in prep:
        X = (matrix(cs, names) - mu) / sd
        y = [c["label"]["sim"] for c in cs]
        prs = [(i, j, y[i] - y[j]) for i in range(len(cs)) for j in range(len(cs)) if y[i] - y[j] > MARGIN]
        if not prs:
            continue
        if len(prs) > MAX_PAIRS:
            prs = rng.sample(prs, MAX_PAIRS)
        tot = sum(w for _, _, w in prs)
        for i, j, w in prs:
            D.append(X[i] - X[j])
            Wt.append(w / tot)
    D = np.array(D)
    Wt = np.array(Wt)
    w = np.zeros(D.shape[1])
    for _ in range(30):  # Newton on the weighted logistic loss
        z = D @ w
        p = 1.0 / (1.0 + np.exp(-z))
        g = -(D * (Wt * (1 - p))[:, None]).sum(0) + l2 * w
        Hm = (D * (Wt * p * (1 - p))[:, None]).T @ D + l2 * np.eye(D.shape[1])
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return dict(kind="linear-pairwise", features=names, weights=w.tolist(), mean=mu.tolist(), std=sd.tolist())


def evaluate(prep, model, blend, pix=None):
    sims, names = [], []
    for row, cs in prep:
        c, lab = S.pick(row, cs, model, blend)
        sims.append(lab["sim"])
        names.append(c["name"] if c else "fallback")
        if pix is not None:
            pix.append(c["pix"] if c else 0.0)
    return sims, names


def summary(prep, model, blend, label):
    base, bn = evaluate(prep, None, 0.0)
    new, nn = evaluate(prep, model, blend)
    oracle = [max([c["label"]["sim"] for c in cs if c["ok"]] or [row["fallback"]["sim"]]) for row, cs in prep]
    changed = sum(1 for a, b in zip(bn, nn) if a != b)
    better = sum(1 for a, b in zip(base, new) if b > a + 1e-9)
    worse = sum(1 for a, b in zip(base, new) if b < a - 1e-9)
    n = max(1, len(prep))
    print(f"{label:<28} n={len(prep):4d}  designer-sim  hand {sum(base) / n:.4f} -> ranked {sum(new) / n:.4f}"
          f"   oracle {sum(oracle) / n:.4f}   pick changed {changed} ({changed / n:.0%}): {better} closer, {worse} further")
    return dict(n=len(prep), hand=sum(base) / n, ranked=sum(new) / n, oracle=sum(oracle) / n, changed=changed,
                better=better, worse=worse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    t = {k: getattr(T, k) for k in ("STACK_MAX_R", "SPLIT_MIN_R", "ROW_MIN_R", "ADAPT_TOL", "ADAPT_PRIOR",
                                     "STACK_KNEE", "SPLIT_PEAK", "ROW_KNEE")}
    rows = S.load("old", "train")
    train = prepared([r for r in rows if r["split"] in ("train", "tune")], t)
    held = {s: prepared([r for r in rows if r["split"] == s], t) for s in ("held", "held2")}
    if a.report:
        m = R.model()
        print("shipped weights:", OUT, "blend", m["blend"])
        summary(train, m, m["blend"], "train (in-sample)")
        for s, p in held.items():
            summary(p, m, m["blend"], f"held-out {s}")
        summary(held["held"] + held["held2"], m, m["blend"], "held-out all")
        return
    names = feature_names(train)
    print(f"training groups {len(train)}  features {len(names)}  thresholds {t}")
    # cross-validation over families: choose l2 and blend by the replayed pick
    res, pixr = {}, {}
    for l2 in L2S:
        sims = {b: [] for b in BLENDS}
        pix = {b: [] for b in BLENDS}
        for k in range(5):
            tr = [x for x in train if fold_of(x[0]["fam"]) != k]
            va = [x for x in train if fold_of(x[0]["fam"]) == k]
            m = fit(tr, names, l2)
            for b in BLENDS:
                sims[b] += evaluate(va, m, b, pix[b])[0]
        for b in BLENDS:
            res[(l2, b)] = sum(sims[b]) / len(sims[b])
            pixr[(l2, b)] = sum(pix[b]) / len(pix[b])
        print(f"  l2 {l2:<6g} " + "  ".join(f"b{b:g}:{res[(l2, b)]:.4f}/px{pixr[(l2, b)]:.1f}" for b in BLENDS),
              flush=True)
    ok = {k: v for k, v in res.items() if pixr[k] <= pixr[(k[0], 0.0)] * (1.0 + PIX_TOL)}
    (l2, blend), cv = max(ok.items(), key=lambda kv: kv[1])
    (ul2, ublend), ucv = max(res.items(), key=lambda kv: kv[1])
    print(f"chosen l2 {l2:g} blend {blend:g}  cv designer-sim {cv:.4f} (hand alone {res[(l2, 0.0)]:.4f}), "
          f"pick pixel penalty {pixr[(l2, blend)]:.1f} (hand alone {pixr[(l2, 0.0)]:.1f})")
    print(f"unconstrained best: l2 {ul2:g} blend {ublend:g}  cv designer-sim {ucv:.4f}, "
          f"pick pixel penalty {pixr[(ul2, ublend)]:.1f}")
    m = fit(train, names, l2)
    m["blend"] = blend
    m["meta"] = dict(margin=MARGIN, l2=l2, cv_sim=round(cv, 4), cv_hand=round(res[(l2, 0.0)], 4), pix_tol=PIX_TOL,
                     cv_pix=round(pixr[(l2, blend)], 2), cv_pix_hand=round(pixr[(l2, 0.0)], 2), groups=len(train),
                     thresholds=t, trained_on="tools/crello train families + even regression families",
                     made_by="python -m tools.ranker_train (tools/ranker_data.py dumps)")
    rnd = lambda v: [round(x, 6) for x in v]  # noqa: E731
    m["weights"], m["mean"], m["std"] = rnd(m["weights"]), rnd(m["mean"]), rnd(m["std"])
    OUT.write_text(json.dumps(m, indent=0))
    print("wrote", OUT, f"({OUT.stat().st_size // 1024} KB)")
    top = sorted(zip(names, m["weights"]), key=lambda x: -abs(x[1]))[:15]
    print("largest weights:", ", ".join(f"{n} {w:+.2f}" for n, w in top))
    summary(train, m, blend, "train (in-sample)")
    for s, p in held.items():
        summary(p, m, blend, f"held-out {s}")
    summary(held["held"] + held["held2"], m, blend, "held-out all")


if __name__ == "__main__":
    main()
