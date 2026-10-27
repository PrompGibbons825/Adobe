"""Train the candidate ranker on Crello designer pairs and ship ranker_weights.json.

    .venv/bin/python -m tools.ranker_train
    .venv/bin/python -m tools.ranker_train --report
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
from tools import ranker_sim as S  # noqa: E402

OUT = ROOT / "engine" / "smart" / "ranker_weights.json"
MARGIN = 0.02
MAX_PAIRS = 200
BLENDS = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0]
L2S = [0.3, 3.0, 30.0]
PIX_TOL = 0.02
N_STUMPS = 40


def fold_of(fam: str, k: int = 5) -> int:
    return int(hashlib.md5(fam.encode()).hexdigest()[:6], 16) % k


def prepared(rows):
    out = []
    for r in rows:
        cs = S.candidates(r)
        if cs:
            out.append((r, cs))
    return out


def feature_names(prep):
    names = set()
    for _, cs in prep:
        for c in cs:
            names.update(c["feats"].keys())
    return sorted(names)


def matrix(cs, names):
    return np.array([[c["feats"].get(n, 0.0) for n in names] for c in cs], dtype=np.float64)


def pairwise_data(prep, names, seed=0):
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
    return np.array(D), np.array(Wt), mu, sd


def fit_linear(prep, names, l2: float, seed: int = 0):
    D, Wt, mu, sd = pairwise_data(prep, names, seed)
    if len(D) < 10:
        return None
    w = np.zeros(D.shape[1])
    for _ in range(40):
        z = D @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = -(D * (Wt * (1 - p))[:, None]).sum(0) + l2 * w
        Hm = (D * (Wt * p * (1 - p))[:, None]).T @ D + l2 * np.eye(D.shape[1])
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return dict(kind="linear-pairwise", features=names, weights=w.tolist(), mean=mu.tolist(), std=sd.tolist())


def fit_stumps(prep, names, n_stumps=N_STUMPS, seed=0):
    rng = random.Random(seed)
    X_all = np.concatenate([matrix(cs, names) for _, cs in prep])
    mu = X_all.mean(0)
    sd = X_all.std(0)
    sd[sd < 1e-9] = 1.0
    labels = []
    feats = []
    for _, cs in prep:
        for c in cs:
            labels.append(c["label"]["sim"])
            feats.append([c["feats"].get(n, 0.0) for n in names])
    y = np.array(labels)
    X = np.array(feats)
    Xn = (X - mu) / sd
    stumps = []
    bias = float(np.median(y))
    pred = np.full(len(y), bias)
    for _ in range(n_stumps):
        best = None
        for j in range(Xn.shape[1]):
            col = Xn[:, j]
            for q in (0.25, 0.5, 0.75):
                thr = float(np.quantile(col, q))
                left = col >= thr
                for sign in (1, -1):
                    w = 0.5 * sign
                    trial = pred + w * (left.astype(float) * 2 - 1)
                    err = np.mean((trial - y) ** 2)
                    if best is None or err < best[0]:
                        best = (err, names[j], thr, w)
        if best is None:
            break
        _, feat, thr, w = best
        stumps.append({"feat": feat, "thr": thr, "w": w})
        col = Xn[:, names.index(feat)]
        pred += w * (np.where(col >= thr, 1.0, -1.0))
    return dict(kind="stumps", features=names, mean=mu.tolist(), std=sd.tolist(), stumps=stumps, bias=bias)


def fit_logistic_raw(prep, names, l2=1.0):
    """Logistic on designer sim >= median within each group (classification)."""
    D, Wt, mu, sd = pairwise_data(prep, names)
    if len(D) < 10:
        return None
    w = np.zeros(D.shape[1])
    for _ in range(30):
        z = D @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = -(D * (Wt * (1 - p))[:, None]).sum(0) + l2 * w
        Hm = (D * (Wt * p * (1 - p))[:, None]).T @ D + l2 * np.eye(D.shape[1])
        w -= np.linalg.solve(Hm, g)
    return dict(kind="linear-pairwise", features=names, weights=w.tolist(), mean=mu.tolist(), std=sd.tolist())


def evaluate(prep, model, blend):
    sims = []
    pix = []
    for row, cs in prep:
        c, lab = S.pick(row, cs, model, blend)
        sims.append(lab["sim"])
        pix.append(c["pix"] if c else 0.0)
    return sims, pix


def summary(prep, model, blend, label):
    base, _ = evaluate(prep, None, 0.0)
    new, pix = evaluate(prep, model, blend)
    oracle = [max([c["label"]["sim"] for c in cs if c["ok"]] or [row["fallback"]["sim"]]) for row, cs in prep]
    n = max(1, len(prep))
    hand_m = sum(base) / n
    rank_m = sum(new) / n
    print(f"{label:<32} n={len(prep):3d}  train-sim hand {hand_m:.4f} -> ranker {rank_m:.4f}  oracle {sum(oracle)/n:.4f}")
    return hand_m, rank_m, sum(pix) / n


def pairwise_acc(prep, model):
    ok = tot = 0
    for _, cs in prep:
        if model is None:
            order = sorted(cs, key=lambda c: -c["hand"])
        else:
            rs = [R.predict(c["feats"], model) for c in cs]
            order = sorted(cs, key=lambda c: -R.predict(c["feats"], model))
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                if abs(order[i]["label"]["sim"] - order[j]["label"]["sim"]) < MARGIN:
                    continue
                tot += 1
                if order[i]["label"]["sim"] >= order[j]["label"]["sim"]:
                    ok += 1
    return ok / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    rows = S.load("old")
    train_prep = prepared([r for r in rows if r["split"] == "tune"])
    held_prep = prepared([r for r in rows if r["split"] == "held"])
    all_prep = prepared(rows)
    if a.report:
        m = R.model()
        if not m:
            print("no weights")
            return
        summary(all_prep, m, m.get("blend", 0), "all")
        summary(held_prep, m, m.get("blend", 0), "held (unseen families)")
        return
    if not all_prep:
        print("run tools.ranker_data first")
        sys.exit(1)
    names = feature_names(all_prep)
    print(f"features={len(names)}  train groups={len(train_prep)}  held={len(held_prep)}")

    candidates_models = []
    for l2 in L2S:
        m = fit_linear(train_prep, names, l2)
        if m:
            candidates_models.append((f"linear-l2={l2}", m))
    m_st = fit_stumps(train_prep, names)
    candidates_models.append(("stumps", m_st))

    # reduced feature set for interaction-heavy linear model
    core = [n for n in names if n.startswith(("a:", "t:", "pos_iou", "keep_", "hidden", "log_s", "font_"))]
    m_core = fit_linear(train_prep, core, l2=3.0)
    if m_core:
        candidates_models.append(("linear-core", m_core))

    best = None
    for tag, m in candidates_models:
        pa = pairwise_acc(train_prep, m)
        print(f"  {tag}: pairwise train acc {pa:.3f}")
        for blend in BLENDS:
            hand_m, rank_m, pix_m = summary(train_prep, m, blend, f"    {tag} blend={blend}")
            _, base_pix = evaluate(train_prep, None, 0.0)
            _, new_pix = evaluate(train_prep, m, blend)
            if blend > 0 and np.mean(new_pix) > np.mean(base_pix) * (1 + PIX_TOL):
                continue
            held_hand, held_rank, _ = summary(held_prep, m, blend, f"      held")
            score = held_rank - 0.5 * max(0, held_hand - held_rank)
            if best is None or score > best[0]:
                best = (score, tag, blend, m, held_rank, held_hand, pa)

    if best is None:
        print("no model passed CV")
        sys.exit(1)
    _, tag, blend, m, held_rank, held_hand, pa = best
    if m["kind"] == "linear-pairwise":
        m = fit_linear(all_prep, m["features"], l2=3.0)
    else:
        m = fit_stumps(all_prep, m["features"])
    m["blend"] = blend
    m["margin"] = 10.0
    m["meta"] = dict(tag=tag, train_pairwise_acc=pa, held_sim=held_rank)
    OUT.write_text(json.dumps(m))
    sz = OUT.stat().st_size
    print(f"\nshipped {OUT} ({sz} bytes) tag={tag} blend={blend}")
    summary(all_prep, m, blend, "final all")
    summary(held_prep, m, blend, "final held")


if __name__ == "__main__":
    main()
