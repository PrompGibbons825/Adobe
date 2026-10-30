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


MARGINS = [4.0, 8.0, 12.0, 20.0]
CV_BLENDS = [5.0, 10.0, 20.0, 40.0, 80.0]


def mean_sim(prep, model, blend, margin):
    if not prep:
        return 0.0
    return sum(S.pick(r, cs, model, blend, margin=margin)[1]["sim"] for r, cs in prep) / len(prep)


def cv_select(tune, names):
    """Pick (l2, blend, margin) by cross-validation over the tune folds only (fold 0 stays unseen)."""
    folds = sorted({fold_of(r["fam"]) for r, _ in tune})
    grid = [(l2, b, mg) for l2 in L2S for b in CV_BLENDS for mg in MARGINS]
    gain = {g: [] for g in grid}
    for k in folds:
        tr = [x for x in tune if fold_of(x[0]["fam"]) != k]
        va = [x for x in tune if fold_of(x[0]["fam"]) == k]
        base = mean_sim(va, None, 0.0, 0.0)
        for l2 in L2S:
            m = fit_linear(tr, names, l2)
            if m is None:
                continue
            for b in CV_BLENDS:
                for mg in MARGINS:
                    gain[(l2, b, mg)].append(mean_sim(va, m, b, mg) - base)
    scored = []
    for g, v in gain.items():
        if len(v) == len(folds):
            mu = float(np.mean(v))
            worst = float(min(v))
            scored.append((mu + 0.5 * min(0.0, worst), mu, worst, g))  # penalise a fold it hurts
    scored.sort(reverse=True)
    for row in scored[:8]:
        print(f"  cv l2={row[3][0]} blend={row[3][1]} margin={row[3][2]}: mean gain {row[1]:+.4f} worst fold {row[2]:+.4f}")
    return scored[0] if scored else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["wide"])
    ap.add_argument("--exclude", nargs="*", default=[], help="family lists to leave out (the dev bench sample)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    excl = set()
    for p in a.exclude:
        excl.update(x["fam"] if isinstance(x, dict) else x for x in json.loads(Path(p).read_text()))
    rows = [r for r in S.load(*a.sets) if r["fam"] not in excl and not r.get("error")]
    tune = prepared([r for r in rows if fold_of(r["fam"]) != 0])
    held = prepared([r for r in rows if fold_of(r["fam"]) == 0])
    if a.report:
        m = R.model()
        print("held (fold 0, unseen families): hand", round(mean_sim(held, None, 0, 0), 4),
              "-> ranker", round(mean_sim(held, m, m["blend"], m["margin"]), 4))
        return
    names = feature_names(tune + held)
    print(f"features={len(names)}  tune groups={len(tune)}  held groups={len(held)}  excluded={len(excl)}")
    best = cv_select(tune, names)
    if best is None or best[0] <= 0:
        print("no setting helps in cross-validation: ship nothing")
        sys.exit(1)
    _, _, _, (l2, blend, margin) = best
    m = fit_linear(tune, names, l2)
    oracle = sum(max([c["label"]["sim"] for c in cs if c["ok"]] or [r["fallback"]["sim"]]) for r, cs in held) / max(1, len(held))
    h0, h1 = mean_sim(held, None, 0, 0), mean_sim(held, m, blend, margin)
    print(f"HELD fold 0 ({len(held)} unseen families): hand {h0:.4f} -> ranker {h1:.4f}  (oracle {oracle:.4f})")
    wins = sum(1 for r, cs in held if S.pick(r, cs, m, blend, margin=margin)[1]["sim"] > S.pick(r, cs)[1]["sim"] + 1e-9)
    loss = sum(1 for r, cs in held if S.pick(r, cs, m, blend, margin=margin)[1]["sim"] < S.pick(r, cs)[1]["sim"] - 1e-9)
    print(f"  held changed picks: {wins} closer to the designer, {loss} further")
    m = fit_linear(tune + held, names, l2)  # ship the model fit on every family
    m.update(blend=blend, margin=margin, meta=dict(
        data="crello-v5 wide families (CDLA-Permissive-2.0)", groups=len(tune) + len(held), l2=l2,
        held_hand=h0, held_ranker=h1, held_wins=wins, held_losses=loss))
    OUT.write_text(json.dumps(m))
    print(f"shipped {OUT} ({OUT.stat().st_size} bytes) l2={l2} blend={blend} margin={margin}")


if __name__ == "__main__":
    main()
