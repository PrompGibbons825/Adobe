"""Fit the kNN feature weights and k on held-out Crello families (box IoU with the designer).

    .venv/bin/python -m tools.learn_fit  -> engine/learn/model.npz (+ prints baselines)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.learn.features import FEATURES, NF  # noqa: E402
from engine.learn.knn import predict  # noqa: E402

TRAIN = Path("/data/agents/work/night/lane-learn/train.npz")


def boxes(X, P, keep):
    """Boxes in unit-area target canvas coordinates, from features X and (pred or truth) P."""
    sa, ta = X[:, 12], X[:, 13]
    W, H = np.exp(sa / 2), np.exp(-sa / 2)
    TW, TH = np.exp(ta / 2), np.exp(-ta / 2)
    w0, h0 = X[:, 4] * W, X[:, 5] * H
    side = np.sqrt(w0 * h0) * np.exp(P[:, 2])
    ar = (w0 / h0) * np.where(keep, 1.0, np.exp(P[:, 3]))
    w, h = side * np.sqrt(ar), side / np.sqrt(ar)
    cx, cy = P[:, 0] * TW, P[:, 1] * TH
    b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    bl = P[:, 5] > 0.5
    b[bl] = np.stack([np.zeros(bl.sum()), np.zeros(bl.sum()), TW[bl], TH[bl]], 1)
    # to canvas fractions
    b[:, [0, 2]] /= TW[:, None]
    b[:, [1, 3]] /= TH[:, None]
    return b


def iou(a, b):
    iw = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def absol(Xv, P):
    P = P.copy()
    P[:, 0] += Xv[:, 2]
    P[:, 1] += Xv[:, 3]
    return P


def score(Xv, Yv, P, keep_pred, mask=None):
    tb = boxes(Xv, absol(Xv, Yv), np.zeros(len(Xv), bool))
    pb = boxes(Xv, absol(Xv, P), keep_pred)
    v = iou(pb, tb)
    return float(v.mean() if mask is None else v[mask].mean())


def baselines(Xv):
    sa, ta = Xv[:, 12], Xv[:, 13]
    P_prop = np.zeros((len(Xv), 6), np.float32)
    P_prop[:, 5] = Xv[:, 1]
    # letterbox: uniform scale min(TW/W, TH/H) / s_geo, centred
    W, H = np.exp(sa / 2), np.exp(-sa / 2)
    TW, TH = np.exp(ta / 2), np.exp(-ta / 2)
    s = np.minimum(TW / W, TH / H)
    P_box = P_prop.copy()
    P_box[:, 2] = np.log(s)
    P_box[:, 0] = ((Xv[:, 2] - 0.5) * W * s + TW / 2) / TW - Xv[:, 2]
    P_box[:, 1] = ((Xv[:, 3] - 0.5) * H * s + TH / 2) / TH - Xv[:, 3]
    return P_prop, P_box


def main():
    z = np.load(TRAIN)
    X, Y, fam = z["X"], z["Y"], z["fam"]
    rng = np.random.default_rng(0)
    fams = np.unique(fam)
    val_f = set(rng.choice(fams, size=len(fams) // 5, replace=False).tolist())
    vm = np.array([f in val_f for f in fam])
    Y = Y.copy()
    Y[:, 0] -= X[:, 2]  # residuals over the proportional anchor
    Y[:, 1] -= X[:, 3]
    Xt, Yt = X[~vm], Y[~vm]
    vi = rng.choice(np.where(vm)[0], size=min(1500, vm.sum()), replace=False)
    Xv, Yv = X[vi], Y[vi]
    keep = Xv[:, 0] < 0.5
    txt = Xv[:, 0] > 0.5
    P_prop, P_box = baselines(Xv)
    print(json.dumps(dict(rows_train=len(Xt), rows_val=len(Xv),
                          base_proportional=round(score(Xv, Yv, P_prop, keep), 4),
                          base_letterbox=round(score(Xv, Yv, P_box, keep), 4))))

    w = np.ones(NF, np.float32)
    w[[12, 13]] = 0.0  # absolute aspects start off; d_ar carries the change
    w[14] = 4.0
    k = 20

    def ev(w, k):
        P = predict(Xv, Xt, Yt, w, k, chunk=256)
        return score(Xv, Yv, P, keep)

    best = ev(w, k)
    print("start", round(best, 4), flush=True)
    t0 = time.time()
    for rnd in range(2):
        improved = False
        for kk in (5, 10, 20, 40):
            if kk != k:
                s = ev(w, kk)
                if s > best + 1e-4:
                    best, k, improved = s, kk, True
        for f in range(NF):
            for mult in (0.0, 0.33, 3.0) if w[f] > 0 else (1.0,):
                w2 = w.copy()
                w2[f] = (w[f] * mult) if w[f] > 0 else mult
                s = ev(w2, k)
                if s > best + 1e-4:
                    best, w, improved = s, w2, True
            print(f"r{rnd} {FEATURES[f]:10s} w={w[f]:.3g} best={best:.4f} k={k} t={time.time()-t0:.0f}s", flush=True)
        if not improved:
            break
    P = predict(Xv, Xt, Yt, w, k, chunk=256)
    print(json.dumps(dict(knn=round(best, 4), knn_text=round(score(Xv, Yv, P, keep, txt), 4),
                          knn_other=round(score(Xv, Yv, P, keep, ~txt), 4),
                          prop_text=round(score(Xv, Yv, P_prop, keep, txt), 4),
                          prop_other=round(score(Xv, Yv, P_prop, keep, ~txt), 4),
                          k=k, w={FEATURES[i]: float(w[i]) for i in range(NF)})))
    out = ROOT / "engine" / "learn" / "model.npz"
    # ship every training row (train + val families) with the fitted weights
    np.savez_compressed(out, X=X.astype(np.float16), Y=Y.astype(np.float16), w=w, k=np.array(k))
    np.savez_compressed(out.with_name("pairs.npz"), pair=z["pair"], fam=z["fam"])
    print("wrote", out, out.stat().st_size)


if __name__ == "__main__":
    main()
