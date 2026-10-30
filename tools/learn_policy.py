"""Fit the designers' median policy on the Crello training table (held-out dev families excluded):
per signed aspect-change bin, the median of each designer move (centre shift, box scale over
sqrt-area, text box aspect change, font scale over sqrt-area). -> engine/learn/policy.json

    .venv/bin/python -m tools.learn_policy
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
z = np.load("/data/agents/work/night/lane-learn/train.npz")
X, Y = z["X"], z["Y"].copy()
Y[:, 0] -= X[:, 2]
Y[:, 1] -= X[:, 3]
EDGES = [-9, -2.0, -1.4, -0.9, -0.5, -0.2, 0.2, 0.5, 0.9, 1.4, 2.0, 9]
d = X[:, 14]
txt = X[:, 0] > 0.5
fg = X[:, 1] < 0.5
out = dict(edges=EDGES, centers=[], n=[], fs=[], text_scale=[], text_ar=[], other_scale=[], bleed_keep=[])
for a, b in zip(EDGES[:-1], EDGES[1:]):
    m = (d >= a) & (d < b)
    out["centers"].append(float(np.median(d[m])) if m.any() else (a + b) / 2)
    out["n"].append(int(m.sum()))
    med = lambda mask, j: float(np.median(Y[mask, j])) if mask.any() else 0.0  # noqa: E731
    out["fs"].append(med(m & txt, 4))
    out["text_scale"].append(med(m & txt, 2))
    out["text_ar"].append(med(m & txt, 3))
    out["other_scale"].append(med(m & ~txt & fg, 2))
    sb = m & (X[:, 1] > 0.5)
    out["bleed_keep"].append(float(Y[sb, 5].mean()) if sb.any() else 1.0)
(ROOT / "engine" / "learn" / "policy.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
