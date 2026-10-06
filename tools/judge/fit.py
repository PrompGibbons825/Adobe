"""Fit the proxy judge's weights to the VLM's pairwise labels (dev tool).

    .venv/bin/python -m tools.judge.fit features   # term vectors per pair -> tools/judge/data/features.jsonl
    .venv/bin/python -m tools.judge.fit report     # agreement before/after, Kendall tau, term importance,
                                                   # and the fitted weights -> tools/judge/data/weights.json

Bradley-Terry with the proxy as the strength: P(a beats b) = sigmoid(w . (t_b - t_a)),
where t are the source-relative penalty terms (tools/judge/terms.py) and w >= 0.
Fitted by projected gradient descent with an L2 penalty; lambda chosen by
cross-validation that holds out whole source designs (never just formats).

A pair's label is the share of its two asks (both orders) that picked a. A pair
is "confident" when both asks agree; agreement is reported on those, and on
every ask.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
WORK = ROOT / "output" / "judge"
sys.path.insert(0, str(ROOT))

from tools.judge.terms import HAND_WEIGHTS, TERMS  # noqa: E402

CRITERIA = ["overall", "collisions", "readable", "brand", "intentional"]


def _jsonl(p: Path) -> List[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def cmd_features(_a) -> None:
    from tools.judge import corpus

    corpus.setup_kit()
    cases = json.loads((WORK / "cases.json").read_text())
    have = {r["pair"] for r in _jsonl(DATA / "features.jsonl")}
    n = 0
    with (DATA / "features.jsonl").open("a") as f:
        for p in corpus.load_pairs():
            if p["pair"] in have or not (WORK / "reports" / f"{p['a']}.json").exists():
                continue
            ft = corpus.features(p)
            rec = {"pair": p["pair"]}
            for side in ("a", "b"):
                case, kind = p[side].split("#")
                c = cases[case]["cands"].get(kind, {})
                rec[side] = {"terms": ft[side]["terms"], "old": ft[side]["old"],
                             "engine": c.get("score") if kind.startswith("e") else None,
                             "plan": c.get("plan")}
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(n, "new feature rows")


# ----------------------------------------------------------------------------- data


def dataset(criterion: str = "overall"):
    pairs = {p["pair"]: p for p in _jsonl(DATA / "pairs.jsonl")}
    feats = {r["pair"]: r for r in _jsonl(DATA / "features.jsonl")}
    votes: Dict[str, List[int]] = defaultdict(list)
    confs: Dict[str, List[int]] = defaultdict(list)
    for l in _jsonl(DATA / "labels.jsonl"):
        p = pairs[l["pair"]]
        votes[l["pair"]].append(1 if l[criterion] == p["a"] else 0)
        confs[l["pair"]].append(int(l.get("confidence", 1)))
    rows = []
    for pid, v in votes.items():
        if pid not in feats:
            continue
        p, ft = pairs[pid], feats[pid]
        ta = np.array([ft["a"]["terms"][k] for k in TERMS])
        tb = np.array([ft["b"]["terms"][k] for k in TERMS])
        rows.append({"pair": pid, "case": p["case"], "source": p["source"], "kind": p["kind"], "view": p["view"],
                     "a": p["a"], "b": p["b"], "x": tb - ta, "y": float(np.mean(v)), "n": len(v),
                     "conf": len(v) == 2 and v[0] == v[1], "votes": v,
                     "strong": len(v) == 2 and v[0] == v[1] and min(confs[pid]) >= 2,
                     "old": (ft["a"]["old"], ft["b"]["old"]), "eng": (ft["a"]["engine"], ft["b"]["engine"])})
    return rows


# ----------------------------------------------------------------------------- model


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


PRIOR = None  # None: shrink towards 0; else a weight vector to shrink towards (hand weights)


def fit(X: np.ndarray, y: np.ndarray, n: np.ndarray, lam: float, iters: int = 4000) -> np.ndarray:
    """Non-negative L2 logistic regression without intercept (projected gradient).
    With PRIOR set, the penalty pulls towards c * PRIOR, c fitted on the data
    first (the hand weights' overall scale), so the data adjusts the hand
    weights instead of rebuilding them from nothing."""
    w0 = np.zeros(X.shape[1])
    if PRIOR is not None:
        z = X @ PRIOR
        c = 0.0
        for _ in range(200):  # 1-D logistic fit of the scale
            g = float(z @ ((sigmoid(c * z) - y) * n)) / n.sum()
            h = float((z * z) @ (sigmoid(c * z) * (1 - sigmoid(c * z)) * n)) / n.sum() + 1e-6
            c = max(0.0, c - g / h)
        w0 = c * PRIOR
    w = w0.copy()
    L = 0.25 * np.linalg.norm(X * np.sqrt(n)[:, None], 2) ** 2 / max(1.0, n.sum()) + 2 * lam
    eta = 1.0 / max(L, 1e-6)
    for _ in range(iters):
        g = X.T @ ((sigmoid(X @ w) - y) * n) / n.sum() + 2 * lam * (w - w0)
        w = np.maximum(0.0, w - eta * g)
    return w


def groups_of(rows, k: int = 5, seed: int = 0) -> Dict[str, int]:
    srcs = sorted({r["source"] for r in rows})
    random.Random(seed).shuffle(srcs)
    return {s: i % k for i, s in enumerate(srcs)}


def cv_predict(rows, lam: float, k: int = 5, seeds=(0, 1, 2)) -> np.ndarray:
    """Out-of-fold P(a wins), source designs held out, averaged over fold seeds."""
    X = np.stack([r["x"] for r in rows])
    y = np.array([r["y"] for r in rows])
    n = np.array([r["n"] for r in rows], float)
    pred = np.zeros(len(rows))
    for seed in seeds:
        g = groups_of(rows, k, seed)
        fold = np.array([g[r["source"]] for r in rows])
        for f in range(k):
            tr, te = fold != f, fold == f
            w = fit(X[tr], y[tr], n[tr], lam)
            pred[te] += sigmoid(X[te] @ w)
    return pred / len(seeds)


# ----------------------------------------------------------------------------- metrics


def agreement(rows, margin: np.ndarray):
    """margin > 0 means the model prefers a. Ties count half."""
    def hit(m, y):
        return 0.5 if abs(m) < 1e-12 else float((m > 0) == (y > 0.5))
    conf = [hit(m, r["y"]) for r, m in zip(rows, margin) if r["conf"]]
    strong = [hit(m, r["y"]) for r, m in zip(rows, margin) if r.get("strong")]
    agreement.strong = (float(np.mean(strong)) if strong else float("nan"), len(strong))
    asks = []
    for r, m in zip(rows, margin):
        for v in r["votes"]:
            asks.append(0.5 if abs(m) < 1e-12 else float((m > 0) == (v == 1)))
    return (float(np.mean(conf)) if conf else float("nan"), len(conf), float(np.mean(asks)) if asks else float("nan"))


def kendall_within_case(rows, margin: np.ndarray):
    """Rank candidates inside each case by the judge (Copeland wins) and by the
    model (Copeland wins of the model's own pairwise calls), then Kendall tau-b.
    Only cases where every pair among >= 3 candidates was labelled."""
    by_case = defaultdict(list)
    for r, m in zip(rows, margin):
        by_case[r["case"]].append((r, m))
    taus = []
    for case, items in by_case.items():
        cands = sorted({c for r, _ in items for c in (r["a"], r["b"])})
        if len(cands) < 3 or len(items) < len(cands) * (len(cands) - 1) // 2:
            continue
        judge = Counter()
        model = Counter()
        for r, m in items:
            judge[r["a"]] += r["y"]
            judge[r["b"]] += 1 - r["y"]
            model[r["a"]] += 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
            model[r["b"]] += 1.0 if m < 0 else (0.5 if m == 0 else 0.0)
        taus.append(_tau_b([judge[c] for c in cands], [model[c] for c in cands]))
    return (float(np.mean(taus)) if taus else float("nan"), len(taus))


def _tau_b(x, y) -> float:
    c = d = tx = ty = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            sx, sy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if sx == 0 and sy == 0:
                continue
            if sx == 0:
                tx += 1
            elif sy == 0:
                ty += 1
            elif sx == sy:
                c += 1
            else:
                d += 1
    den = math.sqrt((c + d + tx) * (c + d + ty))
    return (c - d) / den if den else 0.0


def margins(rows, how: str, w=None) -> np.ndarray:
    out = []
    for r in rows:
        if how == "old":
            a, b = r["old"]
            out.append(0.0 if a is None or b is None else a - b)
        elif how == "engine":
            a, b = r["eng"]
            out.append(0.0 if a is None or b is None else a - b)
        else:
            out.append(float(r["x"] @ w))
    return np.array(out)


# ----------------------------------------------------------------------------- report


def cmd_report(a) -> None:
    global PRIOR
    if a.prior == "hand":
        PRIOR = np.array([HAND_WEIGHTS[k] for k in TERMS])
    rows = dataset("overall")
    if not rows:
        sys.exit("no labelled pairs with features")
    lam_grid = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
    hand = np.array([HAND_WEIGHTS[k] for k in TERMS])
    print(f"pairs labelled: {len(rows)}  asks: {sum(r['n'] for r in rows)}  "
          f"both orders agree (overall): {np.mean([r['conf'] for r in rows if r['n'] == 2]):.3f}")
    labels = _jsonl(DATA / "labels.jsonl")
    shown_a = np.mean([l["overall"] == l["shown_A"] for l in labels])
    print(f"position bias: first-shown version picked {shown_a:.3f} of the time")
    for c in CRITERIA[1:]:
        rc = dataset(c)
        print(f"  both orders agree on {c:12s} {np.mean([r['conf'] for r in rc if r['n'] == 2]):.3f}")

    best = None
    print("\nlambda selection (out-of-fold, designs held out)")
    for lam in lam_grid:
        pr = cv_predict(rows, lam)
        ll = -np.mean([r["y"] * math.log(max(p, 1e-9)) + (1 - r["y"]) * math.log(max(1 - p, 1e-9)) for r, p in zip(rows, pr)])
        ag = agreement(rows, pr - 0.5)
        print(f"  lambda={lam:<7} logloss={ll:.4f}  agree(confident)={ag[0]:.3f}  agree(asks)={ag[2]:.3f}")
        if best is None or ll < best[1]:
            best = (lam, ll, pr)
    lam, _, pr = best

    w_all = fit(np.stack([r["x"] for r in rows]), np.array([r["y"] for r in rows]),
                np.array([r["n"] for r in rows], float), lam)
    eng_rows = [r for r in rows if r["eng"][0] is not None and r["eng"][1] is not None]
    eng_idx = [i for i, r in enumerate(rows) if r in eng_rows]

    def line(name, m, sub=None):
        rr = rows if sub is None else [rows[i] for i in sub]
        mm = m if sub is None else m[sub]
        ag = agreement(rr, mm)
        st = agreement.strong
        kt = kendall_within_case(rr, mm)
        print(f"  {name:40s} {ag[0]:.3f} (n={ag[1]:3d})  {st[0]:.3f} (n={st[1]:3d})  {ag[2]:.3f}    "
              f"{2 * ag[0] - 1:+.3f}    {kt[0]:+.3f} (cases={kt[1]})")

    print(f"\nagreement with the VLM judge (overall), prior={a.prior}, lambda={lam}")
    print("  model                                    confident pairs  strong pairs     all asks  tau(pairs)  tau(within case)")
    line("old proxy (tools/proxy.py, hand weights)", margins(rows, "old"))
    line("source-relative terms, hand weights", margins(rows, "hand", hand))
    line("source-relative, calibrated (CV)", pr - 0.5)
    line("source-relative, calibrated (in-sample)", margins(rows, "fit", w_all))
    if eng_idx:
        print(f"  -- engine top-k pairs only (n={len(eng_idx)})")
        line("engine's own score", margins(rows, "engine"), eng_idx)
        line("old proxy", margins(rows, "old"), eng_idx)
        line("hand weights", margins(rows, "hand", hand), eng_idx)
        line("calibrated (CV)", pr - 0.5, eng_idx)

    print("\nby pair kind: calibrated-CV / hand / old agreement on confident pairs")
    kinds = sorted({r["kind"] for r in rows})
    for kd in kinds:
        idx = [i for i, r in enumerate(rows) if r["kind"] == kd]
        sub = [rows[i] for i in idx]
        a_wins = np.mean([r["y"] for r in sub])
        print(f"  {kd:18s} n={len(idx):3d}  judge prefers a {a_wins:.2f}   "
              f"{agreement(sub, (pr - 0.5)[idx])[0]:.3f} / {agreement(sub, margins(sub, 'hand', hand))[0]:.3f} / "
              f"{agreement(sub, margins(sub, 'old'))[0]:.3f}")

    # which terms matter
    X = np.stack([r["x"] for r in rows])
    base = agreement(rows, cv_predict(rows, lam, seeds=(0,)) - 0.5)[0]
    print("\nterms: fitted weight, pairs where the term differs, mean |diff|, drop-term CV agreement change")
    imp = []
    for j, k in enumerate(TERMS):
        nz = int((np.abs(X[:, j]) > 1e-9).sum())
        keep = [i for i in range(len(TERMS)) if i != j]
        sub = [dict(r, x=r["x"][keep]) for r in rows]
        saved = PRIOR
        PRIOR = None if saved is None else saved[keep]
        pj = cv_predict(sub, lam, seeds=(0,))
        PRIOR = saved
        d = agreement(rows, pj - 0.5)[0] - base
        imp.append((w_all[j] * (np.abs(X[:, j]).mean()), k, w_all[j], nz, float(np.abs(X[:, j]).mean()), d))
    for contrib, k, wk, nz, md, d in sorted(imp, reverse=True):
        print(f"  {k:18s} w={wk:7.3f}  hand={HAND_WEIGHTS[k]:4.1f}  differs in {nz:3d} pairs  mean|diff|={md:.3f}  drop: {d:+.3f}")

    per_crit = {}
    print("\nper criterion (in-sample weights, top terms)")
    for c in CRITERIA[1:]:
        rc = dataset(c)
        wc = fit(np.stack([r["x"] for r in rc]), np.array([r["y"] for r in rc]), np.array([r["n"] for r in rc], float), lam)
        top = sorted(zip(wc, TERMS), reverse=True)[:5]
        per_crit[c] = {k: round(float(v), 3) for v, k in top if v > 0}
        print(f"  {c:12s} " + ", ".join(f"{k} {v:.2f}" for v, k in top if v > 0))

    out = {"weights": {k: round(float(v), 4) for k, v in zip(TERMS, w_all)}, "lambda": lam, "prior": a.prior,
           "pairs": len(rows), "asks": int(sum(r["n"] for r in rows)),
           "cv_agreement_confident": round(agreement(rows, pr - 0.5)[0], 4), "per_criterion_top": per_crit,
           "note": "P(a beats b) = sigmoid(sum_k w_k * (t_b,k - t_a,k)); terms from tools/judge/terms.py"}
    if a.write:
        (DATA / "weights.json").write_text(json.dumps(out, indent=1) + "\n")
        print("\nwrote", DATA / "weights.json")


def cmd_disagree(a) -> None:
    """Pairs where the calibrated proxy and a confident judge disagree."""
    rows = dataset("overall")
    w = json.loads((DATA / "weights.json").read_text())
    wv = np.array([w["weights"][k] for k in TERMS])
    labels = defaultdict(list)
    for l in _jsonl(DATA / "labels.jsonl"):
        labels[l["pair"]].append(l["why"])
    for r in rows:
        m = float(r["x"] @ wv)
        if r["conf"] and (m > 0) != (r["y"] > 0.5) and abs(m) > 1e-9:
            diff = {k: round(float(v), 3) for k, v in zip(TERMS, r["x"]) if abs(v) > 1e-3}
            print(r["pair"], r["kind"], r["a"], "vs", r["b"].split("#")[1], "judge:", "a" if r["y"] > 0.5 else "b",
                  "proxy margin", round(m, 3), diff, "|", " / ".join(labels[r["pair"]])[:220])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("features")
    p = sub.add_parser("report")
    p.add_argument("--write", action="store_true")
    p.add_argument("--prior", default="zero", choices=["zero", "hand"])
    sub.add_parser("disagree")
    a = ap.parse_args()
    {"features": cmd_features, "report": cmd_report, "disagree": cmd_disagree}[a.cmd](a)


if __name__ == "__main__":
    main()
