"""Fit the proxy judge's weights to a VLM's pairwise labels (dev tool).

    .venv/bin/python -m tools.judge.fit features --corpus <dir> --assets <dir> [--ref <features.jsonl>]
        # re-measure every labelled pair's two candidates with this tree's terms
        # -> tools/judge/data/features.jsonl (only pairs still valid, see below)
    .venv/bin/python -m tools.judge.fit report [--write]
        # agreement before/after, Kendall tau, term importance;
        # --write stores the fitted weights in tools/judge/data/weights.json

Bradley-Terry with the proxy as the strength: P(a beats b) = sigmoid(w . (t_b - t_a)),
where t are the source-relative penalty terms (tools/judge/terms.py) and w >= 0.
Fitted by projected gradient descent with an L2 penalty; lambda chosen by
cross-validation that holds out whole source designs (never just formats).

A pair's label is the share of its asks (both orders) that picked a. A pair
is "confident" when both asks agree, "strong" when they also both have
confidence >= 2; agreement is reported on those and on every ask.

The corpus is ws/judge-calibration's `output/judge/` (cands/, tasks/), whose
candidates were rendered and judged on macOS (CoreText glyph advances). A pair
is still valid on this tree when its candidates, source and assets all exist,
and, given `--ref` (the branch's own term vectors), when the terms re-measured
here match the ones measured for the judged image on both sides (within
`--tol`, max |difference| over the terms). A pair whose line breaks moved
(FreeType vs CoreText) describes a different image than the judge saw and is
dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
WORK = ROOT / "output" / "judge"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.judge.terms import HAND_WEIGHTS, TERMS  # noqa: E402

CRITERIA = ["overall", "collisions", "readable", "brand", "intentional"]
# A pair stays valid when no term re-measured on this tree moved more than TOL from
# the term measured for the image the judge saw (glyph rasterisation differs
# between the macOS corpus and Linux; a moved line break changes the image).
TOL = 0.25
LAM_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]


def _jsonl(p: Path) -> List[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


# ----------------------------------------------------------------------------- features


def setup_kit(extra_assets: List[Path], kit: Path = WORK / "kit") -> Path:
    """One brand-kit folder holding the Solstice assets plus any extra asset dirs
    (the Crello cases'), so the renderer finds every asset the corpus uses."""
    (kit / "assets").mkdir(parents=True, exist_ok=True)
    for d in [ROOT / "brand_kit" / "assets", *map(Path, extra_assets)]:
        if not d.is_dir():
            continue
        for f in d.iterdir():
            link = kit / "assets" / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
    if not (kit / "fonts").exists():
        (kit / "fonts").symlink_to((ROOT / "brand_kit" / "fonts").resolve())
    os.environ["CHAMELEON_BRAND_KIT"] = str(kit)
    return kit


def missing_assets(design: dict, kit: Path) -> List[str]:
    return [e.get("asset_id") for e in design.get("elements", [])
            if e.get("type") == "image" and e.get("asset_id") and not (kit / "assets" / e["asset_id"]).is_file()]


def max_diff(t1: Dict[str, float], t2: Dict[str, float]) -> float:
    return max(abs(float(t1.get(k, 0.0)) - float(t2.get(k, 0.0))) for k in TERMS)


def _measure_one(args):
    cid, path, kit = args
    import contextlib
    import io

    os.environ["CHAMELEON_BRAND_KIT"] = kit
    from tools.judge import terms as T

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            return cid, T.measure(json.loads(Path(path).read_text())), None
    except Exception as ex:  # noqa: BLE001
        return cid, None, f"{type(ex).__name__}: {ex}"


def cmd_features(a) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from tools.judge import terms as T

    corpus = Path(a.corpus)
    kit = setup_kit(a.assets)
    pairs = _jsonl(DATA / "pairs.jsonl")
    labelled = {l["pair"] for l in _jsonl(DATA / "labels.jsonl")}
    ref = {r["pair"]: r for r in _jsonl(Path(a.ref))} if a.ref else {}
    drop: Counter = Counter()
    todo, need = [], {}
    for p in pairs:
        if p["kind"] == "check":
            drop["check pair (never for fitting)"] += 1
            continue
        if p["pair"] not in labelled:
            drop["no labels"] += 1
            continue
        paths = {"src": corpus / "tasks" / f"{p['source']}.src.json",
                 "a": corpus / "cands" / f"{p['a']}.json", "b": corpus / "cands" / f"{p['b']}.json"}
        if not all(x.exists() for x in paths.values()):
            drop["candidate or source missing"] += 1
            continue
        gone = set(missing_assets(json.loads(paths["src"].read_text()), kit))  # missing in the source too:
        if any(set(missing_assets(json.loads(paths[s].read_text()), kit)) - gone for s in ("a", "b")):  # a placeholder
            drop["asset missing"] += 1                                     # both then and now
            continue
        todo.append(p)
        need[f"SRC_{p['source']}"] = paths["src"]
        need[p["a"]] = paths["a"]
        need[p["b"]] = paths["b"]
    print(f"{len(todo)} pairs to measure, {len(need)} designs; dropped {dict(drop)}", flush=True)
    reps: Dict[str, Optional[dict]] = {}
    jobs = [(cid, str(path), str(kit)) for cid, path in sorted(need.items())]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, (cid, rep, err) in enumerate(ex.map(_measure_one, jobs, chunksize=4)):
            reps[cid] = rep
            if err:
                print("measure failed", cid, err, flush=True)
            if k % 50 == 0:
                print(k, "/", len(jobs), flush=True)
    rows, diffs = [], []
    for p in todo:
        src, ra, rb = reps.get(f"SRC_{p['source']}"), reps.get(p["a"]), reps.get(p["b"])
        if src is None or ra is None or rb is None:
            drop["measure failed"] += 1
            continue
        rec = {"pair": p["pair"]}
        for side, rep in (("a", ra), ("b", rb)):
            rec[side] = {"terms": T.terms(src, rep)}
        old = ref.get(p["pair"])
        if old is not None:
            d = max(max_diff(rec["a"]["terms"], old["a"]["terms"]), max_diff(rec["b"]["terms"], old["b"]["terms"]))
            diffs.append(d)
            rec["ref_diff"] = round(d, 4)
            for side in ("a", "b"):  # the old proxy and the engine score of the judged candidates
                rec[side].update({k: old[side].get(k) for k in ("old", "engine", "plan")})
        elif ref:
            drop["no reference terms"] += 1
            continue
        rows.append(rec)
    with (DATA / "features.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if diffs:
        q = np.percentile(diffs, [50, 90, 99, 100])
        print(f"re-measured vs reference terms, max |diff| per pair: median {q[0]:.4f}  p90 {q[1]:.4f}  "
              f"p99 {q[2]:.4f}  max {q[3]:.4f}  exact {sum(d < 1e-3 for d in diffs)}/{len(diffs)}")
    print(f"{len(rows)} pairs -> {DATA / 'features.jsonl'}; dropped {dict(drop)}")
    if diffs:
        print("pairs within the default validity tolerance", TOL, ":", sum(d <= TOL for d in diffs))


# ----------------------------------------------------------------------------- data


def dataset(criterion: str = "overall", features: Optional[Path] = None, tol: Optional[float] = None) -> List[dict]:
    """Labelled pairs with term differences x = t_b - t_a. Pairs whose re-measured
    terms moved more than `tol` from the judged image's (ref_diff) are left out."""
    tol = TOL if tol is None else tol
    pairs = {p["pair"]: p for p in _jsonl(DATA / "pairs.jsonl")}
    feats = {r["pair"]: r for r in _jsonl(features or DATA / "features.jsonl")}
    votes: Dict[str, List[int]] = defaultdict(list)
    confs: Dict[str, List[int]] = defaultdict(list)
    for l in _jsonl(DATA / "labels.jsonl"):
        p = pairs[l["pair"]]
        votes[l["pair"]].append(1 if l[criterion] == p["a"] else 0)
        confs[l["pair"]].append(int(l.get("confidence", 1)))
    rows = []
    for pid, v in votes.items():
        if pid not in feats or feats[pid].get("ref_diff", 0.0) > tol:
            continue
        p, ft = pairs[pid], feats[pid]
        ta = np.array([ft["a"]["terms"][k] for k in TERMS])
        tb = np.array([ft["b"]["terms"][k] for k in TERMS])
        rows.append({"pair": pid, "case": p["case"], "source": p["source"], "kind": p["kind"], "view": p["view"],
                     "round": p.get("round"), "a": p["a"], "b": p["b"], "x": tb - ta, "y": float(np.mean(v)),
                     "n": len(v), "conf": len(v) == 2 and v[0] == v[1], "votes": v,
                     "strong": len(v) == 2 and v[0] == v[1] and min(confs[pid]) >= 2,
                     "old": (ft["a"].get("old"), ft["b"].get("old")),
                     "eng": (ft["a"].get("engine"), ft["b"].get("engine"))})
    rows.sort(key=lambda r: r["pair"])
    return rows


# ----------------------------------------------------------------------------- model


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit(X: np.ndarray, y: np.ndarray, n: np.ndarray, lam: float, iters: int = 4000,
        prior: Optional[np.ndarray] = None) -> np.ndarray:
    """Non-negative L2 logistic regression without intercept (projected gradient).
    With `prior` set, the penalty pulls towards c * prior, c fitted on the data
    first (the prior's overall scale), so the data adjusts the prior instead of
    rebuilding it from nothing."""
    w0 = np.zeros(X.shape[1])
    if prior is not None:
        z = X @ prior
        c = 0.0
        for _ in range(200):  # 1-D logistic fit of the scale
            g = float(z @ ((sigmoid(c * z) - y) * n)) / n.sum()
            h = float((z * z) @ (sigmoid(c * z) * (1 - sigmoid(c * z)) * n)) / n.sum() + 1e-6
            c = max(0.0, c - g / h)
        w0 = c * prior
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


def cv_predict(rows, lam: float, k: int = 5, seeds=(0, 1, 2), prior: Optional[np.ndarray] = None) -> np.ndarray:
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
            if not te.any():
                continue
            w = fit(X[tr], y[tr], n[tr], lam, prior=prior)
            pred[te] += sigmoid(X[te] @ w)
    return pred / len(seeds)


def logloss(rows, pr) -> float:
    return float(-np.mean([r["y"] * math.log(max(p, 1e-9)) + (1 - r["y"]) * math.log(max(1 - p, 1e-9))
                           for r, p in zip(rows, pr)]))


def choose_lambda(rows, prior=None, grid=LAM_GRID):
    best = None
    for lam in grid:
        pr = cv_predict(rows, lam, prior=prior)
        ll = logloss(rows, pr)
        if best is None or ll < best[1]:
            best = (lam, ll, pr)
    return best


# ----------------------------------------------------------------------------- metrics


def agreement(rows, margin: np.ndarray) -> dict:
    """margin > 0 means the model prefers a. Ties count half."""
    def hit(m, y):
        return 0.5 if abs(m) < 1e-12 else float((m > 0) == (y > 0.5))
    conf = [hit(m, r["y"]) for r, m in zip(rows, margin) if r["conf"]]
    strong = [hit(m, r["y"]) for r, m in zip(rows, margin) if r.get("strong")]
    asks = [0.5 if abs(m) < 1e-12 else float((m > 0) == (v == 1)) for r, m in zip(rows, margin) for v in r["votes"]]
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
    return {"conf": mean(conf), "n_conf": len(conf), "strong": mean(strong), "n_strong": len(strong), "asks": mean(asks)}


def kendall_within_case(rows, margin: np.ndarray):
    """Rank candidates inside each round-robin case by the judge (Copeland wins)
    and by the model's own pairwise calls, then Kendall tau-b. Only cases where
    every pair among >= 3 candidates was labelled."""
    by_case = defaultdict(list)
    for r, m in zip(rows, margin):
        by_case[r["case"]].append((r, m))
    taus = []
    for items in by_case.values():
        rr = {c for r, _ in items if r["kind"] == "rr" for c in (r["a"], r["b"])}
        if not rr:
            continue
        items = [(r, m) for r, m in items if r["a"] in rr and r["b"] in rr]
        cands = sorted(rr)
        if len(cands) < 3 or len(items) < len(cands) * (len(cands) - 1) // 2:
            continue
        judge, model = Counter(), Counter()
        for r, m in items:
            judge[r["a"]] += r["y"]
            judge[r["b"]] += 1 - r["y"]
            model[r["a"]] += 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
            model[r["b"]] += 1.0 if m < 0 else (0.5 if m == 0 else 0.0)
        taus.append(tau_b([judge[c] for c in cands], [model[c] for c in cands]))
    return (float(np.mean(taus)) if taus else float("nan"), len(taus))


def tau_b(x, y) -> float:
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
        if how in ("old", "engine"):
            a, b = r["old"] if how == "old" else r["eng"]
            out.append(0.0 if a is None or b is None else a - b)
        else:
            out.append(float(r["x"] @ w))
    return np.array(out)


# ----------------------------------------------------------------------------- report


def cmd_report(a) -> None:
    prior = np.array([HAND_WEIGHTS[k] for k in TERMS]) if a.prior == "hand" else None
    rows = dataset("overall", tol=a.tol)
    if not rows:
        sys.exit("no labelled pairs with features (run `features` first)")
    hand = np.array([HAND_WEIGHTS[k] for k in TERMS])
    print(f"pairs labelled: {len(rows)}  asks: {sum(r['n'] for r in rows)}  "
          f"both orders agree (overall): {np.mean([r['conf'] for r in rows if r['n'] == 2]):.3f}")
    for c in CRITERIA[1:]:
        rc = dataset(c, tol=a.tol)
        print(f"  both orders agree on {c:12s} {np.mean([r['conf'] for r in rc if r['n'] == 2]):.3f}")

    print("\nlambda selection (out-of-fold, designs held out)")
    best = None
    for lam in LAM_GRID:
        pr = cv_predict(rows, lam, prior=prior)
        ll = logloss(rows, pr)
        ag = agreement(rows, pr - 0.5)
        print(f"  lambda={lam:<7} logloss={ll:.4f}  agree(confident)={ag['conf']:.3f}  agree(asks)={ag['asks']:.3f}")
        if best is None or ll < best[1]:
            best = (lam, ll, pr)
    lam, _, pr = best
    X = np.stack([r["x"] for r in rows])
    w_all = fit(X, np.array([r["y"] for r in rows]), np.array([r["n"] for r in rows], float), lam, prior=prior)

    def line(name, m, sub=None):
        rr = rows if sub is None else [rows[i] for i in sub]
        mm = m if sub is None else m[sub]
        ag = agreement(rr, mm)
        kt = kendall_within_case(rr, mm)
        print(f"  {name:40s} {ag['conf']:.3f} (n={ag['n_conf']:3d})  {ag['strong']:.3f} (n={ag['n_strong']:3d})  "
              f"{ag['asks']:.3f}    {2 * ag['conf'] - 1:+.3f}    {kt[0]:+.3f} (cases={kt[1]})")

    print(f"\nagreement with the VLM judge (overall), prior={a.prior}, lambda={lam}")
    print("  model                                    confident pairs  strong pairs     all asks  tau(pairs)  tau(within case)")
    if any(r["old"][0] is not None for r in rows):
        line("old proxy (tools/proxy.py, hand weights)", margins(rows, "old"))
    line("source-relative terms, hand weights", margins(rows, "hand", hand))
    line("source-relative, calibrated (CV)", pr - 0.5)
    line("source-relative, calibrated (in-sample)", margins(rows, "fit", w_all))
    eng_idx = [i for i, r in enumerate(rows) if r["eng"][0] is not None and r["eng"][1] is not None]
    if eng_idx:
        print(f"  -- engine top-k pairs only (n={len(eng_idx)})")
        line("engine's own score", margins(rows, "engine"), eng_idx)
        line("hand weights", margins(rows, "hand", hand), eng_idx)
        line("calibrated (CV)", pr - 0.5, eng_idx)

    rounds = sorted({r["round"] for r in rows if r["round"] is not None})
    if len(rounds) > 1:  # prospective: fit on round 1 only, score the later rounds
        tr = [r for r in rows if r["round"] == rounds[0]]
        te = [r for r in rows if r["round"] != rounds[0]]
        lam1 = choose_lambda(tr, prior=prior)[0]
        w1 = fit(np.stack([r["x"] for r in tr]), np.array([r["y"] for r in tr]),
                 np.array([r["n"] for r in tr], float), lam1, prior=prior)
        ag = agreement(te, margins(te, "fit", w1))
        print(f"\nprospective: fit on round {rounds[0]} ({len(tr)} pairs, lambda={lam1}), "
              f"score the rest ({len(te)}): confident {ag['conf']:.3f} (n={ag['n_conf']}), asks {ag['asks']:.3f}")

    base = agreement(rows, cv_predict(rows, lam, seeds=(0,), prior=prior) - 0.5)["conf"]
    print("\nterms: fitted weight, pairs where the term differs, mean |diff|, drop-term CV agreement change")
    imp = []
    for j, k in enumerate(TERMS):
        nz = int((np.abs(X[:, j]) > 1e-9).sum())
        keep = [i for i in range(len(TERMS)) if i != j]
        sub = [dict(r, x=r["x"][keep]) for r in rows]
        pj = cv_predict(sub, lam, seeds=(0,), prior=None if prior is None else prior[keep])
        d = agreement(rows, pj - 0.5)["conf"] - base
        imp.append((w_all[j] * (np.abs(X[:, j]).mean()), k, w_all[j], nz, float(np.abs(X[:, j]).mean()), d))
    for _, k, wk, nz, md, d in sorted(imp, reverse=True):
        print(f"  {k:18s} w={wk:7.3f}  hand={HAND_WEIGHTS[k]:4.1f}  differs in {nz:3d} pairs  mean|diff|={md:.3f}  drop: {d:+.3f}")

    per_crit = {}
    print("\nper criterion (in-sample weights, top terms)")
    for c in CRITERIA[1:]:
        rc = dataset(c, tol=a.tol)
        wc = fit(np.stack([r["x"] for r in rc]), np.array([r["y"] for r in rc]),
                 np.array([r["n"] for r in rc], float), lam, prior=prior)
        top = sorted(zip(wc, TERMS), reverse=True)[:5]
        per_crit[c] = {k: round(float(v), 3) for v, k in top if v > 0}
        print(f"  {c:12s} " + ", ".join(f"{k} {v:.2f}" for v, k in top if v > 0))

    ag = agreement(rows, pr - 0.5)
    out = {"weights": {k: round(float(v), 4) for k, v in zip(TERMS, w_all)}, "lambda": lam, "prior": a.prior,
           "pairs": len(rows), "tol": a.tol, "asks": int(sum(r["n"] for r in rows)),
           "cv_agreement_confident": round(ag["conf"], 4), "cv_agreement_strong": round(ag["strong"], 4),
           "cv_agreement_asks": round(ag["asks"], 4), "per_criterion_top": per_crit,
           "source": "ws/judge-calibration labels (10-06), terms re-measured on this tree (tools.judge.fit features)",
           "note": "P(a beats b) = sigmoid(sum_k w_k * (t_b,k - t_a,k)); terms from tools/judge/terms.py"}
    if a.write:
        (DATA / "weights.json").write_text(json.dumps(out, indent=1) + "\n")
        print("\nwrote", DATA / "weights.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("features")
    p.add_argument("--corpus", required=True, help="a judge corpus folder with cands/ and tasks/")
    p.add_argument("--assets", action="append", default=[], help="extra asset folders (repeatable)")
    p.add_argument("--ref", help="the term vectors measured for the judged images (branch features.jsonl)")
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("report")
    p.add_argument("--write", action="store_true")
    p.add_argument("--prior", default="zero", choices=["zero", "hand"])
    p.add_argument("--tol", type=float, default=TOL, help="max re-measure drift (ref_diff) for a pair to count")
    a = ap.parse_args()
    {"features": cmd_features, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
