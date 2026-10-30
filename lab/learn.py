"""Fit score weights on the Crello family index: choose the candidate whose boxes are closest (mean IoU, bench
`similarity`) to the designer's own resize. Train on families NOT in the tournament set.
  python -m lab.learn gather -o /data/agents/work/night/lane-compose/cand.pkl [--n 300]
  python -m lab.learn fit  cand.pkl"""
import argparse, json, pickle, random, sys, math
from pathlib import Path
from multiprocessing import Pool
sys.path.append("/data/agents/work/night/lane-compose/benchtree")
CASES = Path("/data/agents/work/crello-wide/cases")

def tasks():
    tour = set(json.load(open("/data/agents/work/crello-wide/tournament_set.json"))["tasks"])
    tfam = {t.rsplit("_", 1)[0] for t in tour}
    out = []
    for p in sorted(CASES.glob("*.json")):
        f = json.loads(p.read_text())
        if f["family"] in tfam: continue
        for k, ref in enumerate(f["targets"]):
            out.append((f"{f['family']}_{k}", f["source"], dict(ref["canvas"]), ref))
    return out

def one(t):
    from lab.sim import similarity
    from engine.compose import plan, realise
    name, src, tgt, ref = t
    try:
        ctx = plan(src, tgt)
        rows = []
        for L in ctx["cands"]:
            o = realise(ctx, L); o.pop("_layout", None)
            rows.append((L.name, dict(L.feat), similarity(o, ref)))
        return name, rows
    except Exception as ex:
        return name, []

def gather(a):
    ts = tasks(); random.Random(0).shuffle(ts); ts = ts[:a.n]
    with Pool(a.workers) as p:
        res = dict(p.map(one, ts, chunksize=2))
    pickle.dump(res, open(a.o, "wb")); print(len(res), "tasks", sum(len(v) for v in res.values()), "cands")

def evaluate(res, w, names):
    from engine import scoring as S
    tot = []
    for n in names:
        rows = res[n]
        if not rows: continue
        best = max(rows, key=lambda r: S.dot(r[1], w))
        tot.append(best[2] or 0.0)
    return sum(tot) / max(1, len(tot))

def fit(a):
    from engine import scoring as S
    res = pickle.load(open(a.pkl, "rb")); names = sorted(n for n, v in res.items() if v)
    random.Random(1).shuffle(names)
    half = len(names) // 2
    folds = [names[:half], names[half:]]
    oracle = sum(max((r[2] or 0) for r in res[n]) for n in names) / len(names)
    print("tasks", len(names), "oracle", round(oracle, 4), "default", round(evaluate(res, S.WEIGHTS, names), 4))
    def search(train, seed):
        rng = random.Random(seed); w = dict(S.WEIGHTS); best = evaluate(res, w, train)
        for it in range(400):
            k = rng.choice(list(w)); w2 = dict(w)
            w2[k] = max(0.0, w[k] * math.exp(rng.gauss(0, .6)) + rng.choice([0, 0, .1, -.1]))
            sc = evaluate(res, w2, train)
            if sc > best: best, w = sc, w2
        return w, best
    for i in (0, 1):
        w, tr = search(folds[i], i); te = evaluate(res, w, folds[1 - i])
        base = evaluate(res, S.WEIGHTS, folds[1 - i])
        print(f"fold{i}: train {tr:.4f} heldout {te:.4f} (default on heldout {base:.4f})", {k: round(v, 2) for k, v in w.items()})
    w, tr = search(names, 7); print("ALL", round(tr, 4), json.dumps({k: round(v, 3) for k, v in w.items()}))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="c")
    g = sp.add_parser("gather"); g.add_argument("-o", required=True); g.add_argument("--n", type=int, default=300); g.add_argument("--workers", type=int, default=3)
    f = sp.add_parser("fit"); f.add_argument("pkl")
    a = ap.parse_args(); {"gather": gather, "fit": fit}[a.c](a)
