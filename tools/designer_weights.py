"""Fit the designer term (engine/smart/designer.py) on candidate dumps (tools/designer_dump.py).

    .venv/bin/python -m tools.designer_weights --dumps /data/agents/work/designer/dumps/v0 [--write]

Per task the pick is replayed exactly as plan_design makes it: hand score + designer term on the
top K (ineligible plans and plans below K take -cap), stable re-sort, _diverse (dedupe, 8 places,
2 per archetype), the pixel critic's penalty on those 8, then the first plan that passes the hard
check. The objective is the designer similarity (tools.bench.similarity) of that pick.

Weights are one per decision family (arrangement, drop, anchor, scale, ...), non-negative, fitted
by coordinate ascent over a small grid together with the blend and cap. Folds are by family hash
(5): every number reported as held-out comes from params chosen without that fold. Tournament
families are never in the dumps (designer_dump excludes them) and are re-checked here.
Predictions come from the *current* engine/smart/designer_tree.json, re-applied to each task's
features (the tree itself was fitted by A2 on the same dev families, so its agreement is in-sample
for the tree; only the weights, blend and cap are held out here).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.smart import decisions as dm  # noqa: E402

OUT = ROOT / "engine" / "smart" / "designer_weights.json"
TOURNAMENT = Path("/data/agents/work/designer/tournament_families.txt")
K = 12
TOP = 8
PER_KIND = 2
W_GRID = (0.0, 0.5, 1.0, 2.0)
BLENDS = (2.0, 5.0, 10.0, 20.0, 40.0)
CAPS = (3.0, 6.0, 10.0, 20.0)


def fold_of(fam: str, k: int = 5) -> int:
    return int(hashlib.md5(fam.encode()).hexdigest()[:6], 16) % k


def group(key: str) -> str:
    return key.split(":")[0]


def load(dirs, oracle=False):
    bad = set(TOURNAMENT.read_text().split()) if TOURNAMENT.exists() else set()
    rows = []
    for d in dirs:
        for p in sorted(Path(d).glob("dump_*.jsonl")):
            for line in p.read_text().splitlines():
                r = json.loads(line)
                if not r.get("cands") or r["family"] in bad:
                    continue
                if any(c["sim"] is None for c in r["cands"]):
                    continue
                pred = ({k: (v, 1.0) for k, v in r["ref_decisions"].items()} if oracle  # upper bound only
                        else dm.predict(r["features"]))
                for c in r["cands"]:
                    a = {}
                    for k, (v, conf) in pred.items():
                        if k in c["decisions"] and c["decisions"][k] == v:
                            g = group(k)
                            a[g] = a.get(g, 0.0) + float(conf)
                    c["_a"] = a
                rows.append(r)
    return rows


def pick(r, w=None, blend=0.0, cap=0.0):
    cs = r["cands"]
    n = len(cs)
    sc = [c["score"] for c in cs]
    if w and blend > 0 and cap > 0:
        floor = cs[0]["core_hidden"]
        ag = {}
        for i in range(min(K, n)):
            if cs[i]["core_hidden"] <= floor:
                ag[i] = sum(w.get(g, 0.0) * v for g, v in cs[i]["_a"].items())
        if ag:
            best = max(ag.values())
            sc = [s + (max(-cap, min(0.0, blend * (ag[i] - best))) if i in ag else -cap) for i, s in enumerate(sc)]
    order = sorted(range(n), key=lambda i: -sc[i])  # stable, like list.sort
    seen, uniq = set(), []
    for i in order:
        if cs[i]["key"] not in seen:
            seen.add(cs[i]["key"])
            uniq.append(i)
    head, rest, cnt = [], [], {}
    for i in uniq:
        kd = cs[i]["kind"]
        if len(head) < TOP and cnt.get(kd, 0) < PER_KIND:
            cnt[kd] = cnt.get(kd, 0) + 1
            head.append(i)
        else:
            rest.append(i)
    if r.get("pixels"):
        head.sort(key=lambda i: -(sc[i] - cs[i]["pix"]))
    for i in head + rest:
        if cs[i]["passes"]:
            return i
    return order[0]


def mean_sim(rows, w=None, blend=0.0, cap=0.0):
    return sum(r["cands"][pick(r, w, blend, cap)]["sim"] for r in rows) / max(1, len(rows))


def fit(rows, groups):
    best = (mean_sim(rows), None, 0.0, 0.0)
    w = {g: 1.0 for g in groups}
    for blend in BLENDS:
        for cap in CAPS:
            s = mean_sim(rows, w, blend, cap)
            if s > best[0] + 1e-9:
                best = (s, dict(w), blend, cap)
    if best[1] is None:
        best = (best[0], dict(w), BLENDS[1], CAPS[1])
        best = (mean_sim(rows, *best[1:]), *best[1:])
    for _ in range(2):
        for g in groups:
            for v in W_GRID:
                w2 = dict(best[1])
                w2[g] = v
                if not any(w2.values()):
                    continue
                for blend in (best[2],) + BLENDS:
                    for cap in (best[3],) + CAPS:
                        s = mean_sim(rows, w2, blend, cap)
                        if s > best[0] + 1e-9:
                            best = (s, w2, blend, cap)
    return best


_ROWS, _GROUPS = [], []


def _fit_fold(k):
    tr = [r for r in _ROWS if fold_of(r["family"]) != k] if k < 5 else _ROWS
    return fit(tr, _GROUPS)


def report(rows, w, blend, cap, label):
    base = [r["cands"][pick(r)]["sim"] for r in rows]
    new = [r["cands"][pick(r, w, blend, cap)]["sim"] for r in rows]
    ch = [i for i, r in enumerate(rows) if pick(r) != pick(r, w, blend, cap)]
    wins = sum(1 for i in ch if new[i] > base[i] + 1e-9)
    loss = sum(1 for i in ch if new[i] < base[i] - 1e-9)
    n = max(1, len(rows))
    print(f"{label}: n={len(rows)} main {sum(base)/n:.4f} -> designer {sum(new)/n:.4f} "
          f"(delta {(sum(new)-sum(base))/n:+.4f}); changed {len(ch)} ({len(ch)/n:.0%}) wins {wins} losses {loss}")
    return dict(n=len(rows), main=sum(base) / n, designer=sum(new) / n, changed=len(ch), wins=wins, losses=loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", action="append", default=[])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="predict = the designer's own decisions (diagnostic)")
    a = ap.parse_args()
    rows = load(a.dumps or ["/data/agents/work/designer/dumps/v0"], a.oracle)
    agree_main = sum(1 for r in rows if r.get("main_pick") == r["cands"][pick(r)]["name"]) / max(1, len(rows))
    print(f"{len(rows)} tasks, {len({r['family'] for r in rows})} families; replay matches plan_design's pick "
          f"(by name) on {agree_main:.0%}")
    oracle = sum(max(c["sim"] for c in r["cands"] if c["passes"] or c is r["cands"][0]) for r in rows) / max(1, len(rows))
    print(f"oracle (best passing candidate of the dumped {len(rows[0]['cands']) if rows else 0}): {oracle:.4f}")
    groups = sorted({g for r in rows for c in r["cands"] for g in c["_a"]})
    print("decision groups with predictions:", groups)
    if not groups:
        print("no predictions from the tree: nothing to fit")
        return
    held_base, held_new, held_ch, held_n = 0.0, 0.0, 0, 0
    global _ROWS, _GROUPS
    _ROWS, _GROUPS = rows, groups
    import multiprocessing as mp
    with mp.get_context("fork").Pool(6) as pool:
        fits = pool.map(_fit_fold, range(6))
    for k in range(5):
        va = [r for r in rows if fold_of(r["family"]) == k]
        s, w, blend, cap = fits[k]
        print(f"fold {k}: train {s:.4f} w={w} blend={blend} cap={cap}")
        m = report(va, w, blend, cap, f"  held fold {k}")
        held_base += m["main"] * m["n"]
        held_new += m["designer"] * m["n"]
        held_ch += m["changed"]
        held_n += m["n"]
    print(f"HELD-OUT (5 folds by family): main {held_base/held_n:.4f} -> designer {held_new/held_n:.4f} "
          f"(delta {(held_new-held_base)/held_n:+.4f}); changed {held_ch}/{held_n} ({held_ch/held_n:.0%})")
    s, w, blend, cap = fits[5]
    print(f"ALL: w={w} blend={blend} cap={cap}")
    report(rows, w, blend, cap, "in-sample all")
    if a.write and not a.oracle:
        OUT.write_text(json.dumps(dict(weights=w, blend=blend, cap=cap, K=K, meta=dict(
            tasks=len(rows), held_main=round(held_base / held_n, 4), held_designer=round(held_new / held_n, 4),
            held_changed=held_ch)), indent=1, sort_keys=True) + "\n")
        print("wrote", OUT)


if __name__ == "__main__":
    main()
