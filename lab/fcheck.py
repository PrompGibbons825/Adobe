"""Run engine.fundamentals.check on the tournament tiles (+ optional stress). python -m lab.fcheck [--stress N]"""
import json, sys, collections, argparse, time
from pathlib import Path
from engine.compose import compose
from engine.fundamentals.check import check
from lab.check import check as mine
CASES = Path("/data/agents/work/crello-wide/cases")
ap = argparse.ArgumentParser(); ap.add_argument("--stress", type=int, default=0); ap.add_argument("--pixels", action="store_true"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
tour = json.load(open("/data/agents/work/crello-wide/tournament_set.json"))["tasks"]
if a.limit: tour = tour[:a.limit]
jobs = []
for n in tour:
    fam, k = n.rsplit("_", 1); f = json.load(open(CASES / f"{fam}.json"))
    jobs.append((n, f["source"], dict(f["targets"][int(k)]["canvas"])))
if a.stress:
    import random
    from lab.stress import gen
    from lab.sheet_fmt import FORMATS
    rng = random.Random(5)
    for i in range(a.stress):
        s = gen(rng)
        for nm in rng.sample(list(FORMATS), 3):
            W, H = FORMATS[nm]; jobs.append((f"s{i}_{nm}", s, {"width": W, "height": H}))
cnt = collections.Counter(); tiles = collections.Counter(); ex = {}; missed = 0
for n, src, tg in jobs:
    o = compose(src, tg); o.pop("_layout", None)
    fs = check(src, o, tg, pixels=a.pixels)
    seen = set()
    for f in fs:
        if f.level == "info": continue
        key = (f.level, f.code); cnt[key] += 1; seen.add(key); ex.setdefault(key, (n, str(f)[:140]))
    for key in seen: tiles[key] += 1
    if any(f.level == "hard" for f in fs) and not mine(src, o): missed += 1
print("tiles", len(jobs), "hard-faults-missed-by-lab/check", missed)
for k, v in cnt.most_common(): print(k, v, "tiles", tiles[k], ex[k])
