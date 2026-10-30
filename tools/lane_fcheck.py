"""L0's fundamentals.check (vendored in engine/fundamentals) over input/ x 20 formats, random stress
designs and Crello dev tasks (not the tournament set). Tallies hard and promise faults by code.

    .venv/bin/python -m tools.lane_fcheck [--stress 100] [--crello 60]
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
from engine.fundamentals.check import check  # noqa: E402
from engine.resize import resize  # noqa: E402
from tools.lane_check import FORMATS  # noqa: E402
from tools.lane_stress import rnd_design  # noqa: E402

KIT = ROOT / "brand_kit" / "assets"
WIDE = Path("/data/agents/work/crello-wide")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress", type=int, default=100)
    ap.add_argument("--crello", type=int, default=60)
    a = ap.parse_args()
    jobs = []
    for p in sorted((ROOT / "input").glob("*.json")):
        src = json.loads(p.read_text())["source"]
        for k, (w, h) in FORMATS.items():
            jobs.append(("input", f"{p.stem[18:]}@{k}", src, dict(width=w, height=h), KIT))
    r = random.Random(2)
    for d in range(a.stress):
        src = rnd_design(r)
        for f in r.sample(list(FORMATS), 3):
            jobs.append(("stress", f"s{d}@{f}", src, dict(width=FORMATS[f][0], height=FORMATS[f][1]), KIT))
    tour = set(json.load(open(WIDE / "tournament_set.json"))["tasks"])
    dev = []
    for p in sorted((WIDE / "cases").glob("*.json")):
        d = json.loads(p.read_text())
        for k, t in enumerate(d["targets"]):
            if f"{d['family']}_{k}" not in tour:
                dev.append((d["source"], t["canvas"], f"{d['family']}_{k}"))
    step = max(1, len(dev) // max(1, a.crello))
    for src, tgt, n in dev[::step][: a.crello]:
        jobs.append(("crello", n, src, dict(tgt), WIDE / "cases" / "assets"))
    hard, prom, tiles = collections.Counter(), collections.Counter(), collections.Counter()
    import os
    for grp, name, src, tgt, kit in jobs:
        os.environ["CHAMELEON_BRAND_KIT"] = str(Path(kit).parent)
        out = resize(src, tgt)
        fs = check(src, out, tgt, assets=str(kit))
        h = [f for f in fs if f.level == "hard"]
        for f in h:
            hard[(grp, f.code)] += 1
        for f in fs:
            if f.level == "promise":
                prom[(grp, f.code)] += 1
        tiles[grp] += 1
        for x in h:
            if hard[(grp, x.code)] <= 3:
                print(grp, name, str(x))
    print(json.dumps(dict(tiles=dict(tiles), hard={f"{g}:{c}": n for (g, c), n in sorted(hard.items())},
                          promise={f"{g}:{c}": n for (g, c), n in sorted(prom.items())})))


if __name__ == "__main__":
    main()
