"""Light dev eval: B0's similarity (mean IoU with the designer) + own rule check, per aspect class.
Reporting only; never used to fit or select weights.

    python -m tools.sim_eval [--set tournament_set.json] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BENCH = Path.home() / "code/adobe-hackathon-2026/worktrees/night-bench"
WIDE = Path("/data/agents/work/crello-wide")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(WIDE / "tournament_set.json"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--engine", default="opt", choices=["opt", "fallback"])
    a = ap.parse_args()
    os.environ.setdefault("CHAMELEON_BRAND_KIT", str(WIDE / "cases"))
    src = (BENCH / "tools/bench.py").read_text()
    ns: dict = {"__file__": str(BENCH / "tools/bench.py"), "__name__": "bench"}
    exec(compile(src.split("# ---- run")[0], "bench", "exec"), ns)   # aspect_class, similarity, load_tasks
    sys.path[:] = [x for x in sys.path if not x.startswith(str(BENCH))]
    sys.path.insert(0, str(ROOT))
    for m in [m for m in sys.modules if m == "engine" or m.startswith("engine.")]:
        del sys.modules[m]
    from engine.resize import _fallback, resize
    from tools.lane import check
    tasks = ns["load_tasks"](a.limit, a.set)
    per, faults, times = {}, 0, []
    for n, t in tasks.items():
        t0 = time.time()
        fn = resize if a.engine == "opt" else _fallback
        out = fn(json.loads(json.dumps(t["source"])), dict(t["target"]))
        times.append(time.time() - t0)
        errs = check(t["source"], out, t["target"])
        if errs:
            faults += 1
            print(n, errs[:4], flush=True)
        s = ns["similarity"](out, t["reference"])
        c = ns["aspect_class"](t["target"]["width"], t["target"]["height"])
        if s is not None:
            per.setdefault(c, []).append(s)
    allv = [v for vs in per.values() for v in vs]
    times.sort()
    print(json.dumps({"n": len(tasks), "faults": faults, "sim": round(sum(allv) / len(allv), 4),
                      "per_class": {k: round(sum(v) / len(v), 4) for k, v in sorted(per.items())},
                      "p50": round(times[len(times) // 2], 2), "p95": round(times[int(.95 * len(times))], 2)}))


if __name__ == "__main__":
    main()
