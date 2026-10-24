"""Gate tool: measure the panel's sentinels on dev data, with the real judges, before a gate run.

    loop/capped .venv/bin/python -m tools.gate.sentinel_check --id dev-sent-1 --set crello_old --n 24

Builds the sentinel pool from a dev set (``tools.gate.devsets``; tasks with a designer resize
are the good side, others fall back to nothing: fit-and-centre needs a designer resize), asks
``--n`` decisive sentinels of every family in the panel config (DEFAULT_PANEL, or ``--panel``
JSON overrides) plus the report-only ones at ``small_sentinels`` per decisive ask, and prints
accuracy per family and per perturbation kind. It passes when every family reaches
``--min`` (default 0.85, the gate's sentinel_min) on at least ``--min-n`` decisive sentinels.

Dev data only: it refuses a sealed version name, so no sealed verdict is consumed. The asks run
from ``~/judge-adobe-hackathon-2026/<id>/`` like a gate's, so the usage meter and the Cursor cap
see them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from tools.gate import devsets, sealed
from tools.gate import panel as PN
from tools.gate.engine_run import GATE_ROOT


def pool_for(name: str) -> list:
    tasks, assets = devsets.tasks(name)
    return [{"set": name, "tile": t["task_id"], "design": t["task_id"], "source": t["source"], "base": None,
             "reference": t["reference"], "kit": assets.parent} for t in tasks if t.get("reference")]


def check(gid: str, name: str, n: int, cfg: dict, min_acc: float, min_n: int, judge=None, seed: int = 0) -> dict:
    if sealed.VERSION_RE.match(name) and name in set(sealed.sealed_versions()):
        raise SystemExit(f"{name} is a sealed version: sentinel_check runs on dev data only")
    pool = pool_for(name)
    if not pool:
        raise SystemExit(f"dev set {name!r} has no tasks with a designer resize")
    private = GATE_ROOT / "output" / "sentinel_check" / gid
    pn = PN.Panel(gid, private, cfg, judge=judge, seed=seed)
    fams = list(pn.families)
    sent = pn.sentinels(pool, n, 0, fams)
    small = pn.sentinels(pool, math.ceil(n * float(pn.cfg.get("small_sentinels") or 0)), 0, fams, small=True)
    pn.run([a for v in sent.values() for a in v] + [a for v in small.values() for a in v])
    kinds = pn.sentinel_kinds()
    res = {"id": gid, "set": name, "families": {}, "kinds": kinds, "usage": pn.usage()}
    ok = True
    for f in fams:
        a = pn.sentinel_accuracy(sent[f])
        s = pn.sentinel_accuracy(small[f])
        passed = a["n"] >= min_n and a["accuracy"] is not None and a["accuracy"] >= min_acc
        ok &= passed
        res["families"][f] = {"decisive": a, "report_only": s, "pass": passed}
    res["pass"] = ok
    (private / "result.json").write_text(json.dumps(res, indent=1))
    pn.clean_images()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.gate.sentinel_check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--id", required=True)
    ap.add_argument("--set", default="crello_old", help="a dev set name (tools.gate.devsets)")
    ap.add_argument("--n", type=int, default=24, help="decisive sentinels per family")
    ap.add_argument("--panel", help="JSON overrides of DEFAULT_PANEL")
    ap.add_argument("--min", type=float, default=0.85)
    ap.add_argument("--min-n", type=int, default=20)
    a = ap.parse_args(argv)
    try:
        r = check(a.id, a.set, a.n, json.loads(a.panel) if a.panel else {}, a.min, a.min_n)
    except PN.CapReached as e:
        print(f"SENTINELS PAUSED: {e}")
        return 4
    parts = []
    for f, v in r["families"].items():
        d, s = v["decisive"], v["report_only"]
        acc = f"{d['accuracy']:.2f}" if d["accuracy"] is not None else "n/a"
        racc = f"{s['accuracy']:.2f}" if s["accuracy"] is not None else "n/a"
        parts.append(f"{f} {acc} n={d['n']} {d['by_kind']} report-only {racc} n={s['n']}")
    print(f"SENTINELS {a.set} | " + " | ".join(parts) + f" | cursor_asks {r['usage']['cursor_asks']} | "
          + ("PASS" if r["pass"] else "FAIL"))
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.path.insert(0, str(GATE_ROOT))
    raise SystemExit(main())
