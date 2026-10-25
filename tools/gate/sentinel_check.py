"""Gate tool: measure the panel's sentinels on dev data, with the real judges, before a gate run.

    loop/capped .venv/bin/python -m tools.gate.sentinel_check --id dev-sent-1 --set crello_old --n 24

Builds the sentinel pool from a dev set (``tools.gate.devsets``; tasks with a designer resize
are the good side; with ``--engine`` a task without one gets the current engine's output as the
good side, as a gate's own tiles do, so fit-and-centre is not asked of it; ``--tasks`` samples
that many tasks, seeded), asks
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


def pool_for(name: str, engine: bool = False, n_tasks: int = 0, seed: int = 0) -> list:
    tasks, assets = devsets.tasks(name)
    if not engine:
        tasks = [t for t in tasks if t.get("reference")]
    if n_tasks and len(tasks) > n_tasks:
        import random

        tasks = random.Random(f"{seed}|{name}").sample(tasks, n_tasks)
    pool = []
    for t in tasks:
        base = None
        if not t.get("reference"):
            from engine.resize import resize  # the current engine, read only

            base = resize(t["source"], t["target_canvas"])
        pool.append({"set": name, "tile": t["task_id"], "design": t["task_id"], "source": t["source"], "base": base,
                     "reference": t.get("reference"), "kit": assets.parent})
    return pool


def _screened(pn) -> dict:
    out: dict = {}
    for s in pn.screened:
        k = s["why"].split(":", 1)[0] if s["why"].startswith("good side") else \
            " ".join(w for w in s["why"].split()[:4] if not any(c.isdigit() for c in w))
        d = out.setdefault(s["kind"], {})
        d[k] = d.get(k, 0) + 1
    return out


def check(gid: str, name: str, n: int, cfg: dict, min_acc: float, min_n: int, judge=None, seed: int = 0,
          engine: bool = False, n_tasks: int = 0) -> dict:
    if sealed.VERSION_RE.match(name) and name in set(sealed.sealed_versions()):
        raise SystemExit(f"{name} is a sealed version: sentinel_check runs on dev data only")
    pool = pool_for(name, engine, n_tasks, seed)
    if not pool:
        raise SystemExit(f"dev set {name!r} has no tasks with a designer resize (try --engine)")
    private = GATE_ROOT / "output" / "sentinel_check" / gid
    pn = PN.Panel(gid, private, cfg, judge=judge, seed=seed)
    fams = list(pn.families)
    sent = pn.sentinels(pool, n, 0, fams)
    small = pn.sentinels(pool, math.ceil(n * float(pn.cfg.get("small_sentinels") or 0)), 0, fams, small=True)
    pn.run([a for v in sent.values() for a in v] + [a for v in small.values() for a in v])
    kinds = pn.sentinel_kinds()
    res = {"id": gid, "set": name, "pool": len(pool), "families": {}, "kinds": kinds, "usage": pn.usage(),
           "screened": _screened(pn)}
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


def dry(gid: str, name: str, n: int, cfg: dict, engine: bool = False, n_tasks: int = 0) -> dict:
    """Build and screen the sentinels as ``check`` would, without asking a judge: how many of
    each kind were made, from how many tasks, and why the rest were dropped."""
    pool = pool_for(name, engine, n_tasks)
    private = GATE_ROOT / "output" / "sentinel_check" / gid
    import os
    from unittest import mock

    with mock.patch.dict(os.environ, {"CHAMELEON_JUDGE_ROOT": str(private / "judge")}):  # no judge folder
        pn = PN.Panel(gid, private, cfg, judge=lambda p, m: ("", {}))
    fams = list(pn.families)
    made = {}
    for small, k in ((False, n), (True, math.ceil(n * float(pn.cfg.get("small_sentinels") or 0)))):
        got = pn.sentinels(pool, k, 0, fams, small=small)
        for a in got[fams[0]]:
            ask = pn.asks[a]
            kk = ask["key"].rsplit(".", 1)[-1]
            made[kk] = made.get(kk, 0) + 1
    pn.clean_images()
    return {"pool": len(pool), "made": made, "screened": _screened(pn)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.gate.sentinel_check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--id", required=True)
    ap.add_argument("--set", default="crello_old", help="a dev set name (tools.gate.devsets)")
    ap.add_argument("--n", type=int, default=24, help="decisive sentinels per family")
    ap.add_argument("--panel", help="JSON overrides of DEFAULT_PANEL")
    ap.add_argument("--min", type=float, default=0.85)
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--engine", action="store_true", help="the current engine's output is the good side "
                                                          "of tasks without a designer resize")
    ap.add_argument("--tasks", type=int, default=0, help="sample this many tasks of the set (0: all)")
    ap.add_argument("--dry", action="store_true", help="build and screen the sentinels, ask nobody")
    a = ap.parse_args(argv)
    try:
        if a.dry:
            r = dry(a.id, a.set, a.n, json.loads(a.panel) if a.panel else {}, a.engine, a.tasks)
            print(json.dumps(r, indent=1))
            return 0
        r = check(a.id, a.set, a.n, json.loads(a.panel) if a.panel else {}, a.min, a.min_n,
                  engine=a.engine, n_tasks=a.tasks)
    except PN.CapReached as e:
        print(f"SENTINELS PAUSED: {e}")
        return 4
    parts = []
    for f, v in r["families"].items():
        d, s = v["decisive"], v["report_only"]
        acc = f"{d['accuracy']:.2f}" if d["accuracy"] is not None else "n/a"
        racc = f"{s['accuracy']:.2f}" if s["accuracy"] is not None else "n/a"
        parts.append(f"{f} {acc} n={d['n']} {d['by_kind']} good-side {d.get('by_side')} report-only {racc} n={s['n']}")
    print(f"SENTINELS {a.set} pool {r['pool']} | " + " | ".join(parts) + f" | cursor_asks {r['usage']['cursor_asks']} | "
          + ("PASS" if r["pass"] else "FAIL"))
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.path.insert(0, str(GATE_ROOT))
    raise SystemExit(main())
