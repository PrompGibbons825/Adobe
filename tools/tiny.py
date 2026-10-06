"""Dev tool: what the engine gives up on cramped formats.

    .venv/bin/python -m tools.tiny run --engine <repo root of an engine version> --tag now
    .venv/bin/python -m tools.tiny score --tags base,now

Runs every source design (the samples, tools/cases, the Crello families) at
strip, skyscraper and other tiny ad sizes, and counts per design:

  hidden        elements painted with opacity 0 that were visible in the source
                (missing-asset and void elements are not counted), by inferred role
  visible text  texts that are drawn (opacity > 0, non-empty)
  fallback      the smart engine raised and resize() used the fit-and-centre layout
  repair        the engine repaired its best candidate instead (if it reports that)
  rule errors   tools.stress.check() on the output (small print, copy over copy, ...)

Crello families with an odd index are the held-out half, as in tools.evalset;
the samples and tools/cases count with the tuning half. Results land in
tools/crello/eval/<tag>_tiny.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "tools" / "crello" / "eval"
CRELLO = ROOT / "tools" / "crello" / "cases"

FORMATS = {
    "strip": [(728, 90), (970, 90), (320, 50), (468, 60), (300, 50)],
    "skyscraper": [(160, 600), (120, 600)],
    "tiny": [(300, 250), (336, 280), (250, 250), (320, 100)],
}

RUNNER = r"""
import contextlib, io, json, os, sys
sys.path.insert(0, os.getcwd())
from engine.resize import resize
jobs = json.load(open(sys.argv[1]))
out = []
try:
    from engine.smart import demote as _dm
except Exception:
    _dm = None
for j in jobs:
    src = json.load(open(j["path"]))["source"]
    buf = io.StringIO()
    if _dm is not None:
        _dm.STATS.clear()
    with contextlib.redirect_stderr(buf):
        o = resize(json.loads(json.dumps(src)), {"width": j["w"], "height": j["h"]})
    err = buf.getvalue()
    out.append(dict(j, out=o, fallback="Traceback" in err,
                    repair=bool(_dm is not None and _dm.STATS.get("repair")),
                    log=list(_dm.STATS.get("log", [])) if _dm is not None else []))
json.dump(out, open(sys.argv[2], "w"))
"""


def sources():
    out = []
    for p in sorted((ROOT / "input").glob("*.json")) + sorted((ROOT / "tools" / "cases").glob("*.json")):
        out.append((p.stem, str(p), "tune"))
    for p in sorted(CRELLO.glob("fam??.json")):
        out.append((p.stem, str(p), "tune" if int(p.stem[3:5]) % 2 == 0 else "held"))
    return out


def cmd_run(a):
    EVAL.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, path, split in sources():
        for cls, sizes in FORMATS.items():
            for w, h in sizes:
                jobs.append(dict(name=name, path=path, split=split, cls=cls, w=w, h=h))
    n = max(1, a.jobs)
    chunks = [jobs[i::n] for i in range(n)]
    tmp = EVAL / f".{a.tag}_tiny"
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ, CHAMELEON_BRAND_KIT=str(CRELLO))
    procs = []
    for i, ch in enumerate(chunks):
        jp, op = tmp / f"jobs{i}.json", tmp / f"out{i}.json"
        jp.write_text(json.dumps(ch))
        procs.append((subprocess.Popen([sys.executable, "-c", RUNNER, str(jp), str(op)], cwd=a.engine, env=env), op))
    res = []
    for p, op in procs:
        if p.wait() != 0:
            raise SystemExit(f"runner failed: {op}")
        res += json.loads(op.read_text())
    rows = [_measure(r) for r in res]
    (EVAL / f"{a.tag}_tiny.json").write_text(json.dumps(rows, indent=0))
    print("wrote", len(rows), "runs to", EVAL / f"{a.tag}_tiny.json")


def _measure(r: dict) -> dict:
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("CHAMELEON_BRAND_KIT", str(CRELLO))
    from engine.smart.analyze import analyze
    from tools.stress import check

    src = json.load(open(r["path"]))["source"]
    d = analyze(json.loads(json.dumps(src)))
    role = {}
    for e in d.backgrounds:
        role[e["id"]] = "background"
    for e in d.decor:
        role[e["id"]] = "decor"
    for bd in d.backdrops:
        role[bd.el["id"]] = "backdrop"
    for it in d.items:
        for e in it.els:
            r_ = it.role
            if it.kind in ("group", "composite") and e.get("type") == "text" and it.role in ("panel", "hero", "image"):
                r_ = it.role + "-text"
            role[e["id"]] = r_
    src_vis = {e["id"]: float(e.get("opacity", 1) if e.get("opacity") is not None else 1) > 0 for e in src["elements"]}
    hidden = Counter()
    vis_text = 0
    hid_text = 0
    for e in r["out"]["elements"]:
        op = float(e.get("opacity", 1) if e.get("opacity") is not None else 1)
        rl = role.get(e["id"], "?")
        if op <= 0 and src_vis.get(e["id"], True) and rl not in ("missing", "void"):
            hidden[rl] += 1
            if e.get("type") == "text":
                hid_text += 1
        if e.get("type") == "text" and op > 0 and str(e.get("content", "")).strip():
            vis_text += 1
    errs = check(src, r["out"], {"width": r["w"], "height": r["h"]}, smart=not r["fallback"])
    return dict(name=r["name"], split=r["split"], cls=r["cls"], w=r["w"], h=r["h"], hidden=dict(hidden),
                n_hidden=sum(hidden.values()), hidden_text=hid_text, visible_text=vis_text,
                fallback=r["fallback"], repair=r.get("repair", False), errors=errs, log=r.get("log", []))


def cmd_score(a):
    tags = a.tags.split(",")
    data = {t: json.loads((EVAL / f"{t}_tiny.json").read_text()) for t in tags}
    for split in ("tune", "held", "all"):
        print(f"\n=== split={split}")
        print("tag".ljust(10) + "class".ljust(12) + "runs".rjust(6) + "hidden/des".rjust(12) + "hid-text".rjust(10)
              + "des-w/hid".rjust(11) + "vis-text".rjust(10) + "fallback".rjust(10) + "repair".rjust(8)
              + "rule-err".rjust(10))
        for t in tags:
            for cls in list(FORMATS) + ["ALL"]:
                rs = [r for r in data[t] if (split == "all" or r["split"] == split) and (cls == "ALL" or r["cls"] == cls)]
                if not rs:
                    continue
                n = len(rs)
                print(t.ljust(10) + cls.ljust(12) + f"{n:6d}" + f"{sum(r['n_hidden'] for r in rs) / n:12.2f}"
                      + f"{sum(r['hidden_text'] for r in rs):10d}" + f"{sum(1 for r in rs if r['n_hidden']):11d}"
                      + f"{sum(r['visible_text'] for r in rs):10d}" + f"{sum(r['fallback'] for r in rs):10d}"
                      + f"{sum(r['repair'] for r in rs):8d}" + f"{sum(1 for r in rs if r['errors']):10d}")
    print("\nhidden elements by role (all splits)")
    for t in tags:
        c = Counter()
        for r in data[t]:
            c.update(r["hidden"])
        print(t.ljust(10) + "  ".join(f"{k}={v}" for k, v in c.most_common()))
    if a.show:
        for t in tags:
            for r in data[t]:
                if r["errors"] or r["fallback"]:
                    print(t, r["name"], f"{r['w']}x{r['h']}", "FALLBACK" if r["fallback"] else "", r["errors"][:3])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--engine", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--jobs", type=int, default=3)
    p = sub.add_parser("score")
    p.add_argument("--tags", required=True)
    p.add_argument("--show", action="store_true", help="list fallbacks and rule errors")
    a = ap.parse_args()
    {"run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
