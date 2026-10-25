"""The gate's floor: what any engine version must pass before a judge looks at it.

    # the whole floor for one engine root (suites, Docker, dev tiles), base for the "no rise" rows
    loop/capped .venv/bin/python -m tools.gate.floor run --engine ../worktrees/x --base . \
        --tiles input --tiles tools/cases --json /tmp/floor.json

    # only the tile checks, on outputs someone else already rendered
    .venv/bin/python -m tools.gate.floor tiles --src <task dir> --out <cand outputs> [--base-out <base outputs>] \
        [--kit <brand-kit dir>]

Parts (each reported as PASS / FAIL / SKIP):

* ``suites``  ``tests.run``, ``tools.stress --n 300 --seed 1`` and ``unittest discover``. By default
  the gate's own ``tests/`` and ``tools/`` run against the engine root's ``engine/``
  (``--tests engine`` runs the root's own copies instead), so a candidate cannot pass by
  editing the tests.
* ``docker``  ``docker build`` of the engine root and a ``--network none --cpus 2 --memory 3g``
  run over its ``input/``: every task must yield a JSON and a PNG, and the JSON must pass the
  hard rules.
* ``tiles``   for every (task dir, output dir) pair: ``rules.hard_rule_errors`` and
  ``rules.ghost_text`` must be 0 on every tile (a missing output is a hard error);
  ``promise_errors``, ``contrast_errors``, ``hierarchy_errors`` and ``hidden`` (elements an
  output hides, ``tools/gate/visible.py``, loop/STEERING.md S1) must not rise against the
  base's outputs on the same tasks (with no base they are reported only). A prereg may set
  ``floor_tolerance`` (``{"contrast": 1}``): a counted soft row may then exceed the base by at
  most that many per set before it fails; the excess is recorded under ``tolerated``. Only
  ``TOLERABLE`` rows (promise, contrast, hierarchy) take one, default 0 (no rise at all). Hard
  rules (ids, bounds, aspect), ghost text and hidden elements (S1) never do. ``bleed``
  (``rules.bleed_report``: elements an output lets past the canvas within the S4 bleed policy)
  is reported per tile and in total, never a failure: only the panel judges bleed.
  The must-not-rise rows compare the two engines on the tiles where **both** outputs pass the
  hard rules: a tile one side crashed on carries no soft-row counts for that side, so it would
  otherwise move the other side's count for free.

**Profiles** (a prereg's ``floor_profile``, default ``default``):

* ``default``: everything above; what a change on one engine line is held to.
* ``whole-engine``: one whole engine against another (a line head against main, STEERING S15).
  ``tests.run`` (the finicky suite) and ``unittest`` encode our own engine's choices, and the
  soft rows (promise, contrast, hierarchy) are checks written around our engine's outputs, so a
  different engine must not be blocked by them: the suites part runs ``tools.stress`` only, and
  the soft rows are counted and reported (``reported``), never a failure. Hard rules, ghost
  text, stress, Docker, metamorphic and hidden elements (must not rise) still decide.

A task dir holds brief-format ``<name>.json`` tasks (``source`` + ``target_canvas``); an output
dir holds ``<name>.json`` or ``<name>_output.json``. The floor never decides which directories
those are: the caller (``tools/gate/run.py`` for sealed sets) passes them in. ``--kit`` is the
brand-kit directory (holding ``assets/``) the tiles' images come from.

Measuring code is always the gate's own (this repo's ``tools/rules.py`` and ``engine.smart``
helpers), whatever engine produced the outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from tools.gate.engine_run import GATE_ROOT, load_tasks, output_path, render_tiles

PY = sys.executable
SUITES = [
    ("tests.run", [PY, "-m", "tests.run"]),
    ("stress", [PY, "-m", "tools.stress", "--n", "300", "--seed", "1"]),
    ("unittest", [PY, "-m", "unittest", "discover", "-s", "tests", "-t", "."]),
]
MUST_BE_ZERO = ("hard", "ghost")
MUST_NOT_RISE = ("promise", "contrast", "hierarchy", "hidden")
REPORT_ONLY = ("bleed",)
TOLERABLE = ("promise", "contrast", "hierarchy")  # rows a prereg's floor_tolerance may loosen
PROFILES = {
    "default": {"suites": [n for n, _ in SUITES], "report_rows": ()},
    "whole-engine": {"suites": ["stress"], "report_rows": TOLERABLE},
}


def profile_errors(name) -> List[str]:
    """What is wrong with a prereg's ``floor_profile``."""
    if name is None or name in PROFILES:
        return []
    return [f"unknown floor_profile {name!r} (known: {', '.join(PROFILES)})"]


def tolerance_errors(tol) -> List[str]:
    """What is wrong with a prereg's ``floor_tolerance`` (a dict row -> non-negative int)."""
    if tol is None:
        return []
    if not isinstance(tol, dict):
        return ["floor_tolerance must be an object like {\"contrast\": 1}"]
    errs = []
    for k, v in tol.items():
        if k not in TOLERABLE:
            why = ("is zero-tolerance" if k in MUST_BE_ZERO or k == "hidden" else "is not a floor row")
            errs.append(f"floor_tolerance {k!r} {why} (tolerable: {', '.join(TOLERABLE)})")
        elif isinstance(v, bool) or not isinstance(v, int) or v < 0:
            errs.append(f"floor_tolerance {k!r} must be a non-negative integer")
    return errs
DOCKER_LIMITS = ["--cpus", "2", "--memory", "3g", "--network", "none"]
SKIP_COPY = {".git", ".venv", "output", "__pycache__", "tests", "tools", "worktrees"}


def _tail(text: str, n: int = 6) -> List[str]:
    return [ln for ln in text.strip().splitlines() if ln.strip() and "Fontconfig" not in ln][-n:]


# ----------------------------------------------------------------------------
# suites


def stage(engine_root: Path, dest: Path) -> Path:
    """A throwaway tree: the engine root minus its tests/tools, plus the gate's tests/tools."""
    engine_root = Path(engine_root).resolve()
    for p in engine_root.iterdir():
        if p.name in SKIP_COPY:
            continue
        if p.is_dir():
            shutil.copytree(p, dest / p.name, ignore=shutil.ignore_patterns("__pycache__"), symlinks=True)
        else:
            shutil.copy2(p, dest / p.name)
    for d in ("tests", "tools"):
        shutil.copytree(GATE_ROOT / d, dest / d, ignore=shutil.ignore_patterns("__pycache__", "eval"))
    # The gate's measuring code (tools/rules.py) imports helpers from engine.smart. An engine
    # built differently (another line, e.g. line/jeff) has no engine/smart, so the gate's own
    # copy goes in beside it: the candidate's resize() never imports it, the measurements do.
    if not (dest / "engine" / "smart").exists() and (GATE_ROOT / "engine" / "smart").is_dir():
        shutil.copytree(GATE_ROOT / "engine" / "smart", dest / "engine" / "smart",
                        ignore=shutil.ignore_patterns("__pycache__"))
    (dest / "output").mkdir(exist_ok=True)
    return dest


def run_suites(engine_root: Path, tests_from: str = "gate", only: Optional[List[str]] = None) -> dict:
    res: Dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="gate-floor-") as tmp:
        cwd = Path(engine_root).resolve() if tests_from == "engine" else stage(engine_root, Path(tmp))
        for name, cmd in SUITES:
            if only and name not in only:
                continue
            t0 = time.time()
            p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            ok = p.returncode == 0
            res[name] = {"pass": ok, "secs": round(time.time() - t0), "tail": _tail(p.stdout + "\n" + p.stderr)}
            print(f"  suite {name}: {'PASS' if ok else 'FAIL'} ({res[name]['secs']}s) {res[name]['tail'][-1:]}",
                  flush=True)
    return {"pass": all(r["pass"] for r in res.values()), "parts": res}


# ----------------------------------------------------------------------------
# docker


def run_docker(engine_root: Path, tag: str = "chameleon-gate-floor") -> dict:
    engine_root = Path(engine_root).resolve()
    t0 = time.time()
    b = subprocess.run(["docker", "build", "--platform", "linux/amd64", "-q", "-t", tag, str(engine_root)],
                       capture_output=True, text=True)
    if b.returncode != 0:
        return {"pass": False, "stage": "build", "tail": _tail(b.stdout + b.stderr)}
    tasks = load_tasks(engine_root / "input")
    with tempfile.TemporaryDirectory(prefix="gate-docker-") as tmp:
        inp, out = Path(tmp) / "input", Path(tmp) / "output"
        shutil.copytree(engine_root / "input", inp)
        out.mkdir()
        os.chmod(out, 0o777)
        r = subprocess.run(["docker", "run", "--rm", *DOCKER_LIMITS, "--user", f"{os.getuid()}:{os.getgid()}",
                            "-e", "HOME=/tmp", "-v", f"{inp}:/app/input:ro", "-v", f"{out}:/app/output", tag],
                           capture_output=True, text=True)
        errs = [] if r.returncode == 0 else [f"exit {r.returncode}"]
        if not errs:
            sys.path.insert(0, str(GATE_ROOT))
            from tools.rules import hard_rule_errors

            for tid, t in tasks.items():  # engine.main names outputs by the task file's stem
                js, png = out / f"{tid}_output.json", out / f"{tid}_output.png"
                if not js.is_file() or not png.is_file() or png.stat().st_size == 0:
                    errs.append(f"{tid}: missing output json/png")
                    continue
                errs += [f"{tid}: {e}" for e in hard_rule_errors(t["source"], json.loads(js.read_text()),
                                                                 t["target_canvas"])]
    ok = not errs
    return {"pass": ok, "secs": round(time.time() - t0), "tasks": len(tasks), "errors": errs[:20],
            "tail": [] if ok else _tail(r.stdout + r.stderr)}


# ----------------------------------------------------------------------------
# tiles


def tile_metrics(src_dir: Path, out_dir: Path, kit: Optional[Path] = None) -> Dict[str, dict]:
    """Per task of ``src_dir``: error lists of every floor check on its output in ``out_dir``.
    ``kit`` must be set before the first engine.smart import of this process (the brand-kit
    lookup is cached), so the CLI evaluates each pair in its own process."""
    if kit:
        os.environ["CHAMELEON_BRAND_KIT"] = str(Path(kit).resolve())
    from tools.rules import (bleed_report, contrast_errors, ghost_text, hard_rule_errors, hierarchy_errors,
                             promise_errors)
    from tools.gate.visible import hidden_elements

    assets = Path(kit).resolve() / "assets" if kit else None
    res = {}
    for name, t in load_tasks(src_dir).items():
        src, tgt = t["source"], t["target_canvas"]
        p = output_path(out_dir, name)
        m = {k: [] for k in MUST_BE_ZERO + MUST_NOT_RISE + REPORT_ONLY}
        if p is None:
            m["hard"] = ["no output"]
            res[name] = m
            continue
        try:
            out = json.loads(p.read_text())
            m["hard"] = hard_rule_errors(src, out, tgt)
        except Exception as ex:  # noqa: BLE001  malformed output is a hard error, not a gate crash
            m["hard"] = [f"unreadable output: {type(ex).__name__}: {ex}"]
            res[name] = m
            continue
        if not m["hard"]:  # the other checks assume a structurally valid output
            for key, fn in (("ghost", lambda: ghost_text(out, assets) if assets else ghost_text(out)),
                            ("promise", lambda: promise_errors(src, out, tgt)),
                            ("contrast", lambda: contrast_errors(src, out, tgt)),
                            ("hierarchy", lambda: hierarchy_errors(src, out)),
                            ("hidden", lambda: hidden_elements(src, out, tgt)),
                            ("bleed", lambda: bleed_report(src, out, tgt))):
                try:
                    m[key] = fn()
                except Exception as ex:  # noqa: BLE001
                    m[key] = [f"check crashed: {type(ex).__name__}: {ex}"]
        res[name] = m
    return res


def _metrics_subprocess(src_dir: Path, out_dir: Path, kit: Optional[Path]) -> Dict[str, dict]:
    with tempfile.TemporaryDirectory(prefix="gate-tiles-") as tmp:
        j = Path(tmp) / "m.json"
        cmd = [PY, "-m", "tools.gate.floor", "metrics", "--src", str(src_dir), "--out", str(out_dir), "--json", str(j)]
        if kit:
            cmd += ["--kit", str(kit)]
        subprocess.run(cmd, cwd=GATE_ROOT, check=True, stdout=subprocess.DEVNULL)
        return json.loads(j.read_text())


def compare_tiles(cand: Dict[str, dict], base: Optional[Dict[str, dict]] = None,
                  tolerance: Optional[Dict[str, int]] = None, report_rows=()) -> dict:
    """The tile verdict: zero rows must be 0 everywhere, rise rows must not exceed the base by
    more than their pre-registered ``tolerance`` (default 0), counted on the tiles where both
    sides' outputs pass the hard rules. Rows in ``report_rows`` (a ``whole-engine`` profile)
    are counted and reported under ``reported``, never a failure."""
    errs = tolerance_errors(tolerance)
    bad = set(report_rows) - set(TOLERABLE)
    if bad:
        errs.append(f"report_rows {sorted(bad)} are not soft rows (soft: {', '.join(TOLERABLE)})")
    if errs:
        raise ValueError("; ".join(errs))
    tol = {k: int(v) for k, v in (tolerance or {}).items() if v}
    totals = {k: sum(len(m.get(k, [])) for m in cand.values()) for k in MUST_BE_ZERO + MUST_NOT_RISE + REPORT_ONLY}
    fails = [f"{k}: {totals[k]} (must be 0)" for k in MUST_BE_ZERO if totals[k]]
    base_totals, rises, tolerated, reported, paired = None, {}, {}, {}, None
    if base is not None:
        base_totals = {k: sum(len(m.get(k, [])) for m in base.values()) for k in MUST_NOT_RISE + REPORT_ONLY}
        both = [n for n in cand if n in base and not cand[n].get("hard") and not base[n].get("hard")]
        paired = {"tiles": len(both)}
        for k in MUST_NOT_RISE:
            c, b = (sum(len(side[n].get(k, [])) for n in both) for side in (cand, base))
            paired[k] = {"cand": c, "base": b}
            rise = c - b
            if k in report_rows:
                if rise > 0:
                    reported[k] = {"cand": c, "base": b, "rise": rise}
            elif rise > tol.get(k, 0):
                fails.append(f"{k}: {c} > base {b}" + (f" (tolerance {tol[k]})" if tol.get(k) else ""))
            elif rise > 0:
                tolerated[k] = {"cand": c, "base": b, "rise": rise, "tolerance": tol[k]}
            rises[k] = sorted(n for n in both if len(cand[n][k]) > len(base[n][k]))
    examples = {k: [f"{n}: {e}" for n, m in sorted(cand.items()) for e in m.get(k, [])][:8]
                for k in MUST_BE_ZERO + MUST_NOT_RISE + REPORT_ONLY if totals[k]}
    # report only: per tile, how many elements bleed (S4); bleed never fails the floor
    bleed = {n: len(m.get("bleed", [])) for n, m in sorted(cand.items()) if m.get("bleed")}
    return {"pass": not fails, "tiles": len(cand), "totals": totals, "base_totals": base_totals,
            "fails": fails, "tolerance": tol, "tolerated": tolerated, "reported": reported, "paired": paired,
            "rose_on": rises, "examples": examples, "bleed": bleed}


def check_tiles(src_dir: Path, out_dir: Path, base_out: Optional[Path] = None, kit: Optional[Path] = None,
                tolerance: Optional[Dict[str, int]] = None, report_rows=()) -> dict:
    cand = _metrics_subprocess(src_dir, out_dir, kit)
    base = _metrics_subprocess(src_dir, base_out, kit) if base_out else None
    return compare_tiles(cand, base, tolerance, report_rows)


# ----------------------------------------------------------------------------
# CLI


def _pair(spec: str):
    """``TASKDIR`` or ``TASKDIR:KITDIR``."""
    src, _, kit = spec.partition(":")
    return Path(src).resolve(), (Path(kit).resolve() if kit else None)


def cmd_run(a) -> int:
    report: dict = {"engine": str(Path(a.engine).resolve()), "base": a.base and str(Path(a.base).resolve())}
    skip = set(filter(None, (a.skip or "").split(",")))
    print(f"FLOOR {report['engine']}", flush=True)
    report["suites"] = {"pass": None, "skipped": True} if "suites" in skip else run_suites(a.engine, a.tests)
    if "docker" in skip:
        report["docker"] = {"pass": None, "skipped": True}
    else:
        report["docker"] = run_docker(a.engine)
        print(f"  docker: {'PASS' if report['docker']['pass'] else 'FAIL'} {report['docker'].get('errors', [])[:3]}",
              flush=True)
    tiles = {}
    if "tiles" not in skip:
        with tempfile.TemporaryDirectory(prefix="gate-floor-out-") as tmp:
            for i, spec in enumerate(a.tiles or []):
                src, kit = _pair(spec)
                co, bo = Path(tmp) / f"{i}-cand", Path(tmp) / f"{i}-base"
                render_tiles(a.engine, src, co, kit=kit)
                if a.base:
                    render_tiles(a.base, src, bo, kit=kit)
                r = check_tiles(src, co, bo if a.base else None, kit)
                tiles[str(src)] = r
                print(f"  tiles {src.name}: {'PASS' if r['pass'] else 'FAIL'} n={r['tiles']} {r['totals']}"
                      + (f" base {r['base_totals']}" if r["base_totals"] else "") + (f" {r['fails']}" if r["fails"] else ""),
                      flush=True)
    report["tiles"] = {"pass": all(r["pass"] for r in tiles.values()) if tiles else None, "sets": tiles}
    parts = [report[k]["pass"] for k in ("suites", "docker", "tiles")]
    report["pass"] = all(p is not False for p in parts)
    summary = " | ".join(f"{k} {'SKIP' if report[k]['pass'] is None else 'PASS' if report[k]['pass'] else 'FAIL'}"
                         for k in ("suites", "docker", "tiles"))
    print(f"FLOOR {'PASS' if report['pass'] else 'FAIL'} | {summary}")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    return 0 if report["pass"] else 1


def cmd_tiles(a) -> int:
    r = check_tiles(Path(a.src), Path(a.out), Path(a.base_out) if a.base_out else None,
                    Path(a.kit) if a.kit else None, json.loads(a.tolerance) if a.tolerance else None)
    print(f"TILES {'PASS' if r['pass'] else 'FAIL'} n={r['tiles']} {r['totals']}"
          + (f" base {r['base_totals']}" if r["base_totals"] else "") + (f" {r['fails']}" if r["fails"] else ""))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))
    return 0 if r["pass"] else 1


def cmd_metrics(a) -> int:
    m = tile_metrics(Path(a.src), Path(a.out), Path(a.kit) if a.kit else None)
    Path(a.json).write_text(json.dumps(m))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="the whole floor for one engine root")
    p.add_argument("--engine", required=True, help="repo root of the engine version to check")
    p.add_argument("--base", help="repo root of the base engine, for the must-not-rise rows")
    p.add_argument("--tiles", action="append", help="TASKDIR[:KITDIR], repeatable")
    p.add_argument("--tests", choices=["gate", "engine"], default="gate", help="whose tests/ and tools/ run")
    p.add_argument("--skip", help="comma list of suites,docker,tiles")
    p.add_argument("--json")
    p = sub.add_parser("tiles", help="tile checks on already rendered outputs")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base-out")
    p.add_argument("--kit")
    p.add_argument("--tolerance", help='JSON floor_tolerance, e.g. {"contrast": 1} (default: none)')
    p.add_argument("--json")
    p = sub.add_parser("metrics", help=argparse.SUPPRESS)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kit")
    p.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "tiles": cmd_tiles, "metrics": cmd_metrics}[a.cmd](a)


if __name__ == "__main__":
    sys.path.insert(0, str(GATE_ROOT))
    raise SystemExit(main())
