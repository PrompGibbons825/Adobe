"""Run any engine version on a batch of (source, target) jobs, in its own process.

The gate never imports the engine it judges. It writes the jobs to a file and runs a small
runner with ``cwd`` = the engine's repo root (the ``tools/evalset.py`` pattern), so the code
that answers is exactly that root's ``engine/``, and the code that measures is the gate's own.

    results = run_jobs(engine_root, [{"name": "a", "source": {...}, "target": {"width": 300, "height": 250}}])
    results["a"] -> {"out": {...} | None, "fell_back": bool, "error": str | None, "secs": float}

``fell_back`` means the smart engine raised and resize() returned its safe fallback (a
traceback on stderr). ``error`` means resize() itself raised: a crash.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

GATE_ROOT = Path(__file__).resolve().parents[2]

RUNNER = r"""
import contextlib, io, json, os, sys, time, traceback
sys.path.insert(0, os.getcwd())
jobs_path, out_path, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
jobs = json.load(open(jobs_path))

def one(job):
    from engine.resize import resize
    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(io.StringIO()):
            out = resize(json.loads(json.dumps(job["source"])), dict(job["target"]))
        json.dumps(out)
        err = None
    except BaseException as ex:  # noqa: BLE001
        out, err = None, f"{type(ex).__name__}: {ex}"[:400]
    return job["name"], {"out": out, "fell_back": "Traceback" in buf.getvalue(), "error": err,
                         "secs": round(time.time() - t0, 3)}

if workers > 1 and len(jobs) > 1:
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
        res = dict(pool.imap_unordered(one, jobs, chunksize=1))
else:
    res = dict(one(j) for j in jobs)
json.dump(res, open(out_path, "w"))
"""


def default_workers() -> int:
    """Parallel resizes per runner. The loop's capped scope allows 4 cores."""
    return max(1, min(4, int(os.environ.get("GATE_WORKERS", "4"))))


def run_jobs(engine_root: Path, jobs: List[dict], kit: Optional[Path] = None, hash_seed: str = "0",
             workers: Optional[int] = None, timeout: Optional[float] = None) -> Dict[str, dict]:
    """resize() of ``engine_root`` on every job. ``kit`` is a brand-kit directory holding
    ``assets/`` (exported as CHAMELEON_BRAND_KIT); ``hash_seed`` is the runner's
    PYTHONHASHSEED, so two calls with different seeds probe hash-order dependence."""
    engine_root = Path(engine_root).resolve()
    if not (engine_root / "engine" / "resize.py").is_file():
        raise SystemExit(f"{engine_root} is not an engine root (no engine/resize.py)")
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        raise ValueError("job names must be unique")
    env = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
    if kit:
        env["CHAMELEON_BRAND_KIT"] = str(Path(kit).resolve())
    with tempfile.TemporaryDirectory(prefix="gate-run-") as tmp:
        jp, op = Path(tmp) / "jobs.json", Path(tmp) / "out.json"
        jp.write_text(json.dumps(jobs))
        subprocess.run([sys.executable, "-c", RUNNER, str(jp), str(op), str(workers or default_workers())],
                       cwd=engine_root, env=env, check=True, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return json.loads(op.read_text())


# ----------------------------------------------------------------------------
# task directories


def load_tasks(src_dir: Path) -> Dict[str, dict]:
    """Every ``<name>.json`` task in ``src_dir`` holding ``source`` and ``target_canvas``
    (the brief's task format), by name. Files that are not tasks are skipped."""
    tasks = {}
    for p in sorted(Path(src_dir).glob("*.json")):
        try:
            t = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(t, dict) and "source" in t and "target_canvas" in t:
            tasks[p.stem] = t
    return tasks


def output_path(out_dir: Path, name: str) -> Optional[Path]:
    """Where an engine wrote the output for task ``name``: ``<name>.json`` (render_tiles)
    or ``<name>_output.json`` / ``<task_id>_output.json`` (engine.main)."""
    for cand in (f"{name}.json", f"{name}_output.json"):
        p = Path(out_dir) / cand
        if p.is_file():
            return p
    return None


def render_tiles(engine_root: Path, src_dir: Path, out_dir: Path, kit: Optional[Path] = None,
                 workers: Optional[int] = None) -> Dict[str, dict]:
    """Run ``engine_root`` on every task of ``src_dir`` and write ``out_dir/<name>.json``.
    Crashed tasks get no file (the floor counts a missing output as a hard error).
    Returns the runner's per-task status without the outputs."""
    tasks = load_tasks(src_dir)
    jobs = [{"name": n, "source": t["source"], "target": t["target_canvas"]} for n, t in tasks.items()]
    res = run_jobs(engine_root, jobs, kit=kit, workers=workers) if jobs else {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    for n, r in res.items():
        if r["out"] is not None:
            (out_dir / f"{n}.json").write_text(json.dumps(r["out"]))
        status[n] = {k: v for k, v in r.items() if k != "out"}
    return status


def canonical(out: Optional[dict]) -> str:
    """An output as one comparable string: keys sorted, elements sorted by id."""
    if out is None:
        return "null"
    o = dict(out)
    o["elements"] = sorted(out.get("elements", []), key=lambda e: str(e.get("id")))
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def dev_tasks(n_generated: int = 6, seed: int = 11) -> List[dict]:
    """Dev designs only (never sealed data): the samples, tools/cases, and a few
    tools.designs designs. Each as ``{"name", "source", "target_canvas"}``."""
    out = []
    for d in (GATE_ROOT / "input", GATE_ROOT / "tools" / "cases"):
        for n, t in load_tasks(d).items():
            out.append({"name": n, "source": t["source"], "target_canvas": t["target_canvas"]})
    if n_generated:
        sys.path.insert(0, str(GATE_ROOT))
        from tools.designs import designs

        for t in designs(n_generated, seed=seed):
            out.append({"name": t["task_id"], "source": t["source"], "target_canvas": t["target_canvas"]})
    return out


def iter_names(xs: Iterable[dict]) -> List[str]:
    return [x["name"] for x in xs]
