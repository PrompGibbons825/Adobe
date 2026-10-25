"""Run any engine version on a batch of (source, target) jobs, in its own process.

The gate never imports the engine it judges. It writes the jobs to a file and runs a small
runner with ``cwd`` = the engine's repo root (the ``tools/evalset.py`` pattern), so the code
that answers is exactly that root's ``engine/``, and the code that measures is the gate's own.

    results = run_jobs(engine_root, [{"name": "a", "source": {...}, "target": {"width": 300, "height": 250}}])
    results["a"] -> {"out": {...} | None, "fell_back": bool, "error": str | None, "secs": float}

``fell_back`` means the smart engine raised and resize() returned its safe fallback (a
traceback on stderr). ``error`` means resize() itself raised: a crash.

``run_jobs_isolated`` is the sealed-set render path (tools.gate.run): one process per task,
killed past GATE_TASK_MEM_MB of resident memory or GATE_TASK_TIMEOUT seconds, so one runaway
task is a crash for that task, not a kernel OOM kill of loop/capped's whole scope.
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


ONE_RUNNER = r"""
import contextlib, io, json, os, resource, sys, threading, time, traceback
sys.path.insert(0, os.getcwd())
job_path, out_path = sys.argv[1], sys.argv[2]
job = json.load(open(job_path))
peak = {"kb": 0, "at": None}
main_id = threading.get_ident()

def rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0

def where():
    # the engine/renderer frames active at the peak, outermost first (code names only)
    f, chain = sys._current_frames().get(main_id), []
    while f is not None:
        fn = f.f_code.co_filename
        if "/engine/" in fn or "/renderer/" in fn:
            chain.append(f"{fn.rsplit('/', 1)[-1][:-3]}.{f.f_code.co_name}")
        f = f.f_back
    return " > ".join(reversed(chain[:10])) or None

def sample():
    written = 0
    while True:
        try:
            kb = rss_kb()
            if kb > peak["kb"]:
                peak["kb"], peak["at"] = kb, where()
                if kb - written > 32768:
                    written = kb
                    with open(out_path + ".peak", "w") as f:
                        json.dump(peak, f)
        except Exception:
            pass
        time.sleep(0.05)

if os.path.exists("/proc/self/status"):
    threading.Thread(target=sample, daemon=True).start()
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
json.dump({"out": out, "fell_back": "Traceback" in buf.getvalue(), "error": err,
           "secs": round(time.time() - t0, 3), "peak_at": peak["at"],
           "rss_kb": max(peak["kb"], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}, open(out_path, "w"))
"""


def task_mem_mb() -> int:
    """Resident-memory cap per isolated task (GATE_TASK_MEM_MB, default 1200; 0 = none). Normal
    tasks peak under 300 MB, and 4 workers x 1200 MB stays inside loop/capped's 6 GB, so a
    runaway task is killed by the gate instead of the kernel OOM-killing the whole scope."""
    return max(0, int(os.environ.get("GATE_TASK_MEM_MB", "1200")))


def task_timeout() -> float:
    """Wall-clock cap per isolated task in seconds (GATE_TASK_TIMEOUT, default 300)."""
    return float(os.environ.get("GATE_TASK_TIMEOUT", "300"))


def _rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def run_isolated(engine_root: Path, job: dict, kit: Optional[Path] = None, hash_seed: str = "0",
                 mem_mb: Optional[int] = None, timeout: Optional[float] = None) -> dict:
    """resize() of one job in a fresh process, watched by the parent: killed when its resident
    set passes ``mem_mb`` or its wall time passes ``timeout``. Returns the run_jobs record plus
    ``status`` (ok | error | oom | timeout | died), ``rss_mb`` (peak resident set seen) and
    ``peak_at`` (the engine frames active at the child's own RSS peak: code names only). A
    task that is killed comes back with ``out`` None (a crash); the tasks after it are
    unaffected and all its memory is returned to the OS."""
    import time as _time

    env = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
    if kit:
        env["CHAMELEON_BRAND_KIT"] = str(Path(kit).resolve())
    mem = task_mem_mb() if mem_mb is None else mem_mb
    tmo = task_timeout() if timeout is None else timeout
    with tempfile.TemporaryDirectory(prefix="gate-one-") as tmp:
        jp, op, ep = Path(tmp) / "job.json", Path(tmp) / "out.json", Path(tmp) / "stderr"
        jp.write_text(json.dumps(job))
        t0 = _time.time()
        killed, seen = None, 0.0
        with ep.open("wb") as errf:
            p = subprocess.Popen([sys.executable, "-c", ONE_RUNNER, str(jp), str(op)],
                                 cwd=Path(engine_root).resolve(), env=env,
                                 stdout=subprocess.DEVNULL, stderr=errf)
            while p.poll() is None:
                seen = max(seen, _rss_mb(p.pid))
                if mem and seen > mem:
                    killed = "oom"
                elif tmo and _time.time() - t0 > tmo:
                    killed = "timeout"
                if killed:
                    p.kill()
                    p.wait()
                    break
                _time.sleep(0.05)
        rec, peak = None, {}
        pp = Path(str(op) + ".peak")
        if pp.is_file():
            try:
                peak = json.loads(pp.read_text())
            except ValueError:
                peak = {}
        if op.is_file() and not killed:
            try:
                rec = json.loads(op.read_text())
            except ValueError:
                rec = None
        secs = round(_time.time() - t0, 3)
        if rec is None:
            status = killed or "died"
            rec = {"out": None, "fell_back": False, "error": f"{status} (rc {p.returncode})", "secs": secs,
                   "peak_at": peak.get("at")}
        else:
            status = "ok" if rec["out"] is not None else "error"
        rec["status"] = status
        rec["rss_mb"] = round(max(seen, rec.pop("rss_kb", 0) / 1024, peak.get("kb", 0) / 1024))
        return rec


def run_jobs_isolated(engine_root: Path, jobs: List[dict], kit: Optional[Path] = None, hash_seed: str = "0",
                      done: Optional[Dict[str, dict]] = None, on_result: Optional[callable] = None,
                      mem_mb: Optional[int] = None, timeout: Optional[float] = None,
                      workers: Optional[int] = None) -> Dict[str, dict]:
    """``run_jobs`` with one process per job (``run_isolated``), so memory is returned to the OS
    after every task and one runaway task cannot take the batch down. Jobs named in ``done``
    are skipped (resume); ``on_result(name, rec)`` is called as each job finishes, in the
    calling thread, so the caller can persist results incrementally."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    res: Dict[str, dict] = dict(done or {})
    todo = [j for j in jobs if j["name"] not in res]
    n = max(1, workers or default_workers())

    def one(j):
        return j["name"], run_isolated(engine_root, j, kit=kit, hash_seed=hash_seed, mem_mb=mem_mb, timeout=timeout)

    with ThreadPoolExecutor(n) as ex:
        for fut in as_completed([ex.submit(one, j) for j in todo]):
            name, rec = fut.result()
            res[name] = rec
            if on_result:
                on_result(name, rec)
    return res


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
