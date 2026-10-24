"""Gate tool: the sealed test-set store.

    .venv/bin/python -m tools.gate.sealed add <staging-dir> --set <name>   # seal a staging folder
    .venv/bin/python -m tools.gate.sealed list-versions                    # ids and counts only
    .venv/bin/python -m tools.gate.sealed verify <version>                 # manifest check
    .venv/bin/python -m tools.gate.sealed retire <version> [--reason ...]  # -> tools/gate/dev/

Layout: ``~/chameleon-sealed/<set>/v<N>/`` (every directory mode 700), holding
``items/*.json`` (tasks: ``task_id``, ``source``, ``target_canvas``, optional ``reference`` and
``meta``), ``assets/`` (what the items' image elements name), optional ``set.json`` (build
counts) and ``exclude.json`` (hashes a builder must not reuse), and ``manifest.sha256``. A copy
of every manifest is committed under ``tools/gate/manifests/<set>-v<N>.sha256``, so a later
tamper or a swapped file shows up in ``verify``. A version id is ``<set>-v<N>``.

The rule (loop/ROUND.md rule 5): nobody reads, lists or greps the store by hand. Only
``tools/gate`` code opens it. What this module hands out publicly is aggregate: version ids,
counts, verify results, and ``aggregate()`` over a per-task metric. ``iter_tasks()`` exists for
the gate's own runners (floor, panel), which must render the tasks; they report aggregates too.

``CHAMELEON_SEALED_ROOT``, ``CHAMELEON_GATE_MANIFESTS`` and ``CHAMELEON_GATE_DEV`` override the
three locations (the unit tests use temp dirs).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

GATE = Path(__file__).resolve().parent
VERSION_RE = re.compile(r"^([a-z0-9][a-z0-9_-]*)-v([0-9]+)$")
SET_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")  # no '-': it separates set and version in an id


def root() -> Path:
    return Path(os.environ.get("CHAMELEON_SEALED_ROOT", Path.home() / "chameleon-sealed"))


def manifests_dir() -> Path:
    return Path(os.environ.get("CHAMELEON_GATE_MANIFESTS", GATE / "manifests"))


def dev_dir() -> Path:
    return Path(os.environ.get("CHAMELEON_GATE_DEV", GATE / "dev"))


def _parse(version: str) -> Tuple[str, int]:
    m = VERSION_RE.match(version)
    if not m:
        raise ValueError(f"not a version id (want <set>-v<N>): {version!r}")
    return m.group(1), int(m.group(2))


def _vdir(version: str) -> Path:
    s, n = _parse(version)
    return root() / s / f"v{n}"


def _mkdir700(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
    p.chmod(0o700)


def _lock_down(p: Path) -> None:
    for d, dirs, files in os.walk(p):
        os.chmod(d, 0o700)
        for f in files:
            os.chmod(os.path.join(d, f), 0o600)


# ---- manifests -----------------------------------------------------------------------------

def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_manifest(d: Path) -> List[Tuple[str, str]]:
    out = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name != "manifest.sha256":
            out.append((_sha(p), p.relative_to(d).as_posix()))
    return out


def _format(entries: List[Tuple[str, str]]) -> str:
    return "".join(f"{h}  {rel}\n" for h, rel in entries)


def _read_manifest(p: Path) -> List[Tuple[str, str]]:
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            out.append((h, rel))
    return out


def _counts(entries: List[Tuple[str, str]]) -> Dict[str, int]:
    return {"items": sum(1 for _, r in entries if r.startswith("items/") and r.endswith(".json")),
            "assets": sum(1 for _, r in entries if r.startswith("assets/")),
            "files": len(entries)}


# ---- store operations ----------------------------------------------------------------------

def _validate_staging(stage: Path) -> int:
    items = sorted((stage / "items").glob("*.json")) if (stage / "items").is_dir() else []
    if not items:
        raise ValueError(f"{stage}: no items/*.json")
    assets = stage / "assets"
    for p in items:
        t = json.loads(p.read_text())
        for k in ("task_id", "source", "target_canvas"):
            if k not in t:
                raise ValueError(f"{p.name}: missing {k!r}")
        for d in [t["source"]] + ([t["reference"]] if t.get("reference") else []):
            for e in d.get("elements", []):
                a = e.get("asset_id")
                if e.get("type") == "image" and a and not (assets / a).exists():
                    raise ValueError(f"{p.name}: asset {a!r} not in staging assets/")
    return len(items)


def _taken_numbers(set_name: str) -> List[int]:
    nums = []
    sd = root() / set_name
    if sd.is_dir():
        nums += [int(p.name[1:]) for p in sd.iterdir() if re.fullmatch(r"v[0-9]+", p.name)]
    for p in [manifests_dir()] + ([dev_dir()] if dev_dir().is_dir() else []):
        if p.is_dir():
            for q in p.iterdir():
                m = VERSION_RE.match(q.name.split(".")[0])
                if m and m.group(1) == set_name:
                    nums.append(int(m.group(2)))
    return nums


def add(staging: str | Path, set_name: str) -> Dict:
    """Seal a staging folder as the next version of `set_name`. The staging folder is moved in
    (it no longer exists afterwards). Returns the version id and counts only."""
    if not SET_RE.match(set_name):
        raise ValueError(f"bad set name {set_name!r} (lowercase letters, digits, '_')")
    stage = Path(staging).resolve()
    if not stage.is_dir():
        raise ValueError(f"{stage} is not a directory")
    if root().resolve() in stage.parents:
        raise ValueError("staging folder is already inside the sealed store")
    _validate_staging(stage)
    n = max(_taken_numbers(set_name), default=0) + 1
    version = f"{set_name}-v{n}"
    entries = compute_manifest(stage)
    text = _format(entries)
    (stage / "manifest.sha256").write_text(text)
    _mkdir700(root())
    _mkdir700(root() / set_name)
    dest = root() / set_name / f"v{n}"
    shutil.move(str(stage), str(dest))
    if stage.exists():  # a cross-device move copies; make sure nothing is left behind
        shutil.rmtree(stage)
    _lock_down(dest)
    manifests_dir().mkdir(parents=True, exist_ok=True)
    (manifests_dir() / f"{version}.sha256").write_text(text)
    return {"version": version, **_counts(entries)}


def list_versions() -> List[Dict]:
    """Every sealed and retired version: id, status and counts. Never names a file."""
    out = []
    r = root()
    if r.is_dir():
        for sd in sorted(p for p in r.iterdir() if p.is_dir()):
            for vd in sorted((p for p in sd.iterdir() if re.fullmatch(r"v[0-9]+", p.name)),
                             key=lambda p: int(p.name[1:])):
                m = vd / "manifest.sha256"
                c = _counts(_read_manifest(m)) if m.exists() else {}
                out.append({"version": f"{sd.name}-{vd.name}", "status": "sealed", **c})
    for rec in retired():
        out.append({"version": rec["version"], "status": "retired", "reason": rec.get("reason", "")})
    return out


def sealed_versions(set_name: Optional[str] = None) -> List[str]:
    return [v["version"] for v in list_versions()
            if v["status"] == "sealed" and (set_name is None or _parse(v["version"])[0] == set_name)]


def latest(set_name: str) -> Optional[str]:
    vs = sealed_versions(set_name)
    return max(vs, key=lambda v: _parse(v)[1]) if vs else None


def verify(version: str) -> Dict:
    """Recompute hashes against the store's and the committed manifest. Aggregates only."""
    vd = _vdir(version)
    if not vd.is_dir():
        raise ValueError(f"{version} is not sealed")
    now = dict((rel, h) for h, rel in compute_manifest(vd))
    res = {"version": version, "files": len(now)}
    for name, p in (("store", vd / "manifest.sha256"), ("committed", manifests_dir() / f"{version}.sha256")):
        if not p.exists():
            res[name] = "missing"
            continue
        want = dict((rel, h) for h, rel in _read_manifest(p))
        changed = sum(1 for k in want if k in now and now[k] != want[k])
        missing = sum(1 for k in want if k not in now)
        extra = sum(1 for k in now if k not in want)
        res[name] = "ok" if not (changed or missing or extra) else f"changed {changed} missing {missing} extra {extra}"
    res["ok"] = res.get("store") == "ok" and res.get("committed") == "ok"
    return res


def retired() -> List[Dict]:
    p = dev_dir() / "retired.json"
    return json.loads(p.read_text()) if p.exists() else []


def retire(version: str, reason: str = "") -> Dict:
    """Move a sealed version out of the store into tools/gate/dev/<version>/: from now on it is
    dev data that anyone may read. Its number is never reused."""
    vd = _vdir(version)
    if not vd.is_dir():
        raise ValueError(f"{version} is not sealed")
    dest = dev_dir() / version
    if dest.exists():
        raise ValueError(f"{dest} already exists")
    dev_dir().mkdir(parents=True, exist_ok=True)
    shutil.move(str(vd), str(dest))
    for d, _, files in os.walk(dest):  # dev data is ordinary repo-local data now
        os.chmod(d, 0o755)
        for f in files:
            os.chmod(os.path.join(d, f), 0o644)
    recs = retired()
    recs.append({"version": version, "reason": reason,
                 "retired_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")})
    (dev_dir() / "retired.json").write_text(json.dumps(recs, indent=1) + "\n")
    return {"version": version, "status": "retired", "to": str(dest)}


def exclusions(set_name: str) -> set:
    """Hashes a builder of a new version must not reuse (from every sealed and retired version
    of the set). Hashes only, no content."""
    out = set()
    dirs = []
    sd = root() / set_name
    if sd.is_dir():
        dirs += [p for p in sd.iterdir() if p.is_dir()]
    if dev_dir().is_dir():
        dirs += [p for p in dev_dir().iterdir() if p.is_dir() and p.name.startswith(set_name + "-v")]
    for d in dirs:
        p = d / "exclude.json"
        if p.exists():
            out |= set(json.loads(p.read_text()))
    return out


# ---- for gate code only --------------------------------------------------------------------

def assets_dir(version: str) -> Path:
    return _vdir(version) / "assets"


def iter_tasks(version: str) -> Iterator[Dict]:
    """The tasks of a sealed version, for the gate's own runners (tools/gate/*) only. Whatever
    they report must be aggregate; no builder or orchestrator calls this."""
    for p in sorted((_vdir(version) / "items").glob("*.json")):
        yield json.loads(p.read_text())


def aggregate(version: str, metric: Callable[[Dict, Path], Dict[str, float]]) -> Dict:
    """Run `metric(task, assets_dir)` on every task; return n, errors, and per-key mean / sum /
    min / max. No per-task value leaves this function."""
    vals: Dict[str, List[float]] = {}
    n = errors = 0
    adir = assets_dir(version)
    for t in iter_tasks(version):
        n += 1
        try:
            r = metric(t, adir)
        except Exception:  # noqa: BLE001 - a crash is counted, never shown (it could quote content)
            errors += 1
            continue
        for k, v in (r or {}).items():
            vals.setdefault(k, []).append(float(v))
    return {"version": version, "n": n, "errors": errors,
            "metrics": {k: {"mean": statistics.fmean(v), "sum": sum(v), "min": min(v), "max": max(v), "n": len(v)}
                        for k, v in sorted(vals.items())}}


# ---- CLI -----------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.gate.sealed")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    p.add_argument("staging")
    p.add_argument("--set", required=True, dest="set_name")
    sub.add_parser("list-versions")
    p = sub.add_parser("verify")
    p.add_argument("version")
    p = sub.add_parser("retire")
    p.add_argument("version")
    p.add_argument("--reason", default="")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "add":
            print(json.dumps(add(a.staging, a.set_name)))
        elif a.cmd == "list-versions":
            for v in list_versions():
                print(json.dumps(v))
        elif a.cmd == "verify":
            r = verify(a.version)
            print(json.dumps(r))
            return 0 if r["ok"] else 1
        elif a.cmd == "retire":
            print(json.dumps(retire(a.version, a.reason)))
    except ValueError as e:
        print(f"sealed: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
