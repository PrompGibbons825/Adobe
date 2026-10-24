"""Gate tool: pre-registration and the one-run rule (loop/ROUND.md rule 6).

    # write a prereg (commits resolved to SHAs now), then commit it BEFORE running the gate
    .venv/bin/python -m tools.gate.prereg make --id h0-keep-all --base main --cand exp/h0-keep-all \
        --hypothesis "plans place every element" [--sets crello-v1,novel-v1] [--null] [--builtin fit_centre]
    git add tools/gate/prereg/h0-keep-all.json && git commit -m "Prereg h0-keep-all" && git push

    .venv/bin/python -m tools.gate.prereg check tools/gate/prereg/h0-keep-all.json   # what run.py checks
    .venv/bin/python -m tools.gate.prereg runs                                        # the run log

A prereg holds everything the verdict depends on, fixed before anything renders:

    {"id": "h0-keep-all",                         gate id: [a-z0-9][a-z0-9_-]*, unique
     "hypothesis": "...",
     "base":      {"commit": "<sha>", "label": "main"},
     "candidate": {"commit": "<sha>", "label": "exp/h0-keep-all"}
                  | {"commit": "<sha>", "transform": "nudge"}    null run: A vs A' (1 px + re-encode)
                  | {"builtin": "fit_centre"},                   letterbox baseline (G6)
     "sets": ["novel-v1", "crello-v1"],           sealed versions (tools.gate.sealed)
     "null": false,
     "floor": ["suites", "docker", "metamorphic", "tiles"],
     "rule": {...},                               tools.gate.stats.DEFAULT_RULE, overrides allowed
     "panel": {...},                              tools.gate.panel.DEFAULT_PANEL, written out in full
                                                  by ``make`` (protocol two-family: families, tiebreak,
                                                  batch, agent_daily_cap...), so the models are fixed
     "anchors": {"parity_designs": 30},
     "sentinel_sets": ["crello-v1"],              optional: sealed versions used only as the sentinel
                                                  pool (designer resize vs a defect); never judged,
                                                  so they do not count as a run on that version
     "created": "<UTC>"}

The gate refuses a prereg that is not committed (tracked, clean, in a git repo), and refuses a
second run of the same candidate on any sealed version it already ran on: every run is
appended to ``tools/gate/runs.jsonl`` (override with ``CHAMELEON_GATE_RUNS``) when it starts,
and a crash still counts. ``--resume`` continues the *same* gate id (same prereg content) after
an interruption; it is not a re-run.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tools.gate import sealed

GATE = Path(__file__).resolve().parent
REPO = GATE.parents[1]
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
FLOOR_PARTS = ["suites", "docker", "metamorphic", "tiles"]
TRANSFORMS = ("nudge",)
BUILTINS = ("fit_centre",)


class Refused(Exception):
    """The gate must not run."""


def runs_path() -> Path:
    return Path(os.environ.get("CHAMELEON_GATE_RUNS", GATE / "runs.jsonl"))


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)


def resolve(ref: str, repo: Path = REPO) -> str:
    p = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if p.returncode != 0:
        raise Refused(f"unknown commit or branch {ref!r}")
    return p.stdout.strip()


def cand_key(c: dict) -> str:
    """What the one-run rule calls "the same candidate"."""
    if c.get("builtin"):
        return f"builtin:{c['builtin']}"
    return c["commit"] + (f"+{c['transform']}" if c.get("transform") else "")


def content_hash(p: dict) -> str:
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]


def validate(p: dict) -> List[str]:
    errs = []
    if not ID_RE.match(str(p.get("id", ""))):
        errs.append("id must match [a-z0-9][a-z0-9_-]*")
    for k in ("base", "candidate", "sets"):
        if k not in p:
            errs.append(f"missing {k!r}")
    if errs:
        return errs
    if not p["base"].get("commit"):
        errs.append("base needs a commit")
    c = p["candidate"]
    if c.get("builtin"):
        if c["builtin"] not in BUILTINS:
            errs.append(f"unknown builtin {c['builtin']!r}")
    elif not c.get("commit"):
        errs.append("candidate needs a commit or a builtin")
    if c.get("transform") and c["transform"] not in TRANSFORMS:
        errs.append(f"unknown transform {c['transform']!r}")
    if not p.get("null") and not c.get("transform") and c.get("commit") == p["base"].get("commit"):
        errs.append("candidate and base are the same commit (a null run says so with --null)")
    if p.get("null") and not c.get("transform"):
        errs.append("a null run's candidate is the base with a transform")
    if p.get("null") and c.get("commit") != p["base"].get("commit"):
        errs.append("a null run compares the base with itself")
    if not p["sets"] or not all(sealed.VERSION_RE.match(v) for v in p["sets"]):
        errs.append("sets must be sealed version ids like crello-v1")
    bad = set(p.get("floor", FLOOR_PARTS)) - set(FLOOR_PARTS)
    if bad:
        errs.append(f"unknown floor parts {sorted(bad)}")
    from tools.gate.stats import DEFAULT_RULE

    bad = set(p.get("rule", {})) - set(DEFAULT_RULE)
    if bad:
        errs.append(f"unknown rule keys {sorted(bad)}")
    ss = p.get("sentinel_sets", [])
    if not isinstance(ss, list) or not all(isinstance(v, str) and sealed.VERSION_RE.match(v) for v in ss):
        errs.append("sentinel_sets must be a list of sealed version ids")
    errs += validate_panel(p.get("panel") or {})
    return errs


def validate_panel(cfg: dict) -> List[str]:
    """The two-family protocol (loop/STEERING.md S10): exactly two judge families, each with a
    known CLI and a model, on different CLIs and different models (one Claude, one Cursor); a
    tie-break judge on a third model; 1-12 comparisons per call."""
    from tools.gate.panel import CLIS, DEFAULT_PANEL, PANEL_KEYS, TIEBREAK

    errs = []
    bad = set(cfg) - PANEL_KEYS
    if bad:
        errs.append(f"unknown panel keys {sorted(bad)}")
    c = dict(DEFAULT_PANEL, **cfg)
    if c["protocol"] != "two-family":
        errs.append(f"unknown panel protocol {c['protocol']!r}")
    fams = c.get("families")
    if not isinstance(fams, dict) or len(fams) != 2:
        errs.append("panel families must name exactly two judge families")
    else:
        for name, f in fams.items():
            if name == TIEBREAK or not isinstance(f, dict) or f.get("cli") not in CLIS or not f.get("model"):
                errs.append(f"panel family {name!r} needs a cli ({'/'.join(CLIS)}) and a model")
        if len({(f or {}).get("model") for f in fams.values()}) < 2:
            errs.append("the two panel families must use different models")
        if len({(f or {}).get("cli") for f in fams.values()}) < 2:
            errs.append("the two panel families must use different CLIs (different model vendors)")
    tb = c.get("tiebreak")
    if not isinstance(tb, dict) or tb.get("cli") not in CLIS or not tb.get("model"):
        errs.append("panel tiebreak needs a cli and a model")
    elif isinstance(fams, dict) and tb.get("model") in {(f or {}).get("model") for f in fams.values()}:
        errs.append("the tie-break judge must be a third model")
    from tools.judge.perturb import PERTURBATIONS, SMALL_DEFECTS

    known = set(PERTURBATIONS) | set(SMALL_DEFECTS) | {"fit_centre"}
    for k in ("sentinel_kinds", "report_kinds"):
        if not isinstance(c.get(k), list) or not set(c[k]) <= known:
            errs.append(f"panel {k} must list known perturbations ({', '.join(sorted(known))})")
    if not c.get("sentinel_kinds"):
        errs.append("panel sentinel_kinds must not be empty")
    if not (isinstance(c.get("batch"), int) and 1 <= c["batch"] <= 12):
        errs.append("panel batch must be 1-12 comparisons per call")
    return errs


def committed(path: Path) -> str:
    """The commit that holds ``path`` as it is on disk; Refused if it is untracked or dirty."""
    path = Path(path).resolve()
    if not path.is_file():
        raise Refused(f"no prereg at {path}")
    d = path.parent
    if _git(d, "ls-files", "--error-unmatch", path.name).returncode != 0:
        raise Refused(f"{path.name} is not committed (untracked); commit it before the gate runs")
    st = _git(d, "status", "--porcelain", "--", path.name)
    if st.returncode != 0 or st.stdout.strip():
        raise Refused(f"{path.name} has uncommitted changes; commit them before the gate runs")
    log = _git(d, "log", "-1", "--format=%H", "--", path.name)
    if log.returncode != 0 or not log.stdout.strip():
        raise Refused(f"{path.name} has no commit")
    return log.stdout.strip()


def read_runs() -> List[dict]:
    p = runs_path()
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def check(path: Path, resume: bool = False, need_sealed: bool = True) -> dict:
    """Everything run.py needs before it renders anything. Raises Refused."""
    path = Path(path)
    commit = committed(path)
    try:
        p = json.loads(path.read_text())
    except ValueError as e:
        raise Refused(f"prereg is not JSON: {e}")
    errs = validate(p)
    if errs:
        raise Refused("; ".join(errs))
    if need_sealed:
        have = set(sealed.sealed_versions())
        missing = [v for v in p["sets"] + p.get("sentinel_sets", []) if v not in have]
        if missing:
            raise Refused(f"not sealed (or retired): {missing}")
    key, h = cand_key(p["candidate"]), content_hash(p)
    for r in read_runs():
        if r["id"] == p["id"]:
            if not resume:
                raise Refused(f"gate id {p['id']!r} already ran (use --resume to continue that same run)")
            if r.get("prereg_hash") != h:
                raise Refused(f"gate id {p['id']!r} ran with different prereg content")
            continue
        if r["candidate"] == key:
            again = sorted(set(r["sets"]) & set(p["sets"]))
            if again:
                raise Refused(f"candidate {key} already ran on {again} (gate {r['id']}); "
                              "one run per candidate per sealed version")
    if resume and not any(r["id"] == p["id"] for r in read_runs()):
        raise Refused(f"nothing to resume: gate id {p['id']!r} never started")
    return {"prereg": p, "path": str(path.resolve()), "prereg_commit": commit, "prereg_hash": h, "candidate_key": key}


def record(chk: dict, status: str, **extra) -> None:
    p = chk["prereg"]
    rec = {"id": p["id"], "candidate": chk["candidate_key"], "base": p["base"]["commit"], "sets": p["sets"],
           "null": bool(p.get("null")), "prereg_commit": chk["prereg_commit"], "prereg_hash": chk["prereg_hash"],
           "status": status, "at": _now(), **extra}
    runs_path().parent.mkdir(parents=True, exist_ok=True)
    with runs_path().open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _full_panel(overrides: Optional[dict]) -> dict:
    from tools.gate.panel import DEFAULT_PANEL

    return json.loads(json.dumps(dict(DEFAULT_PANEL, **(overrides or {}))))


def make(gid: str, base: str, cand: Optional[str], sets: Optional[List[str]] = None, hypothesis: str = "",
         null: bool = False, builtin: Optional[str] = None, rule: Optional[dict] = None,
         panel: Optional[dict] = None, floor: Optional[List[str]] = None) -> dict:
    base_sha = resolve(base)
    if null:
        candidate = {"commit": base_sha, "label": f"{base} (nudged)", "transform": "nudge"}
    elif builtin:
        candidate = {"builtin": builtin, "label": builtin}
    else:
        candidate = {"commit": resolve(cand), "label": cand}
    if not sets:
        sets = [v for v in (sealed.latest("novel"), sealed.latest("crello")) if v]
    p = {"id": gid, "hypothesis": hypothesis, "base": {"commit": base_sha, "label": base},
         "candidate": candidate, "sets": list(sets), "null": null,
         "floor": floor if floor is not None else ([] if null else
                                                   ["tiles"] if builtin else list(FLOOR_PARTS)),
         "rule": rule or {}, "panel": _full_panel(panel), "anchors": {"parity_designs": 0 if null else 30},
         "created": _now()}
    if "novel" not in {sealed._parse(v)[0] for v in p["sets"]} and "primary" not in p["rule"]:
        p["rule"]["primary"] = sealed._parse(p["sets"][0])[0]  # no sealed-novel yet: say so in the prereg
    errs = validate(p)
    if errs:
        raise Refused("; ".join(errs))
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.gate.prereg")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--id", required=True)
    m.add_argument("--base", required=True)
    m.add_argument("--cand")
    m.add_argument("--sets", help="comma list of sealed versions (default: latest novel and crello)")
    m.add_argument("--hypothesis", default="")
    m.add_argument("--null", action="store_true")
    m.add_argument("--builtin", choices=BUILTINS)
    m.add_argument("--rule", help="JSON overrides of tools.gate.stats.DEFAULT_RULE")
    m.add_argument("--panel", help="JSON overrides of tools.gate.panel.DEFAULT_PANEL")
    m.add_argument("--out", help="default tools/gate/prereg/<id>.json")
    c = sub.add_parser("check")
    c.add_argument("path")
    c.add_argument("--resume", action="store_true")
    sub.add_parser("runs")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "make":
            if not (a.cand or a.null or a.builtin):
                ap.error("need --cand, --null or --builtin")
            p = make(a.id, a.base, a.cand, a.sets.split(",") if a.sets else None, a.hypothesis, a.null,
                     a.builtin, json.loads(a.rule) if a.rule else None, json.loads(a.panel) if a.panel else None)
            out = Path(a.out) if a.out else GATE / "prereg" / f"{a.id}.json"
            if out.exists():
                raise Refused(f"{out} exists; a prereg is never rewritten")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(p, indent=1) + "\n")
            print(f"wrote {out}; commit and push it before running the gate")
        elif a.cmd == "check":
            r = check(Path(a.path), a.resume)
            print(json.dumps({k: v for k, v in r.items() if k != "prereg"}))
        else:
            for r in read_runs():
                print(json.dumps(r))
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
