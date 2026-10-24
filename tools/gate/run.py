"""The gate: floor -> proxy -> panel, on the sealed sets a committed prereg names.

    loop/capped .venv/bin/python -m tools.gate.run --prereg tools/gate/prereg/<id>.json [--resume] [--wait]

Steps (each one can end the run with REJECT):

1. **prereg** (tools.gate.prereg): committed, valid, sets sealed, and the candidate has not run
   on these sealed versions before. The run is logged in tools/gate/runs.jsonl before anything
   renders; a crash still counts as the run.
2. **engines**: base and candidate are ``git archive``d from this repo at the prereg's commits
   into ``output/gate/<id>/engines/`` (a builtin such as fit_centre is written there instead; a
   null run's candidate is the base's outputs nudged 1 px).
3. **render** every sealed task with both engines (``tools.gate.engine_run``), after
   ``sealed.verify`` of each version (a tampered set refuses the run).
4. **floor** (the parts the prereg lists): suites and Docker on the candidate root, metamorphic
   checks against the base, and the tile checks on every sealed tile (hard / ghost must be 0;
   promise, contrast, hierarchy and hidden elements must not rise vs base).
5. **proxy** (tools.judge.terms) on the changed tiles of the designs the panel would judge: it
   may only reject, when its candidate win rate's upper bound is below ``proxy_reject_upper``.
6. **panel** (tools.gate.panel, two families: S10) in pre-registered random design order with
   sequential looks; the decision is tools.gate.stats.evaluate on the pooled tile scores plus
   the per-family harm test. Sentinels are checked per family on its cumulative count: at an
   early look once it holds ``sentinel_min_n`` sentinels, and always at the look where the run
   stops, so one miss in a small look cannot drop a family. A family below ``sentinel_min`` is
   dropped from the run (its votes leave every tile) and the verdict says so;
   a single-family run can only REJECT or REPORT; when every family fails the run is INVALID.
   At the Cursor cap (or a rate/usage limit) the run is PAUSED (exit 4) and continues with
   ``--resume``; ``--wait`` sleeps to the next UTC day instead. It never accepts on one family.
7. **anchors** (reported only): designer parity on sets with references, and any
   ``anchors/<product>/<task>@<W>x<H>.png`` (Express, Adobe's reference crops) against the
   candidate on that dev task.

Writes ``output/gate/<id>/gate.json`` (aggregates only: no tile ids, no content; ``cursor_asks``
is the number of ``agent`` calls) and prints one verdict line, which carries each family's point
estimate on the win cell and their agreement rate. Everything that ties asks to tiles stays in ``output/gate/<id>/private/`` (mode
700, sealed-derived: nobody reads it by hand).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import io
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tools.gate import prereg as P
from tools.gate import sealed, stats
from tools.gate.baselines import nudge, write_builtin
from tools.gate.engine_run import GATE_ROOT, canonical, run_jobs
from tools.gate.tiers import tier_of

OUT_ROOT = GATE_ROOT / "output" / "gate"
ANCHORS = GATE_ROOT / "anchors"


def _say(msg: str) -> None:
    print(msg, flush=True)


# ---- engines ---------------------------------------------------------------------------------

def materialize(spec: dict, dest: Path, repo: Path = GATE_ROOT) -> Path:
    """An engine root for a prereg engine spec (commit or builtin), cached in ``dest``."""
    dest = Path(dest)
    if spec.get("builtin"):
        return write_builtin(spec["builtin"], dest)
    if (dest / "engine" / "resize.py").is_file():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", spec["commit"]], capture_output=True)
    if p.returncode != 0:
        raise P.Refused(f"git archive {spec['commit']} failed: {p.stderr.decode()[-200:]}")
    with tarfile.open(fileobj=io.BytesIO(p.stdout)) as tf:
        tf.extractall(dest)
    return dest


# ---- sealed tasks ----------------------------------------------------------------------------

def design_id(task: dict) -> str:
    m = task.get("meta") or {}
    d = m.get("design") or task.get("design")
    if d:
        return str(d)
    return hashlib.sha256(json.dumps(task["source"], sort_keys=True).encode()).hexdigest()[:12]


def load_set(version: str) -> dict:
    v = sealed.verify(version)
    if not v["ok"]:
        raise P.Refused(f"{version} fails verify (store {v.get('store')}, committed {v.get('committed')})")
    tasks = list(sealed.iter_tasks(version))
    return {"version": version, "set": sealed._parse(version)[0], "kit": sealed.assets_dir(version).parent,
            "tasks": tasks}


def render_set(s: dict, roots: Dict[str, Path], private: Path, transform: Optional[str]) -> Dict[str, dict]:
    """{task_id: {"base": out|None, "cand": out|None}}, cached under private/out/<version>/."""
    d = private / "out" / s["version"]
    tdir = private / "tasks" / s["version"]
    res: Dict[str, dict] = {}
    for role in ("base", "cand"):
        rd = d / role
        if not (rd / ".done").exists():
            rd.mkdir(parents=True, exist_ok=True)
            if role == "cand" and transform:
                outs = {t["task_id"]: (nudge(res[t["task_id"]]["base"]) if res[t["task_id"]]["base"] else None)
                        for t in s["tasks"]}
            else:
                jobs = [{"name": t["task_id"], "source": t["source"], "target": t["target_canvas"]} for t in s["tasks"]]
                r = run_jobs(roots[role], jobs, kit=s["kit"])
                outs = {k: v["out"] for k, v in r.items()}
            for k, o in outs.items():
                if o is not None:
                    (rd / f"{k}.json").write_text(json.dumps(o))
            (rd / ".done").write_text("")
        for t in s["tasks"]:
            p = rd / f"{t['task_id']}.json"
            res.setdefault(t["task_id"], {})[role] = json.loads(p.read_text()) if p.exists() else None
    tdir.mkdir(parents=True, exist_ok=True)
    for t in s["tasks"]:
        p = tdir / f"{t['task_id']}.json"
        if not p.exists():
            p.write_text(json.dumps({"task_id": t["task_id"], "source": t["source"],
                                     "target_canvas": t["target_canvas"]}))
    return res


# ---- floor -----------------------------------------------------------------------------------

def run_floor(parts: List[str], roots: Dict[str, Path], sets: List[dict], private: Path,
              cand_is_engine: bool) -> dict:
    from tools.gate import floor

    cache = private / "floor.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rep: dict = {"parts": {}}
    if "suites" in parts and cand_is_engine:
        r = floor.run_suites(roots["cand"])
        rep["parts"]["suites"] = {"pass": r["pass"], "failed": [k for k, v in r["parts"].items() if not v["pass"]]}
    if "docker" in parts and cand_is_engine:
        r = floor.run_docker(roots["cand"])
        rep["parts"]["docker"] = {"pass": r["pass"], "errors": len(r.get("errors", []))}
    if "metamorphic" in parts and cand_is_engine:
        from tools.gate import metamorphic as mm

        tiles = mm.build_jobs(4, 11)
        c = mm.run_checks(roots["cand"], tiles)
        b = mm.run_checks(roots["base"], tiles)
        ok, lines = mm.verdict(c, b)
        rep["parts"]["metamorphic"] = {"pass": ok, "lines": lines}
    hidden = {"cand": 0, "base": 0}
    if "tiles" in parts:
        sets_rep, ok = {}, True
        for s in sets:
            src = private / "tasks" / s["version"]
            co, bo = private / "out" / s["version"] / "cand", private / "out" / s["version"] / "base"
            r = floor.check_tiles(src, co, bo, s["kit"])
            ok &= r["pass"]
            # aggregates only: no tile names, no messages (they can quote sealed content)
            sets_rep[s["version"]] = {"pass": r["pass"], "tiles": r["tiles"], "totals": r["totals"],
                                      "base_totals": r["base_totals"],
                                      "fails": [f.split(":")[0] for f in r["fails"]]}
            hidden["cand"] += r["totals"].get("hidden", 0)
            hidden["base"] += (r["base_totals"] or {}).get("hidden", 0)
        rep["parts"]["tiles"] = {"pass": ok, "sets": sets_rep}
        rep["hidden"] = hidden
    rep["pass"] = all(v["pass"] for v in rep["parts"].values())
    cache.write_text(json.dumps(rep))
    return rep


def _bleed_counts(t: dict) -> Optional[dict]:
    """Per tile, how many elements of each output bleed past the canvas within the S4 policy
    (``tools.bleed``). Report only: bleed may cost on the auto grader, only the panel judges it."""
    from tools import bleed

    assets = Path(t["kit"]) / "assets" if t.get("kit") else None
    res = {}
    for side in ("base", "cand"):
        if t.get(side) is not None:
            try:
                res[side] = len(bleed.check(t["source"], t[side], t[side]["canvas"], assets)[1])
            except Exception:  # noqa: BLE001  a malformed output is the floor's business
                res[side] = None
    return res or None


# ---- proxy -----------------------------------------------------------------------------------

def run_proxy(tiles: List[dict], rule: dict, seed: int) -> dict:
    """The calibrated source-relative proxy (tools.judge.terms) on changed tiles. Per tile 1 when
    the candidate's penalty is lower, 0 when higher, 0.5 within ``eps``."""
    from tools.judge import terms

    eps = 1e-3
    recs = []
    src_cache: Dict[str, dict] = {}
    w = terms.calibrated_weights()
    for t in tiles:
        os.environ["CHAMELEON_BRAND_KIT"] = str(Path(t["kit"]).resolve())
        try:
            k = f"{t['set']}/{t['tile']}"
            if k not in src_cache:
                src_cache[k] = terms.measure(t["source"])
            pb = terms.penalty(terms.terms(src_cache[k], terms.measure(t["base"])), w)
            pc = terms.penalty(terms.terms(src_cache[k], terms.measure(t["cand"])), w)
        except Exception:  # noqa: BLE001 - the proxy can only reject; a crash is no evidence
            continue
        s = 0.5 if abs(pb - pc) <= eps else (1.0 if pc < pb else 0.0)
        recs.append({"set": t["set"], "design": t["design"], "tier": t["tier"], "score": {"overall": s}})
    r = stats.rule_with_defaults(rule)
    v = list(stats.design_scores(recs).values())
    b = stats.bounds(v, 0.05, 0.05, r["boot"], seed)
    b["reject"] = bool(b["n"] and b["upper"] < r["proxy_reject_upper"])
    b["tiles"] = len(recs)
    return b


# ---- anchors ---------------------------------------------------------------------------------

def anchor_items(roots: Dict[str, Path]) -> Dict[str, List[dict]]:
    """``anchors/<product>/<task>@<W>x<H>.png`` against the candidate on dev task <task>."""
    if not ANCHORS.is_dir():
        return {}
    from tools.gate import devsets

    dev = {t["task_id"]: t for t in list(devsets.samples()) + list(devsets.cases())}
    out: Dict[str, List[dict]] = {}
    for prod in sorted(p for p in ANCHORS.iterdir() if p.is_dir()):
        items, jobs = [], []
        for png in sorted(prod.glob("*@*x*.png")):
            name, size = png.stem.rsplit("@", 1)
            if name not in dev:
                continue
            w, h = (int(x) for x in size.split("x"))
            key = f"{name}@{w}x{h}"
            jobs.append({"name": key, "source": dev[name]["source"], "target": {"width": w, "height": h}})
            items.append({"key": key, "source": dev[name]["source"], "y_png": png, "kit": devsets.REPO / "brand_kit"})
        if not items:
            continue
        res = run_jobs(roots["cand"], jobs, kit=devsets.REPO / "brand_kit")
        out[prod.name] = [dict(it, x=res[it["key"]]["out"]) for it in items if res[it["key"]]["out"]]
    return out


# ---- the run ---------------------------------------------------------------------------------

def _fmt(c: dict) -> str:
    if not c or not c.get("n"):
        return "n=0"
    return f"{c['point']:.2f} [{c['lower']:.2f},{c['upper']:.2f}] n={c['n']}"


def _pt(c: dict) -> str:
    return f"{c['point']:.2f}" if c and c.get("point") is not None else "n/a"


def verdict_line(rep: dict) -> str:
    p = rep["prereg"]
    cand = p["candidate"].get("label") or P.cand_key(p["candidate"])
    base = p["base"].get("label") or p["base"]["commit"][:10]
    parts = [f"GATE {cand} vs {base}"]
    fl = rep.get("floor") or {}
    h = fl.get("hidden")
    hs = f" hidden {h['cand']} (base {h['base']})" if h else " hidden n/a"
    parts.append(f"floor {'PASS' if fl.get('pass') else 'FAIL' if fl else 'n/a'}{hs}")
    pr = rep.get("proxy")
    if pr:
        parts.append(f"proxy {_fmt(pr)}{' REJECT' if pr.get('reject') else ''}")
    ev = (rep.get("panel") or {}).get("final")
    if ev:
        by_set: Dict[str, List[str]] = {}
        for v in p["sets"]:  # versions of one set pool into the same cells: print them once
            by_set.setdefault(sealed._parse(v)[0], []).append(v)
        for s, vs in by_set.items():
            parts.append(f"{'+'.join(vs)} t1 {_fmt(ev['cells'].get(f'{s} t1'))} t2 {_fmt(ev['cells'].get(f'{s} t2'))}")
        parts.append("harm " + (", ".join(ev["harm"]) if ev["harm"] else "none"))
        fam = ev.get("family") or {}
        if fam:
            ag = ev.get("agreement") or {}
            parts.append("families " + " ".join(f"{f} {_pt(c)}" for f, c in fam.items())
                         + (f" agree {ag['rate']:.2f} n={ag['n']}" if ag else " agree n/a"))
        se = rep["panel"].get("sentinels") or {}
        ss = [f"{f} {v['accuracy']:.2f} n={v['n']}" for f, v in se.items() if v.get("n")]
        if ss:
            parts.append("sentinels " + " ".join(ss))
        if rep["panel"].get("dropped"):
            parts.append("dropped " + ", ".join(f"{f} ({d})" for f, d in rep["panel"]["dropped"].items()))
    u = (rep.get("panel") or {}).get("usage")
    if u:
        parts.append(f"cursor_asks {u.get('cursor_asks', 0)}")
    an = rep.get("anchors") or {}
    if "parity" in an and an["parity"].get("n"):
        parts.append(f"parity {an['parity']['share']:.2f} n={an['parity']['n']}")
    for k, v in an.items():
        if k != "parity" and v.get("n"):
            parts.append(f"{k} {v['share']:.2f} n={v['n']}")
    nl = rep.get("null")
    if nl:
        parts.append(f"null false-win {nl['rate']:.3f} ({nl['false_wins']}/{nl['n_perm']})"
                     + "".join(f" {f} {r['rate']:.3f}" for f, r in (nl.get("families") or {}).items()))
    parts.append(rep["verdict"])
    return " | ".join(parts)


def run(prereg_path: Path, resume: bool = False, judge: Optional[Callable] = None,
        out_root: Path = OUT_ROOT, keep_images: bool = False,
        materializer: Callable[[dict, Path], Path] = materialize, wait: bool = False) -> dict:
    chk = P.check(Path(prereg_path), resume)
    p = chk["prereg"]
    rule = stats.rule_with_defaults(p.get("rule"))
    seed = int(rule["seed"]) ^ int(chk["prereg_hash"][:8], 16)
    if not resume:
        P.record(chk, "started")
    out = Path(out_root) / p["id"]
    private = out / "private"
    private.mkdir(parents=True, exist_ok=True)
    os.chmod(private, 0o700)
    rep: dict = {"id": p["id"], "prereg": p, "prereg_commit": chk["prereg_commit"], "verdict": "REJECT"}

    def finish(verdict: str, why: str = "") -> dict:
        rep["verdict"] = verdict + (f" ({why})" if why else "")
        rep["line"] = verdict_line(rep)
        (out / "gate.json").write_text(json.dumps({k: v for k, v in rep.items()}, indent=1, default=str))
        P.record(chk, "done", verdict=verdict, line=rep["line"])
        _say(rep["line"])
        return rep

    # engines and renders
    transform = p["candidate"].get("transform")
    roots = {"base": materializer(p["base"], out / "engines" / "base")}
    roots["cand"] = roots["base"] if transform else materializer(p["candidate"], out / "engines" / "cand")
    sets = [load_set(v) for v in p["sets"]]
    sent_pool = sentinel_pool([load_set(v) for v in p.get("sentinel_sets", [])])
    outs = {}
    for s in sets:
        _say(f"render {s['version']}: {len(s['tasks'])} tasks x 2")
        outs[s["version"]] = render_set(s, roots, private, transform)

    # floor
    parts = p.get("floor", P.FLOOR_PARTS)
    fl = run_floor(parts, roots, sets, private, cand_is_engine=not p["candidate"].get("builtin") and not transform)
    rep["floor"] = fl
    _say(f"floor {'PASS' if fl['pass'] else 'FAIL'} {json.dumps({k: v['pass'] for k, v in fl['parts'].items()})}")
    if not fl["pass"]:
        return finish("REJECT", "floor")

    # changed tiles, design order
    tiles, pool = [], []
    bled = {"base": 0, "cand": 0}  # elements bleeding within the S4 policy, over every tile (report only)
    for s in sets:
        for t in s["tasks"]:
            o = outs[s["version"]][t["task_id"]]
            rec = {"set": s["set"], "version": s["version"], "tile": t["task_id"], "design": design_id(t),
                   "tier": tier_of(t["target_canvas"]),
                   "source": t["source"], "base": o["base"], "cand": o["cand"], "kit": s["kit"],
                   "reference": t.get("reference")}
            rec["bleed"] = _bleed_counts(rec)
            for side, n in (rec["bleed"] or {}).items():
                bled[side] += n or 0
            if o["base"] is not None:
                pool.append(rec)
            if o["base"] is None or o["cand"] is None or canonical(o["base"]) == canonical(o["cand"]):
                continue
            tiles.append(rec)
    designs = sorted({f"{t['set']}/{t['design']}" for t in tiles})
    rep["bleed"] = bled
    random.Random(seed).shuffle(designs)
    cap = designs[: rule["looks"][-1]]
    rep["changed"] = {"tiles": len(tiles), "designs": len(designs), "judged_cap": len(cap),
                      "by_set": {s["version"]: sum(1 for t in tiles if t["version"] == s["version"]) for s in sets}}
    _say(f"changed {len(tiles)} tiles in {len(designs)} designs")
    if not designs:
        return finish("REJECT", "no changed tiles")
    by_design: Dict[str, List[dict]] = {}
    for t in tiles:
        by_design.setdefault(f"{t['set']}/{t['design']}", []).append(t)

    # proxy (may only reject)
    if p.get("proxy", True):
        rep["proxy"] = run_proxy([t for d in cap for t in by_design[d]], rule, seed)
        _say(f"proxy {_fmt(rep['proxy'])}")
        if rep["proxy"]["reject"]:
            return finish("REJECT", "proxy")

    # panel, sequential looks (two families, S10)
    from tools.gate import panel as PN

    panel = PN.Panel(p["id"], private / "panel", p.get("panel"), judge=judge, seed=seed)
    try:
        return _panel_phase(p, rule, seed, panel, rep, designs, by_design, sent_pool or pool, roots, transform,
                            finish, keep_images, wait)
    except PN.CapReached as e:
        rep["panel"] = dict(rep.get("panel") or {}, usage=panel.usage())
        rep["verdict"] = f"PAUSED ({e}; resume with --resume, or run with --wait)"
        rep["line"] = verdict_line(rep)
        (out / "gate.json").write_text(json.dumps(rep, indent=1, default=str))
        P.record(chk, "paused", line=rep["line"])
        _say(rep["line"])
        return rep


def _capped(fn: Callable, wait: bool):
    """``fn()``, sleeping past the Cursor cap when ``wait`` (the panel resumes where it stopped)."""
    from tools.gate.panel import CapReached

    while True:
        try:
            return fn()
        except CapReached as e:
            if not wait:
                raise
            secs = max(60.0, e.reset_at - __import__("time").time() + 120)
            _say(f"paused: {e}; waiting {secs / 3600:.1f} h")
            __import__("time").sleep(secs)


def sentinel_pool(sets: List[dict]) -> List[dict]:
    """The sentinel-only sets' tasks that carry a designer resize, as panel pool records (the
    reference is the good side; nothing is rendered or judged against an engine)."""
    return [{"set": s["set"], "version": s["version"], "tile": t["task_id"], "design": design_id(t),
             "source": t["source"], "base": None, "reference": t["reference"], "kit": s["kit"]}
            for s in sets for t in s["tasks"] if t.get("reference")]


def sentinel_drops(sent: Dict[str, dict], active: List[str], rule: dict, look: int, stop: bool) -> Dict[str, str]:
    """Families to drop on their cumulative sentinel accuracy: at an early look only once a
    family holds ``sentinel_min_n`` sentinels; at the look where the run stops, whatever it holds."""
    out = {}
    for f in active:
        n = (sent.get(f) or {}).get("n", 0)
        if not n or (not stop and n < int(rule["sentinel_min_n"])):
            continue
        acc = sent[f]["correct"] / n
        if acc < float(rule["sentinel_min"]):
            out[f] = f"sentinels {acc:.2f} < {rule['sentinel_min']} n={n} at look {look}"
    return out


def _panel_phase(p, rule, seed, panel, rep, designs, by_design, pool, roots, transform, finish,
                 keep_images, wait) -> dict:
    from tools.gate.panel import score_tiles

    sizes = stats.look_sizes(len(designs), rule["looks"])
    alphas = stats.spend(rule["alpha_win"], [k / sizes[-1] for k in sizes], rule["spending"])
    active = list(panel.families)
    raw, recs, looks = [], [], []
    sent = {f: {"n": 0, "correct": 0} for f in active}
    small: Dict[str, dict] = {}
    dropped: Dict[str, str] = {}
    ref_pool = [t for t in pool if t.get("reference")] or pool
    verdict, why, ev = "REJECT", "", None
    prev = 0
    for i, (k, a) in enumerate(zip(sizes, alphas)):
        batch = [t for d in designs[prev:k] for t in by_design[d]]
        prev = k
        res = _capped(lambda: panel.judge_tiles(batch, i, pool=ref_pool, share=rule["sentinel_share"],
                                                reencode_cand=bool(transform), families=list(active)), wait)
        raw += res["tiles"]
        for f, s in res["sentinels"].items():
            sent.setdefault(f, {"n": 0, "correct": 0})
            sent[f]["n"] += s["n"]
            sent[f]["correct"] += s["correct"]
        last = i == len(sizes) - 1
        for f, why_f in sentinel_drops(sent, active, rule, i + 1, stop=last).items():
            active.remove(f)
            dropped[f] = why_f
        for f, s in res["small_sentinels"].items():
            m = small.setdefault(f, {"n": 0, "correct": 0})
            m["n"] += s["n"]
            m["correct"] += s["correct"]
        look = {"look": i + 1, "designs": k, "alpha": a, "asks": res["asks"], "failed": res["failed"],
                "tiebreaks": res["tiebreaks"],
                "sentinel_accuracy": {f: s["accuracy"] for f, s in res["sentinels"].items()}}
        looks.append(look)
        if not active:
            verdict, why = "INVALID", "every family failed its sentinels: " + "; ".join(
                f"{f} {r}" for f, r in dropped.items())
            look["decision"] = "INVALID"
            _say(f"look {i + 1}: {why}")
            break
        recs = score_tiles(raw, active)
        ev = stats.evaluate(recs, rule, a, final=last, seed=seed + i)
        if ev["decision"] != "CONTINUE" and not last and not p.get("null"):
            # the run would stop here: every family's sentinels count now, whatever their n
            late = sentinel_drops(sent, active, rule, i + 1, stop=True)
            for f, why_f in late.items():
                active.remove(f)
                dropped[f] = why_f
            if late and not active:
                verdict, why = "INVALID", "every family failed its sentinels: " + "; ".join(
                    f"{f} {r}" for f, r in dropped.items())
                look["decision"] = "INVALID"
                _say(f"look {i + 1}: {why}")
                break
            if late:
                recs = score_tiles(raw, active)
                ev = stats.evaluate(recs, rule, a, final=last, seed=seed + i)
        look["decision"] = ev["decision"]
        look["families"] = list(active)
        _say(f"look {i + 1}: {k} designs, {ev['win_cell']} {_fmt(ev['cells'][ev['win_cell']])} -> {ev['decision']}"
             + "".join(f" {f} sentinels {s['accuracy']:.2f}" for f, s in res["sentinels"].items()
                       if s["accuracy"] is not None)
             + (f" dropped {','.join(dropped)}" if dropped else ""))
        if ev["decision"] != "CONTINUE" and not p.get("null"):
            verdict, why = ev["decision"], "; ".join(ev["reasons"])
            break
        verdict, why = ev["decision"], "; ".join(ev["reasons"])
    if verdict == "ACCEPT" and len(active) < int(rule["families_required"]):  # belt and braces
        verdict, why = "REPORT", f"single-family run ({', '.join(active)})"
    for f, d in (dropped.items() if verdict != "INVALID" else ()):
        why =(why + "; " if why else "") + f"family {f} dropped ({d})"
    for d in (sent, small):
        for v in d.values():
            v["accuracy"] = v["correct"] / v["n"] if v["n"] else None
    rep["panel"] = {"looks": looks, "final": ev and {k: ev[k] for k in ("decision", "win_cell", "win_pass", "cells",
                                                                          "criteria", "harm", "reasons", "family",
                                                                          "families", "coverage", "agreement")},
                    # pairwise votes decide; the 1-10 score deltas and small-defect sentinels are report only
                    "deltas": stats.score_deltas(recs), "sentinels": sent, "small_sentinels": small,
                    # per family and perturbation kind, correct/asked (aggregates: kinds are ours, not sealed)
                    "sentinel_kinds": panel.sentinel_kinds(),
                    "families": list(panel.families), "active": list(active), "dropped": dropped,
                    "usage": panel.usage(),
                    "tiebreak_tiles": sum(1 for r in recs if r.get("tiebreak_used")), "tiles": len(recs)}

    # null run: how often would the rule accept a relabelling of these judgements? (pooled and per family)
    if p.get("null") and verdict != "INVALID":
        order = [d for d in designs[:prev]]
        nl = stats.null_false_win_rate(recs, order, rule, n_perm=200, seed=seed)
        fam = {}
        for j, f in enumerate(active):
            r = stats.null_false_win_rate(stats.family_view(recs, f), order, dict(rule, families_required=1),
                                          n_perm=200, seed=seed + 31 * (j + 1))
            fam[f] = r
        nl["families"] = fam
        nl["pass"] = (nl["rate"] <= 0.05 and nl["observed"] != "ACCEPT"
                      and all(r["rate"] <= 0.05 and r["observed"] != "ACCEPT" for r in fam.values()))
        rep["null"] = nl
        verdict = "NULL PASS" if nl["pass"] else "NULL FAIL"
        why = f"observed {nl['observed']}" + "".join(f"; family {f} dropped ({d})" for f, d in dropped.items())

    # anchors (reported only)
    an: dict = {}
    nd = int((p.get("anchors") or {}).get("parity_designs", 0))
    if nd and verdict not in ("INVALID",):
        items = [t for t in pool if t.get("reference") and t.get("cand") is not None]
        items = sorted(items, key=lambda t: hashlib.sha256(f"{seed}{t['tile']}".encode()).hexdigest())[:nd]
        if items:
            its = [{"key": f"{t['set']}/{t['tile']}", "source": t["source"], "x": t["cand"],
                    "y": t["reference"], "kit": t["kit"]} for t in items]
            r = _capped(lambda: panel.compare("parity", its, 100, families=list(active)), wait)
            an["parity"] = {"n": len(r), "share": sum(1 for x in r if x["x_share"] >= 0.5) / len(r) if r else None}
    for prod, items in anchor_items(roots).items():
        r = _capped(lambda: panel.compare(f"anchor-{prod}", items, 200, families=list(active)), wait)
        an[prod] = {"n": len(r), "share": sum(1 for x in r if x["x_share"] >= 0.5) / len(r) if r else None}
    if not any(k for k in an if k != "parity"):
        an["express"] = {"n": 0, "note": "no anchors/express/ yet"}
    rep["anchors"] = an
    rep["panel"]["usage"] = panel.usage()
    rep["cursor_asks"] = rep["panel"]["usage"]["cursor_asks"]
    if not keep_images:
        panel.clean_images()
    return finish(verdict, why)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.gate.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--resume", action="store_true", help="continue this same gate id after an interruption")
    ap.add_argument("--keep-images", action="store_true", help="keep the blind ask images after the run")
    ap.add_argument("--wait", action="store_true",
                    help="at the Cursor cap, sleep until the next UTC day instead of pausing (exit 4)")
    a = ap.parse_args(argv)
    try:
        rep = run(Path(a.prereg), a.resume, keep_images=a.keep_images, wait=a.wait)
    except P.Refused as e:
        print(f"GATE REFUSED: {e}", file=sys.stderr)
        return 3
    if rep["verdict"].startswith("PAUSED"):
        return 4
    return 0 if rep["verdict"].startswith(("ACCEPT", "NULL PASS")) else 1


if __name__ == "__main__":
    sys.path.insert(0, str(GATE_ROOT))
    raise SystemExit(main())
