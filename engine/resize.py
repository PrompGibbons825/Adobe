"""
Chameleon resize engine - entry point.

`resize(source, target_canvas)` runs two resize engines on the design in parallel, each in its own
process with its own full `engine/` package under `engine/subs/<name>/`:

  night  archetype layouts, candidate scoring, photo-aware crops, learned ranker
  learn  designers' moves learned from Crello multi-size families, then repair

The pick: every output first goes through `engine/combo/strict.clamp` (every box inside the
canvas), then `engine/combo/strict.faults` (the stage-one rules: nothing hidden, nothing
off-canvas). Night's output is kept when it passes; otherwise Night's layout with every unit it
dropped put back (`engine/combo/strict.unhide`, CHAMELEON_UNHIDE=0 turns it off) if that passes;
otherwise Learn's; otherwise the output with the fewest faults, Night first on a tie. If neither
engine answers, a fit-and-centre layout with a type floor.

On the judged official-format tiles this order scored 6.22 against 4.67 for Night alone and 5.66
for Learn alone (broken output = 0), because Night hides elements on dense designs and Learn
almost never does.

Aesthetic mode (CHAMELEON_AESTHETIC=1, off by default): a third engine, jeff (saliency crops, CLIP
logo check, OCR read check, its own critic), runs beside the two. Its output is returned when the
only rule it breaks is allowed bleed: photos and decoration past the canvas edge with at least 60%
of their box on the canvas (`engine/combo/bleed.py`). Anything else it breaks, and the default
pick is returned instead. With the variable unset the output is exactly the default.

Environment: CHAMELEON_BRAND_KIT (default <repo>/brand_kit) is passed to every engine;
CHAMELEON_COMBO_ENGINES limits the default engines (comma list); CHAMELEON_COMBO_TIMEOUT caps one
engine in seconds (default 300); CHAMELEON_COMBO_LOG appends one JSON line per design (pick,
faults, seconds).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
SUBS = Path(__file__).resolve().parent / "subs"
ENGINES = ["night", "learn"]
LEAD = "night"
AESTHETIC = "jeff"

RUNNER = r"""
import json, os, sys
sys.path.insert(0, os.getcwd())
job = json.load(open(sys.argv[1]))
from engine.resize import resize
out = resize(job["source"], job["target"])
with open(sys.argv[2], "w") as f:
    json.dump(out, f)
"""


def _env() -> Dict[str, str]:
    env = dict(os.environ)
    kit = Path(env.get("CHAMELEON_BRAND_KIT") or REPO / "brand_kit").resolve()
    env["CHAMELEON_BRAND_KIT"] = str(kit)
    env.setdefault("CHAMELEON_ASSETS_DIR", str(kit / "assets"))  # Jeff's engine reads this one
    env.setdefault("PYTHONHASHSEED", "0")
    env.pop("CHAMELEON_COMBO_LOG", None)
    return env


def _run(name: str, source: dict, target: dict, timeout: float) -> Dict[str, Any]:
    root = SUBS / name
    env = _env()
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=f"combo-{name}-") as tmp:
        jp, op = Path(tmp) / "job.json", Path(tmp) / "out.json"
        jp.write_text(json.dumps({"source": source, "target": target}))
        err = None
        try:
            p = subprocess.run([sys.executable, "-c", RUNNER, str(jp), str(op)], cwd=root, env=env, timeout=timeout,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if p.returncode != 0:
                err = f"exit {p.returncode}: {(p.stderr or '')[-300:]}"
        except subprocess.TimeoutExpired:
            err = "timeout"
        out = None
        if err is None:
            try:
                out = json.loads(op.read_text())
            except (OSError, ValueError) as ex:
                err = f"no output: {ex}"
    return {"engine": name, "out": out, "error": err, "secs": round(time.time() - t0, 2)}


def choose(source: dict, target: dict, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The pick over finished runs (outputs already clamped); returns the chosen run and why."""
    from engine.combo.strict import faults

    kit = Path(os.environ.get("CHAMELEON_BRAND_KIT") or REPO / "brand_kit") / "assets"
    for r in runs:
        if r["out"] is None:
            r["faults"] = ["no output"]
            continue
        try:
            r["faults"] = faults(source, r["out"], target, kit)
        except Exception as ex:  # noqa: BLE001
            r["faults"] = [f"check failed: {type(ex).__name__}: {ex}"]
    order = sorted((r for r in runs if r["out"] is not None), key=lambda r: ENGINES.index(r["engine"]))
    if not order:
        return {"run": None, "why": "every engine failed"}
    for r in order:
        if not r["faults"]:
            return {"run": r, "why": f"first passing ({r['engine']})"}
        if r["engine"] == LEAD and os.environ.get("CHAMELEON_UNHIDE", "1") != "0":
            # Night's layout reads best; put back the units it dropped before falling back to Learn
            from engine.combo.strict import unhide

            try:
                fixed = unhide(source, r["out"], target)
                f = faults(source, fixed, target, kit) if fixed is not None else ["unhide: no free room"]
            except Exception:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                f = ["unhide failed"]
            if not f:
                return {"run": dict(r, out=fixed, faults=f), "why": f"unhidden ({r['engine']})"}
    from engine.combo.strict import clamp, rescue

    for r in order:  # lift type at 2 px or less and undo opacity hiding, then check again
        fixed = rescue(source, r["out"], target)
        f = faults(source, fixed, target, kit)
        if not f:
            return {"run": dict(r, out=fixed, faults=f), "why": f"rescued ({r['engine']})"}
    fit = clamp(_fallback(source, target), target)
    f = faults(source, fit, target, kit)
    if not f:
        return {"run": {"engine": "fit", "out": fit, "faults": f, "secs": 0, "error": None}, "why": "fit-and-centre passes"}
    return {"run": min(order, key=lambda r: len(r["faults"])), "why": "fewest faults (none pass)"}


def aesthetic_mode() -> bool:
    return os.environ.get("CHAMELEON_AESTHETIC", "").strip().lower() in ("1", "true", "yes", "on")


def beyond_bleed(source: dict, out: dict, target: dict, kit: Path) -> List[str]:
    """Faults of `out` other than allowed bleed (photos and decoration past the canvas edge,
    keeping at least 60% of their box on it). [] = only allowed bleed, or nothing at all."""
    from engine.combo import bleed
    from engine.combo.strict import faults

    errs, _ = bleed.check(source, out, target, kit)
    allowed = bleed.allowed_ids(source, out, target, kit)
    rest = []
    for f in faults(source, out, target, kit):
        i, _, why = f.partition(": ")
        if i in allowed and why in ("off-canvas", "past canvas"):
            continue
        rest.append(f)
    return errs + rest


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    target = {"width": target_canvas["width"], "height": target_canvas["height"]}
    names = [n for n in (os.environ.get("CHAMELEON_COMBO_ENGINES") or ",".join(ENGINES)).split(",") if n in ENGINES]
    timeout = float(os.environ.get("CHAMELEON_COMBO_TIMEOUT") or 300)
    extra = [AESTHETIC] if aesthetic_mode() else []
    aest = None
    try:
        with ThreadPoolExecutor(max_workers=len(names) + len(extra)) as ex:
            done = list(ex.map(lambda n: _run(n, source, target, timeout), names + extra))
        runs = [r for r in done if r["engine"] in ENGINES]
        aest = next((r for r in done if r["engine"] == AESTHETIC), None)
        from engine.combo.strict import clamp

        for r in runs:
            if r["out"] is not None:
                r["out"] = clamp(r["out"], target)
        pick = choose(source, target, runs)
        if aest is not None and aest["out"] is not None:
            kit = Path(os.environ.get("CHAMELEON_BRAND_KIT") or REPO / "brand_kit") / "assets"
            aest["faults"] = beyond_bleed(source, aest["out"], target, kit)
            if not aest["faults"]:
                pick = {"run": aest, "why": "aesthetic mode: only allowed bleed"}
    except Exception:  # noqa: BLE001  never let one odd design take the run down
        traceback.print_exc(file=sys.stderr)
        runs, pick = [], {"run": None, "why": "combiner failed"}
    run = pick["run"]
    out = run["out"] if run is not None else _fallback(source, target)
    if run is None:
        from engine.combo.strict import clamp

        out = clamp(out, target)
    log = os.environ.get("CHAMELEON_COMBO_LOG")
    if log:
        rec = {"pick": run["engine"] if run else "fallback", "why": pick["why"], "secs": round(time.time() - t0, 2),
               "runs": {r["engine"]: {"faults": r.get("faults"), "secs": r["secs"],
                                       "error": r["error"]} for r in runs + ([aest] if aest else [])}}
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return out


def _fallback(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Uniform fit-and-centre: no element hidden, no picture stretched; backgrounds that covered
    the source canvas cover the target (cover crop). Type keeps a floor of 1% of the short side
    (at least 4 px), so a long strip does not shrink copy to nothing."""
    SW, SH = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    W, H = float(target["width"]), float(target["height"])
    s = min(W / SW, H / SH)
    dx, dy = (W - SW * s) / 2, (H - SH * s) / 2
    els = []
    for e in source.get("elements", []):
        o = dict(e)
        x, y, w, h = (float(e.get(k, 0) or 0) for k in ("x", "y", "width", "height"))
        if e.get("type") in ("image", "shape") and x <= 0.5 and y <= 0.5 and x + w >= SW - 0.5 and y + h >= SH - 0.5:
            o.update(x=0, y=0, width=W, height=H)
            if e.get("type") == "image":
                o["fit"] = "cover"
        else:
            o.update(x=dx + x * s, y=dy + y * s, width=w * s, height=h * s)
            if e.get("type") == "text":
                o["font_size"] = max(max(4, int(0.01 * min(W, H))), int(round(float(e.get("font_size", 12)) * s)))
        els.append(o)
    return {"canvas": {"width": target["width"], "height": target["height"]}, "elements": els}
