"""
Chameleon resize engine - entry point (combined engine, branch combo/four).

`resize(source, target_canvas)` runs four resize engines on the design, drops every output that
breaks the rules, and returns one of the rest unchanged. Each engine is a full copy of its own
`engine/` package under `engine/subs/<name>/` and runs in its own process, so the four never share
imports or state:

  main   chameleon main 93cab1a (archetype layouts, candidate scoring)
  night  line/night 3dfabd7 (main's line plus ranker, vision crops, kept elements)
  learn  lane/learn 8ae6a42 (designer moves learned from Crello families)
  jeff   jeff/engine-v1 a7ebd7a (Jeffrey's engine: saliency crops, same-design candidates, OCR check)

The pick was fixed on the final comparison's judged tiles (2026-10-30 data) before the
benchmark that tests it:

  1. Drop outputs that fail `engine/combo/rules.faults`, the stage-one rules as decided on
     2026-10-31 (context/notes/adobe-rules.md): ids, shape and fields, copy byte for byte, fonts
     and colours unchanged, nothing past the canvas, no stretch and no cropped logo or product,
     every element visible (missing assets included).
  2. Keep Night's output unless another passing output's proxy penalty (`engine/combo/terms`, the
     calibrated source-relative judge proxy) is lower by more than MARGIN.
  3. If none passes, the output with the fewest faults, then the lowest penalty.
  4. If every engine failed to answer, a fit-and-centre layout that keeps every element visible.

Environment: CHAMELEON_BRAND_KIT (default <repo>/brand_kit) is passed to every engine;
CHAMELEON_COMBO_ENGINES limits the engines (comma list); CHAMELEON_COMBO_TIMEOUT caps one engine
in seconds (default 300); CHAMELEON_COMBO_LOG appends one JSON line per design (pick, faults,
penalties, seconds).
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
ENGINES = ["night", "main", "learn", "jeff"]
LEAD = "night"
MARGIN = 1.0

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


def _penalty(source: dict, out: dict, rep_src) -> Optional[float]:
    try:
        from engine.combo import terms as T

        return T.penalty(T.terms(rep_src, T.measure(out)), T.calibrated_weights())
    except Exception:  # noqa: BLE001  an unmeasurable output simply cannot win on the proxy
        traceback.print_exc(file=sys.stderr)
        return None


def choose(source: dict, target: dict, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The pick (steps 1-3) over finished runs; returns the chosen run plus the reasons."""
    from engine.combo.rules import faults

    for r in runs:
        if r["out"] is None:
            r["faults"] = ["no output"]
            continue
        try:
            r["faults"] = faults(source, r["out"], target)
        except Exception as ex:  # noqa: BLE001
            r["faults"] = [f"check failed: {type(ex).__name__}: {ex}"]
    answered = [r for r in runs if r["out"] is not None]
    if not answered:
        return {"run": None, "why": "every engine failed"}
    passing = [r for r in answered if not r["faults"]]
    pool = passing or answered
    lead = next((r for r in pool if r["engine"] == LEAD), None) if passing else None
    if passing and len(passing) == 1:
        return {"run": passing[0], "why": "only passing output"}
    rep_src = None
    try:
        from engine.combo import terms as T

        rep_src = T.measure(source)
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
    for r in pool:
        r["penalty"] = _penalty(source, r["out"], rep_src) if rep_src is not None else None
    key = lambda r: (len(r["faults"]), r["penalty"] if r["penalty"] is not None else 1e9, ENGINES.index(r["engine"]))  # noqa: E731
    best = min(pool, key=key)
    if lead is not None and lead["penalty"] is not None and best["penalty"] is not None \
            and lead["penalty"] <= best["penalty"] + MARGIN:
        return {"run": lead, "why": f"{LEAD} within {MARGIN} of the best penalty"}
    if lead is not None and (lead["penalty"] is None or best["penalty"] is None):
        return {"run": lead, "why": f"{LEAD} passes; penalties unavailable"}
    return {"run": best, "why": "lowest penalty among passing" if passing else "fewest faults (none pass)"}


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    target = {"width": target_canvas["width"], "height": target_canvas["height"]}
    names = [n for n in (os.environ.get("CHAMELEON_COMBO_ENGINES") or ",".join(ENGINES)).split(",") if n in ENGINES]
    timeout = float(os.environ.get("CHAMELEON_COMBO_TIMEOUT") or 300)
    try:
        with ThreadPoolExecutor(max_workers=len(names)) as ex:
            runs = list(ex.map(lambda n: _run(n, source, target, timeout), names))
        pick = choose(source, target, runs)
    except Exception:  # noqa: BLE001  never let one odd design take the run down
        traceback.print_exc(file=sys.stderr)
        runs, pick = [], {"run": None, "why": "combiner failed"}
    run = pick["run"]
    out = run["out"] if run is not None else _fallback(source, target)
    log = os.environ.get("CHAMELEON_COMBO_LOG")
    if log:
        rec = {"pick": run["engine"] if run else "fallback", "why": pick["why"], "secs": round(time.time() - t0, 2),
               "runs": {r["engine"]: {"faults": r.get("faults"), "penalty": r.get("penalty"), "secs": r["secs"],
                                       "error": r["error"]} for r in runs}}
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return out


def _fallback(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Uniform fit-and-centre: no element hidden, no picture stretched; backgrounds that covered
    the source canvas cover the target (cover crop)."""
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
                o["font_size"] = max(1, int(round(float(e.get("font_size", 12)) * s)))
        els.append(o)
    return {"canvas": {"width": target["width"], "height": target["height"]}, "elements": els}
