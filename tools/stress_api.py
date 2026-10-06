"""Dev tool: random designs x random constraints through the finale API.

    .venv/bin/python -m tools.stress_api --n 120 --seed 1

Same generator and hard-rule check as tools/stress.py, applied to every
candidate candidates() returns under random pins, priorities, safe insets,
min font sizes, measured text and budgets. Also checks: the output is JSON,
pins are kept (or a warning says why not), must items are never hidden,
min_font_px holds, and with no constraints candidate 1 equals resize().
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.api import candidates  # noqa: E402
from engine.resize import resize  # noqa: E402
from tools.sheet import CATALOGUE  # noqa: E402
from tools.stress import check, random_design  # noqa: E402


def random_constraints(rng: random.Random, src: dict, tgt: dict) -> dict:
    ids = [e["id"] for e in src["elements"]]
    texts = [e for e in src["elements"] if e["type"] == "text" and e.get("content")]
    W, H = tgt["width"], tgt["height"]
    c: dict = {}
    if ids and rng.random() < 0.4:
        pid = rng.choice(ids)
        if rng.random() < 0.5:
            w, h = rng.uniform(0.1, 0.4) * W, rng.uniform(0.05, 0.3) * H
            c["pinned"] = {pid: {"x": rng.uniform(0, W - w), "y": rng.uniform(0, H - h), "width": w, "height": h}}
        else:
            c["pinned"] = {pid: rng.choice(["keep", "position", "size", True])}
    if ids and rng.random() < 0.4:
        c["priority"] = {i: rng.choice(["must", "nice", "optional"]) for i in rng.sample(ids, min(len(ids), 3))}
    if rng.random() < 0.3:
        c["safe_insets"] = rng.choice([[0.05, 0.1, 0.05, 0.15], {"top": 0.12, "bottom": 0.18}, 20])
    if rng.random() < 0.3:
        c["min_font_px"] = rng.choice([12, 16, 24])
    if texts and rng.random() < 0.2:
        t = rng.choice(texts)
        c["measured_text"] = {t["id"]: {"font_size": 24, "width": rng.uniform(0.6, 1.6) * 12 * len(t["content"])}}
    if rng.random() < 0.2:
        c["allow_drop"] = True
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    targets = list(CATALOGUE.values())
    fails = crashes = fallbacks = 0
    times = []
    for i in range(a.n):
        src = random_design(rng)
        W, H = rng.choice(targets)
        tgt = {"width": W, "height": H}
        cons = random_constraints(rng, src, tgt) if i % 4 else {}
        budget = rng.choice([None, None, {"ms": 300}, {"work": 2000}, {"bound": True}])
        t0 = time.time()
        try:
            with redirect_stderr(io.StringIO()):
                cs = candidates(json.loads(json.dumps(src)), tgt, cons or None, k=a.k, budget=budget)
            json.dumps(cs)
        except Exception as ex:  # noqa: BLE001
            crashes += 1
            print(f"[{i}] CRASH {type(ex).__name__}: {ex}  cons={cons}")
            continue
        times.append(time.time() - t0)
        errs = []
        drop = bool(cons.get("allow_drop"))
        if cs[0]["archetype"] == "fallback":
            fallbacks += 1
        for c in cs:
            out = c["design"]
            if drop:  # removed elements are fine; check the rest against a source without them
                kept = {e["id"] for e in out["elements"]}
                src_k = dict(src, elements=[e for e in src["elements"] if e["id"] in kept])
                errs += check(src_k, out, tgt, smart=c["archetype"] != "fallback" and not cons.get("pinned"))
            else:
                errs += check(src, out, tgt, smart=c["archetype"] != "fallback" and not cons.get("pinned"))
            chk = c["checks"]
            if "min_font_ok" in chk and not chk["min_font_ok"] and c["archetype"] != "fallback":
                errs.append(f"min_font {chk['min_font_px']} < {cons.get('min_font_px')}")
            if chk.get("pins_ok") is False and not c["warnings"]:
                errs.append(f"pin not kept, no warning: {chk.get('pins')}")
            for eid, pr in (cons.get("priority") or {}).items():
                role = c["roles"].get(eid, {}).get("role")
                if pr == "must" and eid in c["hidden_ids"] and role not in ("missing", "void") \
                        and c["archetype"] != "fallback":
                    errs.append(f"must {eid} ({role}) hidden")
        if not cons and budget is None:
            with redirect_stderr(io.StringIO()):
                ref = resize(json.loads(json.dumps(src)), tgt)
            if json.dumps(ref, sort_keys=True) != json.dumps(cs[0]["design"], sort_keys=True):
                errs.append("candidate 1 != resize()")
        if errs:
            fails += 1
            print(f"[{i}] {src['canvas']} -> {tgt} cons={json.dumps(cons)[:160]}: {errs[:4]}")
    times.sort()
    print(f"\n{a.n} designs: {fails} rule failures, {crashes} crashes, {fallbacks} fallbacks; "
          f"median {times[len(times) // 2]:.2f}s, p95 {times[int(len(times) * 0.95)]:.2f}s, max {times[-1]:.2f}s")
    return 1 if fails or crashes else 0


if __name__ == "__main__":
    raise SystemExit(main())
