"""Pairwise comparison: ranker pick vs hand-score pick on unseen dev tiles.

    CHAMELEON_RANKER=0 .venv/bin/python -m tools.ranker_compare --split dev
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def family_of(fmt: str, w: int, h: int) -> str:
    r = w / max(1, h)
    if r >= 3:
        return "banners"
    if r <= 1 / 1.6:
        return "stories"
    if 0.9 <= r <= 1.1:
        return "squares"
    if r >= 1.3:
        return "posters"
    return "other"


def run_engine(engine: Path, source: dict, target: dict, ranker_on: bool) -> dict:
    env = os.environ.copy()
    env["CHAMELEON_RANKER"] = "1" if ranker_on else "0"
    script = """
import json, os, sys
sys.path.insert(0, os.getcwd())
os.environ["CHAMELEON_RANKER"] = sys.argv[1]
from engine.resize import resize
src, tgt = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
print(json.dumps(resize(src, tgt)))
"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as s, \
            tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as t:
        json.dump(source, s)
        json.dump(target, t)
        sp, tp = s.name, t.name
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", script, env["CHAMELEON_RANKER"], sp, tp],
            cwd=str(engine), env=env, stderr=subprocess.DEVNULL,
        )
        return json.loads(out)
    finally:
        os.unlink(sp)
        os.unlink(tp)


def score_tile(source: dict, out_a: dict, out_b: dict, target: dict) -> str:
    """Return win_a, win_b, or same using audit flags + check counts."""
    from tools import audit
    from tools import rules

    tgt = {"width": target["width"], "height": target["height"]}
    fa = audit.flags(source, out_a, tgt)
    fb = audit.flags(source, out_b, tgt)
    hard = ("overlap", "hidden", "flat", "speck", "nothing")
    sa = sum(3 for h in hard if h in fa) + len(fa)
    sb = sum(3 for h in hard if h in fb) + len(fb)
    ea = len(rules.hard_rule_errors(source, out_a, tgt))
    eb = len(rules.hard_rule_errors(source, out_b, tgt))
    sa += 10 * ea
    sb += 10 * eb
    if sa < sb - 0.5:
        return "ranker"
    if sb < sa - 0.5:
        return "hand"
    return "same"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--engine", type=Path, default=ROOT)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from tools.unseen import designs_for  # noqa

    tasks = designs_for(a.split)
    if a.limit:
        tasks = tasks[: a.limit]
    by_fam = defaultdict(lambda: {"ranker": 0, "hand": 0, "same": 0})
    totals = {"ranker": 0, "hand": 0, "same": 0}
    for task in tasks:
        src = task["source"]
        for fmt_name, tgt in task["targets"]:
            try:
                o_hand = run_engine(a.engine, src, tgt, False)
                o_rank = run_engine(a.engine, src, tgt, True)
            except Exception as e:
                print("skip", task.get("name"), fmt_name, e)
                continue
            verdict = score_tile(src, o_rank, o_hand, tgt)
            fam = family_of(fmt_name, tgt["width"], tgt["height"])
            by_fam[fam][verdict] += 1
            totals[verdict] += 1
    print("totals", dict(totals))
    for fam, c in sorted(by_fam.items()):
        print(f"  {fam}: ranker wins {c['ranker']}  hand wins {c['hand']}  same {c['same']}")


if __name__ == "__main__":
    main()
