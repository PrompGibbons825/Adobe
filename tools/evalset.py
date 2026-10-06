"""Dev tool: score engine versions on the Crello families with the proxy judge.

    .venv/bin/python -m tools.evalset run --engine <repo root of an engine version> --tag now
    .venv/bin/python -m tools.evalset score --tags v1,now,next --split held

Families with an even index are the tuning half, odd ones are held out (never
tuned against). Outputs land in tools/crello/eval/<tag>/famNN.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tools" / "crello" / "cases"
EVAL = ROOT / "tools" / "crello" / "eval"

RUNNER = r"""
import json, os, sys, io, contextlib
sys.path.insert(0, os.getcwd())
from engine.resize import resize
cases, out = sys.argv[1], sys.argv[2]
for f in sorted(os.listdir(cases)):
    if not f.startswith("fam") or not f.endswith(".json"):
        continue
    fam = json.load(open(os.path.join(cases, f)))
    tgt = fam["targets"][0]
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        o = resize(fam["source"], {"width": tgt["canvas"]["width"], "height": tgt["canvas"]["height"]})
    if "Traceback" in buf.getvalue():
        open(os.path.join(out, "fallbacks.txt"), "a").write(f + " " + buf.getvalue().strip().splitlines()[-1] + "\n")
    json.dump(o, open(os.path.join(out, f), "w"))
    json.dump({"source": fam["source"]}, open(os.path.join(out, f.replace(".json", ".task.json")), "w"))
"""


def split_of(name: str) -> str:
    k = int(name[3:5])
    return "tune" if k % 2 == 0 else "held"


def cmd_run(a):
    out = EVAL / a.tag
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CHAMELEON_BRAND_KIT=str(CASES))
    subprocess.run([sys.executable, "-c", RUNNER, str(CASES), str(out)], cwd=a.engine, env=env, check=True)
    print("wrote", len(list(out.glob("fam??.json"))), "outputs to", out)


def cmd_score(a):
    os.environ["CHAMELEON_PROXY_ASSETS"] = str(CASES / "assets")
    sys.path.insert(0, str(ROOT))
    import contextlib
    import io

    from tools import proxy

    src_cache = EVAL / "source_reports.json"
    src_reps = json.loads(src_cache.read_text()) if src_cache.exists() else {}

    def source_report(fam: str, task_path: str) -> dict:
        if fam not in src_reps:
            with contextlib.redirect_stderr(io.StringIO()):
                src_reps[fam] = proxy.score(task_path)
            src_cache.write_text(json.dumps(src_reps, default=str))
        return src_reps[fam]

    rows = {}
    for tag in a.tags.split(","):
        res = []
        for f in sorted((EVAL / tag).glob("fam??.json")):
            if a.split != "all" and split_of(f.stem) != a.split:
                continue
            task = str(f).replace(".json", ".task.json")
            with contextlib.redirect_stderr(io.StringIO()):
                agg = proxy.aggregate(str(f), task)
                rep = proxy.score(str(f))
            src = source_report(f.stem, task)
            # relative to the source: only what the resize made worse counts
            st = {t["id"]: t for t in src["texts"] if t.get("visible", True) and "wcag_p10" in t}
            live = [t for t in rep["texts"] if t.get("cap_px", 0) >= 3 and "wcag_p10" in t]
            lost_c = sum(1 for t in live if t["id"] in st and
                         t["wcag_p10"] < 0.85 * min(st[t["id"]]["wcag_p10"], 4.5))
            gained_busy = sum(1 for t in live if t["id"] in st and
                              t["busy_p90grad"] > max(1.5 * st[t["id"]]["busy_p90grad"], 0.03))
            src_pairs = {(c[0], c[1]) for c in src["collisions"] if c[3] > 0.05}
            src_pairs |= {(b, a_) for a_, b in src_pairs}
            new_coll = sum(1 for c in rep["collisions"] if c[3] > 0.1 and (c[0], c[1]) not in src_pairs)
            low = sum(1 for t in live if t["wcag_p10"] < 3.0)
            coll = sum(1 for c in rep["collisions"] if c[3] > 0.1)
            res.append(dict(agg, low_contrast_texts=low, visible_texts=len(live), collisions_gt10=coll,
                            contrast_lost=lost_c, busy_gained=gained_busy, new_collisions=new_coll))
        rows[tag] = res
        (EVAL / f"{tag}_{a.split}.json").write_text(json.dumps(res, indent=1))
    keys = ["score", "low_contrast", "text_on_busy", "tiny_text", "collisions", "dropped_text", "offcanvas",
            "aspect_violation", "imbalance"]
    print(f"split={a.split}  n={len(next(iter(rows.values())))}")
    print("raw proxy (tools/proxy.py aggregate, mean per design; collisions/busy count intended layering too)")
    print("tag".ljust(8) + "".join(k[:12].rjust(13) for k in keys))
    for tag, res in rows.items():
        n = max(1, len(res))
        print(tag.ljust(8) + "".join(f"{sum(r[k] for r in res) / n:13.3f}" for k in keys))
    if a.judge:
        judge_table(rows, a)
    print("\nrelative to the source (counts over all designs in the split)")
    print("tag".ljust(8) + "visible-texts".rjust(15) + "wcag<3".rjust(9) + "contrast-lost".rjust(15)
          + "busy-gained".rjust(13) + "new-collisions".rjust(16) + "all-collisions".rjust(16))
    for tag, res in rows.items():
        g = lambda k: sum(r[k] for r in res)  # noqa: E731
        print(tag.ljust(8) + f"{g('visible_texts'):15d}{g('low_contrast_texts'):9d}{g('contrast_lost'):15d}"
              f"{g('busy_gained'):13d}{g('new_collisions'):16d}{g('collisions_gt10'):16d}")


def judge_table(rows, a):
    """The calibrated source-relative proxy (tools/judge): mean penalty per design,
    and how often it prefers each tag's output over the first tag's."""
    import math

    os.environ["CHAMELEON_BRAND_KIT"] = str(CASES)
    from tools.judge import terms as T

    w = T.calibrated_weights()
    cache = EVAL / "judge_source_reports.json"
    src = json.loads(cache.read_text()) if cache.exists() else {}
    pens = {}
    for tag in rows:
        pens[tag] = {}
        for f in sorted((EVAL / tag).glob("fam??.json")):
            if a.split != "all" and split_of(f.stem) != a.split:
                continue
            if f.stem not in src:
                src[f.stem] = T.measure(json.loads(Path(str(f).replace(".json", ".task.json")).read_text())["source"])
                cache.write_text(json.dumps(src))
            pens[tag][f.stem] = T.penalty(T.terms(src[f.stem], T.measure(json.loads(f.read_text()))), w)
    first = next(iter(rows))
    print("\ncalibrated judge (tools/judge/data/weights.json; lower penalty is better)")
    print("tag".ljust(8) + "penalty".rjust(10) + "  P(judge prefers it over " + first + "), mean   better/worse/same designs")
    for tag, pp in pens.items():
        common = [k for k in pp if k in pens[first]]
        d = [pens[first][k] - pp[k] for k in common]
        pw = sum(1 / (1 + math.exp(-x)) for x in d) / max(1, len(d))
        print(tag.ljust(8) + f"{sum(pp.values()) / max(1, len(pp)):10.3f}  {pw:.3f}"
              f"{sum(x > 1e-6 for x in d):24d}/{sum(x < -1e-6 for x in d)}/{sum(abs(x) <= 1e-6 for x in d)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--engine", required=True)
    p.add_argument("--tag", required=True)
    p = sub.add_parser("score")
    p.add_argument("--tags", required=True)
    p.add_argument("--split", default="held", choices=["held", "tune", "all"])
    p.add_argument("--judge", action="store_true", help="also score with the calibrated judge (tools/judge)")
    a = ap.parse_args()
    {"run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
