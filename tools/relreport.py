"""Dev tool: how much of each source design's relation graph the outputs keep,
and how aligned / evenly spaced they look (engine/smart/relations.py).

    .venv/bin/python -m tools.relreport --tags base,ws --split held

Reads tools/crello/eval/<tag>/famNN.json (from tools.evalset run). Prints, per
tag: relation survival (mean and histogram), the share of relations of each
type that still hold exactly, and the look metrics next to the sources' own.
Not part of the engine or the Docker image.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tools" / "crello" / "cases"
EVAL = ROOT / "tools" / "crello" / "eval"
BINS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)


def split_of(name: str) -> str:
    return "tune" if int(name[3:5]) % 2 == 0 else "held"


def rows_for_tag(tag: str, split: str):
    from engine.smart.analyze import analyze
    from engine.smart.relations import evaluate, extract

    out = []
    for f in sorted((EVAL / tag).glob("fam??.json")):
        if split != "all" and split_of(f.stem) != split:
            continue
        o = json.loads(f.read_text())
        src = json.loads(Path(str(f).replace(".json", ".task.json")).read_text())["source"]
        with contextlib.redirect_stderr(io.StringIO()):
            d = analyze(src)
            g = extract(d)
            ev = evaluate(g, d, o["elements"], float(o["canvas"]["width"]), float(o["canvas"]["height"]))
        ev["fam"] = f.stem
        ev.update({"src_" + k: v for k, v in g.look.items()})
        out.append(ev)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--split", default="held", choices=("held", "tune", "all"))
    ap.add_argument("--per-design", action="store_true")
    a = ap.parse_args()
    os.environ["CHAMELEON_BRAND_KIT"] = str(CASES)
    sys.path.insert(0, str(ROOT))
    res = {t: rows_for_tag(t, a.split) for t in a.tags.split(",")}
    n = len(next(iter(res.values())))
    print(f"split={a.split}  n={n}")
    hdr = "tag".ljust(8) + "survival".rjust(10) + "raw".rjust(8) + "   histogram " + " ".join(
        f"<{b:.1f}" if b < 1 else "<=1" for b in BINS[1:])
    print(hdr)
    for t, rows in res.items():
        sv = [r["survival"] for r in rows]
        hist = [sum(1 for v in sv if lo <= v < hi) for lo, hi in zip(BINS, BINS[1:])]
        print(t.ljust(8) + f"{sum(sv) / len(sv):10.3f}{sum(r['raw'] for r in rows) / len(rows):8.3f}   "
              + "           " + " ".join(f"{h:4d}" for h in hist))
    kinds = ["contain", "read", "hier", "align", "eqsize", "order", "band", "gap"]
    print("\nshare of relations holding exactly, per type (mean over designs that have the type)")
    print("tag".ljust(8) + "".join(k.rjust(9) for k in kinds))
    for t, rows in res.items():
        cells = []
        for k in kinds:
            v = [r["held_" + k] for r in rows if "held_" + k in r]
            cells.append(f"{sum(v) / len(v):9.3f}" if v else "        -")
        print(t.ljust(8) + "".join(cells))
    looks = ["near_miss", "x_lines", "gap_cv", "margin_x", "margin_y", "density"]
    print("\nlook metrics (mean; lower is better except density, which should track the source)")
    print("tag".ljust(8) + "".join(k.rjust(11) for k in looks))
    first = next(iter(res.values()))
    print("source".ljust(8) + "".join(f"{sum(r['src_' + k] for r in first) / len(first):11.3f}" for k in looks))
    for t, rows in res.items():
        print(t.ljust(8) + "".join(f"{sum(r['look_' + k] for r in rows) / len(rows):11.3f}" for k in looks))
    if a.per_design:
        print("\nper design: survival per tag")
        fams = [r["fam"] for r in first]
        for i, fam in enumerate(fams):
            print(fam.ljust(8) + "".join(f"{res[t][i]['survival']:8.3f}" for t in res)
                  + "   near_miss " + " ".join(f"{res[t][i]['look_near_miss']:.2f}" for t in res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
