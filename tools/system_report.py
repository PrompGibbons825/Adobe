"""Dev tool: the inferred design system of a design, and how well layouts keep it.

    .venv/bin/python -m tools.system_report infer input/*.json tools/cases/*.json
    .venv/bin/python -m tools.system_report infer --fam 4,10,21
    .venv/bin/python -m tools.system_report compare --tags base,ws --split held

`compare` measures, per Crello family, the designer's own resize and each
evalset tag's output (tools/crello/eval/<tag>/) against the source's system:
type-ratio drift (vs the source ratios and vs the class-compressed ratios),
size-order flips, margin unit, in-group gap drift, groups split apart, rhythm
spread, alignment axes, chip proportions and copy kept on its surface.

Not part of the engine or the Docker image.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tools" / "crello" / "cases"
EVAL = ROOT / "tools" / "crello" / "eval"
sys.path.insert(0, str(ROOT))


def _system(src: dict):
    from engine.smart.analyze import analyze
    from engine.smart.system import infer

    return infer(analyze(json.loads(json.dumps(src))))


def cmd_infer(a):
    files = list(a.files)
    if a.fam:
        files += [str(CASES / f"fam{int(k):02d}.json") for k in a.fam.split(",")]
    for f in files:
        if "crello" in f:
            os.environ["CHAMELEON_BRAND_KIT"] = str(CASES)
        doc = json.loads(Path(f).read_text())
        src = doc.get("source", doc)
        sy = _system(src)
        print(f"{Path(f).stem} {int(sy.W)}x{int(sy.H)}: {sy.describe()}")
        for lv in sy.levels[::-1]:
            print(f"   level {lv.fs:6.1f}px  r={lv.ratio:.2f}  lh={lv.leading:.2f}  {lv.ids}")


KEYS = ["type_dev", "type_dev_c", "type_dev_des", "type_inv", "margin", "margin_dev", "margin_des", "tight_dev",
        "split_groups", "rhythm_cv", "axes", "axes_src", "chip_dev", "on_surface"]


def cmd_compare(a):
    os.environ["CHAMELEON_BRAND_KIT"] = str(CASES)
    from engine.smart.items import Ctx
    from engine.smart.system import instantiate, measure

    tags = [t for t in a.tags.split(",") if t]
    cols = ["designer"] + tags
    rows = {c: [] for c in cols}
    per_fam = []
    for f in sorted(CASES.glob("fam??.json")):
        k = int(f.stem[3:5])
        if a.split == "tune" and k % 2:
            continue
        if a.split == "held" and not k % 2:
            continue
        fam = json.loads(f.read_text())
        src, tgt = fam["source"], fam["targets"][0]
        W, H = tgt["canvas"]["width"], tgt["canvas"]["height"]
        sy = _system(src)
        ts = instantiate(sy, Ctx(W=W, H=H, W0=sy.W, H0=sy.H))
        line = [f.stem, f"{int(sy.W)}x{int(sy.H)}->{W}x{H}"]
        des_ratio = {e["id"]: float(e.get("font_size", 0) or 0) for e in tgt["elements"]
                     if e.get("type") == "text" and float(e.get("opacity", 1) or 0) > 0}
        des_margin = measure(sy, tgt["elements"], W, H)["margin"]
        for c in cols:
            if c == "designer":
                els = tgt["elements"]
            else:
                p = EVAL / c / f.name
                if not p.exists():
                    continue
                els = json.loads(p.read_text())["elements"]
            m = measure(sy, els, W, H)
            m["type_dev_c"] = measure(sy, els, W, H, want_ratio=ts.ratio)["type_dev"]
            m["type_dev_des"] = measure(sy, els, W, H, want_ratio=des_ratio)["type_dev"]
            m["margin_des"] = abs(math.log(m["margin"] / des_margin)) if m["margin"] > 0 and des_margin > 0 else float("nan")
            rows[c].append(m)
            line.append(f"{c}: td {m['type_dev']:.2f} inv {m['type_inv']:.0f} m {m['margin']:.3f} "
                        f"tg {m['tight_dev']:.2f} sp {m['split_groups']:.0f} ax {m['axes']:.0f}")
        per_fam.append(line)
    if a.verbose:
        for line in per_fam:
            print("  ".join(line))
    print(f"split={a.split}  n={len(rows['designer'])}  (means; split_groups, type_inv summed)")
    print("".ljust(10) + "".join(k[:11].rjust(12) for k in KEYS))
    for c in cols:
        rs = rows[c]
        if not rs:
            continue
        vals = []
        for k in KEYS:
            xs = [r[k] for r in rs if not (isinstance(r[k], float) and math.isnan(r[k]))]
            v = sum(xs) if k in ("split_groups", "type_inv") else (sum(xs) / len(xs) if xs else float("nan"))
            vals.append(v)
        print(c.ljust(10) + "".join(f"{v:12.3f}" for v in vals))
        if a.median:
            meds = []
            for k in KEYS:
                xs = sorted(r[k] for r in rs if not (isinstance(r[k], float) and math.isnan(r[k])))
                meds.append(xs[len(xs) // 2] if xs else float("nan"))
            print(("  median").ljust(10) + "".join(f"{v:12.3f}" for v in meds))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("infer")
    p.add_argument("files", nargs="*")
    p.add_argument("--fam", default="")
    p = sub.add_parser("compare")
    p.add_argument("--tags", default="base")
    p.add_argument("--split", default="held", choices=["held", "tune", "all"])
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--median", action="store_true")
    a = ap.parse_args()
    {"infer": cmd_infer, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
