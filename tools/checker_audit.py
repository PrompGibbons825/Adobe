"""Checker audit (L0): tools/rules.py (shares the engine's text model) vs fundamentals.check
(renderer model + rendered pixels), on the same outputs. Dev only.

    .venv/bin/python -m tools.checker_audit --stress 300 --seed 1          # this tree's engine on stress designs
    .venv/bin/python -m tools.checker_audit --bench main,sprint             # outputs already on the bench
    ... --json /tmp/audit.json                                               # every disagreement, for review

Faults are compared per (element id, category). Each disagreement is a real fault one checker
misses or a checker bug; the summary counts them per category and side.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundamentals.check import check  # noqa: E402
from tools.rules import capacity_errors, hard_rule_errors, promise_errors  # noqa: E402

# rules.py message -> category
RULE_CATS = [
    (r"canvas .* != ", "canvas"), (r"ids differ", "ids"), (r"type changed", "schema"), (r": bad \[", "schema"),
    (r"negative size", "schema"), (r"(content|font_family|asset_id|shape_type) changed", "copy_changed"),
    (r"font_size ", "schema"), (r"empty text box", "render_crash"), (r"text_align", "schema"),
    (r"font_weight", "schema"), (r"is not rgba", "colour_format"), (r"colour changed", "colour_changed"),
    (r"image collapsed", "aspect"), (r"aspect ", "aspect"), (r"off-canvas", "off_canvas"),
    (r"small print", "small_print"), (r"glyphs drawn past the canvas", "text_cut"),
    (r"copy over copy", "copy_over_copy"), (r"label left its shape", "label_left_shape"),
    (r"capacity:", "capacity"),
]
FUND_CATS = {
    "canvas": "canvas", "id_missing": "ids", "id_added": "ids", "id_duplicate": "ids", "schema": "schema",
    "content_changed": "copy_changed", "font_family_changed": "copy_changed", "asset_id_changed": "copy_changed",
    "shape_type_changed": "copy_changed", "type_changed": "schema", "render_crash": "render_crash",
    "colour_changed": "colour_changed", "shape_colour": "colour_format", "aspect": "aspect",
    "image_collapsed": "aspect", "distorted": "aspect", "off_canvas": "off_canvas", "small_print": "small_print",
    "text_cut": "text_cut", "copy_over_copy": "copy_over_copy", "hidden": "hidden", "contrast": "contrast",
}


def rule_items(src, out, tgt):
    errs = hard_rule_errors(src, out, tgt) + promise_errors(src, out, tgt) + capacity_errors(out, tgt)
    items = set()
    for m in errs:
        cat = next((c for pat, c in RULE_CATS if re.search(pat, m)), "other")
        ids = re.findall(r"copy over copy: (\S+) x (\S+)", m)
        if ids:
            for i in ids[0]:
                items.add((i, cat, m))
            continue
        mid = m.split(":")[1].strip() if m.startswith("capacity:") else m.split(":")[0].strip()
        items.add((mid, cat, m))
    return items


def fund_items(src, out, tgt, assets):
    items = set()
    for f in check(src, out, tgt, pixels=True, assets=assets):
        if f.level == "info":
            if f.code == "small_print":
                items.add((f.id or "-", "capacity", str(f)))
            continue
        cat = FUND_CATS.get(f.code, f.code)
        if f.code == "copy_over_copy":
            for i in re.findall(r"of (\S+) and (\S+) ", f.msg)[0]:
                items.add((i, cat, str(f)))
            continue
        items.add((f.id or "-", cat, str(f)))
    return items


def compare(name, src, out, tgt, assets, rows):
    try:
        r = rule_items(src, out, tgt)
    except Exception as x:  # noqa: BLE001
        r = {("-", "rules_crash", f"{type(x).__name__}: {x}")}
    try:
        f = fund_items(src, out, tgt, assets)
    except Exception as x:  # noqa: BLE001
        f = {("-", "fund_crash", f"{type(x).__name__}: {x}")}
    rk = {(i, c): m for i, c, m in r}
    fk = {(i, c): m for i, c, m in f}
    for k in set(rk) | set(fk):
        side = "both" if k in rk and k in fk else ("rules_only" if k in rk else "fund_only")
        rows.append(dict(task=name, id=k[0], cat=k[1], side=side, rules=rk.get(k), fund=fk.get(k)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bench", default="", help="comma-separated bench tags")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    logging.disable(logging.WARNING)
    rows, n, t0 = [], 0, time.time()
    if a.stress:
        from engine.resize import resize
        from tools.sheet import CATALOGUE
        from tools.stress import random_design

        rng = random.Random(a.seed)
        targets = list(CATALOGUE.values())
        for i in range(a.stress):  # the same draws as tools.stress --seed
            src = random_design(rng)
            W, H = rng.choice(targets)
            tgt = {"width": W, "height": H}
            buf = io.StringIO()
            try:
                with redirect_stderr(buf):
                    out = resize(json.loads(json.dumps(src)), tgt)
            except Exception:  # noqa: BLE001
                continue
            compare(f"stress{i}", src, out, tgt, None, rows)
            n += 1
    for tag in [t for t in a.bench.split(",") if t]:
        from tools.bench import BENCH, CASES, load_tasks

        tasks = load_tasks()
        res = json.loads((BENCH / tag / "results.json").read_text())
        for k, (name, r) in enumerate(sorted(res.items())):
            if a.limit and k >= a.limit:
                break
            if not r.get("out") or name not in tasks:
                continue
            t = tasks[name]
            compare(f"{tag}:{name}", t["source"], r["out"], t["target"], CASES / "assets", rows)
            n += 1
    c = Counter((r["cat"], r["side"]) for r in rows)
    cats = sorted({r["cat"] for r in rows})
    print(f"checker audit: {n} outputs ({time.time() - t0:.0f}s)")
    print(f"{'category':18s} {'both':>6s} {'rules_only':>11s} {'fund_only':>10s}")
    for cat in cats:
        print(f"{cat:18s} {c[(cat, 'both')]:6d} {c[(cat, 'rules_only')]:11d} {c[(cat, 'fund_only')]:10d}")
    ex = defaultdict(list)
    for r in rows:
        if r["side"] != "both" and len(ex[(r["cat"], r["side"])]) < 3:
            ex[(r["cat"], r["side"])].append(r)
    for (cat, side), rs in sorted(ex.items()):
        for r in rs:
            print(f"  {cat}/{side}: {r['task']} {r['id']}: {(r['rules'] or r['fund'])[:150]}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
