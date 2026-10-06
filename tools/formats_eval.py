"""Dev tool: measure the formats-decor workstream (profiles, zones, decorations, chip sets).

    .venv/bin/python -m tools.formats_eval hits
    .venv/bin/python -m tools.formats_eval run --engine <engine root> --tag base
    .venv/bin/python -m tools.formats_eval score --tags base,ws

``hits``  profile hit rates (exact / near / class) on the size catalogues.
``run``   resizes three design sets with the engine at <root> (outputs land in
          tools/crello/eval/fmt_<tag>/, git-ignored):
            zones  samples + tools/cases + Crello sources -> 7 formats with platform UI
            chips  synthetic designs with 1-7 chips (row and column sources) + Level 3
                   -> the 20 sheet formats
            decor  the stress generator's random designs (decorative shapes) -> 3 formats
``score`` reads every tag with the *current* analyzer, so roles, decorations and
          chip sets are the same for every engine version:
            critical (headline/CTA/logo) and other copy inside avoid rects;
            decoration anchor (corner / edge-centre / centre) and edge contact kept;
            chip sets: one font size, equal heights, equal gaps, no orphan row.

Not part of the engine or the Docker image.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVAL = ROOT / "tools" / "crello" / "eval"
CASES = ROOT / "tools" / "crello" / "cases"
os.environ.setdefault("CHAMELEON_BRAND_KIT", str(CASES))  # Crello assets; the Solstice kit is found via cwd
ZONE_TARGETS = [(1080, 1920), (2560, 1440), (1280, 720), (1500, 500), (1584, 396), (1128, 191),
                (1170, 2532)]
DECOR_TARGETS = [(1600, 400), (1080, 1920), (300, 250)]

# ----------------------------------------------------------------------------
# design sets

LABELS = ["NEW", "LIMITED EDITION", "AWARD WINNING 2026", "SMALL BATCH", "VEGAN", "FREE SHIPPING", "ORGANIC",
          "HANDMADE", "TOP RATED", "GIFT READY"]


def chip_design(n: int, arrangement: str, seed: int) -> dict:
    rng = random.Random(seed)
    W = H = 1080
    els = [{"id": "bg", "type": "image", "x": 0, "y": 0, "width": W, "height": H, "z_index": 0,
            "asset_id": "solstice_bg_gradient.png", "fit": "cover"},
           {"id": "logo", "type": "image", "x": 380, "y": 60, "width": 320, "height": 100, "z_index": 2,
            "asset_id": "solstice_logo_lockup.png", "fit": "contain"},
           {"id": "headline", "type": "text", "x": 90, "y": 220, "width": 900, "height": 150, "z_index": 2,
            "content": "Chase the Light", "font_family": "Solstice Serif", "font_size": 64, "font_weight": "bold",
            "color": "rgba(255,255,255,1)", "text_align": "center"},
           {"id": "cta_bg", "type": "shape", "x": 330, "y": 940, "width": 420, "height": 90, "z_index": 3,
            "shape_type": "rectangle", "fill_color": "rgba(232,106,58,1)", "corner_radius": 45, "group_id": "cta"},
           {"id": "cta_label", "type": "text", "x": 330, "y": 965, "width": 420, "height": 44, "z_index": 4,
            "content": "Shop the Summer Line", "font_family": "Solstice Sans", "font_size": 22,
            "font_weight": "bold", "color": "rgba(255,255,255,1)", "text_align": "center", "group_id": "cta"}]
    labels = rng.sample(LABELS, n)
    h, fs, gap = 50, 15, 20
    widths = [max(120, 14 * len(t) + 60) for t in labels]
    if arrangement == "row":
        scale = min(1.0, (W - 120 - gap * (n - 1)) / sum(widths))
        widths = [w * scale for w in widths]
        x, y = (W - sum(widths) - gap * (n - 1)) / 2, 480
    else:
        x, y = 90, 420
    for i, (t, w) in enumerate(zip(labels, widths)):
        els.append({"id": f"chip{i}_bg", "type": "shape", "x": x, "y": y, "width": w, "height": h, "z_index": 3,
                    "shape_type": "rectangle", "fill_color": rng.choice(["rgba(232,106,58,1)", "rgba(43,27,46,1)"]),
                    "corner_radius": 25, "group_id": f"chip{i}"})
        els.append({"id": f"chip{i}_label", "type": "text", "x": x, "y": y + 12, "width": w, "height": 26,
                    "z_index": 4, "content": t, "font_family": "Solstice Sans", "font_size": fs,
                    "font_weight": "bold", "color": "rgba(255,255,255,1)", "text_align": "center",
                    "group_id": f"chip{i}"})
        if arrangement == "row":
            x += w + gap
        else:
            y += h + 16
    return {"canvas": {"width": W, "height": H}, "elements": els}


def design_sets() -> dict:
    from tools.sheet import CATALOGUE
    from tools.stress import random_design

    sets = {"zones": [], "chips": [], "decor": []}
    srcs = []
    for f in sorted((ROOT / "input").glob("*.json")) + sorted((ROOT / "tools" / "cases").glob("*.json")):
        srcs.append((f.stem[-24:], json.loads(f.read_text())["source"]))
    for f in sorted(CASES.glob("fam*.json")):
        srcs.append((f.stem, json.loads(f.read_text())["source"]))
    for name, src in srcs:
        for W, H in ZONE_TARGETS:
            sets["zones"].append((f"{name}@{W}x{H}", src, (W, H)))
    l3 = json.loads((ROOT / "input" / "chameleon_sample_level3_advanced.json").read_text())["source"]
    chips = [("level3", l3)] + [(f"chips{n}{a[0]}", chip_design(n, a, 100 * n + len(a)))
                                for n in range(1, 8) for a in ("row", "column")]
    for name, src in chips:
        for W, H in CATALOGUE.values():
            sets["chips"].append((f"{name}@{W}x{H}", src, (W, H)))
    rng = random.Random(11)
    for i in range(80):
        src = random_design(rng)
        for W, H in DECOR_TARGETS:
            sets["decor"].append((f"rand{i:02d}@{W}x{H}", src, (W, H)))
    return sets


# ----------------------------------------------------------------------------

RUNNER = r"""
import json, os, sys, io, contextlib
sys.path.insert(0, sys.argv[1])
from engine.resize import resize
jobs = json.load(open(sys.argv[2]))
out = sys.argv[3]
for name, src, (W, H) in jobs:
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(io.StringIO()):
        o = resize(json.loads(json.dumps(src)), {"width": W, "height": H})
    o["_fallback"] = "Traceback" in buf.getvalue()
    json.dump(o, open(os.path.join(out, name.replace("/", "_") + ".json"), "w"))
"""


def cmd_run(a):
    sets = design_sets()
    for set_name, jobs in sets.items():
        out = EVAL / f"fmt_{a.tag}" / set_name
        out.mkdir(parents=True, exist_ok=True)
        jf = out / "_jobs.json"
        jf.write_text(json.dumps(jobs))
        env = dict(os.environ, CHAMELEON_BRAND_KIT=str(CASES))
        subprocess.run([sys.executable, "-c", RUNNER, str(Path(a.engine).resolve()), str(jf), str(out)],
                       cwd=str(ROOT), env=env, check=True)
        jf.unlink()
        print(set_name, len(jobs), "->", out)


# ----------------------------------------------------------------------------


def cmd_hits(_a):
    from engine.smart.formats import SIZES, profile_for

    rel = Path("context") / "research" / "scripts" / "formats" / "express_canvas.json"
    found = [p / rel for p in (ROOT.parent, ROOT.parents[1]) if (p / rel).exists()]
    ex = json.loads(found[0].read_text()) if found else None
    groups = defaultdict(list)
    if ex:
        for _n, task, w, h, unit, _ar in ex:
            if unit == "PIXELS":
                groups["express pixel presets"].append((w, h))
            else:
                k = 25.4 if unit == "MILLIMETERS" else 1.0
                for dpi in (72, 96, 150, 300):
                    groups[f"express physical @{dpi}dpi"].append((round(w / k * dpi), round(h / k * dpi)))
    for f in sorted(CASES.glob("fam*.json")):
        fam = json.loads(f.read_text())
        groups["crello sources"].append((fam["source"]["canvas"]["width"], fam["source"]["canvas"]["height"]))
        for t in fam["targets"]:
            groups["crello designer targets"].append((t["canvas"]["width"], t["canvas"]["height"]))
    from tools.sheet import CATALOGUE

    groups["sheet catalogue"] = list(CATALOGUE.values())
    base = list(SIZES)
    groups["table sizes x1.01"] = [(round(w * 1.01), round(h * 1.01)) for w, h in base]
    groups["table sizes x0.985 w"] = [(round(w * 0.985), h) for w, h in base]
    groups["table sizes x1.05"] = [(round(w * 1.05), round(h * 1.05)) for w, h in base]
    print(f"{'set':28s}{'n':>5s}{'exact':>7s}{'near':>6s}{'class':>7s}{'zones':>7s}")
    for name, sizes in groups.items():
        c = Counter(profile_for(w, h).match for w, h in sizes)
        z = sum(1 for w, h in sizes if profile_for(w, h).avoid)
        print(f"{name:28s}{len(sizes):5d}{c['exact']:7d}{c['near']:6d}{c['class']:7d}{z:7d}")
    print("\nclasses of the table:", dict(Counter(profile_for(w, h).cls for w, h in SIZES)))


# ----------------------------------------------------------------------------


def _roles(src: dict) -> dict:
    """element id -> (role, critical) from the current analyzer."""
    from engine.smart.analyze import analyze

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        d = analyze(src)
    out = {}
    for it in d.items:
        for e in it.els:
            if it.role == "panel" or it.kind == "composite":
                crit = e.get("type") == "text" and it.core
                out[e["id"]] = ("copy" if e.get("type") == "text" else "art", crit)
            elif it.role in ("headline", "cta", "logo"):
                out[e["id"]] = (it.role, e.get("type") == "text" or it.role == "logo" or it.role == "cta")
            else:
                out[e["id"]] = ("copy" if e.get("type") == "text" else "art", False)
    return d, out


def _visible(e: dict) -> bool:
    return float(e.get("opacity", 1) or 0) > 0.01 and e.get("width", 0) > 0 and e.get("height", 0) > 0


def score_zones(tag: str, jobs, cache) -> dict:
    from engine.smart.formats import profile_for
    from engine.smart.geom import Box, el_visual_box

    res = Counter()
    per = defaultdict(Counter)
    for name, src, (W, H) in jobs:
        p = EVAL / f"fmt_{tag}" / "zones" / (name + ".json")
        if not p.exists():
            continue
        out = json.loads(p.read_text())
        res["designs"] += 1
        res["fallbacks"] += bool(out.get("_fallback"))
        key = "roles:" + name.split("@")[0]
        if key not in cache:
            cache[key] = _roles(src)[1]
        roles = cache[key]
        zones = profile_for(W, H).avoid_boxes(W, H)
        for e in out["elements"]:
            if not _visible(e) or e["id"] not in roles:
                continue
            role, crit = roles[e["id"]]
            if not crit and e.get("type") != "text":
                continue
            b = el_visual_box(e)
            frac = sum(b.inter(z) for _, z, _ in zones) / max(1.0, b.area)
            kind = "critical" if crit else "other copy"
            res[kind] += 1
            if frac >= 0.1:
                res[kind + " in zone"] += 1
                per[f"{W}x{H}"][kind] += 1
    return {"total": dict(res), "per_target": {k: dict(v) for k, v in per.items()}}


def _ink(e: dict):
    from engine.smart.decor import _ink_box

    return _ink_box(e)


def score_decor(tag: str, jobs_by_set, cache) -> dict:
    from engine.smart.analyze import analyze
    from engine.smart.decor import anchor

    res = Counter()
    for set_name, jobs in jobs_by_set.items():
        for name, src, (W, H) in jobs:
            p = EVAL / f"fmt_{tag}" / set_name / (name + ".json")
            if not p.exists():
                continue
            out = {e["id"]: e for e in json.loads(p.read_text())["elements"]}
            key = "decor:" + name.split("@")[0] + set_name
            if key not in cache:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    d = analyze(src)
                cache[key] = [e for e in d.decor if not e.get("_slab") and e.get("type") in ("shape", "image")
                              and e["width"] > 0 and e["height"] > 0], d.W, d.H
            decor, W0, H0 = cache[key]
            for e in decor:
                v0 = _ink(e)
                if v0.area <= 0:
                    continue
                o = out.get(e["id"])
                if o is None or not _visible(o):
                    res["hidden"] += 1
                    continue
                a0 = anchor(v0, W0, H0)
                a1 = anchor(_ink(o), W, H)
                same = a0[:2] == a1[:2]
                contact = a0[2] == a1[2]
                res["decorations"] += 1
                res["anchor kept"] += same
                res["edge contact kept"] += contact
                res["both kept"] += same and contact
                if any(a0[2]):
                    res["edge-touching"] += 1
                    res["edge-touching, contact kept"] += contact
    return dict(res)


def score_chips(tag: str, jobs, cache) -> dict:
    from engine.smart.analyze import analyze
    from engine.smart.geom import el_box

    res = Counter()
    for name, src, (W, H) in jobs:
        p = EVAL / f"fmt_{tag}" / "chips" / (name + ".json")
        if not p.exists():
            continue
        out = {e["id"]: e for e in json.loads(p.read_text())["elements"]}
        key = "chips:" + name.split("@")[0]
        if key not in cache:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                d = analyze(src)
            cache[key] = [[(it.container["id"], [l["id"] for l in it.labels]) for it in g if it.container is not None]
                          for g in getattr(d, "repeats", [])]
        for g in cache[key]:
            vis = [(c, ls) for c, ls in g if _visible(out[c])]
            if len(vis) < 2:
                res["sets hidden"] += 1 if not vis else 0
                res["sets with <2 visible"] += 1
                continue
            res["sets"] += 1
            fonts = {out[l]["font_size"] for _, ls in vis for l in ls}
            boxes = [el_box(out[c]) for c, _ in vis]
            hs = [b.h for b in boxes]
            # rows by centre line
            rows = []
            for b in sorted(boxes, key=lambda b: (b.cy, b.x)):
                if rows and abs(rows[-1][0].cy - b.cy) <= 0.5 * b.h:
                    rows[-1].append(b)
                else:
                    rows.append([b])
            gaps = []
            for r in rows:
                r.sort(key=lambda b: b.x)
                gaps += [b.x - a.r for a, b in zip(r, r[1:])]
            vg = [b[0].y - a[0].b for a, b in zip(rows, rows[1:])]
            one_font = len(fonts) == 1
            eq_h = max(hs) - min(hs) <= 1.0
            eq_gap = (max(gaps) - min(gaps) <= 2.0) if len(gaps) > 1 else True
            orphan = len(rows) > 1 and min(len(r) for r in rows) == 1 and len(vis) > len(rows)
            res["one font"] += one_font
            res["equal heights"] += eq_h
            res["equal gaps"] += eq_gap
            res["orphan row"] += orphan
            res["uniform (all four)"] += one_font and eq_h and eq_gap and not orphan
            res["arr:" + ("row" if len(rows) == 1 else ("column" if all(len(r) == 1 for r in rows) else "grid"))] += 1
    return dict(res)


def cmd_score(a):
    sets = design_sets()
    cache: dict = {}
    for tag in a.tags.split(","):
        print(f"\n=== {tag}")
        z = score_zones(tag, sets["zones"], cache)
        t = z["total"]
        print(f"zones: {t.get('designs', 0)} outputs ({t.get('fallbacks', 0)} fallbacks); critical in avoid rect "
              f"{t.get('critical in zone', 0)}/{t.get('critical', 0)}, other copy {t.get('other copy in zone', 0)}"
              f"/{t.get('other copy', 0)}")
        for k in sorted(z["per_target"]):
            print(f"    {k:>10s}  {z['per_target'][k]}")
        dc = score_decor(tag, {"decor": sets["decor"], "zones": sets["zones"], "chips": sets["chips"]}, cache)
        print("decor:", dc)
        ch = score_chips(tag, sets["chips"], cache)
        print("chips:", ch)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hits")
    p = sub.add_parser("run")
    p.add_argument("--engine", required=True)
    p.add_argument("--tag", required=True)
    p = sub.add_parser("score")
    p.add_argument("--tags", required=True)
    a = ap.parse_args()
    {"hits": cmd_hits, "run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
