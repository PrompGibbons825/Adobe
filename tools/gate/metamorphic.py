"""Metamorphic checks: transformations of the input that must not change the answer.

    loop/capped .venv/bin/python -m tools.gate.metamorphic --engine <root> [--base <root>] [--json out.json]

The engine root runs in its own process (``tools/gate/engine_run.py``); the comparisons run
here. The designs are dev data only: the samples, ``tools/cases`` and a few ``tools.designs``
designs, each on its own target plus a handful of real formats. This extends
``tests/test_engine.py`` ``SameShape`` (identity / uniform scale) to properties the engine has
no reason to break:

* ``scale2x``      the source at twice its size (geometry, type, radii, strokes, spacing) gives
                   the same output for the same target (positions within ``TOL_PX``, the same
                   font sizes).
* ``permutation``  the element list in another order, every ``z_index`` kept (and the order of
                   equal-z elements kept, since that order paints), gives a byte-identical output
                   once elements are sorted by id.
* ``rename``       random ids give the byte-identical output after mapping the ids back.
* ``determinism``  the same jobs in a second process with another PYTHONHASHSEED give
                   byte-identical outputs.
* ``nonlatin``     copy swapped for non-Latin scripts never crashes, never falls back, keeps the
                   hard rules, renders, and hides no copy the Latin original showed.
* ``continuity``   a target 0.1% wider (at least 1 px) moves no element more than 15% of the
                   canvas. This one is a count: it must not rise against ``--base``.

Checks named in ``REPORT_ONLY`` fail on main for a reason that is a known engine weakness; they
are reported with their counts but do not fail the run (see each entry's note).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tools.gate.engine_run import GATE_ROOT, canonical, dev_tasks, run_jobs

FORMATS = [(1080, 1920), (728, 90), (300, 250)]
TOL_PX = 1.0
JUMP = 0.15
LAYOUT = 0.02  # a scale2x difference above 2% of the canvas, or a shown/hidden flip, is a re-layout
PX_KEYS = ("x", "y", "width", "height", "font_size", "corner_radius", "stroke_width", "letter_spacing")
SCRIPTS = [
    "夏の夕暮れ、ボトルの中に 金色の時間 今すぐ購入",
    "ساعة ذهبية في زجاجة اطلب الآن عرض الصيف",
    "Золотой час в бутылке закажите сейчас летняя коллекция",
    "सुनहरा पल बोतल में अभी खरीदें गर्मी का संग्रह",
    "שעת הזהב בבקבוק הזמינו עכשיו קולקציית קיץ",
    "ชั่วโมงทองในขวด สั่งซื้อเลย คอลเลกชันฤดูร้อน",
    "황금 시간 병 속에 지금 주문하세요 여름 컬렉션",
    "☀ 🍊 Ελληνικά καλοκαίρι — 夏天 🌅",
]
# check -> why it is report-only on main (a finding, not a gate failure)
REPORT_ONLY: Dict[str, str] = {
    "scale2x": "main (df0bf7d, 10-24) is not scale-invariant: 25/56 dev tiles differ at 2x. Most by 1-5 px "
               "(rounding), but some re-lay out (level3@1080x1920 hides legal_text, story_hero_top@1200x628 "
               "moves 260 px, band_panel@1600x900 126 px): some engine thresholds are absolute px in "
               "source units. A finding, not a gate failure; its re-laid-out count (4 on main) is the one to watch.",
}


# ----------------------------------------------------------------------------
# transformations (pure, unit-tested in tests/test_gate.py)


def scale_source(src: dict, k: float) -> dict:
    """The design at ``k`` times its size: canvas, geometry and every px-valued property."""
    s = json.loads(json.dumps(src))
    s["canvas"] = {"width": round(src["canvas"]["width"] * k), "height": round(src["canvas"]["height"] * k)}
    for e in s["elements"]:
        for key in PX_KEYS:
            v = e.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                e[key] = round(v * k) if isinstance(v, int) and float(v * k).is_integer() else v * k
    return s


def permute_source(src: dict, rng: random.Random) -> dict:
    """The element list shuffled, keeping every z_index and the relative order of equal-z
    elements (the renderer paints those in list order)."""
    s = json.loads(json.dumps(src))
    els = s["elements"]
    order = list(range(len(els)))
    rng.shuffle(order)
    # within each z value, put the original elements back in their original order
    by_z: Dict[object, List[int]] = {}
    for i in range(len(els)):
        by_z.setdefault(_z(els[i]), []).append(i)
    queues = {z: iter(ix) for z, ix in by_z.items()}
    s["elements"] = [els[next(queues[_z(els[i])])] for i in order]
    return s


def _z(e: dict):
    z = e.get("z_index")
    return z if isinstance(z, (int, float)) and not isinstance(z, bool) else None


def rename_source(src: dict, rng: random.Random) -> Tuple[dict, Dict[str, str]]:
    """Every id replaced by a random opaque one; returns the source and new->old."""
    s = json.loads(json.dumps(src))
    back: Dict[str, str] = {}
    used = set()
    for e in s["elements"]:
        while True:
            new = "".join(rng.choice("bcdfghjklmnpqrstvwxz") for _ in range(6)) + str(rng.randint(10, 99))
            if new not in used:
                break
        used.add(new)
        back[new] = e["id"]
        e["id"] = new
    return s, back


def map_ids(out: Optional[dict], back: Dict[str, str]) -> Optional[dict]:
    if out is None:
        return None
    o = json.loads(json.dumps(out))
    for e in o.get("elements", []):
        e["id"] = back.get(e.get("id"), e.get("id"))
    return o


def nonlatin_source(src: dict, offset: int = 0) -> dict:
    """Every text's copy swapped for a non-Latin string of about the same length."""
    s = json.loads(json.dumps(src))
    i = offset
    for e in s["elements"]:
        if e.get("type") != "text" or not str(e.get("content", "")).strip():
            continue
        base, n = SCRIPTS[i % len(SCRIPTS)], len(e["content"])
        i += 1
        e["content"] = (base * (n // len(base) + 1))[:max(1, n)].strip() or base[0]
        runs = [r for r in e.get("style_runs") or [] if isinstance(r, dict)]
        if runs:
            L = len(e["content"])
            e["style_runs"] = [dict(r, start=min(int(r.get("start", 0)), L), end=min(int(r.get("end", 0)), L))
                               for r in runs]
    return s


def max_px_diff(a: dict, b: dict) -> Tuple[float, List[str]]:
    """Largest px difference between two outputs over PX_KEYS, plus non-px mismatches."""
    if a is None or b is None:
        return float("inf"), ["missing output"]
    ia, ib = {e["id"]: e for e in a["elements"]}, {e["id"]: e for e in b["elements"]}
    if set(ia) != set(ib) or a.get("canvas") != b.get("canvas"):
        return float("inf"), ["ids or canvas differ"]
    worst, other = 0.0, []
    for i, ea in ia.items():
        eb = ib[i]
        for key in set(ea) | set(eb):
            va, vb = ea.get(key), eb.get(key)
            if key in PX_KEYS and isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                worst = max(worst, abs(va - vb))
            elif key == "opacity" and isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                if abs(va - vb) > 1e-6:
                    other.append(f"{i}.opacity {va} vs {vb}")
            elif va != vb:
                other.append(f"{i}.{key} {va!r} vs {vb!r}")
    return worst, other


def jumps(a: Optional[dict], b: Optional[dict]) -> List[str]:
    """Elements whose box moved or resized by more than JUMP of the canvas between two
    near-identical targets, or that flipped between shown and hidden."""
    if a is None or b is None:
        return ["missing output"]
    W, H = a["canvas"]["width"], a["canvas"]["height"]
    ib = {e["id"]: e for e in b["elements"]}
    out = []
    for ea in a["elements"]:
        eb = ib.get(ea["id"])
        if eb is None:
            out.append(f"{ea['id']}: missing")
            continue
        va, vb = float(ea.get("opacity", 1) or 0) > 0, float(eb.get("opacity", 1) or 0) > 0
        if va != vb:
            out.append(f"{ea['id']}: {'hidden' if va else 'shown'}")
            continue
        if not va:
            continue
        d = max(abs(ea["x"] - eb["x"]) / W, abs(ea["y"] - eb["y"]) / H,
                abs(ea["width"] - eb["width"]) / W, abs(ea["height"] - eb["height"]) / H)
        if d > JUMP:
            out.append(f"{ea['id']}: moved {d:.0%} of the canvas")
    return out


def _shown_copy(out: Optional[dict]) -> set:
    if not out:
        return set()
    return {e["id"] for e in out["elements"] if e.get("type") == "text"
            and float(e.get("opacity", 1) or 0) > 0 and str(e.get("content", "")).strip()}


# ----------------------------------------------------------------------------
# the run


def build_jobs(n_generated: int, seed: int, formats=FORMATS):
    tiles = []  # (tile name, source, target)
    for t in dev_tasks(n_generated, seed=seed):
        targets = [t["target_canvas"]] + [{"width": w, "height": h} for w, h in formats]
        seen = set()
        for tgt in targets:
            key = (tgt["width"], tgt["height"])
            if key in seen or key == (t["source"]["canvas"]["width"], t["source"]["canvas"]["height"]):
                continue
            seen.add(key)
            tiles.append((f"{t['name']}@{key[0]}x{key[1]}", t["source"], dict(tgt)))
    return tiles


CHECKS = ("scale2x", "permutation", "rename", "determinism", "nonlatin", "continuity")


def run_checks(engine_root: Path, tiles, seed: int = 5, checks=CHECKS) -> dict:
    rng = random.Random(seed)
    base_jobs, a_jobs, b_jobs, backs = [], [], [], {}
    for name, src, tgt in tiles:
        base_jobs.append({"name": name, "source": src, "target": tgt})
        a_jobs.append({"name": f"scale2x|{name}", "source": scale_source(src, 2.0), "target": tgt})
        a_jobs.append({"name": f"permutation|{name}", "source": permute_source(src, rng), "target": tgt})
        ren, back = rename_source(src, rng)
        backs[name] = back
        a_jobs.append({"name": f"rename|{name}", "source": ren, "target": tgt})
        a_jobs.append({"name": f"nonlatin|{name}", "source": nonlatin_source(src, rng.randrange(8)), "target": tgt})
        d = max(1, round(tgt["width"] * 0.001))
        a_jobs.append({"name": f"continuity|{name}", "source": src,
                       "target": {"width": tgt["width"] + d, "height": tgt["height"]}})
        b_jobs.append({"name": f"determinism|{name}", "source": src, "target": tgt})
    a_jobs = [j for j in a_jobs if j["name"].split("|")[0] in checks]
    b_jobs = [j for j in b_jobs if j["name"].split("|")[0] in checks]
    print(f"  {engine_root}: {len(base_jobs) + len(a_jobs) + len(b_jobs)} resizes", flush=True)
    base = run_jobs(engine_root, base_jobs + a_jobs, hash_seed="0")
    if b_jobs:
        base.update(run_jobs(engine_root, b_jobs, hash_seed="4242"))

    res: Dict[str, dict] = {k: {"fail": [], "n": 0} for k in
                            ("crash", "scale2x", "permutation", "rename", "determinism", "nonlatin", "continuity")}
    res["scale2x"]["exact"] = res["scale2x"]["layout"] = 0
    for name, src, tgt in tiles:
        b0 = base[name]
        if b0["error"] or b0["out"] is None:
            res["crash"]["fail"].append(f"{name}: {b0['error']}")
            continue
        o0 = b0["out"]
        c0 = canonical(o0)
        res["crash"]["n"] += 1

        if "scale2x" in checks:
            _scale2x(res["scale2x"], name, o0, c0, base[f"scale2x|{name}"])
        for key in ("permutation", "determinism"):
            if key not in checks:
                continue
            r = base[f"{key}|{name}"]
            res[key]["n"] += 1
            if canonical(r["out"]) != c0:
                d, oth = max_px_diff(o0, r["out"])
                res[key]["fail"].append(f"{name}: max {d:.1f}px" + (f", {oth[:2]}" if oth else ""))
        if "rename" in checks:
            r = base[f"rename|{name}"]
            res["rename"]["n"] += 1
            mapped = map_ids(r["out"], backs[name])
            if canonical(mapped) != c0:
                d, oth = max_px_diff(o0, mapped)
                res["rename"]["fail"].append(f"{name}: max {d:.1f}px" + (f", {oth[:2]}" if oth else ""))
        if "nonlatin" in checks:
            nl_src = next(j["source"] for j in a_jobs if j["name"] == f"nonlatin|{name}")
            _nonlatin(res["nonlatin"], name, o0, nl_src, tgt, base[f"nonlatin|{name}"])
        if "continuity" in checks:
            r = base[f"continuity|{name}"]
            res["continuity"]["n"] += 1
            j = jumps(o0, r["out"]) if r["out"] else ["crash"]
            if j:
                res["continuity"]["fail"].append(f"{name}: {j[:3]}")
            res["continuity"]["jumps"] = res["continuity"].get("jumps", 0) + len(j)
    res["continuity"].setdefault("jumps", 0)
    secs = sorted(v["secs"] for v in base.values())
    res["timing"] = {"median": secs[len(secs) // 2], "max": secs[-1]} if secs else {}
    return res


def _scale2x(res: dict, name: str, o0: dict, c0: str, r: dict) -> None:
    res["n"] += 1
    diff, other = max_px_diff(o0, r["out"])
    if canonical(r["out"]) == c0:
        res["exact"] += 1
    side = max(o0["canvas"]["width"], o0["canvas"]["height"])
    if diff > LAYOUT * side or any(".opacity" in o or ".z_index" in o for o in other):
        res["layout"] += 1
    if diff > TOL_PX or other:
        res["fail"].append(f"{name}: max {diff:.1f}px" + (f", {other[:2]}" if other else ""))


def _nonlatin(res: dict, name: str, o0: dict, nl_src: dict, tgt: dict, r: dict) -> None:
    from tools.rules import hard_rule_errors, render_image

    res["n"] += 1
    if r["error"] or r["out"] is None:
        res["fail"].append(f"{name}: crash {r['error']}")
        return
    why = []
    if r["fell_back"]:
        why.append("fell back")
    why += hard_rule_errors(nl_src, r["out"], tgt)[:2]
    dropped = _shown_copy(o0) - _shown_copy(r["out"])
    if dropped:
        why.append(f"dropped {sorted(dropped)}")
    try:
        render_image(r["out"])
    except Exception as ex:  # noqa: BLE001
        why.append(f"render crash {type(ex).__name__}")
    if why:
        res["fail"].append(f"{name}: " + "; ".join(map(str, why)))


def verdict(cand: dict, base: Optional[dict], checks=CHECKS) -> Tuple[bool, List[str]]:
    lines, ok = [], True
    for key in ("crash",) + tuple(k for k in CHECKS if k != "continuity"):
        if key != "crash" and key not in checks:
            continue
        n_fail = len(cand[key]["fail"])
        status = "PASS" if not n_fail else ("REPORT" if key in REPORT_ONLY else "FAIL")
        if status == "FAIL":
            ok = False
        extra = (f" (exact {cand[key]['exact']}, re-laid-out {cand[key]['layout']})" if key == "scale2x" else "")
        if base is not None:
            extra += f" (base {len(base[key]['fail'])}" + (f", re-laid-out {base[key]['layout']})"
                                                           if key == "scale2x" else ")")
        lines.append(f"{key:12s} {status:6s} {n_fail}/{cand[key]['n']} differ{extra}")
    c = cand["continuity"]["jumps"]
    if "continuity" not in checks:
        pass
    elif base is not None:
        b = base["continuity"]["jumps"]
        st = "PASS" if c <= b else "FAIL"
        ok &= c <= b
        lines.append(f"{'continuity':12s} {st:6s} {c} jumps on {len(cand['continuity']['fail'])} tiles (base {b})")
    else:
        lines.append(f"{'continuity':12s} {'COUNT':6s} {c} jumps on {len(cand['continuity']['fail'])} tiles (no base)")
    return ok, lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="metamorphic checks for one engine root")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--base", help="base engine root, for the continuity count")
    ap.add_argument("--generated", type=int, default=4, help="tools.designs designs to add to the dev set")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--checks", default=",".join(CHECKS), help="comma list of " + ",".join(CHECKS))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    checks = tuple(c for c in a.checks.split(",") if c)
    unknown = set(checks) - set(CHECKS)
    if unknown:
        ap.error(f"unknown checks {sorted(unknown)}")
    tiles = build_jobs(a.generated, a.seed)
    print(f"METAMORPHIC {len(tiles)} tiles", flush=True)
    cand = run_checks(Path(a.engine).resolve(), tiles, checks=checks)
    base = run_checks(Path(a.base).resolve(), tiles, checks=checks) if a.base else None
    ok, lines = verdict(cand, base, checks)
    for ln in lines:
        print("  " + ln)
    for key in ("crash", "scale2x", "permutation", "rename", "determinism", "nonlatin", "continuity"):
        for f in cand[key]["fail"][:4]:
            print(f"    {key}: {f}")
    for key, why in REPORT_ONLY.items():
        print(f"  report-only {key}: {why}")
    print(f"METAMORPHIC {'PASS' if ok else 'FAIL'}")
    if a.json:
        Path(a.json).write_text(json.dumps({"pass": ok, "lines": lines, "cand": cand, "base": base,
                                            "report_only": REPORT_ONLY}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.path.insert(0, str(GATE_ROOT))
    raise SystemExit(main())
