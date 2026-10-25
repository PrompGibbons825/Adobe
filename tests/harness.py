"""Run a finicky case on a target and collect everything wrong with the output."""

from __future__ import annotations

import io
import json
import math
from contextlib import redirect_stderr
from typing import List, Tuple

from engine.resize import resize
from engine.smart.geom import el_box
from engine.smart.textfit import drawn_box
from tests.finicky import ROOMY_SIDE, BleedProbe, Case, Relation, _scaled
from tools.rules import (bleed_report, contrast_errors, ghost_text, hard_rule_errors, hierarchy_errors,
                         promise_errors)


def run(src: dict, tgt: dict) -> Tuple[dict, bool]:
    """resize() on a private copy; True if the smart engine fell back."""
    buf = io.StringIO()
    with redirect_stderr(buf):
        out = resize(json.loads(json.dumps(src)), dict(tgt))
    return out, "Traceback" in buf.getvalue()


def _visible(e: dict) -> bool:
    return float(e.get("opacity", 1) or 0) > 0


def _scale(src_el: dict, out_el: dict) -> float:
    if src_el.get("type") == "text":
        return out_el["font_size"] / max(1.0, float(src_el["font_size"]))
    return out_el["width"] / max(1e-6, float(src_el["width"]))


def _seen(e: dict):
    """Where the element is seen: a text's glyphs, anything else's box."""
    return drawn_box(e) if e.get("type") == "text" else el_box(e)


def _centre(e: dict) -> Tuple[float, float]:
    b = _seen(e)
    return b.cx, b.cy


def together_errors(case: Case, src: dict, out: dict, roomy: bool = True) -> List[str]:
    """Each pair keeps its source arrangement: the offset between the two, in
    units of the first element's scale, is what it was. On a strip a stacked
    pair may be re-set side by side, but the two still have to touch."""
    so = {e["id"]: e for e in src["elements"]}
    oo = {e["id"]: e for e in out["elements"]}
    errs = []
    for a, b in case.together:
        if _visible(oo[a]) != _visible(oo[b]):
            errs.append(f"{a}/{b}: one shown, one hidden")
            continue
        if not _visible(oo[a]):
            continue
        if not roomy:
            ga, gb = _seen(oo[a]), _seen(oo[b])
            gap = max(gb.x - ga.r, ga.x - gb.r, gb.y - ga.b, ga.y - gb.b)
            if gap > 0.5 * max(ga.h, gb.h):
                errs.append(f"{a}/{b}: drifted {gap:.0f} px apart")
            continue
        k = _scale(so[a], oo[a])
        (ax0, ay0), (bx0, by0) = _centre(so[a]), _centre(so[b])
        (ax1, ay1), (bx1, by1) = _centre(oo[a]), _centre(oo[b])
        ex, ey = (bx1 - ax1) - (bx0 - ax0) * k, (by1 - ay1) - (by0 - ay0) * k
        slack = 0.15 * max(math.hypot(oo[i]["width"], oo[i]["height"]) for i in (a, b)) + 2.0
        if math.hypot(ex, ey) > slack:
            errs.append(f"{a}/{b}: arrangement broke (off by {math.hypot(ex, ey):.0f} px, slack {slack:.0f})")
    return errs


def evaluate(case: Case, target: str, render_check: bool = True) -> Tuple[dict, List[str]]:
    src, tgt = case.source, case.target(target)
    out, fell_back = run(src, tgt)
    errs = hard_rule_errors(src, out, tgt)
    if fell_back:
        errs.append("fell back to fit-and-centre")
    else:
        errs += promise_errors(src, out, tgt)
    oo = {e["id"]: e for e in out["elements"]}
    errs += [f"{i}: hidden" for i in case.must_show(target) if not _visible(oo[i])]
    errs += together_errors(case, src, out, roomy=min(tgt.values()) >= ROOMY_SIDE)
    errs += hierarchy_errors(src, out)
    if case.check:
        errs += case.check(src, out, tgt)
    if render_check:
        errs += ghost_text(out) + contrast_errors(src, out, tgt)
    return out, errs


def probe_errors(p: BleedProbe) -> List[str]:
    """What is wrong with the bleed rules on one probe: an allowed bleed the hard rules reject
    (or do not report as bleed), or a forbidden bleed they let through."""
    src, out, tgt = p.source, p.output, p.target
    hard = hard_rule_errors(src, out, tgt)
    if p.allowed:
        errs = [f"allowed bleed rejected: {e}" for e in hard]
        bled = {b.split(":", 1)[0] for b in bleed_report(src, out, tgt)}
        errs += [f"{i}: bleed not reported" for i in p.moves if i not in bled]
        return errs
    off = [e for e in hard if "off-canvas" in e]
    other = [f"unrelated error: {e}" for e in hard if "off-canvas" not in e]
    return other + ([] if off else ["forbidden bleed accepted"])


def relation_errors(r: Relation) -> List[str]:
    """How far the two runs of a relation disagree: visibility, then each element's box (and a
    text's size) in units of the canvas. Tolerances are loose; a flipped layout is far outside."""
    t = r.target
    if r.kind == "scale":
        a, _ = run(r.source, t)
        b, _ = run(_scaled(r.source, r.k), {"width": t["width"] * r.k, "height": t["height"] * r.k})
        f, W, H = 1.0 / r.k, t["width"], t["height"]
        tol = 0.04
    else:
        t2 = {"width": t["width"] + (r.name.endswith("+10")), "height": t["height"] + (r.name.endswith("+01"))}
        a, _ = run(r.source, t)
        b, _ = run(r.source, t2)
        f, W, H = 1.0, t["width"], t["height"]
        tol = 0.03
    oa = {e["id"]: e for e in a["elements"]}
    errs = []
    for e in b["elements"]:
        o = oa[e["id"]]
        if _visible(o) != _visible(e):
            errs.append(f"{e['id']}: shown in one run, hidden in the other")
            continue
        if not _visible(o):
            continue
        d = max(abs(o["x"] - e["x"] * f) / W, abs(o["y"] - e["y"] * f) / H,
                abs(o["width"] - e["width"] * f) / W, abs(o["height"] - e["height"] * f) / H)
        if d > tol:
            errs.append(f"{e['id']}: box moved {d:.0%} of the canvas")
        elif e.get("type") == "text" and abs(o["font_size"] - e["font_size"] * f) > 0.1 * o["font_size"] + 1:
            errs.append(f"{e['id']}: font {o['font_size']:.0f} vs {e['font_size'] * f:.0f}")
    return errs
