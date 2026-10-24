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
from tests.finicky import ROOMY_SIDE, Case
from tools.rules import contrast_errors, ghost_text, hard_rule_errors, hierarchy_errors, promise_errors


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
