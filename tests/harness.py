"""Run a finicky case on a target and collect everything wrong with the output."""

from __future__ import annotations

import io
import json
import math
from contextlib import redirect_stderr
from typing import List, Tuple

from engine.resize import resize
from tests.finicky import Case
from tools.rules import ghost_text, hard_rule_errors, promise_errors


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


def _centre(e: dict) -> Tuple[float, float]:
    return e["x"] + e["width"] / 2.0, e["y"] + e["height"] / 2.0


def together_errors(case: Case, src: dict, out: dict) -> List[str]:
    """Each pair keeps its source arrangement: the offset between the two, in
    units of the first element's scale, is what it was."""
    so = {e["id"]: e for e in src["elements"]}
    oo = {e["id"]: e for e in out["elements"]}
    errs = []
    for a, b in case.together:
        if _visible(oo[a]) != _visible(oo[b]):
            errs.append(f"{a}/{b}: one shown, one hidden")
            continue
        if not _visible(oo[a]):
            continue
        k = _scale(so[a], oo[a])
        (ax0, ay0), (bx0, by0) = _centre(so[a]), _centre(so[b])
        (ax1, ay1), (bx1, by1) = _centre(oo[a]), _centre(oo[b])
        ex, ey = (bx1 - ax1) - (bx0 - ax0) * k, (by1 - ay1) - (by0 - ay0) * k
        slack = 0.15 * math.hypot(oo[a]["width"], oo[a]["height"]) + 2.0
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
    errs += [f"{i}: hidden" for i in case.show if not _visible(oo[i])]
    errs += together_errors(case, src, out)
    if case.check:
        errs += case.check(src, out, tgt)
    if render_check:
        errs += ghost_text(out)
    return out, errs
