"""Finale API: explained candidates, user constraints, and one design -> every channel.

    candidates(source, target, constraints=None, k=5, budget=None) -> [candidate, ...]

A candidate is plain JSON:

    rank, archetype, strategy        "split", "split-left-0.34"
    score, breakdown                 the ranking score and its terms
    roles                            id -> {role, unit, importance, priority}
    hidden_ids, dropped              what the layout gave up, and why
    decisions                        id -> [{rule, before, after, reason, text}, ...]
    layout_decisions, highlights     plan-level reasons; the top one-liners
    design                           the finished layout (the resize() output format)
    checks, warnings, budget         pins kept, min font, safe area; parse warnings

Candidate 1 is exactly what `resize(source, target)` returns when no
constraints and no budget are given. Near-identical layouts (same hidden set,
every box within ~1% of the canvas, same type sizes) are folded together.

Everything here is deterministic: same input, same output (a wall-clock budget
that is actually hit is the one exception, and the candidate says so).
"""

from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stderr
from typing import Any, Dict, List, Optional, Tuple

from engine.smart import budget as _budget
from engine.smart import constraints as _cons
from engine.smart import explain as _explain
from engine.smart import providers as _providers

HEAD = 8  # plans the shipped ranking re-ranks by pixels (engine.smart._pixel_rerank)


# ----------------------------------------------------------------------------
# helpers


def _canvas(target) -> Dict[str, int]:
    if isinstance(target, (tuple, list)):
        return {"width": int(target[0]), "height": int(target[1])}
    return {"width": int(target["width"]), "height": int(target["height"])}


def _sig(design: dict) -> List[tuple]:
    W, H = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    out = []
    for e in design["elements"]:
        if float(e.get("opacity", 1) or 0) <= 0:
            out.append((e["id"], None))
        else:
            out.append((e["id"], (e["x"] / W, e["y"] / H, e["width"] / W, e["height"] / H,
                                  float(e.get("font_size", 0) or 0))))
    return out


def _same(a: List[tuple], b: List[tuple], tol: float = 0.012) -> bool:
    if len(a) != len(b):
        return False
    for (ia, va), (ib, vb) in zip(a, b):
        if ia != ib or (va is None) != (vb is None):
            return False
        if va is None:
            continue
        if max(abs(x - y) for x, y in zip(va[:4], vb[:4])) > tol or abs(va[4] - vb[4]) > 1.0:
            return False
    return True


def _round(v: float, n: int = 3) -> float:
    return float(round(v, n))


def _checks(design: dict, c: Optional[_cons.Constraints], pins_kept: Dict[str, bool]) -> dict:
    from engine.smart.textfit import text_overlaps

    vis_text = [e for e in design["elements"] if e.get("type") == "text" and float(e.get("opacity", 1) or 0) > 0]
    out = {"copy_over_copy": bool(text_overlaps(design["elements"])),
           "min_font_px": min((int(e.get("font_size", 0)) for e in vis_text), default=None)}
    if c is not None and c.min_font_px:
        out["min_font_ok"] = out["min_font_px"] is None or out["min_font_px"] >= c.min_font_px - 0.5
    if pins_kept:
        out["pins_ok"] = all(pins_kept.values())
        out["pins"] = pins_kept
    return out


def _safe_check(design: dict, roles: dict, c: Optional[_cons.Constraints], out: dict) -> None:
    if c is None or not c.has_safe:
        return
    W, H = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    l, t, r, b = c.safe
    bad = []
    for e in design["elements"]:
        if float(e.get("opacity", 1) or 0) <= 0:
            continue
        if roles.get(e["id"], {}).get("role") in ("background", "decor", "slab", "backdrop", "other"):
            continue
        if e["x"] < l - 0.5 or e["y"] < t - 0.5 or e["x"] + e["width"] > W - r + 0.5 or e["y"] + e["height"] > H - b + 0.5:
            bad.append(e["id"])
    out["safe_area_ok"] = not bad
    out["outside_safe"] = bad


def _pins_kept(design: dict, c: Optional[_cons.Constraints], d) -> Dict[str, bool]:
    if c is None or not c.pinned:
        return {}
    by = {e["id"]: e for e in design["elements"]}
    out = {}
    for eid, spec in c.pinned.items():
        e = by.get(eid)
        if e is None:
            continue
        if spec.mode != "box":
            out[eid] = float(e.get("opacity", 1) or 0) > 0
            continue
        bx = spec.box
        tol = 0.02 * max(bx.w, bx.h) + 1.0
        cx, cy = e["x"] + e["width"] / 2.0, e["y"] + e["height"] / 2.0
        out[eid] = abs(cx - bx.cx) <= tol + 0.5 * abs(e["width"] - bx.w) and \
            abs(cy - bx.cy) <= tol + 0.5 * abs(e["height"] - bx.h) and float(e.get("opacity", 1) or 0) > 0
    return out


def _decor_pins(els: List[dict], d, c: Optional[_cons.Constraints]) -> None:
    """Decoration is not laid out, so a pin on it is applied directly."""
    if c is None or not c.pinned:
        return
    decor = {e["id"] for e in d.decor} | {bd.el["id"] for bd in d.backdrops}
    for e in els:
        spec = c.pinned.get(e["id"])
        if spec is not None and spec.mode == "box" and e["id"] in decor:
            e.update(x=spec.box.x, y=spec.box.y, width=spec.box.w, height=spec.box.h)


# ----------------------------------------------------------------------------
# candidates


def _fallback_candidate(source: dict, target: dict, c, why: str, roles: dict) -> dict:
    from engine.resize import _fallback
    from engine.smart import finalize

    design = finalize(source, _fallback(source, target), target)
    lead = _explain._entry("archetype", "", "fallback", why, "safe fit-and-centre", 1000.0)
    dec = {e["id"]: [_explain._entry("fit", "", "", "scaled uniformly to fit and centred", "scaled to fit", 0.0)]
           for e in source.get("elements", [])}
    return {"rank": 1, "archetype": "fallback", "strategy": "fallback", "score": None, "breakdown": {},
            "roles": roles, "hidden_ids": [], "dropped": [], "decisions": _explain.public(dec),
            "layout_decisions": _explain.public({"_": [lead]})["_"], "highlights": ["fallback: " + why],
            "design": design, "duplicates": 0, "checks": _checks(design, c, {}), "warnings": list(c.warnings) if c else []}


def candidates(source: dict, target, constraints: Optional[dict] = None, k: int = 5, *, budget=None,
               explain: bool = True) -> List[dict]:
    """Up to k distinct, explained layouts for `target`, best first."""
    tgt = _canvas(target)
    c = _cons.parse(constraints, tgt)
    b = _budget.make(budget)
    factors = _cons.text_corrections(source, c)
    meas = _providers.CorrectedMeasurer(factors) if factors else None
    with _providers.text_measurer(meas), _cons.using(c), _budget.using(b):
        out = _candidates(source, tgt, c, max(1, int(k)), explain)
        rep = b.report() if b is not None else None
    for cand in out:
        cand["budget"] = rep
    return out


def _candidates(source: dict, target: dict, c, k: int, explain: bool) -> List[dict]:
    from engine.smart import _pixel_rerank, emit, finalize, plan_design
    from engine.smart.analyze import analyze
    from engine.smart.textfit import deliberate_overlaps, text_overlaps

    err = io.StringIO()
    try:
        with redirect_stderr(err):
            d, ctx, plans = plan_design(source, target)
        if not plans:
            raise RuntimeError("no feasible layout")
    except Exception as ex:  # resize() would fall back to fit-and-centre here
        try:
            roles = _explain.roles(analyze(source))
        except Exception:
            roles = {}
        msg = str(ex) or type(ex).__name__
        if "copy over copy" in msg:
            msg = "every candidate draws copy over copy"
        else:
            traceback.print_exc(file=sys.stderr)
        return [_fallback_candidate(source, target, c, msg, roles)]

    roles = _explain.roles(d)
    src_pairs = frozenset(deliberate_overlaps(d.elements))
    chosen: List[Tuple[Any, dict, list]] = []  # (plan, design, signature)
    dups: Dict[int, int] = {}
    rejected = 0

    def take(seq) -> bool:
        nonlocal rejected
        for p in seq:
            els = emit(p, d, ctx)
            if text_overlaps(els, exempt=src_pairs):
                rejected += 1
                continue
            _decor_pins(els, d, c)
            design = finalize(source, {"canvas": target, "elements": els}, target)
            sig = _sig(design)
            hit = next((i for i, (_, _, s) in enumerate(chosen) if _same(s, sig)), None)
            if hit is not None:
                dups[hit] = dups.get(hit, 0) + 1
                continue
            chosen.append((p, design, sig))
            if len(chosen) >= k:
                return True
        return False

    done = take(plans[:HEAD])
    if not done and len(plans) > HEAD:
        tail = list(plans[HEAD:])
        with redirect_stderr(io.StringIO()):
            _pixel_rerank(tail, d, ctx, top=min(len(tail), 4 * k))
        take(tail)

    out = []
    for i, (p, design, _) in enumerate(chosen):
        hidden_ids = [e["id"] for it in p.hidden for e in it.els]
        dropped = []
        for it in p.hidden:
            for e in it.els:
                dropped.append({"id": e["id"], "role": it.role})
        if c is not None and c.allow_drop:
            gone = set(hidden_ids)
            design = dict(design, elements=[e for e in design["elements"] if e["id"] not in gone])
        cand = {
            "rank": i + 1,
            "archetype": _explain.family(p.name),
            "strategy": p.name,
            "score": _round(p.score),
            "breakdown": {kk: _round(v) for kk, v in sorted(p.breakdown.items())},
            "roles": roles,
            "hidden_ids": hidden_ids,
            "dropped": dropped,
            "design": design,
            "duplicates": dups.get(i, 0),
            "warnings": list(c.warnings) if c else [],
        }
        if explain:
            dec, plan_level = _explain.explain(p, plans, d, ctx, design, allow_drop=bool(c and c.allow_drop))
            if i == 0 and len(chosen) > 1:
                cmp_ = _explain.compare(p, chosen[1][0])
                if cmp_:
                    plan_level.append(cmp_)
            for dr in dropped:
                ent = next((e for e in dec.get(dr["id"], []) if e["rule"] == "drop"), None)
                if ent:
                    dr["reason"] = ent["reason"]
            cand["highlights"] = _explain.highlights(dec, plan_level)
            cand["decisions"] = _explain.public(dec)
            cand["layout_decisions"] = _explain.public({"_": plan_level})["_"]
        chk = _checks(design, c, _pins_kept(design, c, d))
        _safe_check(design, roles, c, chk)
        cand["checks"] = chk
        out.append(cand)
    if rejected and out:
        out[0]["rejected_copy_over_copy"] = rejected
    return out


def best(source: dict, target, constraints: Optional[dict] = None, *, budget=None) -> dict:
    """The single best layout (the resize() output format)."""
    return candidates(source, target, constraints, k=1, budget=budget, explain=False)[0]["design"]
