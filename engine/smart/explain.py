"""Why the layout looks the way it does: roles and a per-element decision log.

Everything is derived after the search from what the search already holds
(the plan's realized items, its font sizes and line breaks, the other scored
plans), so explaining costs nothing while planning and a few milliseconds per
explained candidate.

A decision is {rule, before, after, reason, text}: `text` is the one-liner a
panel shows on hover ("headline 56→40 px to fit 2 lines in 420 px").
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .geom import Box, el_box

TERMS = {
    "visible": "size of what is shown", "hidden": "keeps more on the canvas", "small_text": "readable type",
    "lines": "cleaner line breaks", "overlap": "no collisions", "coverage": "fills the canvas",
    "balance": "visual balance", "prior": "suits the format", "pixels": "contrast under the copy",
    "safe_area": "stays in the safe area",
}
MUST_ROLES = ("headline", "cta", "logo")
OPTIONAL_ROLES = ("legal", "watermark", "badge", "ribbon", "shape")


def _dims(W: float, H: float) -> str:
    return f"{int(round(W))}×{int(round(H))}"


def shape_word(W: float, H: float) -> str:
    r = W / max(1e-6, H)
    named = {16 / 9: "16:9", 9 / 16: "9:16", 1.91: "1.91:1", 4 / 5: "4:5", 3.0: "3:1", 2 / 3: "2:3", 4 / 3: "4:3"}
    tag = next((v for k, v in named.items() if abs(math.log(r / k)) < 0.02), None)
    if tag is None:
        tag = f"{r:.1f}:1" if r >= 1 else f"1:{1 / r:.1f}"
    if r >= 5:
        return f"a strip ({tag})"
    if r >= 1.6:
        return f"wide ({tag})"
    if r > 1.15:
        return f"landscape ({tag})"
    if r >= 0.87:
        return "square" if abs(r - 1) < 0.02 else f"near-square ({tag})"
    if r >= 0.6:
        return f"portrait ({tag})"
    return f"tall ({tag})"


def where(fx: float, fy: float) -> str:
    v = "top" if fy < 1 / 3 else ("bottom" if fy > 2 / 3 else "middle")
    h = "left" if fx < 1 / 3 else ("right" if fx > 2 / 3 else "centre")
    if v == "middle":
        return "centre" if h == "centre" else h
    return v if h == "centre" else f"{v} {h}"


def family(name: str) -> str:
    return name.split("-")[0] if name else "?"


def _entry(rule: str, before, after, reason: str, text: str, weight: float = 0.0) -> dict:
    return {"rule": rule, "before": before, "after": after, "reason": reason, "text": text, "_w": weight}


# ----------------------------------------------------------------------------
# roles


def default_priority(it) -> str:
    if getattr(it, "priority", None):
        return it.priority
    if it.role in MUST_ROLES or it.core:
        return "must"
    if it.role in OPTIONAL_ROLES:
        return "optional"
    return "nice"


def roles(d) -> Dict[str, dict]:
    """Element id -> {role, unit, importance, priority} (inferred; priority is the
    user's when one was given)."""
    out: Dict[str, dict] = {}
    for e in d.backgrounds:
        out[e["id"]] = {"role": "background", "unit": e["id"], "importance": 0, "priority": "nice"}
    for e in d.decor:
        out[e["id"]] = {"role": "slab" if e.get("_slab") else "decor", "unit": e["id"], "importance": 0,
                        "priority": "optional"}
    for bd in d.backdrops:
        out[bd.el["id"]] = {"role": "backdrop", "unit": bd.el["id"], "importance": 0, "priority": "nice"}
    for it in d.items:
        pr = default_priority(it)
        for e in it.els:
            out[e["id"]] = {"role": it.role, "unit": it.key, "importance": round(float(it.importance), 1),
                            "priority": pr}
    for e in d.elements:
        out.setdefault(e["id"], {"role": "other", "unit": e["id"], "importance": 0, "priority": "nice"})
    return {e["id"]: out[e["id"]] for e in d.elements}


# ----------------------------------------------------------------------------
# plan level


def archetype_reason(plan, d, ctx) -> Tuple[str, str]:
    """(short, long) reason for the chosen arrangement."""
    name = plan.name if plan is not None else "fallback"
    fam = family(name)
    dims, shp = _dims(ctx.W, ctx.H), shape_word(ctx.W, ctx.H)
    if name == "fallback":
        return "safe fit", "every candidate drew copy over copy, so the source is scaled to fit and centred"
    if fam == "adapt":
        return (f"{dims} keeps the source's shape",
                f"{dims} is close to the source's {_dims(d.W, d.H)} shape: the original arrangement is kept, "
                f"scaled ×{plan.s:.2f}")
    if fam == "stack":
        side = {"stack-start": " set against the left side", "stack-end": " set against the right side"}.get(name, "")
        return f"{dims} is {shp}", f"{dims} is {shp}: one column in reading order{side}"
    if fam == "split":
        side = "left" if "-left-" in name else "right"
        hero = next((r.item.role for r, _, _ in plan.placed if r.item.role in ("hero", "image")), "image")
        fold = ", corner items folded into the copy column" if name.endswith("-fold") else ""
        return (f"{dims} is {shp}", f"{dims} is {shp}: {hero} on the {side}, copy stacked beside it{fold}")
    if fam == "row":
        how = "logo stacked over the copy, the rest on one line" if "stacked" in name else "everything on one line"
        return f"{dims} is {shp}", f"{dims} is {shp}: {how}"
    if fam == "empty":
        return "nothing to lay out", "every element is empty or missing"
    return name, name


def compare(plan, other) -> Optional[dict]:
    """Why `plan` ranked above `other`: the two score terms it gains most on."""
    if other is None:
        return None
    keys = sorted(set(plan.breakdown) | set(other.breakdown))
    deltas = sorted(((plan.breakdown.get(k, 0.0) - other.breakdown.get(k, 0.0), k) for k in keys),
                    key=lambda t: (-t[0], t[1]))
    gains = [(v, k) for v, k in deltas if v > 0.5][:2]
    if not gains:
        return None
    why = " and ".join(f"{TERMS.get(k, k)} (+{v:.0f})" for v, k in gains)
    total = plan.score - other.score
    return _entry("choice", other.name, plan.name, f"ahead of {other.name} by {total:.0f} points on {why}",
                  f"chosen over {other.name}: {why}", 0.0)


# ----------------------------------------------------------------------------
# element level


def _src_lines(el: dict) -> int:
    from . import textfit

    try:
        return textfit.measure(str(el.get("content", "")), textfit.TextStyle.of(el), float(el.get("font_size", 12)),
                               float(el.get("width", 0) or 0), exact=True).n_lines
    except Exception:
        return 1


def _font_entry(it, eid: str, fs: int, blk, plan, d, ctx, pinned: bool) -> Optional[dict]:
    el = d.by_id.get(eid)
    if el is None:
        return None
    fs0 = float(el.get("font_size", 12) or 12)
    l0 = _src_lines(el)
    n = blk.n_lines if blk is not None else l0
    if int(round(fs0)) == fs and n == l0:
        return None
    raw = fs0 * plan.s
    role = it.role if it.kind == "text" else f"{it.role} label"
    if pinned:
        why = "pinned size"
    elif fs > raw * 1.05 and fs <= ctx.judge_floor + 0.5:
        why = f"raised to the {ctx.judge_floor} px legibility floor for {_dims(ctx.W, ctx.H)}"
    elif fs > raw * 1.05:
        why = f"kept readable: at least {fs} px on {_dims(ctx.W, ctx.H)}"
    else:
        why = f"scaled ×{fs / fs0:.2f} with the layout"
    fit = ""
    if blk is not None and n != l0:
        fit = f"to fit {n} line{'s' if n > 1 else ''} in {blk.wrap_width:.0f} px"
    elif blk is not None and n > 1:
        fit = f"on {n} lines"
    text = f"{role} {fs0:g}→{fs} px" + (f" {fit}" if fit else "")
    weight = it.importance * abs(math.log(max(1.0, fs) / max(1.0, fs0))) + (0.1 * it.importance if n != l0 else 0)
    return _entry("font", f"{fs0:g} px, {l0} line{'s' if l0 > 1 else ''}", f"{fs} px, {n} line{'s' if n > 1 else ''}",
                  why + (f"; {fit}" if fit else ""), text, weight)


def _move_entry(it, F: Box, plan, d, ctx, idx: int, n: int, pinned: bool) -> Optional[dict]:
    src = it.src_box
    before = where(src.cx / d.W, src.cy / d.H)
    after = where(F.cx / ctx.W, F.cy / ctx.H)
    fam = family(plan.name)
    dims, shp = _dims(ctx.W, ctx.H), shape_word(ctx.W, ctx.H)
    if pinned:
        return _entry("pin", before, after, "pinned by the user: kept in its box",
                      f"{it.role} pinned ({after})", it.importance * 0.6)
    if it.corner and any(r.item is it for r, _, _ in plan.floaters):
        names = {"tl": "top-left", "tr": "top-right", "bl": "bottom-left", "br": "bottom-right"}
        c = names.get(it.corner, it.corner)
        return _entry("corner", before, after, f"corner item: kept in the {c} corner",
                      f"{it.role} kept {c}", it.importance * 0.1)
    if before == after:
        return None
    if fam == "adapt":
        why = "kept its place relative to the nearest canvas edges"
    elif fam == "split":
        side = "left" if "-left-" in plan.name else "right"
        if it.role in ("hero", "image") and it.kind in ("image", "composite"):
            why = f"takes the {side} side: {dims} is {shp} (split)"
        else:
            other = "right" if side == "left" else "left"
            why = f"moved into the {other} column beside the art: {dims} is {shp} (split)"
    elif fam == "stack":
        why = f"row {idx + 1} of {n} in one column: {dims} is {shp}"
    elif fam == "row":
        part = "left end" if F.cx < ctx.W / 3 else ("right end" if F.cx > 2 * ctx.W / 3 else "middle")
        why = f"{part} of a single row: {dims} is {shp}"
    else:
        why = plan.name
    return _entry("move", before, after, why, f"{it.role} {before} → {after}", it.importance * 0.35)


def _scale_entry(it, r, plan, d, ctx) -> Optional[dict]:
    if it.kind not in ("image", "composite"):
        return None
    k = r.scale
    u = min(ctx.W / d.W, ctx.H / d.H)
    if it.role not in ("hero", "logo") and abs(math.log(max(1e-6, k) / max(1e-6, u))) < math.log(1.25):
        return None
    src = it.src_box
    if it.role == "logo" and k > plan.s * 1.05 and r.h <= 3.4 * ctx.base_fs:
        why = f"kept at least {3.2 * ctx.base_fs:.0f} px tall to read as a logo"
    elif k > u * 1.25:
        why = f"×{k:.2f}: grows into the room the new shape gives it (uniform fit would be ×{u:.2f})"
    elif k < u / 1.25:
        why = f"×{k:.2f}: shrinks to leave room for the copy (uniform fit would be ×{u:.2f})"
    else:
        why = f"×{k:.2f}, about the uniform fit"
    return _entry("scale", f"{src.w:.0f}×{src.h:.0f}", f"{r.w:.0f}×{r.h:.0f}", why,
                  f"{it.role} {src.w:.0f}→{r.w:.0f} px wide", it.importance * 0.2 * abs(math.log(max(1e-6, k / u))))


def _drop_entry(it, plan, plans, d, ctx, allow_drop: bool) -> dict:
    after = "removed" if allow_drop else "hidden (opacity 0)"
    dims = _dims(ctx.W, ctx.H)
    if it.role == "missing":
        return _entry("drop", "visible", after, "asset not in the brand kit: it would paint a grey box",
                      "image asset missing", 1.0)
    if it.role == "void":
        return _entry("drop", "visible", after, "nothing visible in the source", "empty element", 0.0)
    alt = None
    for p in plans:
        if p is plan:
            continue
        if all(h is not it for h in p.hidden) and (alt is None or p.score > alt.score):
            alt = p
    if alt is None:
        why = f"no candidate layout fits it on {dims} with readable type"
    else:
        why = f"keeping it, the best layout ({alt.name}) scores {plan.score - alt.score:.0f} lower"
        head = next((r for r, _, _ in plan.placed + plan.floaters if r.item.role == "headline"), None)
        if head is not None:
            h2 = next((r for r, _, _ in alt.placed + alt.floaters if r.item is head.item), None)
            a, b = max(head.fonts.values(), default=0), max(h2.fonts.values(), default=0) if h2 else 0
            if h2 is not None and b < 0.9 * a:
                why = f"keeping it would shrink the headline {a}→{b} px ({alt.name})"
    pr = getattr(it, "priority", None)
    if pr == "optional":
        why = "marked optional; " + why
    return _entry("drop", "visible", after, why, f"dropped {it.role}", it.importance)


def _background_entry(e: dict, ctx, d) -> dict:
    from . import assets

    b = el_box(e)
    a = assets.info(e.get("asset_id")) if e.get("type") == "image" else None
    if a is not None and a.found and a.size[1] > 0:
        r0 = a.size[0] / a.size[1]
    else:
        r0 = b.w / max(1e-6, b.h)
    r = ctx.W / ctx.H
    crop = 1.0 - min(r0, r) / max(r0, r)
    side = "height" if r > r0 else "width"
    why = "full-bleed background re-covers the new canvas"
    if e.get("type") == "image" and crop > 0.01:
        why += f" (fit: cover, {crop * 100:.0f}% of its {side} cropped)"
    return _entry("cover", _dims(b.w, b.h), _dims(ctx.W, ctx.H), why, "background re-covers the canvas", 0.0)


def explain(plan, plans, d, ctx, design: dict, allow_drop: bool = False) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """(decisions by element id, plan-level decisions)."""
    out: Dict[str, List[dict]] = {e["id"]: [] for e in d.elements}
    plan_level: List[dict] = []
    short, long = archetype_reason(plan, d, ctx)
    name = plan.name if plan is not None else "fallback"
    plan_level.append(_entry("archetype", f"{_dims(d.W, d.H)} ({shape_word(d.W, d.H)})", name, long,
                             f"{family(name)}: {short}", 1000.0))
    if plan is None:
        return out, plan_level
    pinned = set(getattr(plan, "pinned", []) or [])
    placed = plan.placed + plan.floaters
    order = sorted(range(len(placed)), key=lambda i: (placed[i][2] if family(plan.name) != "row" else placed[i][1],
                                                     placed[i][0].item.key))
    rank = {i: k for k, i in enumerate(order)}
    for i, (r, x, y) in enumerate(placed):
        it = r.item
        F = Box(x, y, r.w, r.h)
        pin = it.key in pinned
        unit: List[dict] = []
        m = _move_entry(it, F, plan, d, ctx, rank[i], len(placed), pin)
        if m:
            unit.append(m)
        s = _scale_entry(it, r, plan, d, ctx)
        if s:
            unit.append(s)
        for e in it.els:
            out[e["id"]].extend(unit)
        for eid, fs in sorted(r.fonts.items()):
            f = _font_entry(it, eid, fs, r.blocks.get(eid), plan, d, ctx, pin)
            if f and eid in out:
                out[eid].append(f)
        if len(it.els) > 1:
            lead = it.els[0]["id"]
            for e in it.els[1:]:
                if not out[e["id"]]:
                    out[e["id"]].append(_entry("group", "", it.key, f"moves with its {it.role} unit ({lead})",
                                               f"travels with the {it.role}", 0.0))
    for it in plan.hidden:
        ent = _drop_entry(it, plan, plans, d, ctx, allow_drop)
        for e in it.els:
            out[e["id"]].append(ent)
    for e in d.backgrounds:
        out[e["id"]].append(_background_entry(e, ctx, d))
    by_id = {e["id"]: e for e in design.get("elements", [])}
    for e in d.decor:
        o = by_id.get(e["id"])
        if o is None:
            continue
        b0, b1 = el_box(e), el_box(o)
        k = b1.w / max(1e-6, b0.w)
        if e.get("_slab"):
            why = "flat background slab: keeps spanning the side it spanned"
        else:
            vis = b0.inter_box(Box(0, 0, d.W, d.H))
            bled = vis is not None and vis.area < 0.95 * max(1.0, b0.area)
            why = f"decoration scaled ×{k:.2f}" + ("; it bled off the source canvas, so only its visible part is kept, inside"
                                                   if bled else ", same relative spot")
        out[e["id"]].append(_entry("decor", _dims(b0.w, b0.h), _dims(b1.w, b1.h), why, "decoration rescaled", 0.0))
    for bd in d.backdrops:
        o = by_id.get(bd.el["id"])
        hid = o is not None and float(o.get("opacity", 1) or 0) <= 0
        out[bd.el["id"]].append(_entry("backdrop", "", "hidden" if hid else "redrawn",
                                       "its copy was dropped" if hid else
                                       f"panel redrawn around its {len(bd.members)} member(s) wherever they landed",
                                       "panel redrawn around its copy", 0.0))
    drops = [it for it in plan.hidden if it.role not in ("missing", "void")]
    if drops:
        names: Dict[str, int] = {}
        for it in drops:
            names[it.role] = names.get(it.role, 0) + 1
        what = ", ".join(f"{n} {r}s" if n > 1 else r for r, n in names.items())
        plan_level.append(_entry("drop", f"{len(drops)} unit(s) visible", "hidden",
                                 f"dropped on {_dims(ctx.W, ctx.H)}: {what}", f"dropped {what}",
                                 sum(it.importance for it in drops)))
    return out, plan_level


def highlights(decisions: Dict[str, List[dict]], plan_level: List[dict], n: int = 4) -> List[str]:
    """The most telling decisions, most important first (archetype, drops, then
    the biggest type and position changes)."""
    seen, ranked = set(), []
    for e in plan_level:
        ranked.append((e["_w"], e["text"]))
    for es in decisions.values():
        for e in es:
            if e["rule"] in ("drop",) and e["_w"] < 1.5:
                continue
            if e["rule"] == "drop":  # summarised at plan level
                continue
            ranked.append((e["_w"], e["text"]))
    ranked.sort(key=lambda t: (-t[0], t[1]))
    out = []
    for _, t in ranked:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= n:
            break
    return out


def public(decisions: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Drop the internal salience weights for serialisation."""
    return {k: [{kk: vv for kk, vv in e.items() if kk != "_w"} for e in v] for k, v in decisions.items()}
