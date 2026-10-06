"""Chameleon layout engine: analyse -> generate candidate layouts -> score -> emit."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .analyze import Design, Item, analyze
from .compose import Plan, generate
from .geom import Box, el_box, rotated_aabb
from .items import Ctx
from .score import prune_relaxed, score_plans

__all__ = ["layout_design", "finalize", "plan_design"]


def plan_design(source: dict, target: dict):
    d = analyze(source)
    ctx = Ctx(W=float(target["width"]), H=float(target["height"]), W0=d.W, H0=d.H)
    from . import formats  # [formats-decor] platform safe/avoid zones
    plans = prune_relaxed(generate(d, ctx) + formats.safe_plans(d, ctx, generate))
    score_plans(plans, d, ctx)
    formats.score_zones(plans, d, ctx)
    plans.sort(key=lambda p: -p.score)
    _pixel_rerank(plans, d, ctx)
    _no_text_on_text(plans, d, ctx)
    return d, ctx, plans


def _no_text_on_text(plans, d: Design, ctx: Ctx) -> None:
    """Hard check: the shipped plan has no copy drawn over other copy. Pairs that
    overlapped deliberately in the source (stacked display type) are exempt.
    Take the best plan that passes; if none does, give up (resize() then falls
    back to the safe fit-and-centre layout)."""
    from .textfit import deliberate_overlaps, text_overlaps

    if not plans:
        return
    src_pairs = deliberate_overlaps(d.elements)
    chosen = None
    for i, p in enumerate(plans):
        if not text_overlaps(emit(p, d, ctx), exempt=frozenset(src_pairs)):
            chosen = i
            break
    if chosen is None:  # never ship copy over copy: the safe layout only scales the source
        raise RuntimeError("every candidate draws copy over copy")
    if chosen:
        plans.insert(0, plans.pop(chosen))


def _pixel_rerank(plans, d: Design, ctx: Ctx, top: int = 8) -> None:
    """Re-rank the best candidates by what is actually under their copy."""
    from . import pixels
    from .score import pixel_penalty

    if len(plans) < 1 or not pixels.available():
        return
    src = pixels.measure(d.elements, d.W, d.H, skip=set())
    if not src:
        return
    head = plans[:top]
    for p in head:
        imp = {}
        for r, _, _ in p.placed + p.floaters:
            for e in r.item.els:
                imp[e["id"]] = r.item.importance
        hidden = {e["id"] for it in p.hidden for e in it.els}
        stats = pixels.measure(emit(p, d, ctx), ctx.W, ctx.H, skip=hidden)
        pen = pixel_penalty(stats, src, imp)
        p.breakdown["pixels"] = -pen
        p.score -= pen
    head.sort(key=lambda p: -p.score)
    plans[:top] = head


def layout_design(source: dict, target: dict) -> dict:
    d, ctx, plans = plan_design(source, target)
    if not plans:
        raise RuntimeError("no feasible layout")
    out = emit(plans[0], d, ctx)
    return {"canvas": {"width": int(target["width"]), "height": int(target["height"])}, "elements": out}


# ----------------------------------------------------------------------------


def _strip(el: dict) -> dict:
    return {k: v for k, v in el.items() if k not in ("_idx", "_decor_kind")}


def emit(plan: Plan, d: Design, ctx: Ctx) -> List[dict]:
    out: Dict[str, dict] = {}
    for r, x, y in plan.placed + plan.floaters:
        for e in r.emit(x, y):
            out[e["id"]] = e
    for e in d.backgrounds:
        b = _strip(e)
        b.update(x=0.0, y=0.0, width=ctx.W, height=ctx.H)
        if b.get("type") == "image":
            b["fit"] = "cover"
        out[b["id"]] = b
    from . import decor  # [formats-decor] anchor + coverage placement
    for e in d.decor:
        out[e["id"]] = _slab(e, d, ctx) if e.get("_slab") else (decor.place(e, d, ctx) or _decor(e, d, ctx))
    for bd in d.backdrops:
        out[bd.el["id"]] = _backdrop(bd, plan, d, ctx)
    for it in plan.hidden:
        for e in it.els:
            if e["id"] not in out:
                out[e["id"]] = _hidden(e, d, ctx)  # void text still paints its whole string
    for e in d.elements:  # anything not placed (should not happen) is mapped proportionally
        if e["id"] not in out:
            out[e["id"]] = _proportional(e, d, ctx)
    return [out[e["id"]] for e in d.elements]


def _backdrop(bd, plan: Plan, d: Design, ctx: Ctx) -> dict:
    """Redraw a panel around wherever its members landed, reaching the canvas
    edges it reached in the source when nothing else is in the way."""
    members = {id(m) for m in bd.members}
    mine, others = [], []
    for r, x, y in plan.placed + plan.floaters:
        (mine if id(r.item) in members else others).append((r, Box(x, y, r.w, r.h)))
    if not mine:
        return _hidden(bd.el, d, ctx)
    u = mine[0][1]
    for _, b in mine[1:]:
        u = u.union(b)
    scales = sorted(r.scale for r, _ in mine)
    sc = scales[len(scales) // 2]
    floor = 0.6 * ctx.base_fs
    pl, pt, pr, pb = (max(floor, p * sc) for p in bd.pad)
    box = Box(u.x - pl, u.y - pt, u.w + pl + pr, u.h + pt + pb)
    obstacles = [b for _, b in others]

    def free(region: Box) -> bool:
        return all(region.inter(o) <= 0 for o in obstacles)

    el, et, er, eb = bd.edges
    if el and free(Box(0, box.y, box.x, box.h)):
        box = Box(0, box.y, box.r, box.h)
    if er and free(Box(box.r, box.y, ctx.W - box.r, box.h)):
        box = Box(box.x, box.y, ctx.W - box.x, box.h)
    if et and free(Box(box.x, 0, box.w, box.y)):
        box = Box(box.x, 0, box.w, box.b)
    if eb and free(Box(box.x, box.b, box.w, ctx.H - box.b)):
        box = Box(box.x, box.y, box.w, ctx.H - box.y)
    box = box.inter_box(Box(0, 0, ctx.W, ctx.H)) or box
    if not free(box):  # padding would run into a neighbour: hug the members instead
        tight = u.expanded(0.25 * floor)
        if free(tight):
            box = tight
        else:
            return _hidden(bd.el, d, ctx)
    o = _strip(bd.el)
    o.update(x=box.x, y=box.y, width=box.w, height=box.h)
    if o.get("corner_radius"):
        o["corner_radius"] = float(o["corner_radius"]) * sc
    return o


def _slab(e: dict, d: Design, ctx: Ctx) -> dict:
    """A flat background slab keeps spanning the side it spanned, at the edge it
    touched, with its proportions (it is an image)."""
    b = el_box(e)
    o = _strip(e)
    o.pop("_slab", None)
    r = b.w / max(1e-6, b.h)
    if b.y <= 0.03 * d.H and b.b >= 0.97 * d.H:  # full height
        h = ctx.H
        w = min(ctx.W, h * r)
        h = w / r
        x = 0.0 if b.cx < d.W / 2 else ctx.W - w
        y = (ctx.H - h) / 2.0
    else:  # full width
        w = ctx.W
        h = min(ctx.H, w / r)
        w = h * r
        x = (ctx.W - w) / 2.0
        y = 0.0 if b.cy < d.H / 2 else ctx.H - h
    o.update(x=x, y=y, width=w, height=h)
    return o


def _uniform(d: Design, ctx: Ctx) -> float:
    return min(ctx.W / d.W, ctx.H / d.H)


def _proportional(e: dict, d: Design, ctx: Ctx) -> dict:
    s = _uniform(d, ctx)
    b = el_box(e)
    o = _strip(e)
    w, h = b.w * s, b.h * s
    cx, cy = b.cx / d.W * ctx.W, b.cy / d.H * ctx.H
    o.update(x=cx - w / 2.0, y=cy - h / 2.0, width=w, height=h)
    if o.get("type") == "text":
        o["font_size"] = max(1, int(round(float(o.get("font_size", 12)) * s)))
    if o.get("type") == "shape" and o.get("corner_radius"):
        o["corner_radius"] = float(o["corner_radius"]) * s
    return o


def _hidden(e: dict, d: Design, ctx: Ctx) -> dict:
    """Dropped for this format: kept in the document (same id), not painted.

    opacity 0 says so. The reference renderer ignores opacity on text, so a
    dropped element is also stacked just under an opaque full-canvas
    background, where it cannot show through either way."""
    o = _proportional(e, d, ctx)
    o["opacity"] = 0.0
    floor = _opaque_bg_z(d)
    if floor is not None:
        o["z_index"] = int(floor) - 1
    return o


def _opaque_bg_z(d: Design):
    from . import assets
    from .color import parse_rgba

    zs = []
    for b in d.backgrounds:
        if b.get("opacity", 1.0) < 0.999:
            continue
        if b.get("type") == "image":
            a = assets.info(b.get("asset_id"))
            if a.found and a.opaque:
                zs.append(b["z_index"])
        elif b.get("type") == "shape" and parse_rgba(b.get("fill_color"))[3] >= 0.999:
            zs.append(b["z_index"])
    return min(zs) if zs else None


def _decor(e: dict, d: Design, ctx: Ctx) -> dict:
    """Decorative shapes keep their shape and corner. A shape that bled off the
    source canvas is shrunk to the part that was visible, then tucked inside."""
    b = el_box(e)
    o = _strip(e)
    if b.w <= 0 or b.h <= 0:
        o.update(x=0.0, y=0.0, width=0.0, height=0.0)
        return o
    vis = b.inter_box(Box(0, 0, d.W, d.H)) or b
    s = min(ctx.W / d.W, ctx.H / d.H)
    shrink = max(vis.w / b.w, vis.h / b.h)
    w, h = b.w * s * shrink, b.h * s * shrink
    cx, cy = vis.cx / d.W * ctx.W, vis.cy / d.H * ctx.H
    o.update(x=cx - w / 2.0, y=cy - h / 2.0, width=w, height=h)
    k = s * shrink
    if o.get("type") == "text":
        o["font_size"] = max(1, int(round(float(o.get("font_size", 12)) * k)))
    if o.get("corner_radius"):
        o["corner_radius"] = float(o["corner_radius"]) * k
    if o.get("stroke_width"):
        o["stroke_width"] = float(o["stroke_width"]) * k
    return o


# ----------------------------------------------------------------------------
# hard rules, enforced on any output


def finalize(source: dict, out: dict, target: dict) -> dict:
    """Make the output satisfy every structural rule no matter what produced it."""
    W, H = float(target["width"]), float(target["height"])
    src_by_id = {e["id"]: e for e in source.get("elements", [])}
    got = {e.get("id"): e for e in out.get("elements", []) if isinstance(e, dict)}
    elements = []
    for se in source.get("elements", []):
        e = dict(got.get(se["id"]) or se)
        # immutable fields come from the source, always
        for k in ("id", "type", "content", "font_family", "asset_id", "font_weight", "font_style",
                  "color", "fill_color", "shape_type", "group_id", "style_runs", "rotation"):
            if k in se:
                e[k] = se[k]
        for k in ("x", "y", "width", "height"):
            try:
                v = float(e.get(k, se.get(k, 0)))
            except (TypeError, ValueError):
                v = float(se.get(k, 0) or 0)
            e[k] = v if math.isfinite(v) else 0.0
        e["width"] = max(0.0, e["width"])
        e["height"] = max(0.0, e["height"])
        _keep_ratio(e, se)
        if not e.pop("_bleed", False):  # [formats-decor] decor.DECOR_BLEED (off by default)
            _inside(e, W, H)
        if e.get("type") == "text":
            try:
                fs = float(e.get("font_size", se.get("font_size", 12)))
            except (TypeError, ValueError):
                fs = float(se.get("font_size", 12) or 12)
            e["font_size"] = max(1, int(round(fs)) if math.isfinite(fs) else 12)
        _renderer_safe(e)
        if e.get("type") != "image":  # images keep full precision: thin rules must keep their ratio
            for k in ("x", "y", "width", "height"):
                e[k] = round(e[k], 2)
        e.pop("_idx", None)
        e.pop("_decor_kind", None)
        elements.append(e)
    return {"canvas": {"width": int(target["width"]), "height": int(target["height"])}, "elements": elements}


def _renderer_safe(e: dict) -> None:
    """Never emit a value the reference renderer raises on (one exception there
    aborts every remaining task of the run). Values are only *normalised*: the
    same colour as rgba(), a valid enum for an invalid one."""
    from .color import normalize

    t = e.get("type")
    for k in ("rotation", "opacity"):
        if k in e:
            try:
                v = float(e[k])
                e[k] = v if math.isfinite(v) else (0.0 if k == "rotation" else 1.0)
            except (TypeError, ValueError):
                e[k] = 0.0 if k == "rotation" else 1.0
    if "opacity" in e:
        e["opacity"] = min(1.0, max(0.0, e["opacity"]))
    if "z_index" in e and not isinstance(e["z_index"], int):
        try:
            e["z_index"] = int(e["z_index"])
        except (TypeError, ValueError):
            e["z_index"] = 0
    if t == "text":
        e["width"] = max(1.0, e["width"])
        e["height"] = max(1.0, e["height"])
        if e.get("text_align", "left") not in ("left", "center", "right"):
            e["text_align"] = "left"
        fw = e.get("font_weight")
        if fw is not None and not isinstance(fw, str):
            try:
                e["font_weight"] = "bold" if float(fw) >= 600 else "regular"
            except (TypeError, ValueError):
                e["font_weight"] = "regular"
        fst = e.get("font_style")
        if fst is not None and fst not in ("normal", "italic"):
            e["font_style"] = "italic" if str(fst).lower() == "italic" else "normal"
        if "line_height" in e:
            try:
                lh = float(e["line_height"])
                if not (math.isfinite(lh) and lh > 0):
                    raise ValueError
            except (TypeError, ValueError):
                e["line_height"] = 1.2
        if "letter_spacing" in e:
            try:
                ls = float(e["letter_spacing"])
                e["letter_spacing"] = ls if math.isfinite(ls) else 0.0
            except (TypeError, ValueError):
                e["letter_spacing"] = 0.0
        col = normalize(e.get("color"))
        e["color"] = col or "rgba(0,0,0,1)"
        runs = e.get("style_runs")
        if isinstance(runs, list):
            clean = []
            for r in runs:
                if not isinstance(r, dict):
                    continue
                r = dict(r)
                if "color" in r:
                    c = normalize(r["color"])
                    if c is None:
                        r.pop("color")
                    else:
                        r["color"] = c
                for k in ("font_weight",):
                    if k in r and not isinstance(r[k], str):
                        r.pop(k)
                clean.append(r)
            e["style_runs"] = clean
        elif runs is not None:
            e.pop("style_runs")
    elif t == "shape":
        col = normalize(e.get("fill_color"))
        if col:
            e["fill_color"] = col
        if e.get("stroke_color") is not None:
            sc = normalize(e.get("stroke_color"))
            if sc:
                e["stroke_color"] = sc


def _keep_ratio(e: dict, se: dict) -> None:
    if e.get("type") != "image":
        return
    sw, sh = float(se.get("width", 0) or 0), float(se.get("height", 0) or 0)
    if sw <= 0 or sh <= 0:
        return
    if e["width"] < 1e-3 or e["height"] < 1e-3:  # collapsed: a 1px speck at the source ratio
        cx, cy = e["x"] + e["width"] / 2.0, e["y"] + e["height"] / 2.0
        r = sw / sh
        e["width"], e["height"] = (1.0, 1.0 / r) if r >= 1 else (r, 1.0)
        e["x"], e["y"] = cx - e["width"] / 2.0, cy - e["height"] / 2.0
        return
    full = e["x"] <= 0.5 and e["y"] <= 0.5 and e.get("fit") == "cover"
    if full:
        return
    ra, rb = sw / sh, e["width"] / e["height"]
    if abs(rb / ra - 1.0) > 0.01:
        cx, cy = e["x"] + e["width"] / 2.0, e["y"] + e["height"] / 2.0
        area = e["width"] * e["height"]
        e["width"] = math.sqrt(area * ra)
        e["height"] = e["width"] / ra
        e["x"], e["y"] = cx - e["width"] / 2.0, cy - e["height"] / 2.0


def _inside(e: dict, W: float, H: float) -> None:
    """Shift (and if needed shrink, keeping proportions) so the element stays on canvas,
    including the corners of a rotated element."""
    rot = float(e.get("rotation", 0) or 0)
    for _ in range(3):
        b = Box(e["x"], e["y"], e["width"], e["height"])
        vb = b.union(rotated_aabb(b, rot))
        if vb.w > W + 1e-6 or vb.h > H + 1e-6:
            f = min(W / max(vb.w, 1e-6), H / max(vb.h, 1e-6)) * 0.999
            cx, cy = b.cx, b.cy
            e["width"], e["height"] = b.w * f, b.h * f
            e["x"], e["y"] = cx - e["width"] / 2.0, cy - e["height"] / 2.0
            if e.get("type") == "text":
                e["font_size"] = max(1, int(float(e.get("font_size", 12)) * f))
            continue
        dx = (-vb.x if vb.x < 0 else 0.0) + (W - vb.r if vb.r > W else 0.0)
        dy = (-vb.y if vb.y < 0 else 0.0) + (H - vb.b if vb.b > H else 0.0)
        e["x"] += dx
        e["y"] += dy
        break
