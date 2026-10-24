"""Chameleon layout engine: analyse -> generate candidate layouts -> score -> emit."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .analyze import Design, Item, analyze, as_number
from .color import normalize as normalize_colour
from .compose import Plan, generate
from .geom import Box, el_box, rotated_aabb
from .items import Ctx
from .score import prune_relaxed, score_plans
from .textfit import TextStyle, letter_spacing

__all__ = ["layout_design", "finalize", "plan_design"]


def plan_design(source: dict, target: dict):
    """Analyse, generate candidates, rank them and apply the hard checks; the
    passing plans come back best first.

    Texts locked together (a shadow, a price) are first kept as one piece. If
    no layout works that way, the design is planned again with them as
    separate copy. If still nothing passes, this raises and resize() falls
    back to the safe fit-and-centre layout."""
    reason = "no feasible layout"
    for lockups in (True, False):
        d = analyze(source, lockups=lockups)
        ctx = Ctx(W=float(target["width"]), H=float(target["height"]), W0=d.W, H0=d.H)
        plans = prune_relaxed(generate(d, ctx))
        if plans:
            score_plans(plans, d, ctx)
            plans.sort(key=lambda p: -p.score)
            _pixel_rerank(plans, d, ctx)
            if _promote_passing(plans, source, d, ctx):
                return d, ctx, plans
            reason = "every candidate draws copy over copy or past the canvas"
        if not any(it.lockup for it in d.items):
            break
    raise RuntimeError(reason)


def _promote_passing(plans, source: dict, d: Design, ctx: Ctx) -> bool:
    """Hard check, on the finished output exactly as it will be emitted: no
    copy over copy that the source did not draw (see
    textfit.new_text_overlaps) and no glyph past the canvas edge. Moves the
    best plan that passes to the front; False if none does."""
    from .textfit import drawn_box, new_text_overlaps

    target = {"width": ctx.W, "height": ctx.H}
    canvas = Box(-0.5, -0.5, ctx.W + 1.0, ctx.H + 1.0)

    def passes(p: Plan) -> bool:
        els = finalize(source, {"canvas": target, "elements": emit(p, d, ctx)}, target)["elements"]
        shown = [e for e in els if e.get("type") == "text" and float(e.get("opacity", 1)) > 0
                 and str(e.get("content", "")).strip()]
        if any(drawn_box(e).inter(canvas) < drawn_box(e).area - 1.0 for e in shown):
            return False
        return not new_text_overlaps(els, d.elements)

    chosen = next((i for i, p in enumerate(plans) if passes(p)), None)
    if chosen is None:
        return False
    plans.insert(0, plans.pop(chosen))
    return True


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
    out = emit(plans[0], d, ctx)
    return {"canvas": {"width": int(target["width"]), "height": int(target["height"])}, "elements": out}


# ----------------------------------------------------------------------------


def _strip(el: dict) -> dict:
    return {k: v for k, v in el.items() if k != "_idx"}


def emit(plan: Plan, d: Design, ctx: Ctx) -> List[dict]:
    out: Dict[str, dict] = {}
    for r, x, y in plan.placed + plan.floaters:
        for e in r.emit(x, y):
            out[e["id"]] = e
        for e in r.dropped:
            out[e["id"]] = _hidden(e, d, ctx)
    for e in d.backgrounds:
        b = _strip(e)
        b.update(x=0.0, y=0.0, width=ctx.W, height=ctx.H)
        if b.get("type") == "image":
            b["fit"] = "cover"
        out[b["id"]] = b
    for e in d.decor:
        out[e["id"]] = _slab(e, d, ctx) if e.get("_slab") else _decor(e, d, ctx)
    for bd in d.backdrops:
        out[bd.el["id"]] = _backdrop(bd, plan, d, ctx)
    for it in plan.hidden:
        for e in it.els:
            if e["id"] not in out:
                out[e["id"]] = _hidden(e, d, ctx)  # void text still paints its whole string
    for e in d.elements:  # anything not placed (should not happen) is mapped proportionally
        if e["id"] not in out:
            out[e["id"]] = _proportional(e, d, ctx)
    els = [out[e["id"]] for e in d.elements]
    _tuck_hidden_text(els)
    return els


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

    opacity 0 says so, and it is stacked under everything. The reference
    renderer ignores opacity on text, though, so hidden copy is also set at
    1 px, where the whole string paints a few faint pixels at most (and
    `_tuck_hidden_text` puts it under something opaque when it can)."""
    from .textfit import TextStyle, measure

    o = _proportional(e, d, ctx)
    o["opacity"] = 0.0
    o["z_index"] = min(int(x["z_index"]) for x in d.elements) - 1
    if o.get("type") == "text":
        blk = measure(str(o.get("content", "")), TextStyle.of(o), 1, max(1.0, o["width"]), exact=True)
        o.update(font_size=1, width=max(1.0, blk.ink_width), height=max(1.0, blk.height))
    return o


def _tuck_hidden_text(els: List[dict]) -> None:
    """Move each hidden text under the largest opaque shape or image on the
    canvas, so not even its 1 px glyphs show."""
    from . import assets
    from .color import parse_rgba

    def cover(e: dict) -> Optional[Box]:
        if float(e.get("opacity", 1.0)) < 0.999 or e.get("rotation"):
            return None
        b = el_box(e)
        if e.get("type") == "shape" and parse_rgba(e.get("fill_color"))[3] >= 0.999:
            if e.get("shape_type") == "ellipse":  # the square inscribed in it
                return Box(b.cx - b.w * 0.35, b.cy - b.h * 0.35, b.w * 0.7, b.h * 0.7)
            return b if e.get("shape_type", "rectangle") == "rectangle" else None
        a = assets.info(e.get("asset_id")) if e.get("type") == "image" else None
        if a is not None and a.found and a.opaque:
            return b
        return None

    hidden = [e for e in els if e.get("type") == "text" and float(e.get("opacity", 1.0)) <= 0]
    covers = sorted((c for c in map(cover, els) if c is not None and c.w >= 4 and c.h >= 4), key=lambda b: -b.area)
    if not hidden or not covers:
        return
    c = covers[0]
    for e in hidden:
        w = min(e["width"], c.w - 2.0)
        e.update(x=c.x + 1.0, y=c.y + 1.0, width=max(1.0, w))


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
    """Make the output satisfy every structural rule no matter what produced it
    (and never raise, whatever the source holds)."""
    W, H = max(1.0, as_number(target.get("width"), 1.0)), max(1.0, as_number(target.get("height"), 1.0))
    got = {e.get("id"): e for e in out.get("elements", []) if isinstance(e, dict)}
    elements = []
    for se in source.get("elements") or []:
        if not isinstance(se, dict):
            continue
        e = dict(got.get(se.get("id")) or se)
        # immutable fields come from the source, always
        for k in ("id", "type", "content", "font_family", "asset_id", "font_weight", "font_style",
                  "color", "fill_color", "shape_type", "group_id", "style_runs", "rotation"):
            if k in se:
                e[k] = se[k]
        for k in ("x", "y", "width", "height"):
            e[k] = as_number(e.get(k), as_number(se.get(k), 0.0))
        e["width"] = max(0.0, e["width"])
        e["height"] = max(0.0, e["height"])
        e["rotation"] = as_number(e.get("rotation"), 0.0)
        _keep_ratio(e, se)
        _inside(e, W, H)
        if e.get("type") == "text":
            fs = as_number(e.get("font_size"), as_number(se.get("font_size"), 12.0))
            e["font_size"] = max(1, int(round(fs)))
            if as_number(se.get("letter_spacing"), 0.0):  # tracking scales with the type, whatever produced the size
                e["letter_spacing"] = letter_spacing(TextStyle.of(se), e["font_size"])
        _renderer_safe(e)
        if e.get("type") != "image":  # images keep full precision: thin rules must keep their ratio
            for k in ("x", "y", "width", "height"):
                e[k] = round(e[k], 2)
        e.pop("_idx", None)
        elements.append(e)
    return {"canvas": {"width": int(target["width"]), "height": int(target["height"])}, "elements": elements}


def _renderer_safe(e: dict) -> None:
    """Never emit a value the reference renderer raises on (one exception there
    aborts every remaining task of the run). Values are only *normalised*: the
    same colour as rgba(), the same number as a float, a valid enum for an
    invalid one; a field the renderer cannot read at all is left out."""
    t = e.get("type")
    for k, default in (("rotation", 0.0), ("opacity", 1.0)):
        if k in e:
            e[k] = as_number(e[k], default)
    if "opacity" in e:
        e["opacity"] = min(1.0, max(0.0, e["opacity"]))
    if "z_index" in e and not isinstance(e["z_index"], int):
        e["z_index"] = int(as_number(e["z_index"], 0.0))
    if t == "text":
        e["width"] = max(1.0, e["width"])
        e["height"] = max(1.0, e["height"])
        if not isinstance(e.get("font_family", ""), str):
            e.pop("font_family")  # the renderer falls back to its default family
        if e.get("text_align", "left") not in ("left", "center", "right"):
            e["text_align"] = "left"
        fw = e.get("font_weight")
        if fw is not None and not isinstance(fw, str):
            e["font_weight"] = "bold" if as_number(fw, 400.0) >= 600 else "regular"
        fst = e.get("font_style")
        if fst is not None and fst not in ("normal", "italic"):
            e["font_style"] = "italic" if str(fst).lower() == "italic" else "normal"
        if "line_height" in e:
            lh = as_number(e["line_height"], 1.2)
            e["line_height"] = lh if lh > 0 else 1.2
        if "letter_spacing" in e:
            e["letter_spacing"] = as_number(e["letter_spacing"], 0.0)
        e["color"] = normalize_colour(e.get("color")) or "rgba(0,0,0,1)"
        runs = e.get("style_runs")
        if isinstance(runs, list):
            e["style_runs"] = [r for r in map(_safe_run, runs) if r is not None]
        elif runs is not None:
            e.pop("style_runs")
    elif t == "shape":
        col = normalize_colour(e.get("fill_color"))
        if col:
            e["fill_color"] = col
        if e.get("stroke_color") is not None:
            sc = normalize_colour(e.get("stroke_color"))
            if sc:
                e["stroke_color"] = sc
        for k in ("corner_radius", "stroke_width"):
            if k in e:
                e[k] = max(0.0, as_number(e[k], 0.0))
    elif t == "image":
        if e.get("fit", "contain") not in ("contain", "cover", "fill"):
            e["fit"] = "contain"  # anything else would be stretched to the box
        if e.get("asset_id") is not None and not isinstance(e["asset_id"], str):
            e["asset_id"] = str(e["asset_id"])


def _safe_run(r) -> Optional[dict]:
    """A style run the converter can read, or None."""
    if not isinstance(r, dict) or not all(isinstance(r.get(k, 0), int) for k in ("start", "end")):
        return None
    r = dict(r)
    if "color" in r:
        c = normalize_colour(r["color"])
        if c is None:
            r.pop("color")
        else:
            r["color"] = c
    if "font_weight" in r and not isinstance(r["font_weight"], str):
        r.pop("font_weight")
    return r


def _keep_ratio(e: dict, se: dict) -> None:
    if e.get("type") != "image":
        return
    sw, sh = as_number(se.get("width"), 0.0), as_number(se.get("height"), 0.0)
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
