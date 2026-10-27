"""
CHAMELEON engine entry point.

Pipeline:
  1. analyze   - infer roles (headline, logo, cta, hero, badges, legal...),
                 groups, decoration and background; saliency per image
  2. generate  - many candidate layouts from three templates (stack, split,
                 row) at several "drop levels" (what to hide when space is
                 tight)
  3. finalize  - saliency-aware background crop, decoration, corner-anchored
                 floats (ribbons, small images), keep content in canvas
  4. score     - our own critic (readability, collisions, prominence, text on
                 busy image, balance, what got dropped); return the best
"""

from __future__ import annotations

import copy
import os
import sys
import traceback
from typing import Any

from .analyze import FLOW_ROLES, analyze, norm_rot
from .conform import conform, fault_key, faults, normalize_source, num, type_floor
from .finalize import finalize
from .layout import (Ctx, tmpl_backdrop, tmpl_card_panel, tmpl_grid, tmpl_panel, tmpl_row, tmpl_rows, tmpl_split,
                     tmpl_stack)
from .blocks import Placed
from .score import score

DEBUG = os.environ.get("CHAMELEON_DEBUG") == "1"
DROP_LEVELS = [0.0, 1.3, 3.5, 5.5]


def _candidates(ctx, model):
    """Every candidate keeps every unit (no drops)."""
    flow = [u for u in model["units"] if u.role in FLOW_ROLES]
    if not flow:          # nothing to flow (backgrounds and decoration only): finalize places it all
        def empty():
            P = Placed()
            P.meta = {"template": "decor-only", "t": ctx.t_star}
            return P
        yield 0.0, empty
        return
    has_frame = any(u.kind == "frame" for u in model["units"])
    for alpha in ((0.0, 0.5, 0.9) if has_frame else (0.0,)):
        for item in _templates(ctx, flow, alpha):
            yield item


def _templates(ctx, flow, alpha):
    def with_alpha(make):
        def run():
            ctx.frame_alpha = alpha
            P = make()
            if P is not None:
                P.meta["template"] += f" a{alpha}" if alpha else ""
            return P
        return run
    for th, make in _templates_raw(ctx, flow):
        yield th, with_alpha(make)


def f_hero(flow):
    return [u for u in flow if u.role == "hero"]


def _templates_raw(ctx, flow):
    # grids: units side by side in columns (optionally under a headline tier,
    # optionally beside a full-bleed picture); a separate, fallback stage
    if getattr(ctx, "grids", False):
        # re-stack into rows or columns (strips above all): every unit legible at the type
        # floor, as many lines as it takes; tried only when no faithful layout keeps every promise
        # wide strips also get columns fitted to their height (each text as narrow as it can be)
        fit = (("fit", (1.0,)),) if ctx.W / ctx.H >= 1.5 else ()
        for order in ("reading", "role"):
            for axis, caps in (("rows", (1.0, 0.6, 0.4)), ("cols", (1.0, 0.45, 0.3, 0.2))) + fit:
                for cap in caps:
                    for atom in (False, True, "pieces"):
                        yield 0.0, lambda f=flow, o=order, c=cap, a=axis, t=atom: tmpl_rows(ctx, f, o, c, "center", a, t)
        for k in (2, 3, 4):
            for tiers in (False, True):
                for panel in (None, "left", "right"):
                    if ctx.W / ctx.H >= 1.0:
                        yield 0.0, lambda f=flow, k=k, ti=tiers, pa=panel: tmpl_grid(ctx, f, k, ti, pa)
        return
    for region in ("full", "left", "right", "top", "bottom"):
        for align in ("center", "left"):
            # top/bottom bands on tall canvases, and on clearly wide ones too: on a
            # wide photo the quiet, high-contrast part is often a horizontal band
            if region == "full" or (region in ("top", "bottom") and (ctx.H / ctx.W > 1.2 or ctx.W / ctx.H >= 1.5)) \
                    or (region in ("left", "right") and ctx.W / ctx.H > 1.2):
                yield 0.0, lambda f=flow, a=align, r=region: tmpl_stack(ctx, f, a, r)
                if any(u.role == "hero" for u in f_hero(flow)):
                    # hero-first: hold text back so the hero visual can take the space
                    yield 0.0, lambda f=flow, a=align, r=region: tmpl_stack(ctx, f, a, r, 0.72)
    for side in ("left", "right"):
        for frac in (0.4, 0.5):
            for align in ("left", "center"):
                yield 0.0, lambda f=flow, s=side, fr=frac, a=align: tmpl_split(ctx, f, s, fr, a)
    for region in ("full", "left", "right", "top", "bottom"):
        for align in ("center", "left"):
            if region == "full" or (region in ("left", "right") and ctx.W / ctx.H > 1.2) \
                    or (region in ("top", "bottom") and ctx.H / ctx.W > 1.2):
                yield 0.0, lambda f=flow, a=align, r=region: tmpl_backdrop(ctx, f, a, r)
    for side in ("left", "right", "top", "bottom"):
        for frac in (0.4, 0.55, 0.7):
            yield 0.0, lambda f=flow, s=side, fr=frac: tmpl_card_panel(ctx, f, s, fr)
    for side in ("left", "right", "top", "bottom"):
        for frac in (0.3, 0.45):
            yield 0.0, lambda f=flow, s=side, fr=frac: tmpl_panel(ctx, f, s, fr)
    for scale in (1.0, 0.8, 0.6, 0.4):
        for ncols in (1, 2):
            yield 0.0, lambda f=flow, sc=scale, nc=ncols: tmpl_row(ctx, f, sc, nc)
    # text cards turned into vertical strips (reading bottom to top); text
    # ribbons turned sideways in the source join the row upright
    turned = [u for u in ctx.model["units"] if u.role == "ribbon" and any(e["type"] == "text" for e in u.els)
              and all(abs(abs(norm_rot(e.get("rotation", 0))) - 90) < 2 for e in u.els)]
    for scale in (1.0, 0.8):
        yield 0.0, lambda f=flow + turned, sc=scale: tmpl_row(ctx, f, sc, 1, rotate=True)
    if ctx.W / ctx.H >= 2.2:
        for side in ("left", "right"):
            def rot_panel(f=flow, s=side):
                ctx.rotate = True
                try:
                    P = tmpl_panel(ctx, f, s, 0.45)
                finally:
                    ctx.rotate = False
                if P is not None:
                    P.meta["template"] += "-rot"
                return P
            yield 0.0, rot_panel


def _build(source, target, P):
    out = []
    for el in source["elements"]:
        e = copy.deepcopy(el)
        e.pop("_flow", None)
        for k, v in P.props.get(el["id"], {}).items():
            e[k] = round(v, 2) if isinstance(v, float) else v
        if e["type"] == "text":
            e["font_size"] = max(1, int(round(e["font_size"])))
        out.append(e)
    return {"canvas": {"width": target["width"], "height": target["height"]}, "elements": out}


def _fallback(source, target):
    """Uniform fit-and-centre (no distortion), full-canvas backgrounds cover. Used only when the
    layout search raised; conform() then enforces the hard rules on it."""
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    W, H = target["width"], target["height"]
    s = min(W / sw, H / sh)
    ox, oy = (W - sw * s) / 2, (H - sh * s) / 2
    out = []
    for el in source["elements"]:
        e = dict(el)
        full = el["x"] <= 0 and el["y"] <= 0 and el["x"] + el["width"] >= sw and el["y"] + el["height"] >= sh
        if full and e["type"] in ("image", "shape"):
            e.update(x=0.0, y=0.0, width=W, height=H)
            if e["type"] == "image":
                e["fit"] = "cover"
        else:
            e["x"], e["y"] = ox + el["x"] * s, oy + el["y"] * s
            e["width"], e["height"] = el["width"] * s, el["height"] * s
            if e["type"] == "text":
                e["font_size"] = max(1, int(el["font_size"] * s))
        out.append(e)
    return {"canvas": {"width": W, "height": H}, "elements": out}


def _broken(tight, loose):
    """How many compositions of the tight reading the loose reading splits."""
    where = {}
    for i, u in enumerate(loose["units"]):
        for eid in u.ids:
            where[eid] = i
    return sum(1 for u in tight["units"] if len(u.ids) > 1 and len({where.get(e) for e in u.ids}) > 1)


BREAK_COST = 2.0   # per composition taken apart: it has to buy readability or space
RESTACK_COST = 0.5  # per composition re-stacked along the other axis
FIT_COST = 2.0     # a re-stack into columns as narrow as a strip's height allows
ROTATE_COST = 0.5   # per text piece turned into a vertical strip


def _unwrap(source):
    """Marks texts whose hand-wrapped paragraph breaks can be dropped when the
    text re-flows into a new width (it then wraps cleanly instead of breaking
    at the old points). Text that moves rigidly keeps its content."""
    from .textfit import unwrap_breaks
    src = copy.deepcopy(source)
    sizes = [e["font_size"] for e in src["elements"] if e["type"] == "text"]
    top = max(sizes) if sizes else 0
    for e in src["elements"]:
        # the design's headline (its largest type) keeps its designed lines
        if e["type"] == "text" and isinstance(e.get("content"), str) and e["font_size"] < 0.9 * top:
            flow = unwrap_breaks(e["content"])
            if flow != e["content"]:
                e["_flow"] = flow     # applied only where the text re-flows into a new width
    return src


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """Content never changes (hard rule): hand-set line breaks stay, so _unwrap is not applied."""
    src = normalize_source(source)
    tgt = {"width": max(1.0, num((target_canvas or {}).get("width"), 1.0)),
           "height": max(1.0, num((target_canvas or {}).get("height"), 1.0))}
    for k in ("width", "height"):
        if float(tgt[k]).is_integer():
            tgt[k] = int(tgt[k])
    resize.last_faults = []
    if not src["elements"]:
        return {"canvas": dict(tgt), "elements": [copy.deepcopy(e) for e in src["_other"]]}
    try:
        out = _resize(src, tgt)
    except Exception:  # never let one odd design take the run down; reported, not hidden
        traceback.print_exc(file=sys.stderr)
        resize.last_choice = ({"template": "fallback"}, 0.0, {})
        out = conform(src, _fallback(src, tgt), tgt)
    return out


def _resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """Every candidate keeps every element. Candidates are ranked first by how many of the engine's
    promises their conformed output breaks (small print, glyphs past the canvas, copy over copy,
    a label off its button: ``conform.faults``), then by score. A legible layout always beats an
    illegible one, however well the illegible one scores."""
    resize.last_choice = ({"template": "fallback"}, 0.0, {})
    resize.last_model = None
    resize.last_P = None
    resize.last_errors = []
    W, H = target_canvas["width"], target_canvas["height"]
    tight = analyze(source)
    readings = [(tight, 0)]
    loose = analyze(source, compose=False)
    n = _broken(tight, loose)
    if n:
        readings.append((loose, n))
    best = None    # (key, P, score, breakdown, meta, model, output)
    from .blocks import find_cut
    passes = []
    for model, broken in readings:
        passes.append((model, broken, False))
        carea = model["canvas"][0] * model["canvas"][1]
        if any(u.kind == "group" and u.role in FLOW_ROLES and find_cut(u, carea) for u in model["units"]):
            passes.append((model, broken, True))     # compositions may re-stack
    # grids restructure the design: tried only when no faithful layout keeps every promise
    stages = [False, True]
    for grids, (model, broken, restack) in [(g, p) for g in stages for p in passes]:
        if grids and best is not None and best[0][:2] == (0, 0.0) and best[3].get("readability", 0.0) >= -5.0:
            break
        ctx = Ctx(model, target_canvas)
        ctx.restack = restack
        ctx.grids = grids
        ctx.upright = grids
        for th, make in _candidates(ctx, model):
            try:
                P = make()
                if P is None:
                    continue
                if finalize(ctx, P, model) is None:
                    continue
                s, br = score(ctx, P, model)
            except Exception as ex:  # one bad candidate is skipped, and said so
                resize.last_errors.append(f"{type(ex).__name__}: {ex}")
                print(f"chameleon: candidate skipped: {type(ex).__name__}: {ex}", file=sys.stderr)
                if DEBUG:
                    traceback.print_exc()
                continue
            if broken:
                br["grouping"] = -BREAK_COST * broken
                s += br["grouping"]
            if getattr(P, "lifted", 0):
                br["grouping"] = br.get("grouping", 0.0) - BREAK_COST * P.lifted
                s -= BREAK_COST * P.lifted
            if getattr(P, "rotated", 0):
                br["rotate"] = -ROTATE_COST * P.rotated
                s += br["rotate"]
            if P.meta.get("template", "").startswith("fit-"):
                # narrow columns read worse than rows: they must buy legibility or a clear margin
                br["fit"] = -FIT_COST
                s -= FIT_COST
            if getattr(P, "restacked", 0):
                br["restack"] = -RESTACK_COST * P.restacked
                s += br["restack"]
                P.meta["template"] += " restack"
            out = conform(source, _build(source, target_canvas, P), target_canvas)
            fl = faults(source, out, W, H)
            key = (*fault_key(fl, type_floor(W, H)), -s)
            if DEBUG:
                print(f"    {('loose ' if broken else '') + P.meta['template']:<34} faults={fl[:2]} score={s:7.2f} "
                      + " ".join(f"{k}={v:.1f}" for k, v in br.items()))
            if best is None or key < best[0]:
                meta = dict(P.meta, frame_modes=dict(getattr(P, "frame_modes", {})))
                if broken:
                    meta["template"] = "loose-" + meta["template"]
                best = (key, P, s, br, meta, model, out, fl)
    if best is None:
        raise RuntimeError("no candidate layout placed every element")
    key, P, s, br, meta, model, out, fl = best
    if DEBUG:
        print(f"  -> {meta['template']} faults={len(fl)} score={s:.2f} {fl[:3]}")
    resize.last_choice = (meta, s, br)
    resize.last_P = P
    resize.last_model = model
    resize.last_faults = fl
    return out
