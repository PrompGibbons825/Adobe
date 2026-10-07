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
from typing import Any

from .analyze import FLOW_ROLES, analyze
from .finalize import finalize
from .layout import Ctx, tmpl_backdrop, tmpl_card_panel, tmpl_panel, tmpl_row, tmpl_split, tmpl_stack
from .score import score

DEBUG = os.environ.get("CHAMELEON_DEBUG") == "1"
DROP_LEVELS = [0.0, 1.3, 3.5, 5.5]


def _candidates(ctx, model):
    """Every candidate keeps every unit (no drops)."""
    flow = [u for u in model["units"] if u.role in FLOW_ROLES]
    if not flow:
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
    for region in ("full", "left", "right", "top", "bottom"):
        for align in ("center", "left"):
            # top/bottom bands are allowed on any canvas: on a wide photo the
            # quiet, high-contrast part is often a horizontal band
            if region == "full" or region in ("top", "bottom") or (region in ("left", "right") and ctx.W / ctx.H > 1.2):
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
              and all(abs(abs(e.get("rotation", 0) or 0) - 90) < 2 for e in u.els)]
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
        for k, v in P.props.get(el["id"], {}).items():
            e[k] = round(v, 2) if isinstance(v, float) else v
        if e["type"] == "text":
            e["font_size"] = max(1, int(round(e["font_size"])))
        out.append(e)
    return {"canvas": {"width": target["width"], "height": target["height"]}, "elements": out}


def _fallback(source, target):
    """Uniform contain-scale, centered. Never fails; used only on errors."""
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    W, H = target["width"], target["height"]
    s = min(W / sw, H / sh)
    ox, oy = (W - sw * s) / 2, (H - sh * s) / 2
    out = []
    for el in source["elements"]:
        e = dict(el)
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
ROTATE_COST = 0.5   # per text piece turned into a vertical strip


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    resize.last_choice = ({"template": "fallback"}, 0.0, {})
    resize.last_model = None
    try:
        tight = analyze(source)
        readings = [(tight, 0)]
        loose = analyze(source, compose=False)
        n = _broken(tight, loose)
        if n:
            readings.append((loose, n))
        best, best_s, best_br, best_meta, best_model = None, -1e18, None, None, None
        from .blocks import find_cut
        passes = []
        for model, broken in readings:
            passes.append((model, broken, False))
            carea = model["canvas"][0] * model["canvas"][1]
            if any(u.kind == "group" and u.role in FLOW_ROLES and find_cut(u, carea) for u in model["units"]):
                passes.append((model, broken, True))     # compositions may re-stack
        for model, broken, restack in passes:
            ctx = Ctx(model, target_canvas)
            ctx.restack = restack
            for th, make in _candidates(ctx, model):
                try:
                    P = make()
                    if P is None:
                        continue
                    if finalize(ctx, P, model) is None:
                        continue
                    s, br = score(ctx, P, model)
                    if broken:
                        br["grouping"] = -BREAK_COST * broken
                        s += br["grouping"]
                    if getattr(P, "rotated", 0):
                        br["rotate"] = -ROTATE_COST * P.rotated
                        s += br["rotate"]
                    if getattr(P, "restacked", 0):
                        br["restack"] = -RESTACK_COST * P.restacked
                        s += br["restack"]
                        P.meta["template"] += " restack"
                except Exception:  # one bad candidate never kills the run
                    if DEBUG:
                        import traceback
                        traceback.print_exc()
                    continue
                if DEBUG:
                    print(f"    {('loose ' if broken else '') + P.meta['template']:<34} score={s:7.2f} "
                          + " ".join(f"{k}={v:.1f}" for k, v in br.items()))
                if s > best_s:
                    best, best_s, best_br, best_model = P, s, br, model
                    best_meta = dict(P.meta, frame_modes=dict(getattr(P, "frame_modes", {})))
                    if broken:
                        best_meta["template"] = "loose-" + best_meta["template"]
        if best is None:
            return _fallback(source, target_canvas)
        if DEBUG:
            print(f"  -> {best_meta['template']} score={best_s:.2f}")
        resize.last_choice = (best_meta, best_s, best_br)
        resize.last_P = best
        resize.last_model = best_model
        return _build(source, target_canvas, best)
    except Exception:
        if DEBUG:
            import traceback
            traceback.print_exc()
        return _fallback(source, target_canvas)
