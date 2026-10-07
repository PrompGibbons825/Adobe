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
from .layout import Ctx, tmpl_row, tmpl_split, tmpl_stack
from .score import score

DEBUG = os.environ.get("CHAMELEON_DEBUG") == "1"
DROP_LEVELS = [0.0, 1.3, 3.5, 5.5]


def _candidates(ctx, model):
    """Every candidate keeps every unit (no drops)."""
    flow = [u for u in model["units"] if u.role in FLOW_ROLES]
    if not flow:
        return
    for region in ("full", "left", "right", "top", "bottom"):
        for align in ("center", "left"):
            if region == "full" or (region in ("left", "right") and ctx.W / ctx.H > 1.2) \
                    or (region in ("top", "bottom") and ctx.H / ctx.W > 1.2):
                yield 0.0, lambda f=flow, a=align, r=region: tmpl_stack(ctx, f, a, r)
    for side in ("left", "right"):
        for frac in (0.4, 0.5):
            for align in ("left", "center"):
                yield 0.0, lambda f=flow, s=side, fr=frac, a=align: tmpl_split(ctx, f, s, fr, a)
    for scale in (1.0, 0.8, 0.6, 0.4):
        for ncols in (1, 2):
            yield 0.0, lambda f=flow, sc=scale, nc=ncols: tmpl_row(ctx, f, sc, nc)


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


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    resize.last_choice = ({"template": "fallback"}, 0.0, {})
    try:
        model = analyze(source)
        ctx = Ctx(model, target_canvas)
        best, best_s, best_br, best_meta = None, -1e18, None, None
        for th, make in _candidates(ctx, model):
            try:
                P = make()
                if P is None:
                    continue
                if finalize(ctx, P, model) is None:
                    continue
                s, br = score(ctx, P, model)
            except Exception as ex:  # one bad candidate never kills the run
                if DEBUG:
                    import traceback
                    traceback.print_exc()
                continue
            if DEBUG:
                print(f"    {P.meta['template']:<28} drop>={th:<4} score={s:7.2f} "
                      + " ".join(f"{k}={v:.1f}" for k, v in br.items()))
            if s > best_s:
                best, best_s, best_br, best_meta = P, s, br, P.meta
        if best is None:
            return _fallback(source, target_canvas)
        if DEBUG:
            print(f"  -> {best_meta['template']} score={best_s:.2f}")
        resize.last_choice = (best_meta, best_s, best_br)
        return _build(source, target_canvas, best)
    except Exception:
        if DEBUG:
            import traceback
            traceback.print_exc()
        return _fallback(source, target_canvas)
