"""
Chameleon resize engine - entry point.

`resize(source, target_canvas)` re-lays a design out for a new canvas instead of
stretching it. The work happens in `engine/smart/`:

  analyze.py   infer roles from geometry (background, decor, headline, CTA,
               badge, ribbon, logo, hero, legal, missing asset...)
  textfit.py   measure and wrap text exactly the way the renderer will
  layout.py    a small flexbox-like engine + largest-scale search
  compose.py   candidate layouts per archetype (stack / split / row), with
               optional hiding of low-importance items for tiny formats
  score.py     rank candidates the way the rubric does
  __init__.py  emit the winner; `finalize` enforces every hard rule

If anything in the smart path fails, a conservative fit-and-centre layout is
returned instead, so the engine never crashes and never emits an invalid file.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any

from engine.smart import finalize, layout_design


def resize(source: dict[str, Any], target_canvas: dict[str, Any],
           constraints: dict[str, Any] | None = None) -> dict[str, Any]:
    if constraints:  # finale-api: pins / priorities / safe area go through engine/api.py
        from engine.api import best

        return best(source, target_canvas, constraints)
    try:
        out = layout_design(source, target_canvas)
    except Exception:  # never let one odd design take the run down
        traceback.print_exc(file=sys.stderr)
        out = _fallback(source, target_canvas)
    return finalize(source, out, target_canvas)


def _fallback(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """Uniform fit-and-centre (no distortion), backgrounds stretched to cover."""
    W0, H0 = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    W, H = float(target_canvas["width"]), float(target_canvas["height"])
    s = min(W / W0, H / H0)
    ox, oy = (W - W0 * s) / 2.0, (H - H0 * s) / 2.0
    out = []
    for el in source["elements"]:
        e = dict(el)
        x, y = float(e.get("x", 0) or 0), float(e.get("y", 0) or 0)
        w, h = float(e.get("width", 0) or 0), float(e.get("height", 0) or 0)
        full = x <= 0 and y <= 0 and x + w >= W0 and y + h >= H0 and e.get("type") in ("image", "shape")
        if full:
            e.update(x=0.0, y=0.0, width=W, height=H)
            if e.get("type") == "image":
                e["fit"] = "cover"
        else:
            e.update(x=ox + x * s, y=oy + y * s, width=w * s, height=h * s)
            if e.get("type") == "text":
                e["font_size"] = max(1, round(float(e.get("font_size", 12) or 12) * s))
            if e.get("corner_radius"):
                e["corner_radius"] = float(e["corner_radius"]) * s
        out.append(e)
    return {"canvas": {"width": int(W), "height": int(H)}, "elements": out}
