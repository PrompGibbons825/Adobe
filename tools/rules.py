"""What a valid output is, in one place (dev only; used by tools.stress and tests/).

Two tiers:

* ``hard_rule_errors``  the brief's structural rules. Any output, including the
  safe fallback, must pass these: every id exactly once, everything on the
  canvas, image proportions kept, copy and fonts untouched, and only values the
  reference renderer can draw.
* ``promise_errors``    what the layout engine promises on top (the fallback for
  impossible content does not): no small print, no copy over copy, no text
  drawn past the canvas edge, button labels inside their button.

``ghost_text`` is the one pixel check: the reference renderer ignores text
``opacity``, so a hidden text has to be hidden some other way. It renders the
output with and without its hidden texts and reports any that still paint.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from engine.smart.analyze import normalize
from engine.smart.color import normalize as normalize_colour, parse_rgba
from engine.smart.geom import Box, el_box, rotated_aabb
from engine.smart.items import Ctx
from engine.smart.textfit import drawn_box, new_text_overlaps

ASSETS = Path(__file__).resolve().parents[1] / "brand_kit" / "assets"
FONTS = ASSETS.parent / "fonts"
TOL = 0.6  # px of slack for float rounding at the canvas edge


def _visible_text(e: dict) -> bool:
    return e.get("type") == "text" and float(e.get("opacity", 1) or 0) > 0 and bool(str(e.get("content", "")).strip())


def hard_rule_errors(src: dict, out: dict, tgt: dict) -> List[str]:
    errs: List[str] = []
    W, H = tgt["width"], tgt["height"]
    if out.get("canvas") != {"width": W, "height": H}:
        errs.append(f"canvas {out.get('canvas')} != {W}x{H}")
    sid = [e["id"] for e in src["elements"]]
    oid = [e.get("id") for e in out.get("elements", [])]
    if sorted(map(str, sid)) != sorted(map(str, oid)) or len(set(oid)) != len(oid):
        errs.append("ids differ from the source")
    so = {e["id"]: e for e in src["elements"]}
    for e in out.get("elements", []):
        s = so.get(e.get("id"), {})
        eid = e.get("id")
        if e.get("type") != s.get("type"):
            errs.append(f"{eid}: type changed")
        bad = [k for k in ("x", "y", "width", "height")
               if not isinstance(e.get(k), (int, float)) or not math.isfinite(e[k])]
        if bad:
            errs.append(f"{eid}: bad {bad}")
            continue
        if e["width"] < 0 or e["height"] < 0:
            errs.append(f"{eid}: negative size")
        b = Box(e["x"], e["y"], e["width"], e["height"])
        vb = b.union(rotated_aabb(b, float(e.get("rotation", 0) or 0)))
        if vb.x < -TOL or vb.y < -TOL or vb.r > W + TOL or vb.b > H + TOL:
            errs.append(f"{eid}: off-canvas ({vb.x:.1f},{vb.y:.1f},{vb.r:.1f},{vb.b:.1f})")
        for k in ("content", "font_family", "asset_id", "shape_type"):
            if k in s and e.get(k) != s.get(k):
                errs.append(f"{eid}: {k} changed")
        if e.get("type") == "text":
            errs += _text_safety(e, s)
        if e.get("type") == "image":
            errs += _image_ratio(e, s, W, H)
    return errs


def _text_safety(e: dict, s: dict) -> List[str]:
    errs = []
    eid = e.get("id")
    if not isinstance(e.get("font_size"), int) or e["font_size"] < 1:
        errs.append(f"{eid}: font_size {e.get('font_size')!r}")
    if e["width"] <= 0 or e["height"] <= 0:
        errs.append(f"{eid}: empty text box")
    if e.get("text_align", "left") not in ("left", "center", "right"):
        errs.append(f"{eid}: text_align {e.get('text_align')!r}")
    if "font_weight" in e and not isinstance(e["font_weight"], str):
        errs.append(f"{eid}: font_weight {e['font_weight']!r}")
    col = e.get("color")
    if not (isinstance(col, str) and col.strip().lower().startswith("rgb")):
        errs.append(f"{eid}: colour {col!r} is not rgba()")
    elif normalize_colour(s.get("color")) and parse_rgba(col) != parse_rgba(normalize_colour(s.get("color"))):
        errs.append(f"{eid}: colour changed {s.get('color')!r} -> {col!r}")
    return errs


def _image_ratio(e: dict, s: dict, W: float, H: float) -> List[str]:
    sw, sh = float(s.get("width", 0) or 0), float(s.get("height", 0) or 0)
    if sw <= 0 or sh <= 0:
        return []
    if e["width"] <= 0 or e["height"] <= 0:
        return [f"{e.get('id')}: image collapsed to {e['width']}x{e['height']}"]
    full = e["x"] <= TOL and e["y"] <= TOL and e["width"] >= W - 1 and e["height"] >= H - 1 \
        and e.get("fit") == "cover"
    ra, rb = sw / sh, e["width"] / e["height"]
    if not full and abs(rb / ra - 1) > 0.05:
        return [f"{e.get('id')}: aspect {ra:.3f} -> {rb:.3f}"]
    return []


def promise_errors(src: dict, out: dict, tgt: dict) -> List[str]:
    errs: List[str] = []
    W, H = tgt["width"], tgt["height"]
    floor = Ctx(W=W, H=H, W0=1, H0=1).judge_floor
    texts = [e for e in out["elements"] if _visible_text(e)]
    for e in texts:
        if e["font_size"] < floor:
            errs.append(f"{e['id']}: small print {e['font_size']}px < {floor}px")
        db = drawn_box(e)
        if db.x < -TOL or db.y < -TOL or db.r > W + TOL or db.b > H + TOL:
            errs.append(f"{e['id']}: glyphs drawn past the canvas ({db.x:.0f},{db.y:.0f},{db.r:.0f},{db.b:.0f})")
    src_norm = [normalize(e, i) for i, e in enumerate(src["elements"])]  # as the renderer would draw it
    for a, b, area, _ in new_text_overlaps(out["elements"], src_norm):
        errs.append(f"copy over copy: {a} x {b} ({area:.0f} px2)")
    errs += _labels_in_buttons(src, out)
    return errs


def _labels_in_buttons(src: dict, out: dict) -> List[str]:
    """A text that sat inside a shape of its own group stays inside it."""
    errs = []
    so = {e["id"]: e for e in src["elements"]}
    oo = {e["id"]: e for e in out["elements"]}
    groups: Dict[str, List[dict]] = {}
    for e in src["elements"]:
        if e.get("group_id"):
            groups.setdefault(str(e["group_id"]), []).append(e)
    for members in groups.values():
        shapes = [m for m in members if m.get("type") == "shape"]
        for t in (m for m in members if m.get("type") == "text"):
            for sh in shapes:
                if not _inside(drawn_box(so[t["id"]]), el_box(so[sh["id"]]), 2.0):
                    continue
                ot, osh = oo[t["id"]], oo[sh["id"]]
                if not _visible_text(ot) or float(osh.get("opacity", 1) or 0) <= 0:
                    continue
                if not _inside(drawn_box(ot), el_box(osh), 2.0) and not osh.get("rotation"):
                    errs.append(f"{t['id']}: label left its shape {sh['id']}")
    return errs


def _inside(a: Box, b: Box, tol: float) -> bool:
    return a.x >= b.x - tol and a.y >= b.y - tol and a.r <= b.r + tol and a.b <= b.b + tol


# ----------------------------------------------------------------------------
# rendered checks


def render_image(design: dict, assets: Path = ASSETS):
    """The design through the reference renderer (brand-kit fonts), as an RGB
    PIL image. Each call renders into its own temporary file, so parallel runs
    cannot read each other's half-written PNGs."""
    from PIL import Image

    from renderer.converter.convert import render_design_to_png

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.png"
        render_design_to_png(design, assets, str(path), fonts_dir=FONTS)
        with Image.open(path) as im:
            return im.convert("RGB")


def render(design: dict, assets: Path = ASSETS):
    """The rendered design as a greyscale numpy array."""
    import numpy as np

    return np.asarray(render_image(design, assets).convert("L"), dtype=np.int16)


def ghost_text(out: dict, assets: Path = ASSETS, min_pixels: int = 12) -> List[str]:
    """Hidden texts (opacity 0) that the renderer still paints."""
    hidden = [e for e in out["elements"] if e.get("type") == "text" and float(e.get("opacity", 1) or 0) <= 0
              and str(e.get("content", "")).strip()]
    if not hidden:
        return []
    full = render(out, assets)
    errs = []
    for e in hidden:
        without = dict(out, elements=[x for x in out["elements"] if x is not e])
        diff = int((abs(full - render(without, assets)) > 24).sum())
        if diff >= min_pixels:
            errs.append(f"{e['id']}: hidden text still paints {diff} px")
    return errs


def rule_report(src: dict, out: dict, tgt: dict, smart: bool = True, render_check: bool = False,
                assets: Optional[Path] = None) -> List[str]:
    errs = hard_rule_errors(src, out, tgt)
    if smart:
        errs += promise_errors(src, out, tgt)
    if render_check:
        errs += ghost_text(out, assets or ASSETS)
    return errs
