"""The stage-one structural rules as decided on 2026-10-31 (context/notes/adobe-rules.md, workspace c0b8026).

    from engine.combo.rules import faults
    faults(source, output, target_canvas) -> ["<id>: <why>", ...]   # [] = the output passes

The combined engine drops every engine output that fails this check before it picks, and the
Adobe-oriented benchmark scores with the same function. One rule per line of adobe-rules.md:

* Output shape: only ``canvas`` and ``elements`` at the top level, the canvas is the target, the
  required fields per element type are present (rules 5-6).
* Every source id exactly once; type, ``content`` (byte for byte), ``font_family``, ``asset_id`` and
  ``shape_type`` unchanged; ``font_size`` any number >= 1 (rules 2, 9, 10).
* No recolouring: text ``color`` and shape ``fill_color`` / ``stroke_color`` unchanged (rule 11).
* Nothing past the canvas by more than 0.5 px, decoration included (rule 7).
* Never stretch: an image changes aspect only when full-bleed or ``fit: "cover"``, and a cut-out
  (logo, product) is never cropped (rule 8).
* Every element stays visible (rules 3 and 21): no opacity drop on an element the source showed,
  no type at 2 px or less, no box shrunk to nothing, nothing buried under opaque layers above it.
  An image whose asset file is missing counts too: it must stay as a visible placeholder.

Self-contained (``bleed``, ``visible``, ``fitgeom`` are vendored), so it runs in the image.
Until 2ff0ee6 this module held a looser reading (crops and bleed allowed, whitespace-only copy
changes allowed, missing assets exempt); the benchmark showed that reading let through outputs
that the decided rules zero.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

from . import bleed
from .visible import _covered

REQ = {"text": ["content", "font_family", "font_size", "color"], "image": ["asset_id"], "shape": ["shape_type", "fill_color"]}
COMMON = ["id", "type", "x", "y", "width", "height"]
EDGE = 0.5  # px past the canvas the organizers' self-check tolerates
ASPECT_TOL = 0.05
OPACITY_TOL = 0.05
TINY_TEXT = 2  # px: copy at or below this is near-zero type, i.e. hidden
BURIED = 0.9  # share of a box under opaque layers above it that counts as buried


def _rgba(v) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(v, str):
        return None
    m = re.match(r"\s*rgba?\(([^)]*)\)\s*$", v.strip().lower())
    if not m:
        return None
    try:
        parts = [float(p) for p in m.group(1).replace("/", ",").split(",") if p.strip()]
    except ValueError:
        return None
    if len(parts) == 3:
        parts.append(1.0)
    return tuple(round(p, 3) for p in parts[:4]) if len(parts) >= 4 else None


def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _op(e: dict) -> float:
    v = e.get("opacity", 1)
    return 1.0 if v is None else float(v)


def faults(src: dict, out: dict, tgt: dict, assets=None) -> List[str]:
    W, H = float(tgt["width"]), float(tgt["height"])
    SW, SH = float(src["canvas"]["width"]), float(src["canvas"]["height"])
    if not isinstance(out, dict) or not isinstance(out.get("elements"), list):
        return ["output is not a design"]
    errs: List[str] = []
    if set(out) - {"canvas", "elements"}:
        errs.append(f"extra top-level keys {sorted(set(out) - {'canvas', 'elements'})}")
    c = out.get("canvas") or {}
    if not isinstance(c, dict) or set(c) != {"width", "height"} or _num(c.get("width")) != W or _num(c.get("height")) != H:
        errs.append(f"canvas {c} != {int(W)}x{int(H)}")
    if sorted(str(e.get("id")) for e in src["elements"]) != sorted(str(e.get("id")) for e in out["elements"]):
        errs.append("ids differ from the source")
    so = {str(e.get("id")): e for e in src["elements"]}
    adir = bleed._assets_dir(assets)
    cov_s, cov_o = _covered(src["elements"], assets), _covered(out["elements"], assets)
    for e in out["elements"]:
        i = str(e.get("id"))
        s = so.get(i)
        if s is None:
            continue
        for k in COMMON + REQ.get(s.get("type"), []):
            if k not in e:
                errs.append(f"{i}: missing field {k}")
        if e.get("type") != s.get("type"):
            errs.append(f"{i}: type changed")
        x, y, w, h = (_num(e.get(k)) for k in ("x", "y", "width", "height"))
        if None in (x, y, w, h):
            errs.append(f"{i}: non-numeric geometry")
            continue
        for k in ("content", "font_family", "asset_id", "shape_type"):
            if k in s and e.get(k) != s.get(k):
                errs.append(f"{i}: {k} changed")
        for k in ("color", "fill_color", "stroke_color"):
            if _rgba(s.get(k)) is not None and _rgba(e.get(k)) != _rgba(s.get(k)):
                errs.append(f"{i}: {k} changed")
        if s.get("type") == "text":
            fs = _num(e.get("font_size"))
            if fs is None or fs < 1:
                errs.append(f"{i}: font_size {e.get('font_size')!r}")
            elif fs <= TINY_TEXT < (_num(s.get("font_size")) or 99):
                errs.append(f"{i}: hidden (near-zero type {fs:g}px)")
        if x < -EDGE or y < -EDGE or x + w > W + EDGE or y + h > H + EDGE:
            errs.append(f"{i}: off-canvas")
        sw, sh = _num(s.get("width")) or 0.0, _num(s.get("height")) or 0.0
        if w <= 0 or h <= 0:
            errs.append(f"{i}: hidden (empty box)")
            continue
        # A box under 1 px is hidden when it shrank far more than the canvas did; a hairline rule
        # scaled with the canvas (3 px on a tall post -> 0.6 px on a banner) is not.
        if (w < 1 or h < 1) and sw >= 2 and sh >= 2 and min(w / sw, h / sh) < 0.25 * min(W / SW, H / SH):
            errs.append(f"{i}: hidden (shrunk to {w:.1f}x{h:.1f})")
        if bleed.share_on_canvas(s, SW, SH) > 0 and _op(e) < _op(s) - OPACITY_TOL:
            errs.append(f"{i}: hidden (opacity {_op(s):.2f} -> {_op(e):.2f})")
        src_bg = sw * sh >= 0.9 * SW * SH
        if not src_bg and cov_o.get(i, 0) >= BURIED and cov_s.get(i, 0) < 0.5:
            errs.append(f"{i}: hidden (buried {cov_o[i]:.0%} under opaque layers)")
        if s.get("type") == "image" and sw > 0 and sh > 0 and abs((w / h) / (sw / sh) - 1) > ASPECT_TOL:
            full = x <= EDGE and y <= EDGE and x + w >= W - EDGE and y + h >= H - EDGE
            fit = e.get("fit", "contain")
            cutout = bleed._transparent(str(adir / str(s.get("asset_id", "")))) is True and not src_bg
            if not (full or fit == "cover"):
                errs.append(f"{i}: aspect changed without a crop (fit {fit!r})")
            elif cutout and not full:
                errs.append(f"{i}: logo/product cropped")
    return errs


def edge_any(out: dict, tgt: dict) -> List[str]:
    """Ids whose box is past the canvas by more than 0.5 px (the organizers' own self-check)."""
    W, H = float(tgt["width"]), float(tgt["height"])
    res = []
    for e in out.get("elements", []):
        x, y, w, h = (_num(e.get(k)) for k in ("x", "y", "width", "height"))
        if None in (x, y, w, h):
            continue
        if x < -EDGE or y < -EDGE or x + w > W + EDGE or y + h > H + EDGE:
            res.append(str(e.get("id")))
    return res
