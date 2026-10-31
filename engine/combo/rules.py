"""Adobe's stage-one structural check as we read it on 2026-10-31, plus the no-hiding rule.

    from engine.combo.rules import faults
    faults(source, output, target_canvas) -> ["<id>: <why>", ...]   # [] = the output passes

The brief checks the output JSON "structurally: same element ids as the source (nothing added or
removed), required fields present, nothing off-canvas, images kept in proportion". Read with
Vlatko's rulings (2026-10-31):

* Crops are allowed; only a stretch is a fault. An image box whose ratio changed is fine when the
  renderer crops (``cover``) or letterboxes (``contain``) it, and a fault when it would distort it.
* Hand line breaks do not occur in Adobe's inputs, so copy compares with whitespace collapsed.
* Hiding counts as dropping: opacity lowered on an element the source showed, copy set at 2 px or
  less, or a box collapsed below 1 px far faster than the canvas shrank. An image whose asset file
  is missing (it would paint a grey placeholder) is exempt.
* Off-canvas follows the bleed policy (``engine/combo/bleed.py``): photos and decoration may bleed,
  copy, logos and buttons may not (unless the source bled them).

Self-contained (only ``bleed``), so the combined engine and the benchmark share one definition.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

from . import bleed

ASPECT_TOL = 0.05
OPACITY_TOL = 0.05
TINY_TEXT = 2  # px: copy at or below this is unreadable at any zoom, i.e. hidden


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


def _norm(s) -> str:
    return " ".join(str(s or "").split())


def _op(e: dict) -> float:
    v = e.get("opacity", 1)
    return 1.0 if v is None else float(v)


def faults(src: dict, out: dict, tgt: dict, assets=None) -> List[str]:
    W, H = float(tgt["width"]), float(tgt["height"])
    SW, SH = float(src["canvas"]["width"]), float(src["canvas"]["height"])
    errs: List[str] = []
    if not isinstance(out, dict) or not isinstance(out.get("elements"), list):
        return ["output is not a design"]
    c = out.get("canvas") or {}
    if _num(c.get("width")) != W or _num(c.get("height")) != H:
        errs.append(f"canvas {c} != {int(W)}x{int(H)}")
    sid = [str(e.get("id")) for e in src.get("elements", [])]
    oid = [str(e.get("id")) for e in out["elements"]]
    if sorted(sid) != sorted(oid):
        errs.append("ids differ from the source")
    so = {str(e.get("id")): e for e in src.get("elements", [])}
    adir = bleed._assets_dir(assets)
    placed = []
    for e in out["elements"]:
        i = str(e.get("id"))
        s = so.get(i)
        if s is None:
            continue
        if e.get("type") != s.get("type"):
            errs.append(f"{i}: type changed")
        x, y, w, h = (_num(e.get(k)) for k in ("x", "y", "width", "height"))
        if None in (x, y, w, h):
            errs.append(f"{i}: missing or non-numeric geometry")
            continue
        placed.append(e)
        for k in ("font_family", "asset_id", "shape_type"):
            if k in s and e.get(k) != s.get(k):
                errs.append(f"{i}: {k} changed")
        if "content" in s and _norm(e.get("content")) != _norm(s.get("content")):
            errs.append(f"{i}: content changed")
        if _rgba(s.get("color")) and _rgba(e.get("color")) and _rgba(e.get("color")) != _rgba(s.get("color")):
            errs.append(f"{i}: colour changed")
        sw, sh = _num(s.get("width")) or 0.0, _num(s.get("height")) or 0.0
        # An image whose asset file is missing would paint a grey placeholder; keeping that out of
        # sight hides nothing real (the L3 sample's holiday badge), so no hiding check applies.
        missing = s.get("type") == "image" and s.get("asset_id") and not (adir / str(s["asset_id"])).is_file()
        # A box under 1 px is hidden when it shrank far more than the canvas did; a hairline rule
        # scaled with the canvas (3 px on a tall post -> 0.6 px on a banner) is not.
        if not missing and (w < 1 or h < 1) and sw >= 2 and sh >= 2 \
                and min(max(w, 0) / sw, max(h, 0) / sh) < 0.25 * min(W / SW, H / SH):
            errs.append(f"{i}: hidden (box collapsed to {w:.1f}x{h:.1f})")
            continue
        visible_in_src = bleed.share_on_canvas(s, SW, SH) > 0
        if visible_in_src and not missing and _op(e) < _op(s) - OPACITY_TOL:
            errs.append(f"{i}: hidden (opacity {_op(s):.2f} -> {_op(e):.2f})")
        if e.get("type") == "text":
            fs = _num(e.get("font_size"))
            if fs is None or fs < 1:
                errs.append(f"{i}: font_size {e.get('font_size')!r}")
            elif fs <= TINY_TEXT < (_num(s.get("font_size")) or 99):
                errs.append(f"{i}: hidden (text at {fs:g}px)")
        if e.get("type") == "image" and sw > 0 and sh > 0:
            ra, rb = sw / sh, w / h
            if abs(rb / ra - 1) > ASPECT_TOL and e.get("fit", "contain") not in ("cover", "contain"):
                errs.append(f"{i}: stretched ({ra:.3f} -> {rb:.3f}, fit {e.get('fit')!r})")
    errs += bleed.check(src, dict(out, elements=placed), tgt, assets)[0]
    return errs


def edge_any(out: dict, tgt: dict) -> List[str]:
    """Ids whose box is past the canvas by the organizers' own self-check (engine/main.py: more
    than 0.5 px outside), which is stricter than the bleed policy. Reported, not a fault."""
    W, H = float(tgt["width"]), float(tgt["height"])
    res = []
    for e in out.get("elements", []):
        x, y, w, h = (_num(e.get(k)) for k in ("x", "y", "width", "height"))
        if None in (x, y, w, h):
            continue
        if x < -0.5 or y < -0.5 or x + w > W + 0.5 or y + h > H + 0.5:
            res.append(str(e.get("id")))
    return res
