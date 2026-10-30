"""What a valid output is, in one place (dev only; used by tools.stress and tests/).

Two tiers:

* ``hard_rule_errors``  the brief's structural rules. Any output, including the
  safe fallback, must pass these: every id exactly once, nothing past the
  canvas except allowed bleed (``tools/bleed.py``, loop/STEERING.md S4: never
  text, logos, buttons, badges or cutouts; photos, image panels and decoration
  only while they keep 60% on the canvas), image proportions kept, copy and
  fonts untouched, and only values the reference renderer can draw.
* ``bleed_report``      the allowed bleed, reported (one line per element), never an error.
* ``promise_errors``    what the layout engine promises on top (the fallback for
  impossible content does not): no small print, no copy over copy, no text
  drawn past the canvas edge, button labels inside their button.

Text geometry comes from ``fundamentals`` (the renderer's own wrap and draw rules, pinned by
tests/conformance), not from the engine under test, so this file checks any engine tree.

``ghost_text`` is the one pixel check: the reference renderer ignores text
``opacity``, so a hidden text has to be hidden some other way. It renders the
output with and without its hidden texts and reports any that still paint.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from dataclasses import dataclass

from fundamentals.text import measure_text
from tools import bleed

ASSETS = Path(__file__).resolve().parents[1] / "brand_kit" / "assets"
FONTS = ASSETS.parent / "fonts"
TOL = 0.6  # px of slack for float rounding at the canvas edge


# ----------------------------------------------------------------------------
# geometry from the renderer model (fundamentals), never from an engine's own measurement


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def r(self) -> float:
        return self.x + self.w

    @property
    def b(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


def _f(v, d=0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return d
    return f if math.isfinite(f) else d


def el_box(e: dict) -> Box:
    return Box(_f(e.get("x")), _f(e.get("y")), max(0.0, _f(e.get("width"))), max(0.0, _f(e.get("height"))))


def drawn_box(e: dict) -> Box:
    """Where the renderer puts a text's ink (fundamentals.measure_text: its wrap at the float
    size, glyphs at the int size, letter spacing, runs, alignment, overhang, rotation)."""
    try:
        b = measure_text(e).canvas_ink_box()
    except Exception:  # noqa: BLE001  unmeasurable (bad values): the element box
        return el_box(e)
    if b is None:
        x, y = _f(e.get("x")), _f(e.get("y"))
        return Box(x, y, 0.0, 0.0)
    return Box(b[0], b[1], b[2] - b[0], b[3] - b[1])


def judge_floor(W: float, H: float) -> int:
    """No visible text below this: cap height >= 7 px when the judge sees the canvas at 1024 px on
    the long side (research 04), and never under 10 px."""
    return max(10, math.ceil(10.0 * max(W, H) / 1024.0))


def _rgba(v):
    """(r, g, b, a) of any CSS-ish colour (rgba(), rgb(), hex, named), or None."""
    if not isinstance(v, str):
        return None
    t = v.strip().lower()
    if t.startswith("rgb"):
        try:
            parts = [float(p) for p in t[t.index("(") + 1:t.rindex(")")].split(",")]
            return (int(parts[0]), int(parts[1]), int(parts[2]), round(parts[3] if len(parts) > 3 else 1.0, 3))
        except (ValueError, IndexError):
            return None
    try:
        from PIL import ImageColor

        c = ImageColor.getcolor(t, "RGBA")
        return (c[0], c[1], c[2], round(c[3] / 255.0, 3))
    except (ValueError, AttributeError):
        return None


def text_overlaps(elements, tol: float = 1.0):
    """Pairs of visible texts whose drawn ink boxes overlap (beyond ``tol`` px):
    (a, b, area, area / smaller box)."""
    texts = [e for e in elements if _visible_text(e)]
    boxes = [(e["id"], drawn_box(e)) for e in texts]
    out = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (ia, a), (ib, b) = boxes[i], boxes[j]
            ow = min(a.r, b.r) - max(a.x, b.x)
            oh = min(a.b, b.b) - max(a.y, b.y)
            if ow > tol and oh > tol:
                out.append((ia, ib, ow * oh, ow * oh / max(1.0, min(a.area, b.area))))
    return out


def new_text_overlaps(elements, source_elements, deliberate: float = 0.15, slack: float = 0.02):
    """Copy drawn over copy that the source did not draw that way: the pair overlaps by a larger
    share of the smaller box than in the source (plus ``slack``); pairs the source overlapped by
    ``deliberate`` or more (stacked display type, a drop shadow) are the design."""
    src = {(a, b): f for a, b, _, f in text_overlaps(source_elements)}
    out = []
    for a, b, area, f in text_overlaps(elements):
        f0 = max(src.get((a, b), 0.0), src.get((b, a), 0.0))
        if f0 < deliberate and f > f0 + slack:
            out.append((a, b, area, f))
    return out


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
    off: List[dict] = []
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
        if e["width"] >= 0 and e["height"] >= 0:
            off.append(e)
        for k in ("content", "font_family", "asset_id", "shape_type"):
            if k in s and e.get(k) != s.get(k):
                errs.append(f"{eid}: {k} changed")
        if e.get("type") == "text":
            errs += _text_safety(e, s)
        if e.get("type") == "image":
            errs += _image_ratio(e, s, W, H)
    errs += bleed.check(src, dict(out, elements=off), tgt)[0]
    return errs


def bleed_report(src: dict, out: dict, tgt: dict) -> List[str]:
    """Allowed bleed (S4), one line per element past the canvas. Report only: a little bleed
    may cost on the auto grader, so the gate shows it per tile, but only the panel judges it."""
    return bleed.check(src, out, tgt)[1]


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
    elif _rgba(s.get("color")) and _rgba(col) != _rgba(s.get("color")):
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


def floor_area(texts: List[dict], floor: float) -> float:
    """Least ink-box area the visible ``texts`` can occupy with every one at ``floor`` px: each
    set on a single line (wrapping only adds area) at the floor size, width x line step."""
    total = 0.0
    fs = max(1, int(round(floor)))
    for e in texts:
        lay = measure_text(dict(e, font_size=fs, width=1e9, rotation=0), with_ink=False)
        total += lay.max_line_width * fs * lay.line_height * max(1, lay.n_lines)
    return total


CAPACITY_K = 1.0  # exactly 1.0: above it the texts cannot all fit on the canvas at the floor, provably


def capacity_impossible(out: dict, tgt: dict) -> bool:
    """True when ``floor_area(texts, floor) > 1.0 * W * H``: holding the small-print floor is
    impossible, so small print is reported as ``capacity`` instead of a failure (never as passing)."""
    W, H = tgt["width"], tgt["height"]
    floor = judge_floor(W, H)
    texts = [e for e in out["elements"] if _visible_text(e)]
    return floor_area(texts, floor) > CAPACITY_K * W * H


def capacity_errors(out: dict, tgt: dict) -> List[str]:
    """The small-print findings that ``promise_errors`` exempts because of capacity (else [])."""
    if not capacity_impossible(out, tgt):
        return []
    floor = judge_floor(tgt["width"], tgt["height"])
    return [f"capacity: {e['id']}: {e['font_size']}px < {floor}px, texts need more than the canvas at the floor"
            for e in out["elements"] if _visible_text(e) and e["font_size"] < floor]


def promise_errors(src: dict, out: dict, tgt: dict) -> List[str]:
    errs: List[str] = []
    W, H = tgt["width"], tgt["height"]
    floor = judge_floor(W, H)
    texts = [e for e in out["elements"] if _visible_text(e)]
    cap = capacity_impossible(out, tgt)
    for e in texts:
        if e["font_size"] < floor and not cap:
            errs.append(f"{e['id']}: small print {e['font_size']}px < {floor}px")
        db = drawn_box(e)
        if db.x < -TOL or db.y < -TOL or db.r > W + TOL or db.b > H + TOL:
            errs.append(f"{e['id']}: glyphs drawn past the canvas ({db.x:.0f},{db.y:.0f},{db.r:.0f},{db.b:.0f})")
    for a, b, area, _ in new_text_overlaps(out["elements"], src["elements"]):
        errs.append(f"copy over copy: {a} x {b} ({area:.0f} px2)")
    errs += _labels_in_buttons(src, out)
    return errs


def hierarchy_errors(src: dict, out: dict) -> List[str]:
    """Type hierarchy inversions: a text set at least 25% larger than another
    in the source must not end up (more than a pixel) smaller than it."""
    so = {e["id"]: e for e in src["elements"] if e.get("type") == "text"}
    shown = [e for e in out["elements"] if _visible_text(e) and e["id"] in so]
    errs = []
    for a in shown:
        for b in shown:
            fa, fb = float(so[a["id"]].get("font_size") or 0), float(so[b["id"]].get("font_size") or 0)
            if fa >= 1.25 * fb and a["font_size"] < b["font_size"] - 1:  # a pixel is rounding
                errs.append(f"hierarchy: {a['id']} ({a['font_size']}px) now smaller than {b['id']} ({b['font_size']}px)")
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


def contrast_errors(src: dict, out: dict, tgt: dict) -> List[str]:
    """Copy that lost its contrast: below 3:1 against what is under it and
    clearly worse than in the source (white copy that left its dark panel for
    a white canvas). Measured like the engine's pixel critic: WCAG contrast at
    the 10th percentile of the region, on a small render without the copy."""
    try:
        from engine.smart import pixels
    except ImportError:  # an engine without engine/smart: measure on the renderer's pixels
        return _contrast_errors_rendered(src, out, tgt)

    hidden = {e["id"] for e in out["elements"] if float(e.get("opacity", 1) or 0) <= 0}
    before = pixels.measure(src["elements"], src["canvas"]["width"], src["canvas"]["height"], skip=set())
    after = pixels.measure(out["elements"], tgt["width"], tgt["height"], skip=hidden)
    errs = []
    for tid, (c, _) in after.items():
        c0 = before.get(tid, (4.5, 0.0))[0]
        if c < 3.0 and c < 0.7 * min(c0, 4.5):
            errs.append(f"{tid}: contrast {c:.1f}:1 (source {c0:.1f}:1)")
    return errs


def _contrast_errors_rendered(src: dict, out: dict, tgt: dict) -> List[str]:
    """contrast_errors without the incumbent's pixel critic (fundamentals: median of what is under
    each text's ink, on a render without it)."""
    from fundamentals.check import _Px

    after = _Px(dict(out, canvas={"width": tgt["width"], "height": tgt["height"]}), None, None)
    before = _Px(src, None, None)
    so = {e["id"]: e for e in src["elements"]}
    errs = []
    for e in out["elements"]:
        if not _visible_text(e) or e["id"] not in so:
            continue
        c = after.contrast(e)
        c0 = before.contrast(so[e["id"]]) or 4.5
        if c is not None and c < 3.0 and c < 0.7 * min(c0, 4.5):
            errs.append(f"{e['id']}: contrast {c:.1f}:1 (source {c0:.1f}:1)")
    return errs


def rule_report(src: dict, out: dict, tgt: dict, smart: bool = True, render_check: bool = False,
                assets: Optional[Path] = None) -> List[str]:
    errs = hard_rule_errors(src, out, tgt)
    if smart:
        errs += promise_errors(src, out, tgt)
    if render_check:
        errs += ghost_text(out, assets or ASSETS)
    return errs
