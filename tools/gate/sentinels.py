"""Gate tool: build sentinels a judge can actually see (G6-fix, round 6).

A sentinel is a good design against a copy with one known defect. In round 5 both panel
families fell below 0.85 on the sealed sentinel pool while passing on dev, and a look at the
pairs showed why: the "good" side was often not clean (designer resizes drawn with fallback
fonts overflow their boxes, copy runs into copy or off the canvas), the defect sometimes landed
on a minor text, and some defects barely changed a pixel. Three checks, all general (none looks
at a particular design):

* ``good_ok(source, good, tgt)``: the good side must not already carry the defects sentinels
  plant: no copy over copy at all (``OVERLAP_MAX``), no glyphs drawn past the canvas and no text hidden
  (``tools.gate.visible``), the last two relative to the source (a bleed the source has is the
  design, not a defect). Rendered, the good side's copy must not reach the canvas edge where
  the source's does not (``edge_ink``, ``rendered_ok``): drawn boxes miss the cuts that fallback
  fonts and letter spacing make, and the judges in round 6's dev check named exactly those. A picture or shape the designer dropped is not checked: it is a
  choice, not one of the planted defects.
* ``structural(kind, source, good, bad)``: the planted defect is the one named: ``p_crop`` and
  ``p_clip_letter`` cut the **headline** (the largest text: largest font size, and of near-equal
  sizes the largest drawn area, ``tools.judge.perturb.headline``), which must be prominent
  (``HEADLINE_MIN`` of the canvas), upright (a rotated text's cut is not where the move puts it)
  and not already cut; ``p_crop`` puts at least a quarter of its ink past the edge;
  ``p_onart`` moves a headline that sat mostly off the picture onto it; ``fit_centre`` needs a
  real change of shape (``FIT_MIN_RATIO``).
* ``pixel_ok(good_png, bad_png, canvas, region)``: rendered, the two sides differ visibly: at least
  ``min_diff`` of the pixels change by more than ``DIFF_LEVEL`` (of 255), and, for a defect on
  one element, at least ``REGION_MIN`` of the element's new drawn box changes (its ink shows
  there: not white copy moved onto white).

``tools.gate.panel.Panel.sentinels`` drops every candidate that fails one of them and tries
another (tile, kind). This is measuring code; the engine is only read.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

DIFF_LEVEL = 40        # a channel change above this (of 255) counts as a changed pixel
MIN_DIFF = 0.006       # share of the canvas that must change (panel cfg sentinel_min_diff)
REGION_MIN = 0.06      # share of the moved element's new box that must change
HEADLINE_MIN = 0.004   # the headline's drawn box, as a share of the canvas area, at least
FIT_MIN_RATIO = 1.3    # fit-and-centre only when the aspect ratio changes at least this much
OFF_ART_MAX = 0.25     # p_onart: the headline's share on the picture before the move, at most
OVERLAP_MAX = 0.10     # good side: two texts' drawn boxes may share this much of the smaller, at most
EDGE_STRIP = 0.004     # rendered check: the border strip, as a share of the side
EDGE_MAX = 0.002       # rendered check: text ink on more than this share of the border is a cut
CROP_MIN = 0.25        # p_crop: share of the headline's ink past the edge, at least

ONE_ELEMENT = ("p_crop", "p_onart", "p_clip_letter", "p_edge_touch", "p_lowcon", "p_mis20", "p_mis8",
               "p_typescale", "p_shrinksub")


def _live_text(e: dict) -> bool:
    op = e.get("opacity", 1)
    return (e.get("type") == "text" and (1.0 if op is None else float(op)) > 0 and str(e.get("content", "")).strip()
            and float(e.get("width", 0) or 0) > 1 and float(e.get("height", 0) or 0) > 1)


def headline(design: dict) -> Optional[dict]:
    """The largest text (``tools.judge.perturb.headline``): the text the headline defects move."""
    from tools.judge.perturb import headline as h

    return h(design)


def _on_canvas(b, W: float, H: float) -> float:
    if b.area <= 0:
        return 0.0
    iw = max(0.0, min(b.r, W) - max(b.x, 0.0))
    ih = max(0.0, min(b.b, H) - max(b.y, 0.0))
    return iw * ih / b.area


def good_ok(source: dict, good: dict, tgt: Optional[dict] = None) -> List[str]:
    """Why ``good`` is not a clean good side ([] when it is)."""
    from engine.smart.textfit import drawn_box, text_overlaps
    from tools.gate.visible import hidden_elements

    W = float((tgt or good["canvas"])["width"])
    H = float((tgt or good["canvas"])["height"])
    SW, SH = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    src = {e["id"]: e for e in source["elements"]}
    why = []
    for e in good["elements"]:
        if not _live_text(e):
            continue
        on = _on_canvas(drawn_box(e), W, H)
        s = src.get(e["id"])
        on_s = _on_canvas(drawn_box(s), SW, SH) if s is not None and _live_text(s) else 1.0
        if on < 0.98 and on < on_s - 0.02:
            why.append(f"{e['id']}: glyphs {1 - on:.0%} off the canvas")
    for a, b, _, share in text_overlaps(good["elements"]):
        # even where the source overlaps too: drawn with fallback fonts a designer's overlay
        # often collides, and a judge cannot tell a broken good side from the planted defect.
        # Line boxes touching by a sliver (leading) are not a collision.
        if share > OVERLAP_MAX:
            why.append(f"copy over copy: {a} x {b} ({share:.0%})")
    texts = {str(e["id"]) for e in source["elements"] if _live_text(e)}
    why += [f"hidden {h}" for h in hidden_elements(source, good, {"width": W, "height": H})
            if h.split(":", 1)[0] in texts]
    return why


def _no_text(design: dict) -> dict:
    """``design`` without its texts (only rendered; the renderer paints even a text at opacity 0)."""
    return dict(design, elements=[e for e in design["elements"] if e.get("type") != "text"])


def edge_ink(with_text: Path, without_text: Path) -> float:
    """Share of the canvas border (a strip ``EDGE_STRIP`` of the side, at least 2 px) that a
    text's rendered ink covers: the pixels that change when the texts are removed. Catches copy
    the renderer clips at the edge that drawn boxes miss (fallback fonts, letter spacing)."""
    import numpy as np

    m = _diff(with_text, without_text)
    if m is None:
        return 0.0
    h, w = m.shape
    sx, sy = max(2, int(EDGE_STRIP * w)), max(2, int(EDGE_STRIP * h))
    border = np.zeros_like(m)
    border[:sy, :] = border[-sy:, :] = True
    border[:, :sx] = border[:, -sx:] = True
    return float((m & border).sum()) / float(border.sum())


def rendered_ok(good_edge: float, source_edge: float) -> Optional[str]:
    """Why the good side's rendered copy is cut at the canvas edge (None when it is not, or
    when the source runs its copy to the edge as much)."""
    if good_edge > EDGE_MAX and good_edge > source_edge + EDGE_MAX:
        return f"rendered copy runs off the edge ({good_edge:.2%} of the border; source {source_edge:.2%})"
    return None


def structural(kind: str, source: dict, good: dict, bad: dict) -> Tuple[Optional[str], Optional[dict]]:
    """(why the defect is not the named one or None, the element it changed or None)."""
    from engine.smart.geom import Box
    from engine.smart.textfit import drawn_box

    W, H = float(good["canvas"]["width"]), float(good["canvas"]["height"])
    gi = {e["id"]: e for e in good["elements"]}
    changed = [e for e in bad["elements"] if e["id"] in gi and e != gi[e["id"]]]
    moved = changed[0] if len(changed) == 1 else None
    if kind == "fit_centre":
        r0 = float(source["canvas"]["width"]) / float(source["canvas"]["height"])
        r1 = W / H
        if max(r0 / r1, r1 / r0) < FIT_MIN_RATIO:
            return f"aspect change {max(r0 / r1, r1 / r0):.2f} < {FIT_MIN_RATIO}", None
        return None, None
    if not changed:
        return "nothing changed", None
    if kind in ("p_crop", "p_clip_letter", "p_onart"):
        h = headline(good)
        if h is None or moved is None or moved["id"] != h["id"]:
            return "the defect is not on the headline (the largest text)", moved
        if float(h.get("rotation", 0) or 0) % 360:
            return "the headline is rotated", moved
        gb = drawn_box(h)
        if gb.area < HEADLINE_MIN * W * H:
            return f"headline is {gb.area / (W * H):.2%} of the canvas < {HEADLINE_MIN:.1%}", moved
        if _on_canvas(gb, W, H) < 0.98:
            return "headline already cut on the good side", moved
        nb = drawn_box(moved)
        if kind == "p_crop" and 1 - _on_canvas(nb, W, H) < CROP_MIN:
            return f"only {1 - _on_canvas(nb, W, H):.0%} of the headline cut", moved
        if kind == "p_onart":
            imgs = [e for e in good["elements"] if e.get("type") == "image" and e["id"] != h["id"]]
            best = 0.0
            for im in imgs:
                ib = Box(im["x"], im["y"], im["width"], im["height"])
                if nb.inter(ib) >= 0.3 * max(1.0, nb.area):
                    best = max(best, gb.inter(ib) / max(1.0, gb.area))
                    break
            if best > OFF_ART_MAX:
                return f"headline already {best:.0%} on the picture", moved
    return None, moved


def _diff(a, b):
    import numpy as np
    from PIL import Image

    with Image.open(a) as ia, Image.open(b) as ib:
        x = np.asarray(ia.convert("RGB"), dtype=np.int16)
        y = np.asarray(ib.convert("RGB"), dtype=np.int16)
    if x.shape != y.shape:
        return None
    return (np.abs(x - y).max(axis=2) > DIFF_LEVEL)


def pixel_ok(good_png: Path, bad_png: Path, canvas: dict, region=None, min_diff: float = MIN_DIFF
             ) -> Tuple[Optional[str], float]:
    """(why the two renders are not visibly different or None, the changed share)."""
    m = _diff(good_png, bad_png)
    if m is None:
        return None, 1.0  # different render sizes: visibly different by construction
    share = float(m.mean())
    if share < min_diff:
        return f"only {share:.2%} of the pixels change < {min_diff:.2%}", share
    if region is not None:
        h, w = m.shape
        sx, sy = w / float(canvas["width"]), h / float(canvas["height"])
        x0, y0 = max(0, int(region.x * sx)), max(0, int(region.y * sy))
        x1, y1 = min(w, int(region.r * sx) + 1), min(h, int(region.b * sy) + 1)
        if x1 > x0 and y1 > y0:
            r = float(m[y0:y1, x0:x1].mean())
            if r < REGION_MIN:
                return f"the changed element shows in {r:.0%} of its box < {REGION_MIN:.0%}", share
    return None, share


def region_of(kind: str, moved: Optional[dict]):
    """The box whose pixels must change for a one-element defect (None: whole canvas only)."""
    from engine.smart.geom import Box
    from engine.smart.textfit import drawn_box

    if moved is None or kind not in ONE_ELEMENT:
        return None
    if moved.get("type") == "text":
        return drawn_box(moved)
    return Box(moved["x"], moved["y"], moved["width"], moved["height"])
