"""Bleed policy (loop/STEERING.md S4): which elements may extend past the canvas, and how far.

    from tools.bleed import check
    errors, bled = check(source, output, target_canvas)

The renderer clips at the canvas, so a box larger than the canvas renders as a clean crop. A
designer bleeds photos and edge ornaments on purpose; copy, logos and buttons never.

Roles are inferred from the **source** design only (the input has no labels), from type and
geometry, plus whether an image asset has transparent pixels:

* ``text``   any text. Never bleeds.
* ``info``   carries information: a shape holding a text of its group, or one filling at least
  ``LABEL_FILL`` of it (button, badge with copy), that is not a large panel; an image with transparent pixels (logo, product cutout) unless it covered the
  source canvas; a small opaque image (icon, thumbnail). Bleeds only if the source bled it.
* ``photo``  any other image (backgrounds, photos, image panels). May bleed: a box that covers
  the canvas is a cover crop and needs nothing more; otherwise it is a panel against an edge.
* ``decor``  any other shape (circles, blobs, confetti, rules, colour panels). May bleed.

Every allowed bleed except a cover crop keeps at least ``KEEP`` (60%) of its box on the canvas,
or its source share if that was less (minus ``SLACK``). Anything outside these rules is an error.
Aspect ratios are not checked here (``tools/rules.py`` does that).

Vendored copy of tools/bleed.py (same policy); imports nothing from ``engine/`` or ``tools/``.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TOL = 0.6  # px of slack for float rounding at the canvas edge
KEEP = 0.6  # share of an allowed bleed's box that stays on the canvas
SLACK = 0.05  # share a bleed may lose against its source share ("about the same proportion")
SMALL_IMAGE = 0.10  # an opaque image below this share of the source canvas is an icon / thumbnail
BIG_PANEL = 0.35  # a shape holding copy and at least this share of the canvas is a panel, not a button
LABEL_FILL = 0.12  # a text filling this share of a shape it sits in makes the shape its button / badge
TRANSPARENT = 0.05  # share of an asset's pixels with alpha < 250 that makes it a cutout / logo
COVERING = 0.9  # a transparent image covering this much of the source canvas is a background layer

ROLES = ("text", "info", "photo", "decor")


def _num(v, d=0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return d
    return f if math.isfinite(f) else d


def aabb(e: dict) -> Tuple[float, float, float, float]:
    """x0, y0, x1, y1 of the element's box, grown to its rotated bounding box."""
    x, y, w, h = _num(e.get("x")), _num(e.get("y")), max(0.0, _num(e.get("width"))), max(0.0, _num(e.get("height")))
    x0, y0, x1, y1 = x, y, x + w, y + h
    rot = _num(e.get("rotation"))
    if rot:
        t = math.radians(rot)
        c, s = abs(math.cos(t)), abs(math.sin(t))
        rw, rh = w * c + h * s, w * s + h * c
        cx, cy = x + w / 2.0, y + h / 2.0
        x0, y0, x1, y1 = min(x0, cx - rw / 2), min(y0, cy - rh / 2), max(x1, cx + rw / 2), max(y1, cy + rh / 2)
    return x0, y0, x1, y1


def share_on_canvas(e: dict, W: float, H: float) -> float:
    """Share of the element's (rotated) box that lies on the canvas; 1 for an empty box inside."""
    x0, y0, x1, y1 = aabb(e)
    area = (x1 - x0) * (y1 - y0)
    if area <= 0:
        return 1.0 if (x0 >= -TOL and y0 >= -TOL and x1 <= W + TOL and y1 <= H + TOL) else 0.0
    iw = max(0.0, min(x1, W) - max(x0, 0.0))
    ih = max(0.0, min(y1, H) - max(y0, 0.0))
    return iw * ih / area


def bleeds(e: dict, W: float, H: float) -> bool:
    x0, y0, x1, y1 = aabb(e)
    return x0 < -TOL or y0 < -TOL or x1 > W + TOL or y1 > H + TOL


def covers(e: dict, W: float, H: float) -> bool:
    """The box covers the whole canvas (a cover crop: the canvas edge does the cropping)."""
    x0, y0, x1, y1 = aabb(e)
    return x0 <= TOL and y0 <= TOL and x1 >= W - TOL and y1 >= H - TOL


def _assets_dir(assets) -> Path:
    if assets:
        return Path(assets)
    kit = os.environ.get("CHAMELEON_BRAND_KIT")
    if kit:
        return Path(kit) / "assets"
    from .fonts import find_fonts_dir

    return find_fonts_dir().parent / "assets"


@lru_cache(maxsize=512)
def _transparent(path: str) -> Optional[bool]:
    """Whether the asset has a meaningful share of transparent pixels; None if unreadable."""
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as im:
            if im.mode not in ("RGBA", "LA", "PA") and "transparency" not in im.info:
                return False
            im.thumbnail((256, 256))
            a = np.asarray(im.convert("RGBA"))[..., 3]
        return bool((a < 250).mean() >= TRANSPARENT)
    except Exception:  # noqa: BLE001  missing or broken asset: decide by geometry
        return None


def _inside(a, b, tol: float = 2.0) -> bool:
    return a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol


def roles(design: dict, assets=None) -> Dict[str, str]:
    """id -> role (``ROLES``), from this design's own geometry."""
    W, H = _num(design["canvas"]["width"], 1.0), _num(design["canvas"]["height"], 1.0)
    area_c = max(1e-9, W * H)
    els = design.get("elements", [])
    texts = [(aabb(e), e.get("group_id")) for e in els if e.get("type") == "text" and str(e.get("content", "")).strip()]
    adir = _assets_dir(assets)
    out = {}
    for e in els:
        i, t = str(e.get("id")), e.get("type")
        b = aabb(e)
        share = (b[2] - b[0]) * (b[3] - b[1]) / area_c
        if t == "text":
            out[i] = "text"
        elif t == "image":
            transp = _transparent(str(adir / str(e.get("asset_id", "")))) if e.get("asset_id") else None
            if transp is True:  # cutout or logo, unless it is a background layer
                out[i] = "photo" if share >= COVERING else "info"
            else:
                out[i] = "photo" if share >= SMALL_IMAGE else "info"
        else:
            barea = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
            holds_copy = any(_inside(tb, b) and ((g is not None and g == e.get("group_id"))
                                                 or (tb[2] - tb[0]) * (tb[3] - tb[1]) >= LABEL_FILL * barea)
                             for tb, g in texts)
            out[i] = "info" if holds_copy and share < BIG_PANEL else "decor"
    return out


def verdict(role: str, src_share: float, out_share: float, out_covers: bool, src_bled: bool) -> Optional[str]:
    """None if this bleed is allowed, else why not. Only call it for an element that bleeds."""
    floor = min(KEEP, src_share) - SLACK
    if role == "text":
        return "text past the canvas"
    if role == "info":
        if not src_bled:
            return "never-bleed element (logo, button, badge, cutout) past the canvas"
        if out_share < floor:
            return f"bled further than the source ({out_share:.0%} on canvas, source {src_share:.0%})"
        return None
    if role == "photo" and out_covers:
        return None
    if out_share < floor:
        return f"{role} bleed keeps {out_share:.0%} on canvas (< {min(KEEP, src_share):.0%})"
    return None


def scan(src: dict, out: dict, tgt: dict, assets=None) -> List[Tuple[str, bool, str]]:
    """(id, allowed, message) for every output element that extends past the canvas."""
    W, H = _num(tgt["width"], 1.0), _num(tgt["height"], 1.0)
    SW, SH = _num(src["canvas"]["width"], 1.0), _num(src["canvas"]["height"], 1.0)
    role = roles(src, assets)
    so = {str(e.get("id")): e for e in src.get("elements", [])}
    res = []
    for e in out.get("elements", []):
        i = str(e.get("id"))
        if not bleeds(e, W, H):
            continue
        s = so.get(i)
        r = role.get(i, "text" if e.get("type") == "text" else "info")
        src_share = share_on_canvas(s, SW, SH) if s is not None else 1.0
        on = share_on_canvas(e, W, H)
        why = verdict(r, src_share, on, e.get("type") == "image" and covers(e, W, H),
                      s is not None and bleeds(s, SW, SH))
        x0, y0, x1, y1 = aabb(e)
        where = f"({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})"
        res.append((i, why is None, f"{i}: off-canvas {where}: {why}" if why
                    else f"{i}: {r} bleeds {where}, {on:.0%} on canvas"))
    return res


def check(src: dict, out: dict, tgt: dict, assets=None) -> Tuple[List[str], List[str]]:
    """(errors, bled): errors for forbidden or excessive bleed; one report line per allowed bleed."""
    res = scan(src, out, tgt, assets)
    return [m for _, ok, m in res if not ok], [m for _, ok, m in res if ok]


def allowed_ids(src: dict, out: dict, tgt: dict, assets=None) -> set:
    """Ids whose off-canvas part is allowed bleed (for scorers that must not penalise it)."""
    return {i for i, ok, _ in scan(src, out, tgt, assets) if ok}
