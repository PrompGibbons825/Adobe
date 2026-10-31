"""Read-only analysis of image assets.

Nothing here modifies pixels. We only look at an asset to learn where its
visible (non-transparent) content sits inside the element box, so layout can
space things by what the viewer actually sees rather than by padded boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

from . import paths

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

Frac = Tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of the element box


@dataclass(frozen=True)
class AssetInfo:
    found: bool
    size: Tuple[int, int] = (0, 0)
    alpha_bbox: Optional[Tuple[int, int, int, int]] = None  # in asset pixels, (x0, y0, x1, y1) exclusive
    opaque: bool = True  # no meaningful transparency
    luminance: float = 0.5  # mean luminance 0..1 of visible pixels
    busy: float = 0.0  # 0..1 edge density of visible pixels
    coverage: float = 1.0  # share of the image that is painted (frames and outlines are low)
    ring: float = 0.0  # share of the painted pixels lying in the outer band of the visible bounds


@lru_cache(maxsize=256)
def info(asset_id: Optional[str]) -> AssetInfo:
    path = paths.find_asset(asset_id)
    if path is None or Image is None:
        return AssetInfo(found=False)
    try:
        im = Image.open(path)
        im.load()
    except Exception:
        return AssetInfo(found=False)
    w, h = im.size
    rgba = im.convert("RGBA")
    alpha = rgba.getchannel("A")
    mask = alpha.point(lambda a: 255 if a > 16 else 0)
    bbox = mask.getbbox() or (0, 0, w, h)
    opaque = alpha.getextrema()[0] > 240
    lum, busy = _stats(rgba, mask)
    small = mask.crop(bbox)  # inside the visible bounds: frames stay hollow, cut-out products are solid
    small.thumbnail((128, 128))
    hist = small.histogram()
    coverage = sum(hist[128:]) / max(1, sum(hist))
    sw, sh = small.size
    bx, by = max(1, round(0.15 * sw)), max(1, round(0.15 * sh))
    inner = small.crop((bx, by, max(bx + 1, sw - bx), max(by + 1, sh - by))).histogram()
    painted = sum(hist[128:])
    ring = (painted - sum(inner[128:])) / max(1, painted)
    return AssetInfo(found=True, size=(w, h), alpha_bbox=bbox, opaque=opaque, luminance=lum, busy=busy,
                     coverage=coverage, ring=ring)


def _stats(rgba, mask) -> Tuple[float, float]:
    try:
        from PIL import ImageFilter, ImageStat

        small = rgba.copy()
        small.thumbnail((128, 128))
        m = mask.copy().resize(small.size)
        gray = small.convert("L")
        lum = ImageStat.Stat(gray, m).mean[0] / 255.0
        edges = gray.filter(ImageFilter.FIND_EDGES)
        busy = min(1.0, ImageStat.Stat(edges, m).mean[0] / 40.0)
        return lum, busy
    except Exception:
        return 0.5, 0.0


def ink_fraction(asset_id: Optional[str], box_w: float, box_h: float, fit: str) -> Frac:
    """Where the visible pixels land inside an element box of this size."""
    a = info(asset_id)
    if not a.found or a.alpha_bbox is None or box_w <= 0 or box_h <= 0:
        return (0.0, 0.0, 1.0, 1.0)
    iw, ih = a.size
    x0, y0, x1, y1 = a.alpha_bbox
    if fit == "cover":
        s = max(box_w / iw, box_h / ih)
        ox, oy = (iw * s - box_w) / 2.0, (ih * s - box_h) / 2.0
        fx = lambda v: (v * s - ox) / box_w  # noqa: E731
        fy = lambda v: (v * s - oy) / box_h  # noqa: E731
    elif fit == "fill":
        fx = lambda v: v / iw  # noqa: E731
        fy = lambda v: v / ih  # noqa: E731
    else:  # contain
        s = min(box_w / iw, box_h / ih)
        ox, oy = (box_w - iw * s) / 2.0, (box_h - ih * s) / 2.0
        fx = lambda v: (v * s + ox) / box_w  # noqa: E731
        fy = lambda v: (v * s + oy) / box_h  # noqa: E731
    clamp = lambda v: max(0.0, min(1.0, v))  # noqa: E731
    f = (clamp(fx(x0)), clamp(fy(y0)), clamp(fx(x1)), clamp(fy(y1)))
    if f[2] - f[0] < 0.02 or f[3] - f[1] < 0.02:
        return (0.0, 0.0, 1.0, 1.0)
    return f
