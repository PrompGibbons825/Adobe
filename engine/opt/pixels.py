"""Read (never alter) asset pixels: edge density and luminance under a canvas region.
Owner: pixel evidence for the 'no text on busy pixels' and contrast terms."""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]


def _assets_dirs():
    d = []
    kit = os.environ.get("CHAMELEON_BRAND_KIT")
    if kit:
        d.append(Path(kit) / "assets")
    d += [ROOT / "brand_kit" / "assets", Path("/app/brand_kit/assets")]
    return d


@functools.lru_cache(maxsize=256)
def _load(asset_id: str):
    try:
        from PIL import Image, ImageFilter
    except Exception:  # noqa: BLE001
        return None
    for d in _assets_dirs():
        p = d / asset_id
        if p.is_file():
            try:
                im = Image.open(p).convert("RGBA")
            except Exception:  # noqa: BLE001
                return None
            iw, ih = im.size
            im.thumbnail((160, 160))
            g = im.convert("L")
            edges = g.filter(ImageFilter.FIND_EDGES)
            alpha = im.getchannel("A")
            return iw, ih, im.size, g, edges, alpha
    return None


def region_stats(el: dict, box: Tuple[float, float, float, float], rx0, ry0, rx1, ry1) -> Optional[Tuple[float, float, float]]:
    """(edge density 0..1, mean luminance 0..1, coverage 0..1) of image el drawn in box over the
    canvas region [rx0,rx1]x[ry0,ry1]. None if the asset is unavailable or the region misses it."""
    aid = el.get("asset_id")
    if not aid:
        return None
    L = _load(str(aid))
    if L is None:
        return None
    from PIL import ImageStat
    iw, ih, (tw, th), g, edges, alpha = L
    ex, ey, ew, eh = box
    fit = el.get("fit", "contain")
    if fit == "cover":
        sc = max(ew / iw, eh / ih)
        ox, oy = ex - (iw * sc - ew) / 2, ey - (ih * sc - eh) / 2
        sx = sy = sc
    elif fit == "contain":
        sc = min(ew / iw, eh / ih)
        ox, oy = ex + (ew - iw * sc) / 2, ey + (eh - ih * sc) / 2
        sx = sy = sc
    else:
        sx, sy = ew / iw, eh / ih
        ox, oy = ex, ey
    # clip region to the element box
    rx0, ry0 = max(rx0, ex), max(ry0, ey)
    rx1, ry1 = min(rx1, ex + ew), min(ry1, ey + eh)
    if rx1 - rx0 < 1 or ry1 - ry0 < 1:
        return None
    k = tw / iw
    u0 = int(max(0, (rx0 - ox) / sx * k))
    v0 = int(max(0, (ry0 - oy) / sy * k))
    u1 = int(min(tw, (rx1 - ox) / sx * k + 1))
    v1 = int(min(th, (ry1 - oy) / sy * k + 1))
    if u1 - u0 < 1 or v1 - v0 < 1:
        return None
    crop = (u0, v0, u1, v1)
    a = ImageStat.Stat(alpha.crop(crop)).mean[0] / 255.0
    if a < 0.2:
        return None
    e = ImageStat.Stat(edges.crop(crop)).mean[0] / 255.0
    lum = ImageStat.Stat(g.crop(crop)).mean[0] / 255.0
    return e, lum, a
