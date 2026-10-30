"""Image fit / crop geometry as the converter draws it (renderer/converter/assets.py).

The converter pre-fits the asset into a ``round(w) x round(h)`` bitmap (LANCZOS), and skia then
stretches that bitmap into the float ``w x h`` box. ``fit_geometry`` returns where the asset's
pixels land in element-local coordinates and which part of the asset is shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

Box = Tuple[float, float, float, float]


@dataclass
class Fit:
    fit: str
    content: Box  # element-local box the asset's pixels cover (before rotation)
    crop: Box  # the part of the asset shown, in asset pixels (x0, y0, x1, y1)
    scale_x: float  # drawn px per asset px
    scale_y: float
    distortion: float  # |aspect drawn / aspect asset - 1|; 0 except for fill
    missing: bool = False


def fit_geometry(width: float, height: float, asset_w: int, asset_h: int, fit: str = "contain") -> Fit:
    w, h = max(1, round(width)), max(1, round(height))
    sx, sy = float(width) / w, float(height) / h  # bitmap -> box stretch done by skia
    iw, ih = max(1, int(asset_w)), max(1, int(asset_h))
    if fit == "contain":
        s = min(w / iw, h / ih)
        nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
        ox, oy = (w - nw) // 2, (h - nh) // 2
        content = (ox * sx, oy * sy, (ox + nw) * sx, (oy + nh) * sy)
        dist = abs((nw * sx / (nh * sy)) / (iw / ih) - 1)
        return Fit("contain", content, (0.0, 0.0, float(iw), float(ih)), nw * sx / iw, nh * sy / ih, dist)
    if fit == "cover":
        s = max(w / iw, h / ih)
        nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
        left, top = (nw - w) // 2, (nh - h) // 2
        crop = (left / s, top / s, (left + w) / s, (top + h) / s)
        dist = abs((nw * sx / (nh * sy)) / (iw / ih) - 1)
        return Fit("cover", (0.0, 0.0, float(width), float(height)), crop, nw * sx / iw, nh * sy / ih, dist)
    dist = abs((float(width) / max(1e-9, float(height))) / (iw / ih) - 1)
    return Fit(fit or "fill", (0.0, 0.0, float(width), float(height)), (0.0, 0.0, float(iw), float(ih)),
               float(width) / iw, float(height) / ih, dist)


def asset_size(assets_dir, asset_id) -> Optional[Tuple[int, int]]:
    if not assets_dir or not asset_id:
        return None
    p = Path(assets_dir) / str(asset_id)
    if not p.is_file():
        return None
    from PIL import Image

    with Image.open(p) as im:
        return im.size


def image_fit(e: dict, assets_dir) -> Fit:
    """Fit for an image element; a missing asset draws a placeholder over the whole box."""
    size = asset_size(assets_dir, e.get("asset_id"))
    w, h = float(e.get("width", 1.0)), float(e.get("height", 1.0))
    if size is None:
        return Fit("placeholder", (0.0, 0.0, w, h), (0.0, 0.0, 0.0, 0.0), 1.0, 1.0, 0.0, True)
    return fit_geometry(w, h, size[0], size[1], e.get("fit", "contain"))
