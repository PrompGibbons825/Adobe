"""What is actually under the copy.

"Text never sits on a busy image" and "enough contrast" are rubric rows, and
neither can be judged from boxes. Here a plan's non-text layers (background,
slabs, panels, photos) are rendered small through the real renderer, and each
text box is measured against those pixels the way the proxy judge does it
(tools/proxy.py): WCAG contrast at the 10th percentile of the region, and the
90th-percentile luminance gradient as busy-ness. The same measurement on the
source design is the reference, so a design that was low-contrast to begin with
is not punished for staying that way.

Everything here is optional: without numpy or the renderer it reports nothing.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from . import paths

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

SIDE = 384  # render size: plenty to judge contrast and texture under a text box


def _renderer():
    try:
        from renderer.converter.convert import _get_renderer, convert_chameleon_to_crello  # type: ignore

        fonts = paths.fonts_dir()
        if fonts is None:
            return None
        return _get_renderer(str(fonts)), convert_chameleon_to_crello
    except Exception:
        return None


_R = None


def available() -> bool:
    global _R
    if np is None:
        return False
    if _R is None:
        _R = _renderer() or False
    return bool(_R)


def _small(elements: List[dict], W: float, H: float) -> Tuple[List[dict], float, int, int]:
    f = min(1.0, SIDE / max(W, H))
    out = []
    for e in elements:
        if e.get("type") == "text":
            continue
        e = dict(e)
        for k in ("x", "y", "width", "height"):
            e[k] = float(e.get(k, 0) or 0) * f
        if e.get("corner_radius"):
            e["corner_radius"] = float(e["corner_radius"]) * f
        out.append(e)
    return out, f, max(1, round(W * f)), max(1, round(H * f))


def backdrop(elements: List[dict], W: float, H: float):
    """RGB float array of everything except text, and the scale used."""
    if not available():
        return None, 1.0
    R, convert = _R
    els, f, w, h = _small(elements, W, H)
    d = {"canvas": {"width": w, "height": h}, "elements": els}
    try:
        assets_dir = str(paths.assets_dirs()[0]) if paths.assets_dirs() else None
        ex, _ = convert(d, R.font_manager, assets_dir)
        png = R.render(ex, max_size=max(w, h), format="png")
        from PIL import Image

        arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32) / 255.0
        return arr, f
    except Exception:
        return None, 1.0


def _lum(rgb):
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def text_stats(arr, f: float, e: dict) -> Optional[Tuple[float, float]]:
    """(contrast p10, busy p90 gradient) of one text element against the backdrop."""
    from .color import parse_rgba

    h, w = arr.shape[:2]
    x0, y0 = int(max(0, e["x"] * f)), int(max(0, e["y"] * f))
    x1, y1 = int(min(w, (e["x"] + e["width"]) * f + 1)), int(min(h, (e["y"] + e["height"]) * f + 1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    bg = arr[y0:y1, x0:x1].reshape(-1, 3)
    r, g, b, a = parse_rgba(e.get("color"), default=(0, 0, 0, 1.0))
    t = np.array([r, g, b], dtype=np.float32) / 255.0
    eff = a * t + (1 - a) * bg
    lt, lb = _lum(eff), _lum(bg)
    hi, lo = np.maximum(lt, lb), np.minimum(lt, lb)
    c = (hi + 0.05) / (lo + 0.05)
    gray = _lum(arr[max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1])
    # judge texture at the scale of the glyphs (cap height -> 10 px), so the
    # source and a resized canvas are measured alike
    cap = max(1.0, 0.7 * float(e.get("font_size", 12) or 12) * f)
    k = 10.0 / cap
    if abs(k - 1.0) > 0.05 and gray.size > 4:
        from PIL import Image

        gh, gw = gray.shape
        im = Image.fromarray((gray * 255).astype("uint8")).resize(
            (max(3, round(gw * k)), max(3, round(gh * k))), Image.BILINEAR)
        gray = np.asarray(im, dtype=np.float32) / 255.0
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    busy = float(np.percentile(np.hypot(gx, gy) * 0.5, 90))
    return float(np.percentile(c, 10)), busy


SPLIT_DIST = 0.12  # RGB distance (0..1 units) from a box's median colour past which a pixel is other ground


def split_share(arr, f: float, e: dict) -> float:
    """Share of a text box's backdrop pixels that are not its main ground (farther than
    SPLIT_DIST from the box's median colour): a box half on a plain wall and half on a smooth
    armchair and its cushion straddles two grounds. Neither contrast nor texture sees that
    collision when both grounds are calm and light; this does. Measured under the glyphs'
    own box (textfit.drawn_box), not the element's, which may run past the line."""
    from .textfit import drawn_box

    try:
        g = drawn_box(e)
        bx, by, bw, bh = g.x, g.y, g.w, g.h
    except Exception:  # noqa: BLE001
        bx, by, bw, bh = e["x"], e["y"], e["width"], e["height"]
    h, w = arr.shape[:2]
    x0, y0 = int(max(0, bx * f)), int(max(0, by * f))
    x1, y1 = int(min(w, (bx + bw) * f + 1)), int(min(h, (by + bh) * f + 1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return 0.0
    bg = arr[y0:y1, x0:x1].reshape(-1, 3)
    med = np.median(bg, axis=0)
    return float((np.sqrt(((bg - med) ** 2).sum(-1)) > SPLIT_DIST).mean())


def measure(elements: List[dict], W: float, H: float, skip: set,
            split: bool = False) -> Dict[str, Tuple[float, ...]]:
    """{text id: (contrast p10, busy p90)} against the backdrop; with ``split``, also the
    split_share as a third value."""
    arr, f = backdrop(elements, W, H)
    if arr is None:
        return {}
    out = {}
    for e in elements:
        if e.get("type") != "text" or e["id"] in skip or float(e.get("opacity", 1) or 0) <= 0:
            continue
        st = text_stats(arr, f, e)
        if st:
            out[e["id"]] = st + (split_share(arr, f, e),) if split else st
    return out
