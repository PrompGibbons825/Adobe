"""Pixel truth: draw through the official renderer (``renderer/`` + cr-renderer).

Only this module touches ``renderer``; everything else in ``fundamentals`` is a model of it.
Needs the repo (or /app) root on ``sys.path``, which ``python -m`` from the repo root gives.

``render(design)`` is exactly ``render_design_to_png`` (PNG round trip included).
``Scene(design)`` converts once and then draws any subset of elements, at any scale, padded or
transparent, with cr-renderer's own ``_render_to_surface`` (no PNG encode, converted bitmaps
reused): the cheap way to get per-element footprints and with/without renders.
"""

from __future__ import annotations

import contextlib
import copy
import io
import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from .fonts import find_fonts_dir

MAX_SIZE = 2048  # render_design_to_png's default


def assets_dir(hint=None) -> Path:
    if hint:
        return Path(hint)
    return find_fonts_dir().parent / "assets"


def _quiet():
    for name in ("cr_renderer", "cr_renderer.fonts", "cr_renderer.text_utils", "cr_renderer.renderer", "renderer"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _renderer(fonts=None):
    from renderer.converter.convert import _get_renderer

    _quiet()
    return _get_renderer(str(fonts) if fonts else str(find_fonts_dir()))


def render(design: dict, assets=None, fonts=None, transparent: bool = False,
           max_size: Optional[int] = MAX_SIZE) -> np.ndarray:
    """RGBA uint8 array of ``design`` as the scorer renders it (white canvas, max_size 2048)."""
    import skia  # type: ignore
    from PIL import Image

    from renderer.converter.convert import convert_chameleon_to_crello

    r = _renderer(fonts)
    example, _ = convert_chameleon_to_crello(design, r.font_manager, str(assets_dir(assets)))
    png = r.render(example, max_size=max_size, format="png",
                   canvas_color=None if transparent else skia.ColorWHITE)
    with Image.open(io.BytesIO(png)) as im:
        return np.asarray(im.convert("RGBA"))


_LIST_KEYS = ("left", "top", "width", "height", "angle", "type", "opacity", "text", "font_size", "font",
              "line_height", "text_align", "capitalize", "letter_spacing", "font_bold", "font_italic",
              "text_color", "text_line", "image")


class Scene:
    """One converted design; draw subsets of it through cr-renderer's own drawing code."""

    def __init__(self, design: dict, assets=None, fonts=None):
        from renderer.converter.convert import convert_chameleon_to_crello

        self.r = _renderer(fonts)
        self.W = float(design["canvas"]["width"])
        self.H = float(design["canvas"]["height"])
        ex, self.warnings = convert_chameleon_to_crello(design, self.r.font_manager, str(assets_dir(assets)))
        self.example = ex
        # converter order = elements sorted by z_index (stable); map ids to rows
        order = sorted(design["elements"], key=lambda e: e.get("z_index", 0))
        self.rows = {str(e.get("id")): i for i, e in enumerate(order)}
        self._sk = {}

    def draw(self, ids: Optional[Iterable[str]] = None, scale: float = 1.0, pad: int = 0,
             transparent: bool = False) -> np.ndarray:
        """RGBA of the elements ``ids`` (None = all), canvas grown by ``pad`` px per side, drawn at
        ``scale``. Pixel (x, y) of the canvas is at ``[(y + pad) * scale, (x + pad) * scale]``."""
        import skia  # type: ignore

        import cr_renderer.image_utils as iu
        import cr_renderer.renderer as R

        keep = sorted(self.rows.values()) if ids is None else sorted(self.rows[i] for i in ids if i in self.rows)
        ex = {k: [self.example[k][i] for i in keep] for k in _LIST_KEYS}
        ex["left"] = [v + pad for v in ex["left"]]
        ex["top"] = [v + pad for v in ex["top"]]
        ex["color"] = []
        ex["length"] = len(keep)
        ex["canvas_width"] = self.W + 2 * pad
        ex["canvas_height"] = self.H + 2 * pad
        cw, ch = ex["canvas_width"], ex["canvas_height"]
        max_size = None if scale >= 1.0 else int(round(max(cw, ch) * scale))
        with self._patched(iu):
            snap = R._render_to_surface(self.r.font_manager, ex, max_size, True, "png",
                                        None if transparent else skia.ColorWHITE)
        arr = snap.toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType)
        return arr

    @contextlib.contextmanager
    def _patched(self, iu):
        """Skip the PNG encode of the surface and cache PIL->skia bitmap conversion. Drawing is
        untouched: the same skia calls on the same bitmaps."""
        enc, conv = iu.encode_surface, iu.convert_pil_image_to_skia_image
        cache = self._sk

        def convert(img, format="PNG"):
            k = id(img)
            if k not in cache:
                cache[k] = (img, conv(img, format))
            return cache[k][1]

        iu.encode_surface = lambda surface, format: surface.makeImageSnapshot()
        iu.convert_pil_image_to_skia_image = convert
        try:
            yield
        finally:
            iu.encode_surface, iu.convert_pil_image_to_skia_image = enc, conv


def footprint(e: dict, W: float, H: float, pad: int = 0, assets=None, fonts=None) -> Tuple[np.ndarray, int]:
    """Alpha (uint8) of ``e`` drawn alone at scale 1 on a transparent canvas grown by ``pad``."""
    sc = Scene({"canvas": {"width": W, "height": H}, "elements": [e]}, assets, fonts)
    return sc.draw(pad=pad, transparent=True)[..., 3], pad


def mask_box(mask: np.ndarray, thr: int = 0, offset: int = 0, scale: float = 1.0) -> Optional[Tuple[float, float, float, float]]:
    ys, xs = np.nonzero(mask > thr)
    if len(xs) == 0:
        return None
    return (xs.min() / scale - offset, ys.min() / scale - offset, (xs.max() + 1) / scale - offset, (ys.max() + 1) / scale - offset)
