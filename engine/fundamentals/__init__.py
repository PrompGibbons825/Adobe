"""Fundamentals: an engine-independent model of the official renderer, plus the hard-rule checker.

Imports nothing from ``engine/``. Engines may vendor this folder into ``engine/`` (the Dockerfile
copies only ``engine/``); ``render`` and the pixel half of ``check`` additionally need the repo's
``renderer/`` package, which the image has at /app/renderer. Read QUIRKS.md first.
"""

from .fonts import FontBook, fontbook
from .image import Fit, fit_geometry, image_fit
from .text import TextLayout, measure_text, to_canvas, wrap_text

__all__ = ["FontBook", "fontbook", "Fit", "fit_geometry", "image_fit", "TextLayout", "measure_text",
           "to_canvas", "wrap_text", "check", "Fault"]


def __getattr__(name):  # lazy: check pulls numpy / the renderer only when used
    if name in ("check", "Fault"):
        from . import check as _c
        return getattr(_c, name)
    raise AttributeError(name)
