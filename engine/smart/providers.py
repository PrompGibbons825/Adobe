"""Provider interfaces: where the engine gets font metrics and asset pixels.

Stage one reads the brand kit from disk and measures text with skia, exactly
like the renderer. An Adobe Express add-on cannot read font files and gets
image pixels as blobs, so both are pluggable:

  TextMeasurer      per-character advances + (ascent, descent, cap height)
  CorrectedMeasurer the default measurer scaled per font by factors the host
                    measured (`constraints.measured_text`, the settle loop)
  asset source      asset_id -> PIL.Image (or None when unknown)

Swapping a provider clears the measurement caches. Everything here is
stdlib + Pillow, so it runs under Pyodide.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Optional, Tuple

from . import assets, textfit

FontKey = Tuple[str, str, str]  # family, normalized weight, style key


class TextMeasurer:
    """The default: brand fonts through skia (a conservative estimate without them)."""

    def char_widths(self, family: str, weight: str, style: str, size: float, text: str) -> Tuple[float, ...]:
        return textfit.native_char_widths(family, weight, style, size, text)

    def metrics(self, family: str, weight: str, style: str) -> Tuple[float, float, float]:
        return textfit.book().metrics(family, weight, style)


class CorrectedMeasurer(TextMeasurer):
    """Another measurer with per-font width and height factors."""

    def __init__(self, factors: Dict[FontKey, Tuple[float, float]], base: Optional[TextMeasurer] = None):
        self.factors = dict(factors)
        self.base = base or TextMeasurer()

    def char_widths(self, family, weight, style, size, text):
        kw = self.factors.get((family, weight, style), (1.0, 1.0))[0]
        return tuple(w * kw for w in self.base.char_widths(family, weight, style, size, text))

    def metrics(self, family, weight, style):
        kh = self.factors.get((family, weight, style), (1.0, 1.0))[1]
        a, d, c = self.base.metrics(family, weight, style)
        return a * kh, d * kh, c * kh


def set_text_measurer(m: Optional[TextMeasurer]) -> None:
    textfit._MEASURER = m
    textfit.clear_caches()


def set_asset_source(src: Optional[Callable[[str], object]]) -> None:
    assets._SOURCE = src
    assets.info.cache_clear()


@contextmanager
def text_measurer(m: Optional[TextMeasurer]):
    """Use `m` inside the block (None leaves the current measurer alone)."""
    if m is None:
        yield None
        return
    prev = textfit._MEASURER
    set_text_measurer(m)
    try:
        yield m
    finally:
        set_text_measurer(prev)
