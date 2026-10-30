"""Text measurement with the renderer's own wrap code and fonts (same lookups, same skia)."""
import math
from functools import lru_cache
from pathlib import Path

import skia  # type: ignore

from renderer.converter.brand_fonts import load_brand_font_manager
from renderer.converter.wrap import wrap_text

_FONTS = Path(__file__).resolve().parents[1] / "brand_kit" / "fonts"
_fm = None
_BOLD = {"bold", "semibold", "extrabold", "black"}


def fm():
    global _fm
    if _fm is None:
        _fm = load_brand_font_manager(str(_FONTS))
    return _fm


@lru_cache(maxsize=None)
def _typeface(family, weight, style):
    return skia.Typeface.MakeFromData(fm().lookup(family, weight, style))


def style_of(el):
    w = (el.get("font_weight") or "regular").lower()
    bold = w in _BOLD
    it = (el.get("font_style") or "") == "italic"
    style = "bolditalic" if bold and it else "bold" if bold else "italic" if it else "regular"
    return ("bold" if bold else "regular"), style


@lru_cache(maxsize=200000)
def _widths(family, weight, style, size, text):
    if not text:
        return ()
    font = skia.Font(_typeface(family, weight, style), size)
    return tuple(font.getWidths(font.textToGlyphs(text)))


def layout(el, size, width):
    """Wrap `el`'s content at integer font `size` into `width` like the converter.
    Returns (n_lines, max_line_width)."""
    text = str(el.get("content", ""))
    weight, style = style_of(el)
    fam = el.get("font_family", "Montserrat")
    w = _widths(fam, weight, style, float(size), text)
    wrapped = wrap_text(text, list(w), width)
    ls = float(el.get("letter_spacing", 0) or 0)
    lines = wrapped.split("\n")
    mx, i = 0.0, 0
    for ln in lines:
        lw = sum(w[i:i + len(ln)]) + ls * max(0, len(ln) - 1)
        mx = max(mx, lw)
        i += len(ln) + 1
    return len(lines), mx


def line_pitch(el, size):
    return size * float(el.get("line_height", 1.2) or 1.2)


def fit_text(el, width, fs_hi, max_h=None):
    """Largest int font size <= fs_hi that fits width (no overflowing word) and max_h.
    Returns (fs, n_lines, block_h, ink_w)."""
    fs_hi = max(1, int(math.floor(fs_hi)))

    def ok(fs):
        n, mx = layout(el, fs, width)
        h = n * line_pitch(el, fs)
        return mx <= width + 0.5 and (max_h is None or h <= max_h + 0.5), n, h, mx

    lo, hi = 1, fs_hi
    if ok(hi)[0]:
        lo = hi
    else:
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ok(mid)[0]:
                lo = mid
            else:
                hi = mid - 1
    _, n, h, mx = ok(lo)
    return lo, n, h, mx
