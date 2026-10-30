"""Measure text exactly the way the renderer draws it.

The renderer wraps with `renderer.converter.wrap` (greedy, spaces only, advances measured at the
float size, letter spacing ignored), then draws each line at int(font_size) with letter spacing,
line i at y = i * int(fs) * line_height, baseline at y - ascent. So a text fits its box when
every drawn line is no wider than the box and the last line's descent is above the box bottom.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
_fm = None


def _font_manager():
    global _fm
    if _fm is None:
        try:
            from renderer.converter.brand_fonts import load_brand_font_manager
            _fm = load_brand_font_manager(str(ROOT / "brand_kit" / "fonts"))
        except Exception:  # noqa: BLE001
            _fm = False
    return _fm


def _weight_style(e: Dict[str, Any]) -> Tuple[str, str]:
    w = str(e.get("font_weight") or "regular").lower()
    bold = w in {"bold", "semibold", "extrabold", "black"}
    it = e.get("font_style") == "italic" or e.get("italic") is True
    try:
        from cr_renderer.fonts import FONT_WEIGHTS
        if w not in FONT_WEIGHTS:
            w = "regular"
    except Exception:  # noqa: BLE001
        w = "regular"
    style = "bolditalic" if bold and it else "bold" if bold else "italic" if it else "regular"
    return w, style


@lru_cache(maxsize=4096)
def _typeface(family: str, weight: str, style: str):
    import skia
    fm = _font_manager()
    return skia.Typeface.MakeFromData(fm.lookup(family, weight, style))


@lru_cache(maxsize=20000)
def _widths(family: str, weight: str, style: str, size: float, text: str):
    import skia
    font = skia.Font(_typeface(family, weight, style), size)
    return tuple(font.getWidths(font.textToGlyphs(text)))


@lru_cache(maxsize=4096)
def _metrics(family: str, weight: str, style: str, size: int):
    import skia
    m = skia.Font(_typeface(family, weight, style), size).getMetrics()
    return -m.fAscent, m.fDescent


def _fallback_widths(size: float, text: str):
    return tuple(0.55 * size for _ in text)


def layout(e: Dict[str, Any], fs: float, width: float) -> Tuple[float, float, int]:
    """(max drawn line width, block height, n lines) for element e at font size fs in a box of
    `width`, exactly as the renderer will wrap and draw it."""
    from renderer.converter.wrap import wrap_text
    text = str(e.get("content", ""))
    fam = str(e.get("font_family", "Montserrat"))
    weight, style = _weight_style(e)
    ls = float(e.get("letter_spacing") or 0.0)
    lh = float(e.get("line_height") or 1.2)
    fs_i = max(1, int(fs))
    try:
        if _font_manager() is False:
            raise RuntimeError
        widths = _widths(fam, weight, style, float(fs), text)
        draw_w = _widths(fam, weight, style, float(fs_i), text)
        asc, desc = _metrics(fam, weight, style, fs_i)
    except Exception:  # noqa: BLE001
        widths = _fallback_widths(fs, text)
        draw_w = _fallback_widths(fs_i, text)
        asc, desc = 0.93 * fs_i, 0.25 * fs_i
    if e.get("capitalize"):
        try:
            draw_w = _widths(fam, weight, style, float(fs_i), text.upper())
        except Exception:  # noqa: BLE001
            pass
    wrapped = wrap_text(text, list(widths), width)
    lines = wrapped.split("\n")
    pos, maxw = 0, 0.0
    for ln in lines:
        seg = draw_w[pos:pos + len(ln)]
        lw = sum(seg) + ls * max(0, len(ln.strip()) - 1)
        maxw = max(maxw, lw)
        pos += len(ln) + 1
    n = len(lines)
    height = (n - 1) * fs_i * lh + asc + desc
    return maxw, height, n


def fits(e: Dict[str, Any], fs: float, w: float, h: float) -> bool:
    mw, bh, _ = layout(e, fs, w)
    return mw <= w * 0.995 + 0.5 and bh <= h + 0.5


def max_font(e: Dict[str, Any], w: float, h: float, hi: float, lo: float = 1.0) -> int:
    """Largest integer font size <= hi that fits (w, h); lo if none does."""
    hi_i, lo_i = max(1, int(hi)), max(1, int(lo))
    if hi_i <= lo_i:
        return lo_i
    if fits(e, hi_i, w, h):
        return hi_i
    a, b = lo_i, hi_i  # a may not fit; find the largest fitting in [a, b)
    if not fits(e, a, w, h):
        return a
    while b - a > 1:
        m = (a + b) // 2
        if fits(e, m, w, h):
            a = m
        else:
            b = m
    return a


def natural_width(e: Dict[str, Any], fs: float) -> float:
    """Width of the text on one line."""
    mw, _, _ = layout(e, fs, 1e9)
    return mw


def longest_word(e: Dict[str, Any], fs: float) -> float:
    words = str(e.get("content", "")).split()
    if not words:
        return 0.0
    best = 0.0
    for wd in words:
        best = max(best, natural_width(dict(e, content=wd), fs))
    return best
