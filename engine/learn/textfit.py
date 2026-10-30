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
FONTS = ROOT / "brand_kit" / "fonts"
_fonts = None  # normalized family -> [(weight, style, path)] in sorted file order

# A self-contained copy of the lookup that brand_fonts.py + cr_renderer.fonts.FontManager do
# (importing cr_renderer pulls in huggingface_hub, ~1-15 s): family title-cased with weight words
# stripped, else "Montserrat", else the first family loaded; exact weight+style, else the
# family's first file.
_WEIGHTS = ("thin", "extralight", "light", "regular", "medium", "semibold", "bold", "extrabold", "black")


def _norm_family(name: str) -> str:
    name = name.replace("_", " ").title()
    for w in (" Bold", " Regular", " Light", " Italic", " Medium"):
        name = name.replace(w, "")
    return name


def _load_fonts():
    global _fonts
    if _fonts is None:
        import re
        _fonts = {}
        for path in sorted(FONTS.glob("*")):
            if path.suffix.lower() not in (".ttf", ".otf"):
                continue
            prefix, _, suffix = path.stem.partition("-")
            family = re.sub(r"(?<!^)(?=[A-Z])", " ", prefix).strip()
            suffix = suffix or "Regular"
            italic = bool(re.search("italic", suffix, re.I))
            wt = re.sub("italic", "", suffix, flags=re.I).strip().lower() or "regular"
            wt = wt if wt in _WEIGHTS else "regular"
            st = ("bolditalic" if wt == "bold" else "italic") if italic else ("bold" if wt == "bold" else "regular")
            _fonts.setdefault(_norm_family(family), []).append((wt, st, path))
    return _fonts


def _font_manager():
    try:
        return bool(_load_fonts())
    except Exception:  # noqa: BLE001
        return False


def _weight_style(e: Dict[str, Any]) -> Tuple[str, str]:
    w = str(e.get("font_weight") or "regular").lower()
    bold = w in {"bold", "semibold", "extrabold", "black"}
    it = e.get("font_style") == "italic" or e.get("italic") is True
    if w not in _WEIGHTS:
        w = "regular"
    style = "bolditalic" if bold and it else "bold" if bold else "italic" if it else "regular"
    return w, style


@lru_cache(maxsize=4096)
def _typeface(family: str, weight: str, style: str):
    import skia
    fonts = _load_fonts()
    fam = fonts.get(_norm_family(family)) or fonts.get("Montserrat") or next(iter(fonts.values()))
    t = family.replace("_", " ").title()
    if " Bold" in t:  # cr_renderer get_font_weight/style: a weight word in the family name wins
        weight = "bold"
        style = "bolditalic" if ("italic" in style or " Italic" in t) else "bold"
    elif " Italic" in t:
        style = "bolditalic" if weight == "bold" else "italic"
    path = next((p for (w, s, p) in fam if w == weight and s == style), fam[0][2])
    return skia.Typeface.MakeFromData(skia.Data.MakeFromFileName(str(path)))


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
    text = str(e.get("content", ""))
    fam = str(e.get("font_family", "Montserrat"))
    weight, style = _weight_style(e)
    lh = float(e.get("line_height") or 1.2)
    fs_i = max(1, int(fs))
    # letter spacing is a property of the type: it scales with the size (em-relative) while the
    # engine is still choosing a size; the final px value is written out with the size
    ls = float(e["_lsem"]) * fs_i if e.get("_lsem") is not None else float(e.get("letter_spacing") or 0.0)
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


def wrap_text(text, widths, max_width):
    """Verbatim logic of renderer/converter/wrap.py (greedy, spaces only, in place)."""
    if not text or max_width <= 0:
        return text
    chars = list(text)
    line_width = 0.0
    last_space_idx = None
    for i, c in enumerate(chars):
        if c == "\n":
            line_width = 0.0
            last_space_idx = None
            continue
        w = widths[i] if i < len(widths) else 0.0
        if line_width + w > max_width and last_space_idx is not None:
            chars[last_space_idx] = "\n"
            line_width = sum(widths[j] for j in range(last_space_idx + 1, i + 1) if chars[j] != "\n")
            last_space_idx = None
        else:
            line_width += w
        if c == " ":
            last_space_idx = i
    return "".join(chars)
