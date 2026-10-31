"""Exact text measurement using the same fonts + wrap logic as the renderer.

The official renderer wraps greedily on spaces using skia glyph widths and
draws line i at y = i * font_size * line_height (top-aligned, no clipping).
We reuse its own helpers so our layout math matches what gets drawn.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

FONTS_DIR = Path(os.environ.get("CHAMELEON_FONTS_DIR", "brand_kit/fonts"))
UNWRAP = os.environ.get("CHAMELEON_UNWRAP", "1") == "1"   # hand-wrapped breaks may go where text re-flows; 0 keeps every string exact
FONTS_PICKLE = os.environ.get("CHAMELEON_FONTS_PICKLE")  # dev only: Crello font set
_FM = None


def _manager():
    global _FM
    if _FM is None:
        try:
            if FONTS_PICKLE:
                from cr_renderer.fonts import FontManager

                _FM = FontManager(FONTS_PICKLE)
            else:
                from renderer.converter.brand_fonts import load_brand_font_manager

                _FM = load_brand_font_manager(str(FONTS_DIR))
        except Exception:
            _FM = False
    return _FM


def _weight_style(el: dict) -> tuple[str, str]:
    bold = (el.get("font_weight") or "regular") == "bold"
    italic = el.get("font_style") == "italic"
    weight = "bold" if bold else "regular"
    if bold and italic:
        style = "bolditalic"
    elif bold:
        style = "bold"
    elif italic:
        style = "italic"
    else:
        style = "regular"
    return weight, style


@lru_cache(maxsize=50000)
def _widths(family: str, weight: str, style: str, size: float, text: str) -> tuple:
    fm = _manager()
    if fm:
        try:
            from renderer.converter.wrap import char_widths

            return tuple(char_widths(fm, family, weight, style, size, text))
        except Exception:
            pass
    # Fallback estimate: average glyph ~0.55em (0.6em bold)
    k = 0.6 if weight == "bold" else 0.55
    return tuple((size * (0.3 if c == " " else k)) for c in text)


def _wrap(text: str, widths: tuple, max_width: float) -> str:
    # identical to renderer.converter.wrap.wrap_text
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


def measure(el: dict, fs: int, max_w: float) -> dict:
    """Wrap `el`'s content at font size fs within max_w.
    Returns lines, widest line width, height, longest word width."""
    text = str(el.get("content", ""))
    weight, style = _weight_style(el)
    widths = _widths(el.get("font_family", ""), weight, style, float(fs), text)
    wrapped = _wrap(text, widths, max_w)
    lines = wrapped.split("\n")
    ls = float(el.get("letter_spacing", 0) or 0)
    if ls and el.get("font_size"):
        ls *= fs / float(el["font_size"])          # tracking scales with the type
    line_ws = []
    idx = 0
    for ln in lines:
        line_ws.append(sum(widths[idx: idx + len(ln)]) + ls * max(0, len(ln) - 1))
        idx += len(ln) + 1
    # longest word
    word_w, cur = 0.0, 0.0
    for c, w in zip(text, widths):
        if c in " \n":
            word_w = max(word_w, cur)
            cur = 0.0
        else:
            cur += w + ls
    word_w = max(word_w, cur)
    lh = float(el.get("line_height", 1.2))
    return {
        "lines": len(lines),
        "width": max(line_ws) if line_ws else 0.0,
        "height": len(lines) * fs * lh,
        "word_w": word_w,
        "fs": fs,
    }


def wrapped(el: dict, fs: int, max_w: float) -> str:
    """The content as the renderer will break it at font size fs within max_w."""
    text = str(el.get("content", ""))
    weight, style = _weight_style(el)
    return _wrap(text, _widths(el.get("font_family", ""), weight, style, float(fs), text), max_w)


def unwrap_breaks(content: str) -> str:
    """Drop line breaks that only wrapped a paragraph by hand, so the text can
    re-wrap cleanly in its new box. A break is kept when it ends a sentence
    or clause (. ! ? :), when the next line starts a new piece of information
    (a capital or a number, unless the whole text is set in caps), or when
    the text is display type (fewer than 3 words per line on average)."""
    if not UNWRAP:
        return content
    lines = content.split("\n")
    if len(lines) < 2:
        return content
    words = [len(ln.split()) for ln in lines if ln.strip()]
    if not words or sum(words) / len(words) < 3:
        return content
    caps = content.upper() == content
    out = lines[0]
    for ln in lines[1:]:
        prev, nxt = out.rstrip(), ln.lstrip()
        continues = bool(prev) and bool(nxt) and prev[-1] not in ".!?:" and (
            nxt[0].islower()                                   # "...lovers and / have a good time"
            or (caps and nxt[0].isalpha() and not prev[-1].isdigit()))   # an all-caps sentence
        if continues:
            out = prev + " " + nxt
        else:
            out += "\n" + ln                                  # a new piece of information: kept
    return out


def flow(el: dict) -> dict:
    """The element as it is set when it re-flows into a new width."""
    return dict(el, content=el["_flow"]) if "_flow" in el else el


def single_line_width(el: dict, fs: int) -> float:
    text = str(el.get("content", ""))
    weight, style = _weight_style(el)
    return sum(_widths(el.get("font_family", ""), weight, style, float(fs), text))


def fit(el: dict, max_w: float, max_h: float, fs_max: float, fs_min: int = 1,
        max_lines: int | None = None) -> dict | None:
    """Largest integer font size <= fs_max whose wrapped text fits the box."""
    hi = max(fs_min, int(fs_max))

    def ok(fs):
        m = measure(el, fs, max_w)
        if m["word_w"] > max_w + 0.5 or m["height"] > max_h + 0.5:
            return None
        if max_lines is not None and m["lines"] > max_lines:
            return None
        return m

    m = ok(hi)
    if m:
        return m
    lo = fs_min
    if not ok(lo):
        return None
    # binary search for the boundary, then walk down to handle non-monotone wraps
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return ok(lo)


@lru_cache(maxsize=64)
def _metrics(family: str, weight: str, style: str) -> tuple:
    """(ascent, cap_height, descent) as fractions of font size."""
    fm = _manager()
    if fm:
        try:
            import skia

            tf = skia.Typeface.MakeFromData(fm.lookup(family, weight, style))
            m = skia.Font(tf, 100).getMetrics()
            return (-m.fAscent / 100, (m.fCapHeight or 70) / 100, m.fDescent / 100)
        except Exception:
            pass
    return (1.0, 0.7, 0.3)


def metrics(el: dict) -> tuple:
    w, s = _weight_style(el)
    return _metrics(el.get("font_family", ""), w, s)


def visual_box(el: dict, x: float, y: float, w: float, fs: int, m: dict) -> tuple:
    """Approximate inked box of a placed text element (x0,y0,x1,y1)."""
    asc, cap, desc = metrics(el)
    lh = float(el.get("line_height", 1.2))
    top = y + (asc - cap) * fs
    bottom = y + (m["lines"] - 1) * fs * lh + asc * fs + desc * fs * 0.6
    tw = min(w, m["width"])
    align = el.get("text_align", "left")
    if align == "center":
        x0 = x + (w - tw) / 2
    elif align == "right":
        x0 = x + w - tw
    else:
        x0 = x
    return (x0, top, x0 + tw, bottom)


def optical_top(el: dict, fs: int, lines: int, center_y: float) -> float:
    """y for a text box so its cap-height block is centered on center_y."""
    asc, cap, desc = metrics(el)
    lh = float(el.get("line_height", 1.2))
    block_top_off = (asc - cap) * fs
    block_h = (lines - 1) * fs * lh + cap * fs
    return center_y - block_h / 2 - block_top_off
