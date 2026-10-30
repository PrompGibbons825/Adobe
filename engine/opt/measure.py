"""Text metrics with the renderer's own fonts and wrap rule (owner: text measurement only)."""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]


def _fonts_dir() -> str:
    for cand in (ROOT / "brand_kit" / "fonts", Path(os.environ.get("CHAMELEON_BRAND_KIT", "/nonexistent")) / "fonts",
                 Path("/app/brand_kit/fonts")):
        if cand.is_dir() and any(cand.glob("*.ttf")):
            return str(cand)
    return ""


_FM = None
_OK = None


def _fm():
    global _FM, _OK
    if _OK is None:
        try:
            from renderer.converter.brand_fonts import load_brand_font_manager
            d = _fonts_dir()
            _FM = load_brand_font_manager(d) if d else None
            _OK = _FM is not None
        except Exception:  # noqa: BLE001
            _FM, _OK = None, False
    return _FM


_BOLD = {"bold", "semibold", "extrabold", "black"}


def _style(el) -> Tuple[str, str]:
    return _style_k((el.get("font_weight") or "regular").lower(), el.get("font_style") == "italic" or el.get("italic") is True)


@functools.lru_cache(maxsize=64)
def _style_k(w: str, it: bool) -> Tuple[str, str]:
    try:
        from cr_renderer.fonts import FONT_WEIGHTS
        if w not in FONT_WEIGHTS:
            w = "regular"
    except Exception:  # noqa: BLE001
        pass
    b = w in _BOLD
    st = "bolditalic" if b and it else "bold" if b else "italic" if it else "regular"
    return w, st


@functools.lru_cache(maxsize=4096)
def _typeface(family: str, weight: str, style: str):
    import skia
    fm = _fm()
    return skia.Typeface.MakeFromData(fm.lookup(family, weight, style))


@functools.lru_cache(maxsize=200000)
def _widths(family, weight, style, size, text) -> Tuple[float, ...]:
    import skia
    if not text:
        return ()
    font = skia.Font(_typeface(family, weight, style), size)
    return tuple(font.getWidths(font.textToGlyphs(text)))


@functools.lru_cache(maxsize=4096)
def _metrics(family, weight, style, size):
    import skia
    font = skia.Font(_typeface(family, weight, style), size)
    m = font.getMetrics()
    return m.fAscent, m.fDescent


def _wrap(text: str, widths, max_width: float) -> str:
    # identical to renderer.converter.wrap.wrap_text
    if not text or max_width <= 0:
        return text
    chars = list(text)
    line_width = 0.0
    last = None
    for i, c in enumerate(chars):
        if c == "\n":
            line_width = 0.0
            last = None
            continue
        w = widths[i] if i < len(widths) else 0.0
        if line_width + w > max_width and last is not None:
            chars[last] = "\n"
            line_width = sum(widths[j] for j in range(last + 1, i + 1) if chars[j] != "\n")
            last = None
        else:
            line_width += w
        if c == " ":
            last = i
    return "".join(chars)


def text_block(el: dict, fs: int, width: float) -> Tuple[int, float, float]:
    """(n_lines, max rendered line width, rendered height) for text el at font size fs in a box of width."""
    content = str(el.get("content", ""))
    lh = float(el.get("line_height", 1.2) or 1.2)
    ls = float(el.get("letter_spacing", 0.0) or 0.0) * fs / max(1e-6, float(el.get("font_size", fs) or fs))
    fam = el.get("font_family", "Montserrat")
    if _fm() is None:
        # crude fallback: 0.55 em per char
        cw = 0.55 * fs + ls
        per = max(1, int(width // max(cw, 1e-6)))
        n = max(1, -(-len(content) // per))
        return n, min(width, len(content) * cw), (n - 1) * fs * lh + 1.2 * fs
    w, st = _style(el)
    wf = _widths(fam, w, st, float(fs), content)
    wrapped = _wrap(content, wf, width)
    wi = _widths(fam, w, st, float(int(fs)), content)
    lines = wrapped.split("\n")
    maxw = 0.0
    pos = 0
    for ln in lines:
        seg = wi[pos:pos + len(ln)]
        lw = sum(seg) + ls * max(0, len(ln) - 1)
        maxw = max(maxw, lw)
        pos += len(ln) + 1
    asc, desc = _metrics(fam, w, st, float(int(fs)))
    n = len(lines)
    h = (n - 1) * fs * lh + (desc - asc)
    return n, maxw, h


def natural_width(el: dict, fs: int) -> float:
    """Width of the content on one line."""
    return text_block(el, fs, 1e9)[1]


def longest_word(el: dict, fs: int) -> float:
    content = str(el.get("content", ""))
    best = 0.0
    for wd in content.replace("\n", " ").split(" "):
        if wd:
            e = dict(el)
            e["content"] = wd
            best = max(best, text_block(e, fs, 1e9)[1])
    return best


# ---- fast path for search: advances scale linearly with size; exact check happens at emit ----
_FAST = {}


def _key(el):
    return (str(el.get("content", "")), el.get("font_family", "Montserrat"), el.get("font_weight"),
            el.get("font_style"), el.get("italic"), float(el.get("letter_spacing", 0.0) or 0.0),
            float(el.get("line_height", 1.2) or 1.2))


def _fast_base(el):
    k = _key(el)
    v = _FAST.get(k)
    if v is None:
        content = k[0]
        if _fm() is None:
            wf = tuple(0.55 * 100 for _ in content)
            asc, desc = -90.0, 25.0
        else:
            w, st = _style(el)
            wf = _widths(k[1], w, st, 100.0, content)
            asc, desc = _metrics(k[1], w, st, 100.0)
        words = []
        cur = 0.0
        cnt = 0
        for c, cw in zip(content, wf):
            if c in " \n":
                if cnt:
                    words.append((cur, cnt))
                cur, cnt = 0.0, 0
            else:
                cur += cw
                cnt += 1
        if cnt:
            words.append((cur, cnt))
        v = (wf, asc, desc, words, {})
        _FAST[k] = v
    return v


def fast_block(el, fs: float, width: float):
    """Approximate (n_lines, max line width, height) — linear scaling of size-100 advances."""
    wf, asc, desc, words, cache = _fast_base(el)
    width = round(width, 1)
    fs = int(fs)
    key = (fs, width, float(el.get("font_size", fs) or fs))
    r = cache.get(key)
    if r is not None:
        return r
    content = str(el.get("content", ""))
    lh = float(el.get("line_height", 1.2) or 1.2)
    ls = float(el.get("letter_spacing", 0.0) or 0.0) * fs / max(1e-6, float(el.get("font_size", fs) or fs))
    k = int(fs) / 100.0
    ws = [x * k for x in wf]
    wrapped = _wrap(content, ws, width)
    maxw = 0.0
    pos = 0
    lines = wrapped.split("\n")
    for ln in lines:
        lw = sum(ws[pos:pos + len(ln)]) + ls * max(0, len(ln) - 1)
        maxw = max(maxw, lw)
        pos += len(ln) + 1
    n = len(lines)
    r = (n, maxw, (n - 1) * fs * lh + (desc - asc) * k)
    cache[key] = r
    return r


def fast_longest(el, fs: float) -> float:
    wf, asc, desc, words, cache = _fast_base(el)
    key = ("lw", int(fs), float(el.get("font_size", fs) or fs))
    r = cache.get(key)
    if r is None:
        r = cache[key] = _fast_longest(el, fs, words)
    return r


def _fast_longest(el, fs, words) -> float:
    ls = float(el.get("letter_spacing", 0.0) or 0.0) * fs / max(1e-6, float(el.get("font_size", fs) or fs))
    k = int(fs) / 100.0
    return max((w * k + ls * max(0, c - 1) for w, c in words), default=0.0)


def fast_natural(el, fs: float) -> float:
    wf, asc, desc, words, cache = _fast_base(el)
    content = str(el.get("content", ""))
    ls = float(el.get("letter_spacing", 0.0) or 0.0) * fs / max(1e-6, float(el.get("font_size", fs) or fs))
    k = int(fs) / 100.0
    best = 0.0
    pos = 0
    for ln in content.split("\n"):
        best = max(best, sum(wf[pos:pos + len(ln)]) * k + ls * max(0, len(ln) - 1))
        pos += len(ln) + 1
    return best
