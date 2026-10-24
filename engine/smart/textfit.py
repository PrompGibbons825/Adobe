"""Text measurement that mirrors the official renderer exactly.

The renderer (renderer/converter + cr-renderer) does three things that matter
for layout:

* it word-wraps greedily at real glyph advances, turning spaces into
  newlines, using the element's *width* and the element's base weight/style;
* it draws each line at ``y = i * font_size * line_height`` with the glyph
  ascent hanging below that line, so text is top-aligned and never clipped;
* it truncates ``font_size`` to an int when drawing.

Everything here re-implements (or imports) those rules, so a text box this
engine plans is the text box that gets rendered. If skia or the fonts are not
available we fall back to a conservative width estimate rather than failing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import paths

try:  # the same libraries the renderer uses
    import skia  # type: ignore
except Exception:  # pragma: no cover - only without skia
    skia = None

_BOLD = {"bold", "semibold", "extrabold", "black"}
_WEIGHTS = {"thin", "light", "extralight", "regular", "medium", "semibold", "bold", "extrabold", "black"}
_DEFAULT_LH = 1.2

# Per-family fallback metrics (fraction of the em) if a font cannot be read.
_FALLBACK_METRICS = (1.0, 0.3, 0.7)  # ascent, descent, cap height
_DESCENDERS = set("gjpqy,;()[]{}|/@_$Qµ")


def norm_weight(weight: Optional[str]) -> str:
    if not weight:
        return "regular"
    weight = str(weight).lower()
    return weight if weight in _WEIGHTS else "regular"


def is_italic(value) -> bool:
    if isinstance(value, bool):
        return value
    return value == "italic"


def font_style_key(weight: str, italic: bool) -> str:
    bold = weight in _BOLD
    if bold and italic:
        return "bolditalic"
    if bold:
        return "bold"
    if italic:
        return "italic"
    return "regular"


class _FontBook:
    """Font lookup that resolves (family, weight, style) the way the renderer does."""

    def __init__(self) -> None:
        self.manager = None
        self._typefaces: Dict[Tuple[str, str, str], object] = {}
        self._metrics: Dict[Tuple[str, str, str], Tuple[float, float, float]] = {}
        fonts_dir = paths.fonts_dir()
        if fonts_dir is None or skia is None:
            return
        try:
            from renderer.converter.brand_fonts import load_brand_font_manager  # type: ignore

            self.manager = load_brand_font_manager(str(fonts_dir))
        except Exception:
            self.manager = None

    def typeface(self, family: str, weight: str, style: str):
        key = (family, weight, style)
        if key in self._typefaces:
            return self._typefaces[key]
        tf = None
        if self.manager is not None:
            try:
                tf = skia.Typeface.MakeFromData(self.manager.lookup(family, weight, style))
            except Exception:
                tf = None
        self._typefaces[key] = tf
        return tf

    def metrics(self, family: str, weight: str, style: str) -> Tuple[float, float, float]:
        """(ascent, descent, cap_height) as fractions of the font size."""
        key = (family, weight, style)
        if key not in self._metrics:
            tf = self.typeface(family, weight, style)
            if tf is None:
                self._metrics[key] = _FALLBACK_METRICS
            else:
                m = skia.Font(tf, 100).getMetrics()
                cap = m.fCapHeight if m.fCapHeight > 0 else 70.0
                self._metrics[key] = (-m.fAscent / 100.0, m.fDescent / 100.0, cap / 100.0)
        return self._metrics[key]


_BOOK: Optional[_FontBook] = None


def book() -> _FontBook:
    global _BOOK
    if _BOOK is None:
        _BOOK = _FontBook()
    return _BOOK


@lru_cache(maxsize=4096)
def char_widths(family: str, weight: str, style: str, size: float, text: str) -> Tuple[float, ...]:
    """Per-character advances, measured exactly like renderer.converter.wrap.char_widths."""
    if not text:
        return ()
    tf = book().typeface(family, weight, style)
    if tf is None:
        # Conservative estimate: wide-ish average advance so plans never overflow.
        avg = 0.62 if weight in _BOLD else 0.58
        return tuple((0.3 if c == " " else avg) * size for c in text)
    font = skia.Font(tf, size)
    glyphs = font.textToGlyphs(text)
    widths = list(font.getWidths(glyphs))
    if len(widths) < len(text):  # surrogate pairs etc. - pad conservatively
        widths += [size * 0.6] * (len(text) - len(widths))
    return tuple(widths)


def wrap_text(text: str, widths: Tuple[float, ...], max_width: float) -> str:
    """Identical to renderer.converter.wrap.wrap_text (greedy, spaces -> newlines)."""
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


class _Prepared:
    """Prefix sums and space indexes for one (text, font, size), so wrapping at
    any width costs O(lines * log n) instead of a pass over every character."""

    __slots__ = ("widths", "P", "next_sp", "prev_sp", "paras")

    def __init__(self, text: str, widths: Tuple[float, ...]):
        n = len(text)
        w = list(widths) + [0.0] * max(0, n - len(widths))
        self.widths = w
        P = [0.0] * (n + 1)
        acc = 0.0
        for i in range(n):
            acc += w[i]
            P[i + 1] = acc
        self.P = P
        nxt = [n] * (n + 1)
        for i in range(n - 1, -1, -1):
            nxt[i] = i if text[i] == " " else nxt[i + 1]
        self.next_sp = nxt
        prv = [-1] * n
        last = -1
        for i in range(n):
            if text[i] == " ":
                last = i
            elif text[i] == "\n":
                last = -1
            prv[i] = last
        self.prev_sp = prv
        paras, s = [], 0
        for i, c in enumerate(text):
            if c == "\n":
                paras.append((s, i))
                s = i + 1
        paras.append((s, n))
        self.paras = paras

    def breaks(self, max_w: float) -> List[int]:
        """Indexes of the spaces that wrap_text() would turn into newlines."""
        from bisect import bisect_right

        P, nxt, prv = self.P, self.next_sp, self.prev_sp
        out: List[int] = []
        for s, e in self.paras:
            a = pos = s
            while pos < e:
                k = bisect_right(P, P[a] + max_w, a + 1)  # first k > a with P[k] - P[a] > max_w
                i0 = max(k - 1, pos)
                if i0 >= e:
                    break
                q = nxt[a]
                if q >= e:
                    break
                i_star = max(i0, q + 1)
                if i_star >= e:
                    break
                ls = prv[i_star - 1]
                out.append(ls)
                a = ls + 1
                pos = i_star + 1
        return out


@lru_cache(maxsize=4096)
def _prepared(family: str, weight: str, style: str, size: float, text: str) -> _Prepared:
    return _Prepared(text, char_widths(family, weight, style, size, text))


@dataclass(frozen=True)
class TextStyle:
    family: str
    weight: str  # normalized
    style: str  # regular/bold/italic/bolditalic
    line_height: float
    tracking: float  # letter spacing in ems, so it scales with the type
    runs: Tuple[Tuple[int, int, Optional[bool], Optional[bool]], ...] = ()  # (start, end incl., bold, italic)

    @staticmethod
    def of(el: dict) -> "TextStyle":
        weight = norm_weight(el.get("font_weight"))
        italic = is_italic(el.get("font_style") or el.get("italic"))
        lh = el.get("line_height")
        try:
            lh = float(lh) if lh is not None else _DEFAULT_LH
        except (TypeError, ValueError):
            lh = _DEFAULT_LH
        if not (0.5 <= lh <= 4.0):
            lh = _DEFAULT_LH
        try:
            ls = float(el.get("letter_spacing") or 0.0)
            fs = float(el.get("font_size") or 16.0)
            tracking = ls / fs if math.isfinite(ls) and fs > 0 else 0.0
        except (TypeError, ValueError):
            tracking = 0.0
        return TextStyle(
            family=str(el.get("font_family") or "Montserrat"),
            weight=weight,
            style=font_style_key(weight, italic),
            line_height=lh,
            tracking=tracking,
            runs=_font_runs(el),
        )


def _font_runs(el: dict) -> tuple:
    """The style runs that change the font (bold or italic), read the way the
    converter reads them: ``end`` inclusive, clipped to the text."""
    n = len(str(el.get("content", "")))
    out = []
    for r in el.get("style_runs") or []:
        if not isinstance(r, dict):
            continue
        try:
            a, b = max(0, int(r.get("start", 0))), min(n - 1, int(r.get("end", n - 1)))
        except (TypeError, ValueError):
            continue
        bold = norm_weight(r["font_weight"]) in _BOLD if isinstance(r.get("font_weight"), str) else None
        italic = is_italic(r.get("font_style") or r.get("italic")) if ("font_style" in r or "italic" in r) else None
        if a <= b and (bold is not None or italic is not None):
            out.append((a, b, bold, italic))
    return tuple(out)


@dataclass
class TextBlock:
    """A measured text layout at a given font size and wrap width."""

    font_size: int
    wrap_width: float  # the element width that produces this wrap
    lines: List[str]
    line_widths: List[float]  # rendered advance of each line (incl. letter spacing)
    height: float  # tight box height covering every glyph
    ascent: float  # px
    descent: float  # px
    cap: float  # px
    line_step: float  # px between line tops

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def ink_width(self) -> float:
        return max(self.line_widths) if self.line_widths else 0.0

    def cap_center(self) -> float:
        """Distance from box top to the optical centre of the first line's caps."""
        return self.ascent - self.cap / 2.0

    def last_baseline(self) -> float:
        return (self.n_lines - 1) * self.line_step + self.ascent


def letter_spacing(style: TextStyle, font_size: float) -> float:
    """The letter spacing (px) to emit and draw at ``font_size``."""
    return round(style.tracking * max(1, int(font_size)), 3)


def measure(content: str, style: TextStyle, font_size: float, wrap_width: float, exact: bool = False) -> TextBlock:
    """The block ``content`` makes at ``font_size`` wrapped at ``wrap_width``.
    ``exact`` runs the renderer's own character-level wrap (cached); the
    default fast path gives the same lines (tests.test_engine.TextParity)."""
    if exact:
        return _measure_exact(content, style, max(1, int(font_size)), float(wrap_width))
    return _measure(content, style, font_size, wrap_width, False)


def _drawn_widths(content: str, style: TextStyle, fs: int) -> List[float]:
    """Each character's advance in the font the renderer draws it with."""
    n = len(content)
    bold = [style.weight in _BOLD] * n
    italic = [style.style in ("italic", "bolditalic")] * n
    for a, b, bo, it in style.runs:
        for i in range(a, b + 1):
            if bo is not None:
                bold[i] = bo
            if it is not None:
                italic[i] = it
    out = [0.0] * n
    for key in set(zip(bold, italic)):
        weight = "bold" if key[0] else "regular"
        widths = char_widths(style.family, weight, font_style_key(weight, key[1]), float(fs), content)
        for i in range(n):
            if (bold[i], italic[i]) == key and i < len(widths):
                out[i] = widths[i]
    return out


@lru_cache(maxsize=65536)
def _measure_exact(content: str, style: TextStyle, fs: int, wrap_width: float) -> TextBlock:
    return _measure(content, style, fs, wrap_width, True)


def _measure(content: str, style: TextStyle, font_size: float, wrap_width: float, exact: bool) -> TextBlock:
    fs = max(1, int(font_size))
    if exact:
        widths = char_widths(style.family, style.weight, style.style, float(fs), content)
        wrapped = wrap_text(content, widths, wrap_width)
        cuts = [i for i, c in enumerate(wrapped) if c == "\n"]
        P = [0.0]
        for wv in list(widths) + [0.0] * max(0, len(content) - len(widths)):
            P.append(P[-1] + wv)
    else:
        prep = _prepared(style.family, style.weight, style.style, float(fs), content)
        P = prep.P
        cuts = sorted(set(prep.breaks(wrap_width) if wrap_width > 0 and content else []) |
                      {i for i, c in enumerate(content) if c == "\n"}) if content else []
    if style.runs:  # bold/italic runs are drawn in their own font, though the wrap measured the base one
        D = [0.0]
        for wv in _drawn_widths(content, style, fs):
            D.append(D[-1] + wv)
    else:
        D = P
    lines: List[str] = []
    line_ws: List[float] = []
    start = 0
    spacing = letter_spacing(style, fs)  # drawn between glyphs, ignored by the wrap
    for cut in cuts + [len(content)]:
        seg = content[start:cut]
        lines.append(seg)
        n = len(seg)
        line_ws.append(D[cut] - D[start] + spacing * max(0, n - 1))
        start = cut + 1

    asc, desc, cap = book().metrics(style.family, style.weight, style.style)
    step = fs * style.line_height
    glyph_h = (asc + desc) * fs
    height = (len(lines) - 1) * step + max(step, glyph_h)
    return TextBlock(
        font_size=fs,
        wrap_width=wrap_width,
        lines=lines,
        line_widths=line_ws,
        height=height,
        ascent=asc * fs,
        descent=desc * fs,
        cap=cap * fs,
        line_step=step,
    )


@lru_cache(maxsize=16384)
def width_keeping_breaks(content: str, style: TextStyle, fs0: float, w0: float, fs: float) -> float:
    """A wrap width at ``fs`` that gives the line breaks ``w0`` gives at ``fs0``.

    Glyph advances are not exactly proportional to the size (and sizes are
    whole pixels), so the proportional width can move a word across a line
    break. A few nearby widths are tried before settling for it."""
    w = w0 * max(1, int(fs)) / max(1, int(fs0))
    want = measure(content, style, fs0, w0, exact=True).lines
    for f in (1.0, 0.99, 1.01, 0.98, 1.02, 0.97, 1.03, 0.95, 1.05):
        if measure(content, style, fs, w * f, exact=True).lines == want:
            return w * f
    return w


@lru_cache(maxsize=16384)
def tight_width(content: str, style: TextStyle, font_size: float, width: float) -> float:
    """The narrowest wrap width (at most ``width``) that keeps the line breaks
    ``width`` gives. The renderer's wrap wants a space's advance of room past a
    line's last glyph, so this is a little more than the widest line; with
    letter spacing the drawn lines are wider still (see ``glyph_span``)."""
    blk = measure(content, style, font_size, width, exact=True)
    want = blk.lines
    # never narrower than the widest line's own glyphs (a word that cannot break
    # keeps one line at any width, but its box should still hold it)
    spacing = letter_spacing(style, font_size)
    widest = max((w - spacing * max(0, len(l) - 1) for l, w in zip(blk.lines, blk.line_widths)), default=1.0)
    lo, hi = max(1.0, widest), max(1.0, width, widest)
    for _ in range(16):
        if hi - lo < 0.25:
            break
        mid = (lo + hi) / 2.0
        if measure(content, style, font_size, mid, exact=True).lines == want:
            hi = mid
        else:
            lo = mid
    return min(max(1.0, width, widest), hi + 0.5)  # headroom, so rounding the width cannot add a line


def glyph_span(blk: TextBlock, align: str) -> Tuple[float, float]:
    """Where the glyphs start and end across the element, from its x. Lines are
    aligned by their drawn width, which includes letter spacing the wrap
    ignores, so glyphs can reach past the box on the side(s) away from the
    alignment edge."""
    ink, w = blk.ink_width, blk.wrap_width
    start = 0.0 if align == "left" else ((w - ink) / 2.0 if align == "center" else w - ink)
    return start, start + ink


@lru_cache(maxsize=8192)
def longest_word(content: str, style: TextStyle, font_size: float) -> float:
    fs = float(max(1, int(font_size)))
    best = 0.0
    for word in content.replace("\n", " ").split(" "):
        if word:
            best = max(best, sum(char_widths(style.family, style.weight, style.style, fs, word)))
    return best


def natural_width(content: str, style: TextStyle, font_size: float) -> float:
    """Width of the text without any automatic wrapping (explicit newlines respected)."""
    blk = measure(content, style, font_size, 1e9)
    return blk.ink_width


def balanced(content: str, style: TextStyle, font_size: float, max_width: float) -> TextBlock:
    """Wrap within ``max_width`` using as few lines as possible, then narrow the
    box as far as it goes without adding a line (CSS ``text-wrap: balance``)."""
    return _balanced(content, style, max(1, int(font_size)), round(max(1.0, max_width) * 4) / 4.0)


@lru_cache(maxsize=65536)
def _balanced(content: str, style: TextStyle, fs: int, max_width: float) -> TextBlock:
    full = measure(content, style, fs, max_width)
    n = full.n_lines
    if n <= 1:
        w = math.ceil(full.ink_width + 1.0)
        return measure(content, style, fs, max(w, 1))
    lo = max(1.0, longest_word(content, style, fs))
    hi = max_width
    if lo >= hi:
        return full
    for _ in range(13):
        mid = (lo + hi) / 2.0
        if measure(content, style, fs, mid).n_lines <= n:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.5:
            break
    w = min(max_width, math.ceil(hi + 1.0))
    blk = measure(content, style, fs, w)
    if blk.n_lines > n:  # rounding pushed us over; fall back to the full width wrap
        return full
    return blk


_PHRASE_END = set(",.;:!?—–")
SEMANTIC_SLACK = 1.18  # a break at a phrase edge is preferred unless its lines are this much less even


def phrase_wrap(content: str, style: TextStyle, font_size: float, max_width: float,
                edges: frozenset = frozenset()) -> Optional[TextBlock]:
    """Two lines broken where the copy breaks itself: after punctuation, or
    where a style run starts or ends (indices in ``edges``), if that is no
    more than SEMANTIC_SLACK less even than the most even break. The renderer
    only wraps greedily, so the result is the wrap width that makes the greedy
    wrap break exactly there; None when no break qualifies."""
    fs = max(1, int(font_size))
    widths = char_widths(style.family, style.weight, style.style, float(fs), content)
    if "\n" in content or len(widths) != len(content):
        return None
    P = [0.0]
    for w in widths:
        P.append(P[-1] + w)
    spaces = [i for i, c in enumerate(content) if c == " " and 0 < i < len(content) - 1]
    options = []
    for b in spaces:
        w1, w2 = P[b], P[-1] - P[b + 1]
        nxt = content.find(" ", b + 1)
        reach = P[nxt if nxt > 0 else len(content)]  # line 1 plus the next word, which must not fit
        lo, hi = max(P[b + 1], w2), reach
        if lo + 0.5 >= hi or max(w1, w2) > max_width:
            continue
        semantic = content[b - 1] in _PHRASE_END or b in edges or b + 1 in edges
        options.append((max(w1, w2), semantic, lo + 0.5))
    if not options:
        return None
    best = min(o[0] for o in options)
    chosen = [o for o in options if o[1] and o[0] <= SEMANTIC_SLACK * best]
    if not chosen:
        return None
    blk = measure(content, style, fs, min(chosen)[2], exact=True)
    return blk if blk.n_lines == 2 else None


def run_edges(el: dict) -> frozenset:
    """Indices where a style run (a colour change, say) starts or ends."""
    out = set()
    for r in el.get("style_runs") or []:
        if isinstance(r, dict) and isinstance(r.get("start"), int) and isinstance(r.get("end"), int):
            out.update((r["start"], r["end"] + 1))
    return frozenset(out)


def fit_height(content: str, style: TextStyle, font_size: float, max_height: float,
               max_width: float) -> Optional[TextBlock]:
    """Narrowest balanced block at ``font_size`` whose height fits ``max_height``
    (used for horizontal rows, where width is the scarce resource)."""
    fs = max(1, int(font_size))
    widest = measure(content, style, fs, max_width)
    if widest.height > max_height + 0.5:
        return None
    lo = max(1.0, longest_word(content, style, fs))
    if lo > max_width:
        return None
    hi = max_width
    for _ in range(22):
        mid = (lo + hi) / 2.0
        if measure(content, style, fs, mid).height <= max_height + 0.5:
            hi = mid
        else:
            lo = mid
    blk = balanced(content, style, fs, min(max_width, math.ceil(hi + 1.0)))
    if blk.height > max_height + 0.5:
        blk = measure(content, style, fs, max_width)
    return blk


def has_descender(line: str) -> bool:
    return any(c in _DESCENDERS for c in line)


def drawn_box(el: dict):
    """Where a text element's glyphs actually land, as the renderer draws them:
    exact wrap at the element width, real advances (fallback fonts included),
    line boxes down from y, aligned within the width; a rotated text as the
    bounding box of its glyphs turned about the element's centre."""
    from .geom import Box, rotate_about

    style = TextStyle.of(el)
    w = float(el.get("width", 0) or 0)
    blk = measure(str(el.get("content", "")), style, float(el.get("font_size", 12) or 12), w, exact=True)
    g0, _ = glyph_span(blk, el.get("text_align", "left"))
    x, y = float(el.get("x", 0) or 0), float(el.get("y", 0) or 0)
    b = Box(x + g0, y, max(0.0, blk.ink_width), blk.height)
    rot = float(el.get("rotation", 0) or 0)
    if rot:
        b = rotate_about(b, rot, x + w / 2.0, y + float(el.get("height", 0) or 0) / 2.0)
    return b


def text_overlaps(elements, exempt=frozenset(), tol: float = 1.0):
    """Pairs of visible texts whose drawn glyph boxes overlap (beyond `tol` px)."""
    texts = [e for e in elements if e.get("type") == "text" and float(e.get("opacity", 1) or 0) > 0
             and str(e.get("content", "")).strip()]
    boxes = [(e["id"], drawn_box(e)) for e in texts]
    out = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (ia, a), (ib, b) = boxes[i], boxes[j]
            if (ia, ib) in exempt or (ib, ia) in exempt:
                continue
            ow = min(a.r, b.r) - max(a.x, b.x)
            oh = min(a.b, b.b) - max(a.y, b.y)
            if ow > tol and oh > tol:
                out.append((ia, ib, ow * oh, ow * oh / max(1.0, min(a.area, b.area))))
    return out


def new_text_overlaps(elements, source_elements, deliberate: float = 0.15, slack: float = 0.02):
    """Copy drawn over copy that the source did not draw that way.

    A pair counts when its glyph boxes overlap by a larger share of the
    smaller box than they did in the source (plus ``slack``). Pairs the source
    overlapped substantially (stacked display type, a drop shadow) are the
    design and never count."""
    src = {(a, b): f for a, b, _, f in text_overlaps(source_elements)}
    out = []
    for a, b, area, f in text_overlaps(elements):
        f0 = max(src.get((a, b), 0.0), src.get((b, a), 0.0))
        if f0 < deliberate and f > f0 + slack:
            out.append((a, b, area, f))
    return out
