"""Text layout exactly as the official renderer wraps and draws it.

    from fundamentals import measure_text
    lay = measure_text(element)          # a schema v0.9 text element (dict)
    lay.lines, lay.n_lines               # the lines as drawn
    lay.font_box, lay.ink_box            # element-local (x0, y0, x1, y1); None if nothing drawn
    lay.canvas_ink_box()                 # canvas coordinates, rotation included

Wrap = ``renderer/converter/wrap.py`` (greedy, spaces only, per-char advances at the float size,
element-level face). Draw = ``cr_renderer.text_utils`` (int(font_size), style spans, letter
spacing, alignment per line, line i top at i*fs*lh, baseline at top - ascent). See QUIRKS.md.
"""

from __future__ import annotations

import html
import itertools
import math
from dataclasses import dataclass, field
from itertools import accumulate
from typing import List, Optional, Tuple

from .fonts import BOLD_WEIGHTS, FontBook, face_style, fontbook, is_italic, normalize_weight

Box = Tuple[float, float, float, float]
DEFAULT_LINE_HEIGHT = 1.2


def wrap_text(text: str, widths: List[float], max_width: float) -> str:
    """renderer/converter/wrap.py:wrap_text, verbatim semantics."""
    if not text or max_width <= 0:
        return text
    chars = list(text)
    line_width = 0.0
    last_space = None
    for i, c in enumerate(chars):
        if c == "\n":
            line_width = 0.0
            last_space = None
            continue
        w = widths[i] if i < len(widths) else 0.0
        if line_width + w > max_width and last_space is not None:
            chars[last_space] = "\n"
            line_width = sum(widths[j] for j in range(last_space + 1, i + 1) if chars[j] != "\n")
            last_space = None
        else:
            line_width += w
        if c == " ":
            last_space = i
    return "".join(chars)


@dataclass
class Span:
    text: str
    bold: bool
    italic: bool
    color: str
    x: float = 0.0  # element-local left of the span
    width: float = 0.0


@dataclass
class Line:
    text: str
    top: float
    baseline: float
    x: float
    width: float
    spans: List[Span] = field(default_factory=list)
    ink: Optional[Box] = None
    glyphs: List[Box] = field(default_factory=list)  # element-local ink box of each glyph


@dataclass
class TextLayout:
    element: dict
    wrapped: str
    lines: List[Line]
    font_size: float  # as given
    draw_size: int  # int(font_size): what glyphs are drawn at
    line_height: float
    ascent: float  # positive, of the int-size element face
    descent: float
    font_box: Optional[Box]  # union of line advance boxes x [top, top + asc + desc]
    ink_box: Optional[Box]  # union of glyph bounds as positioned
    resolved_family: str
    family_exact: bool

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def text_lines(self) -> List[str]:
        return [ln.text for ln in self.lines]

    @property
    def max_line_width(self) -> float:
        return max((ln.width for ln in self.lines), default=0.0)

    def overflow(self) -> dict:
        """How far the drawing leaves the element box (px, >0 means outside), by the ink box."""
        e = self.element
        w, h = float(e.get("width", 1.0)), float(e.get("height", 1.0))
        b = self.ink_box
        if b is None:
            return {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
        return {"left": max(0.0, -b[0]), "top": max(0.0, -b[1]), "right": max(0.0, b[2] - w),
                "bottom": max(0.0, b[3] - h)}

    def canvas_box(self, box: Optional[Box]) -> Optional[Box]:
        return to_canvas(self.element, box)

    def canvas_ink_box(self) -> Optional[Box]:
        """Canvas AABB of the ink; under rotation the union of each line's rotated ink box."""
        if not float(self.element.get("rotation", 0.0) or 0.0):
            return to_canvas(self.element, self.ink_box)
        out = None
        for ln in self.lines:
            for g in ln.glyphs:
                out = _union(out, to_canvas(self.element, g))
        return out

    def canvas_font_box(self) -> Optional[Box]:
        return to_canvas(self.element, self.font_box)


def to_canvas(e: dict, box: Optional[Box]) -> Optional[Box]:
    """Element-local box -> canvas AABB, applying translate(x, y) and rotate(deg, w/2, h/2)."""
    if box is None:
        return None
    x, y = float(e.get("x", 0.0)), float(e.get("y", 0.0))
    rot = float(e.get("rotation", 0.0) or 0.0)
    if not rot:
        return (box[0] + x, box[1] + y, box[2] + x, box[3] + y)
    cx, cy = float(e.get("width", 1.0)) / 2.0, float(e.get("height", 1.0)) / 2.0
    t = math.radians(rot)
    c, s = math.cos(t), math.sin(t)
    pts = []
    for px, py in ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])):
        dx, dy = px - cx, py - cy
        pts.append((cx + dx * c - dy * s + x, cy + dx * s + dy * c + y))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _union(a: Optional[Box], b: Optional[Box]) -> Optional[Box]:
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def char_widths(book: FontBook, e: dict, size: Optional[float] = None) -> List[float]:
    """The converter's per-char advances (element face, float size)."""
    content = str(e.get("content", ""))
    if not content:
        return []
    weight = normalize_weight(e.get("font_weight"))
    style = face_style(weight in BOLD_WEIGHTS, is_italic(e.get("font_style") or e.get("italic")))
    font = book.font(e.get("font_family", "Montserrat"), weight, style,
                     float(e.get("font_size", 24)) if size is None else float(size))
    return list(font.getWidths(font.textToGlyphs(content)))


def style_maps(e: dict, n: int) -> Tuple[List[bool], List[bool], List[str]]:
    """converter._append_text's per-char bold / italic / colour (runs inclusive of ``end``)."""
    weight = normalize_weight(e.get("font_weight"))
    italic = is_italic(e.get("font_style") or e.get("italic"))
    style = face_style(weight in BOLD_WEIGHTS, italic)
    bold_f = [weight in BOLD_WEIGHTS] * n
    ital_f = [style in ("italic", "bolditalic")] * n
    colors = [e.get("color", "rgba(0,0,0,1)")] * n
    for run in e.get("style_runs") or []:
        start = max(0, int(run.get("start", 0)))
        end = min(n - 1, int(run.get("end", n - 1)))
        if start > end:
            continue
        if "color" in run:
            for i in range(start, end + 1):
                colors[i] = run["color"]
        if "font_weight" in run:
            b = normalize_weight(run.get("font_weight")) in BOLD_WEIGHTS
            for i in range(start, end + 1):
                bold_f[i] = b
        if "font_style" in run or "italic" in run:
            it = is_italic(run.get("font_style") or run.get("italic"))
            for i in range(start, end + 1):
                ital_f[i] = it
    return bold_f, ital_f, colors


def _groups(values) -> List[Tuple[object, int, int]]:
    out, off = [], 0
    for v, it in itertools.groupby(values):
        k = len(list(it))
        out.append((v, off, off + k - 1))
        off += k
    return out


def _draw_face(bold: bool, italic: bool) -> Tuple[str, str]:
    """text_utils.make_text_blob's weight/style from the span flags."""
    if bold and italic:
        return "bold", "bolditalic"
    if bold:
        return "bold", "bold"
    if italic:
        return "regular", "italic"
    return "regular", "regular"


def measure_text(e: dict, book: Optional[FontBook] = None, fonts_dir: Optional[str] = None,
                 with_ink: bool = True) -> TextLayout:
    """Lay out a text element as the renderer does. ``e`` is a schema v0.9 text element."""
    book = book or fontbook(fonts_dir)
    content = str(e.get("content", ""))
    family = str(e.get("font_family", "Montserrat"))
    size = float(e.get("font_size", 24))
    width = float(e.get("width", 1.0))
    lh = float(e.get("line_height", DEFAULT_LINE_HEIGHT))
    ls = float(e.get("letter_spacing", 0.0))
    align = e.get("text_align", "left")
    cap = bool(e.get("capitalize", False))

    wrapped = wrap_text(content, char_widths(book, e), width)
    n = len(wrapped)
    bold_f, ital_f, colors = style_maps(e, n)
    text_line = list(accumulate(int(c == "\n") for c in wrapped))

    # cr_renderer: TextElement.text is html.unescape()d, the maps keep the converter's indices.
    text = html.unescape(wrapped)
    tl = len(text)

    def dense(groups, default):
        x = [default] * tl
        for v, a, b in groups:
            for i in range(a, b + 1):
                if i < tl:
                    x[i] = v
        return x

    style_map = list(zip(dense(_groups(colors), None), dense(_groups(bold_f), False), dense(_groups(ital_f), False)))
    line_items = _groups(text_line) or [(None, 0, tl - 1)]

    draw_size = int(size)
    e_w, e_s = _draw_face(normalize_weight(e.get("font_weight")) in BOLD_WEIGHTS,
                          is_italic(e.get("font_style") or e.get("italic")))
    path, fam_name, exact = book.resolve(family, e_w, e_s)
    from .fonts import _font  # noqa: PLC0415

    m = _font(path, float(draw_size)).getMetrics()
    asc, desc = -m.fAscent, m.fDescent

    lines: List[Line] = []
    font_box: Optional[Box] = None
    ink_box: Optional[Box] = None
    for li, (_, a, b) in enumerate(line_items):
        line_text = text[a:b + 1]
        line_style = style_map[a:b + 1]
        top = li * size * (lh or 1.0)
        spans: List[Span] = []
        blobs = []
        off = 0
        line_metrics_asc = None
        for st, it in itertools.groupby(line_style):
            k = len(list(it))
            s_text = line_text[off:off + k]
            off += k
            color, bold, italic = st
            t = s_text.upper() if cap else s_text
            t = t.strip("\n")
            w_, s_ = _draw_face(bold, italic)
            font = _font(book.resolve(family, w_, s_)[0], float(draw_size))
            glyphs = font.textToGlyphs(t)
            xpos = [p + ls * i for i, p in enumerate(font.getXPos(glyphs))]
            adv = sum(font.getWidths(glyphs)) + ls * max(0, len(glyphs) - 1)
            fm = font.getMetrics()
            blobs.append((font, glyphs, xpos, adv, fm))
            spans.append(Span(t, bold, italic, color, 0.0, adv))
        total = sum(bl[3] for bl in blobs)
        if align == "center":
            x = (width - total) / 2.0
        elif align == "right":
            x = width - total
        else:
            x = 0.0
        line = Line("".join(s.text for s in spans), top, top + asc, x, total, spans)
        cx = x
        lbox = None
        for sp, (font, glyphs, xpos, adv, fm) in zip(spans, blobs):
            sp.x = cx
            base = top - fm.fAscent  # each span's own ascent, as drawn
            if with_ink and len(glyphs):
                for r, gx in zip(font.getBounds(glyphs), xpos):
                    if r.width() <= 0 or r.height() <= 0:
                        continue
                    g = (cx + gx + r.left(), base + r.top(), cx + gx + r.right(), base + r.bottom())
                    line.glyphs.append(g)
                    lbox = _union(lbox, g)
            cx += adv
        line.ink = lbox
        ink_box = _union(ink_box, lbox)
        if total > 0 or line.text:
            font_box = _union(font_box, (min(x, x + total), top, max(x, x + total), top + asc + desc))
        lines.append(line)

    return TextLayout(e, wrapped, lines, size, draw_size, lh, asc, desc, font_box, ink_box, fam_name, exact)


def text_block_height(e: dict, n_lines: int, book: Optional[FontBook] = None) -> float:
    """Height of the drawn font box of ``n_lines`` lines from the element top."""
    lay = measure_text(dict(e, content="X"), book, with_ink=False)
    return (n_lines - 1) * lay.font_size * lay.line_height + lay.ascent + lay.descent
