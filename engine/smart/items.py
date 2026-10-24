"""Sizing ("realizing") a layout item at a given scale, and emitting its elements.

A realized item knows its footprint (the box the layout engine reserves for
it) and can write out its final elements once the engine has decided where
the footprint goes. Footprints are *visual*: a text's tight line box, an
image's visible (non-transparent) pixels, a button's shape, a rotated
ribbon's bounding box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import textfit
from .analyze import Item
from .geom import Box, el_box, rotate_about, rotated_aabb, union_all
from .textfit import TextBlock, TextStyle

INF = float("inf")

CAPTION_GROWTH = 2.0  # how much larger than its art's scale a caption may be set before it is left out
READING_LINE = 70  # characters: copy set to a longer line than its source's stops here

# Legibility floors, as multiples of the canvas's base readable size.
FLOOR_MULT = {
    "headline": 1.45, "subhead": 1.0, "body": 1.0, "cta": 1.0, "badge": 0.9,
    "ribbon": 0.9, "group": 0.9, "legal": 0.8, "text": 1.0,
}


@dataclass
class Ctx:
    W: float
    H: float
    W0: float
    H0: float
    emphasis: float = 1.0  # how much larger than the rest the headline is set in this plan
    measure: float = 1.0  # how much longer than in the source copy lines may run in this plan

    @property
    def display_max(self) -> float:
        """The largest a display headline is set on this canvas, in px: 15% of
        the short side on posts and stories, up to 42% of a strip's height."""
        e = max(self.W, self.H) / max(1.0, min(self.W, self.H))
        return min(self.W, self.H) * (0.15 + 0.27 * min(1.0, max(0.0, (e - 2.0) / 4.0)))

    @property
    def base_fs(self) -> float:
        """Smallest comfortably readable body size on this canvas, in px."""
        return max(10.0, 0.0115 * max(self.W, self.H), 0.016 * min(self.W, self.H))

    @property
    def judge_floor(self) -> int:
        """No visible text below this: cap height >= 7 px when the judge sees the
        canvas at 1024 px on the long side (research 04), and never under 10 px."""
        return max(10, math.ceil(10.0 * max(self.W, self.H) / 1024.0))

    def floor(self, role: str, relax: float = 1.0) -> float:
        return max(float(self.judge_floor), FLOOR_MULT.get(role, 1.0) * self.base_fs * relax)


def font_for(fs0: float, s: float, role: str, ctx: Ctx, relax: float = 1.0) -> int:
    fs0 = max(1.0, float(fs0))
    floor = min(fs0 * max(1.0, s), ctx.floor(role, relax))
    return max(ctx.judge_floor, int(round(max(fs0 * s, floor))))


@dataclass
class Real:
    item: Item
    w: float
    h: float
    scale: float
    fonts: Dict[str, int] = field(default_factory=dict)
    blocks: Dict[str, TextBlock] = field(default_factory=dict)
    emit_fn: Optional[Callable[[float, float], List[dict]]] = None
    dropped: List[dict] = field(default_factory=list)  # member elements left out of this rendition

    def emit(self, x: float, y: float) -> List[dict]:
        els = self.emit_fn(x, y) if self.emit_fn else []
        src = {e["id"]: e for e in self.item.els}
        for e in els:  # tracking scales with the type
            s = src.get(e["id"])
            if e.get("type") == "text" and s is not None and s.get("letter_spacing"):
                e["letter_spacing"] = textfit.letter_spacing(TextStyle.of(s), e["font_size"])
        return els

    @property
    def oh(self) -> Tuple[float, float, float, float]:
        """How far painted boxes (transparent margins, rotation) reach beyond the
        footprint: (left, top, right, bottom), never negative."""
        cached = getattr(self, "_oh", None)
        if cached is None:
            from .geom import el_visual_box

            els = self.emit(0.0, 0.0)
            if not els:
                cached = (0.0, 0.0, 0.0, 0.0)
            else:
                u = el_visual_box(els[0])
                for e in els[1:]:
                    u = u.union(el_visual_box(e))
                cached = (max(0.0, -u.x), max(0.0, -u.y), max(0.0, u.r - self.w), max(0.0, u.b - self.h))
            self._oh = cached
        return cached


def _copy(el: dict) -> dict:
    return {k: v for k, v in el.items() if k != "_idx"}


# ---------------------------------------------------------------------------
# text

def realize_text(it: Item, s: float, ctx: Ctx, max_w: float, max_h: float = INF,
                 mode: str = "stack", relax: float = 1.0, fs: Optional[int] = None) -> Optional[Real]:
    el = it.els[0]
    content = str(el.get("content", ""))
    style = it.style or TextStyle.of(el)
    if fs is None:
        fs = font_for(it.fs0, s, it.role, ctx, relax)
    rot = it.rotation
    line_max = _line_room(rot, max_w, max_h, fs * style.line_height) if rot else max_w
    if line_max < fs:
        return None
    if mode == "fit" and not rot:
        blk = textfit.fit_height(content, style, fs, max_h, max_w)
        if blk is None:
            return None
    else:
        pref = it.measure_em * fs * 1.06
        if ctx.measure != 1.0:  # a longer line, but no longer than a comfortable reading line
            per_char = textfit.natural_width(content, style, fs) / max(1, len(content))
            pref = max(pref, min(pref * ctx.measure, READING_LINE * per_char))
        word = textfit.longest_word(content, style, fs)
        lim = min(line_max, max(pref, word + 2))
        blk = textfit.balanced(content, style, fs, lim)
        if blk.height > max_h + 0.5 and lim < line_max:  # the source measure is a preference, not a wall
            blk = textfit.balanced(content, style, fs, line_max)
        if blk.n_lines == 2:  # break where the copy breaks itself, if that is nearly as even
            blk = textfit.phrase_wrap(content, style, fs, lim, textfit.run_edges(el)) or blk
    if blk.ink_width > line_max + 0.5 or (not rot and blk.height > max_h + 0.5):
        return None
    max_lines = el.get("max_lines")
    if isinstance(max_lines, int) and max_lines > 0 and blk.n_lines > max_lines:
        return None

    exact = textfit.measure(content, style, blk.font_size, blk.wrap_width, exact=True)
    if exact.lines != blk.lines:  # never trust the fast path over the renderer's own rule
        blk = exact
    w, h = float(blk.wrap_width), float(blk.height)
    fp = text_footprint(Box(0.0, 0.0, w, h), blk, el.get("text_align", "left"), rot)
    if fp.w > max_w + 0.5 or fp.h > max_h + 0.5:
        return None

    def emit(x: float, y: float) -> List[dict]:
        e = _copy(el)
        e.update(x=x - fp.x, y=y - fp.y, width=w, height=h, font_size=blk.font_size)
        return [e]

    return Real(item=it, w=fp.w, h=fp.h, scale=fs / max(1.0, it.fs0), fonts={el["id"]: blk.font_size},
                blocks={el["id"]: blk}, emit_fn=emit)


def text_footprint(box: Box, blk: TextBlock, align: str, rot: float) -> Box:
    """Everything a text element paints or reserves: its box, its glyphs
    (which letter spacing can push past the box) and, if it is turned, both
    turned about the box centre, as the renderer turns it and as finalize()
    keeps it on the canvas."""
    g0, _ = textfit.glyph_span(blk, align)
    glyphs = Box(box.x + g0, box.y, blk.ink_width, box.h)
    fp = box.union(glyphs)
    if rot:
        fp = fp.union(rotated_aabb(box, rot)).union(rotate_about(glyphs, rot, box.cx, box.cy))
    return fp


def _line_room(rot: float, max_w: float, max_h: float, line: float) -> float:
    """The longest line a text turned by ``rot`` can have, if one line of
    height ``line`` is to stay inside max_w x max_h."""
    c, s = abs(math.cos(math.radians(rot))), abs(math.sin(math.radians(rot)))
    room = INF
    if c > 1e-3:
        room = min(room, (max_w - line * s) / c)
    if s > 1e-3:
        room = min(room, (max_h - line * c) / s)
    return max(0.0, room)


# ---------------------------------------------------------------------------
# images (and standalone non-decor shapes, which behave like fixed-ratio art)

def realize_image(it: Item, s: float, ctx: Ctx, max_w: float = INF, max_h: float = INF,
                  k: float = 1.0) -> Optional[Real]:
    el = it.els[0]
    iw0, ih0 = max(1e-3, it.src_box.w), max(1e-3, it.src_box.h)
    iw, ih = iw0 * s * k, ih0 * s * k
    fx0, fy0, fx1, fy1 = it.ink
    fw_, fh_ = max(1e-6, fx1 - fx0), max(1e-6, fy1 - fy0)
    # the visible part must fit the slot, and the whole box (transparent margin
    # included) must still fit on the canvas
    f = min(1.0, max_w / iw if iw > 0 else 1.0, max_h / ih if ih > 0 else 1.0,
            (ctx.W - 2.0) * fw_ / iw if iw > 0 else 1.0, (ctx.H - 2.0) * fh_ / ih if ih > 0 else 1.0)
    iw, ih = iw * f, ih * f
    if max(iw, ih) < 2:  # never vanish entirely (thin rules may stay thin)
        g = 2.0 / max(1e-6, max(iw, ih))
        if iw * g > max_w + 0.5 or ih * g > max_h + 0.5:
            return None
        iw, ih = iw * g, ih * g
        f *= g
    bw, bh = iw / fw_, ih / fh_
    rot = it.rotation
    if rot and el.get("type") == "image":
        # footprint is the rotated AABB; ink == box for rotated art
        bw, bh = el_box(el).w * (iw / iw0), el_box(el).h * (ih / ih0)

    def emit(x: float, y: float) -> List[dict]:
        e = _copy(el)
        if rot:
            cx, cy = x + iw / 2.0, y + ih / 2.0
            e.update(x=cx - bw / 2.0, y=cy - bh / 2.0, width=bw, height=bh)
        else:
            e.update(x=x - fx0 * bw, y=y - fy0 * bh, width=bw, height=bh)
        if e.get("type") == "shape" and e.get("corner_radius"):
            e["corner_radius"] = float(e["corner_radius"]) * (bw / max(1e-6, el_box(el).w))
        return [e]

    return Real(item=it, w=iw, h=ih, scale=s * k * f, emit_fn=emit)


# ---------------------------------------------------------------------------
# groups

def _label_block(lab: dict, fs: int, max_w: float) -> TextBlock:
    style = TextStyle.of(lab)
    content = str(lab.get("content", ""))
    nat = textfit.measure(content, style, fs, 1e9)
    if nat.ink_width + 1 <= max_w:
        return textfit.measure(content, style, fs, math.ceil(nat.ink_width + 1.0))
    return textfit.balanced(content, style, fs, max_w)


def realize_group(it: Item, s: float, ctx: Ctx, max_w: float = INF, max_h: float = INF,
                  relax: float = 1.0, nowrap: bool = False) -> Optional[Real]:
    if it.container is not None and it.centered_label and len(it.labels) == 1:
        return _realize_button(it, s, ctx, max_w, max_h, relax, nowrap)
    r = _realize_generic(it, s, ctx, max_w, max_h, relax)
    if r is None and it.lockup and _stacked(it):
        r = _realize_lockup_line(it, s, ctx, max_w, max_h, relax)
    return r


def _stacked(it: Item) -> bool:
    """A lockup of different lines (not one line set twice, as in a drop
    shadow), so its parts can also be read side by side."""
    lines = [" ".join(str(e.get("content", "")).split()).lower() for e in it.els]
    return len(set(lines)) == len(lines)


def _reading_order(texts: List[dict]) -> List[dict]:
    """Parts of a lockup in reading order: left to right where they sit side by
    side (a figure and its cents), top to bottom where they are stacked."""
    import functools

    boxes = {e["id"]: textfit.drawn_box(e) for e in texts}

    def before(a: dict, b: dict) -> int:
        A, B = boxes[a["id"]], boxes[b["id"]]
        apart = max(B.x - A.r, A.x - B.r) > -0.1 * min(A.w, B.w)
        level = A.y <= B.cy <= A.b or B.y <= A.cy <= B.b
        if apart and level:
            return -1 if A.cx < B.cx else 1
        return -1 if A.cy < B.cy else (1 if A.cy > B.cy else 0)

    return sorted(texts, key=functools.cmp_to_key(before))


def _realize_lockup_line(it: Item, s: float, ctx: Ctx, max_w: float, max_h: float, relax: float) -> Optional[Real]:
    """A stacked lockup ("UP TO / 50% / OFF") that cannot fit stacked, set on
    one line in reading order, centred on a common middle, sizes in proportion."""
    parts = _reading_order(it.els)
    fs0 = [float(e.get("font_size", 16) or 16) for e in parts]
    glyphs = [textfit.drawn_box(e) for e in parts]
    # parts already side by side keep their own spacing (a figure and its cents); the rest get a word space
    beside = [min(a.b, b.b) - max(a.y, b.y) > 0 and b.x >= a.x for a, b in zip(glyphs, glyphs[1:])]
    g = max(s, ctx.judge_floor / min(fs0))
    for _ in range(8):
        blocks = [textfit.measure(str(e.get("content", "")), TextStyle.of(e), max(int(f * g), ctx.judge_floor),
                                  1e9, exact=True) for e, f in zip(parts, fs0)]
        word = 0.3 * min(b.font_size for b in blocks)
        gaps = [max(0.0, (b.x - a.r) * g) if side else word for a, b, side in zip(glyphs, glyphs[1:], beside)]
        w = sum(b.ink_width + 1.0 for b in blocks) + sum(gaps)
        h = max(b.height for b in blocks)
        if w <= max_w + 0.5 and h <= max_h + 0.5:
            break
        if max(fs0) * g < ctx.floor("body", relax):
            return None
        g *= 0.99 * min(max_w / w, max_h / h)
    else:
        return None
    fonts = {e["id"]: b.font_size for e, b in zip(parts, blocks)}

    def emit(x: float, y: float) -> List[dict]:
        out, cx = [], x
        for i, (e, b) in enumerate(zip(parts, blocks)):
            o = _copy(e)
            o.update(x=cx, y=y + (h - b.height) / 2.0, width=b.ink_width + 1.0, height=b.height,
                     font_size=b.font_size)
            out.append(o)
            cx += b.ink_width + 1.0 + (gaps[i] if i < len(gaps) else 0.0)
        return out

    return Real(item=it, w=w, h=h, scale=g, fonts=fonts, blocks={e["id"]: b for e, b in zip(parts, blocks)},
                emit_fn=emit)


def _realize_button(it: Item, s: float, ctx: Ctx, max_w: float, max_h: float, relax: float,
                    nowrap: bool = False) -> Optional[Real]:
    shape = it.container
    lab = it.labels[0]
    cb = el_box(shape)
    fs0 = float(lab.get("font_size", 16) or 16)
    fs = font_for(fs0, s, it.role, ctx, relax)
    g = fs / fs0
    style = TextStyle.of(lab)
    content = str(lab.get("content", ""))
    src_blk = textfit.measure(content, style, fs0, 1e9)
    # Padding in ems keeps the source's proportions; a label held up at the
    # legibility floor (q times larger than the layout scale) gets a tighter
    # pill, as a designer sets a button on a small format, down to a compact
    # 0.8 em a side and 2.6 em tall.
    q = max(1.0, g / max(s, 1e-6))
    pad_em = max(0.8, max(0.0, (cb.w - src_blk.ink_width) / 2.0) / fs0 / q)
    h_em = max(2.6, cb.h / fs0 / q)
    pad_x = max(pad_em * fs, 0.55 * fs)
    rot = it.rotation
    cos_t, sin_t = abs(math.cos(math.radians(rot))), abs(math.sin(math.radians(rot)))

    inner_max = max_w
    if rot:
        inner_max = max_w / max(0.2, cos_t)
    min_pad = 0.5 * fs
    blk = _label_block(lab, fs, max(fs * 2.0, inner_max - 2 * min_pad))
    if nowrap and blk.n_lines > 1:
        return None
    w = max(cb.w * g / q, blk.ink_width + 2 * pad_x)
    if w > inner_max:  # tight canvas: give up padding before anything else
        w = max(blk.ink_width + 2 * min_pad, inner_max)
    h = max(min(cb.h * g, h_em * fs), blk.height + 0.45 * fs)
    if blk.n_lines > 1:
        h = max(h, blk.height + 1.0 * fs)
    h_cap = max_h / max(0.2, cos_t) if rot else max_h
    if h > h_cap:  # short canvas: a slimmer button, never tighter than the text
        h = max(blk.height + 0.3 * fs, h_cap)
    fw, fh = (w, h) if not rot else (w * cos_t + h * sin_t, w * sin_t + h * cos_t)
    if fw > max_w + 0.5 or fh > max_h + 0.5:
        return None
    r0 = float(shape.get("corner_radius", 0) or 0)
    radius = h / 2.0 if r0 >= 0.45 * cb.h else min(h / 2.0, r0 * g)
    stroke = shape.get("stroke_width")

    def emit(x: float, y: float) -> List[dict]:
        cx, cy = x + fw / 2.0, y + fh / 2.0
        sh = _copy(shape)
        sh.update(x=cx - w / 2.0, y=cy - h / 2.0, width=w, height=h)
        if "corner_radius" in shape or r0:
            sh["corner_radius"] = radius
        if stroke:
            sh["stroke_width"] = max(1.0, float(stroke) * g)
        lb = _copy(lab)
        lw = float(blk.wrap_width)
        # optical centre: middle of the cap height of the text block
        vis_mid = (blk.ascent - blk.cap + blk.last_baseline()) / 2.0
        dx, dy = 0.0, (blk.height / 2.0) - vis_mid  # label-centre offset from shape centre (local frame)
        if rot:
            t = math.radians(rot)
            dx, dy = dx * math.cos(t) - dy * math.sin(t), dx * math.sin(t) + dy * math.cos(t)
        lcx, lcy = cx + dx, cy + dy
        lb.update(x=lcx - lw / 2.0, y=lcy - blk.height / 2.0, width=lw, height=blk.height, font_size=blk.font_size)
        out = [sh, lb]
        # any extra members (icons...) scale with the group around the shape centre
        for m in it.els:
            if m is shape or m is lab:
                continue
            out.append(_scaled_member(m, cb, g, cx - w / 2.0, cy - h / 2.0, ctx, it.role, relax))
        return out

    return Real(item=it, w=fw, h=fh, scale=g, fonts={lab["id"]: blk.font_size}, blocks={lab["id"]: blk},
                emit_fn=emit)


def _scaled_member(m: dict, origin: Box, g: float, ox: float, oy: float, ctx: Ctx, role: str, relax: float) -> dict:
    b = el_box(m)
    e = _copy(m)
    e.update(x=ox + (b.x - origin.x) * g, y=oy + (b.y - origin.y) * g, width=b.w * g, height=b.h * g)
    if m.get("type") == "text":
        fs0 = float(m.get("font_size", 16) or 16)
        e["font_size"] = max(1, int(round(fs0 * g)))
    if m.get("type") == "shape" and m.get("corner_radius"):
        e["corner_radius"] = float(m["corner_radius"]) * g
    return e


def _overlap(a: Box, b: Box) -> Tuple[float, float]:
    return min(a.r, b.r) - max(a.x, b.x), min(a.b, b.b) - max(a.y, b.y)


def _separate(boxes: List[Box], allowed: Optional[Dict[Tuple[int, int], Tuple[float, float]]] = None) -> List[Box]:
    """Move text boxes apart, in reading order, until no pair overlaps by more
    than ``allowed[(i, j)]`` = (x, y) overlap it had in the source (nothing, for
    pairs that did not touch). Each excess goes along the axis that needs the
    smaller move: lines stacked too tight move down, a figure and its cents
    move apart sideways."""
    allowed = allowed or {}
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i].y, boxes[i].x))
    out = list(boxes)
    done: List[int] = []
    for i in order:
        b = out[i].copy()
        for _ in range(2 * len(done) + 1):
            moved = False
            for j in done:
                ox, oy = _overlap(b, out[j])
                ax, ay = allowed.get((min(i, j), max(i, j)), (0.0, 0.0))
                ex, ey = ox - ax, oy - ay
                if ox > 0.5 and oy > 0.5 and ex > 0.5 and ey > 0.5:
                    if ex < ey and b.cx > out[j].cx:
                        b.x += ex + 0.5
                    else:
                        b.y += ey + 0.5
                    moved = True
            if not moved:
                break
        out[i] = b
        done.append(i)
    return out


def _realize_generic(it: Item, s: float, ctx: Ctx, max_w: float, max_h: float, relax: float) -> Optional[Real]:
    """Scale the group's internal arrangement uniformly. Every line stays at or
    above the legibility floor and keeps its line breaks and alignment edge;
    lines that grow push later lines down so copy never runs into copy it did
    not overlap in the source, and the footprint grows to hold them.

    A lockup (several texts set as one piece of type) keeps its proportions:
    it is never set so small that its smallest part needs the floor."""
    src = it.src_box
    texts = [m for m in it.els if m.get("type") == "text"]
    fs0 = {m["id"]: float(m.get("font_size", 16) or 16) for m in texts}
    if "glyphs" not in it.memo:  # where each text's glyphs sit in the source, and how they overlap
        g0 = {m["id"]: textfit.drawn_box(m) for m in texts}
        it.memo["glyphs"] = g0
        it.memo["overlap"] = {(i, j): _overlap(g0[a["id"]], g0[b["id"]])
                              for i, a in enumerate(texts) for j, b in enumerate(texts) if i < j}
    glyphs0, overlap0 = it.memo["glyphs"], it.memo["overlap"]
    g = s
    if texts:  # the biggest line on the group must stay legible
        big = max(fs0.values())
        g = max(g, font_for(big, s, "body" if it.role == "panel" else it.role, ctx, relax) / big)
        if it.lockup:
            g = max(g, ctx.judge_floor / min(fs0.values()))

    def layout(g: float):
        local, fonts, blocks = {}, {}, {}
        for m in it.els:
            b = el_box(m)
            local[m["id"]] = Box((b.x - src.x) * g, (b.y - src.y) * g, b.w * g, b.h * g)
        glyphs, starts = {}, {}
        for m in texts:
            content, style = str(m.get("content", "")), TextStyle.of(m)
            fs = max(int(fs0[m["id"]] * g), ctx.judge_floor)  # rounded down, but never small print
            w = textfit.width_keeping_breaks(content, style, fs0[m["id"]], max(1.0, el_box(m).w), fs)
            w = textfit.tight_width(content, style, fs, w)
            blk = textfit.measure(content, style, fs, w, exact=True)
            gb = glyphs0[m["id"]]
            a = m.get("text_align", "left")
            starts[m["id"]] = textfit.glyph_span(blk, a)[0]
            rot = float(m.get("rotation", 0) or 0)
            if rot:  # turned copy keeps its centre; its glyphs turn about the box centre
                b0 = el_box(m)
                cx, cy = (b0.cx - src.x) * g, (b0.cy - src.y) * g
                local[m["id"]] = Box(cx - w / 2.0, cy - blk.height / 2.0, w, blk.height)
                glyphs[m["id"]] = rotate_about(Box(cx - w / 2.0 + starts[m["id"]], cy - blk.height / 2.0,
                                                   blk.ink_width, blk.height), rot, cx, cy)
            else:  # the glyphs keep their alignment edge
                edge = gb.x if a == "left" else (gb.r if a == "right" else gb.cx)
                gx = (edge - src.x) * g - (0.0 if a == "left" else (blk.ink_width if a == "right" else blk.ink_width / 2.0))
                glyphs[m["id"]] = Box(gx, (gb.y - src.y) * g, blk.ink_width, blk.height)
                local[m["id"]] = Box(gx - starts[m["id"]], (gb.y - src.y) * g, w, blk.height)
            fonts[m["id"]] = fs
            blocks[m["id"]] = blk
        if texts:  # copy may overlap only as much as it did in the source
            allowed = {k: (ox * g, oy * g) for k, (ox, oy) in overlap0.items() if ox > 0 and oy > 0}
            sep = _separate([glyphs[m["id"]] for m in texts], allowed)
            for m, b in zip(texts, sep):  # move each element box with its glyphs
                dx, dy = b.x - glyphs[m["id"]].x, b.y - glyphs[m["id"]].y
                glyphs[m["id"]] = b
                local[m["id"]] = local[m["id"]].moved(dx, dy)
        u = union_all(list(local.values()) + list(glyphs.values()) +
                      [text_footprint(local[m["id"]], blocks[m["id"]], m.get("text_align", "left"),
                                      float(m.get("rotation", 0) or 0)) for m in texts])
        return local, fonts, blocks, u

    local, fonts, blocks, u = layout(g)
    for _ in range(6):  # too big: shrink (copy held at the floor shrinks less, so repeat)
        if u.w <= max_w + 0.5 and u.h <= max_h + 0.5:
            break
        g *= 0.99 * min(max_w / max(1e-6, u.w), max_h / max(1e-6, u.h))
        if texts:
            big = max(fs0.values())
            if big * g < ctx.floor("body", relax) * 0.999 and big * g < big:
                return None
        local, fonts, blocks, u = layout(g)
    else:
        return None

    def emit(x: float, y: float) -> List[dict]:
        out = []
        for m in it.els:
            b = local[m["id"]]
            e = _copy(m)
            e.update(x=x + b.x - u.x, y=y + b.y - u.y, width=b.w, height=b.h)
            if m.get("type") == "text":
                e["font_size"] = fonts[m["id"]]
            if m.get("type") == "shape" and m.get("corner_radius"):
                e["corner_radius"] = float(m["corner_radius"]) * g
            out.append(e)
        return out

    return Real(item=it, w=u.w, h=u.h, scale=g, fonts=fonts, blocks=blocks, emit_fn=emit)


def _composite_at(it: Item, t: float, ctx: Ctx, relax: float, sticker_cap: float = INF):
    host = it.host
    rh = realize_image(host, t, ctx)
    if rh is None:
        return None
    hb = host.src_box
    k = rh.w / max(1e-6, hb.w)  # px of output per px of source
    parts = [(rh, 0.0, 0.0)]
    dropped: List[dict] = []
    for sat in it.sats:
        if sat.kind == "text":
            # a secondary caption that would have to grow past twice its art's scale to stay
            # legible no longer belongs on the art: it is left out rather than blown up
            if ctx.judge_floor > CAPTION_GROWTH * sat.fs0 * k and sat.role not in ("headline", "cta") and not sat.core:
                dropped += sat.els
                continue
            # copy on art keeps its place and measure, so captions do not collide
            rs, ox, oy = _scaled_text(sat, k, ctx)
            eb = el_box(sat.els[0])
            parts.append((rs, (eb.x - hb.x) * k + ox, (eb.y - hb.y) * k + oy))
            continue
        rs = realize(sat, t if sat.kind == "image" else min(t, sticker_cap), ctx, relax=relax)
        if rs is None:
            return None
        fx = (sat.src_box.cx - hb.x) / max(1e-6, hb.w)
        fy = (sat.src_box.cy - hb.y) / max(1e-6, hb.h)
        parts.append((rs, fx * rh.w - rs.w / 2.0, fy * rh.h - rs.h / 2.0))
    tix = [i for i, (r, _, _) in enumerate(parts) if r.item.is_copy]
    if len(tix) > 1:  # captions that grew to the floor must not run into each other (any more than designed)
        src = [parts[i][0].item.src_box for i in tix]
        allowed = {}
        for a in range(len(tix)):
            for b in range(a + 1, len(tix)):
                ox, oy = _overlap(src[a], src[b])
                if ox > 0 and oy > 0:
                    allowed[(a, b)] = (ox * k, oy * k)
        sep = _separate([Box(parts[i][1], parts[i][2], parts[i][0].w, parts[i][0].h) for i in tix], allowed)
        for i, b in zip(tix, sep):
            parts[i] = (parts[i][0], b.x, b.y)
    x0 = min(px for _, px, _ in parts)
    y0 = min(py for _, _, py in parts)
    x1 = max(px + r.w for r, px, _ in parts)
    y1 = max(py + r.h for r, _, py in parts)
    return parts, x0, y0, x1 - x0, y1 - y0, rh, dropped


def _scaled_text(it: Item, k: float, ctx: Ctx) -> Tuple[Real, float, float]:
    """A caption scaled with the art it sits on: same place, same line breaks,
    font rounded down but never below small-print legibility. Returns the
    realized caption and where its footprint starts relative to the scaled
    source box (glyphs can reach past the box: letter spacing, rotation)."""
    el = it.els[0]
    content, style = str(el.get("content", "")), it.style or TextStyle.of(el)
    fs0 = max(1.0, it.fs0)
    fs = max(int(fs0 * k), ctx.judge_floor)
    eb = el_box(el)
    width = textfit.width_keeping_breaks(content, style, fs0, max(1.0, eb.w), fs)
    blk = textfit.measure(content, style, fs, width, exact=True)
    rot = it.rotation
    # the box keeps its scaled top-left, or its centre if it is turned (the renderer's pivot)
    bx, by = ((eb.w * k - width) / 2.0, (eb.h * k - blk.height) / 2.0) if rot else (0.0, 0.0)
    fp = text_footprint(Box(bx, by, width, blk.height), blk, el.get("text_align", "left"), rot)

    def emit(x: float, y: float) -> List[dict]:
        e = _copy(el)
        e.update(x=x + bx - fp.x, y=y + by - fp.y, width=width, height=blk.height, font_size=fs)
        return [e]

    real = Real(item=it, w=fp.w, h=fp.h, scale=fs / fs0, fonts={el["id"]: fs}, blocks={el["id"]: blk}, emit_fn=emit)
    return real, fp.x, fp.y


def realize_composite(it: Item, s: float, ctx: Ctx, max_w: float = INF, max_h: float = INF,
                      relax: float = 1.0, k: float = 1.0) -> Optional[Real]:
    """An image with things stuck to it (or a row of shots): scaled as one piece.
    A sticker keeps to the design's scale ``s`` when the art grows past it to
    fill its slot, so a badge never outgrows the copy around it."""
    def fits(res):
        return res is not None and res[3] <= max_w + 0.5 and res[4] <= max_h + 0.5 and \
            res[3] <= ctx.W - 2 and res[4] <= ctx.H - 2

    t = s * k
    res = _composite_at(it, t, ctx, relax, s)
    if not fits(res):
        lo, hi, best = 0.0, t, None
        for _ in range(18):
            mid = (lo + hi) / 2.0
            r = _composite_at(it, mid, ctx, relax, s) if mid > 0 else None
            if fits(r):
                lo, best = mid, r
            else:
                hi = mid
        res = best
    if res is None:
        return None
    parts, x0, y0, w, h, rh, dropped = res
    fonts, blocks = {}, {}
    for r, _, _ in parts:
        fonts.update(r.fonts)
        blocks.update(r.blocks)

    def emit(x: float, y: float) -> List[dict]:
        out = []
        for r, px, py in parts:
            out += r.emit(x + px - x0, y + py - y0)
        return out

    return Real(item=it, w=w, h=h, scale=rh.scale, fonts=fonts, blocks=blocks, emit_fn=emit, dropped=dropped)


def _align_shift(e: dict, w_scaled: float, w: float) -> float:
    """Where a box of width ``w`` starts so its text keeps the alignment edge
    of the scaled box."""
    a = e.get("text_align", "left")
    return 0.0 if a == "left" else (w_scaled - w if a == "right" else (w_scaled - w) / 2.0)


def realize_scaled(it: Item, k: float, ctx: Ctx) -> Real:
    """The item exactly as designed, every element scaled by ``k``: same
    arrangement, same line breaks. Only copy that would fall below the
    legibility floor is set larger (and re-measured at its scaled width)."""
    src = it.src_box
    fonts, blocks, boxes = {}, {}, {}
    for e in it.els:
        b = el_box(e)
        box = Box((b.x - src.x) * k, (b.y - src.y) * k, b.w * k, b.h * k)
        if e.get("type") == "text":
            content, style, fs0 = str(e.get("content", "")), TextStyle.of(e), float(e.get("font_size", 16) or 16)
            fs = max(int(round(fs0 * k)), ctx.judge_floor)
            box.w = textfit.width_keeping_breaks(content, style, fs0, max(1.0, b.w), fs)
            if box.w > b.w * k + 0.5:  # set larger than scale: the box grows only as far as its glyphs need
                box.w = max(b.w * k, textfit.tight_width(content, style, fs, box.w))
            box.x = (b.x - src.x) * k + _align_shift(e, b.w * k, box.w)
            blk = textfit.measure(content, style, fs, box.w, exact=True)
            box.h = blk.height
            fonts[e["id"]], blocks[e["id"]] = fs, blk
        boxes[e["id"]] = box

    def emit(x: float, y: float) -> List[dict]:
        out = []
        for e in it.els:
            b = boxes[e["id"]]
            o = _copy(e)
            o.update(x=x + b.x, y=y + b.y, width=b.w, height=b.h)
            if e["id"] in fonts:
                o["font_size"] = fonts[e["id"]]
            for key in ("corner_radius", "stroke_width"):
                if o.get(key):
                    o[key] = float(o[key]) * k
            out.append(o)
        return out

    return Real(item=it, w=src.w * k, h=src.h * k, scale=k, fonts=fonts, blocks=blocks, emit_fn=emit)


DISPLAY_LINES = 3  # display type sets a short headline in at most this many lines


def _display(it: Item, s: float, ctx: Ctx, max_w: float, max_h: float, relax: float, mode: str) -> Optional[Real]:
    """The headline as display type: up to ``ctx.emphasis`` times the design's
    scale, never past the canvas's display size, and stepped down until it
    takes no more lines than display type should: about two words a line, at
    most DISPLAY_LINES (or its source's count). A longer headline may only
    grow into the lines it already takes at ``s``. None if even a small boost
    does not fit; the caller then sets it at ``s``."""

    def at(t: float) -> Optional[Real]:
        return realize_text(it, t, ctx, max_w, max_h, mode=mode, relax=relax) if it.kind == "text" else \
            realize_group(it, t, ctx, max_w, max_h, relax=relax)

    words = len(" ".join(str(e.get("content", "")) for e in it.els if e.get("type") == "text").split())
    lines = max(it.src_lines, min(DISPLAY_LINES, words, max(2, math.ceil(words / 2))))
    if words > 2 * DISPLAY_LINES:
        plain = at(s)
        if plain is None:
            return None
        lines = max(lines, max(b.n_lines for b in plain.blocks.values()))
    top = max(s, min(s * ctx.emphasis, ctx.display_max / max(1.0, it.main_fs())))
    for f in (1.0, 0.8, 0.6, 0.4, 0.2):
        r = at(s + (top - s) * f)
        if r is not None and all(b.n_lines <= lines for b in r.blocks.values()):
            return r
    return None


_REALIZED: Dict[tuple, Optional[Real]] = {}


def forget_realized() -> None:
    """Drop the memo of realized items (call whenever a design is analysed afresh)."""
    _REALIZED.clear()


def realize(it: Item, s: float, ctx: Ctx, max_w: float = INF, max_h: float = INF,
            relax: float = 1.0, k: float = 1.0, mode: str = "stack", nowrap: bool = False) -> Optional[Real]:
    """``it`` sized at scale ``s`` within max_w x max_h, or None. Memoised: the
    candidates of one design ask for the same items at the same sizes many
    times (a Real is never changed once made). Emphasis only reaches the
    headline and the measure only plain text, so only they key on it."""
    key = (id(it), s, max_w, max_h, relax, k, mode, nowrap, ctx.W, ctx.H, ctx.W0, ctx.H0,
           ctx.emphasis if it.role == "headline" and it.is_copy else 1.0,
           ctx.measure if it.kind == "text" else 1.0)
    if key not in _REALIZED:
        _REALIZED[key] = _realize(it, s, ctx, max_w, max_h, relax, k, mode, nowrap)
    return _REALIZED[key]


def _realize(it: Item, s: float, ctx: Ctx, max_w: float, max_h: float, relax: float, k: float, mode: str,
             nowrap: bool) -> Optional[Real]:
    if ctx.emphasis != 1.0 and it.role == "headline" and it.is_copy:
        r = _display(it, s, ctx, max_w, max_h, relax, mode)
        if r is not None:
            return r
    if it.kind == "composite":
        return realize_composite(it, s, ctx, max_w, max_h, relax=relax, k=k)
    if it.kind == "text":
        return realize_text(it, s, ctx, max_w, max_h, mode=mode, relax=relax)
    if it.kind == "image":
        if it.role == "logo":
            # a logo smaller than this stops reading as a logo
            min_h = 3.2 * ctx.base_fs
            s = max(s, min_h / max(1e-6, it.src_box.h * k))
        return realize_image(it, s, ctx, max_w, max_h, k=k)
    return realize_group(it, s, ctx, max_w, max_h, relax=relax, nowrap=nowrap)
