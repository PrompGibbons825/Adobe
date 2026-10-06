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
from .constraints import min_font as _min_font
from .analyze import Item
from .geom import Box, el_box, rotated_aabb, union_all
from .textfit import TextBlock, TextStyle

INF = float("inf")

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

    @property
    def base_fs(self) -> float:
        """Smallest comfortably readable body size on this canvas, in px."""
        return max(10.0, 0.0115 * max(self.W, self.H), 0.016 * min(self.W, self.H))

    @property
    def judge_floor(self) -> int:
        """No visible text below this: cap height >= 7 px when the judge sees the
        canvas at 1024 px on the long side (research 04), and never under 10 px."""
        return max(10, math.ceil(10.0 * max(self.W, self.H) / 1024.0), _min_font())  # finale-api: min_font_px

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

    def emit(self, x: float, y: float) -> List[dict]:
        return self.emit_fn(x, y) if self.emit_fn else []

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
    if max_w < fs:
        return None
    if mode == "fit":
        blk = textfit.fit_height(content, style, fs, max_h, max_w)
        if blk is None:
            return None
    else:
        pref = it.measure_em * fs * 1.06
        word = textfit.longest_word(content, style, fs)
        lim = min(max_w, max(pref, word + 2))
        blk = textfit.balanced(content, style, fs, lim)
        if blk.height > max_h + 0.5 and lim < max_w:  # the source measure is a preference, not a wall
            blk = textfit.balanced(content, style, fs, max_w)
    if blk.ink_width > max_w + 0.5 or blk.height > max_h + 0.5:
        return None
    max_lines = el.get("max_lines")
    if isinstance(max_lines, int) and max_lines > 0 and blk.n_lines > max_lines:
        return None

    exact = textfit.measure(content, style, blk.font_size, blk.wrap_width, exact=True)
    if exact.lines != blk.lines:  # never trust the fast path over the renderer's own rule
        blk = exact
        if blk.height > max_h + 0.5 or blk.ink_width > max_w + 0.5:
            return None
    w, h = float(blk.wrap_width), float(blk.height)

    def emit(x: float, y: float) -> List[dict]:
        e = _copy(el)
        e.update(x=x, y=y, width=w, height=h, font_size=blk.font_size)
        return [e]

    return Real(item=it, w=w, h=h, scale=fs / max(1.0, it.fs0), fonts={el["id"]: blk.font_size},
                blocks={el["id"]: blk}, emit_fn=emit)


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
    return _realize_generic(it, s, ctx, max_w, max_h, relax)


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
    pad_x0 = max(0.0, (cb.w - src_blk.ink_width) / 2.0)
    pad_x = max(pad_x0 * g, 0.55 * fs)
    rot = it.rotation
    cos_t, sin_t = abs(math.cos(math.radians(rot))), abs(math.sin(math.radians(rot)))

    inner_max = max_w
    if rot:
        inner_max = max_w / max(0.2, cos_t)
    min_pad = 0.5 * fs
    blk = _label_block(lab, fs, max(fs * 2.0, inner_max - 2 * min_pad))
    if nowrap and blk.n_lines > 1:
        return None
    w = max(cb.w * g, blk.ink_width + 2 * pad_x)
    if w > inner_max:  # tight canvas: give up padding before anything else
        w = max(blk.ink_width + 2 * min_pad, inner_max)
    h = max(cb.h * g, blk.height + 0.45 * fs)
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


def _separate(boxes: List[Box]) -> List[Box]:
    """Push text boxes down (in reading order) until none overlaps another."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i].y, boxes[i].x))
    done: List[Box] = []
    out = list(boxes)
    for i in order:
        b = out[i].copy()
        for _ in range(len(done) + 1):
            hit = [d for d in done if min(b.r, d.r) - max(b.x, d.x) > 0.5 and min(b.b, d.b) - max(b.y, d.y) > 0.5]
            if not hit:
                break
            b.y = max(d.b for d in hit) + 0.5
        out[i] = b
        done.append(b)
    return out


def _realize_generic(it: Item, s: float, ctx: Ctx, max_w: float, max_h: float, relax: float) -> Optional[Real]:
    """Scale the group's internal arrangement uniformly. Every line stays at or
    above the legibility floor; lines that grow push later lines down so copy
    never overlaps copy, and the group's footprint grows to hold them."""
    src = it.src_box
    g = s
    if it.labels:  # the biggest line on the group must stay legible
        lab = max(it.labels, key=lambda l: float(l.get("font_size", 16) or 16))
        fs0 = float(lab.get("font_size", 16) or 16)
        g = max(g, font_for(fs0, s, "body" if it.role == "panel" else it.role, ctx, relax) / fs0)

    def layout(g: float):
        local, fonts, blocks = {}, {}, {}
        for m in it.els:
            b = el_box(m)
            local[m["id"]] = Box((b.x - src.x) * g, (b.y - src.y) * g, b.w * g, b.h * g)
        texts = [m for m in it.els if m.get("type") == "text"]
        for m in texts:
            fs0 = float(m.get("font_size", 16) or 16)
            fs = max(int(fs0 * g), ctx.judge_floor)  # rounded down, but never small print
            lb = local[m["id"]]
            w = lb.w * fs / max(1e-6, fs0 * g)
            blk = textfit.measure(str(m.get("content", "")), TextStyle.of(m), fs, w, exact=True)
            if blk.ink_width > w:
                w = blk.ink_width + 1.0
                blk = textfit.measure(str(m.get("content", "")), TextStyle.of(m), fs, w, exact=True)
            a = m.get("text_align", "left")
            x = lb.x if a == "left" else (lb.r - w if a == "right" else lb.cx - w / 2.0)
            local[m["id"]] = Box(x, lb.y, w, blk.height)
            fonts[m["id"]] = fs
            blocks[m["id"]] = blk
        if texts:
            sep = _separate([local[m["id"]] for m in texts])
            for m, b in zip(texts, sep):
                local[m["id"]] = b
        u = union_all(local.values())
        return local, fonts, blocks, u

    local, fonts, blocks, u = layout(g)
    if u.w > max_w + 0.5 or u.h > max_h + 0.5:
        f = min(max_w / max(1e-6, u.w), max_h / max(1e-6, u.h))
        g *= f
        if it.labels:
            fs0 = max(float(l.get("font_size", 16) or 16) for l in it.labels)
            if fs0 * g < ctx.floor("body", relax) * 0.999 and fs0 * g < fs0:
                return None
        local, fonts, blocks, u = layout(g)
        if u.w > max_w + 0.5 or u.h > max_h + 0.5:
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


def _composite_at(it: Item, t: float, ctx: Ctx, relax: float):
    host = it.host
    rh = realize_image(host, t, ctx)
    if rh is None:
        return None
    hb = host.src_box
    k = rh.w / max(1e-6, hb.w)  # px of output per px of source
    parts = [(rh, 0.0, 0.0)]
    for sat in it.sats:
        if sat.kind == "text":
            # copy on art keeps its place and measure, so captions do not collide
            rs = _scaled_text(sat, k, ctx)
            eb = el_box(sat.els[0])
            parts.append((rs, (eb.x - hb.x) * k, (eb.y - hb.y) * k))
            continue
        rs = realize(sat, t, ctx, relax=relax)
        if rs is None:
            return None
        fx = (sat.src_box.cx - hb.x) / max(1e-6, hb.w)
        fy = (sat.src_box.cy - hb.y) / max(1e-6, hb.h)
        parts.append((rs, fx * rh.w - rs.w / 2.0, fy * rh.h - rs.h / 2.0))
    tix = [i for i, (r, _, _) in enumerate(parts) if r.item.kind == "text"]
    if len(tix) > 1:  # captions that grew to the floor must not run into each other
        sep = _separate([Box(parts[i][1], parts[i][2], parts[i][0].w, parts[i][0].h) for i in tix])
        for i, b in zip(tix, sep):
            parts[i] = (parts[i][0], b.x, b.y)
    x0 = min(px for _, px, _ in parts)
    y0 = min(py for _, _, py in parts)
    x1 = max(px + r.w for r, px, _ in parts)
    y1 = max(py + r.h for r, _, py in parts)
    return parts, x0, y0, x1 - x0, y1 - y0, rh


def _scaled_text(it: Item, k: float, ctx: Ctx) -> Real:
    """A caption scaled with the art it sits on: same place, same measure (in
    ems), font rounded down but never below small-print legibility."""
    el = it.els[0]
    style = it.style or TextStyle.of(el)
    fs0 = max(1.0, it.fs0)
    fs = max(int(fs0 * k), ctx.judge_floor)
    eb = el_box(el)
    width = max(1.0, eb.w * fs / fs0)
    blk = textfit.measure(str(el.get("content", "")), style, fs, width, exact=True)
    width = max(width, blk.ink_width + 1.0)
    blk = textfit.measure(str(el.get("content", "")), style, fs, width, exact=True)

    def emit(x: float, y: float) -> List[dict]:
        e = _copy(el)
        e.update(x=x, y=y, width=width, height=blk.height, font_size=fs)
        return [e]

    return Real(item=it, w=width, h=blk.height, scale=fs / fs0, fonts={el["id"]: fs},
                blocks={el["id"]: blk}, emit_fn=emit)


def realize_composite(it: Item, s: float, ctx: Ctx, max_w: float = INF, max_h: float = INF,
                      relax: float = 1.0, k: float = 1.0) -> Optional[Real]:
    """An image with things stuck to it (or a row of shots): scaled as one piece."""
    def fits(res):
        return res is not None and res[3] <= max_w + 0.5 and res[4] <= max_h + 0.5 and \
            res[3] <= ctx.W - 2 and res[4] <= ctx.H - 2

    t = s * k
    res = _composite_at(it, t, ctx, relax)
    if not fits(res):
        lo, hi, best = 0.0, t, None
        for _ in range(18):
            mid = (lo + hi) / 2.0
            r = _composite_at(it, mid, ctx, relax) if mid > 0 else None
            if fits(r):
                lo, best = mid, r
            else:
                hi = mid
        res = best
    if res is None:
        return None
    parts, x0, y0, w, h, rh = res
    fonts, blocks = {}, {}
    for r, _, _ in parts:
        fonts.update(r.fonts)
        blocks.update(r.blocks)

    def emit(x: float, y: float) -> List[dict]:
        out = []
        for r, px, py in parts:
            out += r.emit(x + px - x0, y + py - y0)
        return out

    return Real(item=it, w=w, h=h, scale=rh.scale, fonts=fonts, blocks=blocks, emit_fn=emit)


def realize(it: Item, s: float, ctx: Ctx, max_w: float = INF, max_h: float = INF,
            relax: float = 1.0, k: float = 1.0, mode: str = "stack", nowrap: bool = False) -> Optional[Real]:
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
