"""Layout blocks: each knows how to size itself at a scale and place its
elements. Sizes are VISUAL sizes (inked text, opaque image pixels), so gaps
and alignment are computed on what the eye sees, not on padded boxes."""

from __future__ import annotations

import math

from . import saliency
from . import textfit as tf


def img_vis_frac(el: dict) -> tuple:
    """Visible-pixel rect of an image element, as fractions of its box."""
    info = saliency.analyze(el.get("asset_id") or "")
    if info is None or el["width"] <= 0 or el["height"] <= 0:
        return (0.0, 0.0, 1.0, 1.0)
    iw, ih = info["size"]
    A, B = iw / ih, el["width"] / el["height"]
    fit = el.get("fit", "contain")
    if fit == "contain":
        fw, fh = (1.0, B / A) if A > B else (A / B, 1.0)
        fx0, fy0 = (1 - fw) / 2, (1 - fh) / 2
    elif fit == "cover":
        return (0.0, 0.0, 1.0, 1.0)
    else:
        fx0, fy0, fw, fh = 0.0, 0.0, 1.0, 1.0
    ax0, ay0, ax1, ay1 = info["alpha_box"]
    return (fx0 + ax0 * fw, fy0 + ay0 * fh, fx0 + ax1 * fw, fy0 + ay1 * fh)


def rot_bounds(x, y, w, h, deg):
    """Axis-aligned bounds of a box rotated about its center."""
    a = math.radians(deg or 0)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    bw, bh = w * c + h * s, w * s + h * c
    cx, cy = x + w / 2, y + h / 2
    return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


class Placed:
    """Accumulates output geometry for one candidate."""

    def __init__(self):
        self.props: dict[str, dict] = {}   # id -> updated fields
        self.vis: dict[str, tuple] = {}    # id -> visual box (content elements)

    def set(self, eid, **kw):
        self.props.setdefault(eid, {}).update(kw)


# ---------------------------------------------------------------------------

class TextBlock:
    def __init__(self, unit, src_w):
        self.unit = unit
        self.el = unit.els[0]
        self.wfrac = max(0.55, min(1.0, self.el["width"] / (src_w * 0.86)))

    def _fit(self, t, max_w):
        el = self.el
        box_w = max_w * self.wfrac
        fs = max(1, round(el["font_size"] * t))
        m = tf.measure(el, fs, box_w)
        while m["word_w"] > box_w and fs > 1:
            fs -= 1
            m = tf.measure(el, fs, box_w)
        return box_w, fs, m

    def size(self, t, max_w):
        box_w, fs, m = self._fit(t, max_w)
        asc, cap, desc = tf.metrics(self.el)
        lh = float(self.el.get("line_height", 1.2))
        vh = (m["lines"] - 1) * fs * lh + (cap + desc * 0.6) * fs
        return min(box_w, m["width"]), vh

    def place(self, P, x, y_vis, t, max_w, align):
        box_w, fs, m = self._fit(t, max_w)
        asc, cap, desc = tf.metrics(self.el)
        if align == "center":
            bx = x + (max_w - box_w) / 2
        else:
            bx = x
        y = y_vis - (asc - cap) * fs
        lh = float(self.el.get("line_height", 1.2))
        P.set(self.el["id"], x=bx, y=y, width=box_w, height=m["lines"] * fs * lh,
              font_size=fs, text_align=align)
        el2 = dict(self.el, text_align=align)
        P.vis[self.el["id"]] = tf.visual_box(el2, bx, y, box_w, fs, m)
        return fs


class ImageBlock:
    def __init__(self, unit):
        self.unit = unit
        self.el = unit.els[0]
        self.v = img_vis_frac(self.el)
        self.vw = self.el["width"] * (self.v[2] - self.v[0])
        self.vh = self.el["height"] * (self.v[3] - self.v[1])

    def scale_for(self, s, max_w, max_h=None):
        if self.vw * s > max_w:
            s = max_w / self.vw
        if max_h is not None and self.vh * s > max_h:
            s = max_h / self.vh
        return s

    def size(self, s, max_w):
        s = self.scale_for(s, max_w)
        return self.vw * s, self.vh * s

    def place(self, P, x, y_vis, s, max_w, align):
        s = self.scale_for(s, max_w)
        vw = self.vw * s
        vx = x + (max_w - vw) / 2 if align == "center" else x
        self.place_vis(P, vx, y_vis, s)

    def place_vis(self, P, vx, vy, s):
        bw, bh = self.el["width"] * s, self.el["height"] * s
        bx = vx - self.v[0] * bw
        by = vy - self.v[1] * bh
        P.set(self.el["id"], x=bx, y=by, width=bw, height=bh)
        P.vis[self.el["id"]] = (vx, vy, vx + self.vw * s, vy + self.vh * s)


class GroupBlock:
    """Rigid group (buttons, generic groups): scaled uniformly, text refit."""

    def __init__(self, unit):
        self.unit = unit
        x0, y0, x1, y1 = unit.bbox
        self.x0, self.y0, self.w, self.h = x0, y0, x1 - x0, y1 - y0
        self.shape = unit.shape_el()

    def scale_for(self, u, max_w, max_h=None):
        if self.w * u > max_w:
            u = max_w / self.w
        if max_h is not None and self.h * u > max_h:
            u = max_h / self.h
        return u

    def size(self, u, max_w):
        u = self.scale_for(u, max_w)
        return self.w * u, self.h * u

    def place(self, P, x, y, u, max_w, align):
        u = self.scale_for(u, max_w)
        gx = x + (max_w - self.w * u) / 2 if align == "center" else x
        self.place_at(P, gx, y, u)

    def place_at(self, P, gx, gy, u):
        for e in self.unit.els:
            ex = gx + (e["x"] - self.x0) * u
            ey = gy + (e["y"] - self.y0) * u
            ew, eh = e["width"] * u, e["height"] * u
            if e["type"] == "text":
                src_lines = tf.measure(e, int(e["font_size"]), e["width"])["lines"]
                fs0 = max(1, round(e["font_size"] * u))
                avail_h = eh * 1.6
                if self.shape is not None and len(self.unit.els) == 2:
                    avail_h = self.shape["height"] * u * 0.85
                m = tf.fit(e, ew, avail_h, fs0, 1, max_lines=max(1, src_lines)) or tf.measure(e, fs0, ew)
                fs = m["fs"]
                if self.shape is not None and len(self.unit.els) == 2:
                    scy = gy + (self.shape["y"] - self.y0 + self.shape["height"] / 2) * u
                    ey = tf.optical_top(e, fs, m["lines"], scy)
                lh = float(e.get("line_height", 1.2))
                P.set(e["id"], x=ex, y=ey, width=ew, height=max(eh, m["lines"] * fs * lh), font_size=fs)
                P.vis[e["id"]] = rot_bounds(*tf.visual_box(e, ex, ey, ew, fs, m)[:2],
                                            min(ew, m["width"]), m["lines"] * fs * lh,
                                            e.get("rotation", 0))
            else:
                P.set(e["id"], x=ex, y=ey, width=ew, height=eh)
                if e["type"] == "image":
                    v = img_vis_frac(e)
                    P.vis[e["id"]] = rot_bounds(ex + v[0] * ew, ey + v[1] * eh,
                                                (v[2] - v[0]) * ew, (v[3] - v[1]) * eh, e.get("rotation", 0))
                else:
                    P.vis[e["id"]] = rot_bounds(ex, ey, ew, eh, e.get("rotation", 0))

    def bounds_at(self, gx, gy, u):
        boxes = [rot_bounds(gx + (e["x"] - self.x0) * u, gy + (e["y"] - self.y0) * u,
                            e["width"] * u, e["height"] * u, e.get("rotation", 0)) for e in self.unit.els]
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))


class BadgesBlock:
    """A row of similar pills; wraps onto more rows when narrow."""

    def __init__(self, unit):
        self.unit = unit
        self.items = [GroupBlock(m) for m in sorted(unit.members, key=lambda m: m.bbox[0])]
        gaps = [self.items[i + 1].x0 - (self.items[i].x0 + self.items[i].w) for i in range(len(self.items) - 1)]
        self.gap = max(8.0, sum(gaps) / len(gaps)) if gaps else 16.0

    def _rows(self, u, max_w):
        rows, cur, cw = [], [], 0.0
        g = self.gap * u
        for it in self.items:
            w = it.w * u
            if cur and cw + g + w > max_w:
                rows.append((cur, cw))
                cur, cw = [], 0.0
            cw = cw + (g if cur else 0) + w
            cur.append(it)
        if cur:
            rows.append((cur, cw))
        return rows

    def scale_for(self, u, max_w):
        widest = max(it.w for it in self.items)
        if widest * u > max_w:
            u = max_w / widest
        return u

    def size(self, u, max_w):
        u = self.scale_for(u, max_w)
        rows = self._rows(u, max_w)
        h = max(it.h for it in self.items) * u
        return max(r[1] for r in rows), len(rows) * h + (len(rows) - 1) * self.gap * u

    def place(self, P, x, y, u, max_w, align):
        u = self.scale_for(u, max_w)
        rows = self._rows(u, max_w)
        h = max(it.h for it in self.items) * u
        g = self.gap * u
        for i, (row, rw) in enumerate(rows):
            cx = x + (max_w - rw) / 2 if align == "center" else x
            ry = y + i * (h + g)
            for it in row:
                it.place_at(P, cx, ry, u)
                cx += it.w * u + g


class FrameBlock:
    """A container (frame, card, outline) plus its contents. The container
    may change aspect ratio toward the new canvas; its contents are laid out
    again inside it, as a column or as a row (icons beside a text column)."""

    def __init__(self, unit, ctx):
        self.unit, self.ctx = unit, ctx
        c = unit.container
        self.c = c
        self.cw, self.ch = c["width"], c["height"]
        self.aspect = self.cw / max(1e-6, self.ch)
        boxes = [ch.bbox for ch in unit.children]
        ix0, iy0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        ix1, iy1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        cx0, cy0 = c["x"], c["y"]
        self.pad = (max(0.0, ix0 - cx0) / self.cw, max(0.0, iy0 - cy0) / self.ch,
                    max(0.0, cx0 + self.cw - ix1) / self.cw, max(0.0, cy0 + self.ch - iy1) / self.ch)
        alpha = getattr(ctx, "frame_alpha", 0.0)
        src_ca = ctx.Ws / ctx.Hs
        tgt_ca = ctx.W / ctx.H
        self.t_aspect = self.aspect * (tgt_ca / src_ca) ** alpha
        self.t_aspect = min(max(self.t_aspect, self.aspect / 4), self.aspect * 4)

    def dims(self, t):
        area = self.cw * self.ch * t * t
        return (area * self.t_aspect) ** 0.5, (area / self.t_aspect) ** 0.5

    def scale_for(self, t, max_w, max_h=None):
        w, h = self.dims(t)
        k = 1.0
        if w > max_w:
            k = max_w / w
        if max_h is not None and h * k > max_h:
            k = max_h / h
        return t * k

    def size(self, t, max_w):
        return self.dims(self.scale_for(t, max_w))

    def place(self, P, x, y, t, max_w, align):
        t = self.scale_for(t, max_w)
        w, h = self.dims(t)
        bx = x + (max_w - w) / 2 if align == "center" else x
        self.place_box(P, bx, y, w, h, t)

    def place_box(self, P, bx, by, w, h, t):
        from .layout import inner_layout
        c = self.c
        P.set(c["id"], x=bx, y=by, width=w, height=h)
        P.vis[c["id"]] = (bx, by, bx + w, by + h)
        # padding keeps its proportion on each axis, but never more than it was at uniform scale
        pl = min(self.pad[0] * w, self.pad[0] * self.cw * t * 1.5)
        pr = min(self.pad[2] * w, self.pad[2] * self.cw * t * 1.5)
        pt = min(self.pad[1] * h, self.pad[1] * self.ch * t * 1.5)
        pb = min(self.pad[3] * h, self.pad[3] * self.ch * t * 1.5)
        rect = (bx + pl, by + pt, max(1.0, w - pl - pr), max(1.0, h - pt - pb))
        P.frame_modes = getattr(P, "frame_modes", {})
        P.frame_modes[self.unit.uid] = inner_layout(self.ctx, P, self.unit.children, rect, t)


def make_block(unit, src_w, ctx=None):
    if unit.kind == "frame" and ctx is not None:
        return FrameBlock(unit, ctx)
    if unit.kind == "text":
        return TextBlock(unit, src_w)
    if unit.kind == "image":
        return ImageBlock(unit)
    if unit.kind == "badges":
        return BadgesBlock(unit)
    return GroupBlock(unit)
