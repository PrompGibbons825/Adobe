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


def rotate_box_about(box, cx, cy, deg):
    """Axis-aligned bounds of a box turned by deg about (cx, cy): the inked
    part of a rotated text turns about its element's centre, as the renderer does."""
    if abs(((deg or 0) + 180) % 360 - 180) < 0.5:
        return box
    a = math.radians(deg or 0)
    ca, sa = math.cos(a), math.sin(a)
    pts = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
    rp = [(cx + (x - cx) * ca - (y - cy) * sa, cy + (x - cx) * sa + (y - cy) * ca) for x, y in pts]
    return (min(p[0] for p in rp), min(p[1] for p in rp), max(p[0] for p in rp), max(p[1] for p in rp))


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
        if getattr(unit, "_in_strip", False):
            self.wfrac = 1.0          # in a rotated strip a line uses the strip's full length

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
        # measured on what the pieces cover once rotated (a sideways text is tall, not wide)
        bs = [rot_bounds(e["x"], e["y"], e["width"], e["height"], e.get("rotation", 0)) for e in unit.els] \
            if unit.kind != "badges" else [unit.bbox]
        x0, y0 = min(b[0] for b in bs), min(b[1] for b in bs)
        x1, y1 = max(b[2] for b in bs), max(b[3] for b in bs)
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
                vb = tf.visual_box(e, ex, ey, ew, fs, m)
                P.vis[e["id"]] = rotate_box_about(vb, ex + ew / 2, ey + max(eh, m["lines"] * fs * lh) / 2,
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
        # small text-free pieces sitting in a corner of the frame keep that corner;
        # only the rest is laid out again inside the frame
        cx0, cy0, cw0, ch0 = c["x"], c["y"], c["width"], c["height"]
        self.cornered = []
        flow_children = []
        for ch in unit.children:
            b = ch.bbox
            mx, my = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            near_x = mx - cx0 < 0.25 * cw0 or cx0 + cw0 - mx < 0.25 * cw0
            near_y = my - cy0 < 0.25 * ch0 or cy0 + ch0 - my < 0.25 * ch0
            small = (b[2] - b[0]) * (b[3] - b[1]) < 0.05 * cw0 * ch0
            if near_x and near_y and small and not any(e["type"] == "text" for e in ch.els):
                self.cornered.append(ch)
            else:
                flow_children.append(ch)
        if not flow_children:
            flow_children, self.cornered = list(unit.children), []
        self.flow_children = flow_children
        boxes = [ch.bbox for ch in flow_children]
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
        # A detailed picture is never reshaped. A flat card that fills its box
        # may change shape: it is cropped (fit=cover), never stretched.
        self.crop_card = False
        if c["type"] == "image":
            from .analyze import _fills_box
            if _fills_box(c):
                info = saliency.analyze(c.get("asset_id") or "")
                if info is not None and info["busyness"] > 0.03:
                    self.t_aspect = self.aspect
                else:
                    self.crop_card = True

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
        if self.crop_card and abs(w / max(1e-6, h) - self.aspect) > 0.01 * self.aspect:
            P.set(c["id"], fit="cover")
        P.vis[c["id"]] = (bx, by, bx + w, by + h)
        # padding keeps its proportion on each axis, but never more than it was at uniform scale
        if getattr(self.unit, "_in_strip", False):
            # a strip is short: padding at most 6% of its length
            p6 = 0.06 * max(w, h)
            self.pad = tuple(min(pv, p6 / max(1e-6, w if i % 2 == 0 else h)) for i, pv in enumerate(self.pad))
        pl = min(self.pad[0] * w, self.pad[0] * self.cw * t * 1.5)
        pr = min(self.pad[2] * w, self.pad[2] * self.cw * t * 1.5)
        pt = min(self.pad[1] * h, self.pad[1] * self.ch * t * 1.5)
        pb = min(self.pad[3] * h, self.pad[3] * self.ch * t * 1.5)
        rect = (bx + pl, by + pt, max(1.0, w - pl - pr), max(1.0, h - pt - pb))
        P.frame_modes = getattr(P, "frame_modes", {})
        P.frame_modes[self.unit.uid] = inner_layout(self.ctx, P, self.flow_children, rect, t)
        # corner pieces: same corner, same proportional inset, scaled with the frame
        s = (w * h / max(1e-6, self.cw * self.ch)) ** 0.5
        for ch in self.cornered:
            b = ch.bbox
            left = (b[0] + b[2]) / 2 - c["x"] < c["x"] + self.cw - (b[0] + b[2]) / 2
            top = (b[1] + b[3]) / 2 - c["y"] < c["y"] + self.ch - (b[1] + b[3]) / 2
            ox = (b[0] - c["x"]) / self.cw if left else (c["x"] + self.cw - b[2]) / self.cw
            oy = (b[1] - c["y"]) / self.ch if top else (c["y"] + self.ch - b[3]) / self.ch
            pw, ph = (b[2] - b[0]) * s, (b[3] - b[1]) * s
            px = bx + ox * w if left else bx + w - ox * w - pw
            py = by + oy * h if top else by + h - oy * h - ph
            blk = make_block(ch, self.ctx.Ws, None)
            if isinstance(blk, ImageBlock):
                el = ch.els[0]
                ew, eh = el["width"] * s, el["height"] * s
                P.set(el["id"], x=px, y=py, width=ew, height=eh)
                v = blk.v
                P.vis[el["id"]] = (px + v[0] * ew, py + v[1] * eh, px + v[2] * ew, py + v[3] * eh)
            else:
                gb = GroupBlock(ch)
                gb.place_at(P, px, py, s)


def find_cut(unit, carea):
    """A clean cut through a composition: a horizontal or vertical line that
    no substantial piece crosses, with real content on both sides. Small
    connectors that straddle it go with the side holding their centre.
    Returns (axis, part_a_els, part_b_els) or None."""
    els = unit.els
    if len(els) < 2:
        return None
    boxes = [rot_bounds(e["x"], e["y"], e["width"], e["height"], e.get("rotation", 0)) for e in els]
    gx0, gy0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    gx1, gy1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    garea = max(1.0, (gx1 - gx0) * (gy1 - gy0))
    best = None
    for axis in ("y", "x"):
        lo, hi = (1, 3) if axis == "y" else (0, 2)
        span0, span1 = (gy0, gy1) if axis == "y" else (gx0, gx1)
        tol = 0.01 * (span1 - span0)
        cands = sorted({b[hi] for b in boxes} | {b[lo] for b in boxes})
        for c in cands:
            if c <= span0 + 0.2 * (span1 - span0) or c >= span1 - 0.2 * (span1 - span0):
                continue        # both sides must hold a real part
            crossing = [b for b in boxes if b[lo] < c - tol and b[hi] > c + tol]
            small = all((b[2] - b[0]) * (b[3] - b[1]) < 0.05 * garea for b in crossing)
            if not small:
                continue
            a = [e for e, b in zip(els, boxes) if (b[lo] + b[hi]) / 2 < c]
            bb = [e for e, b in zip(els, boxes) if (b[lo] + b[hi]) / 2 >= c]
            if not a or not bb:
                continue
            bal = abs(len(a) - len(bb)) + abs((c - span0) - (span1 - c)) / (span1 - span0)
            if best is None or bal < best[0]:
                best = (bal, axis, a, bb)
    return best[1:] if best else None


class RestackBlock:
    """A composition re-stacked along the other axis: two parts that were
    stacked go side by side (or the reverse), still touching, same scale,
    centres aligned. Each part keeps its own arrangement."""

    def __init__(self, unit, cut):
        from .analyze import Unit
        axis, a, b = cut
        self.unit = unit
        self.parts = [GroupBlock(Unit(unit.uid + ":a", unit.role, "group", a)),
                      GroupBlock(Unit(unit.uid + ":b", unit.role, "group", b))]
        self.side = axis == "y"            # stacked in the source -> side by side now
        pa, pb = self.parts
        # keep the source gap between the parts (zero when they touched)
        if self.side:
            self.gap = max(0.0, pb.y0 - (pa.y0 + pa.h))
            self.w, self.h = pa.w + self.gap + pb.w, max(pa.h, pb.h)
        else:
            self.gap = max(0.0, pb.x0 - (pa.x0 + pa.w))
            self.w, self.h = max(pa.w, pb.w), pa.h + self.gap + pb.h
        self.x0, self.y0 = unit.bbox[0], unit.bbox[1]

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
        pa, pb = self.parts
        if self.side:
            pa.place_at(P, gx, gy + (self.h - pa.h) * u / 2, u)
            pb.place_at(P, gx + (pa.w + self.gap) * u, gy + (self.h - pb.h) * u / 2, u)
        else:
            pa.place_at(P, gx + (self.w - pa.w) * u / 2, gy, u)
            pb.place_at(P, gx + (self.w - pb.w) * u / 2, gy + (pa.h + self.gap) * u, u)
        P.restacked = getattr(P, "restacked", 0) + 1

    def bounds_at(self, gx, gy, u):
        return (gx, gy, gx + self.w * u, gy + self.h * u)


def make_block(unit, src_w, ctx=None):
    if unit.kind == "frame" and ctx is not None:
        return FrameBlock(unit, ctx)
    if unit.kind == "group" and ctx is not None and getattr(ctx, "restack", False):
        cut = find_cut(unit, ctx.Ws * ctx.Hs)
        if cut:
            return RestackBlock(unit, cut)
    if unit.kind == "text":
        return TextBlock(unit, src_w)
    if unit.kind == "image":
        return ImageBlock(unit)
    if unit.kind == "badges":
        return BadgesBlock(unit)
    return GroupBlock(unit)
