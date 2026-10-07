"""Layout templates. Each returns a Placed (or None if it can't fit).

- stack: one centered/left column, reading order from source (tall/square)
- split: hero image in its own column, text stack beside it (wide)
- row:   everything on one line, logo | hero | text | cta (ultra-wide banners)
"""

from __future__ import annotations

from .blocks import GroupBlock, ImageBlock, Placed, TextBlock, make_block
from . import textfit as tf


class Ctx:
    def __init__(self, model, target):
        self.model = model
        self.Ws, self.Hs = model["canvas"]
        self.W, self.H = target["width"], target["height"]
        self.t_star = ((self.W * self.H) / (self.Ws * self.Hs)) ** 0.5
        self.margin = 0.06 * min(self.W, self.H) + 0.01 * max(self.W, self.H)


def _src_gaps(units):
    gaps = []
    for a, b in zip(units, units[1:]):
        gaps.append(b.bbox[1] - a.bbox[3])
    return gaps


MIN_FS = {"headline": 16, "cta": 12, "subhead": 11, "body": 11, "group": 10, "badges": 9, "legal": 7}


def _src_fs(u):
    els = u.els if u.kind != "badges" else [e for m in u.members for e in m.els]
    fss = [e["font_size"] for e in els if e["type"] == "text"]
    return min(fss) if fss else None


def block_scale(u, t, k, boost=True):
    if u.role == "hero":
        return t * k
    fs = _src_fs(u)
    if boost and fs and u.role in MIN_FS:
        return max(t, MIN_FS[u.role] / fs)
    return t


def _stack_sizes(blocks, units, t, k, max_w, boost=True):
    return [b.size(block_scale(u, t, k, boost), max_w) for b, u in zip(blocks, units)]


def stack(ctx: Ctx, P: Placed, units, rect, align, t_hi):
    """Try with minimum-legibility boosts first; if the content can't fit
    that way, fall back to pure proportional scaling (nothing is dropped)."""
    r = _stack(ctx, P, units, rect, align, t_hi, True)
    if r is None:
        r = _stack(ctx, P, units, rect, align, t_hi, False)
    return r


def _stack(ctx: Ctx, P: Placed, units, rect, align, t_hi, boost):
    """Lay units out top-to-bottom in rect. Returns chosen t or None."""
    if not units:
        return t_hi
    units = sorted(units, key=lambda u: u.cy)
    blocks = [make_block(u, ctx.Ws, ctx) for u in units]
    rx, ry, rw, rh = rect
    base_gaps = [min(max(g, 0.02 * ctx.Hs), 0.08 * ctx.Hs) for g in _src_gaps(units)]
    has_hero = any(u.role == "hero" for u in units)
    k_min = 0.6 if has_hero else 1.0

    def total(t, k):
        sizes = _stack_sizes(blocks, units, t, k, rw, boost)
        return sum(s[1] for s in sizes) + sum(g * t for g in base_gaps), sizes

    lo, hi = 0.0, t_hi
    if total(hi, k_min)[0] <= rh:
        lo = hi
    else:
        for _ in range(30):
            mid = (lo + hi) / 2
            if total(mid, k_min)[0] <= rh:
                lo = mid
            else:
                hi = mid
    t = lo
    if t <= 0.003:
        return None
    k = k_min
    if has_hero:
        klo, khi = k_min, 1.8
        for _ in range(25):
            mid = (klo + khi) / 2
            if total(t, mid)[0] <= rh:
                klo = mid
            else:
                khi = mid
        k = klo
    tot, sizes = total(t, k)
    gaps = [g * t for g in base_gaps]
    slack = rh - tot
    if gaps and sum(gaps) > 0 and slack > 0:
        extra = min(slack, sum(gaps) * 1.5)
        f = 1 + extra / sum(gaps)
        gaps = [g * f for g in gaps]
        slack -= extra
    y = ry + max(0.0, slack) / 2
    for i, (b, u) in enumerate(zip(blocks, units)):
        b.place(P, rx, y, block_scale(u, t, k, boost), rw, align)
        y += sizes[i][1] + (gaps[i] if i < len(gaps) else 0)
    return t


def t_cap(ctx, col_w):
    return min(1.25 * ctx.t_star, 2.0)


def tmpl_stack(ctx, flow, align="center", region="full"):
    P = Placed()
    m = ctx.margin
    W, H = ctx.W, ctx.H
    rect = {
        "full": (m, m, W - 2 * m, H - 2 * m),
        "left": (m, m, W * 0.55 - 1.5 * m, H - 2 * m),
        "right": (W * 0.45 + 0.5 * m, m, W * 0.55 - 1.5 * m, H - 2 * m),
        "top": (m, m, W - 2 * m, H * 0.55 - 1.5 * m),
        "bottom": (m, H * 0.45 + 0.5 * m, W - 2 * m, H * 0.55 - 1.5 * m),
    }[region]
    t = stack(ctx, P, flow, rect, align, t_cap(ctx, rect[2]))
    if t is None:
        return None
    P.meta = {"template": f"stack-{region}-{align}", "t": t}
    return P


def tmpl_split(ctx, flow, side="left", frac=0.45, align="left"):
    if ctx.W / ctx.H < 1.15:
        return None
    heroes = [u for u in flow if u.role == "hero"]
    if not heroes:
        return None
    hero = heroes[0]
    rest = [u for u in flow if u is not hero]
    P = Placed()
    m = ctx.margin
    W, H = ctx.W, ctx.H
    hb = ImageBlock(hero)
    col_w = W * frac - 1.5 * m
    s = hb.scale_for(10.0, col_w, H - 2 * m)
    hw, hh = hb.vw * s, hb.vh * s
    if side == "left":
        hx = m + (col_w - hw) / 2
        text_rect = (W * frac + 0.5 * m, m, W * (1 - frac) - 1.5 * m, H - 2 * m)
    else:
        hx = W * (1 - frac) + 0.5 * m + (col_w - hw) / 2
        text_rect = (m, m, W * (1 - frac) - 1.5 * m, H - 2 * m)
    hb.place_vis(P, hx, (H - hh) / 2, s)
    t = stack(ctx, P, rest, text_rect, align, t_cap(ctx, text_rect[2]))
    if t is None:
        return None
    P.meta = {"template": f"split-{side}-{frac}-{align}", "t": t}
    return P


ROW_ORDER = {"logo": 0, "hero": 1, "text": 2, "headline": 2, "subhead": 2, "body": 2, "group": 3,
             "legal": 3, "badges": 4, "cta": 5}


def _fit_column(texts, col_w, h_avail, fs_cap):
    """Joint fit of a column of texts: one scale t for all (keeps hierarchy)."""
    if not texts:
        return None
    head_fs = max(u.els[0]["font_size"] for u in texts)

    def layout(t):
        ms, tot = [], 0.0
        for i, u in enumerate(texts):
            el = u.els[0]
            fs = max(5, int(el["font_size"] * t))
            m = tf.measure(el, fs, col_w)
            if m["word_w"] > col_w:
                return None
            ms.append(m)
            tot += m["height"] + (fs * 0.3 if i else 0)
        return ms if tot <= h_avail else None

    lo, hi = 0.0, fs_cap / head_fs
    if layout(hi):
        return layout(hi)
    for _ in range(25):
        mid = (lo + hi) / 2
        if layout(mid):
            lo = mid
        else:
            hi = mid
    return layout(lo) if lo > 0 else None


def tmpl_row(ctx, flow, item_scale=1.0, ncols=1):
    """Everything on one line, every unit kept: logo | hero | text column(s) | groups | badges | cta."""
    from .blocks import BadgesBlock
    W, H = ctx.W, ctx.H
    if W / H < 2.2:
        return None
    P = Placed()
    mv = 0.12 * H
    hA = H - 2 * mv
    gut = max(0.015 * W, 0.6 * mv)
    texts = sorted([u for u in flow if u.kind == "text" and u.role != "cta"], key=lambda u: (u.bbox[1], u.bbox[0]))
    others = [u for u in flow if u not in texts]
    items = []  # dict(order, kind, block, w, h, s, unit)
    for u in others:
        it = {"order": ROW_ORDER.get(u.role, 3), "unit": u, "x0": u.bbox[0]}
        if u.kind == "image":
            b = ImageBlock(u)
            fh = {"hero": 1.15, "logo": 0.75}.get(u.role, 0.6)
            sc = b.scale_for(10.0, 0.22 * W * item_scale, hA * fh * item_scale)
            it.update(kind="img", block=b, w=b.vw * sc, h=b.vh * sc, s=sc)
        elif u.kind == "badges":
            b = BadgesBlock(u)
            ih = max(x.h for x in b.items)
            sc = hA * 0.42 * item_scale / ih
            for _ in range(20):
                bw, bh = b.size(sc, 0.3 * W)
                if bh <= hA * 0.95:
                    break
                sc *= 0.9
            it.update(kind="badges", block=b, w=bw, h=bh, s=sc, maxw=0.3 * W)
        elif u.kind == "text":  # text CTA
            el = u.els[0]
            mm = tf.fit(el, 0.25 * W, hA * 0.6, min(hA * 0.45, el["font_size"]), 1, 1)
            if mm is None:
                return None
            it.update(kind="ctatext", block=u, w=mm["width"], h=mm["height"], s=mm)
        elif u.kind == "frame":
            from .blocks import FrameBlock
            b = FrameBlock(u, ctx)
            h = hA * 0.98 * item_scale
            w = min(h * b.t_aspect, 0.5 * W)
            h = w / b.t_aspect
            it.update(kind="frame", block=b, w=w, h=h, s=h / b.dims(1.0)[1])
        else:
            b = GroupBlock(u)
            fh = {"cta": 0.7, "badges": 0.5, "headline": 1.0}.get(u.role, 0.8)
            sc = b.scale_for(10.0, 0.32 * W * item_scale, hA * fh * item_scale)
            it.update(kind="group", block=b, w=b.w * sc, h=b.h * sc, s=sc)
        items.append(it)
    cols = []
    if texts:
        if ncols == 2 and len(texts) >= 2:
            cols = [texts[:1], texts[1:]]
        else:
            cols = [texts]
        for c in cols:
            items.append({"order": 2, "kind": "textcol", "unit": None, "texts": c, "x0": c[0].bbox[0]})
    items.sort(key=lambda it: (it["order"], it["x0"]))

    def pad(it, side):
        if it["kind"] == "img":
            b = it["block"]
            return (b.v[0] if side == 0 else 1 - b.v[2]) * b.el["width"] * it["s"]
        return 0.0
    mL = max(0.03 * W, mv, pad(items[0], 0) + 2)
    mR = max(0.03 * W, mv, pad(items[-1], 1) + 2)
    fixed = sum(it["w"] for it in items if it["kind"] != "textcol")
    n = len(items)
    col_total = W - mL - mR - fixed - gut * (n - 1)
    tcols = [it for it in items if it["kind"] == "textcol"]
    if tcols:
        if col_total < 0.14 * W * len(tcols):
            return None
        shares = [1.0] if len(tcols) == 1 else [0.58, 0.42]
        for it, sh in zip(tcols, shares):
            cw = col_total * sh
            cap = hA * (0.5 if len(it["texts"]) > 1 else 0.62)
            ms = _fit_column(it["texts"], cw, hA, cap)
            if ms is None or ms[0]["fs"] < 7:
                return None
            it.update(w=max(m["width"] for m in ms), h=sum(m["height"] for m in ms)
                      + sum(m["fs"] * 0.3 for m in ms[1:]), ms=ms, colw=cw)
    total_w = sum(it["w"] for it in items)
    free = W - mL - mR - total_w
    g = free / (n - 1) if n > 1 else 0
    if n > 1 and g < gut * 0.5:
        return None
    g = min(g, 0.08 * W)
    x = mL + (W - mL - mR - total_w - g * (n - 1)) / 2
    cy = H / 2
    head_t = 1.0
    for it in items:
        k = it["kind"]
        if k == "img":
            it["block"].place_vis(P, x, cy - it["h"] / 2, it["s"])
        elif k == "group":
            it["block"].place_at(P, x, cy - it["h"] / 2, it["s"])
        elif k == "frame":
            it["block"].place_box(P, x, cy - it["h"] / 2, it["w"], it["h"], it["s"])
        elif k == "badges":
            it["block"].place(P, x, cy - it["h"] / 2, it["s"], it["w"] + 1, "left")
        elif k == "ctatext":
            el, mm = it["unit"].els[0], it["s"]
            y = tf.optical_top(el, mm["fs"], 1, cy)
            P.set(el["id"], x=x, y=y, width=mm["width"] + mm["fs"], height=mm["height"], font_size=mm["fs"], text_align="left")
            P.vis[el["id"]] = tf.visual_box(dict(el, text_align="left"), x, y, mm["width"] + mm["fs"], mm["fs"], mm)
        else:
            y = cy - it["h"] / 2
            ids = []
            for u, m in zip(it["texts"], it["ms"]):
                el = u.els[0]
                asc, cap, desc = tf.metrics(el)
                P.set(el["id"], x=x, y=y, width=it["colw"], height=m["height"], font_size=m["fs"], text_align="left")
                P.vis[el["id"]] = tf.visual_box(dict(el, text_align="left"), x, y, it["colw"], m["fs"], m)
                ids.append(el["id"])
                if u.role == "headline":
                    head_t = m["fs"] / el["font_size"]
                y += m["height"] + m["fs"] * 0.3
            vb = [P.vis[i] for i in ids]
            dy = cy - (min(b[1] for b in vb) + max(b[3] for b in vb)) / 2
            for i in ids:
                P.props[i]["y"] += dy
                v = P.vis[i]
                P.vis[i] = (v[0], v[1] + dy, v[2], v[3] + dy)
        x += it["w"] + g
    P.meta = {"template": f"row-s{item_scale}-c{len(tcols)}", "t": head_t}
    return P


def _inner_row(ctx, P, units, rect, t):
    """Row inside a frame: non-text pieces on the left, one text column on
    the right. Returns the text scale achieved, or None."""
    rx, ry, rw, rh = rect
    texts = sorted([u for u in units if u.kind == "text"], key=lambda u: (u.bbox[1], u.bbox[0]))
    rest = [u for u in units if u.kind != "text"]
    if not rest or not texts:
        return None
    g = 0.06 * rh
    placed_w = 0.0
    blocks = []
    for u in sorted(rest, key=lambda u: (u.bbox[0], u.bbox[1])):
        b = make_block(u, ctx.Ws, ctx)
        if isinstance(b, ImageBlock):
            s = b.scale_for(t * 1.6, 0.3 * rw, 0.8 * rh)
            w, h = b.vw * s, b.vh * s
        elif isinstance(b, GroupBlock):
            s = b.scale_for(t * 1.6, 0.3 * rw, 0.8 * rh)
            w, h = b.w * s, b.h * s
        else:
            s = t
            w, h = b.size(s, 0.3 * rw)
        blocks.append((b, s, w, h))
        placed_w += w + g
    col_w = rw - placed_w
    if col_w < 0.25 * rw:
        return None
    cap = max(u.els[0]["font_size"] for u in texts) * t * 1.6
    ms = _fit_column(texts, col_w, rh, cap)
    if ms is None:
        return None
    tw = max(m["width"] for m in ms)
    th = sum(m["height"] for m in ms) + sum(m["fs"] * 0.3 for m in ms[1:])
    x = rx + (rw - placed_w - tw) / 2
    cy = ry + rh / 2
    for b, s, w, h in blocks:
        if isinstance(b, ImageBlock):
            b.place_vis(P, x, cy - h / 2, s)
        elif isinstance(b, GroupBlock):
            b.place_at(P, x, cy - h / 2, s)
        else:
            b.place(P, x, cy - h / 2, s, w, "left")
        x += w + g
    y = cy - th / 2
    for u, m in zip(texts, ms):
        el = u.els[0]
        P.set(el["id"], x=x, y=y, width=tw + m["fs"], height=m["height"], font_size=m["fs"], text_align="left")
        P.vis[el["id"]] = tf.visual_box(dict(el, text_align="left"), x, y, tw + m["fs"], m["fs"], m)
        y += m["height"] + m["fs"] * 0.3
    return ms[0]["fs"] / texts[0].els[0]["font_size"]


def inner_layout(ctx, P, units, rect, t):
    """Lay out a frame's contents inside rect, trying both a column and a row
    (icons beside a text column); keep whichever lets the text be larger."""
    options = []
    col = Placed()
    tt = stack(ctx, col, units, rect, "center", t * 1.6)
    if tt is not None:
        options.append((tt * 1.2, "column", col))  # slight preference: keeps the designer's structure
    if rect[2] / max(1.0, rect[3]) > 1.3:
        row = Placed()
        tr = _inner_row(ctx, row, units, rect, t)
        if tr is not None:
            options.append((tr, "row", row))
    if not options:
        return None
    _, mode, best = max(options, key=lambda o: o[0])
    for k, v in best.props.items():
        P.set(k, **v)
    P.vis.update(best.vis)
    P.frame_modes = getattr(P, "frame_modes", {})
    P.frame_modes.update(getattr(best, "frame_modes", {}))
    return mode
