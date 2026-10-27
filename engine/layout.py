"""Layout templates. Each returns a Placed (or None if it can't fit).

- stack: one centered/left column, reading order from source (tall/square)
- split: hero image in its own column, text stack beside it (wide)
- row:   everything on one line, logo | hero | text | cta (ultra-wide banners)
"""

from __future__ import annotations

from .analyze import norm_rot
from .conform import type_floor
from .blocks import GroupBlock, ImageBlock, Placed, TextBlock, make_block
from . import textfit as tf


class Ctx:
    def __init__(self, model, target):
        self.model = model
        self.Ws, self.Hs = model["canvas"]
        self.W, self.H = target["width"], target["height"]
        self.t_star = ((self.W * self.H) / (self.Ws * self.Hs)) ** 0.5
        self.margin = 0.06 * min(self.W, self.H) + 0.01 * max(self.W, self.H)
        self.floor = type_floor(self.W, self.H)    # no text is set below this (S1)
        # a border that echoes the canvas edge keeps its inset; content stays inside it
        k = min(self.W, self.H) / min(self.Ws, self.Hs)
        self.border_scale = k
        for u in model["units"]:
            if u.role == "border":
                inner = max(getattr(u, "insets", (0, 0, 0, 0))) * k
                if inner > 0:
                    self.margin = max(self.margin, inner + 0.04 * min(self.W, self.H))


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


def grows(u, units):
    """The pieces that grow into free space: the hero visual and the headline
    (alone or as the composition carrying it)."""
    if u.role == "hero":
        return True
    return u.role == "headline"


def block_scale(u, t, k, boost=True, grow=False, floor=0):
    """The scale a unit is set at. With boost, a unit carrying text is never scaled so far that
    its smallest text falls under its role's minimum or the canvas's type floor."""
    if u.role == "hero":
        return t * k
    fs = _src_fs(u)
    lo = max(MIN_FS.get(u.role, 0), floor) / fs if (boost and fs) else 0.0
    if grow:                       # the headline grows with k but never shrinks below t
        return max(t * max(1.0, k), lo)
    return max(t, lo)


def _stack_sizes(blocks, units, t, k, max_w, boost=True, floor=0):
    return [b.size(block_scale(u, t, k, boost, grows(u, units), floor), max_w) for b, u in zip(blocks, units)]


def stack(ctx: Ctx, P: Placed, units, rect, align, t_hi):
    """Try with minimum-legibility boosts first; if the content can't fit
    that way, fall back to pure proportional scaling (nothing is dropped)."""
    r = _stack(ctx, P, units, rect, align, t_hi, True)
    if r is None:
        r = _stack(ctx, P, units, rect, align, t_hi, False)
    return r


def _line_extra(b, sc, max_w):
    """(above, below, is_text): how far a text's line box reaches past its inked box."""
    if not isinstance(b, TextBlock):
        return 0.0, 0.0, False
    _, fs, _ = b._fit(sc, max_w)
    asc, cap, desc = tf.metrics(b.el)
    lh = float(b.el.get("line_height", 1.2))
    return (asc - cap) * fs, max(0.0, max(fs * lh, (asc + desc) * fs) - (asc + 0.6 * desc) * fs), True


def _stack(ctx: Ctx, P: Placed, units, rect, align, t_hi, boost):
    """Lay units out top-to-bottom in rect. Returns chosen t or None."""
    if not units:
        return t_hi
    units = sorted(units, key=lambda u: u.cy)
    blocks = [make_block(u, ctx.Ws, ctx) for u in units]
    rx, ry, rw, rh = rect
    base_gaps = [min(max(g, 0.02 * ctx.Hs), 0.08 * ctx.Hs) for g in _src_gaps(units)]
    has_hero = any(grows(u, units) for u in units)
    k_min = 0.6 if any(u.role == "hero" for u in units) else 1.0

    def need(t, k):
        """Least gap after each block: two texts' line boxes (what the renderer occupies, larger
        than the ink a stack aligns on) never touch."""
        ex = [_line_extra(b, block_scale(u, t, k, boost, grows(u, units), ctx.floor), rw)
              for b, u in zip(blocks, units)]
        return [ex[i][1] + ex[i + 1][0] if ex[i][2] and ex[i + 1][2] else 0.0 for i in range(len(ex) - 1)]

    def gaps_at(t, k):
        return [max(g * t, n) for g, n in zip(base_gaps, need(t, k))]

    def total(t, k):
        sizes = _stack_sizes(blocks, units, t, k, rw, boost, ctx.floor)
        return sum(s[1] for s in sizes) + sum(gaps_at(t, k)), sizes

    # with no hero image, keep room for the headline to stand 1.4x above the rest
    k_t = 1.4 if k_min == 1.0 and has_hero else k_min
    lo, hi = 0.0, t_hi
    if total(hi, k_t)[0] <= rh:
        lo = hi
    else:
        for _ in range(30):
            mid = (lo + hi) / 2
            if total(mid, k_t)[0] <= rh:
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
    gaps = gaps_at(t, k)
    slack = rh - tot
    if gaps and sum(gaps) > 0 and slack > 0:
        extra = min(slack, sum(gaps) * 1.5)
        f = 1 + extra / sum(gaps)
        gaps = [g * f for g in gaps]
        slack -= extra
    y = ry + max(0.0, slack) / 2
    for i, (b, u) in enumerate(zip(blocks, units)):
        b.place(P, rx, y, block_scale(u, t, k, boost, grows(u, units), ctx.floor), rw, align)
        y += sizes[i][1] + (gaps[i] if i < len(gaps) else 0)
    return t


def t_cap(ctx, col_w):
    # type may grow with the canvas: a small banner's 15px headline becomes ~50px on a big one.
    # Blank space the designer left in the source (content filling little of
    # it) is not a size to preserve: content may grow into the new canvas.
    fill = ctx.model.get("src_fill", 0.55) if hasattr(ctx, "model") else 0.55
    room = max(1.0, (0.5 / max(0.05, fill)) ** 0.5)
    return min(1.25 * ctx.t_star * room, 3.0)


def tmpl_stack(ctx, flow, align="center", region="full", tscale=1.0):
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
    t = stack(ctx, P, flow, rect, align, t_cap(ctx, rect[2]) * tscale)
    if t is None:
        return None
    P.meta = {"template": f"stack-{region}-{align}" + ("-herofirst" if tscale < 1 else ""), "t": t}
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
    is_img = hero.kind == "image"
    hb = ImageBlock(hero) if is_img else make_block(hero, ctx.Ws, ctx)
    col_w = W * frac - 1.5 * m
    s = hb.scale_for(10.0, col_w, H - 2 * m)
    hw, hh = (hb.vw * s, hb.vh * s) if is_img else (hb.w * s, hb.h * s)
    if side == "left":
        hx = m + (col_w - hw) / 2
        text_rect = (W * frac + 0.5 * m, m, W * (1 - frac) - 1.5 * m, H - 2 * m)
    else:
        hx = W * (1 - frac) + 0.5 * m + (col_w - hw) / 2
        text_rect = (m, m, W * (1 - frac) - 1.5 * m, H - 2 * m)
    if is_img:
        hb.place_vis(P, hx, (H - hh) / 2, s)
    else:
        hb.place_at(P, hx, (H - hh) / 2, s)
    t = stack(ctx, P, rest, text_rect, align, t_cap(ctx, text_rect[2]))
    if t is None:
        return None
    P.meta = {"template": f"split-{side}-{frac}-{align}", "t": t}
    return P


ROW_ORDER = {"logo": 0, "hero": 1, "text": 2, "headline": 2, "subhead": 2, "body": 2, "group": 3,
             "legal": 3, "badges": 4, "cta": 5}


def _fit_column(texts, col_w, h_avail, fs_cap, floor=5):
    """Joint fit of a column of texts: one scale t for all (keeps hierarchy); no text under
    the type floor. None if the column cannot hold them even at the floor."""
    if not texts:
        return None
    head_fs = max(u.els[0]["font_size"] for u in texts)

    def layout(t):
        ms, tot = [], 0.0
        for i, u in enumerate(texts):
            el = tf.flow(u.els[0])
            fs = max(floor, int(el["font_size"] * t))
            m = tf.measure(el, fs, col_w)
            if m["word_w"] > col_w:
                return None
            ms.append(m)
            tot += m["height"] + (fs * 0.3 if i else 0)
        return ms if tot <= h_avail else None

    lo, hi = 0.0, max(fs_cap, floor) / head_fs
    if layout(hi):
        return layout(hi)
    if not layout(lo):
        return None
    for _ in range(25):
        mid = (lo + hi) / 2
        if layout(mid):
            lo = mid
        else:
            hi = mid
    return layout(lo)


def _rot_frame(fb, u, L, S):
    """A card in a strip takes the thickness that gives its text the largest size."""
    best = None
    for f in (0.35, 0.5, 0.7, 1.0, 1.4):
        th = min(S, f * L)
        Pv = Placed()
        fb.place_box(Pv, 0.0, 0.0, L, th, 3.0)
        fss = [Pv.props[i]["font_size"] for i in u.ids if "font_size" in Pv.props.get(i, {})]
        if not fss:
            continue
        key = (min(fss), -th)
        if best is None or key > best[0]:
            best = (key, Pv, th)
    if best is None:
        return None
    return best[1], best[2], 0.0


def _rot_layout(ctx, u, L, S):
    """Lay a unit out in a virtual rect L long and up to S thick (text reflows
    to lines L long, using the strip's full length). Returns (placed,
    thickness used, offset) or None."""
    from .blocks import FrameBlock
    if u.kind == "frame":
        marks = [u] + list(u.children)
        for m in marks:
            m._in_strip = True
        try:
            return _rot_frame(FrameBlock(u, ctx), u, L, S)
        finally:
            for m in marks:
                m._in_strip = False
    Pv = Placed()
    t = stack(ctx, Pv, [u], (0.0, 0.0, L, S), "center", t_cap(ctx, L) * 1.5)
    if t is None:
        return None
    bs = [Pv.vis[i] for i in u.ids if i in Pv.vis]
    if not bs:
        return None
    v0, v1 = min(b[1] for b in bs), max(b[3] for b in bs)
    return Pv, v1 - v0, -v0


def _put_rotated(P, Pv, u, model_els, x, y0, L, dv):
    """Map a virtual layout into a vertical strip at (x, y0), turned -90
    degrees so its lines read bottom to top along the strip."""
    for eid in u.ids:
        p = Pv.props.get(eid)
        if p is None:
            continue
        el = model_els[eid]
        w, h = p.get("width", el["width"]), p.get("height", el["height"])
        cu, cv = p["x"] + w / 2, p["y"] + h / 2 + dv
        cx, cy = x + cv, y0 + L - cu
        rot = ((el.get("rotation", 0) or 0) - 90) % 360
        q = dict(p)
        q.update(x=cx - w / 2, y=cy - h / 2, rotation=rot - 360 if rot > 180 else rot)
        P.set(eid, **q)
        if eid in Pv.vis:
            a0, b0, a1, b1 = Pv.vis[eid]
            P.vis[eid] = (x + b0 + dv, y0 + L - a1, x + b1 + dv, y0 + L - a0)


def _upright(u):
    """A copy of a group whose pieces are all turned ~90 degrees, turned back
    upright as a whole about its centre. Returns (unit copy, {id: rotation})."""
    from .analyze import Unit
    from .analyze import norm_rot
    # every text turned the same way (+-90); the card under it turns with it
    trots = [norm_rot(e.get("rotation", 0)) for e in u.els if e["type"] == "text"]
    if not trots:
        return None
    r = trots[0]
    if not all(abs(abs(x) - 90) < 2 and abs(x - r) < 2 for x in trots):
        return None
    x0, y0, x1, y1 = u.bbox
    gx, gy = (x0 + x1) / 2, (y0 + y1) / 2
    import math
    a = math.radians(-r)
    ca, sa = math.cos(a), math.sin(a)
    els, newrot = [], {}
    for e in u.els:
        cx, cy = e["x"] + e["width"] / 2 - gx, e["y"] + e["height"] / 2 - gy
        nx, ny = cx * ca - cy * sa, cx * sa + cy * ca
        nr = norm_rot((e.get("rotation", 0) or 0) - r)
        c = dict(e, rotation=nr)
        c["x"], c["y"] = gx + nx - e["width"] / 2, gy + ny - e["height"] / 2
        els.append(c)
        newrot[e["id"]] = nr
    return Unit(u.uid, u.role, "group", els), newrot


def tmpl_row(ctx, flow, item_scale=1.0, ncols=1, area=None, P=None, rotate=False):
    """Everything on one line, every unit kept: logo | hero | text column(s) | groups | badges | cta.
    `area` = (x, width) restricts the row to a vertical band of the canvas.
    Aligned pieces are left to finalize, which keeps them on their anchor's line."""
    from .blocks import BadgesBlock
    ax, W = area if area else (0.0, ctx.W)
    H = ctx.H
    if W / H < 2.2:
        return None
    flow = [u for u in flow if u.role != "aligned"]
    P = P if P is not None else Placed()
    mv = 0.12 * H
    hA = H - 2 * mv
    gut = max(0.015 * W, 0.6 * mv)
    texts = sorted([u for u in flow if u.kind == "text" and u.role != "cta"], key=lambda u: (u.bbox[1], u.bbox[0]))
    others = [u for u in flow if u not in texts]
    items = []  # dict(order, kind, block, w, h, s, unit)
    for u in others:
        it = {"order": ROW_ORDER.get(u.role, 3), "unit": u, "x0": u.bbox[0]}
        has_txt = any(e["type"] == "text" for e in u.els)
        if rotate and has_txt and u.kind in ("group", "ribbon"):
            up = _upright(u)
            if up is not None:
                b = GroupBlock(up[0])
                sc = b.scale_for(10.0, 0.32 * W * item_scale, hA * 0.9 * item_scale)
                it.update(kind="group", block=b, w=b.w * sc, h=b.h * sc, s=sc, upright=up[1])
                items.append(it)
                continue
        if rotate and has_txt and u.kind in ("frame", "group") and u.role not in ("headline", "subhead", "cta") \
                and not any(abs(norm_rot(e.get("rotation", 0))) > 1 for e in u.els):
            r = _rot_layout(ctx, u, hA, 0.2 * W)
            if r is not None:
                it.update(kind="rot", Pv=r[0], w=r[1], h=hA, dv=r[2])
                items.append(it)
                continue
        if u.kind == "image":
            b = ImageBlock(u)
            fh = {"hero": 1.15, "logo": 0.75}.get(u.role, 0.6)
            sc = b.scale_for(10.0, (0.45 if u.role == "hero" else 0.22) * W * item_scale, hA * fh * item_scale)
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
            b = make_block(u, ctx.Ws, ctx)
            fh = {"cta": 0.7, "badges": 0.5, "headline": 1.0, "hero": 1.1}.get(u.role, 0.8)
            # a group carrying the main picture may take up to 45% of the banner
            wcap = 0.45 if (u.role == "hero" or any(e["type"] == "image" and e["width"] * e["height"] >= 0.15 * ctx.Ws * ctx.Hs
                                                     for e in u.els)) else 0.32
            sc = b.scale_for(10.0, wcap * W * item_scale, hA * fh * item_scale)
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
    if not items:
        return None
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
    # the largest type already set in cards/groups: loose secondary text in a
    # column must not outgrow it (a legal line is never bigger than the headline)
    refs = []          # (source size, size set here) of texts already placed in cards and groups
    for it in items:
        u = it.get("unit")
        if u is None or not any(e["type"] == "text" for e in u.els):
            continue
        for e in u.els:
            if e["type"] != "text":
                continue
            if it["kind"] == "rot" and e["id"] in it["Pv"].props:
                refs.append((e["font_size"], it["Pv"].props[e["id"]].get("font_size", 0)))
            elif it["kind"] in ("group", "frame"):
                refs.append((e["font_size"], e["font_size"] * it["s"]))
    if tcols:
        if col_total < 0.14 * W * len(tcols):
            return None
        shares = [1.0] if len(tcols) == 1 else [0.58, 0.42]
        for it, sh in zip(tcols, shares):
            cw = col_total * sh
            cap = hA * (0.5 if len(it["texts"]) > 1 else 0.62)
            # stay smaller than every text that was larger than this column's in the source
            col_src = max(u.els[0]["font_size"] for u in it["texts"])
            above = [out for src, out in refs if src > 1.05 * col_src and out > 0]
            if above:
                cap = min(cap, max(7.0, 0.9 * min(above)))
            ms = _fit_column(it["texts"], cw, hA, cap, ctx.floor)
            if ms is None:
                return None
            it.update(w=max(m["width"] for m in ms), h=sum(m["height"] for m in ms)
                      + sum(m["fs"] * 0.3 for m in ms[1:]), ms=ms, colw=cw)
    total_w = sum(it["w"] for it in items)
    free = W - mL - mR - total_w
    g = free / (n - 1) if n > 1 else 0
    if n > 1 and g < gut * 0.5:
        return None
    g = min(g, 0.08 * W)
    x = ax + mL + (W - mL - mR - total_w - g * (n - 1)) / 2
    cy = H / 2
    head_t = 1.0
    for it in items:
        k = it["kind"]
        if k == "img":
            it["block"].place_vis(P, x, cy - it["h"] / 2, it["s"])
        elif k == "group":
            it["block"].place_at(P, x, cy - it["h"] / 2, it["s"])
            for eid, r in it.get("upright", {}).items():
                P.set(eid, rotation=r)
            if it.get("upright"):
                P.rotated = getattr(P, "rotated", 0) + 1
            if it["unit"].role == "headline":
                head_t = it["s"]       # a headline lockup: its scale is the text scale
        elif k == "frame":
            it["block"].place_box(P, x, cy - it["h"] / 2, it["w"], it["h"], it["s"])
            if it["unit"].role == "headline":
                head_t = it["s"]
        elif k == "rot":
            _put_rotated(P, it["Pv"], it["unit"], ctx.model["els"], x, cy - it["h"] / 2, it["h"], it["dv"])
            P.rotated = getattr(P, "rotated", 0) + 1
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
                P.set(el["id"], x=x, y=y, width=it["colw"], height=m["height"], font_size=m["fs"], text_align="left",
                      content=tf.flow(el)["content"])
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
    P.meta = {"template": f"row-s{item_scale}-c{len(tcols)}" + ("-rot" if rotate else ""), "t": head_t}
    return P


def _inner_row(ctx, P, units, rect, t):
    """Row inside a frame. One piece: icon | text. Two or more: the pieces
    flank the text (first one left, second right, alternating), and the text
    column sits centered between them. Returns the text scale, or None."""
    rx, ry, rw, rh = rect
    texts = sorted([u for u in units if u.kind == "text"], key=lambda u: (u.bbox[1], u.bbox[0]))
    rest = sorted([u for u in units if u.kind != "text"], key=lambda u: (u.bbox[1], u.bbox[0]))
    if not rest or not texts:
        return None
    g = 0.06 * rh
    sized = []
    for u in rest:
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
        sized.append((b, s, w, h))
    symmetric = len(sized) >= 2
    left = sized[0::2] if symmetric else sized
    right = sized[1::2] if symmetric else []
    wl = sum(x[2] + g for x in left)
    wr = sum(x[2] + g for x in right)
    side = max(wl, wr) if symmetric else wl
    col_w = rw - side * (2 if symmetric else 1)
    if col_w < 0.25 * rw:
        return None
    cap = max(u.els[0]["font_size"] for u in texts) * t * 1.6
    ms = _fit_column(texts, col_w, rh, cap, ctx.floor)
    if ms is None:
        return None
    tw = max(m["width"] for m in ms)
    th = sum(m["height"] for m in ms) + sum(m["fs"] * 0.3 for m in ms[1:])
    cy = ry + rh / 2
    align = "center" if symmetric else "left"
    if symmetric:
        cx = rx + rw / 2                      # text centered in the frame
        tx = cx - tw / 2
        lx_end, rx_start = tx - g, tx + tw + g
    else:
        tx = rx + (rw - wl - tw) / 2 + wl
        lx_end, rx_start = tx - g, None

    def put(b, s, x, w, h):
        if isinstance(b, ImageBlock):
            b.place_vis(P, x, cy - h / 2, s)
        elif isinstance(b, GroupBlock):
            b.place_at(P, x, cy - h / 2, s)
        else:
            b.place(P, x, cy - h / 2, s, w, "left")
    x = lx_end
    for b, s, w, h in reversed(left):        # pack leftwards from the text
        x -= w
        put(b, s, x, w, h)
        x -= g
    x = rx_start if rx_start is not None else 0
    for b, s, w, h in right:
        put(b, s, x, w, h)
        x += w + g
    y = cy - th / 2
    for u, m in zip(texts, ms):
        el = u.els[0]
        # a little slack around the column, but never past the frame's inner edges
        bx = max(rx, tx - m["fs"] / 2)
        bw = min(tw + m["fs"], rx + rw - bx)
        if align == "center":
            bw = 2 * min(tx + tw / 2 - bx, bx + bw - (tx + tw / 2))
            bx = tx + tw / 2 - bw / 2
        P.set(el["id"], x=bx, y=y, width=bw, height=m["height"], font_size=m["fs"], text_align=align,
              content=tf.flow(el)["content"])
        P.vis[el["id"]] = tf.visual_box(dict(el, text_align=align), bx, y, bw, m["fs"], m)
        y += m["height"] + m["fs"] * 0.3
    return ms[0]["fs"] / texts[0].els[0]["font_size"]


def inner_layout(ctx, P, units, rect, t):
    """Lay out a frame's contents inside rect, trying both a column and a row
    (icons beside a text column); keep whichever lets the text be larger."""
    options = []
    col = Placed()
    tt = stack(ctx, col, units, rect, "center", t * 1.6)
    if tt is not None:
        options.append((tt * 1.5, "column", col))  # keeps the designer's structure unless a row is much larger
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


def place_photo_panel(ctx, P, hero, rect, side):
    """Crop a hero that fills its box to fill `rect`, a full-bleed band along one
    canvas edge. The photo keeps its proportions; it is scaled to cover the
    band and slides along the band so its subject sits mid-band. Any excess
    runs off the canvas edges the band touches, never into the content."""
    from . import saliency
    el = hero.els[0]
    info = saliency.analyze(el.get("asset_id") or "")
    if info is None:
        return False
    rx, ry, rw, rh = rect
    s = max(rw / el["width"], rh / el["height"])
    bw, bh = el["width"] * s, el["height"] * s
    sb = info["sal_box"]
    concentrated = (sb[2] - sb[0]) * (sb[3] - sb[1]) < 0.45
    fx, fy = info["centroid"] if concentrated else (0.5, 0.5)
    # a diffuse subject still has an extent: follow it along an axis it doesn't fill
    if not concentrated:
        if sb[2] - sb[0] < 0.8:
            fx = (sb[0] + sb[2]) / 2
        if sb[3] - sb[1] < 0.8:
            fy = (sb[1] + sb[3]) / 2
    if side in ("left", "right"):
        # vertical slide is free (top and bottom are canvas edges)
        y = min(max(ry + rh / 2 - fy * bh, ry + rh - bh), ry)
        # horizontally the inner edge is pinned to the band; excess goes outward
        x = rx + rw - bw if side == "left" else rx
    else:
        x = min(max(rx + rw / 2 - fx * bw, rx + rw - bw), rx)
        y = ry + rh - bh if side == "top" else ry
    P.set(el["id"], x=x, y=y, width=bw, height=bh)
    P.vis[el["id"]] = (rx, ry, rx + rw, ry + rh)
    P.panel = {"id": el["id"], "rect": rect, "side": side}
    return True


def _main_picture(ctx, flow):
    """If no croppable hero exists, offer the design's largest box-filling
    detailed picture as one, lifting it out of its group (the rest of the
    group stays together). Returns (flow, number of groups split)."""
    from . import saliency
    from .analyze import Unit, _fills_box
    if any(u.role == "hero" and getattr(u, "croppable", False) for u in flow):
        return flow, 0
    best = None
    for u in flow:
        for e in u.els:
            if e["type"] != "image" or not _fills_box(e):
                continue
            info = saliency.analyze(e.get("asset_id") or "")
            if info is None or info["busyness"] < 0.03:
                continue
            a = e["width"] * e["height"]
            if a >= 0.15 * ctx.Ws * ctx.Hs and (best is None or a > best[0]):
                best = (a, u, e)
    if best is None:
        return flow, 0
    _, u, e = best
    pic = Unit(e["id"], "hero", "image", [e])
    pic.croppable = True
    if len(u.els) == 1:
        return [pic if v is u else v for v in flow], 0
    rest = Unit(u.uid + ":rest", u.role, "group", [x for x in u.els if x is not e])
    return [v for v in flow if v is not u] + [pic, rest], 1


def tmpl_panel(ctx, flow, side="left", frac=0.4):
    """A croppable hero fills a full-bleed band on one side of the canvas;
    everything else is laid out in the rest (as a row on extreme banners,
    as a stack otherwise)."""
    W, H = ctx.W, ctx.H
    wide = side in ("left", "right")
    if (wide and W / H < 1.2) or (not wide and H / W < 1.2):
        return None
    flow, lifted = _main_picture(ctx, flow)
    heroes = [u for u in flow if u.role == "hero" and getattr(u, "croppable", False)]
    if not heroes:
        return None
    hero = heroes[0]
    rest = [u for u in flow if u is not hero]
    # The photo fills the band's full cross-size and is cropped only along the
    # band (the free axis, which runs off the canvas). If at full cross-size it
    # is longer than the band, the band grows to show it whole; a photo that
    # would need more than 60% of the canvas is a backdrop, not a panel.
    el = hero.els[0]
    A = el["width"] / max(1e-6, el["height"])
    natural = A * H / W if wide else (W / A) / H
    if natural > 0.6:
        return None
    frac = max(frac, natural)
    P = Placed()
    if wide:
        pw = W * frac
        rect = (0.0, 0.0, pw, H) if side == "left" else (W - pw, 0.0, pw, H)
    else:
        ph = H * frac
        rect = (0.0, 0.0, W, ph) if side == "top" else (0.0, H - ph, W, ph)
    if not place_photo_panel(ctx, P, hero, rect, side):
        return None
    m = ctx.margin
    if wide:
        cx0 = rect[2] if side == "left" else 0.0
        cw = W - rect[2]
        if cw / H >= 2.2:
            R = tmpl_row(ctx, rest, 1.0, 1, area=(cx0, cw), P=P, rotate=getattr(ctx, "rotate", False))
            if R is None:
                R = tmpl_row(ctx, rest, 0.7, 1, area=(cx0, cw), P=P, rotate=getattr(ctx, "rotate", False))
            if R is None:
                return None
            t = R.meta["t"]
        else:
            t = stack(ctx, P, rest, (cx0 + m, m, cw - 2 * m, H - 2 * m), "center", t_cap(ctx, cw))
    else:
        cy0 = rect[3] if side == "top" else 0.0
        ch = H - rect[3]
        t = stack(ctx, P, rest, (m, cy0 + m, W - 2 * m, ch - 2 * m), "center", t_cap(ctx, W))
    if t is None:
        return None
    P.meta = {"template": f"panel-{side}-{frac}" + ("-lifted" if lifted else ""), "t": t}
    P.lifted = lifted
    return P


def tmpl_backdrop(ctx, flow, align="center", region="full"):
    """A full-bleed hero photo becomes the backdrop (cropped to cover the
    canvas in finalize, like a background); the rest is stacked over it."""
    heroes = [u for u in flow if u.role == "hero" and getattr(u, "backdrop", False)]
    if not heroes:
        return None
    hero = heroes[0]
    P = tmpl_stack(ctx, [u for u in flow if u is not hero], align, region)
    if P is None:
        return None
    P.backdrop = hero
    P.meta["template"] = "backdrop-" + P.meta["template"]
    return P


def _picture_card_panel(ctx, flow, side, frac):
    """A group made of a plain card carrying a picture (an illustration on a
    coloured card) becomes the band: the card fills the band (cropped, never
    stretched) and the picture scales with the band's cross size."""
    from . import saliency
    from .analyze import _fills_box
    W, H = ctx.W, ctx.H
    wide = side in ("left", "right")
    best = None
    for u in flow:
        if u.kind != "group":
            continue
        cards = [e for e in u.els if e["type"] == "image" and _fills_box(e)
                 and (saliency.analyze(e.get("asset_id") or "") or {}).get("plain", 0) >= 0.85]
        pics = [e for e in u.els if e["type"] == "image"
                and (saliency.analyze(e.get("asset_id") or "") or {}).get("plain", 1) < 0.85]
        if not cards or not pics:
            continue
        c = max(cards, key=lambda e: e["width"] * e["height"])
        b = u.bbox
        if c["width"] * c["height"] < 0.5 * (b[2] - b[0]) * (b[3] - b[1]):
            continue
        if best is None or c["width"] * c["height"] > best[1]["width"] * best[1]["height"]:
            best = (u, c)
    if best is None:
        return None
    u, c = best
    P = Placed()
    if wide:
        rect = (0.0, 0.0, W * frac, H) if side == "left" else (W - W * frac, 0.0, W * frac, H)
        s = H / c["height"]
    else:
        rect = (0.0, 0.0, W, H * frac) if side == "top" else (0.0, H - H * frac, W, H * frac)
        s = W / c["width"]
    rx, ry, rw, rh = rect
    # the group keeps its arrangement around the card's centre, at the band's scale
    ccx, ccy = c["x"] + c["width"] / 2, c["y"] + c["height"] / 2
    tx, ty = rx + rw / 2, ry + rh / 2
    from .blocks import GroupBlock
    gb = GroupBlock(u)
    gb.place_at(P, tx + (gb.x0 - ccx) * s, ty + (gb.y0 - ccy) * s, s)
    P.set(c["id"], x=rx, y=ry, width=rw, height=rh, fit="cover")
    P.vis[c["id"]] = rect
    P.panel = {"id": c["id"], "rect": rect, "side": side}
    rest = [v for v in flow if v is not u]
    m = ctx.margin
    if wide:
        ox = rect[2] if side == "left" else 0.0
        area = (ox + m, m, W - rect[2] - 2 * m, H - 2 * m)
    else:
        oy = rect[3] if side == "top" else 0.0
        area = (m, oy + m, W - 2 * m, H - rect[3] - 2 * m)
    t = stack(ctx, P, rest, area, src_align(rest), t_cap(ctx, area[2])) if rest else 1.0
    if t is None:
        return None
    P.meta = {"template": f"cardpanel-{side}-{frac}", "t": t}
    return P


def tmpl_card_panel(ctx, flow, side="left", frac=0.5):
    """A card (a frame whose container is a flat, box-filling piece) becomes a
    full-bleed band on one side, its text reflowed inside; the background
    shows on the other side. The card is cropped to the band (excess runs off
    the canvas edges it touches), never stretched."""
    from .blocks import FrameBlock
    W, H = ctx.W, ctx.H
    wide = side in ("left", "right")
    if (wide and W / H < 1.2) or (not wide and H / W < 1.2):
        return None
    frames = [u for u in flow if u.kind == "frame"]
    if not frames:
        return _picture_card_panel(ctx, flow, side, frac)
    card = max(frames, key=lambda u: u.container["width"] * u.container["height"])
    fb = FrameBlock(card, ctx)
    if not fb.crop_card and card.container["type"] != "shape":
        return None
    P = Placed()
    if wide:
        bw, bh = W * frac, H
        rect = (0.0, 0.0, bw, bh) if side == "left" else (W - bw, 0.0, bw, bh)
    else:
        bw, bh = W, H * frac
        rect = (0.0, 0.0, bw, bh) if side == "top" else (0.0, H - bh, bw, bh)
    c = card.container
    rx, ry, rw, rh = rect
    if c["type"] == "image":
        s = max(rw / c["width"], rh / c["height"])
        cw_, ch_ = c["width"] * s, c["height"] * s
        # excess runs outward only: pinned to the band's inner edge
        if wide:
            cx = rx + rw - cw_ if side == "left" else rx
            cy = ry + (rh - ch_) / 2
        else:
            cx = rx + (rw - cw_) / 2
            cy = ry + rh - ch_ if side == "top" else ry
        P.set(c["id"], x=cx, y=cy, width=cw_, height=ch_)
    else:
        P.set(c["id"], x=rx, y=ry, width=rw, height=rh)
    P.vis[c["id"]] = rect
    P.panel = {"id": c["id"], "rect": rect, "side": side}
    m = ctx.margin
    inner = (rx + m, ry + m, rw - 2 * m, rh - 2 * m)
    P.frame_modes = {}
    P.frame_modes[card.uid] = inner_layout(ctx, P, card.children, inner, 1.25 * ctx.t_star)
    if P.frame_modes[card.uid] is None:
        return None
    rest = [u for u in flow if u is not card]
    t = 1.0
    pics = [u for u in rest if u.role == "hero" and getattr(u, "croppable", False)]
    others = [u for u in rest if u not in pics and u.role not in ("aligned", "minor")]
    if pics and not others:
        # the card on one side, the picture filling the other edge to edge (no white margins)
        if wide:
            r2 = (rect[2], 0.0, W - rect[2], H) if side == "left" else (0.0, 0.0, W - rect[2], H)
            s2 = "right" if side == "left" else "left"
        else:
            r2 = (0.0, rect[3], W, H - rect[3]) if side == "top" else (0.0, 0.0, W, H - rect[3])
            s2 = "bottom" if side == "top" else "top"
        P2 = Placed()
        if place_photo_panel(ctx, P2, pics[0], r2, s2):
            for k, v in P2.props.items():
                P.set(k, **v)
            P.vis.update(P2.vis)
            P.panels = [P.panel, P2.panel]
            rest = [u for u in rest if u not in pics]
    if rest:
        if wide:
            ox = rect[2] if side == "left" else 0.0
            t = stack(ctx, P, rest, (ox + m, m, W - rect[2] - 2 * m, H - 2 * m), src_align(rest), t_cap(ctx, W))
        else:
            oy = rect[3] if side == "top" else 0.0
            t = stack(ctx, P, rest, (m, oy + m, W - 2 * m, H - rect[3] - 2 * m), src_align(rest), t_cap(ctx, W))
        if t is None:
            return None
    P.meta = {"template": f"cardpanel-{side}-{frac}", "t": t}
    return P


def src_align(units):
    """The alignment most of the text (by ink area) had in the source."""
    tot = {}
    for u in units:
        for e in u.els:
            if e["type"] == "text":
                a = e.get("text_align", "left")
                tot[a] = tot.get(a, 0.0) + e["width"] * e["height"]
    if not tot:
        return "center"
    a = max(tot, key=tot.get)
    return "left" if a == "left" else "center"


def _columns(units, k):
    """Contiguous splits of units (reading order) into k columns, most
    balanced first (by source area). Yields lists of columns."""
    import itertools
    n = len(units)
    if k > n or k < 1:
        return []
    area = [max(1.0, (u.bbox[2] - u.bbox[0]) * (u.bbox[3] - u.bbox[1])) for u in units]
    out = []
    for cuts in itertools.combinations(range(1, n), k - 1):
        b = (0,) + cuts + (n,)
        cols = [units[b[i]:b[i + 1]] for i in range(k)]
        sums = [sum(area[b[i]:b[i + 1]]) for i in range(k)]
        out.append((max(sums) / max(1.0, min(sums)), cols))
    out.sort(key=lambda x: x[0])
    return [c for _, c in out[:2]]


def _stack_t(ctx, units, rect, align):
    """Largest text scale at which units fit in rect (dry run)."""
    return stack(ctx, Placed(), units, rect, align, t_cap(ctx, rect[2]))


def tmpl_grid(ctx, flow, k=2, tiers=False, panel=None, align=None):
    """Units in reading order split into k side-by-side columns sharing one
    text scale. tiers: the headline (and subhead/logo) take a band across
    the top first. panel: 'left'/'right' puts a croppable main picture as a
    full-bleed band on that side. Every text keeps its own line breaks."""
    W, H = ctx.W, ctx.H
    flow = [u for u in flow if u.role != "aligned"]
    align = align or src_align(flow)
    P = Placed()
    m = ctx.margin
    x0, x1 = m, W - m
    if panel:
        flow2, lifted = _main_picture(ctx, flow)
        heroes = [u for u in flow2 if u.role == "hero" and getattr(u, "croppable", False)]
        if not heroes:
            return None
        hero = heroes[0]
        el = hero.els[0]
        A = el["width"] / max(1e-6, el["height"])
        frac = min(0.45, max(0.25, A * H / W))
        pw = W * frac
        rect = (0.0, 0.0, pw, H) if panel == "left" else (W - pw, 0.0, pw, H)
        if not place_photo_panel(ctx, P, hero, rect, panel):
            return None
        P.lifted = lifted
        flow = [u for u in flow2 if u is not hero]
        if panel == "left":
            x0 = pw + m
        else:
            x1 = W - pw - m
    if not flow or x1 - x0 < 0.2 * W:
        return None
    order = sorted(flow, key=lambda u: (u.bbox[1], u.bbox[0]))
    y0, y1 = m, H - m
    top = []
    if tiers:
        top = [u for u in order if u.role in ("headline", "subhead", "logo")]
        rest = [u for u in order if u not in top]
        if not top or not rest:
            return None
        th = 0.42 * (y1 - y0)
        t_top = stack(ctx, P, top, (x0, y0, x1 - x0, th), align, t_cap(ctx, x1 - x0))
        if t_top is None:
            return None
        y0 = y0 + th + 0.04 * H
        order = rest
    if len(order) < k:
        return None
    gut = 0.03 * W
    best = None
    for cols in _columns(order, k):
        widths = [max(u.bbox[2] - u.bbox[0] for u in c) for c in cols]
        tot = sum(widths)
        avail = x1 - x0 - gut * (k - 1)
        rects, x = [], x0
        for wv in widths:
            cw = avail * (0.5 * wv / tot + 0.5 / k)
            rects.append((x, y0, cw, y1 - y0))
            x += cw + gut
        ts = [_stack_t(ctx, c, r, align) for c, r in zip(cols, rects)]
        if any(t is None for t in ts):
            continue
        tc = min(ts)
        if best is None or tc > best[0]:
            best = (tc, cols, rects)
    if best is None:
        return None
    tc, cols, rects = best
    for c, r in zip(cols, rects):
        # each column as large as it can be, but within 1.3x of the tightest one
        tk = _stack_t(ctx, c, r, align) or tc
        if stack(ctx, P, c, r, align, min(tk, 1.3 * tc)) is None:
            return None
    P.meta = {"template": f"grid-{k}" + ("-tiers" if tiers else "") + (f"-panel-{panel}" if panel else ""),
              "t": tc}
    return P


# ---------------------------------------------------------------------------
# rows: re-stack into rows (strips and any canvas where nothing else is legible)


def _unit_floor_scale(u, floor):
    """The least scale at which a unit's smallest text reaches the type floor (0 for no text)."""
    fs = _src_fs(u)
    return floor / fs if fs else 0.0


def _centre_in(t, sh, tol=2.0):
    """A text's centre lies inside a shape's own (unturned) box: both turn about their centres."""
    cx, cy = t["x"] + t["width"] / 2, t["y"] + t["height"] / 2
    return sh["x"] - tol <= cx <= sh["x"] + sh["width"] + tol and sh["y"] - tol <= cy <= sh["y"] + sh["height"] + tol


def _atoms(ctx, flow, fine=False):
    """The design taken apart for the last-resort re-stack: along clean cuts only as far as a part
    is still too big at the floor (`fine` False), or all the way to its smallest readable pieces
    (`fine` True): a shape with the text it carries (a button, a badge) stays one piece; every
    other text, image and shape becomes its own piece. A composition that mixes display type with
    small labels can then give each its own size instead of scaling them as one rigid block, and
    a rigid group whose re-fitted copy outgrows its box can no longer run into its neighbours."""
    out = []
    for u in flow:
        for a in _split(ctx, u):
            if fine and a.kind not in ("text", "image") and len(a.ids) > 1:
                for p in _pieces(a):    # a part's pieces stay together, in the part's place
                    p.parent = a
                    out.append(p)
            else:
                out.append(a)
    return out


def _is_button(u) -> bool:
    """One shape carrying all the unit's other pieces, which are texts centred in it."""
    if u.kind not in ("group", "button", "ribbon") or len(u.els) < 2:
        return False
    shapes = [e for e in u.els if e["type"] == "shape"]
    return len(shapes) == 1 and all(e["type"] == "text" and _centre_in(e, shapes[0])
                                    for e in u.els if e is not shapes[0])


def _fits_at_floor(ctx, u) -> bool:
    """A piece can stay whole: scaled until its smallest text reaches the floor, it still fits
    comfortably on the canvas."""
    s = _unit_floor_scale(u, ctx.floor)
    b = u.bbox
    m = min(ctx.margin, 0.12 * min(ctx.W, ctx.H) + 0.01 * max(ctx.W, ctx.H))
    if (b[2] - b[0]) * s > ctx.W - 2 * m or (b[3] - b[1]) * s > ctx.H - 2 * m:
        return False
    from .blocks import GroupBlock
    if u.kind not in ("group", "frame"):
        return True
    trial = Placed()                 # set as a rigid group, its texts re-fit: do they hold the floor?
    GroupBlock(u).place_at(trial, 0.0, 0.0, s)
    return all(p.get("font_size", ctx.floor) >= ctx.floor for p in trial.props.values())


def _split(ctx, u, depth=0):
    """Take a piece apart only as far as it must: along clean cuts (a card row into its cards,
    a caption staying with its picture) while a part is still too big at the floor; down to
    single pieces only where no clean cut exists."""
    from .analyze import Unit
    from .blocks import find_cut
    if u.kind in ("text", "image") or len(u.ids) <= 1:
        return [u]
    if u.kind == "badges":           # a row of pills: each pill rebuilt around its own label
        return [p for m in u.members for p in _split(ctx, m, depth + 1)]
    if _is_button(u):
        return _pieces(u)            # a button: rebuilt around its label, never a rigid scale
    fss = [e["font_size"] for e in u.els if e["type"] == "text"]
    even = not fss or max(fss) <= 2.5 * min(fss) or all(e["type"] == "text" for e in u.els)
    # a card (a picture with its caption, type of like sizes), a type lockup (texts only) or a part
    # a clean cut produced may stay whole; display type composed with small pills is taken apart
    if (depth > 0 or even) and _fits_at_floor(ctx, u):
        return [u]
    if u.kind in ("group", "frame") and depth < 6:
        els = list(u.els) + ([u.container] if u.kind == "frame" and u.container is not None
                             and u.container not in u.els else [])
        cut = find_cut(Unit(u.uid, u.role, "group", els), ctx.Ws * ctx.Hs)
        if cut:
            parts = []
            for i, side in enumerate(cut[1:]):
                kind = "group" if len(side) > 1 else ("text" if side[0]["type"] == "text" else
                                                      "image" if side[0]["type"] == "image" else "group")
                parts += _split(ctx, Unit(f"{u.uid}/{i}", u.role, kind, list(side)), depth + 1)
            return parts
    return _pieces(u)


def _overlap_share(a, b):
    """The share of a's (unturned) box that lies on b's."""
    ix = min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"])
    iy = min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"])
    return max(0.0, ix) * max(0.0, iy) / max(1e-6, a["width"] * a["height"])


def _pieces(u):
    """Single pieces: a shape with the text it carries is one (a button); all else alone."""
    from .analyze import Unit
    out = []
    els = [e for m in u.members for e in m.els] if u.kind == "badges" else list(u.els)
    if u.kind == "frame" and u.container is not None and u.container not in els:
        els = [u.container] + els
    texts = [e for e in els if e["type"] == "text"]
    taken = set()
    for sh in sorted((e for e in els if e["type"] == "shape"), key=lambda e: e["width"] * e["height"]):
        mine = [t for t in texts if t["id"] not in taken and _centre_in(t, sh)
                and abs(float(t.get("rotation") or 0) - float(sh.get("rotation") or 0)) < 1   # turned with it
                and (t.get("group_id") and t.get("group_id") == sh.get("group_id") or
                     t["width"] * t["height"] >= 0.12 * sh["width"] * sh["height"]
                     and _overlap_share(t, sh) >= 0.5)]   # a shape carries copy that lies on it
        if mine:
            taken.update(t["id"] for t in mine)
            taken.add(sh["id"])
            b = Unit(f"{u.uid}:{sh['id']}", u.role, "group", [sh] + mine)
            b.button = len(mine) == 1 and float(mine[0].get("rotation") or 0) == float(sh.get("rotation") or 0) \
                or (not sh.get("rotation") and not any(t.get("rotation") for t in mine))
            out.append(b)
    # copy set over copy (a drop shadow, an outline under its fill) is one piece: apart, it reads twice
    for t in texts:
        if t["id"] in taken:
            continue
        twins = [v for v in texts if v is not t and v["id"] not in taken
                 and str(v.get("content", "")) == str(t.get("content", ""))
                 and abs(float(v["font_size"]) - float(t["font_size"])) <= 0.1 * float(t["font_size"])
                 and _overlap_share(t, v) >= 0.5 and _overlap_share(v, t) >= 0.5]
        if twins:
            taken.update([t["id"]] + [v["id"] for v in twins])
            out.append(Unit(f"{u.uid}:{t['id']}", u.role, "group", [t] + twins))
    for e in els:
        if e["id"] in taken:
            continue
        kind = "text" if e["type"] == "text" else ("image" if e["type"] == "image" else "group")
        out.append(Unit(f"{u.uid}:{e['id']}", u.role, kind, [e]))
    return out


def _with_carriers(ctx, flow):
    """A shape read as decoration that carries copy of its own group (a pill whose label the
    reading put elsewhere) joins the piece holding that copy, so a re-stack rebuilds them together
    instead of leaving the pill where decoration goes and the label in a row. Returns the flow and
    the ids it now holds that finalize would otherwise place."""
    from .analyze import Unit
    flow = list(flow)
    held = set()
    for d in ctx.model["units"]:
        if d.role != "deco" or len(d.els) != 1 or d.els[0]["type"] != "shape" or not d.els[0].get("group_id"):
            continue
        sh = d.els[0]
        for i, u in enumerate(flow):
            if u.kind in ("badges", "frame"):
                continue
            if any(t["type"] == "text" and t.get("group_id") == sh["group_id"] and _centre_in(t, sh) for t in u.els):
                nu = Unit(u.uid, u.role, "group", list(u.els) + [sh])
                flow[i] = nu
                held.add(sh["id"])
                break
    return flow, held


def tmpl_rows(ctx, flow, order="reading", cap=1.0, align="center", axis="rows", atom=False):
    """Every flow unit, packed in order into as many rows (or columns) as the canvas needs (a
    flex-wrap), all at one scale t: texts at max(floor, size x t), wrapping at most `cap` of the
    canvas width; groups and badges never so small their copy falls under the floor; pictures at
    t, but never shrunk to a speck. The largest t that fits wins. A strip gets legible copy in a
    few rows or columns instead of one row of unreadable copy (STEERING S15); nothing is hidden
    (S1). When even the floor cannot hold everything, the floor steps down (to 60% of it, never
    under 6 px; then 5, 4, 3 px as a last resort) rather than any element going.
    axis: "rows" fills left to right, then wraps down; "cols" fills top to bottom, then wraps
    right (a wide strip). order: "reading" keeps the source's reading order; "role" puts the
    logo first and the call to action last, as banners read."""
    # ribbons and corner pieces carrying copy are packed too: a re-stack has no free corner to
    # promise them, and a float that hops corners on a 1 px change is no layout
    flow = list(flow) + [u for u in ctx.model["units"] if u.role in ("ribbon", "anchored")
                         and any(e["type"] == "text" for e in u.els) and u not in flow]
    flow, held = _with_carriers(ctx, flow)
    extra = 0
    if atom:
        clean = _atoms(ctx, flow)
        flow = _atoms(ctx, flow, fine=True) if atom == "pieces" else clean
        extra = len(flow) - len(clean)   # compositions taken apart beyond the clean cuts
    floors = sorted({max(6, int(ctx.floor * k)) for k in (1.0, 0.9, 0.8, 0.7, 0.6)}, reverse=True)
    for floor in floors + [5, 4, 3]:   # under 6 px only when nothing else keeps every element
        if floor > 3 and not _may_hold(ctx, flow, floor):   # 3 px stays the last resort
            continue
        # margins and gaps tighten before the type gets any smaller
        P = None
        for tight in (1.0, 0.5, 0.25):
            P = _rows(ctx, flow, order, cap, align, axis, floor, tight)
            if P is not None:
                break
        if P is not None:
            if atom:
                P.meta["template"] += "-pieces" if atom == "pieces" else "-atoms"
            P.held = held
            if extra > 0:
                P.lifted = extra             # each costs like any composition taken apart
            return P
    return None


def floor_area(els, floor) -> float:
    """A lower bound on the area the copy needs at the type floor: however a text wraps, its
    block is at least its one-line length times its line height (a wrap only adds slack)."""
    need = 0.0
    for e in els:
        if e["type"] == "text" and str(e.get("content", "")).strip():
            m = tf.measure(e, int(floor), 1e9)
            need += m["width"] * m["height"]
    return need


def _may_hold(ctx, flow, floor) -> bool:
    """Whether the canvas can hold all of the flow's copy at this floor at all (with the tightest
    margins a re-stack uses). Cached per canvas: every re-stack variant asks the same."""
    cache = ctx.__dict__.setdefault("_hold", {})
    els = {e["id"]: e for u in flow for e in u.els}
    key = (floor, tuple(sorted(els)))
    if key not in cache:
        m = 0.25 * min(ctx.margin, 0.12 * min(ctx.W, ctx.H) + 0.01 * max(ctx.W, ctx.H))
        cache[key] = floor_area(els.values(), floor) <= (ctx.W - 2 * m) * (ctx.H - 2 * m)
    return cache[key]


def _under(a, b) -> bool:
    """b is set directly under a: they share most of the narrower one's width and only a small
    gap runs between them (a subhead under its headline)."""
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    share = min(ax1, bx1) - max(ax0, bx0)
    narrow = min(ax1 - ax0, bx1 - bx0)
    gap = by0 - ay1
    return narrow > 0 and share >= 0.5 * narrow and -0.1 * (ay1 - ay0) <= gap <= 0.35 * max(ay1 - ay0, by1 - by0)


def _under_follows(units):
    """Reading order, except that a piece set directly under another follows it at once: rows
    bucketed by centre height would otherwise let a piece to the side cut in between."""
    out, left = [], list(units)
    while left:
        u = left.pop(0)
        out.append(u)
        while True:
            nxt = [v for v in left if _under(u, v)]
            if not nxt:
                break
            v = min(nxt, key=lambda v: v.bbox[1])
            left.remove(v)
            out.append(v)
            u = v
    return out


GROW = 1.06   # a text's slot is this much taller than measured: the Linux renderer may set it taller


def _same_lines(b, t, w, wrap):
    """The narrowest box, from the drawn width w up to the packing's wrap width, that breaks the
    text into the same lines as the packing measured: re-wrapping at the drawn width alone can
    push a word down (a greedy wrap counts the space before a break), and a text one line taller
    than its slot runs into the next piece."""
    n = b._fit(t, wrap)[2]["lines"]
    lo, hi = w + 1.0, max(wrap, w + 1.0)
    if b._fit(t, lo)[2]["lines"] <= n:
        return lo
    for _ in range(10):
        mid = (lo + hi) / 2
        if b._fit(t, mid)[2]["lines"] <= n:
            hi = mid
        else:
            lo = mid
    return hi


def _text_lines(b, t, max_w):
    """A text's width and the height of its line boxes (what the renderer occupies, taller than
    the inked cap-to-descender height a stack aligns on): rows must not let line boxes touch."""
    box_w, fs, m = b._fit(t, max_w)
    asc, capk, desc = tf.metrics(b.el)
    lh = float(b.el.get("line_height", 1.2))
    w = max(m["width"], m["word_w"])     # tracked lines draw wider than the wrap they broke at
    return (w if b.el.get("letter_spacing") else min(box_w, w)), (m["lines"] - 1) * fs * lh + max(fs * lh, (asc + desc) * fs)


def _rows(ctx, flow, order, cap, align, axis, floor, tight=1.0):
    from .blocks import BadgesBlock, FrameBlock, GroupBlock, ImageBlock, RestackBlock, TextBlock
    if not flow:       # aligned pieces are packed too: a re-stack has no anchor line to keep
        return None
    W, H = ctx.W, ctx.H
    m = tight * min(ctx.margin, 0.12 * min(W, H) + 0.01 * max(W, H))
    AW, AH = W - 2 * m, H - 2 * m
    if AW <= 0 or AH <= 0:
        return None
    def at(u):                       # pieces of one part read where their part reads
        return getattr(u, "parent", None) or u
    if order == "role":
        units = sorted(flow, key=lambda u: (ROW_ORDER.get(u.role, 3), at(u).bbox[1], at(u).bbox[0], u.bbox[1], u.bbox[0]))
    else:
        units = _under_follows(sorted(flow, key=lambda u: (round(at(u).cy / max(1.0, 0.04 * ctx.Hs)), at(u).bbox[0],
                                                           round(u.cy / max(1.0, 0.04 * ctx.Hs)), u.bbox[0])))
    blocks = []
    from .blocks import ButtonBlock
    for u in units:
        if not hasattr(u, "button"):
            u.button = _is_button(u) and len({float(e.get("rotation") or 0) for e in u.els}) == 1
        b = ButtonBlock(u, floor) if u.button else make_block(u, ctx.Ws, ctx)
        if isinstance(b, TextBlock):
            b.wfrac = 1.0
            b.floor = floor
        blocks.append(b)
    for b in blocks:                 # a turned pill too big for the space even at the floor turns upright
        if isinstance(b, ButtonBlock) and b.rot:
            bw, bh = b.size(1e-4, AW)
            if bw > AW + 0.5 or bh > AH + 0.5:
                b.set_upright()
    gap = tight * max(0.6 * floor, 0.025 * min(W, H))
    speck = 0.18 * min(AW, AH)       # a picture's long side never under this
    cols = axis in ("cols", "fit")
    main_len, cross_len = (AH, AW) if cols else (AW, AH)

    def scale_of(u, b, t):
        if isinstance(b, ImageBlock):
            return max(t, speck / max(1e-6, max(b.vw, b.vh)))
        if isinstance(b, (GroupBlock, BadgesBlock, RestackBlock, FrameBlock)):
            return max(t, _unit_floor_scale(u, floor))
        return t

    texts = [(u.els[0]["font_size"], b) for u, b in zip(units, blocks) if isinstance(b, TextBlock)]

    def dims(b, sc, hmax):
        """A piece's (width, height) in a column at most hmax tall: a text wraps as narrow as
        that height allows (the narrowest wrap width whose lines still fit), anything else at
        its size. None when it cannot fit at all."""
        if not isinstance(b, TextBlock):
            w, h = b.size(sc, AW)
            return (w, h, AW) if w <= AW + 0.5 and h <= hmax + 0.5 else None
        w, h = _text_lines(b, sc, AW)
        h *= GROW
        if w > AW + 0.5 or h > hmax + 0.5:
            return None
        fs = b._fit(sc, AW)[1]
        lo, hi, best = 1.0, AW, (w, h, AW)
        for _ in range(12):
            mid = (lo + hi) / 2
            w2, h2 = _text_lines(b, sc, mid)
            h2 *= GROW
            # never so narrow that the longest word forces the type smaller
            if h2 <= hmax + 0.5 and w2 <= AW + 0.5 and b._fit(sc, mid)[1] >= fs:
                hi, best = mid, (w2, h2, mid)
            else:
                lo = mid
        return best

    def pack_fit(t):
        """Columns across a wide strip (the strip's height is the scarce side): each piece takes
        the narrowest shape that fits the column, and joins the current column under the last
        piece when that widens the column less than a column of its own would take."""
        lines, cur, ch, cw, thick = [], [], 0.0, 0.0, []
        for u, b in zip(units, blocks):
            sc = scale_of(u, b, t)
            full = dims(b, sc, AH)
            if full is None:
                return None
            put = None
            if cur and AH - ch - gap > 0:
                st = dims(b, sc, AH - ch - gap)
                if st is not None and max(cw, st[0]) - cw <= full[0] + gap:
                    put = st
            if put is None:
                if cur:
                    lines.append((cur, ch))
                    thick.append(cw)
                cur, ch, cw = [], 0.0, 0.0
                put = full
            ch += (gap if cur else 0.0) + put[1]
            cw = max(cw, put[0])
            cur.append((u, b, sc, put[0], put[1], put[2]))
        if cur:
            lines.append((cur, ch))
            thick.append(cw)
        return (lines, thick) if sum(thick) + gap * (len(lines) - 1) <= cross_len + 0.5 else None

    def pack(t):
        # hierarchy: a button label stays no larger than any text set >= 1.25x larger than it
        tfs = [(f0, b._fit(scale_of(None, b, t), AW * cap)[1]) for f0, b in texts]
        for u, b in zip(units, blocks):
            if isinstance(b, ButtonBlock):
                above = [fs for f0, fs in tfs if f0 >= 1.25 * b.fs0]
                b.max_fs = min(above) if above else None
        if axis == "fit":
            return pack_fit(t)
        lines, cur, cl = [], [], 0.0
        for u, b in zip(units, blocks):
            sc = scale_of(u, b, t)
            if isinstance(b, TextBlock):
                w, h = _text_lines(b, sc, AW * cap)
                h *= GROW
            else:
                w, h = b.size(sc, AW)
            if w > AW + 0.5 or h > AH + 0.5:
                return None
            along = h if cols else w
            if cur and cl + gap + along > main_len:
                lines.append((cur, cl))
                cur, cl = [], 0.0
            cl += (gap if cur else 0.0) + along
            cur.append((u, b, sc, w, h, AW * cap))
        if cur:
            lines.append((cur, cl))
        thick = [max((it[3] if cols else it[4]) for it in ln[0]) for ln in lines]
        return (lines, thick) if sum(thick) + gap * (len(lines) - 1) <= cross_len + 0.5 else None

    hi = t_cap(ctx, AW) * 1.5
    if pack(hi):
        lo = hi
    else:
        if not pack(1e-4):
            # legible copy before a pill's tilt: turned pills go upright, then try again
            turned = [b for b in blocks if isinstance(b, ButtonBlock) and b.rot]
            for b in turned:
                b.set_upright()
            if not turned or not pack(1e-4):
                return None
        lo = 1e-4
        for _ in range(16):
            mid = (lo + hi) / 2
            if pack(mid):
                lo = mid
            else:
                hi = mid
    lines, thick = pack(lo)
    P = Placed()
    tot = sum(thick) + gap * (len(lines) - 1)
    c = m + (cross_len - tot) / 2               # cross axis: the block of lines, centred
    for (ln, length), th in zip(lines, thick):
        a = m + ((main_len - length) / 2 if align == "center" or cols else 0.0)
        for u, b, sc, w, h, wrap in ln:
            if cols:
                x, y = c + ((th - w) / 2 if align == "center" else 0.0), a
            else:
                x, y = a, c + (th - h) / 2
            if isinstance(b, TextBlock):           # packed by line boxes, placed by cap top
                ww = _same_lines(b, sc, w, wrap)    # wraps where the packing measured it
                _, fs, _ = b._fit(sc, ww)
                asc, capk, _ = tf.metrics(b.el)
                b.place(P, x, y + (asc - capk) * fs, sc, ww, "center" if cols and align == "center" else "left")
            elif isinstance(b, ImageBlock):
                b.place_vis(P, x, y, b.scale_for(sc, AW))
            else:
                b.place(P, x, y, sc, w, "left")
            a += (h if cols else w) + gap
        c += th + gap
    P.meta = {"template": f"{axis}-{order}-c{cap}-{align}-n{len(lines)}-f{floor}", "t": lo}
    P.restructured = True
    return P
