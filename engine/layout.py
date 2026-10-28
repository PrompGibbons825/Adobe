"""Layout templates. Each returns a Placed (or None if it can't fit).

- stack: one centered/left column, reading order from source (tall/square)
- split: hero image in its own column, text stack beside it (wide)
- row:   everything on one line, logo | hero | text | cta (ultra-wide banners)
"""

from __future__ import annotations

from .analyze import norm_rot
from .blocks import GroupBlock, ImageBlock, Placed, TextBlock, img_vis_frac, make_block
from . import saliency
from . import textfit as tf


class Ctx:
    def __init__(self, model, target):
        self.model = model
        self.Ws, self.Hs = model["canvas"]
        self.W, self.H = target["width"], target["height"]
        self.t_star = ((self.W * self.H) / (self.Ws * self.Hs)) ** 0.5
        self.margin = 0.06 * min(self.W, self.H) + 0.01 * max(self.W, self.H)
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


def block_scale(u, t, k, boost=True, grow=False):
    if u.role == "hero":
        return t * k
    if grow:                       # the headline grows with k but never shrinks below t
        return t * max(1.0, k)
    fs = _src_fs(u)
    if boost and fs and u.role in MIN_FS:
        return max(t, MIN_FS[u.role] / fs)
    return t


def _stack_sizes(blocks, units, t, k, max_w, boost=True):
    return [b.size(block_scale(u, t, k, boost, grows(u, units)), max_w) for b, u in zip(blocks, units)]


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
    has_hero = any(grows(u, units) for u in units)
    k_min = 0.6 if any(u.role == "hero" for u in units) else 1.0

    def total(t, k):
        sizes = _stack_sizes(blocks, units, t, k, rw, boost)
        return sum(s[1] for s in sizes) + sum(g * t for g in base_gaps), sizes

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
    gaps = [g * t for g in base_gaps]
    slack = rh - tot
    if gaps and sum(gaps) > 0 and slack > 0:
        extra = min(slack, sum(gaps) * 1.5)
        f = 1 + extra / sum(gaps)
        gaps = [g * f for g in gaps]
        slack -= extra
    y = ry + max(0.0, slack) / 2
    for i, (b, u) in enumerate(zip(blocks, units)):
        b.place(P, rx, y, block_scale(u, t, k, boost, grows(u, units)), rw, align)
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


def _fit_column(texts, col_w, h_avail, fs_cap):
    """Joint fit of a column of texts: one scale t for all (keeps hierarchy)."""
    if not texts:
        return None
    head_fs = max(u.els[0]["font_size"] for u in texts)

    def layout(t):
        ms, tot = [], 0.0
        for i, u in enumerate(texts):
            el = tf.flow(u.els[0])
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
    ms = _fit_column(texts, col_w, rh, cap)
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
    concentrated = (sb[2] - sb[0]) * (sb[3] - sb[1]) < 0.45 or bool(info.get("eyes"))
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
# The same design, rescaled. Every candidate above rebuilds the layout from the
# inferred units, so a wrong grouping (a ruled grid torn into pieces) becomes a
# wrong layout. This one keeps the source's own geometry, transformed as a
# whole: the design the organizers expect to see, and the one a re-layout has
# to clearly beat.
#   center  the design scaled uniformly to fit, centred; the background fills
#   pinned  each unit keeps to its nearest edge (or the centre line) at its
#           scaled distance from it: corners stay corners
#   spread  each unit at the same uniform scale, moved to the same relative
#           place on the new canvas (uses the room when the shape changes a lot)

SAME_ROLES_SKIP = {"bg", "border"}     # laid out by finalize (cover, edge-hugging)


def tmpl_same(ctx, model, mode="center"):
    from .analyze import norm_rot
    W, H, Ws, Hs = ctx.W, ctx.H, ctx.Ws, ctx.Hs
    s = min(W / Ws, H / Hs)
    ox, oy = (W - Ws * s) / 2, (H - Hs * s) / 2
    P = Placed()
    units = [u for u in model["units"] if u.role not in SAME_ROLES_SKIP]
    blocks = _same_blocks(units, Ws, Hs) if mode != "center" else {id(u): [u] for u in units}
    # a picture covering most of the canvas with a slim inset (a photo in a
    # border) is the backdrop: it fills the new canvas at the same inset,
    # cropped, and everything set on it keeps its place relative to the canvas
    if mode != "center":
        for u in units:
            e = u.els[0]
            if len(u.els) != 1 or e["type"] != "image" or abs(norm_rot(e.get("rotation", 0) or 0)) > 1:
                continue
            l, t = e["x"], e["y"]
            r, b = Ws - e["x"] - e["width"], Hs - e["y"] - e["height"]
            if e["width"] * e["height"] >= 0.6 * Ws * Hs and max(l, t, r, b) <= 0.08 * min(Ws, Hs) \
                    and min(l, t, r, b) >= -0.02 * min(Ws, Hs):
                k = min(W, H) / min(Ws, Hs)
                x, y = l * k, t * k
                w, h = W - (l + r) * k, H - (t + b) * k
                P.set(e["id"], x=x, y=y, width=w, height=h, fit="cover")
                P.vis[e["id"]] = (x, y, x + w, y + h)
                units = [v for v in units if v is not u]
                break
    for u in units:
        # a unit moves with its block: pieces set close together in the source
        # (a ruled grid, a photo and its caption) move as one
        blk = [e for v in blocks[id(u)] for e in v.els]
        xs = [e["x"] for e in blk] + [e["x"] + e["width"] for e in blk]
        ys = [e["y"] for e in blk] + [e["y"] + e["height"] for e in blk]
        ux, uy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        if mode == "pinned":
            # per axis: a block near an edge keeps to it (its margin scaled);
            # anywhere else it keeps its offset from the centre line, so pieces
            # aligned in the source stay aligned; a block spanning the axis
            # lets each of its units do the same on their own

            def pin(lo, hi, size, new):
                near = 0.12 * size          # a designer's margin: up to about an eighth
                a, b = lo, size - hi
                if a <= near and b <= near:
                    return None
                if a <= near:
                    return a * s + (hi - lo) * s / 2
                if b <= near:
                    return new - b * s - (hi - lo) * s / 2
                return new / 2 + ((lo + hi) / 2 - size / 2) * s

            def sides(k):
                """The block split at its widest gap along axis k: (this unit's
                part, the other part), each as (lo, hi); other is None if no gap."""
                spans = sorted(((min(e["x" if k == 0 else "y"] for e in v.els),
                                 max(e["x" if k == 0 else "y"] + e["width" if k == 0 else "height"] for e in v.els))
                                for v in blocks[id(u)]))
                best, cut, reach = 0.0, None, spans[0][1]
                for lo_, hi_ in spans[1:]:
                    if lo_ - reach > best:
                        best, cut = lo_ - reach, lo_
                    reach = max(reach, hi_)
                if cut is None:
                    return (spans[0][0], max(t[1] for t in spans)), None
                me = min(e["x" if k == 0 else "y"] for e in u.els) >= cut
                mine = [t for t in spans if (t[0] >= cut) == me]
                other = [t for t in spans if (t[0] >= cut) != me]
                return (min(t[0] for t in mine), max(t[1] for t in mine)), \
                    (min(t[0] for t in other), max(t[1] for t in other))

            def axis(blo, bhi, k, size, new):
                c = pin(blo, bhi, size, new)
                if c is not None:
                    return c - (blo + bhi) / 2 * s
                (lo_, hi_), other = sides(k)
                if other is None:
                    c = new / 2 + ((lo_ + hi_) / 2 - size / 2) * s
                    return c - (lo_ + hi_) / 2 * s
                # the part nearer its edge anchors there; the other keeps its
                # source gap to it, so spare room goes to the outer margin
                def margin(t):
                    return max(0.0, min(t[0], size - t[1]))      # a box bleeding past the edge is at it
                if abs(margin((lo_, hi_)) - margin(other)) <= 0.03 * size:
                    # a tie: the picture holds its edge and the copy follows it
                    pic_side = _has_picture(blocks[id(u)], k, (lo_, hi_), Ws, Hs)
                    anchor = (lo_, hi_) if pic_side else other
                else:
                    anchor = (lo_, hi_) if margin((lo_, hi_)) <= margin(other) else other
                at_start = anchor[0] <= size - anchor[1]
                a_off = (anchor[0] * s) if at_start else (new - (size - anchor[0]) * s)
                # anchor's own shift; everything in the block shares it
                return a_off - anchor[0] * s
            dx = axis(min(xs), max(xs), 0, Ws, W)
            dy = axis(min(ys), max(ys), 1, Hs, H)
        elif mode == "spread":
            # the unit's centre keeps its relative place; its extent stays inside
            uw, uh = (max(xs) - min(xs)) * s, (max(ys) - min(ys)) * s
            cx = min(max(ux / Ws * W, uw / 2), W - uw / 2)
            cy = min(max(uy / Hs * H, uh / 2), H - uh / 2)
            dx, dy = cx - ux * s, cy - uy * s
        else:
            dx, dy = ox, oy
        for e in u.els:
            x, y, w, h = dx + e["x"] * s, dy + e["y"] * s, e["width"] * s, e["height"] * s
            if e["type"] == "text":
                fs = max(1, int(e["font_size"] * s))          # never larger: keeps the designed lines
                P.set(e["id"], x=x, y=y, width=w, height=h, font_size=fs, content=e["content"])
                if abs(norm_rot(e.get("rotation", 0) or 0)) > 1:
                    P.vis[e["id"]] = _rot_box(x, y, w, h, e.get("rotation", 0))
                else:
                    m = tf.measure(e, fs, w)
                    P.vis[e["id"]] = tf.visual_box(e, x, y, w, fs, m)
            else:
                # what ran to a canvas edge in the source still runs to it: no
                # empty band appears beside a photo or a bar. A picture is
                # cropped to its new box (fit=cover), never stretched.
                tol = 0.02 * min(Ws, Hs)
                ex0, ey0 = e["x"] <= tol, e["y"] <= tol
                ex1, ey1 = e["x"] + e["width"] >= Ws - tol, e["y"] + e["height"] >= Hs - tol
                from .analyze import _fills_box
                # only what fills its box (a photo, a flat band) stretches to the
                # edge; a shape (a circle bleeding off the canvas) keeps its form
                info_ = saliency.analyze(e.get("asset_id") or "") if e["type"] == "image" else None
                tint = info_ is not None and info_.get("alpha_std", 1.0) < 0.05     # uniform see-through layer
                if abs(norm_rot(e.get("rotation", 0) or 0)) <= 1 and (ex0 or ey0 or ex1 or ey1) and (_fills_box(e) or tint):
                    # a surface spanning one axis keeps its share of the other
                    # (a photo filling the height and two thirds of the width
                    # still fills two thirds), anchored to the edge it touches
                    if mode != "center" and ey0 and ey1 and not (ex0 and ex1):
                        nw = max(w, e["width"] / Ws * W)
                        x = x if ex0 else (W - nw if ex1 else x - (nw - w) / 2)
                        w = nw
                    if mode != "center" and ex0 and ex1 and not (ey0 and ey1):
                        nh = max(h, e["height"] / Hs * H)
                        y = y if ey0 else (H - nh if ey1 else y - (nh - h) / 2)
                        h = nh
                    x1, y1 = (W if ex1 else x + w), (H if ey1 else y + h)
                    x, y = (min(x, 0.0) if ex0 else x), (min(y, 0.0) if ey0 else y)
                    src_aspect = e["width"] / max(1e-6, e["height"])
                    if e["type"] == "image" and abs((x1 - x) / max(1e-6, y1 - y) - src_aspect) > 0.01 * src_aspect:
                        # its box changes shape: crop to the canvas side it touches
                        x, y = (0.0 if ex0 else x), (0.0 if ey0 else y)
                        P.set(e["id"], fit="cover")
                    w, h = x1 - x, y1 - y
                P.set(e["id"], x=x, y=y, width=w, height=h)
                if abs(norm_rot(e.get("rotation", 0) or 0)) > 1:
                    P.vis[e["id"]] = _rot_box(x, y, w, h, e.get("rotation", 0))
                else:
                    f = img_vis_frac(e) if e["type"] == "image" else (0.0, 0.0, 1.0, 1.0)
                    P.vis[e["id"]] = (x + f[0] * w, y + f[1] * h, x + f[2] * w, y + f[3] * h)
    P.same = True
    P.meta = {"template": f"same-{mode}", "t": s / ctx.t_star if ctx.t_star else 1.0}
    return P


def _same_blocks(units, Ws, Hs):
    """{id(unit): units of its block}. Units within a small gap of each other
    in the source form a block; a unit spanning the canvas (touching two or
    more edges: a photo, a band) is a surface, not a member, and stays alone."""
    from .analyze import _box
    tol, gap = 0.02 * min(Ws, Hs), 0.07 * min(Ws, Hs)

    def bb(u):
        bs = [_box(e) for e in u.els]
        return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))
    boxes = {id(u): bb(u) for u in units}

    def surface(u):
        b = boxes[id(u)]
        return (b[0] <= tol) + (b[1] <= tol) + (b[2] >= Ws - tol) + (b[3] >= Hs - tol) >= 2
    parent = {id(u): id(u) for u in units}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    # decoration moves on its own: a ring behind a quote is not part of it
    members = [u for u in units if not surface(u) and u.role not in ("deco", "minor")]
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            A, B = boxes[id(a)], boxes[id(b)]
            dx = max(0.0, max(A[0], B[0]) - min(A[2], B[2]))
            dy = max(0.0, max(A[1], B[1]) - min(A[3], B[3]))
            if max(dx, dy) <= gap:
                parent[find(id(a))] = find(id(b))
    out = {}
    for u in units:
        out.setdefault(find(id(u)), []).append(u)
    return {id(u): out[find(id(u))] for u in units}


def _has_picture(units, k, span, Ws, Hs):
    """Whether the part of a block within span (along axis k) holds a sizeable picture."""
    lo, hi = span
    for v in units:
        for e in v.els:
            p0 = e["x"] if k == 0 else e["y"]
            p1 = p0 + (e["width"] if k == 0 else e["height"])
            if e["type"] == "image" and lo - 1 <= p0 and p1 <= hi + 1 and e["width"] * e["height"] > 0.1 * Ws * Hs:
                return True
    return False


def _rot_box(x, y, w, h, rot):
    from .blocks import rot_bounds
    return rot_bounds(x, y, w, h, rot)


# ---------------------------------------------------------------------------
# The design turned onto the new axis. A skyscraper made into a leaderboard (or
# the reverse) is a sequence of bands along its long side: badge, photo, copy
# panel. They are laid out in the same order along the new long side, each in
# a slot sized to its content; a band keeps its own arrangement (scaled to fit
# its slot) unless that leaves its text too small to read, then its units are
# re-stacked to fill the slot.

AXIS_SKIP = {"bg", "border"}


def _bands(els, along, tol):
    """Elements grouped into bands along one axis (0 = x, 1 = y), in order:
    a band ends where nothing crosses the gap to the next."""
    from .analyze import _box
    spans = sorted(((_box(e)[along], _box(e)[along + 2], e) for e in els), key=lambda t: t[0])
    bands, cur, hi = [], [], None
    for lo, h, e in spans:
        # a new band starts where a piece reaches past the current one and
        # overlaps it by less than tol (a panel tucked under a photo's edge)
        if cur and lo > hi - tol and h > hi:
            bands.append(cur)
            cur = []
        hi = h if not cur else max(hi, h)
        cur.append(e)
    if cur:
        bands.append(cur)
    return bands


def _box_is_picture(x):
    """A detailed picture (a photo), not a flat panel."""
    if x["type"] != "image":
        return False
    info = saliency.analyze(x.get("asset_id") or "")
    return bool(info and info["busyness"] > 0.03 and info["opaque_frac"] > 0.6)


def tmpl_axis(ctx, model):
    from .analyze import Unit, _box, norm_rot
    W, H, Ws, Hs = ctx.W, ctx.H, ctx.Ws, ctx.Hs
    src_long = 1 if Hs >= Ws else 0
    new_long = 0 if W >= H else 1
    if src_long == new_long:
        return None
    els = [e for u in model["units"] if u.role not in AXIS_SKIP for e in u.els
           if abs(norm_rot(e.get("rotation", 0) or 0)) <= 1]
    if len(els) < 3:
        return None
    srcL = Hs if src_long == 1 else Ws
    # what runs most of the length (a full-height panel, a frame) is a
    # surface: it fills the new canvas instead of taking a slot
    surfaces = [e for e in els if _box(e)[src_long + 2] - _box(e)[src_long] >= 0.7 * srcL]
    els = [e for e in els if e not in surfaces]
    bands = [b for b in _bands(els, src_long, 0.1 * srcL)]
    # tiny bands (a lone rule, a dot) join their neighbour
    merged = []
    for b in bands:
        bs = [_box(e) for e in b]
        size = max(x[src_long + 2] for x in bs) - min(x[src_long] for x in bs)
        if merged and size < 0.06 * srcL:
            merged[-1] += b
        else:
            merged.append(list(b))
    bands = merged
    if len(bands) < 2 or len(bands) > 5:
        return None
    if not any(e["type"] == "text" for b in bands for e in b):
        return None
    P = Placed()
    L, S = (W, H) if new_long == 0 else (H, W)
    m = 0.06 * S
    gap = 0.02 * L

    def ext(band):
        bs = [_box(e) for e in band]
        return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))
    exts = [ext(b) for b in bands]
    fmin = 11.0 * max(W, H) / 768
    need = []
    for band, e in zip(bands, exts):
        bw, bh = e[2] - e[0], e[3] - e[1]
        short_src, long_src = (bw, bh) if new_long == 1 else (bh, bw)
        k = (S - 2 * m) / max(1e-6, short_src)
        texty = any(x["type"] == "text" for x in band)
        # a band of copy turned onto its side gets room to set its lines across
        need.append(max(long_src * k, 2.2 * (S - 2 * m) if texty else 0.0))
    avail = L - 2 * m - gap * (len(bands) - 1)
    scale = min(1.0, avail / max(1e-6, sum(need)))
    pos = m + (avail - sum(n * scale for n in need)) / 2
    troles = model.get("troles", {})
    for band, e, n in zip(bands, exts, need):
        slot_len = n * scale
        slot = (pos, m, slot_len, S - 2 * m) if new_long == 0 else (m, pos, S - 2 * m, slot_len)
        pos += slot_len + gap
        bw, bh = max(1e-6, e[2] - e[0]), max(1e-6, e[3] - e[1])
        k = min(slot[2] / bw, slot[3] / bh)
        texts = [x for x in band if x["type"] == "text"]
        pic = sum(x["width"] * x["height"] for x in band if x["type"] != "text"
                  and x["width"] * x["height"] >= 0.4 * bw * bh and _box_is_picture(x))
        if texts and min(x["font_size"] * k for x in texts) < fmin and pic == 0:
            # re-set the band: its surface fills the slot, its copy is stacked across it
            barea = bw * bh
            for x in band:
                if x["type"] == "text":
                    continue
                if x["width"] * x["height"] >= 0.4 * barea:
                    sx = slot[0] - m * 0.5 if new_long == 0 else slot[0] - m
                    P.set(x["id"], x=slot[0] - 0.5 * m, y=slot[1] - m, width=slot[2] + m, height=slot[3] + 2 * m,
                          fit="cover")
                    P.vis[x["id"]] = (slot[0], slot[1], slot[0] + slot[2], slot[1] + slot[3])
            tunits = [Unit(x["id"], troles.get(x["id"], "body"), "text", [x]) for x in
                      sorted(texts, key=lambda x: (_box(x)[1], _box(x)[0]))]
            inner = (slot[0] + 0.05 * slot[2], slot[1], 0.9 * slot[2], slot[3])
            sub = Placed()
            # no line larger than the band's biggest type would be at a share of the slot
            t = stack(ctx, sub, tunits, inner, "left", min(t_cap(ctx, inner[2]), 1.6 * k / max(1e-6, ctx.t_star)))
            if t is None:
                return None
            for kk, v in sub.props.items():
                P.set(kk, **v)
            P.vis.update(sub.vis)
            # small marks in the band keep their place relative to it
            for x in band:
                if x["type"] != "text" and x["id"] not in P.props:
                    rx = (x["x"] - e[0]) / bw
                    ry = (x["y"] - e[1]) / bh
                    w_, h_ = x["width"] * k, x["height"] * k
                    P.set(x["id"], x=slot[0] + rx * slot[2], y=slot[1] + ry * slot[3], width=w_, height=h_)
                    P.vis[x["id"]] = (slot[0] + rx * slot[2], slot[1] + ry * slot[3],
                                      slot[0] + rx * slot[2] + w_, slot[1] + ry * slot[3] + h_)
            continue
        ox = slot[0] + (slot[2] - bw * k) / 2 - e[0] * k
        oy = slot[1] + (slot[3] - bh * k) / 2 - e[1] * k
        for x in band:
            nx, ny, nw, nh = ox + x["x"] * k, oy + x["y"] * k, x["width"] * k, x["height"] * k
            if x["type"] == "text":
                fs = max(1, int(x["font_size"] * k))
                P.set(x["id"], x=nx, y=ny, width=nw, height=nh, font_size=fs, content=x["content"])
                P.vis[x["id"]] = tf.visual_box(x, nx, ny, nw, fs, tf.measure(x, fs, nw))
            else:
                P.set(x["id"], x=nx, y=ny, width=nw, height=nh)
                P.vis[x["id"]] = (nx, ny, nx + nw, ny + nh)
    # what was left out (turned pieces) keeps the source's relative place
    for u in model["units"]:
        if u.role in AXIS_SKIP:
            continue
        for x in u.els:
            if x["id"] not in P.props:
                k = min(W / Ws, H / Hs)
                P.set(x["id"], x=x["x"] / Ws * W, y=x["y"] / Hs * H, width=x["width"] * k, height=x["height"] * k)
                P.vis[x["id"]] = _rot_box(x["x"] / Ws * W, x["y"] / Hs * H, x["width"] * k, x["height"] * k,
                                          x.get("rotation", 0))
    for x in surfaces:
        # the surface keeps its insets from the canvas edges (scaled)
        k = min(W, H) / min(Ws, Hs)
        l, t = max(0.0, x["x"]) * k, max(0.0, x["y"]) * k
        r, b = max(0.0, Ws - x["x"] - x["width"]) * k, max(0.0, Hs - x["y"] - x["height"]) * k
        P.set(x["id"], x=l, y=t, width=W - l - r, height=H - t - b, fit="cover")
        P.vis[x["id"]] = (l, t, W - r, H - b)
    P.same = True
    P.meta = {"template": f"axis-{len(bands)}", "t": scale}
    return P
