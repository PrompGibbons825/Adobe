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


def block_scale(u, t, k):
    if u.role == "hero":
        return t * k
    fs = _src_fs(u)
    if fs and u.role in MIN_FS:
        return max(t, MIN_FS[u.role] / fs)
    return t


def _stack_sizes(blocks, units, t, k, max_w):
    return [b.size(block_scale(u, t, k), max_w) for b, u in zip(blocks, units)]


def stack(ctx: Ctx, P: Placed, units, rect, align, t_hi):
    """Lay units out top-to-bottom in rect. Returns chosen t or None."""
    if not units:
        return t_hi
    units = sorted(units, key=lambda u: u.cy)
    blocks = [make_block(u, ctx.Ws) for u in units]
    rx, ry, rw, rh = rect
    base_gaps = [min(max(g, 0.02 * ctx.Hs), 0.08 * ctx.Hs) for g in _src_gaps(units)]
    has_hero = any(u.role == "hero" for u in units)
    k_min = 0.6 if has_hero else 1.0

    def total(t, k):
        sizes = _stack_sizes(blocks, units, t, k, rw)
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
    if t <= 0.01:
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
        b.place(P, rx, y, block_scale(u, t, k), rw, align)
        y += sizes[i][1] + (gaps[i] if i < len(gaps) else 0)
    return t


def t_cap(ctx, col_w):
    return min(1.25 * ctx.t_star, 2.0)


def tmpl_stack(ctx, flow, align="center"):
    P = Placed()
    m = ctx.margin
    rect = (m, m, ctx.W - 2 * m, ctx.H - 2 * m)
    t = stack(ctx, P, flow, rect, align, t_cap(ctx, rect[2]))
    if t is None:
        return None
    P.meta = {"template": f"stack-{align}", "t": t}
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


def tmpl_row(ctx, flow, with_hero=True, with_logo=True, with_sub=True):
    W, H = ctx.W, ctx.H
    if W / H < 2.2:
        return None
    P = Placed()
    mv = 0.14 * H
    mh = max(0.03 * W, mv)
    hA = H - 2 * mv
    gut = max(0.02 * W, mv)
    by_role = {}
    for u in flow:
        by_role.setdefault(u.role, u)
    head = by_role.get("headline")
    if head is None:
        return None
    items = []  # (kind, block, w, h, scale)
    if with_logo and "logo" in by_role:
        b = ImageBlock(by_role["logo"])
        s = b.scale_for(10.0, 0.2 * W, hA * 0.75)
        items.append(("logo", b, b.vw * s, b.vh * s, s))
    if with_hero and "hero" in by_role:
        b = ImageBlock(by_role["hero"])
        s = b.scale_for(10.0, 0.2 * W, H - 2 * mv * 0.5)
        items.append(("hero", b, b.vw * s, b.vh * s, s))
    text_idx = len(items)
    items.append(("text", None, 0, 0, 0))
    cta = by_role.get("cta")
    if cta is not None:
        if cta.kind == "text":
            b = TextBlock(cta, ctx.Ws)
            mm = tf.fit(cta.els[0], 0.25 * W, hA * 0.6, min(hA * 0.45, cta.els[0]["font_size"]), 1, 1)
            if mm:
                items.append(("cta_text", b, mm["width"], hA * 0.6, mm))
        else:
            b = GroupBlock(cta)
            u = b.scale_for(10.0, 0.24 * W, hA * 0.7)
            items.append(("cta", b, b.w * u, b.h * u, u))
    def _pad(it, side):
        if it[0] in ("logo", "hero"):
            b, sc = it[1], it[4]
            return (b.v[0] if side == 0 else 1 - b.v[2]) * b.el["width"] * sc
        return 0.0
    mL = max(mh, _pad(items[0], 0) + 2)
    mR = max(mh, _pad(items[-1], 1) + 2)
    fixed = sum(it[2] for it in items)
    col_w = W - mL - mR - fixed - gut * (len(items) - 1)
    if col_w < 0.18 * W:
        return None
    hel = head.els[0]
    sub = by_role.get("subhead") if with_sub else None
    h_share = 0.62 if sub else 1.0
    hm = tf.fit(hel, col_w, hA * h_share, min(hA * 0.55, hel["font_size"] * 1.2), 1, max_lines=3)
    if hm is None:
        return None
    sm = None
    if sub is not None:
        sel = sub.els[0]
        gap = hm["fs"] * 0.25
        sm = tf.fit(sel, col_w, hA - hm["height"] - gap, min(hm["fs"] * 0.6, sel["font_size"]), 1, max_lines=2)
        if sm is None or sm["fs"] < 9:
            return None
    tw = max(hm["width"], sm["width"] if sm else 0)
    th = hm["height"] + (sm["height"] + hm["fs"] * 0.25 if sm else 0)
    items[text_idx] = ("text", None, tw, th, None)
    total_w = sum(it[2] for it in items)
    n = len(items)
    free = W - mL - mR - total_w
    if free < 0:
        return None
    g = free / (n - 1) if n > 1 else 0
    g = min(g, W * 0.08)
    if n > 1 and g < gut * 0.6:
        return None
    x = mL + (W - mL - mR - total_w - g * (n - 1)) / 2 if n > 1 else (W - total_w) / 2
    cy = H / 2
    for kind, b, w, h, s in items:
        if kind in ("logo", "hero"):
            b.place_vis(P, x, cy - h / 2, s)
        elif kind == "cta":
            b.place_at(P, x, cy - h / 2, s)
        elif kind == "cta_text":
            el = b.el
            y = tf.optical_top(el, s["fs"], 1, cy)
            P.set(el["id"], x=x, y=y, width=w + s["fs"], height=s["fs"] * float(el.get("line_height", 1.2)),
                  font_size=s["fs"], text_align="left")
            P.vis[el["id"]] = tf.visual_box(dict(el, text_align="left"), x, y, w + s["fs"], s["fs"], s)
        else:
            # text column, vertically centered as a block
            asc, cap, desc = tf.metrics(hel)
            lh = float(hel.get("line_height", 1.2))
            top = cy - th / 2
            y = top
            P.set(hel["id"], x=x, y=y, width=col_w, height=hm["height"], font_size=hm["fs"], text_align="left")
            P.vis[hel["id"]] = tf.visual_box(dict(hel, text_align="left"), x, y, col_w, hm["fs"], hm)
            if sm:
                sel = sub.els[0]
                y2 = y + hm["height"] + hm["fs"] * 0.25
                P.set(sel["id"], x=x, y=y2, width=col_w, height=sm["height"], font_size=sm["fs"], text_align="left")
                P.vis[sel["id"]] = tf.visual_box(dict(sel, text_align="left"), x, y2, col_w, sm["fs"], sm)
            # nudge so the inked block is centered rather than the line boxes
            vb = [P.vis[hel["id"]]] + ([P.vis[sub.els[0]["id"]]] if sm else [])
            ink_c = (min(b_[1] for b_ in vb) + max(b_[3] for b_ in vb)) / 2
            dy = cy - ink_c
            for eid in [hel["id"]] + ([sub.els[0]["id"]] if sm else []):
                P.props[eid]["y"] += dy
                v = P.vis[eid]
                P.vis[eid] = (v[0], v[1] + dy, v[2], v[3] + dy)
        x += w + g
    P.meta = {"template": f"row{'-hero' if with_hero else ''}{'-logo' if with_logo else ''}{'-sub' if sm else ''}",
              "t": hm["fs"] / max(1, hel["font_size"])}
    return P
