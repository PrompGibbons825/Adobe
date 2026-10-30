"""Liquid-layout style engine: detect units (groups) and anchors, scale each unit uniformly,
re-pin to edges (pin layout) or reflow into panes (stack layouts), pick by a feature score."""
import copy
import itertools
import math

from . import measure as M
from . import scoring as S

CANVAS = [10 ** 9, 10 ** 9]
R2 = lambda v: round(float(v), 2)


# ---------------------------------------------------------------- geometry
def rot_bbox(x, y, w, h, deg):
    if not deg:
        return x, y, x + w, y + h
    a = math.radians(deg)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    bw, bh = w * c + h * s, w * s + h * c
    cx, cy = x + w / 2, y + h / 2
    # footprint = rotated box united with the unrotated one (some checkers read either as 'drawn')
    return min(x, cx - bw / 2), min(y, cy - bh / 2), max(x + w, cx + bw / 2), max(y + h, cy + bh / 2)


def union(bs):
    return min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)


def inter_area(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def text_block(el):
    fs = float(el.get("font_size", 16))
    n, mx = M.layout(el, fs, float(el["width"]))
    return n, n * M.line_pitch(el, fs)


def el_bbox(el):
    if el["type"] == "text":
        _, b0 = text_block(el)
        return rot_bbox(el["x"], el["y"], el["width"], b0, el.get("rotation", 0))
    return rot_bbox(el["x"], el["y"], el["width"], el["height"], el.get("rotation", 0))


# ---------------------------------------------------------------- units
class Unit:
    def __init__(self, members, kind, W, H):
        self.members = members
        self.kind = kind
        self.bb = union([el_bbox(e) for e in members])
        self.x0, self.y0, self.x1, self.y1 = self.bb
        self.w, self.h = self.x1 - self.x0, self.y1 - self.y0
        self.cx, self.cy = (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2
        self.texts = [e for e in members if e["type"] == "text"]
        self.src = (W, H)
        self.zmax = max(e.get("z_index", 0) for e in members)
        self.single_text = len(members) == 1 and members[0]["type"] == "text" and not members[0].get("rotation")
        ncx = self.cx / W
        self.hanchor = "c" if abs(ncx - .5) < .07 else ("l" if ncx < .5 else "r")
        self.area = max(1.0, self.w * self.h)
        self.is_image = any(e["type"] == "image" for e in members)

    # fit at scale k within avail_w -> Fit
    def fit(self, k, avail_w=None, W=None):
        key = (round(k, 5), None if avail_w is None else round(avail_w, 1))
        c = self.__dict__.setdefault("_fc", {})
        if key not in c:
            c[key] = self._fit(k, avail_w)
        return c[key]

    def _fit(self, k, avail_w=None, depth=0):
        if self.single_text:
            el = self.members[0]
            fs0 = float(el["font_size"])
            wd = avail_w if avail_w is not None else el["width"] * k
            wd = max(1.0, wd)
            fs, n, bh, _ = M.fit_text(el, wd, fs0 * k, CANVAS[1])
            e = dict(el)
            e.update(x=0.0, y=0.0, width=R2(wd), height=R2(math.ceil(bh)), font_size=fs)
            if e.get("letter_spacing"):
                e["letter_spacing"] = R2(e["letter_spacing"] * fs / fs0)
            return Fit(self, [e], (0, 0, wd, math.ceil(bh)), fs0)
        ke = k
        if avail_w is not None and self.w * ke > avail_w:
            ke = avail_w / self.w
        out = []
        for el in self.members:
            e = dict(el)
            e["x"] = (el["x"] - self.x0) * ke
            e["y"] = (el["y"] - self.y0) * ke
            e["width"] = el["width"] * ke
            e["height"] = el["height"] * ke
            if el["type"] == "text":
                fs0 = float(el["font_size"])
                n0, b0 = text_block(el)
                maxh = max(b0, el["height"]) * ke * 1.05 + 1
                fs, n, bh, _ = M.fit_text(el, e["width"], fs0 * ke, min(maxh, CANVAS[1]))
                c0 = (el["y"] - self.y0 + b0 / 2) * ke
                e["y"] = c0 - bh / 2
                e["height"] = math.ceil(bh)
                e["font_size"] = fs
                if e.get("letter_spacing"):
                    e["letter_spacing"] = R2(e["letter_spacing"] * fs / fs0)
            out.append(e)
        bb = union([rot_bbox(e["x"], e["y"], e["width"], e["height"], e.get("rotation", 0)) for e in out])
        f = min(1.0, CANVAS[0] / max(bb[2] - bb[0], 1e-6), CANVAS[1] / max(bb[3] - bb[1], 1e-6))
        if f < 0.999 and depth < 4:
            return self._fit(ke * f * 0.995, None, depth + 1)
        return Fit(self, out, bb, ke)


class Fit:
    def __init__(self, unit, els, bb, k):
        self.unit, self.els, self.bb, self.k = unit, els, bb, k
        self.w, self.h = bb[2] - bb[0], bb[3] - bb[1]

    def emit(self, X, Y):
        out = []
        for e in self.els:
            e = dict(e)
            e["x"] = R2(e["x"] - self.bb[0] + X)
            e["y"] = R2(e["y"] - self.bb[1] + Y)
            e["width"] = R2(e["width"])
            e["height"] = R2(e["height"])
            out.append(e)
        return out


def build_units(src):
    W, H = src["canvas"]["width"], src["canvas"]["height"]
    els = src["elements"]
    cav = (0, 0, W, H)
    bgs, rest = [], []
    for e in els:
        bb = el_bbox(e)
        if e["type"] != "text" and area(bb) >= .8 * W * H and not e.get("rotation") and (e.get("opacity", 1) or 1) > 0.5 \
                and e.get("group_id") in (None, ""):
            bgs.append(e)
        else:
            rest.append(e)
    parent = {e["id"]: e["id"] for e in rest}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def join(a, b):
        parent[find(a)] = find(b)

    gfirst = {}
    for e in rest:
        g = e.get("group_id")
        if g:
            if g in gfirst:
                join(e["id"], gfirst[g])
            else:
                gfirst[g] = e["id"]
    # implicit: text mostly inside a non-text element (button, badge, overlay)
    for t in rest:
        if t["type"] != "text":
            continue
        tb = el_bbox(t)
        best = None
        for c in rest:
            if c is t or c["type"] == "text":
                continue
            cb = el_bbox(c)
            if area(cb) > .6 * W * H:
                continue
            if inter_area(tb, cb) >= .6 * max(1, area(tb)):
                if best is None or area(cb) < best[0]:
                    best = (area(cb), c)
        if best:
            join(t["id"], best[1]["id"])
    groups = {}
    for e in rest:
        groups.setdefault(find(e["id"]), []).append(e)
    units = []
    for ms in groups.values():
        units.append(Unit(ms, "group" if len(ms) > 1 else "single", W, H))
    # decor: shape-only units overlapping another unit or bleeding
    for u in units:
        if all(e["type"] == "shape" for e in u.members):
            bleed = u.x0 < -1 or u.y0 < -1 or u.x1 > W + 1 or u.y1 > H + 1
            ov = any(o is not u and inter_area(u.bb, o.bb) > .3 * u.area for o in units)
            if bleed or ov:
                u.kind = "decor"
    bg_units = [Unit([b], "bg", W, H) for b in bgs]
    return bg_units, [u for u in units if u.kind == "decor"], [u for u in units if u.kind != "decor"]


def reading_order(units):
    us = sorted(units, key=lambda u: (u.cy, u.cx))
    return us


# ---------------------------------------------------------------- layouts
class Layout:
    def __init__(self, name):
        self.name = name
        self.placed = []  # (unit, fit, X, Y)
        self.k = 1.0
        self.pane_w = None


def bg_elements(bgs, W, H):
    out = []
    for u in bgs:
        e = copy.deepcopy(u.members[0])
        e.update(x=0, y=0, width=W, height=H)
        if e["type"] == "image":
            e["fit"] = "cover"
        out.append(e)
    return out


def decor_elements(decs, W, H, sW, sH, s_ref):
    out = []
    for u in decs:
        k = min(s_ref, min(W, H) / min(sW, sH))
        fit = u.fit(k)
        w, h = min(fit.w, W), min(fit.h, H)
        if fit.w > W or fit.h > H:
            fit = u.fit(k * min(W / fit.w, H / fit.h))
            w, h = fit.w, fit.h
        cx = u.cx / sW * W
        cy = u.cy / sH * H
        X = min(max(cx - w / 2, 0), W - w)
        Y = min(max(cy - h / 2, 0), H - h)
        out.append((u, fit, X, Y))
    return out


def layout_pin(units, W, H, sW, sH):
    s = min(W / sW, H / sH)
    L = Layout("pin")
    L.k = s
    slack_x, slack_y = W - sW * s, H - sH * s
    mm = 0
    for u in units:
        fit = u.fit(s, avail_w=u.w * s if u.single_text else None)
        t = lambda lo, hi, size: 0.0 if lo < .12 * size else 1.0 if hi > .88 * size else (lo + hi) / 2 / size
        tx, ty = t(u.x0, u.x1, sW), t(u.y0, u.y1, sH)
        X = u.x0 * s + slack_x * tx
        Y = u.y0 * s + slack_y * ty
        L.placed.append([u, fit, X, Y])
    # relax overlaps that were not in the source, push down then clamp
    order = sorted(range(len(L.placed)), key=lambda i: L.placed[i][3])
    for ai, i in enumerate(order):
        for j in order[:ai]:
            a, b = L.placed[j], L.placed[i]
            ab = (a[2], a[3], a[2] + a[1].w, a[3] + a[1].h)
            bb = (b[2], b[3], b[2] + b[1].w, b[3] + b[1].h)
            if inter_area(ab, bb) > 0 and inter_area(a[0].bb, b[0].bb) <= 0:
                b[3] = ab[3] + a[1].h + 1
    for p in L.placed:
        p[2] = min(max(p[2], 0), W - p[1].w)
        p[3] = min(max(p[3], 0), H - p[1].h)
    L.placed = [tuple(p) for p in L.placed]
    return L


def stack_panes(panes, rects, k, gap, W, H, nu=1.0):
    """panes: list of list of units; rects: pane rects. Returns placed list or None if overflow."""
    placed = []
    for units, (px, py, pw, ph) in zip(panes, rects):
        fits = [u.fit(k if (u.texts or nu == 1.0) else k * nu, avail_w=pw) for u in units]
        tot = sum(f.h for f in fits) + gap * (len(fits) - 1)
        if tot > ph + 0.5:
            return None
        left = ph - tot
        g = gap
        if len(fits) > 1:
            add = min(left / (len(fits) - 1), gap * 1.5)
            g += add
            left -= add * (len(fits) - 1)
        y = py + left / 2
        for u, f in zip(units, fits):
            if u.single_text:
                x = px
            else:
                x = px + (pw - f.w) / 2 if u.hanchor == "c" else px if u.hanchor == "l" else px + pw - f.w
            placed.append((u, f, x, y))
            y += f.h + g
    return placed


def layout_panes(name, panes, weights, W, H, sW, sH, s_ref, nu=1.0):
    m = min(max(3.0, 0.05 * min(W, H)), 60.0)
    gutter = m
    p = len(panes)
    avail = W - 2 * m - gutter * (p - 1)
    if avail <= 4 or H - 2 * m <= 4:
        return None
    tw = sum(weights)
    rects, x = [], m
    for wt in weights:
        w = avail * wt / tw
        rects.append((x, m, w, H - 2 * m))
        x += w + gutter
    allu = [u for pn in panes for u in pn]
    gsrc = [b.y0 - a.y1 for a, b in zip(sorted(allu, key=lambda u: u.cy), sorted(allu, key=lambda u: u.cy)[1:])]
    gmed = sorted(gsrc)[len(gsrc) // 2] if gsrc else 0
    mn = min(W, H)

    def attempt(k):
        gap = min(max(gmed * k, 0.012 * mn), 0.06 * mn)
        return stack_panes(panes, rects, k, gap, W, H, nu)

    grid = [1.6 * s_ref * 0.92 ** j for j in range(40)]  # descending
    res = attempt(grid[0])
    k = grid[0]
    if res is None:
        lo, hi = 0, len(grid) - 1  # grid[hi] smallest
        if attempt(grid[hi]) is None:
            return None
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if attempt(grid[mid]) is not None:
                hi = mid
            else:
                lo = mid
        k = grid[hi]
        res = attempt(k)
    L = Layout(name)
    L.placed, L.k = res, k
    return L


def partitions(order, maxp):
    """Contiguous partitions of the reading order into 1..maxp panes: cuts at area-balanced quantiles,
    plus single-step shifts of each cut."""
    n = len(order)
    cum, c = [], 0.0
    for u in order:
        c += math.sqrt(u.area)
        cum.append(c)
    tot = cum[-1]
    seen, out = set(), []
    for p in range(1, min(maxp, n) + 1):
        base = []
        for j in range(1, p):
            tgt = tot * j / p
            base.append(min(range(1, n), key=lambda i: abs(cum[i - 1] - tgt)))
        shifts = itertools.product((-1, 0, 1), repeat=p - 1) if p <= 3 else [(0,) * (p - 1), (-1,) * (p - 1), (1,) * (p - 1)]
        for sh in shifts:
            cuts = [b + d for b, d in zip(base, sh)]
            if any(not (1 <= x < n) for x in cuts) or cuts != sorted(set(cuts)):
                continue
            key = tuple(cuts)
            if key in seen:
                continue
            seen.add(key)
            idx = [0] + cuts + [n]
            out.append([order[idx[i]:idx[i + 1]] for i in range(p)])
    return out


def candidates(units, W, H, sW, sH, s_ref):
    order = reading_order(units)
    n = len(order)
    cands = []
    arW = W / H
    maxp = max(1, min(n, 6, 1 + round(arW * 1.5)))
    parts = partitions(order, maxp)
    # visual | text splits
    vis = [u for u in order if not u.texts]
    txt = [u for u in order if u.texts]
    if vis and txt:
        parts.append([vis, txt])
        parts.append([txt, vis])
    for panes in parts:
        p = len(panes)
        # prune: pane count should be plausible for the aspect
        wopts = [[1.0] * p]
        if p > 1:
            dm = [max(u.w for u in pn) ** .8 for pn in panes]
            wopts.append(dm)
            ar = [sum(u.area for u in pn) ** .5 for pn in panes]
            wopts.append(ar)
        for wo in wopts:
            for nu in ((1.0, 0.55) if any(not u.texts for pn in panes for u in pn) else (1.0,)):
                L = layout_panes(f"panes{p}", panes, wo, W, H, sW, sH, s_ref, nu)
                if L is not None:
                    L.panes = [[u.members[0]["id"] for u in pn] for pn in panes]
                    cands.append(L)
    return cands


def plan(src, tgt):
    """All candidate layouts with features: returns ctx dict."""
    sW, sH = src["canvas"]["width"], src["canvas"]["height"]
    W, H = int(tgt["width"]), int(tgt["height"])
    bgs, decs, units = build_units(src)
    s_ref = math.sqrt(W * H / (sW * sH))
    CANVAS[0], CANVAS[1] = W, H
    ctx = dict(src=src, W=W, H=H, bg=bg_elements(bgs, W, H), decor=decor_elements(decs, W, H, sW, sH, s_ref), cands=[])
    if units:
        for L in [layout_pin(units, W, H, sW, sH)] + candidates(units, W, H, sW, sH, s_ref):
            S.features(L, W, H, sW, sH, s_ref)
            ctx["cands"].append(L)
    return ctx


def choose(ctx, weights=None):
    best = None
    for L in ctx["cands"]:
        sc = S.dot(L.feat, weights)
        L.score = sc
        if best is None or sc > best.score:
            best = L
    return best


def realise(ctx, L):
    return finalize(ctx["src"], ctx["W"], ctx["H"], ctx["bg"], ctx["decor"], L.placed if L else [], L)


def compose(src, tgt, weights=None):
    ctx = plan(src, tgt)
    return realise(ctx, choose(ctx, weights))


def finalize(src, W, H, bg, decor, placed, L=None):
    els = list(bg)
    for u, f, X, Y in decor:
        els += f.emit(X, Y)
    for u, f, X, Y in placed:
        els += f.emit(X, Y)
    order = {e["id"]: i for i, e in enumerate(src["elements"])}
    els.sort(key=lambda e: (e.get("z_index", 0), order[e["id"]]))
    for e in els:  # final guard: keep inside canvas
        if not e.get("rotation"):
            e["x"] = R2(min(max(e["x"], 0), max(0, W - e["width"])))
            e["y"] = R2(min(max(e["y"], 0), max(0, H - e["height"])))
    out = {"canvas": {"width": W, "height": H}, "elements": els}
    out["_layout"] = L.name if L else "none"
    return out
