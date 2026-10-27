"""Copy that keeps to the quiet part of a photo.

A design that sets its headline on the calm part of a canvas-covering photo
(a pale sky, a plain wall, a dark floor) can lose that calm on another canvas:
the crop moves, the type grows, and the headline lands on the trees. Every
other candidate is a layout of the whole canvas; these are the ones that ask
first where the photo leaves room to read.

The photo is rendered small under the target canvas as a plan draws it (no
copy, nothing changed), and each row and column is marked readable where the
headline's own colour holds WCAG 3:1 against it and the texture is calm.
Runs of readable rows (or columns) are the quiet bands. For each band two
kinds of stack are offered:

  * ``quiet-in``: the whole column inside the band (on a side band, the
    column keeps to that side at the band's width);
  * ``quiet-head``: the column runs from the band onwards, its scale capped
    so the headline itself still sits inside the band; the rest may run on
    past it (a button carries its own fill, as designers set it).

They are only candidates: the pixel critic (pixels.py) and the score choose,
so on a photo whose copy reads anywhere they simply lose. Nothing here touches
a pixel of any asset.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional, Tuple

from .analyze import Design
from .compose import DISPLAY_WORDS, EMPHASIS, WIDE_MEASURE, Plan, _specks, copyfit, margins, overflow, plan_stack, story_band
from .layout import BudgetExceeded
from .items import Ctx
from .structure import lead_text

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

COVER = 0.9  # a background image covering this share of the source canvas is a photo the copy sits on
READ = 3.0  # WCAG contrast a headline needs on its ground (large text)
CALM = 0.07  # luminance gradient (per small-render pixel, smoothed) past which the ground is texture
ROW_OK = 0.85  # a row (column) is readable when this share of some WINDOW of the copy's span is ...
WINDOW = 0.7  # ... (a tree in one corner does not spoil the sky beside it)
MIN_BAND = 0.14  # a quiet band is at least this share of the canvas along its axis ...
MAX_BAND = 0.8  # ... and less than this (a canvas readable almost everywhere needs no help)
BANDS = 2  # the longest bands looked at, per axis
STEPS = 6  # bisection steps for the headline's scale cap
ON_SUBJECT = 0.04  # share of the headline's box a quiet-head plan may set on the photo's subject
WIDE_GAIN = 1.15  # a wider headline measure must buy at least this much scale in a band


def _photo(d: Design) -> bool:
    for e in d.backgrounds:
        if e.get("type") != "image":
            continue
        w, h = float(e.get("width", 0) or 0), float(e.get("height", 0) or 0)
        x, y = float(e.get("x", 0) or 0), float(e.get("y", 0) or 0)
        ix = max(0.0, min(x + w, d.W) - max(x, 0.0))
        iy = max(0.0, min(y + h, d.H) - max(y, 0.0))
        if ix * iy >= COVER * d.W * d.H:
            return True
    return False


def _lum(rgb):
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def _box_mean(a, k: int):
    """Mean over a (2k+1)^2 window, edges clamped (a summed-area table)."""
    if k <= 0:
        return a
    p = np.pad(a, k, mode="edge")
    S = np.zeros((p.shape[0] + 1, p.shape[1] + 1), dtype=np.float64)
    S[1:, 1:] = p.cumsum(0).cumsum(1)
    n = 2 * k + 1
    h, w = a.shape
    return ((S[n:n + h, n:n + w] - S[:h, n:n + w] - S[n:n + h, :w] + S[:h, :w]) / (n * n)).astype(np.float32)


def readable_map(arr, colour: Tuple[float, float, float]):
    """Boolean map: where text of ``colour`` (0..1 RGB) reads on these pixels."""
    lb = _lum(arr)
    lt = float(_lum(np.array(colour, dtype=np.float32)))
    c = (np.maximum(lt, lb) + 0.05) / (np.minimum(lt, lb) + 0.05)
    gx = np.zeros_like(lb)
    gy = np.zeros_like(lb)
    gx[:, 1:-1] = lb[:, 2:] - lb[:, :-2]
    gy[1:-1, :] = lb[2:, :] - lb[:-2, :]
    busy = _box_mean(np.hypot(gx, gy) * 0.5, 2)
    return (c >= READ) & (busy <= CALM)


def bands(ok, lo: int, hi: int, n: int) -> List[Tuple[int, int]]:
    """Runs of readable lines (rows of ``ok`` restricted to columns lo..hi) at least ``n`` long,
    longest first."""
    if hi <= lo:
        return []
    m = ok[:, lo:hi].astype(np.float32)
    k = max(1, int(round(WINDOW * (hi - lo))))
    c = np.concatenate([np.zeros((m.shape[0], 1), np.float32), m.cumsum(axis=1)], axis=1)
    frac = (c[:, k:] - c[:, :-k]).max(axis=1) / k
    good = frac >= ROW_OK
    runs, start = [], None
    for i, g in enumerate(list(good) + [False]):
        if g and start is None:
            start = i
        elif not g and start is not None:
            if i - start >= n:
                runs.append((start, i))
            start = None
    runs.sort(key=lambda r: r[0] - r[1])
    return runs[:BANDS]


def _ground(plan: Plan, d: Design, ctx: Ctx):
    """What the copy of this plan sits on, rendered small: the plan's output without its
    copy, logo and buttons (backgrounds, slabs and decoration as the plan draws them)."""
    from . import emit, pixels

    mine = {e["id"] for r, _, _ in plan.placed + plan.floaters for e in r.item.els}
    els = [e for e in emit(plan, d, ctx) if e["id"] not in mine]
    return pixels.backdrop(els, ctx.W, ctx.H)


def _subject(plan: Plan, d: Design, ctx: Ctx, shape, f: float):
    """Boolean map (the ground's grid): where the photo's subject is drawn under this plan's
    crop: the people (head and shoulders), and the weightiest part of the picture (its
    importance map at or past vision.SUBJECT_Q). A smooth subject (a white cushion, a pale
    bottle) passes the contrast and texture test, yet copy set on it collides with the very
    thing the photo shows: quiet ground is the calm part around the subject, never the subject."""
    from . import vision

    m = np.zeros(shape, dtype=bool)
    for e in d.backgrounds:
        if e.get("type") != "image":
            continue
        v = vision.analyze(e.get("asset_id"))
        if v is None or not v.subject:
            break
        box = plan.fixed.get(e["id"])
        if box is None:
            from . import _decor_rects

            box = vision.bg_box(e, plan, d, ctx, _decor_rects(d, ctx)) or vision.centre_box(v, ctx.W, ctx.H)
        bx, by, bw, bh = box
        h, w = shape
        for fx0, fy0, fx1, fy1 in v.people:
            c0, c1 = max(0, int((bx + fx0 * bw) * f)), min(w, int(math.ceil((bx + fx1 * bw) * f)))
            r0, r1 = max(0, int((by + fy0 * bh) * f)), min(h, int(math.ceil((by + fy1 * bh) * f)))
            if c1 > c0 and r1 > r0:
                m[r0:r1, c0:c1] = True
        imp = v.imp
        ih, iw = imp.shape
        thr = float(np.quantile(imp, vision.SUBJECT_Q))
        xs = ((np.arange(w) + 0.5) / f - bx) / bw
        ys = ((np.arange(h) + 0.5) / f - by) / bh
        cx = np.clip((xs * iw).astype(int), 0, iw - 1)
        cy = np.clip((ys * ih).astype(int), 0, ih - 1)
        inside = ((ys >= 0) & (ys < 1))[:, None] & ((xs >= 0) & (xs < 1))[None, :]
        m |= (imp[cy[:, None], cx[None, :]] >= thr) & inside
        break  # the first canvas-covering photo is the ground
    return m


def _tone(e: dict) -> bool:
    """True for dark type (it needs a light ground), False for light type."""
    from .color import parse_rgba

    r, g, b, _ = parse_rgba(e.get("color"), default=(0, 0, 0, 1.0))
    return float(_lum(np.array((r / 255.0, g / 255.0, b / 255.0), dtype=np.float32))) < 0.18


def _ok(p: Optional[Plan], ctx: Ctx) -> bool:
    return p is not None and max(overflow(p.placed, ctx)) <= 0.5 and max(overflow(p.floaters, ctx)) <= 0.5


def _finish(p: Plan, d: Design, ctx: Ctx, name: str) -> Plan:
    p.name = name
    p.specks = _specks(p, d, ctx)
    copyfit(p, d, ctx)
    return p


def _slides(d: Design, ctx: Ctx) -> List[dict]:
    """Other crops of the canvas-covering photo: slid to either end of what the source showed
    along the axis the target crops (by bleed, as vision.cover_box draws a focal crop), so a
    quiet band the centre crop cuts away (the sky over a landscape) can stay in view. A crop
    that would cut a head is not offered."""
    from . import vision

    if not vision.BLEED:
        return []
    out = []
    for e in d.backgrounds:
        if e.get("type") != "image":
            continue
        if e.get("fit") != "cover":
            break  # finalize keeps bleed only for a cover photo: any other would be pulled inside
        v = vision.analyze(e.get("asset_id"))
        if v is None:
            continue
        iw, ih = v.size
        sc = max(ctx.W / iw, ctx.H / ih)
        bw, bh = iw * sc, ih * sc
        wx, wy = ctx.W / bw, ctx.H / bh
        if wx > 0.985 and wy > 0.985:
            continue
        horiz = wx <= wy
        win = wx if horiz else wy
        room = 1.0 - win
        shown = vision.source_window(e, d.W, d.H, v.size)
        a0, a1 = (shown[0], shown[2]) if horiz else (shown[1], shown[3])
        if a1 - a0 < win:
            continue  # the source showed less than the target will: the centre of it is the crop
        centre = min(max((a0 + a1) / 2.0 - win / 2.0, 0.0), room)
        for t in (max(0.0, a0), min(room, a1 - win)):
            if abs(t - centre) * (bw if horiz else bh) < 0.04 * (ctx.W if horiz else ctx.H):
                continue
            s0, s1 = (v.extent[0], v.extent[2]) if horiz else (v.extent[1], v.extent[3])
            # a photo with a subject keeps it in view (as much of it as the centre crop does)
            kept = lambda t0: max(0.0, min(t0 + win, s1) - max(t0, s0))  # noqa: E731
            cut = v.subject and kept(t) < 0.9 * kept(centre)
            for hd in v.heads:
                h0, h1 = (hd[0], hd[2]) if horiz else (hd[1], hd[3])
                cut = cut or h0 < t - 1e-4 or h1 > t + win + 1e-4
            if cut:
                continue
            box = (-t * bw, -(bh - ctx.H) / 2.0, bw, bh) if horiz else (-(bw - ctx.W) / 2.0, -t * bh, bw, bh)
            out.append({e["id"]: box})
        break  # the first canvas-covering photo is the ground
    return out


def plans(d: Design, ctx: Ctx, field: List[Plan]) -> List[Plan]:
    """Extra stack candidates that keep the headline on the quiet part of a canvas-covering
    photo (see the module docstring); [] when the design has no such photo or the photo
    leaves no quiet band (or is quiet nearly everywhere)."""
    if np is None or not field or not _photo(d):
        return []
    head = next((it for it in d.items if it.role == "headline" and it.is_copy), None)
    if head is None:
        return []
    from . import pixels
    from .color import parse_rgba

    if not pixels.available():
        return []
    base = min(field, key=lambda p: (p.core_lost(), p.lost(), -p.score))
    r, g, b, _ = parse_rgba(lead_text(head).get("color"), default=(0, 0, 0, 1.0))
    colour = (r / 255.0, g / 255.0, b / 255.0)
    out: List[Plan] = []
    try:
        for fixed in [dict(base.fixed)] + [{**base.fixed, **sl} for sl in _slides(d, ctx)]:
            ground = replace(base, fixed=fixed)
            arr, f = _ground(ground, d, ctx)
            if arr is None:
                continue
            sub = _subject(ground, d, ctx, arr.shape[:2], f)
            out += _from_ground(d, ctx, base, head, readable_map(arr, colour), f, fixed, sub)
    except BudgetExceeded:
        pass  # out of budget: the quiet plans found so far join the field
    return out


def _from_ground(d: Design, ctx: Ctx, base: Plan, head, ok, f: float, fixed: dict, sub=None) -> List[Plan]:
    h, w = ok.shape
    mx, my = margins(d, ctx)
    top, bottom = story_band(ctx)
    y0, y1 = max(my, top), ctx.H - max(my, bottom)  # the safe zone, px
    x0, x1 = mx, ctx.W - mx
    hidden = list(base.hidden)
    out: List[Plan] = []
    tag = "-slid" if fixed != base.fixed else ""

    def px(v: int) -> float:
        return v / f

    def attempt(name: str, insets, c: Optional[Ctx] = None, **kw) -> Optional[Plan]:
        p = plan_stack(d, c or ctx, hidden, name=name, insets=insets, **kw)
        if p is None or not _ok(p, ctx):
            return None
        p.fixed = dict(fixed)
        return p

    def head_in(c: Ctx, ins, vside: str, b0: float, b1: float, keys: frozenset) -> Optional[Plan]:
        """The largest stack from the band on whose headline (and the copy in ``keys``) still
        sits inside the band."""
        def inside(p: Optional[Plan]) -> bool:
            if p is None:
                return False
            tol = 0.02 * ctx.H
            seen = 0
            for rr, xx, yy in p.placed + p.floaters:
                if rr.item.key in keys:
                    seen += 1
                    if yy < b0 - tol or yy + rr.h > b1 + tol:
                        return False
                    # and on readable ground across its width too: a band's rows need only most of
                    # the canvas readable, so a long line may still reach the subject beside it
                    cell = ok[max(0, int(yy * f)):max(1, int((yy + rr.h) * f)),
                              max(0, int(xx * f)):max(1, int((xx + rr.w) * f))]
                    if cell.size and float(cell.mean()) < ROW_OK:
                        return False
                    # and off the photo's subject: a smooth cushion reads, but copy on it collides
                    if sub is not None:
                        on = sub[max(0, int(yy * f)):max(1, int((yy + rr.h) * f)),
                                 max(0, int(xx * f)):max(1, int((xx + rr.w) * f))]
                        if on.size and float(on.mean()) > ON_SUBJECT:
                            return False
            return seen > 0

        free = attempt("quiet-head", ins, c, vside=vside)
        if free is None or inside(free):
            return free
        lo, hi, best = 0.0, free.s, None
        for _ in range(STEPS):
            cap = (lo + hi) / 2.0 if lo > 0 else hi * 0.5
            q = attempt("quiet-head", ins, replace(c, scale_cap=cap), vside=vside)
            if inside(q):
                best, lo = q, cap
            else:
                hi = cap
            if lo > 0 and hi - lo < 0.04 * free.s:
                break
        return best

    # the copy that needs the same ground as the headline: plain texts of its tone (a button
    # carries its own fill)
    dark = _tone(lead_text(head))
    kin = frozenset(it.key for it in d.items if it.kind == "text" and it.role not in ("cta", "void", "missing")
                    and _tone(lead_text(it)) == dark) | {head.key}
    sets = [frozenset({head.key})] + ([kin] if len(kin) > 1 else [])
    words = len(str(lead_text(head).get("content", "")).split())
    emph = (1.0,) + (EMPHASIS[:1] if words <= DISPLAY_WORDS else ())
    # rows: a band across the canvas (sky above, floor below)
    span = (int(x0 * f), max(int(x0 * f) + 1, int(x1 * f)))
    for a, z in bands(ok, span[0], span[1], max(2, int(MIN_BAND * h))):
        if (z - a) >= MAX_BAND * h:
            continue
        b0, b1 = max(y0, px(a)), min(y1, px(z))
        if b1 - b0 < MIN_BAND * ctx.H:
            continue
        p = attempt("quiet-in", (0.0, b0 - y0, 0.0, y1 - b1))
        if p is not None:
            out.append(_finish(p, d, ctx, f"quiet-in{tag}-y{int(b0)}"))
        # the headline in the band, the rest run on past it, away from the nearer canvas edge
        down = (b0 - y0) <= (y1 - b1)
        ins = (0.0, b0 - y0, 0.0, 0.0) if down else (0.0, 0.0, 0.0, y1 - b1)
        # a band across the canvas is short and wide: the headline may also run on in longer
        # lines (WIDE_MEASURE), so it fits the band in fewer lines instead of shrinking the column;
        # only when that buys it real size (WIDE_GAIN), else the source's own breaks stand
        for k, keys in enumerate(sets):
            for e in emph:
                narrow = None
                for m in (1.0, WIDE_MEASURE) if k == 0 else (1.0,):
                    c = replace(ctx, emphasis=e, measure=m) if (e, m) != (1.0, 1.0) else ctx
                    p = head_in(c, ins, "top" if down else "bottom", b0, b1, keys)
                    if m == 1.0:
                        narrow = p
                    elif p is not None and narrow is not None and p.s < WIDE_GAIN * narrow.s:
                        continue
                    if p is not None:
                        out.append(_finish(p, d, c, f"quiet-{'copy' if k else 'head'}{tag}-y{int(b0)}"
                                           + (f"~{m:g}" if m != 1.0 else "") + (f"@{e:g}" if e != 1.0 else "")))
    # columns: a band down one side (a plain wall beside the subject)
    span = (int(y0 * f), max(int(y0 * f) + 1, int(y1 * f)))
    for a, z in bands(ok.T, span[0], span[1], max(2, int(MIN_BAND * w))):
        if (z - a) >= MAX_BAND * w:
            continue
        b0, b1 = max(x0, px(a)), min(x1, px(z))
        if b1 - b0 < max(MIN_BAND * ctx.W, 0.25 * (x1 - x0)):
            continue
        side = "start" if (b0 - x0) <= (x1 - b1) else "end"
        p = attempt("quiet-in", (b0 - x0, 0.0, x1 - b1, 0.0), force_align=side)
        if p is not None:
            out.append(_finish(p, d, ctx, f"quiet-in{tag}-x{int(b0)}"))
    return out
