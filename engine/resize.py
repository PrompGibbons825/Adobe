"""
CHAMELEON engine entry point.

Pipeline:
  1. analyze   - infer roles (headline, logo, cta, hero, badges, legal...),
                 groups, decoration and background; saliency per image
  2. generate  - many candidate layouts from three templates (stack, split,
                 row) at several "drop levels" (what to hide when space is
                 tight)
  3. finalize  - saliency-aware background crop, decoration, corner-anchored
                 floats (ribbons, small images), keep content in canvas
  4. score     - our own critic (readability, collisions, prominence, text on
                 busy image, balance, what got dropped); return the best
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any

from .analyze import FLOW_ROLES, analyze, norm_rot
from .finalize import finalize
from .layout import Ctx, tmpl_backdrop, tmpl_card_panel, tmpl_grid, tmpl_panel, tmpl_row, tmpl_split, tmpl_stack
from .score import score

DEBUG = os.environ.get("CHAMELEON_DEBUG") == "1"
DROP_LEVELS = [0.0, 1.3, 3.5, 5.5]


def _candidates(ctx, model):
    """Every candidate keeps every unit (no drops)."""
    if not getattr(ctx, "grids", False) and not getattr(ctx, "upright", False) \
            and not getattr(ctx, "restack", False):
        from .layout import tmpl_same
        # at a near-same shape the design is kept exactly; otherwise its units
        # keep to their edges (pinned), and from a 1.5x change they may also
        # spread to the new proportions
        sev = abs(math.log((ctx.W / ctx.H) / (ctx.Ws / ctx.Hs)))
        modes = ("center",) if sev < math.log(1.15) else \
            ("pinned", "spread") if _fade(ctx) < 1.0 else ("pinned",)
        for mode in modes:
            yield 0.0, lambda m=mode: tmpl_same(ctx, model, m)
    if not getattr(ctx, "grids", False) and _fade(ctx) < 0.5:
        from .layout import tmpl_axis
        yield 0.0, lambda: tmpl_axis(ctx, model)
    flow = [u for u in model["units"] if u.role in FLOW_ROLES]
    if not flow:
        return
    has_frame = any(u.kind == "frame" for u in model["units"])
    for alpha in ((0.0, 0.5, 0.9) if has_frame else (0.0,)):
        for item in _templates(ctx, flow, alpha):
            yield item


def _templates(ctx, flow, alpha):
    def with_alpha(make):
        def run():
            ctx.frame_alpha = alpha
            P = make()
            if P is not None:
                P.meta["template"] += f" a{alpha}" if alpha else ""
            return P
        return run
    for th, make in _templates_raw(ctx, flow):
        yield th, with_alpha(make)


def f_hero(flow):
    return [u for u in flow if u.role == "hero"]


def _templates_raw(ctx, flow):
    # grids: units side by side in columns (optionally under a headline tier,
    # optionally beside a full-bleed picture); a separate, fallback stage
    if getattr(ctx, "grids", False):
        for k in (2, 3, 4):
            for tiers in (False, True):
                for panel in (None, "left", "right"):
                    if ctx.W / ctx.H >= 1.0:
                        yield 0.0, lambda f=flow, k=k, ti=tiers, pa=panel: tmpl_grid(ctx, f, k, ti, pa)
        return
    for region in ("full", "left", "right", "top", "bottom"):
        for align in ("center", "left"):
            # top/bottom bands on tall canvases, and on clearly wide ones too: on a
            # wide photo the quiet, high-contrast part is often a horizontal band
            if region == "full" or (region in ("top", "bottom") and (ctx.H / ctx.W > 1.2 or ctx.W / ctx.H >= 1.5)) \
                    or (region in ("left", "right") and ctx.W / ctx.H > 1.2):
                yield 0.0, lambda f=flow, a=align, r=region: tmpl_stack(ctx, f, a, r)
                if any(u.role == "hero" for u in f_hero(flow)):
                    # hero-first: hold text back so the hero visual can take the space
                    yield 0.0, lambda f=flow, a=align, r=region: tmpl_stack(ctx, f, a, r, 0.72)
    for side in ("left", "right"):
        for frac in (0.4, 0.5):
            for align in ("left", "center"):
                yield 0.0, lambda f=flow, s=side, fr=frac, a=align: tmpl_split(ctx, f, s, fr, a)
    for region in ("full", "left", "right", "top", "bottom"):
        for align in ("center", "left"):
            if region == "full" or (region in ("left", "right") and ctx.W / ctx.H > 1.2) \
                    or (region in ("top", "bottom") and ctx.H / ctx.W > 1.2):
                yield 0.0, lambda f=flow, a=align, r=region: tmpl_backdrop(ctx, f, a, r)
    for side in ("left", "right", "top", "bottom"):
        for frac in (0.4, 0.55, 0.7):
            yield 0.0, lambda f=flow, s=side, fr=frac: tmpl_card_panel(ctx, f, s, fr)
    for side in ("left", "right", "top", "bottom"):
        for frac in (0.3, 0.45):
            yield 0.0, lambda f=flow, s=side, fr=frac: tmpl_panel(ctx, f, s, fr)
    for scale in (1.0, 0.8, 0.6, 0.4):
        for ncols in (1, 2):
            yield 0.0, lambda f=flow, sc=scale, nc=ncols: tmpl_row(ctx, f, sc, nc)
    # text cards turned into vertical strips (reading bottom to top); text
    # ribbons turned sideways in the source join the row upright
    turned = [u for u in ctx.model["units"] if u.role == "ribbon" and any(e["type"] == "text" for e in u.els)
              and all(abs(abs(norm_rot(e.get("rotation", 0))) - 90) < 2 for e in u.els)]
    for scale in (1.0, 0.8):
        yield 0.0, lambda f=flow + turned, sc=scale: tmpl_row(ctx, f, sc, 1, rotate=True)
    if ctx.W / ctx.H >= 2.2:
        for side in ("left", "right"):
            def rot_panel(f=flow, s=side):
                ctx.rotate = True
                try:
                    P = tmpl_panel(ctx, f, s, 0.45)
                finally:
                    ctx.rotate = False
                if P is not None:
                    P.meta["template"] += "-rot"
                return P
            yield 0.0, rot_panel


def _build(source, target, P):
    out = []
    for el in source["elements"]:
        e = copy.deepcopy(el)
        e.pop("_flow", None)
        for k, v in P.props.get(el["id"], {}).items():
            e[k] = round(v, 2) if isinstance(v, float) else v
        if e["type"] == "text":
            e["font_size"] = max(1, int(round(e["font_size"])))
            # tracking is set in ems: it scales with the type
            ls = float(el.get("letter_spacing", 0) or 0)
            if ls and el.get("font_size"):
                e["letter_spacing"] = round(ls * e["font_size"] / float(el["font_size"]), 3)
        out.append(e)
    return {"canvas": {"width": target["width"], "height": target["height"]}, "elements": out}


def _fallback(source, target):
    """Uniform contain-scale, centered. Never fails; used only on errors."""
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    W, H = target["width"], target["height"]
    s = min(W / sw, H / sh)
    ox, oy = (W - sw * s) / 2, (H - sh * s) / 2
    out = []
    for el in source["elements"]:
        e = dict(el)
        e["x"], e["y"] = ox + el["x"] * s, oy + el["y"] * s
        e["width"], e["height"] = el["width"] * s, el["height"] * s
        if e["type"] == "text":
            e["font_size"] = max(1, int(el["font_size"] * s))
        out.append(e)
    return {"canvas": {"width": W, "height": H}, "elements": out}


def _broken(tight, loose):
    """How many compositions of the tight reading the loose reading splits."""
    where = {}
    for i, u in enumerate(loose["units"]):
        for eid in u.ids:
            where[eid] = i
    return sum(1 for u in tight["units"] if len(u.ids) > 1 and len({where.get(e) for e in u.ids}) > 1)


BREAK_COST = 2.0   # per composition taken apart: it has to buy readability or space
RESTACK_COST = 0.5  # per composition re-stacked along the other axis
ROTATE_COST = 0.5   # per text piece turned into a vertical strip
SAME_BONUS = 3.0    # the same design, rescaled, at a near-same shape (fades to 0 by a 4x change)
UPRIGHT_COST = 0.5  # per turned text set upright (a restructure, like turning one)


# The critic compares candidates tier by tier: a visible defect first, then
# whether every text reads, then the designer's arrangement, and only then the
# rest of the score. Each tier is rounded into bands, so a trivial difference
# falls through to the next tier; within the last band the full score decides.
# The arrangement tier fades as the shape change grows: a near-same shape keeps
# the designer's structure, an extreme one (a square into a strip) may flip
# rows and columns or turn text, as long as the text reads.
TIERS = (("dropped", "eyes", "collisions"),
         ("readability", "contrast", "busy_bg"),
         ("subject_hidden", "picture_cover", "arrangement", "subject_relation", "thumbnail",
          "grouping", "restack", "rotate"))
BANDS = (5.0, 6.0, 6.0)   # a band is about one clear defect / one text well below legible / one relation lost
TIERED = os.environ.get("CHAMELEON_TIERS") == "1"   # off: the terms are not yet defect measures (see review/h2h_sprint7pm/T3_changes.png)


def _fade(ctx):
    """1 up to a 1.5x change of aspect, 0 from 4x on."""
    sev = abs(math.log((ctx.W / ctx.H) / (ctx.Ws / ctx.Hs)))
    lo, hi = math.log(1.5), math.log(4.0)
    return min(1.0, max(0.0, (hi - sev) / (hi - lo)))


def _key(br, fade):
    if not TIERED:
        return (sum(br.values()),)
    t = [sum(br.get(k, 0.0) for k in names) for names in TIERS]
    total = sum(br.values()) - (1 - fade) * t[2]
    t[2] *= fade
    # whole bands of penalty: -0.6 is band 0 (no worse than clean), -6.5 is band -1
    return tuple(-math.floor(max(0.0, -t[i]) / BANDS[i]) for i in range(3)) + (total,)


# Read check: the top few candidates are rendered at the judge's view and read
# with OCR (engine/legible.py); a candidate that loses words the source's own
# render keeps loses to one that keeps them, before the score is consulted.
READ_CHECK = os.environ.get("CHAMELEON_READ_CHECK", "1") != "0"
READ_TOP = 4


def _reading_order(boxes):
    """Text ids in reading order: lines top to bottom (boxes that share most of
    their height are one line), left to right within a line."""
    items = sorted(boxes.items(), key=lambda kv: (kv[1][1] + kv[1][3]) / 2)
    lines = []
    for i, b in items:
        for ln in lines:
            lb = ln[0][1]
            ov = min(b[3], lb[3]) - max(b[1], lb[1])
            if ov > 0.5 * min(b[3] - b[1], lb[3] - lb[1]):
                ln.append((i, b))
                break
        else:
            lines.append([(i, b)])
    return [i for ln in lines for i, _ in sorted(ln, key=lambda t: t[1][0])]


def _order_agree(a, b):
    """Share of text pairs read in the same order in a and b."""
    pos = {i: k for k, i in enumerate(b)}
    ids = [i for i in a if i in pos]
    n = ok = 0
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            n += 1
            ok += pos[ids[x]] < pos[ids[y]]
    return ok / n if n else 1.0


def _read_check(source, target, pool):
    """A veto, not a trade: the plain pick stands unless OCR finds that its
    headline or call to action has lost a third or more of the words the
    source's own render reads. Then the next candidates in score order are
    tried, and the first that keeps them is taken, provided it keeps the
    source's reading order and does not shrink the headline below 80%."""
    from . import legible
    pool = sorted(pool, key=lambda c: c[0], reverse=True)
    first = pool[0]
    texts = [e for e in source["elements"] if e["type"] == "text"]
    troles = first[4].get("troles", {})
    sizes = sorted((e["font_size"] for e in texts), reverse=True)
    head = [e["id"] for e in texts if sizes and e["font_size"] >= 0.9 * sizes[0]]
    key_ids = set(head) | {i for i, r in troles.items() if r == "cta"}
    if not key_ids:
        return first[2], first[1], first[3], first[4], first[5]
    plain = {"canvas": source["canvas"], "elements": [{k: v for k, v in e.items() if k != "_flow"}
                                                       for e in source["elements"]]}

    def broken(P):
        rec = legible.recall(_build(source, target, P))
        if rec is None:
            return None
        return [i for i in key_ids if src.get(i, 0) >= 0.6 and rec.get(i, 0.0) < src[i] - 0.34]

    src = legible.recall(plain)
    if src is None:
        return first[2], first[1], first[3], first[4], first[5]
    bad = broken(first[2])
    if not bad:
        return first[2], first[1], first[3], first[4], first[5]
    src_order = _reading_order({e["id"]: (e["x"], e["y"], e["x"] + e["width"], e["y"] + e["height"]) for e in texts})

    def agree(P):
        return _order_agree(_reading_order({e["id"]: P.vis[e["id"]] for e in texts if e["id"] in P.vis}), src_order)

    def head_fs(P):
        return max((P.props.get(i, {}).get("font_size", 0) for i in head), default=0)
    need, floor_fs = agree(first[2]) - 1e-9, 0.8 * head_fs(first[2])
    seen, tried = {first[5]["template"]}, 0
    for c in pool[1:]:
        if c[5]["template"] in seen:
            continue
        seen.add(c[5]["template"])
        if agree(c[2]) < need or head_fs(c[2]) < floor_fs:
            continue
        tried += 1
        if tried > READ_TOP:
            break
        if broken(c[2]) == []:
            return c[2], c[1], dict(c[3], read_veto=1.0), c[4], c[5]
    return first[2], first[1], first[3], first[4], first[5]


def _unwrap(source):
    """Marks texts whose hand-wrapped paragraph breaks can be dropped when the
    text re-flows into a new width (it then wraps cleanly instead of breaking
    at the old points). Text that moves rigidly keeps its content."""
    from .textfit import unwrap_breaks
    src = copy.deepcopy(source)
    sizes = [e["font_size"] for e in src["elements"] if e["type"] == "text"]
    top = max(sizes) if sizes else 0
    for e in src["elements"]:
        # the design's headline (its largest type) keeps its designed lines
        if e["type"] == "text" and isinstance(e.get("content"), str) and e["font_size"] < 0.9 * top:
            flow = unwrap_breaks(e["content"])
            if flow != e["content"]:
                e["_flow"] = flow     # applied only where the text re-flows into a new width
    return src


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    return _resize(_unwrap(source), target_canvas)


def _resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    resize.last_choice = ({"template": "fallback"}, 0.0, {})
    resize.last_model = None
    resize.last_P = None
    try:
        tight = analyze(source)
        readings = [(tight, 0)]
        loose = analyze(source, compose=False)
        n = _broken(tight, loose)
        if n:
            readings.append((loose, n))
        best, best_s, best_br, best_meta, best_model = None, -1e18, None, None, None
        best_key = None
        pool = []
        from .blocks import find_cut
        passes = []
        for model, broken in readings:
            passes.append((model, broken, False))
            carea = model["canvas"][0] * model["canvas"][1]
            if any(u.kind == "group" and u.role in FLOW_ROLES and find_cut(u, carea) for u in model["units"]):
                passes.append((model, broken, True))     # compositions may re-stack
        # grids restructure the design: tried only when no faithful layout
        # can make every text legible
        # A design with turned text also gets candidates with that text set
        # upright: turned text reads far worse at the judge's view (OCR read
        # 29-65% of its words at any size), so horizontal may read better.
        turned = any(e["type"] == "text" and abs(norm_rot(e.get("rotation", 0) or 0)) > 1
                     for e in source["elements"])
        stages = [(False, False)] + ([(False, True)] if turned else []) + [(True, True)]
        for (grids, upright), (model, broken, restack) in [(g, p) for g in stages for p in passes]:
            if grids and best_br is not None and best_br.get("readability", 0.0) >= -5.0:
                break
            ctx = Ctx(model, target_canvas)
            ctx.restack = restack
            ctx.grids = grids
            ctx.upright = upright
            for th, make in _candidates(ctx, model):
                try:
                    P = make()
                    if P is None:
                        continue
                    if finalize(ctx, P, model) is None:
                        continue
                    s, br = score(ctx, P, model)
                    if broken:
                        br["grouping"] = -BREAK_COST * broken
                        s += br["grouping"]
                    if getattr(P, "lifted", 0):
                        br["grouping"] = br.get("grouping", 0.0) - BREAK_COST * P.lifted
                        s -= BREAK_COST * P.lifted
                    if upright and not grids:
                        n_up = sum(1 for e in source["elements"] if e["type"] == "text"
                                   and abs(norm_rot(e.get("rotation", 0) or 0)) > 1
                                   and abs(norm_rot(P.props.get(e["id"], {}).get("rotation", e.get("rotation", 0)) or 0)) <= 1)
                        if not n_up:
                            continue      # nothing was set upright: the plain pass already has it
                        br["upright"] = -UPRIGHT_COST * n_up
                        s += br["upright"]
                    if getattr(P, "rotated", 0):
                        br["rotate"] = -ROTATE_COST * P.rotated
                        s += br["rotate"]
                    if getattr(P, "restacked", 0):
                        br["restack"] = -RESTACK_COST * P.restacked
                        s += br["restack"]
                        P.meta["template"] += " restack"
                except Exception:  # one bad candidate never kills the run
                    if DEBUG:
                        import traceback
                        traceback.print_exc()
                    continue
                if DEBUG:
                    print(f"    {('loose ' if broken else '') + P.meta['template']:<34} score={s:7.2f} "
                          + " ".join(f"{k}={v:.1f}" for k, v in br.items()))
                if P.meta["template"].startswith("same-"):
                    # faithfulness first: for a standard change of format the
                    # designer's own arrangement wins unless clearly beaten
                    br["same_design"] = SAME_BONUS * _fade(ctx)
                    s += br["same_design"]
                key = _key(br, _fade(ctx))
                if READ_CHECK:
                    meta = dict(P.meta, frame_modes=dict(getattr(P, "frame_modes", {})))
                    if broken:
                        meta["template"] = "loose-" + meta["template"]
                    pool.append((key, s, P, br, model, meta))
                if best_key is None or key > best_key:
                    best_key = key
                    best, best_s, best_br, best_model = P, s, br, model
                    best_meta = dict(P.meta, frame_modes=dict(getattr(P, "frame_modes", {})))
                    if broken:
                        best_meta["template"] = "loose-" + best_meta["template"]
        if best is None:
            return _fallback(source, target_canvas)
        if DEBUG:
            print(f"  -> {best_meta['template']} score={best_s:.2f}")
        if READ_CHECK and len(pool) > 1:
            best, best_s, best_br, best_model, best_meta = _read_check(source, target_canvas, pool)
        resize.last_choice = (best_meta, best_s, best_br)
        resize.last_P = best
        resize.last_model = best_model
        return _build(source, target_canvas, best)
    except Exception:
        if DEBUG:
            import traceback
            traceback.print_exc()
        return _fallback(source, target_canvas)
