"""Chameleon resize engine, lane `learn`: re-lay a design out the way Crello's designers resized
similar elements to a similar shape, then repair it so every hard rule holds.

    engine/learn/features.py  geometry/type features of each element (no roles)
    engine/learn/knn.py       nearest designer moves, weights fitted on held-out Crello families
    engine/learn/compose.py   units -> transfer -> bleed -> type -> repair
    engine/learn/textfit.py   measures text exactly like the renderer

Any failure falls back to a uniform fit-and-centre; `_finalize` enforces the hard rules on
whatever comes out.
"""

from __future__ import annotations

import math
import sys
import traceback
from typing import Any, Dict

from engine.learn.features import is_bleed, num


def resize(source: Dict[str, Any], target_canvas: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from engine.learn.compose import layout
        out = layout(source, target_canvas)
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        out = _fallback(source, target_canvas)
    return _finalize(source, out, target_canvas)


def _fallback(source, target):
    W = max(1.0, num(source["canvas"].get("width"), 1.0))
    H = max(1.0, num(source["canvas"].get("height"), 1.0))
    TW, TH = max(1.0, num(target.get("width"), 1.0)), max(1.0, num(target.get("height"), 1.0))
    s = min(TW / W, TH / H)
    ox, oy = (TW - W * s) / 2, (TH - H * s) / 2
    out = []
    for e in source.get("elements") or []:
        e = dict(e)
        if is_bleed(e, W, H):
            e.update(x=0.0, y=0.0, width=TW, height=TH)
            if e.get("type") == "image":
                e["fit"] = "cover"
        else:
            e.update(x=ox + num(e.get("x")) * s, y=oy + num(e.get("y")) * s,
                     width=max(1.0, num(e.get("width"), 1) * s), height=max(1.0, num(e.get("height"), 1) * s))
            if e.get("type") == "text":
                e["font_size"] = max(1, int(num(e.get("font_size"), 12) * s))
        out.append(e)
    return dict(canvas=dict(width=int(TW), height=int(TH)), elements=out)


def _finalize(source, out, target):
    TW, TH = max(1.0, num(target.get("width"), 1.0)), max(1.0, num(target.get("height"), 1.0))
    W = max(1.0, num(source["canvas"].get("width"), 1.0))
    H = max(1.0, num(source["canvas"].get("height"), 1.0))
    by_id = {}
    for e in out.get("elements") or []:
        if isinstance(e, dict) and e.get("id") not in by_id:
            by_id[e.get("id")] = e
    res = []
    for se in source.get("elements") or []:
        e = dict(by_id.get(se.get("id")) or se)
        for k in ("type", "content", "font_family", "asset_id"):
            if k in se:
                e[k] = se[k]
        x, y = num(e.get("x")), num(e.get("y"))
        w, h = max(1.0, num(e.get("width"), 1.0)), max(1.0, num(e.get("height"), 1.0))
        full = (e.get("type") == "image" and e.get("fit") == "cover" and w >= TW - 1 and h >= TH - 1)
        if e.get("type") == "image" and not full:
            # keep the source aspect (re-derive from the source box, fit inside the current box)
            sw, sh = max(1e-3, num(se.get("width"), 1)), max(1e-3, num(se.get("height"), 1))
            if abs(math.log((w / h) / (sw / sh))) > 0.01:
                if is_bleed(se, W, H):
                    e["fit"] = "cover"
                    x, y, w, h = 0.0, 0.0, TW, TH
                else:
                    cx, cy = x + w / 2, y + h / 2
                    side = math.sqrt(w * h)
                    ar = sw / sh
                    w, h = side * math.sqrt(ar), side / math.sqrt(ar)
                    x, y = cx - w / 2, cy - h / 2
        if e.get("type") == "image" and not full and min(num(se.get("width")), num(se.get("height"))) <= 1e-6:
            # a degenerate source box has no aspect to keep: give it a visible square, contained
            cx, cy = x + w / 2, y + h / 2
            w = h = min(max(12.0, min(w, h)), TW, TH)
            x, y = cx - w / 2, cy - h / 2
            e["fit"] = "contain"
        dmin = min(w, h)
        if e.get("type") == "image" and e.get("fit", "contain") == "contain":
            sz_ = _asset_size(e.get("asset_id"))
            if sz_:
                c = min(w / sz_[0], h / sz_[1])
                dmin = min(dmin, sz_[0] * c, sz_[1] * c)
        if e.get("type") == "image" and not full and dmin < 20.0 and max(w, h) < 4.0 * min(w, h):
            # small icons only: a hairline rule (long and thin) stays thin
            # the converter rounds to whole pixels (box, then the contained asset): below ~12 px
            # that alone distorts > 5%
            k = max(1.0, min(20.0 / max(1e-6, dmin), TW / w, TH / h, 60.0 / max(w, h)))
            cx, cy = x + w / 2, y + h / 2
            w, h = w * k, h * k
            x, y = cx - w / 2, cy - h / 2
        if e.get("type") == "image" and not full and e.get("fit", "contain") not in ("contain", "cover"):
            # a 'fill' box off the asset's own aspect would draw it distorted: contain it instead
            sz = _asset_size(e.get("asset_id"))
            if sz and abs((w / h) / (sz[0] / sz[1]) - 1) > 0.03:
                e["fit"] = "contain"
        if w > TW or h > TH:
            s = min(TW / w, TH / h)
            if e.get("type") == "image" and not full:
                w, h = w * s, h * s
            else:
                w, h = min(w, TW), min(h, TH)
        rot = num(e.get("rotation"), 0.0) % 360.0
        if rot and not full:
            # the drawn (rotated) box must stay inside the canvas too
            r = math.radians(rot)
            bw = abs(w * math.cos(r)) + abs(h * math.sin(r))
            bh = abs(w * math.sin(r)) + abs(h * math.cos(r))
            k = min(1.0, TW / max(1e-6, bw), TH / max(1e-6, bh))
            if k < 1.0:
                cx, cy = x + w / 2, y + h / 2
                w, h, bw, bh = w * k, h * k, bw * k, bh * k
                x, y = cx - w / 2, cy - h / 2
                if e.get("type") == "text":
                    e["font_size"] = max(1, int(num(e.get("font_size"), 12) * k))
            cx = min(max(bw / 2, x + w / 2), TW - bw / 2)
            cy = min(max(bh / 2, y + h / 2), TH - bh / 2)
            x, y = cx - w / 2, cy - h / 2
        x = min(max(0.0, x), TW - w)
        y = min(max(0.0, y), TH - h)
        e.update(x=float(x), y=float(y), width=float(w), height=float(h))
        if e.get("type") == "text":
            e["font_size"] = max(1, int(num(e.get("font_size"), 12)))
            fw = str(e.get("font_weight") or "regular").lower()
            if "font_weight" in e and fw not in ("regular", "bold"):
                # the renderer draws only regular/bold: say what it will draw
                e["font_weight"] = "bold" if fw in ("semibold", "extrabold", "black") else "regular"
            try:
                _text_guard(e, TW, TH)
            except Exception:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
        if num(e.get("rotation"), 0.0) % 360.0 and not full:
            _contain_rotated(e, TW, TH)
        for k, v in list(e.items()):
            if hasattr(v, "item") and not isinstance(v, (str, bytes)):
                e[k] = v.item()
        res.append(e)
    return dict(canvas=dict(width=int(TW), height=int(TH)), elements=res)


def _text_guard(e, TW, TH):
    """Fit text; as a last resort (a word that is wider than the canvas from letter spacing alone)
    tighten the letter spacing."""
    for _ in range(8):
        if _text_guard1(e, TW, TH):
            _ink_margin(e, TW, TH)
            return
        ls = num(e.get("letter_spacing"), 0.0)
        if ls <= 0.05:
            return
        e["letter_spacing"] = round(ls / 2, 3)


def _ink_margin(e, TW, TH, m=1.5):
    """Glyph ink overhangs advances by a pixel or so: keep the drawn lines m px off the edges."""
    if num(e.get("rotation"), 0.0) % 360.0:
        return
    from engine.learn import textfit
    mw, bh, _ = textfit.layout(e, e["font_size"], e["width"])
    al = e.get("text_align", "left")
    w = e["width"]
    if w > TW - 2 * m and al in ("center", "right") and mw <= TW - 2 * m - 1:
        # a full-width box cannot move: inset it so the ink clears the edge
        mw2, bh2, _ = textfit.layout(e, e["font_size"], TW - 2 * m)
        if mw2 <= TW - 2 * m:
            e.update(x=m, width=float(TW - 2 * m))
            w, mw, bh = e["width"], mw2, bh2
    off = (w - mw) / 2 if al == "center" else (w - mw) if al == "right" else 0.0
    x0 = e["x"] + off
    if mw <= TW - 2 * m:
        if x0 < m:
            e["x"] += m - x0
        elif x0 + mw > TW - m:
            e["x"] -= x0 + mw - (TW - m)
    if bh <= TH - 2 * m:
        if e["y"] < m:
            e["y"] = m
        if e["y"] + bh > TH - m:
            e["y"] = TH - m - bh
    # the box itself stays on the canvas
    e["x"] = min(max(0.0, e["x"]), max(0.0, TW - w))
    e["y"] = min(max(0.0, e["y"]), max(0.0, TH - e["height"]))


def _text_guard1(e, TW, TH):
    """Last word on text: its drawn lines stay inside its box width and the canvas, measured the
    way the renderer wraps and draws."""
    from engine.learn import textfit
    if not str(e.get("content", "")).strip():
        return True
    x, y, w, h = e["x"], e["y"], e["width"], e["height"]
    fs0 = int(e["font_size"])

    def ok(fs, w, y):
        mw, bh, _ = textfit.layout(e, fs, w)
        # drawn lines leave room for glyph ink overhang at the canvas edges
        return mw <= min(w, TW - 3.0) + 0.5 and bh <= TH - 2.0, bh

    for width, x0 in ((w, x), (TW, 0.0)):
        for fs in range(fs0, 0, -1):
            good, bh = ok(fs, width, y)
            if good:
                ny = min(y, TH - bh) if y + bh > TH else y
                e.update(x=float(x0), y=float(max(0.0, ny)), width=float(width), font_size=fs)
                if not e.get("rotation"):
                    e["height"] = float(min(TH - max(0.0, ny), max(h if width == w else 0, bh)))
                return True
    ups, f = [], float(fs0 + 1)
    while f <= max(fs0 + 1, TH):
        ups.append(int(f))
        f = max(f + 1, f * 1.12)
    for fs in ups:
        good, bh = ok(fs, TW, 0.0)
        if good:
            e.update(x=0.0, y=float(max(0.0, min(y, TH - bh))), width=float(TW), font_size=fs,
                     height=float(min(TH, bh)))
            return True
    # letter spacing the wrap ignores: wrap in a narrower box so the drawn lines fit the canvas,
    # and place the box so the alignment keeps them inside
    al = e.get("text_align", "left")
    for frac in (0.85, 0.7, 0.55, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.005):
        bw = TW * frac
        for fs in [fs0] + ups:
            mw, bh, _ = textfit.layout(e, fs, bw)
            if mw <= TW - 3.0 and bh <= TH - 2.0:
                bx = (TW - bw) / 2 if al == "center" else (TW - bw if al == "right" else 0.0)
                e.update(x=float(bx), y=float(max(0.0, min(y, TH - bh))), width=float(bw), font_size=fs,
                         height=float(min(TH, bh)))
                return True
    return False


_asset_cache = {}


def _asset_size(asset_id):
    if not asset_id:
        return None
    if asset_id in _asset_cache:
        return _asset_cache[asset_id]
    import os
    from pathlib import Path
    dirs = [Path("brand_kit/assets"), Path(__file__).resolve().parents[1] / "brand_kit" / "assets"]
    if os.environ.get("CHAMELEON_BRAND_KIT"):
        dirs.insert(0, Path(os.environ["CHAMELEON_BRAND_KIT"]) / "assets")
    size = None
    for d in dirs:
        p = d / str(asset_id)
        if p.is_file():
            try:
                from PIL import Image
                with Image.open(p) as im:
                    size = im.size
            except Exception:  # noqa: BLE001
                size = None
            break
    _asset_cache[asset_id] = size
    return size


def _rot_extent(e, w, h):
    """AABB (x0, y0, x1, y1) of what gets drawn, relative to the box's top-left, after rotation
    about the box centre. Text: its ink block (top-aligned, wrap and alignment as rendered)."""
    pts = [(0, 0), (w, 0), (0, h), (w, h)]
    if e.get("type") == "text":
        try:
            from engine.learn import textfit
            mw, bh, _ = textfit.layout(e, e["font_size"], w)
            al = e.get("text_align", "left")
            x0 = (w - mw) / 2 if al == "center" else (w - mw) if al == "right" else 0.0
            pts = [(min(0, x0), 0), (max(w, x0 + mw), 0), (min(0, x0), max(h, bh)), (max(w, x0 + mw), max(h, bh))]
        except Exception:  # noqa: BLE001
            pass
    r = math.radians(num(e.get("rotation"), 0.0))
    c, s_ = math.cos(r), math.sin(r)
    cx, cy = w / 2, h / 2
    xs, ys = [], []
    for px, py in pts:
        dx, dy = px - cx, py - cy
        xs.append(cx + dx * c - dy * s_)
        ys.append(cy + dx * s_ + dy * c)
    # the unrotated box too: a checker may read the raw box
    return min(min(xs), 0.0), min(min(ys), 0.0), max(max(xs), w), max(max(ys), h)


def _contain_rotated(e, TW, TH):
    x, y, w0, h0 = e["x"], e["y"], e["width"], e["height"]
    fs0 = e.get("font_size")
    for attempt in range(10):
        w, h = w0, h0
        if fs0 is not None:
            e["font_size"] = fs0
        ok = False
        for _ in range(40):
            x0, y0, x1, y1 = _rot_extent(e, w, h)
            if x1 - x0 <= TW + 0.01 and y1 - y0 <= TH + 0.01:
                ok = True
                break
            k = 0.9
            w, h = w * k, h * k
            if e.get("type") == "text":
                e["font_size"] = max(1, int(num(e.get("font_size"), 12) * k))
        if ok or e.get("type") != "text":
            break
        if attempt == 8:  # nothing fits rotated: lay it flat, then fit as plain text
            e["rotation"] = 0.0
            e.update(width=float(min(w0, TW)), height=float(min(h0, TH)), x=0.0, y=0.0)
            _text_guard(e, TW, TH)
            return
        e["letter_spacing"] = round(num(e.get("letter_spacing"), 0.0) / 2, 3)
    x0, y0, x1, y1 = _rot_extent(e, w, h)
    nx = min(max(-x0, x), TW - x1)
    ny = min(max(-y0, y), TH - y1)
    e.update(x=float(nx), y=float(ny), width=float(w), height=float(h))
