"""Write element boxes from a placed layout and guarantee the hard rules (owner: final JSON)."""
from __future__ import annotations

import copy
import math
from typing import Dict

from .analyse import Scene
from .layout import Placed
from .measure import longest_word, text_block


def _fit_text(el: dict, fs: int, width: float) -> int:
    while fs > 1:
        n, maxw, h = text_block(el, fs, width)
        if maxw <= width + 0.5 and longest_word(el, fs) <= width + 0.5:
            return fs
        fs = fs - 1 if fs <= 12 else int(fs * 0.95)
    return max(1, fs)


def _ink(e):
    """Drawn glyph ink box on the canvas (L0 fundamentals: includes overhang), or None."""
    try:
        from engine.fundamentals.text import measure_text
        return measure_text(e).canvas_ink_box()
    except Exception:  # noqa: BLE001
        return None


def _rot_aabb(x, y, w, h, deg):
    if not deg:
        return x, y, w, h
    t = math.radians(deg)
    c, s = abs(math.cos(t)), abs(math.sin(t))
    rw, rh = w * c + h * s, w * s + h * c
    cx, cy = x + w / 2, y + h / 2
    return cx - rw / 2, cy - rh / 2, rw, rh


def emit(S: Scene, placed: Dict[int, Placed], W: float, H: float) -> dict:
    out = [copy.deepcopy(e) for e in S.els]
    for i in S.backdrop:
        e = out[i]
        ow, oh = float(e.get("width", 1)), float(e.get("height", 1))
        e["x"], e["y"], e["width"], e["height"] = 0.0, 0.0, float(W), float(H)
        if e.get("type") == "image" and abs((W / H) / (ow / oh) - 1) > 0.01:
            e["fit"] = "cover"
    for i in S.panels:
        e = out[i]
        if e.get("type") == "image":
            k = min(W / S.W, H / S.H)
            w, h = float(e["width"]) * k, float(e["height"]) * k
            cx = (float(e["x"]) + float(e["width"]) / 2) / S.W * W
            cy = (float(e["y"]) + float(e["height"]) / 2) / S.H * H
            e["x"], e["y"], e["width"], e["height"] = cx - w / 2, cy - h / 2, w, h
            continue
        e["x"] = e["x"] / S.W * W
        e["y"] = e["y"] / S.H * H
        e["width"] = max(1.0, e["width"] / S.W * W)
        e["height"] = max(1.0, e["height"] / S.H * H)
    sd = min(W / S.W, H / S.H)   # decor keeps its share of the short side
    sd = min(sd, max(W / S.W, H / S.H))
    for i in S.decor:
        e = out[i]
        x, y, w, h = float(e["x"]), float(e["y"]), float(e["width"]), float(e["height"])
        cx, cy = (x + w / 2) / S.W, (y + h / 2) / S.H
        nw, nh = max(1.0, w * sd), max(1.0, h * sd)
        fp = min(20.0, 0.4 * min(W, H)) if e.get("type") == "image" else 1.0
        if min(w * sd, h * sd) < fp:          # bitmap rounding: keep images >= 20 px, aspect kept
            k = fp / max(1e-6, min(w * sd, h * sd))
            k = max(1.0, min(k, 0.9 * W / (w * sd), 0.9 * H / (h * sd)))   # never wider than the canvas
            nw, nh = w * sd * k, h * sd * k
        e["x"], e["y"], e["width"], e["height"] = cx * W - nw / 2, cy * H - nh / 2, nw, nh
        inside = x >= -0.5 and y >= -0.5 and x + w <= S.W + 0.5 and y + h <= S.H + 0.5
        if inside:
            e["x"] = min(max(0.0, e["x"]), W - nw)
            e["y"] = min(max(0.0, e["y"]), H - nh)
        else:
            # bleed no further than the source: keep at least its on-canvas share per axis
            fx = max(0.0, min(x + w, S.W) - max(x, 0.0)) / w
            fy = max(0.0, min(y + h, S.H) - max(y, 0.0)) / h
            ox, oy = (1 - fx) * nw, (1 - fy) * nh
            e["x"] = min(max(-ox, e["x"]), W - nw + ox) if nw - ox <= W else e["x"]
            e["y"] = min(max(-oy, e["y"]), H - nh + oy) if nh - oy <= H else e["y"]
    for b in S.blocks:
        pl = placed.get(b.bid)
        if pl is None:
            continue
        if b.kind == "text":
            e = out[b.text_idx]
            e["x"], e["y"], e["width"], e["height"] = pl.x, pl.y, pl.w, pl.h
            fs = int(pl.fs or e.get("font_size", 16))
            # exact measure: keep the planned box (fast advances can differ by a wrap)
            fs = _fit_text(e, fs, pl.w)
            while fs > 1:
                n, maxw, th = text_block(e, fs, pl.w)
                if th <= pl.h * 1.03 + 1:
                    break
                fs -= 1
            _set_fs(e, fs)
            continue
        s = pl.w / b.w
        for i in b.members:
            e = out[i]
            src = S.els[i]
            nx = pl.x + (float(src["x"]) - b.x) * s
            ny = pl.y + (float(src["y"]) - b.y) * s
            nw = float(src["width"]) * s
            nh = float(src["height"]) * s
            floor_px = 20.0 if e.get("type") == "image" else 1.0   # bitmap rounding stays under 5%
            floor_px = min(floor_px, 0.4 * min(W, H))
            if min(nw, nh) < floor_px:
                k = floor_px / max(1e-6, min(nw, nh))
                k = max(1.0, min(k, 0.9 * W / nw, 0.9 * H / nh))
                nx, ny = nx + nw / 2 - nw * k / 2, ny + nh / 2 - nh * k / 2
                nw, nh = nw * k, nh * k
            e["x"], e["y"], e["width"], e["height"] = nx, ny, nw, nh
            if e.get("type") == "text":
                fs0 = float(src.get("font_size", 16) or 16)
                fs = max(1, int(fs0 * s))
                fs = _fit_text(e, fs, nw)
                _set_fs(e, fs)
                n, maxw, th = text_block(e, fs, nw)
                if maxw > nw:
                    e["x"] = nx + nw / 2 - (maxw + 1) / 2
                    e["width"] = nw = maxw + 1
                    n, maxw, th = text_block(e, fs, nw)
                if th > nh:
                    cy = ny + nh / 2
                    e["y"] = cy - th / 2
                    e["height"] = th
    return finalize(S, {"canvas": {"width": W, "height": H}, "elements": out})


def _set_fs(e: dict, fs: int) -> None:
    """Change font size; letter spacing is in px, so it scales with the font."""
    old = float(e.get("font_size", fs) or fs)
    if e.get("letter_spacing"):
        e["letter_spacing"] = float(e["letter_spacing"]) * fs / max(1e-6, old)
    e["font_size"] = fs


def finalize(S: Scene, design: dict) -> dict:
    W = float(design["canvas"]["width"])
    H = float(design["canvas"]["height"])
    bleed_ok = set(S.decor)
    for k, e in enumerate(design["elements"]):
        for key in ("x", "y", "width", "height"):
            e[key] = float(e.get(key, 0.0))
        if e.get("type") == "text":
            _set_fs(e, int(max(1, round(float(e.get("font_size", 12))))))
            fs = int(e["font_size"])
            # make sure text fits the canvas, then the box covers the rendered text
            w = min(e["width"], W)
            fs = _fit_text(e, fs, w)
            n, maxw, th = text_block(e, fs, w)
            while th > H and fs > 1:
                fs -= 1
                n, maxw, th = text_block(e, fs, w)
            _set_fs(e, fs)
            if maxw > w:
                w2 = min(W, maxw + 1)
                e["x"] -= (w2 - w) / 2
                w = w2
                n, maxw, th = text_block(e, fs, w)
            if maxw > w + 0.5 and e.get("letter_spacing"):
                e["letter_spacing"] = 0.0     # degenerate: spacing alone overflows the canvas
                n, maxw, th = text_block(e, fs, w)
            e["width"] = w
            e["height"] = th      # text draws from the box top; hug the rendered lines
        if k in bleed_ok:
            src = S.els[k]
            sx, sy, sw, sh = float(src["x"]), float(src["y"]), float(src["width"]), float(src["height"])
            if sx < -0.5 or sy < -0.5 or sx + sw > S.W + 0.5 or sy + sh > S.H + 0.5:
                continue
        rot = float(e.get("rotation", 0) or 0)
        ax, ay, aw, ah = _rot_aabb(e["x"], e["y"], e["width"], e["height"], rot)
        if (aw > W or ah > H) and e.get("type") == "text" and not rot:
            fs = int(e["font_size"])
            w = min(e["width"], W)
            while fs > 1:
                n, maxw, th = text_block(e, fs, w)
                if th <= H and maxw <= w + 0.5:
                    break
                fs -= 1
            n, maxw, th = text_block(e, fs, w)
            _set_fs(e, fs)
            e["width"], e["height"] = w, min(H, max(th, 1.0))
            ax, ay, aw, ah = e["x"], e["y"], e["width"], e["height"]
        if aw > W or ah > H:
            f = min(W / aw, H / ah) * 0.999
            cx, cy = e["x"] + e["width"] / 2, e["y"] + e["height"] / 2
            e["width"] *= f
            e["height"] *= f
            e["x"], e["y"] = cx - e["width"] / 2, cy - e["height"] / 2
            if e.get("type") == "text":
                _set_fs(e, max(1, int(e["font_size"] * f)))
            ax, ay, aw, ah = _rot_aabb(e["x"], e["y"], e["width"], e["height"], rot)
        if e.get("type") == "text":
            boxes = [(ax, ay, ax + aw, ay + ah)]
            ink = _ink(e)
            if ink is not None:
                boxes.append(tuple(ink))
            if rot:   # checkers may test the unrotated text box too: keep both inside
                boxes.append((e["x"], e["y"], e["x"] + e["width"], e["y"] + e["height"]))
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
            ax, ay, aw, ah = x0, y0, x1 - x0, y1 - y0
        dx = -ax if ax < 0 else (W - (ax + aw) if ax + aw > W else 0.0)
        dy = -ay if ay < 0 else (H - (ay + ah) if ay + ah > H else 0.0)
        e["x"] += dx
        e["y"] += dy
    return design
