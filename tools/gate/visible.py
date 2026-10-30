"""Gate tool: which source elements an output hides (loop/STEERING.md S1).

    from tools.gate.visible import hidden_elements
    hidden_elements(source, output, target) -> ["<id>: <why>", ...]

An element counts as **visible** when all of these hold (the steering definition):

* it is in the output at all;
* its opacity is unchanged from the source (a drop of more than ``OPACITY_TOL``);
* no more than half of its box is covered by later opaque elements, unless the source covered
  it at least as much (covered share measured on a grid over the box, rotation ignored);
* at least half of its box is on the canvas, unless the source had less on its canvas or the box
  covers the whole canvas (a cover crop, loop/STEERING.md S4);
* text is at or above the floor ``text_floor(W, H)`` (or its source size, if that was smaller);
* an image's shorter side is at least 2% of the canvas's shorter side, or its source share of
  the source canvas's shorter side, whichever is smaller;
* any element keeps at least 1 px on both sides when the source had more.

Opaque coverers are shapes and images at opacity >= 0.95; an image covers a grid point only where
its fitted asset is opaque there (alpha >= 128, through the renderer's fit/crop geometry), so a
product cutout or logo over a background no longer "hides" it. An image whose asset file is
missing may be faded out: the renderer would draw a grey "missing asset" placeholder. Text never
covers. This is measuring code: the gate's own, never the engine's.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

OPACITY_TOL = 0.05
COVER_MAX = 0.5
OPAQUE = 0.95
GRID = 24


def text_floor(W: float, H: float) -> float:
    """The smallest text the gate still calls visible: 6 px, or 1% of the shorter side."""
    return max(6.0, 0.01 * min(W, H))


def _op(e: dict) -> float:
    v = e.get("opacity", 1)
    return 1.0 if v is None else float(v)


def _box(e: dict):
    return float(e.get("x", 0) or 0), float(e.get("y", 0) or 0), float(e.get("width", 0) or 0), float(e.get("height", 0) or 0)


def _on_canvas(e: dict, W: float, H: float) -> float:
    x, y, w, h = _box(e)
    if w <= 0 or h <= 0:
        return 0.0
    iw = max(0.0, min(x + w, W) - max(x, 0.0))
    ih = max(0.0, min(y + h, H) - max(y, 0.0))
    return iw * ih / (w * h)


def _covers(e: dict, W: float, H: float) -> bool:
    x, y, w, h = _box(e)
    return w > 0 and h > 0 and x <= 0.6 and y <= 0.6 and x + w >= W - 0.6 and y + h >= H - 0.6


def _paint_key(i_e):
    i, e = i_e
    return (float(e.get("z_index", 0) or 0), i)


def _assets(assets=None) -> Path:
    from tools import bleed

    return bleed._assets_dir(assets)


@lru_cache(maxsize=256)
def _alpha(path: str):
    """(alpha array of the asset, width, height), or None if unreadable."""
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            a = im.convert("RGBA").split()[-1]
            a.thumbnail((256, 256))
            return np.asarray(a), w, h
    except Exception:  # noqa: BLE001
        return None


def _opaque_at(o: dict, assets) -> Optional[Callable[[float, float], bool]]:
    """For an image, a test "is the drawn asset opaque at canvas point (x, y)"; None = the box."""
    if o.get("type") != "image" or not o.get("asset_id"):
        return None
    got = _alpha(str(_assets(assets) / str(o["asset_id"])))
    if got is None:
        return None
    a, aw, ah = got
    from fundamentals.image import fit_geometry

    ox, oy, ow, oh = _box(o)
    f = fit_geometry(ow, oh, aw, ah, o.get("fit", "contain"))
    cx0, cy0, cx1, cy1 = f.content
    kx0, ky0, kx1, ky1 = f.crop
    th, tw = a.shape

    def test(px: float, py: float) -> bool:
        lx, ly = px - ox, py - oy
        if not (cx0 <= lx < cx1 and cy0 <= ly < cy1):
            return False
        ax = kx0 + (lx - cx0) / max(1e-9, cx1 - cx0) * (kx1 - kx0)
        ay = ky0 + (ly - cy0) / max(1e-9, cy1 - cy0) * (ky1 - ky0)
        i, j = min(th - 1, max(0, int(ay * th / ah))), min(tw - 1, max(0, int(ax * tw / aw)))
        return bool(a[i, j] >= 128)

    return test


def _missing_asset(e: dict, assets) -> bool:
    return e.get("type") == "image" and bool(e.get("asset_id")) and not (_assets(assets) / str(e["asset_id"])).is_file()


def _covered(els: List[dict], assets=None) -> Dict[str, float]:
    """Share of each element's box covered by opaque elements painted after it."""
    order = [e for _, e in sorted(enumerate(els), key=_paint_key)]
    out = {}
    for k, e in enumerate(order):
        x, y, w, h = _box(e)
        if w <= 0 or h <= 0:
            out[str(e.get("id"))] = 0.0
            continue
        later = []
        for o in order[k + 1:]:
            if o.get("type") in ("shape", "image") and _op(o) >= OPAQUE:
                ox, oy, ow, oh = _box(o)
                if ow > 0 and oh > 0 and ox < x + w and ox + ow > x and oy < y + h and oy + oh > y:
                    later.append((ox, oy, ox + ow, oy + oh, _opaque_at(o, assets)))
        if not later:
            out[str(e.get("id"))] = 0.0
            continue
        hit = 0
        for gi in range(GRID):
            px = x + (gi + 0.5) * w / GRID
            for gj in range(GRID):
                py = y + (gj + 0.5) * h / GRID
                if any(a <= px <= c and b <= py <= d and (t is None or t(px, py)) for a, b, c, d, t in later):
                    hit += 1
        out[str(e.get("id"))] = hit / (GRID * GRID)
    return out


def hidden_elements(src: dict, out: dict, tgt: dict, assets=None) -> List[str]:
    W, H = float(tgt["width"]), float(tgt["height"])
    SW, SH = float(src["canvas"]["width"]), float(src["canvas"]["height"])
    oe = {str(e.get("id")): e for e in out.get("elements", [])}
    cov_s, cov_o = _covered(src["elements"], assets), _covered(out.get("elements", []), assets)
    res = []
    for s in src["elements"]:
        i = str(s["id"])
        e = oe.get(i)
        if e is None:
            res.append(f"{i}: missing")
            continue
        if _op(e) < _op(s) - OPACITY_TOL and not _missing_asset(e, assets):
            res.append(f"{i}: opacity {_op(s):.2f} -> {_op(e):.2f}")
            continue
        if cov_o.get(i, 0) > COVER_MAX and cov_o[i] > cov_s.get(i, 0) + 1e-6:
            res.append(f"{i}: {cov_o[i]:.0%} covered (source {cov_s.get(i, 0):.0%})")
            continue
        on_o, on_s = _on_canvas(e, W, H), _on_canvas(s, SW, SH)
        if on_o < 0.5 and on_o < on_s - 1e-6 and not _covers(e, W, H):
            res.append(f"{i}: {on_o:.0%} on canvas")
            continue
        _, _, w, h = _box(e)
        _, _, sw, sh = _box(s)
        if min(w, h) < 1 and min(sw, sh) >= 1:
            res.append(f"{i}: shrunk to {w:.1f}x{h:.1f}")
            continue
        if e.get("type") == "text":
            fs, sfs = float(e.get("font_size", 0) or 0), float(s.get("font_size", 0) or 0)
            floor = min(text_floor(W, H), sfs) if sfs > 0 else text_floor(W, H)
            if fs < floor - 1e-6:
                res.append(f"{i}: text {fs:.1f}px < floor {floor:.1f}")
        elif e.get("type") == "image":
            need = min(0.02 * min(W, H), min(sw, sh) / max(1e-9, min(SW, SH)) * min(W, H))
            if min(w, h) < need - 1e-6:
                res.append(f"{i}: image {min(w, h):.1f}px < {need:.1f}")
    return res
