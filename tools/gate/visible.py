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

Opaque coverers are shapes and images at opacity >= 0.95 (a shape's fill alpha counts: a
translucent card or scrim hides nothing). An image covers only where its own
pixels are opaque as drawn (alpha >= 0.95 after its ``fit`` places the asset in the box), so a
cut-out product's transparent box hides nothing behind it; an image whose asset cannot be read
covers its whole box (the renderer paints a placeholder there). Text never covers.

An element whose own asset file is missing is exempt: the renderer would paint a grey
placeholder for it, so hiding it is the right answer, never a fault.

Assets are looked up in ``assets`` (an argument), else ``$CHAMELEON_BRAND_KIT/assets``, else the
repo's ``brand_kit/assets``. This is measuring code: the gate's own, never the engine's.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
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


def _fill_alpha(e: dict) -> float:
    """The alpha of a shape's fill colour (``rgba(r,g,b,a)`` or ``#rrggbbaa``); 1 if unreadable."""
    c = str(e.get("fill_color") or "").strip().lower()
    try:
        if c.startswith("rgba(") and c.endswith(")"):
            return max(0.0, min(1.0, float(c[5:-1].split(",")[3])))
        if c.startswith("#") and len(c) == 9:
            return int(c[7:9], 16) / 255.0
    except (ValueError, IndexError):
        pass
    return 1.0


def _cover_op(e: dict) -> float:
    """How opaque an element paints: its opacity, times its fill alpha for a shape."""
    return _op(e) * (_fill_alpha(e) if e.get("type") == "shape" else 1.0)


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


def _asset_dirs(assets=None) -> Tuple[Path, ...]:
    dirs = []
    if assets:
        dirs.append(Path(assets))
    kit = os.environ.get("CHAMELEON_BRAND_KIT")
    if kit:
        dirs.append(Path(kit) / "assets")
    dirs.append(REPO / "brand_kit" / "assets")
    return tuple(dirs)


def _find(asset_id, dirs: Sequence[Path]) -> Optional[Path]:
    if not asset_id:
        return None
    for d in dirs:
        p = d / str(asset_id)
        if p.is_file():
            return p
    return None


@lru_cache(maxsize=256)
def _alpha(path: str):
    """The asset's opaque mask (alpha >= OPAQUE) on a grid of at most 128 px, and its pixel size;
    None if unreadable."""
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as im:
            im.load()
            size = im.size
            a = im.convert("RGBA").getchannel("A")
        a.thumbnail((128, 128))
        return np.asarray(a) >= int(OPAQUE * 255), size
    except Exception:  # noqa: BLE001  unreadable: treated as an opaque box
        return None


def _opaque_at(e: dict, mask_size, px: float, py: float) -> bool:
    """Whether the image element paints an opaque pixel at canvas point (px, py)."""
    mask, (iw, ih) = mask_size
    x, y, w, h = _box(e)
    fit = e.get("fit", "contain") or "contain"
    if fit == "fill":
        u, v = (px - x) / w, (py - y) / h
    else:
        s = max(w / iw, h / ih) if fit == "cover" else min(w / iw, h / ih)
        dw, dh = iw * s, ih * s
        u, v = (px - (x + (w - dw) / 2.0)) / dw, (py - (y + (h - dh) / 2.0)) / dh
    if not (0.0 <= u < 1.0 and 0.0 <= v < 1.0):
        return False  # outside the drawn image (contain letterbox)
    mh, mw = mask.shape
    return bool(mask[min(mh - 1, int(v * mh)), min(mw - 1, int(u * mw))])


def _paint_key(i_e):
    i, e = i_e
    return (float(e.get("z_index", 0) or 0), i)


def _covered(els: List[dict], dirs: Sequence[Path] = ()) -> Dict[str, float]:
    """Share of each element's box covered by opaque elements painted after it."""
    order = [e for _, e in sorted(enumerate(els), key=_paint_key)]
    masks = {}
    for o in order:
        if o.get("type") == "image":
            p = _find(o.get("asset_id"), dirs)
            masks[id(o)] = _alpha(str(p)) if p else None
    out = {}
    for k, e in enumerate(order):
        x, y, w, h = _box(e)
        if w <= 0 or h <= 0:
            out[str(e.get("id"))] = 0.0
            continue
        later = []
        for o in order[k + 1:]:
            if o.get("type") in ("shape", "image") and _cover_op(o) >= OPAQUE:
                ox, oy, ow, oh = _box(o)
                if ow > 0 and oh > 0 and ox < x + w and ox + ow > x and oy < y + h and oy + oh > y:
                    later.append((ox, oy, ox + ow, oy + oh, o, masks.get(id(o))))
        if not later:
            out[str(e.get("id"))] = 0.0
            continue
        hit = 0
        for gi in range(GRID):
            px = x + (gi + 0.5) * w / GRID
            for gj in range(GRID):
                py = y + (gj + 0.5) * h / GRID
                if any(a <= px <= c and b <= py <= d and (m is None or _opaque_at(o, m, px, py))
                       for a, b, c, d, o, m in later):
                    hit += 1
        out[str(e.get("id"))] = hit / (GRID * GRID)
    return out


def hidden_elements(src: dict, out: dict, tgt: dict, assets=None) -> List[str]:
    W, H = float(tgt["width"]), float(tgt["height"])
    SW, SH = float(src["canvas"]["width"]), float(src["canvas"]["height"])
    dirs = _asset_dirs(assets)
    oe = {str(e.get("id")): e for e in out.get("elements", [])}
    cov_s, cov_o = _covered(src["elements"], dirs), _covered(out.get("elements", []), dirs)
    res = []
    for s in src["elements"]:
        i = str(s["id"])
        e = oe.get(i)
        if e is None:
            res.append(f"{i}: missing")
            continue
        if s.get("type") == "image" and _find(s.get("asset_id"), dirs) is None:
            continue  # its asset is missing: hiding it beats the renderer's grey placeholder
        if _op(e) < _op(s) - OPACITY_TOL:
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
