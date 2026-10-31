"""The strict stage-one check and the clamp pass used by the combined engine's pick.

    from engine.combo.strict import faults, clamp
    faults(source, output, target_canvas, assets_dir) -> ["<id>: <why>", ...]   # [] = passes
    clamp(output, target_canvas) -> output with every box inside the canvas

Read with the rules decided on 2026-10-31 (context/notes/adobe-rules.md):

* ``rules.faults``: ids exactly once, canvas, numeric geometry, type/font/asset/shape unchanged,
  colour unchanged, no hiding (opacity, text at 2 px or less, collapsed boxes), the bleed policy.
* Content is compared byte for byte (rule 9), not with whitespace collapsed.
* A shape's ``fill_color`` is unchanged (rule 11).
* An image keeps its source aspect within 5% unless it is ``fit: "cover"`` or covers the whole
  canvas (rule 8); a ``contain`` box of another shape is a fault here.
* Nothing more than 0.5 px past the canvas, decorations included (rule 7: clamp, no bleed).
* Nothing hidden by opacity or colour alpha <= 0.05, the missing-asset placeholder included (rule 3).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from . import rules as R

EDGE = 0.5


def _hidden(src: dict, out: dict) -> List[str]:
    so = {str(e.get("id")): e for e in src.get("elements", [])}
    bad = []
    for e in out.get("elements", []):
        s = so.get(str(e.get("id")))
        if s is None:
            continue
        op = R._num(e.get("opacity"))
        op = 1.0 if op is None else op
        sop = R._num(s.get("opacity"))
        sop = 1.0 if sop is None else sop
        c, sc = R._rgba(e.get("color")), R._rgba(s.get("color")) or (0, 0, 0, 1)
        w, h = R._num(e.get("width")) or 0.0, R._num(e.get("height")) or 0.0
        if op <= 0.05 < sop:
            bad.append(f"{e['id']}: opacity 0")
        elif c and c[3] <= 0.05 < sc[3]:
            bad.append(f"{e['id']}: alpha 0")
        elif e.get("type") == "image" and (w < 2 or h < 2) and (R._num(s.get("width")) or 0) >= 2:
            bad.append(f"{e['id']}: image collapsed")
    return bad


@lru_cache(maxsize=256)
def _asset_solid(path: str) -> bool:
    """True when the asset paints (nearly) every pixel of its box: 95% of pixels fully opaque.
    A missing file paints a solid grey placeholder."""
    p = Path(path)
    if not p.is_file():
        return True
    try:
        from PIL import Image

        im = Image.open(p)
        if "A" not in im.getbands() and im.mode != "P":
            return True
        a = im.convert("RGBA").getchannel("A").histogram()
        return sum(a[250:]) >= 0.95 * sum(a)
    except Exception:  # noqa: BLE001  an unreadable asset is not counted as covering anything
        return False


def _opaque(e: dict, adir: Path) -> bool:
    """Whether the element paints its whole box: a rectangle with an opaque fill, or a solid
    picture that fills its box (cover, or drawn at its own proportions)."""
    if e.get("type") == "text" or (R._num(e.get("opacity")) if e.get("opacity") is not None else 1.0) < 0.95:
        return False
    if e.get("type") == "shape":
        c = R._rgba(e.get("fill_color"))
        return str(e.get("shape_type", "rectangle")).lower() in ("rectangle", "rect", "square") and bool(c) and c[3] >= 0.95
    if e.get("type") != "image" or not e.get("asset_id"):
        return False
    return _asset_solid(str(adir / str(e["asset_id"])))


def _buried(d: dict, assets=None) -> set:
    """Ids whose box is at least 98% covered (no visible rim) by an opaque shape or image drawn above it."""
    adir = R.bleed._assets_dir(assets)
    els = [e for e in d.get("elements", []) if None not in (R._num(e.get(k)) for k in ("x", "y", "width", "height"))]
    res = set()
    for a in els:
        ax, ay, aw, ah = (float(a[k]) for k in ("x", "y", "width", "height"))
        if aw <= 0 or ah <= 0:
            continue
        for b in els:
            if b is a or (R._num(b.get("z_index")) or 0) <= (R._num(a.get("z_index")) or 0) or not _opaque(b, adir):
                continue
            bx, by, bw, bh = (float(b[k]) for k in ("x", "y", "width", "height"))
            ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
            iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
            if ix * iy >= 0.98 * aw * ah:
                res.add(str(a.get("id")))
                break
    return res


def faults(src: dict, out: dict, tgt: dict, assets=None) -> List[str]:
    errs = R.faults(src, out, tgt, assets)
    if not isinstance(out, dict) or not isinstance(out.get("elements"), list):
        return errs
    W, H = float(tgt["width"]), float(tgt["height"])
    so = {str(e.get("id")): e for e in src.get("elements", [])}
    for e in out["elements"]:
        i = str(e.get("id"))
        s = so.get(i)
        if s is None:
            continue
        if "content" in s and e.get("content") != s.get("content") and f"{i}: content changed" not in errs:
            errs.append(f"{i}: content changed (exact)")
        if "fill_color" in s and e.get("fill_color") != s.get("fill_color"):
            errs.append(f"{i}: fill_color changed")
        x, y, w, h = (R._num(e.get(k)) for k in ("x", "y", "width", "height"))
        if None in (x, y, w, h):
            continue
        if x < -EDGE or y < -EDGE or x + w > W + EDGE or y + h > H + EDGE:
            errs.append(f"{i}: past canvas")
        sw, sh = R._num(s.get("width")) or 0.0, R._num(s.get("height")) or 0.0
        if e.get("type") == "image" and sw > 0 and sh > 0 and w > 0 and h > 0:
            full = x <= 1 and y <= 1 and x + w >= W - 1 and y + h >= H - 1
            if not full and e.get("fit") != "cover" and abs((w / h) / (sw / sh) - 1) > R.ASPECT_TOL:
                errs.append(f"{i}: aspect changed ({sw / sh:.3f} -> {w / h:.3f}, fit {e.get('fit')!r})")
    errs += [x for x in _hidden(src, out) if x not in errs]
    errs += [f"{i}: buried under an opaque layer" for i in sorted(_buried(out, assets) - _buried(src, assets))]
    return errs


def rescue(src: dict, out: dict, tgt: dict, floor: float = 3.0) -> dict:
    """Last resort when no engine's output passes: lift every text set at or below 2 px to
    ``floor`` px (its box grows by the same factor, kept inside the canvas) and give an element
    hidden by opacity its source opacity back. Collisions may follow; a dropped element may not."""
    so = {str(e.get("id")): e for e in src.get("elements", [])}
    W, H = float(tgt["width"]), float(tgt["height"])
    els = []
    for e in out.get("elements", []):
        o = dict(e)
        s = so.get(str(e.get("id")), {})
        sop = R._num(s.get("opacity"))
        if (R._num(e.get("opacity")) if e.get("opacity") is not None else 1.0) <= 0.05 < (1.0 if sop is None else sop):
            o["opacity"] = 1.0 if sop is None else sop
        fs = R._num(e.get("font_size"))
        if e.get("type") == "text" and fs is not None and fs <= R.TINY_TEXT:
            k = floor / max(fs, 0.5)
            w = min(W, float(e["width"]) * k)
            h = min(H, float(e["height"]) * k)
            o.update(font_size=floor, width=w, height=h)
        els.append(o)
    return clamp(dict(out, elements=els), tgt)


def clamp(out: dict, tgt: dict) -> dict:
    """Every box inside the canvas: a box larger than the canvas shrinks evenly (its type with it),
    then any box past an edge moves back inside. A box that covers the canvas becomes the canvas
    (an image there crops with ``cover``). Boxes already inside are left exactly as they are."""
    if not isinstance(out, dict) or not isinstance(out.get("elements"), list):
        return out
    W, H = float(tgt["width"]), float(tgt["height"])
    els = []
    for e in out["elements"]:
        x, y, w, h = (R._num(e.get(k)) for k in ("x", "y", "width", "height"))
        if None in (x, y, w, h) or (x >= -EDGE and y >= -EDGE and x + w <= W + EDGE and y + h <= H + EDGE):
            els.append(e)
            continue
        o = dict(e)
        if e.get("type") in ("image", "shape") and x <= 0.5 and y <= 0.5 and x + w >= W - 0.5 and y + h >= H - 0.5:
            o.update(x=0, y=0, width=W, height=H)
            if e.get("type") == "image":
                o["fit"] = "cover"
            els.append(o)
            continue
        k = min(1.0, W / w if w > 0 else 1.0, H / h if h > 0 else 1.0)
        if k < 1.0:
            w, h = w * k, h * k
            if e.get("type") == "text" and R._num(e.get("font_size")):
                o["font_size"] = max(1.0, float(e["font_size"]) * k)
        o.update(x=min(max(x, 0.0), W - w), y=min(max(y, 0.0), H - h), width=w, height=h)
        els.append(o)
    return dict(out, elements=els)
