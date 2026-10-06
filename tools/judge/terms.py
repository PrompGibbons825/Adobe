"""Source-relative proxy judge (dev tool, not engine code).

`tools/proxy.py` scores an output on its own: every pixel overlap between two
elements without a shared `group_id` is a collision, so copy sitting on its own
panel (the designer's intent) counts against the layout. This module scores an
output *against its source*: only what the resize made worse is penalised.

    rep_src = measure(source_design)          # cacheable, JSON-serialisable
    rep_out = measure(output_design)
    t = terms(rep_src, rep_out)               # {term: value >= 0}
    s = penalty(t, weights)                   # sum of weight * term (lower is better)

What `measure` records per element comes from the official converter and
cr-renderer: each element is rendered alone on a padded transparent canvas
(exact ink, including ink that falls off the canvas), occlusion by opaque
layers above it is resolved, and every text is measured against the render of
the stack below it (WCAG contrast at the 10th percentile of glyph + halo,
p90 luminance gradient after resampling the region so the cap height is
10 px, as engine/smart/pixels.py does).

Overlaps that already existed in the source (copy on its panel, a label on its
button, a sticker on a product) are allowed; copy that spills off the surface
it sat on, or overlaps more than it did, is not.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LIMIT = 800  # long side the design is measured at (plenty for the judge's 512-1024 view)

# Hand-set starting weights (research 04 §4.4, adapted to the source-relative terms).
HAND_WEIGHTS = {
    "dropped_text": 2.0, "dropped_art": 1.5, "contrast_loss": 2.5, "busy_gain": 2.0, "abs_low_contrast": 0.0,
    "abs_busy": 0.0, "coll_text_art": 3.0, "coll_text_text": 3.0, "coll_art": 1.0, "spill": 1.5,
    "cut_text": 2.0, "cut_art": 1.0, "tiny512": 1.0, "tiny1024": 2.0, "text_shrink": 1.0,
    "art_shrink": 0.5, "hierarchy": 1.0, "align_lost": 0.5, "near_miss": 0.5, "tight_margin": 0.5,
    "imbalance": 0.5, "underfill": 0.5, "overfill": 0.5, "aspect": 3.0, "order_flip": 0.5, "bare_canvas": 1.0,
}


# ----------------------------------------------------------------------------- rendering


def kit_dir() -> Path:
    env = os.environ.get("CHAMELEON_BRAND_KIT")
    return Path(env) if env else REPO / "brand_kit"


_R = None


def renderer():
    global _R
    if _R is None:
        from renderer.converter.convert import _get_renderer

        fonts = kit_dir() / "fonts"
        if not fonts.is_dir():
            fonts = REPO / "brand_kit" / "fonts"
        _R = _get_renderer(str(fonts))
    return _R


def _raw_surfaces() -> None:
    """Let cr-renderer hand back the surface pixels (format="raw") instead of a
    PNG: identical values, and PNG encoding was half of the measuring time."""
    import cr_renderer.image_utils as iu
    import skia

    if getattr(iu, "_judge_raw", False):
        return
    orig = iu.encode_surface

    def encode(surface, format):
        if format == "raw":
            return surface.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType,
                                                       alphaType=skia.kUnpremul_AlphaType)
        return orig(surface, format)

    iu.encode_surface = encode
    iu._judge_raw = True


def render(design: dict, transparent: bool = False, pad: int = 0, max_size: int = 4096) -> np.ndarray:
    """RGBA float array in [0, 1]; `pad` grows the canvas on every side so ink
    that falls off the canvas can be measured."""
    from renderer.converter.convert import convert_chameleon_to_crello
    import skia

    _raw_surfaces()
    d = design
    if pad:
        d = {"canvas": {"width": design["canvas"]["width"] + 2 * pad, "height": design["canvas"]["height"] + 2 * pad},
             "elements": [dict(e, x=e["x"] + pad, y=e["y"] + pad) for e in design["elements"]]}
    R = renderer()
    ex, _ = convert_chameleon_to_crello(d, R.font_manager, str(kit_dir() / "assets"))
    raw = R.render(ex, max_size=max_size, format="raw", canvas_color=None if transparent else skia.ColorWHITE)
    return raw.astype(np.float32) / 255.0


def render_png(design: dict, path: Path, max_size: int = 2048) -> None:
    """Exactly what engine/main.py writes (render_design_to_png): white canvas, max 2048."""
    from renderer.converter.convert import convert_chameleon_to_crello

    R = renderer()
    ex, _ = convert_chameleon_to_crello(design, R.font_manager, str(kit_dir() / "assets"))
    Path(path).write_bytes(R.render(ex, max_size=max_size, format="png"))


def downscale(design: dict, limit: int = LIMIT) -> dict:
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    f = min(1.0, limit / max(W, H))
    if f >= 1.0:
        return design
    els = []
    for e in design["elements"]:
        e = dict(e)
        for k in ("x", "y", "width", "height"):
            e[k] = float(e.get(k, 0) or 0) * f
        if "font_size" in e:
            e["font_size"] = max(1, int(round(float(e["font_size"]) * f)))
        for k in ("corner_radius", "stroke_width", "letter_spacing"):
            if e.get(k):
                e[k] = float(e[k]) * f
        els.append(e)
    return {"canvas": {"width": max(1, round(W * f)), "height": max(1, round(H * f))}, "elements": els}


# ----------------------------------------------------------------------------- pixel maths


def lum(rgb: np.ndarray) -> np.ndarray:
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    m = mask.copy()
    for _ in range(max(0, r)):
        n = m.copy()
        n[1:, :] |= m[:-1, :]
        n[:-1, :] |= m[1:, :]
        n[:, 1:] |= m[:, :-1]
        n[:, :-1] |= m[:, 1:]
        m = n
    return m


def busy_p90(gray: np.ndarray, cap_px: float) -> float:
    """p90 luminance gradient at the scale of the glyphs (cap height -> 10 px)."""
    k = 10.0 / max(1.0, cap_px)
    if abs(k - 1.0) > 0.05 and gray.size > 4:
        gh, gw = gray.shape
        im = Image.fromarray((np.clip(gray, 0, 1) * 255).astype("uint8")).resize(
            (max(3, round(gw * k)), max(3, round(gh * k))), Image.BILINEAR)
        gray = np.asarray(im, dtype=np.float32) / 255.0
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return float(np.percentile(np.hypot(gx, gy) * 0.5, 90))


# ----------------------------------------------------------------------------- measurement


FAINT = 0.03  # alpha above which an element counts as present (faint decoration included)


def _inside(e: dict, W: float, H: float) -> bool:
    from engine.smart.geom import Box, rotated_aabb

    b = Box(float(e["x"]), float(e["y"]), float(e["width"]), float(e["height"]))
    rot = float(e.get("rotation", 0) or 0)
    if rot:
        b = b.union(rotated_aabb(b, rot))
    return b.x >= -1 and b.y >= -1 and b.r <= W + 1 and b.b <= H + 1


def _paint_order(els: List[dict]) -> List[dict]:
    return [e for _, e in sorted(enumerate(els), key=lambda t: (int(t[1].get("z_index", 0) or 0), t[0]))]


def measure(design: dict, limit: int = LIMIT) -> dict:
    """Everything the terms need, as a small JSON-serialisable dict."""
    from renderer.converter.colorutil import parse_rgba

    Wn, Hn = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    dz = downscale(design, limit)
    W, H = int(dz["canvas"]["width"]), int(dz["canvas"]["height"])
    s = W / Wn
    els = _paint_order(dz["elements"])
    native = {e["id"]: e for e in design["elements"]}
    PAD = max(16, int(0.25 * max(W, H)))
    canvas = {"width": W, "height": H}

    alpha, off = {}, {}
    for e in els:
        pad = PAD if (e.get("type") == "text" or not _inside(e, W, H)) else 0
        a = render({"canvas": canvas, "elements": [e]}, transparent=True, pad=pad)[..., 3]
        m = a > FAINT
        tot = int(m.sum())
        inside = int(m[pad:pad + H, pad:pad + W].sum())
        off[e["id"]] = (1.0 - inside / tot) if tot else 0.0
        alpha[e["id"]] = a[pad:pad + H, pad:pad + W]
    masks = {i: a > 0.25 for i, a in alpha.items()}  # solid ink: collisions, glyphs
    inks = {i: a > FAINT for i, a in alpha.items()}  # any ink: presence, extent

    # occlusion by opaque layers painted later
    occ = np.zeros((H, W), bool)
    vis = {}
    for e in reversed(els):
        m = inks[e["id"]]
        area = int(m.sum())
        vis[e["id"]] = float((m & ~occ).sum() / area) if area else 0.0
        occ |= alpha[e["id"]] > 0.95

    rep: dict = {"W": Wn, "H": Hn, "s": s, "els": {}, "texts": {}, "pairs": {}}
    for e in els:
        i = e["id"]
        m = inks[i]
        area = int(m.sum())
        rec = {"type": e.get("type"), "area": area / float(W * H), "off": round(off[i], 4), "vis": round(vis[i], 4)}
        rec["solid"] = round(float(masks[i].sum()) / area, 4) if area else 0.0
        rec["bg"] = bool(area > 0.85 * W * H)
        rec["live"] = bool(area > 0 and vis[i] >= 0.2)
        if area:
            ys, xs = np.where(m)
            rec["bbox"] = [float(xs.min()) / W, float(ys.min()) / H, float(xs.max() + 1) / W, float(ys.max() + 1) / H]
        if e.get("type") == "text":
            rec["fs"] = float(native[i].get("font_size", 12) or 12)
            rec["bold"] = str(native[i].get("font_weight", "regular")) == "bold"
        if e.get("type") == "image":
            ne = native[i]
            rec["box_aspect"] = float(ne.get("width", 0) or 0) / max(1e-6, float(ne.get("height", 0) or 0))
            rec["fit"] = ne.get("fit", "contain")
            rec["opacity"] = float(ne.get("opacity", 1) if ne.get("opacity") is not None else 1)
        rep["els"][i] = rec

    # per text: contrast and busy-ness against the stack below it
    for k, t in enumerate(els):
        i = t["id"]
        if t.get("type") != "text" or not rep["els"][i]["live"]:
            continue
        glyph = masks[i]
        under = render({"canvas": canvas, "elements": els[:k]})[..., :3]
        fs = float(t.get("font_size", 12) or 12)
        halo = dilate(glyph, max(2, int(0.15 * fs))) & ~glyph
        r, g, b, a = parse_rgba(t.get("color"))
        trgb = np.array([r, g, b], dtype=np.float32) / 255.0
        bgpix = under[halo | glyph]
        eff = a * trgb + (1 - a) * bgpix
        lt, lb = lum(eff), lum(bgpix)
        c = (np.maximum(lt, lb) + 0.05) / (np.minimum(lt, lb) + 0.05)
        ys, xs = np.where(glyph)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        gray = lum(under[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4])
        rep["texts"][i] = {"wcag": round(float(np.percentile(c, 10)), 3),
                           "busy": round(busy_p90(gray, 0.7 * fs), 4)}

    # pairwise ink overlaps between live, non-background elements
    live = [e["id"] for e in els if rep["els"][e["id"]]["live"] and not rep["els"][e["id"]]["bg"]]
    for x in range(len(live)):
        for y in range(x + 1, len(live)):
            a_, b_ = live[x], live[y]
            inter = int((masks[a_] & masks[b_]).sum())
            if inter == 0:
                continue
            na, nb = int(masks[a_].sum()), int(masks[b_].sum())
            rep["pairs"][f"{a_}|{b_}"] = [round(inter / max(1, min(na, nb)), 4), round(inter / max(1, na), 4),
                                         round(inter / max(1, nb), 4)]

    painted = np.zeros((H, W), bool)
    for e in els:
        painted |= masks[e["id"]]
    rep["bare"] = round(1.0 - float(painted.mean()), 4)  # canvas no element paints (white bands, voids)

    occ_live = np.zeros((H, W), bool)
    for i in live:
        occ_live |= masks[i]
    ys, xs = np.where(occ_live)
    if len(xs):
        S = min(W, H)
        rep["centroid"] = [float(xs.mean()) / W, float(ys.mean()) / H]
        rep["coverage"] = float(occ_live.mean())
        rep["margins"] = [float(xs.min()) / S, float(ys.min()) / S, float(W - 1 - xs.max()) / S, float(H - 1 - ys.max()) / S]
    else:
        rep["centroid"], rep["coverage"], rep["margins"] = [0.5, 0.5], 0.0, [0.5] * 4
    return rep


# ----------------------------------------------------------------------------- terms
# The term definitions are shared with the engine's box-level re-rank.

from engine.smart.judge import TERMS, penalty, terms  # noqa: E402,F401


WEIGHTS_FILE = Path(__file__).resolve().parent / "data" / "weights.json"


def calibrated_weights() -> Dict[str, float]:
    return json.loads(WEIGHTS_FILE.read_text())["weights"]


if __name__ == "__main__":
    # python -m tools.judge.terms <task.json with "source"> <output.json>
    task = json.loads(Path(sys.argv[1]).read_text())
    out = json.loads(Path(sys.argv[2]).read_text())
    tt = terms(measure(task["source"]), measure(out))
    print(json.dumps(tt, indent=1))
    print("hand-weighted penalty", round(penalty(tt, HAND_WEIGHTS), 3))
