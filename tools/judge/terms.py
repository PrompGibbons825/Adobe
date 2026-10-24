"""Source-relative proxy judge (dev tool, not engine code).

`tools/proxy.py` scores an output on its own: every pixel overlap between two
elements without a shared `group_id` is a collision, so copy sitting on its own
panel (the designer's intent) counts against the layout. This module scores an
output *against its source*: only what the resize made worse is penalised.

    rep_src = measure(source_design)          # cacheable, JSON-serialisable
    rep_out = measure(output_design)
    t = terms(rep_src, rep_out)               # {term: value >= 0}, the 26 TERMS
    s = penalty(t, calibrated_weights())      # sum of weight * term (lower is better)

    .venv/bin/python -m tools.judge.terms input/<task>.json output/<task>_output.json

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

This module imports nothing from `engine/`: it only needs the renderer, so it
can score any engine root's output without import cycles. The term definitions
were ported from ws/judge-calibration (engine/smart/judge.py there, 10-06); the
engine re-rank flag that used them was not ported.
"""

from __future__ import annotations

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

TERMS = [
    "dropped_text", "dropped_art", "contrast_loss", "busy_gain", "abs_low_contrast", "abs_busy",
    "coll_text_art", "coll_text_text", "coll_art", "spill", "cut_text", "cut_art",
    "tiny512", "tiny1024", "text_shrink", "art_shrink", "hierarchy", "align_lost", "near_miss",
    "tight_margin", "imbalance", "underfill", "overfill", "aspect", "order_flip", "bare_canvas",
]

# Hand-set starting weights (research 04 §4.4, adapted to the source-relative terms).
HAND_WEIGHTS = {
    "dropped_text": 2.0, "dropped_art": 1.5, "contrast_loss": 2.5, "busy_gain": 2.0, "abs_low_contrast": 0.0,
    "abs_busy": 0.0, "coll_text_art": 3.0, "coll_text_text": 3.0, "coll_art": 1.0, "spill": 1.5,
    "cut_text": 2.0, "cut_art": 1.0, "tiny512": 1.0, "tiny1024": 2.0, "text_shrink": 1.0,
    "art_shrink": 0.5, "hierarchy": 1.0, "align_lost": 0.5, "near_miss": 0.5, "tight_margin": 0.5,
    "imbalance": 0.5, "underfill": 0.5, "overfill": 0.5, "aspect": 3.0, "order_flip": 0.5, "bare_canvas": 1.0,
}

WEIGHTS_FILE = Path(__file__).resolve().parent / "data" / "weights.json"


def calibrated_weights() -> Dict[str, float]:
    """Bradley-Terry weights fitted by tools/judge/fit.py (data/weights.json)."""
    return json.loads(WEIGHTS_FILE.read_text())["weights"]


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
    """The element's box (and its rotated bounding box) lies on the canvas."""
    x, y = float(e["x"]), float(e["y"])
    w, h = float(e["width"]), float(e["height"])
    x0, y0, x1, y1 = x, y, x + w, y + h
    rot = float(e.get("rotation", 0) or 0)
    if rot:
        t = math.radians(rot)
        c, s = abs(math.cos(t)), abs(math.sin(t))
        rw, rh = w * c + h * s, w * s + h * c
        cx, cy = x + w / 2.0, y + h / 2.0
        x0, y0 = min(x0, cx - rw / 2.0), min(y0, cy - rh / 2.0)
        x1, y1 = max(x1, cx + rw / 2.0), max(y1, cy + rh / 2.0)
    return x0 >= -1 and y0 >= -1 and x1 <= W + 1 and y1 <= H + 1


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
        if not glyph.any():
            continue
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


def _pair(rep: dict, a: str, b: str) -> Optional[list]:
    v = rep["pairs"].get(f"{a}|{b}")
    if v is not None:
        return v
    v = rep["pairs"].get(f"{b}|{a}")
    if v is not None:
        return [v[0], v[2], v[1]]
    return None


def _need(rec: dict) -> float:
    fs = rec.get("fs", 12)
    return 3.0 if fs >= 24 or (rec.get("bold") and fs >= 18.66) else 4.5


def _stacked(b1, b2, axis: int) -> bool:
    """Boxes separated along `axis` (1 = one above the other, 0 = side by side)."""
    lo, hi = (1, 3) if axis == 1 else (0, 2)
    return b1[hi] <= b2[lo] or b2[hi] <= b1[lo]


def _relations(rep: dict, ids: Iterable[str]) -> Dict[tuple, float]:
    """Alignment relations in normalised coordinates: (a, b, kind) -> |offset|."""
    out = {}
    ids = [i for i in ids if "bbox" in rep["els"][i]]
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            a, b = ids[x], ids[y]
            ba, bb = rep["els"][a]["bbox"], rep["els"][b]["bbox"]
            if _stacked(ba, bb, 1):  # vertical stack: left / centre / right lines
                for kind, f in (("l", lambda q: q[0]), ("r", lambda q: q[2]), ("cx", lambda q: (q[0] + q[2]) / 2)):
                    out[(a, b, kind)] = abs(f(ba) - f(bb))
            if _stacked(ba, bb, 0):  # side by side: top / middle / bottom lines
                for kind, f in (("t", lambda q: q[1]), ("b", lambda q: q[3]), ("cy", lambda q: (q[1] + q[3]) / 2)):
                    out[(a, b, kind)] = abs(f(ba) - f(bb))
    return out


def _near_miss_rate(rep: dict, ids: List[str]) -> float:
    """Share of stacked pairs whose best alignment is almost, but not quite, exact
    (O'Donovan: a near miss reads as a mistake, a clear offset as a choice)."""
    W, H = rep["W"], rep["H"]
    best: Dict[tuple, float] = {}
    for (a, b, kind), d in _relations(rep, ids).items():
        span = W if kind in ("l", "r", "cx") else H
        px = d * span
        key = (a, b, kind in ("l", "r", "cx"))
        best[key] = min(best.get(key, 1e9), px)
    if not best:
        return 0.0
    L = max(W, H)
    tol = max(2.0, 0.004 * L)
    near = sum(1 for px in best.values() if tol < px <= 0.03 * L)
    return near / len(best)


def terms(src: dict, out: dict) -> Dict[str, float]:
    """The 26 source-relative penalty terms (each >= 0; 0 means no worse than the source)."""
    t = {k: 0.0 for k in TERMS}
    S, O = src["els"], out["els"]
    ids = [i for i in S if i in O]

    # what the viewer loses
    texts_s = [i for i in ids if S[i]["type"] == "text" and S[i]["live"]]
    max_fs = max([S[i]["fs"] for i in texts_s] or [1.0])
    wsum = sum(S[i]["fs"] / max_fs for i in texts_s)
    if wsum:
        t["dropped_text"] = sum(S[i]["fs"] / max_fs for i in texts_s if not O[i]["live"]) / wsum
    art_s = [i for i in ids if S[i]["type"] != "text" and S[i]["live"] and not S[i]["bg"]]
    aw = {i: math.sqrt(S[i]["area"]) * max(0.1, S[i].get("solid", 1.0)) for i in art_s}  # faint decor counts little
    asum = sum(aw.values())
    if asum:
        t["dropped_art"] = sum(aw[i] for i in art_s if not O[i]["live"]) / asum

    # contrast and texture under the copy, relative to the source
    both = [i for i in texts_s if O[i]["live"] and i in src["texts"] and i in out["texts"]]
    live_o = [i for i in ids if O[i]["type"] == "text" and O[i]["live"] and i in out["texts"]]
    if texts_s:
        cl = bg = 0.0
        for i in both:
            need = _need(O[i])
            c0, c1 = src["texts"][i]["wcag"], out["texts"][i]["wcag"]
            cl += max(0.0, min(need, c0) - c1) / need
            b0, b1 = src["texts"][i]["busy"], out["texts"][i]["busy"]
            bg += min(1.0, max(0.0, b1 - max(b0, 0.03)) / 0.08)
        t["contrast_loss"] = cl / len(texts_s)
        t["busy_gain"] = bg / len(texts_s)
    if live_o:
        t["abs_low_contrast"] = sum(max(0.0, _need(O[i]) - out["texts"][i]["wcag"]) / _need(O[i]) for i in live_o) / len(live_o)
        t["abs_busy"] = sum(min(1.0, out["texts"][i]["busy"] / 0.08) for i in live_o) / len(live_o)

    # collisions the source did not have; overlaps it had must not grow or spill
    for key, (fmin, fa, fb) in out["pairs"].items():
        a, b = key.split("|")
        if a not in S or b not in S or S[a]["bg"] or S[b]["bg"]:
            continue  # anything over what was a full-canvas background in the source is intended
        sp = (_pair(src, a, b) if (S[a]["live"] and S[b]["live"]) else None) or [0.0, 0.0, 0.0]
        ta, tb = S[a]["type"] == "text", S[b]["type"] == "text"
        on_surface = False
        for txt, cin0, cin1 in ((a, sp[1], fa), (b, sp[2], fb)):
            if S[txt]["type"] == "text" and cin0 >= 0.6:  # copy on its panel / photo / button
                t["spill"] += max(0.0, cin0 - cin1)
                on_surface = True
        if on_surface:
            continue
        grown = fmin - sp[0]
        if grown <= 0.05:  # what the source already had (resolution noise included)
            continue
        if ta and tb:
            t["coll_text_text"] += grown
        elif ta or tb:
            t["coll_text_art"] += grown
        else:
            t["coll_art"] += grown
    for k in ("coll_text_art", "coll_text_text", "coll_art", "spill"):
        t[k] = min(3.0, t[k])

    # cut off at the canvas edge (decoration that bled in the source may keep bleeding)
    for i in ids:
        if not O[i]["live"]:
            continue
        d = max(0.0, O[i]["off"] - S[i]["off"])
        if d > 0.002:
            t["cut_text" if O[i]["type"] == "text" else "cut_art"] += d
    t["cut_text"] = min(2.0, t["cut_text"])
    t["cut_art"] = min(2.0, t["cut_art"])

    # size: legibility at the judge's view, and prominence relative to the canvas
    def cap_pen(fs, W, H, view):
        cap = 0.7 * fs * min(1.0, view / max(W, H))
        return min(1.0, max(0.0, (7.0 - cap) / 7.0))

    texts_o = [i for i in ids if O[i]["type"] == "text" and O[i]["live"] and S[i]["live"]]
    if texts_o:
        for view, key in ((512, "tiny512"), (1024, "tiny1024")):
            t[key] = sum(max(0.0, cap_pen(O[i]["fs"], out["W"], out["H"], view) - cap_pen(S[i]["fs"], src["W"], src["H"], view))
                         for i in texts_o) / len(texts_o)
        ns, no = math.sqrt(src["W"] * src["H"]), math.sqrt(out["W"] * out["H"])
        acc = wt = 0.0
        for i in texts_o:
            r = (O[i]["fs"] / no) / max(1e-9, S[i]["fs"] / ns)
            w = S[i]["fs"] / max_fs
            acc += w * min(1.0, max(0.0, -math.log(max(1e-6, r)) / math.log(2)))
            wt += w
        t["text_shrink"] = acc / max(1e-9, wt)
        inv = pairs = 0
        for x in range(len(texts_o)):
            for y in range(x + 1, len(texts_o)):
                a, b = texts_o[x], texts_o[y]
                d0 = S[a]["fs"] - S[b]["fs"]
                if abs(d0) < 1.0:
                    continue
                pairs += 1
                inv += d0 * (O[a]["fs"] - O[b]["fs"]) < 0
        t["hierarchy"] = inv / max(1, pairs)
    arts = [i for i in art_s if O[i]["live"]]
    if arts:
        acc = wt = 0.0
        for i in arts:
            r = math.sqrt(max(1e-9, O[i]["area"]) / max(1e-9, S[i]["area"]))
            w = aw[i]
            acc += w * min(1.0, max(0.0, -math.log(r) / math.log(2)))
            wt += w
        t["art_shrink"] = acc / max(1e-9, wt)

    # alignment: source relations that survive where the arrangement survived
    keep = [i for i in ids if S[i]["live"] and O[i]["live"] and not S[i]["bg"] and not O[i]["bg"]]
    rs = _relations(src, keep)
    ro = _relations(out, keep)
    Ls, Lo = max(src["W"], src["H"]), max(out["W"], out["H"])
    lost = n = 0.0
    for key, d0 in rs.items():
        span0 = src["W"] if key[2] in ("l", "r", "cx") else src["H"]
        if d0 * span0 > max(1.5, 0.004 * Ls) or key not in ro:
            continue
        span1 = out["W"] if key[2] in ("l", "r", "cx") else out["H"]
        px = ro[key] * span1
        tol = max(1.5, 0.002 * Lo)
        n += 1
        lost += min(1.0, max(0.0, px - tol) / (0.02 * Lo))
    t["align_lost"] = lost / n if n else 0.0
    t["near_miss"] = max(0.0, _near_miss_rate(out, keep) - _near_miss_rate(src, keep))

    # the source's reading order: what sat left of / above what still does (where the
    # arrangement survived); a mirrored layout reads as a different design
    flips = n = 0
    for x in range(len(keep)):
        for y in range(x + 1, len(keep)):
            a, b = keep[x], keep[y]
            if "bbox" not in S[a] or "bbox" not in S[b] or "bbox" not in O[a] or "bbox" not in O[b]:
                continue
            sa, sb, oa, ob = S[a]["bbox"], S[b]["bbox"], O[a]["bbox"], O[b]["bbox"]
            for axis, lo, hi in ((0, 0, 2), (1, 1, 3)):
                if _stacked(sa, sb, axis) and _stacked(oa, ob, axis):
                    n += 1
                    flips += (sa[hi] <= sb[lo]) != (oa[hi] <= ob[lo])
    t["order_flip"] = flips / n if n else 0.0
    t["bare_canvas"] = max(0.0, out.get("bare", 0.0) - src.get("bare", 0.0))

    # whitespace and balance, relative to the source
    t["tight_margin"] = sum(1 for m0, m1 in zip(src["margins"], out["margins"]) if m1 < 0.02 and m0 >= 0.02) / 4.0
    imb = lambda r: 2 * math.hypot(r["centroid"][0] - 0.5, r["centroid"][1] - 0.5)  # noqa: E731
    t["imbalance"] = max(0.0, imb(out) - imb(src))
    c0, c1 = src["coverage"] + 0.02, out["coverage"] + 0.02
    t["underfill"] = min(2.0, max(0.0, math.log(c0 / c1)))
    t["overfill"] = min(2.0, max(0.0, math.log(c1 / c0)))

    # pictures keep their proportions (a full-bleed cover background may crop)
    imgs = [i for i in ids if O[i]["type"] == "image" and O[i].get("fit") != "cover" and O[i]["live"]]
    if imgs:
        t["aspect"] = sum(abs(O[i]["box_aspect"] / max(1e-6, S[i]["box_aspect"]) - 1) > 0.05 for i in imgs) / len(imgs)
    return {k: round(v, 4) for k, v in t.items()}


def penalty(t: Dict[str, float], weights: Dict[str, float]) -> float:
    return float(sum(weights.get(k, 0.0) * v for k, v in t.items()))


def score(source: dict, output: dict, weights: Optional[Dict[str, float]] = None) -> dict:
    """Measure both designs and return {"terms", "penalty"} (calibrated weights by default)."""
    t = terms(measure(source), measure(output))
    return {"terms": t, "penalty": round(penalty(t, calibrated_weights() if weights is None else weights), 4)}


if __name__ == "__main__":
    # python -m tools.judge.terms <task.json with "source"> <output.json>
    if len(sys.argv) != 3:
        sys.exit("usage: python -m tools.judge.terms <task.json> <output.json>")
    task = json.loads(Path(sys.argv[1]).read_text())
    out = json.loads(Path(sys.argv[2]).read_text())
    res = score(task.get("source", task), out)
    print(json.dumps(res["terms"], indent=1))
    print("calibrated penalty", res["penalty"])
    print("hand-weighted penalty", round(penalty(res["terms"], HAND_WEIGHTS), 3))
