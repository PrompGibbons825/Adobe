"""Proxy evaluator for Chameleon outputs (dev tool, not engine code).

Vendored from context/research/scripts/perception/proxy.py (2026-10-05 research
session) with three changes: asset/font folders are configurable
(CHAMELEON_PROXY_ASSETS), designs larger than 1600 px are scaled down before
scoring (speed), and the module no longer hardcodes the repo path.

Original description:

Renders each element in isolation with the official converter + cr-renderer to get
exact per-element alpha masks, then measures what the VLM judge is likely to punish:
text contrast against the real pixels under the glyphs, text over busy pixels,
glyph collisions, cut-off / out-of-canvas ink, tiny text, logo scale, etc.

Usage: python proxy.py <output.json> [<output.json> ...]
Run with the chameleon repo venv, cwd = chameleon/.
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import os

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from renderer.converter.convert import _get_renderer, convert_chameleon_to_crello  # noqa: E402
from renderer.converter.colorutil import parse_rgba  # noqa: E402

ASSETS = os.environ.get("CHAMELEON_PROXY_ASSETS", str(REPO / "brand_kit/assets"))
FONTS = str(REPO / "brand_kit/fonts")
R = _get_renderer(FONTS)


def render(design: dict, transparent: bool = False, pad: int = 0) -> np.ndarray:
    """RGBA float array in [0,1]. `pad` enlarges the canvas on every side so ink
    that spills off-canvas can be measured."""
    d = design
    if pad:
        d = {
            "canvas": {"width": design["canvas"]["width"] + 2 * pad, "height": design["canvas"]["height"] + 2 * pad},
            "elements": [dict(e, x=e["x"] + pad, y=e["y"] + pad) for e in design["elements"]],
        }
    ex, _ = convert_chameleon_to_crello(d, R.font_manager, ASSETS)
    import skia

    png = R.render(ex, max_size=4096, format="png", canvas_color=None if transparent else skia.ColorWHITE)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"), dtype=np.float32) / 255.0


def srgb_lum(rgb: np.ndarray) -> np.ndarray:
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def wcag(l1, l2):
    hi, lo = np.maximum(l1, l2), np.minimum(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def apca_lc(text_rgb, bg_rgb):
    """APCA 0.0.98G-4g (W3 version) lightness contrast Lc, signed. Inputs 0..1 sRGB."""
    def y(rgb):
        r, g, b = (np.asarray(rgb)[..., i] ** 2.4 for i in range(3))
        yy = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        return np.where(yy > 0.022, yy, yy + np.clip(0.022 - yy, 0, None) ** 1.414)
    yt, yb = y(text_rgb), y(bg_rgb)
    normal = yb > yt
    s_norm = (yb ** 0.56 - yt ** 0.57) * 1.14
    s_rev = (yb ** 0.65 - yt ** 0.62) * 1.14
    s = np.where(normal, s_norm, s_rev)
    lc = np.where(np.abs(s) < 0.1, 0.0, np.where(normal, s - 0.027, s + 0.027))
    return lc * 100.0


def grad_mag(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.hypot(gx, gy) * 0.5


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    m = mask.copy()
    for _ in range(r):
        n = m.copy()
        n[1:, :] |= m[:-1, :]
        n[:-1, :] |= m[1:, :]
        n[:, 1:] |= m[:, :-1]
        n[:, :-1] |= m[:, 1:]
        m = n
    return m


def downscale(design: dict, limit: int = 1600) -> dict:
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    f = min(1.0, limit / max(W, H))
    if f >= 1.0:
        return design
    els = []
    for e in design["elements"]:
        e = dict(e)
        for k in ("x", "y", "width", "height"):
            e[k] = e[k] * f
        if "font_size" in e:
            e["font_size"] = max(1, int(round(e["font_size"] * f)))
        if e.get("corner_radius"):
            e["corner_radius"] = e["corner_radius"] * f
        els.append(e)
    return {"canvas": {"width": max(1, round(W * f)), "height": max(1, round(H * f))}, "elements": els}


def score(path: str) -> dict:
    out = json.loads(Path(path).read_text())
    design = out.get("source", out) if "elements" not in out else out
    design = downscale(design)
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    els = sorted(design["elements"], key=lambda e: e.get("z_index", 0))
    PAD = max(32, int(0.25 * max(W, H)))

    full = render(design)
    masks, inks = {}, {}
    for e in els:
        a = render({"canvas": design["canvas"], "elements": [e]}, transparent=True, pad=PAD)[..., 3]
        inks[e["id"]] = a
        masks[e["id"]] = a[PAD:PAD + H, PAD:PAD + W] > 0.25
    rep = {"file": Path(path).name, "canvas": [W, H], "texts": [], "collisions": [], "offcanvas": []}

    # --- off-canvas ink (cut off) -------------------------------------------------
    for e in els:
        a = inks[e["id"]] > 0.25
        tot = a.sum()
        if tot == 0:
            continue
        inside = a[PAD:PAD + H, PAD:PAD + W].sum()
        frac = 1 - inside / tot
        if frac > 0.002:
            rep["offcanvas"].append((e["id"], round(float(frac), 3)))

    # --- per text: contrast & busyness against what is underneath -------------------
    by_id = {e["id"]: e for e in els}
    for t in [e for e in els if e["type"] == "text"]:
        below = els[:els.index(t)]
        under = render({"canvas": design["canvas"], "elements": below})[..., :3]
        glyph = masks[t["id"]]
        if glyph.sum() == 0:
            rep["texts"].append({"id": t["id"], "visible": False})
            continue
        halo = dilate(glyph, max(2, int(0.15 * t["font_size"]))) & ~glyph
        r, g, b, a = parse_rgba(t.get("color"))
        trgb = np.array([r, g, b]) / 255.0
        # the text's own alpha blends it with the background
        alpha = a  # element opacity is ignored for text by cr-renderer (see research 06, A3)
        bgpix = under[halo | glyph]
        eff = alpha * trgb + (1 - alpha) * bgpix
        lt, lb = srgb_lum(eff), srgb_lum(bgpix)
        c = wcag(lt, lb)
        lc = np.abs(apca_lc(eff, bgpix))
        ys, xs = np.where(glyph)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        gray = srgb_lum(under[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4])
        busy = float(np.percentile(grad_mag(gray), 90))
        std = float(gray.std())
        # cap height in px ~ glyph mask height of a single line is noisy; use font metrics proxy
        cap_px = 0.70 * float(t["font_size"])
        rep["texts"].append({
            "id": t["id"], "fs": t["font_size"], "cap_px": round(cap_px, 1),
            "wcag_p10": round(float(np.percentile(c, 10)), 2), "wcag_med": round(float(np.median(c)), 2),
            "apca_p10": round(float(np.percentile(lc, 10)), 1),
            "busy_p90grad": round(busy, 4), "bg_lum_std": round(std, 4),
            "ink_box": [int(x0), int(y0), int(x1), int(y1)],
            "box_overflow_px": [
                int(max(0, t["x"] - x0)), int(max(0, t["y"] - y0)),
                int(max(0, x1 - (t["x"] + t["width"]))), int(max(0, y1 - (t["y"] + t["height"])))],
        })

    # --- pixel collisions between elements (ignoring same group and full-canvas bgs) -------
    canvas_area = W * H
    bgs = {e["id"] for e in els if masks[e["id"]].sum() > 0.85 * canvas_area}
    ids = [e["id"] for e in els if e["id"] not in bgs]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ea, eb = by_id[a], by_id[b]
            if ea.get("group_id") and ea.get("group_id") == eb.get("group_id"):
                continue
            inter = (masks[a] & masks[b]).sum()
            if inter == 0:
                continue
            small = min(masks[a].sum(), masks[b].sum())
            rep["collisions"].append((a, b, int(inter), round(float(inter / max(1, small)), 3)))

    # --- whitespace / balance --------------------------------------------------------
    occ = np.zeros((H, W), bool)
    for i in ids:
        occ |= masks[i]
    ys, xs = np.where(occ)
    if len(xs):
        cx, cy = xs.mean() / W, ys.mean() / H
        rep["ink_centroid"] = [round(float(cx), 3), round(float(cy), 3)]
        rep["ink_coverage"] = round(float(occ.mean()), 3)
        rep["margins_px"] = [int(xs.min()), int(ys.min()), int(W - 1 - xs.max()), int(H - 1 - ys.max())]
    return rep


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(score(p), indent=1, default=str))


# ---------------------------------------------------------------------------------------
# Aggregate: deterministic proxy of the rubric (weights are a starting point to calibrate)
# ---------------------------------------------------------------------------------------
def _ink_area(rep_masks, i):
    return int(rep_masks[i].sum())


def aggregate(out_path: str, src_task_path: str) -> dict:
    task = json.loads(Path(src_task_path).read_text())
    src = task["source"]
    rep = score(out_path)
    out = downscale(json.loads(Path(out_path).read_text()))
    W, H = out["canvas"]["width"], out["canvas"]["height"]
    S = min(W, H)
    pen = {}

    # 1. readability: cap height in px (renderer truncates fs to int)
    caps = {t["id"]: t.get("cap_px", 0) for t in rep["texts"]}
    # floor in *judge* pixels: assume the VLM sees the PNG downscaled to ~1024 px on the long edge
    judge_scale = min(1.0, 1024.0 / max(W, H))
    floor = 7.0 / judge_scale if judge_scale < 1 else 7.0  # cap height >= ~7 judge-px (~10 px Poppins)
    dropped = [i for i, c in caps.items() if c < 3.0]
    small = [i for i, c in caps.items() if 3.0 <= c < floor]
    pen["tiny_text"] = len(small) / max(1, len(caps))

    # 2. contrast against real pixels (WCAG on 10th percentile, large text gets 3:1)
    bad = 0.0
    for t in rep["texts"]:
        if not t.get("visible", True) or t.get("cap_px", 0) < 3.0:
            continue
        need = 3.0 if t["fs"] >= 24 or (t["fs"] >= 18.66) else 4.5
        bad += max(0.0, (need - t["wcag_p10"]) / need)
    pen["low_contrast"] = bad / max(1, len(rep["texts"]))

    # 3. text over busy pixels (gradient p90 under the text block; 0.02 ~ visible texture)
    live = [t for t in rep["texts"] if t.get("cap_px", 0) >= 3.0]
    pen["text_on_busy"] = sum(min(1.0, t.get("busy_p90grad", 0) / 0.08) for t in live) / max(1, len(live))
    # aspect: non-cover images must keep the source box aspect within 5% (structural check)
    sb = {e["id"]: e for e in src["elements"]}
    viol = 0
    imgs = [e for e in out["elements"] if e["type"] == "image" and e.get("fit", "contain") != "cover"]
    for e in imgs:
        s0 = sb[e["id"]]
        a0 = s0["width"] / max(1e-6, s0["height"]); a1 = e["width"] / max(1e-6, e["height"])
        viol += abs(a1 / a0 - 1) > 0.05
    pen["aspect_violation"] = viol / max(1, len(imgs))

    # 4. collisions (pixel-level, excluding same-group pairs and full-canvas backgrounds)
    pen["collisions"] = min(1.0, sum(c[3] for c in rep["collisions"]))

    # 5. cut off / off-canvas ink
    pen["offcanvas"] = min(1.0, sum(f for _, f in rep["offcanvas"]))

    # 6. faithfulness: element retention + font-size order preserved
    src_fs = {e["id"]: e["font_size"] for e in src["elements"] if e["type"] == "text"}
    out_fs = {e["id"]: e["font_size"] for e in out["elements"] if e["type"] == "text"}
    ids = sorted(src_fs)
    inv = 0
    pairs = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if abs(src_fs[a] - src_fs[b]) < 1e-6:
                continue
            pairs += 1
            if (src_fs[a] - src_fs[b]) * (out_fs[a] - out_fs[b]) < 0:
                inv += 1
    pen["hierarchy_inversions"] = inv / max(1, pairs)
    pen["dropped_text"] = len(dropped) / max(1, len(caps))  # hidden (fs~1) = element lost to the viewer

    # 7. margins & balance
    m = rep.get("margins_px", [0, 0, 0, 0])
    pen["tight_margins"] = sum(1 for v in m if v < 0.02 * S) / 4
    cx, cy = rep.get("ink_centroid", [0.5, 0.5])
    pen["imbalance"] = min(1.0, 2 * math.hypot(cx - 0.5, cy - 0.5))

    w = {"tiny_text": 2.0, "low_contrast": 2.0, "text_on_busy": 1.5, "collisions": 2.0, "offcanvas": 2.0,
         "hierarchy_inversions": 1.0, "dropped_text": 1.0, "aspect_violation": 3.0, "tight_margins": 0.5, "imbalance": 0.5}
    total = 10.0 - sum(w[k] * v for k, v in pen.items())
    return {"file": rep["file"], "score": round(total, 2), **{k: round(v, 3) for k, v in pen.items()}}


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--agg":
    pass
