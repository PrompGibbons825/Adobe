"""Image analysis: saliency (where the eye goes) and busyness (where text
would be hard to read). Classical methods only: no weights, runs offline.

- Spectral residual saliency (Hou & Zhang 2007): the log amplitude spectrum
  of natural images is smooth; whatever sticks out of that smooth curve is
  "unexpected" content, i.e. the subject.
- Busyness: local edge density. A smooth gradient scores ~0, a photo of
  foliage scores high. This is the "text never sits on a busy image" check.
- For transparent PNG cutouts, alpha IS the object mask.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ASSETS_DIR = Path(os.environ.get("CHAMELEON_ASSETS_DIR", "brand_kit/assets"))
_WORK = 128  # analysis resolution (long side)


def asset_file(asset_id: str | None) -> Path | None:
    if not asset_id:
        return None
    p = ASSETS_DIR / asset_id
    return p if p.is_file() else None


@lru_cache(maxsize=64)
def _load(asset_id: str):
    p = asset_file(asset_id)
    if p is None:
        return None
    try:
        im = Image.open(p).convert("RGBA")
    except Exception:
        return None
    return im


def image_size(asset_id: str) -> tuple[int, int] | None:
    im = _load(asset_id)
    return im.size if im else None


_CASCADES = None


def _faces(im: Image.Image) -> list:
    """Faces in a picture, as (x0, y0, x1, y1) fractions. OpenCV's bundled Haar
    cascades (frontal x2 + profile) on a 900px copy; a hit counts only if at
    least 45% of its box is skin-toned (YCrCb) and two eyes are found in its
    upper part, which drops skin, fabric and lights that merely look
    face-like. Returns [] if OpenCV is unavailable."""
    global _CASCADES
    try:
        import cv2
    except Exception:
        return []
    if _CASCADES is None:
        _CASCADES = [cv2.CascadeClassifier(cv2.data.haarcascades + n) for n in
                     ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml",
                      "haarcascade_profileface.xml", "haarcascade_eye.xml")]
    eye = _CASCADES[3]
    w, h = im.size
    if w * h < 120 * 120:
        return []
    sc = 900 / max(w, h)
    rgb = np.asarray(im.convert("RGB").resize((max(1, int(w * sc)), max(1, int(h * sc)))))
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    mn = max(30, int(min(gray.shape) * 0.05))
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin = (ycc[..., 1] >= 135) & (ycc[..., 1] <= 180) & (ycc[..., 2] >= 85) & (ycc[..., 2] <= 135)
    out = []
    for k, c in enumerate(_CASCADES[:3]):
        for (x, y, fw, fh) in c.detectMultiScale(gray, 1.08, 4 if k < 2 else 5, minSize=(mn, mn)):
            if skin[y:y + fh, x:x + fw].mean() < 0.45:
                continue
            # a face has two eyes in its upper part (drops skin, fabric, lights)
            roi = gray[y:y + int(fh * 0.6), x:x + fw]
            if len(eye.detectMultiScale(roi, 1.1, 3, minSize=(max(5, fw // 10), max(5, fw // 10)))) < 2:
                continue
            box = (x / rgb.shape[1], y / rgb.shape[0], (x + fw) / rgb.shape[1], (y + fh) / rgb.shape[0])
            if not any(max(0, min(box[2], o[2]) - max(box[0], o[0])) * max(0, min(box[3], o[3]) - max(box[1], o[1]))
                       > 0.3 * (box[2] - box[0]) * (box[3] - box[1]) for o in out):
                out.append(box)
    return out


@lru_cache(maxsize=64)
def analyze(asset_id: str) -> dict | None:
    """Returns dict with saliency map, busy map (both HxW float 0..1 at work
    res), alpha bbox (fractions), and salient-box (fractions)."""
    im = _load(asset_id)
    if im is None:
        return None
    w, h = im.size
    s = _WORK / max(w, h)
    sw, sh = max(8, round(w * s)), max(8, round(h * s))
    small = im.resize((sw, sh), Image.LANCZOS)
    arr = np.asarray(small).astype(np.float32) / 255.0
    alpha = arr[..., 3]
    gray = (arr[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)) * alpha

    # --- spectral residual ---
    f = np.fft.fft2(gray)
    log_amp = np.log(np.abs(f) + 1e-8)
    phase = np.angle(f)
    k = 3
    pad = np.pad(log_amp, k // 2, mode="edge")
    avg = np.zeros_like(log_amp)
    for dy in range(k):
        for dx in range(k):
            avg += pad[dy: dy + log_amp.shape[0], dx: dx + log_amp.shape[1]]
    avg /= k * k
    resid = log_amp - avg
    sal = np.abs(np.fft.ifft2(np.exp(resid + 1j * phase))) ** 2
    sal_img = Image.fromarray((sal / (sal.max() + 1e-8) * 255).astype(np.uint8))
    sal = np.asarray(sal_img.filter(ImageFilter.GaussianBlur(radius=max(2, sw // 24)))).astype(np.float32)
    sal = sal / (sal.max() + 1e-8)

    # --- busyness: gradient magnitude, locally averaged ---
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad_img = Image.fromarray(np.clip(grad * 4 * 255, 0, 255).astype(np.uint8))
    busy = np.asarray(grad_img.filter(ImageFilter.BoxBlur(radius=max(2, sw // 20)))).astype(np.float32) / 255.0

    # weight saliency by alpha (transparent pixels are never the subject)
    has_alpha = alpha.min() < 0.98
    if has_alpha:
        sal = 0.5 * sal * alpha + 0.5 * alpha
    sal = sal / (sal.max() + 1e-8)

    # alpha bbox
    ys, xs = np.where(alpha > 0.05)
    if len(xs):
        abox = (xs.min() / sw, ys.min() / sh, (xs.max() + 1) / sw, (ys.max() + 1) / sh)
    else:
        abox = (0, 0, 1, 1)

    # salient box: smallest box holding ~80% of saliency mass above threshold
    thr = np.quantile(sal, 0.85)
    mask = sal >= thr
    ys, xs = np.where(mask)
    if len(xs) and float(sal[mask].sum()) > 0:
        sbox = (np.quantile(xs, 0.05) / sw, np.quantile(ys, 0.05) / sh,
                (np.quantile(xs, 0.95) + 1) / sw, (np.quantile(ys, 0.95) + 1) / sh)
        wts = sal[mask]
        cx = float((xs * wts).sum() / wts.sum() + 0.5) / sw
        cy = float((ys * wts).sum() / wts.sum() + 0.5) / sh
    else:
        sbox, cx, cy = (0.25, 0.25, 0.75, 0.75), 0.5, 0.5

    # faces are the subject: the crop centres on them (head and shoulders)
    faces = _faces(im) if alpha.mean() > 0.9 else []
    if faces:
        x0 = min(f[0] for f in faces); x1 = max(f[2] for f in faces)
        y0 = min(f[1] for f in faces); y1 = max(f[3] for f in faces)
        fw, fh = x1 - x0, y1 - y0
        sbox = (max(0.0, x0 - 0.3 * fw), max(0.0, y0 - 0.3 * fh), min(1.0, x1 + 0.3 * fw), min(1.0, y1 + 1.2 * fh))
        wts = [(f[2] - f[0]) * (f[3] - f[1]) for f in faces]
        cx = sum(((f[0] + f[2]) / 2) * wt for f, wt in zip(faces, wts)) / sum(wts)
        cy = sum(((f[1] + f[3]) / 2) * wt for f, wt in zip(faces, wts)) / sum(wts)

    # relative luminance (WCAG), for text contrast; transparent pixels count as unknown (nan)
    lin = np.where(arr[..., :3] <= 0.04045, arr[..., :3] / 12.92, ((arr[..., :3] + 0.055) / 1.055) ** 2.4)
    lum = lin @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    lum = np.where(alpha > 0.5, lum, np.nan).astype(np.float32)
    mean_lum = float(np.nanmean(lum)) if np.isfinite(lum).any() else 1.0

    # plain: share of the visible pixels in the single most common colour
    # (a flat circle or card ~1.0, a picture well under 0.6)
    vis = arr[..., 3] > 0.4
    if vis.sum() >= 20:
        q = (arr[..., :3][vis] * 255 // 24).astype(np.int32)
        keys = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
        plain = float(np.bincount(keys - keys.min()).max() / len(keys))
    else:
        plain = 1.0

    return {
        "size": (w, h),
        "plain": plain,
        "faces": faces,
        "lum": lum,
        "mean_lum": mean_lum,
        "sal": sal,
        "busy": busy,
        "alpha_box": abox,
        "sal_box": sbox,
        "centroid": (cx, cy),
        "has_alpha": bool(has_alpha),
        "busyness": float(busy.mean()),
        "opaque_frac": float((alpha > 0.5).mean()),
        "edge_cover": _edge_cover(_alpha_maxpooled(im)),
        "corners": _corners(_alpha_maxpooled(im)),
    }


def _alpha_maxpooled(im: Image.Image, long_side: int = 256) -> np.ndarray:
    """Alpha downsampled with max-pooling so thin/dotted lines survive."""
    a = im.getchannel("A")
    k = max(1, int(max(a.size) / long_side))
    if k > 1:
        a = a.filter(ImageFilter.MaxFilter(2 * (k // 2) + 1))
        a = a.resize((max(8, a.width // k), max(8, a.height // k)), Image.NEAREST)
    return np.asarray(a).astype(np.float32) / 255.0


def _edge_cover(alpha: np.ndarray) -> float:
    """How much of each side of the box is traced by opaque pixels (min over
    the four sides). ~1 for a rectangular border, low for a hexagon/circle."""
    h, w = alpha.shape
    bx, by = max(1, int(round(w * 0.06))), max(1, int(round(h * 0.06)))
    m = alpha > 0.3
    sides = [m[:by, :].any(axis=0).mean(), m[-by:, :].any(axis=0).mean(),
             m[:, :bx].any(axis=1).mean(), m[:, -bx:].any(axis=1).mean()]
    return float(min(sides))


def _corners(alpha: np.ndarray) -> bool:
    """True when all four corners of the box carry ink (rectangular frames do;
    hexagons, circles and diamonds don't)."""
    h, w = alpha.shape
    bx, by = max(1, int(round(w * 0.06))), max(1, int(round(h * 0.06)))
    m = alpha > 0.3
    return bool(m[:by, :bx].any() and m[:by, -bx:].any() and m[-by:, :bx].any() and m[-by:, -bx:].any())


def region_mean(map_: np.ndarray, fx0: float, fy0: float, fx1: float, fy1: float) -> float:
    """Mean of map over a fractional region of the image."""
    h, w = map_.shape
    x0, x1 = int(max(0, min(w, fx0 * w))), int(max(0, min(w, np.ceil(fx1 * w))))
    y0, y1 = int(max(0, min(h, fy0 * h))), int(max(0, min(h, np.ceil(fy1 * h))))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(map_[y0:y1, x0:x1].mean())
