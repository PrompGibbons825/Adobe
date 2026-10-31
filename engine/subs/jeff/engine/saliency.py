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

ASSETS_DIR = Path(os.environ.get("CHAMELEON_ASSETS_DIR")
                  or (Path(os.environ["CHAMELEON_BRAND_KIT"]) / "assets" if os.environ.get("CHAMELEON_BRAND_KIT")
                      else "brand_kit/assets"))
_WORK = 128  # analysis resolution (long side)
# the learned subject mask (engine/vision.py): off until the critic can use it
# without flipping unrelated layout choices (see review/h2h_sprint7pm)
SUBJECT_MASK = os.environ.get("CHAMELEON_SUBJECT_MASK") == "1"


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
_YUNET = None
YUNET_MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"


def _faces(im: Image.Image) -> list:
    """Faces in a picture: [(box, eye)] with box = (x0, y0, x1, y1) and eye =
    (x, y), the midpoint of the eyes, as fractions. YuNet (a 230 KB CNN
    shipped with the engine, run by OpenCV) finds faces turned, tilted, half
    lit or behind hair that the Haar cascades miss; the cascades are the
    fallback. Returns [] if OpenCV is unavailable."""
    try:
        import cv2
    except Exception:
        return []
    w, h = im.size
    if w * h < 96 * 96:
        return []
    found = _yunet(cv2, im)
    if found is None:
        # no landmarks: the eyes sit about 40% down a face box
        return [(f, ((f[0] + f[2]) / 2, f[1] + 0.4 * (f[3] - f[1]))) for f in _haar(cv2, im)]
    # tiny faces (a crowd far away) are not the subject
    found = [(f, e) for f, _, e in sorted(found, key=lambda t: -t[1]) if (f[3] - f[1]) * h >= 0.035 * min(w, h)]
    out = []
    for f, e in found:
        if not any(_iou(f, o[0]) > 0.3 for o in out):
            out.append((f, e))
    return out[:12]


def _iou(a, b):
    i = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _yunet(cv2, im):
    """[(box, score, eye)] from YuNet at two scales, or None if it cannot run."""
    global _YUNET
    if _YUNET is False or not YUNET_MODEL.is_file():
        return None
    w, h = im.size
    out = []
    for side in (640, 320):
        sc = min(1.0, side / max(w, h))
        sz = (max(1, int(w * sc)), max(1, int(h * sc)))
        bgr = cv2.cvtColor(np.asarray(im.convert("RGB").resize(sz, Image.BILINEAR)), cv2.COLOR_RGB2BGR)
        try:
            if _YUNET is None:
                _YUNET = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", sz, 0.8, 0.3, 50)
            _YUNET.setInputSize(sz)
            _, res = _YUNET.detect(bgr)
        except Exception:
            _YUNET = False
            return None
        for r in ([] if res is None else res):
            x, y, fw, fh, s = (float(v) for v in (r[0], r[1], r[2], r[3], r[14]))
            if fw > 0 and fh > 0 and np.isfinite([x, y, fw, fh, s]).all():
                eye = ((float(r[4]) + float(r[6])) / 2 / sz[0], (float(r[5]) + float(r[7])) / 2 / sz[1])
                out.append(((max(0.0, x / sz[0]), max(0.0, y / sz[1]),
                             min(1.0, (x + fw) / sz[0]), min(1.0, (y + fh) / sz[1])), s, eye))
    return out


def _haar(cv2, im: Image.Image) -> list:
    """OpenCV's bundled Haar cascades (frontal x2 + profile) on a 900px copy;
    a hit counts only if at least 45% of its box is skin-toned (YCrCb) and two
    eyes are found in its upper part, which drops skin, fabric and lights that
    merely look face-like."""
    global _CASCADES
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
    elif SUBJECT_MASK and w * h >= 160 * 160 and float(busy.mean()) > 0.02:
        # a photo: the foreground a person would cut out of it (a small
        # salient-object model) leads; the spectral residual fills in. A flat
        # illustration has no foreground, and keeps the spectral map alone.
        from . import vision
        mk = vision.mask_from_image(im)
        if mk is not None:
            mk = np.asarray(Image.fromarray((mk * 255).astype(np.uint8)).resize((sw, sh), Image.BILINEAR),
                            dtype=np.float32) / 255.0
            if 0.02 <= float((mk > 0.5).mean()) <= 0.85:
                sal = 0.6 * mk + 0.4 * sal / (sal.max() + 1e-8)
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

    # a diffuse map (a textured surface: marble, wood, paper) finds no subject;
    # then what stands out from the surface's own tone (dark or coloured
    # objects on pale stone) is the subject, if it is a clear part of the
    # picture. Only the box and centroid change; the map stays as it was.
    if (sbox[2] - sbox[0]) * (sbox[3] - sbox[1]) > 0.75 and not has_alpha:
        rgb = arr[..., :3]
        g = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        sat = rgb.max(-1) - rgb.min(-1)
        dev = np.abs(g - np.median(g)) + np.maximum(0.0, sat - np.median(sat))
        fg = np.asarray(Image.fromarray((dev > 0.22).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 0
        if 0.05 <= float(fg.mean()) <= 0.55:
            def band(share):
                """The longest run of rows/columns that are mostly object."""
                best, cur = (0, 0), None
                for i, v in enumerate(list(share) + [0.0]):
                    if v > 0.3:
                        cur = (cur[0], i + 1) if cur else (i, i + 1)
                    elif cur:
                        best = max(best, cur, key=lambda r: r[1] - r[0])
                        cur = None
                return best
            x0, x1 = band(fg.mean(0))
            y0, y1 = band(fg[:, x0:x1].mean(1)) if x1 > x0 else (0, 0)
            box = (x0 / sw, y0 / sh, x1 / sw, y1 / sh)
            area = (box[2] - box[0]) * (box[3] - box[1])
            if 0.04 <= area < 0.6:
                sbox = box
                sub = fg[y0:y1, x0:x1]
                ys, xs = np.where(sub)
                cx, cy = float(xs.mean() + x0 + 0.5) / sw, float(ys.mean() + y0 + 0.5) / sh

    # faces are the subject and the eyes are where the viewer looks first:
    # the crop keeps head and shoulders and aims at the eye line
    found = _faces(im) if alpha.mean() > 0.9 else []
    faces = [f for f, _ in found]
    eyes = [e for _, e in found]
    if faces:
        x0 = min(f[0] for f in faces); x1 = max(f[2] for f in faces)
        y0 = min(f[1] for f in faces); y1 = max(f[3] for f in faces)
        fw, fh = x1 - x0, y1 - y0
        sbox = (max(0.0, x0 - 0.3 * fw), max(0.0, y0 - 0.3 * fh), min(1.0, x1 + 0.3 * fw), min(1.0, y1 + 1.2 * fh))
        wts = [(f[2] - f[0]) * (f[3] - f[1]) for f in faces]
        cx = sum(e[0] * wt for e, wt in zip(eyes, wts)) / sum(wts)
        cy = sum(e[1] * wt for e, wt in zip(eyes, wts)) / sum(wts)

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

    # interior structure: rows/columns of ink crossing the middle of an
    # otherwise transparent drawing (dividers inside an outline frame)
    a_in = alpha[int(0.12 * sh):int(0.88 * sh), int(0.12 * sw):int(0.88 * sw)] > 0.5
    rules = float(max((a_in.mean(1) > 0.5).mean(), (a_in.mean(0) > 0.5).mean())) if a_in.size else 0.0

    return {
        "size": (w, h),
        "rules": rules,
        "plain": plain,
        "faces": faces,
        "eyes": eyes,
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
        "alpha_mean": float(alpha.mean()),
        "alpha_std": float(alpha.std()),       # ~0 for a full-bleed tint, high for a cut-out shape
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


@lru_cache(maxsize=256)
def inner_rect(asset_id: str, quarter_turns: int = 0):
    """The largest rectangle inside a shape's visible pixels (fractions of its
    box), for setting copy inside an arch, a circle or a badge; turned by the
    given number of quarter turns (numpy's rot90 sense). None if unknown."""
    im = _load(asset_id)
    if im is None:
        return None
    n = 32
    a = np.asarray(im.split()[-1].resize((n, n), Image.BILINEAR)) > 128
    a = np.rot90(a, quarter_turns)
    if a.mean() > 0.97:
        return (0.0, 0.0, 1.0, 1.0)
    # largest all-opaque rectangle (row histogram + stack), on the coarse grid
    best, h = (0, None), np.zeros(n, int)
    for y in range(n):
        h = np.where(a[y], h + 1, 0)
        st = []
        for x in range(n + 1):
            cur = h[x] if x < n else 0
            start = x
            while st and st[-1][1] >= cur:
                sx, sh = st.pop()
                area = sh * (x - sx)
                if area > best[0]:
                    best = (area, (sx, y - sh + 1, x, y + 1))
                start = sx
            st.append((start, cur))
    if best[1] is None:
        return None
    x0, y0, x1, y1 = best[1]
    return (x0 / n, y0 / n, x1 / n, y1 / n)
