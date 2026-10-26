"""Where the subject of a photo is, so a resize keeps it and keeps copy off it.

Read-only: nothing here changes a pixel. A photo is only *looked at*, small:

  * an importance map at 128 px on the long side, in the manner of
    smartcrop.js (MIT): detail (edges), skin tone and saturation, plus
    spectral-residual saliency (Hou & Zhang 2007: whatever sticks out of the
    smooth log-amplitude spectrum is the unexpected part, the subject);
  * faces, from YuNet (OpenCV zoo, MIT, models/face_detection_yunet_2023mar.onnx),
    or OpenCV's bundled Haar cascade with a skin and two-eye check when YuNet
    cannot load. Each face is padded to the head (never cut) and to head and
    shoulders (copy keeps off it);
  * U2-Netp saliency (Apache-2.0, models/u2netp.onnx) merged into the importance
    map for cropping and copy penalties;
  * PP-OCRv4 text detection only (Apache-2.0, models/ch_PP-OCRv4_det_infer.onnx):
    baked-in text on a photo is treated like a face (never cropped through or
    covered by live copy).

A background photo that covers the canvas is then cropped by *bleed*
(``cover_box``): drawn at its own aspect, scaled to cover the canvas, and slid
so no head is cut, as much as possible of what the source showed stays, and the
subject sits on the third line opposite the copy (centred when the copy is).
The box is larger than the canvas at a negative offset, so the renderer's
in-box crop does nothing and the canvas edge does the cropping. A flat or
diffuse picture (a gradient, a texture) keeps the plain centre crop.

``penalty`` charges a plan for copy, logo or button set over a face or on the
busy part of the photo, so every candidate is ranked with it.

Everything degrades: without numpy, OpenCV or the asset, nothing here has an
opinion, and the engine behaves as it did before.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import paths

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

MAP = 128  # importance map, long side
FACE_SIDE = 640  # face detection, long side
MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
FLAT = 0.45  # a subject box (85th-percentile importance) wider than this share of the image is diffuse
STEPS = 48  # crop positions tried along the free axis
# The focal crop draws the photo past the canvas edges (allowed bleed for a canvas-covering photo,
# STEERING S4/S5). CHAMELEON_NO_BLEED=1 keeps every background at the renderer's centre crop
# (the copy still keeps off faces).
BLEED = os.environ.get("CHAMELEON_NO_BLEED", "") in ("", "0")

Frac = Tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of the image


@dataclass(frozen=True, eq=False)  # hashed by identity (cached per asset)
class Vision:
    size: Tuple[int, int]  # native pixels
    imp: object  # importance map (h, w) float32 in 0..1
    integral: object  # summed-area table of imp, (h + 1, w + 1)
    faces: Tuple[Frac, ...]  # detected faces
    heads: Tuple[Frac, ...]  # faces padded to the whole head: never cut
    people: Tuple[Frac, ...]  # head and shoulders: copy keeps off
    baked: Tuple[Frac, ...]  # text baked into the raster: never crop through or cover
    focal: Tuple[float, float]  # the subject's centre
    subject: bool  # a concentrated subject (or a face): worth cropping for
    extent: Frac = (0.0, 0.0, 1.0, 1.0)  # where the subject is (the heads, or the weightiest part)

    def mass(self, f: Frac) -> float:
        """Mean importance over a fractional region."""
        h, w = self.imp.shape
        x0, x1 = int(max(0, min(w, math.floor(f[0] * w)))), int(max(0, min(w, math.ceil(f[2] * w))))
        y0, y1 = int(max(0, min(h, math.floor(f[1] * h)))), int(max(0, min(h, math.ceil(f[3] * h))))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        S = self.integral
        tot = S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]
        return float(tot) / ((x1 - x0) * (y1 - y0))


# ----------------------------------------------------------------------------
# analysis (cached per asset file)


def analyze(asset_id: Optional[str]) -> Optional[Vision]:
    if np is None or not asset_id:
        return None
    p = paths.find_asset(asset_id)
    if p is None:
        return None
    try:
        st = os.stat(p)
    except OSError:
        return None
    return _analyze(str(p), st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=64)
def _analyze(path: str, mtime: int, size: int) -> Optional[Vision]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.load()
            rgba = im.convert("RGBA")
    except Exception:
        return None
    try:
        return _vision(rgba, path)
    except Exception:  # never let a picture break a resize
        return None


def _vision(rgba, path: str = "") -> Optional[Vision]:
    from PIL import Image

    iw, ih = rgba.size
    if iw < 8 or ih < 8:
        return None
    s = MAP / max(iw, ih)
    sw, sh = max(8, round(iw * s)), max(8, round(ih * s))
    arr = np.asarray(rgba.resize((sw, sh), Image.BILINEAR), dtype=np.float32) / 255.0
    alpha = arr[..., 3]
    rgb = arr[..., :3]
    imp = _importance(rgb, alpha)
    from . import vision_models as _vm

    u2 = _vm.u2_saliency(rgb, alpha, sh, sw)
    if u2 is not None:
        imp = np.maximum(imp, 0.68 * u2)
    faces = _faces(rgba) if alpha.mean() > 0.9 else []
    heads = tuple(_pad(f, 0.25, 0.45, 0.25, 0.25) for f in faces)
    people = tuple(_pad(f, 0.5, 0.45, 0.5, 0.8) for f in faces)
    baked: Tuple[Frac, ...] = ()
    if path:
        try:
            st = os.stat(path)
            raw_baked = _vm.baked_text(path, st.st_mtime_ns, st.st_size)
            baked = tuple(f for f in raw_baked if (f[2] - f[0]) * (f[3] - f[1]) >= 0.003
                          and not any(_iou(f, h) > 0.25 for h in heads))
        except OSError:
            baked = ()
    for f in baked:
        x0, y0, x1, y1 = (int(round(v)) for v in (f[0] * sw, f[1] * sh, f[2] * sw, f[3] * sh))
        imp[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = np.maximum(
            imp[max(0, y0):max(0, y1), max(0, x0):max(0, x1)], 0.92)
    for f in heads:  # the faces are what matters most in the picture
        x0, y0, x1, y1 = (int(round(v)) for v in (f[0] * sw, f[1] * sh, f[2] * sw, f[3] * sh))
        imp[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = np.maximum(
            imp[max(0, y0):max(0, y1), max(0, x0):max(0, x1)], 0.85)
    integral = np.zeros((sh + 1, sw + 1), dtype=np.float64)
    integral[1:, 1:] = imp.astype(np.float64).cumsum(0).cumsum(1)

    extent = (0.0, 0.0, 1.0, 1.0)
    if faces:
        wts = [(f[2] - f[0]) * (f[3] - f[1]) for f in faces]
        fx = sum((f[0] + f[2]) / 2 * w for f, w in zip(faces, wts)) / sum(wts)
        fy = sum((f[1] + f[3]) / 2 * w for f, w in zip(faces, wts)) / sum(wts)
        subject = True
        extent = (min(h[0] for h in heads), min(h[1] for h in heads), max(h[2] for h in heads),
                  max(h[3] for h in heads))
    else:
        thr = float(np.quantile(imp, 0.85))
        ys, xs = np.nonzero(imp >= thr)
        if len(xs) < 4 or float(imp.max()) < 0.15:
            fx, fy, subject = 0.5, 0.5, False
        else:
            extent = (float(np.quantile(xs, 0.05)) / sw, float(np.quantile(ys, 0.05)) / sh,
                      float(np.quantile(xs, 0.95) + 1) / sw, float(np.quantile(ys, 0.95) + 1) / sh)
            bw = (np.quantile(xs, 0.95) - np.quantile(xs, 0.05) + 1) / sw
            bh = (np.quantile(ys, 0.95) - np.quantile(ys, 0.05) + 1) / sh
            wt = imp[ys, xs]
            fx = float((xs * wt).sum() / wt.sum() + 0.5) / sw
            fy = float((ys * wt).sum() / wt.sum() + 0.5) / sh
            # concentrated, and standing out from the rest of the picture
            subject = bw * bh < FLAT and float(imp[ys, xs].mean()) > 2.0 * float(imp.mean()) + 0.05
    return Vision(size=(iw, ih), imp=imp, integral=integral, faces=tuple(faces), heads=heads, people=people,
                  baked=baked, focal=(fx, fy), subject=bool(subject), extent=extent)


def _pad(f: Frac, l: float, t: float, r: float, b: float) -> Frac:
    w, h = f[2] - f[0], f[3] - f[1]
    return (max(0.0, f[0] - l * w), max(0.0, f[1] - t * h), min(1.0, f[2] + r * w), min(1.0, f[3] + b * h))


def _blur(a, k: int):
    """Box blur (separable, edge-padded), k odd."""
    if k <= 1:
        return a
    p = k // 2
    for ax in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[ax] = (p + 1, p)
        c = np.pad(a, pad, mode="edge").cumsum(axis=ax)
        a = (np.take(c, range(k, c.shape[ax]), axis=ax) - np.take(c, range(0, c.shape[ax] - k), axis=ax)) / k
    return a


def _importance(rgb, alpha):
    """smartcrop-style importance (detail, skin, saturation) plus spectral
    residual saliency, 0..1, zero where the picture is transparent."""
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    # detail: local edge energy (Laplacian), on an absolute scale so a smooth gradient stays ~0
    lap = np.zeros_like(lum)
    lap[1:-1, 1:-1] = np.abs(4 * lum[1:-1, 1:-1] - lum[:-2, 1:-1] - lum[2:, 1:-1] - lum[1:-1, :-2] - lum[1:-1, 2:])
    detail = np.clip(_blur(lap, 5) / 0.12, 0.0, 1.0)
    # skin: close to a skin hue, at a skin brightness (smartcrop's test, in normalised rgb)
    mag = np.sqrt((rgb ** 2).sum(-1)) + 1e-6
    skin_c = np.array([0.78, 0.57, 0.44], dtype=np.float32)
    skin_c = skin_c / np.linalg.norm(skin_c)
    dist = np.sqrt((((rgb / mag[..., None]) - skin_c) ** 2).sum(-1))
    skin = np.clip(1.0 - dist / 0.1, 0.0, 1.0) * ((lum > 0.2) & (lum < 0.95))
    # saturation, at a mid brightness
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    satur = np.clip((sat - 0.4) / 0.6, 0.0, 1.0) * ((lum > 0.05) & (lum < 0.9))
    # spectral residual saliency, scaled by how much structure the picture has
    f = np.fft.fft2(lum)
    log_amp = np.log(np.abs(f) + 1e-8)
    resid = log_amp - _blur(log_amp, 3)
    sr = np.abs(np.fft.ifft2(np.exp(resid + 1j * np.angle(f)))) ** 2
    sr = _blur(_blur(sr, 7), 7)
    sr = sr / (sr.max() + 1e-9)
    structure = min(1.0, float(detail.mean()) / 0.08)
    imp = 0.35 * detail + 0.25 * skin * (0.3 + 0.7 * detail) + 0.1 * satur + 0.3 * sr * structure
    imp = _blur(imp, 5) * alpha
    return np.clip(imp / max(0.35, float(imp.max())), 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------
# faces


_DETECTOR = None  # False once it has failed to load


def _faces(rgba) -> List[Frac]:
    try:
        import cv2
    except Exception:
        return []
    from PIL import Image

    w, h = rgba.size
    if w * h < 96 * 96:
        return []
    out: List[Tuple[Frac, float]] = []
    for side in (FACE_SIDE, 320):
        sc = min(1.0, side / max(w, h))
        sz = (max(1, int(w * sc)), max(1, int(h * sc)))
        bgr = cv2.cvtColor(np.asarray(rgba.convert("RGB").resize(sz, Image.BILINEAR)), cv2.COLOR_RGB2BGR)
        for f, score in _yunet(cv2, bgr) if _DETECTOR is not False else []:
            out.append((f, score))
        if _DETECTOR is False:
            out = [(f, 1.0) for f in _haar(cv2, bgr)]
            break
    # merge the two scales; tiny faces (a crowd far away) are not the subject
    keep: List[Frac] = []
    for f, _ in sorted(out, key=lambda t: -t[1]):
        if (f[3] - f[1]) * h < 0.035 * min(w, h):
            continue
        if not any(_iou(f, k) > 0.3 for k in keep):
            keep.append(f)
    return keep[:12]


def _yunet(cv2, bgr) -> List[Tuple[Frac, float]]:
    global _DETECTOR
    h, w = bgr.shape[:2]
    try:
        if _DETECTOR is None:
            _DETECTOR = cv2.FaceDetectorYN.create(str(MODEL), "", (w, h), 0.8, 0.3, 50)
        _DETECTOR.setInputSize((w, h))
        _, res = _DETECTOR.detect(bgr)
    except Exception:
        _DETECTOR = False
        return []
    if res is None:
        return []
    out = []
    for r in res:
        x, y, fw, fh, sc = (float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[14]))
        if not all(math.isfinite(c) for c in (x, y, fw, fh, sc)) or fw <= 0 or fh <= 0:
            continue
        out.append(((max(0.0, x / w), max(0.0, y / h), min(1.0, (x + fw) / w), min(1.0, (y + fh) / h)), sc))
    return out


_CASCADES = None


def _haar(cv2, bgr) -> List[Frac]:
    """OpenCV's bundled frontal cascade (a fallback for YuNet); a hit counts
    only if it is skin-toned and two eyes are found in its upper part, or it
    is mostly skin-toned."""
    global _CASCADES
    try:
        if _CASCADES is None:
            _CASCADES = [cv2.CascadeClassifier(cv2.data.haarcascades + n) for n in
                         ("haarcascade_frontalface_default.xml", "haarcascade_eye.xml")]
        face, eye = _CASCADES
        gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        skin = (ycc[..., 1] >= 135) & (ycc[..., 1] <= 180) & (ycc[..., 2] >= 85) & (ycc[..., 2] <= 135)
        mn = max(24, int(min(gray.shape) * 0.05))
        out = []
        H, W = gray.shape
        for (x, y, fw, fh) in face.detectMultiScale(gray, 1.08, 4, minSize=(mn, mn)):
            sk = skin[y:y + fh, x:x + fw].mean()
            if sk < 0.25:
                continue
            roi = gray[y:y + int(fh * 0.6), x:x + fw]
            eyes = len(eye.detectMultiScale(roi, 1.1, 3, minSize=(max(5, fw // 10), max(5, fw // 10))))
            if eyes < 2 and sk < 0.45:  # skin-toned and two eyes, or very skin-toned
                continue
            out.append((x / W, y / H, (x + fw) / W, (y + fh) / H))
        return out
    except Exception:
        return []


def _iou(a: Frac, b: Frac) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------------------
# cropping by bleed


Rect = Tuple[float, float, float, float]  # x, y, w, h on the canvas


def source_window(e: dict, W0: float, H0: float, size: Tuple[int, int]) -> Frac:
    """The part of the image the source design showed, as image fractions."""
    iw, ih = size
    x, y, w, h = (float(e.get(k, 0) or 0) for k in ("x", "y", "width", "height"))
    if w <= 0 or h <= 0:
        return (0.0, 0.0, 1.0, 1.0)
    fit = e.get("fit", "contain")
    if fit == "cover":
        s = max(w / iw, h / ih)
    elif fit == "contain":
        s = min(w / iw, h / ih)
    else:
        s = None
    if s is None:  # fill: the image is stretched to the box
        ox, oy, dw, dh = x, y, w, h
    else:
        dw, dh = iw * s, ih * s
        ox, oy = x + (w - dw) / 2.0, y + (h - dh) / 2.0
    # visible: inside the box (cover crops to it) and on the canvas
    vx0, vy0 = max(0.0, x if fit == "cover" else ox), max(0.0, y if fit == "cover" else oy)
    vx1 = min(W0, x + w if fit == "cover" else ox + dw)
    vy1 = min(H0, y + h if fit == "cover" else oy + dh)
    if vx1 <= vx0 or vy1 <= vy0:
        return (0.0, 0.0, 1.0, 1.0)
    f = ((vx0 - ox) / dw, (vy0 - oy) / dh, (vx1 - ox) / dw, (vy1 - oy) / dh)
    return tuple(min(1.0, max(0.0, v)) for v in f)  # type: ignore[return-value]


def _thirds(c: Optional[float]) -> float:
    """Where the subject should land along an axis, given the copy's centre
    there (0..1): on the third line opposite the copy, centred if it is."""
    if c is None:
        return 0.5
    if c < 0.42:
        return 2.0 / 3.0
    if c > 0.58:
        return 1.0 / 3.0
    return 0.5


def _masses(v: Vision, x0, y0, x1, y1):
    """Mean importance over fractional regions (numpy arrays of equal shape)."""
    h, w = v.imp.shape
    S = v.integral
    X0 = np.clip(np.floor(x0 * w), 0, w).astype(int)
    X1 = np.clip(np.ceil(x1 * w), 0, w).astype(int)
    Y0 = np.clip(np.floor(y0 * h), 0, h).astype(int)
    Y1 = np.clip(np.ceil(y1 * h), 0, h).astype(int)
    n = np.maximum(0, X1 - X0) * np.maximum(0, Y1 - Y0)
    tot = S[Y1, X1] - S[Y0, X1] - S[Y1, X0] + S[Y0, X0]
    return np.where(n > 0, tot / np.maximum(1, n), 0.0)


def cover_box(v: Optional[Vision], W: float, H: float, copy: Sequence[Rect] = (),
              shown: Frac = (0.0, 0.0, 1.0, 1.0)) -> Optional[Rect]:
    """Where to draw a canvas-covering photo: its own aspect, scaled to cover
    W x H, slid along the free axis. None means the plain centre crop."""
    if v is None:
        return None
    return _cover_box(v, float(W), float(H), tuple(tuple(round(c, 1) for c in r) for r in copy),
                      tuple(round(c, 4) for c in shown))


@lru_cache(maxsize=512)
def _cover_box(v: Vision, W: float, H: float, copy: Tuple[Rect, ...], shown: Frac) -> Optional[Rect]:
    iw, ih = v.size
    s = max(W / iw, H / ih)
    bw, bh = iw * s, ih * s
    wx, wy = W / bw, H / bh  # the visible window, as image fractions
    if wx > 0.985 and wy > 0.985:
        return None
    horiz = wx <= wy  # the axis with room to slide
    win = wx if horiz else wy
    room = 1.0 - win
    size = bw if horiz else bh
    centre = room / 2.0
    a0, a1 = (shown[0], shown[2]) if horiz else (shown[1], shown[3])
    if not v.subject:
        # nothing to aim at: keep what the source showed in view, centred on it
        t = min(max((a0 + a1) / 2.0 - win / 2.0, 0.0), room)
        best_t = t
    else:
        T = np.linspace(0.0, room, STEPS + 1)
        T = np.append(T, centre)
        ones, zeros = np.ones_like(T), np.zeros_like(T)

        def region(lo, hi, c0, c1):
            """Image-fraction region of a canvas span [lo, hi] (fractions of the window) at each T."""
            if horiz:
                return T + lo * win, c0 * ones, T + hi * win, c1 * ones
            return c0 * ones, T + lo * win, c1 * ones, T + hi * win

        total = v.mass((0.0, 0.0, 1.0, 1.0)) + 1e-6
        sc = _masses(v, *region(0.0, 1.0, 0.0, 1.0)) * win / total  # the picture's weight that stays
        ov = np.maximum(0.0, np.minimum(T + win, a1) - np.maximum(T, a0)) / max(1e-6, min(win, a1 - a0))
        sc = sc + 0.6 * ov  # what the source showed stays in view
        area = sum(r[2] * r[3] for r in copy)
        if area > 0:
            cc = sum((r[0] + r[2] / 2.0 if horiz else r[1] + r[3] / 2.0) * r[2] * r[3] for r in copy) / area
            target = _thirds(cc / (W if horiz else H))
        else:
            target = 0.5
        focal = v.focal[0] if horiz else v.focal[1]
        s0, s1 = (v.extent[0], v.extent[2]) if horiz else (v.extent[1], v.extent[3])
        held = min(centre + win, s1) - max(centre, s0) >= 0.9 * (s1 - s0)
        if not held:  # the centre crop cuts the subject: put it on its third line (else leave it be)
            sc = sc - 0.5 * np.abs((focal - T) / win - target)
        for hd in v.heads:  # never cut a head
            h0, h1 = (hd[0], hd[2]) if horiz else (hd[1], hd[3])
            cut = np.maximum(0.0, T - h0) + np.maximum(0.0, h1 - (T + win))
            sc = sc - np.where(cut > 1e-4, 4.0 + 20.0 * cut / max(1e-6, h1 - h0), 0.0)
        for bd in v.baked:  # or text baked into the asset
            h0, h1 = (bd[0], bd[2]) if horiz else (bd[1], bd[3])
            cut = np.maximum(0.0, T - h0) + np.maximum(0.0, h1 - (T + win))
            sc = sc - np.where(cut > 1e-4, 3.0 + 16.0 * cut / max(1e-6, h1 - h0), 0.0)
        for r in copy:  # copy keeps to the quiet part, and off the people
            if horiz:
                lo, hi, c0, c1 = r[0] / W, (r[0] + r[2]) / W, r[1] / H, (r[1] + r[3]) / H
            else:
                lo, hi, c0, c1 = r[1] / H, (r[1] + r[3]) / H, r[0] / W, (r[0] + r[2]) / W
            f = region(lo, hi, c0, c1)
            sc = sc - 2.5 * _masses(v, *f) * min(1.0, 6.0 * r[2] * r[3] / max(1.0, W * H))
            for pp in v.people:
                ix = np.maximum(0.0, np.minimum(f[2], pp[2]) - np.maximum(f[0], pp[0]))
                iy = np.maximum(0.0, np.minimum(f[3], pp[3]) - np.maximum(f[1], pp[1]))
                sc = sc - 1.5 * ix * iy / max(1e-6, (pp[2] - pp[0]) * (pp[3] - pp[1]))
        k = int(np.argmax(sc[:-1]))
        best_t = float(T[k]) if sc[k] > sc[-1] + 0.03 else centre  # only move when it is worth it
    if abs(best_t - centre) * size < 2.0:
        return None
    if horiz:
        return (-best_t * bw, -(bh - H) / 2.0, bw, bh)
    return (-(bw - W) / 2.0, -best_t * bh, bw, bh)


def drawn_faces(v: Vision, box: Rect, which: str = "people") -> List[Rect]:
    """Face regions on the canvas for a picture drawn to fill ``box`` at its own aspect."""
    x, y, w, h = box
    return [(x + f[0] * w, y + f[1] * h, (f[2] - f[0]) * w, (f[3] - f[1]) * h) for f in getattr(v, which)]


def centre_box(v: Vision, W: float, H: float) -> Rect:
    """The box the renderer's centre crop amounts to, at the picture's own aspect."""
    iw, ih = v.size
    s = max(W / iw, H / ih)
    bw, bh = iw * s, ih * s
    return ((W - bw) / 2.0, (H - bh) / 2.0, bw, bh)


# ----------------------------------------------------------------------------
# a plan's background crop, and what it costs to set things on the subject

FACE = 1.0  # x importance, for an item over a head (share of the smaller box)
BAKED = 0.85  # x importance, for live copy over text baked into the photo
PEOPLE = 0.3  # x importance, over head and shoulders
QUIET = 0.8  # x importance, per unit of picture weight under an item past the picture's mean
QUIET_SLACK = 0.1  # picture weight past the mean that costs nothing (the map's noise on plain areas)
CAP = 1.5  # x importance, at most, per item
DECOR = 30.0  # importance of a decoration set on the photo


def _placed_rects(plan) -> List[Rect]:
    return [(x, y, r.w, r.h) for r, x, y in plan.placed + plan.floaters if r.w > 0 and r.h > 0]


def bg_box(e: dict, plan, d, ctx, extra: Sequence[Rect] = ()) -> Optional[Rect]:
    """The bleed box for a canvas-covering background photo under this plan,
    or None for the plain centre crop (0, 0, W, H, cover). ``extra``: other
    things set on it whatever the plan (decoration)."""
    if e.get("type") != "image" or not BLEED:
        return None
    v = analyze(e.get("asset_id"))
    if v is None:
        return None
    try:
        return cover_box(v, ctx.W, ctx.H, _placed_rects(plan) + list(extra),
                         source_window(e, d.W, d.H, v.size))
    except Exception:
        return None


def item_penalties(plan, d, ctx, extra: Sequence[Rect] = ()) -> Tuple[dict, float]:
    """What each placed item of this plan costs on a background photo with a
    subject, as this plan crops it: over a face, over the people, or on the
    busy part of the picture (``{item key: cost}``), plus what the decoration
    (``extra``, importance DECOR) costs."""
    per: dict = {}
    deco = 0.0
    for e in d.backgrounds:
        if e.get("type") != "image":
            continue
        v = analyze(e.get("asset_id"))
        if v is None or not v.subject:
            continue
        box = bg_box(e, plan, d, ctx, extra) or centre_box(v, ctx.W, ctx.H)
        heads, people = drawn_faces(v, box, "heads"), drawn_faces(v, box, "people")
        baked = drawn_faces(v, box, "baked")
        avg = v.mass((0.0, 0.0, 1.0, 1.0))
        bx, by, bw, bh = box
        things = [(x, y, r.w, r.h, float(getattr(r.item, "importance", 40.0)), getattr(r.item, "key", id(r.item)))
                  for r, x, y in plan.placed + plan.floaters] + [tuple(rc) + (DECOR, None) for rc in extra]
        for x, y, w, h, imp, key in things:
            if w <= 0 or h <= 0:
                continue
            a = w * h
            k = 0.0
            for (fx, fy, fw, fh), wt in [(f, FACE) for f in heads] + [(f, BAKED) for f in baked] + \
                    [(f, PEOPLE) for f in people]:
                ix = max(0.0, min(x + w, fx + fw) - max(x, fx))
                iy = max(0.0, min(y + h, fy + fh) - max(y, fy))
                if ix * iy > 0:
                    k += wt * ix * iy / max(1.0, min(a, fw * fh))
            q = v.mass(((x - bx) / bw, (y - by) / bh, (x + w - bx) / bw, (y + h - by) / bh))
            k += QUIET * max(0.0, q - avg - QUIET_SLACK)
            if key is None:
                deco += imp * min(CAP, k)
            else:
                per[key] = per.get(key, 0.0) + imp * min(CAP, k)
    return per, deco


def penalty(plan, d, ctx, extra: Sequence[Rect] = ()) -> float:
    """Items and decoration set over a face, over the people, or on the busy
    part of a background photo, as this plan crops it."""
    per, deco = item_penalties(plan, d, ctx, extra)
    return sum(per.values()) + deco


def rank(plans, d, ctx, extra: Sequence[Rect] = ()) -> None:
    """Charge every candidate for what it sets on the subject (see
    item_penalties). An item a plan leaves out pays what it costs where the
    other candidates set it best, so copy is moved off the subject but never
    dropped for it, and dropping stays exactly as attractive as it was."""
    if np is None or not any(e.get("type") == "image" for e in d.backgrounds):
        return
    costs = []
    best: dict = {}
    for p in plans:
        try:
            per, deco = item_penalties(p, d, ctx, extra)
        except Exception:
            per, deco = {}, 0.0
        costs.append((per, deco))
        for k, c in per.items():
            best[k] = min(best.get(k, c), c)
    for p, (per, deco) in zip(plans, costs):
        if not per and not deco:
            continue
        pen = sum(per.values()) + deco + sum(best.get(getattr(it, "key", None), 0.0) for it in p.hidden)
        if pen:
            p.breakdown["subject"] = -pen
            p.score -= pen


def photo_subject(d) -> bool:
    """A background photo with a subject worth keeping copy off."""
    for e in d.backgrounds:
        if e.get("type") == "image":
            v = analyze(e.get("asset_id"))
            if v is not None and v.subject:
                return True
    return False
