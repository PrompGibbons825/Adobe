"""Extra photo signals: U2-Netp saliency (Apache-2.0) and PP-OCRv4 text boxes (Apache-2.0).

Read-only; models live in engine/smart/models/. OpenCV DNN runs both on CPU with no
network at run time. Everything degrades to empty when numpy, OpenCV or a model is missing.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

Frac = Tuple[float, float, float, float]

_DIR = Path(__file__).resolve().parent / "models"
U2NET = _DIR / "u2netp.onnx"
OCR_DET = _DIR / "ch_PP-OCRv4_det_infer.onnx"
U2_SIDE = 320
OCR_MIN = 960  # detector resize, long side at least this (RapidOCR-style)

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

_U2 = None  # False once load or inference fails
_OCR = None


def u2_saliency(rgb, alpha, out_h: int, out_w: int):
    """Salient-object map from U2-Netp, resized to (out_h, out_w), values 0..1."""
    if np is None:
        return None
    try:
        import cv2
    except Exception:
        return None
    global _U2
    h, w = rgb.shape[:2]
    if h < 4 or w < 4 or _U2 is False:
        return None
    try:
        if _U2 is None:
            if not U2NET.is_file():
                _U2 = False
                return None
            _U2 = cv2.dnn.readNetFromONNX(str(U2NET))
        sc = U2_SIDE / max(h, w)
        sw, sh = max(8, round(w * sc)), max(8, round(h * sc))
        small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(small, scalefactor=1.0 / 255.0, size=(sw, sh),
                                     mean=(0, 0, 0), swapRB=True, crop=False)
        _U2.setInput(blob)
        out = _U2.forward()[0, 0]
        out = cv2.resize(out, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        out = out * alpha[:out_h, :out_w] if alpha is not None else out
        mx = float(out.max()) + 1e-9
        return np.clip(out / mx, 0.0, 1.0).astype(np.float32)
    except Exception:
        _U2 = False
        return None


def _ocr_preprocess(img):
    import cv2

    h, w = img.shape[:2]
    if min(h, w) < OCR_MIN:
        ratio = float(OCR_MIN) / min(h, w)
    else:
        ratio = 1.0
    rh, rw = int(round(h * ratio / 32) * 32), int(round(w * ratio / 32) * 32)
    rh, rw = max(32, rh), max(32, rw)
    resized = cv2.resize(img, (rw, rh))
    blob = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    return blob.transpose(2, 0, 1)[None], (h, w), (rh, rw)


def _boxes_from_map(pred, src_hw, resized_hw, thresh=0.35, box_thresh=0.55) -> List[Frac]:
    """DB-style boxes as axis-aligned fractions of the source image."""
    import cv2

    src_h, src_w = src_hw
    rh, rw = resized_hw
    bitmap = (pred[0, 0] > thresh).astype(np.uint8) * 255
    outs = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[0] if len(outs) == 2 else outs[1]
    boxes: List[Frac] = []
    sx, sy = src_w / rw, src_h / rh
    for c in contours[:200]:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 4 or bh < 4:
            continue
        patch = pred[0, 0, y:y + bh, x:x + bw]
        if patch.size == 0 or float(patch.mean()) < box_thresh:
            continue
        x0, y0 = x * sx / src_w, y * sy / src_h
        x1, y1 = (x + bw) * sx / src_w, (y + bh) * sy / src_h
        if (x1 - x0) * (y1 - y0) < 0.0008:
            continue
        boxes.append((max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)))
    return _merge_boxes(boxes)


def _merge_boxes(boxes: List[Frac], iou=0.35) -> List[Frac]:
    keep: List[Frac] = []
    for f in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if not any(_iou(f, k) > iou for k in keep):
            keep.append(f)
    return keep[:24]


def _iou(a: Frac, b: Frac) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@lru_cache(maxsize=64)
def baked_text(path: str, mtime: int, size: int) -> Tuple[Frac, ...]:
    """Text baked into a raster asset (detector only, no recognition)."""
    if np is None:
        return ()
    try:
        import cv2
        from PIL import Image
    except Exception:
        return ()
    global _OCR
    if _OCR is False:
        return ()
    try:
        if _OCR is None:
            if not OCR_DET.is_file():
                _OCR = False
                return ()
            _OCR = cv2.dnn.readNetFromONNX(str(OCR_DET))
        with Image.open(path) as im:
            im.load()
            rgb = np.asarray(im.convert("RGB"))
        blob, src_hw, res_hw = _ocr_preprocess(rgb)
        _OCR.setInput(blob)
        pred = _OCR.forward()
        return tuple(_boxes_from_map(pred, src_hw, res_hw))
    except Exception:
        _OCR = False
        return ()


def quiet_band(v_imp, canvas_rect, thr_ratio=0.35) -> Frac:
    """Fractional region of the image that is quieter than ``thr_ratio`` of the mean mass."""
    if np is None or v_imp is None:
        return (0.0, 0.0, 1.0, 1.0)
    h, w = v_imp.shape
    avg = float(v_imp.mean())
    thr = avg * thr_ratio
    mask = v_imp <= max(thr, float(np.quantile(v_imp, 0.25)))
    if not mask.any():
        return (0.0, 0.0, 1.0, 1.0)
    ys, xs = np.nonzero(mask)
    return (float(xs.min()) / w, float(ys.min()) / h, float(xs.max() + 1) / w, float(ys.max() + 1) / h)
