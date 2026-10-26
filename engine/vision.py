"""Small offline vision models, run on the CPU with onnxruntime. Every model
ships in engine/models/; nothing is fetched at run time. Each helper returns
None when its model or runtime is unavailable, and the engine then falls back
to its classical image analysis.

  subject_mask(asset)  where the subject is: IS-Net (general use), the
                       foreground a person would cut out of the picture
  read_text(asset)     words drawn into a picture (PP-OCRv4 mobile, RapidOCR)
  relevance(asset, s)  which part of a picture the copy is about (CLIP ViT-B/32)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

MODELS = Path(__file__).resolve().parent / "models"
MASK_MODEL = os.environ.get("CHAMELEON_MASK_MODEL", "u2netp")
_SESS = {}


def _session(name):
    if name in _SESS:
        return _SESS[name]
    s = None
    try:
        import onnxruntime as ort
        p = MODELS / name
        if p.is_file():
            o = ort.SessionOptions()
            o.log_severity_level = 3
            s = ort.InferenceSession(str(p), o, providers=["CPUExecutionProvider"])
    except Exception:
        s = None
    _SESS[name] = s
    return s


# --- subject mask ----------------------------------------------------------

_MASK_CFG = {
    "isnet-general-use": (1024, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "u2netp": (320, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def mask_from_image(im: Image.Image, model: str = MASK_MODEL):
    """Foreground probability (H x W float32, 0..1) at the picture's size, or None."""
    s = _session(model + ".onnx")
    if s is None:
        return None
    side, mean, std = _MASK_CFG[model]
    x = np.asarray(im.convert("RGB").resize((side, side), Image.BILINEAR)).astype(np.float32) / 255.0
    x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)
    x = x.transpose(2, 0, 1)[None]
    out = s.run(None, {s.get_inputs()[0].name: x})[0][0, 0]
    lo, hi = float(out.min()), float(out.max())
    out = (out - lo) / (hi - lo) if hi > lo else np.zeros_like(out)
    return np.asarray(Image.fromarray((out * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR),
                      dtype=np.float32) / 255.0


# --- OCR -------------------------------------------------------------------

_OCR = None


def _ocr():
    global _OCR
    if _OCR is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR = RapidOCR()
        except Exception:
            _OCR = False
    return _OCR or None


def read_text(im: Image.Image):
    """(words, run-together text) OCR finds in a picture, lower-case; None if
    OCR is unavailable. OCR often drops the spaces between words, so a word
    counts as read if it appears inside the run-together text as well."""
    import re
    o = _ocr()
    if o is None:
        return None
    try:
        res, _ = o(np.asarray(im.convert("RGB"))[:, :, ::-1].copy())
    except Exception:
        return None
    lines = [r[1] for r in (res or [])]
    toks = [w for ln in lines for w in re.findall(r"[a-z0-9]+", ln.lower())]
    return toks, re.sub(r"[^a-z0-9]", "", " ".join(lines).lower())
