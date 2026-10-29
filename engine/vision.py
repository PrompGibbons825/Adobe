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


# --- CLIP ------------------------------------------------------------------

_TOK = None
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def _clip_image(ims):
    s = _session("clip/vision_model_quantized.onnx")
    if s is None:
        return None
    batch = []
    for im in ims:
        im = im.convert("RGB")
        k = 224 / min(im.size)
        im = im.resize((max(224, round(im.width * k)), max(224, round(im.height * k))), Image.BICUBIC)
        l, t = (im.width - 224) // 2, (im.height - 224) // 2
        x = (np.asarray(im.crop((l, t, l + 224, t + 224))).astype(np.float32) / 255 - _MEAN) / _STD
        batch.append(x.transpose(2, 0, 1))
    e = s.run(None, {"pixel_values": np.stack(batch)})[0]
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def _clip_text(texts):
    global _TOK
    s = _session("clip/text_model_quantized.onnx")
    if s is None:
        return None
    try:
        if _TOK is None:
            from tokenizers import Tokenizer
            _TOK = Tokenizer.from_file(str(MODELS / "clip" / "tokenizer.json"))
    except Exception:
        return None
    ids = [_TOK.encode(t.lower()).ids[:77] for t in texts]
    n = max(len(i) for i in ids)
    arr = np.array([i + [49407] * (n - len(i)) for i in ids], np.int64)
    e = s.run(None, {"input_ids": arr})[0]
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def relevance(im: Image.Image, boxes, text: str):
    """CLIP similarity of each crop box (fractions of im) to the text, or None."""
    t = _clip_text([text])
    if t is None:
        return None
    W, H = im.size
    crops = [im.crop((int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H))) for b in boxes]
    v = _clip_image(crops)
    return None if v is None else (v @ t[0]).tolist()


_LOGO_LABELS = ["a brand logo", "a company logo or emblem", "a decorative shape", "a geometric icon",
                "a photograph", "a line or divider", "a pattern of confetti or stars"]
_LOGO_T = None


def logo_prob(im: Image.Image):
    """CLIP's probability that a small image is a logo or emblem (on white), or None."""
    global _LOGO_T
    if _LOGO_T is None:
        _LOGO_T = _clip_text(_LOGO_LABELS)
        if _LOGO_T is None:
            _LOGO_T = False
    if _LOGO_T is False:
        return None
    rgba = im.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    bg.alpha_composite(rgba)
    v = _clip_image([bg.convert("RGB")])
    if v is None:
        return None
    p = np.exp(100 * (v @ _LOGO_T.T))[0]
    p /= p.sum()
    return float(p[0] + p[1])


_ORIENT_LABELS = ["a photo of a person", "a photo of people", "a portrait of a face",
                  "a human hand", "a photo of a building", "architecture or a city street",
                  "a room interior", "a landscape with sky",
                  "a flower", "a product or object", "food", "an abstract texture or pattern",
                  "a plant or leaves"]
_ORIENT_UPRIGHT = 8          # the first eight labels have a top and a bottom
_ORIENT_T = None


def needs_upright(im: Image.Image):
    """CLIP's probability that a picture has a natural up (people, faces,
    buildings, rooms, landscapes) and must not be turned; None if unavailable."""
    global _ORIENT_T
    if _ORIENT_T is None:
        _ORIENT_T = _clip_text(_ORIENT_LABELS)
        if _ORIENT_T is None:
            _ORIENT_T = False
    if _ORIENT_T is False:
        return None
    v = _clip_image([im.convert("RGB")])
    if v is None:
        return None
    p = np.exp(100 * (v @ _ORIENT_T.T))[0]
    p /= p.sum()
    return float(p[:_ORIENT_UPRIGHT].sum())
