"""Download the model files the engine uses that are too large for git.
Run once after cloning (the Docker build runs it too):  python tools/fetch_models.py

  engine/models/clip/   CLIP ViT-B/32 (quantized ONNX) + tokenizer: logo detection
  engine/models/u2netp.onnx   small salient-object mask (optional; off by default)
YuNet (faces) is in git. OCR models ship inside the rapidocr wheel.
Without these files the engine still runs; it just skips what needs them.
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "engine" / "subs" / "jeff" / "engine" / "models"
FILES = {
    "clip/vision_model_quantized.onnx":
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx",
    "clip/text_model_quantized.onnx":
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model_quantized.onnx",
    "clip/tokenizer.json":
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json",
    "u2netp.onnx": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
}

for rel, url in FILES.items():
    dst = ROOT / rel
    if dst.is_file() and dst.stat().st_size > 1000:
        print("ok     ", rel)
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    print("fetch  ", rel, file=sys.stderr)
    urllib.request.urlretrieve(url, dst)
    print("fetched", rel, dst.stat().st_size // 1024, "KB")
