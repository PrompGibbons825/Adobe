"""Download the model files that are too large for git. The Docker build runs this; nothing is
downloaded at run time. To run outside Docker:  python tools/fetch_models.py

  clip/vision_model_quantized.onnx, clip/text_model_quantized.onnx, clip/tokenizer.json
      CLIP ViT-B/32 (quantized ONNX) and its tokenizer: logo and orientation checks
  u2netp.onnx
      small salient-object mask: where the subject of a photo is, for focal crops

They land in engine/subs/jeff/engine/models/. Each file is checked against a pinned SHA-256.
The YuNet face model is in git, and the OCR models ship inside the rapidocr wheel.
If a download fails after three tries, the build goes on with a warning: the engine still runs
and simply skips the checks that need the missing file.
"""
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "engine" / "subs" / "jeff" / "engine" / "models"
HF = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"
FILES = {
    "clip/vision_model_quantized.onnx": (f"{HF}/onnx/vision_model_quantized.onnx",
                                         "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299"),
    "clip/text_model_quantized.onnx": (f"{HF}/onnx/text_model_quantized.onnx",
                                       "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a"),
    "clip/tokenizer.json": (f"{HF}/tokenizer.json",
                            "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd"),
    "u2netp.onnx": ("https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
                    "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


missing = []
for rel, (url, digest) in FILES.items():
    dst = ROOT / rel
    if dst.is_file() and sha256(dst) == digest:
        print("ok     ", rel)
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, dst)
            if sha256(dst) != digest:
                raise ValueError("checksum mismatch")
            print("fetched", rel, dst.stat().st_size // 1024, "KB")
            break
        except Exception as ex:  # noqa: BLE001
            print(f"retry   {rel}: {ex}", file=sys.stderr)
            dst.unlink(missing_ok=True)
            time.sleep(2 * (attempt + 1))
    else:
        missing.append(rel)

if missing:
    print("WARNING: could not fetch", ", ".join(missing), "- the engine will skip the checks that use them",
          file=sys.stderr)
