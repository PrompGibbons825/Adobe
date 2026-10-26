"""Calibrate readability: render outputs at the size a judge probably sees,
read them with OCR, and relate each text's word recall to its letter height.

usage (repo root):
  python tools/legibility.py OUT_DIR [--side 1024] [--csv out.csv]
OUT_DIR holds <root>_p<k>_<W>x<H>.json outputs; roots crello/ and crello_holdout/.
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.crello_compare import render, renderer_with  # noqa: E402
from engine import textfit as tf  # noqa: E402


def words(s):
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) >= 2]


def close(a, b):
    """One edit apart counts as read (OCR drops or swaps a letter)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1 or min(len(a), len(b)) < 4:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


_OCR = None


def ocr_tokens(im):
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    res, _ = _OCR(np.asarray(im.convert("RGB"))[:, :, ::-1].copy())
    return [w for r in (res or []) for w in words(r[1])], [(r[0], r[1], float(r[2])) for r in (res or [])]


def found(w, toks, flat):
    """A word is read if OCR returned it (one edit allowed), or, since OCR often
    drops the spaces between words, if it appears inside the run-together text."""
    return any(close(w, t) for t in toks) or (len(w) >= 3 and w in flat)


def main():
    args = sys.argv[1:]
    side = int(args[args.index("--side") + 1]) if "--side" in args else 1024
    out_csv = args[args.index("--csv") + 1] if "--csv" in args else None
    d = Path(args[0])
    png = d / "_legibility"
    png.mkdir(exist_ok=True)
    rows = []
    cur = None
    for f in sorted(d.glob("crello*.json")):
        key = f.stem
        root = "crello_holdout" if key.startswith("crello_holdout") else "crello"
        if root != cur:
            renderer_with(f"{root}/fonts_subset.pickle", Path(root) / "assets")
            cur = root
        design = json.load(open(f))
        im = render(design, Path(root) / "assets", png / f"{key}.png")
        k = side / max(im.size)
        small = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
        toks, raw = ocr_tokens(small)
        flat = re.sub(r"[^a-z0-9]", "", " ".join(r[1] for r in raw).lower())
        for e in design["elements"]:
            if e["type"] != "text":
                continue
            ws = words(e.get("content"))
            if not ws:
                continue
            got = sum(found(w, toks, flat) for w in ws)
            asc, cap, desc = tf.metrics(e)
            rows.append(dict(key=key, id=e["id"], text=" ".join(ws)[:40], words=len(ws), recall=got / len(ws),
                             cap_px=round(e["font_size"] * cap * k, 1), fs=e["font_size"],
                             rot=abs(float(e.get("rotation", 0) or 0)) > 1))
    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    up = [r for r in rows if not r["rot"]]
    print(f"{len(rows)} texts in {len(set(r['key'] for r in rows))} designs at {side}px; upright {len(up)}")
    edges = [0, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 1e9]
    print(f"{'cap height px':<16}{'texts':>6}{'mean recall':>13}{'read >=80%':>12}")
    for a, b in zip(edges, edges[1:]):
        g = [r for r in up if a <= r["cap_px"] < b]
        if g:
            print(f"{f'{a}-{b if b < 1e9 else 999}':<16}{len(g):>6}{np.mean([r['recall'] for r in g]):>13.2f}"
                  f"{np.mean([r['recall'] >= 0.8 for r in g]):>12.2f}")
    rot = [r for r in rows if r["rot"]]
    if rot:
        print(f"turned texts: {len(rot)}, mean recall {np.mean([r['recall'] for r in rot]):.2f}")


if __name__ == "__main__":
    main()
