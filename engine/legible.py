"""Does the copy actually read? Render a candidate the way the judge will see
it (the long side at JUDGE px), read it with OCR, and report each text's share
of words read. Measured on our renders: below about 4 px of letter height at a
512-768 px view most words are lost; turned (vertical) text reads poorly at any
size; and among larger texts the misses are text on busy or low-contrast ground,
which no size rule sees.

A text only has to read if it reads in the source rendered at the same view:
a script signature or a decorative face that OCR cannot read in the designer's
own version is not held against the resize.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from PIL import Image

from . import saliency, vision

JUDGE = int(os.environ.get("CHAMELEON_JUDGE_SIDE", "768"))
_RENDERERS = {}


def _words(s):
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) >= 2]


def _close(a, b):
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1 or min(len(a), len(b)) < 4:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


def render(design: dict):
    """The design rendered by the official renderer, as a PIL image, or None."""
    try:
        from renderer.converter import convert
        from . import textfit
        assets = saliency.ASSETS_DIR
        if textfit.FONTS_PICKLE:
            # a dev font set (the Crello cases): one renderer per pickle
            key = str(Path(assets).parent / "fonts")
            if key not in _RENDERERS:
                from cr_renderer import CrelloV5Renderer
                from cr_renderer.fonts import FontManager
                r = CrelloV5Renderer.__new__(CrelloV5Renderer)
                r.features = convert._STUB_FEATURES
                r.font_manager = FontManager(textfit.FONTS_PICKLE)
                convert._renderer_cache[key] = r
                _RENDERERS[key] = r
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.png"
            convert.render_design_to_png(design, assets, str(p))
            return Image.open(p).convert("RGB")
    except Exception:
        return None


def recall(design: dict):
    """{text id: share of its words OCR reads at the judge's view}, or None."""
    im = render(design)
    if im is None:
        return None
    k = JUDGE / max(im.size)
    if k < 1:
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    got = vision.read_text(im)
    if got is None:
        return None
    toks, flat = got
    out = {}
    for e in design["elements"]:
        if e["type"] != "text":
            continue
        ws = _words(e.get("content"))
        if ws:
            out[e["id"]] = sum(any(_close(w, t) for t in toks) or (len(w) >= 3 and w in flat) for w in ws) / len(ws)
    return out
