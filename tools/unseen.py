"""Dev tool: the unseen test suite. It defines "pass" for a resize, tile by tile.

    .venv/bin/python -m tools.unseen --split dev                      # 24 generated designs x 8 targets
    .venv/bin/python -m tools.unseen --split crello-dev               # Jeff's real Crello pairs
    .venv/bin/python -m tools.unseen --split dev --engine ../sprint-a --sheet /tmp/u.png --json /tmp/u.json
    .venv/bin/python -m tools.unseen --split dev --formats tier1 --jobs 4 --sheet-all /tmp/all.png

Splits: ``dev`` (seeds 1-3) and ``holdout`` (seeds 101-103) are generated here, written
independently of tools/designs.py (which the engine has been tuned against): eight families of
marketing designs on real Crello images and the Solstice kit, one design per family per seed,
sources in the tier-1 formats, targets = the other tier-1 formats + 728x90, 300x250, 1500x500.
``crello-dev`` / ``crello-holdout`` are Jeff's designer pairs (~/chameleon-data/jeff-crello):
every design of a pair is a source; targets = the other designer size + the tier-1 formats.
Holdout designs are never generated, printed or rendered unless ``--split holdout`` (or
``crello-holdout``) is asked for.

``--engine`` is any engine root (a worktree holding engine/resize.py); default: the current
directory if it is one, else this tool's own repo. The engine runs in its own process with
cwd = that root (tools/gate/engine_run.py), so the code that answers is that root's engine and
the code that measures is this repo's.

Per-tile checks, each a named check (all must hold for a tile to pass):

  C1 rules     tools.rules.hard_rule_errors and promise_errors empty, and no FALLBACK / crash
  C2 hidden    tools.gate.visible.hidden_elements == [] (alpha-aware covers, missing assets exempt)
  C3 legible   every visible text >= the engine's judge floor; on tier-1 targets the headline
               (most prominent source text) has a cap height >= 7 px at 512 px on the long side
  C4 on_photo  each text over an image: p10 WCAG >= 3.0 or no worse than the source (-0.3), and
               busy-ness no worse than the source (+0.02 over max(src, 0.03)) (tools.judge.terms)
  C5 faces     a face in a source image keeps > 85% of its area on the canvas and is < 10% covered
               by copy, logos or buttons painted above it (relative to the source, +5% slack)
  C6 audit     tools.audit.flags: overlap / hidden / flat / speck / nothing are failures; empty,
               sparse, lopsided, cramped are soft (reported, budget <= 10% of tiles)

The kit (~/chameleon-data/unseen-kit: assets/, fonts/, models/) is assembled on first use; the
engine and renderer find it through CHAMELEON_BRAND_KIT. Dev-only: nothing here ships.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = Path(os.environ.get("CHAMELEON_DATA", str(Path.home() / "chameleon-data")))
KIT = DATA / "unseen-kit"
JEFF = DATA / "jeff-crello"
CRELLO_ASSETS = JEFF / "dev" / "assets"
YUNET = KIT / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
BRAND = ROOT / "brand_kit"

TIER1 = {"ig-square": (1080, 1080), "ig-portrait": (1080, 1350), "story": (1080, 1920),
         "yt-thumb": (1280, 720), "hd-16x9": (1920, 1080), "fb-link": (1200, 628)}
EXTRA = {"leaderboard": (728, 90), "med-rect": (300, 250), "x-header": (1500, 500)}
SEEDS = {"dev": (1, 2, 3), "holdout": (101, 102, 103)}
SPLITS = ("dev", "holdout", "crello-dev", "crello-holdout")
MISSING_ASSET = "unseen_award_seal_2026.png"  # referenced on purpose, never in the kit

CHECKS = ("C1", "C2", "C3", "C4", "C5", "C6")
CHECK_NAMES = {"C1": "rules", "C2": "hidden", "C3": "legible", "C4": "on_photo", "C5": "faces", "C6": "audit"}
AUDIT_HARD = ("overlap", "hidden", "flat", "speck", "nothing")
AUDIT_SOFT = ("empty", "sparse", "lopsided", "cramped")
CAP_RATIO = 0.70  # Solstice Sans / Serif cap height over font size (measured on 'H')

# Faces the detectors miss in the illustrated people (annotated by eye, 2026-10-25), in asset px.
KNOWN_FACES = {"p5_r464_7.png": [(80, 125, 85, 90)], "p5_r465_7.png": [(85, 130, 85, 90)]}


# ============================================================================ kit


def ensure_kit() -> Path:
    """Copy the Crello dev images and the Solstice kit into KIT (once), fonts included."""
    a, f = KIT / "assets", KIT / "fonts"
    a.mkdir(parents=True, exist_ok=True)
    f.mkdir(parents=True, exist_ok=True)
    for p in sorted(CRELLO_ASSETS.glob("*.png")) + sorted((BRAND / "assets").glob("*.png")):
        q = a / p.name
        if not q.exists():
            shutil.copy2(p, q)
    for p in sorted((BRAND / "fonts").glob("*.[ot]tf")):
        q = f / p.name
        if not q.exists():
            shutil.copy2(p, q)
    if not YUNET.exists():
        try:
            import urllib.request

            YUNET.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(YUNET_URL, YUNET)
        except Exception:  # noqa: BLE001  offline: Haar + annotations only
            pass
    return KIT


def _rgba(path: Path):
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGBA")


@lru_cache(maxsize=None)
def classify(path: str) -> dict:
    """What an asset is, from its pixels: photo, texture, cutout, person, decor, tiny."""
    import numpy as np

    p = Path(path)
    im = _rgba(p)
    w, h = im.size
    im.thumbnail((256, 256))
    a = np.asarray(im)
    al = a[..., 3]
    tr, zero = float((al < 250).mean()), float((al < 16).mean())
    vis = (a[..., :3][al > 16] // 32).astype(np.int32)
    ncol = len(np.unique(vis[:, 0] * 64 + vis[:, 1] * 8 + vis[:, 2])) if len(vis) else 0
    n = p.name
    if n.startswith("solstice_logo"):
        kind = "logo"
    elif n.startswith("solstice_bg"):
        kind = "gradient"
    elif n.startswith("solstice_bottle"):
        kind = "product"
    elif min(w, h) < 40:
        kind = "tiny"
    elif tr < 0.02:
        kind = "photo" if ncol >= 20 and min(w, h) >= 280 else ("texture" if min(w, h) >= 280 else "tiny")
    elif 0.12 <= tr <= 0.55 and ncol >= 40:
        kind = "person" if faces(path) else "cutout"
    elif tr > 0.5 and zero > 0.3 and min(w, h) >= 40:
        kind = "decor"
    else:
        kind = "other"
    return {"kind": kind, "size": (w, h), "transparent": tr}


def kit_assets(kind: str) -> List[str]:
    return sorted(p.name for p in (KIT / "assets").glob("*.png") if classify(str(p))["kind"] == kind)


@lru_cache(maxsize=None)
def asset_size(name: str, kit: str = "") -> Tuple[int, int]:
    from PIL import Image

    with Image.open(Path(kit or KIT) / "assets" / name) as im:
        return im.size


@lru_cache(maxsize=None)
def faces(path: str) -> Tuple[Tuple[float, float, float, float], ...]:
    """Face boxes (x, y, w, h) in asset pixels: YuNet (score >= 0.7) union OpenCV's Haar frontal
    cascade with strict parameters, plus KNOWN_FACES. Transparent pixels are composited on grey."""
    import numpy as np

    p = Path(path)
    if not p.is_file():
        return ()
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return tuple(KNOWN_FACES.get(p.name, ()))
    from PIL import Image

    im = _rgba(p)
    if min(im.size) < 48:
        return ()
    bg = Image.new("RGBA", im.size, (128, 128, 128, 255))
    bg.alpha_composite(im)
    k = min(1.0, 1024 / max(im.size))
    if k < 1:
        bg = bg.resize((round(im.width * k), round(im.height * k)))
    bgr = np.asarray(bg.convert("RGB"))[:, :, ::-1].copy()
    h, w = bgr.shape[:2]
    found: List[Tuple[float, float, float, float]] = []
    if YUNET.exists():
        try:
            yn = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.7)
            yn.setInputSize((w, h))
            _, f = yn.detect(bgr)
            for r in (f if f is not None else []):
                found.append(tuple(float(v) / k for v in r[:4]))
        except Exception:  # noqa: BLE001
            pass
    try:
        haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ms = max(24, int(0.06 * min(w, h)))
        for (x, y, fw, fh) in haar.detectMultiScale(g, 1.1, 8, minSize=(ms, ms)):
            b = (x / k, y / k, fw / k, fh / k)
            if all(_iou(b, o) < 0.3 for o in found):
                found.append(b)
    except Exception:  # noqa: BLE001
        pass
    for b in KNOWN_FACES.get(p.name, ()):
        if all(_iou(b, o) < 0.3 for o in found):
            found.append(tuple(map(float, b)))
    return tuple(found)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1e-9, a[2] * a[3] + b[2] * b[3] - i)


@lru_cache(maxsize=None)
def _gray(name: str):
    import numpy as np

    im = _rgba(KIT / "assets" / name).convert("L")
    im.thumbnail((256, 256))
    return np.asarray(im, dtype=np.float32) / 255.0


def quiet_side(name: str, bw: float, bh: float, horizontal: bool) -> Tuple[str, float, float]:
    """Of a cover-cropped photo in a bw x bh box: the quieter half ('left'/'right' or
    'top'/'bottom'), and the mean luminance of each half (quiet, other)."""
    import numpy as np

    g = _gray(name)
    ih, iw = g.shape
    s = max(bw / iw, bh / ih)
    cw, ch = bw / s, bh / s
    x0, y0 = int((iw - cw) / 2), int((ih - ch) / 2)
    c = g[y0:y0 + max(2, int(ch)), x0:x0 + max(2, int(cw))]
    e = np.abs(np.diff(c, axis=0))[:, :-1] + np.abs(np.diff(c, axis=1))[:-1, :]
    if horizontal:
        m = c.shape[1] // 2
        a, b, la, lb = e[:, :m].mean(), e[:, m:].mean(), c[:, :m].mean(), c[:, m:].mean()
        return ("left", la, lb) if a <= b else ("right", lb, la)
    m = c.shape[0] // 2
    a, b, la, lb = e[:m].mean(), e[m:].mean(), c[:m].mean(), c[m:].mean()
    return ("top", la, lb) if a <= b else ("bottom", lb, la)


def region_stats(name: str, bw: float, bh: float, x0: float, y0: float, x1: float, y1: float) -> Tuple[float, float]:
    """Mean luminance and mean gradient of a cover-cropped photo (bw x bh box at 0,0) inside a rect."""
    g = _gray(name)
    ih, iw = g.shape
    s = max(bw / iw, bh / ih)
    ox, oy = (iw * s - bw) / 2, (ih * s - bh) / 2
    c = g[max(0, int((y0 + oy) / s)):max(1, int((y1 + oy) / s)), max(0, int((x0 + ox) / s)):max(1, int((x1 + ox) / s))]
    if c.shape[0] < 3 or c.shape[1] < 3:
        return float(c.mean()) if c.size else 0.5, 0.0
    e = abs(c[1:, :-1] - c[:-1, :-1]) + abs(c[:-1, 1:] - c[:-1, :-1])
    return float(c.mean()), float(e.mean())


# ============================================================================ copy and colour

SERIF, SANS = "Solstice Serif", "Solstice Sans"

COPY = {
    "home": {
        "head": ["Soft Days", "Slow Living", "The Linen Edit", "Rest, Reimagined",
                 "Make every corner of your home feel like a weekend morning",
                 "Handwoven throws and cushions for the cosiest season yet",
                 "Everything you need for a calmer, softer, warmer bedroom this autumn"],
        "sub": ["Natural fibres, made to last.", "Free delivery on orders over $50.",
                "Our new collection of washed linen, wool and cotton pieces is designed to be lived in, "
                "loved and passed on."],
        "cta": ["Shop now", "Explore the edit", "Shop the collection", "See what's new"],
        "kicker": ["NEW SEASON", "AUTUMN 2026", "JUST LANDED"],
        "badge": ["NEW", "ORGANIC", "HANDMADE", "BESTSELLER", "FREE SHIPPING"],
        "event": ("Home & Craft Fair", "SAT 14 NOV · 10 AM – 6 PM", "The Old Mill, Riverside Lane"),
    },
    "outdoors": {
        "head": ["Go Further", "Find Your Trail", "Wild Weekends", "Into the Hills",
                 "Pack light, walk far and come home with better stories",
                 "Three hundred kilometres of marked trails, one week, no signal"],
        "sub": ["Guided hikes for every level.", "Book before Friday and save 15%.",
                "Small groups, expert local guides and cabins with a view, all included in one simple price."],
        "cta": ["Book a trip", "Plan your route", "Join a hike", "Get the guide"],
        "kicker": ["SPRING ESCAPES", "LIMITED PLACES", "NEW ROUTES"],
        "badge": ["-15%", "NEW", "GUIDED", "ALL LEVELS"],
        "event": ("Mountain Film Night", "FRI 3 OCT · DOORS 7 PM", "Alpine Hall, 12 Station Road"),
    },
    "eco": {
        "head": ["Grow Green", "Less, but Better", "Plant a Future",
                 "Small changes at home add up to a big change for the planet",
                 "Join ten thousand neighbours planting trees this spring",
                 "Refill, reuse and repair: a practical guide to a lower-waste kitchen"],
        "sub": ["Every order plants a tree.", "Volunteers welcome, no experience needed.",
                "Bring a friend, a pair of gloves and a reusable bottle. We'll bring the saplings, tools and lunch."],
        "cta": ["Get involved", "Sign up", "Learn more", "Donate today"],
        "kicker": ["EARTH WEEK", "COMMUNITY", "VOLUNTEER"],
        "badge": ["FREE", "ZERO WASTE", "LOCAL", "100% RECYCLED"],
        "event": ("Community Planting Day", "SUN 22 MAR · 9 AM", "Northside Park, Gate C"),
    },
    "coffee": {
        "head": ["Fresh Roast", "Morning Ritual", "Brewed Slow",
                 "Single-origin beans, roasted every Tuesday in small batches",
                 "Your new favourite cup is one subscription away"],
        "sub": ["Free grinder with every annual plan.", "Roasted to order, shipped within 48 hours.",
                "Tasting notes of dark cherry, cocoa and orange blossom, from a family farm in Huila."],
        "cta": ["Order now", "Subscribe", "Taste the range", "Find a café"],
        "kicker": ["HOUSE BLEND", "NEW ORIGIN", "LIMITED LOT"],
        "badge": ["NEW", "-20%", "FAIR TRADE", "2 FOR 1"],
        "event": ("Latte Art Throwdown", "THU 9 APR · 6:30 PM", "Grindhouse Roastery, Dock Street"),
    },
    "books": {
        "head": ["Read More", "Stories for Winter", "The Big Book Sale",
                 "Lose yourself in the best new fiction of the year",
                 "Signed first editions, rare finds and a reading corner with free tea"],
        "sub": ["Members save 10% every day.", "Open late on Thursdays.",
                "Meet the author, hear a reading from the new novel and get your copy signed on the night."],
        "cta": ["Browse books", "Reserve a seat", "Join the club", "Shop the sale"],
        "kicker": ["AUTHOR EVENT", "STAFF PICKS", "NEW RELEASES"],
        "badge": ["SIGNED", "-30%", "NEW", "BOOK OF THE MONTH"],
        "event": ("An Evening with Mara Quill", "WED 18 FEB · 7 PM", "Lantern Books, 4 Market Square"),
    },
}

LEGAL = ["Terms apply. Offer valid while stocks last.",
         "Prices include VAT. Free delivery applies to mainland addresses only. See website for full terms.",
         "© 2026 Solstice Co. All rights reserved.",
         "Offer ends 31 December 2026. Cannot be combined with any other promotion or discount code."]

# (name, background, text, soft text, accent, text on accent)
LIGHT = [("cream", (250, 244, 234), (38, 30, 44), (70, 60, 76), (214, 92, 52), (255, 255, 255)),
         ("mint", (226, 242, 234), (20, 54, 44), (44, 84, 72), (24, 120, 96), (255, 255, 255)),
         ("sky", (232, 240, 252), (22, 36, 72), (52, 66, 102), (40, 92, 220), (255, 255, 255)),
         ("blush", (252, 232, 232), (66, 24, 40), (96, 52, 66), (200, 48, 92), (255, 255, 255))]
DARK = [("ink", (24, 22, 36), (255, 255, 255), (220, 216, 230), (246, 176, 64), (24, 22, 36)),
        ("forest", (18, 52, 40), (250, 246, 236), (214, 226, 214), (240, 196, 92), (18, 52, 40)),
        ("plum", (58, 22, 60), (255, 255, 255), (232, 214, 236), (255, 120, 90), (255, 255, 255)),
        ("navy", (16, 30, 64), (255, 255, 255), (214, 222, 240), (255, 92, 92), (255, 255, 255))]
SALE = [((255, 214, 64), (30, 24, 20), (220, 40, 60)), ((232, 52, 72), (255, 255, 255), (255, 214, 64)),
        ((30, 30, 34), (255, 255, 255), (255, 88, 64)), ((88, 60, 220), (255, 255, 255), (255, 214, 64))]


def rgba(c, a: float = 1.0) -> str:
    return f"rgba({c[0]},{c[1]},{c[2]},{a:g})"


class Copy:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.domain = rng.choice(sorted(COPY))
        self.d = COPY[self.domain]
        r = rng.random()  # copy length: a third short, a third medium, a third long
        heads = self.d["head"]
        short = [h for h in heads if len(h) <= 22]
        long_ = [h for h in heads if len(h) > 22]
        self.head = rng.choice(short if r < 0.4 else long_)
        self.sub = rng.choice(self.d["sub"])
        self.cta = rng.choice(self.d["cta"])
        self.kicker = rng.choice(self.d["kicker"])
        self.badges = rng.sample(self.d["badge"], min(3, len(self.d["badge"])))
        self.legal = rng.choice(LEGAL)
        self.event = self.d["event"]


# ============================================================================ builder


class B:
    """Collects elements in paint order; text boxes are sized to the renderer's own wrap."""

    def __init__(self, W: int, H: int, rng: random.Random, s: float = 1.0):
        self.W, self.H, self.rng, self.s, self.els, self.z = W, H, rng, s, [], 0

    def add(self, el: dict) -> dict:
        self.z += 1
        el["z_index"] = self.z
        for k in ("x", "y", "width", "height"):
            el[k] = round(float(el[k]), 1)
        self.els.append(el)
        return el

    @staticmethod
    def _style(content, size, family, bold, **kw):
        el = {"content": content, "font_family": family, "font_size": int(size), **kw}
        if bold:
            el["font_weight"] = "bold"
        return el

    def measure(self, content, size, family, bold, width, **kw):
        from engine.smart.textfit import TextStyle, measure

        el = self._style(content, size, family, bold, **kw)
        return measure(content, TextStyle.of(el), int(size), width, exact=True)

    def text_h(self, content, size, family, bold, width, **kw) -> float:
        blk = self.measure(content, size, family, bold, width, **kw)
        return math.ceil(max(blk.height, blk.n_lines * blk.line_step)) + 2

    def ink_w(self, content, size, family, bold, **kw) -> float:
        blk = self.measure(content, size, family, bold, 1e6, **kw)
        return max(blk.line_widths) if blk.line_widths else 0.0

    def text(self, id, x, y, w, content, size, color, family=SANS, bold=False, align="left", **kw) -> dict:
        size = max(1, int(round(size)))
        h = self.text_h(content, size, family, bold, w, **kw)
        el = {"id": id, "type": "text", "x": x, "y": y, "width": w, "height": h, **self._style(content, size, family, bold),
              "color": rgba(color) if isinstance(color, tuple) else color, "text_align": align, **kw}
        return self.add(el)

    def rect(self, id, x, y, w, h, fill, radius=0.0, shape="rectangle", **kw) -> dict:
        el = {"id": id, "type": "shape", "x": x, "y": y, "width": w, "height": h, "shape_type": shape,
              "fill_color": rgba(fill) if isinstance(fill, tuple) else fill, **kw}
        if radius:
            el["corner_radius"] = round(radius, 1)
        return self.add(el)

    def img(self, id, x, y, w, h, asset, fit="contain", **kw) -> dict:
        return self.add({"id": id, "type": "image", "x": x, "y": y, "width": w, "height": h, "asset_id": asset,
                         "fit": fit, **kw})

    def img_fit(self, id, x, y, maxw, maxh, asset, ax=0.5, ay=0.5, **kw) -> dict:
        """An image box with the asset's own aspect, as large as fits maxw x maxh, anchored."""
        iw, ih = asset_size(asset)
        k = min(maxw / iw, maxh / ih)
        w, h = iw * k, ih * k
        return self.img(id, x + (maxw - w) * ax, y + (maxh - h) * ay, w, h, asset, **kw)

    def button(self, gid, x, y, label, size, fill, fg, align="left", w=None, pill=None, ids=None) -> Tuple[dict, dict]:
        size = int(round(size))
        h = round(size * self.rng.uniform(2.2, 2.7))
        w = w or max(h * 2.4, self.ink_w(label, size, SANS, True) + 2 * size * self.rng.uniform(1.0, 1.5))
        if align == "center":
            x -= w / 2
        elif align == "right":
            x -= w
        pill = self.rng.random() < 0.5 if pill is None else pill
        bid, lid = ids or (f"{gid}_bg", f"{gid}_label")
        bg = self.rect(bid, x, y, w, h, fill, radius=h / 2 if pill else size * 0.3, group_id=gid)
        th = self.text_h(label, size, SANS, True, w)
        lab = self.text(lid, x, y + (h - th) / 2, w, label, size, fg, bold=True, align="center", group_id=gid)
        return bg, lab

    def cta(self, x, y, label, size, fill, fg, align="left", **kw):
        return self.button("cta", x, y, label, size, fill, fg, align=align, ids=("cta_bg", "cta_label"), **kw)


def head_size(W: int, H: int, content: str, s: float = 1.0) -> float:
    n = len(content)
    k = 0.17 if n <= 12 else 0.135 if n <= 22 else 0.095 if n <= 45 else 0.076 if n <= 70 else 0.064
    short = min(W, H)
    if W > 1.4 * H:  # landscape copy sits in a column, roughly half the width: wrap less, set larger
        short = min(1.15 * H, 0.62 * W)
    return k * short * s


class Column:
    """A vertical stack of parts placed as one block: (kind, id, payload, gap_after)."""

    def __init__(self, b: B, w: float, align: str):
        self.b, self.w, self.align, self.parts = b, w, align, []

    def text(self, id, content, size, color, family=SANS, bold=False, gap=0.5, **kw):
        self.parts.append(("text", id, dict(content=content, size=int(round(size)), color=color, family=family,
                                            bold=bold, **kw), gap))
        return self

    def cta(self, label, size, fill, fg, gap=0.5):
        self.parts.append(("cta", "cta", dict(label=label, size=int(round(size)), fill=fill, fg=fg), gap))
        return self

    def logo(self, h, gap=0.6):
        self.parts.append(("logo", "logo", dict(h=h), gap))
        return self

    def _h(self, p) -> float:
        kind, _, d, _ = p
        if kind == "text":
            extra = {k: v for k, v in d.items() if k not in ("content", "size", "color", "family", "bold")}
            return self.b.text_h(d["content"], d["size"], d["family"], d["bold"], self.w, **extra)
        if kind == "cta":
            return d["size"] * 2.5
        iw, ih = asset_size("solstice_logo_lockup.png")
        return min(d["h"], self.w * ih / iw)

    def height(self) -> float:
        hs = [self._h(p) for p in self.parts]
        gaps = [p[3] * (p[2].get("size", 30) if p[0] != "logo" else p[2]["h"]) for p in self.parts[:-1]]
        return sum(hs) + sum(gaps)

    def place(self, x: float, y: float) -> float:
        b = self.b
        ax = {"left": 0.0, "center": 0.5, "right": 1.0}[self.align]
        for k, p in enumerate(self.parts):
            kind, id, d, gap = p
            if kind == "text":
                extra = {kk: v for kk, v in d.items() if kk not in ("content", "size", "color", "family", "bold")}
                el = b.text(id, x, y, self.w, d["content"], d["size"], d["color"], family=d["family"],
                            bold=d["bold"], align=self.align, **extra)
                y += el["height"]
            elif kind == "cta":
                bg, _ = b.cta(x + ax * self.w, y, d["label"], d["size"], d["fill"], d["fg"], align=self.align)
                y += bg["height"]
            else:
                h = self._h(p)
                el = b.img_fit("logo", x, y, self.w * 0.5, h, "solstice_logo_lockup.png", ax=ax, ay=0.0)
                y += el["height"]
            if k < len(self.parts) - 1:
                y += gap * (d.get("size", 30) if kind != "logo" else d["h"])
        return y

    def place_in(self, x: float, top: float, bottom: float, va: float = 0.5) -> float:
        h = self.height()
        return self.place(x, top + max(0.0, (bottom - top - h)) * va)


def _theme(rng, dark=None):
    dark = rng.random() < 0.5 if dark is None else dark
    name, bg, fg, soft, acc, on = rng.choice(DARK if dark else LIGHT)
    return {"dark": dark, "bg": bg, "fg": fg, "soft": soft, "acc": acc, "on": on, "name": name}


def _landscape(W, H) -> bool:
    return W > 1.15 * H


# ============================================================================ families


def fam_photo_person(b: B, c: Copy, rng) -> str:
    """1. full-bleed photo (with a person cut-out on the busy side), copy on the quiet side."""
    W, H, s = b.W, b.H, b.s
    photo = rng.choice(kit_assets("photo"))
    b.img("background_photo", 0, 0, W, H, photo, fit="cover")
    land = _landscape(W, H)
    side, _, _ = quiet_side(photo, W, H, horizontal=land)
    m = rng.uniform(0.05, 0.08) * min(W, H)
    if land:
        rect = (0, 0, W * 0.5, H) if side == "left" else (W * 0.5, 0, W, H)
    else:
        rect = (0, 0, W, H * 0.5) if side == "top" else (0, H * 0.5, W, H)
    lum, busy = region_stats(photo, W, H, *rect)
    scrim = busy > 0.035 and rng.random() < 0.7  # a designer darkens a busy quiet side
    if scrim:
        b.rect("band_scrim", *rect[:2], rect[2] - rect[0], rect[3] - rect[1],
               rgba((16, 14, 22), round(rng.uniform(0.5, 0.65), 2)))
        lum = lum * 0.4
    dark_text = lum > 0.5
    fg = (30, 26, 36) if dark_text else (255, 255, 255)
    acc = rng.choice([(214, 92, 52), (40, 92, 220), (24, 120, 96)]) if dark_text else rng.choice(
        [(246, 176, 64), (255, 255, 255), (232, 90, 70)])
    on = (255, 255, 255) if acc != (255, 255, 255) and acc != (246, 176, 64) else (30, 26, 36)
    persons = kit_assets("person")
    has_person = persons and rng.random() < 0.75
    if land:
        cw = W * rng.uniform(0.4, 0.48) - m
        cx = m if side == "left" else W - m - cw
        px0, px1 = (W * 0.52, W - m * 0.5) if side == "left" else (m * 0.5, W * 0.48)
        ptop, pbot = H * 0.12, H
    else:
        cw = W - 2 * m
        cx = m
        px0, px1 = W * 0.15, W * 0.85
        ptop, pbot = (H * 0.5, H) if side == "top" else (0.0, H * 0.5)
    if has_person:
        b.img_fit("person", px0, ptop, px1 - px0, pbot - ptop, rng.choice(persons), ax=0.5, ay=1.0 if side == "top" or land else 0.0)
    align = rng.choice(["left", "center"]) if land else "center"
    col = Column(b, cw, align)
    hs = head_size(W, H, c.head, s)
    if rng.random() < 0.5:
        col.text("kicker", c.kicker, hs * 0.28, acc if acc != (255, 255, 255) else fg, bold=True, gap=0.6,
                 letter_spacing=2)
    col.text("headline", c.head, hs, fg, family=rng.choice([SERIF, SANS]), bold=True, gap=0.35)
    if rng.random() < 0.7:
        col.text("subhead", c.sub, max(hs * 0.32, 18 * s), fg, gap=0.9)
    col.cta(c.cta, max(hs * 0.28, 18 * s), acc, on, gap=0.9)
    if rng.random() < 0.6:
        col.logo(min(W, H) * 0.128)
    if land:
        col.place_in(cx, m, H - m, va=rng.choice([0.5, 0.35]))
    elif side == "top":
        col.place_in(cx, m, H * 0.5, va=0.3)
    else:
        col.place_in(cx, H * 0.5, H - m, va=0.7)
    return "photo_person"


def fam_split_panel(b: B, c: Copy, rng) -> str:
    """2. half photo, half colour panel holding the copy."""
    W, H, s = b.W, b.H, b.s
    t = _theme(rng)
    photo = rng.choice(kit_assets("photo"))
    land = _landscape(W, H)
    frac = rng.uniform(0.45, 0.58)
    first = rng.random() < 0.5
    if land:
        pw = W * frac
        px = 0 if first else W - pw
        b.img("photo", px, 0, pw, H, photo, fit="cover")
        b.rect("band_panel", pw if first else 0, 0, W - pw, H, t["bg"])
        x0, x1, y0, y1 = (pw if first else 0), (W if first else W - pw), 0, H
    else:
        ph = H * frac
        py = 0 if first else H - ph
        b.img("photo", 0, py, W, ph, photo, fit="cover")
        b.rect("band_panel", 0, ph if first else 0, W, H - ph, t["bg"])
        x0, x1, y0, y1 = 0, W, (ph if first else 0), (H if first else H - ph)
    m = rng.uniform(0.07, 0.1) * min(x1 - x0, y1 - y0) + 0.03 * min(W, H)
    align = rng.choice(["left", "center"])
    col = Column(b, x1 - x0 - 2 * m, align)
    hs = head_size(W, H, c.head, s) * (0.85 if land else 0.8)
    if rng.random() < 0.6:
        col.logo(min(W, H) * 0.112)
    col.text("headline", c.head, hs, t["fg"], family=rng.choice([SERIF, SANS]), bold=rng.random() < 0.7, gap=0.4)
    col.text("subhead", c.sub, max(hs * 0.34, 18 * s), t["soft"], gap=0.9)
    col.cta(c.cta, max(hs * 0.3, 18 * s), t["acc"], t["on"], gap=0.8)
    if rng.random() < 0.35:
        col.text("legal", c.legal, max(hs * 0.18, 13 * s), t["soft"], gap=0)
    col.place_in(x0 + m, y0 + m, y1 - m, va=0.5)
    return "split_panel"


def fam_products(b: B, c: Copy, rng) -> str:
    """3. product cut-outs on the gradient, round badges and a price lockup ("$24" + ".99")."""
    W, H, s = b.W, b.H, b.s
    b.img("background", 0, 0, W, H, "solstice_bg_gradient.png", fit="cover")
    fg, acc, on = (255, 255, 255), rng.choice([(43, 27, 46), (232, 106, 58), (255, 255, 255)]), None
    on = (255, 255, 255) if acc != (255, 255, 255) else (43, 27, 46)
    land = _landscape(W, H)
    m = 0.06 * min(W, H)
    n = rng.choice([1, 2, 3])
    if land:
        zone = (m, m, W * 0.5 - m, H - 2 * m)
        cx0, cw = W * 0.5, W * 0.5 - m
    else:
        zone = (m, H * 0.22, W - 2 * m, H * 0.42)
        cx0, cw = m, W - 2 * m
    zx, zy, zw, zh = zone
    if n == 1:
        b.img_fit("product", zx, zy, zw, zh, "solstice_bottle.png")
    else:
        side = zh * 0.78
        offs = [-1, 1] if n == 2 else [-1, 1, 0]
        for k, o in enumerate(offs):
            sz = zh if o == 0 or n == 2 else side
            sz = min(sz, zw / (1.6 if n == 2 else 1.9))
            x = zx + zw / 2 - sz / 2 + o * zw * (0.22 if n == 2 else 0.3)
            y = zy + (zh - sz)
            pid = "product" if (o == 0 or (n == 2 and k == 0)) else f"product_{k + 1}"
            b.img(pid, x, y, sz, sz, "solstice_bottle.png")
        # the centre bottle paints last (on top), ids stay unique
    # round badge sticker(s)
    nb = rng.choice([1, 2])
    for k in range(nb):
        d = min(W, H) * rng.uniform(0.15, 0.2) * (1.0 if not land else 0.9)
        bx = zx + (zw - d if k == 0 else 0) + rng.uniform(-0.02, 0.02) * W
        by = zy + (0 if k == 0 else zh * 0.55)
        gid = f"badge{k + 1}"
        col = rng.choice([(220, 40, 60), (246, 176, 64), (255, 255, 255)])
        tc = (43, 27, 46) if col != (220, 40, 60) else (255, 255, 255)
        b.rect(f"{gid}_bg", bx, by, d, d, col, shape="ellipse", group_id=gid)
        label = c.badges[k] if len(c.badges[k]) <= 8 else c.badges[k].split()[0]
        fs = d * (0.26 if len(label) <= 4 else 0.17)
        th = b.text_h(label, fs, SANS, True, d * 0.8)
        b.text(f"{gid}_label", bx + d * 0.1, by + (d - th) / 2, d * 0.8, label, fs, tc, bold=True, align="center",
               group_id=gid)
    hs = head_size(W, H, c.head, s) * 0.85
    if land:
        col = Column(b, cw - m, "left")
    else:
        col = Column(b, cw, "center")
    col.logo(min(W, H) * 0.112, gap=0.5)
    col.text("headline", c.head, hs, fg, family=SERIF, bold=True, gap=0.3)
    if rng.random() < 0.5:
        col.text("subhead", c.sub, max(hs * 0.32, 18 * s), (255, 240, 230), gap=0.4)
    if land:
        y = col.place_in(cx0, m, H * 0.62, va=0.6)
        py = y + 0.04 * H
        px, align = cx0, "left"
    else:
        col.place_in(cx0, m, H * 0.22 - 0.01 * H, va=0.5)
        py = H * 0.66
        px, align = None, "center"
    # price lockup: "$24" big and ".99" raised beside it
    ps = min(W, H) * rng.uniform(0.11, 0.14) * s
    main, cents = rng.choice([("$24", ".99"), ("$9", ".50"), ("€39", ".95"), ("$129", ".00")])
    mw = b.ink_w(main, ps, SANS, True) + 4
    cw2 = b.ink_w(cents, ps * 0.45, SANS, True) + 4
    if px is None:
        px = (W - mw - cw2) / 2
    b.text("price_main", px, py, mw, main, ps, fg, bold=True, group_id="price")
    b.text("price_cents", px + mw + 2, py + ps * 0.12, cw2, cents, ps * 0.45, fg, bold=True, group_id="price")
    ph = b.text_h(main, ps, SANS, True, mw)
    by = py + ph + 0.025 * H
    b.cta(px if align == "left" else W / 2, by, c.cta, max(ps * 0.26, 18 * s), acc, on, align=align)
    return "products"


def fam_collage(b: B, c: Copy, rng) -> str:
    """4. three or four photos in a collage, with a headline and a CTA."""
    W, H, s = b.W, b.H, b.s
    t = _theme(rng)
    b.rect("background", 0, 0, W, H, t["bg"])
    photos = rng.sample(kit_assets("photo"), 4)
    n = rng.choice([3, 4])
    g = 0.015 * min(W, H) * rng.choice([0, 1, 1.5])
    m = 0.05 * min(W, H)
    land = _landscape(W, H)
    if land:
        gx0, gy0, gw, gh = m, m, W * 0.58 - m, H - 2 * m
        cx0, cy0, cw, ch = W * 0.58 + m, m, W * 0.42 - 2 * m, H - 2 * m
    else:
        gx0, gy0, gw, gh = m, m, W - 2 * m, H * 0.6 - m
        cx0, cy0, cw, ch = m, H * 0.6 + m * 0.6, W - 2 * m, H * 0.4 - 1.6 * m
    cells = []
    if n == 3 and rng.random() < 0.5:  # one big + two stacked
        bw = gw * 0.6
        cells = [(gx0, gy0, bw - g / 2, gh), (gx0 + bw + g / 2, gy0, gw - bw - g / 2, gh / 2 - g / 2),
                 (gx0 + bw + g / 2, gy0 + gh / 2 + g / 2, gw - bw - g / 2, gh / 2 - g / 2)]
    elif n == 3:  # a row of three
        cwid = (gw - 2 * g) / 3
        cells = [(gx0 + k * (cwid + g), gy0, cwid, gh) for k in range(3)]
    else:  # 2 x 2
        cwid, chei = (gw - g) / 2, (gh - g) / 2
        cells = [(gx0 + i * (cwid + g), gy0 + j * (chei + g), cwid, chei) for j in range(2) for i in range(2)]
    for k, (x, y, w, h) in enumerate(cells):
        b.img(f"photo_{k + 1}", x, y, w, h, photos[k], fit="cover")
    align = "left" if land else rng.choice(["left", "center"])
    col = Column(b, cw, align)
    hs = head_size(W, H, c.head, s) * (0.75 if land else 0.7)
    col.text("headline", c.head, hs, t["fg"], family=rng.choice([SERIF, SANS]), bold=True, gap=0.35)
    if rng.random() < 0.5:
        col.text("subhead", c.sub, max(hs * 0.36, 18 * s), t["soft"], gap=0.8)
    col.cta(c.cta, max(hs * 0.32, 18 * s), t["acc"], t["on"], gap=0.7)
    if rng.random() < 0.5:
        col.logo(min(W, H) * 0.096)
    col.place_in(cx0, cy0, cy0 + ch, va=0.5)
    return "collage"


def fam_event(b: B, c: Copy, rng) -> str:
    """5. an event poster: long headline, date and venue lines, logo, legal."""
    W, H, s = b.W, b.H, b.s
    kind = rng.choice(["photo", "gradient", "solid"])
    t = _theme(rng, dark=True if kind != "solid" else None)
    if kind == "photo":
        b.img("background_photo", 0, 0, W, H, rng.choice(kit_assets("photo")), fit="cover")
        b.rect("band_scrim", 0, 0, W, H, rgba(t["bg"], round(rng.uniform(0.55, 0.7), 2)))
    elif kind == "gradient":
        b.img("background", 0, 0, W, H, "solstice_bg_gradient.png", fit="cover")
        t["acc"] = (255, 236, 200)
    else:
        b.rect("background", 0, 0, W, H, t["bg"])
    title, date, venue = c.event
    head = title if rng.random() < 0.4 else f"{title}: {c.head}" if len(c.head) < 40 else c.head
    land = _landscape(W, H)
    m = 0.07 * min(W, H)
    align = "left" if land else rng.choice(["left", "center"])
    cw = (W * 0.62 if land else W) - 2 * m
    col = Column(b, cw, align)
    hs = head_size(W, H, head, s)
    if rng.random() < 0.6:
        col.text("kicker", c.kicker, max(hs * 0.25, 16 * s), t["acc"], bold=True, gap=0.7, letter_spacing=3)
    col.text("headline", head, hs, t["fg"], family=SERIF, bold=rng.random() < 0.6, gap=0.5)
    col.text("date_line", date, max(hs * 0.34, 18 * s), t["acc"], bold=True, gap=0.3)
    col.text("venue_line", venue, max(hs * 0.28, 16 * s), t["soft"], gap=0.6)
    col.place_in(m, m, H * 0.85, va=rng.choice([0.4, 0.55]))
    # logo and legal pinned to the foot
    lh = min(W, H) * 0.11
    ax = {"left": 0.0, "center": 0.5}[align]
    lg = b.img_fit("logo", m, H - m - lh, cw * 0.4, lh, "solstice_logo_lockup.png", ax=ax, ay=1.0)
    lw = cw if land is False else W - 2 * m
    if land:
        b.text("legal", W - m - W * 0.3, H - m - b.text_h(c.legal, max(hs * 0.15, 12 * s), SANS, False, W * 0.3),
               W * 0.3, c.legal, max(hs * 0.15, 12 * s), t["soft"], align="right")
    else:
        fs = max(hs * 0.15, 12 * s)
        th = b.text_h(c.legal, fs, SANS, False, lw)
        lg["y"] = round(H - m - th - lg["height"] - 0.015 * H, 1)
        b.text("legal", m, H - m - th, lw, c.legal, fs, t["soft"], align=align)
    if rng.random() < 0.5:
        b.rect("decor_rule", m if align == "left" else W / 2 - W * 0.1, H * 0.86 - m * 0.3, W * 0.2, max(2, 0.004 * H),
               t["acc"])
    return "event"


def fam_sale(b: B, c: Copy, rng) -> str:
    """6. a sale promo: big "50% OFF" lockup, round sticker, ribbon, CTA, decor bleeding off corners."""
    W, H, s = b.W, b.H, b.s
    bg, fg, acc = rng.choice(SALE)
    b.rect("background", 0, 0, W, H, bg)
    for k in range(rng.choice([2, 3])):
        d = min(W, H) * rng.uniform(0.35, 0.6)
        cx, cy = rng.choice([(0, 0), (W, 0), (0, H), (W, H)])
        col = tuple(min(255, int(v * 0.85 + a * 0.15)) for v, a in zip(bg, acc))
        b.rect(f"decor_{k + 1}", cx - d / 2, cy - d / 2, d, d, rgba(col, 0.9), shape="ellipse")
    land = _landscape(W, H)
    m = 0.08 * min(W, H)
    pct = rng.choice(["50%", "30%", "70%", "25%"])
    big = min(W * (0.3 if land else 0.42), H * 0.34) * s
    align = "left" if land else "center"
    x0 = m * 1.5 if land else 0
    cw = (W * 0.58 if land else W) - (m * 1.5 if land else 0)
    pw = b.ink_w(pct, big, SANS, True) + 6
    off_fs = big * 0.42
    ow = b.ink_w("OFF", off_fs, SANS, True) + 6
    kick_fs = max(big * 0.16, 18 * s)
    kh = b.text_h("SUMMER SALE", kick_fs, SANS, True, cw)
    ph = b.text_h(pct, big, SANS, True, pw)
    sub = rng.choice(["On everything, this weekend only", "Selected styles, while stocks last",
                      "Our biggest sale of the year starts now. Online and in every store."])
    sub_fs = max(big * 0.13, 18 * s)
    sw = min(cw - (0 if land else 2 * m), W - 2 * m)
    sh = b.text_h(sub, sub_fs, SANS, False, sw)
    cta_fs = max(big * 0.12, 18 * s)
    total = kh + 0.2 * kick_fs + ph + 0.2 * big + sh + cta_fs * 3.4
    y = max(m, (H - total) / 2)
    kx = x0 if land else (W - sw) / 2
    b.text("kicker", kx, y, sw, rng.choice(["SUMMER SALE", "FLASH SALE", "END OF SEASON"]), kick_fs, fg, bold=True,
           align=align, letter_spacing=3)
    y += kh + 0.2 * kick_fs
    lx = x0 if land else (W - pw - ow) / 2
    b.text("headline", lx, y, pw, pct, big, fg, bold=True, group_id="deal")
    b.text("deal_off", lx + pw, y + ph - b.text_h("OFF", off_fs, SANS, True, ow) - big * 0.08, ow, "OFF", off_fs, acc,
           bold=True, group_id="deal")
    y += ph + 0.2 * big
    b.text("subhead", kx, y, sw, sub, sub_fs, fg, align=align)
    y += sh + cta_fs
    b.cta(x0 if land else W / 2, y, rng.choice(["Shop the sale", "Shop now", "Grab the deal"]), cta_fs, acc,
          (255, 255, 255) if acc != (255, 214, 64) else (30, 24, 20), align=align)
    # round sticker
    d = min(W, H) * rng.uniform(0.17, 0.22)
    sx = W - m - d if not land else W * 0.72
    sy = H * 0.6 if not land else H * 0.5 - d / 2
    if not land:
        sx, sy = rng.choice([(W - m * 0.6 - d, H * 0.08 + 1.1 * m), (m * 0.6, H - d - m)])
    rot = rng.choice([0, -12, 10])
    b.rect("sticker_bg", sx, sy, d, d, (255, 255, 255) if bg != (255, 255, 255) else acc, shape="ellipse",
           group_id="sticker", rotation=rot)
    label = rng.choice(["NEW IN", "FREE SHIPPING", "TODAY ONLY", "EXTRA 10%"])
    fs = d * 0.15
    th = b.text_h(label, fs, SANS, True, d * 0.74)
    b.text("sticker_label", sx + d * 0.13, sy + (d - th) / 2, d * 0.74, label, fs, (30, 24, 20), bold=True,
           align="center", group_id="sticker", rotation=rot)
    # ribbon across the top-left corner
    rw, rh = min(W, H) * 0.42, min(W, H) * 0.075
    rx, ry = -rw * 0.12, min(W, H) * 0.06
    b.rect("ribbon_bg", rx, ry, rw, rh, acc, group_id="ribbon", rotation=-30)
    rfs = rh * 0.42
    rth = b.text_h("LIMITED", rfs, SANS, True, rw * 0.6)
    b.text("ribbon_label", rx + rw * 0.2, ry + (rh - rth) / 2, rw * 0.6, "LIMITED", rfs,
           (255, 255, 255) if acc != (255, 214, 64) else (30, 24, 20), bold=True, align="center", group_id="ribbon",
           rotation=-30)
    return "sale"


def fam_glass_card(b: B, c: Copy, rng) -> str:
    """7. a translucent card (a shape with alpha) over a photo, the copy inside the card."""
    W, H, s = b.W, b.H, b.s
    photo = rng.choice(kit_assets("photo"))
    b.img("background_photo", 0, 0, W, H, photo, fit="cover")
    dark = rng.random() < 0.5
    card = (20, 20, 28) if dark else (255, 255, 255)
    fg = (255, 255, 255) if dark else (30, 26, 36)
    acc = rng.choice([(246, 176, 64), (232, 90, 70)]) if dark else rng.choice([(214, 92, 52), (40, 92, 220)])
    on = (30, 26, 36) if acc == (246, 176, 64) else (255, 255, 255)
    land = _landscape(W, H)
    m = 0.07 * min(W, H)
    if land:
        cw, ch = W * rng.uniform(0.42, 0.5), H - 2 * m
        side, _, _ = quiet_side(photo, W, H, True)
        cx = m if side == "left" else W - m - cw
        cy = m
    else:
        cw, ch = W - 2 * m, H * rng.uniform(0.45, 0.55)
        cx = m
        cy = rng.choice([m, H - m - ch, (H - ch) / 2])
    b.rect("card", cx, cy, cw, ch, rgba(card, round(rng.uniform(0.62, 0.82), 2)), radius=0.04 * min(cw, ch))
    pad = 0.09 * min(cw, ch) + 0.02 * min(W, H)
    align = rng.choice(["left", "center"])
    col = Column(b, cw - 2 * pad, align)
    hs = head_size(W, H, c.head, s) * 0.75
    if rng.random() < 0.5:
        col.logo(min(W, H) * 0.096, gap=0.5)
    if rng.random() < 0.5:
        col.text("kicker", c.kicker, max(hs * 0.27, 16 * s), acc, bold=True, gap=0.6, letter_spacing=2)
    col.text("headline", c.head, hs, fg, family=rng.choice([SERIF, SANS]), bold=True, gap=0.4)
    col.text("subhead", c.sub, max(hs * 0.34, 18 * s), fg, gap=0.9)
    col.cta(c.cta, max(hs * 0.3, 18 * s), acc, on)
    col.place_in(cx + pad, cy + pad, cy + ch - pad, va=0.5)
    return "glass_card"


def fam_dense(b: B, c: Copy, rng) -> str:
    """8. Level-3 density: logo, product, headline, subhead, three badges, CTA, ribbon, a watermark
    whose asset is missing, and a legal line."""
    W, H, s = b.W, b.H, b.s
    if rng.random() < 0.6:
        b.img("background", 0, 0, W, H, "solstice_bg_gradient.png", fit="cover")
        fg, soft, acc, dark = (255, 255, 255), (255, 240, 232), (232, 106, 58), (43, 27, 46)
    else:
        t = _theme(rng, dark=False)
        b.rect("background", 0, 0, W, H, t["bg"])
        fg, soft, acc, dark = t["fg"], t["soft"], t["acc"], t["fg"]
    land = _landscape(W, H)
    m = 0.05 * min(W, H)
    head = c.head if len(c.head) > 30 else f"{c.head} — {COPY[c.domain]['head'][-1]}"
    hs = head_size(W, H, head, s) * 0.8
    if land:
        b.img_fit("product", m, H * 0.18, W * 0.34, H * 0.62, "solstice_bottle.png")
        cx, cw = W * 0.4, W * 0.6 - m
    else:
        b.img_fit("product", W * 0.3, H * 0.13, W * 0.4, H * 0.27, "solstice_bottle.png")
        cx, cw = m * 1.5, W - 3 * m
    lh = min(W, H) * 0.11
    b.img_fit("logo", cx if land else W / 2 - W * 0.15, m, W * 0.3 if not land else cw * 0.4, lh,
              "solstice_logo_lockup.png", ax=0.0 if land else 0.5, ay=0.0)
    col = Column(b, cw, "left" if land else "center")
    col.text("headline", head, hs, fg, family=SERIF, bold=True, gap=0.4)
    col.text("subhead", COPY[c.domain]["sub"][-1], max(hs * 0.5, 16 * s), soft, gap=0.6)
    top = (m + lh + 0.03 * H) if land else H * 0.42
    y = col.place(cx, top)
    # three badges in a row
    bfs = max(min(W, H) * 0.02, 13) * s
    bh = bfs * 2.6
    labels = [x.upper() for x in c.badges[:3]]
    widths = [b.ink_w(lab, bfs, SANS, True) + 2.4 * bfs for lab in labels]
    gap = 0.02 * W
    total = sum(widths) + gap * 2
    bx = cx if land else (W - total) / 2
    y += 0.025 * H
    for k, (lab, w) in enumerate(zip(labels, widths)):
        b.button(f"badge{k + 1}", bx, y, lab, bfs, acc if k != 1 else dark, (255, 255, 255), w=w, pill=True)
        bx += w + gap
    y += bh + 0.03 * H
    cfs = max(min(W, H) * 0.022, 16) * s
    b.cta(cx if land else W / 2, y, c.cta, cfs, acc, (255, 255, 255), align="left" if land else "center", pill=True)
    # ribbon, rotated, top-right
    rw, rh = min(W, H) * 0.3, min(W, H) * 0.055
    rx, ry = W - rw - m * 0.4, m * 0.7
    b.rect("ribbon_bg", rx, ry, rw, rh, (220, 60, 50), group_id="ribbon", rotation=-12)
    rfs = rh * 0.42
    rth = b.text_h("SUNSET SALE", rfs, SANS, True, rw)
    b.text("ribbon_label", rx, ry + (rh - rth) / 2, rw, rng.choice(["SUNSET SALE", "HOT DEAL", "NEW"]), rfs,
           (255, 255, 255), bold=True, align="center", group_id="ribbon", rotation=-12)
    # watermark whose asset is missing, legal at the foot
    b.img("watermark", W - m - min(W, H) * 0.16, H - m - min(W, H) * 0.09 - 0.04 * H, min(W, H) * 0.16,
          min(W, H) * 0.075, MISSING_ASSET)
    lfs = max(min(W, H) * 0.011, 11) * s
    lth = b.text_h(c.legal, lfs, SANS, False, W - 2 * m)
    b.text("legal_text", m, H - m * 0.6 - lth, W - 2 * m, c.legal, lfs, soft, align="center")
    return "dense"


FAMILIES = [fam_photo_person, fam_split_panel, fam_products, fam_collage, fam_event, fam_sale, fam_glass_card,
            fam_dense]
FAMILY_NAMES = ["photo_person", "split_panel", "products", "collage", "event", "sale", "glass_card", "dense"]


def _source_ok(d: dict) -> List[str]:
    """A generated source must itself be clean: copy inside the canvas, no copy over copy,
    labels inside their shapes, no element so small it vanishes."""
    from engine.smart.geom import el_box
    from engine.smart.textfit import drawn_box

    W, H = d["canvas"]["width"], d["canvas"]["height"]
    errs = []
    texts = [e for e in d["elements"] if e["type"] == "text"]
    boxes = {}
    for e in texts:
        db = drawn_box(e)
        boxes[e["id"]] = db
        if not e.get("rotation") and (db.x < -0.5 or db.y < -0.5 or db.r > W + 0.5 or db.b > H + 0.5):
            errs.append(f"{e['id']} off canvas")
        if e["y"] + e["height"] > H + 0.5 or e["x"] + e["width"] > W + 0.5:
            errs.append(f"{e['id']} box off canvas")
    for i, a in enumerate(texts):
        for t in texts[i + 1:]:
            if a.get("group_id") and a.get("group_id") == t.get("group_id"):
                continue
            A, Bx = boxes[a["id"]], boxes[t["id"]]
            if A.x < Bx.r and Bx.x < A.r and A.y < Bx.b and Bx.y < A.b:
                errs.append(f"{a['id']} x {t['id']}")
    for e in d["elements"]:
        if e["type"] != "text" and e["id"].startswith(("logo", "product", "cta", "badge", "person")):
            b = el_box(e)
            if b.x < -0.5 or b.y < -0.5 or b.r > W + 0.5 or b.b > H + 0.5:
                errs.append(f"{e['id']} off canvas")
    return errs


def build(family: int, seed: int, fmt: str) -> dict:
    W, H = TIER1[fmt]
    errs: List[str] = []
    for s in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        rng = random.Random(f"unseen|{family}|{seed}|{fmt}")
        b = B(W, H, rng, s)
        c = Copy(rng)
        FAMILIES[family](b, c, rng)
        d = {"canvas": {"width": W, "height": H}, "elements": b.els}
        errs = _source_ok(d)
        if not errs:
            d["_scale"] = s
            return d
    raise RuntimeError(f"family {FAMILY_NAMES[family]} seed {seed} {fmt}: no clean source ({errs[:3]})")


def generated(split: str) -> List[dict]:
    """The generated designs of a split, each with its targets."""
    assert split in SEEDS, split
    ensure_kit()
    out = []
    for seed in SEEDS[split]:
        for f in range(len(FAMILIES)):
            rng = random.Random(f"fmt|{f}|{seed}")
            fmt = rng.choice(sorted(TIER1))
            src = build(f, seed, fmt)
            scale = src.pop("_scale", 1.0)
            targets = [(k, {"width": w, "height": h}) for k, (w, h) in list(TIER1.items()) + list(EXTRA.items())
                       if k != fmt]
            out.append({"name": f"{FAMILY_NAMES[f]}-s{seed}-{fmt}", "family": FAMILY_NAMES[f], "source": src,
                        "targets": targets, "kit": str(KIT), "scale": scale})
    return out


def crello(split: str) -> List[dict]:
    """Jeff's Crello pairs: every member a source; targets = the other designer size + tier-1."""
    sub = split.split("-", 1)[1]
    base = JEFF / sub
    pairs = json.loads((base / "index.json").read_text())
    out = []
    for pi, fam in enumerate(pairs):
        for m in fam:
            src = json.loads((base / "designs" / m["file"]).read_text())
            W, H = src["canvas"]["width"], src["canvas"]["height"]
            targets = [(f"designer-{o['w']}x{o['h']}", {"width": o["w"], "height": o["h"]}) for o in fam if o is not m]
            for k, (w, h) in TIER1.items():
                if (w, h) != (W, H) and all(t[1] != {"width": w, "height": h} for t in targets):
                    targets.append((k, {"width": w, "height": h}))
            out.append({"name": f"{Path(m['file']).stem}", "family": f"pair{pi}", "source": src, "targets": targets,
                        "kit": str(base)})
    return out


def designs_for(split: str) -> List[dict]:
    if split not in SPLITS:
        raise SystemExit(f"unknown split {split}")
    return crello(split) if split.startswith("crello") else generated(split)


# ============================================================================ checks


def _shown(e: dict) -> bool:
    return float(e.get("opacity", 1) if e.get("opacity") is not None else 1) > 0


def _paint(els: List[dict]) -> List[dict]:
    return [e for _, e in sorted(enumerate(els), key=lambda t: (float(t[1].get("z_index", 0) or 0), t[0]))]


def headline_id(src: dict) -> Optional[str]:
    best, key = None, -1.0
    for e in src["elements"]:
        if e.get("type") == "text" and str(e.get("content", "")).strip() and _shown(e):
            k = float(e.get("font_size", 0) or 0) * (1.15 if str(e.get("font_weight", "")) == "bold" else 1.0)
            if k > key:
                best, key = e["id"], k
    return best


def check_c1(src, out, tgt, rec) -> List[str]:
    from tools.rules import hard_rule_errors, promise_errors

    if rec.get("error"):
        return [f"crash: {rec['error'][:80]}"]
    errs = []
    if rec.get("fell_back"):
        errs.append("FALLBACK")
    errs += hard_rule_errors(src, out, tgt)
    errs += promise_errors(src, out, tgt)
    return errs


def check_c2(src, out, tgt, kit: Path) -> List[str]:
    from tools.gate.visible import hidden_elements

    try:
        return hidden_elements(src, out, tgt, assets=kit / "assets")
    except TypeError:  # an older visible.py without the asset fixes
        return hidden_elements(src, out, tgt)


def check_c3(src, out, tgt) -> List[str]:
    from engine.smart.items import Ctx
    from tools.gate.tiers import tier_of

    W, H = tgt["width"], tgt["height"]
    floor = Ctx(W=W, H=H, W0=1, H0=1).judge_floor
    errs = []
    for e in out["elements"]:
        if e.get("type") == "text" and _shown(e) and str(e.get("content", "")).strip():
            if float(e.get("font_size", 0) or 0) < floor:
                errs.append(f"{e['id']}: {e.get('font_size')}px < floor {floor}px")
    hid = headline_id(src)
    if hid and tier_of(tgt) == 1:
        o = {e["id"]: e for e in out["elements"]}.get(hid)
        if o is None or not _shown(o):
            errs.append(f"headline {hid} not shown")
        else:
            cap = CAP_RATIO * float(o.get("font_size", 0) or 0) * 512.0 / max(W, H)
            if cap < 7.0:
                errs.append(f"headline {hid}: cap {cap:.1f}px < 7px at 512")
    return errs


def _ink_box(e: dict, kit: Path):
    """Where an element paints, in canvas px (text: glyphs; image: its non-transparent pixels)."""
    from engine.smart.geom import Box, el_box
    from engine.smart.textfit import drawn_box

    if e.get("type") == "text":
        return drawn_box(e)
    b = el_box(e)
    if e.get("type") == "image":
        from engine.smart.assets import ink_fraction

        f = ink_fraction(e.get("asset_id"), b.w, b.h, e.get("fit", "contain"))
        return Box(b.x + f[0] * b.w, b.y + f[1] * b.h, (f[2] - f[0]) * b.w, (f[3] - f[1]) * b.h)
    return b


def _inter(a, b) -> float:
    return max(0.0, min(a.r, b.r) - max(a.x, b.x)) * max(0.0, min(a.b, b.b) - max(a.y, b.y))


def texts_on_images(design: dict, kit: Path) -> List[str]:
    """Visible texts whose glyph box lies at least 30% over an image painted below them."""
    order = _paint(design["elements"])
    res = []
    for k, t in enumerate(order):
        if t.get("type") != "text" or not _shown(t) or not str(t.get("content", "")).strip():
            continue
        tb = _ink_box(t, kit)
        area = max(1e-6, tb.w * tb.h)
        for e in order[:k]:
            if e.get("type") == "image" and _shown(e) and _inter(tb, _ink_box(e, kit)) >= 0.3 * area:
                res.append(t["id"])
                break
    return res


C4_LIMIT = 1200  # long side the copy is measured at: lower re-wraps tight lines (glyph advances round up at small sizes)


def _downscale(design: dict, limit: int) -> dict:
    """tools.judge.terms.downscale, but font sizes stay fractional: rounding 9.85 px up to 10 px
    re-wraps a button label onto two lines that hang out of its button (seen on 10-25)."""
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    f = min(1.0, limit / max(W, H))
    if f >= 1.0:
        return design
    els = []
    for e in design["elements"]:
        e = dict(e)
        for k in ("x", "y", "width", "height"):
            e[k] = float(e.get(k, 0) or 0) * f
        if e.get("type") == "text":
            # small sizes round glyph advances, so a tightly fitted line can re-wrap: give it 8%
            # more room on the side its alignment grows towards (the glyphs stay where they were)
            grow = 0.08 * e["width"]
            al = e.get("text_align", "left")
            e["x"] -= grow / 2 if al == "center" else grow if al == "right" else 0.0
            e["width"] += grow
            e["font_size"] = max(0.5, float(e.get("font_size", 12) or 12) * f)
        for k in ("corner_radius", "stroke_width", "letter_spacing"):
            if e.get(k):
                e[k] = float(e[k]) * f
        els.append(e)
    return {"canvas": {"width": max(1, round(W * f)), "height": max(1, round(H * f))}, "elements": els}


def text_contrast(design: dict, limit: int = C4_LIMIT) -> Dict[str, dict]:
    """Per visible text: p10 WCAG contrast of glyph + halo against what is under it and the p90
    gradient there, exactly as tools.judge.terms.measure computes them, but against one render of
    the design without its copy (2 renders + 1 per text instead of ~3 per element)."""
    import numpy as np
    from renderer.converter.colorutil import parse_rgba

    from tools.judge import terms

    dz = _downscale(design, limit)
    W, H = int(dz["canvas"]["width"]), int(dz["canvas"]["height"])
    canvas = {"width": W, "height": H}
    els = terms._paint_order(dz["elements"])
    under = terms.render({"canvas": canvas, "elements": [e for e in els if e.get("type") != "text"]})[..., :3]
    res = {}
    for t in els:
        if t.get("type") != "text" or not _shown(t) or not str(t.get("content", "")).strip():
            continue
        glyph = terms.render({"canvas": canvas, "elements": [t]}, transparent=True)[..., 3] > 0.25
        if not glyph.any():
            continue
        fs = float(t.get("font_size", 12) or 12)
        halo = terms.dilate(glyph, max(2, int(0.15 * fs))) & ~glyph
        r, g, b_, a = parse_rgba(t.get("color"))
        trgb = np.array([r, g, b_], dtype=np.float32) / 255.0
        bgpix = under[halo | glyph]
        eff = a * trgb + (1 - a) * bgpix
        lt, lb = terms.lum(eff), terms.lum(bgpix)
        c = (np.maximum(lt, lb) + 0.05) / (np.minimum(lt, lb) + 0.05)
        ys, xs = np.where(glyph)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        gray = terms.lum(under[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4])
        res[str(t["id"])] = {"wcag": round(float(np.percentile(c, 10)), 3),
                             "busy": round(terms.busy_p90(gray, 0.7 * fs), 4)}
    return res


def check_c4(src, out, tgt, kit: Path, cache: dict) -> List[str]:
    ids = texts_on_images(out, kit)
    if not ids:
        return []
    if "src_contrast" not in cache:
        cache["src_contrast"] = text_contrast(src)
    rs, ro = cache["src_contrast"], text_contrast(out)
    errs = []
    for i in ids:
        o, s = ro.get(i), rs.get(i)
        if o is None:
            continue
        c0 = s["wcag"] if s else 4.5
        if o["wcag"] < 3.0 and o["wcag"] < c0 - 0.3:
            errs.append(f"{i}: contrast {o['wcag']:.1f} (src {c0:.1f})")
        b0 = s["busy"] if s else 0.0
        if o["busy"] > max(b0, 0.03) + 0.02:
            errs.append(f"{i}: busy {o['busy']:.3f} (src {b0:.3f})")
    return errs


def _face_regions(e: dict, kit: Path, W: float, H: float):
    """Each face of an image element as (mapped face box, visible part) in canvas px."""
    from engine.smart.geom import Box

    p = kit / "assets" / str(e.get("asset_id"))
    fs = faces(str(p)) if p.is_file() else ()
    if not fs:
        return []
    iw, ih = asset_size(str(e["asset_id"]), str(kit))
    x, y, w, h = float(e["x"]), float(e["y"]), float(e["width"]), float(e["height"])
    if w <= 0 or h <= 0:
        return []
    fit = e.get("fit", "contain") or "contain"
    if fit == "fill":
        sx, sy, ox, oy = w / iw, h / ih, x, y
    else:
        sc = max(w / iw, h / ih) if fit == "cover" else min(w / iw, h / ih)
        sx = sy = sc
        ox, oy = x + (w - iw * sc) / 2, y + (h - ih * sc) / 2
    res = []
    for fx, fy, fw, fh in fs:
        fb = Box(ox + fx * sx, oy + fy * sy, fw * sx, fh * sy)
        vx0, vy0 = max(fb.x, x, 0.0), max(fb.y, y, 0.0)
        vx1, vy1 = min(fb.r, x + w, W), min(fb.b, y + h, H)
        vis = Box(vx0, vy0, max(0.0, vx1 - vx0), max(0.0, vy1 - vy0))
        res.append((fb, vis))
    return res


def _face_stats(design: dict, kit: Path, roles: Dict[str, str]) -> Dict[Tuple[str, int], Tuple[float, float]]:
    """(image id, face index) -> (share of the face lost off the canvas / its box, share covered by
    copy, logos or buttons painted above it)."""
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    order = _paint(design["elements"])
    out = {}
    for k, e in enumerate(order):
        if e.get("type") != "image" or not _shown(e):
            continue
        for j, (fb, vis) in enumerate(_face_regions(e, kit, W, H)):
            area = max(1e-6, fb.w * fb.h)
            lost = 1.0 - vis.w * vis.h / area
            covers = [_ink_box(o, kit) for o in order[k + 1:]
                      if _shown(o) and roles.get(str(o["id"])) in ("text", "info")]
            cov = 0.0
            if covers and vis.w > 0 and vis.h > 0:
                n, hit = 16, 0
                for gi in range(n):
                    px = vis.x + (gi + 0.5) * vis.w / n
                    for gj in range(n):
                        py = vis.y + (gj + 0.5) * vis.h / n
                        if any(c.x <= px <= c.r and c.y <= py <= c.b for c in covers):
                            hit += 1
                cov = hit / (n * n) * (vis.w * vis.h) / area
            out[(str(e["id"]), j)] = (lost, cov)
    return out


def check_c5(src, out, tgt, kit: Path, cache: dict) -> List[str]:
    from tools import bleed

    if "roles" not in cache:
        cache["roles"] = bleed.roles(src, kit / "assets")
        cache["src_faces"] = _face_stats(src, kit, cache["roles"])
    sf = cache["src_faces"]
    if not sf:
        return []
    of = _face_stats(out, kit, cache["roles"])
    errs = []
    for key, (l0, c0) in sf.items():
        if l0 > 0.5 or key not in of:
            continue
        l1, c1 = of[key]
        if l1 > 0.15 and l1 > l0 + 0.05:
            errs.append(f"{key[0]} face{key[1]}: {l1:.0%} cut")
        if c1 >= 0.10 and c1 > c0 + 0.05:
            errs.append(f"{key[0]} face{key[1]}: {c1:.0%} covered")
    return errs


def check_c6(src, out, tgt) -> Tuple[List[str], List[str]]:
    from tools.audit import flags

    hard, soft = [], []
    for f in flags(src, out, tgt):
        k = f.split()[0]
        (hard if k in AUDIT_HARD else soft).append(f)
    return hard, soft


def check_tile(src: dict, out: Optional[dict], tgt: dict, rec: dict, kit: Path, cache: dict) -> dict:
    res = {c: [] for c in CHECKS}
    res["soft"] = []
    if out is None:
        res["C1"] = [f"crash: {(rec.get('error') or 'no output')[:80]}"]
        for c in CHECKS[1:]:
            res[c] = ["no output"]
        return res
    for c, fn in (("C1", lambda: check_c1(src, out, tgt, rec)), ("C2", lambda: check_c2(src, out, tgt, kit)),
                  ("C3", lambda: check_c3(src, out, tgt)), ("C4", lambda: check_c4(src, out, tgt, kit, cache)),
                  ("C5", lambda: check_c5(src, out, tgt, kit, cache))):
        try:
            res[c] = fn()
        except Exception as ex:  # noqa: BLE001  a measuring failure is reported, never hidden
            res[c] = [f"check error {type(ex).__name__}: {ex}"[:120]]
    try:
        res["C6"], res["soft"] = check_c6(src, out, tgt)
    except Exception as ex:  # noqa: BLE001
        res["C6"] = [f"check error {type(ex).__name__}: {ex}"[:120]]
    return res


# ============================================================================ running


def engine_root(path: Optional[str]) -> Path:
    if path:
        return Path(path).resolve()
    cwd = Path.cwd()
    return cwd if (cwd / "engine" / "resize.py").is_file() else ROOT


def tiles_of(designs: List[dict], formats: str) -> List[dict]:
    jobs = []
    for d in designs:
        for fmt, tgt in d["targets"]:
            if formats == "tier1" and fmt not in TIER1 and not _tier1(tgt):
                continue
            jobs.append({"name": f"{d['name']}|{fmt}", "design": d["name"], "family": d["family"], "fmt": fmt,
                         "target": tgt})
    return jobs


def _tier1(tgt: dict) -> bool:
    from tools.gate.tiers import tier_of

    return tier_of(tgt) == 1


def _thumb(design: dict, kit: Path, cell: int) -> bytes:
    from PIL import Image

    from tools.rules import render_image

    im = render_image(design, kit / "assets")
    k = min(cell / im.width, cell / im.height)
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _measure_design(arg) -> List[dict]:
    """Every check on every tile of one design (one worker; the source measure is shared)."""
    d, tiles, outs, thumbs, cell = arg
    kit = Path(d["kit"])
    os.environ["CHAMELEON_BRAND_KIT"] = str(kit)
    cache: dict = {}
    rows = []
    src_thumb = None
    for t in tiles:
        rec = outs.get(t["name"], {"out": None, "error": "not run"})
        t0 = time.time()
        res = check_tile(d["source"], rec.get("out"), t["target"], rec, kit, cache)
        failed = [c for c in CHECKS if res[c]]
        row = dict(t, checks=res, failed=failed, secs=rec.get("secs"), measure_secs=round(time.time() - t0, 2))
        want = thumbs == "all" and _tier1(t["target"]) or (thumbs in ("failing", "all") and failed)
        if want and rec.get("out"):
            try:
                if src_thumb is None:
                    src_thumb = _thumb(d["source"], kit, cell)
                row["thumbs"] = (src_thumb, _thumb(rec["out"], kit, cell))
            except Exception:  # noqa: BLE001
                pass
        rows.append(row)
    return rows


def run_suite(designs: List[dict], formats: str = "all", engine: Optional[str] = None, jobs: int = 4,
              thumbs: str = "none", cell: int = 220, log=print) -> List[dict]:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    from tools.gate.engine_run import run_jobs

    root = engine_root(engine)
    tiles = tiles_of(designs, formats)
    by_design = {d["name"]: d for d in designs}
    outs: Dict[str, dict] = {}
    t0 = time.time()
    for kit in sorted({d["kit"] for d in designs}):
        js = [{"name": t["name"], "source": by_design[t["design"]]["source"], "target": t["target"]}
              for t in tiles if by_design[t["design"]]["kit"] == kit]
        outs.update(run_jobs(root, js, kit=Path(kit), workers=jobs, timeout=3600))
    log(f"engine {root}: {len(tiles)} tiles in {time.time() - t0:.0f}s")
    t0 = time.time()
    groups = []
    for d in designs:
        ts = [t for t in tiles if t["design"] == d["name"]]
        if ts:
            groups.append((d, ts, {t["name"]: outs.get(t["name"]) for t in ts}, thumbs, cell))
    rows: List[dict] = []
    if jobs > 1:
        with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
            for r in ex.map(_measure_design, groups):
                rows += r
    else:
        for g in groups:
            rows += _measure_design(g)
    log(f"checks: {time.time() - t0:.0f}s")
    return rows


# ============================================================================ report


def _tier_name(r: dict) -> str:
    return "tier1" if _tier1(r["target"]) else "tier2"


def report(rows: List[dict], log=print) -> dict:
    n = len(rows)
    passed = sum(1 for r in rows if not r["failed"])
    soft = sum(1 for r in rows if r["checks"].get("soft"))
    log(f"\n{n} tiles, {passed} pass all checks ({100 * passed / max(1, n):.0f}%); soft audit flags on {soft} "
        f"({100 * soft / max(1, n):.0f}%, budget 10%)")
    log("\ncheck        fail  pass")
    summary = {"tiles": n, "pass": passed, "soft": soft, "checks": {}}
    for c in CHECKS:
        f = sum(1 for r in rows if r["checks"][c])
        summary["checks"][c] = f
        log(f"{c} {CHECK_NAMES[c]:9s} {f:4d} {n - f:5d}")

    def table(key, label):
        groups: Dict[str, List[dict]] = {}
        for r in rows:
            groups.setdefault(key(r), []).append(r)
        log(f"\n{label:14s} tiles  pass " + " ".join(f"{c:>4s}" for c in CHECKS) + "  soft")
        for g in sorted(groups):
            rs = groups[g]
            log(f"{g:14s} {len(rs):5d} {sum(1 for r in rs if not r['failed']):5d} "
                + " ".join(f"{sum(1 for r in rs if r['checks'][c]):4d}" for c in CHECKS)
                + f"  {sum(1 for r in rs if r['checks'].get('soft')):4d}")
        return {g: {"tiles": len(rs), "pass": sum(1 for r in rs if not r["failed"]),
                    **{c: sum(1 for r in rs if r["checks"][c]) for c in CHECKS}} for g, rs in groups.items()}

    summary["family"] = table(lambda r: r["family"], "family")
    summary["tier"] = table(_tier_name, "tier")
    summary["format"] = table(lambda r: r["fmt"], "format")
    log("\nfailing tiles")
    for r in rows:
        if r["failed"]:
            why = "; ".join(f"{c} {'; '.join(r['checks'][c])[:90]}" for c in r["failed"])
            log(f"  {r['name']:44s} {why}")
    return summary


def sheet(rows: List[dict], path: str, cell: int = 220, cols: int = 3, only_failing: bool = True) -> int:
    from PIL import Image, ImageDraw

    rows = [r for r in rows if r.get("thumbs") and (r["failed"] or not only_failing)]
    if not rows:
        return 0
    pw, ph = 2 * cell + 12, cell + 44
    S = Image.new("RGB", (cols * pw, ((len(rows) + cols - 1) // cols) * ph), (200, 200, 200))
    for k, r in enumerate(rows):
        t = Image.new("RGB", (pw - 4, ph - 4), (240, 240, 240))
        a, b = (Image.open(io.BytesIO(x)) for x in r["thumbs"])
        t.paste(a, (2, 40))
        t.paste(b, (cell + 8, 40))
        d = ImageDraw.Draw(t)
        d.text((3, 2), f"{r['name'][:70]}", fill=(0, 0, 0))
        fails = ", ".join(f"{c}:{CHECK_NAMES[c]}" for c in r["failed"]) or "pass"
        d.text((3, 14), fails, fill=(170, 20, 20) if r["failed"] else (20, 120, 20))
        first = r["checks"][r["failed"][0]][0] if r["failed"] else "; ".join(r["checks"].get("soft", []))
        d.text((3, 26), first[:72], fill=(90, 40, 40))
        S.paste(t, ((k % cols) * pw + 2, (k // cols) * ph + 2))
    S.save(path)
    return len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--engine", default=None, help="engine root (default: cwd if it holds engine/, else this repo)")
    ap.add_argument("--formats", default="all", choices=["tier1", "all"])
    ap.add_argument("--sheet", default="", help="contact sheet of the failing tiles (source | output)")
    ap.add_argument("--sheet-all", default="", help="contact sheet of every tier-1 tile")
    ap.add_argument("--json", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="", help="comma-separated substrings of design names to keep")
    ap.add_argument("--cell", type=int, default=220)
    a = ap.parse_args(argv)
    if a.split in SEEDS:
        ensure_kit()
        os.environ["CHAMELEON_BRAND_KIT"] = str(KIT)
    else:
        os.environ["CHAMELEON_BRAND_KIT"] = str(JEFF / a.split.split("-", 1)[1])
    ds = designs_for(a.split)
    if a.only:
        keep = a.only.split(",")
        ds = [d for d in ds if any(k in d["name"] for k in keep)]
    thumbs = "all" if a.sheet_all else ("failing" if a.sheet else "none")
    rows = run_suite(ds, a.formats, a.engine, max(1, a.jobs), thumbs, a.cell)
    summary = report(rows)
    if a.sheet:
        print("failing sheet:", a.sheet, sheet(rows, a.sheet, a.cell), "tiles")
    if a.sheet_all:
        print("tier-1 sheet:", a.sheet_all, sheet([r for r in rows if _tier1(r["target"])], a.sheet_all, a.cell,
                                                  only_failing=False), "tiles")
    if a.json:
        slim = [{k: v for k, v in r.items() if k != "thumbs"} for r in rows]
        Path(a.json).write_text(json.dumps({"split": a.split, "engine": str(engine_root(a.engine)),
                                            "summary": summary, "tiles": slim}, indent=1, default=str))
        print("json:", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
