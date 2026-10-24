"""Gate tool: format tiers (loop/STEERING.md S2).

Tier 1 is what the gate optimises: the most common social formats. Everything else is tier 2,
which must not regress. A target is tier 1 when its aspect ratio is within ``ASPECT_TOL`` of a
tier-1 format's and its shorter side is at least ``MIN_SHORT`` px (a 50x50 square is not an
Instagram post), so designer sizes like 1200x630 or 1080x1080 at a different scale still count.
"""

from __future__ import annotations

TIER1 = {
    "ig_post": (1080, 1080),
    "ig_portrait": (1080, 1350),
    "story": (1080, 1920),
    "yt_thumb": (1280, 720),
    "landscape_16_9": (1920, 1080),
    "link_post": (1200, 628),
}
ASPECT_TOL = 0.02
MIN_SHORT = 400


def tier_of(canvas: dict) -> int:
    w, h = float(canvas["width"]), float(canvas["height"])
    if min(w, h) < MIN_SHORT:
        return 2
    ar = w / h
    for W, H in TIER1.values():
        if abs(ar / (W / H) - 1) <= ASPECT_TOL:
            return 1
    return 2
