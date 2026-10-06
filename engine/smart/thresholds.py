"""Constants that decide which archetypes are tried, how strongly each is
preferred for a target shape, and how far type scales.

They used to be literals in compose.py and score.py. Kept in one place so a
dev tool (tools/thresholds.py) can sweep and fit them on designer-made Crello
retargets, and so the fitted values can be compared with the hand-set ones.
Aspect values are width / height of the target canvas (r) or source (r0).
"""

from __future__ import annotations

# --- which archetypes are generated (compose.generate) ----------------------
STACK_MAX_R = 2.2  # one column only on targets up to this wide
SPLIT_MIN_R = 1.15  # hero beside the copy from this wide
ROW_MIN_R = 1.7  # one line of everything from this wide
ADAPT_TOL = 1.2  # keep the source arrangement while r / r0 is within this factor

# --- per-archetype prior (score._prior) --------------------------------------
ADAPT_PRIOR = 40.0  # bonus for the source arrangement at r == r0, fading to 0 at ADAPT_TOL
STACK_KNEE = 1.25  # a column starts losing favour beyond this width
SPLIT_PEAK = 2.0  # split is most at home at this width
ROW_KNEE = 3.6  # a single row is penalised below this width

# --- type scale caps (compose.scale_bounds) ---------------------------------
AREA_GAIN = 1.0  # scale cap = AREA_GAIN * sqrt(target area / source area)
STRIP_R = 6.0  # strips at least this elongated (either way) ...
STRIP_BOOST = 1.4  # ... get type this much above the sqrt-area prior


def snapshot() -> dict:
    return {k: v for k, v in globals().items() if k.isupper()}
