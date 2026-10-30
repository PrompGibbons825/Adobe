"""The source's design system, held across every target format.

A resize is consistent when every format reads as the same design: the same
type scale (the ratios between its text sizes), the same headline-to-copy
contrast. This module infers the source's type scale once and scores how far
a candidate plan drifts from it.

The parameters are learned, not set by hand. On Crello (cyberagent/crello,
CDLA-Permissive-2.0), 1,361 size families (the same template re-made by its
designer at two or more canvas sizes; the sealed gate families excluded),
158k matched text pairs:

  * every text scales by one common factor, ~ sqrt(area ratio)
    (log f1/f0 = 0.47 log S1/S0 + 0.52 log L1/L0, held-out R^2 0.94; neither
    a text's rank in the type scale nor its length adds anything);
  * so the type scale is held: the headline-to-second-text contrast changes
    by a median factor of 1.000 whatever the change of aspect, and the
    drift of a pair's size ratio, |log(r1/r0)|, has a median that grows with
    the change of aspect |log(ar1/ar0)| = a:
        a < 0.25: 0.029    0.25-0.7: 0.041    0.7-1.5: 0.055    > 1.5: 0.121
    (DRIFT_TOL below is a line through these).

Exempt: a pair whose smaller text sits at the legibility floor (the floor,
not the layout, set it), and text the plan hides.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from .analyze import Design
from .items import Ctx

DRIFT_TOL = (0.025, 0.045)  # free drift of a pair's log size ratio: a + b * |log aspect change| (Crello medians)
STEP = math.log(1.1)  # texts whose source sizes differ by less than this are one level of the scale
FLOOR_SLACK = 1.03  # a text within 3% of its legibility floor is held there by the floor


def type_scale(d: Design) -> Dict[str, float]:
    """The source's type scale: each text element's size (px)."""
    out = {}
    for tid, e in d.by_id.items():
        if e.get("type") == "text":
            fs = float(e.get("font_size") or 0)
            if fs > 0:
                out[tid] = fs
    return out


def tolerance(d: Design, ctx: Ctx) -> float:
    a = abs(math.log((ctx.W / max(1.0, ctx.H)) / (d.W / max(1.0, d.H))))
    return DRIFT_TOL[0] + DRIFT_TOL[1] * a


def drift(placed, d: Design, ctx: Ctx, src: Dict[str, float]) -> Tuple[float, float]:
    """Importance-weighted mean excess drift of the plan's type ratios from the
    source's (log units past the learned tolerance), and the total weight."""
    out: Dict[str, float] = {}
    imp: Dict[str, float] = {}
    held: Dict[str, bool] = {}
    for r, _, _ in placed:
        it = r.item
        role = it.role if it.role in ("legal", "badge", "ribbon", "headline") else "body"
        fl = ctx.floor(role)
        for tid, fs in r.fonts.items():
            if tid in src and fs > 0:
                out[tid], imp[tid] = float(fs), it.importance
                held[tid] = fs <= FLOOR_SLACK * fl
    tol = tolerance(d, ctx)
    ids = sorted(out, key=lambda t: -src[t])
    num = den = 0.0
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            r0 = math.log(src[a] / src[b])
            if r0 < STEP:
                continue  # one level: the pair carries no ratio to hold
            r1 = math.log(out[a] / out[b])
            if held[b] and r1 < r0:
                continue  # the floor raised the smaller text: forced, not chosen
            w = min(imp[a], imp[b])
            num += w * max(0.0, abs(r1 - r0) - tol)
            den += w
    return (num / den if den else 0.0), den


SCALE_TOL = 0.243  # designers' type scale sits within exp(+-0.243) of sqrt(area ratio) 80% of the time (Crello)


def scale_shortfall(placed, d: Design, ctx: Ctx, src: Dict[str, float]) -> float:
    """How far (log units, past SCALE_TOL) the plan's type scale falls short of
    the designers' law: every text scaled by sqrt(target area / source area).
    The plan's type scale is the importance-weighted geometric mean of its
    texts' size factors. Only a shortfall counts: the legibility floor raises
    type on small canvases, and growth is the other terms' business."""
    k = math.sqrt((ctx.W * ctx.H) / max(1.0, d.W * d.H))
    num = den = 0.0
    for r, _, _ in placed:
        for tid, fs in r.fonts.items():
            if tid in src and fs > 0:
                num += r.item.importance * math.log(fs / (src[tid] * k))
                den += r.item.importance
    if not den:
        return 0.0
    return max(0.0, -num / den - SCALE_TOL)

