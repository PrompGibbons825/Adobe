"""Chameleon resize engine, lane `learn`: re-lay a design out the way Crello's designers resized
similar elements to a similar shape, then repair it so every hard rule holds.

    engine/learn/features.py  geometry/type features of each element (no roles)
    engine/learn/knn.py       nearest designer moves, weights fitted on held-out Crello families
    engine/learn/compose.py   units -> transfer -> bleed -> type -> repair
    engine/learn/textfit.py   measures text exactly like the renderer

Any failure falls back to a uniform fit-and-centre; `_finalize` enforces the hard rules on
whatever comes out.
"""

from __future__ import annotations

import math
import sys
import traceback
from typing import Any, Dict

from engine.learn.features import is_bleed, num


def resize(source: Dict[str, Any], target_canvas: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from engine.learn.compose import layout
        out = layout(source, target_canvas)
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        out = _fallback(source, target_canvas)
    return _finalize(source, out, target_canvas)


def _fallback(source, target):
    W = max(1.0, num(source["canvas"].get("width"), 1.0))
    H = max(1.0, num(source["canvas"].get("height"), 1.0))
    TW, TH = max(1.0, num(target.get("width"), 1.0)), max(1.0, num(target.get("height"), 1.0))
    s = min(TW / W, TH / H)
    ox, oy = (TW - W * s) / 2, (TH - H * s) / 2
    out = []
    for e in source.get("elements") or []:
        e = dict(e)
        if is_bleed(e, W, H):
            e.update(x=0.0, y=0.0, width=TW, height=TH)
            if e.get("type") == "image":
                e["fit"] = "cover"
        else:
            e.update(x=ox + num(e.get("x")) * s, y=oy + num(e.get("y")) * s,
                     width=max(1.0, num(e.get("width"), 1) * s), height=max(1.0, num(e.get("height"), 1) * s))
            if e.get("type") == "text":
                e["font_size"] = max(1, int(num(e.get("font_size"), 12) * s))
        out.append(e)
    return dict(canvas=dict(width=int(TW), height=int(TH)), elements=out)


def _finalize(source, out, target):
    TW, TH = max(1.0, num(target.get("width"), 1.0)), max(1.0, num(target.get("height"), 1.0))
    W = max(1.0, num(source["canvas"].get("width"), 1.0))
    H = max(1.0, num(source["canvas"].get("height"), 1.0))
    by_id = {}
    for e in out.get("elements") or []:
        if isinstance(e, dict) and e.get("id") not in by_id:
            by_id[e.get("id")] = e
    res = []
    for se in source.get("elements") or []:
        e = dict(by_id.get(se.get("id")) or se)
        for k in ("type", "content", "font_family", "asset_id"):
            if k in se:
                e[k] = se[k]
        x, y = num(e.get("x")), num(e.get("y"))
        w, h = max(1.0, num(e.get("width"), 1.0)), max(1.0, num(e.get("height"), 1.0))
        full = (e.get("type") == "image" and e.get("fit") == "cover" and w >= TW - 1 and h >= TH - 1)
        if e.get("type") == "image" and not full:
            # keep the source aspect (re-derive from the source box, fit inside the current box)
            sw, sh = max(1e-3, num(se.get("width"), 1)), max(1e-3, num(se.get("height"), 1))
            if abs(math.log((w / h) / (sw / sh))) > 0.01:
                if is_bleed(se, W, H):
                    e["fit"] = "cover"
                    x, y, w, h = 0.0, 0.0, TW, TH
                else:
                    cx, cy = x + w / 2, y + h / 2
                    side = math.sqrt(w * h)
                    ar = sw / sh
                    w, h = side * math.sqrt(ar), side / math.sqrt(ar)
                    x, y = cx - w / 2, cy - h / 2
        if w > TW or h > TH:
            s = min(TW / w, TH / h)
            if e.get("type") == "image" and not full:
                w, h = w * s, h * s
            else:
                w, h = min(w, TW), min(h, TH)
        x = min(max(0.0, x), TW - w)
        y = min(max(0.0, y), TH - h)
        e.update(x=float(x), y=float(y), width=float(w), height=float(h))
        if e.get("type") == "text":
            e["font_size"] = max(1, int(num(e.get("font_size"), 12)))
        res.append(e)
    return dict(canvas=dict(width=int(TW), height=int(TH)), elements=res)
