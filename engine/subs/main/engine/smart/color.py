"""CSS colour parsing (same grammar the renderer accepts)."""

from __future__ import annotations

import re
from typing import Optional, Tuple

_RGBA_RE = re.compile(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)")
_HEX_RE = re.compile(r"#([0-9a-fA-F]{6})([0-9a-fA-F]{2})?$")


def parse_rgba(value: Optional[str], default=(0, 0, 0, 1.0)) -> Tuple[int, int, int, float]:
    if not value or not isinstance(value, str):
        return default
    v = value.strip()
    m = _RGBA_RE.match(v)
    if m:
        r, g, b = (int(float(m.group(i))) for i in (1, 2, 3))
        a = float(m.group(4)) if m.group(4) is not None else 1.0
        return (r, g, b, a)
    m = _HEX_RE.match(v)
    if m:
        h = m.group(1)
        a = int(m.group(2), 16) / 255.0 if m.group(2) else 1.0
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)
    return default


def luminance(rgb) -> float:
    def ch(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb[:3]
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(l1: float, l2: float) -> float:
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


_NAMED = {
    "white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0), "green": (0, 128, 0),
    "blue": (0, 0, 255), "yellow": (255, 255, 0), "orange": (255, 165, 0), "gray": (128, 128, 128),
    "grey": (128, 128, 128), "purple": (128, 0, 128), "pink": (255, 192, 203), "brown": (165, 42, 42),
}


def normalize(value) -> "Optional[str]":
    """The same colour as an ``rgba()`` string the renderer can parse, or None.

    The reference renderer crashes on named / #rrggbbaa text colours and paints
    hex shape fills grey; rgba() is what the schema documents."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    low = v.lower()
    if low.startswith("rgba(") or low.startswith("rgb("):
        m = _RGBA_RE.match(low)
        if m:
            r, g, b = (max(0, min(255, int(float(m.group(i))))) for i in (1, 2, 3))
            a = float(m.group(4)) if m.group(4) is not None else 1.0
            a = max(0.0, min(1.0, a))
            if v.startswith("rgba(") or v.startswith("rgb("):
                return v  # already fine, keep the designer's exact string
            return f"rgba({r},{g},{b},{a:g})"
        return None
    if low in _NAMED:
        r, g, b = _NAMED[low]
        return f"rgba({r},{g},{b},1)"
    if low == "transparent":
        return "rgba(0,0,0,0)"
    h = low.lstrip("#")
    if all(c in "0123456789abcdef" for c in h) and len(h) in (3, 4, 6, 8):
        if len(h) in (3, 4):
            h = "".join(c * 2 for c in h)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        a = int(h[6:8], 16) / 255.0 if len(h) == 8 else 1.0
        return f"rgba({r},{g},{b},{a:.3g})"
    return None
