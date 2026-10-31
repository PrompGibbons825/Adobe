"""Small geometry helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def r(self) -> float:
        return self.x + self.w

    @property
    def b(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def inter(self, o: "Box") -> float:
        iw = min(self.r, o.r) - max(self.x, o.x)
        ih = min(self.b, o.b) - max(self.y, o.y)
        return iw * ih if iw > 0 and ih > 0 else 0.0

    def inter_box(self, o: "Box") -> Optional["Box"]:
        x0, y0 = max(self.x, o.x), max(self.y, o.y)
        x1, y1 = min(self.r, o.r), min(self.b, o.b)
        if x1 <= x0 or y1 <= y0:
            return None
        return Box(x0, y0, x1 - x0, y1 - y0)

    def union(self, o: "Box") -> "Box":
        x0, y0 = min(self.x, o.x), min(self.y, o.y)
        x1, y1 = max(self.r, o.r), max(self.b, o.b)
        return Box(x0, y0, x1 - x0, y1 - y0)

    def moved(self, dx: float, dy: float) -> "Box":
        return Box(self.x + dx, self.y + dy, self.w, self.h)

    def expanded(self, d: float) -> "Box":
        return Box(self.x - d, self.y - d, self.w + 2 * d, self.h + 2 * d)

    def copy(self) -> "Box":
        return Box(self.x, self.y, self.w, self.h)


def union_all(boxes: Iterable[Box]) -> Optional[Box]:
    out = None
    for b in boxes:
        out = b.copy() if out is None else out.union(b)
    return out


def rotated_aabb(b: Box, deg: float) -> Box:
    if not deg or abs(deg) < 1e-6:
        return b.copy()
    t = math.radians(deg)
    c, s = abs(math.cos(t)), abs(math.sin(t))
    w = b.w * c + b.h * s
    h = b.w * s + b.h * c
    return Box(b.cx - w / 2.0, b.cy - h / 2.0, w, h)


def rotate_about(b: Box, deg: float, cx: float, cy: float) -> Box:
    """Bounding box of ``b`` rotated by ``deg`` (clockwise, as the renderer
    does) about the point (cx, cy)."""
    if not deg or abs(deg) < 1e-6:
        return b.copy()
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    xs, ys = [], []
    for px, py in ((b.x, b.y), (b.r, b.y), (b.x, b.b), (b.r, b.b)):
        dx, dy = px - cx, py - cy
        xs.append(cx + dx * c - dy * s)
        ys.append(cy + dx * s + dy * c)
    return Box(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def el_box(el: dict) -> Box:
    return Box(float(el.get("x", 0) or 0), float(el.get("y", 0) or 0),
               max(0.0, float(el.get("width", 0) or 0)), max(0.0, float(el.get("height", 0) or 0)))


def el_visual_box(el: dict) -> Box:
    """Box including rotation (what is actually painted, ignoring transparency)."""
    b = el_box(el)
    rb = rotated_aabb(b, float(el.get("rotation", 0) or 0))
    return b.union(rb)
