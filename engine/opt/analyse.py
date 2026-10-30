"""Perceive structure (owner: backdrop / decor / panel / block membership, reading rows).

Evidence: geometry, z order, opacity, group_id. No pixels, no roles beyond structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Block:
    bid: int
    members: List[int]                 # element indices
    x: float
    y: float
    w: float
    h: float
    kind: str                          # text | image | shape | group
    z: float = 0.0
    text_idx: Optional[int] = None     # single-text block
    area_share: float = 0.0
    row: int = 0


@dataclass
class Scene:
    W: float
    H: float
    els: List[dict]
    backdrop: List[int] = field(default_factory=list)
    decor: List[int] = field(default_factory=list)
    panels: List[int] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    rows: List[List[int]] = field(default_factory=list)   # block ids per reading row
    halign: str = "center"


def _box(e):
    return float(e.get("x", 0)), float(e.get("y", 0)), max(1.0, float(e.get("width", 1))), max(1.0, float(e.get("height", 1)))


def _inter(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    return max(0.0, iw) * max(0.0, ih)


def analyse(source: dict) -> Scene:
    W = float(source["canvas"]["width"])
    H = float(source["canvas"]["height"])
    els = source["elements"]
    S = Scene(W, H, els)
    A = W * H
    cv = (0.0, 0.0, W, H)
    n = len(els)
    boxes = [_box(e) for e in els]
    order = sorted(range(n), key=lambda i: (els[i].get("z_index", 0), i))

    # backdrop: non-text, covers >= 85% of canvas, below every text
    zt = min([els[i].get("z_index", 0) for i in order if els[i].get("type") == "text"] or [float("inf")])
    for i in order:
        e = els[i]
        if e.get("type") == "text" or e.get("z_index", 0) >= zt:
            break
        if _inter(boxes[i], cv) >= 0.85 * A and not e.get("rotation"):
            S.backdrop.append(i)
    # a large photo at the bottom of the foreground stack acts as the backdrop (designers bleed it)
    for i in order:
        if i in S.backdrop:
            continue
        e = els[i]
        if e.get("type") == "image" and _inter(boxes[i], cv) >= 0.45 * A:
            S.backdrop.append(i)
        break
    rest = [i for i in order if i not in S.backdrop]

    for i in rest:
        e = els[i]
        x, y, w, h = boxes[i]
        if e.get("type") == "text":
            continue
        vis = _inter(boxes[i], cv) / (w * h)
        faint = float(e.get("opacity", 1.0) or 1.0) * _fill_alpha(e) < 0.4 and e.get("type") == "shape"
        if vis < 0.9 or faint:
            # bleeding or faint non-text: decor, unless something sits inside it (then it is a container)
            holds = any(j != i and els[j].get("type") == "text" and _inter(boxes[i], boxes[j]) > 0.8 * boxes[j][2] * boxes[j][3]
                        for j in rest)
            if not holds or vis < 0.5:
                S.decor.append(i)
                continue
        if (e.get("type") == "shape" and w * h >= 0.45 * A) or _inter(boxes[i], cv) >= 0.6 * A:
            S.panels.append(i)
    fg = [i for i in rest if i not in S.decor and i not in S.panels]

    # union-find blocks
    parent = {i: i for i in fg}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    groups: Dict[str, List[int]] = {}
    for i in fg:
        g = els[i].get("group_id")
        if g:
            groups.setdefault(str(g), []).append(i)
    for mem in groups.values():
        for j in mem[1:]:
            union(mem[0], j)
    for a in fg:
        for b in fg:
            if a >= b:
                continue
            ia = boxes[a][2] * boxes[a][3]
            ib = boxes[b][2] * boxes[b][3]
            inter = _inter(boxes[a], boxes[b])
            small, big = (a, b) if ia <= ib else (b, a)
            sa, ba = min(ia, ib), max(ia, ib)
            if inter <= 0:
                continue
            tb = els[big].get("type")
            ts = els[small].get("type")
            if ts == "text" and tb != "text" and ba <= 0.6 * A and inter >= 0.6 * sa:
                union(a, b)       # text on its plate / art: one block
            elif ba <= 0.30 * A and inter >= 0.6 * sa:
                # small thing on a modest container (badge, button, composite art)
                if tb == "text" and ts == "text" and inter < 0.9 * sa:
                    continue
                union(a, b)
            elif ts != "text" and tb != "text" and inter >= 0.85 * sa and ba <= 0.45 * A:
                union(a, b)
    comp: Dict[int, List[int]] = {}
    for i in fg:
        comp.setdefault(find(i), []).append(i)
    for k, mem in enumerate(sorted(comp.values(), key=lambda m: min(m))):
        xs = [boxes[i][0] for i in mem]
        ys = [boxes[i][1] for i in mem]
        x2 = [boxes[i][0] + boxes[i][2] for i in mem]
        y2 = [boxes[i][1] + boxes[i][3] for i in mem]
        x, y = min(xs), min(ys)
        w, h = max(x2) - x, max(y2) - y
        types = {els[i].get("type") for i in mem}
        kind = types.pop() if len(types) == 1 and len(mem) == 1 else "group"
        b = Block(k, sorted(mem), x, y, max(1, w), max(1, h), kind,
                  z=max(els[i].get("z_index", 0) for i in mem),
                  text_idx=mem[0] if kind == "text" else None,
                  area_share=(w * h) / A)
        S.blocks.append(b)

    # reading rows: sort by top, rows by vertical-interval overlap
    bl = sorted(S.blocks, key=lambda b: (b.y + b.h / 2, b.x))
    rows: List[List[Block]] = []
    for b in bl:
        placed = False
        for r in rows:
            ry0 = min(o.y for o in r)
            ry1 = max(o.y + o.h for o in r)
            ov = min(ry1, b.y + b.h) - max(ry0, b.y)
            if ov > 0.5 * min(b.h, ry1 - ry0):
                # no horizontal overlap with members
                if all(min(o.x + o.w, b.x + b.w) - max(o.x, b.x) <= 0.1 * min(o.w, b.w) for o in r):
                    r.append(b)
                    placed = True
                    break
        if not placed:
            rows.append([b])
    rows.sort(key=lambda r: min(o.y for o in r))
    S.rows = []
    for ri, r in enumerate(rows):
        r.sort(key=lambda o: o.x)
        for o in r:
            o.row = ri
        S.rows.append([o.bid for o in r])

    # dominant horizontal alignment of content
    votes = {"left": 0.0, "center": 0.0, "right": 0.0}
    for b in S.blocks:
        dl = b.x
        dr = W - (b.x + b.w)
        dc = abs((b.x + b.w / 2) - W / 2)
        wt = b.w * b.h
        if dc <= 0.04 * W:
            votes["center"] += wt
        elif dl < dr:
            votes["left"] += wt
        else:
            votes["right"] += wt
    if S.blocks:
        S.halign = max(votes, key=votes.get)
    return S


def _fill_alpha(e) -> float:
    c = str(e.get("fill_color") or "")
    if c.startswith("rgba"):
        try:
            return float(c[c.index("(") + 1:c.index(")")].split(",")[3])
        except Exception:  # noqa: BLE001
            return 1.0
    return 1.0
