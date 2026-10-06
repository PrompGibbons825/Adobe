"""Cross-format consistency: one design, many canvases, one family.

Adobe resizes one page at a time. When a design goes to twenty channels the
set should still read as one campaign: the logo in the same corner class,
the same reading order, the same headline-to-body type ratio, the art on the
same side of the copy, the copy aligned the same way. Each target already has
its top-k candidates; here one is picked per target to maximise

    sum_t quality_t(c_t) + weight * mean_{u != t} sim(c_t, c_u)

quality_t is the candidate's score relative to the target's own best (in
units of ~8% of that score), and a candidate more than ~6% below the target's
best is never taken. The search is iterated conditional modes from the
per-target winners, sweeping targets in a fixed order: deterministic, and
it only moves away from a winner when the family gains more than the target
loses.

Features are read from the finished designs and the inferred roles, so this
works for any candidate generator.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

WEIGHT = 2.0  # consistency vs. each target's own score (tuned on the tuning half)
MAX_LOSS = 0.06  # never give up more than this share of a target's best score
SCALE = 0.08  # quality unit: this share of a target's best score

COPY = ("headline", "subhead", "body", "cta")
SKIP = ("background", "decor", "slab", "backdrop", "other", "missing", "void")


def _box(e) -> Tuple[float, float, float, float]:
    return float(e["x"]), float(e["y"]), float(e["width"]), float(e["height"])


def _union(bs):
    x0 = min(b[0] for b in bs)
    y0 = min(b[1] for b in bs)
    x1 = max(b[0] + b[2] for b in bs)
    y1 = max(b[1] + b[3] for b in bs)
    return x0, y0, x1 - x0, y1 - y0


def _cls(v: float) -> int:
    return 0 if v < 1 / 3 else (2 if v > 2 / 3 else 1)


def features(cand: dict) -> dict:
    """What a viewer would compare between two sizes of the same design."""
    design, roles = cand["design"], cand["roles"]
    W, H = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    units: Dict[str, list] = {}
    unit_role: Dict[str, str] = {}
    fonts: Dict[str, List[float]] = {}
    for e in design["elements"]:
        if float(e.get("opacity", 1) or 0) <= 0 or e["width"] <= 0 or e["height"] <= 0:
            continue
        r = roles.get(e["id"], {})
        role = r.get("role", "other")
        if role in SKIP:
            continue
        u = r.get("unit", e["id"])
        units.setdefault(u, []).append(_box(e))
        unit_role[u] = role
        if e.get("type") == "text":
            fonts.setdefault(role, []).append(float(e.get("font_size", 0) or 0))
    boxes = {u: _union(bs) for u, bs in units.items()}
    f: dict = {"archetype": cand.get("archetype")}
    logo = [boxes[u] for u in sorted(boxes) if unit_role[u] == "logo"]
    if logo:
        x, y, w, h = _union(logo)
        f["logo"] = (_cls((y + h / 2) / H), _cls((x + w / 2) / W))
    # reading order of what is read (not the hero art): XY-cut, columns left to right
    f["order"] = _xy_order({u: b for u, b in boxes.items() if unit_role[u] not in ("hero", "image")})
    head = max(fonts.get("headline", []), default=0.0)
    rest = sorted(v for r in ("subhead", "body") for v in fonts.get(r, []))
    if head > 0 and rest:
        f["type_ratio"] = head / rest[len(rest) // 2]
    copy = [boxes[u] for u in boxes if unit_role[u] in COPY]
    hero = [boxes[u] for u in boxes if unit_role[u] == "hero"]
    if copy and hero:
        cx, cy, cw, ch = _union(copy)
        hx, hy, hw, hh = _union(hero)
        ix = min(cx + cw, hx + hw) - max(cx, hx)
        iy = min(cy + ch, hy + hh) - max(cy, hy)
        if ix > 0 and iy > 0 and ix * iy > 0.3 * min(cw * ch, hw * hh):
            f["hero_side"] = "behind"
        else:
            dx = (hx + hw / 2 - (cx + cw / 2)) / W
            dy = (hy + hh / 2 - (cy + ch / 2)) / H
            if abs(dx) >= abs(dy):
                f["hero_side"] = "left" if dx < 0 else "right"
            else:
                f["hero_side"] = "above" if dy < 0 else "below"
    texts = [boxes[u] for u in boxes if unit_role[u] in ("headline", "subhead", "body")]
    if len(texts) >= 2:
        spread = {
            "start": _spread([b[0] for b in texts]),
            "center": _spread([b[0] + b[2] / 2 for b in texts]),
            "end": _spread([b[0] + b[2] for b in texts]),
        }
        best = min(spread, key=lambda k: (spread[k], k))
        if spread[best] <= 0.02 * W + 2:
            f["align"] = best
    return f


def _cuts(boxes: Dict[str, tuple], axis: int) -> List[List[str]]:
    """Split into groups along an axis wherever nothing straddles the gap."""
    keys = sorted(boxes, key=lambda u: (boxes[u][axis], u))
    groups: List[List[str]] = []
    end = None
    for u in keys:
        a, size = boxes[u][axis], boxes[u][axis + 2]
        if groups and a < end - 0.15 * min(size, 1e9):
            groups[-1].append(u)
            end = max(end, a + size)
        else:
            groups.append([u])
            end = a + size
    return groups


def _xy_order(boxes: Dict[str, tuple], depth: int = 0) -> List[str]:
    if len(boxes) <= 1 or depth > 8:
        return sorted(boxes, key=lambda u: (boxes[u][1], boxes[u][0], u))
    for axis in (1, 0):  # bands first, then columns
        groups = _cuts(boxes, axis)
        if len(groups) > 1:
            return [u for g in groups for u in _xy_order({v: boxes[v] for v in g}, depth + 1)]
    return sorted(boxes, key=lambda u: (boxes[u][1], boxes[u][0], u))


def _spread(v: List[float]) -> float:
    return max(v) - min(v)


_AXIS = {"left": "h", "right": "h", "above": "v", "below": "v", "behind": "z"}


def similarity(a: dict, b: dict) -> Tuple[float, float]:
    """(similarity in [0, 1], total weight of the features both sides define)."""
    parts: List[Tuple[float, float]] = []
    if "logo" in a and "logo" in b:
        parts.append((2.0, 0.5 * (a["logo"][0] == b["logo"][0]) + 0.5 * (a["logo"][1] == b["logo"][1])))
    common = [u for u in a["order"] if u in set(b["order"])]
    if len(common) >= 2:
        pb = {u: i for i, u in enumerate(u for u in b["order"] if u in set(common))}
        n = conc = 0
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                n += 1
                conc += pb[common[i]] < pb[common[j]]
        parts.append((1.5, conc / n))
    if "type_ratio" in a and "type_ratio" in b:
        parts.append((1.0, math.exp(-abs(math.log(a["type_ratio"] / b["type_ratio"])) / 0.2)))
    if "hero_side" in a and "hero_side" in b and _AXIS[a["hero_side"]] == _AXIS[b["hero_side"]]:
        parts.append((1.5, float(a["hero_side"] == b["hero_side"])))
    if "align" in a and "align" in b:
        parts.append((1.0, float(a["align"] == b["align"])))
    w = sum(p[0] for p in parts)
    if w <= 0:
        return 1.0, 0.0
    return sum(p[0] * p[1] for p in parts) / w, w


def select(per: Dict[str, List[dict]], weight: Optional[float] = None, sweeps: int = 12) -> Dict[str, int]:
    """Index of the chosen candidate per target (0 = the target's own best)."""
    lam = WEIGHT if weight is None else weight
    names = list(per)
    feats = {n: [features(c) for c in per[n]] for n in names}
    qual: Dict[str, List[Optional[float]]] = {}
    for n in names:
        s0 = per[n][0].get("score")
        q = []
        for c in per[n]:
            s = c.get("score")
            if s0 is None or s is None:
                q.append(0.0 if c is per[n][0] else None)
                continue
            unit = max(1.0, SCALE * abs(s0))
            q.append((s - s0) / unit if s >= s0 - MAX_LOSS * abs(s0) else None)
        qual[n] = q
    pick = {n: 0 for n in names}
    sim_cache: Dict[tuple, Tuple[float, float]] = {}

    def sim(n, i, m, j):
        key = (n, i, m, j) if n < m else (m, j, n, i)
        if key not in sim_cache:
            sim_cache[key] = similarity(feats[n][i], feats[m][j])
        return sim_cache[key]

    def local(n, i):
        tot = wsum = 0.0
        for m in names:
            if m == n:
                continue
            s, w = sim(n, i, m, pick[m])
            tot += s * w
            wsum += w
        return (tot / wsum) if wsum > 0 else 0.0

    for _ in range(sweeps):
        changed = False
        for n in names:
            best_i, best_v = pick[n], None
            for i, q in enumerate(qual[n]):
                if q is None:
                    continue
                v = q + lam * local(n, i)
                if best_v is None or v > best_v + 1e-9:
                    best_i, best_v = i, v
            if best_i != pick[n]:
                pick[n] = best_i
                changed = True
        if not changed:
            break
    return pick


def metric(chosen: Dict[str, dict]) -> dict:
    """Family consistency of one pick per target, over all target pairs."""
    names = list(chosen)
    feats = {n: features(chosen[n]) for n in names}
    sims, logo, order, side, align = [], [], [], [], []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fa, fb = feats[a], feats[b]
            s, w = similarity(fa, fb)
            if w > 0:
                sims.append(s)
            if "logo" in fa and "logo" in fb:
                logo.append(float(fa["logo"] == fb["logo"]))
            common = [u for u in fa["order"] if u in set(fb["order"])]
            if len(common) >= 2:
                pb = {u: k for k, u in enumerate(u for u in fb["order"] if u in set(common))}
                pairs = [(common[x], common[y]) for x in range(len(common)) for y in range(x + 1, len(common))]
                order.append(sum(pb[p] < pb[q] for p, q in pairs) / len(pairs))
            if "hero_side" in fa and "hero_side" in fb and _AXIS[fa["hero_side"]] == _AXIS[fb["hero_side"]]:
                side.append(float(fa["hero_side"] == fb["hero_side"]))
            if "align" in fa and "align" in fb:
                align.append(float(fa["align"] == fb["align"]))
    ratios = [math.log(feats[n]["type_ratio"]) for n in names if "type_ratio" in feats[n]]

    def mean(v):
        return round(sum(v) / len(v), 4) if v else None

    sd = None
    if len(ratios) >= 2:
        m = sum(ratios) / len(ratios)
        sd = round(math.sqrt(sum((r - m) ** 2 for r in ratios) / len(ratios)), 4)
    return {"similarity": mean(sims), "logo_same_class": mean(logo), "reading_order": mean(order),
            "hero_same_side": mean(side), "same_alignment": mean(align), "type_ratio_log_sd": sd,
            "pairs": len(sims)}


def house_style(chosen: Dict[str, dict]) -> dict:
    """The family's majority traits (what the batch has in common)."""
    votes: Dict[str, Dict[str, int]] = {}
    for c in chosen.values():
        f = features(c)
        for k in ("logo", "hero_side", "align", "archetype"):
            if k in f and f[k] is not None:
                v = "-".join(("top", "middle", "bottom")[f[k][0]:f[k][0] + 1] + ("left", "centre", "right")[f[k][1]:f[k][1] + 1]) \
                    if k == "logo" else str(f[k])
                votes.setdefault(k, {})
                votes[k][v] = votes[k].get(v, 0) + 1
    return {k: max(sorted(v.items()), key=lambda t: t[1])[0] for k, v in sorted(votes.items())}


# ----------------------------------------------------------------------------
# anchoring the logo


def _logo_ids(cand: dict) -> List[str]:
    return [i for i, r in sorted(cand["roles"].items()) if r.get("role") == "logo"]


def logo_home(source: dict, per: Dict[str, List[dict]]) -> Optional[Tuple[int, int]]:
    """The family's logo cell: where the designer put it in the source, unless
    that is the dead centre (then the most common cell among the winners)."""
    first = next(iter(per.values()))[0]
    ids = set(_logo_ids(first))
    els = [e for e in source.get("elements", []) if e.get("id") in ids]
    if not els:
        return None
    W, H = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    x, y, w, h = _union([_box(e) for e in els])
    cell = (_cls((y + h / 2) / H), _cls((x + w / 2) / W))
    if cell != (1, 1):
        return cell
    votes: Dict[Tuple[int, int], int] = {}
    for cs in per.values():
        c = features(cs[0]).get("logo")
        if c is not None:
            votes[c] = votes.get(c, 0) + 1
    return max(sorted(votes.items()), key=lambda t: t[1])[0] if votes else None


def anchor_box(cell: Tuple[int, int], cand: dict) -> Optional[Tuple[str, dict]]:
    """(logo id, box): the logo as big as in `cand`, moved into `cell`."""
    ids = set(_logo_ids(cand))
    design = cand["design"]
    els = [e for e in design["elements"] if e["id"] in ids and float(e.get("opacity", 1) or 0) > 0]
    if not els:
        return None
    W, H = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    x, y, w, h = _union([_box(e) for e in els])
    if w >= 0.6 * W or h >= 0.6 * H:
        return None
    m = max(6.0, 0.05 * min(W, H))
    v, hz = cell
    bx = (m, (W - w) / 2.0, W - m - w)[hz]
    by = (m, (H - h) / 2.0, H - m - h)[v]
    bx, by = min(max(0.0, bx), W - w), min(max(0.0, by), H - h)
    return sorted(ids)[0], {"x": round(bx, 2), "y": round(by, 2), "width": round(w, 2), "height": round(h, 2)}
