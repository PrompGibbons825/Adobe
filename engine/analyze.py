"""Turn a raw element list into a semantic model of the design.

No labels are given, so roles are inferred from geometry, type, font size,
group_id, stacking and asset names. Output: a list of Units (things that
move together) each with a role and a priority.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import saliency

CTA_WORDS = re.compile(
    r"\b(shop|buy|order|learn|get|sign|try|download|join|book|start|discover|explore|"
    r"subscribe|register|reserve|claim|grab|see|watch|read|call|visit|apply|donate)\b", re.I)

CTA_LEAD = re.compile(r"^\W*" + CTA_WORDS.pattern, re.I)  # "Shop now", not "We'll get through"

PRIORITY = {
    "headline": 10, "logo": 9, "cta": 8.5, "hero": 7, "subhead": 5, "body": 4,
    "group": 4, "badges": 3, "ribbon": 2.5, "minor": 1.5, "aligned": 1.5, "legal": 1.2, "missing": 0,
}
# Roles that are laid out in the main flow (others float or fill)
FLOW_ROLES = {"headline", "logo", "cta", "hero", "subhead", "body", "group", "badges", "legal", "aligned"}


@dataclass
class Unit:
    uid: str
    role: str
    kind: str          # text | image | button | group | badges | bg | deco | ribbon
    els: list          # source element dicts
    members: list = field(default_factory=list)  # for badges: list[Unit]
    prio: float = 0.0
    container: dict | None = None                  # for frames: the enclosing element
    children: list = field(default_factory=list)   # for frames: inner units (re-laid out)

    @property
    def ids(self):
        if self.kind == "badges":
            return [i for m in self.members for i in m.ids]
        return [e["id"] for e in self.els]

    @property
    def bbox(self):
        if self.kind == "badges":
            boxes = [m.bbox for m in self.members]
        else:
            boxes = [(e["x"], e["y"], e["x"] + e["width"], e["y"] + e["height"]) for e in self.els]
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    @property
    def w(self):
        b = self.bbox
        return b[2] - b[0]

    @property
    def h(self):
        b = self.bbox
        return b[3] - b[1]

    @property
    def cy(self):
        b = self.bbox
        return (b[1] + b[3]) / 2

    def text_el(self):
        for e in self.els:
            if e["type"] == "text":
                return e
        return None

    def shape_el(self):
        for e in self.els:
            if e["type"] == "shape":
                return e
        return None


def norm_rot(r):
    """Rotation in degrees mapped to (-180, 180]: 269.86 is -90.14."""
    r = ((r or 0) + 180.0) % 360.0 - 180.0
    return 180.0 if r == -180.0 else r


def _poly(e):
    """The element's real footprint: its rectangle turned about its centre."""
    import math
    r = norm_rot(e.get("rotation", 0))
    x, y, w, h = e["x"], e["y"], e["width"], e["height"]
    if abs(r) <= 1:
        b = _box(e)
        return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
    cx, cy = x + w / 2, y + h / 2
    a = math.radians(r)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)
            for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def _poly_area(pts):
    return abs(sum(pts[i][0] * pts[i - 1][1] - pts[i - 1][0] * pts[i][1] for i in range(len(pts)))) / 2


def _clip(subject, clip):
    """Sutherland-Hodgman: subject polygon clipped by a convex clip polygon."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def cross(p, q, a, b):
        x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
        x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return q
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    # orient the clip polygon counter-clockwise in screen coordinates
    if sum(clip[i][0] * clip[i - 1][1] - clip[i - 1][0] * clip[i][1] for i in range(len(clip))) > 0:
        clip = clip[::-1]
    out = subject
    for i in range(len(clip)):
        a, b = clip[i - 1], clip[i]
        inp, out = out, []
        if not inp:
            break
        for j in range(len(inp)):
            p, q = inp[j - 1], inp[j]
            if inside(q, a, b):
                if not inside(p, a, b):
                    out.append(cross(p, q, a, b))
                out.append(q)
            elif inside(p, a, b):
                out.append(cross(p, q, a, b))
    return out


def _area(e):
    if abs(norm_rot(e.get("rotation", 0))) > 1:
        return e["width"] * e["height"]
    b = _box(e)
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _box(e):
    r = norm_rot(e.get("rotation", 0))
    if abs(r) > 1:
        # a turned piece covers its rotated footprint (rotation is about the box centre)
        from .blocks import rot_bounds
        return rot_bounds(e["x"], e["y"], e["width"], e["height"], r)
    h = e["height"]
    if e["type"] == "text":
        # what the letters actually cover: inked height, and inked width placed
        # by the text's alignment inside its box
        h = min(h, _ink_h(e))
        iw = _ink_w(e)
        al = e.get("text_align", "left")
        x0 = e["x"] + (e["width"] - iw) * (0.5 if al == "center" else 1.0 if al == "right" else 0.0)
        return (x0, e["y"], x0 + iw, e["y"] + h)
    return (e["x"], e["y"], e["x"] + e["width"], e["y"] + h)


def _ink_w(e):
    from . import textfit
    try:
        return min(e["width"], textfit.measure(e, max(1, int(e["font_size"])), e["width"])["width"])
    except Exception:
        return e["width"]


def _ink_h(e):
    from . import textfit
    try:
        m = textfit.measure(e, max(1, int(e["font_size"])), e["width"])
        return m["height"]
    except Exception:
        return e["height"]


def _inter(a, b):
    """Overlap area of two elements' real footprints (turned pieces included)."""
    A, B = _box(a), _box(b)
    ix = max(0.0, min(A[2], B[2]) - max(A[0], B[0]))
    iy = max(0.0, min(A[3], B[3]) - max(A[1], B[1]))
    if ix * iy <= 0:
        return 0.0
    if abs(norm_rot(a.get("rotation", 0))) <= 1 and abs(norm_rot(b.get("rotation", 0))) <= 1:
        return ix * iy
    pts = _clip(_poly(a), _poly(b))
    return _poly_area(pts) if len(pts) >= 3 else 0.0


def _gap(a, b):
    A, B = _box(a), _box(b)
    dx = max(0.0, max(A[0], B[0]) - min(A[2], B[2]))
    dy = max(0.0, max(A[1], B[1]) - min(A[3], B[3]))
    return max(dx, dy)


def _tol(o):
    """How much of a piece may stick out of a card and still belong to it:
    a text mostly on a card (60%+) is on it; text widths are only estimates."""
    return 0.4 if o["type"] == "text" else 0.15


def _inside(inner, outer, tol=0.15):
    return _inter(inner, outer) >= (1 - tol) * max(1e-6, _area(inner))


def _corners(e):
    info = saliency.analyze(e.get("asset_id") or "")
    return info["corners"] if info else False


def _edge_cover(e):
    info = saliency.analyze(e.get("asset_id") or "")
    return info["edge_cover"] if info else 0.0


def _ink_density(e):
    """Opaque share inside the image's own visible (alpha) box: low for line art."""
    if e["type"] != "image":
        return 1.0
    info = saliency.analyze(e.get("asset_id") or "")
    if info is None:
        return 1.0
    ab = info["alpha_box"]
    return min(1.0, info["opaque_frac"] / max(1e-3, (ab[2] - ab[0]) * (ab[3] - ab[1])))


def _opaque_frac(e):
    if e["type"] != "image":
        return 1.0
    info = saliency.analyze(e.get("asset_id") or "")
    return info["opaque_frac"] if info else 1.0


class _UF:
    def __init__(self, ids):
        self.p = {i: i for i in ids}

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _text_roles(texts):
    """Headline / subhead / body / legal / cta for every text element."""
    roles = {}
    if not texts:
        return roles
    # a call to action is a button-sized label: the design's largest type
    # (clearly bigger than the rest) is the headline even if it starts with a verb
    fs_sorted = sorted((e["font_size"] for e in texts), reverse=True)
    biggest = fs_sorted[0] if fs_sorted else 0
    dominant = len(fs_sorted) == 1 or biggest >= 1.3 * fs_sorted[1]
    cta = [e for e in texts if CTA_LEAD.match(e["content"]) and len(e["content"]) <= 30
           and not (dominant and e["font_size"] == biggest)]
    others = [e for e in texts if e not in cta]
    for e in cta:
        roles[e["id"]] = "cta"
    if not others:
        return roles
    head = max(others, key=lambda e: (e["font_size"] * (1.15 if e.get("font_weight") == "bold" else 1), -e["y"]))
    roles[head["id"]] = "headline"
    hfs = head["font_size"]
    rest = sorted([e for e in others if e is not head], key=lambda e: e["font_size"], reverse=True)
    sub_done = False
    for e in rest:
        small = e["font_size"] <= 0.4 * hfs
        faint = (e.get("opacity", 1) or 1) < 0.8 or re.search(r",\s*0?\.[0-6]\d*\)", e.get("color", "")) is not None
        if small and (len(e["content"]) > 40 or faint) and e is rest[-1]:
            roles[e["id"]] = "legal"
        elif not sub_done and e["font_size"] >= 0.3 * hfs:
            roles[e["id"]], sub_done = "subhead", True
        else:
            roles[e["id"]] = "body"
    return roles


ROLE_ORDER = ["headline", "cta", "subhead", "body", "legal"]


def _same_line(a, b):
    """Two texts set side by side on one line (HOHOHO | AGENCY): same baseline,
    similar size, a word space apart. They read as one line."""
    if any(abs(norm_rot(e.get("rotation", 0))) > 1 for e in (a, b)):
        return False
    fa, fb = a["font_size"], b["font_size"]
    if abs(fa - fb) > 0.2 * max(fa, fb):
        return False
    from . import textfit
    try:
        ma = textfit.measure(a, max(1, int(fa)), a["width"])
        mb = textfit.measure(b, max(1, int(fb)), b["width"])
    except Exception:
        return False
    if ma["lines"] != 1 or mb["lines"] != 1:
        return False
    A, B = _box(a), _box(b)
    if abs(A[3] - B[3]) > 0.25 * max(fa, fb):
        return False
    # inked horizontal extents, by alignment inside each box
    def ink(e, m, bx):
        w = min(e["width"], m["width"])
        al = e.get("text_align", "left")
        x0 = bx[0] + (e["width"] - w) * (0.5 if al == "center" else 1.0 if al == "right" else 0.0)
        return x0, x0 + w
    a0, a1 = ink(a, ma, A)
    b0, b1 = ink(b, mb, B)
    gap = max(a0, b0) - min(a1, b1)
    return -0.1 * max(fa, fb) <= gap <= 0.8 * max(fa, fb)


def _fragments(a, b):
    """Two texts that are pieces of one word: same style, touching, and at
    least one is a fragment of a few letters."""
    if any(abs(norm_rot(e.get("rotation", 0))) > 1 for e in (a, b)):
        return False
    short = min(len(re.sub(r"\s", "", e["content"])) for e in (a, b))
    if short == 0 or short > 3:
        return False
    fa, fb = a["font_size"], b["font_size"]
    if abs(fa - fb) > 0.1 * max(fa, fb):
        return False
    for k in ("font_family", "font_weight", "font_style"):
        if a.get(k) != b.get(k):
            return False
    if re.sub(r"\s", "", a.get("color", "")) != re.sub(r"\s", "", b.get("color", "")):
        return False
    return _gap(a, b) <= 0.3 * min(fa, fb)


def _contact(a, b, mind, carea):
    """Two substantial pieces sharing an edge: no gap, and side by side along
    at least half of the shorter one. They read as one object."""
    if a["type"] == "text" or b["type"] == "text":
        return False
    if min(_area(a), _area(b)) < 0.02 * carea:
        return False
    A, B = _box(a), _box(b)
    tol = 0.004 * mind
    dx = max(A[0], B[0]) - min(A[2], B[2])     # > 0: horizontal gap, < 0: overlap
    dy = max(A[1], B[1]) - min(A[3], B[3])
    span_y = min(A[3], B[3]) - max(A[1], B[1])
    span_x = min(A[2], B[2]) - max(A[0], B[0])
    side = abs(dx) <= tol and span_y >= 0.5 * min(A[3] - A[1], B[3] - B[1])
    stacked = abs(dy) <= tol and span_x >= 0.5 * min(A[2] - A[0], B[2] - B[0])
    return side or stacked


def _picture(e, carea):
    """A large detailed picture (the thing the loose reading frees content from)."""
    if e["type"] != "image" or _area(e) < 0.15 * carea or not _fills_box(e):
        return False
    info = saliency.analyze(e.get("asset_id") or "")
    return info is not None and info["busyness"] > 0.03


def _flush(a, b, mind):
    """Touching or overlapping pieces that share two or more edge lines (same
    left, right, top or bottom, or one's edge on the other's): designed as one."""
    A, B = _box(a), _box(b)
    tol = 0.01 * mind
    if max(A[0], B[0]) - min(A[2], B[2]) > tol or max(A[1], B[1]) - min(A[3], B[3]) > tol:
        return False
    lines = sum(abs(A[i] - B[i]) <= tol for i in range(4))
    lines += abs(A[2] - B[0]) <= tol or abs(A[0] - B[2]) <= tol
    lines += abs(A[3] - B[1]) <= tol or abs(A[1] - B[3]) <= tol
    return lines >= 2


def _fills_box(e):
    """Opaque edge to edge (a photo, a flat panel, a full-frame illustration):
    it can be cropped without cutting a silhouette."""
    if e["type"] == "shape":
        return e.get("shape_type", "rectangle") == "rectangle" and not e.get("corner_radius")
    if e["type"] != "image":
        return False
    info = saliency.analyze(e.get("asset_id") or "")
    return info is not None and info["opaque_frac"] > 0.97


def _is_backdrop(e, cw, ch):
    if e["type"] == "text" or not _fills_box(e):
        return False
    x0, y0, x1, y1 = _box(e)
    if (min(x1, cw) - max(x0, 0)) * (min(y1, ch) - max(y0, 0)) < 0.5 * cw * ch:
        return False
    t = 0.01 * min(cw, ch)
    return (x0 <= t) + (y0 <= t) + (x1 >= cw - t) + (y1 >= ch - t) >= 2


def _cluster(cand, carea, mind, cw_=None, ch_=None, compose=True, on_scene=False):
    """Union-find over elements, in three passes.

    strong:  designer groups, word fragments, overlapping texts, and enclosure
             (a piece belongs to the enclosure directly beneath it)
    contact: substantial pieces sharing an edge
    weak:    partial overlaps and small ornaments, strongest first. A weak
             link never bridges two groups that each frame their own text:
             a sketch overlapping two cards does not fuse the cards."""
    if not cand:
        return []
    uf = _UF([e["id"] for e in cand])
    opaque = {e["id"]: _opaque_frac(e) for e in cand}
    has_text = {e["id"]: e["type"] == "text" for e in cand}
    framed = {e["id"]: False for e in cand}

    def union(x, y):
        rx, ry = uf.find(x), uf.find(y)
        if rx == ry:
            return
        uf.union(rx, ry)
        r = uf.find(rx)
        has_text[r] = has_text[rx] or has_text[ry]
        framed[r] = framed[rx] or framed[ry]
    # backdrops: content floating on them is not glued to them. A backdrop
    # fills its box, covers half the canvas or more and runs off two or more edges.
    # Anything else (a circle, a cross, a card) is an object or a container.
    # Nothing is a backdrop when it sits on a detailed full-canvas background
    # (a band over a photo is a panel on the scene, not the scene).
    big = {e["id"] for e in cand if cw_ and not on_scene and _is_backdrop(e, cw_, ch_)}
    if not compose:
        # the loose reading: a large *detailed* picture is not a container either
        # (text stays readable on a flat card, not on a detailed picture)
        big |= {e["id"] for e in cand if e["type"] == "image" and _fills_box(e) and _area(e) > 0.15 * carea
                and (saliency.analyze(e.get("asset_id") or "") or {}).get("busyness", 0) > 0.03}
    enclosers: dict[str, list] = {}
    contacts, weak, overlaps, loose_joins = [], [], [], []
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            if a.get("group_id") and a.get("group_id") == b.get("group_id"):
                union(a["id"], b["id"])
                continue
            if a.get("group_id") and b.get("group_id"):
                continue
            ta, tb = a["type"] == "text", b["type"] == "text"
            if not ta and not tb and _flush(a, b, mind):
                if compose:
                    union(a["id"], b["id"])   # designed together: two or more shared edge lines
                elif not (_picture(a, carea) or _picture(b, carea)):
                    loose_joins.append((a["id"], b["id"]))
                continue
            if a["id"] in big or b["id"] in big:
                if not ta and not tb and _contact(a, b, mind, carea):
                    contacts.append((a["id"], b["id"]))
                continue
            if ta and tb and (_fragments(a, b) or _same_line(a, b)):
                union(a["id"], b["id"])   # one word split across texts ("PA" / "IN"), or one line
                continue
            inter = _inter(a, b)
            small = min(_area(a), _area(b))
            for f, o in ((a, b), (b, a)):
                if f["type"] != "text" and _area(f) > _area(o) * 1.5 and _inside(o, f, _tol(o)):
                    enclosers.setdefault(o["id"], []).append(f)
                    break
            else:
                if not ta and not tb and (_inside(a, b, 0.05) or _inside(b, a, 0.05)) \
                        and max(_area(a), _area(b)) <= 1.5 * min(_area(a), _area(b)):
                    union(a["id"], b["id"])   # layers of one object (a card and its inset outline)
                elif ta and tb:
                    if inter >= 0.2 * small:
                        union(a["id"], b["id"])
                elif _contact(a, b, mind, carea):
                    contacts.append((a["id"], b["id"]))
                else:
                    if not ta and not tb and inter > 0 and small >= 0.02 * carea:
                        overlaps.append((a["id"], b["id"]))   # joins if both hold text (see below)
                    if inter >= 0.15 * small:
                        weak.append((inter / max(1e-6, small), a["id"], b["id"],
                                     inter / max(1e-6, _area(a)), inter / max(1e-6, _area(b))))
                    elif not (ta and tb) and (
                            (_gap(a, b) <= 0.012 * mind and small < 0.015 * carea)
                            or (_gap(a, b) <= 0.025 * mind and small < 0.06 * carea
                                and min(_ink_density(a), _ink_density(b)) < 0.4)):
                        weak.append((0.1, a["id"], b["id"], 0.0, 0.0))  # ornaments / line-art pieces of a lockup
    byid = {e["id"]: e for e in cand}
    for oid, fs in enclosers.items():
        z = byid[oid]["z_index"]
        if byid[oid]["type"] == "text":
            # only something solid can hold text (sparse line art behind it does not)
            fs = [f for f in fs if f["type"] != "image" or opaque[f["id"]] >= 0.3] or fs
        below = [f for f in fs if f["z_index"] < z]
        f = max(below, key=lambda f: f["z_index"]) if below else min(fs, key=_area)
        union(oid, f["id"])
        if byid[oid]["type"] == "text":
            framed[uf.find(oid)] = True
    for x, y in contacts:
        if compose:
            union(x, y)
        elif not (_picture(byid[x], carea) or _picture(byid[y], carea)):
            loose_joins.append((x, y))
    if loose_joins:
        # the loose reading keeps touching cards together as one object, unless
        # together they tile most of the canvas (then they are the page grid)
        tmp = _UF([e["id"] for e in cand])
        for x, y in loose_joins:
            tmp.union(x, y)
        comp = {}
        for e in cand:
            comp.setdefault(tmp.find(e["id"]), []).append(e)
        for members in comp.values():
            if len(members) < 2:
                continue
            bs = [_box(e) for e in members]
            ext = (max(b[2] for b in bs) - min(b[0] for b in bs)) * (max(b[3] for b in bs) - min(b[1] for b in bs))
            if ext < 0.6 * carea:
                for e in members[1:]:
                    union(members[0]["id"], e["id"])
    if compose:
        # two text cards that overlap at all, however little, are one composition
        for x, y in overlaps:
            if framed[uf.find(x)] and framed[uf.find(y)]:
                union(x, y)
    for _, x, y, fx, fy in sorted(weak, reverse=True):
        rx, ry = uf.find(x), uf.find(y)
        if rx == ry:
            continue
        if framed[rx] and framed[ry]:
            continue          # a bridge between two framed text blocks
        # a sparse line-art piece that mostly lies outside a text card is
        # decoration around the card; a solid picture overlapping it is part of
        # the composition
        if (framed[ry] and not has_text[rx] and fx < 0.5 and _ink_density(byid[x]) < 0.4) or \
                (framed[rx] and not has_text[ry] and fy < 0.5 and _ink_density(byid[y]) < 0.4):
            continue
        union(x, y)
    comps: dict[str, list] = {}
    for e in cand:
        comps.setdefault(uf.find(e["id"]), []).append(e)
    out = list(comps.values())
    for m in out:
        m.sort(key=lambda e: e["z_index"])
    return out


def _find_container(members):
    """The element that encloses most of the others (a frame, card, pill)."""
    best = None
    for f in members:
        if f["type"] == "text" or abs(norm_rot(f.get("rotation", 0))) > 1:
            continue
        others = [o for o in members if o is not f]
        inside = [o for o in others if _area(f) > _area(o) * 1.5 and _inside(o, f, _tol(o))]
        if others and len(inside) >= max(1, 0.7 * len(others)) and len(inside) == len(others):
            if best is None or _area(f) > _area(best):
                best = f
    return best


def _group_unit(members, troles, carea, mind, depth=0):
    gid = members[0].get("group_id") or ("cl_" + members[0]["id"])
    txt = [m for m in members if m["type"] == "text"]
    non = [m for m in members if m["type"] != "text"]
    rotated = any(abs(norm_rot(m.get("rotation", 0))) > 1 for m in members)
    explicit = all(m.get("group_id") for m in members)
    if rotated and explicit and len(members) <= 3:
        return Unit(gid, "ribbon", "ribbon", members)
    if len(txt) == 1 and len(non) == 1 and not rotated and _inside(txt[0], non[0], 0.25) \
            and _area(non[0]) < 0.12 * carea and len(txt[0]["content"]) <= 40:
        return Unit(gid, "button", "button", members)
    rs = [troles.get(t["id"], "body") for t in txt]
    role = min(rs, key=ROLE_ORDER.index) if rs else "group"
    if not txt:
        aid = " ".join((m.get("asset_id") or "").lower() for m in non)
        role = "logo" if "logo" in aid else "group"
    cont = _find_container(members) if depth < 2 and not rotated else None
    if cont is not None and (len(members) >= 3 or txt):   # a card holding even one text reflows
        inner = [m for m in members if m is not cont]
        children = []
        for sub in _cluster(inner, carea, mind):
            if len(sub) == 1:
                e = sub[0]
                if e["type"] == "text":
                    children.append(Unit(e["id"], troles.get(e["id"], "body"), "text", [e]))
                elif e["type"] == "image":
                    children.append(Unit(e["id"], "minor", "image", [e]))
                else:
                    children.append(Unit(e["id"], "group", "group", [e]))
            else:
                children.append(_group_unit(sub, troles, carea, mind, depth + 1))
        return Unit(gid, role, "frame", members, container=cont, children=children)
    return Unit(gid, role, "group", members)


def _ink_box(e):
    """Box of what an element actually draws (an image's opaque pixels)."""
    x0, y0, x1, y1 = _box(e)
    if e["type"] == "image":
        info = saliency.analyze(e.get("asset_id") or "")
        if info is not None:
            ab = info["alpha_box"]
            w, h = x1 - x0, y1 - y0
            return (x0 + ab[0] * w, y0 + ab[1] * h, x0 + ab[2] * w, y0 + ab[3] * h)
    return (x0, y0, x1, y1)


def coverage(boxes, region, n=64):
    """Share of `region` covered by the union of `boxes` (coarse raster)."""
    import numpy as np
    rx0, ry0, rx1, ry1 = region
    if rx1 <= rx0 or ry1 <= ry0:
        return 0.0
    g = np.zeros((n, n), bool)
    for b in boxes:
        i0 = int(np.clip((b[0] - rx0) / (rx1 - rx0) * n, 0, n))
        i1 = int(np.clip(np.ceil((b[2] - rx0) / (rx1 - rx0) * n), 0, n))
        j0 = int(np.clip((b[1] - ry0) / (ry1 - ry0) * n, 0, n))
        j1 = int(np.clip(np.ceil((b[3] - ry0) / (ry1 - ry0) * n), 0, n))
        g[j0:j1, i0:i1] = True
    return float(g.mean())


ANCHOR_ROLES = {"headline", "subhead", "body", "cta", "legal", "logo"}


def _unit_ink(u):
    """Inked horizontal extent and vertical extent of a unit, in source px."""
    boxes = []
    for e in (u.els if u.kind != "badges" else [x for m in u.members for x in m.els]):
        boxes.append(_box(e))      # _box is already the inked extent for text
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _aligned_anchor(u, units, cw, mind):
    """(anchor unit, 'center'|'left'|'right') if u sits right above or below a
    text-bearing unit on one of its lines, else None."""
    pb = _unit_ink(u)
    pw = pb[2] - pb[0]
    best = None
    for a in units:
        if a is u or a.role not in ANCHOR_ROLES:
            continue
        ab = _unit_ink(a)
        aw = ab[2] - ab[0]
        if pw > aw:
            continue                       # a satellite is smaller than its anchor
        if pb[3] <= ab[1]:
            gap = ab[1] - pb[3]
        elif pb[1] >= ab[3]:
            gap = pb[1] - ab[3]
        else:
            continue                       # side by side or overlapping: not this rule
        if gap > 0.08 * mind:
            continue
        tol = 0.02 * cw
        lines = (("center", abs((pb[0] + pb[2]) / 2 - (ab[0] + ab[2]) / 2)),
                 ("left", abs(pb[0] - ab[0])), ("right", abs(pb[2] - ab[2])))
        line, d = min(lines, key=lambda x: x[1])
        if d <= tol and (best is None or gap < best[2]):
            best = (a, line, gap)
    return best[:2] if best else None


def analyze(source: dict, compose: bool = True) -> dict:
    """compose=False leaves compositions (pieces joined only by contact, flush
    edges or overlapping text cards) as separate units: the engine also tries
    that reading, so a composition never forces unreadable text."""
    cw, ch = source["canvas"]["width"], source["canvas"]["height"]
    els = [dict(e) for e in source["elements"]]
    for i, e in enumerate(els):
        e.setdefault("z_index", i)
    carea = cw * ch
    mind = min(cw, ch)
    units: list[Unit] = []
    used = set()

    # 1a. Borders: outlines that echo the canvas edge (even insets on all sides).
    #     A designer keeps these hugging the edge at the same inset, whatever the shape.
    borders = []
    for e in els:
        if e["type"] == "text" or e.get("group_id") or abs(norm_rot(e.get("rotation", 0))) > 1:
            continue
        ins = (e["x"], e["y"], cw - e["x"] - e["width"], ch - e["y"] - e["height"])
        outline = (_opaque_frac(e) < 0.35 and _edge_cover(e) > 0.4 and _corners(e)) if e["type"] == "image" else (
            "rgba(0,0,0,0)" in e.get("fill_color", "").replace(" ", "") or (e.get("stroke_width") or 0) > 0
            and re.search(r",\s*0(\.0+)?\)", e.get("fill_color", "")) is not None)
        if outline and min(ins) >= -0.01 * mind and max(ins) <= 0.15 * mind \
                and max(ins) - min(ins) <= 0.05 * mind and _area(e) >= 0.55 * carea:
            borders.append((e, ins))
    for e, ins in borders:
        u = Unit(e["id"], "border", "bg", [e])
        u.insets = ins
        units.append(u)
        used.add(e["id"])

    # 1b. Backgrounds: full-bleed layers at the bottom of the stack
    for e in sorted(els, key=lambda e: e["z_index"]):
        if e["id"] in used:
            continue
        if e["type"] in ("image", "shape") and _area(e) >= 0.85 * carea and not e.get("group_id"):
            if _opaque_frac(e) < 0.3:
                u = Unit(e["id"], "border", "bg", [e])  # full-canvas overlay/frame
                u.insets = (e["x"], e["y"], cw - e["x"] - e["width"], ch - e["y"] - e["height"])
                units.append(u)
            else:
                units.append(Unit(e["id"], "bg", "bg", [e]))
            used.add(e["id"])

    # 2. Decoration: faint layers, and loose shapes that hold no text
    texts_all = [e for e in els if e["type"] == "text" and e["id"] not in used]
    for e in els:
        if e["id"] in used or e["type"] == "text" or e.get("group_id"):
            continue
        faint = (e.get("opacity", 1) or 1) < 0.5
        loose_shape = e["type"] == "shape" and not any(_inside(t, e) for t in texts_all)
        if faint or loose_shape:
            units.append(Unit(e["id"], "deco", "deco", [e]))
            used.add(e["id"])

    # 3. Text roles are global (the headline may sit inside a lockup)
    troles = _text_roles([e for e in texts_all])

    # 3b. Separators: dots, dashes, thin rules. They belong to the text next to
    #     them and ride along with it; they never glue groups together.
    seps = []
    for e in els:
        if e["id"] in used or e["type"] == "text":
            continue
        b = _box(e)
        w, h = b[2] - b[0], b[3] - b[1]
        tiny = w * h < 0.0008 * carea
        thin = min(w, h) < 0.008 * mind and max(w, h) / max(1e-6, min(w, h)) > 5 and w * h < 0.02 * carea
        if thin:
            # a line that spans most of the canvas (and is not inside a card) is
            # page structure (a grid rule), not an ornament of nearby text
            horiz = w >= h
            span = w / cw if horiz else h / ch
            in_card = any(o is not e and o["type"] != "text" and o["id"] not in used
                          and _area(o) > 2 * w * h and _inside(e, o, 0.05) for o in els)
            if span >= 0.8 and not in_card:
                units.append(Unit(e["id"], "rule", "bg", [e]))
                used.add(e["id"])
                continue
        if tiny or thin:
            seps.append(e)
            used.add(e["id"])

    # 3c. Edge-anchored pieces: non-text elements that sit in a canvas corner
    #     (touching two adjacent edges). A designer keeps them in that corner.
    tol = 0.012 * mind
    for e in els:
        if e["id"] in used or e["type"] == "text" or abs(norm_rot(e.get("rotation", 0))) > 1:
            continue
        b = _box(e)
        touch_l, touch_t = b[0] <= tol, b[1] <= tol
        touch_r, touch_b = b[2] >= cw - tol, b[3] >= ch - tol
        corner = (touch_l or touch_r) and (touch_t or touch_b) and not (touch_l and touch_r) and not (touch_t and touch_b)
        # a corner card carrying text stays in its corner and its text travels
        # with it; in the loose reading it is an ordinary container instead
        def _on(t):
            cx, cy = t["x"] + t["width"] / 2, t["y"] + t["height"] / 2
            return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
        riders = [t for t in els if t["type"] == "text" and t["id"] not in used and _on(t)]
        if corner and riders and not compose:
            continue
        if corner and 0.03 * carea <= _area(e) <= 0.5 * carea:
            u = Unit(e["id"], "anchored", "group" if riders else "image", [e] + riders)
            for t in riders:
                used.add(t["id"])
            u.edges = ("l" if touch_l else "r") + ("t" if touch_t else "b")
            units.append(u)
            used.add(e["id"])

    # 4. Geometric grouping: what travels together
    cand = [e for e in els if e["id"] not in used]
    for e in list(cand):
        if e["type"] == "image" and saliency.asset_file(e.get("asset_id")) is None:
            units.append(Unit(e["id"], "missing", "image", [e]))
            used.add(e["id"])
            cand.remove(e)
    singles_img = []
    on_scene = any(u.role == "bg" and u.els[0]["type"] == "image"
                   and (saliency.analyze(u.els[0].get("asset_id") or "") or {}).get("busyness", 0) > 0.03
                   for u in units)
    for members in _cluster(cand, carea, mind, cw, ch, compose, on_scene):
        for e in members:
            used.add(e["id"])
        if len(members) == 1:
            e = members[0]
            if e["type"] == "text":
                units.append(Unit(e["id"], troles.get(e["id"], "body"), "text", [e]))
            elif e["type"] == "image":
                singles_img.append(e)
            else:
                units.append(Unit(e["id"], "deco", "deco", [e]))
            continue
        units.append(_group_unit(members, troles, carea, mind))

    # 5. Single images: missing / logo / hero / minor
    real = []
    for e in singles_img:
        if saliency.asset_file(e.get("asset_id")) is None:
            units.append(Unit(e["id"], "missing", "image", [e]))
        else:
            real.append(e)
    logo_ids = set()
    for e in real:
        aid = (e.get("asset_id") or "").lower()
        if "logo" in aid or "wordmark" in aid or "lockup" in aid:
            logo_ids.add(e["id"])
    if not logo_ids and not any(u.role == "logo" for u in units):
        cands = [e for e in real if e["width"] / max(1, e["height"]) >= 1.8
                 and _area(e) < 0.08 * carea and e["y"] < 0.25 * ch]
        if cands:
            logo_ids.add(min(cands, key=lambda e: e["y"])["id"])
    rest = [e for e in real if e["id"] not in logo_ids]
    hero_id = max(rest, key=_area)["id"] if rest else None
    for e in real:
        if e["id"] in logo_ids:
            role = "logo"
        elif e["id"] == hero_id and _area(e) >= 0.04 * carea:
            role = "hero"
        else:
            role = "minor"
        units.append(Unit(e["id"], role, "image", [e]))

    # 6. Buttons: CTA vs badges (clusters of similar pills form a badge row)
    buttons = [u for u in units if u.kind == "button"]
    clusters: list[list[Unit]] = []
    for b in sorted(buttons, key=lambda u: u.bbox[0]):
        for c in clusters:
            ref = c[0]
            if abs(b.h - ref.h) <= 0.2 * ref.h and abs(b.cy - ref.cy) <= 0.5 * ref.h:
                c.append(b)
                break
        else:
            clusters.append([b])
    has_text_cta = any(u.role == "cta" and u.kind == "text" for u in units)
    singles = [c[0] for c in clusters if len(c) == 1]
    cta_unit = None
    if singles and not has_text_cta:
        def cta_score(u):
            t = u.text_el()
            kw = 1 if (t and CTA_WORDS.search(t.get("content", ""))) else 0
            return (kw, u.w * u.h)
        cta_unit = max(singles, key=cta_score)
    for c in clusters:
        if len(c) >= 2:
            for b in c:
                units.remove(b)
                b.role = "badge"
            units.append(Unit("badges:" + c[0].uid, "badges", "badges", [], members=c))
        else:
            c[0].role = "cta" if c[0] is cta_unit else "badges"

    # 7. Hero visual and hero text by visual weight (inked area). Every
    #    composition gets one of each; multi-piece illustrations qualify.
    def ink_w(e):
        b = _box(e)
        a = (b[2] - b[0]) * (b[3] - b[1]) / carea
        if e["type"] == "text":
            return 2.0 * a
        if e["type"] != "image":
            return a
        info = saliency.analyze(e.get("asset_id") or "")
        if info is None:
            return 0.0
        ab = info["alpha_box"]
        return a * min(1.0, max(0.03, info["opaque_frac"] / max(1e-3, (ab[2] - ab[0]) * (ab[3] - ab[1]))))

    def unit_weight(u, kind):
        els_ = u.els if u.kind != "badges" else [e for m in u.members for e in m.els]
        return sum(ink_w(e) for e in els_ if (e["type"] == "text") == (kind == "text"))

    def pictorial(u):
        """Carries something to look at: at least one image that is not a plain shape."""
        for e in u.els:
            if e["type"] == "image":
                info = saliency.analyze(e.get("asset_id") or "")
                if info is not None and info.get("plain", 1.0) < 0.85:
                    return True
        return False

    for u in units:                     # a plain shape is a surface, not the hero visual
        if u.role == "hero" and not pictorial(u):
            u.role = "minor"
    # a composition that is mostly picture (its text, counted double, under 40% of its
    # ink) is a visual even if it carries a caption
    visual_cands = [u for u in units if u.role in ("hero", "minor", "group", "subhead", "body") and pictorial(u)
                    and u.kind in ("image", "group") and unit_weight(u, "text") <= 0.4 * unit_weight(u, "image")]
    if visual_cands:
        hv = max(visual_cands, key=lambda u: unit_weight(u, "image"))
        tot_img = sum(unit_weight(u, "image") for u in visual_cands)
        if unit_weight(hv, "image") >= 0.03 and unit_weight(hv, "image") >= 0.35 * tot_img:
            for u in units:
                if u.role == "hero" and u is not hv:
                    u.role = "minor"
            hv.role = "hero"
    # Decoration is decided by what a unit is, not how many pieces it has: a
    # text-free group that is not the hero visual or a logo is decoration,
    # exactly like a lone small image.
    for u in units:
        if u.role == "group" and u.kind == "group" and not any(e["type"] == "text" for e in u.els):
            u.role = "minor"
    text_units = [u for u in units if u.kind == "text" and u.role in ("headline", "subhead", "body")]
    if text_units:
        def tw(u):
            e = u.els[0]
            return ink_w(e) * (e["font_size"] / max(1.0, max(t.els[0]["font_size"] for t in text_units))) ** 0.5
        ht = max(text_units, key=tw)
        cur = next((u for u in text_units if u.role == "headline"), None)
        if cur is not None and ht is not cur and tw(ht) > 1.5 * tw(cur):
            cur.role, ht.role = "subhead", "headline"
            troles[cur.els[0]["id"]], troles[ht.els[0]["id"]] = "subhead", "headline"

    # 8. Separators join the unit of the nearest element. A text that carries
    #    separators (dashed / dotted / ruled type) becomes one fixed piece, so
    #    the ornaments stay exactly where the designer put them.
    def _attach(sp, u):
        if u.kind == "frame":
            for i, ch_ in enumerate(u.children):
                if anchor["id"] in ch_.ids:
                    if ch_.kind in ("text", "image"):
                        u.children[i] = Unit(ch_.uid, ch_.role, "group", ch_.els + [sp])
                    else:
                        ch_.els.append(sp)
                    break
            else:  # anchored to the frame itself: its own small piece inside the frame
                u.children.append(Unit(sp["id"], "minor", "image", [sp]))
            u.els.append(sp)
        elif u.kind in ("text", "image"):
            u.kind = "group"          # now a fixed piece: the element plus its ornaments
            u.els.append(sp)
        elif u.kind == "badges":
            for m in u.members:
                if anchor["id"] in m.ids:
                    m.els.append(sp)
        else:
            u.els.append(sp)

    owner_el = [(e, u) for u in units if u.role not in ("bg", "border", "deco", "missing", "anchored", "rule")
                for e in (u.els if u.kind != "badges" else [x for m in u.members for x in m.els])]
    for sp in seps:
        if not owner_el:
            units.append(Unit(sp["id"], "deco", "deco", [sp]))
            continue
        # prefer the nearest text (a rule under a headline, dots between words);
        # fall back to the nearest element of any kind
        texts_near = [eu for eu in owner_el if eu[0]["type"] == "text" and _gap(sp, eu[0]) <= 0.08 * mind]
        anchor, au = min(texts_near or owner_el, key=lambda eu: _gap(sp, eu[0]))
        if _gap(sp, anchor) > 0.08 * mind:
            units.append(Unit(sp["id"], "deco", "deco", [sp]))   # a stray speck: decoration
            continue
        _attach(sp, au)

    deco_ids = {e["id"] for u in units if u.role in ("bg", "border", "deco", "missing", "rule") for e in u.els}

    # 8b. Alignment inheritance: a small piece right above or below a text unit,
    #     on the same centre / left / right line, stays on that line.
    for u in units:
        if u.role != "minor" or u.kind not in ("image", "group"):
            continue
        if any(abs(norm_rot(e.get("rotation", 0))) > 1 for e in u.els):
            continue
        hit = _aligned_anchor(u, units, cw, mind)
        if hit:
            u.role, u.anchor_uid, u.align_line = "aligned", hit[0].uid, hit[1]

    # 8c. A hero image that fills its box edge to edge (photo or full-frame
    #     illustration) may be cropped to fill a panel of the canvas: it is a
    #     window onto something, not an object with a silhouette.
    for u in units:
        if u.role == "hero" and u.kind == "image":
            info = saliency.analyze(u.els[0].get("asset_id") or "")
            # cropping shows part of a picture: it needs detail to show (a flat
            # colour block is a card, not a picture)
            u.croppable = _fills_box(u.els[0]) and abs(norm_rot(u.els[0].get("rotation", 0))) < 1 \
                and info is not None and info["busyness"] > 0.03 and info.get("plain", 1.0) < 0.85
            # a hero that bleeds off three or more edges and fills much of the
            # canvas is the scene everything sits in: it may become a backdrop
            x0, y0, x1, y1 = _box(u.els[0])
            t_ = 0.01 * mind
            edges = (x0 <= t_) + (y0 <= t_) + (x1 >= cw - t_) + (y1 >= ch - t_)
            u.backdrop = u.croppable and edges >= 3 and _area(u.els[0]) >= 0.4 * carea
            if u.backdrop:
                # how much of the photo the designer covered with opaque pieces
                ph = u.els[0]
                over = [_ink_box(e) for e in els if e["type"] != "text" and e is not ph
                        and e["z_index"] > ph["z_index"] and e["id"] not in deco_ids]
                u.src_covered = coverage(over, (max(0, x0), max(0, y0), min(cw, x1), min(ch, y1)))

    for u in units:
        u.prio = PRIORITY.get(u.role, 0)

    # 9. Which units sit on the background photo's subject in the source
    photos = [u for u in units if u.role == "bg" and u.els[0]["type"] == "image"]
    infos = [(u, saliency.analyze(u.els[0].get("asset_id") or "")) for u in photos]
    infos = [(u, i) for u, i in infos if i is not None]
    if infos:
        bu, info = max(infos, key=lambda ui: ui[1]["busyness"])
        sb = info["sal_box"]
        if (sb[2] - sb[0]) * (sb[3] - sb[1]) < 0.45:
            e = bu.els[0]
            fx = e["x"] + info["centroid"][0] * e["width"]
            fy = e["y"] + info["centroid"][1] * e["height"]
            for u in units:
                if u.role in ("bg", "border", "deco", "missing", "separator", "rule"):
                    continue
                b = u.bbox
                u.on_subject = b[0] <= fx <= b[2] and b[1] <= fy <= b[3] and u.w * u.h < 0.6 * carea

    content = [u for u in units if u.role not in ("bg", "border", "deco", "missing", "rule")]
    centered = False
    if content:
        # area-weighted horizontal center of the content, and whether the key units sit on the axis
        tot = sum(max(1.0, u.w * u.h) for u in content)
        cx = sum((u.bbox[0] + u.bbox[2]) / 2 * max(1.0, u.w * u.h) for u in content) / tot
        key = [u for u in content if u.role in ("headline", "logo", "cta", "subhead")] or content
        on_axis = all(abs((u.bbox[0] + u.bbox[2]) / 2 - cw / 2) < 0.08 * cw for u in key)
        centered = abs(cx - cw / 2) < 0.06 * cw and on_axis
    # how much of the canvas the designer's content spanned (decoration excluded)
    core = [u for u in content if u.role not in ("minor", "anchored", "ribbon", "aligned")]
    src_fill = 0.55
    if core:
        bx0 = max(0.0, min(u.bbox[0] for u in core)); by0 = max(0.0, min(u.bbox[1] for u in core))
        bx1 = min(cw, max(u.bbox[2] for u in core)); by1 = min(ch, max(u.bbox[3] for u in core))
        src_fill = max(0.0, bx1 - bx0) * max(0.0, by1 - by0) / carea
    return {"canvas": (cw, ch), "units": units, "src_fill": src_fill, "els": {e["id"]: e for e in els}, "troles": troles,
            "separators": [e["id"] for e in seps],
            "centered": centered}
