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
    "group": 4, "badges": 3, "ribbon": 2.5, "minor": 1.5, "legal": 1.2, "missing": 0,
}
# Roles that are laid out in the main flow (others float or fill)
FLOW_ROLES = {"headline", "logo", "cta", "hero", "subhead", "body", "group", "badges", "legal"}


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


def _area(e):
    b = _box(e)
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _box(e):
    h = e["height"]
    if e["type"] == "text":
        h = min(h, _ink_h(e))
    return (e["x"], e["y"], e["x"] + e["width"], e["y"] + h)


def _ink_h(e):
    from . import textfit
    try:
        m = textfit.measure(e, max(1, int(e["font_size"])), e["width"])
        return m["height"]
    except Exception:
        return e["height"]


def _inter(a, b):
    A, B = _box(a), _box(b)
    ix = max(0.0, min(A[2], B[2]) - max(A[0], B[0]))
    iy = max(0.0, min(A[3], B[3]) - max(A[1], B[1]))
    return ix * iy


def _gap(a, b):
    A, B = _box(a), _box(b)
    dx = max(0.0, max(A[0], B[0]) - min(A[2], B[2]))
    dy = max(0.0, max(A[1], B[1]) - min(A[3], B[3]))
    return max(dx, dy)


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
    cta = [e for e in texts if CTA_LEAD.match(e["content"]) and len(e["content"]) <= 30]
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


def _cluster(cand, carea, mind):
    """Union-find over elements: designer groups, frames enclosing content,
    overlapping pieces, and small ornaments touching their neighbour."""
    if not cand:
        return []
    uf = _UF([e["id"] for e in cand])
    opaque = {e["id"]: _opaque_frac(e) for e in cand}
    # large *detailed* images (photos) are backdrops, not glue; large *flat*
    # panels (cards, colour blocks) are containers and keep their contents
    def _detailed(e):
        if e["type"] != "image":
            return False
        info = saliency.analyze(e.get("asset_id") or "")
        return info is not None and info["busyness"] > 0.03
    big = {e["id"] for e in cand if e["type"] != "text" and _area(e) > 0.3 * carea
           and opaque[e["id"]] >= 0.3 and _detailed(e)}
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            if a.get("group_id") and a.get("group_id") == b.get("group_id"):
                uf.union(a["id"], b["id"])
                continue
            if a.get("group_id") and b.get("group_id"):
                continue
            if a["id"] in big or b["id"] in big:
                continue
            ta, tb = a["type"] == "text", b["type"] == "text"
            inter = _inter(a, b)
            small = min(_area(a), _area(b))
            for f, o in ((a, b), (b, a)):
                if f["type"] != "text" and _area(f) > _area(o) * 1.5 and _inside(o, f):
                    uf.union(a["id"], b["id"])
                    break
            else:
                if ta and tb:
                    if inter >= 0.2 * small:
                        uf.union(a["id"], b["id"])
                elif inter >= 0.15 * small:
                    uf.union(a["id"], b["id"])
                elif not (ta and tb) and (
                        (_gap(a, b) <= 0.012 * mind and small < 0.015 * carea)
                        or (_gap(a, b) <= 0.025 * mind and small < 0.06 * carea
                            and min(_ink_density(a), _ink_density(b)) < 0.4)):
                    uf.union(a["id"], b["id"])  # ornaments / line-art pieces of a lockup
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
        if f["type"] == "text" or abs(f.get("rotation", 0) or 0) > 1:
            continue
        others = [o for o in members if o is not f]
        inside = [o for o in others if _area(f) > _area(o) * 1.5 and _inside(o, f)]
        if others and len(inside) >= max(1, 0.7 * len(others)) and len(inside) == len(others):
            if best is None or _area(f) > _area(best):
                best = f
    return best


def _group_unit(members, troles, carea, mind, depth=0):
    gid = members[0].get("group_id") or ("cl_" + members[0]["id"])
    txt = [m for m in members if m["type"] == "text"]
    non = [m for m in members if m["type"] != "text"]
    rotated = any(abs(m.get("rotation", 0) or 0) > 1 for m in members)
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
    if cont is not None and len(members) >= 3:
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


def analyze(source: dict) -> dict:
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
        if e["type"] == "text" or e.get("group_id") or abs(e.get("rotation", 0) or 0) > 1:
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
        if tiny or thin:
            seps.append(e)
            used.add(e["id"])

    # 3c. Edge-anchored pieces: non-text elements that sit in a canvas corner
    #     (touching two adjacent edges). A designer keeps them in that corner.
    tol = 0.012 * mind
    for e in els:
        if e["id"] in used or e["type"] == "text" or abs(e.get("rotation", 0) or 0) > 1:
            continue
        b = _box(e)
        touch_l, touch_t = b[0] <= tol, b[1] <= tol
        touch_r, touch_b = b[2] >= cw - tol, b[3] >= ch - tol
        corner = (touch_l or touch_r) and (touch_t or touch_b) and not (touch_l and touch_r) and not (touch_t and touch_b)
        if corner and 0.03 * carea <= _area(e) <= 0.5 * carea:
            u = Unit(e["id"], "anchored", "image", [e])
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
    for members in _cluster(cand, carea, mind):
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

    visual_cands = [u for u in units if u.role in ("hero", "minor", "group")
                    and u.kind in ("image", "group") and unit_weight(u, "text") <= 0.25 * unit_weight(u, "image")]
    if visual_cands:
        hv = max(visual_cands, key=lambda u: unit_weight(u, "image"))
        tot_img = sum(unit_weight(u, "image") for u in visual_cands)
        if unit_weight(hv, "image") >= 0.03 and unit_weight(hv, "image") >= 0.35 * tot_img:
            for u in units:
                if u.role == "hero" and u is not hv:
                    u.role = "minor"
            hv.role = "hero"
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

    owner_el = [(e, u) for u in units if u.role not in ("bg", "border", "deco", "missing", "anchored")
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
                if u.role in ("bg", "border", "deco", "missing", "separator"):
                    continue
                b = u.bbox
                u.on_subject = b[0] <= fx <= b[2] and b[1] <= fy <= b[3] and u.w * u.h < 0.6 * carea

    content = [u for u in units if u.role not in ("bg", "border", "deco", "missing")]
    centered = False
    if content:
        # area-weighted horizontal center of the content, and whether the key units sit on the axis
        tot = sum(max(1.0, u.w * u.h) for u in content)
        cx = sum((u.bbox[0] + u.bbox[2]) / 2 * max(1.0, u.w * u.h) for u in content) / tot
        key = [u for u in content if u.role in ("headline", "logo", "cta", "subhead")] or content
        on_axis = all(abs((u.bbox[0] + u.bbox[2]) / 2 - cw / 2) < 0.08 * cw for u in key)
        centered = abs(cx - cw / 2) < 0.06 * cw and on_axis
    return {"canvas": (cw, ch), "units": units, "els": {e["id"]: e for e in els}, "troles": troles,
            "separators": [e["id"] for e in seps],
            "centered": centered}
