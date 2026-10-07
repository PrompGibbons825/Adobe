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
    return max(0.0, e["width"]) * max(0.0, e["height"])


def _inside(inner, outer, tol=0.15):
    ix0, iy0 = inner["x"], inner["y"]
    ix1, iy1 = ix0 + inner["width"], iy0 + inner["height"]
    ox0, oy0 = outer["x"], outer["y"]
    ox1, oy1 = ox0 + outer["width"], oy0 + outer["height"]
    tw, th = inner["width"] * tol, inner["height"] * tol
    return ix0 >= ox0 - tw and iy0 >= oy0 - th and ix1 <= ox1 + tw and iy1 <= oy1 + th


def analyze(source: dict) -> dict:
    cw, ch = source["canvas"]["width"], source["canvas"]["height"]
    els = [dict(e) for e in source["elements"]]
    for i, e in enumerate(els):
        e.setdefault("z_index", i)
    carea = cw * ch
    units: list[Unit] = []
    used = set()

    # 1. Backgrounds: full-bleed images / shapes at the bottom of the stack
    for e in sorted(els, key=lambda e: e["z_index"]):
        if e["type"] in ("image", "shape") and _area(e) >= 0.85 * carea and not e.get("group_id"):
            units.append(Unit(e["id"], "bg", "bg", [e]))
            used.add(e["id"])

    # 2. Explicit groups
    groups: dict[str, list] = {}
    for e in els:
        if e["id"] in used:
            continue
        g = e.get("group_id")
        if g:
            groups.setdefault(g, []).append(e)
    for g, members in groups.items():
        for e in members:
            used.add(e["id"])
        rotated = any(abs(e.get("rotation", 0) or 0) > 1 for e in members)
        has_shape = any(m["type"] == "shape" for m in members)
        has_text = any(m["type"] == "text" for m in members)
        if rotated:
            kind, role = "ribbon", "ribbon"
        elif has_shape and has_text and len(members) == 2:
            kind, role = "button", "button"
        else:
            kind, role = "group", "group"
        units.append(Unit(g, role, kind, members))

    # 3. Implicit containers: ungrouped text sitting inside an ungrouped small shape
    free_shapes = [e for e in els if e["type"] == "shape" and e["id"] not in used
                   and _area(e) < 0.4 * carea and (e.get("opacity", 1) or 1) > 0.5]
    for s in free_shapes:
        inner = [t for t in els if t["type"] == "text" and t["id"] not in used and _inside(t, s)]
        if len(inner) == 1:
            used.update([s["id"], inner[0]["id"]])
            units.append(Unit(s["id"] + "+", "button", "button", [s, inner[0]]))

    # 4. Remaining shapes -> decoration
    for e in els:
        if e["id"] not in used and e["type"] == "shape":
            used.add(e["id"])
            units.append(Unit(e["id"], "deco", "deco", [e]))

    # 5. Images
    imgs = [e for e in els if e["id"] not in used and e["type"] == "image"]
    for e in imgs:
        used.add(e["id"])
    real = []
    for e in imgs:
        if saliency.asset_file(e.get("asset_id")) is None:
            units.append(Unit(e["id"], "missing", "image", [e]))
        else:
            real.append(e)
    logo_ids = set()
    for e in real:
        aid = (e.get("asset_id") or "").lower()
        if "logo" in aid or "wordmark" in aid or "lockup" in aid:
            logo_ids.add(e["id"])
    if not logo_ids:
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

    # 6. Texts
    texts = [e for e in els if e["id"] not in used and e["type"] == "text"]
    for e in texts:
        used.add(e["id"])
    text_units = [Unit(e["id"], "body", "text", [e]) for e in texts]
    if text_units:
        def fs(u):
            return u.els[0]["font_size"]
        cta_text = [u for u in text_units if CTA_WORDS.search(u.els[0]["content"])
                    and len(u.els[0]["content"]) <= 40]
        others = [u for u in text_units if u not in cta_text]
        if others:
            head = max(others, key=lambda u: (fs(u), -u.els[0]["y"]))
            head.role = "headline"
            hfs = fs(head)
            rest = sorted([u for u in others if u is not head], key=fs, reverse=True)
            sub_assigned = False
            for u in rest:
                e = u.els[0]
                small = fs(u) <= 0.4 * hfs
                faint = (e.get("opacity", 1) or 1) < 0.8 or "0.6)" in e.get("color", "")
                if small and (len(e["content"]) > 40 or faint) and u is rest[-1]:
                    u.role = "legal"
                elif not sub_assigned:
                    u.role, sub_assigned = "subhead", True
                else:
                    u.role = "body"
        for u in cta_text:
            u.role = "cta"
    units.extend(text_units)

    # 7. Buttons: CTA vs badges (clusters of similar pills form a badge row)
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
    has_text_cta = any(u.role == "cta" for u in units)
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
            bu = Unit("badges:" + c[0].uid, "badges", "badges", [], members=c)
            units.append(bu)
        else:
            c[0].role = "cta" if c[0] is cta_unit else "badges"
            if c[0].role == "badges":
                c[0].kind = "button"

    for u in units:
        u.prio = PRIORITY.get(u.role, 0)

    return {"canvas": (cw, ch), "units": units, "els": {e["id"]: e for e in els}}
