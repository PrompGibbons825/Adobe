"""Synthetic designs aimed at the finicky parts of the problem.

Each case is built from the starter brand kit (the hidden designs use the same
kit) and names the one thing it probes: renderer quirks in text measurement,
rotation, bleed, translucent layers, deliberate overlaps, missing assets,
malformed input, extreme targets. A case also says what must survive every
resize (``show``) and which elements must keep their arrangement relative to
each other (``together``).

    .venv/bin/python -m tests.finicky --write tests/out/cases    # dump as task JSON for tools.sheet
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from engine.smart.geom import el_box, el_visual_box
from engine.smart.textfit import drawn_box
from tools.make_cases import GOLD, ORANGE, PLUM, WHITE, button, text
from tools.sheet import CATALOGUE

Check = Callable[[dict, dict, dict], List[str]]

# Target sets, by the kind of trouble they cause
ROOMY = ("ig-square", "ig-portrait", "story", "fb-link", "yt-thumb", "pinterest", "pres-16x9")
STRIPS = ("leaderboard", "mobile-banner", "large-leader", "billboard", "li-banner", "fb-cover")
TALL = ("skyscraper", "half-page")
SMALL = ("med-rect", "email-header")
SPREAD = ("ig-square", "story", "fb-link", "leaderboard", "mobile-banner", "billboard", "skyscraper",
          "med-rect", "a4")
ROOMY_SIDE = 200  # below this (strips) only the core message has to survive


@dataclass
class Case:
    name: str
    probe: str
    W: int
    H: int
    elements: List[dict]
    targets: Sequence[str] = SPREAD
    show: Tuple[str, ...] = ()  # visible on every target
    show_roomy: Tuple[str, ...] = ()  # visible wherever the short side is at least ROOMY_SIDE
    together: Tuple[Tuple[str, str], ...] = ()
    check: Optional[Check] = None
    background: bool = True
    bg_fit: str = "cover"  # the source background's fit
    # Targets on which the current engine is KNOWN to fail this case (a failure class found by the
    # loop, not fixed yet). The suite reports them as XFAIL and stays green; once the engine
    # passes one, the run reports XPASS and fails, so the mark is removed with the fix.
    xfail: Tuple[str, ...] = ()
    # Resource budget for one resize (engine robustness, r10): wall seconds and growth of the
    # process's peak memory. None = not checked.
    seconds: Optional[float] = None
    max_mb: Optional[float] = None

    @property
    def source(self) -> dict:
        els = [dict(e) for e in self.elements]
        if self.background:
            els.insert(0, {"id": "bg", "type": "image", "x": 0, "y": 0, "width": self.W, "height": self.H,
                           "z_index": 0, "asset_id": "solstice_bg_gradient.png", "fit": self.bg_fit})
        return {"canvas": {"width": self.W, "height": self.H}, "elements": els}

    def target(self, name: str) -> dict:
        w, h = _size(name)
        return {"width": w, "height": h}

    def must_show(self, name: str) -> Tuple[str, ...]:
        return self.show + (self.show_roomy if min(_size(name)) >= ROOMY_SIDE else ())


def _size(name: str) -> Tuple[int, int]:
    """A catalogue format, or a bare 'WxH' target."""
    if name in CATALOGUE:
        return CATALOGUE[name]
    w, h = name.split("x")
    return int(w), int(h)


def image(id, x, y, w, h, asset="solstice_bottle.png", z=2, **kw) -> dict:
    return {"id": id, "type": "image", "x": x, "y": y, "width": w, "height": h, "z_index": z,
            "asset_id": asset, "fit": "contain", **kw}


def logo(x, y, w=300, h=100, z=3) -> dict:
    return image("logo", x, y, w, h, "solstice_logo_lockup.png", z)


def shape(id, x, y, w, h, fill, z=1, kind="rectangle", **kw) -> dict:
    return {"id": id, "type": "shape", "x": x, "y": y, "width": w, "height": h, "z_index": z,
            "shape_type": kind, "fill_color": fill, **kw}


# ----------------------------------------------------------------------------
# extra checks


def _box(e: dict) -> Tuple[float, float, float, float]:
    return e["x"], e["y"], e["x"] + e["width"], e["y"] + e["height"]


def full_bleed(*ids: str) -> Check:
    """These elements still cover the whole canvas."""
    def check(src, out, tgt):
        errs = []
        for e in out["elements"]:
            if e["id"] in ids:
                x0, y0, x1, y1 = _box(e)
                if x0 > 0.5 or y0 > 0.5 or x1 < tgt["width"] - 0.5 or y1 < tgt["height"] - 0.5:
                    errs.append(f"{e['id']}: no longer full-bleed ({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})")
        return errs
    return check


def focal_on_canvas(pid: str, fx: float, fy: float) -> Check:
    """A photo positioned past the canvas (bleed, S4/S5) keeps its subject, at (fx, fy) of the
    photo's box, on the canvas. A box kept on the canvas is cropped by the renderer, not by the
    engine, and is not checked here (S5 tightens this once the engine chooses crop windows)."""
    def check(src, out, tgt):
        e = next((x for x in out["elements"] if x["id"] == pid), None)
        if e is None or float(e.get("opacity", 1)) <= 0:
            return []
        x0, y0, x1, y1 = _box(e)
        if x0 >= -0.6 and y0 >= -0.6 and x1 <= tgt["width"] + 0.6 and y1 <= tgt["height"] + 0.6:
            return []
        px, py = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
        if not (0 <= px <= tgt["width"] and 0 <= py <= tgt["height"]):
            return [f"{pid}: bled so far its subject ({px:.0f},{py:.0f}) is off the canvas"]
        return []
    return check


def same_size(*ids: str, tol: float = 0.12) -> Check:
    """Alike elements (a set of badges, two CTAs) stay alike in size."""
    def check(src, out, tgt):
        hs = [e["height"] for e in out["elements"] if e["id"] in ids and float(e.get("opacity", 1)) > 0]
        if len(hs) > 1 and max(hs) > (1 + tol) * min(hs):
            return [f"{ids}: heights {', '.join(f'{h:.0f}' for h in hs)} no longer match"]
        return []
    return check


def whole_units(*units: Tuple[str, ...]) -> Check:
    """Each unit is shown or hidden as a whole, never in part; on a roomy target
    at least one shows."""
    def check(src, out, tgt):
        shown = {e["id"]: float(e.get("opacity", 1)) > 0 for e in out["elements"]}
        errs = [f"{u}: partly hidden" for u in units if len({shown[i] for i in u}) > 1]
        if min(tgt.values()) >= ROOMY_SIDE and not any(shown[u[0]] for u in units):
            errs.append("every unit hidden")
        return errs
    return check


def balanced(tol: float = 0.12) -> Check:
    """A design whose content was centred across the canvas still is, whatever
    its text alignment, on every target that reflows it into a column."""
    def check(src, out, tgt):
        W = tgt["width"]
        if W / tgt["height"] > 2.2:  # strips are rows; their spread is the row's business
            return []
        boxes = [drawn_box(e) if e["type"] == "text" else el_visual_box(e) for e in out["elements"]
                 if e["id"] != "bg" and float(e.get("opacity", 1)) > 0 and float(e.get("font_size", 2) or 2) > 1]
        if not boxes:
            return []
        cx = (min(b.x for b in boxes) + max(b.r for b in boxes)) / 2.0
        if abs(cx - W / 2.0) > tol * W:
            return [f"content centred at {cx / W:.2f} of the width, not the middle"]
        return []
    return check


def stays_under(top: str, under: str) -> Check:
    """``under`` reads straight after ``top``: just below it (within a line of
    its own type), sharing its column. Re-wrapping either is fine."""
    def check(src, out, tgt):
        oo = {e["id"]: e for e in out["elements"]}
        a, b = oo[top], oo[under]
        if float(a.get("opacity", 1)) <= 0 or float(b.get("opacity", 1)) <= 0:
            return []
        A, B = drawn_box(a), drawn_box(b)
        gap = B.y - A.b
        share = min(A.r, B.r) - max(A.x, B.x)
        if not (-2.0 <= gap <= 1.5 * float(b["font_size"])) or share < 0.5 * min(A.w, B.w):
            return [f"{under} no longer reads after {top} (gap {gap:.0f} px, shared width {share:.0f} px)"]
        return []
    return check


def leads_strips(first: str, head: str = "headline") -> Check:
    """On a strip, ``first`` leads: it opens the row (nothing visible starts left
    of it) or heads the copy, set right above ``head``."""
    def check(src, out, tgt):
        if tgt["width"] / tgt["height"] < 3.0:
            return []
        canvas = tgt["width"] * tgt["height"]
        shown = [e for e in out["elements"] if float(e.get("opacity", 1)) > 0 and float(e.get("font_size", 2) or 2) > 1
                 and e["width"] * e["height"] < 0.85 * canvas]  # a background opens nothing
        boxes = {e["id"]: (drawn_box(e) if e["type"] == "text" else el_visual_box(e)) for e in shown}
        if first not in boxes:
            return [f"{first} hidden on a strip"]
        lead = min(boxes, key=lambda k: boxes[k].x)
        f, h = boxes[first], boxes.get(head)
        heads_copy = h is not None and f.b <= h.y + 1.0 and min(f.r, h.r) - max(f.x, h.x) > 0
        return [] if lead == first or heads_copy else [f"{lead} opens the row and {first} does not head the copy"]
    return check


def sticks_to(sticker: str, art: str) -> Check:
    """A shown sticker sits on its art (its centre within the art's box, give
    or take a quarter); it may be left out where it could not be read."""
    def check(src, out, tgt):
        oo = {e["id"]: e for e in out["elements"]}
        s_, a = oo[sticker], oo[art]
        if float(s_.get("opacity", 1)) <= 0 or float(s_.get("font_size", 2) or 2) <= 1:
            return []
        if float(a.get("opacity", 1)) <= 0:
            return [f"{sticker} shown without {art}"]
        b = drawn_box(s_)
        box = el_box(a)
        mx, my = 0.25 * box.w, 0.25 * box.h
        if not (box.x - mx <= b.cx <= box.r + mx and box.y - my <= b.cy <= box.b + my):
            return [f"{sticker} came off {art}"]
        return []
    return check


def stays_round(shape_id: str, tol: float = 0.05) -> Check:
    """A circle (a sticker, a round badge) is still a circle wherever it shows."""
    def check(src, out, tgt):
        e = next(e for e in out["elements"] if e["id"] == shape_id)
        if float(e.get("opacity", 1)) <= 0:
            return []
        r = e["width"] / max(1e-6, e["height"])
        return [] if abs(r - 1.0) <= tol else [f"{shape_id} is no longer round ({e['width']:.0f}x{e['height']:.0f})"]
    return check


def framed(panel: str, members: Tuple[str, ...], least: float = 4.0) -> Check:
    """A card keeps padding around what it frames, and one that stood free of
    the canvas edges in the source still does (no run-off, no clipped corners)."""
    def check(src, out, tgt):
        oo = {e["id"]: e for e in out["elements"]}
        p_ = oo[panel]
        if float(p_.get("opacity", 1)) <= 0:
            return []
        pb = el_box(p_)
        errs = []
        for m in members:
            e = oo[m]
            if float(e.get("opacity", 1)) <= 0 or float(e.get("font_size", 2) or 2) <= 1:
                continue
            b = drawn_box(e) if e["type"] == "text" else el_box(e)
            if min(b.x - pb.x, b.y - pb.y, pb.r - b.r, pb.b - b.b) < least:
                errs.append(f"{m} sits within {least:.0f} px of the {panel}'s edge")
        if pb.x <= 0.5 or pb.y <= 0.5 or pb.r >= tgt["width"] - 0.5 or pb.b >= tgt["height"] - 0.5:
            errs.append(f"{panel} runs into the canvas edge")
        return errs
    return check


def never_stretched(*ids: str, tol: float = 0.05) -> Check:
    """An image with fit "fill" is drawn stretched to its box, so its box must keep the source
    box's aspect ratio (within tol). A full-bleed background that must cover a different shape
    has to become fit "cover" (cropped) instead."""
    def check(src, out, tgt):
        so = {e["id"]: e for e in src["elements"]}
        errs = []
        for e in out["elements"]:
            if e["id"] in ids and e.get("fit") == "fill" and float(e.get("opacity", 1)) > 0:
                a0 = so[e["id"]]["width"] / so[e["id"]]["height"]
                a1 = e["width"] / max(1e-6, e["height"])
                if abs(a1 / a0 - 1) > tol:
                    errs.append(f"{e['id']}: fit fill stretched, aspect {a0:.2f} -> {a1:.2f}")
        return errs
    return check


def stays_band(*ids: str) -> Check:
    """A band that ran edge to edge in the source still runs edge to edge (one axis, both
    sides) on the target: it does not shrink into a floating card or an L shape."""
    def check(src, out, tgt):
        errs = []
        for e in out["elements"]:
            if e["id"] in ids and float(e.get("opacity", 1)) > 0:
                x0, y0, x1, y1 = _box(e)
                W, H = tgt["width"], tgt["height"]
                across = x0 <= 0.5 and x1 >= W - 0.5
                down = y0 <= 0.5 and y1 >= H - 0.5
                if not (across or down):
                    errs.append(f"{e['id']}: band became a card ({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})")
        return errs
    return check


def at_least_share(pid: str, share: float) -> Check:
    """This element's longer side is at least `share` of the target's shorter side."""
    def check(src, out, tgt):
        e = next(x for x in out["elements"] if x["id"] == pid)
        if float(e.get("opacity", 1)) <= 0:
            return []
        side = max(e["width"], e["height"])
        need = share * min(tgt["width"], tgt["height"])
        return [f"{pid}: {side:.0f} px, wants {need:.0f}"] if side < need else []
    return check


def label_at_least(pid: str, share: float) -> Check:
    """A text's font size is at least `share` of the target's shorter side."""
    def check(src, out, tgt):
        e = next(x for x in out["elements"] if x["id"] == pid)
        need = share * min(tgt["width"], tgt["height"])
        return [f"{pid}: font {e['font_size']:.0f}, wants {need:.0f}"] if e["font_size"] < need else []
    return check


def style_survives(*ids: str) -> Check:
    """Every property of a text except its geometry, size and layer is what the source had
    (colour, weight, style, alignment, line height, letter spacing, content, font)."""
    skip = {"x", "y", "width", "height", "font_size", "z_index", "opacity", "letter_spacing", "rotation"}

    def check(src, out, tgt):
        so = {e["id"]: e for e in src["elements"]}
        errs = []
        for e in out["elements"]:
            if e["id"] in ids and e["id"] in so:
                for k in sorted((set(so[e["id"]]) | set(e)) - skip):
                    if so[e["id"]].get(k) != e.get(k):
                        errs.append(f"{e['id']}: {k} {so[e['id']].get(k)!r} -> {e.get(k)!r}")
                s_ = so[e["id"]]  # letter spacing is in px, so it scales with the type
                r0 = float(s_.get("letter_spacing") or 0) / max(1.0, float(s_["font_size"]))
                r1 = float(e.get("letter_spacing") or 0) / max(1.0, float(e["font_size"]))
                if abs(r1 - r0) > 0.01 + 0.15 * abs(r0):
                    errs.append(f"{e['id']}: letter spacing {r0:.3f} em -> {r1:.3f} em")
                if float(s_.get("rotation") or 0) != float(e.get("rotation") or 0):
                    errs.append(f"{e['id']}: rotation {s_.get('rotation')!r} -> {e.get('rotation')!r}")
        return errs
    return check


def margin_clear(grow: float = 0.05) -> Check:
    """Safe if the renderer sets text a few percent taller than we measured (FreeType vs CoreText):
    with every shown text's height grown by `grow`, texts that did not overlap in the source still
    do not overlap each other, no text leaves the canvas, and a button label stays on its button."""
    def grown(e):
        b = drawn_box(e)
        return b.x, b.y, b.r, b.b + grow * b.h

    def hit(a, b):
        return min(a[2], b[2]) - max(a[0], b[0]) > 0.5 and min(a[3], b[3]) - max(a[1], b[1]) > 0.5

    def check(src, out, tgt):
        so = {e["id"]: e for e in src["elements"]}
        shown = [e for e in out["elements"] if e["type"] == "text" and float(e.get("opacity", 1)) > 0
                 and float(e.get("font_size", 2) or 2) > 1 and e["id"] in so]
        errs = []
        for i, a in enumerate(shown):
            ga = grown(a)
            if ga[3] > tgt["height"] + 0.5:
                errs.append(f"{a['id']}: grown by {grow:.0%} it leaves the canvas bottom")
            for b in shown[i + 1:]:
                if hit(ga, grown(b)) and not hit(grown(so[a["id"]]), grown(so[b["id"]])):
                    errs.append(f"{a['id']}/{b['id']}: touch once text runs {grow:.0%} taller")
            plate = next((x for x in out["elements"] if x["id"] == a["id"].replace("_label", "_bg")
                          and x["id"] != a["id"]), None)
            if plate is not None and float(plate.get("opacity", 1)) > 0 and ga[3] > plate["y"] + plate["height"] + 0.5:
                errs.append(f"{a['id']}: grown by {grow:.0%} it leaves its button")
        return errs
    return check


def all_of(*checks: Check) -> Check:
    def check(src, out, tgt):
        return [e for c in checks for e in c(src, out, tgt)]
    return check


# ----------------------------------------------------------------------------
# the cases


def cases() -> List[Case]:
    return _base_cases() + round5_cases() + round7_cases() + round10_cases() + sprint_b_cases()


def _base_cases() -> List[Case]:
    return [
        Case("unbreakable_words",
             "a word wider than a narrow column: the renderer never breaks inside a word, so the type "
             "must shrink or the column widen; nothing may draw past the canvas",
             1080, 1080, [
                 logo(390, 70),
                 text("headline", 90, 300, 900, 240, "Sonnenuntergangsgetränke für alle", 80, "Solstice Serif", True),
                 text("url", 140, 620, 800, 50, "solstice.com/summer-collection-2026/limited", 30,
                      color="rgba(255,255,255,0.85)"),
                 *button("cta", 340, 760, 400, 90, "Bestellen", 28),
             ], targets=SPREAD + TALL, show=("headline", "cta_label"), show_roomy=("logo",)),

        Case("explicit_newlines",
             "designer line breaks ('\\n') are kept by the renderer's wrap; measurement must honour them "
             "and a single-word line must not be treated as one long line",
             1080, 1350, [
                 logo(390, 60),
                 text("headline", 90, 300, 900, 420, "Chase\nthe\nLight", 120, "Solstice Serif", True),
                 text("subhead", 140, 780, 800, 120, "Botanical sparkling infusions.\nCrafted for golden hour.", 32,
                      color="rgba(255,255,255,0.85)"),
                 *button("cta", 340, 1000, 400, 90, "Shop now", 28),
             ], show=("headline", "cta_label"), show_roomy=("logo",)),

        Case("letter_spaced_caps",
             "letter spacing is drawn but ignored by the renderer's wrap, so a tracked line is wider "
             "than its box; it must still fit the canvas and not hit its neighbours",
             1080, 1080, [
                 text("kicker", 140, 200, 800, 50, "LIMITED SUMMER EDITION", 30, bold=True, color=GOLD,
                      letter_spacing=14),
                 text("headline", 90, 290, 900, 260, "Taste the longest day", 92, "Solstice Serif", True),
                 text("subhead", 160, 600, 760, 80, "Three botanical infusions, only until the equinox.", 28,
                      color="rgba(255,255,255,0.85)", letter_spacing=2),
                 *button("cta", 340, 760, 400, 90, "FIND A STORE", 26),
                 logo(390, 920, 300, 100),
             ], show=("headline", "cta_label"), show_roomy=("logo",)),

        Case("accents_and_symbols",
             "non-ASCII copy: accents, dashes, currency, a symbol the brand fonts may not carry; measured "
             "advances must match the renderer's or wraps differ",
             1080, 1080, [
                 text("headline", 90, 240, 900, 240, "Crème brûlée — l'été en bouteille", 76, "Solstice Serif", True),
                 text("price", 340, 560, 400, 90, "€4,99 · £3.99 · ¥650", 44, bold=True, color=GOLD),
                 text("subhead", 140, 690, 800, 80, "Ünïcödé süßigkeiten ☀ ✦ 🍊", 30, color="rgba(255,255,255,0.85)"),
                 *button("cta", 340, 830, 400, 90, "Jetzt kaufen", 28),
             ], show=("headline", "cta_label"), show_roomy=("price",)),

        Case("blank_and_empty_text",
             "texts with no visible content (empty, spaces, a lone newline) must survive as ids without "
             "crashing measurement or taking up layout room",
             1080, 1080, [
                 logo(390, 80),
                 text("headline", 90, 380, 900, 200, "Chase the Light", 90, "Solstice Serif", True),
                 text("empty", 90, 600, 900, 40, "", 28),
                 text("spaces", 90, 650, 900, 40, "     ", 28),
                 text("newline", 90, 700, 900, 40, "\n", 28),
                 *button("cta", 340, 800, 400, 90, "Shop now", 28),
             ], show=("headline", "cta_label"), show_roomy=("logo",)),

        Case("shadowed_headline",
             "a drop shadow made of a second copy of the headline, offset a few px underneath: the pair "
             "must move as one, or the shadow shows up as a duplicate headline",
             1080, 1080, [
                 logo(390, 70),
                 text("headline_shadow", 96, 346, 900, 220, "Golden Hour Sale", 104, "Solstice Serif", True,
                      color="rgba(0,0,0,0.45)", z=3),
                 text("headline", 90, 340, 900, 220, "Golden Hour Sale", 104, "Solstice Serif", True, z=4),
                 text("subhead", 140, 600, 800, 80, "Every bottle, this weekend only.", 32,
                      color="rgba(255,255,255,0.85)"),
                 *button("cta", 340, 760, 400, 90, "Shop the sale", 28),
             ], show=("headline", "cta_label"), show_roomy=("headline_shadow",),
             together=(("headline", "headline_shadow"),)),

        Case("stacked_discount",
             "display type stacked on purpose ('UP TO' over a huge '50%' over 'OFF'), overlapping in the "
             "source: allowed to overlap, but it must stay one lockup",
             1080, 1350, [
                 logo(390, 60),
                 text("upto", 340, 260, 400, 60, "UP TO", 44, bold=True, color=GOLD),
                 text("big", 190, 300, 700, 330, "50%", 300, "Solstice Serif", True),
                 text("off", 340, 610, 400, 80, "OFF", 64, bold=True, color=GOLD),
                 text("subhead", 140, 760, 800, 80, "On every botanical sparkler, all weekend.", 30,
                      color="rgba(255,255,255,0.85)"),
                 *button("cta", 340, 920, 400, 90, "Shop the sale", 28),
             ], show=("big", "cta_label"), show_roomy=("upto", "off"),
             together=(("upto", "big"), ("big", "off"))),

        Case("price_lockup",
             "a price built from three texts (big dollars, small cents, per-unit): tiny gaps that are "
             "part of the design, not collisions, and must not be torn apart",
             1200, 628, [
                 image("product", 60, 54, 520, 520),
                 text("headline", 640, 80, 520, 140, "Golden hour, bottled.", 54, "Solstice Serif", True, align="left"),
                 text("dollars", 640, 250, 160, 120, "$29", 96, "Solstice Serif", True, align="left", color=GOLD),
                 text("cents", 800, 262, 80, 50, ".99", 40, bold=True, align="left", color=GOLD),
                 text("per", 800, 318, 160, 40, "per case", 24, align="left", color="rgba(255,255,255,0.8)"),
                 *button("cta", 640, 430, 300, 70, "Order now", 24),
             ], show=("dollars", "cta_label"), show_roomy=("headline", "cents"),
             together=(("dollars", "cents"), ("cents", "per"))),

        Case("rotated_everything",
             "a ribbon at -12 degrees in a corner, vertical text at 90 degrees along an edge, a tilted "
             "product: rotated corners must stay on the canvas and off the copy",
             1080, 1350, [
                 logo(390, 60),
                 image("product", 290, 220, 500, 500, rotation=8),
                 *[dict(e, rotation=-12, z_index=6 + i) for i, e in enumerate(
                     button("ribbon", 760, 40, 280, 56, "SUMMER SALE", 22, fill="rgba(220,60,50,1)", radius=0))],
                 text("vertical", -160, 600, 400, 50, "EST. 2026 · SMALL BATCH", 26, bold=True,
                      color="rgba(255,255,255,0.7)", rotation=-90),
                 text("headline", 90, 780, 900, 150, "Summer sale on every bottle", 62, "Solstice Serif", True),
                 *button("cta", 340, 1080, 400, 90, "Shop the sale", 28),
             ], show=("headline", "cta_label"), show_roomy=("logo",)),

        Case("bleeding_decor",
             "translucent circles that bleed off two corners and a big blob behind the product: decoration "
             "must stay decoration (never pushes copy around) and stay on the canvas",
             1080, 1080, [
                 shape("circle_tl", -140, -140, 520, 520, "rgba(255,255,255,0.14)", kind="ellipse"),
                 shape("circle_br", 760, 760, 420, 420, "rgba(240,178,66,0.18)", kind="ellipse"),
                 shape("blob", 240, 180, 600, 600, "rgba(232,106,58,0.22)", kind="ellipse"),
                 logo(390, 60),
                 image("product", 340, 230, 400, 400, z=3),
                 text("headline", 90, 680, 900, 140, "Chase the Light", 76, "Solstice Serif", True),
                 *button("cta", 340, 860, 400, 90, "Shop now", 28),
             ], show=("headline", "cta_label"), show_roomy=("logo", "product")),

        Case("source_bleeds_decor",
             "Adobe's Level 2 move: a decorative circle that bleeds off the top-left corner in the source "
             "(at -100,-100, 64% on canvas) and one fully inside: either may bleed in the output by about "
             "the same proportion (S4), neither may push copy, and the badge and CTA never bleed",
             1080, 1080, [
                 shape("deco_circle_1", -100, -100, 500, 500, "rgba(255,255,255,0.18)", kind="ellipse"),
                 shape("deco_circle_2", 700, 700, 300, 300, "rgba(240,178,66,0.25)", kind="ellipse"),
                 logo(380, 60, 320, 110),
                 image("product", 290, 260, 500, 500),
                 text("headline", 90, 780, 900, 140, "Golden Hour Sparkle", 72, "Solstice Serif", True),
                 *button("badge", 40, 40, 220, 56, "NEW FLAVOUR", 18, fill="rgba(43,27,46,1)"),
                 *button("cta", 330, 950, 420, 90, "Shop now", 28),
             ], show=("headline", "cta_label"), show_roomy=("logo",)),

        Case("focal_photo_background",
             "a full-bleed photo background with its subject off-centre (upper right, at 72%/30% of the "
             "photo) and copy in the quiet lower left: the photo keeps covering the canvas, and if the "
             "engine frames it by bleeding the box past the canvas (S4/S5) the subject stays on it",
             1200, 1500, [
                 text("headline", 80, 980, 700, 180, "Into the long light", 72, "Solstice Serif", True,
                      align="left"),
                 text("subhead", 80, 1170, 640, 70, "Botanical sparkling, made for evenings.", 28, align="left"),
                 *button("cta", 80, 1300, 360, 90, "Discover", 28),
             ], targets=("story", "ig-square", "fb-link", "leaderboard", "billboard", "skyscraper"),
             show=("headline", "cta_label"),
             check=all_of(full_bleed("bg"), focal_on_canvas("bg", 0.72, 0.30))),

        Case("no_background",
             "nothing paints the canvas (the renderer leaves it white) and copy is dark: hidden copy has "
             "no background to hide under, and every visible text needs contrast against white",
             1080, 1080, [
                 logo(390, 60),
                 image("product", 340, 200, 400, 400),
                 text("headline", 90, 640, 900, 120, "Chase the Light", 76, "Solstice Serif", True, color=PLUM),
                 text("subhead", 140, 770, 800, 70, "Botanical sparkling infusions, crafted for golden hour.", 26,
                      color="rgba(43,27,46,0.8)"),
                 *button("cta", 340, 870, 400, 90, "Shop now", 28),
                 text("legal", 60, 1030, 960, 30,
                      "Please drink responsibly. Botanical Sparkling Co. 2026. Void where prohibited.", 14,
                      color="rgba(43,27,46,0.6)"),
             ], background=False, show=("headline", "cta_label"), show_roomy=("logo",)),

        Case("translucent_overlay",
             "a full-canvas dark scrim over the background photo (rgba alpha 0.45): it is part of the "
             "background and must stay full-bleed under the copy, not become a panel or decoration",
             1080, 1350, [
                 shape("scrim", 0, 0, 1080, 1350, "rgba(10,5,15,0.45)", z=1),
                 logo(390, 80),
                 image("product", 290, 250, 500, 500, z=2),
                 text("headline", 90, 820, 900, 160, "After sunset, it sparkles", 66, "Solstice Serif", True),
                 *button("cta", 340, 1080, 400, 90, "Shop now", 28),
             ], show=("headline", "cta_label"), show_roomy=("logo",), check=full_bleed("bg", "scrim")),

        Case("split_background",
             "two flat half-canvas panels form the background (plum left, orange right) with copy on one "
             "and the product on the other: copy must stay on its own colour",
             1200, 628, [
                 shape("left_half", 0, 0, 600, 628, "rgba(43,27,46,1)", z=1),
                 shape("right_half", 600, 0, 600, 628, "rgba(232,106,58,1)", z=1),
                 text("headline", 60, 150, 480, 180, "Two moods, one bottle", 58, "Solstice Serif", True,
                      align="left"),
                 text("subhead", 60, 350, 460, 80, "Blood orange or wild honey.", 26, align="left",
                      color="rgba(255,255,255,0.85)"),
                 *button("cta", 60, 460, 260, 70, "Choose yours", 22, fill=GOLD, color=PLUM),
                 image("product", 700, 64, 400, 500, z=2),
             ], background=False, show=("headline", "cta_label"), show_roomy=("product",)),

        Case("two_ctas",
             "a primary and a secondary button side by side: both are buttons, they must keep the same "
             "height and neither may be dropped on a roomy format",
             1080, 1080, [
                 logo(390, 70),
                 image("product", 340, 210, 400, 400),
                 text("headline", 90, 650, 900, 120, "Chase the Light", 76, "Solstice Serif", True),
                 *button("cta", 150, 840, 360, 90, "Shop now", 28),
                 *button("cta2", 570, 840, 360, 90, "Learn more", 28, fill="rgba(255,255,255,0.16)"),
             ], show=("headline", "cta_label"), show_roomy=("logo",), check=same_size("cta_bg", "cta2_bg")),

        Case("badge_grid",
             "six alike badges in a 3x2 grid (a variable-count set): they must stay alike and in sets, "
             "never be scattered one by one",
             1080, 1350, [
                 logo(390, 60),
                 text("headline", 90, 210, 900, 160, "Everything you love about summer", 64, "Solstice Serif", True),
                 *[e for i, (lab, x, y) in enumerate([
                     ("VEGAN", 110, 470), ("NO SUGAR", 400, 470), ("ORGANIC", 690, 470),
                     ("LOCAL", 110, 560), ("SMALL BATCH", 400, 560), ("ZERO WASTE", 690, 560)])
                   for e in button(f"badge{i}", x, y, 280, 60, lab, 20, fill=PLUM if i % 2 else ORANGE)],
                 image("product", 390, 680, 300, 300),
                 *button("cta", 340, 1100, 400, 90, "Shop now", 28),
             ], show=("headline", "cta_label"), show_roomy=("logo",),
             check=same_size(*[f"badge{i}_bg" for i in range(6)])),

        Case("product_cards",
             "three product cards, each an explicit group of panel + image + name + price: a group that "
             "is not a button must still travel as a unit",
             1200, 628, [
                 text("headline", 60, 40, 1080, 80, "Pick your sunset", 56, "Solstice Serif", True),
                 *[e for i, (x, name, price) in enumerate([(60, "Blood Orange", "$4.99"),
                                                            (440, "Wild Honey", "$5.49"),
                                                            (820, "Smoked Rosemary", "$5.99")])
                   for e in (
                       shape(f"card{i}", x, 150, 320, 420, "rgba(255,255,255,0.10)", z=1, corner_radius=24,
                             group_id=f"card{i}"),
                       image(f"bottle{i}", x + 70, 170, 180, 260, z=2, group_id=f"card{i}"),
                       text(f"name{i}", x + 20, 450, 280, 40, name, 26, bold=True, group_id=f"card{i}"),
                       text(f"price{i}", x + 20, 500, 280, 40, price, 26, color=GOLD, group_id=f"card{i}"))],
             ], show=("headline",), check=whole_units(*[(f"card{i}", f"bottle{i}", f"name{i}", f"price{i}")
                                                         for i in range(3)]),
             together=(("bottle0", "name0"), ("bottle1", "name1"), ("bottle2", "name2"))),

        Case("wall_of_copy",
             "a long body paragraph and legal line in a story: on strips something has to give, but the "
             "headline and CTA never go, and nothing drops to small print",
             1080, 1920, [
                 logo(390, 100),
                 text("headline", 90, 300, 900, 260, "The long summer evening, in a bottle", 80, "Solstice Serif",
                      True),
                 text("body", 120, 640, 840, 560,
                      "Small-batch botanical infusions blended with wild honey, blood orange zest and a whisper "
                      "of smoked rosemary. Bottled at the peak of golden hour for a taste that lingers long after "
                      "sunset. Serve over ice with a slice of orange, or straight from the bottle on the porch. "
                      "Every case ships in recycled packaging from our solar-powered bottling house.", 32,
                      color="rgba(255,255,255,0.85)", align="left"),
                 *button("cta", 290, 1360, 500, 110, "Shop the collection", 32),
                 text("legal", 90, 1800, 900, 60,
                      "Please drink responsibly. Botanical Sparkling Co., 2026. Void where prohibited. "
                      "See solstice.com/terms for full details.", 18, color="rgba(255,255,255,0.6)"),
             ], targets=SPREAD + TALL, show=("headline", "cta_label")),

        Case("tiny_source",
             "a 300x250 ad scaled up to a story and A4: type should grow to suit the canvas instead of a "
             "small design floating in space",
             300, 250, [
                 image("logo", 100, 12, 100, 33, "solstice_logo_lockup.png", 3),
                 image("product", 20, 60, 110, 140),
                 text("headline", 140, 70, 145, 80, "Chase the Light", 22, "Solstice Serif", True, align="left"),
                 *button("cta", 140, 165, 130, 34, "Shop now", 13),
             ], targets=("story", "a4", "ig-square", "billboard", "pres-16x9"),
             show=("headline", "cta_label"), show_roomy=("logo", "product")),

        Case("wide_source",
             "a 728x90 leaderboard rebuilt as tall and square formats: a row must turn into a column",
             728, 90, [
                 image("logo", 12, 15, 150, 60, "solstice_logo_lockup.png", 3),
                 image("product", 170, 5, 60, 80),
                 text("headline", 245, 20, 300, 50, "Chase the Light — new botanical sparklers", 17,
                      "Solstice Serif", True, align="left"),
                 *button("cta", 570, 25, 145, 40, "Shop now", 15),
             ], targets=("ig-square", "story", "ig-portrait", "med-rect", "half-page", "skyscraper"),
             show=("headline", "cta_label"), show_roomy=("logo", "product")),

        Case("missing_hero_asset",
             "the large hero asset is not in the brand kit (the renderer would paint a grey box) next to "
             "a small missing watermark: neither may crash the engine",
             1080, 1080, [
                 logo(390, 60),
                 image("hero", 240, 200, 600, 400, asset="summer_campaign_photo.jpg"),
                 image("watermark", 880, 980, 160, 70, asset="holiday_badge_2025.png", z=5),
                 text("headline", 90, 650, 900, 120, "Chase the Light", 76, "Solstice Serif", True),
                 *button("cta", 340, 850, 400, 90, "Shop now", 28),
             ], show=("headline", "cta_label"), show_roomy=("logo",)),

        Case("thin_rules_and_lines",
             "hairline dividers (a 2px rectangle and a 'line' shape) between copy: they must stay thin "
             "lines, not swell into bars or vanish",
             1080, 1350, [
                 logo(390, 70),
                 text("headline", 90, 260, 900, 200, "Golden hour, bottled", 84, "Solstice Serif", True),
                 shape("rule", 440, 500, 200, 2, GOLD, z=3),
                 text("subhead", 140, 540, 800, 90, "Sparkling botanicals with blood orange and wild honey.", 30,
                      color="rgba(255,255,255,0.85)"),
                 shape("line", 340, 680, 400, 10, GOLD, z=3, kind="line", stroke_color=GOLD, stroke_width=2),
                 *button("cta", 340, 740, 400, 90, "Shop now", 28),
             ], show=("headline", "cta_label"), show_roomy=("logo",), check=lambda s, o, t: _thin("rule", s, o)),

        Case("sticker_on_product",
             "a big round sticker on the product's corner, outside its visible pixels, next to a smaller "
             "pill button: the pill is the CTA, and the sticker travels with the product",
             1280, 720, [
                 text("headline", 60, 240, 560, 150, "Join the club that makes Monday feel good", 52,
                      "Solstice Serif", True, align="left"),
                 *button("cta", 60, 430, 150, 48, "Join now", 18),
                 image("product", 820, 90, 400, 540),
                 *[dict(e, z_index=7 + i) for i, e in enumerate(
                     button("sticker", 1100, 90, 170, 170, "FREE", 40, fill=GOLD, color=PLUM, radius=85))],
             ], show=("headline", "cta_label"), show_roomy=("product",),
             check=all_of(sticks_to("sticker_label", "product"), stays_round("sticker_bg"))),

        Case("left_aligned_strip",
             "a banner set flush left (logo | headline over its subhead | button): as a column its "
             "block belongs in the middle, left-aligned inside, and the subhead stays under the headline "
             "even though only a sliver of space runs between them and gutters set the columns apart",
             1584, 396, [
                 logo(40, 140, 240, 80),
                 text("headline", 340, 40, 720, 260, "Open house this Sunday", 104, "Solstice Serif", True,
                      align="left"),
                 text("subhead", 340, 305, 720, 50, "Call us to book a private tour.", 40, align="left",
                      color="rgba(255,255,255,0.85)"),
                 *button("cta", 1150, 160, 380, 76, "Book a viewing", 30),
             ], targets=("ig-square", "ig-portrait", "ig-landscape", "fb-link", "story", "med-rect", "skyscraper",
                         "half-page"),
             show=("headline", "cta_label"), show_roomy=("subhead",),
             check=all_of(balanced(), stays_under("headline", "subhead"))),

        Case("type_wordmark",
             "the brand set in type (no logo artwork) above the product, as in the brief's reference post: "
             "it is the logo, so it stays on every format and opens the row on a strip; the subhead under "
             "the headline is copy, not small print",
             1080, 1080, [
                 shape("cream", 0, 0, 1080, 1080, "rgba(248,238,222,1)", z=0),
                 text("wordmark", 90, 98, 900, 80, "SOLSTICE", 67, "Solstice Serif", True, color=PLUM),
                 image("product", 286, 144, 507, 507),
                 text("headline", 90, 612, 900, 132, "Chase the Light", 110, "Solstice Serif", True, color=PLUM),
                 text("subhead", 90, 772, 900, 42, "Botanical sparkling infusions", 35,
                      color="rgba(43,27,46,0.75)"),
                 *button("cta", 352, 887, 374, 90, "Shop the Line", 32),
             ], targets=SPREAD + ("large-leader", "li-banner"), show=("headline", "cta_label", "wordmark"),
             show_roomy=("subhead",), check=leads_strips("wordmark"), background=False),

        Case("card_on_canvas",
             "copy on a free-standing rounded card on a light canvas: the card is redrawn around its copy "
             "on every format, keeping its padding, never shrink-wrapped and never run off the canvas",
             1280, 720, [
                 shape("cream", 0, 0, 1280, 720, "rgba(248,238,222,1)", z=0),
                 text("wordmark", 340, 40, 600, 40, "SOLSTICE", 26, "Solstice Serif", True, color=PLUM,
                      letter_spacing=4),
                 shape("card", 220, 150, 840, 300, "rgba(255,255,255,0.95)", z=1, corner_radius=22),
                 text("headline", 300, 190, 680, 90, "Sun Festival", 64, "Solstice Serif", True, color=PLUM),
                 *button("cta", 545, 320, 190, 56, "Learn more", 20),
             ], targets=SPREAD + ("fb-link", "billboard", "half-page"), show=("headline", "cta_label"),
             check=framed("card", ("headline", "cta_label")), background=False),

        Case("lockup_on_photo",
             "a date set as a two-line lockup ('Jan 9' over '5 PM - 6.30 PM') on the corner of a product "
             "photo, beside a headline: the lockup is copy, not a sticker, so it stays on the art wherever "
             "it can be read, and it keeps to the design's type scale when the photo grows",
             1920, 1080, [
                 image("product", 1120, 160, 760, 760),
                 text("date", 1150, 200, 260, 44, "Jan 9", 36, "Solstice Serif", True, z=5, align="left"),
                 text("time", 1150, 244, 260, 30, "5 PM - 6.30 PM", 24, "Solstice Serif", z=5, align="left"),
                 text("headline", 120, 360, 900, 140, "Botanical Tasting", 104, "Solstice Serif", True,
                      align="left"),
                 *button("cta", 120, 560, 360, 96, "Book a seat", 34),
             ], targets=("leaderboard", "billboard", "fb-link", "ig-square", "story"),
             show=("headline", "cta_label"), show_roomy=("date", "time")),

        Case("crowded_l3",
             "the Level 3 load in miniature: a run-on headline, three badges, a ribbon, a legal line and a "
             "missing watermark, to the extreme strips",
             1080, 1350, [
                 logo(380, 50, 320, 100),
                 image("product", 340, 180, 400, 400, z=3),
                 text("headline", 90, 620, 900, 220,
                      "Introducing the All-New Limited-Batch Solstice Botanical Sparkling Collection", 40,
                      "Solstice Serif", True, z=5),
                 *button("badge1", 60, 1020, 290, 50, "LIMITED EDITION", 15),
                 *button("badge2", 370, 1020, 340, 50, "AWARD WINNING 2026", 15, fill=PLUM),
                 *button("badge3", 730, 1020, 260, 50, "SMALL BATCH", 15),
                 *button("cta", 330, 1100, 420, 90, "Shop the Summer Line", 22),
                 *[dict(e, rotation=-12, z_index=6 + i) for i, e in enumerate(
                     button("ribbon", 740, 30, 300, 54, "SUNSET SALE", 20, fill="rgba(220,60,50,1)", radius=0))],
                 image("watermark", 850, 1250, 180, 80, asset="solstice_holiday_badge_2025.png", z=2),
                 text("legal", 60, 1310, 960, 30, "Please drink responsibly. See solstice.com/terms.", 12,
                      color="rgba(255,255,255,0.6)", z=6),
             ], targets=STRIPS + TALL + SMALL, show=("headline", "cta_label"), show_roomy=("logo",)),
    ]


def round5_cases() -> List[Case]:
    """Failure classes named by the gate in rounds 1-4 (loop/weaknesses.md). Several are known
    failures of the engine (``xfail``); each is a general class, built from the brand kit."""
    wide_copy = [
        logo(1250, 90, 300, 100),
        text("headline", 100, 120, 1000, 200, "Taste the longest day", 96, "Solstice Serif", True, align="left"),
        text("subhead", 100, 340, 1000, 80, "Three botanical infusions, until the equinox.", 32,
             color="rgba(255,255,255,0.85)", align="left"),
        *button("cta", 100, 480, 420, 90, "Shop now", 30),
    ]
    sq_copy = [
        logo(390, 70),
        image("product", 340, 200, 400, 520),
        text("headline", 90, 760, 900, 130, "Golden Hour", 84, "Solstice Serif", True),
        *button("cta", 340, 930, 400, 90, "Shop now", 28),
    ]
    band_src = [
        shape("band", 0, 380, 1080, 320, "rgba(40,20,70,0.92)", z=1),
        logo(390, 60),
        text("headline", 90, 410, 900, 160, "Summer sale", 96, "Solstice Serif", True, z=5),
        text("subhead", 140, 580, 800, 70, "Up to 40% off every infusion", 34, z=5),
        *button("cta", 340, 780, 400, 90, "Shop now", 28),
    ]
    return [
        Case("fill_background_wide",
             "a full-bleed background with fit 'fill' (the Crello export default) on a wide design: "
             "re-laid on any other shape it is cropped, never stretched",
             1600, 700, wide_copy, targets=("ig-square", "story", "pinterest", "skyscraper", "half-page",
                                           "leaderboard", "billboard", "med-rect"),
             show=("headline", "cta_label"), bg_fit="fill",
             check=all_of(never_stretched("bg"), full_bleed("bg")),
             ),  # passes today (cropped as cover); kept as a guard

        Case("fill_background_square",
             "a square design whose background is fit 'fill', to strips and tall formats",
             1080, 1080, sq_copy, targets=("story", "leaderboard", "billboard", "skyscraper", "fb-link",
                                           "mobile-banner", "a4"),
             show=("headline", "cta_label"), bg_fit="fill",
             check=all_of(never_stretched("bg"), full_bleed("bg")),
             ),

        Case("band_stays_band_on_squares",
             "a full-width colour band behind the copy: on a square or a small rectangle it stays a band "
             "(edge to edge), not an L shape or a floating card",
             1080, 1350, band_src, targets=("ig-square", "med-rect", "fb-link", "pres-16x9", "email-header",
                                            "ig-portrait"),
             show=("headline", "cta_label"), check=stays_band("band"),
             xfail=("fb-link", "pres-16x9", "email-header")),

        Case("tiny_items_kept_on_small_canvases",
             "a discount sticker and a legal line are small in the source but part of the offer: on a "
             "square and on 300x250 they are kept (smaller type allowed), not hidden",
             1080, 1080, [
                 logo(390, 50, 300, 100),
                 image("product", 320, 190, 440, 520),
                 text("headline", 90, 740, 900, 120, "Summer Sale", 80, "Solstice Serif", True),
                 *button("sticker", 800, 190, 180, 180, "-40%", 40, fill=PLUM, radius=90),
                 *button("cta", 340, 890, 400, 80, "Shop now", 28),
                 text("legal", 90, 1010, 900, 30, "Offer ends 31 August. Terms apply.", 14,
                      color="rgba(255,255,255,0.75)", z=5),
             ], targets=("ig-square", "med-rect", "fb-link", "half-page", "ig-portrait"),
             show=("headline", "cta_label", "sticker_label", "legal"),
             xfail=("med-rect",)),

        Case("product_not_a_thumbnail",
             "one big product and a short headline: on 300x250 and square formats the product keeps a "
             "real share of the canvas instead of shrinking to a thumbnail beside the copy",
             1080, 1350, [
                 logo(390, 50, 300, 100),
                 image("product", 240, 200, 600, 780),
                 text("headline", 90, 1020, 900, 110, "Golden Hour", 84, "Solstice Serif", True),
                 *button("cta", 340, 1180, 400, 90, "Shop now", 28),
             ], targets=("med-rect", "ig-square", "fb-link", "half-page", "email-header"),
             show=("headline", "cta_label", "product"), check=at_least_share("product", 0.5)),

        Case("button_label_not_minimum_on_small",
             "a call to action on 300x250: the label is set at a size a person can read (at least 6% "
             "of the short side), not the renderer's minimum",
             1080, 1080, sq_copy, targets=("med-rect", "email-header"),
             show=("headline", "cta_label"), check=label_at_least("cta_label", 0.06),
             xfail=("med-rect", "email-header")),
    ]


def round7_cases() -> List[Case]:
    """Failure classes from rounds 5-6 (research note 15, loop/weaknesses.md): fill-fit images that
    are not the first element, group splits, FreeType text-height margin, text style that must
    survive. Built from the brand kit, no sealed data. Cases failing on the engine are xfail."""
    fill_art = {"fit": "fill"}
    return [
        Case("fill_bg_odd_id_bleeding",
             "Crello-style background: a fill-fit image with its own id and a layer above zero, a few "
             "px past the canvas on every side; on any shape it is cropped to cover, never stretched",
             1080, 1350, [
                 image("backdrop", -4, -4, 1088, 1358, "solstice_bg_gradient.png", z=0, fit="fill"),
                 logo(390, 70),
                 text("headline", 90, 600, 900, 130, "Golden Hour", 84, "Solstice Serif", True),
                 *button("cta", 340, 800, 400, 90, "Shop now", 28),
             ], targets=("ig-square", "story", "leaderboard", "skyscraper", "pres-16x9", "med-rect", "a4"),
             background=False, show=("headline", "cta_label"), check=never_stretched("backdrop")),

        Case("fill_bg_with_fill_panel",
             "a fill-fit background and a fill-fit photo panel: neither is stretched on any target "
             "(a photo may be cropped or scaled uniformly, never squashed)",
             1600, 900, [
                 image("backdrop", 0, 0, 1600, 900, "solstice_bg_gradient.png", z=0, fit="fill"),
                 image("photo", 100, 120, 640, 660, "solstice_bottle.png", z=2, fit="fill"),
                 text("headline", 820, 260, 700, 160, "Golden Hour", 84, "Solstice Serif", True, align="left"),
                 *button("cta", 820, 520, 380, 90, "Shop now", 28),
             ], targets=("ig-square", "story", "leaderboard", "skyscraper", "fb-link", "med-rect"),
             background=False, show=("headline", "cta_label"),
             check=all_of(never_stretched("backdrop", "photo"), full_bleed("backdrop"))),

        Case("fill_bg_tall_source",
             "a tall story with a fill-fit background re-laid for wide strips and a square",
             1080, 1920, [
                 image("backdrop", 0, 0, 1080, 1920, "solstice_bg_gradient.png", z=0, fit="fill"),
                 logo(390, 120),
                 text("headline", 90, 800, 900, 260, "Summer is here", 100, "Solstice Serif", True),
                 *button("cta", 340, 1400, 400, 100, "Shop now", 30),
             ], targets=("leaderboard", "billboard", "mobile-banner", "ig-square", "pres-16x9", "fb-cover"),
             background=False, show=("headline", "cta_label"),
             check=all_of(never_stretched("backdrop"), full_bleed("backdrop"))),

        Case("plate_text_not_split",
             "a headline set on a rounded plate: wherever the plate shows, the headline stays on it with "
             "padding (the pair is one unit, not two things placed separately)",
             1080, 1080, [
                 logo(390, 60, 300, 100),
                 image("product", 340, 200, 400, 420),
                 shape("plate", 140, 660, 800, 200, "rgba(43,27,46,0.9)", z=3, kind="rectangle", corner_radius=40),
                 text("headline", 180, 690, 720, 140, "Golden Hour", 80, "Solstice Serif", True, z=5),
                 *button("cta", 340, 920, 400, 90, "Shop now", 28),
             ], targets=SPREAD, show=("headline", "cta_label"), together=(("plate", "headline"),),
             check=framed("plate", ("headline",), least=2.0)),

        Case("caption_stays_with_photo",
             "a photo with a caption set right under it, plus a separate headline: the caption keeps "
             "reading straight after its photo (they are never split across the layout)",
             1080, 1350, [
                 image("photo", 200, 140, 680, 640, "solstice_bottle.png"),
                 text("caption", 200, 800, 680, 50, "Citrus and elderflower, bottled at dusk", 28, align="left"),
                 text("headline", 90, 960, 900, 130, "Golden Hour", 84, "Solstice Serif", True),
                 *button("cta", 340, 1150, 400, 90, "Shop now", 28),
             ], targets=SPREAD, show=("headline", "cta_label"), show_roomy=("photo",),
             together=(("photo", "caption"),), xfail=("fb-link", "leaderboard", "mobile-banner", "billboard")),

        Case("freetype_margin_stack",
             "stacked copy set tight (headline, subhead, label, button): if the renderer sets text 5% "
             "taller than measured, nothing touches its neighbour, leaves the canvas or leaves its button",
             1080, 1350, [
                 logo(390, 60, 300, 100),
                 text("headline", 90, 240, 900, 270, "Taste the longest day of summer", 88, "Solstice Serif", True),
                 text("subhead", 90, 530, 900, 110, "Three botanical infusions, until the equinox.", 36),
                 text("tag", 90, 650, 900, 50, "LIMITED RUN", 26, letter_spacing=3),
                 *button("cta", 340, 780, 400, 80, "Shop now", 30),
             ], targets=SPREAD + ("pres-16x9", "ig-portrait"), show=("headline", "cta_label"),
             check=margin_clear(0.05)),

        Case("freetype_margin_narrow",
             "wrapping copy in narrow and short formats keeps 5% of slack under every text",
             1200, 628, [
                 logo(60, 40, 240, 80),
                 text("headline", 60, 150, 620, 220, "Fresh drinks for long evenings outside", 64, "Solstice Serif",
                      True, align="left"),
                 text("body", 60, 400, 620, 100, "Small batch, seasonal, always cold.", 30, align="left"),
                 *button("cta", 60, 530, 320, 70, "Order now", 28),
             ], targets=("med-rect", "mobile-banner", "skyscraper", "half-page", "fb-link", "ig-square", "email-header"),
             show=("headline", "cta_label"), check=margin_clear(0.05), xfail=("email-header",)),

        Case("text_style_survives",
             "every non-size text property (colour, weight, italic, alignment, line height, letter "
             "spacing) is carried through unchanged on every target",
             1080, 1350, [
                 logo(390, 60, 300, 100),
                 text("headline", 90, 240, 900, 240, "Golden Hour in the garden", 84, "Solstice Serif", True,
                      align="left", line_height=1.05, letter_spacing=-1, color="rgba(255,236,200,1)"),
                 text("body", 90, 520, 900, 140, "Citrus, elderflower and a little honey.", 34, align="right",
                      line_height=1.5, font_style="italic", color="rgba(255,255,255,0.9)"),
                 text("tag", 90, 700, 900, 50, "NEW SEASON", 24, align="center", letter_spacing=4,
                      font_weight="bold", color=GOLD),
                 *button("cta", 340, 860, 400, 90, "Shop now", 28),
             ], targets=SPREAD, show=("headline", "cta_label"),
             check=style_survives("headline", "body", "tag", "cta_label")),
    ]


def deep_panel(rows: int = 6, per: int = 3, W: int = 1080, H: int = 1350) -> List[dict]:
    """A menu card: a logo, a product shot and one panel holding a headline, ``rows`` rows of
    ``per`` side-by-side texts and a button, plus a few decorative shapes. Every row of copy is a
    flex row inside the panel's column, the panel sits in the layout's own row or column, so
    nested width searches stack three deep (the r9 blow-up class, novel-v5)."""
    els = [logo(60, 40, 220, 72),
           image("product", 640, 140, 380, 420),
           shape("card", 40, 600, W - 80, H - 660, "rgba(43,27,46,0.85)", z=2, corner_radius=24),
           text("headline", 70, 630, 900, 100, "The summer tasting menu", 64, "Solstice Serif", True, align="left"),
           shape("rule", 60, 130, 300, 4, GOLD, z=2),
           shape("dot", 960, 40, 60, 60, GOLD, z=2, kind="ellipse")]
    cw = (W - 140) // per
    y = 750
    for r in range(rows):
        for k in range(per):
            els.append(text(f"t{r}_{k}", 70 + k * cw, y, cw - 30, 44, f"Course {r + 1}{'abc'[k % 3]} citrus thyme",
                            24, bold=(k == 0), align="left" if k == 0 else "right"))
        y += 70
    els += button("cta", 70, min(y + 10, H - 150), 360, 70, "Book a table", 26)
    for i in range(6):  # small decorative ticks along the bottom edge
        els.append(shape(f"tick{i}", 300 + 120 * i, H - 40, 80, 6, GOLD, z=2))
    return els


def round10_cases() -> List[Case]:
    """Engine robustness (r10, H17): a design whose layout tree nests flex rows three deep must
    resize in bounded time and memory, with every hard rule held."""
    return [
        Case("deep_panel_rows",
             "a panel of many side-by-side text rows beside a product, on a 16:9-ish landscape: the "
             "nested width searches must not blow up (bounded time and memory), and the output is valid",
             1080, 1350, deep_panel(), targets=("1640x924", "pres-16x9", "leaderboard"),
             show=("headline", "cta_label"), seconds=120.0, max_mb=1500.0),
    ]


# A real photo with a face (Crello, CDLA-Permissive-2.0; see tests/assets/SOURCES.txt), found
# through the brand kit's assets directory so the engine and the renderer both read it.
FACE_PHOTO = "../../tests/assets/face_photo_wide.jpg"
FACE_PHOTO_SIZE = (512, 268)
FACE_BOX = (0.62, 0.04, 0.80, 0.46)  # the face, as fractions of the photo (YuNet's box)


def face_kept(pid: str, face=FACE_BOX, size=FACE_PHOTO_SIZE, cut: float = 0.08) -> Check:
    """A background photo's face stays on the canvas (at most ``cut`` of its width or height
    off it) and no visible copy is drawn over it."""
    def check(src, out, tgt):
        e = next((x for x in out["elements"] if x["id"] == pid), None)
        if e is None or float(e.get("opacity", 1)) <= 0:
            return [f"{pid}: photo hidden"]
        iw, ih = size
        if e.get("fit") == "cover":
            s = max(e["width"] / iw, e["height"] / ih)
        elif e.get("fit") == "contain":
            s = min(e["width"] / iw, e["height"] / ih)
        else:
            return []
        dw, dh = iw * s, ih * s
        dx, dy = e["x"] + (e["width"] - dw) / 2.0, e["y"] + (e["height"] - dh) / 2.0
        if e.get("fit") == "cover":  # the renderer crops to the box first
            bx0, by0, bx1, by1 = _box(e)
        else:
            bx0, by0, bx1, by1 = dx, dy, dx + dw, dy + dh
        fx0, fy0 = dx + face[0] * dw, dy + face[1] * dh
        fx1, fy1 = dx + face[2] * dw, dy + face[3] * dh
        vx0, vy0 = max(fx0, bx0, 0.0), max(fy0, by0, 0.0)
        vx1, vy1 = min(fx1, bx1, tgt["width"]), min(fy1, by1, tgt["height"])
        errs = []
        if vx1 - vx0 < (1 - cut) * (fx1 - fx0) or vy1 - vy0 < (1 - cut) * (fy1 - fy0):
            errs.append(f"{pid}: face cut by the crop ({fx0:.0f},{fy0:.0f},{fx1:.0f},{fy1:.0f})")
        area = max(1.0, (fx1 - fx0) * (fy1 - fy0))
        for t in out["elements"]:
            if t.get("type") != "text" or float(t.get("opacity", 1)) <= 0 or not str(t.get("content", "")).strip():
                continue
            db = drawn_box(t)
            ix = max(0.0, min(db.r, fx1) - max(db.x, fx0))
            iy = max(0.0, min(db.b, fy1) - max(db.y, fy0))
            if ix * iy > 0.05 * area:
                errs.append(f"{t['id']}: copy over the face")
        return errs
    return check


def sprint_b_cases() -> List[Case]:
    """Subject-aware photos (S5): a canvas-covering photo is cropped by bleed so its subject
    survives, and copy keeps off a face."""
    W, H = 1200, 628
    photo = {"id": "photo", "type": "image", "x": 0, "y": 0, "width": W, "height": H, "z_index": 0,
             "asset_id": FACE_PHOTO, "fit": "cover"}
    return [
        Case("face_photo_background",
             "a wide photo of a person covers the canvas, copy on the plain wall beside her: on a "
             "square or a story the centre crop cuts her face; the crop must keep it whole and the "
             "copy must not land on it",
             W, H,
             [photo,
              text("kicker", 110, 200, 400, 34, "HIRING BEST", 30, bold=True, align="left"),
              text("headline", 105, 260, 460, 64, "CANDIDATES", 57, bold=True, align="left"),
              text("body", 110, 360, 300, 40, "since 1980", 30, align="left")],
             targets=("ig-square", "ig-portrait", "story", "yt-thumb", "fb-link"),
             show=("headline",), background=False, check=face_kept("photo")),
    ]


# ----------------------------------------------------------------------------
# relations: two runs of the engine that must agree (loop/weaknesses.md, r1). Not about one
# output's quality but about the engine being a well-behaved function of its input.


@dataclass
class Relation:
    name: str
    probe: str
    kind: str  # "scale": design and target both k times larger; "step": target one pixel wider/taller
    source: dict
    target: dict
    k: int = 2
    xfail: bool = False


def _scaled(src: dict, k: float) -> dict:
    out = json.loads(json.dumps(src))
    out["canvas"] = {"width": src["canvas"]["width"] * k, "height": src["canvas"]["height"] * k}
    for e in out["elements"]:
        for f in ("x", "y", "width", "height", "font_size", "letter_spacing", "corner_radius"):
            if isinstance(e.get(f), (int, float)):
                e[f] = e[f] * k
    return out


def relations() -> List[Relation]:
    by = {c.name: c for c in cases()}
    pick = ("explicit_newlines", "letter_spaced_caps", "crowded_l3", "band_stays_band_on_squares",
            "tiny_items_kept_on_small_canvases")
    out: List[Relation] = []
    for n in pick:
        c = by[n]
        for t in ("ig-square", "story", "med-rect", "leaderboard", "fb-link"):
            out.append(Relation(f"scale2x/{n}@{t}",
                                "design and target both doubled: the same layout, doubled",
                                "scale", c.source, c.target(t), 2, xfail=(n, t) in SCALE_XFAIL))
    for n in ("explicit_newlines", "crowded_l3", "band_stays_band_on_squares", "tiny_items_kept_on_small_canvases"):
        c = by[n]
        for (w, h) in ((1080, 1080), (300, 250), (728, 90), (1200, 628), (1080, 1920), (970, 250)):
            for dw, dh in ((1, 0), (0, 1)):
                out.append(Relation(f"step/{n}@{w}x{h}+{dw}{dh}",
                                    "one pixel more of width or height must not flip the layout",
                                    "step", c.source, {"width": w, "height": h}, 1,
                                    xfail=(n, w, h, dw, dh) in STEP_XFAIL))
    r7 = ("fill_bg_odd_id_bleeding", "plate_text_not_split", "caption_stays_with_photo", "freetype_margin_stack")
    for n in r7:  # r7: scale invariance at 3x, and a continuity step, on the new cases
        c = by[n]
        for t in ("ig-square", "story", "med-rect", "leaderboard", "fb-link"):
            out.append(Relation(f"scale3x/{n}@{t}", "design and target both tripled: the same layout, tripled",
                                "scale", c.source, c.target(t), 3, xfail=(n, t) in SCALE3_XFAIL))
        for (w, h) in ((1080, 1080), (300, 250), (728, 90), (1200, 628), (1080, 1920), (970, 250)):
            for dw, dh in ((1, 0), (0, 1)):
                out.append(Relation(f"step/{n}@{w}x{h}+{dw}{dh}",
                                    "one pixel more of width or height must not flip the layout",
                                    "step", c.source, {"width": w, "height": h}, 1,
                                    xfail=(n, w, h, dw, dh) in STEP7_XFAIL))
    return out


# Measured failures of the engine on 2026-10-24 (absolute-pixel thresholds, hard cut-offs).
SCALE_XFAIL = {("explicit_newlines", "med-rect"), ("explicit_newlines", "leaderboard"),
               ("letter_spaced_caps", "med-rect"), ("letter_spaced_caps", "leaderboard"),
               ("crowded_l3", "med-rect"), ("crowded_l3", "leaderboard"), ("crowded_l3", "fb-link"),
               ("band_stays_band_on_squares", "med-rect"),
               ("tiny_items_kept_on_small_canvases", "med-rect"),
               ("tiny_items_kept_on_small_canvases", "leaderboard")}
SCALE3_XFAIL = {("fill_bg_odd_id_bleeding", "med-rect"), ("plate_text_not_split", "med-rect"),
                ("plate_text_not_split", "leaderboard"), ("caption_stays_with_photo", "med-rect"),
                ("caption_stays_with_photo", "leaderboard"), ("caption_stays_with_photo", "fb-link"),
                ("freetype_margin_stack", "med-rect"), ("freetype_margin_stack", "leaderboard")}  # measured 2026-10-25 (r7)
STEP7_XFAIL: set = set()
STEP_XFAIL = {("crowded_l3", 1200, 628, 1, 0), ("crowded_l3", 1080, 1920, 1, 0),
              ("crowded_l3", 1080, 1920, 0, 1)}


# ----------------------------------------------------------------------------
# bleed probes (loop/STEERING.md S4): hand-made outputs, no engine. Each says whether the
# hard rules must accept it (allowed bleed) or reject it (forbidden or excessive bleed).


@dataclass
class BleedProbe:
    name: str
    source: dict
    target: dict
    moves: Dict[str, dict]  # id -> fields of that element in the output
    allowed: bool

    @property
    def output(self) -> dict:
        """The source re-set on the target: each element kept where it was (scaled to fit the
        target), then the probe's moves applied."""
        W, H = self.target["width"], self.target["height"]
        sw, sh = self.source["canvas"]["width"], self.source["canvas"]["height"]
        k = min(W / sw, H / sh)
        els = []
        for e in self.source["elements"]:
            o = dict(e, x=e["x"] * k, y=e["y"] * k, width=e["width"] * k, height=e["height"] * k)
            if e["type"] == "text":
                o["font_size"] = max(1, int(e["font_size"] * k))
            if e["id"] == "bg":
                o.update(x=0, y=0, width=W, height=H)
            els.append(dict(o, **self.moves.get(e["id"], {})))
        for o in els:  # nothing else may leave the canvas by accident
            if o["id"] not in self.moves and o["id"] != "bg":
                o["x"] = min(max(0.0, o["x"]), W - o["width"])
                o["y"] = min(max(0.0, o["y"]), H - o["height"])
        return {"canvas": dict(self.target), "elements": els}


def bleed_probes() -> List[BleedProbe]:
    sq = {"canvas": {"width": 1080, "height": 1080}, "elements": [
        {"id": "bg", "type": "image", "x": 0, "y": 0, "width": 1080, "height": 1080, "z_index": 0,
         "asset_id": "solstice_bg_gradient.png", "fit": "cover"},
        shape("deco_bled", -100, -100, 500, 500, "rgba(255,255,255,0.18)", kind="ellipse"),
        shape("deco_inside", 760, 700, 260, 260, "rgba(240,178,66,0.25)", kind="ellipse"),
        image("panel", 600, 300, 400, 500, asset="solstice_bg_gradient.png"),
        logo(380, 60, 320, 110),
        image("product", 120, 300, 400, 400),
        text("headline", 90, 780, 900, 120, "Golden Hour", 72, "Solstice Serif", True),
        *button("cta", 330, 950, 420, 90, "Shop now", 28),
    ]}
    photo = {"canvas": {"width": 1200, "height": 1500}, "elements": [
        {"id": "bg", "type": "image", "x": 0, "y": 0, "width": 1200, "height": 1500, "z_index": 0,
         "asset_id": "solstice_bg_gradient.png", "fit": "cover"},
        text("headline", 80, 1000, 700, 160, "Into the long light", 64, "Solstice Serif", True, align="left"),
    ]}
    cut = dict(sq, elements=[dict(e, x=-80) if e["id"] == "product" else e for e in sq["elements"]])
    story, lb, wide = {"width": 1080, "height": 1920}, {"width": 728, "height": 90}, {"width": 1600, "height": 400}
    P = BleedProbe
    return [
        P("decor_bled_in_source_keeps_bleeding", sq, wide,
          {"deco_bled": dict(x=-40, y=-50, width=250, height=250)}, True),
        P("plain_decor_bleeds_a_little", sq, wide, {"deco_inside": dict(x=1500, y=200, width=160, height=160)}, True),
        P("photo_cover_crop_framed", photo, story, {"bg": dict(x=-228, y=0, width=1536, height=1920)}, True),
        P("photo_cover_crop_on_a_strip", photo, lb, {"bg": dict(x=0, y=-400, width=728, height=910)}, True),
        P("image_panel_against_an_edge", sq, wide, {"panel": dict(x=1400, y=0, width=320, height=400)}, True),
        P("cutout_that_bled_in_source", cut, wide, {"product": dict(x=-20, y=40, width=110, height=110)}, True),
        P("text_past_the_edge", sq, wide, {"headline": dict(x=-40)}, False),
        P("logo_past_the_edge", sq, wide, {"logo": dict(x=1500, y=10, width=200, height=69)}, False),
        P("button_past_the_edge", sq, wide, {"cta_bg": dict(x=1450, y=300, width=210, height=45),
                                              "cta_label": dict(x=1450, y=311, width=210, height=22)}, False),
        P("cutout_past_the_edge", sq, wide, {"product": dict(x=-60, y=40, width=200, height=200)}, False),
        P("decor_mostly_off", sq, wide, {"deco_inside": dict(x=1530, y=200, width=160, height=160)}, False),
        P("panel_mostly_off", sq, wide, {"panel": dict(x=1480, y=0, width=320, height=400)}, False),
        P("decor_bled_much_further", sq, wide, {"deco_bled": dict(x=-180, y=-180, width=250, height=250)}, False),
    ]


def _thin(rid: str, src: dict, out: dict) -> List[str]:
    s = next(e for e in src["elements"] if e["id"] == rid)
    o = next(e for e in out["elements"] if e["id"] == rid)
    if float(o.get("opacity", 1)) <= 0:
        return []
    ratio = (o["width"] / max(1e-6, o["height"])) / (s["width"] / s["height"])
    if not 0.95 <= ratio <= 1.05 or o["height"] > 8:
        return [f"{rid}: rule became {o['width']:.1f}x{o['height']:.1f}"]
    return []


def malformed() -> List[Tuple[str, dict, dict]]:
    """Inputs that break the schema in ways real exports do. Only the hard rules
    apply: the engine must not crash and must return a valid design."""
    base = cases()[0].source
    out = []

    def variant(name, mutate, tgt=(728, 90)):
        src = json.loads(json.dumps(base))
        mutate(src["elements"])
        out.append((name, src, {"width": tgt[0], "height": tgt[1]}))

    def strings(els):
        for e in els:
            for k in ("x", "y", "width", "height"):
                e[k] = str(e[k])

    def drop_optional(els):
        for e in els:
            for k in ("z_index", "text_align", "fit", "font_weight"):
                e.pop(k, None)

    def nones(els):
        for e in els:
            e["rotation"] = None
            e["opacity"] = None
            if e["type"] == "text":
                e["line_height"] = None
                e["letter_spacing"] = None

    def odd_values(els):
        for e in els:
            if e["type"] == "text":
                e.update(font_weight=700, text_align="justify", color="#ffffff", font_style="oblique")
            if e["type"] == "shape":
                e.update(fill_color="orange")

    def degenerate(els):
        els[1].update(width=0, height=0)
        els[2].update(width=-50)
        els.append({"id": "nan", "type": "text", "x": math.nan, "y": 10, "width": 100, "height": 20,
                    "content": "NaN box", "font_family": "Solstice Sans", "font_size": 20,
                    "color": "rgba(255,255,255,1)"})
        els.append({"id": "far", "type": "image", "x": 99999, "y": -99999, "width": 100, "height": 100,
                    "asset_id": "solstice_bottle.png"})
        els.append({"id": "weird", "type": "group", "x": 0, "y": 0, "width": 10, "height": 10})

    def zero_font(els):
        for e in els:
            if e["type"] == "text":
                e["font_size"] = 0

    variant("numbers_as_strings", strings, (1080, 1920))
    variant("missing_optional_fields", drop_optional)
    variant("null_fields", nones, (1080, 1080))
    variant("odd_enum_and_colour_values", odd_values)
    variant("degenerate_geometry", degenerate, (300, 250))
    variant("zero_font_size", zero_font, (970, 250))
    return out


def hostile() -> List[Tuple[str, dict, dict]]:
    """Documents broken past the schema (no ids, no canvas, junk entries, words
    where numbers go). Nothing can be promised about them except that the
    engine returns a renderable design for the target and does not take the
    run down: one exception would abort every remaining task."""
    text = {"type": "text", "x": 0, "y": 0, "width": 10, "height": 10, "content": "no id",
            "font_family": "Solstice Sans", "font_size": 12, "color": "rgba(0,0,0,1)"}
    shape = {"id": "a", "type": "shape", "x": 0, "y": 0, "width": 10, "height": 10, "shape_type": "rectangle",
             "fill_color": "rgba(0,0,0,1)"}
    return [
        ("element_without_id", {"canvas": {"width": 100, "height": 100}, "elements": [text]}),
        ("no_elements", {"canvas": {"width": 100, "height": 100}, "elements": []}),
        ("zero_canvas", {"canvas": {"width": 0, "height": 100}, "elements": [shape]}),
        ("no_canvas", {"elements": [shape]}),
        ("words_for_numbers", {"canvas": {"width": 100, "height": 100}, "elements": [
            dict(text, id="t", content=None, font_family=None, font_size="big", color=5, rotation="x"),
            {"id": 7, "type": "image", "x": 0, "y": 0, "width": "w", "height": 10}]}),
        ("junk_entries", {"canvas": {"width": 100, "height": 100}, "elements": ["junk", None, {"id": "q", "type": "text"}]}),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", help="directory to write each case as a task JSON (for tools.sheet)")
    a = ap.parse_args()
    if a.write:
        d = Path(a.write)
        d.mkdir(parents=True, exist_ok=True)
        for c in cases():
            t = c.target(c.targets[0])
            (d / f"{c.name}.json").write_text(json.dumps({"task_id": c.name, "source": c.source,
                                                          "target_canvas": t}, indent=1))
        print(f"wrote {len(cases())} cases to {d}")
    else:
        for c in cases():
            print(f"{c.name:22s} {c.probe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
