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

    @property
    def source(self) -> dict:
        els = [dict(e) for e in self.elements]
        if self.background:
            els.insert(0, {"id": "bg", "type": "image", "x": 0, "y": 0, "width": self.W, "height": self.H,
                           "z_index": 0, "asset_id": "solstice_bg_gradient.png", "fit": "cover"})
        return {"canvas": {"width": self.W, "height": self.H}, "elements": els}

    def target(self, name: str) -> dict:
        w, h = CATALOGUE[name]
        return {"width": w, "height": h}

    def must_show(self, name: str) -> Tuple[str, ...]:
        return self.show + (self.show_roomy if min(CATALOGUE[name]) >= ROOMY_SIDE else ())


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


def all_of(*checks: Check) -> Check:
    def check(src, out, tgt):
        return [e for c in checks for e in c(src, out, tgt)]
    return check


# ----------------------------------------------------------------------------
# the cases


def cases() -> List[Case]:
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
