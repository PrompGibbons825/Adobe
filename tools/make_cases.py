"""Dev tool: write extra source designs (not the official samples) to tools/cases/.

These exist to keep the engine honest about generalising: different source
ratios, alignments, colour schemes and element mixes, all built from the same
brand kit the samples use.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "cases"

BG = {"id": "bg", "type": "image", "x": 0, "y": 0, "z_index": 0, "asset_id": "solstice_bg_gradient.png", "fit": "cover"}
WHITE = "rgba(255,255,255,1)"
ORANGE = "rgba(232,106,58,1)"
GOLD = "rgba(240,178,66,1)"
PLUM = "rgba(43,27,46,1)"


def text(id, x, y, w, h, content, size, family="Solstice Sans", bold=False, color=WHITE, align="center", z=4, **kw):
    e = {"id": id, "type": "text", "x": x, "y": y, "width": w, "height": h, "z_index": z, "content": content,
         "font_family": family, "font_size": size, "color": color, "text_align": align}
    if bold:
        e["font_weight"] = "bold"
    e.update(kw)
    return e


def button(prefix, x, y, w, h, label, size, fill=ORANGE, color=WHITE, radius=None, group=True):
    g = prefix if group else None
    shape = {"id": f"{prefix}_bg", "type": "shape", "x": x, "y": y, "width": w, "height": h, "z_index": 3,
             "shape_type": "rectangle", "fill_color": fill, "corner_radius": h / 2 if radius is None else radius}
    lab = text(f"{prefix}_label", x, y + (h - size * 1.4) / 2, w, size * 1.4, label, size, bold=True, color=color)
    if g:
        shape["group_id"] = g
        lab["group_id"] = g
    return [shape, lab]


def task(name, W, H, elements, target=(1080, 1080)):
    bg = dict(BG, width=W, height=H)
    return name, {"task_id": name, "source": {"canvas": {"width": W, "height": H}, "elements": [bg] + elements},
                  "target_canvas": {"width": target[0], "height": target[1]}}


def cases():
    yield task("landscape_split", 1200, 628, [
        {"id": "product", "type": "image", "x": 60, "y": 54, "width": 520, "height": 520, "z_index": 2,
         "asset_id": "solstice_bottle.png", "fit": "contain"},
        {"id": "logo", "type": "image", "x": 640, "y": 60, "width": 240, "height": 80, "z_index": 3,
         "asset_id": "solstice_logo_lockup.png", "fit": "contain"},
        text("headline", 640, 180, 500, 140, "Golden hour, bottled.", 58, "Solstice Serif", True, align="left"),
        text("subhead", 640, 330, 480, 80, "Sparkling botanicals with blood orange and wild honey.", 24, align="left",
             color="rgba(255,255,255,0.85)"),
        *button("cta", 640, 450, 300, 70, "Order now", 24),
    ], target=(1080, 1350))
    yield task("text_only_light", 1080, 1080, [
        {"id": "panel", "type": "shape", "x": 0, "y": 0, "width": 1080, "height": 1080, "z_index": 1,
         "shape_type": "rectangle", "fill_color": "rgba(250,240,228,1)"},
        text("kicker", 140, 250, 800, 40, "SUMMER EDITION", 26, bold=True, color=ORANGE, letter_spacing=4),
        text("headline", 100, 320, 880, 260, "Taste the longest day of the year", 84, "Solstice Serif", True, color=PLUM),
        text("subhead", 160, 620, 760, 90, "Three new botanical sparkling infusions, only until the equinox.", 28,
             color="rgba(43,27,46,0.8)"),
        *button("cta", 365, 780, 350, 84, "Find a store", 26),
    ], target=(1584, 396))
    yield task("story_hero_top", 1080, 1920, [
        {"id": "logo", "type": "image", "x": 400, "y": 90, "width": 280, "height": 93, "z_index": 3,
         "asset_id": "solstice_logo_lockup.png", "fit": "contain"},
        {"id": "product", "type": "image", "x": 140, "y": 240, "width": 800, "height": 800, "z_index": 2,
         "asset_id": "solstice_bottle.png", "fit": "contain"},
        text("headline", 90, 1110, 900, 200, "Made for the golden hour", 76, "Solstice Serif", True,
             style_runs=[{"start": 13, "end": 24, "color": GOLD}]),
        text("subhead", 140, 1330, 800, 110, "Botanical sparkling infusions with wild honey and blood orange.", 32,
             color="rgba(255,255,255,0.85)"),
        *button("cta", 290, 1520, 500, 110, "Shop the collection", 32),
        text("legal", 90, 1820, 900, 30, "Please enjoy responsibly. Offer valid while supplies last.", 18,
             color="rgba(255,255,255,0.6)"),
    ], target=(1200, 628))
    yield task("two_products", 1080, 1080, [
        text("headline", 90, 80, 900, 120, "Pick your sunset", 72, "Solstice Serif", True),
        {"id": "product_a", "type": "image", "x": 110, "y": 260, "width": 420, "height": 420, "z_index": 2,
         "asset_id": "solstice_bottle.png", "fit": "contain"},
        {"id": "product_b", "type": "image", "x": 550, "y": 260, "width": 420, "height": 420, "z_index": 2,
         "asset_id": "solstice_bottle.png", "fit": "contain"},
        text("subhead", 140, 730, 800, 60, "Blood orange or wild honey - or both.", 30, color="rgba(255,255,255,0.85)"),
        *button("cta", 340, 850, 400, 90, "Shop the duo", 28),
        {"id": "logo", "type": "image", "x": 830, "y": 990, "width": 210, "height": 70, "z_index": 3,
         "asset_id": "solstice_logo_lockup.png", "fit": "contain"},
    ], target=(1920, 1080))
    yield task("leaderboard_src", 728, 90, [
        {"id": "logo", "type": "image", "x": 12, "y": 15, "width": 180, "height": 60, "z_index": 3,
         "asset_id": "solstice_logo_lockup.png", "fit": "contain"},
        text("headline", 200, 22, 330, 46, "Chase the Light - new botanical sparklers", 17, "Solstice Serif", True),
        *button("cta", 560, 25, 150, 40, "Shop now", 15),
    ], target=(1080, 1080))
    yield task("band_panel", 1080, 1350, [
        {"id": "product", "type": "image", "x": 240, "y": 60, "width": 600, "height": 600, "z_index": 2,
         "asset_id": "solstice_bottle.png", "fit": "contain"},
        {"id": "band", "type": "shape", "x": 0, "y": 760, "width": 1080, "height": 590, "z_index": 3,
         "shape_type": "rectangle", "fill_color": "rgba(26,16,30,0.85)"},
        text("headline", 90, 820, 900, 160, "The taste of a long summer evening", 60, "Solstice Serif", True),
        text("subhead", 120, 1000, 840, 90, "Sparkling botanicals, crafted in small batches.", 28,
             color="rgba(255,255,255,0.85)"),
        *button("cta", 340, 1140, 400, 90, "Discover", 28),
    ], target=(1600, 900))
    yield task("left_aligned_corner_logo", 1080, 1080, [
        text("headline", 80, 90, 640, 300, "Sunset, served cold.", 88, "Solstice Serif", True, align="left"),
        text("subhead", 80, 420, 560, 120, "Our new botanical sparkling infusions are here for the season.", 30,
             align="left", color="rgba(255,255,255,0.85)"),
        *button("cta", 80, 590, 320, 80, "Shop now", 26),
        {"id": "product", "type": "image", "x": 600, "y": 440, "width": 560, "height": 560, "z_index": 2,
         "asset_id": "solstice_bottle.png", "fit": "contain"},
        {"id": "logo", "type": "image", "x": 60, "y": 960, "width": 210, "height": 70, "z_index": 3,
         "asset_id": "solstice_logo_lockup.png", "fit": "contain"},
        text("legal", 600, 1035, 440, 24, "Enjoy responsibly.", 16, color="rgba(255,255,255,0.6)", align="right"),
    ], target=(1080, 1920))
    yield task("sticker_divider", 1080, 1350, [
        {"id": "logo", "type": "image", "x": 390, "y": 60, "width": 300, "height": 100, "z_index": 3,
         "asset_id": "solstice_logo_lockup.png", "fit": "contain"},
        {"id": "product", "type": "image", "x": 290, "y": 220, "width": 500, "height": 500, "z_index": 2,
         "asset_id": "solstice_bottle.png", "fit": "contain", "rotation": 8},
        {"id": "sticker", "type": "shape", "x": 720, "y": 250, "width": 180, "height": 180, "z_index": 5,
         "shape_type": "ellipse", "fill_color": GOLD},
        text("sticker_text", 720, 315, 180, 50, "-20%", 40, bold=True, color=PLUM, z=6),
        text("headline", 90, 780, 900, 150, "Summer sale on every bottle", 62, "Solstice Serif", True),
        {"id": "divider", "type": "shape", "x": 440, "y": 960, "width": 200, "height": 4, "z_index": 3,
         "shape_type": "rectangle", "fill_color": GOLD},
        text("subhead", 140, 990, 800, 80, "Through August 31st, online and in stores.", 28,
             color="rgba(255,255,255,0.85)"),
        *button("cta", 340, 1120, 400, 90, "Shop the sale", 28),
    ], target=(970, 250))


def reference_post():
    """The Instagram post from the brief's "Reference output - one design, four
    canvases" (page 3), rebuilt from the kit at the measured positions and type
    widths, so the engine's resizes can be laid beside Adobe's own (story,
    poster, leaderboard). A check, never a tuning target."""
    W = H = 1080
    cream = {"id": "bg", "type": "shape", "x": 0, "y": 0, "width": W, "height": H, "z_index": 0,
             "shape_type": "rectangle", "fill_color": "rgba(248,238,222,1)"}
    return "reference_post", {"task_id": "reference_post", "source": {"canvas": {"width": W, "height": H}, "elements": [
        cream,
        text("wordmark", 90, 98, 900, 80, "SOLSTICE", 67, "Solstice Serif", True, color=PLUM),
        {"id": "product", "type": "image", "x": 286, "y": 144, "width": 507, "height": 507, "z_index": 2,
         "asset_id": "solstice_bottle.png", "fit": "contain"},
        text("headline", 90, 612, 900, 132, "Chase the Light", 110, "Solstice Serif", True, color=PLUM),
        text("subhead", 90, 772, 900, 42, "Botanical sparkling infusions", 35, color="rgba(43,27,46,0.75)"),
        *button("cta", 352, 887, 374, 90, "Shop the Line", 32),
    ]}, "target_canvas": {"width": 1080, "height": 1920}}


def main():
    OUT.mkdir(exist_ok=True)
    for name, t in [*cases(), reference_post()]:
        (OUT / f"{name}.json").write_text(json.dumps(t, indent=2))
        print("wrote", name)


if __name__ == "__main__":
    main()
