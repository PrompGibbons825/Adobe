"""tools.gate.visible: an image covers only where its fitted asset is opaque (loop/setup-issues.md,
2026-10-25 r11 S1 artefacts), and a missing asset may be faded out."""

from __future__ import annotations

import unittest

from tools.gate.visible import hidden_elements

BG = dict(id="bg", type="shape", shape_type="rectangle", fill_color="rgba(10,20,30,1)", x=0, y=0, width=1080, height=1080, z_index=0)
BOTTLE = dict(id="p", type="image", asset_id="solstice_bottle.png", fit="contain", x=0, y=0, width=1080, height=1080, z_index=1)
LOGO = dict(id="l", type="image", asset_id="solstice_bg_gradient.png", fit="cover", x=0, y=0, width=1080, height=1080, z_index=1)
SRC = {"canvas": {"width": 1080, "height": 1080}, "elements": [BG, dict(BOTTLE, x=600, width=400)]}
T = {"width": 1080, "height": 1080}


class VisibleAlpha(unittest.TestCase):
    def test_cutout_does_not_hide_background(self):
        out = {"canvas": T, "elements": [BG, BOTTLE]}
        self.assertEqual([h for h in hidden_elements(SRC, out, T) if h.startswith("bg")], [])

    def test_opaque_image_still_hides(self):
        src = {"canvas": T, "elements": [BG, dict(LOGO, width=300)]}
        out = {"canvas": T, "elements": [BG, LOGO]}
        self.assertTrue(any(h.startswith("bg") for h in hidden_elements(src, out, T)))

    def test_missing_asset_may_fade(self):
        miss = dict(id="m", type="image", asset_id="nope.png", x=10, y=10, width=100, height=100)
        src = {"canvas": T, "elements": [BG, miss]}
        out = {"canvas": T, "elements": [BG, dict(miss, opacity=0.0)]}
        self.assertEqual([h for h in hidden_elements(src, out, T) if h.startswith("m")], [])


if __name__ == "__main__":
    unittest.main()
