"""tools/gate/visible.py: the two artefacts fixed on 2026-10-25 (loop/setup-issues.md, r11 S1).

(a) a transparent cut-out's box does not hide what lies behind it; only its opaque pixels do;
(b) an element whose asset file is missing is exempt (hiding beats a grey placeholder).

    .venv/bin/python -m unittest tests.test_visible
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.gate.visible import hidden_elements  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "brand_kit" / "assets"


def _src():
    return {"canvas": {"width": 1000, "height": 1000}, "elements": [
        {"id": "bg", "type": "image", "x": 0, "y": 0, "width": 1000, "height": 1000, "z_index": 0,
         "asset_id": "solstice_bg_gradient.png", "fit": "cover"},
        {"id": "bottle", "type": "image", "x": 400, "y": 300, "width": 200, "height": 400, "z_index": 1,
         "asset_id": "solstice_bottle.png", "fit": "contain"},
        {"id": "seal", "type": "image", "x": 800, "y": 850, "width": 150, "height": 100, "z_index": 2,
         "asset_id": "no_such_asset_on_disk.png", "fit": "contain"},
        {"id": "t", "type": "text", "x": 100, "y": 100, "width": 300, "height": 40, "z_index": 3,
         "content": "Hello", "font_family": "Solstice Sans", "font_size": 30, "color": "rgba(0,0,0,1)"},
    ]}


def _copy(d):
    return json.loads(json.dumps(d))


class CutoutAlpha(unittest.TestCase):
    def test_transparent_cutout_box_hides_nothing(self):
        src = _src()
        out = _copy(src)
        b = out["elements"][1]
        b.update(x=0, y=0, width=1000, height=1000)  # the bottle's box now spans the whole canvas
        hid = hidden_elements(src, out, {"width": 1000, "height": 1000}, assets=ASSETS)
        self.assertFalse([h for h in hid if h.startswith("bg:")], hid)

    def test_opaque_image_still_hides(self):
        src = _src()
        out = _copy(src)
        out["elements"].append({"id": "lid", "type": "image", "x": 0, "y": 0, "width": 1000, "height": 1000,
                                "z_index": 9, "asset_id": "solstice_bg_gradient.png", "fit": "cover"})
        hid = hidden_elements(src, out, {"width": 1000, "height": 1000}, assets=ASSETS)
        self.assertTrue([h for h in hid if h.startswith("bg:")], hid)
        self.assertTrue([h for h in hid if h.startswith("bottle:")], hid)

    def test_opaque_shape_still_hides(self):
        src = _src()
        out = _copy(src)
        out["elements"].append({"id": "lid", "type": "shape", "x": 0, "y": 0, "width": 1000, "height": 1000,
                                "z_index": 9, "shape_type": "rectangle", "fill_color": "rgba(0,0,0,1)"})
        hid = hidden_elements(src, out, {"width": 1000, "height": 1000}, assets=ASSETS)
        self.assertTrue([h for h in hid if h.startswith("bg:")], hid)

    def test_translucent_shape_hides_nothing(self):
        src = _src()
        out = _copy(src)
        out["elements"].append({"id": "scrim", "type": "shape", "x": 0, "y": 0, "width": 1000, "height": 1000,
                                "z_index": 9, "shape_type": "rectangle", "fill_color": "rgba(0,0,0,0.6)"})
        hid = hidden_elements(src, out, {"width": 1000, "height": 1000}, assets=ASSETS)
        self.assertEqual(hid, [])

    def test_contain_letterbox_is_transparent(self):
        # a wide logo letterboxed in a tall box covers only its drawn band
        src = _src()
        src["elements"].insert(1, {"id": "logo", "type": "image", "x": 0, "y": 0, "width": 1000, "height": 100,
                                   "z_index": 1, "asset_id": "solstice_logo_lockup.png", "fit": "contain"})
        out = _copy(src)
        out["elements"][1].update(height=1000)
        hid = hidden_elements(src, out, {"width": 1000, "height": 1000}, assets=ASSETS)
        self.assertFalse([h for h in hid if h.startswith("bg:")], hid)


class MissingAsset(unittest.TestCase):
    def test_missing_asset_hidden_is_exempt(self):
        src = _src()
        out = _copy(src)
        out["elements"][2]["opacity"] = 0
        hid = hidden_elements(src, out, {"width": 1000, "height": 1000}, assets=ASSETS)
        self.assertFalse([h for h in hid if h.startswith("seal:")], hid)

    def test_missing_asset_shrunk_or_offcanvas_is_exempt(self):
        src = _src()
        out = _copy(src)
        out["elements"][2].update(x=2000, width=0.5)
        hid = hidden_elements(src, out, {"width": 1000, "height": 1000}, assets=ASSETS)
        self.assertEqual(hid, [])

    def test_present_asset_hidden_still_counts(self):
        src = _src()
        out = _copy(src)
        out["elements"][1]["opacity"] = 0
        hid = hidden_elements(src, out, {"width": 1000, "height": 1000}, assets=ASSETS)
        self.assertTrue([h for h in hid if h.startswith("bottle:")], hid)


if __name__ == "__main__":
    unittest.main()
