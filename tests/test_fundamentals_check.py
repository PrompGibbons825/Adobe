"""fundamentals.check: known-bad outputs fail, known-good outputs pass (L0 checker audit).

    .venv/bin/python -m unittest tests.test_fundamentals_check
"""

from __future__ import annotations

import copy
import json
import logging
import unittest
from pathlib import Path

from fundamentals.check import check

REPO = Path(__file__).resolve().parents[1]
SAMPLES = sorted((REPO / "input").glob("*.json"))


def _src():
    return json.loads(SAMPLES[1].read_text())["source"]  # level 2: text, images, shapes, groups


def codes(src, out, tgt, pixels=False, level="hard"):
    return {f.code for f in check(src, out, tgt, pixels=pixels) if f.level == level}


class KnownGood(unittest.TestCase):
    def test_sources_are_valid_outputs_of_themselves(self):
        logging.disable(logging.WARNING)
        for p in SAMPLES:
            src = json.loads(p.read_text())["source"]
            tgt = dict(src["canvas"])
            self.assertEqual(codes(src, copy.deepcopy(src), tgt, pixels=True), set(), p.name)


class KnownBad(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.WARNING)
        self.src = _src()
        self.tgt = dict(self.src["canvas"])
        self.out = copy.deepcopy(self.src)
        self.byid = {e["id"]: e for e in self.out["elements"]}
        self.text = next(e for e in self.out["elements"] if e["type"] == "text")
        self.image = next(e for e in self.out["elements"] if e["type"] == "image" and e.get("fit") != "cover")
        self.shape = next(e for e in self.out["elements"] if e["type"] == "shape")

    def bad(self, code, pixels=False, level="hard"):
        self.assertIn(code, codes(self.src, self.out, self.tgt, pixels, level))

    def test_missing_added_duplicate(self):
        self.out["elements"].pop()
        self.bad("id_missing")
        self.out["elements"].append(dict(self.text, id="new"))
        self.bad("id_added")
        self.out["elements"].append(copy.deepcopy(self.text))
        self.bad("id_duplicate")

    def test_canvas(self):
        self.out["canvas"] = {"width": 1, "height": 1}
        self.bad("canvas")

    def test_copy_and_font_changed(self):
        self.text["content"] += "!"
        self.bad("content_changed")
        self.text["font_family"] = "Lora"
        self.bad("font_family_changed")

    def test_image_aspect(self):
        self.image["width"] *= 1.5
        self.bad("aspect")

    def test_text_past_canvas(self):
        self.text["x"] = self.tgt["width"] - 5
        self.bad("text_cut")
        self.bad("text_cut", pixels=True)

    def test_renderer_crashes_and_wrong_draws(self):
        self.text["color"] = "white"
        self.bad("render_crash")
        self.text["color"] = "rgba(0,0,0,1)"
        self.text["width"] = 0
        self.bad("render_crash")
        self.shape["fill_color"] = "#ff0000"
        self.bad("shape_colour")

    def test_hidden_under_an_opaque_shape(self):
        cover = dict(self.shape, id=self.shape["id"], x=0, y=0, width=self.tgt["width"], height=self.tgt["height"],
                     z_index=10 ** 6, fill_color="rgba(20,20,20,1)", shape_type="rectangle", rotation=0)
        self.out["elements"] = [cover if e is self.shape else e for e in self.out["elements"]]
        self.assertIn("hidden", codes(self.src, self.out, self.tgt, True, "hard"))

    def test_copy_over_copy(self):
        texts = [e for e in self.out["elements"] if e["type"] == "text"]
        texts[1].update(x=texts[0]["x"], y=texts[0]["y"])
        self.bad("copy_over_copy", level="promise")


if __name__ == "__main__":
    unittest.main()
