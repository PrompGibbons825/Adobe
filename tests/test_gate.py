"""Fast tests of the gate's own logic (tools/gate): transformations and verdicts. No engine runs."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from tools.gate import floor, metamorphic as mm
from tools.gate.engine_run import canonical, load_tasks, output_path

SRC = {"canvas": {"width": 1000, "height": 500}, "elements": [
    {"id": "bg", "type": "shape", "x": 0, "y": 0, "width": 1000, "height": 500, "z_index": 0,
     "shape_type": "rectangle", "fill_color": "rgba(0,0,0,1)", "corner_radius": 4},
    {"id": "a", "type": "shape", "x": 10, "y": 10, "width": 50, "height": 50, "z_index": 1,
     "shape_type": "ellipse", "fill_color": "rgba(9,9,9,1)"},
    {"id": "b", "type": "shape", "x": 70, "y": 10, "width": 50, "height": 50, "z_index": 1,
     "shape_type": "ellipse", "fill_color": "rgba(8,8,8,1)"},
    {"id": "t", "type": "text", "x": 100, "y": 100, "width": 400, "height": 60, "z_index": 2,
     "content": "Golden hour", "font_family": "Solstice Sans", "font_size": 40, "color": "rgba(255,255,255,1)",
     "letter_spacing": 2, "style_runs": [{"start": 0, "end": 11, "font_weight": "bold"}]},
]}


class Transformations(unittest.TestCase):
    def test_scale_doubles_px_values_only(self):
        s = mm.scale_source(SRC, 2.0)
        self.assertEqual(s["canvas"], {"width": 2000, "height": 1000})
        t = s["elements"][3]
        self.assertEqual((t["x"], t["font_size"], t["letter_spacing"]), (200, 80, 4))
        self.assertEqual(t["z_index"], 2)
        self.assertEqual(s["elements"][0]["corner_radius"], 8)
        self.assertEqual(SRC["elements"][3]["x"], 100)  # the original is untouched

    def test_permutation_keeps_z_and_equal_z_order(self):
        for seed in range(20):
            s = mm.permute_source(SRC, random.Random(seed))
            ids = [e["id"] for e in s["elements"]]
            self.assertEqual(sorted(ids), ["a", "b", "bg", "t"])
            self.assertLess(ids.index("a"), ids.index("b"))  # same z: paint order kept
            self.assertEqual({e["id"]: e["z_index"] for e in s["elements"]},
                             {e["id"]: e["z_index"] for e in SRC["elements"]})

    def test_rename_round_trips(self):
        s, back = mm.rename_source(SRC, random.Random(1))
        self.assertFalse({e["id"] for e in s["elements"]} & {"a", "b", "bg", "t"})
        self.assertEqual(canonical(mm.map_ids(s, back)), canonical(SRC))

    def test_nonlatin_swaps_copy_and_keeps_runs_in_range(self):
        s = mm.nonlatin_source(SRC)
        t = s["elements"][3]
        self.assertNotEqual(t["content"], "Golden hour")
        self.assertTrue(any(ord(c) > 0x2000 for c in t["content"]))
        self.assertLessEqual(t["style_runs"][0]["end"], len(t["content"]))

    def test_canonical_ignores_element_order(self):
        rev = dict(SRC, elements=list(reversed(SRC["elements"])))
        self.assertEqual(canonical(rev), canonical(SRC))

    def test_jumps(self):
        moved = json.loads(json.dumps(SRC))
        moved["elements"][1]["x"] += 200  # 20% of 1000
        moved["elements"][2]["x"] += 100  # 10%
        self.assertEqual(len(mm.jumps(SRC, moved)), 1)
        hidden = json.loads(json.dumps(SRC))
        hidden["elements"][3]["opacity"] = 0
        self.assertEqual(mm.jumps(SRC, hidden), ["t: hidden"])

    def test_max_px_diff(self):
        o = json.loads(json.dumps(SRC))
        o["elements"][1]["y"] += 0.4
        d, other = mm.max_px_diff(SRC, o)
        self.assertAlmostEqual(d, 0.4)
        self.assertEqual(other, [])
        o["elements"][3]["font_size"] = 41
        self.assertEqual(mm.max_px_diff(SRC, o)[0], 1)


class FloorVerdict(unittest.TestCase):
    @staticmethod
    def _m(**counts):
        m = {k: [] for k in floor.MUST_BE_ZERO + floor.MUST_NOT_RISE}
        for k, n in counts.items():
            m[k] = ["x"] * n
        return m

    def test_zero_rows_and_rise_rows(self):
        base = {"a": self._m(promise=1), "b": self._m()}
        self.assertTrue(floor.compare_tiles({"a": self._m(), "b": self._m(promise=1)}, base)["pass"])
        self.assertFalse(floor.compare_tiles({"a": self._m(promise=1), "b": self._m(promise=1)}, base)["pass"])
        self.assertFalse(floor.compare_tiles({"a": self._m(ghost=1), "b": self._m()}, base)["pass"])
        self.assertTrue(floor.compare_tiles({"a": self._m(contrast=3), "b": self._m()}, None)["pass"])

    def test_missing_output_is_a_hard_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "src", Path(tmp) / "out"
            src.mkdir()
            out.mkdir()
            (src / "t1.json").write_text(json.dumps({"source": SRC, "target_canvas": {"width": 1000, "height": 500}}))
            (src / "notes.json").write_text("[]")
            self.assertEqual(list(load_tasks(src)), ["t1"])
            self.assertIsNone(output_path(out, "t1"))
            m = floor.tile_metrics(src, out)
            self.assertEqual(m["t1"]["hard"], ["no output"])
            (out / "t1_output.json").write_text(json.dumps(SRC))
            self.assertEqual(floor.tile_metrics(src, out)["t1"]["hard"], [])


if __name__ == "__main__":
    unittest.main()
