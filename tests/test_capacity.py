"""B0 floor fix: small print is exempt only when the floor is provably unholdable (capacity)."""

import unittest
from unittest import mock

from tools import rules


def _text(i, content, fs):
    return {"id": f"t{i}", "type": "text", "x": 0, "y": 0, "width": 300, "height": fs * 2, "z_index": i,
            "content": content, "font_family": "Solstice Sans", "font_size": fs, "color": "rgba(0,0,0,1)"}


def _design(W, H, texts):
    return {"canvas": {"width": W, "height": H}, "elements": texts}


class Capacity(unittest.TestCase):
    TGT = {"width": 320, "height": 50}

    def test_crowd_on_a_banner_is_capacity(self):
        out = _design(320, 50, [_text(i, "Golden hour sparkling botanical summer sale", 6) for i in range(6)])
        self.assertTrue(rules.capacity_impossible(out, self.TGT))
        self.assertEqual(len(rules.capacity_errors(out, self.TGT)), 6)
        self.assertEqual([e for e in rules.promise_errors(out, out, self.TGT) if "small print" in e], [])

    def test_one_short_text_is_not_capacity(self):
        out = _design(320, 50, [_text(0, "Sale", 6)])
        self.assertFalse(rules.capacity_impossible(out, self.TGT))
        self.assertEqual(rules.capacity_errors(out, self.TGT), [])
        self.assertTrue([e for e in rules.promise_errors(out, out, self.TGT) if "small print" in e])

    def test_k_is_exactly_one(self):
        out = _design(320, 50, [_text(0, "Sale", 6)])
        W, H = 320, 50
        with mock.patch.object(rules, "floor_area", return_value=W * H):
            self.assertFalse(rules.capacity_impossible(out, self.TGT))  # equal is holdable in principle
        with mock.patch.object(rules, "floor_area", return_value=W * H + 0.01):
            self.assertTrue(rules.capacity_impossible(out, self.TGT))
        self.assertEqual(rules.CAPACITY_K, 1.0)

    def test_capacity_does_not_hide_other_errors(self):
        out = _design(320, 50, [_text(i, "Golden hour sparkling botanical summer sale", 12) for i in range(8)])
        out["elements"][0]["x"] = 5000  # drawn past the canvas
        self.assertTrue(any("past the canvas" in e for e in rules.promise_errors(out, out, self.TGT)))

    def test_floor_area_grows_with_text(self):
        a = rules.floor_area([_text(0, "Sale", 6)], 10)
        b = rules.floor_area([_text(0, "Sale", 6), _text(1, "Sale now", 6)], 10)
        self.assertGreater(b, a)
        self.assertGreater(a, 0)


if __name__ == "__main__":
    unittest.main()
