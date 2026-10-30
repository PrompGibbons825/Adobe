"""tools.bench: similarity, summary maths and the tile selection are deterministic and sane."""

import unittest

from tools import bench


def _d(W, H, els):
    return {"canvas": {"width": W, "height": H}, "elements": els}


def _img(i, x, y, w, h):
    return {"id": f"e{i}", "type": "image", "x": x, "y": y, "width": w, "height": h, "z_index": i}


def _txt(i, c, x, y, w, h):
    return {"id": f"e{i}", "type": "text", "content": c, "x": x, "y": y, "width": w, "height": h, "z_index": i}


class Similarity(unittest.TestCase):
    def test_same_layout_is_one_even_when_layers_are_reordered(self):
        ref = _d(100, 100, [_img(0, 0, 0, 50, 50), _img(1, 50, 50, 20, 40), _txt(2, "Hi", 10, 80, 30, 10)])
        ours = _d(100, 100, [_img(0, 50, 50, 20, 40), _img(1, 0, 0, 50, 50), _txt(2, "hi ", 10, 80, 30, 10)])
        self.assertAlmostEqual(bench.similarity(ours, ref), 1.0)

    def test_disjoint_is_zero_and_nothing_matched_is_none(self):
        self.assertEqual(bench.similarity(_d(100, 100, [_img(0, 0, 0, 10, 10)]), _d(100, 100, [_img(0, 80, 80, 10, 10)])), 0.0)
        self.assertIsNone(bench.similarity(_d(100, 100, [_txt(0, "a", 0, 0, 5, 5)]), _d(100, 100, [_txt(0, "b", 0, 0, 5, 5)])))

    def test_scale_free(self):
        a = _d(100, 100, [_img(0, 10, 10, 40, 40)])
        b = _d(200, 200, [_img(0, 20, 20, 80, 80)])
        self.assertAlmostEqual(bench.similarity(a, b), 1.0)


class Summary(unittest.TestCase):
    def test_win_rate_counts_ties_as_half_and_ci_brackets_it(self):
        s = bench._summ({"a": {"score_a": 1.0}, "b": {"score_a": 0.5}, "c": {"score_a": 0.0}, "d": {"score_a": 1.0}})
        self.assertAlmostEqual(s["win_rate"], 0.625)
        self.assertEqual((s["wins"], s["ties"], s["losses"]), (2, 1, 1))
        self.assertLessEqual(s["ci95"][0], s["win_rate"])
        self.assertGreaterEqual(s["ci95"][1], s["win_rate"])
        self.assertAlmostEqual(s["share_ge"], 0.75)

    def test_empty(self):
        self.assertIsNone(bench._summ({})["win_rate"])

    def test_pick_is_deterministic(self):
        names = [f"n{i}" for i in range(50)]
        self.assertEqual(bench._rng("pick", "a", "b").sample(names, 5), bench._rng("pick", "a", "b").sample(names, 5))


if __name__ == "__main__":
    unittest.main()
