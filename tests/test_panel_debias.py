"""The panel's position-bias fixes (round 7): balanced orders, both-order asks, the mats that
stop the cursor judge reading B as a diff of A, the A_file echo against misfiled verdicts, and
spreading one REF over several calls. No real judge is called."""

import json
import re
import unittest

from PIL import Image

from tests.test_gate_run import TempEnv, _design
from tools.gate import baselines, prereg
from tools.gate import panel as PN


class BiasedJudge:
    """Always answers "A" (a pure position bias), or the hidden good side with ``truthful``;
    scores follow the choice, confidence ``conf`` (per model)."""

    def __init__(self, truthful=False, conf=2):
        self.truthful, self.conf = truthful, conf
        self.calls, self.prompts = 0, []
        self.panel = None

    def bind(self, panel):
        self.panel = panel

    def __call__(self, prompt, model):
        self.calls += 1
        self.prompts.append(prompt)
        ids = list(dict.fromkeys(re.findall(r"([0-9a-f]{8})_A\.png", prompt)))
        out = {}
        for aid in ids:
            pick = self.panel.asks[aid]["x_side"] if self.truthful else "A"
            other = PN._other(pick)
            out[aid] = {**{c: pick for c in PN.CRITERIA}, "confidence": self.conf, "why": "fake",
                        "scores": {pick: {c: 8 for c in PN.CRITERIA}, other: {c: 5 for c in PN.CRITERIA}}}
        return json.dumps(out), {"input": 100}


class Balance(unittest.TestCase):
    def test_balanced_side_halves_every_batch_and_prefix(self):
        for batch in range(5):
            sides = [PN.balanced_side(0, "g", "sentinel", batch, i) for i in range(20)]
            self.assertEqual(sides.count("A"), 10)
            for k in range(1, 21):
                self.assertLessEqual(abs(sides[:k].count("A") - sides[:k].count("B")), 1)
        # seeded, not a fixed pattern
        runs = {tuple(PN.balanced_side(s, "g", "pair", 0, i) for i in range(8)) for s in range(10)}
        self.assertGreater(len(runs), 1)


class Combine(unittest.TestCase):
    ONE = {c: 1.0 for c in PN.CRITERIA}
    ZERO = {c: 0.0 for c in PN.CRITERIA}

    def test_agreeing_orders_keep_the_vote(self):
        self.assertEqual(PN.combine_votes([self.ONE, self.ONE], [None, None], [2, 2]), self.ONE)
        self.assertEqual(PN.combine_votes([self.ZERO, None], [None, None], [2, 0]), self.ZERO)
        self.assertIsNone(PN.combine_votes([None, None], [None, None], [0, 0]))

    def test_a_split_goes_by_score_deltas_then_confidence_else_half(self):
        d_x = {c: 3.0 for c in PN.CRITERIA}   # the ask that chose x scored it 3 higher
        d_y = {c: -1.0 for c in PN.CRITERIA}  # the ask that chose y, by 1
        v = PN.combine_votes([self.ONE, self.ZERO], [d_x, d_y], [2, 2])
        self.assertEqual(v["overall"], 1.0)
        v = PN.combine_votes([self.ONE, self.ZERO], [{c: 1.0 for c in PN.CRITERIA}, {c: -4.0 for c in PN.CRITERIA}],
                             [3, 1])
        self.assertEqual(v["overall"], 0.0)  # deltas before confidence
        even = [{c: 3.0 for c in PN.CRITERIA}, {c: -3.0 for c in PN.CRITERIA}]
        self.assertEqual(PN.combine_votes([self.ONE, self.ZERO], even, [3, 1])["overall"], 1.0)
        self.assertEqual(PN.combine_votes([self.ONE, self.ZERO], even, [1, 3])["overall"], 0.0)
        self.assertEqual(PN.combine_votes([self.ONE, self.ZERO], even, [2, 2])["overall"], 0.5)


class Parse(unittest.TestCase):
    def test_a_misfiled_verdict_is_dropped(self):
        ok = {c: "A" for c in PN.CRITERIA}
        text = json.dumps({"aaaaaaaa": dict(ok, A_file="aaaaaaaa_A.png"),
                           "bbbbbbbb": dict(ok, A_file="cccccccc_A.png"),  # describes another comparison
                           "cccccccc": dict(ok)})                            # no echo: accepted
        files = {"aaaaaaaa": "/x/aaaaaaaa_A.png", "bbbbbbbb": "/x/bbbbbbbb_A.png", "cccccccc": "/x/cccccccc_A.png"}
        got = PN.parse_batch(text, list(files), files)
        self.assertIsNotNone(got["aaaaaaaa"])
        self.assertIsNone(got["bbbbbbbb"])
        self.assertIsNotNone(got["cccccccc"])
        self.assertIsNotNone(PN.parse_batch(text, list(files))["bbbbbbbb"])  # no files given: as before

    def test_seen_fields_are_kept(self):
        r = PN.parse_reply({**{c: "B" for c in PN.CRITERIA}, "seen_A": "logo, cut headline", "seen_B": "all"})
        self.assertEqual((r["seen_A"], r["seen_B"]), ("logo, cut headline", "all"))


class Chunks(unittest.TestCase):
    def test_one_ref_spreads_over_calls_in_as_few_calls(self):
        ids = [f"{g}{i}" for g, k in (("a", 3), ("b", 2), ("c", 1), ("d", 1), ("e", 1)) for i in range(k)]
        out = PN.spread_chunks(ids, 3, lambda a: a[0])
        self.assertEqual(len(out), 3)
        self.assertEqual(sorted(x for c in out for x in c), sorted(ids))
        self.assertTrue(all(len(c) <= 3 for c in out))
        for c in out:
            self.assertEqual(len({a[0] for a in c}), len(c), out)  # no REF twice in one call
        self.assertEqual(PN.spread_chunks([], 8, str), [])
        self.assertEqual([len(c) for c in PN.spread_chunks(list("abcdefghij"), 8, lambda a: "g")], [5, 5])


class Mat(TempEnv):
    def test_matted_pads_and_keeps_the_design(self):
        src = self.tmp / "d.png"
        Image.new("RGB", (40, 30), (200, 10, 10)).save(src)
        out = PN.matted(src, (10, 23, 23, 10), self.tmp / "mat")
        with Image.open(out) as im:
            self.assertEqual(im.size, (40 + 33, 30 + 33))
            self.assertEqual(im.getpixel((10, 23)), (200, 10, 10))
            self.assertEqual(im.getpixel((49, 52)), (200, 10, 10))
            self.assertEqual(im.getpixel((9, 22)), PN.MAT_LINE_RGB)
            self.assertEqual(im.getpixel((0, 0)), PN.MAT_FILL)
        self.assertEqual(PN.matted(src, (10, 23, 23, 10), self.tmp / "mat"), out)  # cached
        # every two images sit at different offsets on both axes or at least one
        pads = list(PN.MAT_PADS.values())
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertNotEqual(pads[i][:2], pads[j][:2])

    def test_prompt_versions(self):
        its = [{"id": "aaaaaaaa", "ref": "/x/aaaaaaaa_ref.png", "A": "/x/aaaaaaaa_A.png", "B": "/x/aaaaaaaa_B.png"}]
        v1 = PN.prompt_for(its, "claude", "v1")
        self.assertNotIn("gray mat", v1)
        self.assertNotIn("at random", v1)
        v2 = PN.prompt_for(its, "agent", "v2", mat=True)
        self.assertIn("assigned at random", v2)
        self.assertIn("gray mat", v2)
        self.assertIn('"A_file": "aaaaaaaa_A.png"', v2)
        self.assertIn("seen_B", v2)


class Protocol(TempEnv):
    def _tiles(self, n=4):
        out = []
        for i in range(n):
            src = _design(i)
            base = baselines.fit_centre(src, {"width": 1080, "height": 1080})
            out.append({"set": "novel", "tile": f"i{i}", "design": f"d{i}", "tier": 1, "source": src,
                        "base": base, "cand": baselines.nudge(base), "kit": None})
        return out

    def _panel(self, name, cfg, judge, draw_id=None):
        base = {"view": 192, "min_sentinels": 4, "small_sentinels": 0}
        return PN.Panel(name, self.tmp / name, dict(base, **cfg), judge=judge, draw_id=draw_id)

    def test_balanced_orders_per_family(self):
        pn = self._panel("bal", {}, BiasedJudge(truthful=True))
        pn.judge_tiles(self._tiles(6), 0, share=0)
        for f in ("claude", "cursor"):
            sides = [a["x_side"] for a in pn.asks.values() if a["family"] == f and a["kind"] == "pair"]
            self.assertEqual(sides.count("A"), 3, f)
        by_tile = {}
        for a in pn.asks.values():
            by_tile.setdefault(a["key"], {})[a["family"]] = a["x_side"]
        self.assertTrue(all(v["claude"] != v["cursor"] for v in by_tile.values()))
        # the images on A and B carry different mats, and which side gets which varies
        lo = PN.MAT_LO - 1
        a_lo = set()
        for a in pn.asks.values():
            with Image.open(a["A"]) as ia, Image.open(a["B"]) as ib:
                la, lb = ia.convert("RGB").getpixel((lo, lo)), ib.convert("RGB").getpixel((lo, lo))
            self.assertNotEqual(la == PN.MAT_LINE_RGB, lb == PN.MAT_LINE_RGB)
            a_lo.add(la == PN.MAT_LINE_RGB)
        self.assertEqual(a_lo, {True, False})

    def test_a_pure_position_bias_shows_and_both_orders_neutralise_it(self):
        # one order, balanced: a judge that always says A is right exactly half the time
        pn = self._panel("one", {}, BiasedJudge())
        res = pn.judge_tiles(self._tiles(4), 0)
        for f in ("claude", "cursor"):
            s = res["sentinels"][f]
            self.assertEqual((s["n"], s["accuracy"], s["prior_A"]), (4, 0.5, 1.0))
            self.assertEqual(s["by_side"], {"A": 1.0, "B": 0.0})
        # both orders: every comparison splits, the deltas cancel, confidence ties -> 0.5 each,
        # and pairs come out at 0.5 rather than at whatever side the candidate sat on
        j = BiasedJudge()
        pn = self._panel("two", {"orders": 2}, j)
        res = pn.judge_tiles(self._tiles(4), 0)
        s = res["sentinels"]["cursor"]
        self.assertEqual((s["n"], s["asks"], s["accuracy"], s["correct"]), (4, 8, 0.5, 2))
        recs = PN.score_tiles(res["tiles"], pn.families)
        self.assertTrue(all(t["family"][f]["overall"] == 0.5 for t in recs for f in ("claude", "cursor")))
        self.assertTrue(all(t["asks"] == 4 for t in res["tiles"]))
        # the second order of a comparison never shares a call with the first
        for p in j.prompts:
            ids = set(re.findall(r"([0-9a-f]{8})_A\.png", p))
            keys = [(pn.asks[a]["kind"], pn.asks[a]["key"], pn.asks[a]["family"]) for a in ids]
            self.assertEqual(len(keys), len(set(keys)))

    def test_both_orders_with_a_good_judge(self):
        pn = self._panel("good2", {"orders": 2}, BiasedJudge(truthful=True))
        res = pn.judge_tiles(self._tiles(4), 0)
        self.assertEqual(res["sentinels"]["claude"]["accuracy"], 1.0)
        recs = PN.score_tiles(res["tiles"], pn.families)
        self.assertTrue(all(t["score"]["overall"] == 1.0 for t in recs))
        self.assertEqual(sum(v["n"] for v in pn.sentinel_kinds()["sentinel"]["claude"].values()), 4)
        # a resumed panel asks nothing again
        j2 = BiasedJudge()
        pn2 = self._panel("good2", {"orders": 2}, j2)
        pn2.judge_tiles(self._tiles(4), 0)
        self.assertEqual(j2.calls, 0)

    def test_draw_id_reproduces_the_draw_under_another_gate_id(self):
        a = self._panel("ga", {}, BiasedJudge(), draw_id="shared")
        b = self._panel("gb", {}, BiasedJudge(), draw_id="shared")
        c = self._panel("gc", {}, BiasedJudge())
        tiles = self._tiles(4)
        sa = a.sentinels(tiles, 6, 0, a.families)
        sb = b.sentinels(tiles, 6, 0, b.families)
        sig = lambda pn, s: [(pn.asks[x]["key"], pn.asks[x]["x_side"]) for x in s["cursor"]]  # noqa: E731
        self.assertEqual(len(sa["cursor"]), 6)
        self.assertEqual(sig(a, sa), sig(b, sb))
        self.assertEqual(c.draw_id, "gc")

    def test_validate_panel_knows_the_new_keys(self):
        self.assertEqual(prereg.validate_panel({"prompt": "v2", "orders": 2, "balance": True, "mat": True}), [])
        errs = prereg.validate_panel({"prompt": "v9", "orders": 3, "balance": "yes", "mat": 1})
        self.assertEqual(len(errs), 4, errs)


if __name__ == "__main__":
    unittest.main()
