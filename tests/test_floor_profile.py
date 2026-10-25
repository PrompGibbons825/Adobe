"""Setup track (G6): the whole-engine floor profile and paired soft-row counts."""

import json
import unittest
from pathlib import Path
from unittest import mock

from tools.gate import floor as F
from tools.gate import prereg as P
from tools.gate import run as R


def _m(**counts):
    m = {k: [] for k in F.MUST_BE_ZERO + F.MUST_NOT_RISE + F.REPORT_ONLY}
    for k, n in counts.items():
        m[k] = [f"{k} {i}" for i in range(n)]
    return m


class PairedCounts(unittest.TestCase):
    def test_a_tile_the_base_crashed_on_does_not_count_against_the_candidate(self):
        base = {"a": _m(hard=1), "b": _m(contrast=1)}  # base crashed on a: no soft counts there
        cand = {"a": _m(contrast=2, hidden=1), "b": _m(contrast=1)}
        r = F.compare_tiles(cand, base)
        self.assertTrue(r["pass"], r["fails"])
        self.assertEqual(r["paired"]["tiles"], 1)
        self.assertEqual(r["totals"]["contrast"], 3)  # the report still shows every tile

    def test_a_crashed_candidate_tile_still_fails_hard(self):
        r = F.compare_tiles({"a": _m(hard=1), "b": _m()}, {"a": _m(contrast=3), "b": _m()})
        self.assertFalse(r["pass"])
        self.assertEqual(r["fails"], ["hard: 1 (must be 0)"])


class WholeEngineProfile(unittest.TestCase):
    def test_soft_rows_are_reported_not_failed(self):
        rr = F.PROFILES["whole-engine"]["report_rows"]
        r = F.compare_tiles({"a": _m(contrast=4, promise=2, hierarchy=1)}, {"a": _m()}, None, rr)
        self.assertTrue(r["pass"], r["fails"])
        self.assertEqual(r["reported"]["contrast"], {"cand": 4, "base": 0, "rise": 4})

    def test_hidden_hard_and_ghost_still_decide(self):
        rr = F.PROFILES["whole-engine"]["report_rows"]
        self.assertFalse(F.compare_tiles({"a": _m(hidden=1)}, {"a": _m()}, None, rr)["pass"])
        self.assertFalse(F.compare_tiles({"a": _m(ghost=1)}, {"a": _m()}, None, rr)["pass"])
        self.assertFalse(F.compare_tiles({"a": _m(hard=1)}, {"a": _m()}, None, rr)["pass"])
        with self.assertRaises(ValueError):
            F.compare_tiles({"a": _m()}, {"a": _m()}, None, ("hidden",))

    def test_profile_suites(self):
        self.assertEqual(F.PROFILES["whole-engine"]["suites"], ["stress"])
        self.assertEqual(F.PROFILES["default"]["suites"], ["tests.run", "stress", "unittest"])
        self.assertEqual(F.profile_errors("whole-engine"), [])
        self.assertEqual(F.profile_errors(None), [])
        self.assertTrue(F.profile_errors("lenient"))

    def test_run_floor_uses_the_profile(self):
        import tempfile

        seen = {}

        def fake_suites(root, tests_from="gate", only=None):
            seen["only"] = only
            return {"pass": True, "parts": {}}

        def fake_check(src, co, bo, kit, tolerance=None, report_rows=()):
            seen["report_rows"] = report_rows
            return F.compare_tiles({"a": _m(contrast=2)}, {"a": _m()}, tolerance, report_rows)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(F, "run_suites", side_effect=fake_suites), \
                mock.patch.object(F, "check_tiles", side_effect=fake_check):
            rep = R.run_floor(["suites", "tiles"], {"cand": Path(tmp), "base": Path(tmp)},
                              [{"version": "novel-v1", "kit": None}], Path(tmp), True, None, "whole-engine")
        self.assertTrue(rep["pass"])
        self.assertEqual(seen["only"], ["stress"])
        self.assertEqual(rep["profile"], "whole-engine")
        self.assertEqual(rep["parts"]["tiles"]["sets"]["novel-v1"]["reported"]["contrast"]["rise"], 2)
        line = R.verdict_line({"prereg": {"candidate": {"label": "c"}, "base": {"commit": "b" * 40}, "sets": []},
                               "floor": rep, "verdict": "REJECT"})
        self.assertIn("profile whole-engine (reported contrast +2)", line)

    def test_prereg_validates_and_make_writes_it_only_when_asked(self):
        self.assertTrue(any("floor_profile" in e for e in P.validate(
            {"id": "x", "base": {"commit": "b" * 40}, "candidate": {"commit": "c" * 40}, "sets": ["novel-v1"],
             "floor_profile": "lenient"})))
        with mock.patch.object(P, "resolve", side_effect=lambda ref, repo=P.REPO: {"b": "b" * 40, "c": "c" * 40}[ref]):
            self.assertNotIn("floor_profile", P.make("mk", "b", "c", sets=["novel-v1"]))
            m = P.make("mk2", "b", "c", sets=["novel-v1"], floor_profile="whole-engine")
            self.assertEqual(m["floor_profile"], "whole-engine")
            self.assertEqual(P.make("mk3", "b", "c", sets=["novel-v1"], floor=[])["floor"], [])


if __name__ == "__main__":
    unittest.main()
