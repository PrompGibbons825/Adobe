"""S14 (only the Cursor models the subscription includes) and the pre-registered floor tolerance.

No model is ever called: the Cursor CLI is mocked and the panel runs on the fake judge of
tests.test_gate_run.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from unittest import mock

from tests.test_gate_run import FakeJudge, TempEnv, _design, _p
from tools.gate import baselines, floor as F, panel as PN, prereg as P, run as R

ALLOWED = ["grok-4.7-low", "grok-4.7-medium", "grok-4.7-high", "grok-4.7-xhigh", "cursor-grok-4.6-low",
           "cursor-grok-4.6-medium", "cursor-grok-4.6-high", "cursor-grok-4.6-xhigh", "cursor-grok-4.5-high",
           "composer-2.5"]
REFUSED = ["auto", "grok-4.7-high-fast", "composer-2.5-fast", "cursor-grok-4.5-high-fast", "cursor-grok-4.6-xhigh-fast",
           "gpt-5.6-sol-high", "gemini-3.7-flash-high", "claude-opus-5-5-high", "claude-sonnet-5-thinking-high",
           "grok-4.7", "grok-4.7-max", "cursor-grok-4.5-low", "composer-2", "GROK-4.7-HIGH", " grok-4.7-high", "", None]


class AllowList(TempEnv):
    def test_allow_list(self):
        for m in ALLOWED:
            self.assertTrue(PN.cursor_model_ok(m), m)
            self.assertEqual(PN.require_cursor_model(m), m)
        for m in REFUSED:
            self.assertFalse(PN.cursor_model_ok(m), m)
            with self.assertRaises(PN.ModelNotAllowed):
                PN.require_cursor_model(m)

    def test_defaults_are_grok_and_composer(self):
        self.assertEqual(PN.DEFAULT_PANEL["families"]["cursor"], {"cli": "agent", "model": "grok-4.7-high"})
        self.assertEqual(PN.DEFAULT_PANEL["tiebreak"], {"cli": "agent", "model": "composer-2.5"})
        self.assertEqual(PN.DEFAULT_PANEL["families"]["claude"]["cli"], "claude")
        self.assertTrue(6 <= PN.DEFAULT_PANEL["batch"] <= 12)
        self.assertEqual(PN.DEFAULT_PANEL["agent_daily_cap"], 200)
        self.assertEqual(PN.DEFAULT_PANEL["prompt"], "v2")
        self.assertEqual(P.validate_panel(dict(PN.DEFAULT_PANEL)), [])

    def test_cursor_judge_never_runs_a_refused_model(self):
        cj = PN.CursorJudge(self.tmp, 10)
        with mock.patch.object(PN.subprocess, "run") as run:
            for m in ("gpt-5.6-sol-high", "auto", "grok-4.7-high-fast"):
                with self.assertRaises(PN.ModelNotAllowed):
                    cj("p", m)
            run.assert_not_called()

    def test_panel_refuses_a_refused_cursor_model_even_with_a_fake_judge(self):
        for cfg in ({"families": {"claude": {"cli": "claude", "model": "claude-sonnet-5-5"},
                                  "cursor": {"cli": "agent", "model": "gpt-5.6-sol-high"}}},
                    {"tiebreak": {"cli": "agent", "model": "gemini-3.7-flash-high"}},
                    {"tiebreak": {"cli": "agent", "model": "auto"}}):
            with self.assertRaises(PN.ModelNotAllowed):
                PN.Panel("s14", self.tmp / "priv", cfg, judge=FakeJudge())
        # the claude CLI is not Cursor: its model is not on the Cursor list and need not be
        PN.Panel("s14ok", self.tmp / "priv2", {}, judge=FakeJudge())

    def test_prereg_refuses_a_refused_cursor_model(self):
        for panel, who in (({"families": {"claude": {"cli": "claude", "model": "claude-sonnet-5-5"},
                                          "cursor": {"cli": "agent", "model": "gpt-5.6-sol-high"}}}, "cursor"),
                           ({"tiebreak": {"cli": "agent", "model": "composer-2.5-fast"}}, "tiebreak")):
            errs = P.validate_panel(panel)
            self.assertTrue([e for e in errs if who in e and "not an included Cursor model" in e], errs)
        self.seal(2)
        with self.assertRaisesRegex(P.Refused, "not an included Cursor model"):
            P.check(self.prereg(_p(gid="old", panel={"tiebreak": {"cli": "agent", "model": "gemini-3.7-flash-high"}})))

    def test_cursor_limit_texts_pause(self):
        cj = PN.CursorJudge(self.tmp, 10)
        for err in ("Error: You've hit your usage limit", "You have reached your spend limit",
                    "This model requires on-demand usage", "rate limit exceeded", "Upgrade your plan to continue"):
            with mock.patch.object(PN.subprocess, "run",
                                   return_value=subprocess.CompletedProcess([], 1, stdout="", stderr=err)):
                with self.assertRaises(PN.CapReached, msg=err):
                    cj("p", "grok-4.7-high")


class S14Panel(TempEnv):
    """The panel on the S14 roster: per-model cursor_asks, and a Cursor limit pauses the run with
    no ask moved to another model."""

    def _tiles(self, n=4):
        out = []
        for i in range(n):
            src, tgt = _design(i), {"width": 1080, "height": 1080}
            base = baselines.fit_centre(src, tgt)
            out.append({"set": "novel", "tile": f"i{i}", "design": f"d{i}", "tier": 1, "source": src,
                        "base": base, "cand": baselines.nudge(base), "kit": None})
        return out

    def test_cursor_asks_per_model(self):
        j = FakeJudge(models={"grok-4.7-high": (0.0, 1.0), "claude-sonnet-5-5": (1.0, 1.0)})  # the two split
        pn = PN.Panel("s14u", self.tmp / "p1", {"view": 192, "batch": 6}, judge=j)
        pn.judge_tiles(self._tiles(4), 0, share=0)
        u = pn.usage()
        self.assertEqual(set(u["cursor_asks_by_model"]), {"grok-4.7-high", "composer-2.5"})
        self.assertEqual(sum(u["cursor_asks_by_model"].values()), u["cursor_asks"])
        led = [json.loads(ln) for ln in PN.agent_ledger().read_text().splitlines()]
        self.assertEqual({r["model"] for r in led}, {"grok-4.7-high", "composer-2.5"})
        self.assertNotIn("gpt-5.6-sol-high", j.by_model)

    def test_a_grok_limit_pauses_and_nothing_falls_back(self):
        fj = FakeJudge()

        def limited(prompt, model):
            if model == "grok-4.7-high":
                raise PN.CapReached("agent grok-4.7-high: usage limit")
            return fj(prompt, model)

        limited.bind = fj.bind
        pn = PN.Panel("s14l", self.tmp / "p2", {"view": 192}, judge=limited)
        with self.assertRaises(PN.CapReached) as cm:
            pn.judge_tiles(self._tiles(2), 0, share=0)
        self.assertGreater(cm.exception.reset_at, PN.time.time())  # waits for the next UTC day
        self.assertFalse(any(pn.asks[a]["family"] == "cursor" for a in pn.labels))
        self.assertEqual(set(fj.by_model) - {"claude-sonnet-5-5"}, set())  # no other model was asked
        self.assertTrue(all(pn.asks[a]["model"] == "grok-4.7-high" for a in pn.asks if pn.asks[a]["family"] == "cursor"))


# ---- floor tolerance ---------------------------------------------------------------------------

def _m(**counts):
    m = {k: [] for k in F.MUST_BE_ZERO + F.MUST_NOT_RISE + F.REPORT_ONLY}
    for k, n in counts.items():
        m[k] = [f"{k} {i}" for i in range(n)]
    return m


def _run(**counts):
    """Five tiles; the first carries the counts."""
    return {f"t{i}": (_m(**counts) if i == 0 else _m()) for i in range(5)}


class FloorTolerance(unittest.TestCase):
    def test_default_is_no_rise(self):
        r = F.compare_tiles(_run(contrast=5), _run(contrast=4))
        self.assertFalse(r["pass"])
        self.assertEqual(r["fails"], ["contrast: 5 > base 4"])
        self.assertEqual((r["tolerance"], r["tolerated"]), ({}, {}))
        self.assertEqual(F.compare_tiles(_run(contrast=5), _run(contrast=4), {"contrast": 0})["pass"], False)

    def test_tolerance_lets_a_small_rise_through_and_records_it(self):
        r = F.compare_tiles(_run(contrast=5), _run(contrast=4), {"contrast": 1})
        self.assertTrue(r["pass"], r["fails"])
        self.assertEqual(r["tolerated"], {"contrast": {"cand": 5, "base": 4, "rise": 1, "tolerance": 1}})
        r = F.compare_tiles(_run(contrast=6), _run(contrast=4), {"contrast": 1})
        self.assertFalse(r["pass"])
        self.assertEqual(r["fails"], ["contrast: 6 > base 4 (tolerance 1)"])
        # no rise: nothing tolerated
        self.assertEqual(F.compare_tiles(_run(contrast=4), _run(contrast=4), {"contrast": 1})["tolerated"], {})
        # the tolerance is per row: a hierarchy rise still fails
        r = F.compare_tiles(_run(contrast=5, hierarchy=1), _run(contrast=4), {"contrast": 1})
        self.assertEqual(r["fails"], ["hierarchy: 1 > base 0"])

    def test_hard_rules_stay_zero_tolerance(self):
        r = F.compare_tiles(_run(hard=1, contrast=5), _run(hard=1, contrast=4), {"contrast": 3})
        self.assertFalse(r["pass"])
        self.assertIn("hard: 1 (must be 0)", r["fails"])
        for bad in ({"hard": 1}, {"ghost": 2}, {"hidden": 1}, {"bleed": 1}, {"nope": 1}, {"contrast": -1},
                    {"contrast": 1.5}, {"contrast": True}, [1]):
            self.assertTrue(F.tolerance_errors(bad), bad)
            with self.assertRaises(ValueError):
                F.compare_tiles(_run(), _run(), bad)
        self.assertEqual(F.tolerance_errors(None), [])
        self.assertEqual(F.tolerance_errors({"contrast": 2, "promise": 0, "hierarchy": 1}), [])


class FloorTolerancePrereg(TempEnv):
    def test_prereg_schema(self):
        self.seal(2)
        for gid, tol, msg in (("t1", {"hard": 1}, "zero-tolerance"), ("t2", {"hidden": 1}, "zero-tolerance"),
                              ("t3", {"contrast": -2}, "non-negative"), ("t4", {"x": 1}, "not a floor row")):
            with self.assertRaisesRegex(P.Refused, msg):
                P.check(self.prereg(_p(gid=gid, cand=gid[-1] * 40, floor_tolerance=tol)))
        self.assertEqual(P.check(self.prereg(_p(gid="t5", cand="5" * 40, floor_tolerance={"contrast": 1})))
                         ["prereg"]["floor_tolerance"], {"contrast": 1})

    def test_make_writes_it_only_when_asked(self):
        with mock.patch.object(P, "resolve", side_effect=lambda ref, repo=P.REPO: {"b": "b" * 40, "c": "c" * 40}[ref]):
            self.assertNotIn("floor_tolerance", P.make("mk", "b", "c", sets=["novel-v1"]))
            m = P.make("mk2", "b", "c", sets=["novel-v1"], floor_tolerance={"contrast": 1})
            self.assertEqual(m["floor_tolerance"], {"contrast": 1})
            with self.assertRaisesRegex(P.Refused, "zero-tolerance"):
                P.make("mk3", "b", "c", sets=["novel-v1"], floor_tolerance={"hard": 1})

    def test_run_floor_passes_and_records_the_tolerance(self):
        seen = []

        def fake_check(src, co, bo, kit, tolerance=None):
            seen.append(tolerance)
            return F.compare_tiles(_run(contrast=5), _run(contrast=4), tolerance)

        sets = [{"version": "novel-v1", "kit": None}]
        (self.tmp / "a").mkdir()
        (self.tmp / "b").mkdir()
        with mock.patch.object(F, "check_tiles", side_effect=fake_check):
            a = R.run_floor(["tiles"], {}, sets, self.tmp / "a", False)
            b = R.run_floor(["tiles"], {}, sets, self.tmp / "b", False, tolerance={"contrast": 1})
        self.assertEqual(seen, [None, {"contrast": 1}])
        self.assertFalse(a["pass"])
        self.assertTrue(b["pass"])
        self.assertEqual(b["parts"]["tiles"]["tolerance"], {"contrast": 1})
        self.assertEqual(b["parts"]["tiles"]["sets"]["novel-v1"]["tolerated"]["contrast"]["rise"], 1)
        line = R.verdict_line({"prereg": {"candidate": {"label": "c"}, "base": {"label": "b"}, "sets": ["novel-v1"]},
                               "floor": b, "verdict": "ACCEPT"})
        self.assertIn("floor PASS", line)
        self.assertIn('tolerated contrast +1/1 (tolerance {"contrast": 1})', line)


if __name__ == "__main__":
    unittest.main()
