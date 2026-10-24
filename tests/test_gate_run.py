"""Tests of the gate's decision path (tools/gate/{stats,prereg,panel,run,visible,tiers,baselines}).

No model is ever called: the panel runs with fake judges that look up the hidden side. The
end-to-end tests use a throwaway sealed store and two tiny builtin-style engines.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.gate import baselines, panel as PN, prereg as P, run as R, sealed, stats
from tools.gate.tiers import tier_of
from tools.gate.visible import hidden_elements


# ---- helpers ---------------------------------------------------------------------------------

def _design(i: int, w: int = 500, h: int = 400) -> dict:
    rng = random.Random(i)
    return {"canvas": {"width": w, "height": h}, "elements": [
        {"id": "bg", "type": "shape", "x": 0, "y": 0, "width": w, "height": h, "z_index": 0,
         "shape_type": "rectangle", "fill_color": f"rgba({rng.randrange(150, 255)},230,210,1)"},
        {"id": "card", "type": "shape", "x": 40, "y": 60 + rng.randrange(40), "width": 200, "height": 120,
         "z_index": 1, "shape_type": "rectangle", "fill_color": "rgba(30,40,90,1)", "corner_radius": 8},
        {"id": "t", "type": "text", "x": 60, "y": 250, "width": 360, "height": 60, "z_index": 2,
         "content": f"Golden hour {i}", "font_family": "Solstice Sans", "font_size": 40,
         "color": "rgba(20,20,20,1)"},
    ]}


TOP_ENGINE = '''
import json

def resize(source, target_canvas):
    W, H = float(target_canvas["width"]), float(target_canvas["height"])
    sw, sh = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    s = min(W / sw, H / sh)
    els = []  # fit, but top-left instead of centred
    for e in source["elements"]:
        o = json.loads(json.dumps(e))
        for k in ("x", "y", "width", "height", "font_size", "corner_radius"):
            if isinstance(o.get(k), (int, float)):
                o[k] = o[k] * s
        if "font_size" in o:
            o["font_size"] = max(1, int(round(o["font_size"])))
        els.append(o)
    return {"canvas": {"width": target_canvas["width"], "height": target_canvas["height"]}, "elements": els}
'''


def _engine(root: Path, src: str) -> Path:
    (root / "engine").mkdir(parents=True, exist_ok=True)
    (root / "engine" / "__init__.py").write_text("")
    (root / "engine" / "resize.py").write_text(src)
    return root


class FakeJudge:
    """Chooses the hidden "x" side with probability ``p_x`` on pairs (x = the candidate) and
    ``p_sent`` on sentinels (x = the good design); scores follow the choice."""

    def __init__(self, p_x: float = 1.0, p_sent: float = 1.0, seed: int = 0):
        self.p_x, self.p_sent, self.rng, self.calls = p_x, p_sent, random.Random(seed), 0
        self.panel = None

    def bind(self, panel):
        self.panel = panel

    def __call__(self, prompt: str, model: str):
        self.calls += 1
        a_path = re.search(r"A\s+(\S+_A\.png)", prompt).group(1)
        aid = Path(a_path).name.split("_")[0]
        ask = self.panel.asks[aid]
        p = self.p_sent if ask["kind"] == "sentinel" else self.p_x
        x = ask["x_side"]
        y = "B" if x == "A" else "A"
        pick = x if self.rng.random() < p else y
        other = y if pick == x else x
        rep = {c: pick for c in PN.CRITERIA}
        rep.update(confidence=2, why="fake", scores={pick: {c: 8 for c in PN.CRITERIA},
                                                     other: {c: 5 for c in PN.CRITERIA}})
        return "Here you go:\n" + json.dumps(rep), {"input": 9000, "cache_read": 1000, "output": 120}


class TempEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = {"CHAMELEON_SEALED_ROOT": str(self.tmp / "store"),
               "CHAMELEON_GATE_MANIFESTS": str(self.tmp / "manifests"),
               "CHAMELEON_GATE_DEV": str(self.tmp / "dev"),
               "CHAMELEON_GATE_RUNS": str(self.tmp / "runs.jsonl"),
               "CHAMELEON_JUDGE_ROOT": str(self.tmp / "judges")}
        p = mock.patch.dict(os.environ, env)
        p.start()
        self.addCleanup(p.stop)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
            subprocess.run(["git", "-C", str(self.repo), *args], check=True)

    def seal(self, n_designs: int = 6, set_name: str = "novel", targets=((1080, 1080),)) -> str:
        st = self.tmp / f"stage-{set_name}"
        (st / "items").mkdir(parents=True)
        (st / "assets").mkdir()
        k = 0
        for i in range(n_designs):
            for (w, h) in targets:
                t = {"task_id": f"i{k:03d}", "source": _design(i), "target_canvas": {"width": w, "height": h},
                     "meta": {"design": f"d{i}"}}
                (st / "items" / f"i{k:03d}.json").write_text(json.dumps(t))
                k += 1
        return sealed.add(st, set_name)["version"]

    def prereg(self, p: dict, commit: bool = True, name: str = None) -> Path:
        path = self.repo / f"{name or p['id']}.json"
        path.write_text(json.dumps(p))
        if commit:
            subprocess.run(["git", "-C", str(self.repo), "add", path.name], check=True)
            subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "prereg"], capture_output=True)
        return path


# ---- stats -----------------------------------------------------------------------------------

def _tiles(scores, set_name="novel", tier=1, crit_scores=None):
    out = []
    for i, s in enumerate(scores):
        sc = {c: s for c in stats.CRITERIA}
        sc.update(crit_scores or {})
        out.append({"tile": f"t{i}", "set": set_name, "design": f"d{i}", "tier": tier, "score": sc})
    return out


class Stats(unittest.TestCase):
    def test_spending_adds_up_and_is_stingy_early(self):
        a = stats.spend(0.05, [1 / 3, 2 / 3, 1])
        self.assertAlmostEqual(sum(a), 0.05, places=6)
        self.assertLess(a[0], 0.002)
        self.assertEqual(stats.spend(0.05, [1.0]), [0.05])

    def test_look_sizes(self):
        self.assertEqual(stats.look_sizes(200, [40, 80, 120]), [40, 80, 120])
        self.assertEqual(stats.look_sizes(90, [40, 80, 120]), [40, 80, 90])
        self.assertEqual(stats.look_sizes(25, [40, 80, 120]), [25])
        self.assertEqual(stats.look_sizes(0, [40, 80, 120]), [])

    def test_design_is_the_unit(self):
        tiles = [{"set": "novel", "design": "a", "tier": 1, "score": {"overall": 1.0}} for _ in range(10)]
        tiles.append({"set": "novel", "design": "b", "tier": 1, "score": {"overall": 0.0}})
        self.assertEqual(sorted(stats.design_scores(tiles).values()), [0.0, 1.0])

    def test_clear_win_accepts_and_tie_rejects(self):
        r = {"boot": 2000}
        win = stats.evaluate(_tiles([1.0] * 30 + [0.0] * 6), r, 0.05, final=True)
        self.assertEqual(win["decision"], "ACCEPT")
        tie = stats.evaluate(_tiles([1.0, 0.0] * 20), r, 0.05, final=True)
        self.assertEqual(tie["decision"], "REJECT")
        self.assertIn("no significant win", tie["reasons"][0])

    def test_point_floor(self):
        # many designs at 0.55: the lower bound clears 0.5 but the point is below 0.58
        ev = stats.evaluate(_tiles([0.55] * 400), {"boot": 2000}, 0.05, final=True)
        self.assertGreater(ev["cells"]["novel t1"]["lower"], 0.5)
        self.assertEqual(ev["decision"], "REJECT")

    def test_harm_on_tier2_or_criterion_rejects_a_win(self):
        t = _tiles([1.0] * 30) + _tiles([0.0] * 10, tier=2)
        for x in t[30:]:
            x["design"] += "x"
        ev = stats.evaluate(t, {"boot": 2000}, 0.05, final=True)
        self.assertIn("novel t2", ev["harm"])
        self.assertEqual(ev["decision"], "REJECT")
        ev = stats.evaluate(_tiles([1.0] * 30, crit_scores={"readable": 0.0}), {"boot": 2000}, 0.05, final=True)
        self.assertEqual(ev["harm"], ["criterion readable"])
        self.assertEqual(ev["decision"], "REJECT")
        crello = _tiles([0.0] * 10, set_name="crello")
        ev = stats.evaluate(_tiles([1.0] * 30) + crello, {"boot": 2000}, 0.05, final=True)
        self.assertIn("crello t1", ev["harm"])

    def test_too_few_designs_never_win(self):
        ev = stats.evaluate(_tiles([1.0] * 10), {"boot": 500}, 0.05, final=True)
        self.assertEqual(ev["decision"], "REJECT")
        self.assertIn("(< 20)", ev["reasons"][0])

    def test_small_harm_cells_are_not_significant(self):
        t = _tiles([1.0] * 30) + [dict(x, design="z" + x["design"]) for x in _tiles([0.0] * 3, tier=2)]
        self.assertEqual(stats.evaluate(t, {"boot": 2000}, 0.05, final=True)["decision"], "ACCEPT")

    def test_sequential_stops_early_on_a_clear_win_and_futility(self):
        rule = {"boot": 2000}
        t = _tiles([1.0] * 120)
        order = [f"novel/d{i}" for i in range(120)]
        r = stats.sequential(t, order, rule)
        self.assertEqual((r["decision"], r["look"]), ("ACCEPT", 1))
        r = stats.sequential(_tiles([0.0, 0.2, 0.4, 0.3] * 30), order, rule)
        self.assertEqual((r["decision"], r["look"]), ("REJECT", 1))

    def test_null_false_win_rate_is_controlled(self):
        """Pure-noise panels (each design a coin flip over 2-4 votes) are accepted <= 5% of the time."""
        rng = np.random.default_rng(3)
        rule = {"boot": 1000}
        wins = 0
        sims = 120
        for s in range(sims):
            sc = [float(np.mean(rng.integers(0, 2, size=rng.choice([2, 4])))) for _ in range(120)]
            t = _tiles(sc)
            order = [f"novel/d{i}" for i in range(120)]
            wins += stats.sequential(t, order, rule, seed=s)["decision"] == "ACCEPT"
        self.assertLessEqual(wins / sims, 0.05)

    def test_null_permutation_rate(self):
        rng = np.random.default_rng(5)
        t = _tiles([float(x) for x in rng.integers(0, 2, size=80)])
        order = [f"novel/d{i}" for i in range(80)]
        r = stats.null_false_win_rate(t, order, {"boot": 500}, n_perm=40, seed=1, boot=500)
        self.assertLessEqual(r["rate"], 0.1)
        strong = stats.null_false_win_rate(_tiles([1.0] * 80), order, {"boot": 500}, n_perm=10, boot=500)
        self.assertEqual(strong["observed"], "ACCEPT")


# ---- tiers, visibility, baselines ------------------------------------------------------------

class Small(unittest.TestCase):
    def test_tiers(self):
        for w, h in ((1080, 1080), (1080, 1350), (1080, 1920), (1280, 720), (1920, 1080), (1200, 628),
                     (1200, 630), (540, 960)):
            self.assertEqual(tier_of({"width": w, "height": h}), 1, (w, h))
        for w, h in ((728, 90), (300, 250), (160, 600), (1500, 500), (1000, 1500), (50, 50), (300, 300)):
            self.assertEqual(tier_of({"width": w, "height": h}), 2, (w, h))

    def test_hidden_elements(self):
        src = _design(1)
        tgt = {"width": 500, "height": 400}
        self.assertEqual(hidden_elements(src, json.loads(json.dumps(src)), tgt), [])
        out = json.loads(json.dumps(src))
        out["elements"][2]["opacity"] = 0
        self.assertTrue(hidden_elements(src, out, tgt)[0].startswith("t: opacity"))
        out = json.loads(json.dumps(src))
        out["elements"][2]["font_size"] = 3
        self.assertIn("floor", hidden_elements(src, out, tgt)[0])
        out = json.loads(json.dumps(src))
        out["elements"].append({"id": "lid", "type": "shape", "x": 0, "y": 0, "width": 500, "height": 400,
                                "z_index": 9, "shape_type": "rectangle", "fill_color": "rgba(0,0,0,1)"})
        self.assertEqual(len(hidden_elements(src, out, tgt)), 3)  # bg, card and text under the lid
        out = json.loads(json.dumps(src))
        out["elements"] = out["elements"][:2]
        self.assertEqual(hidden_elements(src, out, tgt), ["t: missing"])

    def test_fit_centre_and_nudge(self):
        src = _design(2, 500, 400)
        o = baselines.fit_centre(src, {"width": 1000, "height": 1000})
        bg = o["elements"][0]
        self.assertEqual((bg["x"], bg["y"], bg["width"], bg["height"]), (0, 100, 1000, 800))
        self.assertEqual(o["elements"][2]["font_size"], 80)
        n = baselines.nudge(o)
        self.assertEqual((n["elements"][0]["x"], n["elements"][0]["y"]), (0, 101))  # full width: moves down
        self.assertEqual(n["elements"][1]["x"], o["elements"][1]["x"] + 1)
        full = baselines.nudge(baselines.fit_centre(src, {"width": 500, "height": 400}))
        self.assertEqual(full["elements"][0]["x"], 0)


# ---- prereg ----------------------------------------------------------------------------------

def _p(gid="g1", cand="c" * 40, sets=("novel-v1",), **kw):
    p = {"id": gid, "base": {"commit": "b" * 40, "label": "base"}, "candidate": {"commit": cand, "label": "cand"},
         "sets": list(sets), "floor": ["tiles"], "rule": {"boot": 500}, "panel": {"view": 256},
         "anchors": {"parity_designs": 0}}
    p.update(kw)
    return p


class Prereg(TempEnv):
    def test_refuses_uncommitted_and_dirty(self):
        self.seal(2)
        path = self.prereg(_p(), commit=False)
        with self.assertRaisesRegex(P.Refused, "not committed"):
            P.check(path)
        subprocess.run(["git", "-C", str(self.repo), "add", path.name], check=True)
        with self.assertRaisesRegex(P.Refused, "uncommitted"):
            P.check(path)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "x"], check=True)
        self.assertEqual(P.check(path)["candidate_key"], "c" * 40)
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(P.Refused, "uncommitted"):
            P.check(path)

    def test_refuses_unsealed_sets_and_bad_rules(self):
        self.seal(2)
        with self.assertRaisesRegex(P.Refused, "not sealed"):
            P.check(self.prereg(_p(sets=["novel-v2"])))
        with self.assertRaisesRegex(P.Refused, "unknown rule keys"):
            P.check(self.prereg(_p(gid="g2", rule={"alpha": 0.5})))
        with self.assertRaisesRegex(P.Refused, "null run"):
            P.check(self.prereg(_p(gid="g3", null=True)))

    def test_one_run_per_candidate_per_sealed_version(self):
        self.seal(2)
        chk = P.check(self.prereg(_p()))
        P.record(chk, "started")
        with self.assertRaisesRegex(P.Refused, "already ran"):
            P.check(self.prereg(_p()))
        with self.assertRaisesRegex(P.Refused, "one run per candidate"):
            P.check(self.prereg(_p(gid="g1-again")))
        self.assertEqual(P.check(self.prereg(_p()), resume=True)["prereg"]["id"], "g1")
        # another candidate is fine; so is the same candidate on a new sealed version
        P.check(self.prereg(_p(gid="other", cand="d" * 40)))
        self.seal(2)
        P.check(self.prereg(_p(gid="g1-v2", sets=["novel-v2"])))
        with self.assertRaisesRegex(P.Refused, "nothing to resume"):
            P.check(self.prereg(_p(gid="never", cand="f" * 40)), resume=True)

    def test_null_and_builtin_keys(self):
        self.assertEqual(P.cand_key({"commit": "a", "transform": "nudge"}), "a+nudge")
        self.assertEqual(P.cand_key({"builtin": "fit_centre"}), "builtin:fit_centre")


# ---- panel -----------------------------------------------------------------------------------

class Panel(TempEnv):
    def test_parse_reply(self):
        good = {c: "A" for c in PN.CRITERIA}
        self.assertEqual(PN.parse_reply("noise " + json.dumps(good) + " tail")["overall"], "A")
        self.assertIsNone(PN.parse_reply(json.dumps(dict(good, brand="C"))))
        self.assertIsNone(PN.parse_reply("I cannot see the images"))
        r = PN.parse_reply(json.dumps(dict(good, scores={"A": {c: 7 for c in PN.CRITERIA},
                                                         "B": {c: "6" for c in PN.CRITERIA}})))
        self.assertEqual(r["scores"]["B"]["overall"], 6.0)

    def test_prompt_is_small_and_blind(self):
        pr = PN.prompt_for(Path("/j/x_ref.png"), Path("/j/x_A.png"), Path("/j/x_B.png"))
        self.assertLess(len(pr), 4000)
        for word in ("candidate", "base", "engine version", "sealed"):
            self.assertNotIn(word, pr.lower())

    def _tiles(self, n=4):
        out = []
        for i in range(n):
            src = _design(i)
            tgt = {"width": 1080, "height": 1080}
            base = baselines.fit_centre(src, tgt)
            cand = baselines.nudge(base)
            out.append({"set": "novel", "tile": f"i{i}", "design": f"d{i}", "tier": 1, "source": src,
                        "base": base, "cand": cand, "kit": None})
        return out

    def test_judge_tiles_with_a_fake_judge(self):
        j = FakeJudge(p_x=1.0)
        pn = PN.Panel("g", self.tmp / "private", {"view": 192}, judge=j)
        res = pn.judge_tiles(self._tiles(), 0)
        self.assertEqual(len(res["tiles"]), 4)
        self.assertTrue(all(t["score"]["overall"] == 1.0 for t in res["tiles"]))
        self.assertFalse(any(t["second_judge"] for t in res["tiles"]))
        self.assertEqual(res["sentinels"]["n"], 4)  # min_sentinels
        self.assertEqual(res["sentinels"]["accuracy"], 1.0)
        self.assertEqual(j.calls, 8 + 4)
        self.assertAlmostEqual(res["tiles"][0]["delta"]["overall"], 3.0)
        # the judge folder holds only blind names
        names = [p.name for p in (self.tmp / "judges" / "g" / "asks").iterdir()]
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{8}_(ref|A|B)\.png", n) for n in names), names[:3])
        # sides are randomised per tile
        sides = {pn.asks[a]["x_side"] for a in pn.asks if pn.asks[a]["kind"] == "pair"}
        self.assertEqual(sides, {"A", "B"})
        # a resumed panel re-asks nothing
        j2 = FakeJudge()
        pn2 = PN.Panel("g", self.tmp / "private", {"view": 192}, judge=j2)
        pn2.judge_tiles(self._tiles(), 0)
        self.assertEqual(j2.calls, 0)
        self.assertEqual(pn2.usage()["context_tokens_mean"], 10000)

    def test_second_judge_only_on_disagreement(self):
        j = FakeJudge(p_x=0.5, seed=4)
        pn = PN.Panel("g2", self.tmp / "private2", {"view": 192}, judge=j)
        res = pn.judge_tiles(self._tiles(6), 0, share=0)
        seconds = [t for t in res["tiles"] if t["second_judge"]]
        self.assertTrue(seconds)
        for t in res["tiles"]:
            self.assertEqual(t["asks"], 4 if t["second_judge"] else 2)
            if not t["second_judge"]:
                self.assertIn(t["score"]["overall"], (0.0, 1.0))

    def test_failed_replies_are_retried_then_dropped(self):
        calls = []

        def bad(prompt, model):
            calls.append(1)
            return "no json here", {}

        pn = PN.Panel("g3", self.tmp / "private3", {"view": 192, "retries": 1}, judge=bad)
        res = pn.judge_tiles(self._tiles(1), 0, share=0)
        self.assertEqual(res["tiles"], [])
        self.assertEqual(len(calls), 4)
        self.assertEqual(pn.usage()["failed"], 2)


# ---- run end to end --------------------------------------------------------------------------

class Run(TempEnv):
    def setUp(self):
        super().setUp()
        self.version = self.seal(6, targets=((1080, 1080), (728, 90)))
        self.roots = {"b" * 40: baselines.write_builtin("fit_centre", self.tmp / "eng-base"),
                      "c" * 40: _engine(self.tmp / "eng-top", TOP_ENGINE)}

    def mat(self, spec, dest):
        if spec.get("builtin"):
            return baselines.write_builtin(spec["builtin"], dest)
        return self.roots[spec["commit"]]

    def go(self, p, judge, **kw):
        out = self.tmp / "out"
        rep = R.run(self.prereg(p), judge=judge, out_root=out, materializer=self.mat, **kw)
        gate = json.loads((out / p["id"] / "gate.json").read_text())
        return rep, gate

    def test_accept_with_a_judge_that_prefers_the_candidate(self):
        p = _p(sets=[self.version], proxy=False, rule={"boot": 500, "looks": [3, 6], "min_win_designs": 3})
        rep, gate = self.go(p, FakeJudge(p_x=1.0))
        self.assertTrue(rep["verdict"].startswith("ACCEPT"), rep["line"])
        self.assertRegex(rep["line"], r"^GATE cand vs base \| floor PASS hidden \d+ \(base \d+\) \| ")
        self.assertIn(f"{self.version} t1 1.00", rep["line"])
        self.assertIn("t2 1.00", rep["line"])
        self.assertIn("sentinels 1.00", rep["line"])
        self.assertEqual(gate["panel"]["looks"][0]["designs"], 3)
        self.assertIn("hidden", gate["floor"]["parts"]["tiles"]["sets"][self.version]["totals"])
        # aggregates only: no tile id leaves the private folder
        self.assertNotRegex(json.dumps({k: v for k, v in gate.items() if k != "prereg"}), r'"i0\d\d')
        runs = P.read_runs()
        self.assertEqual([r["status"] for r in runs], ["started", "done"])
        self.assertEqual(runs[-1]["verdict"], "ACCEPT")
        # the blind images are gone after the run
        self.assertFalse((self.tmp / "judges" / "g1" / "asks").exists())
        # and the same candidate cannot run again on this version
        with self.assertRaises(P.Refused):
            self.go(_p(gid="g1b", sets=[self.version], proxy=False), FakeJudge())

    def test_reject_when_the_judge_prefers_the_base(self):
        p = _p(sets=[self.version], proxy=False, rule={"boot": 500, "looks": [6]})
        rep, gate = self.go(p, FakeJudge(p_x=0.0))
        self.assertTrue(rep["verdict"].startswith("REJECT"), rep["line"])
        self.assertIn("harm", rep["line"])

    def test_bad_sentinels_invalidate(self):
        p = _p(sets=[self.version], proxy=False, rule={"boot": 500, "looks": [6]})
        rep, _ = self.go(p, FakeJudge(p_x=1.0, p_sent=0.0))
        self.assertTrue(rep["verdict"].startswith("INVALID"), rep["line"])

    def test_refuses_without_committed_prereg(self):
        path = self.prereg(_p(sets=[self.version]), commit=False)
        with self.assertRaisesRegex(P.Refused, "not committed"):
            R.run(path, judge=FakeJudge(), out_root=self.tmp / "out", materializer=self.mat)
        self.assertEqual(P.read_runs(), [])
        self.assertFalse((self.tmp / "out").exists())

    def test_null_run(self):
        p = _p(gid="null1", cand="b" * 40, sets=[self.version], proxy=False, null=True, floor=[],
               rule={"boot": 300, "looks": [6]})
        p["candidate"]["transform"] = "nudge"
        rep, gate = self.go(p, FakeJudge(p_x=0.5, seed=1))
        self.assertIn("null false-win", rep["line"])
        self.assertTrue(rep["verdict"].startswith("NULL"), rep["line"])
        self.assertEqual(gate["null"]["n_perm"], 200)

    def test_builtin_candidate_and_proxy(self):
        p = {"id": "fc", "base": {"commit": "c" * 40, "label": "top"}, "candidate": {"builtin": "fit_centre"},
             "sets": [self.version], "floor": ["tiles"], "rule": {"boot": 300, "looks": [6]},
             "panel": {"view": 192}, "anchors": {"parity_designs": 0}}
        rep, gate = self.go(p, FakeJudge(p_x=0.0))
        self.assertIn("proxy", gate)
        self.assertTrue(rep["verdict"].startswith("REJECT"), rep["line"])

    def test_floor_failure_stops_before_the_panel(self):
        bad = _engine(self.tmp / "eng-bad", TOP_ENGINE.replace("els.append(o)", 'o["x"] += 5000\n        els.append(o)'))
        self.roots["e" * 40] = bad
        j = FakeJudge()
        rep, gate = self.go(_p(gid="bad", cand="e" * 40, sets=[self.version]), j)
        self.assertEqual(rep["verdict"], "REJECT (floor)")
        self.assertEqual(j.calls, 0)
        self.assertNotIn("panel", gate)


if __name__ == "__main__":
    unittest.main()
