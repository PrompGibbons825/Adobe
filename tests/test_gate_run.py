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


CLAUDE_M, CURSOR_M, TIE_M = "claude-sonnet-5-5", "grok-4.7-high", "composer-2.5"


class FakeJudge:
    """Chooses the hidden "x" side with probability ``p_x`` on pairs (x = the candidate) and
    ``p_sent`` on sentinels (x = the good design); scores follow the choice. ``models`` overrides
    (p_x, p_sent) per model. Handles a batch prompt (every ``<id>_A.png`` in it) and answers one
    JSON object keyed by id. Choices are a hash of (seed, ask), so thread order does not matter."""

    def __init__(self, p_x: float = 1.0, p_sent: float = 1.0, seed: int = 0, models=None):
        self.p_x, self.p_sent, self.seed, self.models = p_x, p_sent, seed, models or {}
        self.calls, self.items, self.by_model = 0, 0, {}
        self.panel = None

    def bind(self, panel):
        self.panel = panel

    def __call__(self, prompt: str, model: str):
        self.calls += 1
        self.by_model[model] = self.by_model.get(model, 0) + 1
        ids = list(dict.fromkeys(re.findall(r"([0-9a-f]{8})_A\.png", prompt)))  # the reply example names one again
        self.items += len(ids)
        p_x, p_sent = self.models.get(model, (self.p_x, self.p_sent))
        out = {}
        for aid in ids:
            ask = self.panel.asks[aid]
            p = p_sent if ask["kind"] == "sentinel" else p_x
            x = ask["x_side"]
            y = "B" if x == "A" else "A"
            pick = x if PN._seeded(self.seed, aid).random() < p else y
            other = y if pick == x else x
            rep = {c: pick for c in PN.CRITERIA}
            rep.update(confidence=2, why="fake", scores={pick: {c: 8 for c in PN.CRITERIA},
                                                         other: {c: 5 for c in PN.CRITERIA}})
            out[aid] = rep
        return "Here you go:\n" + json.dumps(out), {"input": 9000, "cache_read": 1000, "output": 120}


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

    def seal(self, n_designs: int = 6, set_name: str = "novel", targets=((1080, 1080),), ref: bool = False) -> str:
        st = self.tmp / f"stage-{set_name}"
        (st / "items").mkdir(parents=True)
        (st / "assets").mkdir()
        k = 0
        for i in range(n_designs):
            for (w, h) in targets:
                t = {"task_id": f"i{k:03d}", "source": _design(i), "target_canvas": {"width": w, "height": h},
                     "meta": {"design": f"d{i}"}}
                if ref:
                    t["reference"] = baselines.nudge(baselines.fit_centre(t["source"], t["target_canvas"]))
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

    def test_thin_family_coverage_cannot_accept(self):
        t = _tiles([1.0] * 30)
        for i, x in enumerate(t):
            x["family"] = {"claude": {"overall": 1.0}} if i % 2 else {"claude": {"overall": 1.0}, "cursor": {"overall": 1.0}}
        ev = stats.evaluate(t, {"boot": 500}, 0.05, final=True)
        self.assertEqual(ev["decision"], "REPORT")
        self.assertAlmostEqual(ev["coverage"]["cursor"], 0.5)
        self.assertIn("cursor voted on 50%", "; ".join(ev["reasons"]))

    def test_family_harm_and_single_family(self):
        r = {"boot": 2000}
        t = _tiles([1.0] * 30)
        for x in t:
            x["family"] = {"claude": dict(x["score"]), "cursor": dict(x["score"], overall=0.0)}
        ev = stats.evaluate(t, r, 0.05, final=True)
        self.assertEqual(ev["harm"], ["family cursor"])
        self.assertEqual(ev["decision"], "REJECT")
        self.assertEqual(ev["agreement"], {"n": 30, "rate": 0.0})
        for x in t:
            x["family"]["cursor"] = dict(x["score"])
        ev = stats.evaluate(t, r, 0.05, final=True)
        self.assertEqual((ev["decision"], ev["family"]["cursor"]["point"]), ("ACCEPT", 1.0))
        one = stats.family_view(t, "claude")
        ev = stats.evaluate(one, r, 0.05, final=True)
        self.assertEqual(ev["decision"], "REPORT")
        self.assertIn("single-family", ev["reasons"][-1])
        self.assertEqual(stats.evaluate(one, dict(r, families_required=1), 0.05, final=True)["decision"], "ACCEPT")
        self.assertEqual(stats.sequential(one, [f"novel/d{i}" for i in range(30)], dict(r, looks=[30]))["decision"],
                         "REPORT")
        fl = stats.flip(t, {"novel/d0"})
        self.assertEqual(fl[0]["family"]["cursor"]["overall"], 0.0)

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

    def test_panel_schema(self):
        self.seal(2)
        for gid, panel, msg in (("p1", {"judges": 3}, "unknown panel keys"),
                                ("p2", {"protocol": "claude-only"}, "unknown panel protocol"),
                                ("p3", {"families": {"claude": {"cli": "claude", "model": "m"}}}, "exactly two"),
                                ("p4", {"families": {"a": {"cli": "claude", "model": "m"},
                                                     "b": {"cli": "agent", "model": "m"}}}, "different models"),
                                ("p5", {"tiebreak": {"cli": "agent", "model": "grok-4.7-high"}}, "third model"),
                                ("pa", {"tiebreak": {"cli": "agent", "model": "gpt-5.6-sol-high"}}, "included pool"),
                                ("pb", {"families": {"claude": {"cli": "claude", "model": "claude-sonnet-5-5"},
                                                     "cursor": {"cli": "agent", "model": "auto"}}}, "included pool"),
                                ("p6", {"batch": 20}, "batch must be"),
                                ("p8", {"sentinel_kinds": ["p_nope"]}, "known perturbations"),
                                ("p9", {"sentinel_kinds": []}, "must not be empty"),
                                ("p7", {"families": {"a": {"cli": "claude", "model": "m1"},
                                                     "b": {"cli": "claude", "model": "m2"}}}, "different CLIs")):
            with self.assertRaisesRegex(P.Refused, msg):
                P.check(self.prereg(_p(gid=gid, cand=gid[-1] * 40, panel=panel)))

    def test_make_writes_the_full_panel(self):
        with mock.patch.object(P, "resolve", side_effect=lambda ref, repo=P.REPO: {"b": "b" * 40, "c": "c" * 40}[ref]):
            m = P.make("mk", "b", "c", sets=["novel-v1"], panel={"batch": 10})
        self.assertEqual(m["panel"]["batch"], 10)
        self.assertEqual(m["panel"]["families"]["cursor"]["model"], "grok-4.7-high")
        self.assertEqual(m["panel"]["tiebreak"]["model"], "composer-2.5")
        self.assertEqual(m["panel"]["protocol"], "two-family")

    def test_null_and_builtin_keys(self):
        self.assertEqual(P.cand_key({"commit": "a", "transform": "nudge"}), "a+nudge")
        self.assertEqual(P.cand_key({"builtin": "fit_centre"}), "builtin:fit_centre")


# ---- panel -----------------------------------------------------------------------------------

class Panel(TempEnv):
    def test_parse_reply_and_batch(self):
        good = {c: "A" for c in PN.CRITERIA}
        self.assertEqual(PN.parse_reply("noise " + json.dumps(good) + " tail")["overall"], "A")
        self.assertIsNone(PN.parse_reply(json.dumps(dict(good, brand="C"))))
        self.assertIsNone(PN.parse_reply("I cannot see the images"))
        r = PN.parse_reply(json.dumps(dict(good, scores={"A": {c: 7 for c in PN.CRITERIA},
                                                         "B": {c: "6" for c in PN.CRITERIA}})))
        self.assertEqual(r["scores"]["B"]["overall"], 6.0)
        b = PN.parse_batch("ok:\n" + json.dumps({"aa": good, "bb": dict(good, overall="X")}) + "\n", ["aa", "bb", "cc"])
        self.assertEqual(b["aa"]["overall"], "A")
        self.assertIsNone(b["bb"])
        self.assertIsNone(b["cc"])
        lst = PN.parse_batch(json.dumps([dict(good, id="aa"), dict(good, id="bb", overall="B")]), ["aa", "bb"])
        self.assertEqual(lst["bb"]["overall"], "B")
        self.assertEqual(PN.parse_batch(json.dumps(good), ["solo"])["solo"]["overall"], "A")
        self.assertEqual(PN.parse_batch("", ["aa"]), {"aa": None})

    def test_prompt_is_small_and_blind(self):
        items = [{"id": f"{i:08x}", "ref": f"/j/{i:08x}_ref.png", "A": f"/j/{i:08x}_A.png",
                  "B": f"/j/{i:08x}_B.png"} for i in range(12)]
        pr = PN.prompt_for(items, "claude")
        self.assertLess(len(pr), 6000)
        self.assertIn("/j/0000000b_B.png", pr)
        for word in ("candidate", "base", "engine version", "sealed", "family", "gpt", "claude", "cursor"):
            self.assertNotIn(word, pr.lower())
        ag = PN.prompt_for(items[:2], "agent")
        self.assertNotIn("/j/", ag)  # the agent CLI runs from the asks folder and opens files by name
        self.assertIn(" A 00000001_A.png", ag)

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

    def test_judge_tiles_two_families(self):
        j = FakeJudge(p_x=1.0)
        pn = PN.Panel("g", self.tmp / "private", {"view": 192, "min_sentinels": 4}, judge=j)
        res = pn.judge_tiles(self._tiles(), 0)
        recs = PN.score_tiles(res["tiles"], pn.families)
        self.assertEqual(len(recs), 4)
        self.assertTrue(all(t["score"]["overall"] == 1.0 for t in recs))
        self.assertTrue(all(set(t["family"]) == {"claude", "cursor"} for t in recs))
        self.assertEqual(res["tiebreaks"], 0)
        # per family: 4 tile asks, 4 decisive sentinels (min_sentinels), 2 small-defect (S6)
        for f in ("claude", "cursor"):
            self.assertEqual(res["sentinels"][f]["n"], 4)
            self.assertEqual(res["sentinels"][f]["accuracy"], 1.0)
            self.assertEqual(res["small_sentinels"][f]["n"], 2)
        self.assertEqual(j.items, 2 * (4 + 4 + 2))
        # batched: 10 asks per family at batch 8 -> 2 calls per family
        self.assertEqual(j.by_model, {CLAUDE_M: 2, CURSOR_M: 2})
        self.assertEqual(pn.usage()["cursor_asks"], 2)
        self.assertAlmostEqual(recs[0]["delta"]["overall"], 3.0)
        # the judge folder holds only blind names
        names = [p.name for p in (self.tmp / "judges" / "g" / "asks").iterdir()]
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{8}_(ref|A|B)\.png", n) for n in names), names[:3])
        # per tile the two families see opposite orders, and which one gets A is random over tiles
        pairs = [a for a in pn.asks.values() if a["kind"] == "pair"]
        by_tile = {}
        for a in pairs:
            by_tile.setdefault(a["key"], {})[a["family"]] = a["x_side"]
        self.assertTrue(all(v["claude"] != v["cursor"] for v in by_tile.values()))
        self.assertEqual({v["claude"] for v in by_tile.values()}, {"A", "B"})
        # a resumed panel re-asks nothing
        j2 = FakeJudge()
        pn2 = PN.Panel("g", self.tmp / "private", {"view": 192, "min_sentinels": 4}, judge=j2)
        pn2.judge_tiles(self._tiles(), 0)
        self.assertEqual(j2.calls, 0)
        self.assertEqual(pn2.usage()["context_tokens_mean"], 10000)
        self.assertEqual(pn2.usage()["by_family"]["cursor"]["calls"], 2)

    def test_tiebreak_only_on_a_split_and_majority_of_three(self):
        j = FakeJudge(models={CLAUDE_M: (1.0, 1.0), CURSOR_M: (0.0, 1.0), TIE_M: (1.0, 1.0)})
        pn = PN.Panel("g2", self.tmp / "private2", {"view": 192}, judge=j)
        res = pn.judge_tiles(self._tiles(6), 0, share=0)
        self.assertEqual(res["tiebreaks"], 6)
        recs = PN.score_tiles(res["tiles"], pn.families)
        self.assertTrue(all(t["score"]["overall"] == 1.0 and t["tiebreak_used"] for t in recs))
        self.assertTrue(all(t["family"]["cursor"]["overall"] == 0.0 for t in recs))
        self.assertEqual(stats.family_agreement(recs)["rate"], 0.0)
        # agreeing families need no tie-break
        j = FakeJudge(p_x=0.0)
        pn = PN.Panel("g2b", self.tmp / "private2b", {"view": 192}, judge=j)
        res = pn.judge_tiles(self._tiles(3), 0, share=0)
        self.assertEqual((res["tiebreaks"], j.by_model.get(TIE_M)), (0, None))
        self.assertTrue(all(t["score"]["overall"] == 0.0 for t in PN.score_tiles(res["tiles"], pn.families)))

    def test_score_tile_rules(self):
        one = {c: 1.0 for c in PN.CRITERIA}
        zero = {c: 0.0 for c in PN.CRITERIA}
        rec = {"votes": {"claude": one, "cursor": dict(one, overall=0.0, brand=0.0), "tiebreak": dict(zero)},
               "deltas": {}}
        s = PN.score_tile(rec, ["claude", "cursor"])
        self.assertEqual((s["score"]["overall"], s["score"]["brand"], s["score"]["readable"]), (0.0, 0.0, 1.0))
        rec = {"votes": {"claude": one, "cursor": dict(one, brand=0.0)}, "deltas": {}}
        self.assertEqual(PN.score_tile(rec, ["claude", "cursor"])["score"]["brand"], 0.5)
        # a dropped family leaves the tile: the survivor's vote alone, the tie-break ignored
        rec = {"votes": {"claude": one, "cursor": zero, "tiebreak": zero}, "deltas": {}}
        s = PN.score_tile(rec, ["claude"])
        self.assertEqual((s["score"]["overall"], list(s["family"])), (1.0, ["claude"]))
        self.assertIsNone(PN.score_tile({"votes": {"cursor": zero}}, ["claude"]))

    def test_failed_replies_are_retried_then_dropped(self):
        calls = []

        def bad(prompt, model):
            calls.append(model)
            return "no json here", {}

        pn = PN.Panel("g3", self.tmp / "private3", {"view": 192, "retries": 1}, judge=bad)
        res = pn.judge_tiles(self._tiles(1), 0, share=0)
        self.assertEqual(res["tiles"], [])
        self.assertEqual(len(calls), 4)  # one call per family, retried once
        self.assertEqual(pn.usage()["failed"], 2)

    def test_partial_reply_retries_only_the_missing_items(self):
        fj = FakeJudge()
        seen = []

        def half(prompt, model):
            text, u = fj(prompt, model)
            j = json.loads(text[text.index("{"):])
            seen.append(len(j))
            keep = dict(list(j.items())[: max(1, len(j) // 2)])
            return json.dumps(keep), u

        half.bind = fj.bind
        pn = PN.Panel("g3b", self.tmp / "private3b", {"view": 192, "retries": 2}, judge=half)
        res = pn.judge_tiles(self._tiles(4), 0, share=0)
        self.assertEqual(len(res["tiles"]), 4)
        self.assertEqual(sorted(seen), [1, 1, 2, 2, 4, 4])

    def test_cursor_cap_pauses_and_resumes(self):
        j = FakeJudge()
        cfg = {"view": 192, "batch": 2, "agent_daily_cap": 1}
        pn = PN.Panel("g4", self.tmp / "private4", cfg, judge=j)
        with self.assertRaises(PN.CapReached) as cm:
            pn.judge_tiles(self._tiles(4), 0, share=0)
        self.assertGreater(cm.exception.reset_at, 0)
        self.assertEqual(j.by_model.get(CURSOR_M), 1)
        self.assertEqual(PN.agent_calls_today(), 1)
        n_lab = len(pn.labels)
        self.assertFalse(any(not lab["ok"] for lab in pn.labels.values()))  # nothing marked failed
        # next day (a fresh ledger): the same panel resumes and asks only what is missing
        PN.agent_ledger().unlink()
        pn2 = PN.Panel("g4", self.tmp / "private4", dict(cfg, agent_daily_cap=50), judge=j)
        res = pn2.judge_tiles(self._tiles(4), 0, share=0)
        self.assertEqual(len(PN.score_tiles(res["tiles"], pn2.families)), 4)
        self.assertEqual(len(pn2.labels), 8)
        self.assertGreater(len(pn2.labels), n_lab)

    def test_rate_limit_from_the_judge_pauses(self):
        fj = FakeJudge()

        def limited(prompt, model):
            if model == CURSOR_M:
                raise PN.CapReached("agent: usage limit")
            return fj(prompt, model)

        limited.bind = fj.bind
        pn = PN.Panel("g5", self.tmp / "private5", {"view": 192}, judge=limited)
        with self.assertRaises(PN.CapReached):
            pn.judge_tiles(self._tiles(2), 0, share=0)
        self.assertFalse(any(pn.asks[a]["family"] == "cursor" for a in pn.labels))

    def test_claude_limit_pauses_instead_of_dropping_to_one_family(self):
        cj = PN.ClaudeJudge(self.tmp, 10)
        hit = subprocess.CompletedProcess([], 1, stdout=json.dumps(
            {"type": "result", "is_error": True, "result": "Claude AI usage limit reached|1761350400"}), stderr="")
        with mock.patch.object(PN.subprocess, "run", return_value=hit):
            with self.assertRaises(PN.CapReached) as cm:
                cj("p", "claude-x")
        self.assertLess(cm.exception.reset_at, PN.time.time() + 2 * PN.CLAUDE_RETRY_SECS)
        hit2 = subprocess.CompletedProcess([], 1, stdout="", stderr="You've hit your limit - resets 3am")
        with mock.patch.object(PN.subprocess, "run", return_value=hit2):
            with self.assertRaises(PN.CapReached):
                cj("p", "claude-x")
        other = subprocess.CompletedProcess([], 1, stdout="", stderr="some crash")
        with mock.patch.object(PN.subprocess, "run", return_value=other):
            text, u = cj("p", "claude-x")
        self.assertEqual(u["exit"], 1)
        # in a panel: the claude limit pauses the run; no claude ask is marked failed
        fj = FakeJudge()

        def limited(prompt, model):
            if model == CLAUDE_M:
                raise PN.CapReached("claude: usage limit", PN.time.time() + 60)
            return fj(prompt, model)

        limited.bind = fj.bind
        pn = PN.Panel("g5c", self.tmp / "private5c", {"view": 192}, judge=limited)
        with self.assertRaises(PN.CapReached):
            pn.judge_tiles(self._tiles(2), 0, share=0)
        self.assertFalse(any(pn.asks[a]["family"] == "claude" for a in pn.labels))

    def test_fit_centre_sentinels_only_against_a_designer_resize(self):
        pn = PN.Panel("g6s", self.tmp / "private6s", {"view": 192, "sentinel_kinds": ["fit_centre"]}, judge=FakeJudge())
        tiles = self._tiles(2)
        self.assertEqual(pn.sentinels(tiles, 3, 0, pn.families), {"claude": [], "cursor": []})
        for t in tiles:
            t["reference"] = t["base"]
            t["source"] = dict(t["source"], canvas={"width": 400, "height": 900})
        got = pn.sentinels(tiles, 3, 0, pn.families)
        self.assertEqual([len(v) for v in got.values()], [3, 3])

    def test_sentinel_check_on_dev_data(self):
        from tools.gate import sentinel_check as SC

        pool = []
        for t in self._tiles(3):
            pool.append({"task_id": t["tile"], "source": t["source"], "target_canvas": t["base"]["canvas"],
                         "reference": t["base"]})
        with mock.patch.object(SC.devsets, "tasks", return_value=(pool, self.tmp / "assets")), \
                mock.patch.object(SC, "GATE_ROOT", self.tmp):
            r = SC.check("dev-sent", "devx", 6, {"view": 192}, 0.85, 4, judge=FakeJudge())
            self.assertTrue(r["pass"], r["families"])
            self.assertEqual(r["families"]["cursor"]["decisive"]["n"], 6)
            r = SC.check("dev-sent2", "devx", 6, {"view": 192}, 0.85, 4, judge=FakeJudge(p_sent=0.0))
            self.assertFalse(r["pass"])
        with mock.patch.object(SC.sealed, "sealed_versions", return_value=["novel-v1"]):
            with self.assertRaisesRegex(SystemExit, "dev data only"):
                SC.check("dev-sent3", "novel-v1", 6, {"view": 192}, 0.85, 4, judge=FakeJudge())

    def test_cursor_judge_spots_limits(self):
        cj = PN.CursorJudge(self.tmp, 10)
        done = subprocess.CompletedProcess([], 1, stdout="", stderr="Error: You've hit your usage limit")
        with mock.patch.object(PN.subprocess, "run", return_value=done):
            with self.assertRaises(PN.CapReached):
                cj("p", "m")
        ok = subprocess.CompletedProcess([], 0, stdout=json.dumps(
            {"type": "result", "is_error": False, "result": "{}", "duration_api_ms": 1200,
             "usage": {"inputTokens": 12000, "outputTokens": 40, "cacheReadTokens": 5, "cacheWriteTokens": 0}}),
            stderr="")
        with mock.patch.object(PN.subprocess, "run", return_value=ok) as m:
            text, u = cj("p", "gpt-x")
        self.assertEqual((text, u["input"], u["output"]), ("{}", 12000, 40))
        cmd = m.call_args[0][0]
        self.assertEqual(cmd[1:6], ["-p", "--trust", "--mode=ask", "--model", "gpt-x"])
        self.assertEqual(m.call_args[1]["cwd"], self.tmp)


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
        self.assertRegex(rep["line"], r"families claude 1\.00 cursor 1\.00 agree 1\.00 n=\d+")
        self.assertRegex(rep["line"], r"sentinels claude 1\.00 n=\d+ cursor 1\.00 n=\d+")
        self.assertRegex(rep["line"], r"cursor_asks [1-9]")
        self.assertEqual(gate["cursor_asks"], gate["panel"]["usage"]["cursor_asks"])
        self.assertEqual(gate["panel"]["active"], ["claude", "cursor"])
        self.assertEqual(gate["panel"]["looks"][0]["designs"], 3)
        sk = gate["panel"]["sentinel_kinds"]["sentinel"]
        self.assertEqual(sorted(sk), ["claude", "cursor"])
        self.assertTrue(all(m["correct"] == m["n"] > 0 for m in sk["claude"].values()))
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

    def test_sentinel_sets_feed_only_the_sentinel_pool(self):
        cv = self.seal(4, set_name="crello", targets=((1080, 1920),), ref=True)
        p = _p(sets=[self.version], proxy=False, sentinel_sets=[cv],
               rule={"boot": 500, "looks": [6], "min_win_designs": 3})
        rep, gate = self.go(p, FakeJudge(p_x=1.0))
        self.assertTrue(rep["verdict"].startswith("ACCEPT"), rep["line"])
        asks = PN.Panel("g1", self.tmp / "out" / "g1" / "private" / "panel", judge=FakeJudge()).asks.values()
        self.assertTrue(any(a["kind"] == "sentinel" for a in asks))
        self.assertNotIn(cv, gate["changed"]["by_set"])
        # a sentinel-only set is no run on that version: the same candidate may still be judged on it
        P.check(self.prereg(_p(gid="g1-crello", sets=[cv])))
        with self.assertRaisesRegex(P.Refused, "not sealed"):
            P.check(self.prereg(_p(gid="g1-bad", cand="9" * 40, sets=[self.version], sentinel_sets=["crello-v9"])))

    def test_bad_sentinels_invalidate(self):
        p = _p(sets=[self.version], proxy=False, rule={"boot": 500, "looks": [6]})
        rep, _ = self.go(p, FakeJudge(p_x=1.0, p_sent=0.0))
        self.assertTrue(rep["verdict"].startswith("INVALID"), rep["line"])

    def test_a_family_failing_sentinels_is_dropped_and_cannot_accept(self):
        p = _p(sets=[self.version], proxy=False, rule={"boot": 500, "looks": [6], "min_win_designs": 3})
        rep, gate = self.go(p, FakeJudge(p_x=1.0, models={CURSOR_M: (1.0, 0.0)}))
        self.assertTrue(rep["verdict"].startswith("REPORT"), rep["line"])
        self.assertIn("single-family", rep["verdict"])
        self.assertIn("family cursor dropped", rep["verdict"])
        self.assertIn("dropped cursor (sentinels 0.00", rep["line"])
        self.assertEqual(gate["panel"]["active"], ["claude"])
        self.assertEqual(list(gate["panel"]["final"]["family"]), ["claude"])

    def test_one_early_sentinel_miss_does_not_drop_a_family(self):
        rule = stats.rule_with_defaults({})
        sent = {"claude": {"n": 8, "correct": 7}, "cursor": {"n": 8, "correct": 8}}
        self.assertEqual(R.sentinel_drops(sent, ["claude", "cursor"], rule, 1, stop=False), {})
        # at the stop, or with enough sentinels, the bar applies to the cumulative count
        self.assertEqual(R.sentinel_drops(sent, ["claude", "cursor"], rule, 1, stop=True), {})  # 0.875
        sent["claude"] = {"n": 10, "correct": 8}
        self.assertEqual(list(R.sentinel_drops(sent, ["claude", "cursor"], rule, 1, stop=False)), [])
        self.assertEqual(list(R.sentinel_drops(sent, ["claude", "cursor"], rule, 1, stop=True)), ["claude"])
        sent["claude"] = {"n": 30, "correct": 24}
        self.assertIn("n=30", R.sentinel_drops(sent, ["claude", "cursor"], rule, 2, stop=False)["claude"])

    def test_single_family_can_still_reject(self):
        p = _p(sets=[self.version], proxy=False, rule={"boot": 500, "looks": [6]})
        rep, _ = self.go(p, FakeJudge(p_x=0.0, models={CLAUDE_M: (0.0, 0.0)}))
        self.assertTrue(rep["verdict"].startswith("REJECT"), rep["line"])
        self.assertIn("family claude dropped", rep["verdict"])

    def test_harm_within_one_family_blocks_a_pooled_win(self):
        """Claude likes the candidate, GPT does not, the tie-break sides with Claude: pooled 1.0,
        but significant harm within the cursor family is a REJECT."""
        p = _p(sets=[self.version], proxy=False, rule={"boot": 500, "looks": [6], "min_win_designs": 3})
        rep, gate = self.go(p, FakeJudge(models={CLAUDE_M: (1.0, 1.0), CURSOR_M: (0.0, 1.0), TIE_M: (1.0, 1.0)}))
        self.assertTrue(rep["verdict"].startswith("REJECT"), rep["line"])
        self.assertIn("family cursor", gate["panel"]["final"]["harm"])
        self.assertTrue(gate["panel"]["final"]["win_pass"])
        self.assertRegex(rep["line"], r"families claude 1\.00 cursor 0\.00 agree 0\.00")
        self.assertGreater(gate["panel"]["tiebreak_tiles"], 0)

    def test_cursor_cap_pauses_and_resume_finishes(self):
        p = _p(sets=[self.version], proxy=False, rule={"boot": 500, "looks": [6], "min_win_designs": 3},
               panel={"view": 192, "batch": 12, "agent_daily_cap": 1})
        path = self.prereg(p)
        out = self.tmp / "out"
        rep = R.run(path, judge=FakeJudge(), out_root=out, materializer=self.mat)
        self.assertTrue(rep["verdict"].startswith("PAUSED"), rep["line"])
        self.assertEqual(P.read_runs()[-1]["status"], "paused")
        self.assertTrue((self.tmp / "judges" / "g1" / "asks").exists())  # images kept for the resume
        PN.agent_ledger().unlink()  # the next UTC day
        rep = R.run(path, resume=True, judge=FakeJudge(), out_root=out, materializer=self.mat)
        self.assertTrue(rep["verdict"].startswith("ACCEPT"), rep["line"])
        self.assertEqual([r["status"] for r in P.read_runs()], ["started", "paused", "done"])

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
        self.assertEqual(sorted(gate["null"]["families"]), ["claude", "cursor"])
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
