"""Finale API: candidates, constraints, consistency, budget, determinism.

    .venv/bin/python -m unittest tests.test_finale_api -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.api import _same, _sig, candidates, resize_many  # noqa: E402
from engine.resize import resize  # noqa: E402
from engine.smart import textfit  # noqa: E402


def sample(level: int) -> dict:
    name = {1: "level1_warmup", 2: "level2_core", 3: "level3_advanced"}[level]
    return json.loads((ROOT / "input" / f"chameleon_sample_{name}.json").read_text())["source"]


def by_id(design: dict) -> dict:
    return {e["id"]: e for e in design["elements"]}


def dump(o) -> str:
    return json.dumps(o, sort_keys=True)


class Candidates(unittest.TestCase):
    def test_first_candidate_is_resize(self):
        for lvl, tgt in ((1, (1080, 1920)), (2, (300, 250)), (2, (728, 90)), (3, (728, 90))):
            src = sample(lvl)
            t = {"width": tgt[0], "height": tgt[1]}
            cs = candidates(src, t, k=3)
            self.assertEqual(dump(cs[0]["design"]), dump(resize(src, t)), (lvl, tgt))

    def test_distinct_complete_serialisable(self):
        src = sample(2)
        cs = candidates(src, {"width": 300, "height": 250}, k=5)
        self.assertGreaterEqual(len(cs), 2)
        ids = sorted(e["id"] for e in src["elements"])
        for i, c in enumerate(cs):
            self.assertEqual(sorted(by_id(c["design"])), ids)
            self.assertFalse(c["checks"]["copy_over_copy"])
            for j in range(i):
                self.assertFalse(_same(_sig(c["design"]), _sig(cs[j]["design"])))
            for key in ("archetype", "strategy", "score", "breakdown", "roles", "hidden_ids", "decisions",
                        "layout_decisions", "highlights"):
                self.assertIn(key, c)
        json.dumps(cs)
        self.assertEqual(cs[0]["roles"]["headline"]["role"], "headline")
        self.assertTrue(any(e["rule"] == "font" for e in cs[0]["decisions"]["headline"]))

    def test_dropped_items_have_reasons(self):
        c = candidates(sample(3), {"width": 728, "height": 90}, k=1)[0]
        self.assertTrue(c["dropped"])
        self.assertTrue(all(d.get("reason") for d in c["dropped"]))


class Constraints(unittest.TestCase):
    def test_pin_box(self):
        box = {"x": 60, "y": 60, "width": 300, "height": 120}
        c = candidates(sample(2), {"width": 1080, "height": 1920}, {"pinned": {"logo": box}}, k=1)[0]
        e = by_id(c["design"])["logo"]
        self.assertTrue(c["checks"]["pins_ok"])
        self.assertGreaterEqual(e["x"], 59)
        self.assertLessEqual(e["x"] + e["width"], 361)
        self.assertLessEqual(e["y"] + e["height"], 181)

    def test_pinned_text_size(self):
        pin = {"x": 80, "y": 1500, "width": 920, "height": 300, "font_size": 90}
        c = candidates(sample(2), {"width": 1080, "height": 1920}, {"pinned": {"headline": pin}}, k=1)[0]
        e = by_id(c["design"])["headline"]
        self.assertEqual(e["font_size"], 90)
        self.assertAlmostEqual(e["x"], 80, delta=1)
        self.assertFalse(c["checks"]["copy_over_copy"])

    def test_must_is_never_hidden(self):
        src = sample(3)
        t = {"width": 728, "height": 90}
        self.assertIn("subhead", candidates(src, t, k=1)[0]["hidden_ids"])
        c = candidates(src, t, {"priority": {"subhead": "must"}}, k=1)[0]
        self.assertNotIn("subhead", c["hidden_ids"])

    def test_min_font(self):
        c = candidates(sample(2), {"width": 300, "height": 250}, {"min_font_px": 14}, k=1)[0]
        self.assertGreaterEqual(c["checks"]["min_font_px"], 14)

    def test_safe_insets(self):
        c = candidates(sample(2), {"width": 1080, "height": 1920}, {"safe_insets": {"top": 250, "bottom": 340}}, k=1)[0]
        self.assertTrue(c["checks"]["safe_area_ok"], c["checks"])

    def test_allow_drop_removes(self):
        src = sample(3)
        c = candidates(src, {"width": 728, "height": 90}, {"allow_drop": True}, k=1)[0]
        self.assertTrue(c["hidden_ids"])
        self.assertFalse(set(c["hidden_ids"]) & set(by_id(c["design"])))

    def test_measured_text_is_scoped(self):
        src = sample(2)
        t = {"width": 1280, "height": 720}
        before = dump(resize(src, t))
        wide = candidates(src, t, {"measured_text": {"headline": {"font_size": 56, "width": 700}}}, k=1)[0]
        self.assertIsNone(textfit._MEASURER)
        self.assertEqual(dump(resize(src, t)), before)
        h0, h1 = by_id(json.loads(before))["headline"], by_id(wide["design"])["headline"]
        self.assertNotEqual((h0["width"], h0["font_size"]), (h1["width"], h1["font_size"]))

    def test_resize_with_constraints(self):
        out = resize(sample(1), {"width": 1080, "height": 1080}, {"min_font_px": 30})
        self.assertTrue(all(e["font_size"] >= 30 for e in out["elements"] if e["type"] == "text"))


class Batch(unittest.TestCase):
    def test_resize_many(self):
        targets = {"ig": {"width": 1080, "height": 1080}, "yt": {"width": 1280, "height": 720},
                   "fb": {"width": 1200, "height": 628}, "mr": {"width": 300, "height": 250}}
        r = resize_many(sample(2), targets, k=4)
        self.assertEqual(sorted(r["results"]), sorted(targets))
        ind, joint = r["consistency"]["independent"], r["consistency"]["joint"]
        self.assertGreaterEqual(joint["similarity"], ind["similarity"] - 1e-9)
        json.dumps(r)

    def test_budget(self):
        cs = candidates(sample(3), {"width": 1280, "height": 720}, k=2, budget={"work": 1})
        self.assertTrue(cs[0]["budget"]["hit"])
        self.assertEqual(len(cs[0]["design"]["elements"]), len(sample(3)["elements"]))


class Determinism(unittest.TestCase):
    def test_across_hash_seeds(self):
        code = ("import json,sys; sys.path.insert(0, '.'); from engine.api import candidates; "
                "src=json.load(open('input/chameleon_sample_level2_core.json'))['source']; "
                "print(json.dumps([c['design'] for c in candidates(src, {'width':300,'height':250}, k=4)], sort_keys=True))")
        outs = set()
        for seed in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            outs.add(subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                                    text=True, check=True).stdout)
        self.assertEqual(len(outs), 1)


if __name__ == "__main__":
    unittest.main()
