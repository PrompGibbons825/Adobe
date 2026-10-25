"""Tests for the judge tooling (tools/judge): terms, fit, perturb, views.

    .venv/bin/python -m unittest tests.test_judge
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.judge import fit as F
from tools.judge import perturb as P
from tools.judge import terms as T
from tools.judge import views as V

REPO = Path(__file__).resolve().parents[1]
L1 = REPO / "input" / "chameleon_sample_level1_warmup.json"


def _task():
    return json.loads(L1.read_text())


def _el(design, i):
    return next(e for e in design["elements"] if e["id"] == i)


class Terms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = _task()["source"]
        cls.rep_src = T.measure(cls.src)

    def test_term_list_and_weights(self):
        self.assertEqual(len(T.TERMS), 26)
        self.assertEqual(len(set(T.TERMS)), 26)
        self.assertEqual(set(T.HAND_WEIGHTS), set(T.TERMS))
        w = T.calibrated_weights()
        self.assertEqual(set(w), set(T.TERMS))
        self.assertTrue(all(v >= 0 for v in w.values()))
        self.assertGreater(sum(w.values()), 0)

    def test_no_engine_import(self):
        code = ("import sys; import tools.judge.terms, tools.judge.fit, tools.judge.views, tools.judge.perturb; "
                "print(sorted(m for m in sys.modules if m == 'engine' or m.startswith('engine.')))")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "[]")

    def test_identity_scores_zero(self):
        t = T.terms(self.rep_src, self.rep_src)
        self.assertEqual(set(t), set(T.TERMS))
        # abs_* are absolute, not source-relative; align_lost keeps the branch's asymmetric
        # tolerances (source 0.004 L, output 0.002 L), so a near-aligned pair can leak a little.
        rel = {k: v for k, v in t.items() if not k.startswith("abs_") and k != "align_lost"}
        self.assertTrue(all(v == 0 for v in rel.values()), {k: v for k, v in rel.items() if v})
        self.assertLess(t["align_lost"], 0.05)

    def test_scores_an_engine_output(self):
        from engine.resize import resize

        task = _task()
        out = resize(task["source"], task["target_canvas"])
        res = T.score(task["source"], out)
        self.assertEqual(set(res["terms"]), set(T.TERMS))
        self.assertTrue(all(np.isfinite(v) and v >= 0 for v in res["terms"].values()))
        self.assertTrue(np.isfinite(res["penalty"]) and res["penalty"] >= 0)

    def test_dropped_copy_is_penalised(self):
        d = json.loads(json.dumps(self.src))
        _el(d, "cta")["content"] = ""  # no ink: the copy is gone
        t = T.terms(self.rep_src, T.measure(d))
        self.assertGreater(t["dropped_text"], 0)

    def test_cut_headline_is_penalised(self):
        d = json.loads(json.dumps(self.src))
        h = _el(d, "headline")
        h["x"] = d["canvas"]["width"] - 0.5 * h["width"]
        t = T.terms(self.rep_src, T.measure(d))
        self.assertGreater(t["cut_text"], 0.05)
        self.assertGreater(T.penalty(t, T.calibrated_weights()), 0)

    def test_inside_handles_rotation(self):
        e = {"x": 10, "y": 10, "width": 80, "height": 10, "rotation": 0}
        self.assertTrue(T._inside(e, 100, 100))
        self.assertFalse(T._inside(dict(e, rotation=90), 100, 30))


class BleedInTerms(unittest.TestCase):
    """loop/STEERING.md S4: allowed bleed is no cut in the proxy; forbidden bleed still is."""

    def test_allowed_bleed_is_not_cut_art(self):
        from tests.finicky import bleed_probes

        ps = {p.name: p for p in bleed_probes()}
        for name, allowed in (("image_panel_against_an_edge", True), ("photo_cover_crop_framed", True),
                              ("panel_mostly_off", False)):
            with self.subTest(probe=name):
                p = ps[name]
                t = T.terms(T.measure(p.source), T.measure(p.output))
                if allowed:
                    self.assertEqual(t["cut_art"], 0.0)
                else:
                    self.assertGreater(t["cut_art"], 0.1)


class Fit(unittest.TestCase):
    def _synthetic(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        w_true = np.zeros(len(T.TERMS))
        w_true[:3] = [3.0, 1.5, 0.0]
        rows = []
        for i in range(n):
            x = np.zeros(len(T.TERMS))
            x[:3] = rng.normal(size=3)
            y = float(rng.random() < F.sigmoid(x @ w_true))
            rows.append({"x": x, "y": y, "n": 1, "source": f"s{i % 20}", "conf": True, "votes": [int(y)],
                         "strong": True, "case": f"c{i}", "kind": "eng", "a": "a", "b": "b"})
        return rows, w_true

    def test_bradley_terry_recovers_weights(self):
        rows, w_true = self._synthetic()
        X = np.stack([r["x"] for r in rows])
        y = np.array([r["y"] for r in rows])
        w = F.fit(X, y, np.ones(len(rows)), lam=1e-4)
        self.assertTrue((w >= 0).all())
        self.assertGreater(w[0], w[1])
        self.assertGreater(w[1], w[2])
        self.assertLess(abs(w[0] - 3.0), 1.0)

    def test_prior_pulls_towards_scaled_prior(self):
        rows, _ = self._synthetic(n=60)
        X = np.stack([r["x"] for r in rows])
        y = np.array([r["y"] for r in rows])
        prior = np.ones(len(T.TERMS))
        w = F.fit(X, y, np.ones(len(rows)), lam=100.0, prior=prior)
        self.assertGreater(w[10], 0)  # an unused term keeps its prior share under heavy shrinkage

    def test_cv_holds_out_sources(self):
        rows, _ = self._synthetic(n=100)
        g = F.groups_of(rows, k=5, seed=0)
        self.assertEqual(sorted(set(g.values())), [0, 1, 2, 3, 4])
        pr = F.cv_predict(rows, 1e-3, seeds=(0,))
        self.assertEqual(pr.shape, (100,))
        self.assertTrue(((pr > 0) & (pr < 1)).all())

    def test_metrics(self):
        self.assertAlmostEqual(F.tau_b([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(F.tau_b([1, 2, 3], [3, 2, 1]), -1.0)
        rows = [{"y": 1.0, "conf": True, "strong": True, "votes": [1, 1]},
                {"y": 0.0, "conf": True, "strong": False, "votes": [0, 0]}]
        ag = F.agreement(rows, np.array([1.0, 0.0]))
        self.assertEqual(ag["conf"], 0.75)  # one hit, one tie counted half
        self.assertEqual(ag["n_conf"], 2)

    def test_committed_corpus(self):
        rows = F.dataset("overall")
        self.assertGreaterEqual(len(rows), 150)
        self.assertTrue(all(r["x"].shape == (26,) for r in rows))
        self.assertLessEqual(len(F.dataset("overall", tol=0.05)), len(rows))
        self.assertGreaterEqual(len(F.dataset("overall", tol=99)), len(rows))
        w = json.loads(T.WEIGHTS_FILE.read_text())
        self.assertEqual(w["pairs"], len(rows))
        self.assertGreater(w["cv_agreement_confident"], 0.6)


class Perturb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from engine.resize import resize

        task = _task()
        cls.out = resize(task["source"], task["target_canvas"])

    def test_every_kind_keeps_ids_and_input(self):
        before = json.dumps(self.out, sort_keys=True)
        ids = sorted(e["id"] for e in self.out["elements"])
        got = P.all_perturbations(self.out)
        self.assertIn("p_crop", got)
        for kind, d in got.items():
            self.assertEqual(sorted(e["id"] for e in d["elements"]), ids, kind)
            self.assertNotEqual(json.dumps(d, sort_keys=True), before, kind)
        self.assertEqual(json.dumps(self.out, sort_keys=True), before)

    def test_known_effects(self):
        from engine.smart.textfit import drawn_box

        W = self.out["canvas"]["width"]
        crop = P.perturb("p_crop", self.out, random.Random(0))
        head = max((e for e in crop["elements"] if e["type"] == "text"), key=lambda e: e["font_size"])
        self.assertGreater(drawn_box(head).r, W)
        mis = P.perturb("p_mis20", self.out, random.Random(0))
        moved = [(a["x"] - b["x"]) for a, b in zip(mis["elements"], self.out["elements"]) if a["x"] != b["x"]]
        self.assertEqual(len(moved), 1)
        self.assertAlmostEqual(abs(moved[0]), 20.0)
        self.assertTrue(set(P.SENTINELS) <= set(P.PERTURBATIONS))
        self.assertTrue(set(P.SMALL_DEFECTS) <= set(P.PERTURBATIONS))
        clip = P.perturb("p_clip_letter", self.out, random.Random(0))
        head = max((e for e in clip["elements"] if e["type"] == "text"), key=lambda e: e["font_size"])
        self.assertAlmostEqual(drawn_box(head).r, W + 0.5 * head["font_size"], delta=1.0)
        with self.assertRaises(ValueError):
            P.perturb("p_nope", self.out)

    def test_perturbation_scores_worse(self):
        task = _task()
        rs = T.measure(task["source"])
        w = T.calibrated_weights()
        good = T.penalty(T.terms(rs, T.measure(self.out)), w)
        bad = T.penalty(T.terms(rs, T.measure(P.perturb("p_crop", self.out))), w)
        self.assertGreater(bad, good)


class Views(unittest.TestCase):
    def test_downsample(self):
        im = Image.new("RGB", (2000, 1000), "white")
        self.assertEqual(V.downsample(im, 512).size, (512, 256))
        self.assertEqual(V.downsample(Image.new("RGB", (300, 250)), 512).size, (300, 250))

    def test_views_and_blind_ask(self):
        task = _task()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            v = V.views(task["source"], "L1#src", d / "views")
            self.assertEqual(sorted(v), [512, 1024])
            for size, p in v.items():
                with Image.open(p) as im:
                    self.assertEqual(max(im.size), size)
            self.assertEqual(sorted(p.name for p in (d / "views").iterdir()), ["L1#src@1024.png", "L1#src@512.png"])
            rng = random.Random(1)
            taken = set()
            asks = [V.blind_ask(v[512], v[1024], v[512], d / "asks", rng, taken) for _ in range(3)]
            self.assertEqual(len({a["id"] for a in asks}), 3)
            names = sorted(p.name for p in (d / "asks").iterdir())
            self.assertEqual(len(names), 9)
            for n in names:
                stem, tag = n[:-4].split("_")
                self.assertRegex(stem, r"^[0-9a-f]{8}$")
                self.assertIn(tag, ("ref", "A", "B"))
                self.assertNotIn("L1", n)


if __name__ == "__main__":
    unittest.main()
