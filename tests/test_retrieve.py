"""B1: designer-resize retrieval, the learned score term and their switches (CHAMELEON_OFF)."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from engine.resize import resize
from engine.smart import candidates as C
from engine.smart import ranker
from engine.smart.analyze import analyze
from engine.smart.items import Ctx
from tools.rules import hard_rule_errors

REPO = Path(__file__).resolve().parents[1]
SRC = json.loads((REPO / "input" / "chameleon_sample_level2_core.json").read_text())["source"]
TARGETS = [(1080, 1080), (1080, 1920), (1280, 720)]


def _body(d):
    return [it for it in d.items if not it.corner and it.role not in ("missing", "void")]


class Retrieval(unittest.TestCase):
    def test_index_is_v2_and_loads(self):
        fams = C._layout_index()
        self.assertGreater(len(fams), 100)
        for f in fams[:50]:
            n = {len(m["s"]) for m in f["m"]}
            self.assertEqual(len(n), 1, "members of a family must have aligned slots")

    def test_plans_are_valid(self):
        d = analyze(SRC)
        for W, H in TARGETS:
            ctx = Ctx(W=float(W), H=float(H), W0=d.W, H0=d.H)
            found = C._retrieve(d, ctx, _body(d))
            self.assertLessEqual(len(found), C.RETRIEVE_K)
            for k in range(len(found)):
                p = C.plan_retrieve(d, ctx, [], name=f"retrieve-{k}", rank=k)
                if p is None:
                    continue
                for r, x, y in p.placed:
                    self.assertGreaterEqual(x, -0.5)
                    self.assertGreaterEqual(y, -0.5)
                    self.assertLessEqual(x + r.w, W + 0.5)
                    self.assertLessEqual(y + r.h, H + 0.5)

    def test_exclude_hides_a_family(self):
        d = analyze(SRC)
        ctx = Ctx(W=1080.0, H=1080.0, W0=d.W, H0=d.H)
        found = C._retrieve(d, ctx, _body(d))
        if not found:
            self.skipTest("no family near this design")
        C.EXCLUDE.add(found[0][1])
        try:
            self.assertNotIn(found[0][1], [f[1] for f in C._retrieve(d, ctx, _body(d))])
        finally:
            C.EXCLUDE.clear()


class Switches(unittest.TestCase):
    def setUp(self):
        os.environ["CHAMELEON_ON"] = "search,retrieve"

    def tearDown(self):
        os.environ.pop("CHAMELEON_OFF", None)
        os.environ.pop("CHAMELEON_ON", None)

    def test_off_removes_variants(self):
        d = analyze(SRC)
        ctx = Ctx(W=1080.0, H=1920.0, W0=d.W, H0=d.H)
        names = [v[0] for v in C.extra_variants(d, ctx, True)]
        self.assertTrue(any(n.startswith("retrieve") for n in names))
        os.environ["CHAMELEON_OFF"] = "search,retrieve"
        self.assertEqual([v[0] for v in C.extra_variants(d, ctx, True)], [])
        os.environ["CHAMELEON_OFF"] = "ranker"
        self.assertFalse(ranker.enabled())

    def test_each_switch_still_valid(self):
        for off in ("", "search", "retrieve", "ranker", "search,retrieve,ranker"):
            os.environ["CHAMELEON_OFF"] = off
            out = resize(SRC, {"width": 1080, "height": 1920})
            self.assertEqual(hard_rule_errors(SRC, out, {"width": 1080, "height": 1920}), [], off)


class Defaults(unittest.TestCase):
    def test_search_and_retrieve_default_off(self):
        os.environ.pop("CHAMELEON_ON", None)
        os.environ.pop("CHAMELEON_OFF", None)
        d = analyze(SRC)
        ctx = Ctx(W=1080.0, H=1920.0, W0=d.W, H0=d.H)
        self.assertEqual([v[0] for v in C.extra_variants(d, ctx, True)], [])


class LearnedTerm(unittest.TestCase):
    def test_bounded(self):
        hand = [50.0, 49.0, 20.0]
        out = ranker.blended(hand, [0.0, 1.0, 2.0], 2.0, 20.0)
        self.assertTrue(all(h - 20.0 - 1e-9 <= o <= h + 1e-9 for h, o in zip(hand, out)))
        self.assertGreater(out[1], out[0])  # a near-tie the designers prefer flips
        self.assertLess(out[2], out[0])  # a clear critic loss never wins


if __name__ == "__main__":
    unittest.main()
