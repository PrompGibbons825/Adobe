"""Engine tests (stdlib unittest, no extra dependencies).

    .venv/bin/python -m unittest discover -s tests -t .          # everything (~4 min)
    .venv/bin/python -m unittest tests.test_engine.TextParity     # one group

The finicky cases (tests/finicky.py) run on each of their targets through
tests/harness.py; `python -m tests.run` prints the same checks as one report.
"""

from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

from engine.smart.geom import el_box
from engine.smart.items import Ctx
from engine.smart.paths import find_asset
from engine.smart.textfit import drawn_box
from tests.finicky import bleed_probes, cases, malformed, relations
from tests.harness import evaluate, probe_errors, relation_errors, run
from tools.rules import hard_rule_errors, promise_errors, render_image

REPO = Path(__file__).resolve().parents[1]
SAMPLES = sorted((REPO / "input").glob("*.json")) + sorted((REPO / "tools" / "cases").glob("*.json"))


def _sources():
    return [(p.stem, json.loads(p.read_text())["source"]) for p in SAMPLES]


def _in_kit(asset_id: str) -> bool:
    return find_asset(asset_id) is not None


class FinickyCases(unittest.TestCase):
    def test_every_case_on_every_target(self):
        for case in cases():
            for target in case.targets:
                with self.subTest(case=case.name, target=target):
                    _, errs = evaluate(case, target)
                    if target in case.xfail:  # a known failure of the engine, see Case.xfail
                        self.assertTrue(errs, f"{case.name}@{target} passes now: remove it from xfail")
                    else:
                        self.assertEqual(errs, [], case.probe)


class Relations(unittest.TestCase):
    """Scale invariance and continuity: two runs that must agree."""

    def test_relations(self):
        for r in relations():
            with self.subTest(relation=r.name):
                errs = relation_errors(r)
                if r.xfail:
                    self.assertTrue(errs, f"{r.name} passes now: remove it from the xfail sets")
                else:
                    self.assertEqual(errs, [], r.probe)


class BleedPolicy(unittest.TestCase):
    """loop/STEERING.md S4: allowed bleed passes the hard rules and is reported; forbidden fails."""

    def test_probes(self):
        for p in bleed_probes():
            with self.subTest(probe=p.name):
                self.assertEqual(probe_errors(p), [])


class MalformedInput(unittest.TestCase):
    def test_hard_rules_hold_and_output_renders(self):
        for name, src, tgt in malformed():
            with self.subTest(variant=name):
                out, _ = run(src, tgt)
                self.assertEqual(hard_rule_errors(src, out, tgt), [])
                render_image(out)  # the reference renderer must not raise on anything we emit


    def test_hostile_documents_do_not_take_the_run_down(self):
        from tests.finicky import hostile

        tgt = {"width": 300, "height": 250}
        for name, src in hostile():
            with self.subTest(document=name):
                out, _ = run(src, tgt)
                self.assertEqual(out["canvas"], tgt)
                render_image(out)


class SameShape(unittest.TestCase):
    """When the shape does not change, the designer's own layout is the answer."""

    def _check_scaled(self, name, src, k, tol):
        W, H = src["canvas"]["width"], src["canvas"]["height"]
        tgt = {"width": round(W * k), "height": round(H * k)}
        floor = Ctx(W=tgt["width"], H=tgt["height"], W0=W, H0=H).judge_floor
        small = [e["id"] for e in src["elements"] if e["type"] == "text" and e["font_size"] * k < floor]
        if small:
            self.skipTest(f"{small} must be set larger than designed, so the design may reflow")
        out, fell_back = run(src, tgt)
        self.assertFalse(fell_back)
        so = {e["id"]: e for e in src["elements"]}
        side = max(tgt["width"], tgt["height"])
        for e in out["elements"]:
            s = so[e["id"]]
            if float(e.get("opacity", 1) or 1) <= 0 and s.get("asset_id") and not _in_kit(s["asset_id"]):
                continue  # a missing asset may be dropped
            if e["type"] == "text":  # text boxes are emitted tight: compare where the glyphs land
                a, b = drawn_box(e), drawn_box(s)
                got, want = (a.x, a.y, a.w, a.h), (b.x * k, b.y * k, b.w * k, b.h * k)
            else:
                b = el_box(s)
                if b.x < 0 or b.y < 0 or b.r > W or b.b > H:
                    continue  # bleed is pulled onto the canvas
                got, want = (e["x"], e["y"], e["width"], e["height"]), (b.x * k, b.y * k, b.w * k, b.h * k)
            for key, g, w in zip(("x", "y", "w", "h"), got, want):
                self.assertLessEqual(abs(g - w), tol * side + 1.0, f"{name}: {e['id']}.{key} {g:.1f} vs {w:.1f}")

    def test_identity(self):
        for name, src in _sources():
            with self.subTest(design=name):
                self._check_scaled(name, src, 1.0, 0.02)

    def test_uniform_scale(self):
        for name, src in _sources():
            for k in (0.5, 2.0):
                with self.subTest(design=name, k=k):
                    self._check_scaled(name, src, k, 0.04)


class TextParity(unittest.TestCase):
    """Text is measured exactly the way the reference renderer wraps it."""

    def test_fast_wrap_equals_renderer_wrap(self):
        from engine.smart import textfit
        from renderer.converter.wrap import char_widths, wrap_text

        rng = random.Random(7)
        words = ("golden hour light sparkling botanical Sonnenuntergangsgetränke crème brûlée — €4,99 "
                 "solstice.com/terms ☀ a of the and LIMITED EDITION 2026").split()
        book = textfit.book()
        fonts = [("Solstice Serif", "bold", "bold"), ("Solstice Sans", "regular", "regular"),
                 ("Solstice Sans", "bold", "bold"), ("Solstice Sans", "regular", "italic")]
        for i in range(300):
            content = " ".join(rng.choice(words) for _ in range(rng.randint(1, 18)))
            if rng.random() < 0.15:
                content = content.replace(" ", "\n", 1)
            family, weight, style = rng.choice(fonts)
            size = rng.randint(8, 120)
            width = rng.uniform(20, 1200)
            st = textfit.TextStyle(family, weight, style, 1.2, 0.0)
            ours = textfit.measure(content, st, size, width)
            ref = wrap_text(content, char_widths(book.manager, family, weight, style, float(size), content), width)
            with self.subTest(i=i, content=content, size=size, width=round(width)):
                self.assertEqual(ours.lines, ref.split("\n"))


    def test_drawn_width_matches_the_render(self):
        """Measured ink width (letter spacing, bold and italic runs included)
        is what the renderer paints."""
        import numpy as np

        from engine.smart.textfit import TextStyle, measure

        base = {"id": "t", "type": "text", "x": 20, "y": 10, "width": 1500, "height": 120, "z_index": 1,
                "content": "Made for the golden hour", "font_family": "Solstice Serif", "font_size": 64,
                "color": "rgba(0,0,0,1)", "text_align": "left"}
        variants = [{}, {"letter_spacing": 6}, {"style_runs": [{"start": 13, "end": 23, "font_weight": "bold"}]},
                    {"style_runs": [{"start": 0, "end": 3, "font_style": "italic"}], "font_family": "Solstice Sans"}]
        for v in variants:
            el = dict(base, **v)
            with self.subTest(variant=v):
                im = np.asarray(render_image({"canvas": {"width": 1600, "height": 140}, "elements": [el]}).convert("L"))
                cols = np.where((im < 128).any(axis=0))[0]
                drawn = cols.max() - cols.min() + 1
                measured = measure(el["content"], TextStyle.of(el), el["font_size"], el["width"], exact=True).ink_width
                self.assertLess(abs(drawn - measured), 0.04 * el["font_size"] + 2)  # side bearings


class Determinism(unittest.TestCase):
    def test_same_input_same_output(self):
        for name, src in _sources()[:4]:
            for tgt in ({"width": 1600, "height": 400}, {"width": 1080, "height": 1920}):
                with self.subTest(design=name, target=tgt):
                    a, _ = run(src, tgt)
                    b, _ = run(src, tgt)
                    self.assertEqual(a, b)


class Samples(unittest.TestCase):
    """The official samples and the extra kit cases on their own targets."""

    def test_samples(self):
        for p in SAMPLES:
            task = json.loads(p.read_text())
            src, tgt = task["source"], task["target_canvas"]
            with self.subTest(design=p.stem):
                out, fell_back = run(src, tgt)
                self.assertFalse(fell_back)
                self.assertEqual(hard_rule_errors(src, out, tgt) + promise_errors(src, out, tgt), [])


# (sample, format) whose content does not fit even at the explicit minimum type size, measured
# 2026-10-25 (sprint A): only there may the last-resort repair still leave an element out. A pair
# that starts keeping everything should be struck from here; a pair not listed fails the test.
UNFIT = {
    ("chameleon_sample_level2_core", "mobile-banner"), ("chameleon_sample_level3_advanced", "mobile-banner"),
    ("band_panel", "mobile-banner"), ("sticker_divider", "mobile-banner"), ("story_hero_top", "mobile-banner"),
    ("two_products", "mobile-banner"),
}  # all 320x50: logo, product, headline, subhead and button do not share 50 px even at the minimum


class NothingHidden(unittest.TestCase):
    """S1 (loop/STEERING.md): no sample or kit case hides an element at any catalogue format
    (tools/rules.hidden_errors: opacity 0, 1 px type or a speck; a missing asset is exempt),
    except the few pairs in UNFIT, where the content cannot fit at all."""

    def test_every_sample_on_every_format(self):
        from engine.resize import resize
        from tools.rules import hidden_errors
        from tools.sheet import CATALOGUE

        bad, fixed = [], set(UNFIT)
        for name, src in _sources():
            for fmt, (w, h) in CATALOGUE.items():
                tgt = {"width": w, "height": h}
                errs = hidden_errors(src, resize(json.loads(json.dumps(src)), tgt))
                if errs:
                    fixed.discard((name, fmt))
                if (name, fmt) not in UNFIT:
                    bad += [f"{name}@{fmt}: {e}" for e in errs]
        self.assertEqual(bad, [])
        if fixed:
            print(f"\nNothingHidden: now keeps everything, strike from UNFIT: {sorted(fixed)}")


class WorkBudget(unittest.TestCase):
    """The resize's work budget (engine/smart/layout.Budget, r10): the economy layout and the
    out-of-budget cut-off still give valid output, and a deep design stays bounded."""

    TARGETS = ({"width": 1640, "height": 924}, {"width": 1080, "height": 1350}, {"width": 728, "height": 90})

    def _with(self, **attrs):
        from engine.smart.items import Budget

        old = {k: getattr(Budget, k) for k in attrs}
        for k, v in attrs.items():
            setattr(Budget, k, v)
        self.addCleanup(lambda: [setattr(Budget, k, v) for k, v in old.items()])

    def _valid(self, src, tgt):
        out, fell_back = run(src, tgt)
        self.assertEqual(hard_rule_errors(src, out, tgt), [])
        self.assertEqual(sorted(e["id"] for e in out["elements"]), sorted(e["id"] for e in src["elements"]))
        return out, fell_back

    def test_economy_from_the_start(self):
        self._with(ECONOMY=0)
        for name, src in _sources()[:5]:
            for tgt in self.TARGETS:
                with self.subTest(design=name, target=tgt):
                    _, fell_back = self._valid(src, tgt)
                    self.assertFalse(fell_back)

    def test_budget_spent_early(self):
        self._with(ECONOMY=200, LIMIT=2_000)
        for name, src in _sources()[:5]:
            for tgt in self.TARGETS:
                with self.subTest(design=name, target=tgt):
                    self._valid(src, tgt)

    def test_deep_design_is_bounded(self):
        """Judged by work, not the clock, so a loaded machine does not fail it: the deep design
        must switch to economy and finish well inside the hard limit, with a full layout (no
        fit-and-centre fallback, no clock cut-off)."""
        from engine.smart.items import BUDGET, Budget
        from tests.finicky import deep_panel

        self._with(SECONDS=1e9)  # the wall-clock guard off: only the deterministic budget counts
        src = {"canvas": {"width": 1080, "height": 1350}, "elements": deep_panel()}
        _, fell_back = self._valid(src, {"width": 1640, "height": 924})
        self.assertFalse(fell_back)
        self.assertTrue(BUDGET.economy)
        self.assertLess(BUDGET.work, Budget.LIMIT)


class GeneratedDesigns(unittest.TestCase):
    """Designs the engine was never looked at on (tools.designs, a seed the audit
    does not use): every rule holds and the core message survives everywhere."""

    FORMATS = ("ig-portrait", "story", "fb-link", "leaderboard", "med-rect", "skyscraper")

    def test_rules_and_core_message(self):
        from tools.designs import designs
        from tools.sheet import CATALOGUE

        for t in designs(12, seed=7):
            src = t["source"]
            for fmt in self.FORMATS:
                w, h = CATALOGUE[fmt]
                tgt = {"width": w, "height": h}
                with self.subTest(design=t["task_id"], notes=t["notes"], target=fmt):
                    out, fell_back = run(src, tgt)
                    self.assertFalse(fell_back)
                    self.assertEqual(hard_rule_errors(src, out, tgt) + promise_errors(src, out, tgt), [])
                    shown = {e["id"] for e in out["elements"] if float(e.get("opacity", 1)) > 0}
                    core = {"headline", "cta_label"} & {e["id"] for e in src["elements"]}
                    self.assertEqual(core - shown, set())


if __name__ == "__main__":
    unittest.main()
