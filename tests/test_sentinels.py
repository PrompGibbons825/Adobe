"""Tests for the gate's sentinel screening (tools/gate/sentinels.py, Panel.sentinels)."""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.gate import sentinels as SN
from tools.judge import perturb as P


def _text(i, x, y, w, size, content="Golden hour sale", rot=0):
    e = {"id": i, "type": "text", "x": x, "y": y, "width": w, "height": size * 1.4, "z_index": 2,
         "content": content, "font_family": "Solstice Sans", "font_size": size, "color": "rgba(20,20,20,1)"}
    if rot:
        e["rotation"] = rot
    return e


def _design(w=600, h=400, texts=None, image=None):
    els = [{"id": "bg", "type": "shape", "x": 0, "y": 0, "width": w, "height": h, "z_index": 0,
            "shape_type": "rectangle", "fill_color": "rgba(240,230,210,1)"}]
    if image:
        els.append(dict({"id": "im", "type": "image", "z_index": 1, "src": "x.png"}, **image))
    els += texts if texts is not None else [_text("head", 40, 40, 400, 48, "Big headline"),
                                            _text("sub", 40, 200, 400, 18, "a smaller subhead line")]
    return {"canvas": {"width": w, "height": h}, "elements": els}


class Headline(unittest.TestCase):
    def test_largest_font_wins_and_near_ties_go_to_the_larger_ink(self):
        d = _design()
        self.assertEqual(P.headline(d)["id"], "head")
        d = _design(texts=[_text("tag", 40, 40, 400, 50, "Hi"), _text("title", 40, 150, 500, 48, "A long title here")])
        self.assertEqual(P.headline(d)["id"], "title")  # 48 is within 10% of 50, and far wider
        d = _design(texts=[_text("tag", 40, 40, 400, 80, "Hi"), _text("title", 40, 150, 500, 40, "A long title here")])
        self.assertEqual(P.headline(d)["id"], "tag")  # clearly larger type is the headline

    def test_p_crop_moves_the_headline(self):
        d = _design(texts=[_text("tag", 40, 40, 400, 50, "Hi"), _text("title", 40, 150, 500, 48, "A long title here")])
        bad = P.perturb("p_crop", d, random.Random(0))
        moved = [e["id"] for e, g in zip(bad["elements"], d["elements"]) if e != g]
        self.assertEqual(moved, ["title"])


class Structural(unittest.TestCase):
    def test_crop_on_the_headline_passes(self):
        d = _design()
        bad = P.perturb("p_crop", d, random.Random(0))
        why, moved = SN.structural("p_crop", d, d, bad)
        self.assertIsNone(why)
        self.assertEqual(moved["id"], "head")

    def test_rotated_tiny_or_already_cut_headline_is_refused(self):
        d = _design(texts=[_text("head", 40, 40, 300, 48, "Big headline", rot=90)])
        self.assertIn("rotated", SN.structural("p_crop", d, d, P.perturb("p_crop", d))[0])
        d = _design(w=2000, h=2000, texts=[_text("head", 40, 40, 100, 8, "tiny")])
        self.assertIn("of the canvas", SN.structural("p_crop", d, d, P.perturb("p_crop", d))[0])
        d = _design(texts=[_text("head", 450, 40, 400, 48, "Big headline that runs off")])
        self.assertIn("already cut", SN.structural("p_crop", d, d, P.perturb("p_crop", d))[0])

    def test_a_defect_off_the_headline_is_refused(self):
        d = _design()
        bad = json.loads(json.dumps(d))
        bad["elements"][2]["x"] += 500  # the subhead, not the headline
        self.assertIn("not on the headline", SN.structural("p_crop", d, d, bad)[0])

    def test_onart_needs_a_headline_that_was_off_the_picture(self):
        on = _design(image={"x": 0, "y": 0, "width": 500, "height": 300})
        bad = P.perturb("p_onart", on)
        self.assertIsNotNone(bad)
        self.assertIn("already", SN.structural("p_onart", on, on, bad)[0])
        off = _design(texts=[_text("head", 40, 300, 400, 48, "Big headline")],
                      image={"x": 300, "y": 20, "width": 280, "height": 200})
        bad = P.perturb("p_onart", off)
        self.assertIsNone(SN.structural("p_onart", off, off, bad)[0])

    def test_fit_centre_needs_a_real_change_of_shape(self):
        src = _design(600, 400)
        same = _design(660, 400)
        self.assertIn("aspect", SN.structural("fit_centre", src, same, same)[0])
        tall = _design(400, 900)
        self.assertIsNone(SN.structural("fit_centre", src, tall, tall)[0])


class GoodSide(unittest.TestCase):
    def test_clean_good_side(self):
        d = _design()
        self.assertEqual(SN.good_ok(d, d), [])

    def test_good_side_with_planted_defects_is_refused(self):
        src = _design()
        cut = json.loads(json.dumps(src))
        cut["elements"][1]["x"] = 450
        self.assertTrue(any("off the canvas" in w for w in SN.good_ok(src, cut)))
        over = json.loads(json.dumps(src))
        over["elements"][2]["y"] = 45
        self.assertTrue(any("copy over copy" in w for w in SN.good_ok(src, over)))
        hid = json.loads(json.dumps(src))
        hid["elements"][2]["opacity"] = 0
        self.assertTrue(any("hidden sub" in w for w in SN.good_ok(src, hid)))

    def test_the_sources_own_bleed_is_the_design_but_overlap_never_is(self):
        src = _design(texts=[_text("head", 450, 40, 400, 48, "Big headline that runs off"),
                             _text("sub", 40, 200, 300, 18, "below")])
        self.assertEqual(SN.good_ok(src, src), [])
        src = _design(texts=[_text("head", 40, 40, 400, 48, "Big headline"), _text("sub", 60, 50, 300, 18, "on top")])
        self.assertTrue(any("copy over copy" in w for w in SN.good_ok(src, src)))

    def test_a_dropped_picture_is_the_designers_choice(self):
        src = _design(image={"x": 300, "y": 20, "width": 280, "height": 200})
        good = json.loads(json.dumps(src))
        good["elements"][1]["opacity"] = 0
        self.assertEqual(SN.good_ok(src, good), [])


class Pixels(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def png(self, name, boxes):
        a = np.full((100, 200, 3), 255, np.uint8)
        for x0, y0, x1, y1 in boxes:
            a[y0:y1, x0:x1] = 0
        p = self.tmp / f"{name}.png"
        Image.fromarray(a).save(p)
        return p

    def test_invisible_and_visible_changes(self):
        from engine.smart.geom import Box

        canvas = {"width": 400, "height": 200}
        g = self.png("g", [(10, 10, 60, 30)])
        same = self.png("s", [(10, 10, 60, 30)])
        why, share = SN.pixel_ok(g, same, canvas)
        self.assertIn("pixels change", why)
        self.assertEqual(share, 0.0)
        moved = self.png("m", [(140, 10, 190, 30)])
        self.assertIsNone(SN.pixel_ok(g, moved, canvas, Box(280, 20, 100, 40))[0])
        # the element "moved" to a box where nothing shows (covered, or the same colour)
        self.assertIn("shows in", SN.pixel_ok(g, moved, canvas, Box(20, 140, 100, 40))[0])
        self.assertIn("pixels change", SN.pixel_ok(g, moved, canvas, min_diff=0.5)[0])


class PanelScreen(unittest.TestCase):
    def setUp(self):
        import os
        from unittest import mock

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        p = mock.patch.dict(os.environ, {"CHAMELEON_JUDGE_ROOT": str(self.tmp / "judges")})
        p.start()
        self.addCleanup(p.stop)

    def _pool(self):
        clean = _design()
        messy = _design(texts=[_text("head", 40, 40, 400, 48, "Big headline"),
                               _text("sub", 40, 45, 400, 18, "a smaller subhead line")])
        src = _design()
        return [{"set": "dev", "tile": "clean", "design": "clean", "source": src, "base": None, "reference": clean},
                {"set": "dev", "tile": "messy", "design": "messy", "source": src, "base": None, "reference": messy}]

    def test_screened_sentinels_use_only_clean_good_sides(self):
        from tools.gate import panel as PN

        pn = PN.Panel("scr", self.tmp / "priv", {"view": 256, "sentinel_kinds": ["p_crop"]}, judge=lambda p, m: ("", {}))
        got = pn.sentinels(self._pool(), 3, 0, pn.families)
        self.assertEqual([len(v) for v in got.values()], [3, 3])
        self.assertTrue(pn.screened)
        self.assertTrue(all(s["tile"] == "dev.messy" for s in pn.screened))
        self.assertTrue(all("good side" in s["why"] for s in pn.screened))

    def test_screen_off_keeps_the_old_behaviour(self):
        from tools.gate import panel as PN

        pn = PN.Panel("scr2", self.tmp / "priv2", {"view": 256, "sentinel_kinds": ["p_crop"], "sentinel_screen": False},
                      judge=lambda p, m: ("", {}))
        got = pn.sentinels(self._pool(), 4, 0, pn.families)
        self.assertEqual([len(v) for v in got.values()], [4, 4])
        self.assertEqual(pn.screened, [])


if __name__ == "__main__":
    unittest.main()


class SentinelCheckDry(PanelScreen):
    def test_dry_run_and_engine_pool(self):
        from unittest import mock

        from tools.gate import sentinel_check as SC

        tasks = [{"task_id": "a", "source": _design(), "target_canvas": {"width": 600, "height": 400},
                  "reference": _design()},
                 {"task_id": "b", "source": _design(), "target_canvas": {"width": 600, "height": 400}}]
        with mock.patch.object(SC.devsets, "tasks", return_value=(tasks, self.tmp / "assets")), \
                mock.patch.object(SC, "GATE_ROOT", self.tmp):
            self.assertEqual(len(SC.pool_for("devx")), 1)
            pool = SC.pool_for("devx", engine=True)
            self.assertEqual(len(pool), 2)
            self.assertIsNotNone(pool[1]["base"])
            self.assertEqual(len(SC.pool_for("devx", engine=True, n_tasks=1)), 1)
            r = SC.dry("dry1", "devx", 4, {"view": 256, "sentinel_kinds": ["p_crop"]})
            self.assertEqual(r["made"].get("p_crop"), 4)
        self.assertFalse((self.tmp / "judges" / "dry1").exists())

    def test_prereg_checks_the_screen_thresholds(self):
        from tools.gate import prereg as PR
        from tools.gate.panel import DEFAULT_PANEL

        self.assertFalse([e for e in PR.validate_panel(dict(DEFAULT_PANEL)) if "min_diff" in e])
        self.assertTrue([e for e in PR.validate_panel(dict(DEFAULT_PANEL, sentinel_min_diff=0.9)) if "min_diff" in e])


class EdgeInk(Pixels):
    def test_rendered_copy_at_the_edge(self):
        plain = self.png("plain", [])
        inside = self.png("inside", [(50, 40, 120, 60)])
        cut = self.png("cut", [(150, 40, 200, 60)])
        self.assertEqual(SN.edge_ink(inside, plain), 0.0)
        e = SN.edge_ink(cut, plain)
        self.assertGreater(e, SN.EDGE_MAX)
        self.assertIn("off the edge", SN.rendered_ok(e, 0.0))
        self.assertIsNone(SN.rendered_ok(e, e))  # the source runs its copy to the edge too

    def test_no_text_drops_texts_only(self):
        d = _design()
        self.assertEqual([e["id"] for e in SN._no_text(d)["elements"]], ["bg"])
        self.assertEqual(len(d["elements"]), 3)


class NoSentinels(unittest.TestCase):
    def test_a_family_with_no_sentinels_at_the_stop_is_dropped(self):
        from tools.gate import run as R, stats

        rule = stats.rule_with_defaults({})
        sent = {"claude": {"n": 0, "correct": 0}, "cursor": {"n": 8, "correct": 8}}
        self.assertEqual(R.sentinel_drops(sent, ["claude", "cursor"], rule, 1, stop=False), {})
        self.assertIn("no sentinels", R.sentinel_drops(sent, ["claude", "cursor"], rule, 1, stop=True)["claude"])
        rule = dict(rule, sentinel_share=0)
        self.assertEqual(R.sentinel_drops(sent, ["claude", "cursor"], rule, 1, stop=True), {})
