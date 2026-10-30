"""The inferred type system (engine/smart/system.py): drift and scale shortfall."""

import math
import unittest
from types import SimpleNamespace as NS

from engine.smart import system


def _design(sizes, W=1080, H=1080):
    return NS(W=W, H=H, by_id={k: {"type": "text", "font_size": v} for k, v in sizes.items()})


def _ctx(W=1080, H=1080, floor=10.0):
    return NS(W=W, H=H, floor=lambda role: floor)


def _placed(fonts):
    return [(NS(item=NS(role="body", importance=50.0), fonts={k: v}), 0, 0) for k, v in fonts.items()]


class SystemTest(unittest.TestCase):
    def test_held_ratios_cost_nothing(self):
        d = _design({"a": 60, "b": 30, "c": 15})
        src = system.type_scale(d)
        dr, w = system.drift(_placed({"a": 90, "b": 45, "c": 22.5}), d, _ctx(), src)
        self.assertEqual(dr, 0.0)
        self.assertGreater(w, 0)

    def test_added_contrast_costs(self):
        d = _design({"a": 60, "b": 30})
        src = system.type_scale(d)
        dr, _ = system.drift(_placed({"a": 96, "b": 30}), d, _ctx(), src)
        self.assertAlmostEqual(dr, math.log(1.6) - system.DRIFT_TOL[0], places=6)

    def test_floor_raised_text_is_exempt(self):
        d = _design({"a": 60, "b": 12})
        src = system.type_scale(d)
        dr, _ = system.drift(_placed({"a": 30, "b": 10}), d, _ctx(floor=10.0), src)
        self.assertEqual(dr, 0.0)

    def test_scale_shortfall(self):
        d = _design({"a": 60, "b": 30})
        src = system.type_scale(d)
        self.assertEqual(system.scale_shortfall(_placed({"a": 60, "b": 30}), d, _ctx(), src), 0.0)
        short = system.scale_shortfall(_placed({"a": 30, "b": 15}), d, _ctx(), src)
        self.assertAlmostEqual(short, math.log(2) - system.SCALE_TOL, places=6)


if __name__ == "__main__":
    unittest.main()
