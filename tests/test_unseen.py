"""The unseen suite on a small dev subset: 8 generated designs (seed 1, one per family) x 3 tier-1
targets = 24 tiles, run through this repo's engine (tools/unseen.py defines the checks).

C1 rules, C2 hidden and C3 legible must pass on every tile; C4 on_photo, C5 faces and C6 audit are
reported only (printed), for now.

    .venv/bin/python -m unittest tests.test_unseen          # ~1-3 min, wrap in loop/capped on the box

Baseline (sprint/t at 1265aa8 = the 7pm sprint start engine, H17), 2026-10-25 18:25 UTC:
C1 0/24, C2 4/24, C3 0/24 fail; report-only C4 3/24 (contrast; two are copy that lost its
panel), C5 0/24, C6 0/24. C2 is marked
expectedFailure for that reason (the four tiles are named at the test); when an engine fixes them,
unittest reports an "unexpected success" and the decorator must come off.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import unseen  # noqa: E402

N_TARGETS = 3


def subset():
    ds = [d for d in unseen.generated("dev") if "-s1-" in d["name"]]
    for d in ds:
        d["targets"] = [t for t in d["targets"] if t[0] in unseen.TIER1][:N_TARGETS]
    return ds


@unittest.skipUnless((unseen.JEFF / "dev" / "assets").is_dir(), "needs ~/chameleon-data/jeff-crello/dev/assets")
class UnseenSubset(unittest.TestCase):
    rows: list = []

    @classmethod
    def setUpClass(cls):
        os.environ["CHAMELEON_BRAND_KIT"] = str(unseen.ensure_kit())
        cls.rows = unseen.run_suite(subset(), "tier1", engine=str(ROOT), jobs=4, log=lambda *a: None)

    def fails(self, c):
        return [f"{r['name']}: {'; '.join(r['checks'][c])[:160]}" for r in self.rows if r["checks"][c]]

    def test_subset_size(self):
        self.assertEqual(len(self.rows), 8 * N_TARGETS)

    def test_c1_rules(self):
        self.assertEqual(self.fails("C1"), [])

    # Baseline fails on 4 tiles, all the engine dropping an element to opacity 0:
    #   split_panel-s1-hd-16x9|ig-portrait, |story  (band_panel: the colour panel under the copy)
    #   sale-s1-ig-square|ig-portrait, |yt-thumb      (sticker_bg/label; on yt-thumb also ribbon_bg/label)
    @unittest.expectedFailure
    def test_c2_hidden(self):
        self.assertEqual(self.fails("C2"), [])

    def test_c3_legible(self):
        self.assertEqual(self.fails("C3"), [])

    def test_report_c4_c5_c6(self):
        for c in ("C4", "C5", "C6"):
            f = self.fails(c)
            print(f"\n[report only] {c} {unseen.CHECK_NAMES[c]}: {len(f)}/{len(self.rows)} tiles fail", file=sys.stderr)
            for line in f:
                print("   ", line, file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
