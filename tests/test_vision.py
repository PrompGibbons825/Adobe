"""Subject-aware photos (engine/smart/vision.py, S5): the focal crop by bleed is legal, keeps a
face whole, and leaves flat pictures alone.

    .venv/bin/python -m unittest tests.test_vision
"""

from __future__ import annotations

import unittest

from engine.resize import resize
from engine.smart import vision
from tests.finicky import FACE_BOX, FACE_PHOTO, FACE_PHOTO_SIZE, sprint_b_cases
from tools.rules import hard_rule_errors

try:
    import cv2  # noqa: F401

    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False


class Vision(unittest.TestCase):
    def test_flat_gradient_keeps_centre_crop(self):
        v = vision.analyze("solstice_bg_gradient.png")
        self.assertIsNotNone(v)
        self.assertFalse(v.subject)
        for W, H in ((1080, 1920), (1920, 1080), (728, 90)):
            self.assertIsNone(vision.cover_box(v, W, H))

    def test_missing_asset_has_no_opinion(self):
        self.assertIsNone(vision.analyze("no_such_asset.png"))
        self.assertIsNone(vision.cover_box(None, 1080, 1920))

    @unittest.skipUnless(HAVE_CV2, "OpenCV not installed")
    def test_face_found_and_kept_whole(self):
        v = vision.analyze(FACE_PHOTO)
        self.assertIsNotNone(v)
        self.assertEqual(v.size, FACE_PHOTO_SIZE)
        self.assertEqual(len(v.faces), 1)
        f = v.faces[0]
        self.assertTrue(all(abs(a - b) < 0.05 for a, b in zip(f, FACE_BOX)), f)
        for W, H in ((1080, 1080), (1080, 1350), (1080, 1920)):
            box = vision.cover_box(v, W, H) or vision.centre_box(v, W, H)
            x, y, w, h = box
            self.assertLessEqual(x, 0.5)
            self.assertLessEqual(y, 0.5)
            self.assertGreaterEqual(x + w, W - 0.5)
            self.assertGreaterEqual(y + h, H - 0.5)
            self.assertAlmostEqual(w / h, FACE_PHOTO_SIZE[0] / FACE_PHOTO_SIZE[1], places=3)
            for hx, hy, hw, hh in vision.drawn_faces(v, box, "faces"):
                self.assertGreaterEqual(hx, -1.0)
                self.assertLessEqual(hx + hw, W + 1.0)

    def test_bleed_output_passes_the_hard_rules(self):
        case = sprint_b_cases()[0]
        src = case.source
        for name in case.targets:
            tgt = case.target(name)
            out = resize(src, tgt)
            self.assertEqual(hard_rule_errors(src, out, tgt), [], name)
            photo = next(e for e in out["elements"] if e["id"] == "photo")
            self.assertEqual(photo.get("fit"), "cover")
            self.assertLessEqual(photo["x"], 0.5)
            self.assertGreaterEqual(photo["x"] + photo["width"], tgt["width"] - 0.5)


if __name__ == "__main__":
    unittest.main()
