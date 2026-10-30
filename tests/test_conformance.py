"""Renderer conformance (tests/conformance): fundamentals predicts what the renderer draws.

    .venv/bin/python -m unittest tests.test_conformance
"""

from __future__ import annotations

import logging
import unittest

from tests.conformance import prop
from tests.conformance.features import CASES


class Features(unittest.TestCase):
    pass


def _make(name, fn):
    def test(self):
        logging.disable(logging.WARNING)
        errs = fn()
        self.assertEqual(errs, [], name)
    return test


for _name, _fn in CASES.items():
    setattr(Features, "test_" + "".join(c if c.isalnum() else "_" for c in _name.lower()), _make(_name, _fn))


class Property(unittest.TestCase):
    def test_random_texts(self):
        logging.disable(logging.WARNING)
        self.assertEqual(prop.main(["--n", "150", "--seed", "7", "--show", "3"]), 0)


if __name__ == "__main__":
    unittest.main()
