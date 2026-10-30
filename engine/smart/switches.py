"""Mechanism switches: CHAMELEON_OFF=<name>[,<name>] turns named mechanisms off (default: all on).

B1's names: ``search`` (anchor / flex / grid / tile / strip and extra split and stack parameters),
``retrieve`` (designer-resize retrieval from the Crello family index), ``ranker`` (the learned
designer-preference term inside the score).
"""

from __future__ import annotations

import os


def off(name: str) -> bool:
    return name in {s.strip() for s in os.environ.get("CHAMELEON_OFF", "").split(",") if s.strip()}
