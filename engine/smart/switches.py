"""Mechanism switches: CHAMELEON_OFF=<name>[,<name>] turns named mechanisms off (default: on, except DEFAULT_OFF; CHAMELEON_ON=<name> turns one of those on).

B1's names: ``search`` (anchor / flex / grid / tile / strip and extra split and stack parameters),
``retrieve`` (designer-resize retrieval from the Crello family index), ``ranker`` (the learned
designer-preference term inside the score).
"""

from __future__ import annotations

import os


# Off unless CHAMELEON_ON names them: the extra candidate generators still break finicky floors
# (pill-label font, photo coverage, tiny-item relations) and are not yet floor-green.
DEFAULT_OFF = frozenset({"search", "retrieve"})


def _names(var: str) -> set:
    return {s.strip() for s in os.environ.get(var, "").split(",") if s.strip()}


def off(name: str) -> bool:
    return name in _names("CHAMELEON_OFF") or (name in DEFAULT_OFF and name not in _names("CHAMELEON_ON"))
