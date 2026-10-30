"""Mechanism switches: ``CHAMELEON_OFF=<name>[,<name>]`` turns named mechanisms off (default on).

Read at call time so a bench can run leave-one-out in one tree.
"""

from __future__ import annotations

import os


def off(name: str) -> bool:
    return name in {s.strip() for s in os.environ.get("CHAMELEON_OFF", "").split(",") if s.strip()}


def on(name: str) -> bool:
    return not off(name)
