"""A per-target planning budget with early exit (finale API; off by default).

Candidate generation walks drop levels (hide nothing, then the least important
unit, ...) and, per level, every archetype variant. Most of the time goes to
the first levels on roomy canvases, where the winner usually is anyway; on
tiny canvases the deep levels win but fail or succeed fast. So the budget is
checked before each planner attempt and, once at least one plan exists, stops
generating and ranks what it has.

Two limits, either or both:

  ms     wall-clock milliseconds for one target. Same input -> same output as
         long as it is not hit; when it is, the result says so (`hit`).
  work   text measurements (textfit.measure calls), a deterministic proxy for
         time: the same input always stops at the same point.

`resize()` never sets a budget, so stage-one output is unchanged.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

_ACTIVE: ContextVar[Optional["Budget"]] = ContextVar("chameleon_budget", default=None)


@dataclass
class Budget:
    ms: Optional[float] = None
    work: Optional[int] = None
    t0: float = field(default_factory=time.perf_counter)
    w0: int = 0
    attempts: int = 0
    hit: bool = False

    def start(self) -> "Budget":
        from . import textfit

        self.t0 = time.perf_counter()
        self.w0 = textfit.measure_calls()
        self.attempts = 0
        self.hit = False
        return self

    def used_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def used_work(self) -> int:
        from . import textfit

        return textfit.measure_calls() - self.w0

    def report(self) -> dict:
        return {"ms": round(self.used_ms(), 1), "work": self.used_work(), "attempts": self.attempts,
                "hit": self.hit, "limit_ms": self.ms, "limit_work": self.work}


def make(spec) -> Optional[Budget]:
    """None, a Budget, a number (ms) or {"ms": .., "work": ..}."""
    if spec is None or spec is False:
        return None
    if isinstance(spec, Budget):
        return Budget(ms=spec.ms, work=spec.work)
    if isinstance(spec, (int, float)):
        return Budget(ms=float(spec))
    if isinstance(spec, dict):
        ms, work = spec.get("ms"), spec.get("work")
        return Budget(ms=float(ms) if ms else None, work=int(work) if work else None)
    raise TypeError(f"budget: expected ms, dict or Budget, got {spec!r}")


def active() -> Optional[Budget]:
    return _ACTIVE.get()


@contextmanager
def using(b: Optional[Budget]):
    if b is not None:
        b.start()
    token = _ACTIVE.set(b)
    try:
        yield b
    finally:
        _ACTIVE.reset(token)


def exhausted(plans) -> bool:
    """Called before each planner attempt. Never stops before a first plan exists."""
    b = _ACTIVE.get()
    if b is None:
        return False
    b.attempts += 1
    if not plans:
        return False
    if (b.ms is not None and b.used_ms() > b.ms) or (b.work is not None and b.used_work() > b.work):
        b.hit = True
        return True
    return False
