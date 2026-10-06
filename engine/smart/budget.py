"""A per-target planning budget with early exit (finale API; off by default).

Candidate generation walks drop levels (hide nothing, then the least important
unit, ...) and, per level, every archetype variant. Most of the time goes to
the first levels on roomy canvases, where the winner usually is anyway; on
tiny canvases the deep levels win but fail or succeed fast. So the budget is
checked before each planner attempt and, once at least one plan exists, stops
generating and ranks what it has.

Branch and bound (on whenever a budget is given, `bound=False` to switch it
off): drop levels are cumulative, so the best score a deeper level can reach
only falls. Its ceiling is the importance of what stays visible, minus that
of what is hidden, plus the largest archetype prior; every other score term is
a penalty. Once a found plan scores above that ceiling, deeper levels are
skipped. (Scores are provisional while planning, so this is a heuristic cut,
not a proof; tools/channels.py budget measures how often it changes a pick.)

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
    bound: bool = True
    t0: float = field(default_factory=time.perf_counter)
    w0: int = 0
    attempts: int = 0
    hit: bool = False
    pruned: int = 0  # drop levels skipped by the bound

    def start(self) -> "Budget":
        from . import textfit

        self.t0 = time.perf_counter()
        self.w0 = textfit.measure_calls()
        self.attempts = 0
        self.hit = False
        self.pruned = 0
        return self

    def used_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def used_work(self) -> int:
        from . import textfit

        return textfit.measure_calls() - self.w0

    def report(self) -> dict:
        return {"ms": round(self.used_ms(), 1), "work": self.used_work(), "attempts": self.attempts,
                "hit": self.hit, "pruned_levels": self.pruned, "limit_ms": self.ms, "limit_work": self.work}


def make(spec) -> Optional[Budget]:
    """None, a Budget, a number (ms) or {"ms": .., "work": ..}."""
    if spec is None or spec is False:
        return None
    if isinstance(spec, Budget):
        return Budget(ms=spec.ms, work=spec.work, bound=spec.bound)
    if isinstance(spec, (int, float)):
        return Budget(ms=float(spec))
    if isinstance(spec, dict):
        ms, work = spec.get("ms"), spec.get("work")
        return Budget(ms=float(ms) if ms else None, work=int(work) if work else None,
                      bound=bool(spec.get("bound", True)))
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


def beyond_bound(d, ctx, hidden, plans) -> bool:
    """True when no plan hiding `hidden` can outscore the best plan found so far."""
    b = _ACTIVE.get()
    if b is None or not b.bound or not plans:
        return False
    from .score import _prior, score_plans

    hid = {id(h) for h in hidden}
    keep = sum(it.importance for it in d.items if id(it) not in hid and it.role not in ("missing", "void"))
    lost = sum(it.importance for it in d.items if id(it) in hid and it.role != "missing")
    r, r0 = ctx.W / ctx.H, d.W / d.H
    prior = max(_prior(n, r, r0) for n in ("adapt", "stack", "split-left-0.34", "row"))
    ceiling = keep - lost + max(0.0, prior)
    # only a plan that will survive counts: full legibility (prune_relaxed drops
    # the others) and no copy over copy (the shipped ranking skips those)
    full = [p for p in plans if p.relax >= 1.0]
    if not full:
        return False
    score_plans(plans, d, ctx)  # provisional; generate()'s caller scores again at the end
    from . import emit
    from .textfit import deliberate_overlaps, text_overlaps

    exempt = None
    for p in sorted(full, key=lambda p: -p.score):
        if p.score <= ceiling:
            return False
        if exempt is None:
            exempt = frozenset(deliberate_overlaps(d.elements))
        if not text_overlaps(emit(p, d, ctx), exempt=exempt):
            b.pruned += 1
            return True
    return False
