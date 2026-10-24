"""Gate tool: the dev sets. Open data, for direction and debugging; never evidence for the gate.

    .venv/bin/python -m tools.gate.devsets            # each dev set with its task count

Every dev set yields tasks in the sealed store's shape (``task_id``, ``source``,
``target_canvas``, optional ``reference``), with the assets folder they resolve against:

  samples       input/*.json, the brief's three samples, at their given target
  cases         tools/cases/*.json (tools/make_cases.py), at their given target
  crello_old    the 64 Crello families of tools/crello/families.json, all dev now (both the old
                tuning and held halves), first target with the designer's remake as reference;
                re-create the data with `python -m tools.crello meta && ... pick && ... fetch`
  finicky       tests/finicky.py, every case at each of its targets
  designs       tools/designs.py (n=40, seed=1) at every tools/sheet.CATALOGUE format, as
                tools/audit does
  <set>-v<N>    any sealed version retired into tools/gate/dev/ (tools.gate.sealed retire)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BRAND_ASSETS = REPO / "brand_kit" / "assets"
CRELLO_CASES = REPO / "tools" / "crello" / "cases"


def _given(paths) -> Iterator[Dict]:
    for p in paths:
        t = json.loads(p.read_text())
        yield {"task_id": t.get("task_id", p.stem), "source": t["source"], "target_canvas": t["target_canvas"]}


def samples() -> Iterator[Dict]:
    return _given(sorted((REPO / "input").glob("*.json")))


def cases() -> Iterator[Dict]:
    return _given(sorted((REPO / "tools" / "cases").glob("*.json")))


def crello_old() -> Iterator[Dict]:
    for p in sorted(CRELLO_CASES.glob("fam*.json")):
        fam = json.loads(p.read_text())
        if not fam["targets"]:
            continue
        ref = fam["targets"][0]
        yield {"task_id": p.stem, "source": fam["source"], "target_canvas": dict(ref["canvas"]), "reference": ref}


def finicky() -> Iterator[Dict]:
    from tests import finicky as f

    for c in f.cases():
        for name in c.targets:
            yield {"task_id": f"{c.name}@{name}", "source": c.source, "target_canvas": c.target(name)}


def designs(n: int = 40, seed: int = 1) -> Iterator[Dict]:
    from tools import designs as d
    from tools.sheet import CATALOGUE

    for t in d.designs(n, seed):
        for fmt, (w, h) in CATALOGUE.items():
            yield {"task_id": f"{t['task_id']}@{fmt}", "source": t["source"], "target_canvas": {"width": w, "height": h}}


def _retired(path: Path) -> Callable[[], Iterator[Dict]]:
    def gen():
        for p in sorted((path / "items").glob("*.json")):
            yield json.loads(p.read_text())
    return gen


def registry() -> Dict[str, Tuple[Callable[[], Iterator[Dict]], Path]]:
    """name -> (task generator, assets dir)."""
    from tools.gate import sealed

    reg = {
        "samples": (samples, BRAND_ASSETS),
        "cases": (cases, BRAND_ASSETS),
        "crello_old": (crello_old, CRELLO_CASES / "assets"),
        "finicky": (finicky, BRAND_ASSETS),
        "designs": (designs, BRAND_ASSETS),
    }
    dev = sealed.dev_dir()
    if dev.is_dir():
        for p in sorted(dev.iterdir()):
            if p.is_dir() and sealed.VERSION_RE.match(p.name):
                reg[p.name] = (_retired(p), p / "assets")
    return reg


def tasks(name: str) -> Tuple[List[Dict], Path]:
    gen, assets = registry()[name]
    return list(gen()), assets


def main() -> int:
    for name, (gen, assets) in registry().items():
        n = sum(1 for _ in gen())
        print(f"{name:14s} {n:5d} tasks  assets {assets}{'' if assets.is_dir() else '  (missing)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
