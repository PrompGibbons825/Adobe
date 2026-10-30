"""Ranker training data from the wide Crello family index (dev only).

    .venv/bin/python -m tools.ranker_wide [--workers 4] [--limit N] [--exclude sample.json]

Source: families.jsonl from tools/crello_wide.py (Crello v5, CDLA-Permissive-2.0; sealed families
already dropped). Geometry and copy only: image elements carry no pixels here, so the candidates are
judged on layout alone. One task per family: a designer's member -> the same family's member at the
most different aspect ratio (the hardest resize the designer actually made). While a family is
being dumped, it is excluded from the retrieval index, so a "retrieve" candidate never sees the
answer. Splits are by family hash (fold_of): fold 0 is held out.

Output: tools/crello/ranker/wide.jsonl, rows in the format of tools/ranker_data.py.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tools" / "crello" / "ranker"
WIDE = Path(os.environ.get("CRELLO_WIDE", "/data/agents/work/crello-wide"))
FOLDS = 5


def fold_of(fam: str, k: int = FOLDS) -> int:
    return int(hashlib.md5(fam.encode()).hexdigest()[:6], 16) % k


def design(mem: dict) -> dict:
    """A family member as a Chameleon design (images are geometry only: their assets are absent)."""
    els = []
    for e in mem["els"]:
        b = dict(id=f"e{e['i']}", x=e["x"], y=e["y"], width=max(1.0, e["w"]), height=max(1.0, e["h"]),
                 z_index=e["i"], rotation=e["angle"], opacity=e["opacity"])
        if e["kind"] == "TextElement":
            b.update(type="text", content=e["text"], font_family=e.get("font") or "Roboto", font_size=e["fs"],
                     color=e["color"] or "rgb(0,0,0)",
                     text_align=e["align"] if e["align"] in ("left", "center", "right") else "left",
                     line_height=1.2, letter_spacing=0.0)
            if e["bold"]:
                b["font_weight"] = "bold"
        else:
            b.update(type="image", asset_id=f"absent_{mem['id']}_{e['i']}.png", fit="fill")
        els.append(b)
    return {"canvas": {"width": mem["W"], "height": mem["H"]}, "elements": els}


def pair(fam: dict):
    """(source, target) members: the pair with the largest aspect change, ties to the larger source."""
    mem = fam["members"]
    best = None
    for a in mem:
        for b in mem:
            if a is b:
                continue
            k = (abs(math.log(a["ar"] / b["ar"])), a["W"] * a["H"])
            if best is None or k > best[0]:
                best = (k, a, b)
    return best[1], best[2]


def _init():
    os.environ["CHAMELEON_RANKER"] = "0"
    sys.path.insert(0, str(ROOT))


def one(fam: dict):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from engine.smart import candidates
        from tools.ranker_data import dump

        candidates.EXCLUDE.clear()
        candidates.EXCLUDE.add(fam["fam"])
        a, b = pair(fam)
        row = dict(fam=fam["fam"], ti=0, split="held" if fold_of(fam["fam"]) == 0 else "tune",
                   src_id=a["id"], tgt_id=b["id"])
        try:
            return dump(design(a), design(b), row)
        except Exception as e:  # a dev tool: record and move on
            row.update(error=repr(e), cands=[])
            return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--families", default=str(WIDE / "families.jsonl"))
    a = ap.parse_args()
    excl = set()
    for p in a.exclude:
        excl.update(x["fam"] if isinstance(x, dict) else x for x in json.loads(Path(p).read_text()))
    fams = [json.loads(l) for l in open(a.families)]
    fams = [f for f in fams if f["fam"] not in excl]
    fams.sort(key=lambda f: hashlib.md5(f["fam"].encode()).hexdigest())  # a stable, unbiased order
    out = OUT / "wide.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["fam"])
            except ValueError:
                pass
    todo = [f for f in fams if f["fam"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    with open(out, "a") as fh, ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
        for i, row in enumerate(ex.map(one, todo, chunksize=1)):
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            print(i, row["fam"][:8], len(row.get("cands", [])), f"{row.get('secs', 0):.1f}s", flush=True)


if __name__ == "__main__":
    main()
