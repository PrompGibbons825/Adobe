"""Bake designers' resizes into engine/smart/data/layout_index.json (dev only).

    .venv/bin/python -m tools.build_layout_index [--families PATH] [--exclude sample.json ...]

Source: the wide Crello v5 family index (cyberagent/crello, CDLA-Permissive-2.0) written by
tools/crello_wide.py: every design that exists at two or more canvas sizes, sealed families already
dropped. Each family is stored with its elements aligned across its sizes (matched by kind, text and
z-order), as normalized boxes only: no pixels, no copy, no fonts go into the image.
Families named in --exclude files (the dev bench sample) are left out so the bench stays honest.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "engine" / "smart" / "data" / "layout_index.json"
WIDE = Path(os.environ.get("CRELLO_WIDE", "/data/agents/work/crello-wide"))


def _key(e):
    return ("t" if e["kind"] == "TextElement" else "i", (e.get("text") or "").strip().lower())


def aligned(fam):
    """Members' element boxes aligned on a common element order, or None when they do not align."""
    mems = fam["members"]
    order = None
    per = []
    for m in mems:
        groups = collections.defaultdict(list)
        for e in sorted(m["els"], key=lambda e: e["i"]):
            groups[_key(e)].append(e)
        per.append(groups)
        sig = sorted((k, len(v)) for k, v in groups.items())
        if order is None:
            order = sig
        elif sig != order:
            return None
    keys = [k for k, n in order for _ in range(n)]
    idx = [i for _, n in order for i in range(n)]
    out = []
    for m, groups in zip(mems, per):
        W, H = m["W"], m["H"]
        slots = []
        for k, i in zip(keys, idx):
            e = groups[k][i]
            slots.append([round(e["x"] / W, 4), round(e["y"] / H, 4), round(e["w"] / W, 4), round(e["h"] / H, 4), k[0]])
        out.append(dict(ar=round(W / H, 4), s=slots))
    # elements nobody can see (zero opacity / zero size) are dropped from every member alike
    keep = [j for j in range(len(keys))
            if all(o["s"][j][2] > 0 and o["s"][j][3] > 0 for o in out)]
    for o in out:
        o["s"] = [o["s"][j] for j in keep]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=str(WIDE / "families.jsonl"))
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--exclude-fold", type=int, default=None, help="leave out one hash fold (dev holdout)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    excl = set()
    for p in a.exclude:
        data = json.loads(Path(p).read_text())
        excl.update(x["fam"] if isinstance(x, dict) else x for x in data)
    sys.path.insert(0, str(ROOT))
    from tools.ranker_wide import fold_of

    fams, n_bad = [], 0
    for line in open(a.families):
        f = json.loads(line)
        if f["fam"] in excl or (a.exclude_fold is not None and fold_of(f["fam"]) == a.exclude_fold):
            continue
        m = aligned(f)
        if m is None or not m[0]["s"]:
            n_bad += 1
            continue
        fams.append(dict(id=f["fam"], m=m))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(v=2, source="crello-v5 CDLA-Permissive-2.0", families=fams),
                                      separators=(",", ":")))
    print(f"wrote {len(fams)} families ({sum(len(f['m']) for f in fams)} sizes), {n_bad} unaligned, "
          f"{len(excl)} excluded -> {a.out} ({Path(a.out).stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
