"""Dev tool: measure designer decisions on Crello families and fit the shallow decision trees.

    /data/agents/work/designer/venv-dev/bin/python -m tools.designer_fit dataset   # -> rows.jsonl
    /data/agents/work/designer/venv-dev/bin/python -m tools.designer_fit fit       # CV table + designer_tree.json

Data: /data/agents/work/crello-wide/families.jsonl (geometry only). Tournament families are
excluded from every fit. Rows: one per ordered pair (source member, target member) of a family.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.smart import decisions as D  # noqa: E402

FAMILIES = Path(os.environ.get("FAMILIES", "/data/agents/work/crello-wide/families.jsonl"))
TOURNAMENT = Path("/data/agents/work/designer/tournament_families.txt")
WORK = Path(os.environ.get("DESIGNER_WORK", "/data/agents/work/designer/tree"))
TREE_OUT = ROOT / "engine" / "smart" / "designer_tree.json"


def member_to_design(m: dict) -> dict:
    """families.jsonl member -> Chameleon-schema design. TextElement -> text (content, font_size),
    ImageElement/SvgElement/SvgMaskElement -> image, others -> shape. ``_kind`` keeps the Crello kind
    for matching; ids are unique per member so matching across members goes by content/kind."""
    els = []
    for e in m["els"]:
        k = e["kind"]
        base = {"id": f"{m['id']}_{e['i']}", "x": e["x"], "y": e["y"], "width": e["w"], "height": e["h"],
                "z_index": e["i"], "rotation": e.get("angle", 0.0), "opacity": e.get("opacity", 1.0), "_kind": k}
        if k == "TextElement":
            base.update(type="text", content=e.get("text", ""), font_size=e.get("fs") or 0.0,
                        font_family=e.get("font", ""), text_align=e.get("align") or "left")
        elif k in ("ImageElement", "SvgElement", "SvgMaskElement"):
            base.update(type="image", asset_id="", fit="fill")
        else:
            base.update(type="shape", shape_type="rectangle", fill_color=e.get("color") or "rgba(0,0,0,1)")
        els.append(base)
    return {"canvas": {"width": m["W"], "height": m["H"]}, "elements": els}


def tournament() -> set:
    return {l.strip() for l in TOURNAMENT.read_text().split() if l.strip()}


def fold_of(fam: str, k: int = 5) -> int:
    return int(hashlib.md5(fam.encode()).hexdigest(), 16) % k


def iter_families():
    with FAMILIES.open() as fh:
        for line in fh:
            yield json.loads(line)


def aspect_class(x: float) -> str:
    return "small" if x < 0.3 else ("mid" if x <= 0.9 else "large")


def build_rows(out_path: Path):
    tour = tournament()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as fo:
        for f in iter_families():
            designs = [member_to_design(m) for m in f["members"]]
            for i, s in enumerate(designs):
                for j, t in enumerate(designs):
                    if i == j:
                        continue
                    feats = D.source_features(s, t)
                    dec = D.decisions_of(s, t)
                    keep = D.keep_source(s)
                    fo.write(json.dumps({"fam": f["fam"], "tour": f["fam"] in tour, "n_sizes": f["n_sizes"],
                                         "src": f["members"][i]["id"], "tgt": f["members"][j]["id"],
                                         "feats": feats, "dec": dec, "keep": keep}) + "\n")
                    n += 1
    return n


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    if cmd == "dataset":
        print(build_rows(WORK / "rows.jsonl"), "rows")
