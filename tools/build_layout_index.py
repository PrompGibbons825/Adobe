"""Bake Crello eval layouts into engine/smart/data/layout_index.json (dev only).

Reads designer-resized families from tools/crello/eval/r30 when present (run
tools/crello fetch first on a machine with cache). Normalized slot geometry only;
no assets are copied into the image.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "tools" / "crello" / "eval" / "r30"
OUT = ROOT / "engine" / "smart" / "data" / "layout_index.json"


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from engine.smart.geom import el_box

    if not EVAL.is_dir():
        print(f"missing {EVAL}; copy eval families or run crello fetch", file=sys.stderr)
        return 1
    out = []
    seen = set()
    for p in sorted(EVAL.glob("fam*.json")):
        if p.name.endswith(".task.json"):
            continue
        d = json.loads(p.read_text())
        W, H = d["canvas"]["width"], d["canvas"]["height"]
        key = (W, H, len(d["elements"]))
        if key in seen:
            continue
        seen.add(key)
        slots = []
        for e in d["elements"]:
            if float(e.get("opacity", 1) or 0) <= 0:
                continue
            b = el_box(e)
            if b.area <= 0:
                continue
            kind = "text" if e.get("type") == "text" else "image"
            slots.append({
                "kind": kind,
                "cx": b.cx / W,
                "cy": b.cy / H,
                "w": b.w / W,
                "h": b.h / H,
                "area": b.area / (W * H),
            })
        slots.sort(key=lambda s: -s["area"])
        out.append({"id": p.stem, "ar": W / H, "n": len(slots), "slots": slots})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {len(out)} layouts -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
