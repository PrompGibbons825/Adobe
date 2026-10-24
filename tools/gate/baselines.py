"""Gate tool: reference "engines" and output transforms the gate compares against.

* ``fit_centre(source, target)``: the whole design scaled uniformly to fit the target and
  centred (letterboxed). The naive baseline G6 checks the gate rejects, and the bad side of
  the designer-vs-fit-and-centre sentinels.
* ``nudge(output)``: the null run's A' — every element moved 1 px (right, else left, else down,
  else up; a full-canvas element stays). The panel also re-encodes A' views.
* ``write_builtin(name, root)``: a minimal engine root (``engine/resize.py``) running a builtin,
  so ``tools.gate.engine_run`` can run it like any engine version.
"""

from __future__ import annotations

import json
from pathlib import Path

PX_KEYS = ("x", "y", "width", "height", "font_size", "corner_radius", "stroke_width", "letter_spacing")

FIT_CENTRE_SRC = '''
import json

PX_KEYS = ("x", "y", "width", "height", "font_size", "corner_radius", "stroke_width", "letter_spacing")


def fit_centre(source, target):
    W, H = float(target["width"]), float(target["height"])
    sw, sh = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    s = min(W / sw, H / sh)
    ox, oy = (W - sw * s) / 2, (H - sh * s) / 2
    els = []
    for e in source["elements"]:
        o = json.loads(json.dumps(e))
        for k in PX_KEYS:
            v = o.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                o[k] = v * s
        if isinstance(o.get("font_size"), (int, float)):
            o["font_size"] = max(1, int(round(o["font_size"])))  # the schema wants whole px
        o["x"] = o.get("x", 0) + ox
        o["y"] = o.get("y", 0) + oy
        for r in o.get("style_runs") or []:
            if isinstance(r.get("font_size"), (int, float)):
                r["font_size"] = max(1, int(round(r["font_size"] * s)))
        els.append(o)
    return {"canvas": {"width": target["width"], "height": target["height"]}, "elements": els}
'''

_ns: dict = {}
exec(FIT_CENTRE_SRC, _ns)  # noqa: S102 - our own constant, shared with the builtin engine root
fit_centre = _ns["fit_centre"]

BUILTINS = {"fit_centre": FIT_CENTRE_SRC + "\n\ndef resize(source, target_canvas):\n"
                                           "    return fit_centre(source, target_canvas)\n"}


def write_builtin(name: str, root: Path) -> Path:
    root = Path(root)
    (root / "engine").mkdir(parents=True, exist_ok=True)
    (root / "engine" / "__init__.py").write_text("")
    (root / "engine" / "resize.py").write_text(BUILTINS[name])
    return root


def nudge(out: dict) -> dict:
    o = json.loads(json.dumps(out))
    W, H = float(o["canvas"]["width"]), float(o["canvas"]["height"])
    for e in o["elements"]:
        x, w = float(e.get("x", 0)), float(e.get("width", 0))
        y, h = float(e.get("y", 0)), float(e.get("height", 0))
        if x + w + 1 <= W:
            e["x"] = x + 1
        elif x - 1 >= 0:
            e["x"] = x - 1
        elif y + h + 1 <= H:
            e["y"] = y + 1
        elif y - 1 >= 0:
            e["y"] = y - 1
    return o
