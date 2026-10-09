"""Flag pictures that shrank to thumbnails: for every image covering at least
8% of the source canvas, compare its visible share of the canvas in the
source with its share in our output (the part on canvas, after any crop). Cut-outs are
compared with the largest share they could have whole on the new canvas.

usage (repo root, CHAMELEON_* env set): python tools/thumbs.py crello_holdout
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import engine.resize as R  # noqa: E402
from engine import saliency  # noqa: E402

root = sys.argv[1]
idx = json.loads(Path(f"{root}/index.json").read_text())


def share(e, W, H):
    x0, y0 = max(0, e["x"]), max(0, e["y"])
    x1, y1 = min(W, e["x"] + e["width"]), min(H, e["y"] + e["height"])
    return max(0, x1 - x0) * max(0, y1 - y0) / (W * H)


rows = []
for k, pair in enumerate(idx):
    a, b = pair[0], pair[1]
    src = json.loads(Path(f"{root}/designs/{a['file']}").read_text())
    out = R.resize(src, {"width": b["w"], "height": b["h"]})
    o = {e["id"]: e for e in out["elements"]}
    Ws, Hs = src["canvas"]["width"], src["canvas"]["height"]
    for e in src["elements"]:
        if e["type"] != "image":
            continue
        info = saliency.analyze(e.get("asset_id") or "")
        if info is None or info["busyness"] < 0.03:   # pictures only, not flat shapes
            continue
        s0 = share(e, Ws, Hs)
        if s0 < 0.08 or s0 > 0.95:                     # skip small pieces and full backgrounds
            continue
        s1 = share(o[e["id"]], b["w"], b["h"])
        # a picture with a silhouette can't be cropped: compare with the largest
        # share it can have whole on this canvas
        ref = s0
        if info["opaque_frac"] <= 0.97:
            A = e["width"] / max(1e-6, e["height"])
            ref = min(s0, A * b["h"] / b["w"] if A <= b["w"] / b["h"] else b["w"] / (A * b["h"]))
        rows.append((s1 / ref, f"p{k}", e["id"], round(s0, 2), round(s1, 3), R.resize.last_choice[0]["template"]))
rows.sort()
for r in rows:
    flag = "THUMB" if r[0] < 0.35 else ""
    print(f"{r[1]:4} {r[2]:4} src {r[3]:5} out {r[4]:6} ratio {r[0]:.2f} {flag:5} {r[5]}")
