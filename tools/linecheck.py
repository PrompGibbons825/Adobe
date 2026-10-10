"""Check that every text in our output wraps into exactly the lines it had in
the source (same words per line). usage: python tools/linecheck.py <root>|official"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import engine.resize as R  # noqa: E402
from engine import textfit as tf  # noqa: E402

root = sys.argv[1]
if root == "official":
    cases = [(f.stem, json.loads(f.read_text())["source"], json.loads(f.read_text())["target_canvas"])
             for f in sorted(Path("input").glob("*.json"))]
else:
    idx = json.loads(Path(f"{root}/index.json").read_text())
    cases = [(f"p{k}", json.loads(Path(f"{root}/designs/{p[0]['file']}").read_text()),
              {"width": p[1]["w"], "height": p[1]["h"]}) for k, p in enumerate(idx)]
bad = 0
total = 0
for name, src, tgt in cases:
    out = R.resize(src, tgt)
    o = {e["id"]: e for e in out["elements"]}
    for e in src["elements"]:
        if e["type"] != "text" or (o[e["id"]].get("opacity", 1) == 0):
            continue
        total += 1
        s = tf._breaks(tf.measure(e, float(e["font_size"]), float(e["width"]))["wrapped"])
        q = o[e["id"]]
        r = tf._breaks(tf.measure(q, float(q["font_size"]), float(q["width"]))["wrapped"])
        if s != r:
            bad += 1
            print(f"{name} {e['id']} src {s} -> out {r}  {e['content'][:40]!r}  {R.resize.last_choice[0]['template']}")
print(f"{root}: {total - bad}/{total} texts keep their source lines")
