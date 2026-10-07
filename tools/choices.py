"""Print the winning template for every pair in a Crello export (regression aid).

usage: python tools/choices.py crello_holdout   (with CHAMELEON_* env vars set)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import resize as R  # noqa: E402

root = sys.argv[1]
idx = json.load(open(f"{root}/index.json"))
worst = 0.0
for i, pair in enumerate(idx):
    a, b = pair[0], pair[1]
    src = json.load(open(f"{root}/designs/{a['file']}"))
    t = time.time()
    out = R.resize(src, {"width": b["w"], "height": b["h"]})
    worst = max(worst, time.time() - t)
    assert [e["id"] for e in out["elements"]] == [e["id"] for e in src["elements"]]
    print(f"p{i}", R.resize.last_choice[0]["template"])
print(f"max {worst:.2f}s", file=sys.stderr)
