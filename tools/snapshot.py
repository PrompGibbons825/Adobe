"""Render every pair of one Crello export (or the official samples) with the
current engine and keep the results under review/steps/<tag>/, so the
progress page can show each pair across every step.

usage (repo root), one set per process (the engine reads its asset and font
locations at import):
  CHAMELEON_ASSETS_DIR=crello_holdout/assets CHAMELEON_FONTS_PICKLE=crello_holdout/fonts_subset.pickle \
      python tools/snapshot.py <tag> crello_holdout
  python tools/snapshot.py <tag> official
or tools/snapshot.sh <tag> for all three sets.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import engine.resize as R  # noqa: E402

tag, root = sys.argv[1], sys.argv[2]
out = Path("review/steps") / tag
out.mkdir(parents=True, exist_ok=True)
logf = out / "templates.json"
log = json.loads(logf.read_text()) if logf.exists() else {}

if root == "official":
    from engine.main import render_to_png
    for f in sorted(Path("input").glob("*.json")):
        task = json.loads(f.read_text())
        res = R.resize(task["source"], task["target_canvas"])
        name = "L" + f.stem.split("level")[1][0]
        render_to_png(res, Path("brand_kit/assets"), out / f"{name}.png")
        log[name] = R.resize.last_choice[0]["template"]
else:
    from tools.crello_compare import render, renderer_with
    renderer_with(f"{root}/fonts_subset.pickle", Path(root) / "assets")
    idx = json.loads(Path(f"{root}/index.json").read_text())
    for k, pair in enumerate(idx):
        a, b = pair[0], pair[1]
        src = json.loads(Path(f"{root}/designs/{a['file']}").read_text())
        res = R.resize(src, {"width": b["w"], "height": b["h"]})
        name = f"{'h' if root.endswith('holdout') else 'd'}{k}"
        render(res, Path(root) / "assets", out / f"{name}.png")
        log[name] = R.resize.last_choice[0]["template"]
logf.write_text(json.dumps(log, indent=1))
print(out, len(log), "renders")
