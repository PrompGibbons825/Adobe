"""Run any engine checkout over the Crello designer pairs and save its outputs,
for tools/scoreboard.py and tools/beforeafter.py.

usage:
  python tools/h2h.py ENGINE_REPO LABEL DATA_ROOT OUT_DIR
Writes OUT_DIR/<root>_p<k>_<W>x<H>.json for every pair in DATA_ROOT/index.json
(the first variant is the source, every other variant a target size), plus
OUT_DIR/_times_<root>.json. Run it with the engine repo as the working
directory, and the engine's own environment (CHAMELEON_ASSETS_DIR /
CHAMELEON_FONTS_PICKLE for this repo, CHAMELEON_BRAND_KIT for Vlatko's).
"""

import json
import sys
import time
from pathlib import Path


def main(repo, label, root, out):
    sys.path.insert(0, str(Path(repo).resolve()))
    from engine.resize import resize
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    times = {}
    idx = json.loads((root / "index.json").read_text())
    for p, vs in enumerate(idx):
        src = json.loads((root / "designs" / vs[0]["file"]).read_text())
        for v in vs[1:]:
            key = f"{root.name}_p{p}_{v['w']}x{v['h']}"
            t = time.time()
            try:
                res = resize(src, {"width": v["w"], "height": v["h"]})
            except Exception as e:      # an engine that crashes on a pair: record it, go on
                print(f"{key}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            times[key] = round(time.time() - t, 2)
            (out / f"{key}.json").write_text(json.dumps(res))
    (out / f"_times_{root.name}.json").write_text(json.dumps(times, indent=1))
    print(f"{label}: {len(times)} outputs, median {sorted(times.values())[len(times) // 2] if times else 0}s")


if __name__ == "__main__":
    main(*sys.argv[1:5])
