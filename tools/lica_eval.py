"""Resize every LICA design into its target sizes and tile  [Original | ours]
sheets. There is no designer version to score against, so this reports crashes,
self_check warnings and time; the sheets are for judging by eye.

usage (repo root):
  CHAMELEON_ASSETS_DIR=lica/assets CHAMELEON_FONTS_PICKLE=lica/fonts_subset.pickle \
      python tools/lica_eval.py lica
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.main import self_check  # noqa: E402
from engine.resize import resize  # noqa: E402
from tools.crello_compare import render, renderer_with, tile  # noqa: E402


def main(root):
    root = Path(root)
    assets = root / "assets"
    renderer_with(str(root / "fonts_subset.pickle"), assets)
    out = root / "eval"
    out.mkdir(exist_ok=True)
    rows = []
    for p, (src_v, *targets) in enumerate(json.loads((root / "index.json").read_text())):
        src = json.loads((root / "designs" / src_v["file"]).read_text())
        try:
            src_png = render(src, assets, out / f"p{p}_original.png")
        except Exception as e:
            rows.append({"p": p, "cat": src_v["format"], "status": "source_render_failed", "error": repr(e)[:200]})
            continue
        for v in targets:
            r = {"p": p, "cat": src_v["format"], "target": f"{v['w']}x{v['h']}"}
            t = time.time()
            try:
                ours = resize(src, {"width": v["w"], "height": v["h"]})
                r["secs"] = round(time.time() - t, 1)
                r["warnings"] = self_check(src, ours)
                r["template"] = getattr(resize, "last_choice", ({"template": "?"},))[0].get("template")
                png = render(ours, assets, out / f"p{p}_{v['w']}x{v['h']}.png")
                tile([src_png, png], [f"Original {src_v['w']}x{src_v['h']} {src_v['format']}",
                                      f"Ours {r['template']}"]).save(out / f"p{p}_{v['w']}x{v['h']}_sheet.png")
                r["status"] = "ok"
            except Exception as e:
                r.update(status="crash", error=repr(e)[:200])
            rows.append(r)
            print(r, flush=True)
            (out / "results.json").write_text(json.dumps(rows, indent=1))
    print(Counter(r["status"] for r in rows), "with warnings:", sum(bool(r.get("warnings")) for r in rows))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lica")
