"""For each multi-size Crello design: take the smallest-ratio version as the
source, resize it with our engine into every other size the designer made,
and render  [source | designer's version | ours]  side by side.

usage (from repo root):
  CHAMELEON_ASSETS_DIR=crello/assets CHAMELEON_FONTS_PICKLE=crello/fonts_subset.pickle \
      python tools/crello_compare.py crello
"""

import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cr_renderer import CrelloV5Renderer  # noqa: E402
from cr_renderer.fonts import FontManager  # noqa: E402
from renderer.converter import convert  # noqa: E402

from engine.resize import resize  # noqa: E402
from engine.main import self_check  # noqa: E402


def renderer_with(fonts_pickle, assets_dir):
    r = CrelloV5Renderer.__new__(CrelloV5Renderer)
    r.features = convert._STUB_FEATURES
    r.font_manager = FontManager(fonts_pickle)
    convert._renderer_cache[str(Path(assets_dir).parent / "fonts")] = r
    return r


def render(design, assets_dir, path):
    convert.render_design_to_png(design, Path(assets_dir), str(path))
    return Image.open(path).convert("RGB")


def tile(ims, labels, h=420):
    scaled = []
    for im in ims:
        s = h / im.height
        if im.width * s > 700:
            s = 700 / im.width
        scaled.append(im.resize((max(1, int(im.width * s)), max(1, int(im.height * s)))))
    W = sum(i.width for i in scaled) + 20 * (len(scaled) + 1)
    out = Image.new("RGB", (W, h + 40), (238, 238, 238))
    d = ImageDraw.Draw(out)
    x = 20
    for im, lab in zip(scaled, labels):
        d.text((x, 8), lab, fill=(0, 0, 0))
        out.paste(im, (x, 30))
        x += im.width + 20
    return out


def main(root):
    root = Path(root)
    assets = root / "assets"
    renderer_with(str(root / "fonts_subset.pickle"), assets)
    outdir = root / "compare"
    outdir.mkdir(exist_ok=True)
    index = json.loads((root / "index.json").read_text())
    rows = []
    for p, variants in enumerate(index):
        src_v = variants[0]
        src = json.loads((root / "designs" / src_v["file"]).read_text())
        src_png = render(src, assets, outdir / f"p{p}_source.png")
        for v in variants[1:]:
            ref = json.loads((root / "designs" / v["file"]).read_text())
            target = {"width": v["w"], "height": v["h"]}
            ours = resize(src, target)
            meta = getattr(resize, "last_choice", ({"template": "?"}, 0, {}))[0]
            probs = self_check(src, ours)
            (outdir / f"p{p}_{v['w']}x{v['h']}_ours.json").write_text(json.dumps(ours, indent=1))
            ref_png = render(ref, assets, outdir / f"p{p}_{v['w']}x{v['h']}_designer.png")
            our_png = render(ours, assets, outdir / f"p{p}_{v['w']}x{v['h']}_ours.png")
            sheet = tile([src_png, ref_png, our_png],
                         [f"SOURCE {src_v['w']}x{src_v['h']}", f"DESIGNER {v['w']}x{v['h']}",
                          f"OURS [{meta['template']}]"])
            sheet.save(outdir / f"p{p}_{v['w']}x{v['h']}_compare.png")
            rows.append(sheet)
            print(f"p{p}: {src_v['w']}x{src_v['h']} -> {v['w']}x{v['h']}  {meta['template']}  "
                  f"{len(probs)} warnings")
    if rows:
        Wm = max(r.width for r in rows)
        big = Image.new("RGB", (Wm, sum(r.height for r in rows)), (238, 238, 238))
        y = 0
        for r in rows:
            big.paste(r, (0, y))
            y += r.height
        big.save(outdir / "ALL_compare.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "crello")
