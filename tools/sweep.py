"""Dev harness: run every sample design into many real-world formats and
build one contact sheet per design (not part of the submitted engine)."""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.resize import resize  # noqa: E402
from engine.main import self_check  # noqa: E402
from renderer.converter.convert import render_design_to_png  # noqa: E402

TARGETS = {
    "IG post 1080x1080": (1080, 1080),
    "IG portrait 1080x1350": (1080, 1350),
    "Story 1080x1920": (1080, 1920),
    "Landscape 1920x1080": (1920, 1080),
    "FB cover 1640x624": (1640, 624),
    "Twitter header 1500x500": (1500, 500),
    "Banner 1600x400": (1600, 400),
    "Leaderboard 728x90": (728, 90),
    "Med rect 300x250": (300, 250),
    "Skyscraper 160x600": (160, 600),
    "Billboard 970x250": (970, 250),
    "Pinterest 1000x1500": (1000, 1500),
}

OUT = Path("output/sweep")


def main(only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path("input").glob("*.json")):
        task = json.loads(f.read_text())
        src = task["source"]
        tiles = []
        targets = dict(TARGETS)
        tc = task["target_canvas"]
        targets[f"TASK {tc['width']}x{tc['height']}"] = (tc["width"], tc["height"])
        for name, (w, h) in targets.items():
            if only and only not in name:
                continue
            out = resize(src, {"width": w, "height": h})
            probs = self_check(src, out)
            meta, sc, br = getattr(resize, "last_choice", ({"template": "fallback"}, 0, {}))
            png = OUT / f"{f.stem}_{w}x{h}.png"
            render_design_to_png(out, Path("brand_kit/assets"), str(png))
            (OUT / f"{f.stem}_{w}x{h}.json").write_text(json.dumps(out, indent=1))
            print(f"{f.stem[18:]:<16} {name:<26} {meta['template']:<26} {sc:6.1f} "
                  f"{'; '.join(probs)[:90]}")
            tiles.append((name + f"  [{meta['template']}]", png))
        # contact sheet
        th = 360
        ims = []
        for label, p in tiles:
            im = Image.open(p).convert("RGB")
            s = min(th / im.height, 560 / im.width)
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
            ims.append((label, im))
        cols = 4
        cw = 580
        rows = (len(ims) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cw, rows * (th + 30)), (235, 235, 235))
        d = ImageDraw.Draw(sheet)
        for i, (label, im) in enumerate(ims):
            x, y = (i % cols) * cw + 10, (i // cols) * (th + 30)
            d.text((x, y + 4), label, fill=(0, 0, 0))
            sheet.paste(im, (x, y + 22))
        sheet.save(OUT / f"sheet_{f.stem}.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
