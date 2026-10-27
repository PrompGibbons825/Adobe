"""Does CLIP pick the designer's crop? For every pair whose source has a big
photo and whose shape changes enough that the photo must be cropped, compare
three windows on the photo along the axis that gets cropped:
  designer  what the designer's own resize shows of the photo
  ours      what our engine's output shows
  clip      the window CLIP finds most like the copy
Prints per pair and a summary; --sheet OUT.png renders the cases.

usage (repo root):
  python tools/clip_crop_test.py ROOT:OURS_DIR [ROOT:OURS_DIR ...] [--sheet OUT.png]
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import vision  # noqa: E402
from tools.scoreboard import look, look_sim  # noqa: E402

STEPS = 11


def window(e, W, H):
    """Visible part of an image element on a W x H canvas, as fractions of the image."""
    w, h = max(1e-6, e["width"]), max(1e-6, e["height"])
    return (max(0.0, -e["x"] / w), max(0.0, -e["y"] / h), min(1.0, (W - e["x"]) / w), min(1.0, (H - e["y"]) / h))


def photo(root, d):
    W, H = d["canvas"]["width"], d["canvas"]["height"]
    best = None
    for e in d["elements"]:
        if e["type"] != "image":
            continue
        a = max(0.0, min(e["x"] + e["width"], W) - max(e["x"], 0)) * max(0.0, min(e["y"] + e["height"], H) - max(e["y"], 0))
        if a >= 0.4 * W * H and (best is None or a > best[0]):
            best = (a, e)
    return best[1] if best else None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sheet = sys.argv[sys.argv.index("--sheet") + 1] if "--sheet" in sys.argv else None
    if sheet in args:
        args.remove(sheet)
    rows, tiles = [], []
    for spec in args:
        root, ours_dir = spec.split(":")
        idx = json.load(open(f"{root}/index.json"))
        for p, vs in enumerate(idx):
            src = json.load(open(f"{root}/designs/{vs[0]['file']}"))
            sp = photo(root, src)
            if sp is None:
                continue
            texts = sorted((e for e in src["elements"] if e["type"] == "text"), key=lambda e: -e["font_size"])
            if not texts:
                continue
            head = " ".join(" ".join(e["content"].split()) for e in texts[:2])
            try:
                im = Image.open(f"{root}/assets/{sp['asset_id']}").convert("RGB")
            except Exception:
                continue
            for v in vs[1:]:
                key = f"{root}_p{p}_{v['w']}x{v['h']}"
                ref = json.load(open(f"{root}/designs/{v['file']}"))
                # the designer's copy of this photo (files are re-exported per variant)
                lk = look(root, sp["asset_id"])
                cands = [e for e in ref["elements"] if e["type"] == "image"]
                dp = max(cands, key=lambda e: look_sim(lk, look(root, e.get("asset_id"))), default=None)
                if dp is None or look_sim(lk, look(root, dp.get("asset_id"))) < 0.75:
                    continue
                of = Path(ours_dir) / f"{key}.json"
                if not of.is_file():
                    continue
                ours = json.load(open(of))
                op = next((e for e in ours["elements"] if e["id"] == sp["id"]), None)
                if op is None or op.get("fit") == "contain":
                    continue
                dw = window(dp, v["w"], v["h"])
                ow = window(op, v["w"], v["h"])
                # the axis the designer cropped
                k = 1 if (dw[3] - dw[1]) < (dw[2] - dw[0]) else 0
                span = dw[2 + k] - dw[k]
                if span > 0.85:
                    continue               # barely cropped: nothing to choose
                starts = [i * (1 - span) / (STEPS - 1) for i in range(STEPS)]
                boxes = [((0, s, 1, s + span) if k == 1 else (s, 0, s + span, 1)) for s in starts]
                r = vision.relevance(im, boxes, head)
                if r is None:
                    print("CLIP unavailable")
                    return
                best = starts[int(np.argmax(r))]
                c = lambda s: s + span / 2
                d_c = abs(c(best) - (dw[k] + dw[2 + k]) / 2)
                d_o = abs((ow[k] + ow[2 + k]) / 2 - (dw[k] + dw[2 + k]) / 2)
                spread = max(r) - min(r)
                rows.append((key, head[:40], spread, d_o, d_c))
                tiles.append((key, im, k, dw, ow, (boxes[int(np.argmax(r))]), spread, d_o, d_c, head))
    print(f"{'pair':<34}{'spread':>8}{'ours->dsg':>11}{'clip->dsg':>11}  copy")
    for key, head, sp_, do, dc in sorted(rows, key=lambda r: -r[2]):
        print(f"{key:<34}{sp_:8.3f}{do:11.3f}{dc:11.3f}  {head}")
    for thr in (0.0, 0.03, 0.05):
        g = [r for r in rows if r[2] >= thr]
        if g:
            print(f"spread >= {thr:.2f}: {len(g)} pairs, mean offset ours {np.mean([r[3] for r in g]):.3f}"
                  f"  clip {np.mean([r[4] for r in g]):.3f};  clip closer on {sum(r[4] < r[3] - 0.02 for r in g)},"
                  f" ours closer on {sum(r[3] < r[4] - 0.02 for r in g)}")
    if sheet and tiles:
        tiles.sort(key=lambda t: -t[6])
        F = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        rowsimg = []
        for key, im, k, dw, ow, cb, spread, do, dc, head in tiles[:16]:
            th = im.copy()
            th.thumbnail((360, 360))
            d = ImageDraw.Draw(th)
            for box, col in ((dw, (40, 200, 40)), (ow, (40, 120, 255)), (cb, (255, 60, 60))):
                d.rectangle([box[0] * th.width, box[1] * th.height, box[2] * th.width - 1, box[3] * th.height - 1],
                            outline=col, width=3)
            r = Image.new("RGB", (900, th.height + 40), "white")
            r.paste(th, (0, 34))
            dd = ImageDraw.Draw(r)
            dd.text((0, 4), f"{key}  spread {spread:.3f}  ours off {do:.2f}  clip off {dc:.2f}", fill=(0, 0, 0), font=F)
            dd.text((380, 40), "green = designer\nblue = ours\nred = CLIP", fill=(60, 60, 60), font=F)
            dd.text((380, 120), head[:80], fill=(60, 60, 60), font=F)
            rowsimg.append(r)
        S = Image.new("RGB", (900, sum(x.height for x in rowsimg)), "white")
        y = 0
        for x in rowsimg:
            S.paste(x, (0, y))
            y += x.height
        S.save(sheet)
        print(sheet)


if __name__ == "__main__":
    main()
