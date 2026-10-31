"""Figures for the README appendix: one design through the whole pipeline, drawn
from the engine's own data (roles, groups, photo maps, candidates, scores, OCR).

usage (repo root):
  CHAMELEON_BLEED=1 CHAMELEON_ASSETS_DIR=crello_val/assets CHAMELEON_FONTS_PICKLE=crello_val/fonts_subset.pickle \
      python tools/pipeline_viz.py crello_val 3 600 200 docs/examples/pipeline
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import resize as R  # noqa: E402
from engine import saliency, vision  # noqa: E402
from tools.crello_compare import render, renderer_with  # noqa: E402

COL = {"headline": "#ff2d55", "subhead": "#ff9500", "body": "#ffcc00", "cta": "#34c759", "legal": "#8e8e93",
       "logo": "#5856d6", "hero": "#00c7be", "bg": "#007aff", "deco": "#af52de", "group": "#ffffff"}
Rm = sys.modules["engine.resize"]


def font(n):
    return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", n)


def label(d, x, y, text, col, f):
    b = d.textbbox((x, y), text, font=f)
    d.rectangle((b[0] - 3, b[1] - 2, b[2] + 3, b[3] + 2), fill=col)
    d.text((x, y), text, fill="#000000" if col not in ("#5856d6", "#007aff", "#000000") else "#ffffff", font=f)


def bbox(u):
    return u.bbox() if callable(u.bbox) else u.bbox


def boxes(base, items, scale):
    """items: (box, text, colour) in design units."""
    im = base.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    f = font(max(15, im.width // 45))
    for b, t, c in items:
        q = [v * scale for v in b]
        d.rectangle(q, outline=c, width=max(3, im.width // 300))
        label(d, max(q[0], 0) + 5, max(q[1], 0) + 5, t, c, f)
    return im


def heat(photo, m, tint):
    m = np.asarray(Image.fromarray((np.clip(m / max(m.max(), 1e-6), 0, 1) * 255).astype("uint8")).resize(photo.size, Image.BICUBIC)) / 255.0
    a = np.asarray(photo.convert("L").convert("RGB")).astype(float) * 0.45
    return Image.fromarray((a * (1 - m[..., None]) + np.array(tint) * m[..., None]).astype("uint8"))


def titled(im, t):
    o = Image.new("RGB", (im.width, im.height + 40), "white")
    ImageDraw.Draw(o).text((4, 8), t, fill="black", font=font(22))
    o.paste(im, (0, 40))
    return o


def row(ims, gap=24):
    o = Image.new("RGB", (sum(i.width for i in ims) + gap * (len(ims) - 1), max(i.height for i in ims)), "white")
    x = 0
    for i in ims:
        o.paste(i, (x, 0))
        x += i.width + gap
    return o


def main(root, p, w, h, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    src = json.loads((root / "designs" / json.loads((root / "index.json").read_text())[p][0]["file"]).read_text())
    renderer_with(str(root / "fonts_subset.pickle"), str(root / "assets"))
    assets = str(root / "assets")
    tmp = out / "_t.png"

    seen, last = [], {}
    score0, key0 = Rm.score, Rm._key

    def score(ctx, P, model):
        last["P"] = copy.deepcopy(P)
        return score0(ctx, P, model)

    def key(br, fade):
        k = key0(br, fade)
        seen.append((k, last["P"], dict(br)))
        return k
    Rm.score, Rm._key = score, key
    output = Rm.resize(src, {"width": w, "height": h})
    meta, _, chosen = Rm.resize.last_choice
    model = Rm.resize.last_model

    S = render(src, assets, tmp).convert("RGB")
    sc = 900 / S.height
    S = S.resize((int(S.width * sc), 900), Image.LANCZOS)
    k = S.width / src["canvas"]["width"]

    # 1 element types, as given
    typ = {"text": "#ff2d55", "image": "#00c7be", "shape": "#ffcc00"}
    a = boxes(S, [((e["x"], e["y"], e["x"] + e["width"], e["y"] + e["height"]), e["type"], typ[e["type"]])
                  for e in src["elements"] if e["width"] * e["height"] > 0], k)
    # 2 roles and groups, as inferred
    items = []
    for u in model["units"]:
        c = COL.get(u.role, "#ffffff")
        items.append((bbox(u), f"{u.role} ({u.kind})", c))
    b = boxes(S, items, k)
    row([titled(a, "Input: unlabelled boxes"), titled(b, "Inferred roles and groups")]).save(out / "1_analyze.png")

    # 3 photo maps
    pic = max((e for e in src["elements"] if e["type"] == "image"), key=lambda e: e["width"] * e["height"])
    info = saliency.analyze(pic["asset_id"])
    photo = Image.open(saliency.asset_file(pic["asset_id"])).convert("RGB")
    photo = photo.resize((int(photo.width * 700 / photo.height), 700), Image.LANCZOS)
    f1 = heat(photo, info["sal"], (255, 60, 40))
    d = ImageDraw.Draw(f1)
    sb = info["sal_box"]
    d.rectangle((sb[0] * f1.width, sb[1] * f1.height, sb[2] * f1.width, sb[3] * f1.height), outline="#ffffff", width=4)
    cx, cy = info["centroid"]
    d.ellipse((cx * f1.width - 10, cy * f1.height - 10, cx * f1.width + 10, cy * f1.height + 10), fill="#ffffff")
    f2 = heat(photo, info["busy"], (60, 200, 255))
    row([titled(photo, "Photo"), titled(f1, "Saliency, subject box, centroid"), titled(f2, "Busyness")]).save(out / "2_photo.png")

    # 4 candidates, best per layout
    by = {}
    for kk, P, br in seen:
        t = P.meta["template"]
        if t not in by or kk > by[t][0]:
            by[t] = (kk, P, br)
    ranked = sorted(by.values(), key=lambda c: c[0], reverse=True)
    win = meta["template"].replace("loose-", "")
    cells = []
    for i, (kk, P, br) in enumerate(ranked[:12]):
        im = render(Rm._build(src, {"width": w, "height": h}, P), assets, tmp).convert("RGB")
        im = im.resize((480, int(im.height * 480 / im.width)), Image.LANCZOS)
        t = P.meta["template"]
        c = Image.new("RGB", (500, im.height + 56), "#34c759" if t == win else "white")
        c.paste(im, (10, 46))
        ImageDraw.Draw(c).text((10, 12), f"{i + 1}. {t}  {sum(br.values()):.1f}", fill="black", font=font(21))
        cells.append(c)
    rows = [row(cells[i:i + 3], 12) for i in range(0, len(cells), 3)]
    o = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 12 for r in rows)), "white")
    y = 0
    for r in rows:
        o.paste(r, (0, y))
        y += r.height + 12
    o.save(out / "3_candidates.png")

    # 5 the winner's score terms
    terms = sorted(((a, v) for a, v in chosen.items() if abs(v) > 0.01), key=lambda t: t[1])
    H = 40 + 38 * len(terms)
    o = Image.new("RGB", (1100, H), "white")
    d = ImageDraw.Draw(o)
    m = max(abs(v) for _, v in terms)
    z = 720
    for i, (a_, v) in enumerate(terms):
        y = 20 + i * 38
        d.text((10, y), a_, fill="black", font=font(21))
        d.rectangle((z, y, z + v / m * 340, y + 24) if v >= 0 else (z + v / m * 340, y, z, y + 24), fill="#34c759" if v > 0 else "#ff3b30")
        d.text((270, y), f"{v:+.1f}", fill="black", font=font(21))
    o.save(out / "4_score.png")

    # 6 final, with the overhang and what OCR read
    Rd = render(output, assets, tmp).convert("RGB")
    W, Hh = output["canvas"]["width"], output["canvas"]["height"]
    q = 1200 / W
    Fin = Rd.resize((1200, int(Hh * q)), Image.LANCZOS)
    ov = [e for e in output["elements"] if e["x"] < -1 or e["y"] < -1 or e["x"] + e["width"] > W + 1 or e["y"] + e["height"] > Hh + 1]
    x0 = min([0] + [e["x"] for e in ov]); y0 = min([0] + [e["y"] for e in ov])
    x1 = max([W] + [e["x"] + e["width"] for e in ov]); y1 = max([Hh] + [e["y"] + e["height"] for e in ov])
    sm = min(1200 / (x1 - x0), 700 / (y1 - y0))
    C = Image.new("RGB", (int((x1 - x0) * sm) + 2, int((y1 - y0) * sm) + 2), "#eeeeee")
    d = ImageDraw.Draw(C)
    for e in ov:
        if e["type"] == "image":
            im = Image.open(saliency.asset_file(e["asset_id"])).convert("RGB").resize((int(e["width"] * sm), int(e["height"] * sm)))
            C.paste(Image.blend(im, Image.new("RGB", im.size, "white"), 0.55), (int((e["x"] - x0) * sm), int((e["y"] - y0) * sm)))
    C.paste(Rd.resize((int(W * sm), int(Hh * sm)), Image.LANCZOS), (int(-x0 * sm), int(-y0 * sm)))
    d.rectangle((-x0 * sm, -y0 * sm, (W - x0) * sm, (Hh - y0) * sm), outline="#ff2d55", width=4)
    C.save(out / "5_bleed.png")
    res, _ = vision._ocr()(np.asarray(Fin)[:, :, ::-1].copy())
    d = ImageDraw.Draw(Fin)
    for r in res or []:
        d.polygon([tuple(p) for p in r[0]], outline="#34c759", width=3)
        label(d, r[0][1][0] + 6, r[0][0][1], r[1], "#34c759", font(17))
    Fin.save(out / "6_final.png")
    tmp.unlink()
    print(meta["template"], len(seen), "candidates", len(by), "layouts")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
