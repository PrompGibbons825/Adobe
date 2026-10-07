"""Two-page PDF report on Crello designer pairs.

Page 1: source | designer's resize | ours
Page 2: how each design was grouped (source) and where each group went (ours)

usage (repo root):
  CHAMELEON_ASSETS_DIR=crello/assets CHAMELEON_FONTS_PICKLE=crello/fonts_subset.pickle \
      python tools/crello_report.py crello p0 p1 p2 p3
"""

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.crello_compare import render, renderer_with  # noqa: E402
from engine.analyze import analyze  # noqa: E402
from engine.resize import resize  # noqa: E402

PALETTE = [(230, 57, 70), (29, 120, 220), (42, 157, 92), (245, 160, 20), (142, 68, 173),
           (0, 160, 170), (220, 80, 160), (120, 120, 120), (180, 110, 40), (90, 90, 220)]
PAGE_W = 2480  # A4-ish at 300dpi, landscape-free portrait page
PAGE_H = 3508
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_PATHS[1 if bold else 0], size)
    except Exception:
        return ImageFont.load_default()


def unit_label(u, modes):
    if u.kind == "frame":
        m = modes.get(u.uid)
        from collections import Counter
        cnt = Counter(ch.role for ch in u.children)
        parts = ", ".join(f"{r}×{n}" if n > 1 else r for r, n in cnt.items())
        return f"{u.role} · frame" + (f" → {m}" if m else "") + f"  [{parts}]"
    if u.kind in ("group", "button", "badges", "ribbon"):
        return f"{u.role} · {u.kind} ({len(u.ids)})"
    return f"{u.role}"


def boxes_for(ids, design, skip_hidden=True):
    els = {e["id"]: e for e in design["elements"]}
    bs = []
    for i in ids:
        e = els.get(i)
        if e is None or (skip_hidden and e.get("opacity", 1) == 0):
            continue
        a = math.radians(e.get("rotation", 0) or 0)
        c, s = abs(math.cos(a)), abs(math.sin(a))
        w, h = e["width"], e["height"]
        bw, bh = w * c + h * s, w * s + h * c
        cx, cy = e["x"] + w / 2, e["y"] + h / 2
        bs.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    if not bs:
        return None
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def annotate(img, design, units, modes, scale):
    im = img.copy().convert("RGB")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    f = font(22, True)
    for n, u in enumerate(units):
        col = PALETTE[n % len(PALETTE)]
        b = boxes_for(u.ids, design)
        if b is None:
            continue
        b = [v * scale for v in b]
        b = [max(0, b[0]), max(0, b[1]), min(im.width - 1, b[2]), min(im.height - 1, b[3])]
        d.rectangle(b, fill=col + (40,), outline=col + (255,), width=4)
        for ch in getattr(u, "children", []):
            cb = boxes_for(ch.ids, design)
            if cb:
                cb = [v * scale for v in cb]
                d.rectangle(cb, outline=col + (200,), width=2)
        tag = f"G{n + 1}"
        tw = d.textlength(tag, font=f)
        d.rectangle([b[0], b[1], b[0] + tw + 12, b[1] + 30], fill=col + (255,))
        d.text((b[0] + 6, b[1] + 2), tag, fill=(255, 255, 255), font=f)
    return Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")


def fit(im, w, h):
    s = min(w / im.width, h / im.height)
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS), s


def main(root, picks):
    root = Path(root)
    assets = root / "assets"
    renderer_with(str(root / "fonts_subset.pickle"), assets)
    index = json.loads((root / "index.json").read_text())
    out = root / "report"
    out.mkdir(exist_ok=True)
    cases = []
    for p in picks:
        k = int(p.lstrip("p"))
        variants = index[k]
        src_v, ref_v = variants[0], variants[1]
        src = json.loads((root / "designs" / src_v["file"]).read_text())
        ref = json.loads((root / "designs" / ref_v["file"]).read_text())
        ours = resize(src, {"width": ref_v["w"], "height": ref_v["h"]})
        meta = resize.last_choice[0]
        (out / f"{p}_ours.json").write_text(json.dumps(ours, indent=1))
        cases.append({
            "name": p, "title": src_v.get("title", ""), "src": src, "ours": ours, "meta": meta,
            "src_v": src_v, "ref_v": ref_v,
            "src_png": render(src, assets, out / f"{p}_source.png"),
            "ref_png": render(ref, assets, out / f"{p}_designer.png"),
            "our_png": render(ours, assets, out / f"{p}_ours.png"),
            "units": [u for u in analyze(src)["units"] if u.role not in ("bg", "frame_bg", "deco", "missing")],
        })
        print(p, meta["template"], meta.get("frame_modes"))

    # ---------------- page 1 ----------------
    pages = []
    margin, gap = 90, 40
    row_h = (PAGE_H - 2 * margin - 160) // len(cases) - 70
    pg = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(pg)
    d.text((margin, margin - 10), "Chameleon engine vs. the designer, on real Crello designs", fill=(20, 20, 20), font=font(54, True))
    d.text((margin, margin + 62), "Each row: the source design, the designer's own resize, and our engine's resize to the same size.",
           fill=(90, 90, 90), font=font(30))
    y = margin + 150
    for c in cases:
        d.text((margin, y), f"{c['name']} · {c['title']}   {c['src_v']['w']}×{c['src_v']['h']} → {c['ref_v']['w']}×{c['ref_v']['h']}",
               fill=(30, 30, 30), font=font(30, True))
        y += 45
        avail = PAGE_W - 2 * margin - 2 * gap
        widths = [avail * 0.22, avail * 0.39, avail * 0.39]
        x = margin
        for im, lab, w in zip([c["src_png"], c["ref_png"], c["our_png"]],
                              ["Source", "Designer", f"Ours · {c['meta']['template']}"], widths):
            t, _ = fit(im, int(w), row_h - 40)
            pg.paste(t, (int(x), y + 36))
            d.rectangle([int(x), y + 36, int(x) + t.width - 1, y + 36 + t.height - 1], outline=(200, 200, 200), width=2)
            d.text((int(x), y), lab, fill=(110, 110, 110), font=font(26))
            x += w + gap
        y += row_h + 25
    pages.append(pg)

    # ---------------- page 2 ----------------
    pg = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(pg)
    d.text((margin, margin - 10), "How each design was grouped, and where the groups went", fill=(20, 20, 20), font=font(54, True))
    d.text((margin, margin + 62), "Coloured boxes are groups found from the raw JSON (thin boxes: contents of a frame). "
           "Same colour = same group before and after.", fill=(90, 90, 90), font=font(28))
    y = margin + 150
    for c in cases:
        modes = c["meta"].get("frame_modes", {})
        d.text((margin, y), f"{c['name']} · {c['title']}", fill=(30, 30, 30), font=font(30, True))
        y += 45
        avail = PAGE_W - 2 * margin - 2 * gap
        widths = [avail * 0.25, avail * 0.45, avail * 0.30]
        x = margin
        t, s = fit(c["src_png"], int(widths[0]), row_h - 40)
        sc = t.width / c["src"]["canvas"]["width"]
        pg.paste(annotate(t, c["src"], c["units"], modes, sc), (int(x), y + 36))
        d.text((int(x), y), "Source, grouped", fill=(110, 110, 110), font=font(26))
        x += widths[0] + gap
        t, s = fit(c["our_png"], int(widths[1]), row_h - 40)
        sc = t.width / c["ours"]["canvas"]["width"]
        pg.paste(annotate(t, c["ours"], c["units"], modes, sc), (int(x), y + 36))
        d.text((int(x), y), "Ours, groups repositioned", fill=(110, 110, 110), font=font(26))
        x += widths[1] + gap
        ly = y + 40
        for n, u in enumerate(c["units"]):
            col = PALETTE[n % len(PALETTE)]
            d.rectangle([int(x), ly + 6, int(x) + 26, ly + 32], fill=col)
            d.text((int(x) + 38, ly + 2), f"G{n + 1}  {unit_label(u, modes)}", fill=(40, 40, 40), font=font(26))
            ly += 40
            if ly > y + row_h - 20:
                break
        y += row_h + 25
    pages.append(pg)
    pdf = out / "chameleon_crello_report.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=300)
    for i, p in enumerate(pages):
        p.save(out / f"page{i + 1}.png")
    print("wrote", pdf)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["p0", "p1", "p2", "p3"])
