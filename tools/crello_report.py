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
from engine.debug import source_placed  # noqa: E402
from engine import saliency  # noqa: E402

PALETTE = [(230, 57, 70), (29, 120, 220), (42, 157, 92), (245, 160, 20), (142, 68, 173),
           (0, 160, 170), (220, 80, 160), (120, 120, 120), (180, 110, 40), (90, 90, 220)]
PAGE_W = 2480  # A4-ish at 300dpi, landscape-free portrait page
PAGE_H = 3508
FONT_PATHS = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]


def font(size, bold=False):
    for pair in FONT_PATHS:
        try:
            return ImageFont.truetype(pair[1 if bold else 0], size)
        except Exception:
            continue
    return ImageFont.load_default()


def unit_label(u, modes):
    if u.role == "border":
        return "border · hugs canvas edge"
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


def _role_of(model, eid):
    for u in model["units"]:
        if eid in u.ids:
            r = model.get("troles", {}).get(eid) if u.kind == "frame" else None
            return r or u.role
    return "?"


def weight_lines(P, model, design):
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    items = getattr(P, "weight_items", None) or []
    tot = sum(i[3] for i in items) or 1.0
    agg = {}
    for eid, cx, cy, w, kind in items:
        name = "photo subject" if kind == "subject" else f"{_role_of(model, eid)} ({'text' if kind == 'text' else 'image'})"
        agg[name] = agg.get(name, 0.0) + w
    out = [f"{n:<24} {w / tot * 100:4.0f}%" for n, w in sorted(agg.items(), key=lambda kv: -kv[1])[:6]]
    cm = getattr(P, "cm", None)
    if cm:
        dx, dy = (cm[0] / W - 0.5) * 100, (cm[1] / H - 0.5) * 100
        out.append(f"mass centre: {abs(dx):.0f}% {'left' if dx < 0 else 'right'}, {abs(dy):.0f}% {'up' if dy < 0 else 'down'}")
    return out


def _arrow(d, a, b, col, w=4):
    import math as _m
    d.line([a, b], fill=col, width=w)
    ang = _m.atan2(b[1] - a[1], b[0] - a[0])
    for s_ in (-0.5, 0.5):
        d.line([b, (b[0] - 18 * _m.cos(ang + s_), b[1] - 18 * _m.sin(ang + s_))], fill=col, width=w)


def diag_panel(render_img, design, P, model, assets, pw, ph, ghost):
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    bi = getattr(P, "bg_info", None)
    x0, y0, x1, y1 = 0.0, 0.0, float(W), float(H)
    if ghost and bi:
        bx, by, bw, bh = bi["box"]
        x0, y0, x1, y1 = min(x0, bx), min(y0, by), max(x1, bx + bw), max(y1, by + bh)
    sc = min(pw / (x1 - x0), ph / (y1 - y0))
    PW, PH = max(1, int((x1 - x0) * sc)), max(1, int((y1 - y0) * sc))
    panel = Image.new("RGB", (PW, PH), (236, 236, 236))

    def T(px, py):
        return ((px - x0) * sc, (py - y0) * sc)
    if ghost and bi and bi.get("asset"):
        try:
            ap = Path(assets) / bi["asset"]
            g = Image.open(ap).convert("RGB")
            bx, by, bw, bh = bi["box"]
            g = g.resize((max(1, int(bw * sc)), max(1, int(bh * sc))))
            g = Image.blend(g, Image.new("RGB", g.size, (236, 236, 236)), 0.6)
            panel.paste(g, tuple(int(v) for v in T(bx, by)))
        except Exception:
            pass
    cx0, cy0 = T(0, 0)
    canvas = render_img.convert("RGB").resize((max(1, int(W * sc)), max(1, int(H * sc))))
    panel.paste(canvas, (int(cx0), int(cy0)))
    ov = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    if bi and bi.get("concentrated") and bi.get("asset"):
        info = saliency.analyze(bi["asset"])
        if info is not None:
            bx, by, bw, bh = bi["box"]
            hm = Image.fromarray((info["sal"] * 255).astype("uint8")).resize((max(1, int(bw * sc)), max(1, int(bh * sc))))
            red = Image.new("RGBA", hm.size, (230, 40, 40, 0))
            red.putalpha(hm.point(lambda v: 0 if v < 140 else int((v - 140) * 1.4)))
            ov.alpha_composite(red, tuple(max(0, int(v)) for v in T(bx, by)) if bx >= x0 else (0, 0))
            sb = bi["sal_box"]
            d.rectangle([*T(sb[0], sb[1]), *T(sb[2], sb[3])], outline=(230, 40, 40, 255), width=3)
            fx, fy = T(*bi["focal"])
            d.line([(fx - 16, fy), (fx + 16, fy)], fill=(230, 40, 40, 255), width=4)
            d.line([(fx, fy - 16), (fx, fy + 16)], fill=(230, 40, 40, 255), width=4)
            if ghost:
                gx, gy = T(*bi["focal_centered"])
                if abs(gx - fx) + abs(gy - fy) > 6:
                    d.ellipse([gx - 10, gy - 10, gx + 10, gy + 10], outline=(230, 40, 40, 255), width=3)
                    _arrow(d, (gx, gy), (fx, fy), (230, 40, 40, 255))
    for k in (1, 2):  # thirds
        xa, _ = T(W * k / 3, 0)
        _, ya = T(0, H * k / 3)
        d.line([(xa, cy0), (xa, cy0 + H * sc)], fill=(120, 120, 120, 150), width=2)
        d.line([(cx0, ya), (cx0 + W * sc, ya)], fill=(120, 120, 120, 150), width=2)
    cols = {"text": (29, 120, 220), "image": (245, 150, 20), "shape": (245, 150, 20), "subject": (230, 40, 40)}
    A = W * H * sc * sc
    for eid, ex, ey, w, kind in getattr(P, "weight_items", None) or []:
        r = max(3.0, (w * A * 0.35 / 3.1416) ** 0.5)
        px, py = T(ex, ey)
        col = cols.get(kind, (120, 120, 120))
        d.ellipse([px - r, py - r, px + r, py + r], fill=col + (70,), outline=col + (230,), width=2)
    cx, cy = T(W / 2, H / 2)
    d.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], outline=(110, 110, 110, 255), width=3)
    cm = getattr(P, "cm", None)
    if cm:
        mx, my = T(*cm)
        for wdt, colr in ((9, (255, 255, 255, 255)), (5, (0, 0, 0, 255))):
            d.line([(mx - 20, my), (mx + 20, my)], fill=colr, width=wdt)
            d.line([(mx, my - 20), (mx, my + 20)], fill=colr, width=wdt)
    d.rectangle([cx0 - 2, cy0 - 2, cx0 + W * sc + 1, cy0 + H * sc + 1], outline=(30, 30, 30, 255), width=3)
    return Image.alpha_composite(panel.convert("RGBA"), ov).convert("RGB")


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
        ourP = getattr(resize, "last_P", None)
        model = analyze(src)
        (out / f"{p}_ours.json").write_text(json.dumps(ours, indent=1))
        cases.append({
            "name": p, "title": src_v.get("title", ""), "src": src, "ours": ours, "meta": meta,
            "src_v": src_v, "ref_v": ref_v,
            "src_png": render(src, assets, out / f"{p}_source.png"),
            "ref_png": render(ref, assets, out / f"{p}_designer.png"),
            "our_png": render(ours, assets, out / f"{p}_ours.png"),
            "units": [u for u in model["units"] if u.role not in ("bg", "frame_bg", "deco", "missing", "rule")],
            "model": model, "srcP": source_placed(model), "ourP": ourP,
        })
        print(p, meta["template"], meta.get("frame_modes"))

    # ---------------- per-pair panels (for the review gallery) ----------------
    for c in cases:
        modes = c["meta"].get("frame_modes", {})
        for tag, png, design in (("src", c["src_png"], c["src"]), ("ours", c["our_png"], c["ours"])):
            t, _ = fit(png, 1000, 700)
            sc = t.width / design["canvas"]["width"]
            annotate(t, design, c["units"], modes, sc).save(out / f"{c['name']}_groups_{tag}.png")
        legend = [f"G{n + 1}  {unit_label(u, modes)}" for n, u in enumerate(c["units"])]
        panels = {}
        for tag, png, design, P, ghost in (("src", c["src_png"], c["src"], c["srcP"], False),
                                           ("ours", c["our_png"], c["ours"], c["ourP"], True)):
            if P is None:
                continue
            diag_panel(png, design, P, c["model"], assets, 1000, 700, ghost=ghost).save(out / f"{c['name']}_weight_{tag}.png")
            panels[tag] = weight_lines(P, c["model"], design)
        (out / f"{c['name']}_meta.json").write_text(json.dumps({
            "title": c["title"], "src": [c["src_v"]["w"], c["src_v"]["h"]], "dst": [c["ref_v"]["w"], c["ref_v"]["h"]],
            "template": c["meta"]["template"], "legend": legend,
            "palette": [PALETTE[n % len(PALETTE)] for n in range(len(c["units"]))], "weights": panels}))

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
            d.rectangle([int(x) - 2, y + 34, int(x) + t.width + 1, y + 36 + t.height + 1], outline=(40, 40, 40), width=3)
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
        d.rectangle([int(x) - 2, y + 34, int(x) + t.width + 1, y + 36 + t.height + 1], outline=(40, 40, 40), width=3)
        d.text((int(x), y), "Source, grouped", fill=(110, 110, 110), font=font(26))
        x += widths[0] + gap
        t, s = fit(c["our_png"], int(widths[1]), row_h - 40)
        sc = t.width / c["ours"]["canvas"]["width"]
        pg.paste(annotate(t, c["ours"], c["units"], modes, sc), (int(x), y + 36))
        d.rectangle([int(x) - 2, y + 34, int(x) + t.width + 1, y + 36 + t.height + 1], outline=(40, 40, 40), width=3)
        d.text((int(x), y), "Ours, groups repositioned", fill=(110, 110, 110), font=font(26))
        x += widths[1] + gap
        ly = y + 40
        for n, u in enumerate(c["units"]):
            col = PALETTE[n % len(PALETTE)]
            d.rectangle([int(x), ly + 6, int(x) + 26, ly + 32], fill=col)
            import textwrap
            lines = textwrap.wrap(f"G{n + 1}  {unit_label(u, modes)}", 34)
            for j, ln in enumerate(lines):
                d.text((int(x) + 38 + (0 if j == 0 else 44), ly + 2), ln, fill=(40, 40, 40), font=font(26))
                ly += 34
            ly += 6
            if ly > y + row_h - 20:
                break
        y += row_h + 25
    pages.append(pg)
    # ---------------- page 3+: visual weight and crop ----------------
    per_page = 4
    for start in range(0, len(cases), per_page):
        chunk = cases[start:start + per_page]
        pg = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        d = ImageDraw.Draw(pg)
        d.text((margin, margin - 10), "Visual weight, the photo's subject, and how the background was cropped",
               fill=(20, 20, 20), font=font(48, True))
        key = ("Red heat = saliency (what the eye goes to); red box = subject area; red + = subject centre.  "
               "Circles = visual weight (area = weight): blue text, orange images, red the photo subject.  "
               "Black + = centre of visual mass, grey ring = canvas centre.  Faded area = background cropped away; "
               "arrow = where the subject moved vs. a plain centred crop.  Grey lines = thirds.")
        import textwrap
        ky = margin + 55
        for ln in textwrap.wrap(key, 150):
            d.text((margin, ky), ln, fill=(90, 90, 90), font=font(26))
            ky += 34
        y = ky + 30
        rh = (PAGE_H - y - margin) // len(chunk) - 60
        for c in chunk:
            d.text((margin, y), f"{c['name']} · {c['title']}", fill=(30, 30, 30), font=font(30, True))
            y += 45
            avail = PAGE_W - 2 * margin - 2 * gap
            widths = [avail * 0.27, avail * 0.45, avail * 0.28]
            x = margin
            sp = diag_panel(c["src_png"], c["src"], c["srcP"], c["model"], assets, int(widths[0]), rh, ghost=False)
            pg.paste(sp, (int(x), y + 36))
            d.text((int(x), y), "Source", fill=(110, 110, 110), font=font(26))
            x += widths[0] + gap
            if c["ourP"] is not None:
                op = diag_panel(c["our_png"], c["ours"], c["ourP"], c["model"], assets, int(widths[1]), rh, ghost=True)
                pg.paste(op, (int(x), y + 36))
            d.text((int(x), y), "Ours (faded = cropped-away background)", fill=(110, 110, 110), font=font(26))
            x += widths[1] + gap
            ly = y + 40
            for label, P, dz in (("Source", c["srcP"], c["src"]), ("Ours", c["ourP"], c["ours"])):
                if P is None:
                    continue
                d.text((int(x), ly), label, fill=(30, 30, 30), font=font(26, True))
                ly += 34
                for line in weight_lines(P, c["model"], dz):
                    d.text((int(x) + 10, ly), line, fill=(50, 50, 50), font=font(23))
                    ly += 29
                ly += 10
            y += rh + 25
        pages.append(pg)

    pdf = out / "chameleon_crello_report.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=300)
    for i, p in enumerate(pages):
        p.save(out / f"page{i + 1}.png")
    print("wrote", pdf)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["p0", "p1", "p2", "p3"])
