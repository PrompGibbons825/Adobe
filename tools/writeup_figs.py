"""Figures for the algorithm write-up, one per step, on real designs.

usage (repo root):
  CHAMELEON_ASSETS_DIR=crello/assets CHAMELEON_FONTS_PICKLE=crello/fonts_subset.pickle python tools/writeup_figs.py crello
  python tools/writeup_figs.py solstice          # step 4 (text roles) on the Solstice sample
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.crello_report import font, annotate, diag_panel, PALETTE  # noqa: E402
from engine.analyze import analyze, FLOW_ROLES  # noqa: E402
from engine.debug import source_placed  # noqa: E402
from engine import saliency  # noqa: E402

OUT = Path("writeup_figs")
INK, QUIET = (25, 25, 25), (95, 95, 95)
BLUE, ORANGE, RED, GREEN, PURPLE, GRAY = (29, 120, 220), (235, 140, 10), (220, 50, 50), (30, 150, 80), (130, 70, 190), (150, 150, 150)


def fit(im, w, h):
    s = min(w / im.width, h / im.height)
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS), s


def boxes(img, scale, items):
    """items: (box x0,y0,x1,y1 in design px, colour, label or None, width, fill_alpha)."""
    im = img.convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    f = font(20, True)
    for b, col, lab, w, fa in items:
        bb = [b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale]
        d.rectangle(bb, fill=col + (fa,), outline=col + (255,), width=w)
        if lab:
            tw = d.textlength(lab, font=f)
            ty = bb[1] - 28 if bb[1] > 30 else bb[3] + 2
            d.rectangle([bb[0], ty, bb[0] + tw + 12, ty + 26], fill=col + (255,))
            d.text((bb[0] + 6, ty + 2), lab, fill=(255, 255, 255), font=f)
    return Image.alpha_composite(im, ov).convert("RGB")


def outline(pg, x, y, im):
    pg.paste(im, (x, y))
    ImageDraw.Draw(pg).rectangle([x - 2, y - 2, x + im.width + 1, y + im.height + 1], outline=(40, 40, 40), width=3)


def sheet(panels, title=None, h=520, gap=48, label_h=44):
    """panels: [(image, label)] laid out left to right at a common height."""
    ims = [(fit(im, 2000, h)[0], lab) for im, lab in panels]
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    slots = [max(i.width, int(probe.textlength(lab, font=font(24)))) for i, lab in ims]
    W = sum(slots) + gap * (len(ims) + 1)
    top = 70 if title else 20
    pg = Image.new("RGB", (W, h + label_h + top + 30), "white")
    d = ImageDraw.Draw(pg)
    if title:
        d.text((gap, 18), title, fill=INK, font=font(32, True))
    x = gap
    for (im, lab), slot in zip(ims, slots):
        d.text((x, top), lab, fill=QUIET, font=font(24))
        outline(pg, x, top + label_h, im)
        x += slot + gap
    return pg


def render(design, assets, name):
    from tools.crello_compare import render as r
    return r(design, assets, OUT / f"_{name}.png")


def bbox_of(model, ids, P=None):
    bs = []
    for i in ids:
        if P is not None and i in P.vis:
            bs.append(P.vis[i])
        elif P is None:
            e = model["els"][i]
            bs.append((e["x"], e["y"], e["x"] + e["width"], e["y"] + e["height"]))
    if not bs:
        return None
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def load(root, k, which=0):
    idx = json.loads((Path(root) / "index.json").read_text())
    v = idx[k][which]
    return json.loads((Path(root) / "designs" / v["file"]).read_text()), idx[k]


def crello(root):
    from tools.crello_compare import renderer_with
    from engine.resize import resize, _candidates, _build
    from engine.layout import Ctx
    from engine.finalize import finalize
    from engine.score import score
    assets = Path(root) / "assets"
    renderer_with(str(Path(root) / "fonts_subset.pickle"), assets)

    # ---- Step 1: stored boxes vs inked boxes (p3) ----
    src, _ = load(root, 3)
    m = analyze(src)
    P = source_placed(m)
    im = render(src, assets, "p3_src")
    t, s = fit(im, 560, 820)
    raw = [((e["x"], e["y"], e["x"] + e["width"], e["y"] + e["height"]), (240, 200, 40), None, 3, 0) for e in src["elements"]
           if not (e["width"] * e["height"] > 0.8 * src["canvas"]["width"] * src["canvas"]["height"])]
    ink = [(P.vis[i], (90, 170, 255) if m["els"][i]["type"] == "text" else ORANGE, None, 3, 0) for i in P.vis]
    sheet([(boxes(t, s, raw), "Stored boxes"), (boxes(t, s, ink), "Inked boxes")],
          ).save(OUT / "step1_ink.png")

    # ---- Step 2: background, borders (p2) ----
    src, _ = load(root, 2)
    m = analyze(src)
    im = render(src, assets, "p2_src")
    t, s = fit(im, 900, 700)
    items = []
    for u in m["units"]:
        e = u.els[0]
        b = (e["x"], e["y"], e["x"] + e["width"], e["y"] + e["height"])
        if u.role == "border":
            items.append((b, GREEN, "border", 4, 0))
    items.append(((8, 8, 120, 60), (60, 60, 60), "background photo", 0, 0))
    sheet([(boxes(t, s, items), "Background photo + two borders that echo the canvas edge")]).save(OUT / "step2_stage.png")

    # ---- Step 3: separators (p4) and corner pieces (p6) ----
    src4, _ = load(root, 4)
    m4 = analyze(src4)
    t4, s4 = fit(render(src4, assets, "p4_src"), 560, 820)
    it4 = [((m4["els"][i]["x"] - 3, m4["els"][i]["y"] - 3, m4["els"][i]["x"] + m4["els"][i]["width"] + 3,
             m4["els"][i]["y"] + m4["els"][i]["height"] + 3), RED, None, 3, 90) for i in m4["separators"]]
    src6, _ = load(root, 6)
    m6 = analyze(src6)
    t6, s6 = fit(render(src6, assets, "p6_src"), 560, 820)
    it6 = [((m6["els"][i]["x"] - 3, m6["els"][i]["y"] - 3, m6["els"][i]["x"] + m6["els"][i]["width"] + 3,
             m6["els"][i]["y"] + m6["els"][i]["height"] + 3), RED, None, 3, 90) for i in m6["separators"]]
    for u in m6["units"]:
        if u.role == "anchored":
            e = u.els[0]
            it6.append(((e["x"], e["y"], e["x"] + e["width"], e["y"] + e["height"]), PURPLE, "corner piece", 4, 30))
    sheet([(boxes(t4, s4, it4), "Separators (red): dashes and dots between letters"),
           (boxes(t6, s6, it6), "Corner pieces (purple) and separator dots (red)")]).save(OUT / "step3_small.png")

    # ---- Step 5: groups (p2 lockup, p3 frame) ----
    panels = []
    for k, lab in ((2, "One lockup: 9 pieces glued by overlap and touch"), (3, "A frame: the box + what it encloses")):
        src, _ = load(root, k)
        m = analyze(src)
        units = [u for u in m["units"] if u.role not in ("bg", "frame_bg", "border", "deco", "missing")]
        t, s = fit(render(src, assets, f"p{k}_src"), 700, 820)
        panels.append((annotate(t, src, units, {}, s), lab))
    sheet(panels).save(OUT / "step5_groups.png")

    # ---- Step 6: visual weight, hero visual + hero text (p5) ----
    src, _ = load(root, 5)
    m = analyze(src)
    P = source_placed(m)
    im = render(src, assets, "p5_src")
    dp = diag_panel(im, src, P, m, assets, 520, 860, ghost=False)
    t, s = fit(im, 520, 860)
    hero = next(u for u in m["units"] if u.role == "hero")
    head = next(u for u in m["units"] if u.role == "headline")
    hl = boxes(t, s, [(bbox_of(m, hero.ids), ORANGE, "hero visual (3 pieces)", 4, 25),
                      (bbox_of(m, head.ids, P), BLUE, "hero text", 4, 25)])
    sheet([(dp, "Visual weight: circle area = weight"), (hl, "Heaviest image unit and heaviest text unit")]).save(OUT / "step6_weight.png")

    # ---- Step 7: saliency and the subject (p1), on-subject frame (p3) ----
    src, _ = load(root, 1)
    m = analyze(src)
    P = source_placed(m)
    im = render(src, assets, "p1_src")
    P.weight_items, P.cm = [], None
    sal1 = diag_panel(im, src, P, m, assets, 480, 860, ghost=False)
    src3, _ = load(root, 3)
    m3 = analyze(src3)
    P3 = source_placed(m3)
    P3.weight_items, P3.cm = [], None
    sal3 = diag_panel(render(src3, assets, "p3_src"), src3, P3, m3, assets, 560, 860, ghost=False)
    fr = next((u for u in m3["units"] if getattr(u, "on_subject", False)), None)
    if fr is not None:
        sal3 = boxes(sal3, sal3.width / src3["canvas"]["width"], [(fr.bbox, GREEN, "sits on the subject", 4, 0)])
    sheet([(sal1, "Subject = the chair (red box, + at its centre)"), (sal3, "Subject = the heart; the frame sits on it")]).save(OUT / "step7_subject.png")

    # ---- Step 8: candidates for p3 -> 432x288 ----
    src, variants = load(root, 3)
    tgt = {"width": variants[1]["w"], "height": variants[1]["h"]}
    m = analyze(src)
    ctx = Ctx(m, tgt)
    cands = []
    for th, make in _candidates(ctx, m):
        try:
            Pc = make()
            if Pc is None or finalize(ctx, Pc, m) is None:
                continue
            sc, br = score(ctx, Pc, m)
            cands.append((sc, Pc.meta["template"], Pc, br))
        except Exception:
            continue
    cands.sort(key=lambda c: -c[0])
    seen, pick = set(), []
    for sc, name, Pc, br in cands:
        fam = name.split(" ")[0].split("-")[0] + ("-a" + name.split(" a")[-1] if " a" in name else "")
        if fam in seen and len(pick) > 0:
            continue
        seen.add(fam)
        pick.append((sc, name, Pc, br))
        if len(pick) == 4:
            break
    panels = []
    for n, (sc, name, Pc, br) in enumerate(pick):
        out = _build(src, tgt, Pc)
        img = render(out, assets, f"cand{n}")
        panels.append((img, f"{'WINNER  ' if n == 0 else ''}{name}  ·  score {sc:.1f}"))
    sheet(panels[:2], h=420).save(OUT / "step8_cands_a.png")
    sheet(panels[2:4], h=420).save(OUT / "step8_cands_b.png")
    (OUT / "step8_scores.json").write_text(json.dumps([(name, round(sc, 2), {k: round(float(v), 2) for k, v in br.items()}) for sc, name, Pc, br in pick], indent=1))

    # ---- Step 9: finishing (p1 crop by thirds, p6 corners in result) ----
    src, variants = load(root, 1)
    tgt = {"width": variants[1]["w"], "height": variants[1]["h"]}
    out = resize(src, tgt)
    P = resize.last_P
    m = analyze(src)
    crop = diag_panel(render(out, assets, "p1_out"), out, P, m, assets, 900, 700, ghost=True)
    src6, v6 = load(root, 6)
    out6 = resize(src6, {"width": v6[1]["w"], "height": v6[1]["h"]})
    P6 = resize.last_P
    m6 = analyze(src6)
    t6, s6 = fit(render(out6, assets, "p6_out"), 700, 700)
    it = [(P6.vis[u.els[0]["id"]], PURPLE, "stays in its corner", 4, 30) for u in m6["units"] if u.role == "anchored" and u.els[0]["id"] in P6.vis]
    sheet([(crop, "Photo crop: subject slid to the third opposite the text"), (boxes(t6, s6, it), "Corner pieces pinned to their corners")], h=560).save(OUT / "step9_finish.png")

    # ---- Step 10: centred vs left (p0) ----
    src, variants = load(root, 0)
    tgt = {"width": variants[1]["w"], "height": variants[1]["h"]}
    m = analyze(src)
    ctx = Ctx(m, tgt)
    best = {}
    for th, make in _candidates(ctx, m):
        try:
            Pc = make()
            if Pc is None or finalize(ctx, Pc, m) is None:
                continue
            sc, br = score(ctx, Pc, m)
        except Exception:
            continue
        name = Pc.meta["template"]
        if name in ("stack-full-center a0.5", "stack-full-left a0.5"):
            best[name] = (sc, Pc, br)
    panels = []
    for name in ("stack-full-left a0.5", "stack-full-center a0.5"):
        if name in best:
            sc, Pc, br = best[name]
            img = render(_build(src, tgt, Pc), assets, name.replace(" ", "_"))
            panels.append((img, f"{name}: busy {br['busy_bg']:.1f}, balance {br['weight_balance']:.1f}, total {sc:.1f}"))
    sheet(panels, h=300).save(OUT / "step10_choose.png")
    print("crello figures done")


def solstice():
    from renderer.converter.convert import render_design_to_png
    d = json.loads(Path("input/chameleon_sample_level3_advanced.json").read_text())["source"]
    m = analyze(d)
    render_design_to_png(d, Path("brand_kit/assets"), str(OUT / "_sol3.png"))
    t, s = fit(Image.open(OUT / "_sol3.png").convert("RGB"), 620, 900)
    cols = {"headline": BLUE, "subhead": (40, 160, 200), "legal": GRAY, "cta": RED, "badges": ORANGE,
            "logo": PURPLE, "hero": GREEN, "ribbon": (200, 60, 140)}
    items = []
    for u in m["units"]:
        if u.role in cols:
            b = u.bbox
            items.append((b, cols[u.role], u.role, 4, 25))
    sheet([(boxes(t, s, items), "Roles inferred from size, position, words and grouping")]).save(OUT / "step4_roles.png")
    print("solstice figure done")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1] == "solstice":
        solstice()
    else:
        crello(sys.argv[1])
