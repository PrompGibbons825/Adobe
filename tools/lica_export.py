"""LICA layouts in CHAMELEON schema, for robustness runs on non-Crello designs.

LICA (CC BY 4.0, arXiv 2603.16098) has no designer resizes, so each design gets
synthetic target sizes instead. index.json has the same shape as crello/index.json:
the first variant is the source, every other variant a target.

Takes static layouts whose fonts are all in the Crello fonts.pickle, an even
draw across categories. SVGs and clipped GROUP shapes are rasterised with skia.

usage (repo root):
  python tools/lica_export.py OUT_DIR --data lica-data --fonts ~/crello_raw/resources/fonts.pickle [--n 36]
"""

import argparse
import csv
import json
import pickle
import random
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import skia
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STILL = (".png", ".jpg", ".jpeg", ".svg")
SKIP_CATEGORIES = {"Videos"}


def px(v, default=0.0):
    m = re.match(r"-?[\d.]+", str(v))
    return float(m.group()) if m else default


def family(c):
    return str(c.get("fontFamily", "")).split("--")[0].strip()


def is_static(layout):
    if layout.get("duration") and any("animation" in k for c in layout["components"] for k in c):
        return False
    return all(c["type"] != "IMAGE" or c.get("src", "").lower().endswith(STILL) for c in layout["components"])


def fetch(url):
    return subprocess.run(["curl", "-fsSL", "-m", "60", url], check=True, capture_output=True).stdout


def save_surface(surface, path):
    surface.makeImageSnapshot().save(str(path), skia.kPNG)


def svg_to_png(data, w, h, path):
    dom = skia.SVGDOM.MakeFromStream(skia.MemoryStream(data))
    if dom is None:
        raise ValueError("bad svg")
    dom.setContainerSize(skia.Size(w, h))
    surface = skia.Surface(w, h)
    dom.render(surface.getCanvas())
    save_surface(surface, path)


def color_of(s):
    m = re.findall(r"[\d.]+", str(s))
    if len(m) < 3:
        return None
    a = float(m[3]) if len(m) > 3 else 1.0
    return skia.Color(int(float(m[0])), int(float(m[1])), int(float(m[2])), int(a * 255))


def path_box(d):
    """Bounds of an absolute M/L/C/Q path; None for anything else."""
    if re.search(r"[^MLCQZ\d\s,.eE+-]", d):
        return None
    n = [float(x) for x in re.findall(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", d)]
    if len(n) < 4:
        return None
    xs, ys = n[::2], n[1::2]
    return min(xs), min(ys), max(max(xs) - min(xs), 1e-6), max(max(ys) - min(ys), 1e-6)


def shape_to_png(c, w, h, path):
    m = re.search(r'path\("(.*)"\)', str(c.get("clipPath", "")))
    box = path_box(m.group(1)) if m else None
    col = c.get("background") or c.get("backgroundColor") or "none"
    if box and color_of(col) is not None:
        vb = " ".join(f"{v:.4f}" for v in box)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" preserveAspectRatio="none">'
               f'<path d="{m.group(1)}" fill="{col}"/></svg>')
        svg_to_png(svg.encode(), w, h, path)
        return
    surface = skia.Surface(w, h)
    surface.getCanvas().clear(color_of(col) or skia.ColorTRANSPARENT)
    save_surface(surface, path)


def transform_of(c):
    t = str(c.get("transform") or "")
    m = re.search(r"rotate\((-?[\d.]+)deg\)", t)
    rot = float(m.group(1)) if m else 0.0
    return rot, "scale(-1, 1)" in t, "scale(1, -1)" in t


def text_element(c, base):
    size = px(c.get("fontSize"), 16) or 16
    weight = str(c.get("fontWeight", "400"))
    e = dict(base, type="text", content=c.get("text", ""), font_family=family(c), font_size=size,
             font_weight="bold" if weight == "bold" or px(weight) >= 600 else "regular",
             font_style="italic" if c.get("fontStyle") == "italic" else "normal",
             color=c.get("color", "rgb(0, 0, 0)"),
             text_align=c.get("textAlign") if c.get("textAlign") in ("left", "center", "right") else "left",
             line_height=round(px(c.get("lineHeight"), size * 1.2) / size, 3),
             letter_spacing=px(c.get("letterSpacing")) * size,
             capitalize=c.get("textTransform") == "uppercase")
    return e


def image_element(c, base, W, H, asset_dir, prefix, i):
    aid = f"{prefix}_{i}.png"
    w, h = max(1, int(round(base["width"]))), max(1, int(round(base["height"]))),
    path = asset_dir / aid
    if c["type"] == "GROUP":
        shape_to_png(c, w, h, path)
    else:
        data = fetch(c["src"])
        if c["src"].lower().endswith(".svg"):
            svg_to_png(data, w, h, path)
        else:
            path_in = asset_dir / (aid + ".src")
            path_in.write_bytes(data)
            Image.open(path_in).convert("RGBA").save(path)
            path_in.unlink()
    _, fx, fy = transform_of(c)
    if fx or fy:
        im = Image.open(path)
        im = ImageOps.mirror(im) if fx else im
        (ImageOps.flip(im) if fy else im).save(path)
    # A full-bleed backdrop is cropped to fit, see crello-export-fit.
    bleed = base["width"] >= W * 0.98 and base["height"] >= H * 0.98
    return dict(base, type="image", asset_id=aid, fit="cover" if bleed else "fill")


def to_chameleon(layout, asset_dir, prefix):
    W, H = int(px(layout["width"])), int(px(layout["height"]))
    els = []
    bg = color_of(layout.get("background"))
    if bg is not None:
        aid = f"{prefix}_bg.png"
        surface = skia.Surface(8, 8)
        surface.getCanvas().clear(bg)
        save_surface(surface, asset_dir / aid)
        els.append({"id": "bg", "type": "image", "x": 0, "y": 0, "width": W, "height": H, "z_index": 0,
                    "opacity": 1.0, "asset_id": aid, "fit": "cover"})
    for i, c in enumerate(layout["components"]):
        rot, _, _ = transform_of(c)
        base = {"id": f"e{i}", "x": round(px(c.get("left")), 2), "y": round(px(c.get("top")), 2),
                "width": round(px(c.get("width")), 2), "height": round(px(c.get("height")), 2),
                "z_index": i + 1, "opacity": float(c.get("opacity", 1) or 1)}
        if abs(rot) > 0.01:
            base["rotation"] = rot
        if base["width"] <= 0 or base["height"] <= 0:
            continue
        if c["type"].startswith("TEXT"):
            if str(c.get("text", "")).strip():
                els.append(text_element(c, base))
        else:
            els.append(image_element(c, base, W, H, asset_dir, prefix, i))
    return {"canvas": {"width": W, "height": H}, "elements": els}


def targets_for(W, H):
    r = W / H
    if r > 1.2:
        return [(1080, 1920), (1080, 1080)]
    if r < 0.83:
        return [(1920, 1080), (1080, 1080)]
    return [(1080, 1920), (1920, 1080)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--data", required=True)
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out, data = Path(a.out), Path(a.data)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    (out / "designs").mkdir(parents=True, exist_ok=True)
    with open(Path(a.fonts).expanduser(), "rb") as f:
        fonts = pickle.load(f)
    known = {k.lower(): k for k in fonts}

    by_cat, skipped = defaultdict(list), defaultdict(int)
    for row in csv.DictReader(open(data / "metadata.csv")):
        if row["category"] in SKIP_CATEGORIES:
            continue
        p = data / "layouts" / row["template_id"] / f"{row['layout_id']}.json"
        if not p.is_file():
            continue
        layout = json.loads(p.read_text())
        fams = {family(c) for c in layout["components"] if c["type"].startswith("TEXT")}
        if not is_static(layout):
            skipped["moving"] += 1
        elif not fams or any(x.lower() not in known for x in fams):
            skipped["fonts"] += 1
        else:
            by_cat[row["category"]].append((row, layout))
    print("eligible", {k: len(v) for k, v in by_cat.items()}, "skipped", dict(skipped))

    # One template per pick: layouts of a template are near-copies.
    rng = random.Random(a.seed)
    pools = {k: rng.sample(v, len(v)) for k, v in sorted(by_cat.items())}
    picks, seen = [], set()
    while len(picks) < a.n and any(pools.values()):
        for cat, pool in pools.items():
            while pool:
                row, layout = pool.pop()
                if row["template_id"] not in seen:
                    seen.add(row["template_id"])
                    picks.append((row, layout))
                    break
            if len(picks) == a.n:
                break

    def one(job):
        p, (row, layout) = job
        try:
            d = to_chameleon(layout, out / "assets", f"p{p}")
        except Exception as e:
            print(f"p{p} {row['layout_id']}: {type(e).__name__}: {e}", file=sys.stderr)
            return None
        return row, d

    used, index = set(), []
    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(one, enumerate(picks)):
            if res is None:
                continue
            row, d = res
            for e in d["elements"]:
                if e["type"] == "text":
                    e["font_family"] = known[e["font_family"].lower()]
                    used.add(e["font_family"])
            W, H = d["canvas"]["width"], d["canvas"]["height"]
            k = len(index)
            fn = f"p{k}_{W}x{H}.json"
            (out / "designs" / fn).write_text(json.dumps(d))
            index.append([{"file": fn, "w": W, "h": H, "title": row["layout_id"], "format": row["category"]}]
                         + [{"w": w, "h": h} for w, h in targets_for(W, H)])
    (out / "index.json").write_text(json.dumps(index, indent=1))
    keep = used | {"Montserrat"}
    with open(out / "fonts_subset.pickle", "wb") as f:
        pickle.dump({k: v for k, v in fonts.items() if k in keep}, f)
    print(f"exported {len(index)} designs, {len(keep)} font families")


if __name__ == "__main__":
    main()
