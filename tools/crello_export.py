"""Runs on a machine that can reach Hugging Face.

Streams Crello v5, finds designs the designer made in more than one size
(same text content, different aspect ratio), and exports each version in
CHAMELEON schema with its pixel assets, plus a fonts.pickle subset holding
only the families those designs use.

usage: python crello_export.py OUT_DIR [--split test] [--scan 4000] [--pairs 12]
"""

import argparse
import io
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def normalize_family(name: str) -> str:
    """Normalize font name."""
    name = name.replace("_", " ").title()
    name = name.replace(" Bold", "")
    name = name.replace(" Regular", "")
    name = name.replace(" Light", "")
    name = name.replace(" Italic", "")
    name = name.replace(" Medium", "")

    FONT_MAP = {
        "Arkana Script": "Arkana",
        "Blogger": "Blogger Sans",
        "Delius Swash": "Delius Swash Caps",
        "Elsie Swash": "Elsie Swash Caps",
        "Gluk Glametrix": "Gluk Foglihtenno06",
        "Gluk Znikomitno25": "Gluk Foglihtenno06",
        "Im Fell": "Im Fell Dw Pica Sc",
        "Medieval Sharp": "Medievalsharp",
        "Playlist Caps": "Playlist",
        "Rissa Typeface": "Rissatypeface",
        "Selima": "Selima Script",
        "Six": "Six Caps",
        "V T323": "Vt323",
        # The rest is unknown.
        "Different Summer": "Montserrat",
        "Dukomdesign Constantine": "Montserrat",
        "Sunday": "Montserrat",
    }
    return FONT_MAP.get(name, name)


def norm_text(t):
    return re.sub(r"\s+", " ", t or "").strip().lower()


def decode(features, key, v):
    f = features.get(key)
    try:
        feat = f.feature if hasattr(f, "feature") else f
        if hasattr(feat, "int2str"):
            return [feat.int2str(x) if isinstance(x, int) else x for x in v] if isinstance(v, list) else feat.int2str(v)
    except Exception:
        pass
    return v


def to_chameleon(ex, features, asset_dir: Path, prefix: str):
    W, H = int(ex["canvas_width"]), int(ex["canvas_height"])
    n = int(ex["length"])
    g = {k: decode(features, k, ex[k]) for k in ex if k not in ("image", "preview")}
    norm = max((g["left"][i] + g["width"][i]) for i in range(n)) <= 1.5 if n else False
    sx, sy = (W, H) if norm else (1, 1)
    els = []
    for i in range(n):
        typ = g["type"][i]
        x, y = g["left"][i] * sx, g["top"][i] * sy
        w, h = g["width"][i] * sx, g["height"][i] * sy
        ang = g["angle"][i]
        rot = ang * 180 / math.pi if abs(ang) <= 2 * math.pi + 1e-6 else ang
        e = {"id": f"e{i}", "x": round(x, 2), "y": round(y, 2), "width": round(w, 2),
             "height": round(h, 2), "z_index": i, "opacity": float(g["opacity"][i]),
             "_crello_type": typ}
        if abs(rot) > 0.01:
            e["rotation"] = round(rot, 2)
        text = g["text"][i] if "text" in g else ""
        if typ == "TextElement" and text:
            colors = g.get("text_color", [[]] * n)[i] or ["rgba(0,0,0,1)"]
            bold = g.get("font_bold", [[]] * n)[i] or [False]
            ital = g.get("font_italic", [[]] * n)[i] or [False]
            e.update({
                "type": "text", "content": text, "font_family": g["font"][i],
                "font_size": float(g["font_size"][i]),
                "font_weight": "bold" if all(bold) else "regular",
                "font_style": "italic" if all(ital) else "normal",
                "color": colors[0] if isinstance(colors[0], str) else "rgba(0,0,0,1)",
                "text_align": g["text_align"][i] if g["text_align"][i] in ("left", "center", "right") else "left",
                "line_height": float(g["line_height"][i] or 1.2),
                "letter_spacing": float(g["letter_spacing"][i] or 0),
                "capitalize": bool(g["capitalize"][i]) if not isinstance(g["capitalize"][i], str) else g["capitalize"][i] == "true",
            })
            # per-character colour changes -> style_runs
            runs, start = [], 0
            for j in range(1, len(colors) + 1):
                if j == len(colors) or colors[j] != colors[start]:
                    if colors[start] != e["color"]:
                        runs.append({"start": start, "end": j, "color": colors[start]})
                    start = j
            if runs:
                e["style_runs"] = runs
        else:
            img = ex["image"][i]
            aid = f"{prefix}_{i}.png"
            if img is not None:
                img.convert("RGBA").save(asset_dir / aid)
            e.update({"type": "image", "asset_id": aid, "fit": "fill"})
        els.append(e)
    return {"canvas": {"width": W, "height": H}, "elements": els}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scan", type=int, default=4000)
    ap.add_argument("--pairs", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    (out / "designs").mkdir(parents=True, exist_ok=True)

    ds = load_dataset("cyberagent/crello", revision="5.0.0", split=a.split, streaming=True)
    features = ds.features
    groups = defaultdict(list)
    keep = {}
    for k, ex in enumerate(ds):
        if k >= a.scan:
            break
        texts = tuple(sorted(norm_text(t) for t in ex["text"] if norm_text(t)))
        if not texts or sum(len(t) for t in texts) < 12:
            continue
        ratio = ex["canvas_width"] / ex["canvas_height"]
        groups[texts].append((k, ratio))
        keep[k] = ex
        # drop rows that can no longer form a pair to bound memory
        if len(keep) > 6000:
            for kk in list(keep)[:1000]:
                if all(len(groups[t]) < 2 for t in groups if any(r[0] == kk for r in groups[t])):
                    keep.pop(kk, None)
    pairs = []
    for texts, rows in groups.items():
        rows = [r for r in rows if r[0] in keep]
        rows.sort(key=lambda r: r[1])
        if len(rows) >= 2 and rows[-1][1] / rows[0][1] > 1.25:
            pairs.append(rows)
    print(f"scanned {a.scan} rows, found {len(pairs)} multi-size designs")
    pairs.sort(key=lambda rows: -(rows[-1][1] / rows[0][1]))
    families = set()
    index = []
    for p, rows in enumerate(pairs[: a.pairs]):
        variants = []
        for k, ratio in rows:
            ex = keep[k]
            d = to_chameleon(ex, features, out / "assets", f"p{p}_r{k}")
            fn = f"p{p}_{d['canvas']['width']}x{d['canvas']['height']}_r{k}.json"
            (out / "designs" / fn).write_text(json.dumps(d))
            ex["preview"].convert("RGB").save(out / "designs" / fn.replace(".json", "_preview.jpg")) if ex.get("preview") else None
            families.update(e["font_family"] for e in d["elements"] if e["type"] == "text")
            variants.append({"file": fn, "w": d["canvas"]["width"], "h": d["canvas"]["height"], "id": ex.get("id")})
        index.append(variants)
    (out / "index.json").write_text(json.dumps(index, indent=1))

    fp = hf_hub_download(repo_id="cyberagent/opencole", filename="resources/fonts.pickle", repo_type="dataset")
    with open(fp, "rb") as f:
        fonts = pickle.load(f)
    keys = {normalize_family(x) for x in families} | {normalize_family("Montserrat")}
    sub = {k: v for k, v in fonts.items() if k in keys}
    with open(out / "fonts_subset.pickle", "wb") as f:
        pickle.dump(sub, f)
    print(f"exported {len(index)} designs, {len(sub)} font families")


if __name__ == "__main__":
    main()
