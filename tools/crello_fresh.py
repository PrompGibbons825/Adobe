"""Held-out designer pairs: Crello designs made in several sizes that no engine
has been tuned on. Excludes every title in crello/ and crello_holdout/ (ours)
and every design id listed in Vlatko's gate manifests, then exports the rest
in CHAMELEON schema like tools/crello_export.py does.

usage (repo root):
  python tools/crello_fresh.py OUT_DIR --parquet ~/crello_raw/data/*.parquet \
      --fonts ~/crello_raw/resources/fonts.pickle [--manifests DIR]
"""

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.crello_export import norm_text, normalize_family, to_chameleon  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--manifests", default=None, help="Vlatko's tools/gate/manifests (design ids to exclude)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    (out / "designs").mkdir(parents=True, exist_ok=True)

    used = set()
    for r in ("crello", "crello_holdout"):
        p = Path(r) / "index.json"
        if p.is_file():
            used |= {vs[0]["title"] for vs in json.load(open(p))}
    theirs = set()
    if a.manifests:
        for f in Path(a.manifests).glob("*.sha256"):
            theirs |= set(re.findall(r"assets/([0-9a-f]{24})_", f.read_text()))

    ds = load_dataset("parquet", data_files=a.parquet, split="train")
    features = ds.features
    meta = ds.remove_columns([c for c in ("image", "preview") if c in ds.column_names])
    by_title = defaultdict(list)
    for k, ex in enumerate(meta):
        if ex["title"]:
            texts = {norm_text(t) for t in ex["text"] if norm_text(t)}
            by_title[ex["title"]].append((k, ex["canvas_width"] / ex["canvas_height"], texts, ex["id"]))
    fams = []
    for title, rows in by_title.items():
        if len(rows) < 2 or title in used:
            continue
        rows.sort(key=lambda r: r[1])
        base = rows[0]
        same = [base] + [r for r in rows[1:] if base[2] and r[2]
                         and len(base[2] & r[2]) / max(1, min(len(base[2]), len(r[2]))) >= 0.5
                         and max(r[1], base[1]) / min(r[1], base[1]) > 1.2]
        if len(same) >= 2 and not any(r[3] in theirs for r in same):
            fams.append(same)
    print(f"{len(fams)} unused designs with designer-made resizes "
          f"({len(used)} titles of ours and {len(theirs)} of Vlatko's design ids excluded)")

    families, index = set(), []
    for p, rows in enumerate(fams):
        variants = []
        for k, ratio, _, _ in rows:
            ex = ds[k]
            d = to_chameleon(ex, features, out / "assets", f"p{p}_r{k}")
            fn = f"p{p}_{d['canvas']['width']}x{d['canvas']['height']}_r{k}.json"
            (out / "designs" / fn).write_text(json.dumps(d))
            families.update(e["font_family"] for e in d["elements"] if e["type"] == "text")
            variants.append({"file": fn, "w": d["canvas"]["width"], "h": d["canvas"]["height"],
                             "title": ex["title"], "format": features["format"].int2str(ex["format"]),
                             "id": ex["id"]})
        index.append(variants)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    with open(a.fonts, "rb") as f:
        fonts = pickle.load(f)
    keys = {normalize_family(x) for x in families} | {normalize_family("Montserrat")}
    with open(out / "fonts_subset.pickle", "wb") as f:
        pickle.dump({k: v for k, v in fonts.items() if k in keys}, f)
    print(f"exported {len(index)} designs, {sum(len(v) - 1 for v in index)} pairs")


if __name__ == "__main__":
    main()
