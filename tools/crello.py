"""Dev tool: real designs from Crello as a regression set for the engine.

Crello (huggingface.co/datasets/cyberagent/crello, CDLA-Permissive-2.0) holds
templates that designers re-made at several canvas sizes. Grouping designs by
their text gives "families": one source design plus the designer's own answer
for other sizes. This tool converts a small, varied sample into Chameleon tasks
so the engine can be compared against what a designer actually did.

    .venv/bin/python -m tools.crello meta      # light columns of every shard (~50 MB) -> cache
    .venv/bin/python -m tools.crello pick      # choose a varied sample -> tools/crello/families.json
    .venv/bin/python -m tools.crello fetch     # images for the sample only -> tools/crello/cases/
    .venv/bin/python -m tools.crello sheet     # source | designer | ours, per target -> sheet png

Dev-only: nothing here is used by the engine or copied into the image. The
engine never looks designs up; this is evaluation only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "crello"
CACHE = ROOT / "cache"
CASES = ROOT / "cases"
BASE = "https://huggingface.co/datasets/cyberagent/crello/resolve/refs%2Fconvert%2Fparquet/default"
LIGHT = ["id", "length", "format", "canvas_width", "canvas_height", "title", "type", "left", "top", "width",
         "height", "angle", "opacity", "color", "text", "font", "font_size", "font_bold", "font_italic",
         "text_line", "text_color", "text_align", "capitalize", "line_height", "letter_spacing"]


def shards():
    import urllib.request

    d = json.load(urllib.request.urlopen("https://datasets-server.huggingface.co/parquet?dataset=cyberagent/crello"))
    return [(f["split"], f["filename"]) for f in d["parquet_files"]]


def features():
    import urllib.request

    p = CACHE / "info.json"
    if not p.exists():
        info = json.load(urllib.request.urlopen("https://datasets-server.huggingface.co/info?dataset=cyberagent/crello"))
        p.write_text(json.dumps(info["dataset_info"]["default"]["features"]))
    return json.loads(p.read_text())


def _names(feats, key):
    f = feats[key]
    f = f[0] if isinstance(f, list) else f.get("feature", f)
    return f["names"]


def _open(split, fn):
    import fsspec
    import pyarrow.parquet as pq

    f = fsspec.open(f"{BASE}/{split}/{fn}", block_size=2 ** 20).open()
    return pq.ParquetFile(f)


def cmd_meta(_a):
    import pyarrow as pa
    import pyarrow.parquet as pq

    CACHE.mkdir(parents=True, exist_ok=True)
    features()

    def one(sf):
        split, fn = sf
        out = CACHE / f"meta_{split}_{fn}"
        if out.exists():
            return out
        pf = _open(split, fn)
        tables = []
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=LIGHT)
            n = t.num_rows
            t = t.append_column("split", pa.array([split] * n)).append_column("shard", pa.array([fn] * n))
            t = t.append_column("rg", pa.array([rg] * n)).append_column("row", pa.array(list(range(n))))
            tables.append(t)
        pq.write_table(pa.concat_tables(tables), out)
        print("meta", split, fn, flush=True)
        return out

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, shards()))


def load_meta():
    import pandas as pd

    return pd.concat([pd.read_parquet(p) for p in sorted(CACHE.glob("meta_*.parquet"))], ignore_index=True)


def cmd_pick(a):
    import pandas as pd

    feats = features()
    tnames = _names(feats, "type")
    df = load_meta()
    df["ar"] = df.canvas_width / df.canvas_height

    def sig(texts):
        t = sorted(str(s).strip().lower() for s in texts if s and str(s).strip())
        return "\x1f".join(t) if len(t) >= 2 else None

    df["sig"] = df.text.map(sig)
    df = df[df.sig.notna()]
    df["types"] = df.type.map(lambda ts: [tnames[t] for t in ts])
    df["has_photo"] = df.types.map(lambda ts: "ImageElement" in ts)
    fams = []
    for s, g in df.groupby("sig"):
        if g[["canvas_width", "canvas_height"]].drop_duplicates().shape[0] < 2:
            continue
        g = g.drop_duplicates(["canvas_width", "canvas_height"])
        if len(g) > 8 or g.length.max() > 25:
            continue
        # source: the member closest to square/portrait (how most source posts look)
        src = g.loc[(g.ar.map(lambda r: abs(math.log(r / 0.9)))).idxmin()]
        tgts = g[g.id != src.id]
        if tgts.empty:
            continue
        gid = {r.id: (r.split, r.shard, int(r.rg)) for r in g.itertuples()}
        # one target per family: the most extreme reshaping (what the engine finds hardest)
        t = tgts.loc[(tgts.ar.map(lambda r: abs(math.log(r / src.ar)))).idxmax()]
        fams.append(dict(sig=s, src=src.id, src_ar=float(src.ar), tgts=[t.id], tgt_ar=[float(t.ar)],
                         groups=sorted({gid[src.id], gid[t.id]}), photo=bool(src.has_photo), n_el=int(src.length),
                         wide=bool(t.ar >= 2.4), tall=bool(t.ar <= 0.62)))
    print("families with >1 size:", len(fams))
    # greedy: download the row groups (~57 MB of images each) that complete the most families
    import collections

    chosen, picked = set(), []
    left = [f for f in fams if f["wide"] or f["tall"] or abs(math.log(f["tgt_ar"][0] / f["src_ar"])) > 0.5]
    while len(chosen) < a.groups and left:
        gain = collections.Counter()
        for f in left:
            need = [tuple(x) for x in f["groups"] if tuple(x) not in chosen]
            w = 1.0 + (1.0 if f["wide"] else 0) + (0.5 if f["photo"] else 0)
            for x in need:
                gain[x] += w / len(need)
        best = gain.most_common(1)[0][0]
        chosen.add(best)
        done = [f for f in left if all(tuple(x) in chosen for x in f["groups"])]
        picked += done
        left = [f for f in left if f not in done]
    picked = picked[: a.n]
    print("row groups", len(chosen), "families", len(picked),
          "wide", sum(f["wide"] for f in picked), "tall", sum(f["tall"] for f in picked),
          "photo", sum(f["photo"] for f in picked))
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "families.json").write_text(json.dumps(picked, indent=1))


def _to_chameleon(row, tnames, aligns, fonts, assets_dir: Path):
    """One Crello v5 design -> Chameleon schema (every non-text element becomes an image)."""
    W, H = int(row["canvas_width"]), int(row["canvas_height"])
    els = []
    for i in range(int(row["length"])):
        t = tnames[row["type"][i]]
        x, y, w, h = (float(row[k][i]) for k in ("left", "top", "width", "height"))
        base = dict(id=f"e{i}", x=x, y=y, width=max(1.0, w), height=max(1.0, h), z_index=i,
                    rotation=float(row["angle"][i]), opacity=float(row["opacity"][i]))
        if t == "TextElement":
            txt = str(row["text"][i])
            colors = row["text_color"][i]
            col = colors[0] if len(colors) else "rgb(0,0,0)"
            bold = bool(len(row["font_bold"][i]) and row["font_bold"][i][0])
            base.update(type="text", content=txt, font_family=fonts[row["font"][i]], font_size=float(row["font_size"][i]),
                        color=col, text_align=aligns[row["text_align"][i]],
                        line_height=float(row["line_height"][i]) or 1.2, letter_spacing=float(row["letter_spacing"][i]))
            if bold:
                base["font_weight"] = "bold"
            if base["text_align"] not in ("left", "center", "right"):
                base["text_align"] = "left"
        else:
            img = row["image"][i]
            data = img["bytes"] if isinstance(img, dict) else img
            name = f"{row['id']}_{i}.png"
            if data:
                (assets_dir / name).write_bytes(data)
            base.update(type="image", asset_id=name, fit="fill")
        els.append(base)
    return {"canvas": {"width": W, "height": H}, "elements": els}


def cmd_fetch(_a):
    import pyarrow.parquet as pq

    feats = features()
    tnames = _names(feats, "type")
    aligns = _names(feats, "text_align")
    fonts = _names(feats, "font")
    fams = json.loads((ROOT / "families.json").read_text())
    meta = load_meta().set_index("id")
    need = {}
    for f in fams:
        for d in [f["src"]] + f["tgts"]:
            r = meta.loc[d]
            need.setdefault((r.split, r.shard, int(r.rg)), set()).add(d)
    assets = CASES / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    designs = {}

    def one(key):
        split, fn, rg = key
        t = _open(split, fn).read_row_group(rg, columns=LIGHT + ["image"]).to_pylist()
        return [row for row in t if row["id"] in need[key]]

    with ThreadPoolExecutor(4) as ex:
        for rows in ex.map(one, list(need)):
            for row in rows:
                designs[row["id"]] = _to_chameleon(row, tnames, aligns, fonts, assets)
            print("fetched", len(designs), flush=True)
    for k, f in enumerate(fams):
        out = {"family": k, "source": designs[f["src"]], "targets": [designs[t] for t in f["tgts"] if t in designs],
               "photo": f["photo"]}
        (CASES / f"fam{k:02d}.json").write_text(json.dumps(out))
    print("wrote", len(fams), "families to", CASES)


def cmd_sheet(a):
    import os

    os.environ["CHAMELEON_BRAND_KIT"] = str(CASES)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from PIL import Image, ImageDraw

    from engine.resize import resize
    from renderer.converter.convert import render_design_to_png

    def render(design, cell):
        p = Path(f"/tmp/_crello_render_{os.getpid()}.png")  # per process: sessions run in parallel
        render_design_to_png(design, CASES / "assets", p, fonts_dir=Path(__file__).resolve().parents[1] / "brand_kit" / "fonts")
        im = Image.open(p).convert("RGB")
        p.unlink(missing_ok=True)
        f = min(cell / im.width, cell * 0.75 / im.height)
        return im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))))

    files = sorted(CASES.glob("fam*.json"))
    if a.only:
        files = [f for f in files if any(f.stem.endswith(f"{int(x):02d}") for x in a.only.split(","))]
    rows = []
    for fpath in files:
        fam = json.loads(fpath.read_text())
        src = fam["source"]
        for tgt in fam["targets"][: a.max_targets]:
            W, H = tgt["canvas"]["width"], tgt["canvas"]["height"]
            if a.wide and W / H < 2.4:
                continue
            ours = resize(json.loads(json.dumps(src)), {"width": W, "height": H})
            ims = [render(src, a.cell), render(tgt, a.cell), render(ours, a.cell)]
            rows.append((f"{fpath.stem} {src['canvas']['width']}x{src['canvas']['height']} -> {W}x{H}", ims))
    if not rows:
        print("nothing to show")
        return
    cw = a.cell + 6
    heights = [max(im.height for im in ims) + 20 for _, ims in rows]
    sheet = Image.new("RGB", (3 * cw, sum(heights)), (210, 210, 210))
    y = 0
    for (label, ims), h in zip(rows, heights):
        d = ImageDraw.Draw(sheet)
        d.text((4, y + 3), label + "    [source | designer | ours]", fill=(0, 0, 0))
        for j, im in enumerate(ims):
            sheet.paste(im, (j * cw, y + 18))
        y += h
    sheet.save(a.out)
    print(a.out, sheet.size, len(rows), "rows")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("meta")
    p = sub.add_parser("pick")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--groups", type=int, default=20)
    sub.add_parser("fetch")
    p = sub.add_parser("sheet")
    p.add_argument("--out", default="/tmp/crello_sheet.png")
    p.add_argument("--cell", type=int, default=300)
    p.add_argument("--max-targets", type=int, default=2)
    p.add_argument("--only", default="")
    p.add_argument("--wide", action="store_true")
    a = ap.parse_args()
    {"meta": cmd_meta, "pick": cmd_pick, "fetch": cmd_fetch, "sheet": cmd_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
