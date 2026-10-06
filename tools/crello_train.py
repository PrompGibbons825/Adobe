"""Dev tool: a larger Crello family set for training the candidate ranker.

tools/crello.py samples 64 families (one extreme target each) as a regression
set. This picks several hundred *more* families, with up to three designer
retargets each, for learning from: which candidate layout is closest to what
the designer did at that size.

    .venv/bin/python -m tools.crello meta            # light columns (shared cache)
    .venv/bin/python -m tools.crello_train pick      # -> tools/crello/train_families.json
    .venv/bin/python -m tools.crello_train fetch     # images -> tools/crello/train/

Splits are strictly by family (a family = one text signature):
  * the 64 families of tools/crello/families.json keep their meaning: even =
    tuning half, odd = held out (never trained on);
  * the new families are split by a hash of their signature: 1 in 5 is a new
    held-out set ("held2"), the rest is training data.

Extreme retargets (>= 2.4:1 wide, <= 0.62 tall) are over-sampled by the greedy
row-group choice. Dev-only: the engine never sees any of this.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.crello import LIGHT, ROOT, _names, _open, _to_chameleon, features, load_meta

TRAIN = ROOT / "train"
FAMS = ROOT / "train_families.json"


def split_of_sig(sig: str) -> str:
    h = int(hashlib.md5(sig.encode("utf-8")).hexdigest()[:8], 16)
    return "held2" if h % 5 == 0 else "train"


def _lar(a: float, b: float) -> float:
    return abs(math.log(a / b))


def cmd_pick(a):
    feats = features()
    tnames = _names(feats, "type")
    df = load_meta()
    df["ar"] = df.canvas_width / df.canvas_height

    def sig(texts):
        t = sorted(str(s).strip().lower() for s in texts if s and str(s).strip())
        return "\x1f".join(t) if len(t) >= 2 else None

    df["sig"] = df.text.map(sig)
    df = df[df.sig.notna()]
    df["has_photo"] = df.type.map(lambda ts: "ImageElement" in [tnames[t] for t in ts])
    old = {f["sig"] for f in json.loads((ROOT / "families.json").read_text())}
    rg = lambda r: (r.split, r.shard, int(r.rg))  # noqa: E731
    fams = []
    for s, g in df.groupby("sig"):
        if s in old:
            continue
        g = g.drop_duplicates(["canvas_width", "canvas_height"])
        g = g[g.length <= 25]
        if len(g) < 2:
            continue
        src = g.loc[(g.ar.map(lambda r: _lar(r, 0.9))).idxmin()]
        tg = g[(g.id != src.id) & (g.ar.map(lambda r: _lar(r, src.ar)) >= 0.05)]
        if tg.empty:
            continue
        tg = tg.assign(ext=tg.ar.map(lambda r: _lar(r, src.ar))).sort_values("ext", ascending=False)
        first = tg.iloc[0]
        wide, tall = bool(first.ar >= 2.4), bool(first.ar <= 0.62)
        fams.append(dict(sig=s, src=src.id, src_ar=float(src.ar), src_rg=rg(src), photo=bool(src.has_photo),
                         n_el=int(src.length), wide=wide, tall=tall,
                         tgts=[(r.id, float(r.ar), rg(r)) for r in tg.itertuples()]))
    print("candidate families:", len(fams))
    # greedy over row groups: complete (source, most extreme target), weighted to extremes
    chosen, picked = set(), []
    left = list(fams)
    while len(picked) < a.n and left:
        gain = collections.Counter()
        for f in left:
            need = {f["src_rg"], f["tgts"][0][2]} - chosen
            w = 1.0 + (1.0 if f["wide"] else 0.0) + (0.7 if f["tall"] else 0.0) + (0.3 if f["photo"] else 0.0)
            for x in need:
                gain[x] += w / len(need)
        best = gain.most_common(1)[0][0]
        chosen.add(best)
        done = [f for f in left if {f["src_rg"], f["tgts"][0][2]} <= chosen]
        picked += done
        ids = {id(f) for f in done}
        left = [f for f in left if id(f) not in ids]
    picked = picked[: a.n]
    out = []
    for f in picked:
        # extra targets come free when their row group is downloaded anyway
        tg = [t for t in f["tgts"] if t[2] in chosen][: a.max_targets]
        out.append(dict(sig=f["sig"], split=split_of_sig(f["sig"]), src=f["src"], src_ar=f["src_ar"],
                        tgts=[t[0] for t in tg], tgt_ar=[t[1] for t in tg],
                        groups=sorted({tuple(f["src_rg"])} | {tuple(t[2]) for t in tg}),
                        photo=f["photo"], n_el=f["n_el"], wide=f["wide"], tall=f["tall"]))
    n_t = sum(len(f["tgts"]) for f in out)
    ars = [r for f in out for r in f["tgt_ar"]]
    print(f"row groups {len(chosen)}  families {len(out)}  targets {n_t}  "
          f"held2 {sum(f['split'] == 'held2' for f in out)}  "
          f"targets wide>=2.4 {sum(r >= 2.4 for r in ars)}  strip>=6 {sum(r >= 6 for r in ars)}  "
          f"tall<=0.62 {sum(r <= 0.62 for r in ars)}  photo {sum(f['photo'] for f in out)}")
    FAMS.write_text("[\n" + ",\n".join(json.dumps(f) for f in out) + "\n]\n")  # one family per line


def cmd_fetch(a):
    feats = features()
    tnames = _names(feats, "type")
    aligns = _names(feats, "text_align")
    fonts = _names(feats, "font")
    fams = json.loads(FAMS.read_text())
    meta = load_meta().set_index("id")
    need = {}
    for f in fams:
        for d in [f["src"]] + f["tgts"]:
            r = meta.loc[d]
            need.setdefault((r.split, r.shard, int(r.rg)), set()).add(d)
    assets = TRAIN / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    designs_dir = TRAIN / "designs"
    designs_dir.mkdir(parents=True, exist_ok=True)
    todo = [k for k in need if not all((designs_dir / f"{d}.json").exists() for d in need[k])]
    print("row groups to read:", len(todo), "of", len(need), flush=True)

    def one(key):
        split, fn, g = key
        rows = _open(split, fn).read_row_group(g, columns=LIGHT + ["image"]).to_pylist()
        n = 0
        for row in rows:
            if row["id"] in need[key]:
                d = _to_chameleon(row, tnames, aligns, fonts, assets)
                (designs_dir / f"{row['id']}.json").write_text(json.dumps(d))
                n += 1
        return n

    done = 0
    with ThreadPoolExecutor(a.threads) as ex:
        for i, n in enumerate(ex.map(one, todo)):
            done += n
            if i % 10 == 0:
                print(f"row group {i + 1}/{len(todo)}  designs {done}", flush=True)
    missing = 0
    cases = TRAIN / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(fams):
        try:
            src = json.loads((designs_dir / f"{f['src']}.json").read_text())
            tg = [json.loads((designs_dir / f"{t}.json").read_text()) for t in f["tgts"]]
        except FileNotFoundError:
            missing += 1
            continue
        out = {"family": f"t{k:03d}", "sig": f["sig"], "split": f["split"], "source": src, "targets": tg,
               "photo": f["photo"]}
        (cases / f"t{k:03d}.json").write_text(json.dumps(out))
    print("wrote", len(fams) - missing, "families to", cases, "missing", missing)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--max-targets", type=int, default=3)
    p = sub.add_parser("fetch")
    p.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    {"pick": cmd_pick, "fetch": cmd_fetch}[a.cmd](a)


if __name__ == "__main__":
    main()
