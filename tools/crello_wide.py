"""Dev tool (bet B0): the widest designer-resize data in Crello v5.

    .venv/bin/python -m tools.crello_wide index   # every family with >=2 canvas sizes -> families.jsonl
    .venv/bin/python -m tools.crello_wide sample  # stratified dev sample (<=400 families) -> sample.json
    .venv/bin/python -m tools.crello_wide fetch   # images for the sample only -> cases/ + assets/

Data: ``cyberagent/crello`` v5, CDLA-Permissive-2.0. Index = geometry and metadata of all
~23k templates, no pixels. A *family* is the same design (same sorted element types + text) at
two or more canvas sizes; each member is a designer's own resize of the others.
Every family whose signature hash is in ``tools.gate.sealed.exclusions("crello")`` is dropped
(the function returns hashes only; the sealed store is never opened here).

Output dir: $CRELLO_WIDE (default /data/agents/work/crello-wide). The metadata cache is
tools/crello/cache (``python -m tools.crello meta``), or $CRELLO_META_CACHE.
families.jsonl: one family per line:
  {"fam": <24-hex sig hash>, "n_sizes": k, "photo": bool, "members": [
     {"id", "W", "H", "ar", "loc": [split, shard, rg], "n_el",
      "els": [{"i", "kind", "role", "x","y","w","h","angle","opacity","fs","text","bold","color","align"}]}]}
Roles are crude hints (bg, headline, subhead, body, small, photo, logo_or_art, shape), inferred from
geometry and font size only, as the engine must do.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import crello  # noqa: E402
from tools.gate import sealed  # noqa: E402
from tools.gate.crello_sealed import sig_hash, sig_of  # noqa: E402

OUT = Path(os.environ.get("CRELLO_WIDE", "/data/agents/work/crello-wide"))
MAX_FAMILY_ELEMENTS = 40
MAX_SIZES = 12


def _use_cache():
    c = os.environ.get("CRELLO_META_CACHE")
    if c:
        crello.CACHE = Path(c)


def _lar(a, b):
    return abs(math.log(a / b))


def _roles(els, W, H):
    texts = [e for e in els if e["kind"] == "TextElement"]
    big = max((e["fs"] * max(1, len(e["text"])) ** 0.5 for e in texts), default=0)
    mx = max((e["fs"] for e in texts), default=0)
    for e in els:
        a = e["w"] * e["h"] / (W * H)
        k = e["kind"]
        if k == "TextElement":
            r = ("headline" if e["fs"] >= 0.85 * mx and mx > 0 else
                 "subhead" if e["fs"] >= 0.55 * mx else "small" if e["fs"] < 0.025 * max(W, H) else "body")
        elif a >= 0.85 and e["x"] <= 0.02 * W and e["y"] <= 0.02 * H:
            r = "bg"
        elif k == "ImageElement":
            r = "photo" if a >= 0.08 else "logo_or_art"
        elif k == "ColoredBackground":
            r = "bg"
        elif k == "SvgElement":
            r = "logo_or_art" if a < 0.08 else "shape"
        else:
            r = "shape"
        e["role"] = r
    return els


def member_record(row, tnames, aligns, fonts):
    W, H = int(row.canvas_width), int(row.canvas_height)
    els = []
    for i in range(int(row.length)):
        k = tnames[row.type[i]]
        e = dict(i=i, kind=k, x=round(float(row.left[i]), 1), y=round(float(row.top[i]), 1),
                 w=round(float(row.width[i]), 1), h=round(float(row.height[i]), 1),
                 angle=round(float(row.angle[i]), 2), opacity=round(float(row.opacity[i]), 3),
                 fs=0.0, text="", bold=False, color=None, align=None)
        if k == "TextElement":
            e["fs"] = round(float(row.font_size[i]), 1)
            e["text"] = str(row.text[i])
            cs = row.text_color[i]
            e["color"] = str(cs[0]) if len(cs) else None
            e["bold"] = bool(len(row.font_bold[i]) and row.font_bold[i][0])
            e["align"] = aligns[row.text_align[i]]
            e["font"] = fonts[row.font[i]]
        els.append(e)
    return dict(id=row.id, W=W, H=H, ar=round(W / H, 4), loc=[row.split, row.shard, int(row.rg)],
                n_el=len(els), els=_roles(els, W, H))


def cmd_index(a):
    _use_cache()
    feats = crello.features()
    tnames, aligns, fonts = (crello._names(feats, k) for k in ("type", "text_align", "font"))
    df = crello.load_meta()
    n_all = len(df)
    df["sig"] = df.text.map(sig_of)
    df = df[df.sig.notna()]
    df["ekey"] = [tuple(sorted(zip((tnames[t] for t in ty), (str(x or "").strip().lower() for x in tx))))
                  for ty, tx in zip(df.type, df.text)]
    df["has_photo"] = df.type.map(lambda ts: any(tnames[t] == "ImageElement" for t in ts))
    excl = sealed.exclusions("crello")
    stats = collections.Counter(templates=n_all, with_text_sig=len(df))
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT / "families.jsonl", "w") as fh:
        for (s, ek), g in df.groupby(["sig", "ekey"], sort=True):
            g = g.drop_duplicates(["canvas_width", "canvas_height"])
            if len(g) < 2:
                continue
            stats["families_2plus_sizes"] += 1
            h = sig_hash(s)
            if h in excl:
                stats["excluded_sealed"] += 1
                continue
            if len(g) > MAX_SIZES or g.length.max() > MAX_FAMILY_ELEMENTS:
                stats["skipped_too_big"] += 1
                continue
            mem = [member_record(r, tnames, aligns, fonts) for r in g.itertuples()]
            mem.sort(key=lambda m: m["ar"])
            fh.write(json.dumps(dict(fam=h, n_sizes=len(mem), photo=bool(g.has_photo.any()), members=mem)) + "\n")
            n += 1
            stats["written"] += 1
            stats["members"] += len(mem)
            stats[f"sizes_{min(len(mem), 6)}"] += 1
            stats["pairs"] += len(mem) * (len(mem) - 1)
    (OUT / "index_summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats), indent=1))


def _load_fams():
    return [json.loads(line) for line in open(OUT / "families.jsonl")]


def _pick_pair(f, rng):
    """Source = member nearest a 0.9 aspect (how most source posts look); target = the other
    member of the most different aspect, plus, when there are 3+ sizes, a second at random."""
    mem = f["members"]
    src = min(mem, key=lambda m: _lar(m["ar"], 0.9))
    others = sorted((m for m in mem if m is not src), key=lambda m: -_lar(m["ar"], src["ar"]))
    tg = [others[0]]
    if len(others) > 1:
        tg.append(rng.choice(others[1:]))
    return src, tg


def cmd_sample(a):
    rng = random.Random(a.seed)
    fams = _load_fams()
    strata = collections.defaultdict(list)
    for f in fams:
        src, tg = _pick_pair(f, rng)
        t = tg[0]
        ext = _lar(t["ar"], src["ar"])
        ar_cls = "wide" if t["ar"] >= 2.4 else "tall" if t["ar"] <= 0.62 else "far" if ext > 0.5 else "near"
        nb = "few" if src["n_el"] <= 5 else "mid" if src["n_el"] <= 10 else "many"
        strata[(ar_cls, f["photo"], nb)].append((f, src, tg))
    # allocation: equal share per stratum, wide and tall weighted x2 (the hard, scored shapes)
    keys = sorted(strata)
    w = {k: (2.0 if k[0] in ("wide", "tall") else 1.0) for k in keys}
    pool, picked = {k: list(v) for k, v in strata.items()}, []
    for k in keys:
        rng.shuffle(pool[k])
    tot = sum(w.values())
    quota = {k: max(1, round(a.n * w[k] / tot)) for k in keys}
    for k in keys:
        picked += pool[k][: quota[k]]
    rng.shuffle(picked)
    picked = picked[: a.n]
    out = [dict(fam=f["fam"], photo=f["photo"], src=s["id"], tgts=[t["id"] for t in tg],
                loc=sorted({tuple(m["loc"]) for m in [s] + tg})) for f, s, tg in picked]
    (OUT / "sample.json").write_text(json.dumps(out))
    print(len(out), "families;", "strata", {"/".join(map(str, k)): len(v) for k, v in strata.items()})
    print("sampled", collections.Counter((f["photo"]) for f in out), "row groups", len({tuple(l) for f in out for l in f["loc"]}))


def cmd_fetch(a):
    _use_cache()
    feats = crello.features()
    tnames, aligns, fonts = (crello._names(feats, k) for k in ("type", "text_align", "font"))
    sample = json.loads((OUT / "sample.json").read_text())
    need = {}
    for f in sample:
        for d in [f["src"]] + f["tgts"]:
            for loc in f["loc"]:
                need.setdefault(tuple(loc), set()).add(d)
    cases = OUT / "cases"
    assets = cases / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    designs = {}

    def one(key):
        t = crello._open(*key[:2]).read_row_group(key[2], columns=crello.LIGHT + ["image"]).to_pylist()
        return [r for r in t if r["id"] in need[key]]

    with ThreadPoolExecutor(4) as ex:
        for rows in ex.map(one, list(need)):
            for r in rows:
                designs[r["id"]] = crello._to_chameleon(r, tnames, aligns, fonts, assets)
            print("fetched", len(designs), flush=True)
    kept = 0
    for f in sample:
        if f["src"] not in designs or not all(t in designs for t in f["tgts"]):
            continue
        (cases / f"{f['fam']}.json").write_text(json.dumps(dict(
            family=f["fam"], photo=f["photo"], source_id=f["src"], target_ids=f["tgts"],
            source=designs[f["src"]], targets=[designs[t] for t in f["tgts"]])))
        kept += 1
    size = sum(p.stat().st_size for p in cases.rglob("*") if p.is_file())
    print("wrote", kept, "family cases,", round(size / 1e6), "MB")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    p = sub.add_parser("sample")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    sub.add_parser("fetch")
    a = ap.parse_args()
    {"index": cmd_index, "sample": cmd_sample, "fetch": cmd_fetch}[a.cmd](a)


if __name__ == "__main__":
    main()
