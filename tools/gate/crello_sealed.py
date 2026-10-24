"""Gate tool: build a sealed-crello version into a staging folder, ready for `sealed add`.

    .venv/bin/python -m tools.gate.crello_sealed pick  --out /data/agents/work/g2/new_families.json
    .venv/bin/python -m tools.gate.crello_sealed build --stage /data/agents/work/g2/stage \\
        --new /data/agents/work/g2/new_families.json
    .venv/bin/python -m tools.gate.sealed add /data/agents/work/g2/stage --set crello

A family is one Crello text signature (see tools/crello.py): a source design plus the
designer's own remakes at other sizes. A sealed item is one (source, target canvas) pair with
the designer's remake as the reference. Only families where the designer kept the same
element set (the same multiset of (type, text) elements) are used, and the target is the
most extreme aspect change among those remakes.

Version 1 is made of
  * the held2 families of the learned-ranker split (~/chameleon-data/crello-train, 1 in 5 by
    signature hash), and
  * new families picked from the Crello metadata cache, excluding every signature already
    used: the 64 dev families (tools/crello/families.json), the 600 train/held2 families, and
    every signature already in the sealed store (sealed.exclusions).

Item and asset names are opaque (i000.json, <sha256>.png) so the committed manifest says
nothing about which designs are in the set. Nothing is printed about individual items.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools import crello
from tools.gate import sealed

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO / "tools" / "crello" / "cache"
MAIN_CACHE = Path.home() / "code" / "adobe-hackathon-2026" / "chameleon" / "tools" / "crello" / "cache"
TRAIN_DATA = Path.home() / "chameleon-data" / "crello-train"
MIN_EXT = 0.25  # |log(target ar / source ar)|: below this a "resize" is barely a reshape


def sig_of(texts) -> str | None:
    t = sorted(str(s).strip().lower() for s in texts if s and str(s).strip())
    return "\x1f".join(t) if len(t) >= 2 else None


def sig_hash(sig: str) -> str:
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:24]


def _lar(a: float, b: float) -> float:
    return abs(math.log(a / b))


def _use_cache(path: Path) -> None:
    crello.CACHE = path


def used_signatures() -> set:
    old = {f["sig"] for f in json.loads((REPO / "tools" / "crello" / "families.json").read_text())}
    train = {json.loads(p.read_text())["sig"] for p in (TRAIN_DATA / "cases").glob("*.json")}
    return old | train


# ---- element-set check on converted designs ----------------------------------------------

def _key(e: dict):
    return (e["type"], str(e.get("content", "")).strip().lower())


def same_element_set(a: dict, b: dict) -> bool:
    return sorted(map(_key, a["elements"])) == sorted(map(_key, b["elements"]))


def remap_ids(src: dict, ref: dict) -> dict:
    """Give the designer's remake the source's element ids (same key, in z order)."""
    pool = collections.defaultdict(list)
    for e in sorted(src["elements"], key=lambda e: e.get("z_index", 0)):
        pool[_key(e)].append(e["id"])
    out = json.loads(json.dumps(ref))
    for e in sorted(out["elements"], key=lambda e: e.get("z_index", 0)):
        e["id"] = pool[_key(e)].pop(0)
    return out


def _ar(d: dict) -> float:
    return d["canvas"]["width"] / d["canvas"]["height"]


# ---- pick new families from the metadata cache -------------------------------------------

def cmd_pick(a):
    _use_cache(Path(a.cache))
    feats = crello.features()
    tnames = crello._names(feats, "type")
    df = crello.load_meta()
    df["ar"] = df.canvas_width / df.canvas_height
    df["sig"] = df.text.map(sig_of)
    df = df[df.sig.notna()]
    df["has_photo"] = df.type.map(lambda ts: "ImageElement" in [tnames[t] for t in ts])
    df["ekey"] = [tuple(sorted(zip((tnames[t] for t in ty), (str(x or "").strip().lower() for x in tx))))
                  for ty, tx in zip(df.type, df.text)]
    used = used_signatures()
    used_h = {sig_hash(s) for s in used} | sealed.exclusions("crello")
    rg = lambda r: (r.split, r.shard, int(r.rg))  # noqa: E731
    fams = []
    for s, g in df.groupby("sig"):
        if s in used or sig_hash(s) in used_h:
            continue
        g = g.drop_duplicates(["canvas_width", "canvas_height"])
        g = g[g.length <= 25]
        if len(g) < 2 or len(g) > 8:
            continue
        src = g.loc[(g.ar.map(lambda r: _lar(r, 0.9))).idxmin()]
        tg = g[(g.id != src.id) & g.ekey.map(lambda k: k == src.ekey)]
        tg = tg[tg.ar.map(lambda r: _lar(r, src.ar)) >= MIN_EXT]
        if tg.empty:
            continue
        t = tg.loc[tg.ar.map(lambda r: _lar(r, src.ar)).idxmax()]
        ext = _lar(t.ar, src.ar)
        fams.append(dict(sig=s, src=src.id, tgt=t.id, src_ar=float(src.ar), tgt_ar=float(t.ar), ext=ext,
                         groups=sorted({rg(src), rg(t)}), photo=bool(src.has_photo), n_el=int(src.length)))
    print("candidate families (unused, same element set, reshaped):", len(fams))
    # greedy over row groups (each ~57 MB of images): complete the most (weighted) families per download
    chosen, picked, left = set(), [], list(fams)
    while len(picked) < a.n and left:
        gain = collections.Counter()
        for f in left:
            need = {tuple(x) for x in f["groups"]} - chosen
            w = 1.0 + min(f["ext"], 2.0) + (0.3 if f["photo"] else 0.0)
            for x in need:
                gain[x] += w / len(need)
        chosen.add(gain.most_common(1)[0][0])
        done = [f for f in left if {tuple(x) for x in f["groups"]} <= chosen]
        picked += done
        ids = {id(f) for f in done}
        left = [f for f in left if id(f) not in ids]
    picked = picked[: a.n]
    ext = [f["ext"] for f in picked]
    print(f"row groups {len(chosen)}  families {len(picked)}  ext>0.5 {sum(e > 0.5 for e in ext)}  "
          f"ext>1 {sum(e > 1 for e in ext)}  photo {sum(f['photo'] for f in picked)}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(picked))


def fetch_new(fams: list, cache: Path, work: Path, threads: int) -> dict:
    """Download and convert the picked designs -> {design id: chameleon design}; assets in work/assets."""
    _use_cache(cache)
    feats = crello.features()
    tnames, aligns, fonts = (crello._names(feats, k) for k in ("type", "text_align", "font"))
    meta = crello.load_meta().set_index("id")
    need = collections.defaultdict(set)
    for f in fams:
        for d in (f["src"], f["tgt"]):
            r = meta.loc[d]
            need[(r.split, r.shard, int(r.rg))].add(d)
    assets = work / "assets"
    designs = work / "designs"
    assets.mkdir(parents=True, exist_ok=True)
    designs.mkdir(parents=True, exist_ok=True)
    todo = [k for k in need if not all((designs / f"{d}.json").exists() for d in need[k])]
    print("row groups to read:", len(todo), "of", len(need), flush=True)

    def one(key):
        split, fn, g = key
        rows = crello._open(split, fn).read_row_group(g, columns=crello.LIGHT + ["image"]).to_pylist()
        for row in rows:
            if row["id"] in need[key]:
                d = crello._to_chameleon(row, tnames, aligns, fonts, assets)
                (designs / f"{row['id']}.json").write_text(json.dumps(d))
        return key

    with ThreadPoolExecutor(threads) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            print(f"row group {i + 1}/{len(todo)}", flush=True)
    out = {}
    for f in fams:
        for d in (f["src"], f["tgt"]):
            p = designs / f"{d}.json"
            if p.exists():
                out[d] = json.loads(p.read_text())
    return out


# ---- build the staging folder ------------------------------------------------------------

def _copy_assets(design: dict, src_dir: Path, dst_dir: Path, names: dict) -> bool:
    """Copy every referenced asset under an opaque content-hash name. False if one is missing."""
    for e in design["elements"]:
        if e["type"] != "image":
            continue
        a = e["asset_id"]
        if a not in names:
            p = src_dir / a
            if not p.exists():
                return False
            data = p.read_bytes()
            n = hashlib.sha256(data).hexdigest()[:20] + Path(a).suffix
            if not (dst_dir / n).exists():
                (dst_dir / n).write_bytes(data)
            names[a] = n
    return True


def _rename(design: dict, names: dict) -> dict:
    d = json.loads(json.dumps(design))
    for e in d["elements"]:
        if e["type"] == "image":
            e["asset_id"] = names[e["asset_id"]]
    return d


def cmd_build(a):
    stage = Path(a.stage)
    if stage.exists() and any(stage.iterdir()):
        raise SystemExit(f"{stage} is not empty")
    work = Path(a.work)
    (stage / "items").mkdir(parents=True, exist_ok=True)
    (stage / "assets").mkdir(parents=True, exist_ok=True)
    pairs = []  # (origin, sig, source, reference, asset dir)
    drop = collections.Counter()

    # held2: the learned-ranker's held-out fifth
    for p in sorted((TRAIN_DATA / "cases").glob("*.json")):
        fam = json.loads(p.read_text())
        if fam["split"] != "held2":
            continue
        src = fam["source"]
        ok = [t for t in fam["targets"] if same_element_set(src, t) and _lar(_ar(t), _ar(src)) > 0.0]
        if not ok:
            drop["held2: designer changed the element set"] += 1
            continue
        ref = max(ok, key=lambda t: _lar(_ar(t), _ar(src)))
        pairs.append(("held2", fam["sig"], src, ref, TRAIN_DATA / "assets"))

    # new families
    if a.new:
        fams = json.loads(Path(a.new).read_text())
        designs = fetch_new(fams, Path(a.cache), work, a.threads)
        for f in fams:
            src, ref = designs.get(f["src"]), designs.get(f["tgt"])
            if src is None or ref is None:
                drop["new: not fetched"] += 1
                continue
            if not same_element_set(src, ref):
                drop["new: element set differs after conversion"] += 1
                continue
            pairs.append(("new", f["sig"], src, ref, work / "assets"))

    rng = random.Random(a.seed)
    rng.shuffle(pairs)
    names: dict = {}
    sigs, exts, counts, k = [], [], collections.Counter(), 0
    for origin, sig, src, ref, adir in pairs:
        local: dict = {}
        if not (_copy_assets(src, adir, stage / "assets", local) and _copy_assets(ref, adir, stage / "assets", local)):
            drop[f"{origin}: missing asset"] += 1
            continue
        names.update(local)
        s, r = _rename(src, local), remap_ids(src, _rename(ref, local))
        tid = f"i{k:03d}"
        k += 1
        item = {"task_id": tid, "source": s, "target_canvas": dict(r["canvas"]), "reference": r,
                "meta": {"origin": origin, "src_ar": _ar(s), "tgt_ar": _ar(r), "ext": _lar(_ar(r), _ar(s)),
                         "n_el": len(s["elements"]),
                         "photo": any(e["type"] == "image" for e in s["elements"][1:])}}
        (stage / "items" / f"{tid}.json").write_text(json.dumps(item))
        sigs.append(sig_hash(sig))
        counts[origin] += 1
        exts.append(item["meta"]["ext"])
    info = {"set": "crello", "items": k, "by_origin": dict(counts), "dropped": dict(drop),
            "ext_gt_0.5": sum(e > 0.5 for e in exts), "ext_gt_1": sum(e > 1 for e in exts),
            "builder": "tools/gate/crello_sealed.py", "seed": a.seed}
    (stage / "set.json").write_text(json.dumps(info, indent=1))
    (stage / "exclude.json").write_text(json.dumps(sorted(sigs)))
    print(json.dumps(info))
    if a.clean_work:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    cache = str(DEFAULT_CACHE if (DEFAULT_CACHE / "info.json").exists() else MAIN_CACHE)
    p = sub.add_parser("pick")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=160)
    p.add_argument("--cache", default=cache)
    p = sub.add_parser("build")
    p.add_argument("--stage", required=True)
    p.add_argument("--work", default="/data/agents/work/g2-crello-fetch")
    p.add_argument("--new", default="")
    p.add_argument("--cache", default=cache)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean-work", action="store_true")
    a = ap.parse_args()
    {"pick": cmd_pick, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
