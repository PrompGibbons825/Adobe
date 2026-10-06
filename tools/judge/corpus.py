"""Pair corpus for calibrating the proxy judge (dev tool, not engine code).

    .venv/bin/python -m tools.judge.corpus cases            # cases + candidates -> output/judge/cands/
    .venv/bin/python -m tools.judge.corpus pairs --round 1  # choose pairs -> tools/judge/data/pairs.jsonl
    .venv/bin/python -m tools.judge.corpus measure          # proxy terms for every paired candidate
    .venv/bin/python -m tools.judge.corpus asks --round 1   # judge views + blind ask files + batches

Sources: the three samples, tools/cases/*.json, and the *tuning* half of the
Crello families (even index; the odd half stays held out for tools.evalset).
Targets: ten catalogue formats. Per case the candidates are the engine's top-k
plans (in the engine's own order, after its pixel re-rank and copy-over-copy
check), the safe fit-and-centre fallback, and controlled perturbations of the
engine's pick (Haraguchi-style): copy moved onto the photo, subhead shrunk,
copy misaligned by 2/8/20 px, logo hidden, headline cut at the edge, copy moved
to where its contrast is worst.

Every candidate is rendered by the official renderer and shown to the judge
downsampled to 512 or 1024 px on the long side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "output" / "judge"
DATA = Path(__file__).resolve().parent / "data"
KIT = WORK / "kit"
CRELLO = ROOT / "tools" / "crello" / "cases"

TARGETS = {
    "ig-square": (1080, 1080), "ig-portrait": (1080, 1350), "story": (1080, 1920), "yt-thumb": (1280, 720),
    "fb-link": (1200, 628), "li-banner": (1584, 396), "leaderboard": (728, 90), "med-rect": (300, 250),
    "skyscraper": (160, 600), "billboard": (970, 250),
}
PERTURBATIONS = ["p_onart", "p_shrinksub", "p_mis2", "p_mis8", "p_mis20", "p_hidelogo", "p_crop", "p_lowcon"]


def setup_kit() -> Path:
    """One brand-kit folder that holds both the Solstice assets and the Crello
    assets, so the engine, the renderer and the proxy all see the same files."""
    (KIT / "assets").mkdir(parents=True, exist_ok=True)
    for d in (ROOT / "brand_kit" / "assets", CRELLO / "assets"):
        if not d.is_dir():
            continue
        for f in d.iterdir():
            link = KIT / "assets" / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
    if not (KIT / "fonts").exists():
        (KIT / "fonts").symlink_to((ROOT / "brand_kit" / "fonts").resolve())
    os.environ["CHAMELEON_BRAND_KIT"] = str(KIT)
    os.environ["CHAMELEON_PROXY_ASSETS"] = str(KIT / "assets")
    return KIT


def sources(held: bool = False) -> Dict[str, dict]:
    """Calibration sources; held=True adds the odd (held-out) Crello families,
    used only for the blind check of the engine's picks, never for fitting."""
    out = {}
    for f in sorted((ROOT / "input").glob("*.json")):
        name = {"level1": "L1", "level2": "L2", "level3": "L3"}[f.stem.split("_")[2]]
        out[name] = json.loads(f.read_text())["source"]
    for f in sorted((ROOT / "tools" / "cases").glob("*.json")):
        out[f.stem] = json.loads(f.read_text())["source"]
    for f in sorted(CRELLO.glob("fam??.json")):
        if held or int(f.stem[3:5]) % 2 == 0:  # tuning half only
            out[f.stem] = json.loads(f.read_text())["source"]
    return out


# ----------------------------------------------------------------------------- candidates


def engine_candidates(source: dict, target: dict, k: int = 4) -> List[dict]:
    """The engine's top-k plans, finalized, in the order the engine prefers them.
    The first one that passes the copy-over-copy check is what resize() ships."""
    from engine.smart import Ctx, analyze, emit, finalize, generate, prune_relaxed, score_plans
    from engine.smart import _pixel_rerank
    from engine.smart.textfit import deliberate_overlaps, text_overlaps

    d = analyze(source)
    ctx = Ctx(W=float(target["width"]), H=float(target["height"]), W0=d.W, H0=d.H)
    plans = prune_relaxed(generate(d, ctx))
    score_plans(plans, d, ctx)
    plans.sort(key=lambda p: -p.score)
    _pixel_rerank(plans, d, ctx)
    exempt = frozenset(deliberate_overlaps(d.elements))
    rows = []
    for p in plans:
        els = emit(p, d, ctx)
        ok = not text_overlaps(els, exempt=exempt)
        out = finalize(source, {"canvas": target, "elements": els}, target)
        rows.append({"plan": p.name, "score": round(p.score, 3), "ok": ok, "out": out,
                     "breakdown": {k_: round(v, 3) for k_, v in p.breakdown.items()}})
    passing = [r for r in rows if r["ok"]]
    ordered = ([passing[0]] if passing else []) + [r for r in rows if not (passing and r is passing[0])]
    seen, uniq = set(), []
    for r in ordered:
        sig = json.dumps([[round(e["x"]), round(e["y"]), round(e["width"]), round(e["height"]),
                           e.get("font_size"), e.get("opacity", 1)] for e in r["out"]["elements"]])
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(r)
    return uniq[:k], bool(passing)


def fallback(source: dict, target: dict) -> dict:
    from engine.resize import _fallback
    from engine.smart import finalize

    return finalize(source, _fallback(source, target), target)


def _live(e: dict) -> bool:
    return float(e.get("opacity", 1) if e.get("opacity") is not None else 1) > 0 and e["width"] > 1 and e["height"] > 1


def _texts(out: dict) -> List[dict]:
    return sorted([e for e in out["elements"] if e["type"] == "text" and _live(e) and str(e.get("content", "")).strip()],
                  key=lambda e: -float(e.get("font_size", 0)))


def _images(out: dict) -> List[dict]:
    W, H = out["canvas"]["width"], out["canvas"]["height"]
    return sorted([e for e in out["elements"] if e["type"] == "image" and _live(e) and e["width"] * e["height"] < 0.7 * W * H],
                  key=lambda e: -(e["width"] * e["height"]))


def _drawn(e: dict):
    from engine.smart.textfit import drawn_box

    return drawn_box(e)


def _clone(out: dict) -> dict:
    return json.loads(json.dumps(out))


def _get(out: dict, i: str) -> dict:
    return next(e for e in out["elements"] if e["id"] == i)


def perturb(kind: str, out: dict, rng: random.Random) -> Optional[dict]:
    from engine.smart.geom import Box

    W, H = out["canvas"]["width"], out["canvas"]["height"]
    texts, imgs = _texts(out), _images(out)
    o = _clone(out)
    if kind == "p_onart":  # the headline moved onto the photo / product
        if not texts or not imgs:
            return None
        t, im = _get(o, texts[0]["id"]), imgs[0]
        db = _drawn(t)
        dx = (im["x"] + im["width"] / 2) - (db.x + db.w / 2)
        dy = (im["y"] + im["height"] / 2) - (db.y + db.h / 2)
        t["x"] = min(max(0.0, t["x"] + dx), W - t["width"])
        t["y"] = min(max(0.0, t["y"] + dy), H - t["height"])
        nb = _drawn(t)
        if nb.inter(Box(im["x"], im["y"], im["width"], im["height"])) < 0.3 * max(1.0, nb.area):
            return None
        return o
    if kind == "p_shrinksub":  # the second-largest copy shrunk to 55%
        if len(texts) < 2:
            return None
        t = _get(o, texts[1]["id"])
        t["font_size"] = max(1, int(round(float(t["font_size"]) * 0.55)))
        return o
    if kind.startswith("p_mis"):  # one aligned element nudged sideways
        n = float(kind[5:])
        cand = []
        live = [e for e in o["elements"] if _live(e) and e["width"] * e["height"] < 0.7 * W * H]
        boxes = {e["id"]: (_drawn(e) if e["type"] == "text" else Box(e["x"], e["y"], e["width"], e["height"])) for e in live}
        for t in texts:
            bt = boxes[t["id"]]
            for e in live:
                if e["id"] == t["id"]:
                    continue
                be = boxes[e["id"]]
                stacked = bt.b <= be.y or be.b <= bt.y
                if stacked and min(abs(bt.x - be.x), abs(bt.r - be.r), abs(bt.cx - be.cx)) <= 2.0:
                    cand.append(t["id"])
                    break
        tid = cand[0] if cand else (texts[0]["id"] if texts else None)
        if tid is None:
            return None
        t = _get(o, tid)
        sgn = 1 if t["x"] + t["width"] + n <= W else -1
        if t["x"] + sgn * n < 0:
            return None
        t["x"] += sgn * n
        return o
    if kind == "p_hidelogo":  # the smallest picture (usually the logo) disappears
        if not imgs or (len(imgs) < 2 and imgs[-1]["width"] * imgs[-1]["height"] > 0.08 * W * H):
            return None
        _get(o, imgs[-1]["id"])["opacity"] = 0.0
        return o
    if kind == "p_crop":  # 30% of the headline's ink pushed past the right edge
        if not texts:
            return None
        t = _get(o, texts[0]["id"])
        db = _drawn(t)
        t["x"] += (W - 0.7 * db.w) - db.x
        return o
    if kind == "p_lowcon":  # copy moved to where it reads worst
        from engine.smart import pixels

        if not texts or not pixels.available():
            return None
        arr, f = pixels.backdrop(o["elements"], W, H)
        if arr is None:
            return None
        others = {e["id"]: _drawn(e) for e in texts}
        best = None
        for t in texts[:3]:
            cur = pixels.text_stats(arr, f, t)
            if not cur:
                continue
            for gx in range(9):
                for gy in range(9):
                    x = (W - t["width"]) * gx / 8.0
                    y = (H - t["height"]) * gy / 8.0
                    m = dict(t, x=x, y=y)
                    db = _drawn(m)
                    if db.x < 0 or db.y < 0 or db.r > W or db.b > H:
                        continue
                    if any(db.inter(b) > 0 for i, b in others.items() if i != t["id"]):
                        continue
                    st = pixels.text_stats(arr, f, m)
                    if st and st[0] < 0.6 * cur[0] and (best is None or st[0] < best[0]):
                        best = (st[0], t["id"], x, y)
        if best is None:
            return None
        t = _get(o, best[1])
        t["x"], t["y"] = best[2], best[3]
        return o
    raise ValueError(kind)


def cmd_cases(a) -> None:
    setup_kit()
    sys.path.insert(0, str(ROOT))
    import contextlib
    import io

    rng = random.Random(a.seed)
    srcs = sources()
    cdir = WORK / "cands"
    cdir.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, src in srcs.items():
        n = a.per_crello if name.startswith("fam") else a.per_kit
        r0 = src["canvas"]["width"] / src["canvas"]["height"]
        names = list(TARGETS)
        rng.shuffle(names)
        # mostly real reshapes; keep at most one near-same-shape target
        picked, near = [], 0
        for tn in names:
            w, h = TARGETS[tn]
            close = abs(math.log((w / h) / r0)) < math.log(1.3)
            if close and near >= 1:
                continue
            near += close
            picked.append(tn)
            if len(picked) >= n:
                break
        for tn in picked:
            w, h = TARGETS[tn]
            tgt = {"width": w, "height": h}
            case = f"{name}@{tn}"
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                try:
                    eng, passing = engine_candidates(src, tgt)
                except Exception as ex:  # noqa: BLE001
                    print("engine failed", case, ex)
                    eng, passing = [], False
            cands = {}
            for j, r in enumerate(eng):
                cands[f"e{j}"] = dict(r, out=None)
                (cdir / f"{case}#e{j}.json").write_text(json.dumps(r["out"]))
            fb = fallback(src, tgt)
            (cdir / f"{case}#fb.json").write_text(json.dumps(fb))
            cands["fb"] = {"plan": "fallback"}
            if not passing:  # resize() ships the fallback: it is the pick
                cands["fb"]["shipped"] = True
            base = json.loads((cdir / f"{case}#e0.json").read_text()) if passing else fb
            for kind in PERTURBATIONS:
                with contextlib.redirect_stderr(buf):
                    p = perturb(kind, base, rng)
                if p is not None:
                    (cdir / f"{case}#{kind}.json").write_text(json.dumps(p))
                    cands[kind] = {"plan": kind}
            index[case] = {"source": name, "target": tn, "canvas": [w, h], "passing": passing,
                           "cands": {k: {kk: vv for kk, vv in v.items() if kk != "out"} for k, v in cands.items()}}
            print(case, sorted(cands))
    (WORK / "cases.json").write_text(json.dumps(index, indent=1))
    print(len(index), "cases;", sum(len(v["cands"]) for v in index.values()), "candidates")


# ----------------------------------------------------------------------------- pairs


def _pid(a: str, b: str) -> str:
    return hashlib.sha1(f"{a}|{b}".encode()).hexdigest()[:10]


def load_pairs() -> List[dict]:
    p = DATA / "pairs.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def cmd_pairs(a) -> None:
    cases = json.loads((WORK / "cases.json").read_text())
    rng = random.Random(1000 + a.round)
    have = load_pairs()
    used = {(p["a"], p["b"]) for p in have} | {(p["b"], p["a"]) for p in have}
    new: List[dict] = []

    def add(case: str, x: str, y: str, kind: str) -> bool:
        ca, cb = f"{case}#{x}", f"{case}#{y}"
        if (ca, cb) in used:
            return False
        used.add((ca, cb))
        used.add((cb, ca))
        new.append({"pair": _pid(ca, cb), "case": case, "source": cases[case]["source"], "a": ca, "b": cb,
                    "kind": kind, "round": a.round, "view": rng.choice([512, 1024])})
        return True

    names = sorted(cases)
    rng.shuffle(names)
    if a.round == 1:
        quota = {"eng": 50, "fallback": 20, "pert": 62, "mix": 10}
        pert_quota = {k: quota["pert"] // len(PERTURBATIONS) + 1 for k in PERTURBATIONS}
        n = {k: 0 for k in quota}
        for case in names:
            c = cases[case]["cands"]
            pick = "e0" if cases[case]["passing"] else "fb"
            if n["eng"] < quota["eng"] and "e1" in c and add(case, "e0", "e1", "eng"):
                n["eng"] += 1
            if n["fallback"] < quota["fallback"] and pick == "e0" and rng.random() < 0.35 and add(case, "e0", "fb", "fallback"):
                n["fallback"] += 1
            perts = [k for k in PERTURBATIONS if k in c and pert_quota[k] > 0]
            rng.shuffle(perts)
            for k in perts[:2]:
                if n["pert"] < quota["pert"] and add(case, pick, k, "pert:" + k):
                    pert_quota[k] -= 1
                    n["pert"] += 1
            if n["mix"] < quota["mix"] and rng.random() < 0.15:
                pool = [k for k in c if k != pick]
                if len(pool) >= 2:
                    x, y = rng.sample(pool, 2)
                    if add(case, x, y, "mix"):
                        n["mix"] += 1
    else:
        # round-robin cases (proper within-case Kendall tau) plus more engine pairs
        rr = 0
        for case in names:
            c = cases[case]["cands"]
            if not cases[case]["passing"] or "e1" not in c or rr >= a.round_robin:
                continue
            perts = [k for k in PERTURBATIONS if k in c and k not in ("p_mis2",)]
            if not perts:
                continue
            members = ["e0", "e1", "fb", rng.choice(perts)]
            added = 0
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    added += add(case, members[i], members[j], "rr")
            rr += 1
        extra = 0
        for case in names:
            c = cases[case]["cands"]
            if extra >= a.extra:
                break
            for x, y in (("e0", "e2"), ("e1", "e2"), ("e0", "e3")):
                if x in c and y in c and extra < a.extra and add(case, x, y, "eng"):
                    extra += 1
    with (DATA / "pairs.jsonl").open("a") as f:
        for p in new:
            f.write(json.dumps(p) + "\n")
    from collections import Counter

    print(len(new), "new pairs", Counter(p["kind"].split(":")[0] for p in new), Counter(p["kind"] for p in new))


# ----------------------------------------------------------------------------- measurement


def _measure_one(args):
    cid, path, task_path, old = args
    setup_kit()
    sys.path.insert(0, str(ROOT))
    import contextlib
    import io

    from tools.judge import terms as T

    out = {"old": old} if old else {}
    with contextlib.redirect_stderr(io.StringIO()):
        design = json.loads(Path(path).read_text())
        out["rep"] = T.measure(design)
        if task_path and not old:
            from tools import proxy

            out["old"] = proxy.aggregate(path, task_path)
    return cid, out


def cmd_measure(a) -> None:
    setup_kit()
    from concurrent.futures import ProcessPoolExecutor

    cases = json.loads((WORK / "cases.json").read_text())
    srcs = sources()
    rdir = WORK / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    tdir = WORK / "tasks"
    tdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    need = set()
    for p in load_pairs():
        need.add(p["a"])
        need.add(p["b"])
    def cached_old(cid):
        f = rdir / f"{cid}.json"
        return json.loads(f.read_text()).get("old") if f.exists() else None

    for cid in sorted(need):
        if (rdir / f"{cid}.json").exists() and not a.refresh:
            continue
        case = cid.split("#")[0]
        task = tdir / f"{cases[case]['source']}.task.json"
        if not task.exists():
            task.write_text(json.dumps({"source": srcs[cases[case]["source"]]}))
        jobs.append((cid, str(WORK / "cands" / f"{cid}.json"), str(task), cached_old(cid)))
    for name in sorted({cases[c.split("#")[0]]["source"] for c in need}):
        if not (rdir / f"SRC_{name}.json").exists() or a.refresh:
            sp = tdir / f"{name}.src.json"
            sp.write_text(json.dumps(srcs[name]))
            jobs.append((f"SRC_{name}", str(sp), None, None))
    print(len(jobs), "to measure")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, (cid, res) in enumerate(ex.map(_measure_one, jobs, chunksize=2)):
            (rdir / f"{cid}.json").write_text(json.dumps(res))
            if k % 25 == 0:
                print(k, cid, flush=True)


def box_report(cid: str) -> dict:
    """What the engine itself can measure for a candidate (or a source, cid
    "SRC_<name>"): element geometry plus the pixel stats under each text."""
    from engine.smart import judge, pixels

    f = WORK / "reports" / f"box_{cid}.json"
    if f.exists():
        return json.loads(f.read_text())
    if cid.startswith("SRC_"):
        design = sources()[cid[4:]]
    else:
        design = json.loads((WORK / "cands" / f"{cid}.json").read_text())
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    els = design["elements"]
    hidden = {e["id"] for e in els if e.get("opacity") is not None and float(e["opacity"]) <= 0}
    rep = judge.box_report(els, W, H, pixels.measure(els, W, H, skip=hidden))
    f.write_text(json.dumps(rep))
    return rep


def features(pair: dict) -> dict:
    """Term vectors (render-measured and box-measured) and the old proxy for both sides of a pair."""
    from tools.judge import terms as T

    rdir = WORK / "reports"
    src = json.loads((rdir / f"SRC_{pair['source']}.json").read_text())["rep"]
    bsrc = box_report(f"SRC_{pair['source']}")
    res = {}
    for side in ("a", "b"):
        r = json.loads((rdir / f"{pair[side]}.json").read_text())
        res[side] = {"terms": T.terms(src, r["rep"]), "box": T.terms(bsrc, box_report(pair[side])),
                     "old": r.get("old", {}).get("score")}
    return res


# ----------------------------------------------------------------------------- asks


def _views(design: dict, cid: str) -> Dict[int, Path]:
    """The candidate as the judge sees it: the official render (native size, max
    2048 like engine/main.py), downsampled with Lanczos to 512 and 1024 px on the
    long side. The native render is not kept (disk)."""
    from PIL import Image

    from tools.judge import terms as T

    vdir = WORK / "views"
    vdir.mkdir(parents=True, exist_ok=True)
    out = {v: vdir / f"{cid}@{v}.png" for v in (512, 1024)}
    if all(p.exists() for p in out.values()):
        return out
    tmp = vdir / "_native.png"
    T.render_png(design, tmp)
    im = Image.open(tmp).convert("RGB")
    for v, p in out.items():
        f = min(1.0, v / max(im.size))
        iv = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS) if f < 1 else im
        iv.save(p, optimize=True)
    tmp.unlink()
    return out


def cmd_asks(a) -> None:
    setup_kit()
    sys.path.insert(0, str(ROOT))

    srcs = sources(held=True)
    rng = random.Random(2000 + a.round)
    pairs = [p for p in load_pairs() if p["round"] == a.round]
    adir = WORK / "asks"
    adir.mkdir(parents=True, exist_ok=True)

    def view(cid: str, v: int) -> Path:
        design = json.loads((WORK / "cands" / f"{cid}.json").read_text()) if "#" in cid else srcs[cid]
        return _views(design, cid)[v]

    def link(src: Path, dest: Path) -> None:
        # a hard link: no extra disk, and nothing in the ask folder names the candidate
        if dest.exists():
            dest.unlink()
        os.link(src, dest)

    asks = json.loads((WORK / "asks.json").read_text()) if (WORK / "asks.json").exists() else {}
    fresh = []
    for p in pairs:
        if any(v["pair"] == p["pair"] for v in asks.values()):
            continue
        first = rng.random() < 0.5
        for order in ((p["a"], p["b"]) if first else (p["b"], p["a"]), (p["b"], p["a"]) if first else (p["a"], p["b"])):
            aid = "".join(rng.choice("0123456789abcdef") for _ in range(8))
            link(view(p["source"], 512), adir / f"{aid}_ref.png")
            link(view(order[0], p["view"]), adir / f"{aid}_A.png")
            link(view(order[1], p["view"]), adir / f"{aid}_B.png")
            asks[aid] = {"pair": p["pair"], "A": order[0], "B": order[1], "view": p["view"], "round": a.round}
            fresh.append(aid)
    (WORK / "asks.json").write_text(json.dumps(asks, indent=1))
    # batches: the two orders of a pair never land in the same batch
    by_pair: Dict[str, List[str]] = {}
    for aid in fresh:
        by_pair.setdefault(asks[aid]["pair"], []).append(aid)
    firsts = [v[0] for v in by_pair.values()]
    seconds = [v[1] for v in by_pair.values() if len(v) > 1]
    rng.shuffle(firsts)
    rng.shuffle(seconds)
    bdir = WORK / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    k0 = len(list(bdir.glob("batch_*.json")))
    k = k0
    for group in (firsts, seconds):
        for i in range(0, len(group), a.batch):
            (bdir / f"batch_{k:02d}.json").write_text(json.dumps(group[i:i + a.batch]))
            k += 1
    print(len(fresh), "asks in", k - k0, "batches", f"(batch_{k0:02d}..batch_{k - 1:02d})")


def cmd_check(a) -> None:
    """Blind check of an engine change: for every Crello family whose pick differs
    between two tools.evalset tags, one pair (round 3, kind "check"), never used
    for fitting."""
    ev = ROOT / "tools" / "crello" / "eval"
    t0, t1 = a.tags.split(",")
    rng = random.Random(3000)
    have = {p["pair"] for p in load_pairs()}
    new = []
    for f in sorted((ev / t0).glob("fam??.json")):
        o0, o1 = json.loads(f.read_text()), json.loads((ev / t1 / f.name).read_text())
        if o0 == o1:
            continue
        case = f"{f.stem}@eval"
        for tag, o in ((t0, o0), (t1, o1)):
            (WORK / "cands" / f"{case}#{tag}.json").write_text(json.dumps(o))
        ca, cb = f"{case}#{t0}", f"{case}#{t1}"
        rec = {"pair": _pid(ca, cb), "case": case, "source": f.stem, "a": ca, "b": cb, "kind": "check",
               "round": a.round, "view": rng.choice([512, 1024]),
               "split": "tune" if int(f.stem[3:5]) % 2 == 0 else "held"}
        if rec["pair"] not in have:
            new.append(rec)
    with (DATA / "pairs.jsonl").open("a") as fh:
        for r in new:
            fh.write(json.dumps(r) + "\n")
    print(len(new), "check pairs:", [r["source"] for r in new])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cases")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--per-kit", type=int, default=4)
    p.add_argument("--per-crello", type=int, default=2)
    p = sub.add_parser("pairs")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--round-robin", type=int, default=10)
    p.add_argument("--extra", type=int, default=40)
    p = sub.add_parser("measure")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--refresh", action="store_true", help="re-measure the terms (the old proxy stays cached)")
    p = sub.add_parser("asks")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--batch", type=int, default=24)
    p = sub.add_parser("check")
    p.add_argument("--tags", required=True, help="two tools.evalset tags, e.g. base,judge")
    p.add_argument("--round", type=int, default=3)
    a = ap.parse_args()
    {"cases": cmd_cases, "pairs": cmd_pairs, "measure": cmd_measure, "asks": cmd_asks, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
