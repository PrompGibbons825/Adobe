"""Dump engine candidates with designer-similarity labels (dev only).

    .venv/bin/python -m tools.ranker_data [--workers 4]

Output: tools/crello/ranker/old.jsonl — one line per (family, target).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRELLO = ROOT / "tools" / "crello"
OUT = CRELLO / "ranker"


def _jobs():
    for f in sorted((CRELLO / "cases").glob("fam??.json")):
        k = int(f.stem[3:])
        yield str(f), 0, ("tune" if k % 2 == 0 else "held"), f.stem


def _init(kit: str):
    os.environ["CHAMELEON_BRAND_KIT"] = kit
    os.environ["CHAMELEON_RANKER"] = "0"
    sys.path.insert(0, str(ROOT))


def one(job):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return _one(job)


def _one(job):
    path, ti, split, fam_id = job
    from engine.resize import _fallback
    from engine.smart import emit, finalize, pixels, ranker
    from engine.smart.analyze import analyze
    from engine.smart.compose import generate, plan_scaled
    from engine.smart.items import Ctx
    from engine.smart.score import _core_hidden, pixel_penalty, score_plans
    from engine.smart.textfit import drawn_box, new_text_overlaps
    from tools.designer_sim import match, similarity

    t0 = time.time()
    fam = json.loads(Path(path).read_text())
    src, des = fam["source"], fam["targets"][ti]
    target = {"width": des["canvas"]["width"], "height": des["canvas"]["height"]}
    mapping = match(src["elements"], des["elements"])
    row = dict(
        fam=fam_id, ti=ti, split=split, W=target["width"], H=target["height"],
        W0=src["canvas"]["width"], H0=src["canvas"]["height"], n_el=len(src["elements"]),
        n_match=len(mapping),
    )
    fb = finalize(src, _fallback(src, target), target)
    row["fallback"] = similarity(fb["elements"], des, src, mapping, text_box=drawn_box)
    try:
        d = analyze(src)
        ctx = Ctx(W=float(target["width"]), H=float(target["height"]), W0=d.W, H0=d.H)
        plans = []
        for settle in (True, False):
            plans.extend(generate(d, ctx, settle=settle))
        seen = set()
        uniq = []
        for p in plans:
            key = (p.name, round(p.s, 4), p.relax, tuple(sorted(h.key for h in p.hidden)))
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        plans = uniq
        sp = plan_scaled(d, ctx, [it for it in d.items if it.role in ("missing", "void")])
        if sp:
            plans.append(sp)
        score_plans(plans, d, ctx)
    except Exception as e:
        row["error"] = repr(e)
        row["cands"] = []
        return row
    feats = ranker.all_features(plans, d, ctx)
    src_px = pixels.measure(d.elements, d.W, d.H, skip=set()) if pixels.available() else {}
    cands = []
    for p, f in zip(plans, feats):
        out = emit(p, d, ctx)
        fin = finalize(src, {"elements": out}, target)["elements"]
        imp = {}
        for r, _, _ in p.placed + p.floaters:
            for e in r.item.els:
                imp[e["id"]] = r.item.importance
        hidden = {e["id"] for it in p.hidden for e in it.els}
        pix = 0.0
        if src_px:
            pix = pixel_penalty(pixels.measure(out, ctx.W, ctx.H, skip=hidden), src_px, imp)
        bd = dict(p.breakdown)
        cands.append(dict(
            name=p.name, relax=p.relax, core_hidden=_core_hidden(p), s=p.s, hand=p.score,
            items=[(r.item.key, r.item.kind, r.item.importance, r.scale) for r, _, _ in p.placed + p.floaters],
            hidden_imp=-bd.get("hidden", 0.0), breakdown=bd, feats=f, pix=pix,
            ok=not new_text_overlaps(out, d.elements),
            label=similarity(fin, des, src, mapping, text_box=drawn_box),
        ))
    row["cands"] = cands
    row["secs"] = time.time() - t0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    kit = str(CRELLO / "cases")
    jobs = list(_jobs())
    if a.limit:
        jobs = jobs[: a.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "old.jsonl"
    done, keep = set(), []
    if out.exists() and not a.fresh:
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            done.add((r["fam"], r["ti"]))
            keep.append(line)
    todo = [j for j in jobs if (j[3], j[1]) not in done]
    print(f"{len(done)} done, {len(todo)} to run")
    if todo:
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(kit,)) as ex:
            for row in ex.map(one, todo, chunksize=1):
                keep.append(json.dumps(row, default=str))
                print(row["fam"], len(row.get("cands", [])), f"{row.get('secs', 0):.1f}s")
    out.write_text("\n".join(keep) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
