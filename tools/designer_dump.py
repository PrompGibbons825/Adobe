"""Dump main's ranked candidates per dev task, for fitting the designer term (tools/designer_weights.py).

    .venv/bin/python -m tools.designer_dump --shard 0 --nshards 48 \
        --cases /data/agents/work/crello-wide/cases --cases /data/agents/work/designer/cases-extra \
        --out /data/agents/work/designer/dumps

Tasks: every (family source -> designer's other size) in the cases dirs, *minus* the tournament
families (/data/agents/work/designer/tournament_families.txt). A family goes to shard
md5(family) % nshards. One JSON line per task in <out>/dump_<shard>of<nshards>.jsonl:

    task, family, target, features (decisions.source_features), ref_decisions (the designer's),
    main_pick (name of the plan plan_design actually leads with), pixels (pixel critic ran),
    cands: the top N plans in hand-score order, each with name, kind, key (the _diverse dedupe key,
    hashed), score, breakdown, core_hidden, passes (_promote_passing's check), pix (the pixel
    critic's penalty), decisions (decisions_of on its finalized output), agree (key -> confidence
    if it matches the tree's prediction else 0), sim (tools.bench.similarity to the designer's).

The pipeline replayed is plan_design's up to the sort, with the designer term *off*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOURNAMENT = Path("/data/agents/work/designer/tournament_families.txt")
N_DUMP = 24


def shard_of(fam: str, n: int) -> int:
    return int(hashlib.md5(fam.encode()).hexdigest()[:8], 16) % n


def load_tasks(dirs):
    bad = set(TOURNAMENT.read_text().split()) if TOURNAMENT.exists() else set()
    tasks = {}
    for dr in dirs:
        for p in sorted(Path(dr).glob("*.json")):
            try:
                f = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if f.get("family") in bad or "targets" not in f:
                continue
            for k, ref in enumerate(f["targets"]):
                tasks[f"{f['family']}_{k}"] = dict(source=f["source"], target=dict(ref["canvas"]), reference=ref,
                                                   family=f["family"], kit=str(Path(dr)))
    return tasks


def dump_task(t: dict) -> dict:
    from engine.smart import _promote_passing, emit, finalize, pixels  # noqa: F401
    from engine.smart.analyze import analyze
    from engine.smart import decisions as dm
    from engine.smart.compose import display_room, generate
    from engine.smart.designer import core_hidden
    from engine.smart.items import Ctx, forget_realized
    from engine.smart.score import pixel_penalty, prune_relaxed, score_plans
    from engine.smart.textfit import drawn_box, new_text_overlaps
    from engine.smart.geom import Box
    from tools.bench import similarity

    source, target, ref = t["source"], t["target"], t["reference"]
    feats = dm.source_features(source, target)
    pred = dm.predict(feats)
    row = dict(features=feats, ref_decisions=dm.decisions_of(source, ref),
               pred={k: list(v) for k, v in pred.items()})
    W, H = float(target["width"]), float(target["height"])
    canvas = Box(-0.5, -0.5, W + 1.0, H + 1.0)
    for lockups in (True, False):
        forget_realized()
        d = analyze(source, lockups=lockups)
        ctx = Ctx(W=W, H=H, W0=d.W, H0=d.H, display_room=display_room(d, W, H))
        plans = prune_relaxed(generate(d, ctx))
        if not plans:
            if not any(it.lockup for it in d.items):
                break
            continue
        score_plans(plans, d, ctx)
        plans.sort(key=lambda p: -p.score)
        src_px = pixels.measure(d.elements, d.W, d.H, skip=set()) if pixels.available() else None
        cands = []
        for p in plans[:N_DUMP]:
            raw = emit(p, d, ctx)
            out = finalize(source, {"canvas": target, "elements": raw}, target)
            els = out["elements"]
            shown = [e for e in els if e.get("type") == "text" and float(e.get("opacity", 1)) > 0
                     and str(e.get("content", "")).strip()]
            ok = not any(drawn_box(e).inter(canvas) < drawn_box(e).area - 1.0 for e in shown) \
                and not new_text_overlaps(els, d.elements)
            pix = 0.0
            if src_px:
                imp = {e["id"]: r.item.importance for r, _, _ in p.placed + p.floaters for e in r.item.els}
                hidden = {e["id"] for it in p.hidden for e in it.els}
                pix = pixel_penalty(pixels.measure(raw, ctx.W, ctx.H, skip=hidden), src_px, imp)
            key = tuple(sorted((r.item.key, round(x), round(y), round(r.w), round(r.h))
                               for r, x, y in p.placed + p.floaters))
            dec = dm.decisions_of(source, out)
            cands.append(dict(
                name=p.name, kind=p.name.split("@")[0].split("~")[0],
                key=hashlib.md5(repr(key).encode()).hexdigest()[:12],
                score=p.score, breakdown=p.breakdown, core_hidden=core_hidden(p), passes=ok, pix=pix,
                decisions=dec,
                agree={k: (float(c) if dec.get(k) == v else 0.0) for k, (v, c) in pred.items() if k in dec},
                sim=similarity(out, ref)))
        if any(c["passes"] for c in cands) or not any(it.lockup for it in d.items):
            row.update(lockups=lockups, pixels=bool(src_px), n_plans=len(plans), cands=cands)
            break
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--cases", action="append", default=[])
    ap.add_argument("--out", default="/data/agents/work/designer/dumps")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dirs = [c for c in (a.cases or ["/data/agents/work/crello-wide/cases"]) if Path(c).is_dir()]
    tasks = load_tasks(dirs)
    names = [n for n in sorted(tasks) if shard_of(tasks[n]["family"], a.nshards) == a.shard]
    if a.limit:
        names = names[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"dump_{a.shard:03d}of{a.nshards}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                done.add(json.loads(line)["task"])
            except Exception:  # noqa: BLE001
                pass
    os.environ["CHAMELEON_OFF"] = ",".join(filter(None, [os.environ.get("CHAMELEON_OFF", ""), "designer"]))
    from engine.smart import paths
    from engine.smart import plan_design
    t_all = time.time()
    with path.open("a") as fh:
        for n in names:
            if n in done:
                continue
            t = tasks[n]
            os.environ["CHAMELEON_BRAND_KIT"] = t["kit"]
            paths.assets_dirs.cache_clear() if hasattr(paths.assets_dirs, "cache_clear") else None
            t0 = time.time()
            try:
                row = dump_task(t)
                try:
                    _, _, plans = plan_design(t["source"], t["target"])
                    row["main_pick"] = plans[0].name
                except Exception:  # noqa: BLE001
                    row["main_pick"] = None
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row = dict(error=repr(e))
            row.update(task=n, family=t["family"], target=t["target"], secs=round(time.time() - t0, 2))
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    print(f"shard {a.shard}/{a.nshards}: {len(names)} tasks in {time.time() - t_all:.0f}s -> {path}")


if __name__ == "__main__":
    main()
