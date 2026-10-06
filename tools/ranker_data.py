"""Dev tool: every candidate the engine generates, labelled by the designer.

For each (family, designer target) the engine is run with its archetype gates
opened wide, so that one dump can replay any gate setting later
(tools/ranker_train.py, tools/thresholds.py). Per candidate it records:

  * what the hand score needs to be recomputed for a subset of candidates
    (per-item scales for the "visible" term, the inputs of the archetype
    prior, relax level and hidden core items for prune_relaxed);
  * the ranker features (engine/smart/ranker.py);
  * the pixel penalty the engine's pixel re-rank would add, and whether the
    candidate passes the no-copy-over-copy hard check;
  * its similarity to the designer's version (tools/designer_sim.py), the label.

    .venv/bin/python -m tools.ranker_data --set old     # the 64 regression families
    .venv/bin/python -m tools.ranker_data --set train   # tools/crello/train cases

Output: tools/crello/ranker/<set>.jsonl, one line per (family, target).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRELLO = ROOT / "tools" / "crello"
OUT = CRELLO / "ranker"

# gates wide open: a superset of the candidates any sane setting would generate
OPEN = dict(STACK_MAX_R=1e9, SPLIT_MIN_R=0.8, ROW_MIN_R=1.2, ADAPT_TOL=2.0)


def _jobs(which: str):
    if which == "old":
        for f in sorted((CRELLO / "cases").glob("fam??.json")):
            k = int(f.stem[3:])
            yield str(f), 0, ("tune" if k % 2 == 0 else "held"), f.stem
    else:
        for f in sorted((CRELLO / "train" / "cases").glob("t???.json")):
            fam = json.loads(f.read_text())
            for i in range(len(fam["targets"])):
                yield str(f), i, fam["split"], f.stem


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
    from engine.smart import thresholds as T
    from engine.smart.analyze import analyze
    from engine.smart.compose import generate
    from engine.smart.items import Ctx
    from engine.smart.score import _core_hidden, pixel_penalty, score_plans
    from engine.smart.textfit import deliberate_overlaps, drawn_box, text_overlaps
    from tools.designer_sim import match, similarity

    t0 = time.time()
    fam = json.loads(Path(path).read_text())
    src, des = fam["source"], fam["targets"][ti]
    target = {"width": des["canvas"]["width"], "height": des["canvas"]["height"]}
    mapping = match(src["elements"], des["elements"])
    row = dict(fam=fam_id, ti=ti, split=split, W=target["width"], H=target["height"],
               W0=src["canvas"]["width"], H0=src["canvas"]["height"], n_el=len(src["elements"]),
               n_match=len(mapping))
    fb = finalize(src, _fallback(src, target), target)
    row["fallback"] = similarity(fb["elements"], des, src, mapping, text_box=drawn_box)
    saved = T.snapshot()
    try:
        for k, v in OPEN.items():
            setattr(T, k, v)
        d = analyze(src)
        ctx = Ctx(W=float(target["width"]), H=float(target["height"]), W0=d.W, H0=d.H)
        plans = generate(d, ctx)
        score_plans(plans, d, ctx)
    except Exception as e:  # the engine would fall back here too
        row["error"] = repr(e)
        row["cands"] = []
        return row
    finally:
        for k, v in saved.items():
            setattr(T, k, v)
    feats = ranker.all_features(plans, d, ctx)
    src_px = pixels.measure(d.elements, d.W, d.H, skip=set()) if pixels.available() else {}
    exempt = frozenset(deliberate_overlaps(d.elements))
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
            name=p.name, relax=p.relax, core_hidden=_core_hidden(p), s=p.s,
            items=[(r.item.key, r.item.kind, r.item.importance, r.scale) for r, _, _ in p.placed + p.floaters],
            hidden_imp=-bd.get("hidden", 0.0), rest=sum(v for k, v in bd.items() if k not in ("visible", "prior")),
            breakdown=bd, feats=f, pix=pix, ok=not text_overlaps(out, exempt=exempt),
            label=similarity(fin, des, src, mapping, text_box=drawn_box)))
    row["cands"] = cands
    row["secs"] = time.time() - t0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["old", "train"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="start over instead of resuming")
    a = ap.parse_args()
    kit = str(CRELLO / "cases") if a.set == "old" else str(CRELLO / "train")
    jobs = list(_jobs(a.set))
    if a.limit:
        jobs = jobs[: a.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.set}.jsonl"
    done, keep = set(), []
    if out.exists() and not a.fresh:  # resume: keep complete lines, redo the rest
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            done.add((r["fam"], r["ti"]))
            keep.append(line)
        out.write_text("".join(x + "\n" for x in keep))
    jobs = [j for j in jobs if (j[3], j[1]) not in done]
    print("groups to do:", len(jobs), "already done:", len(done), flush=True)
    t0 = time.time()
    n = 0
    with open(out, "a") as fh, ProcessPoolExecutor(a.workers, initializer=_init, initargs=(kit,)) as ex:
        for row in ex.map(one, jobs, chunksize=2):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n += 1
            if n % 25 == 0:
                print(f"{n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    print("wrote", n, "groups to", out, f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
