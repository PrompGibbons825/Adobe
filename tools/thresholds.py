"""Dev tool: fit the engine's hand-set thresholds on designer-made retargets.

Two sources, both restricted to *training* families (never the 32 held-out
regression families, never the "held2" families of tools/crello_train.py):

  meta   designers' own behaviour, from the light Crello columns of every
         multi-size family (no images needed, ~1,000 training families):
           * type scale relative to sqrt(target area / source area), by target
             aspect -> AREA_GAIN, STRIP_R, STRIP_BOOST
           * how often the designer kept the source arrangement, by the size
             of the aspect change -> ADAPT_TOL (keep vs reflow)
  tune   the engine's own choice replayed on the dumped candidates
         (tools/ranker_sim.py): coordinate search over the archetype gates and
         the prior's knees for the best mean designer similarity of the pick
           -> STACK_MAX_R, SPLIT_MIN_R, ROW_MIN_R, ADAPT_TOL, ADAPT_PRIOR,
              STACK_KNEE, SPLIT_PEAK, ROW_KNEE

    .venv/bin/python -m tools.thresholds meta
    .venv/bin/python -m tools.thresholds tune      # needs tools/crello/ranker/*.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BINS = [0.0, 0.4, 0.62, 0.85, 1.2, 1.7, 2.4, 3.5, 5.0, 7.0, 1e9]


def _bin(r: float) -> str:
    for lo, hi in zip(BINS, BINS[1:]):
        if lo <= r < hi:
            return f"{lo:g}-{hi:g}" if hi < 1e8 else f">={lo:g}"
    return "?"


def _held_sigs():
    from tools.crello import ROOT as CR
    from tools.crello_train import split_of_sig

    old = json.loads((CR / "families.json").read_text())
    held = {f["sig"] for k, f in enumerate(old) if k % 2 == 1}
    tune = {f["sig"] for k, f in enumerate(old) if k % 2 == 0}
    return held, tune, split_of_sig


def _meta_design(row) -> dict:
    els = []
    for i in range(int(row["length"])):
        x, y, w, h = (float(row[k][i]) for k in ("left", "top", "width", "height"))
        t = "text" if row["text"][i] and str(row["text"][i]).strip() else "image"
        e = dict(id=f"e{i}", type=t, x=x, y=y, width=max(1.0, w), height=max(1.0, h))
        if t == "text":
            e.update(content=str(row["text"][i]), font_size=float(row["font_size"][i]))
        els.append(e)
    return {"canvas": {"width": int(row["canvas_width"]), "height": int(row["canvas_height"])}, "elements": els}


def pairs():
    """(source design, target design) for every training family, source = the
    member closest to 0.9 (as in tools/crello.py), every other size a target."""
    from tools.crello import load_meta

    held, tune, split_of_sig = _held_sigs()
    df = load_meta()
    df["ar"] = df.canvas_width / df.canvas_height

    def sig(texts):
        t = sorted(str(s).strip().lower() for s in texts if s and str(s).strip())
        return "\x1f".join(t) if len(t) >= 2 else None

    df["sig"] = df.text.map(sig)
    df = df[df.sig.notna()]
    for s, g in df.groupby("sig"):
        if s in held or (s not in tune and split_of_sig(s) != "train"):
            continue
        g = g.drop_duplicates(["canvas_width", "canvas_height"])
        if len(g) < 2:
            continue
        src = g.loc[(g.ar.map(lambda r: abs(math.log(r / 0.9)))).idxmin()]
        sd = _meta_design(src)
        for _, t in g[g.id != src.id].iterrows():
            yield s, sd, _meta_design(t)


def cmd_meta(_a):
    from tools.designer_sim import _box, _giou01, _is_bg, _rel, match
    from engine.smart import thresholds as T

    fs_rows, keep_rows = [], []
    fams = set()
    for s, sd, td in pairs():
        fams.add(s)
        W0, H0 = sd["canvas"]["width"], sd["canvas"]["height"]
        W, H = td["canvas"]["width"], td["canvas"]["height"]
        r, r0 = W / H, W0 / H0
        la = 0.5 * math.log(W * H / (W0 * H0))
        m = match(sd["elements"], td["elements"])
        src_by = {e["id"]: e for e in sd["elements"]}
        ks = [math.log(de["font_size"] / src_by[sid]["font_size"]) - la for sid, de in m.items()
              if de.get("type") == "text" and src_by[sid].get("font_size", 0) > 0 and de.get("font_size", 0) > 0]
        if ks:
            fs_rows.append((r, r0, statistics.median(ks)))
        # did the designer keep the source arrangement?
        bx = [(_box(src_by[sid], W0, H0), _box(de, W, H)) for sid, de in m.items()]
        bx = [(a, b) for a, b in bx if not _is_bg(a) and not _is_bg(b)]
        if len(bx) >= 3:
            gi = sum(_giou01(a, b) for a, b in bx) / len(bx)
            agree = tot = 0
            for i in range(len(bx)):
                for j in range(i + 1, len(bx)):
                    ra, rb = _rel(bx[i][0], bx[j][0]), _rel(bx[i][1], bx[j][1])
                    agree += (ra[0] == rb[0]) + (ra[1] == rb[1])
                    tot += 2
            keep_rows.append((abs(math.log(r / r0)), gi, agree / tot))
    print(f"training families {len(fams)}  pairs with matched text {len(fs_rows)}  pairs for keep/reflow {len(keep_rows)}")

    # --- type scale vs sqrt(area), by target aspect ------------------------------
    by = collections.defaultdict(list)
    for r, r0, k in fs_rows:
        by[_bin(r)].append(k)
    print("\ndesigner type scale / sqrt(area ratio), by target aspect (median, IQR, n)")
    for lo, hi in zip(BINS, BINS[1:]):
        b = f"{lo:g}-{hi:g}" if hi < 1e8 else f">={lo:g}"
        v = sorted(by.get(b, []))
        if v:
            q = lambda p: math.exp(v[int(p * (len(v) - 1))])  # noqa: E731
            print(f"  r {b:>9}  x{q(0.5):.2f}  [{q(0.25):.2f}, {q(0.75):.2f}]  n={len(v)}")

    def strip(r, sr):
        return r >= sr or r <= 1.0 / sr

    base = [k for r, _, k in fs_rows if not strip(r, 3.0)]
    gain = math.exp(statistics.median(base))
    best = None
    for sr in (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0):
        ins = [k for r, _, k in fs_rows if strip(r, sr)]
        if len(ins) < 15:
            continue
        boost = math.exp(statistics.median(ins) - math.log(gain))
        err = sum(abs(k - math.log(gain) - (math.log(boost) if strip(r, sr) else 0.0)) for r, _, k in fs_rows)
        print(f"  strip >= {sr:.1f}:1  boost x{boost:.2f}  n={len(ins)}  mean |log err| {err / len(fs_rows):.4f}")
        if best is None or err < best[0]:
            best = (err, sr, boost)
    old_err = sum(abs(k - (math.log(1.4) if strip(r, 6.0) else 0.0)) for r, _, k in fs_rows) / len(fs_rows)
    print(f"  old: gain x{T.AREA_GAIN:.2f}, strip >= {T.STRIP_R}:1 boost x{T.STRIP_BOOST}  "
          f"mean |log err| {old_err:.4f}")
    print(f"  fit: gain x{gain:.2f} (median off-strip), strip >= {best[1]}:1 boost x{best[2]:.2f}  "
          f"mean |log err| {best[0] / len(fs_rows):.4f}")

    # --- keep vs reflow -----------------------------------------------------------
    print("\nkept the source arrangement, by |log(r / r0)|:  'same' = mean GIoU of normalised boxes"
          " >= 0.85 and >= 90% of pairwise relations;  'relations' = >= 90% of pairwise relations")
    edges = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 1.0, 1.5, 10]
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        sel = [(gi, ag) for d, gi, ag in keep_rows if lo <= d < hi]
        if not sel:
            continue
        same = sum(1 for gi, ag in sel if gi >= 0.85 and ag >= 0.9) / len(sel)
        kept = sum(1 for gi, ag in sel if ag >= 0.9) / len(sel)
        rows.append((lo, hi, kept, len(sel)))
        print(f"  {lo:.2f}-{hi:.2f}  (x{math.exp(lo):.2f}-x{math.exp(hi):.2f})  same {same:4.0%}  relations {kept:4.0%}  "
              f"mean GIoU {statistics.mean(g for g, _ in sel):.2f}  n={len(sel)}")
    # boundary: where the kept share crosses 50%, interpolated between bin centres
    cross = None
    for (a0, a1, ka, _), (b0, b1, kb, _) in zip(rows, rows[1:]):
        if ka >= 0.5 > kb:
            ca, cb = (a0 + a1) / 2, (b0 + b1) / 2
            cross = ca + (ka - 0.5) / (ka - kb) * (cb - ca)
            break
    if cross is not None:
        print(f"  old ADAPT_TOL x{T.ADAPT_TOL}  (|log| {math.log(T.ADAPT_TOL):.3f});  designers switch at "
              f"|log| {cross:.3f} -> ADAPT_TOL x{math.exp(cross):.2f}")


def cmd_tune(a):
    from tools import ranker_sim as S
    from engine.smart import thresholds as T

    rows = [r for r in S.load("old", "train") if r["split"] in ("train", "tune") and r["cands"]]
    held = [r for r in S.load("old", "train") if r["split"] in ("held", "held2")]
    grid = dict(
        ADAPT_TOL=[1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0],
        ADAPT_PRIOR=[0.0, 20.0, 40.0, 60.0, 90.0, 130.0],
        STACK_MAX_R=[1.4, 1.7, 2.0, 2.2, 2.6, 3.2, 4.0, 1e9],
        SPLIT_MIN_R=[0.8, 1.0, 1.15, 1.3, 1.5, 1.8],
        ROW_MIN_R=[1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0],
        STACK_KNEE=[0.9, 1.0, 1.1, 1.25, 1.5, 1.8, 2.2],
        SPLIT_PEAK=[1.4, 1.7, 2.0, 2.4, 3.0],
        ROW_KNEE=[2.4, 3.0, 3.6, 4.5, 6.0],
    )
    fixed = [k for k in (a.fix or "").split(",") if k]
    for k in fixed:
        grid.pop(k)
    t = {k: getattr(T, k) for k in grid}
    base = S.evaluate(rows, t)["sim"]
    print(f"training groups {len(rows)}  start (hand-set) mean designer similarity {base:.4f}")
    cur = base
    for rnd in range(a.rounds):
        moved = False
        for k, vals in grid.items():
            res = []
            for v in vals:
                tt = dict(t, **{k: v})
                res.append((S.evaluate(rows, tt)["sim"], v))
            s, v = max(res, key=lambda x: (round(x[0], 4), -abs(math.log(x[1] / t[k])) if t[k] > 0 and x[1] > 0 else 0))
            if s > cur + 1e-4 and v != t[k]:
                print(f"  round {rnd + 1}: {k} {t[k]:g} -> {v:g}   {cur:.4f} -> {s:.4f}")
                t[k], cur, moved = v, s, True
        if not moved:
            break
    old = {k: getattr(T, k) for k in grid}
    hb = S.evaluate(held, old)
    ht = S.evaluate(held, t)
    print("\nthreshold   hand-set   fitted")
    for k in grid:
        print(f"  {k:<12} {old[k]:>8g} {t[k]:>8g}")
    print(f"\ntrain mean designer similarity  {base:.4f} -> {cur:.4f}")
    print(f"held-out (held + held2, n={hb['n']})  {hb['sim']:.4f} -> {ht['sim']:.4f}   oracle {hb['oracle']:.4f} -> {ht['oracle']:.4f}")
    (S.DATA / "thresholds_fit.json").write_text(json.dumps(dict(old=old, new=t, train=[base, cur],
                                                                held=[hb["sim"], ht["sim"]]), indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("meta")
    p = sub.add_parser("tune")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--fix", default="", help="comma-separated thresholds to keep at their current value")
    a = ap.parse_args()
    {"meta": cmd_meta, "tune": cmd_tune}[a.cmd](a)


if __name__ == "__main__":
    main()
