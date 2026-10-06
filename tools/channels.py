"""Dev/demo tool: one design -> every channel, consistent and explained.

    .venv/bin/python -m tools.channels sheet input/chameleon_sample_level3_advanced.json -o output/channels.png
    .venv/bin/python -m tools.channels sheet <task.json> --independent        # each target's own winner
    .venv/bin/python -m tools.channels metric --set tune --limit 8 --weights 0.5,1,2
    .venv/bin/python -m tools.channels budget --set samples --ms 1500

`sheet` renders the source plus the ~20-format catalogue (tools/sheet.py) as a
contact sheet; each tile says its archetype and its top two decisions, the
footer the family consistency with and without the joint pick. `metric`
measures consistency and what it costs in score; `budget` measures time and
quality under a per-target budget. Candidates are cached per design under
output/channels/ so a weight sweep does not re-plan.

Sets: samples = input/*.json + tools/cases/*.json; tune / held = the even /
odd Crello families (tools/crello/cases, `--limit` of them, evenly spaced).
Not part of the engine or the Docker image.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tempfile
import time
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CRELLO = ROOT / "tools" / "crello" / "cases"
OUT = ROOT / "output" / "channels"


def _is_crello(path: Path) -> bool:
    return CRELLO in path.resolve().parents


def designs(which: str, limit: int):
    if which == "samples":
        return sorted((ROOT / "input").glob("*.json")) + sorted((ROOT / "tools" / "cases").glob("*.json"))
    fams = sorted(CRELLO.glob("fam??.json"))
    fams = [f for f in fams if (int(f.stem[3:5]) % 2 == 0) == (which == "tune")]
    if limit and len(fams) > limit:
        step = len(fams) / limit
        fams = [fams[int(i * step)] for i in range(limit)]
    return fams


def _setup(paths) -> None:
    if any(_is_crello(Path(p)) for p in paths):
        os.environ["CHAMELEON_BRAND_KIT"] = str(CRELLO)


def _source(path: Path) -> dict:
    return json.loads(path.read_text())["source"]


def catalogue():
    from tools.sheet import CATALOGUE

    return dict(CATALOGUE)


# ----------------------------------------------------------------------------
# rendering


_UNICODE = False  # does the label font have arrows (→)?


def _font(size: int):
    from PIL import ImageFont

    global _UNICODE
    for p in (Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"), Path("/Library/Fonts/Arial Unicode.ttf"),
              Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if p.exists():
            _UNICODE = True
            return ImageFont.truetype(str(p), size)
    for f in ("SolsticeSans-Regular.ttf", "SolsticeSans-Bold.ttf"):
        p = ROOT / "brand_kit" / "fonts" / f
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _txt(s: str) -> str:
    return s if _UNICODE else s.replace("→", "->").replace("×", "x")


def render(design: dict, crello: bool, box: int):
    from PIL import Image

    from renderer.converter.convert import render_design_to_png

    assets = CRELLO / "assets" if crello else ROOT / "brand_kit" / "assets"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r.png"
        with redirect_stderr(io.StringIO()):
            render_design_to_png(design, assets, p, fonts_dir=ROOT / "brand_kit" / "fonts")
        im = Image.open(p).convert("RGB")
        im.load()
    f = min(box / im.width, box / im.height)
    return im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)


def _wrap(draw, text: str, font, width: int):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def tile(im, head: str, sub: str, notes, cell: int, box: int):
    from PIL import Image, ImageDraw

    f_head, f_sub, f_note = _font(15), _font(13), _font(12)
    head, sub, notes = _txt(head), _txt(sub), [_txt(n) for n in notes]
    t = Image.new("RGB", (cell, 1), (255, 255, 255))
    d = ImageDraw.Draw(t)
    lines = []
    for n in notes:
        for i, ln in enumerate(_wrap(d, n, f_note, cell - 22)[:2]):
            lines.append(("• " if i == 0 else "  ") + ln)
    text_h = 44 + 16 * len(lines)
    t = Image.new("RGB", (cell, box + text_h + 14), (250, 250, 248))
    d = ImageDraw.Draw(t)
    t.paste(im, ((cell - im.width) // 2, 8 + (box - im.height) // 2))
    y = box + 14
    d.text((10, y), head, fill=(20, 20, 20), font=f_head)
    d.text((10, y + 19), sub, fill=(90, 90, 90), font=f_sub)
    for i, ln in enumerate(lines):
        d.text((10, y + 40 + 16 * i), ln, fill=(40, 40, 40), font=f_note)
    return t


def grid(tiles, cols: int, footer: str = ""):
    from PIL import Image, ImageDraw

    cell = tiles[0].width
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    hs = [max(t.height for t in r) for r in rows]
    width = cols * (cell + 6) + 6
    font = _font(15)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    flines = [ln for part in footer.split("\n") for ln in _wrap(probe, _txt(part), font, width - 24)] if footer else []
    foot = 12 + 20 * len(flines) if flines else 0
    sheet = Image.new("RGB", (width, sum(h + 6 for h in hs) + 6 + foot), (214, 214, 210))
    y = 6
    for r, h in zip(rows, hs):
        for i, t in enumerate(r):
            sheet.paste(t, (6 + i * (cell + 6), y))
        y += h + 6
    d = ImageDraw.Draw(sheet)
    for i, ln in enumerate(flines):
        d.text((12, y + 4 + 20 * i), ln, fill=(20, 20, 20), font=font)
    return sheet


def _fmt_metric(m: dict) -> str:
    def f(k):
        v = m.get(k)
        return "n/a" if v is None else f"{v:.2f}"
    return (f"similarity {f('similarity')}, logo same class {f('logo_same_class')}, reading order {f('reading_order')}, "
            f"art same side {f('hero_same_side')}, alignment {f('same_alignment')}, type-ratio sd {f('type_ratio_log_sd')}")


def cmd_sheet(a) -> int:
    from engine.api import resize_many

    path = Path(a.design)
    _setup([path])
    crello = _is_crello(path)
    src = _source(path)
    cat = catalogue()
    if a.only:
        cat = {k: v for k, v in cat.items() if k in a.only.split(",")}
    t0 = time.time()
    res = resize_many(src, {k: {"width": w, "height": h} for k, (w, h) in cat.items()}, k=a.k,
                      consistency=not a.independent, budget=a.ms or None)
    dt = time.time() - t0
    tiles = [tile(render(src, crello, a.box), f"SOURCE {src['canvas']['width']}×{src['canvas']['height']}",
                  path.stem[-34:], [f"family style: " + ", ".join(f"{k} {v}" for k, v in res["style"].items())],
                  a.cell, a.box)]
    for name, cand in res["results"].items():
        W, H = cand["design"]["canvas"]["width"], cand["design"]["canvas"]["height"]
        pick = res["choice"][name]
        sub = f"{cand['archetype']} · {cand['strategy']}" + (f" · pick #{pick}" if pick > 1 else "")
        notes = [h for h in cand.get("highlights", [])[1:3]]
        tiles.append(tile(render(cand["design"], crello, a.box), f"{name}  {W}×{H}", sub, notes, a.cell, a.box))
    ind, joint = res["consistency"]["independent"], res["consistency"]["joint"]
    footer = (f"{'Independent' if a.independent else 'Joint (family-consistent)'} picks: {len(cat)} formats in "
              f"{dt:.0f}s.\nIndependent: {_fmt_metric(ind)}." + ("" if a.independent else f"\nJoint: {_fmt_metric(joint)}."))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid(tiles, a.cols, footer).save(out)
    print(out, f"{dt:.1f}s")
    print("independent", ind)
    print("joint      ", joint)
    return 0


# ----------------------------------------------------------------------------
# metric


def _cached(path: Path, k: int, budget=None) -> dict:
    """name -> list of slim candidates (score, archetype, strategy, roles, design)."""
    from engine.api import candidates

    key = f"{path.stem}_k{k}" + (f"_b{budget}" if budget else "")
    cp = OUT / "cache" / f"{key}.json"
    if cp.exists():
        return json.loads(cp.read_text())
    src = _source(path)
    per = {}
    for name, (w, h) in catalogue().items():
        cs = candidates(src, {"width": w, "height": h}, k=k, budget=budget)
        per[name] = [{kk: c.get(kk) for kk in ("score", "archetype", "strategy", "roles", "design", "highlights")}
                     for c in cs]
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(per))
    return per


def _anchored_planner(path: Path, k: int):
    """candidates() for family_pick's anchored re-plans, cached per design."""
    from engine.api import candidates

    cp = OUT / "cache" / f"{path.stem}_k{k}_anchor.json"
    memo = json.loads(cp.read_text()) if cp.exists() else {}

    def plan(source, tgt, cons, k=k, budget=None):
        key = f"{tgt['width']}x{tgt['height']}|" + json.dumps(cons, sort_keys=True)
        if key not in memo:
            cs = candidates(source, tgt, cons, k=k, budget=budget)
            memo[key] = [{kk: c.get(kk) for kk in ("score", "archetype", "strategy", "roles", "design", "highlights")}
                         for c in cs]
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(memo))
        return memo[key]

    return plan


def cmd_metric(a) -> int:
    from engine.api import family_pick
    from engine.smart import consistency as cs

    paths = designs(a.set, a.limit)
    _setup(paths)
    targets = {n: {"width": w, "height": h} for n, (w, h) in catalogue().items()}
    ws = [float(w) for w in a.weights.split(",")]
    modes = [("indep", 0.0, False)] + [(f"w={w:g}", w, False) for w in ws] + [(f"w={w:g}+anc", w, True) for w in ws]
    rows = {m[0]: [] for m in modes}
    for p in paths:
        t = time.time()
        per = _cached(p, a.k)
        src = _source(p)
        planner = _anchored_planner(p, a.k)
        for label, w, anc in modes:
            if label == "indep":
                pool, pick = per, {n: 0 for n in per}
            else:
                with redirect_stderr(io.StringIO()):
                    pool, pick, _ = family_pick(src, per, targets, k=a.k, weight=w, anchor=anc, plan=planner)
            m = cs.metric({n: pool[n][pick[n]] for n in pool})
            loss = []
            for n in per:
                s0, s = per[n][0]["score"], pool[n][pick[n]]["score"]
                if s0 is not None and s is not None:
                    loss.append((s0 - s) / max(1e-6, abs(s0)))
            m["switched"] = sum(1 for n in per if pool[n][pick[n]]["design"] != per[n][0]["design"]) / len(per)
            m["loss_mean"] = sum(loss) / max(1, len(loss))
            m["loss_max"] = max(loss, default=0.0)
            rows[label].append(m)
        print(f"  {p.stem}: {time.time() - t:.1f}s", file=sys.stderr)
    keys = ["similarity", "logo_same_class", "reading_order", "hero_same_side", "same_alignment",
            "type_ratio_log_sd", "switched", "loss_mean", "loss_max"]
    print(f"set={a.set} designs={len(paths)} formats={len(catalogue())} k={a.k}  "
          f"(loss = share of the target's own best score given up; negative = the anchored option scored higher)")
    print("mode".ljust(11) + "".join(k[:13].rjust(14) for k in keys))
    for label, ms in rows.items():
        line = label.ljust(11)
        for k in keys:
            vals = [m[k] for m in ms if m.get(k) is not None]
            line += (f"{sum(vals) / len(vals):14.3f}" if vals else "n/a".rjust(14))
        print(line)
    return 0


# ----------------------------------------------------------------------------
# budget


def cmd_budget(a) -> int:
    from engine.api import candidates
    from engine.smart import textfit

    paths = designs(a.set, a.limit)
    _setup(paths)
    budget = {"ms": a.ms} if a.ms else {"work": a.work}
    rows = []
    for p in paths:
        src = _source(p)
        for name, (w, h) in catalogue().items():
            tgt = {"width": w, "height": h}
            t = time.time()
            w0 = textfit.measure_calls()
            full = candidates(src, tgt, k=a.k)
            t_full, work = time.time() - t, textfit.measure_calls() - w0
            t = time.time()
            fast = candidates(src, tgt, k=a.k, budget=budget)
            t_fast = time.time() - t
            same = json.dumps(full[0]["design"], sort_keys=True) == json.dumps(fast[0]["design"], sort_keys=True)
            s0, s1 = full[0]["score"], fast[0]["score"]
            rows.append((p.stem, name, t_full, t_fast, work, same, s0, s1, fast[0]["budget"]["hit"]))
            if a.verbose:
                print(f"  {p.stem[:24]:24s} {name:14s} full {t_full:5.2f}s fast {t_fast:5.2f}s work {work:7d} "
                      f"same={same} {s0} {s1}", file=sys.stderr)

    def pct(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(q * len(v)))]

    tf, tb = [r[2] for r in rows], [r[3] for r in rows]
    hit = [r for r in rows if r[8]]
    changed = [r for r in rows if not r[5]]
    rel = [(r[6] - r[7]) / abs(r[6]) for r in changed if r[6] is not None and r[7] is not None and r[6]]
    print(f"set={a.set} designs={len(paths)} targets={len(rows)} budget={budget}")
    print(f"time full: p50 {pct(tf, .5):.2f}s p95 {pct(tf, .95):.2f}s max {max(tf):.2f}s total {sum(tf):.0f}s")
    print(f"time budget: p50 {pct(tb, .5):.2f}s p95 {pct(tb, .95):.2f}s max {max(tb):.2f}s total {sum(tb):.0f}s")
    print(f"budget hit on {len(hit)} targets; top-1 changed on {len(changed)}; "
          f"score loss when changed: mean {100 * sum(rel) / max(1, len(rel)):.1f}% max {100 * max(rel, default=0):.1f}%")
    work_rate = sum(r[4] for r in rows) / max(1e-6, sum(tf))
    print(f"text measurements per second (full runs): {work_rate:.0f}")
    for r in changed:
        print(f"  changed: {r[0]} {r[1]} {r[6]} -> {r[7]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sheet")
    p.add_argument("design")
    p.add_argument("-o", "--out", default=str(OUT / "sheet.png"))
    p.add_argument("--independent", action="store_true")
    p.add_argument("--only", default="")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--ms", type=float, default=0.0, help="per-target budget in ms (0 = none)")
    p.add_argument("--cell", type=int, default=300)
    p.add_argument("--box", type=int, default=260)
    p.add_argument("--cols", type=int, default=6)
    p = sub.add_parser("metric")
    p.add_argument("--set", default="samples", choices=["samples", "tune", "held"])
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--weights", default="1,2,4")
    p = sub.add_parser("budget")
    p.add_argument("--set", default="samples", choices=["samples", "tune", "held"])
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--ms", type=float, default=0.0)
    p.add_argument("--work", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    return {"sheet": cmd_sheet, "metric": cmd_metric, "budget": cmd_budget}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
