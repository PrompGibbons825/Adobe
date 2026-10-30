"""Property test: thousands of random text elements, model vs the official renderer.

    python -m tests.conformance.prop --n 2000 --seed 1

For each element: (1) the model's wrapped string equals the converter's, (2) the model's ink box
is within TOL px of the drawn pixels' box. Prints the mismatch rate per kind.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import time
from typing import List, Optional

import numpy as np

from fundamentals import measure_text
from fundamentals.fonts import fontbook
from fundamentals.render import Scene, mask_box

TOL = 1.5
CW, CH, PAD = 1600, 1600, 300
ALPHA = string.ascii_letters + string.digits + ".,!?'-:%&$@" + "éüñ"


def _word(rng: random.Random) -> str:
    n = max(1, int(rng.expovariate(1 / 5)))
    w = "".join(rng.choice(ALPHA) for _ in range(n))
    return w.upper() if rng.random() < 0.15 else w


def random_text(rng: random.Random) -> dict:
    words = [_word(rng) for _ in range(rng.randint(1, 14))]
    content = ""
    for i, w in enumerate(words):
        sep = "" if i == 0 else rng.choices([" ", "  ", "\n"], [0.9, 0.04, 0.06])[0]
        content += sep + w
    size = rng.choice([rng.randint(6, 120), round(rng.uniform(6, 120), 2)])
    e = {
        "id": "t", "type": "text", "x": 100.0 + rng.random() * 20, "y": 100.0 + rng.random() * 20,
        "width": round(rng.uniform(15, 900), 1), "height": round(rng.uniform(10, 300), 1),
        "content": content, "font_size": size, "color": "rgba(0,0,0,1)",
        "font_family": rng.choices(["Solstice Sans", "Solstice Serif", "Montserrat"], [0.5, 0.4, 0.1])[0],
        "font_weight": rng.choice(["regular", "bold"]),
        "font_style": rng.choices(["normal", "italic"], [0.8, 0.2])[0],
        "text_align": rng.choice(["left", "center", "right"]),
        "line_height": round(rng.uniform(0.8, 2.0), 2),
        "letter_spacing": rng.choices([0, round(rng.uniform(-2, 12), 1)], [0.6, 0.4])[0],
    }
    if rng.random() < 0.1:
        e["capitalize"] = True
    if rng.random() < 0.15:
        a = rng.randrange(len(content))
        b = rng.randrange(a, len(content))
        run = {"start": a, "end": b}
        run[rng.choice(["font_weight", "font_style", "color"])] = rng.choice(["bold", "italic", "rgba(255,0,0,1)"])
        if run.get("font_weight") == "italic":
            run["font_weight"] = "bold"
        if run.get("font_style") == "bold":
            run["font_style"] = "italic"
        if run.get("color") in ("bold", "italic"):
            run["color"] = "rgba(255,0,0,1)"
        e["style_runs"] = [run]
    if rng.random() < 0.05:
        e["rotation"] = rng.choice([90, -15, 30])
    return e


def rotated_err(e: dict, mb, pb) -> float:
    """Rotated text: the model rotates each glyph's bounding rect, so it may only be larger than the
    ink, by at most 8% of the font size. Returns 0 when that holds, else the worst excess (px)."""
    under = max(mb[0] - pb[0], mb[1] - pb[1], pb[2] - mb[2], pb[3] - mb[3])
    over = max(pb[0] - mb[0], pb[1] - mb[1], mb[2] - pb[2], mb[3] - pb[3])
    if under <= TOL and over <= 0.08 * float(e.get("font_size", 12)) + TOL:
        return 0.0
    return max(under, over)


def fit_pad(lay, w, h, margin: int = 24) -> int:
    """A pad that keeps the model's whole drawing (font box, rotated) on the test surface."""
    b = lay.canvas_box(lay.font_box) or (0, 0, 0, 0)
    ib = lay.canvas_ink_box() or b
    lo = min(b[0], b[1], ib[0], ib[1], 0)
    hi = max(b[2] - w, b[3] - h, ib[2] - w, ib[3] - h, 0)
    return int(max(-lo, hi)) + margin


def one(e: dict, conv) -> dict:
    from renderer.converter.convert import convert_chameleon_to_crello

    r = conv
    lay = measure_text(e)
    ex, _ = convert_chameleon_to_crello({"canvas": {"width": CW, "height": CH}, "elements": [e]}, r.font_manager)
    res = {"wrap_ok": lay.wrapped == ex["text"][0]}
    pad = fit_pad(lay, CW, CH)
    a = Scene({"canvas": {"width": CW, "height": CH}, "elements": [e]}).draw(pad=pad, transparent=True)[..., 3]
    edge = a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any()
    pb = mask_box(a, 0, pad)
    mb = lay.canvas_ink_box()
    if edge:
        res["skip"] = True
        return res
    if pb is None or mb is None:
        res["box_err"] = 0.0 if (pb is None and mb is None) else float("inf")
    else:
        res["box_err"] = rotated_err(e, mb, pb) if e.get("rotation") else max(abs(x - y) for x, y in zip(mb, pb))
    res["lines"] = lay.n_lines
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", default=None, help="write per-case results here")
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args(argv)
    from fundamentals.render import _renderer

    r = _renderer(str(fontbook().dir))
    rng = random.Random(args.seed)
    rows, t0 = [], time.time()
    for k in range(args.n):
        e = random_text(rng)
        tol = TOL
        res = one(e, r)
        res["tol"] = tol
        res["i"] = k
        rows.append((e, res))
    done = [x for x in rows if not x[1].get("skip")]
    wrap_bad = [x for x in rows if not x[1]["wrap_ok"]]
    box_bad = [x for x in done if x[1]["box_err"] > x[1]["tol"]]
    errs = np.array([x[1]["box_err"] for x in done if np.isfinite(x[1]["box_err"])])
    print(f"conformance property set: n={args.n} seed={args.seed} ({time.time() - t0:.0f}s)")
    print(f"  wrap string != converter: {len(wrap_bad)}/{len(rows)} ({len(wrap_bad) / max(1, len(rows)):.2%})")
    print(f"  ink box err > tol:        {len(box_bad)}/{len(done)} ({len(box_bad) / max(1, len(done)):.2%}); "
          f"skipped (past the pad) {len(rows) - len(done)}")
    if len(errs):
        print(f"  ink box err px: median {np.median(errs):.2f} p95 {np.percentile(errs, 95):.2f} max {errs.max():.2f}")
    for e, res in (wrap_bad + box_bad)[: args.show]:
        print("   ", res, json.dumps(e)[:300])
    if args.json:
        with open(args.json, "w") as f:
            json.dump([{"element": e, **res} for e, res in rows], f)
    return 0 if not wrap_bad and not box_bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
