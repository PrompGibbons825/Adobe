"""Controlled perturbations of an output design (dev tool, for sentinels).

A sentinel ask shows the judge a good layout against a copy of it with one known
defect. A judge (or a panel batch) that cannot tell them apart is not to be
trusted. Ported from ws/judge-calibration (tools/judge/corpus.py, 10-06), where
the blind VLM judge preferred the unperturbed layout this often:

    p_crop       headline cut 30% past the edge       100%
    p_lowcon     copy moved to its worst contrast     100%
    p_mis20      misaligned by 20 px                   94%
    p_onart      headline moved onto the photo         88%
    p_hidelogo   logo hidden                           69%
    p_shrinksub  subhead at 55%                        69%
    p_mis8       misaligned by 8 px                    62%
    p_mis2       misaligned by 2 px                    56% (chance: not a sentinel)

`SENTINELS` keeps only the kinds the judge caught at least ~85% of the time.

`SMALL_DEFECTS` (loop/STEERING.md S6) are the details VLM judges are known to miss
(DesignSense: GPT-5 0.70 accuracy): one element hidden, one letter clipped at the
edge, copy touching the canvas edge, a broken type scale. The panel asks them as
report-only sentinels, so their accuracy is measured without invalidating a run.

    p_hide_one     one small element (a minor text, else the smallest picture) hidden
    p_clip_letter  the headline pushed half an em past the right edge
    p_edge_touch   a text moved so its ink touches the left canvas edge (no margin)
    p_typescale    one of two equal-size texts set 25% larger (else the second text)

    from tools.judge.perturb import perturb
    bad = perturb("p_crop", output_design, random.Random(0))   # None when it does not apply

The engine is only read (text measurement, the pixel backdrop); never changed.
"""

from __future__ import annotations

import json
import random
from typing import List, Optional

SMALL_DEFECTS = ["p_hide_one", "p_clip_letter", "p_edge_touch", "p_typescale"]
PERTURBATIONS = ["p_onart", "p_shrinksub", "p_mis2", "p_mis8", "p_mis20", "p_hidelogo", "p_crop", "p_lowcon",
                 *SMALL_DEFECTS]
SENTINELS = ["p_crop", "p_lowcon", "p_mis20", "p_onart"]


def _live(e: dict) -> bool:
    return float(e.get("opacity", 1) if e.get("opacity") is not None else 1) > 0 and e["width"] > 1 and e["height"] > 1


def _texts(out: dict) -> List[dict]:
    return sorted([e for e in out["elements"] if e["type"] == "text" and _live(e) and str(e.get("content", "")).strip()],
                  key=lambda e: -float(e.get("font_size", 0)))


def _images(out: dict) -> List[dict]:
    W, H = out["canvas"]["width"], out["canvas"]["height"]
    return sorted([e for e in out["elements"] if e["type"] == "image" and _live(e) and e["width"] * e["height"] < 0.7 * W * H],
                  key=lambda e: -(e["width"] * e["height"]))


HEADLINE_NEAR = 0.9  # font sizes within this ratio of the largest count as equally large


def headline(out: dict) -> Optional[dict]:
    """The largest text: the largest font size; among texts within ``HEADLINE_NEAR`` of it, the
    one with the largest drawn area (a short tag set as large as a long title is not the
    headline). The text p_crop, p_clip_letter and p_onart move."""
    texts = _texts(out)
    if not texts:
        return None
    top = float(texts[0].get("font_size", 0))
    near = [e for e in texts if float(e.get("font_size", 0)) >= HEADLINE_NEAR * top]
    return max(near, key=lambda e: (_drawn(e).area, float(e.get("font_size", 0))))


def _drawn(e: dict):
    from engine.smart.textfit import drawn_box

    return drawn_box(e)


def _clone(out: dict) -> dict:
    return json.loads(json.dumps(out))


def _get(out: dict, i: str) -> dict:
    return next(e for e in out["elements"] if e["id"] == i)


def perturb(kind: str, out: dict, rng: Optional[random.Random] = None) -> Optional[dict]:
    """A copy of `out` with one defect of `kind`, or None when the design has
    nothing to apply it to. `out` itself is never modified. `rng` is accepted
    for API stability; every kind is deterministic today."""
    from engine.smart.geom import Box

    if kind not in PERTURBATIONS:
        raise ValueError(kind)
    W, H = out["canvas"]["width"], out["canvas"]["height"]
    texts, imgs = _texts(out), _images(out)
    o = _clone(out)
    if kind == "p_onart":  # the headline moved onto the photo / product
        if not texts or not imgs:
            return None
        t, im = _get(o, headline(out)["id"]), imgs[0]
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
    if kind == "p_hide_one":  # one small element disappears
        minor = texts[1:]
        if minor:
            _get(o, minor[-1]["id"])["opacity"] = 0.0
            return o
        if not imgs:
            return None
        _get(o, imgs[-1]["id"])["opacity"] = 0.0
        return o
    if kind == "p_clip_letter":  # about one letter of the headline past the right edge
        if not texts:
            return None
        t = _get(o, headline(out)["id"])
        db = _drawn(t)
        t["x"] += (W + 0.5 * float(t["font_size"])) - db.r
        return o
    if kind == "p_edge_touch":  # copy flush against the left canvas edge
        for src_t in texts[1:] + texts[:1]:
            t = _get(o, src_t["id"])
            db = _drawn(t)
            if db.x >= 0.02 * W:
                t["x"] -= db.x
                return o
        return None
    if kind == "p_typescale":  # one of a matched pair of type sizes breaks the scale
        if len(texts) < 2:
            return None
        sizes: dict = {}
        for e in texts:
            sizes.setdefault(int(e["font_size"]), []).append(e["id"])
        twins = [ids for ids in sizes.values() if len(ids) > 1]
        tid = twins[0][-1] if twins else texts[1]["id"]
        t = _get(o, tid)
        t["font_size"] = max(int(t["font_size"]) + 2, int(round(float(t["font_size"]) * 1.25)))
        return o
    if kind == "p_crop":  # 30% of the headline's ink pushed past the right edge
        if not texts:
            return None
        t = _get(o, headline(out)["id"])
        db = _drawn(t)
        t["x"] += (W - 0.7 * db.w) - db.x
        return o
    # p_lowcon: copy moved to where it reads worst
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


def all_perturbations(out: dict, kinds: List[str] = PERTURBATIONS, seed: int = 0) -> dict:
    """{kind: perturbed design} for every kind that applies to `out`."""
    rng = random.Random(seed)
    res = {}
    for k in kinds:
        p = perturb(k, out, rng)
        if p is not None:
            res[k] = p
    return res
