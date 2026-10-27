"""Scoreboard: how close each engine's resize is to the designer's own resize,
measured only on the elements both versions share.

Designer resizes often drop elements or add new ones (24 of our 49 pairs keep
the same text and element count), and our rules forbid both. So the designer's
version is only evidence about the elements it kept: text is matched by its
words, pictures by their appearance (each design file re-exports its images),
and anything unmatched is left out. A pair whose designer kept less than 70%
of the source's words is not scored at all.

Per pair, over the matched elements (canvas coordinates as fractions):
  pos    mean distance between element centres            (lower is better)
  size   mean |log| of the ratio of canvas shares           (lower is better)
  order  share of clear left/right and above/below relations kept (higher)
  rank   share of clear type-size orderings kept            (higher)
  dist   pos + 0.25 size + 0.5 (1 - order) + 0.5 (1 - rank) (lower)

usage (repo root):
  python tools/scoreboard.py OUT_DIR [OUT_DIR ...]
OUT_DIR holds <root>_p<k>_<W>x<H>.json outputs (as written by the h2h runner);
roots are read from crello/ and crello_holdout/.
"""

import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image

ROOTS = ["crello", "crello_holdout"]
if "--roots" in sys.argv:
    ROOTS = sys.argv[sys.argv.index("--roots") + 1].split(",")
MIN_WORDS = 0.7


def words(s):
    return re.findall(r"[\w']+", (s or "").lower())


@lru_cache(maxsize=None)
def look(root, asset):
    """Appearance signature: colour histogram of the visible pixels, the
    visible share and the visible box's aspect."""
    try:
        im = Image.open(f"{root}/assets/{asset}").convert("RGBA")
    except Exception:
        return None
    im.thumbnail((96, 96))
    a = np.asarray(im).astype(np.float32) / 255
    vis = a[..., 3] > 0.5
    if vis.sum() < 4:
        return None
    q = (a[..., :3][vis] * 3.999).astype(int)
    hist = np.bincount(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], minlength=64).astype(np.float32)
    ys, xs = np.where(vis)
    asp = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
    return hist / hist.sum(), float(vis.mean()), float(asp)


def look_sim(a, b):
    if a is None or b is None:
        return 0.0
    h = float(np.minimum(a[0], b[0]).sum())
    cov = 1 - min(1.0, abs(a[1] - b[1]) * 2)
    asp = math.exp(-abs(math.log(max(1e-3, a[2]) / max(1e-3, b[2]))))
    return h * (0.6 + 0.2 * cov + 0.2 * asp)


def match(root, src, ref):
    """{source id: designer element}, greedy on similarity."""
    cand = []
    for e in src["elements"]:
        for r in ref["elements"]:
            if e["type"] == "text" and r["type"] == "text":
                a, b = words(e.get("content")), words(r.get("content"))
                if not a or not b:
                    continue
                s = len(set(a) & set(b)) / len(set(a) | set(b))
                if " ".join(a) == " ".join(b):
                    s = 2.0
                if s >= 0.6:
                    cand.append((s, e["id"], r["id"]))
            elif e["type"] != "text" and r["type"] != "text":
                if e["type"] == "image" and r["type"] == "image":
                    s = look_sim(look(root, e.get("asset_id")), look(root, r.get("asset_id")))
                    if s >= 0.75:
                        cand.append((s, e["id"], r["id"]))
    rmap = {r["id"]: r for r in ref["elements"]}
    used_s, used_r, out = set(), set(), {}
    for s, i, j in sorted(cand, reverse=True):
        if i in used_s or j in used_r:
            continue
        used_s.add(i)
        used_r.add(j)
        out[i] = rmap[j]
    return out


def word_cover(src, ref):
    sw = [w for e in src["elements"] if e["type"] == "text" for w in words(e.get("content"))]
    rw = set(w for e in ref["elements"] if e["type"] == "text" for w in words(e.get("content")))
    return sum(w in rw for w in sw) / max(1, len(sw))


def geom(e, W, H):
    return ((e["x"] + e["width"] / 2) / W, (e["y"] + e["height"] / 2) / H,
            max(1e-6, e["width"] * e["height"] / (W * H)))


def pair_metrics(src, ours, ref, m):
    W, H = ref["canvas"]["width"], ref["canvas"]["height"]
    oe = {e["id"]: e for e in ours["elements"]}
    ids = [i for i in m if i in oe]
    if len(ids) < 2:
        return None
    g_o = {i: geom(oe[i], W, H) for i in ids}
    g_d = {i: geom(m[i], W, H) for i in ids}
    pos = float(np.mean([math.hypot(g_o[i][0] - g_d[i][0], g_o[i][1] - g_d[i][1]) for i in ids]))
    size = float(np.mean([abs(math.log(g_o[i][2] / g_d[i][2])) for i in ids]))
    keep = tot = 0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            for k in (0, 1):
                d = g_d[i][k] - g_d[j][k]
                if abs(d) > 0.05:
                    tot += 1
                    keep += (g_o[i][k] - g_o[j][k]) * d > 0
    order = keep / tot if tot else 1.0
    texts = [i for i in ids if oe[i]["type"] == "text"]
    keep = tot = 0
    for a in range(len(texts)):
        for b in range(a + 1, len(texts)):
            i, j = texts[a], texts[b]
            di, dj = m[i]["font_size"], m[j]["font_size"]
            if max(di, dj) > 1.1 * min(di, dj):
                tot += 1
                keep += (oe[i]["font_size"] - oe[j]["font_size"]) * (di - dj) > 0
    rank = keep / tot if tot else 1.0
    dist = pos + 0.25 * size + 0.5 * (1 - order) + 0.5 * (1 - rank)
    return dict(pos=pos, size=size, order=order, rank=rank, dist=dist, n=len(ids))


BRAND_FONTS = {"sprint"}   # engines that measure text in the brand-kit fonts (Vlad's; also any v_* dir)


def readability(dirs, res, keys):
    """read: share of words OCR reads at a 768 px view, over the texts the
    source's own render reads (>= 60%); lowread: share of those texts read
    under 40%. Each engine is rendered with the fonts it measured with."""
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from engine import legible, saliency, textfit
    from renderer.converter import convert
    src_cache = {}
    for root in ROOTS:
        saliency.ASSETS_DIR = pathlib.Path(f"{root}/assets")
        textfit.FONTS_PICKLE, textfit._FM = f"{root}/fonts_subset.pickle", None
        legible._RENDERERS.clear()
        convert._renderer_cache.clear()
        idx = json.load(open(f"{root}/index.json"))
        for key in [k for k in keys if k.startswith(root + "_p")]:
            p = int(key[len(root) + 2:].split("_")[0])
            if (root, p) not in src_cache:
                src_cache[(root, p)] = legible.recall(json.load(open(f"{root}/designs/" + idx[p][0]["file"])))
            src = src_cache[(root, p)] or {}
            want = [i for i, v in src.items() if v >= 0.6]
            for d in dirs:
                fd = pathlib.Path("brand_kit/fonts") if d.name in BRAND_FONTS or d.name.startswith("v_") else None
                rec = legible.recall(json.load(open(d / f"{key}.json")), fd) or {}
                vals = [rec.get(i, 0.0) for i in want]
                res[d.name][key]["read"] = float(np.mean(vals)) if vals else 1.0
                res[d.name][key]["lowread"] = float(np.mean([v < 0.4 for v in vals])) if vals else 0.0


def _anchors(e):
    """Edges and centre lines an element can align on (x0, xc, x1, y0, yc, y1).
    A text aligns by the edge its alignment uses, not its whole box."""
    x0, y0, w, h = e["x"], e["y"], e["width"], e["height"]
    a = {"x0": x0, "xc": x0 + w / 2, "x1": x0 + w, "y0": y0, "yc": y0 + h / 2, "y1": y0 + h}
    if e["type"] == "text":
        al = e.get("text_align", "left")
        keep = {"left": "x0", "center": "xc", "right": "x1"}.get(al, "x0")
        a = {k: v for k, v in a.items() if k[0] == "y" or k == keep}
    return a


def alignments(d):
    """Pairs of elements sharing an edge or centre line: {(i, j, key)}."""
    W, H = d["canvas"]["width"], d["canvas"]["height"]
    tol = 0.006 * min(W, H)
    els = [e for e in d["elements"] if abs(float(e.get("rotation", 0) or 0)) <= 1
           and e["width"] * e["height"] < 0.6 * W * H and e["width"] > 0 and e["height"] > 0]
    out = set()
    an = {e["id"]: _anchors(e) for e in els}
    for i, a in enumerate(els):
        for b in els[i + 1:]:
            for k, v in an[a["id"]].items():
                if k in an[b["id"]] and abs(v - an[b["id"]][k]) <= tol:
                    out.add((a["id"], b["id"], k))
    return out


def kept_alignments(src, out):
    rel = alignments(src)
    if not rel:
        return None
    W, H = out["canvas"]["width"], out["canvas"]["height"]
    tol = 0.01 * min(W, H)
    oe = {e["id"]: e for e in out["elements"]}
    ok = 0
    for i, j, k in rel:
        if i in oe and j in oe:
            ai, aj = _anchors(oe[i]), _anchors(oe[j])
            ok += k in ai and k in aj and abs(ai[k] - aj[k]) <= tol
    return ok / len(rel)


def subject_cover(root, d):
    """Share of the main photo's subject (faces, else the salient box) under copy."""
    from engine import saliency
    W, H = d["canvas"]["width"], d["canvas"]["height"]
    best = None
    for e in d["elements"]:
        if e["type"] == "image" and e["width"] * e["height"] >= 0.3 * W * H:
            if best is None or e["width"] * e["height"] > best["width"] * best["height"]:
                best = e
    if best is None:
        return None
    info = saliency.analyze(best.get("asset_id") or "")
    if info is None:
        return None
    sb = info["faces"][0] if info.get("faces") else info["sal_box"]
    if not info.get("faces") and (sb[2] - sb[0]) * (sb[3] - sb[1]) > 0.45:
        return None
    bx = (best["x"] + sb[0] * best["width"], best["y"] + sb[1] * best["height"],
          best["x"] + sb[2] * best["width"], best["y"] + sb[3] * best["height"])
    bx = (max(0, bx[0]), max(0, bx[1]), min(W, bx[2]), min(H, bx[3]))
    area = max(1e-6, (bx[2] - bx[0]) * (bx[3] - bx[1]))
    if bx[2] <= bx[0] or bx[3] <= bx[1]:
        return None
    n = 48
    grid = np.zeros((n, n), bool)
    for e in d["elements"]:
        if e["type"] != "text":
            continue
        x0 = int(np.clip((e["x"] - bx[0]) / (bx[2] - bx[0]) * n, 0, n)); x1 = int(np.clip((e["x"] + e["width"] - bx[0]) / (bx[2] - bx[0]) * n, 0, n))
        y0 = int(np.clip((e["y"] - bx[1]) / (bx[3] - bx[1]) * n, 0, n)); y1 = int(np.clip((e["y"] + e["height"] - bx[1]) / (bx[3] - bx[1]) * n, 0, n))
        grid[y0:y1, x0:x1] = True
    return float(grid.mean())


def structure(dirs, res, keys):
    """align: share of the source's alignments kept; cover+: copy over the
    photo's subject beyond the source's own."""
    import pathlib
    from engine import saliency
    for root in ROOTS:
        saliency.ASSETS_DIR = pathlib.Path(f"{root}/assets")
        saliency.analyze.cache_clear()
        saliency._load.cache_clear()
        idx = json.load(open(f"{root}/index.json"))
        for key in [k for k in keys if k.startswith(root + "_p")]:
            p = int(key[len(root) + 2:].split("_")[0])
            src = json.load(open(f"{root}/designs/" + idx[p][0]["file"]))
            sc = subject_cover(root, src)
            for d in dirs:
                out = json.load(open(d / f"{key}.json"))
                ka = kept_alignments(src, out)
                oc = subject_cover(root, out) if sc is not None else None
                res[d.name][key]["align"] = ka
                res[d.name][key]["cover+"] = None if oc is None else max(0.0, oc - sc)


def main(dirs):
    dirs = [Path(d) for d in dirs]
    res = {d.name: {} for d in dirs}
    skipped = []
    for root in ROOTS:
        idx = json.load(open(f"{root}/index.json"))
        for p, vs in enumerate(idx):
            src = json.load(open(f"{root}/designs/" + vs[0]["file"]))
            for v in vs[1:]:
                key = f"{root}_p{p}_{v['w']}x{v['h']}"
                ref = json.load(open(f"{root}/designs/" + v["file"]))
                cov = word_cover(src, ref)
                if cov < MIN_WORDS:
                    skipped.append((key, cov))
                    continue
                m = match(root, src, ref)
                for d in dirs:
                    f = d / f"{key}.json"
                    if f.is_file():
                        r = pair_metrics(src, json.load(open(f)), ref, m)
                        if r:
                            res[d.name][key] = r
    keys = sorted(set.intersection(*(set(r) for r in res.values())))
    if "--read" in sys.argv:
        readability(dirs, res, keys)
    if "--struct" in sys.argv:
        structure(dirs, res, keys)
        for name, r in res.items():
            al = [r[k]["align"] for k in keys if r[k].get("align") is not None]
            cv = [r[k]["cover+"] for k in keys if r[k].get("cover+") is not None]
            print(f"{name:<24} align kept {np.mean(al):.3f} (n={len(al)})   copy over subject +{np.mean(cv):.3f} (n={len(cv)})")
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as fh:
            json.dump({n: {k: r[k] for k in keys} for n, r in res.items()}, fh, indent=1)
    print(f"scored {len(keys)} pairs; skipped {len(skipped)} where the designer kept < {MIN_WORDS:.0%} of the words")
    cols = ("pos", "size", "order", "rank", "dist") + (("read", "lowread") if "--read" in sys.argv else ())
    print(f"{'engine':<12}" + "".join(f"{k:>8}" for k in cols))
    for name, r in res.items():
        print(f"{name:<12}" + "".join(f"{np.mean([r[k][m] for k in keys]):8.3f}" for m in cols))
    names = list(res)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            A, B = res[names[a]], res[names[b]]
            wa = sum(A[k]["dist"] < B[k]["dist"] - 0.01 for k in keys)
            wb = sum(B[k]["dist"] < A[k]["dist"] - 0.01 for k in keys)
            print(f"{names[a]} closer on {wa}, {names[b]} closer on {wb}, ties {len(keys) - wa - wb}")
    if "-v" in sys.argv:
        for k in keys:
            print(f"  {k:<34}" + "".join(f"{res[n][k]['dist']:8.3f}" for n in names) + f"   n={res[names[0]][k]['n']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    for opt in ("--roots", "--json"):
        if opt in args:
            i = args.index(opt)
            args = args[:i] + args[i + 2:]
    main([a for a in args if not a.startswith("-")])
