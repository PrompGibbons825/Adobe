"""Designer decisions: what a resize did to a design, and what a designer would do.

Pure Python, stdlib only. Works on plain Chameleon-schema dicts
(``{"canvas": {"width", "height"}, "elements": [...]}``), so the same code describes a
designer's resize (Crello families) and an engine output.

    source_features(source, target) -> {name: float|str}   # source type + target shape
    decisions_of(source, out)        -> {decision: str}     # what `out` did relative to `source`
    predict(features)                -> {decision: (value, confidence)}  # designer_tree.json

Roles are geometric and computed here (never the engine's analyze() roles):
  background  image/shape covering >= 85% of the canvas
  headline    the text with the largest font size
  small       text with font size < 0.5x the headline's
  body        every other text
  hero        the largest non-background image covering >= 6% of the canvas
  art         every other image (logos, icons, illustrations)
  shape       every other shape

Elements with opacity 0, area < 4 px^2, or fully outside the canvas count as dropped, as do
source elements with no match in ``out``. Matching: by id when every source id is present in
``out`` with the same type (and the same content for texts); otherwise texts by normalized
content (duplicates in order) and the rest greedily by kind (``_kind`` if given, else type) in
z order, to the unused element with the closest aspect ratio and canvas share.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

ROLES = ("background", "headline", "body", "small", "hero", "art", "shape")
ANCHOR_ROLES = ("headline", "hero", "art", "small")
SCALE_ROLES = ("headline", "body", "hero", "art")
_TREE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "designer_tree.json")
_TREES = None


# ---------------------------------------------------------------- geometry helpers

def _canvas(d) -> Tuple[float, float]:
    c = d.get("canvas") or {}
    return float(c.get("width") or 1), float(c.get("height") or 1)


def _box(e) -> Tuple[float, float, float, float]:
    x, y = float(e.get("x") or 0), float(e.get("y") or 0)
    return (x, y, x + float(e.get("width") or 0), y + float(e.get("height") or 0))


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _clip(b, W, H):
    return (max(0.0, b[0]), max(0.0, b[1]), min(W, b[2]), min(H, b[3]))


def _union(bs):
    bs = list(bs)
    if not bs:
        return None
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def _visible(e, W, H) -> bool:
    if float(e.get("opacity", 1) if e.get("opacity") is not None else 1) <= 0.0:
        return False
    b = _box(e)
    if _area(b) < 4.0:
        return False
    return _area(_clip(b, W, H)) > 0.0


def _norm(s) -> str:
    return " ".join(str(s or "").lower().split())


def _fs(e) -> float:
    return float(e.get("font_size") or 0) or max(1.0, (_box(e)[3] - _box(e)[1]) * 0.8)


# ---------------------------------------------------------------- roles

def roles(d) -> Dict[str, str]:
    """id -> geometric role (see module docstring)."""
    W, H = _canvas(d)
    els = d.get("elements") or []
    out = {}
    texts = [e for e in els if e.get("type") == "text"]
    mx = max((_fs(e) for e in texts), default=0.0)
    head = None
    if texts:
        head = max(texts, key=lambda e: (_fs(e), _area(_box(e))))
    imgs = []
    for e in els:
        t = e.get("type")
        b = _box(e)
        cov = _area(_clip(b, W, H)) / (W * H)
        if t == "text":
            if e is head:
                r = "headline"
            elif _fs(e) < 0.5 * mx:
                r = "small"
            else:
                r = "body"
        elif cov >= 0.85:
            r = "background"
        elif t == "image":
            r = "art"
            if cov >= 0.06:
                imgs.append((cov, e))
        else:
            r = "shape"
        out[e.get("id")] = r
    if imgs:
        out[max(imgs, key=lambda ce: ce[0])[1].get("id")] = "hero"
    return out


# ---------------------------------------------------------------- matching

def _kind(e) -> str:
    return e.get("_kind") or e.get("type") or ""


def match(source, out) -> Dict[str, Optional[dict]]:
    """source id -> matching element of ``out`` (or None)."""
    se = source.get("elements") or []
    oe = out.get("elements") or []
    by_id = {e.get("id"): e for e in oe}
    ok = all(e.get("id") in by_id and by_id[e.get("id")].get("type") == e.get("type")
             and (e.get("type") != "text" or _norm(by_id[e.get("id")].get("content")) == _norm(e.get("content")))
             for e in se)
    if ok:
        return {e.get("id"): by_id[e.get("id")] for e in se}
    res: Dict[str, Optional[dict]] = {}
    Ws, Hs = _canvas(source)
    Wo, Ho = _canvas(out)
    txt: Dict[str, List[dict]] = {}
    oth: Dict[str, List[dict]] = {}
    for e in sorted(oe, key=lambda e: e.get("z_index", 0)):
        if e.get("type") == "text":
            txt.setdefault(_norm(e.get("content")), []).append(e)
        else:
            oth.setdefault(_kind(e), []).append(e)
    used = set()

    def sig(e, W, H):
        b = _box(e)
        w, h = max(1e-6, b[2] - b[0]), max(1e-6, b[3] - b[1])
        return math.log(w / h), math.log(max(1e-6, w * h / (W * H)))

    for e in sorted(se, key=lambda e: e.get("z_index", 0)):
        if e.get("type") == "text":
            lst = txt.get(_norm(e.get("content")), [])
            res[e.get("id")] = lst.pop(0) if lst else None
            continue
        a0, s0 = sig(e, Ws, Hs)
        best, bd = None, 1e9
        for c in oth.get(_kind(e), []):
            if id(c) in used:
                continue
            a1, s1 = sig(c, Wo, Ho)
            dd = abs(a1 - a0) + 0.5 * abs(s1 - s0)
            if dd < bd:
                best, bd = c, dd
        if best is not None:
            used.add(id(best))
        res[e.get("id")] = best
    return res


# ---------------------------------------------------------------- XY cut on raw boxes

def _segments(boxes: List[tuple], axis: int) -> List[List[tuple]]:
    lo, hi = (1, 3) if axis == 1 else (0, 2)
    order = sorted(boxes, key=lambda b: b[lo])
    segs: List[List[tuple]] = []
    end = seg_min = 0.0
    for b in order:
        size = max(1.0, b[hi] - b[lo])
        if segs and b[lo] < end - 0.15 * min(size, seg_min):
            segs[-1].append(b)
            end = max(end, b[hi])
            seg_min = min(seg_min, size)
            continue
        segs.append([b])
        end, seg_min = b[hi], size
    return segs


def _gap(segs, axis) -> float:
    lo, hi = (1, 3) if axis == 1 else (0, 2)
    return max((min(b[lo] for b in s2) - max(b[hi] for b in s1) for s1, s2 in zip(segs, segs[1:])), default=0.0)


def top_cut(boxes: List[tuple]) -> Tuple[str, List[List[tuple]]]:
    """('v' bands | 'h' columns | 'leaf', segments) at the top level of an XY-cut."""
    if len(boxes) < 2:
        return "leaf", [list(boxes)]
    bands, cols = _segments(boxes, 1), _segments(boxes, 0)
    if len(bands) > 1 and (len(cols) < 2 or _gap(bands, 1) >= _gap(cols, 0)):
        return "v", bands
    if len(cols) > 1:
        return "h", cols
    return "leaf", [list(boxes)]


# ---------------------------------------------------------------- design summary

def _summary(d, rmap: Dict[str, str], keep: Optional[Dict[str, dict]] = None) -> dict:
    """Geometry of one design by source role. ``keep``: source id -> element in ``d`` (resizes)."""
    W, H = _canvas(d)
    if keep is None:
        keep = {e.get("id"): e for e in d.get("elements") or []}
    by_role: Dict[str, List[dict]] = {r: [] for r in ROLES}
    for sid, e in keep.items():
        if e is not None and _visible(e, W, H):
            by_role[rmap.get(sid, "shape")].append(e)
    S = {"W": W, "H": H, "by_role": by_role}
    copy = (by_role["headline"] + by_role["body"]) or by_role["small"]  # main copy block
    S["copy"] = _union(_clip(_box(e), W, H) for e in copy)
    hero = by_role["hero"]
    S["hero"] = _clip(_box(hero[0]), W, H) if hero else None
    S["hero_raw"] = _box(hero[0]) if hero else None
    fg = [_clip(_box(e), W, H) for r in ROLES if r != "background" for e in by_role[r]]
    S["fg"] = fg
    S["layout"] = _layout(S)
    return S


def _layout(S) -> str:
    """stack | row | split | overlay | single: how the copy sits relative to the hero (or, with no
    hero, the top-level XY cut of the foreground)."""
    W, H = S["W"], S["H"]
    c, h = S["copy"], S["hero"]
    if c is not None and h is not None:
        if _inter(c, h) >= 0.3 * min(_area(c), _area(h)) or _area(h) >= 0.6 * W * H:
            return "overlay"
        ov_x = max(0.0, min(c[2], h[2]) - max(c[0], h[0])) / max(1.0, min(c[2] - c[0], h[2] - h[0]))
        ov_y = max(0.0, min(c[3], h[3]) - max(c[1], h[1])) / max(1.0, min(c[3] - c[1], h[3] - h[1]))
        if ov_y > ov_x:
            return "split" if (h[3] - h[1]) >= 0.9 * H else "row"
        return "stack"
    kind, segs = top_cut(S["fg"])
    return {"v": "stack", "h": "row"}.get(kind, "single")


def _zone(b, W, H) -> str:
    cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
    col = "l" if cx < 1 / 3 else ("r" if cx > 2 / 3 else "c")
    row = "t" if cy < 1 / 3 else ("b" if cy > 2 / 3 else "m")
    return row + col


def _aspect_class(ar: float) -> str:
    if ar < 0.6:
        return "tall"
    if ar < 0.85:
        return "portrait"
    if ar <= 1.18:
        return "square"
    if ar <= 2.0:
        return "landscape"
    return "wide"


def _band(x: float, lo: float, hi: float, names=("down", "same", "up")) -> str:
    return names[0] if x < lo else (names[2] if x > hi else names[1])


def _align_of(els, W) -> str:
    if not els:
        return "none"
    votes = {}
    for e in els:
        a = e.get("text_align")
        if a not in ("left", "center", "right"):
            b = _box(e)
            a = "center" if abs((b[0] + b[2]) / 2 - W / 2) < 0.05 * W else ("left" if b[0] < W - b[2] else "right")
        votes[a] = votes.get(a, 0) + (_box(e)[2] - _box(e)[0])
    a, v = max(votes.items(), key=lambda kv: kv[1])
    return a if v >= 0.7 * sum(votes.values()) else "mixed"


def _photo(S) -> str:
    W, H = S["W"], S["H"]
    imgs = [e for e in S["by_role"]["background"] if e.get("type") == "image"] + S["by_role"]["hero"]
    if not imgs:
        return "none"
    b = _box(imgs[0])
    cov = _area(_clip(b, W, H)) / (W * H)
    if cov >= 0.85:
        return "crop_bleed" if _area(b) > 1.1 * W * H else "bleed"
    return "framed"


def _order(S) -> List[str]:
    copy = S["by_role"]["headline"] + S["by_role"]["body"] + S["by_role"]["small"]
    return [_norm(e.get("content")) for e in sorted(copy, key=lambda e: ((_box(e)[1] + _box(e)[3]) / 2, _box(e)[0]))]


# ---------------------------------------------------------------- public API

def source_features(source: dict, target: dict) -> Dict[str, object]:
    """Source type (geometry only) plus the target shape. ``target`` may be a canvas dict
    ({"width","height"}) or a design ({"canvas": ...})."""
    W, H = _canvas(source)
    tc = target.get("canvas", target)
    Wt, Ht = float(tc.get("width") or 1), float(tc.get("height") or 1)
    rmap = roles(source)
    S = _summary(source, rmap)
    br = S["by_role"]
    bgk = "none"
    for e in br["background"]:
        bgk = "image" if e.get("type") == "image" else ("shape" if bgk == "none" else bgk)
    copy = br["headline"] + br["body"] + br["small"]
    imgs = [e for e in br["background"] + br["hero"] + br["art"] if e.get("type") == "image"]
    on_photo = any(_inter(_box(t), _box(i)) >= 0.5 * max(1.0, _area(_box(t))) for t in copy for i in imgs)
    kind, segs = top_cut(S["fg"])
    head = br["headline"][0] if br["headline"] else None
    sa, ta = W / H, Wt / Ht
    lac = math.log(ta / sa)
    feats = {
        "bg_kind": bgk,
        "copy_on_photo": "yes" if on_photo else "no",
        "cut": kind,
        "cut_n": float(len(segs)),
        "layout": S["layout"],
        "n_images": float(len(br["hero"]) + len([e for e in br["art"] if e.get("type") == "image"])),
        "text_only": 1.0 if not imgs else 0.0,
        "headline_share": (_area(_clip(_box(head), W, H)) / (W * H)) if head else 0.0,
        "hero_share": (_area(S["hero"]) / (W * H)) if S["hero"] else 0.0,
        "n_text": float(len(copy)),
        "src_aspect": _aspect_class(sa),
        "tgt_aspect": _aspect_class(ta),
        "log_src_ar": math.log(sa),
        "log_tgt_ar": math.log(ta),
        "log_aspect_change": lac,
        "abs_aspect_change": abs(lac),
        "area_ratio": math.log((Wt * Ht) / (W * H)),
    }
    # the source's own value of each decision (anchor zones, alignment, photo, coverage...), so a
    # tree can learn "keep it unless the shape change says otherwise"
    for k, v in keep_source(source).items():
        if not k.startswith("drop:") and k not in ("arrangement", "group_kept", "order_kept", "emphasis"):
            feats["src_" + k] = v
    return feats


def decisions_of(source: dict, out: dict) -> Dict[str, str]:
    """The decisions visible in ``out`` relative to ``source`` (categorical values)."""
    rmap = roles(source)
    m = match(source, out)
    S = _summary(source, rmap)
    O = _summary(out, rmap, keep=m)
    Ws, Hs, Wo, Ho = S["W"], S["H"], O["W"], O["H"]
    lin = math.sqrt((Wo * Ho) / (Ws * Hs))
    d: Dict[str, str] = {}
    # arrangement
    ls, lo = S["layout"], O["layout"]
    if lo == ls or lo == "single":
        d["arrangement"] = "kept"
    else:
        d["arrangement"] = {"stack": "restack", "row": "to_row", "split": "to_split", "overlay": "overlay"}[lo]
    # groupings: each top-level source segment (>=2 elements) still bounds only its own members
    src_ids = [sid for sid in m if rmap.get(sid) != "background"]
    sb = {sid: _clip(_box(e), Ws, Hs) for e in source.get("elements") or [] for sid in [e.get("id")] if sid in src_ids}
    _, segs = top_cut(list(sb.values()))
    gk = True
    if len(segs) > 1:
        seg_of = {}
        for k, seg in enumerate(segs):
            for sid, b in sb.items():
                if b in seg:
                    seg_of.setdefault(sid, k)
        ob = {sid: _clip(_box(m[sid]), Wo, Ho) for sid in sb if m.get(sid) is not None and _visible(m[sid], Wo, Ho)}
        for k in range(len(segs)):
            mem = [sid for sid in ob if seg_of.get(sid) == k]
            if len(mem) < 2:
                continue
            u = _union(ob[s] for s in mem)
            for sid, b in ob.items():
                if seg_of.get(sid) != k:
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    if u[0] < cx < u[2] and u[1] < cy < u[3] and _area(b) < 0.5 * Wo * Ho:
                        gk = False
    d["group_kept"] = "yes" if gk else "no"
    so, oo = _order(S), _order(O)
    oo_f = [t for t in oo if t in so]
    so_f = [t for t in so if t in oo_f]
    d["order_kept"] = "yes" if so_f == oo_f else "no"
    # drops, anchors, scales
    for r in ROLES:
        src_r = [sid for sid, rr in rmap.items() if rr == r]
        if not src_r:
            continue
        kept = [sid for sid in src_r if m.get(sid) is not None and _visible(m[sid], Wo, Ho)]
        d["drop:" + r] = "no" if len(kept) == len(src_r) else ("all" if not kept else "some")
        if r in ANCHOR_ROLES and kept:
            z = _zone(_union(_clip(_box(m[s]), Wo, Ho) for s in kept), Wo, Ho)
            d["anchor:" + r] = z
            d["anchor_row:" + r] = {"t": "top", "m": "middle", "b": "bottom"}[z[0]]
            d["anchor_col:" + r] = {"l": "left", "c": "center", "r": "right"}[z[1]]
        if r in SCALE_ROLES and kept:
            sid = max(kept, key=lambda s: _area(_box(next(e for e in source["elements"] if e.get("id") == s))))
            se = next(e for e in source["elements"] if e.get("id") == sid)
            if r in ("headline", "body"):
                f = _fs(m[sid]) / max(1e-6, _fs(se))
            else:
                f = math.sqrt(max(1e-6, _area(_box(m[sid]))) / max(1e-6, _area(_box(se))))
            d["scale:" + r] = _band(f / lin, 0.8, 1.25)
    # emphasis: headline vs next-largest text
    def ratio(Sx):
        ts = sorted((_fs(e) for e in Sx["by_role"]["headline"] + Sx["by_role"]["body"] + Sx["by_role"]["small"]), reverse=True)
        return ts[0] / ts[1] if len(ts) >= 2 and ts[1] > 0 else None
    rs, ro = ratio(S), ratio(O)
    d["emphasis"] = _band(ro / rs, 0.85, 1.18) if rs and ro else "same"
    d["align"] = _align_of(O["by_role"]["headline"] + O["by_role"]["body"] + O["by_role"]["small"], Wo)
    d["photo"] = _photo(O)
    cu = _union(O["fg"])
    cov = _area(cu) / (Wo * Ho) if cu else 0.0
    d["coverage"] = "low" if cov < 0.4 else ("mid" if cov < 0.75 else "high")
    return d


def keep_source(source: dict) -> Dict[str, str]:
    """The no-change value of every decision (what decisions_of would say for a pure rescale)."""
    return decisions_of(source, source)


# ---------------------------------------------------------------- tree prediction

def _load():
    global _TREES
    if _TREES is None:
        try:
            with open(_TREE_PATH) as fh:
                _TREES = json.load(fh).get("trees", {})
        except (OSError, ValueError):
            _TREES = {}
    return _TREES


def _walk(node, feats):
    while "leaf" not in node:
        f = node["f"]
        if "eq" in node:  # one-hot categorical: left = not equal
            go_left = str(feats.get(f)) != node["eq"]
        else:
            v = feats.get(f)
            v = float(v) if isinstance(v, (int, float)) else 0.0
            go_left = v <= node["t"]
        node = node["l"] if go_left else node["r"]
    return node["leaf"]


def predict(features: Dict[str, object]) -> Dict[str, Tuple[str, float]]:
    """decision -> (value, confidence = leaf purity). Empty if designer_tree.json is missing.
    Decisions about a role the source lacks are still returned; ignore them. Leaves with value "="
    mean "keep the source's own value" and are resolved from features["src_<decision>"]."""
    out = {}
    for name, tree in _load().items():
        leaf = _walk(tree, features)
        v = leaf["v"]
        if v == "=":  # relative leaf: keep the source's own value
            v = str(features.get("src_" + name, "n/a"))
        out[name] = (v, float(leaf["p"]))
    return out
