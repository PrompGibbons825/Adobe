"""Layout energy after O'Donovan et al. 2014 (owner: preference between placed layouts).

E(layout) = w . f(layout). Every feature is >= 0 and 'more is worse'. Weights come from
weights.json (fit on Crello designer resizes: designer beats alternative), else DEFAULT.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

from .analyse import Scene
from .layout import Placed, text_floor

NAMES = ["overlap", "outside", "fill_deficit", "span_deficit", "shrink", "grow", "legib", "order",
         "drift", "balance", "margin_uneven", "hier", "lines", "relsize", "align", "smallest", "busy", "contrast"]

DEFAULT = {"overlap": 30.0, "outside": 50.0, "fill_deficit": 2.0, "span_deficit": 1.0, "shrink": 2.0,
           "grow": 0.5, "legib": 6.0, "order": 3.0, "drift": 1.0, "balance": 1.0, "margin_uneven": 1.0,
           "hier": 3.0, "lines": 0.5, "relsize": 1.0, "align": 0.5, "smallest": 2.0,
           "busy": 20.0, "contrast": 4.0}

_W = None


def weights() -> Dict[str, float]:
    global _W
    if _W is None:
        p = Path(__file__).with_name("weights.json")
        w = dict(DEFAULT)
        if p.is_file():
            try:
                w.update(json.loads(p.read_text())["w"])
            except Exception:  # noqa: BLE001
                pass
        _W = w
    return _W


def _inter(a, b):
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, iw) * max(0.0, ih)


def _align_lines(boxes, W, H=None):
    """Distinct alignment lines (left/centre/right in x, top/centre/bottom in y) per box; a box
    that shares any line with another box is aligned on that axis. Lower = more aligned."""
    if not boxes:
        return 0.0

    def count(vals):
        vals.sort()
        k = 1
        for a, b in zip(vals, vals[1:]):
            if b - a > 0.012:
                k += 1
        return k

    kx = count([v for x, y, w, h in boxes for v in (x / W, (x + w / 2) / W, (x + w) / W)])
    if H is None:
        return kx / len(boxes)
    ky = count([v for x, y, w, h in boxes for v in (y / H, (y + h / 2) / H, (y + h) / H)])
    return min(kx, ky) / len(boxes)


def features(S: Scene, placed: Dict[int, Placed], W: float, H: float) -> List[float]:
    A = W * H
    A0 = S.W * S.H
    bl = [b for b in S.blocks if b.bid in placed]
    n = len(bl)
    f = dict.fromkeys(NAMES, 0.0)
    if n == 0:
        return [0.0] * len(NAMES)
    src = {b.bid: (b.x, b.y, b.w, b.h) for b in bl}
    tgt = {b.bid: (placed[b.bid].x, placed[b.bid].y, placed[b.bid].w, placed[b.bid].h) for b in bl}

    # overlap of blocks that did not overlap in the source
    ov = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = bl[i].bid, bl[j].bid
            if _inter(src[a], src[b]) > 0.02 * min(src[a][2] * src[a][3], src[b][2] * src[b][3]):
                continue
            ov += _inter(tgt[a], tgt[b]) / max(1.0, min(tgt[a][2] * tgt[a][3], tgt[b][2] * tgt[b][3]))
    f["overlap"] = ov
    out = 0.0
    for x, y, w, h in tgt.values():
        inside = _inter((x, y, w, h), (0, 0, W, H))
        out += 1 - inside / max(1.0, w * h)
    f["outside"] = out

    fill_s = min(1.0, sum(b[2] * b[3] for b in src.values()) / A0)
    fill_t = min(1.0, sum(b[2] * b[3] for b in tgt.values()) / A)
    f["fill_deficit"] = max(0.0, math.log((fill_s + 0.02) / (fill_t + 0.02)))

    def span(boxes, CW, CH):
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        return (x1 - x0) / CW, (y1 - y0) / CH, x0 / CW, 1 - x1 / CW, y0 / CH, 1 - y1 / CH

    ss = span(list(src.values()), S.W, S.H)
    ts = span(list(tgt.values()), W, H)
    f["span_deficit"] = max(0.0, ss[0] - ts[0]) + max(0.0, ss[1] - ts[1])
    f["margin_uneven"] = max(0.0, abs(ts[2] - ts[3]) - abs(ss[2] - ss[3])) + max(0.0, abs(ts[4] - ts[5]) - abs(ss[4] - ss[5]))

    s_ref = math.sqrt(A / A0)
    logs = []
    for b in bl:
        sb = math.sqrt(tgt[b.bid][2] * tgt[b.bid][3] / max(1.0, src[b.bid][2] * src[b.bid][3]))
        logs.append(math.log(max(1e-6, sb / s_ref)))
    mean = sum(logs) / n
    f["shrink"] = max(0.0, -mean)
    f["grow"] = max(0.0, mean)
    f["relsize"] = math.sqrt(sum((v - mean) ** 2 for v in logs) / n)
    f["smallest"] = max(0.0, -min(logs) - 0.7)

    texts = []
    leg = 0.0
    for b in bl:
        if b.kind == "text":
            el = S.els[b.text_idx]
            fs0 = float(el.get("font_size", 16) or 16)
            fs = placed[b.bid].fs or fs0
            leg += max(0.0, math.log(text_floor(S, fs0, W, H) / max(1.0, fs)))
            texts.append((fs0, fs, b.bid))
        elif b.kind == "group":
            sb = math.sqrt(tgt[b.bid][2] * tgt[b.bid][3] / max(1.0, src[b.bid][2] * src[b.bid][3]))
            for i in b.members:
                el = S.els[i]
                if el.get("type") == "text":
                    fs0 = float(el.get("font_size", 16) or 16)
                    leg += max(0.0, math.log(text_floor(S, fs0, W, H) / max(1.0, fs0 * sb)))
    f["legib"] = leg
    hv, hp = 0, 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, c = texts[i], texts[j]
            if abs(a[0] - c[0]) < 0.08 * max(a[0], c[0]):
                continue
            hp += 1
            if (a[0] - c[0]) * (a[1] - c[1]) < 0:
                hv += 1
    f["hier"] = hv / hp if hp else 0.0

    # reading order: compare source order (rows, x) with target (centres)
    so = sorted(bl, key=lambda b: (b.row, b.x))
    rank = {b.bid: k for k, b in enumerate(so)}
    inv, pairs = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, c = so[i].bid, so[j].bid   # a before c in source
            ta, tc = tgt[a], tgt[c]
            pairs += 1
            above = tc[1] + tc[3] <= ta[1] + 0.25 * min(ta[3], tc[3])
            same_band = not above and not (ta[1] + ta[3] <= tc[1] + 0.25 * min(ta[3], tc[3]))
            left = tc[0] + tc[2] <= ta[0] + 0.25 * min(ta[2], tc[2])
            if above and not (tc[0] >= ta[0] + ta[2] - 1):
                inv += 1
            elif same_band and left:
                inv += 1
    f["order"] = inv / pairs if pairs else 0.0
    del rank

    d = 0.0
    for b in bl:
        sx, sy = (src[b.bid][0] + src[b.bid][2] / 2) / S.W, (src[b.bid][1] + src[b.bid][3] / 2) / S.H
        tx, ty = (tgt[b.bid][0] + tgt[b.bid][2] / 2) / W, (tgt[b.bid][1] + tgt[b.bid][3] / 2) / H
        d += math.hypot(sx - tx, sy - ty)
    f["drift"] = d / n

    def centroid(boxes, CW, CH):
        m = sum(b[2] * b[3] for b in boxes) or 1.0
        return (sum((b[0] + b[2] / 2) * b[2] * b[3] for b in boxes) / m / CW,
                sum((b[1] + b[3] / 2) * b[2] * b[3] for b in boxes) / m / CH)

    cs = centroid(list(src.values()), S.W, S.H)
    ct = centroid(list(tgt.values()), W, H)
    f["balance"] = max(0.0, math.hypot(ct[0] - .5, ct[1] - .5) - math.hypot(cs[0] - .5, cs[1] - .5))

    ls = 0.0
    nt = 0
    for b in bl:
        if b.kind == "text":
            from .measure import fast_block
            el = S.els[b.text_idx]
            n0 = fast_block(el, int(round(float(el.get("font_size", 16) or 16))), b.w)[0]
            n1 = placed[b.bid].lines or n0
            ls += abs(math.log(max(1, n1) / max(1, n0)))
            nt += 1
    f["lines"] = ls / nt if nt else 0.0
    f["align"] = max(0.0, _align_lines(list(tgt.values()), W, H) - _align_lines(list(src.values()), S.W, S.H))
    f["busy"], f["contrast"] = _pixel_terms(S, placed, W, H, src, tgt)
    return [f[k] for k in NAMES]


def _lum_of(color) -> float:
    c = str(color or "rgba(0,0,0,1)")
    try:
        v = [float(t) for t in c[c.index("(") + 1:c.index(")")].split(",")[:3]]
        return (0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]) / 255.0
    except Exception:  # noqa: BLE001
        return 0.0


_RS = {}


def _rs(el, box, tb):
    from .pixels import region_stats
    box = tuple(float(round(v)) for v in box)
    tb = tuple(float(round(v)) for v in tb)
    key = (el.get("asset_id"), el.get("fit"), box, tb)
    r = _RS.get(key, 0)
    if r == 0:
        if len(_RS) > 50000:
            _RS.clear()
        r = _RS[key] = region_stats(el, box, tb[0], tb[1], tb[0] + tb[2], tb[1] + tb[3])
    return r


def _under(S, boxes_img, tb, tz):
    best = None
    for z, el, box in boxes_img:
        if z >= tz:
            continue
        r = _rs(el, box, tb)
        if r is not None and r[2] > 0.5 and (best is None or z > best[3]):
            best = (r[0], r[1], r[2], z)
    return best


def _pixel_terms(S, placed, W, H, src, tgt):
    imgs_s, imgs_t = [], []
    for i in S.backdrop:
        e = S.els[i]
        if e.get("type") == "image":
            imgs_s.append((e.get("z_index", 0), e, (float(e["x"]), float(e["y"]), float(e["width"]), float(e["height"]))))
            imgs_t.append((e.get("z_index", 0), dict(e, fit="cover"), (0.0, 0.0, W, H)))
    for b in S.blocks:
        if b.bid not in placed:
            continue
        if b.kind == "image":
            e = S.els[b.members[0]]
            imgs_s.append((e.get("z_index", 0), e, src[b.bid]))
            imgs_t.append((e.get("z_index", 0), e, tgt[b.bid]))
        elif b.kind == "group":
            k = tgt[b.bid][2] / max(1e-6, b.w)
            for i in b.members:
                e = S.els[i]
                if e.get("type") != "image":
                    continue
                x, y, w, h = float(e["x"]), float(e["y"]), float(e["width"]), float(e["height"])
                imgs_s.append((e.get("z_index", 0), e, (x, y, w, h)))
                imgs_t.append((e.get("z_index", 0), e, (tgt[b.bid][0] + (x - b.x) * k, tgt[b.bid][1] + (y - b.y) * k, w * k, h * k)))
    if not imgs_s:
        return 0.0, 0.0
    busy = con = 0.0
    for b in S.blocks:
        if b.kind != "text" or b.bid not in placed:
            continue
        el = S.els[b.text_idx]
        tz = el.get("z_index", 0)
        us = _under(S, imgs_s, src[b.bid], tz)
        ut = _under(S, imgs_t, tgt[b.bid], tz)
        if ut is None:
            continue
        es = us[0] if us else 0.0
        busy += max(0.0, ut[0] - es - 0.01)
        lt = _lum_of(el.get("color"))
        cs = abs(lt - us[1]) if us else 1.0
        ct = abs(lt - ut[1])
        con += max(0.0, min(cs, 0.5) - ct)
    return busy, con


def energy(S: Scene, placed: Dict[int, Placed], W: float, H: float) -> float:
    w = weights()
    fv = features(S, placed, W, H)
    return sum(w[k] * v for k, v in zip(NAMES, fv))
