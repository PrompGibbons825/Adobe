"""The hard rules, enforced on every output, plus the type floor and the promise checks that
candidate selection ranks by.

* ``normalize_source``  the source as the renderer will read it: numbers as floats, colours as
  rgba(), valid enums, junk entries dropped. Ids, content and font families are never touched.
* ``type_floor``        the smallest text size any layout may set (explicit, STEERING S1).
* ``conform``           the last pass on the chosen layout: values the renderer can draw, image
  proportions kept (only a canvas-covering ``cover`` image may crop), and nothing past the canvas
  except the bleed loop/STEERING.md S4 allows (photos and decoration keep 60% on the canvas; text,
  logos, buttons, badges and cutouts never bleed unless the source bled them).
* ``faults``            what a layout breaks of the engine's promises (small print, glyphs past the
  canvas, new copy over copy, a label off its button). Candidates are ranked by it before score.

Bleed roles mirror the S4 policy and come from the source geometry only (the input has no labels).
"""

from __future__ import annotations

import copy
import math
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from . import textfit as tf

TOL = 0.6          # px of slack at the canvas edge
KEEP = 0.6         # share of an allowed bleed's box that stays on the canvas
SLACK = 0.05       # share a bleed may lose against its source share
SMALL_IMAGE = 0.10  # an opaque image under this share of the source canvas is an icon / thumbnail
BIG_PANEL = 0.35   # a shape holding copy and at least this share of the canvas is a panel
LABEL_FILL = 0.12  # a text filling this share of a shape it sits in makes it a button / badge
COVERING = 0.9     # a transparent image covering this share of the canvas is a background layer


def num(v, default=0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def type_floor(W: float, H: float) -> int:
    """No visible text below this size: a cap height of about 7 px when the canvas is seen at
    1024 px on its long side (research 04), and never under 10 px."""
    return max(10, math.ceil(10.0 * max(W, H) / 1024.0))


# ----------------------------------------------------------------------------
# colours and values

_RGBA = re.compile(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)")
_NAMED = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0), "green": (0, 128, 0),
          "blue": (0, 0, 255), "yellow": (255, 255, 0), "orange": (255, 165, 0), "gray": (128, 128, 128),
          "grey": (128, 128, 128), "purple": (128, 0, 128), "pink": (255, 192, 203), "brown": (165, 42, 42)}


def colour(value) -> Optional[str]:
    """The same colour as an rgba() string the renderer parses, or None if it is not a colour."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    low = v.lower()
    if low.startswith("rgb"):
        m = _RGBA.match(low)
        if not m:
            return None
        if v.startswith("rgba(") or v.startswith("rgb("):
            return v
        r, g, b = (max(0, min(255, int(float(m.group(i))))) for i in (1, 2, 3))
        a = max(0.0, min(1.0, float(m.group(4)))) if m.group(4) is not None else 1.0
        return f"rgba({r},{g},{b},{a:g})"
    if low in _NAMED:
        r, g, b = _NAMED[low]
        return f"rgba({r},{g},{b},1)"
    if low == "transparent":
        return "rgba(0,0,0,0)"
    h = low.lstrip("#")
    if h and all(c in "0123456789abcdef" for c in h) and len(h) in (3, 4, 6, 8):
        if len(h) in (3, 4):
            h = "".join(c * 2 for c in h)
        a = int(h[6:8], 16) / 255.0 if len(h) == 8 else 1.0
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a:.3g})"
    return None


def _safe_values(e: dict) -> None:
    """Normalise one element's values in place (same meaning, a form the renderer accepts)."""
    t = e.get("type")
    for k in ("x", "y"):
        e[k] = num(e.get(k), 0.0)
    for k in ("width", "height"):
        e[k] = max(0.0, num(e.get(k), 0.0))
    e["rotation"] = num(e.get("rotation"), 0.0)
    if "opacity" in e:
        e["opacity"] = min(1.0, max(0.0, num(e.get("opacity"), 1.0)))
    if t == "text":
        if "font_family" in e and not isinstance(e["font_family"], str):
            e.pop("font_family")      # the renderer falls back to its default family
        e["font_size"] = max(1.0, num(e.get("font_size"), 16.0))
        if e.get("text_align", "left") not in ("left", "center", "right"):
            e["text_align"] = "left"
        fw = e.get("font_weight")
        if fw is not None and not isinstance(fw, str):
            e["font_weight"] = "bold" if num(fw, 400.0) >= 600 else "regular"
        elif isinstance(fw, str) and fw not in ("regular", "bold"):
            e["font_weight"] = "bold" if fw.lower() in ("bold", "semibold", "extrabold", "black", "heavy") \
                or num(fw, 400.0) >= 600 else "regular"
        fs = e.get("font_style")
        if fs is not None and fs not in ("normal", "italic"):
            e["font_style"] = "italic" if str(fs).lower() == "italic" else "normal"
        if "line_height" in e:
            lh = num(e.get("line_height"), 1.2)
            e["line_height"] = lh if lh > 0 else 1.2
        if "letter_spacing" in e:
            e["letter_spacing"] = num(e.get("letter_spacing"), 0.0)
        e["color"] = colour(e.get("color")) or "rgba(0,0,0,1)"
        runs = e.get("style_runs")
        if isinstance(runs, list):
            e["style_runs"] = [r for r in map(_safe_run, runs) if r is not None]
        elif runs is not None:
            e.pop("style_runs")
    elif t == "shape":
        c = colour(e.get("fill_color"))
        if c:
            e["fill_color"] = c
        if e.get("stroke_color") is not None:
            c = colour(e.get("stroke_color"))
            if c:
                e["stroke_color"] = c
        for k in ("corner_radius", "stroke_width"):
            if k in e:
                e[k] = max(0.0, num(e[k], 0.0))
    elif t == "image":
        if e.get("fit", "contain") not in ("contain", "cover", "fill"):
            e["fit"] = "contain"


def _safe_run(r) -> Optional[dict]:
    if not isinstance(r, dict) or not all(isinstance(r.get(k, 0), int) for k in ("start", "end")):
        return None
    r = dict(r)
    if "color" in r:
        c = colour(r["color"])
        if c is None:
            r.pop("color")
        else:
            r["color"] = c
    if "font_weight" in r and not isinstance(r["font_weight"], str):
        r.pop("font_weight")
    return r


def normalize_source(source: dict) -> dict:
    """A copy of the source the engine can lay out: every element a dict with numeric geometry,
    valid enums and rgba() colours. Ids, content, font families and assets are kept as given."""
    src = copy.deepcopy(source) if isinstance(source, dict) else {}
    cv = src.get("canvas") if isinstance(src.get("canvas"), dict) else {}
    els, other = [], []
    for i, e in enumerate(src.get("elements") or []):
        if not isinstance(e, dict):
            continue
        if e.get("type") not in ("text", "image", "shape"):
            if "id" in e:
                other.append(e)      # a type the schema does not know: passed through untouched
            continue
        if "id" not in e:
            e["id"] = f"_element_{i}"   # a broken export: it still needs a name to be laid out
        if not isinstance(e.get("z_index"), (int, float)) or not math.isfinite(e["z_index"]):
            e["z_index"] = i
        _safe_values(e)
        els.append(e)
    W0 = max(1.0, num(cv.get("width"), 0.0))
    H0 = max(1.0, num(cv.get("height"), 0.0))
    if not cv:
        xs = [e["x"] + e["width"] for e in els] or [1.0]
        ys = [e["y"] + e["height"] for e in els] or [1.0]
        W0, H0 = max(1.0, max(xs)), max(1.0, max(ys))
    src["canvas"] = {"width": W0, "height": H0}
    src["elements"] = els
    src["_other"] = other
    return src


# ----------------------------------------------------------------------------
# geometry


def aabb(e: dict) -> Tuple[float, float, float, float]:
    x, y, w, h = num(e.get("x")), num(e.get("y")), max(0.0, num(e.get("width"))), max(0.0, num(e.get("height")))
    rot = num(e.get("rotation"))
    if not rot:
        return x, y, x + w, y + h
    t = math.radians(rot)
    c, s = abs(math.cos(t)), abs(math.sin(t))
    rw, rh = w * c + h * s, w * s + h * c
    cx, cy = x + w / 2, y + h / 2
    return min(x, cx - rw / 2), min(y, cy - rh / 2), max(x + w, cx + rw / 2), max(y + h, cy + rh / 2)


def share_on(e: dict, W: float, H: float) -> float:
    x0, y0, x1, y1 = aabb(e)
    a = (x1 - x0) * (y1 - y0)
    if a <= 0:
        return 1.0 if x0 >= -TOL and y0 >= -TOL and x1 <= W + TOL and y1 <= H + TOL else 0.0
    return max(0.0, min(x1, W) - max(x0, 0.0)) * max(0.0, min(y1, H) - max(y0, 0.0)) / a


def _bleeds(b, W, H) -> bool:
    return b[0] < -TOL or b[1] < -TOL or b[2] > W + TOL or b[3] > H + TOL


def _covers(b, W, H) -> bool:
    return b[0] <= TOL and b[1] <= TOL and b[2] >= W - TOL and b[3] >= H - TOL


def glyph_box(e: dict) -> Tuple[float, float, float, float]:
    """Where a text's glyphs land as the renderer draws them: wrapped at the element width, lines
    down from y, aligned in the width (an unbreakable word can reach past it), turned about the
    element centre when rotated."""
    fs = max(1, int(num(e.get("font_size"), 12)))
    w = num(e.get("width"))
    m = tf.measure(e, fs, w)
    asc, cap, desc = tf.metrics(e)
    lh = num(e.get("line_height"), 1.2) or 1.2
    h = (m["lines"] - 1) * fs * lh + max(fs * lh, (asc + desc) * fs)
    ink = m["width"]
    al = e.get("text_align", "left")
    x0 = 0.0 if al == "left" else ((w - ink) / 2 if al == "center" else w - ink)
    x, y = num(e.get("x")), num(e.get("y"))
    b = (x + x0, y, x + x0 + ink, y + h)
    rot = num(e.get("rotation"))
    if rot:
        from .blocks import rotate_box_about
        b = rotate_box_about(b, x + w / 2, y + num(e.get("height")) / 2, rot)
    return b


# ----------------------------------------------------------------------------
# roles (S4)


@lru_cache(maxsize=512)
def _transparent(asset_id: str) -> Optional[bool]:
    from .saliency import _load
    try:
        im = _load(asset_id)
    except Exception:  # noqa: BLE001  an unreadable asset is decided by geometry
        return None
    if im is None:
        return None
    import numpy as np
    small = im.copy()
    small.thumbnail((256, 256))
    a = np.asarray(small.getchannel("A"))
    return bool((a < 250).mean() >= 0.05)


def _inside(a, b, tol=2.0) -> bool:
    return a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol


def roles(src: dict) -> Dict[str, str]:
    """id -> text | info | photo | decor, from the source geometry (STEERING S4)."""
    W, H = src["canvas"]["width"], src["canvas"]["height"]
    area = max(1e-9, W * H)
    texts = [(aabb(e), e.get("group_id")) for e in src["elements"]
             if e["type"] == "text" and str(e.get("content", "")).strip()]
    out = {}
    for e in src["elements"]:
        b = aabb(e)
        share = (b[2] - b[0]) * (b[3] - b[1]) / area
        if e["type"] == "text":
            out[e["id"]] = "text"
        elif e["type"] == "image":
            tr = _transparent(str(e.get("asset_id"))) if e.get("asset_id") else None
            if tr is True:
                out[e["id"]] = "photo" if share >= COVERING else "info"
            else:
                out[e["id"]] = "photo" if share >= SMALL_IMAGE else "info"
        else:
            ba = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
            holds = any(_inside(tb, b) and ((g is not None and g == e.get("group_id"))
                                            or (tb[2] - tb[0]) * (tb[3] - tb[1]) >= LABEL_FILL * ba)
                        for tb, g in texts)
            out[e["id"]] = "info" if holds and share < BIG_PANEL else "decor"
    return out


# ----------------------------------------------------------------------------
# conform


def _fit_text(e: dict, W: float, H: float) -> None:
    """A text whose glyphs cannot fit the canvas gets a narrower box or a smaller size."""
    if num(e["width"]) > W:
        e["width"] = W
    fs = int(e["font_size"])
    while fs > 1:
        b = glyph_box(dict(e, font_size=fs))
        if b[2] - b[0] <= W + TOL and b[3] - b[1] <= H + TOL:
            break
        fs -= 1
    e["font_size"] = fs


def _shift_inside(boxes, W, H) -> Tuple[float, float]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    dx = -x0 if x0 < 0 else (W - x1 if x1 > W else 0.0)
    dy = -y0 if y0 < 0 else (H - y1 if y1 > H else 0.0)
    return dx, dy


def _scale_about(e: dict, k: float, cx: float, cy: float) -> None:
    e["x"] = cx + (e["x"] - cx) * k
    e["y"] = cy + (e["y"] - cy) * k
    e["width"] *= k
    e["height"] *= k
    if e["type"] == "text":
        e["font_size"] = max(1, int(e["font_size"] * k))
    if e.get("corner_radius"):
        e["corner_radius"] *= k


def _keep_ratio(e: dict, s: dict, W: float, H: float) -> None:
    sw, sh = num(s.get("width")), num(s.get("height"))
    if sw <= 0 or sh <= 0:
        return
    if e["width"] < 1e-3 or e["height"] < 1e-3:
        cx, cy = e["x"] + e["width"] / 2, e["y"] + e["height"] / 2
        r = sw / sh
        e["width"], e["height"] = (1.0, 1.0 / r) if r >= 1 else (r, 1.0)
        e["x"], e["y"] = cx - e["width"] / 2, cy - e["height"] / 2
        return
    if _covers(aabb(e), W, H):
        e["fit"] = "cover"    # a canvas-covering image is a cover crop: never stretched
        return
    ra, rb = sw / sh, e["width"] / e["height"]
    if abs(rb / ra - 1) > 0.01:
        cx, cy = e["x"] + e["width"] / 2, e["y"] + e["height"] / 2
        a = e["width"] * e["height"]
        e["width"] = math.sqrt(a * ra)
        e["height"] = e["width"] / ra
        e["x"], e["y"] = cx - e["width"] / 2, cy - e["height"] / 2


def _text_extent(e: dict):
    """Box plus glyphs of a text (both must stay on the canvas)."""
    a, g = aabb(e), glyph_box(e)
    return (min(a[0], g[0]), min(a[1], g[1]), max(a[2], g[2]), max(a[3], g[3]))


def conform(src: dict, out: dict, target: dict) -> dict:
    """Enforce the hard rules on a layout (in place; also returned)."""
    W, H = max(1.0, num(target.get("width"), 1.0)), max(1.0, num(target.get("height"), 1.0))
    SW, SH = src["canvas"]["width"], src["canvas"]["height"]
    so = {e["id"]: e for e in src["elements"]}
    by_id = {e.get("id"): e for e in out.get("elements", [])}
    els = []
    for s in src["elements"]:            # every source id exactly once, in source order
        e = by_id.get(s["id"])
        e = dict(s) if e is None else e
        for k in ("id", "type", "content", "font_family", "asset_id", "shape_type"):
            if k in s:
                e[k] = s[k]
        _safe_values(e)
        if e["type"] == "text":
            e["font_size"] = max(1, int(round(e["font_size"])))
            e["width"], e["height"] = max(1.0, e["width"]), max(1.0, e["height"])
        if e["type"] == "image":
            _keep_ratio(e, s, W, H)
        els.append(e)
    role = roles(src)

    # text, buttons, badges, logos and cutouts stay on the canvas (whole groups move together)
    def fixed(e):
        r = role.get(e["id"], "info")
        return r == "text" or (r == "info" and not _bleeds(aabb(so[e["id"]]), SW, SH))

    for e in els:
        if e["type"] == "text":
            _fit_text(e, W, H)
            g = glyph_box(e)              # the box need be no taller than its lines
            e["height"] = max(1.0, min(e["height"], max(g[3] - e["y"], 1.0)))
    groups: Dict[str, List[dict]] = {}
    for e in els:
        if fixed(e):
            groups.setdefault(str(so[e["id"]].get("group_id") or "\0" + str(e["id"])), []).append(e)
    for members in groups.values():
        for _ in range(3):
            boxes = [_text_extent(m) if m["type"] == "text" else aabb(m) for m in members]
            bw = max(b[2] for b in boxes) - min(b[0] for b in boxes)
            bh = max(b[3] for b in boxes) - min(b[1] for b in boxes)
            if bw > W + TOL or bh > H + TOL:
                k = 0.999 * min(W / max(bw, 1e-6), H / max(bh, 1e-6))
                cx = (max(b[2] for b in boxes) + min(b[0] for b in boxes)) / 2
                cy = (max(b[3] for b in boxes) + min(b[1] for b in boxes)) / 2
                for m in members:
                    _scale_about(m, k, cx, cy)
                continue
            dx, dy = _shift_inside(boxes, W, H)
            for m in members:
                m["x"] += dx
                m["y"] += dy
            if not dx and not dy:
                break
    # photos and decoration may bleed, keeping their meaningful part on the canvas
    for e in els:
        if fixed(e):
            continue
        b = aabb(e)
        if not _bleeds(b, W, H) or (e["type"] == "image" and _covers(b, W, H) and role.get(e["id"]) == "photo"):
            continue
        s = so[e["id"]]
        src_bled = _bleeds(aabb(s), SW, SH)
        r = role.get(e["id"], "info")
        need = min(KEEP, share_on(s, SW, SH)) if (r != "info" or src_bled) else 1.0
        need = max(0.0, need - SLACK) + 0.01 if need < 1.0 else 1.0
        if share_on(e, W, H) >= need:
            continue
        bw, bh = b[2] - b[0], b[3] - b[1]
        if bw > W or bh > H:
            k = 0.999 * min(W / max(bw, 1e-6), H / max(bh, 1e-6))
            if need >= 1.0 or k * k < need:   # shrink only as far as needed
                _scale_about(e, k, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            b = aabb(e)
        dx, dy = _shift_inside([b], W, H)
        lo, hi = 0.0, 1.0
        x0, y0 = e["x"], e["y"]
        for _ in range(20):             # the least shift that keeps enough on the canvas
            mid = (lo + hi) / 2
            e["x"], e["y"] = x0 + dx * mid, y0 + dy * mid
            if share_on(e, W, H) >= need:
                hi = mid
            else:
                lo = mid
        e["x"], e["y"] = x0 + dx * hi, y0 + dy * hi
    for e in els:
        for k in ("x", "y", "width", "height"):
            e[k] = round(float(e[k]), 2)
        if e["type"] == "text":
            e["width"], e["height"] = max(1.0, e["width"]), max(1.0, e["height"])
    out["canvas"] = {"width": int(W) if float(W).is_integer() else W, "height": int(H) if float(H).is_integer() else H}
    out["elements"] = els + [copy.deepcopy(e) for e in src.get("_other", [])]
    return out


# ----------------------------------------------------------------------------
# promises (ranked before score)


def fault_key(fl: List[str], floor: int) -> Tuple[int, float]:
    """Defects (glyphs past the canvas, copy over copy, a label off its button) count one each;
    small print counts by how far under the floor it is, so 9 px beats 3 px."""
    defects, short = 0, 0.0
    for f in fl:
        m = re.search(r"small print (\d+) < (\d+)", f)
        if m:
            short += 1.0 - int(m.group(1)) / int(m.group(2))
        else:
            defects += 1
    return defects, round(short, 1)


def faults(src: dict, out: dict, W: float, H: float) -> List[str]:
    """What this layout breaks of the engine's promises. Selection prefers fewer (fault_key)."""
    errs = []
    floor = type_floor(W, H)
    texts = [e for e in out["elements"] if e["type"] == "text" and num(e.get("opacity"), 1.0) > 0
             and str(e.get("content", "")).strip()]
    boxes = {}
    for e in texts:
        if e["font_size"] < floor:
            errs.append(f"{e['id']}: small print {e['font_size']} < {floor}")
        g = glyph_box(e)
        boxes[e["id"]] = g
        if g[0] < -TOL or g[1] < -TOL or g[2] > W + TOL or g[3] > H + TOL:
            errs.append(f"{e['id']}: glyphs past the canvas")
    so = {e["id"]: e for e in src["elements"]}
    sbox = {}
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i]["id"], texts[j]["id"]
            f = _ov_frac(boxes[a], boxes[b])
            if f <= 0:
                continue
            for k in (a, b):
                if k not in sbox:
                    sbox[k] = glyph_box(so[k])
            f0 = _ov_frac(sbox[a], sbox[b])
            if f0 < 0.15 and f > f0 + 0.02:
                errs.append(f"copy over copy: {a} x {b}")
    oo = {e["id"]: e for e in out["elements"]}
    for t in texts:
        g = so[t["id"]].get("group_id")
        if not g:
            continue
        for sh in src["elements"]:
            if sh["type"] != "shape" or sh.get("group_id") != g or num(oo[sh["id"]].get("opacity"), 1.0) <= 0:
                continue
            if not _inside(glyph_box(so[t["id"]]), aabb(so[sh["id"]])):
                continue
            if not oo[sh["id"]].get("rotation") and not _inside(boxes[t["id"]], aabb(oo[sh["id"]])):
                errs.append(f"{t['id']}: label left its shape {sh['id']}")
    return errs


def _ov_frac(a, b) -> float:
    ow = min(a[2], b[2]) - max(a[0], b[0])
    oh = min(a[3], b[3]) - max(a[1], b[1])
    if ow <= 1.0 or oh <= 1.0:
        return 0.0
    amin = max(1.0, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))
    return ow * oh / amin
