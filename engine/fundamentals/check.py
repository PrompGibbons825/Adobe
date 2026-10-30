"""check(source, output, target) -> faults: is an output valid, judged by the renderer, not an engine.

    from fundamentals.check import check
    faults = check(source, output, target_canvas)               # JSON + renderer model
    faults = check(source, output, target_canvas, pixels=True)  # + rendered-pixel checks
    hard = [f for f in faults if f.level == "hard"]

Levels:
* ``hard``     the brief, RULES.md and the schema (ids, canvas, copy/fonts/assets unchanged, image
               proportions, nothing off-canvas but the S4 bleed policy, values the renderer can
               draw without crashing or drawing the wrong thing, no element hidden).
* ``promise``  what any good layout promises (no copy over copy that the source did not have, no
               small print, text contrast not lost, brand colours kept).
* ``info``     allowed bleed, text drawn outside its own box, model/pixel disagreements.

Text geometry comes from ``fundamentals.text`` (the renderer's own wrap and draw rules); with
``pixels=True`` clipping, overlap, visibility and contrast are measured on rendered pixels (each
element drawn alone on a padded transparent canvas, and the full design with and without it).
Imports nothing from ``engine/`` or ``tools/``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import bleed
from .fonts import fontbook
from .image import asset_size, image_fit
from .text import measure_text

TOL = 0.6  # px slack for float rounding at the canvas edge
ASPECT_TOL = 0.05
SCHEMA_ENUMS = {
    "type": ("text", "image", "shape"),
    "font_weight": ("regular", "bold"),
    "font_style": ("normal", "italic"),
    "text_align": ("left", "center", "right"),
    "fit": ("contain", "cover", "fill"),
    "shape_type": ("rectangle", "ellipse", "line"),
}
REQUIRED = {"text": ("content", "font_family", "font_size", "color"), "image": ("asset_id",),
            "shape": ("shape_type", "fill_color")}
KEEP_FIELDS = ("type", "content", "font_family", "asset_id", "shape_type")
CAP_AT_1024 = 7.0  # px of cap height when the judge views the canvas at 1024 px on the long side
VISIBLE_MIN = 0.10  # share of an element's own pixels that must survive in the full render (S1)
DIFF = 24  # channel difference that counts as "this element changed the pixel"


@dataclass
class Fault:
    level: str  # hard | promise | info
    code: str
    id: Optional[str]
    msg: str
    via: str = "json"  # json | model | pixels

    def __str__(self) -> str:
        return f"[{self.level}:{self.code}] {self.id or '-'}: {self.msg}"


# --------------------------------------------------------------------------- colours

def parse_color(name) -> Tuple[int, int, int, float]:
    """cr_renderer.color_utils.parse_color's acceptance (text colours). Raises like it does."""
    if not isinstance(name, str):
        raise TypeError(f"colour {name!r} is not a string")
    if name.startswith("rgba("):
        v = tuple(map(float, name.lstrip("rgba(").rstrip(")").split(",")))
        return int(v[0]), int(v[1]), int(v[2]), float(v[3])
    if name.startswith("rgb("):
        v = tuple(map(float, name.lstrip("rgb(").rstrip(")").split(",")))
        return int(v[0]), int(v[1]), int(v[2]), 1.0
    if name.startswith("cmyk"):
        raise ValueError("cmyk colours: avoid")
    if name.startswith("#") or len(name) in (3, 6):
        h = name.lstrip("#").replace("none", "000000").replace("undefined", "000000")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) != 6:
            raise ValueError(f"invalid hex {name!r}")
        n = int(h, 16)
        return (n >> 16) & 255, (n >> 8) & 255, n & 255, 1.0
    raise NotImplementedError(f"unknown colour format {name!r}")


def _shape_rgba(value) -> Optional[Tuple[int, int, int, float]]:
    """renderer/converter/colorutil.parse_rgba: None where the converter falls back to grey."""
    import re

    if not value or not isinstance(value, str):
        return None
    m = re.match(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)", value.strip())
    if not m:
        return None
    return int(float(m.group(1))), int(float(m.group(2))), int(float(m.group(3))), float(m.group(4) or 1.0)


def _same_colour(a, b) -> bool:
    try:
        ca, cb = _any_rgba(a), _any_rgba(b)
    except Exception:  # noqa: BLE001
        return a == b
    return ca[:3] == cb[:3] and abs(ca[3] - cb[3]) < 0.02


def _any_rgba(v):
    s = _shape_rgba(v)
    return s if s is not None else parse_color(v)


def _lum(rgb) -> float:
    def ch(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * ch(rgb[0]) + 0.7152 * ch(rgb[1]) + 0.0722 * ch(rgb[2])


def contrast(a, b) -> float:
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


# --------------------------------------------------------------------------- helpers

def _num(v) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def _inter(a, b) -> float:
    if a is None or b is None:
        return 0.0
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _visible_text(e: dict) -> bool:
    return e.get("type") == "text" and bool(str(e.get("content", "")).strip())


def _line_boxes(e: dict) -> List[tuple]:
    try:
        lay = measure_text(e)
    except Exception:  # noqa: BLE001
        return []
    return [b for b in (lay.canvas_box(ln.ink) for ln in lay.lines) if b is not None]


def _text_overlaps(design: dict) -> Dict[Tuple[str, str], float]:
    texts = [(str(e["id"]), _line_boxes(e)) for e in design.get("elements", []) if _visible_text(e)]
    out = {}
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            area = sum(_inter(a, b) for a in texts[i][1] for b in texts[j][1])
            if area > 4.0:
                out[tuple(sorted((texts[i][0], texts[j][0])))] = area
    return out


def small_print_floor(W: float, H: float, cap_em: float = 0.70) -> int:
    """Smallest int font size whose cap height is CAP_AT_1024 px at a 1024 px long-side view."""
    return max(1, math.ceil(CAP_AT_1024 * max(W, H) / 1024.0 / cap_em))


def capacity_bound(texts: List[dict], floor: int, W: float, H: float) -> float:
    """Area all copy needs at the floor size on one line each, over the canvas area. Above 1 the
    content provably cannot be set at the floor (wrapping only adds area)."""
    need = 0.0
    for e in texts:
        lay = measure_text(dict(e, font_size=floor, width=1e9, rotation=0), with_ink=False)
        pitch = floor * lay.line_height
        need += sum(ln.width for ln in lay.lines) * pitch
    return need / max(1.0, W * H)


# --------------------------------------------------------------------------- JSON + model

def check(source: dict, output: dict, target: dict, pixels: bool = False, assets=None,
          fonts=None) -> List[Fault]:
    src = source.get("source", source)
    faults: List[Fault] = []
    F = faults.append
    if not isinstance(output, dict) or not isinstance(output.get("elements"), list) or "canvas" not in output:
        return [Fault("hard", "structure", None, "output lacks canvas/elements")]
    W, H = target["width"], target["height"]
    if output.get("canvas") != {"width": W, "height": H}:
        F(Fault("hard", "canvas", None, f"canvas {output.get('canvas')} != {{'width': {W}, 'height': {H}}}"))
    sids = [str(e.get("id")) for e in src["elements"]]
    oids = [str(e.get("id")) for e in output["elements"]]
    for i in sorted(set(sids) - set(oids)):
        F(Fault("hard", "id_missing", i, "source element missing from the output"))
    for i in sorted(set(oids) - set(sids)):
        F(Fault("hard", "id_added", i, "element not in the source"))
    for i in sorted({i for i in oids if oids.count(i) > 1}):
        F(Fault("hard", "id_duplicate", i, f"appears {oids.count(i)} times"))

    so = {str(e.get("id")): e for e in src["elements"]}
    drawable = []
    for e in output["elements"]:
        eid = str(e.get("id"))
        s = so.get(eid, {})
        t = e.get("type")
        for k in KEEP_FIELDS:
            if k in s and e.get(k) != s.get(k):
                F(Fault("hard", f"{k}_changed", eid, f"{k} {s.get(k)!r} -> {e.get(k)!r}"))
        if t not in SCHEMA_ENUMS["type"]:
            F(Fault("hard", "schema", eid, f"type {t!r}"))
            continue
        geo = {k: _num(e.get(k)) for k in ("x", "y", "width", "height")}
        bad = [k for k, v in geo.items() if v is None]
        if bad:
            F(Fault("hard", "schema", eid, f"missing or non-finite {bad}"))
            continue
        if geo["width"] < 0 or geo["height"] < 0:
            F(Fault("hard", "schema", eid, f"negative size {geo['width']}x{geo['height']}"))
            continue
        for k in REQUIRED.get(t, ()):
            if k not in e:
                F(Fault("hard", "schema", eid, f"missing {k}"))
        for k in ("font_weight", "font_style", "text_align", "fit", "shape_type"):
            if k in e and e[k] not in SCHEMA_ENUMS[k]:
                F(Fault("hard", "schema", eid, f"{k} {e[k]!r} not in {SCHEMA_ENUMS[k]}"))
        for k in ("rotation", "opacity"):
            if k in e and _num(e[k]) is None:
                F(Fault("hard", "schema", eid, f"{k} {e[k]!r}"))
        if "opacity" in e and _num(e["opacity"]) is not None and not 0 <= e["opacity"] <= 1:
            F(Fault("hard", "schema", eid, f"opacity {e['opacity']} outside [0, 1]"))
        if "z_index" in e and (isinstance(e["z_index"], bool) or not isinstance(e["z_index"], int)):
            F(Fault("hard", "schema", eid, f"z_index {e['z_index']!r} is not an integer"))
        ok = True
        if t == "text":
            ok = _check_text(e, s, F)
        elif t == "image":
            _check_image(e, s, W, H, assets, F)
        else:
            _check_shape(e, s, F)
        if ok:
            drawable.append(e)

    for msg in bleed.check(src, dict(output, elements=[e for e in drawable]), target, assets)[0]:
        F(Fault("hard", "off_canvas", msg.split(":")[0], msg))
    for msg in bleed.check(src, dict(output, elements=[e for e in drawable]), target, assets)[1]:
        F(Fault("info", "bleed", msg.split(":")[0], msg))

    texts = [e for e in drawable if e.get("type") == "text" and _visible_text(e)]
    floor = small_print_floor(W, H)
    cap = capacity_bound(texts, floor, W, H) if texts else 0.0
    for e in texts:
        eid = str(e["id"])
        lay = measure_text(e)
        ib = lay.canvas_ink_box()
        if ib and (ib[0] < -TOL or ib[1] < -TOL or ib[2] > W + TOL or ib[3] > H + TOL):
            F(Fault("hard", "text_cut", eid, f"glyphs drawn past the canvas ({ib[0]:.0f},{ib[1]:.0f},{ib[2]:.0f},{ib[3]:.0f})", "model"))
        ov = lay.overflow()
        if ov["left"] > 1 or ov["right"] > 1:
            F(Fault("info", "text_wider_than_box", eid, f"ink leaves its box by {max(ov['left'], ov['right']):.0f}px horizontally", "model"))
        if ov["bottom"] > 1:
            F(Fault("info", "text_below_box", eid, f"ink runs {ov['bottom']:.0f}px below its box (height is not a clip)", "model"))
        if lay.draw_size < floor and _alpha(e) > 0:
            lvl, why = ("info", f", capacity {cap:.2f} > 1: cannot hold the floor") if cap > 1.0 else ("promise", "")
            F(Fault(lvl, "small_print", eid, f"drawn at {lay.draw_size}px < {floor}px floor{why}", "model"))
    src_ov = _text_overlaps(src)
    for (a, b), area in _text_overlaps(dict(output, elements=texts)).items():
        if (a, b) not in src_ov:
            F(Fault("promise", "copy_over_copy", a, f"ink of {a} and {b} overlap ({area:.0f} px2)", "model"))

    if pixels:
        faults += pixel_faults(src, output, target, assets, fonts, [e for e in drawable])
    return faults


def _alpha(e: dict) -> float:
    try:
        return parse_color(e.get("color"))[3]
    except Exception:  # noqa: BLE001
        return 1.0


def _check_text(e: dict, s: dict, F) -> bool:
    eid = str(e.get("id"))
    ok = True
    fs = _num(e.get("font_size"))
    if fs is None or fs < 1:
        F(Fault("hard", "schema", eid, f"font_size {e.get('font_size')!r} < 1"))
        ok = False
    elif fs != int(fs):
        F(Fault("info", "fractional_font_size", eid, f"font_size {fs} draws at {int(fs)}px (wrap measured at {fs})"))
    if geo_zero(e):
        F(Fault("hard", "render_crash", eid, "text width/height <= 0 crashes the renderer (pydantic gt=0)"))
        ok = False
    lh = e.get("line_height", 1.2)
    if _num(lh) is None or lh <= 0:
        F(Fault("hard", "render_crash", eid, f"line_height {lh!r} <= 0 crashes the renderer"))
        ok = False
    if e.get("text_align", "left") not in ("left", "center", "right", "justify"):
        F(Fault("hard", "render_crash", eid, f"text_align {e.get('text_align')!r} crashes the renderer"))
        ok = False
    cols = [("color", e.get("color"))] + [("style_run color", r.get("color")) for r in e.get("style_runs") or [] if "color" in r]
    for what, c in cols:
        try:
            parse_color(c)
        except Exception as x:  # noqa: BLE001
            F(Fault("hard", "render_crash", eid, f"{what} {c!r} crashes the renderer ({type(x).__name__})"))
            ok = False
    if s.get("color") and e.get("color") and not _same_colour(e.get("color"), s.get("color")):
        F(Fault("promise", "colour_changed", eid, f"{s.get('color')!r} -> {e.get('color')!r}"))
    if "&" in str(e.get("content", "")) and ";" in str(e.get("content", "")):
        import html
        if html.unescape(str(e["content"])) != str(e["content"]):
            F(Fault("info", "html_entity", eid, "content has HTML entities: drawn unescaped, line map shifted"))
    if _num(e.get("opacity", 1)) is not None and e.get("opacity", 1) < 1:
        F(Fault("info", "text_opacity_ignored", eid, f"opacity {e.get('opacity')} is ignored on text"))
    return ok


def geo_zero(e: dict) -> bool:
    return (_num(e.get("width")) or 0) <= 0 or (_num(e.get("height")) or 0) <= 0


def _check_image(e: dict, s: dict, W: float, H: float, assets, F) -> None:
    eid = str(e.get("id"))
    sw, sh = _num(s.get("width")) or 0, _num(s.get("height")) or 0
    w, h = float(e["width"]), float(e["height"])
    if w <= 0 or h <= 0:
        F(Fault("hard", "image_collapsed", eid, f"{w}x{h}"))
        return
    full = e["x"] <= TOL and e["y"] <= TOL and e["x"] + w >= W - TOL and e["y"] + h >= H - TOL \
        and e.get("fit") == "cover"
    if sw > 0 and sh > 0 and not full:
        ra, rb = sw / sh, w / h
        if abs(rb / ra - 1) > ASPECT_TOL:
            F(Fault("hard", "aspect", eid, f"box aspect {ra:.3f} -> {rb:.3f}"))
    from .render import assets_dir

    fit = image_fit(e, assets_dir(assets))
    if fit.missing:
        F(Fault("info", "missing_asset", eid, f"{e.get('asset_id')!r} not found: renders a grey placeholder"))
    elif fit.distortion > ASPECT_TOL:
        F(Fault("hard", "distorted", eid, f"fit {fit.fit!r} draws the asset {fit.distortion:.0%} off its aspect"))


def _check_shape(e: dict, s: dict, F) -> None:
    eid = str(e.get("id"))
    fc = e.get("fill_color")
    if fc is not None and _shape_rgba(fc) is None:
        F(Fault("hard", "shape_colour", eid, f"fill_color {fc!r} is not rgba()/rgb(): the converter draws grey"))
    if s.get("fill_color") and fc and not _same_colour(fc, s.get("fill_color")):
        F(Fault("promise", "colour_changed", eid, f"{s.get('fill_color')!r} -> {fc!r}"))
    if e.get("shape_type") != "line" and e.get("stroke_color") and (_num(e.get("stroke_width")) or 0) > 0:
        F(Fault("info", "stroke_ignored", eid, "stroke on a rectangle/ellipse is not drawn"))


# --------------------------------------------------------------------------- pixels

VIEW = 512  # long side of the low-resolution renders used for visibility and contrast


class _Px:
    """Renders of one design at the view scale: full, without-one, alone."""

    def __init__(self, design: dict, assets, fonts):
        from .render import Scene

        self.sc = Scene(design, assets, fonts)
        self.s = min(1.0, VIEW / max(self.sc.W, self.sc.H))
        self.full = self.sc.draw(scale=self.s).astype("int16")
        self.ids = list(self.sc.rows)

    def without(self, eid):
        return self.sc.draw([i for i in self.ids if i != eid], scale=self.s).astype("int16")

    def alone(self, eid):
        return self.sc.draw([eid], scale=self.s, transparent=True)[..., 3]

    def visible(self, eid) -> Optional[float]:
        import numpy as np

        own = self.alone(eid) > 0
        n = int(own.sum())
        if n == 0:
            return None
        changed = np.abs(self.full - self.without(eid))[..., :3].max(axis=2) > DIFF
        return float((changed & own).sum()) / n

    def contrast(self, e) -> Optional[float]:
        import numpy as np

        try:
            rgb = parse_color(e.get("color"))[:3]
        except Exception:  # noqa: BLE001
            return None
        own = self.alone(str(e["id"])) > 64
        if not own.any():
            return None
        bg = self.without(str(e["id"]))[..., :3][own]
        return contrast(rgb, tuple(np.percentile(bg, 50, axis=0)))


def pixel_faults(src: dict, output: dict, target: dict, assets=None, fonts=None,
                 drawable: Optional[List[dict]] = None) -> List[Fault]:
    from .render import mask_box

    W, H = int(target["width"]), int(target["height"])
    els = drawable if drawable is not None else output["elements"]
    design = {"canvas": {"width": W, "height": H}, "elements": els}
    faults: List[Fault] = []
    px = _Px(design, assets, fonts)
    spx: List[Optional[_Px]] = [None]

    def source():
        if spx[0] is None:
            spx[0] = _Px(src, assets, fonts)
        return spx[0]

    pad = 64
    roles = bleed.roles(src, assets)
    src_ov = _text_overlaps(src)
    texts = [e for e in els if _visible_text(e)]
    tfp = {}
    for e in texts:  # text alone at scale 1 on a padded canvas: clipping and model check
        eid = str(e["id"])
        a = px.sc.draw([eid], pad=pad, transparent=True)[..., 3]
        tfp[eid] = a
        total = int((a > 0).sum())
        if total == 0:
            continue
        on = int((a[pad:pad + H, pad:pad + W] > 0).sum())
        if total - on > 2:
            faults.append(Fault("hard", "text_cut", eid, f"{total - on} ink px past the canvas, ink box "
                                f"{tuple(round(v) for v in mask_box(a, 0, pad))}", "pixels"))
        mb, pb = measure_text(e).canvas_ink_box(), mask_box(a, 0, pad)
        if mb and pb and max(abs(m - p) for m, p in zip(mb, pb)) > 2.0 and not float(e.get("rotation", 0) or 0):
            faults.append(Fault("info", "model_mismatch", eid, f"model ink {tuple(round(v) for v in mb)} vs pixels "
                                f"{tuple(round(v) for v in pb)}", "pixels"))
    for e in els:
        eid = str(e["id"])
        if roles.get(eid) == "photo" and e.get("type") == "image" and bleed.covers(e, W, H):
            continue  # a background layer is covered by design
        vis = px.visible(eid)
        if vis is not None and vis < VISIBLE_MIN:
            v0 = source().visible(eid)
            hard = v0 is not None and v0 >= VISIBLE_MIN
            faults.append(Fault("hard" if hard else "info", "hidden", eid,
                                f"only {vis:.0%} of its pixels show" + (f" (source {v0:.0%})" if v0 is not None else ""), "pixels"))
    for e in texts:
        c = px.contrast(e)
        if c is not None and c < 3.0:
            c0 = source().contrast(e) if str(e["id"]) in source().sc.rows else None
            if c0 is None or c < 0.7 * min(c0, 4.5):
                faults.append(Fault("promise", "contrast", str(e["id"]), f"contrast {c:.1f}:1 against what is under it"
                                    + (f" (source {c0:.1f}:1)" if c0 else ""), "pixels"))
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = str(texts[i]["id"]), str(texts[j]["id"])
            n = int(((tfp[a] > 64) & (tfp[b] > 64)).sum())
            if n > 12 and tuple(sorted((a, b))) not in src_ov:
                faults.append(Fault("promise", "copy_over_copy", a, f"ink of {a} and {b} share {n} px", "pixels"))
    return faults
