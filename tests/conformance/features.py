"""Conformance: every schema feature, drawn by the official renderer, against what
``fundamentals`` predicts. Each case returns a list of mismatch strings (empty = conforms).

    python -m tests.conformance.run            # features + property set, one report
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from fundamentals import fit_geometry, image_fit, measure_text
from fundamentals.check import check, parse_color
from fundamentals.fonts import fontbook
from fundamentals.render import Scene, assets_dir, mask_box, render

TOL = 1.5  # px between the model's float ink box and the anti-aliased pixel box
W, H = 640, 360
PAD = 64


def T(**k) -> dict:
    e = dict(id="t", type="text", x=60, y=50, width=240, height=80, content="Hello world wide web of type",
             font_family="Solstice Sans", font_size=30, color="rgba(0,0,0,1)")
    e.update(k)
    return e


def S(**k) -> dict:
    e = dict(id="s", type="shape", x=60, y=50, width=120, height=70, shape_type="rectangle",
             fill_color="rgba(255,0,0,1)")
    e.update(k)
    return e


def I(**k) -> dict:
    e = dict(id="i", type="image", x=40, y=30, width=300, height=150, asset_id="solstice_bottle.png", fit="contain")
    e.update(k)
    return e


def alone(e: dict, w=W, h=H) -> np.ndarray:
    return Scene({"canvas": {"width": w, "height": h}, "elements": [e]}).draw(pad=PAD, transparent=True)


def _box_err(model, px) -> float:
    if model is None or px is None:
        return 0.0 if model is None and px is None else float("inf")
    return max(abs(a - b) for a, b in zip(model, px))


def text_case(e: dict, w=W, h=H) -> List[str]:
    from tests.conformance.prop import fit_pad

    lay = measure_text(e)
    pad = fit_pad(lay, w, h)
    img = Scene({"canvas": {"width": w, "height": h}, "elements": [e]}).draw(pad=pad, transparent=True)
    pb = mask_box(img[..., 3], 0, pad)
    mb = lay.canvas_ink_box()
    from tests.conformance.prop import rotated_err

    err = rotated_err(e, mb, pb) if (mb and pb and float(e.get("rotation", 0) or 0)) else _box_err(mb, pb)
    tol = TOL
    return [] if err <= tol else [f"ink box model {_r(mb)} vs pixels {_r(pb)} (err {err:.1f}px)"]


def _r(b):
    return None if b is None else tuple(round(v, 1) for v in b)


# --------------------------------------------------------------------------- feature cases

def wrap_rule() -> List[str]:
    """The model's wrapped string is the converter's, char for char."""
    from renderer.converter.convert import _get_renderer, convert_chameleon_to_crello

    r = _get_renderer(str(fontbook().dir))
    errs = []
    for e in (T(), T(width=90), T(content="Supercalifragilistic word", width=60), T(content="a\nb  c   d", width=20),
              T(font_size=30.9), T(font_weight="bold", font_style="italic", font_family="Solstice Serif"),
              T(font_weight="light"), T(font_family="Montserrat")):
        ex, _ = convert_chameleon_to_crello({"canvas": {"width": W, "height": H}, "elements": [e]}, r.font_manager)
        got = measure_text(e).wrapped
        if got != ex["text"][0]:
            errs.append(f"wrap {e['content']!r} w={e['width']}: model {got!r} != converter {ex['text'][0]!r}")
    return errs


def line_height() -> List[str]:
    return [m for lh in (0.8, 1.0, 1.2, 1.6, 2.4) for m in text_case(T(line_height=lh, height=200))]


def letter_spacing() -> List[str]:
    return [m for ls in (-2, 0, 3, 8, 15) for m in text_case(T(letter_spacing=ls))]


def alignment() -> List[str]:
    return [m for a in ("left", "center", "right", "justify") for m in text_case(T(text_align=a))]


def font_size_truncation() -> List[str]:
    a = alone(T(content="MMMMMMMM", font_size=20.99, width=600))
    b = alone(T(content="MMMMMMMM", font_size=20, width=600))
    errs = [] if np.array_equal(a, b) else ["20.99 px does not draw like 20 px"]
    return errs + [m for fs in (9.5, 12, 20.99, 47.3, 96) for m in text_case(T(font_size=fs, width=500))]


def fonts_and_fallback() -> List[str]:
    """Face resolution returns the renderer's bytes; unknown families fall back to Solstice Sans."""
    from renderer.converter.convert import _get_renderer

    fm = _get_renderer(str(fontbook().dir)).font_manager
    book = fontbook()
    errs = []
    for fam in ("Solstice Sans", "Solstice Serif", "Montserrat", "Lora", "SolsticeSans", "solstice_serif",
                "Solstice Sans Bold", "Solstice Serif Italic"):
        for wt in ("regular", "bold", "light", "semibold"):
            for st in ("regular", "bold", "italic", "bolditalic"):
                path = book.resolve(fam, wt, st)[0]
                if Path(path).read_bytes() != fm.lookup(fam, wt, st):
                    errs.append(f"lookup({fam!r}, {wt}, {st}) -> {path} differs from FontManager")
    for fam in ("Montserrat", "Lora"):
        errs += text_case(T(font_family=fam))
    for wt, st in (("bold", "normal"), ("bold", "italic"), ("regular", "italic"), ("light", "normal"), ("semibold", "normal")):
        errs += text_case(T(font_weight=wt, font_style=st))
        errs += text_case(T(font_weight=wt, font_style=st, font_family="Solstice Serif"))
    return errs


def text_opacity_ignored() -> List[str]:
    a, b = alone(T()), alone(T(opacity=0.2))
    return [] if np.array_equal(a, b) else ["text opacity changes the drawing"]


def colours() -> List[str]:
    errs = []
    for c in ("rgba(255,0,0,1)", "rgb(255,0,0)", "#ff0000", "#f00"):
        img = alone(T(color=c))
        ink = img[img[..., 3] == 255][..., :3]
        if not len(ink) or tuple(np.median(ink, axis=0).astype(int)) != (255, 0, 0):
            errs.append(f"text colour {c} not drawn red")
    for c in ("red", " rgba(0,0,0,1)"):
        try:
            alone(T(color=c))
            errs.append(f"colour {c!r} rendered (expected a crash)")
        except Exception:  # noqa: BLE001
            pass
        try:
            parse_color(c)
            errs.append(f"fundamentals.parse_color accepted {c!r}")
        except Exception:  # noqa: BLE001
            pass
    return errs


def style_runs() -> List[str]:
    errs = []
    e = T(content="AAAA BBBB", width=500, style_runs=[{"start": 0, "end": 2, "color": "rgba(255,0,0,1)"}])
    img = alone(e)
    red = (img[..., 0] > 150) & (img[..., 1] < 100) & (img[..., 3] > 128)
    lay = measure_text(e)
    # inclusive end: chars 0..2 are red; predicted red extent = ink of "AAA"
    three = measure_text(T(content="AAA", width=500)).canvas_ink_box()
    rb = mask_box(red, 0, PAD)
    if rb is None or abs(rb[2] - three[2]) > TOL:
        errs.append(f"run end is not inclusive: red ink ends at {rb and rb[2]}, 'AAA' ink ends at {three[2]:.1f}")
    errs += text_case(T(style_runs=[{"start": 0, "end": 10, "font_weight": "bold"}]))
    errs += text_case(T(style_runs=[{"start": 3, "end": 8, "font_style": "italic"}], text_align="center"))
    return errs + ([] if lay.n_lines >= 1 else ["no lines"])


def capitalize_and_entities() -> List[str]:
    return text_case(T(capitalize=True)) + text_case(T(content="Fish &amp; chips &lt;3 tonight"))


def overflow_and_height() -> List[str]:
    """Long words overflow, height is not a clip, centred overflow spills both sides."""
    return (text_case(T(content="Unbreakablewordhere", width=80, text_align="center"))
            + text_case(T(content="one two three four five six seven", width=60, height=10))
            + text_case(T(content="Right overflow", width=40, text_align="right")))


def rotation() -> List[str]:
    errs = []
    for deg in (15, 90, -30, 180):
        errs += text_case(T(rotation=deg))
        img = alone(S(rotation=deg))
        pb = mask_box(img[..., 3], 0, PAD)
        from fundamentals.text import to_canvas
        mb = to_canvas(S(rotation=deg), (0, 0, 120, 70))
        if _box_err(mb, pb) > 2.0:
            errs.append(f"shape rotated {deg}: model {_r(mb)} vs pixels {_r(pb)}")
    return errs


def shapes() -> List[str]:
    errs = []
    for k in ({}, {"shape_type": "ellipse"}, {"corner_radius": 20}, {"x": 60.4, "width": 120.6}):
        e = S(**k)
        pb = mask_box(alone(e)[..., 3], 0, PAD)
        mb = (e["x"], e["y"], e["x"] + e["width"], e["y"] + e["height"])
        if _box_err(mb, pb) > 1.0:
            errs.append(f"shape {k}: box {_r(mb)} vs pixels {_r(pb)}")
    img = alone(S(fill_color="#00ff00"))
    if tuple(img[PAD + 80, PAD + 100, :3]) != (200, 200, 200):
        errs.append(f"hex fill expected grey, got {tuple(img[PAD + 80, PAD + 100, :3])}")
    a = alone(S(fill_color="rgba(255,0,0,0.5)"))
    b = alone(S(opacity=0.5))
    if not np.array_equal(a, b):
        errs.append("fill alpha and opacity are not equivalent")
    img = alone(S(stroke_color="rgba(0,0,255,1)", stroke_width=6))
    if (img[..., 2] > 200).sum() and (img[..., 0] < 50).sum():
        errs.append("stroke drawn on a rectangle")
    ln = S(shape_type="line", stroke_width=4, stroke_color="rgba(0,0,0,1)")
    pb = mask_box(alone(ln)[..., 3], 0, PAD)
    y0 = ln["y"] + ln["height"] // 2
    if pb is None or abs(pb[1] - (y0 - 2)) > 1.5 or abs(pb[3] - (y0 + 2)) > 1.5:
        errs.append(f"line drawn at {pb}, expected rows {y0 - 2}..{y0 + 2}")
    return errs


def images_fit() -> List[str]:
    errs = []
    adir = assets_dir()
    for asset in ("solstice_bottle.png", "solstice_logo_lockup.png", "solstice_bg_gradient.png"):
        for fit in ("contain", "cover", "fill"):
            for w, h in ((300, 150), (120, 260), (200.6, 99.4)):
                e = I(asset_id=asset, fit=fit, width=w, height=h)
                f = image_fit(e, adir)
                img = alone(e)
                # the content box: where the bitmap (incl. its own transparent pixels) is drawn.
                # check against the alpha of the asset itself drawn with fill into f.content
                if fit == "contain":
                    from PIL import Image
                    with Image.open(adir / asset) as im:
                        aw, ah = im.size
                    g = fit_geometry(w, h, aw, ah, "contain")
                    cw, ch = g.content[2] - g.content[0], g.content[3] - g.content[1]
                    ref = alone(dict(e, fit="fill", x=e["x"] + g.content[0], y=e["y"] + g.content[1], width=cw, height=ch))
                    pa, pr = mask_box(img[..., 3], 8, PAD), mask_box(ref[..., 3], 8, PAD)
                    if _box_err(pa, pr) > 2.0:
                        errs.append(f"{asset} contain {w}x{h}: drawn {_r(pa)} vs predicted content {_r(pr)}")
                if fit in ("contain", "cover") and f.distortion > 0.02:
                    errs.append(f"{asset} {fit} {w}x{h}: predicted distortion {f.distortion:.3f}")
                if fit == "cover":
                    pb = mask_box(img[..., 3], 0, PAD)
                    if asset == "solstice_bg_gradient.png" and _box_err(pb, (e["x"], e["y"], e["x"] + w, e["y"] + h)) > 1.0:
                        errs.append(f"{asset} cover {w}x{h}: drawn {_r(pb)} does not fill the box")
    a = alone(I(opacity=0.5))
    b = alone(I())
    if not (a[..., 3].max() < b[..., 3].max()):
        errs.append("image opacity ignored")
    miss = alone(I(asset_id="nope.png"))
    pb = mask_box(miss[..., 3], 0, PAD)
    if _box_err(pb, (40, 30, 340, 180)) > 1.0 or tuple(miss[PAD + 60, PAD + 60, :3]) != (214, 214, 214):
        errs.append(f"missing asset placeholder at {pb}, colour {tuple(miss[PAD + 60, PAD + 60, :3])}")
    return errs


def z_order() -> List[str]:
    """Stable sort by z_index (missing = 0): later wins ties."""
    a = dict(S(id="a", fill_color="rgba(255,0,0,1)"))
    b = dict(S(id="b", fill_color="rgba(0,0,255,1)"))
    errs = []
    for els, want in (([a, b], (0, 0, 255)), ([b, a], (255, 0, 0)), ([dict(a, z_index=2), b], (255, 0, 0)),
                      ([dict(a, z_index=1), dict(b, z_index=1)], (0, 0, 255)), ([dict(b, z_index=-1), a], (255, 0, 0))):
        img = render({"canvas": {"width": W, "height": H}, "elements": els}, max_size=None)
        got = tuple(int(v) for v in img[80, 100, :3])
        if got != want:
            errs.append(f"z order {[(e['id'], e.get('z_index')) for e in els]}: top {got}, expected {want}")
    return errs


def off_canvas_and_scale() -> List[str]:
    errs = []
    img = render({"canvas": {"width": 3000, "height": 500}, "elements": [S(x=0, y=0, width=3000, height=10)]})
    if img.shape[:2] != (341, 2048):
        errs.append(f"3000x500 canvas rendered at {img.shape[:2]}, expected (341, 2048)")
    e = T(x=-30, y=-10)
    faults = [f for f in check({"canvas": {"width": W, "height": H}, "elements": [T()]},
                               {"canvas": {"width": W, "height": H}, "elements": [e]}, {"width": W, "height": H},
                               pixels=True) if f.code == "text_cut"]
    vias = {f.via for f in faults}
    if vias != {"model", "pixels"}:
        errs.append(f"text past the canvas: text_cut via {vias}, expected model and pixels")
    return errs


def scene_equals_render() -> List[str]:
    """The cheap Scene path draws exactly what render_design_to_png draws."""
    import json
    from pathlib import Path

    errs = []
    for p in sorted(Path("input").glob("*.json"))[:3]:
        src = json.loads(p.read_text())["source"]
        a = render(src, max_size=None)
        b = Scene(src).draw()
        if a.shape != b.shape or np.abs(a.astype(int) - b.astype(int))[..., :3].max() > 1:
            errs.append(f"{p.name}: Scene differs from render ({np.abs(a.astype(int) - b.astype(int)).max() if a.shape == b.shape else 'shape'})")
    return errs


CASES: Dict[str, Callable[[], List[str]]] = {
    "wrap rule": wrap_rule,
    "line height": line_height,
    "letter spacing": letter_spacing,
    "alignment": alignment,
    "font size truncation": font_size_truncation,
    "fonts and fallback": fonts_and_fallback,
    "text opacity ignored": text_opacity_ignored,
    "colours": colours,
    "style runs": style_runs,
    "capitalize, HTML entities": capitalize_and_entities,
    "overflow, height not a clip": overflow_and_height,
    "rotation": rotation,
    "shapes": shapes,
    "images: fit, opacity, missing": images_fit,
    "z-order": z_order,
    "off-canvas, >2048 scale": off_canvas_and_scale,
    "Scene == render": scene_equals_render,
}
