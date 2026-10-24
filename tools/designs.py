"""Dev tool: plausible source designs from a small design grammar.

    .venv/bin/python -m tools.designs --n 40 --seed 1 -o /tmp/designs   # writes <name>.json tasks

Each design is laid out the way a template designer would lay it out: one of a
handful of arrangements (centred column, left column beside a product, product
beside copy, a banner row, a card or a band over the background), copy of any
length from a few domains, a light or dark theme, and optional extras (kicker,
subhead, legal, badge, decoration, a coloured run in the headline). Nothing in
the engine is tuned to these; they are an evaluation set the engine has not
seen, used by tools.audit. Element ids name their role, so the audit can read
them; the engine never sees ids as roles.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.smart.textfit import TextStyle, measure  # noqa: E402
from tools.sheet import CATALOGUE  # noqa: E402

SERIF, SANS = "Solstice Serif", "Solstice Sans"
DARK = {"bg": "image", "fg": "rgba(255,255,255,1)", "soft": "rgba(255,255,255,0.82)", "accent": "rgba(232,106,58,1)",
        "gold": "rgba(240,178,66,1)", "on_accent": "rgba(255,255,255,1)", "panel": "rgba(26,16,30,0.86)"}
LIGHT = {"bg": "rgba(250,240,228,1)", "fg": "rgba(43,27,46,1)", "soft": "rgba(43,27,46,0.78)",
         "accent": "rgba(232,106,58,1)", "gold": "rgba(196,120,20,1)", "on_accent": "rgba(255,255,255,1)",
         "panel": "rgba(255,255,255,0.92)"}

COPY = {
    "drinks": {
        "headline": ["Golden hour, bottled", "Chase the Light", "Sip the sunset", "Made for long summer evenings",
                     "A taste of the longest day of the year, in every single bottle",
                     "New botanical sparkling infusions for warm nights and good company",
                     "Cold", "Summer, served"],
        "sub": ["Sparkling botanicals with blood orange and wild honey.", "Small-batch, lightly sparkling.",
                "Three new flavours, only until the equinox. Crafted with real fruit, no added sugar, and a "
                "little bit of sunshine."],
        "cta": ["Shop now", "Order today", "Find a store", "Shop the summer collection"],
        "kicker": ["NEW", "SUMMER EDITION", "LIMITED RELEASE"],
        "badge": ["-20%", "NEW", "2 for 1"],
    },
    "property": {
        "headline": ["Your new home is waiting", "Open house this Sunday",
                     "Bright two-bedroom apartment with a garden view, minutes from the station", "Sold"],
        "sub": ["Viewings from 10am to 4pm, no appointment needed.", "Call us to book a private tour.",
                "Spacious living room, renovated kitchen, private balcony and secure parking for two cars."],
        "cta": ["Book a viewing", "Call now", "See the listing"],
        "kicker": ["JUST LISTED", "OPEN HOUSE"],
        "badge": ["NEW", "SOLD"],
    },
    "fitness": {
        "headline": ["Stronger every day", "Your first class is free", "Train smarter, not longer",
                     "Join the club that makes Monday mornings feel like a reward"],
        "sub": ["Classes every morning at 6am.", "No contracts. Cancel any time.",
                "Strength, mobility and conditioning programmes for every level, coached by people who care."],
        "cta": ["Join now", "Start free trial", "Book"],
        "kicker": ["NEW YEAR OFFER", "THIS WEEK ONLY"],
        "badge": ["FREE", "-50%"],
    },
    "events": {
        "headline": ["Sun Festival", "Live at the harbour", "Three days of music, food and fireworks by the sea",
                     "Tonight"],
        "sub": ["June 21-23, Harbour Park.", "Tickets on sale now. Kids under 12 go free.",
                "Forty artists on four stages, a night market, and the biggest fireworks show of the summer."],
        "cta": ["Get tickets", "Learn more", "RSVP"],
        "kicker": ["SUMMER 2027", "LINE-UP ANNOUNCED"],
        "badge": ["SOLD OUT", "VIP"],
    },
}

SOURCES = ["ig-square", "ig-square", "ig-portrait", "story", "fb-link", "yt-thumb", "leaderboard", "billboard",
           "li-banner", "pinterest", "med-rect", "email-header", "pres-16x9"]


class Builder:
    """Collects elements with increasing z and the right shapes for text and buttons."""

    def __init__(self, W, H, theme, rng):
        self.W, self.H, self.t, self.rng, self.els, self.z = W, H, theme, rng, [], 0

    def add(self, el):
        self.z += 1
        el["z_index"] = self.z
        self.els.append(el)
        return el

    def block(self, content, size, family, bold, width, **kw):
        size = round(size)
        el = {"content": content, "font_family": family, "font_size": size, "font_weight": "bold" if bold else None,
              **kw}
        return measure(content, TextStyle.of(el), size, width, exact=True)

    def text_h(self, content, size, family, bold, width, **kw):
        blk = self.block(content, size, family, bold, width, **kw)
        return blk.n_lines * blk.line_step

    def text(self, id, x, y, w, content, size, family=SANS, bold=False, color=None, align="center", **kw):
        size = round(size)
        h = self.text_h(content, size, family, bold, w, **kw)
        el = {"id": id, "type": "text", "x": round(x, 1), "y": round(y, 1), "width": round(w, 1),
              "height": round(h, 1), "content": content, "font_family": family, "font_size": round(size),
              "color": color or self.t["fg"], "text_align": align, **kw}
        if bold:
            el["font_weight"] = "bold"
        return self.add(el)

    def button(self, x, y, label, size, align="center", w=None):
        h = size * 2.4
        w = w or max(h * 2.6, self.block(label, size, SANS, True, 1e6).ink_width + 2.2 * size)
        if align == "center":
            x -= w / 2
        g = "cta"
        self.add({"id": "cta_bg", "type": "shape", "x": round(x, 1), "y": round(y, 1), "width": round(w, 1),
                  "height": round(h, 1), "shape_type": "rectangle", "fill_color": self.t["accent"],
                  "corner_radius": round(h / 2, 1), "group_id": g})
        lh = size * 1.2
        self.add({"id": "cta_label", "type": "text", "x": round(x, 1), "y": round(y + (h - lh) / 2, 1),
                  "width": round(w, 1), "height": round(lh, 1), "content": label, "font_family": SANS,
                  "font_size": round(size), "font_weight": "bold", "color": self.t["on_accent"],
                  "text_align": "center", "group_id": g})
        return w, h

    def logo(self, x, y, w, h):
        """The brand mark: the logo artwork on a dark theme; on a light one (where
        the white artwork would vanish) a letter-spaced wordmark, as a text."""
        if self.t is DARK:
            return self.image("logo", x, y, w, h, "solstice_logo_lockup.png")
        size = max(10, round(h * 0.42))
        return self.text("logo", x, y + (h - size * 1.2) / 2, w, "SOLSTICE", size, SERIF, True, self.t["fg"],
                         "left" if x < self.W * 0.3 else "center", letter_spacing=round(size * 0.18, 1))

    def image(self, id, x, y, w, h, asset, **kw):
        return self.add({"id": id, "type": "image", "x": round(x, 1), "y": round(y, 1), "width": round(w, 1),
                         "height": round(h, 1), "asset_id": asset, "fit": "contain", **kw})


def _pick(rng, xs):
    return xs[rng.randrange(len(xs))]


def _copy(rng):
    c = COPY[_pick(rng, sorted(COPY))]
    return {
        "headline": _pick(rng, c["headline"]),
        "sub": _pick(rng, c["sub"]) if rng.random() < 0.75 else None,
        "cta": _pick(rng, c["cta"]) if rng.random() < 0.9 else None,
        "kicker": _pick(rng, c["kicker"]) if rng.random() < 0.35 else None,
        "badge": _pick(rng, c["badge"]) if rng.random() < 0.25 else None,
        "legal": "Terms apply. See website for details." if rng.random() < 0.2 else None,
    }


def _headline_size(W, H, words, rng):
    short = min(W, H)
    base = short * (0.11 if words <= 3 else 0.085 if words <= 6 else 0.06)
    if max(W, H) / short > 3:  # strips: by height
        base = H * (0.34 if words <= 4 else 0.24)
    return base * rng.uniform(0.85, 1.15)


def _headline_kw(b, copy, rng):
    kw = {}
    words = copy["headline"].split()
    if len(words) >= 3 and rng.random() < 0.3:
        start = len(" ".join(words[: len(words) // 2])) + 1
        kw["style_runs"] = [{"start": start, "end": len(copy["headline"]) - 1, "color": b.t["gold"]}]
    return kw


def column(b, copy, x, y, w, align, hs, product=None, gap=None):
    """A column of copy from ``y``; returns the bottom. ``product`` (w, h) goes after the headline block."""
    W, H, rng = b.W, b.H, b.rng
    gap = gap or hs * 0.45
    cx = x + w / 2 if align == "center" else x
    if copy["kicker"]:
        ks = max(10, hs * 0.28)
        el = b.text("kicker", x, y, w, copy["kicker"], ks, SANS, True, b.t["accent"], align,
                    letter_spacing=round(ks * 0.15, 1))
        y += el["height"] + gap * 0.6
    el = b.text("headline", x, y, w, copy["headline"], hs, SERIF, True, None, align, **_headline_kw(b, copy, rng))
    y += el["height"] + gap
    if product:
        pw, ph = product
        px = cx - pw / 2 if align == "center" else x
        b.image("product", px, y, pw, ph, "solstice_bottle.png")
        y += ph + gap
    if copy["sub"]:
        el = b.text("subhead", x, y, w, copy["sub"], max(10, hs * 0.36), SANS, False, b.t["soft"], align)
        y += el["height"] + gap
    if copy["cta"]:
        bs = max(10, hs * 0.3)
        _, bh = b.button(cx, y, copy["cta"], bs, "center" if align == "center" else "left")
        y += bh + gap
    return y - gap


def centred(b, top, bottom, place, first=None):
    """Run ``place`` (which adds elements from ``top`` and returns where it ended)
    and centre what it added in [top, bottom]."""
    k = len(b.els) if first is None else first
    end = place()
    dy = max(0.0, (bottom - top - (end - top)) / 2)
    for e in b.els[k:]:
        e["y"] = round(e["y"] + dy, 1)


def _fits(e) -> bool:
    """A text's widest line fits its box (an unbreakable word would spill past it)."""
    if e["type"] != "text":
        return True
    blk = measure(e["content"], TextStyle.of(e), e["font_size"], e["width"], exact=True)
    return blk.ink_width <= e["width"] + 0.5


def build(rng: random.Random, name: str) -> dict:
    """One design; the type scale steps down until the source itself fits its canvas."""
    state = rng.getstate()
    for scale in (1.0, 0.85, 0.72, 0.6, 0.5, 0.4):
        rng.setstate(state)
        t = _build(rng, name, scale)
        W, H = t["source"]["canvas"]["width"], t["source"]["canvas"]["height"]
        content = [e for e in t["source"]["elements"] if not e["id"].startswith(("decor", "background"))]
        if all(e["y"] + e["height"] <= H and e["x"] + e["width"] <= W + 0.5 and _fits(e) for e in content):
            break
    return t


def _build(rng: random.Random, name: str, scale: float) -> dict:
    fmt = _pick(rng, SOURCES)
    W, H = CATALOGUE[fmt]
    theme = DARK if rng.random() < 0.6 else LIGHT
    b = Builder(W, H, theme, rng)
    if theme["bg"] == "image":
        b.add({"id": "background", "type": "image", "x": 0, "y": 0, "width": W, "height": H,
               "asset_id": "solstice_bg_gradient.png", "fit": "cover"})
    else:
        b.add({"id": "background", "type": "shape", "x": 0, "y": 0, "width": W, "height": H,
               "shape_type": "rectangle", "fill_color": theme["bg"]})
    copy = _copy(rng)
    words = len(copy["headline"].split())
    hs = _headline_size(W, H, words, rng) * scale
    m = min(W, H) * rng.uniform(0.06, 0.1)
    r = W / H
    for i in range(rng.choice([0, 0, 1, 2])):
        s = min(W, H) * rng.uniform(0.25, 0.5)
        cx, cy = rng.choice([(0, 0), (W, 0), (0, H), (W, H)])
        b.add({"id": f"decor{i}", "type": "shape", "x": round(cx - s / 2), "y": round(cy - s / 2), "width": round(s),
               "height": round(s), "shape_type": "ellipse", "fill_color": theme["gold"], "opacity": 0.18})
    logo = rng.random() < 0.7
    if r > 3:  # strip: logo | headline (+sub) | cta
        x = m
        if logo:
            lw = min(H * 0.6 * 3, W * 0.16)
            b.logo(m, (H - lw / 3) / 2, lw, lw / 3)
            x += lw + m
        cta_w = 0
        if copy["cta"]:
            bs = H * 0.17
            while bs > 9 and b.block(copy["cta"], bs, SANS, True, 1e6).ink_width + 2.2 * bs > W * 0.22:
                bs *= 0.92
            cta_w = max(bs * 2.4 * 2.6, b.block(copy["cta"], bs, SANS, True, 1e6).ink_width + 2.2 * bs)
            b.button(W - m - cta_w, (H - bs * 2.4) / 2, copy["cta"], bs, "left", cta_w)
        w = W - x - m - cta_w - m
        sub = copy["sub"] if copy["sub"] and len(copy["sub"]) < 60 else None
        hs = min(hs, H * 0.34)
        while hs > 10 and (b.block(copy["headline"], hs, SERIF, True, w).n_lines > 2
                           or b.text_h(copy["headline"], hs, SERIF, True, w)
                           + (b.text_h(sub, hs * 0.5, SANS, False, w) if sub else 0) > H * 0.84):
            hs *= 0.92
        hh = b.text_h(copy["headline"], hs, SERIF, True, w)
        sh = b.text_h(sub, hs * 0.5, SANS, False, w) if sub else 0
        y = (H - hh - sh) / 2
        b.text("headline", x, y, w, copy["headline"], hs, SERIF, True, None, "left")
        if sub:
            b.text("subhead", x, y + hh, w, sub, hs * 0.5, SANS, False, theme["soft"], "left")
        layout = "row"
    else:
        layout = _pick(rng, ["center", "center", "left", "product_left", "card", "band"] if r >= 1.3
                       else ["center", "center", "left", "card", "band"] if r >= 0.9
                       else ["center", "center", "card", "band", "left"])
        y = m
        if logo:
            lh = min(W, H) * 0.09
            lw = lh * 3
            lx = (W - lw) / 2 if layout in ("center", "card", "band") else m
            b.logo(lx, m, lw, lh)
            y += lh + m * 0.6
        bottom = H - m * (1.8 if copy["legal"] else 1.0)
        if layout == "center":
            ps = min(W, bottom - y) * 0.38 if rng.random() < 0.6 else 0
            w = W - 2 * m * 1.5
            centred(b, y, bottom, lambda: column(b, copy, (W - w) / 2, y, w, "center", hs,
                                                 (ps * 0.75, ps) if ps else None))
        elif layout == "left":
            if r >= 0.9:
                w = W * 0.5 - m
                ps = min(W * 0.42, H - 2 * m)
                b.image("product", W - m - ps * 0.75, (H - ps) / 2, ps * 0.75, ps, "solstice_bottle.png")
                centred(b, y, bottom, lambda: column(b, copy, m, y, w, "left", hs * 0.9))
            else:
                w = W - 2 * m
                ps = min(W * 0.6, H * 0.42)
                b.image("product", W - m - ps * 0.75, bottom - ps, ps * 0.75, ps, "solstice_bottle.png")
                centred(b, y, bottom - ps - m, lambda: column(b, copy, m, y, w, "left", hs * 0.9))
        elif layout == "product_left":
            ps = H - 2 * m
            b.image("product", m, m, ps * 0.75, ps, "solstice_bottle.png")
            x = m * 2 + ps * 0.75
            centred(b, y, bottom, lambda: column(b, copy, x, y, W - x - m, "left", hs * 0.85))
        elif layout == "card":
            cw = W * rng.uniform(0.62, 0.8)
            cx, pad = (W - cw) / 2, cw * 0.08
            card = b.add({"id": "card", "type": "shape", "x": round(cx), "y": round(y), "width": round(cw),
                          "height": 0, "shape_type": "rectangle", "fill_color": theme["panel"],
                          "corner_radius": round(min(W, H) * 0.03)})

            def card_column():
                end = column(b, copy, cx + pad, y + pad, cw - 2 * pad, "center", hs * 0.8)
                card["height"] = round(end - y + pad)
                return end + pad
            centred(b, y, bottom, card_column, first=len(b.els) - 1)
        else:  # band: product above, copy in a band at the bottom
            b.t = DARK
            band = b.add({"id": "band", "type": "shape", "x": 0, "y": 0, "width": W, "height": 0,
                          "shape_type": "rectangle", "fill_color": DARK["panel"]})
            k = len(b.els)
            pad = m * 0.9
            end = column(b, copy, m * 1.5, pad, W - 3 * m, "center", hs * 0.75)
            band_h = max(H * 0.36, end + pad + (H - bottom))
            top = H - band_h
            band["y"], band["height"] = round(top), round(band_h)
            for e in b.els[k:]:
                e["y"] = round(e["y"] + top + (band_h - (H - bottom) - end - pad) / 2, 1)
            ps = min(W * 0.5, top - y - m)
            if ps > H * 0.12:
                b.image("product", (W - ps * 0.75) / 2, y + (top - y - ps) / 2, ps * 0.75, ps, "solstice_bottle.png")
    if copy["badge"] and any(e["id"] == "product" for e in b.els):
        p = next(e for e in b.els if e["id"] == "product")
        s = p["width"] * 0.42
        bx = min(p["x"] + p["width"] - s * 0.6, W - s - 2)
        b.add({"id": "badge_bg", "type": "shape", "x": round(bx), "y": round(p["y"]),
               "width": round(s), "height": round(s), "shape_type": "ellipse", "fill_color": theme["gold"],
               "group_id": "badge"})
        b.add({"id": "badge_label", "type": "text", "x": round(bx),
               "y": round(p["y"] + s * 0.32), "width": round(s), "height": round(s * 0.36),
               "content": copy["badge"], "font_family": SANS, "font_size": round(s * 0.24), "font_weight": "bold",
               "color": "rgba(43,27,46,1)", "text_align": "center", "group_id": "badge"})
    if copy["legal"] and r <= 3:
        ls = max(9, min(W, H) * 0.018)
        b.text("legal", m, H - m * 0.6 - ls * 1.3, W - 2 * m, copy["legal"], ls, SANS, False, b.t["soft"], "center")
    src = {"canvas": {"width": W, "height": H}, "elements": b.els}
    return {"task_id": name, "notes": f"{fmt} {layout} {'dark' if theme is DARK else 'light'}", "source": src,
            "target_canvas": {"width": 1080, "height": 1080}}


def designs(n: int, seed: int):
    rng = random.Random(seed)
    for i in range(n):
        yield build(rng, f"gen{seed:02d}_{i:03d}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in designs(a.n, a.seed):
        (out / f"{t['task_id']}.json").write_text(json.dumps(t, indent=1))
    print(f"wrote {a.n} designs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
