"""Turn a raw element list into layout units with inferred roles.

The input has no semantic labels, so roles come from geometry, typography,
stacking order and grouping (and, as a weak hint only, asset file names):

  background  full-canvas image/shape at the bottom of the stack
  decor       free-floating shapes with no text (translucent circles, blobs)
  headline    the most prominent standalone text
  subhead / body / legal   the other standalone texts, by prominence and position
  cta / badge / ribbon     a filled shape carrying a short label (rotated -> ribbon)
  logo / hero / image / watermark   standalone images
  missing     an image whose asset is not in the brand kit (hidden, never drawn
              as a grey placeholder)

Each unit ("Item") is what the layout engine moves around. Grouped elements and
a shape with a label inside it travel together as one item.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import assets, textfit
from .geom import Box, el_box, el_visual_box, rotated_aabb, union_all
from .textfit import TextStyle

IMPORTANCE = {
    "headline": 100, "cta": 90, "logo": 88, "hero": 80, "subhead": 62, "body": 55,
    "image": 50, "badge": 42, "group": 45, "ribbon": 36, "shape": 30, "legal": 22,
    "watermark": 12, "missing": 0, "void": 0, "panel": 60,
}


@dataclass
class Item:
    key: str
    kind: str  # text | image | group
    role: str
    els: List[dict]
    src_box: Box  # visual footprint in the source (text: tight box, image: ink box)
    rotation: float = 0.0
    importance: float = 50.0
    corner: Optional[str] = None  # tl/tr/bl/br when pinned to a canvas corner
    # text
    style: Optional[TextStyle] = None
    fs0: float = 0.0
    measure_em: float = 0.0  # source wrap width in ems
    src_lines: int = 1
    # image
    ink: tuple = (0.0, 0.0, 1.0, 1.0)
    aspect: float = 1.0  # element box w/h
    # group
    container: Optional[dict] = None  # the shape that frames the label(s)
    labels: List[dict] = field(default_factory=list)
    centered_label: bool = False
    hidden_reason: Optional[str] = None
    core: bool = False  # carries the headline or CTA: never dropped
    lockup: bool = False  # several texts set as one piece of type (a shadow, a price)
    # composite (an image carrying stickers / a row of product shots)
    host: Optional["Item"] = None
    sats: List["Item"] = field(default_factory=list)
    memo: dict = field(default_factory=dict, repr=False, compare=False)  # per-item derived values

    @property
    def is_text(self) -> bool:
        return self.kind == "text"

    @property
    def is_copy(self) -> bool:
        """Copy that flows with the text: a text, or texts locked up as one."""
        return self.kind == "text" or self.lockup

    @property
    def ids(self) -> List[str]:
        return [e["id"] for e in self.els]

    def main_fs(self) -> float:
        if self.kind == "text":
            return self.fs0
        if self.labels:
            return max(float(l.get("font_size", 12)) for l in self.labels)
        return 0.0


@dataclass
class Backdrop:
    """A shape that frames other items (a dark band behind the copy, a card).
    It is not laid out itself; it is redrawn around wherever its members land."""

    el: dict
    members: List[Item]
    edges: tuple  # touched canvas edges in the source: (left, top, right, bottom)
    pad: tuple  # source padding around the members' union: (l, t, r, b)


@dataclass
class Design:
    W: float
    H: float
    elements: List[dict]  # normalized copies, original order
    backgrounds: List[dict]
    decor: List[dict]
    items: List[Item]
    by_id: Dict[str, dict]
    backdrops: List[Backdrop] = field(default_factory=list)
    split: Optional[object] = None  # regions.Split: a photo region beside a flat panel carrying the copy

    def role_items(self, *roles: str) -> List[Item]:
        return [it for it in self.items if it.role in roles]


# ----------------------------------------------------------------------------

def as_number(v, default=0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def normalize(el: dict, idx: int) -> dict:
    e = dict(el)
    for k in ("x", "y", "width", "height"):
        e[k] = as_number(e.get(k), 0.0)
    e["width"] = max(0.0, e["width"])
    e["height"] = max(0.0, e["height"])
    e["rotation"] = as_number(e.get("rotation"), 0.0)
    e["opacity"] = min(1.0, max(0.0, as_number(e.get("opacity"), 1.0)))
    if not isinstance(e.get("z_index"), (int, float)):
        e["z_index"] = idx
    if e.get("type") == "text":
        # measure exactly what finalize() will emit and the renderer will draw
        if e.get("text_align", "left") not in ("left", "center", "right"):
            e["text_align"] = "left"
        fw = e.get("font_weight")
        if fw is not None and not isinstance(fw, str):
            try:
                e["font_weight"] = "bold" if float(fw) >= 600 else "regular"
            except (TypeError, ValueError):
                e["font_weight"] = "regular"
        fst = e.get("font_style")
        if fst is not None and fst not in ("normal", "italic"):
            e["font_style"] = "italic" if str(fst).lower() == "italic" else "normal"
    e["_idx"] = idx
    return e


def _fill_alpha(el: dict) -> float:
    from .color import parse_rgba

    return parse_rgba(el.get("fill_color"))[3]


def text_tight_box(el: dict, style: TextStyle) -> tuple:
    """Where the source text actually sits inside its (often loose) box."""
    b = el_box(el)
    blk = textfit.measure(str(el.get("content", "")), style, el.get("font_size", 16), b.w)
    ink = blk.ink_width
    align = el.get("text_align", "left")
    if align == "center":
        x = b.x + (b.w - ink) / 2.0
    elif align == "right":
        x = b.r - ink
    else:
        x = b.x
    return Box(x, b.y, max(1.0, ink), blk.height), blk


def analyze(source: dict, lockups: bool = True) -> Design:
    W0 = float(source["canvas"]["width"])
    H0 = float(source["canvas"]["height"])
    canvas = Box(0, 0, W0, H0)
    raw = source.get("elements") or []
    els = [normalize(e, i) for i, e in enumerate(raw)]
    by_id = {e["id"]: e for e in els}

    texts = [e for e in els if e.get("type") == "text"]
    zmin_text = min((e["z_index"] for e in texts), default=10 ** 9)

    # --- backgrounds: full-canvas images/shapes below every text ----------------
    backgrounds = []
    for e in sorted(els, key=lambda e: e["z_index"]):
        if e.get("type") not in ("image", "shape") or e.get("group_id"):
            continue
        cov = el_box(e).inter(canvas) / max(1.0, canvas.area)
        if cov >= 0.88 and abs(e["rotation"]) < 1 and e["z_index"] <= zmin_text:
            backgrounds.append(e)
    regions = [] if backgrounds else _background_regions(els, canvas, zmin_text)
    if regions:  # the region under the headline is the background; the others frame their content
        backgrounds.append(regions[0])
    bg_ids = {e["id"] for e in backgrounds}
    region_ids = {e["id"] for e in regions[1:]}

    # --- explicit groups -----------------------------------------------------------
    groups: Dict[str, List[dict]] = {}
    for e in els:
        if e["id"] in bg_ids:
            continue
        g = e.get("group_id")
        if g:
            groups.setdefault(str(g), []).append(e)
    grouped_ids = {e["id"] for members in groups.values() for e in members if len(members) > 1}
    for g in [g for g, m in groups.items() if len(m) < 2]:
        del groups[g]

    # --- implicit containers: a small shape with a short label on it (button, sticker)
    shapes = [e for e in els if e.get("type") == "shape" and e["id"] not in bg_ids and e["id"] not in grouped_ids]
    free_texts = [e for e in texts if e["id"] not in grouped_ids]
    claimed = set()
    for sh in sorted(shapes, key=lambda e: el_box(e).area):
        sb = el_box(sh)
        if sb.area <= 0 or sb.area > 0.12 * canvas.area:
            continue
        if sh["opacity"] * _fill_alpha(sh) < 0.5:  # a faint shape behind text is decoration, not a button
            continue
        inside = []
        for t in free_texts:
            if t["id"] in claimed or t["z_index"] < sh["z_index"]:
                continue
            tight, _ = text_tight_box(t, TextStyle.of(t))
            if tight.area > 0 and tight.inter(sb) >= 0.8 * tight.area:
                inside.append(t)
        if len(inside) == 1 and len(str(inside[0].get("content", ""))) <= 40 and \
                sb.area <= 30 * max(1.0, text_tight_box(inside[0], TextStyle.of(inside[0]))[0].area):
            key = f"__auto_{sh['id']}"
            groups[key] = [sh] + inside
            claimed.update(t["id"] for t in inside)
            grouped_ids.update([sh["id"]] + [t["id"] for t in inside])

    # --- asset availability ---------------------------------------------------------
    images = [e for e in els if e.get("type") == "image"]
    found_any = any(assets.info(e.get("asset_id")).found for e in images)

    items: List[Item] = []
    decor: List[dict] = []
    pending_backdrops: List[dict] = []

    for gkey, members in groups.items():
        items.append(_group_item(gkey, members))

    for e in els:
        if e["id"] in bg_ids or e["id"] in grouped_ids:
            continue
        t = e.get("type")
        if t in ("image", "shape") and _edge_rule(e, canvas):  # a frame line, not a row of the layout
            e["_slab"] = True
            decor.append(e)
            continue
        if t == "text":
            style = TextStyle.of(e)
            tight, blk = text_tight_box(e, style)
            if e["rotation"]:  # where its glyphs are painted
                tight = textfit.drawn_box(e)
            fs0 = float(e.get("font_size", 16) or 16)
            items.append(Item(
                key=e["id"], kind="text", role="text", els=[e], src_box=tight,
                rotation=e["rotation"], style=style, fs0=fs0,
                measure_em=max(1.0, el_box(e).w) / max(1.0, fs0), src_lines=blk.n_lines,
            ))
        elif t == "image":
            b = el_box(e)
            fit = e.get("fit", "contain")
            ink = assets.ink_fraction(e.get("asset_id"), b.w, b.h, fit)
            ink_box = Box(b.x + ink[0] * b.w, b.y + ink[1] * b.h, (ink[2] - ink[0]) * b.w, (ink[3] - ink[1]) * b.h)
            if e["rotation"]:
                ink_box = el_visual_box(e)
                ink = (0.0, 0.0, 1.0, 1.0)
            it = Item(key=e["id"], kind="image", role="image", els=[e], src_box=ink_box,
                      rotation=e["rotation"], ink=ink, aspect=(b.w / b.h) if b.h > 0 else 1.0)
            if found_any and not assets.info(e.get("asset_id")).found and b.area < 0.06 * canvas.area:
                # a small image whose file is not in the brand kit would paint a grey
                # "missing asset" box; a large one is laid out normally in case the
                # scorer's renderer has the file
                it.role = "missing"
                it.hidden_reason = "asset not in brand kit"
            items.append(it)
        elif e["id"] in region_ids:
            pending_backdrops.append(e)
        elif t == "shape":
            if _frames_something(e, els, bg_ids, grouped_ids, canvas):
                pending_backdrops.append(e)
            elif _is_decor(e, canvas):
                decor.append(e)
            else:
                b = el_box(e)
                items.append(Item(key=e["id"], kind="image", role="shape", els=[e],
                                  src_box=el_visual_box(e), rotation=e["rotation"],
                                  aspect=(b.w / b.h) if b.h > 0 else 1.0))
        else:
            decor.append(e)  # unknown type: leave it alone, scaled like decor

    design = Design(W=W0, H=H0, elements=els, backgrounds=backgrounds, decor=decor, items=items, by_id=by_id)
    _assign_roles(design)
    for it in design.items:
        if it.src_box.w < 1 or it.src_box.h < 1 or it.src_box.inter(canvas) <= 0:
            it.role = "void"  # nothing to show (zero-size or entirely off the source canvas)
        it.importance = IMPORTANCE.get(it.role, 40)
    if lockups:
        _make_lockups(design)
    _make_panels(design)
    _make_composites(design)
    _make_backdrops(design, pending_backdrops)
    _assign_corners(design)
    from .regions import find_split
    design.split = find_split(design)
    return design


def _locked(a: Item, b: Item) -> bool:
    """Two texts set as one piece of type. Separate lines of copy keep at least
    a word space apart; these overlap on purpose (a shadow, stacked display
    type) or sit closer than that (a big figure beside its cents, display
    lines stacked tight in one typeface). A subhead set right under its
    headline in another face is copy that follows it, not one piece: it
    stays with it in a column (see compose._room) but wraps on its own."""
    A, B = a.src_box, b.src_box
    if A.inter(B) >= 0.15 * min(A.area, B.area):
        return True
    fs = min(a.fs0, b.fs0)
    gap_x = max(B.x - A.r, A.x - B.r)
    gap_y = max(B.y - A.b, A.y - B.b)
    side_by_side = min(A.b, B.b) - max(A.y, B.y) > 0.3 * min(A.h, B.h)
    stacked = min(A.r, B.r) - max(A.x, B.x) > 0.3 * min(A.w, B.w)
    one_face = a.style is not None and b.style is not None and a.style.family == b.style.family
    return (side_by_side and gap_x < 0.3 * fs) or (stacked and gap_y < 0.1 * fs and one_face)


def _make_lockups(d: Design) -> None:
    """Texts locked together become one unit that scales as designed."""
    texts = [it for it in d.items if it.kind == "text" and it.role != "void" and not it.rotation]
    parent = {id(t): id(t) for t in texts}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            if _locked(a, b):
                parent[find(id(a))] = find(id(b))
    clusters: Dict[int, List[Item]] = {}
    for t in texts:
        clusters.setdefault(find(id(t)), []).append(t)
    for members in clusters.values():
        if len(members) < 2:
            continue
        lead = max(members, key=lambda m: (m.importance, m.fs0))
        els = sorted((e for m in members for e in m.els), key=lambda e: e["z_index"])
        lock = Item(key=f"lockup:{lead.key}", kind="group", role=lead.role, els=els,
                    src_box=union_all(m.src_box for m in members), importance=lead.importance,
                    style=lead.style, fs0=lead.fs0, labels=els, lockup=True,
                    core=any(m.role in ("headline", "cta") for m in members))
        d.items = [it for it in d.items if all(it is not m for m in members)] + [lock]


def _background_regions(els: List[dict], canvas: Box, zmin_text: float) -> List[dict]:
    """Flat panels that together are the background: a canvas split into a
    dark half and a bright half, say. Each spans the canvas one way, sits
    under all copy, and together they cover it. The one under the most
    prominent copy comes first; [] if the canvas is not split this way."""
    def flat(e: dict) -> bool:
        if e.get("type") == "shape":
            return e.get("shape_type", "rectangle") == "rectangle"
        a = assets.info(e.get("asset_id"))
        return a.found and a.opaque and a.busy < 0.25

    out = []
    for e in els:
        if e.get("type") not in ("image", "shape") or e.get("group_id") or abs(e["rotation"]) >= 1 \
                or e["z_index"] > zmin_text or not flat(e):
            continue
        b = el_box(e).inter_box(canvas)
        if b is None or b.area < 0.15 * canvas.area:
            continue
        full_h = b.y <= 0.02 * canvas.h and b.b >= 0.98 * canvas.h
        full_w = b.x <= 0.02 * canvas.w and b.r >= 0.98 * canvas.w
        if full_h or full_w:
            out.append(e)
    if len(out) < 2 or _covered(out, canvas) < 0.9:
        return []
    texts = [e for e in els if e.get("type") == "text" and str(e.get("content", "")).strip()]
    if texts:
        lead = max(texts, key=lambda e: float(e.get("font_size", 0) or 0) * (1.12 if e.get("font_weight") == "bold" else 1))
        lb = el_box(lead)
        out.sort(key=lambda e: -el_box(e).inter(lb))
    return out


def _covered(els: List[dict], canvas: Box, n: int = 24) -> float:
    """Share of the canvas covered by the union of these boxes (on an n x n grid)."""
    boxes = [el_box(e) for e in els]
    hits = 0
    for i in range(n):
        for j in range(n):
            x, y = canvas.x + (i + 0.5) * canvas.w / n, canvas.y + (j + 0.5) * canvas.h / n
            hits += any(b.x <= x <= b.r and b.y <= y <= b.b for b in boxes)
    return hits / (n * n)


def _frames_something(sh: dict, els: List[dict], bg_ids, grouped_ids, canvas: Box) -> bool:
    """A free shape with text on top of it that is not a small button: a panel.
    An ellipse that bleeds off the canvas is a corner ornament whatever sits
    over it (redrawn around that copy it would become a stretched halo)."""
    sb = el_box(sh)
    if sb.area <= 0:
        return False
    if sh.get("shape_type") == "ellipse" and sb.inter(canvas) < 0.95 * sb.area:
        return False
    n = 0
    for e in els:
        if e is sh or e["id"] in bg_ids or e["z_index"] < sh["z_index"]:
            continue
        if e.get("type") == "text":
            tb, _ = text_tight_box(e, TextStyle.of(e))
            if tb.area > 0 and tb.inter(sb) >= 0.8 * tb.area:
                n += 1
    return n >= 1


def _make_backdrops(d: Design, shapes: List[dict]) -> None:
    for sh in shapes:
        sb = el_box(sh)
        members = [it for it in d.items if it.src_box.area > 0 and it.src_box.inter(sb) >= 0.8 * it.src_box.area
                   and min(e["z_index"] for e in it.els) >= sh["z_index"]]
        if not members:
            d.decor.append(sh)
            continue
        u = union_all(m.src_box for m in members)
        edges = (sb.x <= 2, sb.y <= 2, sb.r >= d.W - 2, sb.b >= d.H - 2)
        pad = (max(0.0, u.x - sb.x), max(0.0, u.y - sb.y), max(0.0, sb.r - u.r), max(0.0, sb.b - u.b))
        d.backdrops.append(Backdrop(el=sh, members=members, edges=edges, pad=pad))


def _make_panels(d: Design) -> None:
    """Copy that sits on a coloured block or a secondary image travels with it.

    Real designs put text on panels all the time (a red block behind the
    headline, a pink card with the offer). Moved apart, white copy lands on a
    white canvas and disappears, so the panel, its text and anything small on
    it become one unit that scales together."""
    canvas = Box(0, 0, d.W, d.H)
    texts = [it for it in d.items if it.is_copy]

    def flat(it: Item) -> bool:  # a solid block, not a product shot or photo
        e = it.els[0]
        if e.get("type") == "shape":
            return True
        a = assets.info(e.get("asset_id"))
        return a.found and a.busy < 0.25

    def hollow(it: Item) -> bool:  # an outline or frame
        return not _solid(it)

    for it in list(d.items):  # a frame around (nearly) the whole design is decoration
        if it.kind == "image" and it.role not in ("logo", "missing", "void") and hollow(it) and \
                el_box(it.els[0]).inter(canvas) >= 0.6 * canvas.area:
            d.items.remove(it)
            d.decor.append(it.els[0])

    # flat full-span slabs (a grey half, a coloured column) are background, not art
    slabs = []
    for it in d.items:
        if it.kind != "image" or it.role in ("logo", "missing", "void") or not flat(it):
            continue
        b = el_box(it.els[0])
        span_h = b.y <= 0.03 * d.H and b.b >= 0.97 * d.H
        span_w = b.x <= 0.03 * d.W and b.r >= 0.97 * d.W
        if (span_h or span_w) and b.inter(canvas) >= 0.15 * canvas.area:
            slabs.append(it)
    hosts = [it for it in d.items if it.kind == "image" and it.role not in ("logo", "missing", "void")
             and (el_box(it.els[0]).area <= 0.6 * canvas.area or
                  (flat(it) and el_box(it.els[0]).area <= 0.8 * canvas.area))]
    z = lambda it: min(e["z_index"] for e in it.els)  # noqa: E731

    # 1. each text belongs to the surface right under it (highest host below it)
    by_host: Dict[int, List[Item]] = {}
    for t in texts:
        tb, tz = t.src_box, min(e["z_index"] for e in t.els)
        under = [h for h in hosts if z(h) < tz and el_box(h.els[0]).x <= tb.cx <= el_box(h.els[0]).r and
                 el_box(h.els[0]).y <= tb.cy <= el_box(h.els[0]).b and tb.inter(el_box(h.els[0])) >= 0.5 * tb.area]
        if under:
            h = max(under, key=z)
            by_host.setdefault(id(h), []).append(t)
    used_hosts = [h for h in hosts if id(h) in by_host]
    # 2. surfaces carrying copy that overlap form one unit (a bar on a block, a frame on a panel)
    parent = {id(h): id(h) for h in used_hosts}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(used_hosts):
        for b in used_hosts[i + 1:]:
            ba, bb = el_box(a.els[0]), el_box(b.els[0])
            if ba.inter(bb) >= 0.2 * min(ba.area, bb.area):
                parent[find(id(a))] = find(id(b))
    clusters: Dict[int, List[Item]] = {}
    for h in used_hosts:
        clusters.setdefault(find(id(h)), []).append(h)
    taken = set()
    groups = []
    for hs in clusters.values():
        hs.sort(key=lambda h: -el_box(h.els[0]).area)
        zmin = min(z(h) for h in hs)
        members = list(hs) + [t for h in hs for t in by_host[id(h)]]
        ids = {id(m) for m in members}
        for o in d.items:  # small art lying on the unit (rules, icons, stickers)
            if id(o) in ids or id(o) in taken or o.kind not in ("image", "group") or \
                    o.role in ("logo", "missing", "void", "cta") or z(o) <= zmin:
                continue
            if any(o.src_box.area < 0.4 * el_box(h.els[0]).area and
                   o.src_box.inter(el_box(h.els[0])) >= 0.9 * max(1.0, o.src_box.area) for h in hs):
                members.append(o)
                ids.add(id(o))
        taken.update(ids)
        groups.append(members)
    for sl in slabs:
        if id(sl) not in taken:
            taken.add(id(sl))
            e = sl.els[0]
            e["_slab"] = True
            d.decor.append(e)
    if not taken:
        return
    new_items = [it for it in d.items if id(it) not in taken]
    for members in groups:
        host = members[0]
        best = max(members, key=lambda m: m.importance)
        core = any(m.role in ("headline", "cta") for m in members)
        if not flat(host) and not hollow(host):
            # a photo with copy on it: still the photo (it can be the hero), the copy rides along
            comp = Item(key=f"comp:{host.key}", kind="composite", role=host.role, els=[e for m in members for e in m.els],
                        src_box=union_all(m.src_box for m in members), importance=best.importance + 2,
                        host=host, sats=members[1:], aspect=host.aspect, core=core)
            new_items.append(comp)
            continue
        els = [e for m in members for e in m.els]
        labels = [e for m in members if m.is_copy for e in m.els]
        it = Item(key=f"panel:{host.key}", kind="group", role="panel", els=sorted(els, key=lambda e: e["z_index"]),
                  src_box=union_all(m.src_box for m in members).union(el_box(host.els[0])),
                  importance=best.importance + 2, labels=labels, core=core)
        new_items.append(it)
    d.items = new_items
    if not any(it.role == "hero" for it in d.items):  # the "hero" was really a panel: pick again
        imgs = [it for it in d.items if it.kind == "image" and it.role == "image" and _solid(it)]
        if imgs:
            h = max(imgs, key=lambda it: it.src_box.area)
            if h.src_box.area >= 0.03 * canvas.area:
                h.role, h.importance = "hero", IMPORTANCE["hero"]


def _make_composites(d: Design) -> None:
    """Stickers sitting on a product travel with it; a row of product shots
    is one unit."""
    from .structure import rows_for

    imgs = [it for it in d.items if it.kind == "image" and it.role in ("hero", "image")]
    if not imgs:
        return
    attached: Dict[int, List[Item]] = {}
    used = set()
    for x in d.items:
        if x in imgs or x.kind == "text" or x.role in ("missing", "cta", "logo"):
            continue
        xb = x.src_box
        best = None
        for h in imgs:
            hb = el_box(h.els[0])  # the frame the designer placed the sticker against, margins included
            if xb.inter(hb) >= 0.25 * max(1.0, xb.area) and xb.area <= 0.35 * hb.area and \
                    min(e["z_index"] for e in x.els) > max(e["z_index"] for e in h.els):
                if best is None or hb.area > best.src_box.area:
                    best = h
        if best is not None:
            attached.setdefault(id(best), []).append(x)
            used.add(id(x))
    hero = next((i for i in imgs if i.role == "hero"), None)
    if hero is not None:
        for row in rows_for(imgs):
            if hero in row and len(row) > 1:
                for o in row:
                    if o is hero:
                        continue
                    attached.setdefault(id(hero), []).append(o)
                    used.add(id(o))
                    attached[id(hero)] += attached.pop(id(o), [])  # its stickers come along
    new_items = []
    for it in d.items:
        if id(it) in used:
            continue
        sats = attached.get(id(it))
        if sats:
            comp = Item(key=f"comp:{it.key}", kind="composite", role=it.role, els=it.els + [e for s in sats for e in s.els],
                        src_box=union_all([it.src_box] + [s.src_box for s in sats]), importance=it.importance + 2,
                        host=it, sats=sats, aspect=it.aspect)
            new_items.append(comp)
        else:
            new_items.append(it)
    d.items = new_items


def _edge_rule(e: dict, canvas: Box) -> bool:
    """A hairline running most of the way across or down the canvas (a frame
    line, a side rule): at least 15:1 and spanning 85% of that side."""
    b = el_box(e)
    if b.w <= 0 or b.h <= 0 or abs(e["rotation"]) >= 1:
        return False
    return (b.h >= 15 * b.w and b.h >= 0.85 * canvas.h) or (b.w >= 15 * b.h and b.w >= 0.85 * canvas.w)


def _is_decor(e: dict, canvas: Box) -> bool:
    b = el_box(e)
    vis = el_visual_box(e)
    alpha = e["opacity"] * _fill_alpha(e)
    if alpha < 0.6:
        return True
    if e.get("shape_type") == "ellipse":
        return True
    if vis.inter(canvas) < 0.95 * max(1.0, vis.area):  # bleeds off the canvas
        return True
    if b.area > 0.35 * canvas.area:
        return True
    return False


def _group_item(gkey: str, members: List[dict]) -> Item:
    members = sorted(members, key=lambda e: e["z_index"])
    shapes = [m for m in members if m.get("type") == "shape"]
    labels = [m for m in members if m.get("type") == "text"]
    container = None
    centered = False
    if shapes and labels:
        cand = max(shapes, key=lambda e: el_box(e).area)
        cb = el_box(cand)
        ok = True
        for t in labels:
            tight, _ = text_tight_box(t, TextStyle.of(t))
            if tight.inter(cb) < 0.7 * max(1.0, tight.area):
                ok = False
        if ok:
            container = cand
            if len(labels) == 1:
                t = labels[0]
                tb = el_box(t)
                centered = (abs(tb.cx - cb.cx) <= 0.08 * cb.w + 2 and abs(tb.cy - cb.cy) <= 0.2 * cb.h + 2
                            and t.get("text_align", "left") == "center")
    vis = union_all(el_visual_box(m) for m in members) or Box(0, 0, 1, 1)
    rot = 0.0
    rots = [m["rotation"] for m in members]
    if rots and all(abs(r - rots[0]) < 0.5 for r in rots):
        rot = rots[0]
    return Item(key=f"group:{gkey}", kind="group", role="group", els=members, src_box=vis,
                rotation=rot, container=container, labels=labels, centered_label=centered)


def _prominence(it: Item) -> float:
    fs = it.fs0
    bold = it.style is not None and it.style.weight in ("bold", "semibold", "extrabold", "black")
    return fs * (1.12 if bold else 1.0)


def _assign_roles(d: Design) -> None:
    W0, H0 = d.W, d.H
    canvas_area = W0 * H0

    # ---- groups: buttons, badges, ribbons ------------------------------------
    buttons = []
    for it in d.items:
        if it.kind != "group":
            continue
        short = all(len(str(l.get("content", ""))) <= 40 for l in it.labels)
        art = any(e.get("type") == "image" for e in it.els)  # a card with a picture is content, not a button
        if it.container is not None and it.labels and short and not art:
            if abs(it.rotation) >= 3:
                it.role = "ribbon"
            else:
                buttons.append(it)
        else:
            it.role = "group"
    if buttons:
        bg = {b["id"] for b in d.backgrounds}
        pictures = [e for e in d.elements if e.get("type") == "image" and e["id"] not in bg]

        def sticker(it: Item) -> bool:
            """Round, or stuck on a picture: a badge, which a real button outranks for CTA."""
            c = it.container
            cb = el_box(c)
            on_art = any(el_box(e).inter(cb) >= 0.3 * max(1.0, cb.area) and c["z_index"] > e["z_index"]
                         for e in pictures)
            return c.get("shape_type") == "ellipse" or on_art

        def btn_score(it: Item) -> float:
            b = el_box(it.container)
            lower = b.cy / H0
            return b.area / canvas_area * 10 + it.main_fs() / 100.0 + 0.3 * lower
        ranked = sorted(buttons, key=lambda it: (not sticker(it), btn_score(it)), reverse=True)
        primary = ranked[0]
        pb = primary.src_box
        corner_top = pb.cy < 0.22 * H0 and (pb.cx < 0.3 * W0 or pb.cx > 0.7 * W0)
        if len(ranked) == 1 and corner_top:
            primary.role = "badge"
        else:
            primary.role = "cta"
        for it in ranked[1:]:  # a second button beside the CTA, of its size, is a CTA too ("Learn more")
            b = it.src_box
            beside = min(b.b, pb.b) - max(b.y, pb.y) > 0.5 * min(b.h, pb.h)
            alike = abs(b.h / max(1.0, pb.h) - 1.0) < 0.15 and it.main_fs() == primary.main_fs()
            it.role = "cta" if primary.role == "cta" and beside and alike else "badge"

    # ---- standalone texts ----------------------------------------------------
    texts = [it for it in d.items if it.kind == "text"]
    if texts:
        ranked = sorted(texts, key=lambda it: (-_prominence(it), it.src_box.y))
        head = ranked[0]
        head.role = "headline"
        hp = _prominence(head)
        rest = ranked[1:]
        others = [x for x in d.items if x.role not in ("missing",)]
        for it in rest:
            p = _prominence(it)
            near_bottom = it.src_box.b >= 0.86 * H0 or it.src_box.y >= 0.9 * H0
            lowest = all(it.src_box.y >= o.src_box.cy for o in others if o is not it)
            faint = it.els[0]["opacity"] * _text_alpha(it.els[0]) < 0.75
            # small print by size alone only when very small; a smaller line that sits with the
            # copy (a subhead under its headline) is legal only if it is faint or set at the very bottom
            if p <= 0.22 * hp or (p <= 0.45 * hp and ((near_bottom and lowest) or faint)):
                it.role = "legal"
            else:
                it.role = "body"
        bodies = sorted([it for it in rest if it.role == "body"], key=lambda it: -_prominence(it))
        if bodies:
            bodies[0].role = "subhead"

    # ---- standalone images -------------------------------------------------------
    imgs = [it for it in d.items if it.kind == "image" and it.role == "image"]
    if imgs:
        def logo_score(it: Item) -> float:
            e = it.els[0]
            name = str(e.get("asset_id", "")).lower()
            b = el_box(e)
            a = assets.info(e.get("asset_id"))
            if a.found and a.opaque and a.busy < 0.25 and any(  # a flat block carrying copy is a panel
                    b.x <= t.src_box.cx <= b.r and b.y <= t.src_box.cy <= b.b and
                    min(x["z_index"] for x in t.els) > e["z_index"] for t in texts):
                return 0.0
            sc = 0.0
            if "logo" in name or "wordmark" in name or "lockup" in name:
                sc += 3.0
            if b.area < 0.08 * canvas_area:
                sc += 1.0
            if it.src_box.cy < 0.2 * H0 or it.src_box.cy > 0.85 * H0:
                sc += 1.0
            if it.aspect >= 1.6:
                sc += 0.8
            return sc
        best_logo = max(imgs, key=logo_score)
        if logo_score(best_logo) >= 2.0:
            best_logo.role = "logo"
        rest = [it for it in imgs if it.role == "image" and _solid(it)]
        if rest:
            # the largest solid picture is the hero; so is a design's only picture, however small
            # (it is the product the design features), unless it is a speck
            hero = max(rest, key=lambda it: it.src_box.area)
            if hero.src_box.area >= 0.03 * canvas_area or (len(rest) == 1 and hero.src_box.area >= 0.01 * canvas_area):
                hero.role = "hero"
        for it in imgs:
            if it.role != "image":
                continue
            e = it.els[0]
            small = it.src_box.area < 0.04 * canvas_area
            faint = e["opacity"] < 0.6
            name = str(e.get("asset_id", "")).lower()
            if faint or (small and ("watermark" in name or "badge" in name or "stamp" in name)):
                it.role = "watermark"
            elif small and (it.src_box.b > 0.88 * H0) and e["z_index"] < max(
                    (x["z_index"] for x in d.elements if x.get("type") == "text"), default=0):
                it.role = "watermark"

    # ---- a brand set in type ------------------------------------------------------
    if texts and not any(it.role == "logo" for it in d.items):
        mark = _wordmark(d, texts)
        if mark is not None:
            was, mark.role = mark.role, "logo"
            if was == "subhead":  # the next body copy takes its place
                bodies = sorted([it for it in texts if it.role == "body"], key=lambda it: -_prominence(it))
                if bodies:
                    bodies[0].role = "subhead"


def _wordmark(d: Design, texts: List[Item]) -> Optional[Item]:
    """A design with no logo artwork often sets its brand in type: one or two
    words, styled as a mark (capitals, tracking or bold), leading the reading
    order (topmost; leftmost on a strip) and standing apart from the headline,
    with something between them or a clear gap. A kicker sits right above
    its headline, so it stays copy."""
    head = next((t for t in texts if t.role == "headline"), None)
    if head is None:
        return None
    wide = d.W / max(1.0, d.H) > 2.5
    content = [it for it in d.items if it.role not in ("missing", "void")]
    for t in texts:
        if t.role not in ("body", "subhead") or t.rotation:
            continue
        el = t.els[0]
        words = str(el.get("content", "")).split()
        if not 0 < len(words) <= 2 or len(" ".join(words)) > 20:
            continue
        style = t.style or TextStyle.of(el)
        marked = " ".join(words).isupper() or style.tracking >= 0.05 or style.weight in (
            "bold", "semibold", "extrabold", "black")
        if not marked:
            continue
        b, hb = t.src_box, head.src_box
        others = [o for o in content if o is not t]
        if wide:
            leads = all(b.x <= o.src_box.x + 1.0 for o in others)
            gap = hb.x - b.r
            between = any(b.r <= o.src_box.cx <= hb.x for o in others if o is not head)
        else:
            leads = all(b.y <= o.src_box.y + 1.0 for o in others)
            gap = hb.y - b.b
            between = any(b.b <= o.src_box.cy <= hb.y for o in others if o is not head)
        if leads and (between or gap > 1.5 * b.h):
            return t
    return None


def _solid(it: Item) -> bool:
    """Art with substance (a product, a photo, a cluster of objects), as opposed
    to an outline or frame: low coverage with the ink hugging the bounds."""
    e = it.els[0]
    if e.get("type") != "image":
        return True
    a = assets.info(e.get("asset_id"))
    return not a.found or a.coverage >= 0.25 or (a.ring < 0.6 and a.coverage >= 0.02)


def _text_alpha(el: dict) -> float:
    from .color import parse_rgba

    return parse_rgba(el.get("color"))[3]


def _assign_corners(d: Design) -> None:
    """Pin small items that sit in a canvas corner (badge, ribbon, watermark...).
    Copy turned on its side ("EST. 2026" up an edge) cannot join the reading
    flow at all: it keeps to the corner of the edge it ran along."""
    W0, H0 = d.W, d.H
    for it in d.items:
        b = it.src_box
        if it.kind == "text" and 45 <= abs(it.rotation) % 180 <= 135 and it.role not in ("headline", "cta"):
            it.corner = ("t" if b.cy < H0 / 2.0 else "b") + ("l" if b.cx < W0 / 2.0 else "r")
            continue
        if it.role in ("headline", "hero", "subhead", "body", "cta"):
            continue
        if b.w > 0.5 * W0 or b.h > 0.25 * H0:
            continue
        if abs(b.cx - W0 / 2.0) < 0.12 * W0:
            continue  # horizontally centred: part of the main column
        near_l = b.x < 0.15 * W0
        near_r = W0 - b.r < 0.15 * W0
        near_t = b.y < 0.13 * H0
        near_b = H0 - b.b < 0.13 * H0
        if near_t and near_l:
            it.corner = "tl"
        elif near_t and near_r:
            it.corner = "tr"
        elif near_b and near_l:
            it.corner = "bl"
        elif near_b and near_r:
            it.corner = "br"
