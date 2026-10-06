"""The source's latent design system, recovered and rebuilt at the target size.

A designer who re-makes a template at another size keeps its *system*, not its
coordinates: the margin unit, the spacing rhythm (tight pairs stay tight), the
alignment axis the copy hangs on, the type scale, the button proportions and
which surface carries which copy. Elements move; those proportions survive,
which is what makes the result read as "the same design".

The input has no labels, so the system is inferred from geometry with
clustering and tolerances, and every part has a default for messy designs:

    infer(d)                  -> DesignSystem  (source px, ems, fractions of S)
    instantiate(sys, d, ctx)  -> TargetSystem  (target px, per-text size factors)
    measure(...)              -> how far any layout (ours, a designer's) is from it
    consistency(plan, ...)    -> a scorer term: deviation from the target system

S is the short side of a canvas, "em" the font size of the reference text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .geom import Box, el_box, union_all

# Measured median max:min font spread per aspect class (research 05 §B2/§C5).
SPREAD_TARGET = {
    "strip": 2.5, "skyscraper": 2.5, "banner": 2.8, "near-square": 2.8,
    "landscape": 3.3, "square": 3.3, "portrait": 4.5, "tall": 4.5,
}
# Module switches (CHAMELEON_DS="type=floor,gaps=1,margins=1,score=1" overrides them, for experiments).
CONFIG = {"enabled": 1, "type": "floor", "gaps": 1, "margins": 1, "score": 1.0}
for _kv in __import__("os").environ.get("CHAMELEON_DS", "").split(","):
    if "=" in _kv:
        _k, _v = _kv.split("=", 1)
        CONFIG[_k.strip()] = _v.strip() if _k.strip() == "type" else float(_v)

MIN_STEP = 1.25  # headline : second level never closer than the smallest modular ratio
TIGHT_EM = 0.9  # text-to-text gaps up to this many ems of the upper text form one group
TIGHT_MIN_EM = 0.1  # ... and never touching at the target (drawn boxes already carry the leading)
RHYTHM_TOL = 1.25  # gaps within 25% of each other are the same spacing step


def aspect_class(W: float, H: float) -> str:
    r = W / max(1e-6, H)
    if r >= 5.0:
        return "strip"
    if r >= 2.4:
        return "banner"
    if r >= 1.4:
        return "landscape"
    if r >= 1.1:
        return "near-square"
    if r >= 0.9:
        return "square"
    if r >= 0.62:
        return "portrait"
    if r >= 0.4:
        return "tall"
    return "skyscraper"


def _cluster(values: Sequence[float], ratio: float) -> List[List[int]]:
    """Single-linkage clusters of positive values: neighbours (sorted) within
    ``ratio`` of each other join. Returns index lists, smallest values first."""
    idx = sorted(range(len(values)), key=lambda i: values[i])
    out: List[List[int]] = []
    for i in idx:
        if out and values[i] <= values[out[-1][-1]] * ratio + 1e-9:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ----------------------------------------------------------------------------
# the inferred system


@dataclass
class Level:
    fs: float  # representative size, source px
    ratio: float  # fs / headline fs
    ids: List[str]
    leading: float  # line_height (multiple of the font size)


@dataclass
class Gap:
    upper: Tuple[str, ...]  # element ids of the row above
    lower: Tuple[str, ...]
    px: float
    ref_fs: float  # font of the text the gap hangs from (0: no text on either side)
    tight: bool  # part of one text group (headline -> subhead), not a step between groups
    step: float  # the rhythm step (px) this gap was snapped to

    @property
    def em(self) -> float:
        return self.px / self.ref_fs if self.ref_fs > 0 else float("nan")


@dataclass
class Axis:
    kind: str  # left | center | right
    x: float  # fraction of W
    ids: List[str]
    weight: float


@dataclass
class Chip:
    shape: str
    label: str
    pad_x_em: float  # horizontal padding around the label, in label ems
    pad_y_em: float
    radius_ratio: float  # corner radius / height (0.5 = pill)
    height_em: float


@dataclass
class DesignSystem:
    W: float
    H: float
    margins: Tuple[float, float, float, float]  # l, t, r, b as fractions of S (nan: nothing near)
    margin_unit: float  # fraction of S: the second-tightest side
    levels: List[Level]
    head_id: Optional[str]
    head_fs: float
    spread: float  # max / min visible font size
    ratio: Dict[str, float]  # text id -> fs / headline fs
    gaps: List[Gap]
    steps: List[float]  # rhythm steps, source px, smallest first
    axes: List[Axis]
    dominant_axis: str
    columns: int
    chips: List[Chip]
    surfaces: List[Tuple[str, str]]  # (surface id, text id): copy designed on that surface
    content_ids: List[str]  # what margins are measured on (no backgrounds, decor, slabs)
    leading: Dict[str, float] = field(default_factory=dict)
    role: Dict[str, str] = field(default_factory=dict)  # text id -> role of the unit it is in
    min_fs: float = 0.0  # smallest visible font size
    min_id: Optional[str] = None

    @property
    def S(self) -> float:
        return min(self.W, self.H)

    def describe(self) -> str:
        lv = " : ".join(f"{l.ratio:.2f}" for l in self.levels[::-1])
        m = ", ".join("-" if math.isnan(v) else f"{v:.3f}" for v in self.margins)
        tight = [g for g in self.gaps if g.tight]
        loose = [g for g in self.gaps if not g.tight]
        chips = "; ".join(f"pad {c.pad_x_em:.2f}x{c.pad_y_em:.2f}em r/h {c.radius_ratio:.2f}" for c in self.chips[:3])
        return (f"margin unit {self.margin_unit:.3f}S (l,t,r,b {m}) | type {len(self.levels)} levels "
                f"{lv} spread {self.spread:.2f} head {self.head_fs:.0f}px | gaps tight "
                f"{[round(g.em, 2) for g in tight]}em loose {[round(g.px) for g in loose]}px steps "
                f"{[round(s) for s in self.steps]} | axis {self.dominant_axis} ({len(self.axes)} axes, "
                f"{self.columns} col) | chips {chips or '-'} | surfaces {len(self.surfaces)}")


def _row_fs(row, edge: str) -> float:
    """Font size of the copy at the bottom (or top) edge of a row of items; 0 if none."""
    from .textfit import drawn_box

    best, fs = None, 0.0
    for it in row:
        if it.kind != "text":  # a button or a photo caption is its own component, not running copy
            continue
        for e in it.els:
            if not _visible_text(e):
                continue
            tb = drawn_box(e)
            key = tb.b if edge == "bottom" else -tb.y
            if best is None or key > best:
                best, fs = key, float(e["font_size"])
    return fs


def _visible_text(e: dict) -> bool:
    return e.get("type") == "text" and str(e.get("content", "")).strip() != "" and \
        float(e.get("opacity", 1) or 0) > 0 and float(e.get("font_size", 0) or 0) > 0


def infer(d) -> DesignSystem:
    """Recover the design system of an analysed source design (analyze.Design)."""
    from .structure import rows_for
    from .textfit import TextStyle, drawn_box, measure

    W, H = d.W, d.H
    S = min(W, H)
    live = [it for it in d.items if it.role not in ("void", "missing")]

    # --- type scale -------------------------------------------------------------
    texts = [e for it in live for e in it.els if _visible_text(e)]
    role = {e["id"]: it.role for it in live for e in it.els if _visible_text(e)}
    head = next((it for it in live if it.role == "headline"), None)
    sizes = [float(e["font_size"]) for e in texts]
    head_fs = head.fs0 if head is not None else (max(sizes) if sizes else 0.0)
    head_id = head.els[0]["id"] if head is not None else None
    levels: List[Level] = []
    for grp in _cluster(sizes, 1.1):
        fs = _median([sizes[i] for i in grp])
        ids = [texts[i]["id"] for i in grp]
        lead = _median([TextStyle.of(texts[i]).line_height for i in grp])
        levels.append(Level(fs=fs, ratio=fs / head_fs if head_fs else 1.0, ids=ids, leading=lead))
    ratio = {e["id"]: float(e["font_size"]) / head_fs for e in texts} if head_fs else {}
    spread = max(sizes) / max(1e-6, min(sizes)) if sizes else 1.0

    # --- margins: the tightest deliberate distance to each edge -------------------
    content = [it for it in live]
    eps = 0.012 * S  # closer than this to an edge is bleed, not a margin
    sides: List[float] = []
    for k in range(4):
        ds = []
        for it in content:
            b = it.src_box
            dist = (b.x, b.y, W - b.r, H - b.b)[k]
            if dist > eps:
                ds.append(dist)
        sides.append(min(ds) / S if ds else float("nan"))
    # the unit is what at least two sides agree on: one stray item near an edge
    # does not set it (on the Crello tuning half this tracks the designers'
    # own target margins better than the single tightest side)
    known = sorted(v for v in sides if not math.isnan(v))
    unit = known[1] if len(known) > 1 else (known[0] if known else 0.05)

    # --- spacing rhythm, read down the stack rows -----------------------------------
    body = [it for it in live if it.corner is None]
    rows = rows_for(body)
    gaps: List[Gap] = []

    for a, b in zip(rows, rows[1:]):
        ua, ub = union_all(i.src_box for i in a), union_all(i.src_box for i in b)
        g = ub.y - ua.b
        if g < -0.3 * min(ua.h, ub.h):  # side by side, not stacked
            continue
        fa, fb = _row_fs(a, "bottom"), _row_fs(b, "top")
        ref = fa or fb
        g = max(0.0, g)
        tight = bool(fa and fb and g <= TIGHT_EM * fa)
        gaps.append(Gap(upper=tuple(e["id"] for i in a for e in i.els), lower=tuple(e["id"] for i in b for e in i.els),
                        px=g, ref_fs=ref, tight=tight, step=g))
    loose = [g for g in gaps if not g.tight and g.px > 0]
    steps: List[float] = []
    for grp in _cluster([g.px for g in loose], RHYTHM_TOL):
        rep = _median([loose[i].px for i in grp])
        steps.append(rep)
        for i in grp:
            loose[i].step = rep

    # --- alignment axes the copy hangs on -------------------------------------------
    decl = []
    for it in live:
        for e in it.els:
            if not _visible_text(e):
                continue
            tb = drawn_box(e)
            a = e.get("text_align", "left")
            x = tb.x if a == "left" else (tb.r if a == "right" else tb.cx)
            decl.append((a if a in ("left", "center", "right") else "left", x / W, e["id"], it.importance))
    axes: List[Axis] = []
    for kind in ("left", "center", "right"):
        mine = [t for t in decl if t[0] == kind]
        xs = [t[1] + 1.0 for t in mine]  # shift so the ratio test works near 0
        for grp in _cluster(xs, 1.012):
            axes.append(Axis(kind=kind, x=_median([mine[i][1] for i in grp]), ids=[mine[i][2] for i in grp],
                             weight=sum(mine[i][3] for i in grp)))
    axes.sort(key=lambda a: -a.weight)
    dominant = axes[0].kind if axes else "center"
    lefts = [it.src_box.x / W + 1.0 for it in body if it.src_box.w < 0.8 * W]
    columns = sum(1 for g in _cluster(lefts, 1.015) if len(g) >= 2) or 1

    # --- chips: buttons and badges keep their padding and corner ---------------------
    chips: List[Chip] = []
    for it in live:
        if it.kind != "group" or it.container is None or len(it.labels) != 1:
            continue
        sh, lab = it.container, it.labels[0]
        fs = float(lab.get("font_size", 0) or 0)
        cb = el_box(sh)
        if fs <= 0 or cb.h <= 0:
            continue
        blk = measure(str(lab.get("content", "")), TextStyle.of(lab), fs, el_box(lab).w, exact=True)
        chips.append(Chip(shape=sh["id"], label=lab["id"], pad_x_em=max(0.0, cb.w - blk.ink_width) / 2.0 / fs,
                          pad_y_em=max(0.0, cb.h - blk.height) / 2.0 / fs,
                          radius_ratio=float(sh.get("corner_radius", 0) or 0) / cb.h, height_em=cb.h / fs))

    # --- colour-block structure: which surface carries which copy -------------------
    surfaces: List[Tuple[str, str]] = []
    for bd in d.backdrops:
        for m in bd.members:
            surfaces += [(bd.el["id"], e["id"]) for e in m.els if _visible_text(e)]
    for it in live:
        if it.role == "panel" and it.els:
            host = max((e for e in it.els if e.get("type") != "text"), key=lambda e: el_box(e).area, default=None)
            if host is not None:
                surfaces += [(host["id"], e["id"]) for e in it.labels if _visible_text(e)]
        elif it.kind == "composite" and it.host is not None:
            hid = it.host.els[0]["id"]
            surfaces += [(hid, e["id"]) for s in it.sats for e in s.els if _visible_text(e)]

    content_ids = [e["id"] for it in live for e in it.els]
    return DesignSystem(W=W, H=H, margins=tuple(sides), margin_unit=unit, levels=levels, head_id=head_id,
                        head_fs=head_fs, spread=spread, ratio=ratio, gaps=gaps, steps=steps, axes=axes,
                        dominant_axis=dominant, columns=columns, chips=chips, surfaces=surfaces,
                        content_ids=content_ids,
                        leading={e["id"]: TextStyle.of(e).line_height for e in texts}, role=role,
                        min_fs=min(sizes) if sizes else 0.0,
                        min_id=texts[sizes.index(min(sizes))]["id"] if sizes else None)


# ----------------------------------------------------------------------------
# the same system at the target size


@dataclass
class TargetSystem:
    src: DesignSystem
    W: float
    H: float
    cls: str
    alpha: float  # class compression exponent of the type scale (1 = the source's own ratios)
    ratio: Dict[str, float]  # text id -> target fs / headline fs (class compression only)
    margin: float  # px
    type_mode: str = "floor"
    _tight: Dict[tuple, bool] = field(default_factory=dict)

    # --- type scale ------------------------------------------------------------

    def exponent(self, s: Optional[float] = None, ctx=None) -> float:
        """Compression exponent at global scale ``s``. 'class' squeezes the
        spread to the aspect class's typical one (research 05 §C5). 'floor'
        squeezes it only as far as legibility needs: when the smallest level
        would fall under its floor, every level moves up along the same
        log-scale instead of the small ones piling up at the floor size."""
        a = self.alpha if self.type_mode in ("class", "both") else 1.0
        src = self.src
        h = src.head_fs
        if self.type_mode in ("floor", "both") and s is not None and ctx is not None and 0 < src.min_fs < h:
            floor = ctx.floor(src.role.get(src.min_id, "body"))
            rmin = src.min_fs / h
            if h * s * rmin < floor:
                a = min(a, math.log(floor / (h * s)) / math.log(rmin) if h * s > floor else 0.0)
        return max(0.5, min(1.0, a))

    def size_mult(self, fs0: float, s: Optional[float] = None, ctx=None) -> float:
        """Size factor for a text of source size ``fs0`` on top of the uniform
        scale: r_t / r_s on the re-instantiated type scale."""
        h = self.src.head_fs
        if self.type_mode == "off" or h <= 0 or fs0 <= 0 or fs0 >= h:
            return 1.0
        a = self.exponent(s, ctx)
        if a >= 1.0:
            return 1.0
        r = fs0 / h
        rt = min(r ** a, 1.0 / MIN_STEP if r <= 1.0 / MIN_STEP else r)  # keep a visible step under the headline
        return max(1.0, rt / r)

    def item_mult(self, it, s: float, ctx) -> float:
        if it.kind == "text":
            return self.size_mult(it.fs0, s, ctx)
        if it.container is not None and it.centered_label:  # a chip: its label sets its size
            return self.size_mult(it.main_fs(), s, ctx)
        return 1.0  # a panel is mostly art; its copy keeps its own floor

    def want_ratio(self, s: Optional[float] = None, ctx=None) -> Dict[str, float]:
        h = self.src.head_fs
        return {i: r * self.size_mult(r * h, s, ctx) for i, r in self.src.ratio.items()}

    # --- spacing ---------------------------------------------------------------

    def tight(self, upper, lower, gap: float) -> bool:
        """Two stacked rows (lists of items) that read as one text group: both
        carry copy and the gap is at most TIGHT_EM ems of the upper copy."""
        key = (tuple(i.key for i in upper), tuple(i.key for i in lower), round(gap, 1))
        hit = self._tight.get(key)
        if hit is None:
            fa, fb = _row_fs(upper, "bottom"), _row_fs(lower, "top")
            hit = bool(fa and fb and gap <= TIGHT_EM * fa)
            self._tight[key] = hit
        return hit

    def snap(self, gap: float) -> float:
        """A loose gap snapped onto the source's spacing rhythm."""
        best = None
        for st in self.src.steps:
            if st > 0 and max(gap, st) <= RHYTHM_TOL * min(gap, st) + 1e-9:
                if best is None or abs(math.log(gap / st)) < abs(math.log(gap / best)):
                    best = st
        return best if best is not None else gap


def instantiate(sys: DesignSystem, ctx, type_mode: Optional[str] = None) -> TargetSystem:
    cls = aspect_class(ctx.W, ctx.H)
    target = SPREAD_TARGET[cls]
    alpha = 1.0
    if sys.spread > target and sys.spread > 1.0:
        alpha = min(1.0, max(0.5, math.log(target) / math.log(sys.spread)))
    S = min(ctx.W, ctx.H)
    m = min(max(sys.margin_unit * S, max(4.0, 0.04 * S)), 0.08 * S)
    t = TargetSystem(src=sys, W=ctx.W, H=ctx.H, cls=cls, alpha=alpha, ratio={}, margin=m,
                     type_mode=type_mode or CONFIG["type"])
    for tid, r in sys.ratio.items():
        rt = min(r ** alpha, 1.0 / MIN_STEP if r <= 1.0 / MIN_STEP else r) if r < 1.0 else r
        t.ratio[tid] = max(r, rt)
    return t


# ----------------------------------------------------------------------------
# measuring any layout against the system


def _boxes(elements: Iterable[dict]) -> Dict[str, Box]:
    from .textfit import drawn_box

    out = {}
    for e in elements:
        if float(e.get("opacity", 1) or 0) <= 0:
            continue
        out[e["id"]] = drawn_box(e) if e.get("type") == "text" else el_box(e)
    return out


def measure(sys: DesignSystem, elements: List[dict], W: float, H: float,
            want_ratio: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """How well a layout keeps the source system. ``want_ratio`` is the
    re-instantiated type scale (default: the source's own ratios).

    type_dev     mean |log(r_out / r_want)| over visible texts (0 = scale kept)
    type_inv     pairs whose size order flipped (beyond 1 px)
    margin       smallest content-to-edge distance / S
    margin_dev   |log(margin / source margin unit)|
    tight_dev    mean |log(em_out / em_src)| over tight (in-group) gaps still stacked
    split_groups tight pairs that ended further apart than a step between groups
    rhythm_cv    spread of gaps that were one rhythm step in the source
    axes         distinct alignment axes the copy uses (source: sys.axes)
    chip_dev     mean |log| change of chip padding (x, y) and radius ratio
    on_surface   share of designed text-on-surface pairs that still hold
    """
    from .textfit import TextStyle, measure as tmeasure

    by_id = {e["id"]: e for e in elements}
    boxes = _boxes(elements)
    S = min(W, H)
    want = want_ratio or sys.ratio
    out: Dict[str, float] = {}

    # type scale
    fs = {i: float(by_id[i].get("font_size", 0) or 0) for i in sys.ratio if i in by_id and i in boxes}
    fs = {i: f for i, f in fs.items() if f > 0}
    hid = sys.head_id if sys.head_id in fs else (max(fs, key=fs.get) if fs else None)
    devs = []
    if hid and want.get(hid, 0) > 0:
        for i, f in fs.items():
            if i == hid or want.get(i, 0) <= 0:
                continue
            devs.append(abs(math.log((f / fs[hid]) / (want[i] / want[hid]))))
    out["type_dev"] = sum(devs) / len(devs) if devs else 0.0
    inv = 0
    ids = list(fs)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            si, sj = sys.ratio[i], sys.ratio[j]
            if abs(si - sj) * sys.head_fs <= 1.0:
                continue
            if (si - sj) * (fs[i] - fs[j]) < 0 and abs(fs[i] - fs[j]) > 1.0:
                inv += 1
    out["type_inv"] = float(inv)

    # margins
    cb = [boxes[i] for i in sys.content_ids if i in boxes and boxes[i].w > 0 and boxes[i].h > 0]
    eps = 0.012 * S
    ds = [v for b in cb for v in (b.x, b.y, W - b.r, H - b.b) if v > eps]
    out["margin"] = min(ds) / S if ds else float("nan")
    out["margin_dev"] = abs(math.log(out["margin"] / max(1e-6, sys.margin_unit))) if ds else 0.0

    # spacing: tight groups stay tight, rhythm steps stay equal
    def ubox(ids_):
        bs = [boxes[i] for i in ids_ if i in boxes]
        return union_all(bs)

    tdev, split, n_tight = [], 0, 0
    loose_now = []
    stepgap: Dict[float, List[float]] = {}
    for g in sys.gaps:
        ua, ub = ubox(g.upper), ubox(g.lower)
        if ua is None or ub is None:
            continue
        now = ub.y - ua.b
        if now < -0.3 * min(ua.h, ub.h) or min(ub.r, ua.r) - max(ub.x, ua.x) < -0.05 * W:
            continue  # re-flowed side by side: not a vertical gap any more
        now = max(0.0, now)
        if g.tight:
            n_tight += 1
            ref = [float(by_id[i].get("font_size", 0) or 0) for i in g.upper if i in fs]
            f = max(ref) if ref else 0.0
            if f > 0 and g.px > 0:
                tdev.append(abs(math.log(max(0.05, now / f) / max(0.05, g.em))))
        else:
            loose_now.append(now)
            stepgap.setdefault(g.step, []).append(now)
    for g in sys.gaps:  # a pair that belonged together is now further apart than groups are
        if not g.tight:
            continue
        ua, ub = ubox(g.upper), ubox(g.lower)
        if ua is None or ub is None or not loose_now:
            continue
        now = ub.y - ua.b
        if now > 0 and now > 0.95 * min(loose_now):
            split += 1
    out["tight_dev"] = sum(tdev) / len(tdev) if tdev else 0.0
    out["split_groups"] = float(split)
    cvs = []
    for vals in stepgap.values():
        if len(vals) >= 2 and sum(vals) > 0:
            mu = sum(vals) / len(vals)
            cvs.append(math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) / max(1e-6, mu))
    out["rhythm_cv"] = sum(cvs) / len(cvs) if cvs else 0.0

    # alignment axes used by the copy
    decl = []
    for i in fs:
        e, tb = by_id[i], boxes[i]
        a = e.get("text_align", "left")
        x = tb.x if a == "left" else (tb.r if a == "right" else tb.cx)
        decl.append((a, x / W + 1.0))
    n_axes = 0
    for kind in ("left", "center", "right"):
        xs = [x for k, x in decl if k == kind]
        n_axes += len(_cluster(xs, 1.012)) if xs else 0
    out["axes"] = float(n_axes)
    out["axes_src"] = float(len(sys.axes))

    # chips
    cd = []
    for c in sys.chips:
        sh, lab = by_id.get(c.shape), by_id.get(c.label)
        if sh is None or lab is None or c.shape not in boxes or c.label not in boxes:
            continue
        f = float(lab.get("font_size", 0) or 0)
        b = el_box(sh)
        if f <= 0 or b.h <= 0:
            continue
        blk = tmeasure(str(lab.get("content", "")), TextStyle.of(lab), f, el_box(lab).w, exact=True)
        px_ = max(0.0, b.w - blk.ink_width) / 2.0 / f
        py_ = max(0.0, b.h - blk.height) / 2.0 / f
        rr = float(sh.get("corner_radius", 0) or 0) / b.h
        for a_, b_ in ((px_, c.pad_x_em), (py_, c.pad_y_em)):
            cd.append(abs(math.log((a_ + 0.1) / (b_ + 0.1))))
        cd.append(abs(rr - c.radius_ratio) * 2.0)
    out["chip_dev"] = sum(cd) / len(cd) if cd else 0.0

    # copy still on the surface it was designed on
    held = tot = 0
    for s_id, t_id in sys.surfaces:
        if s_id not in boxes or t_id not in boxes:
            continue
        tot += 1
        tb = boxes[t_id]
        if tb.area > 0 and tb.inter(boxes[s_id]) >= 0.9 * tb.area:
            held += 1
    out["on_surface"] = held / tot if tot else 1.0
    return out


# ----------------------------------------------------------------------------
# placing a column with its text groups kept rigid


def chain_blocks(targets: List[float], weights: List[float], sizes: List[float], gaps: List[float],
                 tight: Sequence[Optional[float]], lo: float, hi: float) -> List[float]:
    """layout.chain_positions, with rows joined by a tight (in-group) gap moved
    as one block: the gaps between groups stretch, the ones inside a group do not."""
    from .layout import chain_positions

    n = len(sizes)
    blocks: List[List[int]] = [[0]]
    for i in range(1, n):
        if i - 1 < len(tight) and tight[i - 1] is not None:
            blocks[-1].append(i)
        else:
            blocks.append([i])
    if len(blocks) == n:
        return chain_positions(targets, weights, sizes, gaps, lo, hi)
    b_sizes, b_targets, b_weights, b_gaps, offs = [], [], [], [], {}
    for bi, blk in enumerate(blocks):
        off = 0.0
        for j, i in enumerate(blk):
            offs[i] = off
            off += sizes[i] + (gaps[i] if j < len(blk) - 1 else 0.0)
        b_sizes.append(off)
        wsum = sum(weights[i] for i in blk)
        b_weights.append(wsum)
        b_targets.append(sum(weights[i] * (targets[i] - offs[i]) for i in blk) / max(1e-9, wsum))
        if bi:
            b_gaps.append(gaps[blk[0] - 1])
    ys = chain_positions(b_targets, b_weights, b_sizes, b_gaps, lo, hi)
    out = [0.0] * n
    for y, blk in zip(ys, blocks):
        for i in blk:
            out[i] = y + offs[i]
    return out


# ----------------------------------------------------------------------------
# scorer term


def consistency(plan, d, ctx) -> float:
    """How far a candidate drifts from the re-instantiated system (0 = none).

    type   importance-weighted |log| drift of each text's size ratio to the
           headline from the target type scale; a text held up by its
           legibility floor is not blamed for being relatively too big
    groups |log| drift of the gaps inside text groups from their ems
    """
    tsys = ctx.system
    src = tsys.src
    fonts: Dict[str, float] = {}
    imp: Dict[str, float] = {}
    foot: Dict[str, Box] = {}
    for r, x, y in plan.placed + plan.floaters:
        for i, f in r.fonts.items():
            fonts[i] = float(f)
            imp[i] = r.item.importance
        for e in r.item.els:
            foot[e["id"]] = Box(x, y, r.w, r.h)
    pen = 0.0
    hid = src.head_id
    if hid in fonts and src.head_fs > 0:
        want = tsys.want_ratio(plan.s, ctx)
        num = den = 0.0
        for i, f in fonts.items():
            if i == hid or i not in want or want.get(hid, 0) <= 0:
                continue
            dev = math.log((f / fonts[hid]) / (want[i] / want[hid]))
            if dev > 0 and f <= ctx.floor(src.role.get(i, "body")) + 0.5:
                dev = 0.0  # held up by the floor
            w = imp.get(i, 40.0)
            num += w * abs(dev)
            den += w
        if den:
            pen += 30.0 * num / den
    devs = []
    for g in src.gaps:
        if not g.tight:
            continue
        ua = union_all(foot[i] for i in g.upper if i in foot)
        ub = union_all(foot[i] for i in g.lower if i in foot)
        f = max((fonts[i] for i in g.upper if i in fonts), default=0.0)
        if ua is None or ub is None or f <= 0 or ub.y < ua.b - 0.3 * min(ua.h, ub.h):
            continue
        want_gap = max(g.em if g.ref_fs > 0 else 0.0, TIGHT_MIN_EM) * f
        devs.append(abs(math.log((max(0.0, ub.y - ua.b) + 0.1 * f) / (want_gap + 0.1 * f))))
    if devs:
        pen += 10.0 * sum(devs) / len(devs)
    return pen
