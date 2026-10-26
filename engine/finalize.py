"""Second pass for a candidate: background crop, decoration, floating
elements (ribbons, small images), hiding what didn't make the cut, and
keeping content inside the canvas."""

from __future__ import annotations

import numpy as np

from . import saliency
from .blocks import GroupBlock, ImageBlock, img_vis_frac, rot_bounds

HIDE_ROLES = {"missing"}


def _text_center(P, model):
    tot, sx, sy = 0.0, 0.0, 0.0
    for eid, v in P.vis.items():
        if model["els"].get(eid, {}).get("type") != "text":
            continue
        a = max(1.0, (v[2] - v[0]) * (v[3] - v[1]))
        tot += a
        sx += a * (v[0] + v[2]) / 2
        sy += a * (v[1] + v[3]) / 2
    return (sx / tot, sy / tot) if tot else None


def _thirds_target(text_c, size):
    """Where the image's focal point should land on one axis: on the third
    line opposite the text, or centered when the text is centered."""
    if text_c is None:
        return 0.5
    t = text_c / size
    if t < 0.42:
        return 2 / 3
    if t > 0.58:
        return 1 / 3
    return 0.5


def _src_visible(ctx, el, model, shape):
    """How much each cell of a background picture counted in the source design
    (at the saliency map's resolution): 1 where it was seen, 0 where it was off
    the canvas or under an opaque piece, 0.3 under copy (the designer kept that
    part for words, so it is not the subject)."""
    from .analyze import _fills_box, norm_rot
    h, w = shape
    xs = el["x"] + (np.arange(w) + 0.5) / w * el["width"]
    ys = el["y"] + (np.arange(h) + 0.5) / h * el["height"]
    vis = np.outer((ys >= 0) & (ys <= ctx.Hs), (xs >= 0) & (xs <= ctx.Ws)).astype(np.float32)
    z = el.get("z_index", 0)
    for o in model["els"].values():
        if o is el or o.get("z_index", 0) <= z or abs(norm_rot(o.get("rotation", 0))) > 1:
            continue
        if o["type"] == "text":
            k = 0.3
        elif (o.get("opacity", 1) or 1) >= 0.9 and _fills_box(o) and (
                o["type"] != "image" or (saliency.analyze(o.get("asset_id") or "") or {}).get("alpha_mean", 1) >= 0.9):
            k = 0.0               # a tint over the photo still shows it
        else:
            continue
        cx = (xs >= o["x"]) & (xs <= o["x"] + o["width"])
        cy = (ys >= o["y"]) & (ys <= o["y"] + o["height"])
        vis[np.ix_(cy, cx)] *= k
    return vis


def _best_window(ctx, P, el, model, info, pick, wx, wy, bw, bh):
    """The crop of a background picture, weighing the copy against the picture.
    Each window is charged for texture and subject under the copy beyond what
    the designer accepted there (the score's own busy-background terms), and
    credited for the share of the subject it keeps, counting only what the
    source design showed. A face is never cut. The rule-of-thirds pick stands
    unless another window is clearly better.

    Only when the designer set the copy beside the subject: copy set over it
    (a quote across a landscape, a title on the heart) is the design, and the
    subject keeps its place in the frame. And the copy stays on the side of the
    subject the designer put it: under the trees stays under the trees."""
    from .score import _plates, _src_busy
    sb = _eye_band(info) or info["sal_box"]
    ew, eh = max(1e-6, el["width"]), max(1e-6, el["height"])
    on, tot = 0.0, 0.0
    src = None
    for t in model["els"].values():
        if t["type"] != "text" or t.get("z_index", 0) <= el.get("z_index", 0):
            continue
        r = ((t["x"] - el["x"]) / ew, (t["y"] - el["y"]) / eh,
             (t["x"] + t["width"] - el["x"]) / ew, (t["y"] + t["height"] - el["y"]) / eh)
        tot += (r[2] - r[0]) * (r[3] - r[1])
        on += _box_overlap(r, sb)
        src = r if src is None else (min(src[0], r[0]), min(src[1], r[1]), max(src[2], r[2]), max(src[3], r[3]))
    if tot <= 0 or on > 0.3 * tot:
        return pick
    side, depth = _side(src, sb)
    if side is None:
        return pick
    # the copy may reach up to half way into the subject's box (a loose box:
    # its edge is mostly ground), never across it
    reach = (sb[3] - sb[1]) if side in ("above", "below") else (sb[2] - sb[0])
    sal, busy = info["sal"], info["busy"]
    vis = _src_visible(ctx, el, model, sal.shape)
    salv = sal * vis
    total = float(salv.sum()) or 1.0
    seen = [k for k, f in enumerate(info["faces"]) if saliency.region_mean(vis, *f) >= 0.5]
    faces = [info["faces"][k] for k in seen]
    eyes = [info["eyes"][k] for k in seen] if info.get("eyes") else []
    plates = _plates(P, model)
    texts, cover = [], []           # copy on the photo; everything copy-like on it (incl. its cards)
    for eid, v in P.vis.items():
        t = model["els"].get(eid)
        if t is None or t["type"] != "text":
            continue
        cover.append(v)
        z = t.get("z_index", 0)
        on_plate = [pb for pb, pz in plates
                    if pz < z and _box_overlap(v, pb) >= 0.9 * max(1.0, (v[2] - v[0]) * (v[3] - v[1]))]
        if on_plate:
            cover.extend(on_plate)
            continue
        texts.append((v, max(0.06, _src_busy(model, eid))))
    if not cover:
        return pick
    out = (min(v[0] for v in cover) / bw, min(v[1] for v in cover) / bh,
           max(v[2] for v in cover) / bw, max(v[3] for v in cover) / bh)
    hh, ww = sal.shape

    def value(x0, y0):
        c = 0.0
        for v, ok in texts:
            r = ((v[0] / bw) + x0, (v[1] / bh) + y0, (v[2] / bw) + x0, (v[3] / bh) + y0)
            c -= 10 * max(0.0, saliency.region_mean(busy, *r) - ok)
            c -= 8 * max(0.0, saliency.region_mean(sal, *r) - 0.3)
        c0, c1 = int(x0 * ww), int(np.ceil((x0 + wx) * ww))
        r0, r1 = int(y0 * hh), int(np.ceil((y0 + wy) * hh))
        c += 3 * float(salv[r0:r1, c0:c1].sum()) / total
        for f in faces:
            fa = max(1e-6, (f[2] - f[0]) * (f[3] - f[1]))
            inside = max(0.0, min(f[2], x0 + wx) - max(f[0], x0)) * max(0.0, min(f[3], y0 + wy) - max(f[1], y0))
            c -= 20 * (1 - inside / fa)
        for e in eyes:            # the eyes above all: never cropped, never under copy or a card
            if not (x0 + 0.02 <= e[0] <= x0 + wx - 0.02 and y0 + 0.02 <= e[1] <= y0 + wy - 0.02):
                c -= 40
            ex, ey = (e[0] - x0) * bw, (e[1] - y0) * bh
            if any(v[0] <= ex <= v[2] and v[1] <= ey <= v[3] for v in cover):
                c -= 40
        return c

    # the subject may be pushed partly out of frame for the copy, never more
    # than half of what the plain pick shows of it
    sba = max(1e-6, (sb[2] - sb[0]) * (sb[3] - sb[1]))

    def shown(x0, y0):
        return (max(0.0, min(sb[2], x0 + wx) - max(sb[0], x0))
                * max(0.0, min(sb[3], y0 + wy) - max(sb[1], y0))) / sba
    floor = 0.5 * shown(*pick)

    gx = np.linspace(0, 1 - wx, 9) if wx < 0.98 else [pick[0]]
    gy = np.linspace(0, 1 - wy, 9) if wy < 0.98 else [pick[1]]
    base = value(*pick)
    best, best_v = pick, base + 0.15
    for x0 in gx:
        for y0 in gy:
            if shown(float(x0), float(y0)) < floor:
                continue
            o = (out[0] + x0, out[1] + y0, out[2] + x0, out[3] + y0)
            if _overlap_on(o, sb, side) > max(0.0, depth) + 0.5 * reach:
                continue
            v = value(float(x0), float(y0))
            if v > best_v:
                best, best_v = (float(x0), float(y0)), v
    return best


def _overlap_on(copy, subject, side):
    """How far the copy reaches into the subject from the given side (image
    fractions; negative is a gap)."""
    return {"below": subject[3] - copy[1], "above": copy[3] - subject[1],
            "right": subject[2] - copy[0], "left": copy[2] - subject[0]}[side]


def _side(copy, subject):
    """Which side of the subject the copy sits on (image fractions) and how far
    it reaches into it; (None, 0) when it is not clearly on any side."""
    side = min(("below", "above", "right", "left"), key=lambda k: _overlap_on(copy, subject, k))
    d = _overlap_on(copy, subject, side)
    return (side, d) if d <= 0.15 else (None, 0.0)


def _eye_band(info):
    """The band across the eyes of every face found (image fractions), or None."""
    if not info.get("eyes"):
        return None
    bands = [(f[0], e[1] - 0.15 * (f[3] - f[1]), f[2], e[1] + 0.15 * (f[3] - f[1]))
             for f, e in zip(info["faces"], info["eyes"])]
    return (min(b[0] for b in bands), min(b[1] for b in bands), max(b[2] for b in bands), max(b[3] for b in bands))


def _box_overlap(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def place_bg(ctx, P, unit, model=None):
    el = unit.els[0]
    W, H = ctx.W, ctx.H
    info = saliency.analyze(el.get("asset_id") or "") if el["type"] == "image" else None
    if info is None:
        P.set(el["id"], x=0, y=0, width=W, height=H)
        return
    iw, ih = info["size"]
    s = max(W / iw, H / ih)
    bw, bh = iw * s, ih * s
    wx, wy = W / bw, H / bh  # visible window, as fraction of image
    cx0, cy0 = (1 - wx) / 2, (1 - wy) / 2
    sb = info["sal_box"]
    concentrated = (sb[2] - sb[0]) * (sb[3] - sb[1]) < 0.45 or bool(info.get("eyes"))
    x0, y0 = cx0, cy0
    if concentrated and not (wx > 0.98 and wy > 0.98):
        # rule of thirds: slide the crop so the subject sits on the third line
        # opposite the text (only along the axis that has room to slide)
        fx, fy = info["centroid"]
        tc = _text_center(P, model) if model else None
        if wx < 0.98:
            tx = _thirds_target(tc[0] if tc else None, W)
            x0 = min(max(fx - tx * wx, 0.0), 1 - wx)
        if wy < 0.98:
            ty = _thirds_target(tc[1] if tc else None, H)
            y0 = min(max(fy - ty * wy, 0.0), 1 - wy)
    if model is not None and (wx < 0.98 or wy < 0.98):
        x0, y0 = _best_window(ctx, P, el, model, info, (x0, y0), wx, wy, bw, bh)
    if abs(x0 - cx0) < 1e-3 and abs(y0 - cy0) < 1e-3 and el.get("fit", "contain") == "cover":
        P.set(el["id"], x=0, y=0, width=W, height=H)
    else:
        P.set(el["id"], x=-x0 * bw, y=-y0 * bh, width=bw, height=bh)
    P.bg_map = (info["busy"], -x0 * bw, -y0 * bh, bw, bh)
    P.bg_lum = (info["lum"], -x0 * bw, -y0 * bh, bw, bh)
    P.bg_sal = info["sal"] if concentrated else None
    fx_, fy_ = info["centroid"]
    P.bg_info = {
        "id": el["id"], "asset": el.get("asset_id"), "box": (-x0 * bw, -y0 * bh, bw, bh),
        "concentrated": concentrated,
        "focal": (-x0 * bw + fx_ * bw, -y0 * bh + fy_ * bh),
        "focal_centered": (-cx0 * bw + fx_ * bw, -cy0 * bh + fy_ * bh),
        "sal_box": (-x0 * bw + sb[0] * bw, -y0 * bh + sb[1] * bh, -x0 * bw + sb[2] * bw, -y0 * bh + sb[3] * bh),
    }
    eb = _eye_band(info)
    if eb:
        # with eyes found, what cards must leave visible is the band across the
        # eyes; a mask, a chin or shoulders may take a card
        P.bg_info["sal_box"] = (-x0 * bw + eb[0] * bw, -y0 * bh + eb[1] * bh, -x0 * bw + eb[2] * bw, -y0 * bh + eb[3] * bh)
    if concentrated:
        fx, fy = info["centroid"]
        fcx, fcy = -x0 * bw + fx * bw, -y0 * bh + fy * bh
        # visual weight of the subject: its on-canvas size x how strongly it stands out
        sa = ((sb[2] - sb[0]) * bw * (sb[3] - sb[1]) * bh) / (W * H)
        strength = float(info["sal"].max() - info["sal"].mean())
        P.bg_focal = (fcx, fcy, min(1.0, sa) * (0.5 + strength))


def place_border(ctx, P, unit):
    """Borders keep hugging the canvas edge at the same (scaled) inset."""
    el = unit.els[0]
    k = ctx.border_scale
    l, t, r, b = (max(0.0, v) * k for v in getattr(unit, "insets", (0, 0, 0, 0)))
    P.set(el["id"], x=l, y=t, width=max(1.0, ctx.W - l - r), height=max(1.0, ctx.H - t - b))


def place_rule(ctx, P, unit):
    """A structural line keeps its proportional place on the page: its length
    follows the canvas along its axis, its thickness scales like a border."""
    el = unit.els[0]
    sx, sy = ctx.W / ctx.Ws, ctx.H / ctx.Hs
    k = max(1e-6, ctx.border_scale)
    if el["width"] >= el["height"]:
        P.set(el["id"], x=el["x"] * sx, y=el["y"] * sy, width=el["width"] * sx, height=max(1.0, el["height"] * k))
    else:
        P.set(el["id"], x=el["x"] * sx, y=el["y"] * sy, width=max(1.0, el["width"] * k), height=el["height"] * sy)


def place_deco(ctx, P, unit):
    el = unit.els[0]
    sx, sy = ctx.W / ctx.Ws, ctx.H / ctx.Hs
    s = min(ctx.t_star, 0.9 * min(ctx.W, ctx.H) / max(1, max(el["width"], el["height"])))
    cx = (el["x"] + el["width"] / 2) * sx
    cy = (el["y"] + el["height"] / 2) * sy
    w, h = el["width"] * s, el["height"] * s
    P.set(el["id"], x=cx - w / 2, y=cy - h / 2, width=w, height=h)


def _overlap(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def place_float(ctx, P, unit, s0):
    """Corner-anchored placement for ribbons / small images. Returns False
    if it can't be placed without colliding with content."""
    W, H = ctx.W, ctx.H
    occupied = [v for k, v in P.vis.items()]
    x0, y0, x1, y1 = unit.bbox
    if unit.kind == "image":
        blk = ImageBlock(unit)
        el = unit.els[0]
        sb = rot_bounds(el["x"], el["y"], el["width"], el["height"], el.get("rotation", 0))
    else:
        blk = GroupBlock(unit)
        sb = blk.bounds_at(x0, y0, 1.0)
    left = sb[0] < ctx.Ws - sb[2]
    top = sb[1] < ctx.Hs - sb[3]
    pad_x = (sb[0] if left else ctx.Ws - sb[2])
    pad_y = (sb[1] if top else ctx.Hs - sb[3])
    s = s0
    # its own corner first, then the mirrored ones, before shrinking
    corners = [(left, top), (not left, top), (left, not top), (not left, not top)]
    for attempt in range(4):
        last = attempt == 3
        bw, bh = (sb[2] - sb[0]) * s, (sb[3] - sb[1]) * s
        px = max(4, pad_x * s)
        py = max(4, pad_y * s)
        for ci, (cl, ct) in enumerate(corners):
            tx = px if cl else W - px - bw
            ty = py if ct else H - py - bh
            tx = min(max(tx, 0), max(0, W - bw))
            ty = min(max(ty, 0), max(0, H - bh))
            box = (tx, ty, tx + bw, ty + bh)
            free = all(_overlap(box, o) <= 0.03 * _area(box) for o in occupied) and bw <= W and bh <= H
            if free or (last and ci == len(corners) - 1):
                if not free:
                    tx, ty = (px if left else W - px - bw), (py if top else H - py - bh)
                    tx = min(max(tx, 0), max(0, W - bw))
                    ty = min(max(ty, 0), max(0, H - bh))
                break
        else:
            s *= 0.75
            continue
        if unit.kind == "image":
            el = unit.els[0]
            ew, eh = el["width"] * s, el["height"] * s
            P.set(el["id"], x=tx, y=ty, width=ew, height=eh)
            v = blk.v
            P.vis[el["id"]] = (tx + v[0] * ew, ty + v[1] * eh, tx + v[2] * ew, ty + v[3] * eh)
        else:
            b = blk.bounds_at(0, 0, s)
            blk.place_at(P, tx - b[0], ty - b[1], s)
        return True
    return False


def place_scatter(ctx, P, unit, s0, content):
    """Decoration keeps its proportional place on the canvas, scaled with the
    design. If it lands on content it slides toward its nearest edge, then
    shrinks. It may bleed past an edge only as far as it did in the source."""
    W, H = ctx.W, ctx.H
    if unit.kind == "image":
        blk = ImageBlock(unit)
        el = unit.els[0]
        sb = rot_bounds(el["x"], el["y"], el["width"], el["height"], el.get("rotation", 0))
    else:
        blk = GroupBlock(unit)
        sb = blk.bounds_at(unit.bbox[0], unit.bbox[1], 1.0)
    sw, sh = max(1e-6, sb[2] - sb[0]), max(1e-6, sb[3] - sb[1])
    bleed = (max(0.0, -sb[0]) / sw, max(0.0, -sb[1]) / sh,
             max(0.0, sb[2] - ctx.Ws) / sw, max(0.0, sb[3] - ctx.Hs) / sh)
    fx, fy = (sb[0] + sb[2]) / 2 / ctx.Ws, (sb[1] + sb[3]) / 2 / ctx.Hs
    step = 0.04 * min(W, H)
    # edges it touched in the source: it stays pinned to them (bleed included)
    tol = 0.01 * min(ctx.Ws, ctx.Hs)
    pin = (sb[0] <= tol, sb[1] <= tol, sb[2] >= ctx.Ws - tol, sb[3] >= ctx.Hs - tol)
    # it lay under content in the source (lower in the stack, overlapping it):
    # it may lie under content again, so it is not pushed out of the way
    model = getattr(ctx, "model", None)
    under = False
    if model is not None:
        from .analyze import _inter, _area as _earea
        mine = set(unit.ids)
        above = [e for u in model["units"] if u.role not in ("bg", "deco", "border", "rule", "minor", "missing")
                 and not (set(u.ids) & mine) for e in u.els]
        for d in unit.els:          # a piece substantially covered by higher content
            cov = sum(_inter(d, e) for e in above if e.get("z_index", 0) > d.get("z_index", 0))
            if cov >= 0.3 * max(1.0, _earea(d)):
                under = True
                break

    def clamp(x, y, bw, bh):
        x = min(max(x, -bleed[0] * bw), W - bw + bleed[2] * bw)
        y = min(max(y, -bleed[1] * bh), H - bh + bleed[3] * bh)
        if pin[0]:
            x = -bleed[0] * bw
        elif pin[2]:
            x = W - bw + bleed[2] * bw
        if pin[1]:
            y = -bleed[1] * bh
        elif pin[3]:
            y = H - bh + bleed[3] * bh
        return x, y

    def free(x, y, bw, bh):
        if under:
            return True
        box = (x, y, x + bw, y + bh)
        return all(_overlap(box, o) <= 0.005 * min(_area(box), _area(o)) for o in content)

    s = s0
    found = None
    for attempt in range(4):
        bw, bh = sw * s, sh * s
        x, y = clamp(fx * W - bw / 2, fy * H - bh / 2, bw, bh)
        last = (x, y, s)
        if free(x, y, bw, bh):
            found = (x, y, s)
            break
        # slide toward the edges, nearest first
        cx, cy = x + bw / 2, y + bh / 2
        dirs = sorted([(cx, -1, 0), (W - cx, 1, 0), (cy, 0, -1), (H - cy, 0, 1)])
        for _, dx, dy in dirs:
            nx, ny = x, y
            for _ in range(int(max(W, H) / step) + 1):
                px, py = nx, ny
                nx, ny = clamp(nx + dx * step, ny + dy * step, bw, bh)
                if (nx, ny) == (px, py):
                    break
                if free(nx, ny, bw, bh):
                    found = (nx, ny, s)
                    break
            if found:
                break
        if found:
            break
        s *= 0.75
    x, y, s = found or last      # nowhere free: the smallest try, in its own place
    if unit.kind == "image":
        el = unit.els[0]
        ew, eh = el["width"] * s, el["height"] * s
        P.set(el["id"], x=x, y=y, width=ew, height=eh)
        v = blk.v
        P.vis[el["id"]] = (x + v[0] * ew, y + v[1] * eh, x + v[2] * ew, y + v[3] * eh)
    else:
        b = blk.bounds_at(0, 0, s)
        blk.place_at(P, x - b[0], y - b[1], s)


def hide(P, el, W, H):
    # Kept in the JSON (ids must survive) but invisible and out of the way.
    w, h = el["width"], el["height"]
    k = 1.0 / max(1.0, w, h)
    P.set(el["id"], x=0, y=0, width=w * k, height=h * k, opacity=0)
    if el["type"] == "text":
        # The official renderer ignores `opacity` on text, so also tuck it
        # under the background (lowest z) at 1px on a single line.
        P.set(el["id"], font_size=1, width=min(W, 400), height=2, z_index=P.min_z - 1)
    P.vis.pop(el["id"], None)
    P.hidden.add(el["id"])


def clamp_inside(ctx, P, model):
    W, H = ctx.W, ctx.H
    skip = {e["id"] for u in model["units"] if u.role in ("bg", "deco", "border", "anchored", "rule", "minor") for e in u.els}
    for pn in getattr(P, "panels", None) or ([P.panel] if getattr(P, "panel", None) else []):
        skip.add(pn["id"])        # a full-bleed panel runs off the canvas edges it touches
    if getattr(P, "backdrop", None) is not None:
        skip.update(P.backdrop.ids)
    for eid, p in P.props.items():
        if eid in skip or eid in P.hidden:
            continue
        el = model["els"][eid]
        w, h = p.get("width", el["width"]), p.get("height", el["height"])
        rot = p.get("rotation", el.get("rotation", 0))      # as placed
        b = rot_bounds(p["x"], p["y"], w, h, rot)
        bw, bh = b[2] - b[0], b[3] - b[1]                  # its footprint on the canvas
        if bw > W or bh > H:
            k = min(W / bw, H / bh)
            cx, cy = p["x"] + w / 2, p["y"] + h / 2
            p["width"], p["height"] = w * k, h * k
            p["x"], p["y"] = cx - w * k / 2, cy - h * k / 2
            if "font_size" in p:
                p["font_size"] = max(1, int(p["font_size"] * k))
            w, h = p["width"], p["height"]
            b = rot_bounds(p["x"], p["y"], w, h, rot)
        dx = -b[0] if b[0] < 0 else (W - b[2] if b[2] > W else 0)
        dy = -b[1] if b[1] < 0 else (H - b[3] if b[3] > H else 0)
        if dx or dy:
            p["x"] += dx
            p["y"] += dy
            if eid in P.vis:
                v = P.vis[eid]
                P.vis[eid] = (v[0] + dx, v[1] + dy, v[2] + dx, v[3] + dy)


def place_frame_bg(ctx, P, unit):
    """A border/frame layer over the whole canvas: stretch it to the new canvas."""
    el = unit.els[0]
    sx, sy = ctx.W / ctx.Ws, ctx.H / ctx.Hs
    P.set(el["id"], x=el["x"] * sx, y=el["y"] * sy, width=el["width"] * sx, height=el["height"] * sy)


def _vis_box(P, ids):
    bs = [P.vis[i] for i in ids if i in P.vis]
    if not bs:
        return None
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def place_aligned(ctx, P, unit, model):
    """Keep a small piece on its anchor's line (centre, left or right), on the
    same side of it and at the same gap, scaled with the anchor. Returns
    False when that spot is off the canvas or taken."""
    from .analyze import _unit_ink
    anchor = next((u for u in model["units"] if u.uid == unit.anchor_uid), None)
    if anchor is None:
        return False
    nb = _vis_box(P, anchor.ids)
    if nb is None:
        return False
    ob, pb = _unit_ink(anchor), _unit_ink(unit)
    s = (nb[3] - nb[1]) / max(1e-6, ob[3] - ob[1])
    pw, ph = (pb[2] - pb[0]) * s, (pb[3] - pb[1]) * s
    line = unit.align_line
    # the anchor's own alignment may have changed (a left-aligned row column)
    al = {model["els"][i].get("text_align") for i in anchor.ids if model["els"][i]["type"] == "text"}
    new_al = {P.props.get(i, {}).get("text_align") for i in anchor.ids if model["els"][i]["type"] == "text"} - {None}
    if line == "center" and new_al and new_al != {"center"} and al != new_al:
        line = "left" if "left" in new_al else "right"
    x = {"center": (nb[0] + nb[2]) / 2 - pw / 2, "left": nb[0], "right": nb[2] - pw}[line]
    above = pb[3] <= ob[1]
    gap = (ob[1] - pb[3] if above else pb[1] - ob[3]) * s
    y = nb[1] - gap - ph if above else nb[3] + gap
    box = (x, y, x + pw, y + ph)
    edge = 0.01 * min(ctx.W, ctx.H)
    if box[0] < edge or box[1] < edge or box[2] > ctx.W - edge or box[3] > ctx.H - edge:
        return False
    if any(_overlap(box, v) > 0.03 * _area(box) for v in P.vis.values()):
        return False
    if unit.kind == "image":
        el = unit.els[0]
        P.set(el["id"], x=x, y=y, width=pw, height=ph)
        v = img_vis_frac(el)
        P.vis[el["id"]] = (x + v[0] * pw, y + v[1] * ph, x + v[2] * pw, y + v[3] * ph)
    else:
        blk = GroupBlock(unit)
        b = blk.bounds_at(0, 0, s)
        blk.place_at(P, x - b[0], y - b[1], s)
    return True


def pin_edges(ctx, P, model):
    """Content that touched or bled off a canvas edge in the source goes back
    to that edge (same proportional bleed), if it can without landing on other
    content (photo panels included). Left/right and top/bottom independently;
    never both sides."""
    Ws, Hs = ctx.Ws, ctx.Hs
    W, H = ctx.W, ctx.H
    tol = 0.01 * min(Ws, Hs)
    panel_ids = {pn["id"] for pn in (getattr(P, "panels", None) or ([P.panel] if getattr(P, "panel", None) else []))}
    for u in model["units"]:
        if u.role not in ("headline", "subhead", "body", "group", "cta", "logo", "legal", "badges") or u.kind == "badges":
            continue
        ids = [i for i in u.ids if i in P.vis and i in P.props and i not in P.hidden]
        if not ids or set(ids) & panel_ids:
            continue
        b = u.bbox
        L, T, R_, B = b[0] <= tol, b[1] <= tol, b[2] >= Ws - tol, b[3] >= Hs - tol
        vb = [P.vis[i] for i in ids]
        cur = (min(v[0] for v in vb), min(v[1] for v in vb), max(v[2] for v in vb), max(v[3] for v in vb))
        sw, sh = max(1e-6, b[2] - b[0]), max(1e-6, b[3] - b[1])
        dx = dy = 0.0
        if L != R_:
            want = -max(0.0, -b[0]) / sw * (cur[2] - cur[0]) if L else W - (cur[2] - cur[0]) + max(0.0, b[2] - Ws) / sw * (cur[2] - cur[0])
            dx = want - cur[0]
        if T != B:
            want = -max(0.0, -b[1]) / sh * (cur[3] - cur[1]) if T else H - (cur[3] - cur[1]) + max(0.0, b[3] - Hs) / sh * (cur[3] - cur[1])
            dy = want - cur[1]
        if not dx and not dy:
            continue
        moved = (cur[0] + dx, cur[1] + dy, cur[2] + dx, cur[3] + dy)
        others = [v for i, v in P.vis.items() if i not in u.ids and i not in P.hidden
                  and model["els"].get(i, {}).get("type") is not None
                  and not any(i in w.ids for w in model["units"] if w.role in ("bg", "deco", "border", "rule", "minor"))]
        if any(_overlap(moved, o) > 0.02 * _area(o) for o in others):
            continue
        for i in u.ids:
            if i in P.props:
                P.props[i]["x"] = P.props[i].get("x", model["els"][i]["x"]) + dx
                P.props[i]["y"] = P.props[i].get("y", model["els"][i]["y"]) + dy
            if i in P.vis:
                v = P.vis[i]
                P.vis[i] = (v[0] + dx, v[1] + dy, v[2] + dx, v[3] + dy)


def finalize(ctx, P, model, float_threshold=0.0):
    """Returns P, or None if the template failed to place some unit
    (nothing may be dropped)."""
    P.hidden = set()
    P.min_z = min(e.get("z_index", 0) for e in model["els"].values())
    P.bg_map = None
    P.bg_lum = None
    P.bg_sal = None
    P.bg_focal = None
    P.bg_info = None
    t = P.meta.get("t", ctx.t_star)
    def _detail(u):
        info = saliency.analyze(u.els[0].get("asset_id") or "") if u.els[0]["type"] == "image" else None
        return info["busyness"] if info else 0.0
    # place background layers least-detailed first, so the photo (most detail)
    # defines the crop, the subject and the busy map used for scoring
    for u in sorted((u for u in model["units"] if u.role == "bg"), key=_detail):
        place_bg(ctx, P, u, model)
    if getattr(P, "backdrop", None) is not None:
        place_bg(ctx, P, P.backdrop, model)   # over the plain backgrounds; defines the busy map
    # the same-design candidate already placed every unit by the source's own
    # geometry; only the background and the canvas-edge borders follow the canvas
    same = getattr(P, "same", False)
    for u in model["units"]:
        if u.role == "bg":
            continue
        elif u.role == "border":
            place_border(ctx, P, u)
        elif same:
            continue
        elif u.role == "rule":
            place_rule(ctx, P, u)
        elif u.role == "deco":
            place_deco(ctx, P, u)
    for u in model["units"]:
        if u.role == "anchored" and not same:
            place_anchored(ctx, P, u)
    for u in model["units"]:
        if u.role == "aligned" and u.els[0]["id"] not in P.props:
            if not place_aligned(ctx, P, u, model):
                place_float(ctx, P, u, min(t, 1.5))
    content = list(P.vis.values())
    for u in sorted(model["units"], key=lambda u: -u.prio):
        if u.role == "ribbon" and u.els[0]["id"] not in P.props:   # not already in the row
            place_float(ctx, P, u, min(t, 1.5))
        elif u.role == "minor" and not same:
            # decoration scales with the canvas's shape-preserving scale (not the text's)
            place_scatter(ctx, P, u, min(ctx.W / ctx.Ws, ctx.H / ctx.Hs), content)
    for u in model["units"]:
        if u.role in HIDE_ROLES:
            for eid in u.ids:
                hide(P, model["els"][eid], ctx.W, ctx.H)  # missing asset: skip gracefully (FAQ)
    clamp_inside(ctx, P, model)
    for u in model["units"]:
        if u.role == "separator":
            place_satellite(ctx, P, u, model)
    for u in model["units"]:
        for eid in u.ids:
            if eid not in P.props:
                return None
    if not same:
        pin_edges(ctx, P, model)      # the same design already keeps its edges
    clamp_inside(ctx, P, model)
    return P


def place_anchored(ctx, P, unit):
    """Corner pieces stay in their corner, scaled with the canvas."""
    el = unit.els[0]
    k = min(ctx.W / ctx.Ws, ctx.H / ctx.Hs) ** 0.5 * ctx.t_star ** 0.5
    k = min(k, 0.6 * ctx.W / el["width"], 0.6 * ctx.H / el["height"])
    w, h = el["width"] * k, el["height"] * k
    edges = getattr(unit, "edges", "lt")
    # keep any bleed past the edge proportional (a piece pushed 5px off stays a bit off)
    ox = el["x"] * k if "l" in edges else (ctx.Ws - el["x"] - el["width"]) * k
    oy = el["y"] * k if "t" in edges else (ctx.Hs - el["y"] - el["height"]) * k
    x = ox if "l" in edges else ctx.W - w - ox
    y = oy if "t" in edges else ctx.H - h - oy
    if len(unit.els) > 1:
        # a corner card with its text: the whole piece scales and pins together
        blk = GroupBlock(unit)
        blk.place_at(P, x - (el["x"] - blk.x0) * k, y - (el["y"] - blk.y0) * k, k)
        return
    P.set(el["id"], x=x, y=y, width=w, height=h)
    v = img_vis_frac(el)
    P.vis[el["id"]] = (x + v[0] * w, y + v[1] * h, x + v[2] * w, y + v[3] * h)


def place_satellite(ctx, P, unit, model):
    """Separators (dots, dashes, rules) keep their position relative to the
    element they belong to, scaled with it."""
    el = unit.els[0]
    a = model["els"].get(unit.anchor)
    pa = P.props.get(unit.anchor)
    if a is None or pa is None:
        sx, sy = ctx.W / ctx.Ws, ctx.H / ctx.Hs
        P.set(el["id"], x=el["x"] * sx, y=el["y"] * sy, width=el["width"] * min(sx, sy), height=el["height"] * min(sx, sy))
        return
    def bounds(e, props=None):
        x = (props or e).get("x", e["x"])
        y = (props or e).get("y", e["y"])
        w = (props or e).get("width", e["width"])
        h = (props or e).get("height", e["height"])
        if e["type"] == "text":
            from . import textfit as tf
            fs = int((props or e).get("font_size", e["font_size"]))
            h = min(h, tf.measure(e, max(1, fs), w)["height"])
        return rot_bounds(x, y, w, h, (props or e).get("rotation", e.get("rotation", 0)))
    ob = bounds(a)
    nb = bounds(a, pa)
    sx = (nb[2] - nb[0]) / max(1e-6, ob[2] - ob[0])
    sy = (nb[3] - nb[1]) / max(1e-6, ob[3] - ob[1])
    s = min(sx, sy) if min(sx, sy) > 0 else 1.0
    rx, ry, rw, rh = unit.rel
    w, h = el["width"] * s, el["height"] * s
    x = nb[0] + rx * (nb[2] - nb[0])
    y = nb[1] + ry * (nb[3] - nb[1])
    P.set(el["id"], x=x, y=y, width=w, height=h)
