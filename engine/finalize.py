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
    concentrated = (sb[2] - sb[0]) * (sb[3] - sb[1]) < 0.45
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

    def clamp(x, y, bw, bh):
        x = min(max(x, -bleed[0] * bw), W - bw + bleed[2] * bw)
        y = min(max(y, -bleed[1] * bh), H - bh + bleed[3] * bh)
        return x, y

    def free(x, y, bw, bh):
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
    if getattr(P, "panel", None):
        skip.add(P.panel["id"])   # a photo panel bleeds off the canvas edges it touches
    if getattr(P, "backdrop", None) is not None:
        skip.update(P.backdrop.ids)
    for eid, p in P.props.items():
        if eid in skip or eid in P.hidden:
            continue
        el = model["els"][eid]
        w, h = p.get("width", el["width"]), p.get("height", el["height"])
        if w > W or h > H:
            k = min(W / w, H / h)
            p["width"], p["height"] = w * k, h * k
            if "font_size" in p:
                p["font_size"] = max(1, int(p["font_size"] * k))
            w, h = p["width"], p["height"]
        b = rot_bounds(p["x"], p["y"], w, h, el.get("rotation", 0))
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
    for u in model["units"]:
        if u.role == "bg":
            continue
        elif u.role == "border":
            place_border(ctx, P, u)
        elif u.role == "rule":
            place_rule(ctx, P, u)
        elif u.role == "deco":
            place_deco(ctx, P, u)
    for u in model["units"]:
        if u.role == "anchored":
            place_anchored(ctx, P, u)
    for u in model["units"]:
        if u.role == "aligned" and u.els[0]["id"] not in P.props:
            if not place_aligned(ctx, P, u, model):
                place_float(ctx, P, u, min(t, 1.5))
    content = list(P.vis.values())
    for u in sorted(model["units"], key=lambda u: -u.prio):
        if u.role == "ribbon" and u.els[0]["id"] not in P.props:   # not already in the row
            place_float(ctx, P, u, min(t, 1.5))
        elif u.role == "minor":
            # decoration never outgrows the canvas's own shape-preserving scale
            place_scatter(ctx, P, u, min(t, 1.5, ctx.W / ctx.Ws, ctx.H / ctx.Hs), content)
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
        return rot_bounds(x, y, w, h, e.get("rotation", 0))
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
