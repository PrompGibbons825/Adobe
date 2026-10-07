"""Second pass for a candidate: background crop, decoration, floating
elements (ribbons, small images), hiding what didn't make the cut, and
keeping content inside the canvas."""

from __future__ import annotations

import numpy as np

from . import saliency
from .blocks import GroupBlock, ImageBlock, rot_bounds

HIDE_ROLES = {"missing"}


def place_bg(ctx, P, unit):
    el = unit.els[0]
    W, H = ctx.W, ctx.H
    info = saliency.analyze(el.get("asset_id") or "") if el["type"] == "image" else None
    if info is None:
        P.set(el["id"], x=0, y=0, width=W, height=H)
        P.bg_map = None
        return
    iw, ih = info["size"]
    s = max(W / iw, H / ih)
    bw, bh = iw * s, ih * s
    wx, wy = W / bw, H / bh  # visible window, as fraction of image
    sal = info["sal"]
    tot = float(sal.sum()) + 1e-8

    def kept(x0, y0):
        return saliency.region_mean(sal, x0, y0, x0 + wx, y0 + wy) * (wx * wy) * sal.size / tot

    cx0, cy0 = (1 - wx) / 2, (1 - wy) / 2
    sb = info["sal_box"]
    concentrated = (sb[2] - sb[0]) * (sb[3] - sb[1]) < 0.45
    if not concentrated or kept(cx0, cy0) >= 0.85 or (wx > 0.98 and wy > 0.98):
        x0, y0 = cx0, cy0
    else:
        # slide the window to keep the salient centroid as central as possible
        cx, cy = info["centroid"]
        x0 = min(max(cx - wx / 2, 0.0), 1 - wx)
        y0 = min(max(cy - wy / 2, 0.0), 1 - wy)
    if abs(x0 - cx0) < 1e-3 and abs(y0 - cy0) < 1e-3 and el.get("fit", "contain") == "cover":
        P.set(el["id"], x=0, y=0, width=W, height=H)
    else:
        P.set(el["id"], x=-x0 * bw, y=-y0 * bh, width=bw, height=bh)
    P.bg_map = (info["busy"], -x0 * bw, -y0 * bh, bw, bh)


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
    for _ in range(3):
        bw, bh = (sb[2] - sb[0]) * s, (sb[3] - sb[1]) * s
        px = max(4, pad_x * s)
        py = max(4, pad_y * s)
        tx = px if left else W - px - bw
        ty = py if top else H - py - bh
        tx = min(max(tx, 0), max(0, W - bw))
        ty = min(max(ty, 0), max(0, H - bh))
        box = (tx, ty, tx + bw, ty + bh)
        if all(_overlap(box, o) <= 0.03 * _area(box) for o in occupied) and bw <= W and bh <= H:
            if unit.kind == "image":
                el = unit.els[0]
                ew, eh = el["width"] * s, el["height"] * s
                P.set(el["id"], x=tx + (ew - bw) / 2 * 0, y=ty, width=ew, height=eh)
                v = blk.v
                P.vis[el["id"]] = (tx + v[0] * ew, ty + v[1] * eh, tx + v[2] * ew, ty + v[3] * eh)
            else:
                b = blk.bounds_at(0, 0, s)
                blk.place_at(P, tx - b[0], ty - b[1], s)
            return True
        s *= 0.75
    return False


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
    skip = {e["id"] for u in model["units"] if u.role in ("bg", "deco") for e in u.els}
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


def finalize(ctx, P, model, float_threshold=0.0):
    P.hidden = set()
    P.min_z = min(e.get("z_index", 0) for e in model["els"].values())
    P.bg_map = None
    t = P.meta.get("t", ctx.t_star)
    for u in model["units"]:
        if u.role == "bg":
            place_bg(ctx, P, u)
        elif u.role == "deco":
            place_deco(ctx, P, u)
    for u in sorted(model["units"], key=lambda u: -u.prio):
        if u.role in ("ribbon", "minor"):
            ok = u.prio >= float_threshold and place_float(ctx, P, u, min(t, 1.5))
            if not ok:
                for e in u.els:
                    hide(P, e, ctx.W, ctx.H)
    for u in model["units"]:
        for eid in u.ids:
            if u.role in HIDE_ROLES or eid not in P.props:
                hide(P, model["els"][eid], ctx.W, ctx.H)
    clamp_inside(ctx, P, model)
    return P
