"""Our own critic: approximates the rubric so we can pick the best of many
candidate layouts. Higher is better. Returns (score, breakdown)."""

from __future__ import annotations

from . import saliency
from .finalize import _area, _overlap

MIN_ABS = {"headline": 16, "cta": 12, "subhead": 11, "body": 11, "group": 10,
           "badges": 9, "badge": 9, "ribbon": 9, "legal": 7}
MIN_REL = {"headline": 0.045, "cta": 0.022, "subhead": 0.018, "body": 0.016, "group": 0.016,
           "badges": 0.014, "badge": 0.014, "ribbon": 0.014, "legal": 0.008}


def score(ctx, P, model):
    W, H = ctx.W, ctx.H
    br = {}
    units = [u for u in model["units"] if u.role not in ("bg", "deco")]
    owner = {}
    for u in units:
        for eid in u.ids:
            owner[eid] = u

    # 1. what survived (importance-weighted)
    dropped = 0.0
    for u in units:
        if all(eid in P.hidden for eid in u.ids):
            dropped += u.prio
    br["dropped"] = -1.6 * dropped

    # 2. readability
    read = 0.0
    hfs = None
    for u in units:
        for eid in u.ids:
            el = model["els"][eid]
            if el["type"] != "text" or eid in P.hidden:
                continue
            fs = P.props[eid].get("font_size", el["font_size"])
            role = model.get("troles", {}).get(eid, u.role) if u.kind == "frame" else u.role
            mn = max(MIN_ABS.get(role, 10), MIN_REL.get(role, 0.015) * min(W, H))
            if role == "headline" and (hfs is None or fs > hfs):
                hfs = fs
            if role == "headline":
                mn = min(mn, el["font_size"] * ctx.t_star * 0.9) if mn > 16 else mn
            if fs < mn:
                read -= (u.prio + 3) * 3 * (1 - fs / mn)
    br["readability"] = read

    # 3. prominence: headline + hero reasonably large vs the ideal scale
    prom = 0.0
    head = next((u for u in units if u.role == "headline"), None)
    if head and hfs and any(e["type"] == "text" for e in head.els):
        ideal = max(e["font_size"] for e in head.els if e["type"] == "text") * ctx.t_star
        prom += 4 * min(1.0, hfs / ideal)
    hero = next((u for u in units if u.role == "hero"), None)
    if hero and hero.els[0]["id"] in P.vis:
        from .blocks import img_vis_frac
        e = hero.els[0]
        v = img_vis_frac(e)
        src_a = e["width"] * e["height"] * (v[2] - v[0]) * (v[3] - v[1])
        a = _area(P.vis[e["id"]])
        prom += 3 * min(1.0, a / (src_a * ctx.t_star ** 2 * 0.8))
    br["prominence"] = prom

    # 4. collisions between different units
    col = 0.0
    ids = [i for i in P.vis if i not in P.hidden and i in owner]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if owner[a] is owner[b]:
                continue
            if owner[a].kind == "badges" and owner[b] is owner[a]:
                continue
            va, vb = P.vis[a], P.vis[b]
            ov = _overlap(va, vb)
            if ov <= 0:
                continue
            frac = ov / max(1.0, min(_area(va), _area(vb)))
            ta, tb = model["els"][a]["type"], model["els"][b]["type"]
            w = 15 if "text" in (ta, tb) else 5
            col -= w * frac
    br["collisions"] = col

    # 5. text on busy background
    busy = 0.0
    if P.bg_map is not None:
        bmap, bx, by, bw, bh = P.bg_map
        for eid in ids:
            if model["els"][eid]["type"] != "text":
                continue
            v = P.vis[eid]
            b = saliency.region_mean(bmap, (v[0] - bx) / bw, (v[1] - by) / bh, (v[2] - bx) / bw, (v[3] - by) / bh)
            busy -= 10 * max(0.0, b - 0.06)
            if P.bg_sal is not None:
                sm = saliency.region_mean(P.bg_sal, (v[0] - bx) / bw, (v[1] - by) / bh, (v[2] - bx) / bw, (v[3] - by) / bh)
                busy -= 8 * max(0.0, sm - 0.3)
    br["busy_bg"] = busy

    # 6. balance: content should use the canvas, and stay off the edges
    vis = [P.vis[i] for i in ids]
    bal = 0.0
    if vis:
        x0 = min(v[0] for v in vis)
        y0 = min(v[1] for v in vis)
        x1 = max(v[2] for v in vis)
        y1 = max(v[3] for v in vis)
        fill = ((x1 - x0) * (y1 - y0)) / (W * H)
        bal += 3 * min(1.0, fill / 0.55)
        edge = 0.015 * min(W, H)
        for v in vis:
            if v[0] < edge or v[1] < edge or v[2] > W - edge or v[3] > H - edge:
                bal -= 0.5
    br["balance"] = bal

    return sum(br.values()), br
