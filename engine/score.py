"""Our own critic: approximates the rubric so we can pick the best of many
candidate layouts. Higher is better. Returns (score, breakdown)."""

from __future__ import annotations

from . import saliency
from .finalize import _area, _overlap

MIN_ABS = {"headline": 16, "cta": 12, "subhead": 11, "body": 11, "group": 10,
           "badges": 9, "badge": 9, "ribbon": 9, "legal": 7}
MIN_REL = {"headline": 0.045, "cta": 0.022, "subhead": 0.018, "body": 0.016, "group": 0.016,
           "badges": 0.014, "badge": 0.014, "ribbon": 0.014, "legal": 0.008}


def ink_fraction(el):
    """Share of an image's visible box that actually carries pixels: ~1 for a
    photo or solid icon, a few % for a thin outline frame."""
    if el["type"] != "image":
        return 1.0
    info = saliency.analyze(el.get("asset_id") or "")
    if info is None or el.get("fit") == "cover":
        return 1.0
    ab = info["alpha_box"]
    box_frac = max(1e-3, (ab[2] - ab[0]) * (ab[3] - ab[1]))
    return float(min(1.0, max(0.03, info["opaque_frac"] / box_frac)))


def weight_items(P, model, W, H, ids, units):
    """Visual weight of every visible element (inked area share; text x2)
    plus the photo subject. Returns (items, center_of_mass)
    with items = [(eid, cx, cy, weight, kind)]."""
    items = []
    for eid in ids:
        v = P.vis[eid]
        a = (v[2] - v[0]) * (v[3] - v[1]) / (W * H)
        el = model["els"][eid]
        typ = el["type"]
        w = a * (2.0 if typ == "text" else ink_fraction(el))
        items.append((eid, (v[0] + v[2]) / 2, (v[1] + v[3]) / 2, w, typ))
    if getattr(P, "bg_focal", None) is not None:
        fcx, fcy, fw = P.bg_focal
        items.append(("photo subject", fcx, fcy, fw, "subject"))
    tot = sum(i[3] for i in items)
    if tot <= 0:
        return items, None
    return items, (sum(i[1] * i[3] for i in items) / tot, sum(i[2] * i[3] for i in items) / tot)


def _rel_lum_rgb(r, g, b):
    def lin(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _color_lum(col):
    """Relative luminance of a CSS colour (rgb()/rgba()/#hex), or None."""
    import re
    if not col:
        return None
    m = re.match(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", col)
    if m:
        return _rel_lum_rgb(*(float(v) for v in m.groups()))
    m = re.match(r"#([0-9a-fA-F]{6})", col)
    if m:
        h = m.group(1)
        return _rel_lum_rgb(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    return None


def _surface_lum(el):
    """Luminance of what a card or plate shows, or None if unknown."""
    if el["type"] == "shape":
        return _color_lum(el.get("fill_color", ""))
    info = saliency.analyze(el.get("asset_id") or "")
    return info["mean_lum"] if info else None


def _behind_lum(P, model, eid, v):
    """Luminance behind a text's inked box: the topmost card or plate under
    it, else the background image under it, else the plain background."""
    import numpy as np
    z = model["els"][eid].get("z_index", 0)
    best = None
    for i, pv in P.vis.items():
        el = model["els"].get(i)
        if el is None or el["type"] == "text" or i in P.hidden or el.get("z_index", 0) >= z:
            continue
        if (el.get("opacity", 1) or 1) < 0.6:
            continue
        if _overlap(v, pv) < 0.6 * max(1.0, _area(v)):
            continue
        if el["type"] == "image":
            info = saliency.analyze(el.get("asset_id") or "")
            if info is None or info["opaque_frac"] < 0.6 or info["busyness"] > 0.03:
                continue          # a detailed picture is read through the image map below
        if best is None or el.get("z_index", 0) > best[0]:
            best = (el.get("z_index", 0), el)
    if best is not None:
        return _surface_lum(best[1])
    if getattr(P, "bg_lum", None) is not None:
        lmap, bx, by, bw, bh = P.bg_lum
        hh, ww = lmap.shape
        x0 = int(np.clip((v[0] - bx) / bw * ww, 0, ww - 1)); x1 = int(np.clip((v[2] - bx) / bw * ww, x0 + 1, ww))
        y0 = int(np.clip((v[1] - by) / bh * hh, 0, hh - 1)); y1 = int(np.clip((v[3] - by) / bh * hh, y0 + 1, hh))
        reg = lmap[y0:y1, x0:x1]
        if np.isfinite(reg).any():
            return float(np.nanmean(reg))
    bgs = [u for u in model["units"] if u.role == "bg"]
    if bgs:
        return _surface_lum(bgs[0].els[0])
    return 1.0                    # an empty canvas renders white


def _plates(P, model):
    """Opaque flat pieces (cards, bars, plates) as placed: (box, z)."""
    from .analyze import _fills_box
    out = []
    for i, v in P.vis.items():
        el = model["els"].get(i)
        if el is None or el["type"] == "text" or i in P.hidden:
            continue
        if (el.get("opacity", 1) or 1) < 0.8 or not _fills_box(el):
            continue
        if el["type"] == "image":
            info = saliency.analyze(el.get("asset_id") or "")
            if info is None or info["busyness"] > 0.03:
                continue
        out.append((v, el.get("z_index", 0)))
    return out


def score(ctx, P, model):
    W, H = ctx.W, ctx.H
    br = {}
    units = [u for u in model["units"] if u.role not in ("bg", "deco", "border", "rule")]
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
            # each text by its own role (a group's caption is not held to headline size)
            role = model.get("troles", {}).get(eid, u.role) if u.kind != "text" else u.role
            mn = max(MIN_ABS.get(role, 10), MIN_REL.get(role, 0.015) * min(W, H))
            if role == "headline" and (hfs is None or fs > hfs):
                hfs = fs
            if role == "headline":
                mn = min(mn, el["font_size"] * ctx.t_star * 0.9) if mn > 16 else mn
            if fs < mn:
                read -= (u.prio + 3) * 3 * (1 - fs / mn)
    br["readability"] = read

    # hierarchy: text sizes keep their source proportions (two texts set equal
    # stay equal; a headline stays bigger than its subhead by the same factor)
    import math
    sizes = []
    for u in units:
        if u.kind not in ("text", "group", "button", "frame"):
            continue
        for eid in u.ids:
            el = model["els"][eid]
            if el["type"] == "text" and eid not in P.hidden and eid in P.props:
                sizes.append((el["font_size"], P.props[eid].get("font_size", el["font_size"])))
    sizes.sort(reverse=True)
    sizes = sizes[:4]          # the type that carries the message
    drift = 0.0
    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            (s1, o1), (s2, o2) = sizes[i], sizes[j]          # s1 >= s2 in the source
            r_src, r_out = s1 / max(1e-6, s2), o1 / max(1e-6, o2)
            if r_src < 1.05:
                drift += abs(math.log(max(1e-6, r_out)))           # set equal: stay equal
            elif r_out < 1.0:
                drift += math.log(1.0 / max(1e-6, r_out))          # order flipped
            elif r_out > r_src:
                drift += 0.5 * math.log(r_out / r_src)             # secondary text shrank away
            # small text growing toward the headline (legibility) is free
    br["hierarchy"] = -2.0 * drift

    # 3. prominence: headline + hero reasonably large vs the ideal scale
    prom = 0.0
    head = next((u for u in units if u.role == "headline"), None)
    if head and hfs and any(e["type"] == "text" for e in head.els):
        ideal = max(e["font_size"] for e in head.els if e["type"] == "text") * ctx.t_star
        prom += 4 * min(1.0, hfs / ideal)
    hero = next((u for u in units if u.role == "hero"), None)
    if hero and all(i in P.vis for i in hero.ids if model["els"][i]["type"] != "text"):
        from .blocks import img_vis_frac
        src_a, a = 0.0, 0.0
        for e in hero.els:
            if e["type"] == "text":
                continue
            v = img_vis_frac(e)
            src_a += e["width"] * e["height"] * (v[2] - v[0]) * (v[3] - v[1])
            a += _area(P.vis[e["id"]])
        if src_a > 0:
            prom += 2 * min(1.0, a / (src_a * ctx.t_star ** 2 * 0.8))
            # the hero visual should command the canvas: reward up to ~30% of it
            prom += 3 * min(1.0, a / (0.30 * W * H))
    if hero and getattr(P, "backdrop", None) is hero:
        prom += 5      # a backdrop fills the canvas: as prominent as a hero gets
        # ...but only if it stays about as visible as the designer left it
        from .analyze import coverage
        over = [P.vis[i] for i in P.vis if i in owner and owner[i] is not hero
                and i not in P.hidden and model["els"][i]["type"] != "text"]
        covered = coverage(over, (0, 0, W, H))
        prom -= 10 * max(0.0, covered - getattr(hero, "src_covered", 0.0) - 0.1)
    br["prominence"] = prom

    # share of the canvas: the headline unit and the hero keep at least ~70%
    # of the share of the canvas they had in the source
    share = 0.0
    Ws, Hs = model["canvas"]
    for u in units:
        if u.role not in ("headline", "hero"):
            continue
        bs = [P.vis[i] for i in u.ids if i in P.vis]
        if not bs:
            continue
        if getattr(P, "backdrop", None) is u or (getattr(P, "panel", None) and P.panel["id"] in u.ids):
            continue
        x0, y0, x1, y1 = u.bbox
        src = max(0.0, min(x1, Ws) - max(x0, 0)) * max(0.0, min(y1, Hs) - max(y0, 0)) / (Ws * Hs)
        out = (max(b[2] for b in bs) - min(b[0] for b in bs)) * (max(b[3] for b in bs) - min(b[1] for b in bs)) / (W * H)
        if src > 0:
            share -= 6 * max(0.0, 0.7 - out / src)
    br["share"] = share

    # pictures never collapse into thumbnails: every picture covering 8%+ of the
    # source keeps at least 40% of its source share of the canvas (the part
    # actually on canvas, so cropping into a panel or backdrop counts)
    thumb = 0.0
    for eid, el in model["els"].items():
        if el["type"] != "image" or eid in P.hidden or eid not in P.props:
            continue
        info = saliency.analyze(el.get("asset_id") or "")
        if info is None or info["busyness"] < 0.03:
            continue
        sx0, sy0 = max(0.0, el["x"]), max(0.0, el["y"])
        sx1, sy1 = min(Ws, el["x"] + el["width"]), min(Hs, el["y"] + el["height"])
        s_src = max(0.0, sx1 - sx0) * max(0.0, sy1 - sy0) / (Ws * Hs)
        if s_src < 0.08 or s_src > 0.95:
            continue
        p = P.props[eid]
        x, y = p.get("x", el["x"]), p.get("y", el["y"])
        w, h = p.get("width", el["width"]), p.get("height", el["height"])
        s_out = max(0.0, min(W, x + w) - max(0.0, x)) * max(0.0, min(H, y + h) - max(0.0, y)) / (W * H)
        target = 0.4
        from .analyze import _fills_box
        if not _fills_box(el):
            # a cut-out can't be cropped: aim at most of the largest share it can have whole
            A = el["width"] / max(1e-6, el["height"])
            fit = W / (A * H) if A >= W / H else A * H / W
            target = min(target, 0.7 * min(1.0, fit) / s_src)
        thumb -= 25 * max(0.0, target - s_out / s_src) * min(1.0, s_src / 0.3)
    br["thumbnail"] = thumb

    # faithfulness: reshaping a frame has to pay for itself
    import math
    reshape = 0.0
    for u in units:
        if u.kind != "frame" or u.container is None:
            continue
        c = u.container
        pc = P.props.get(c["id"])
        if not pc or getattr(P, "panel", None) and P.panel["id"] == c["id"]:
            continue
        a_src = c["width"] / max(1e-6, c["height"])
        a_out = pc.get("width", c["width"]) / max(1e-6, pc.get("height", c["height"]))
        reshape -= 1.5 * abs(math.log(max(1e-6, a_out / a_src)))
    br["reshape"] = reshape

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
            if owner[a].role == "minor" and owner[b].role == "minor":
                continue          # decoration layered on decoration is the designer's texture
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
        plates = _plates(P, model)
        for eid in ids:
            if model["els"][eid]["type"] != "text":
                continue
            v = P.vis[eid]
            z = model["els"][eid].get("z_index", 0)
            if any(pz < z and _overlap(v, pb) >= 0.9 * max(1.0, _area(v)) for pb, pz in plates):
                continue          # text on its own card or plate is not on the background
            b = saliency.region_mean(bmap, (v[0] - bx) / bw, (v[1] - by) / bh, (v[2] - bx) / bw, (v[3] - by) / bh)
            busy -= 10 * max(0.0, b - 0.06)
            if P.bg_sal is not None:
                sm = saliency.region_mean(P.bg_sal, (v[0] - bx) / bw, (v[1] - by) / bh, (v[2] - bx) / bw, (v[3] - by) / bh)
                busy -= 8 * max(0.0, sm - 0.3)
    br["busy_bg"] = busy

    # contrast: text colour against what is actually behind it (WCAG ratio)
    con = 0.0
    for eid in ids:
        el = model["els"][eid]
        if el["type"] != "text":
            continue
        tl = _color_lum(el.get("color", ""))
        bl = _behind_lum(P, model, eid, P.vis[eid])
        if tl is None or bl is None:
            continue
        ratio = (max(tl, bl) + 0.05) / (min(tl, bl) + 0.05)
        if ratio < 3.0:
            con -= (owner[eid].prio + 3) * (3.0 - ratio) / 2.0
    br["contrast"] = con

    # 6. balance: content should use the canvas, and stay off the edges
    vis = [P.vis[i] for i in ids]
    bal = 0.0
    if vis:
        cvis = [P.vis[i] for i in ids if owner[i].role != "minor"] or vis   # content, not decoration
        x0 = min(v[0] for v in cvis)
        y0 = min(v[1] for v in cvis)
        x1 = max(v[2] for v in cvis)
        y1 = max(v[3] for v in cvis)
        fill = ((x1 - x0) * (y1 - y0)) / (W * H)
        # use the canvas about as much as the designer did (never asked for more than 55%)
        target = min(0.55, max(0.15, 1.1 * model.get("src_fill", 0.55)))
        bal += 3 * min(1.0, fill / target)
        edge = 0.015 * min(W, H)
        panel = getattr(P, "panel", None)
        for eid, v in zip(ids, vis):
            if panel and eid == panel["id"]:
                continue          # a full-bleed photo panel is meant to touch the edges
            if owner[eid].role == "minor":
                continue          # decoration may sit at (or bleed past) the edge
            if v[0] < edge or v[1] < edge or v[2] > W - edge or v[3] > H - edge:
                bal -= 0.5
    br["balance"] = bal

    # 7. visual weight balance (center of mass of text, images and the photo's subject)
    #    and the rule of thirds for the subject
    items, cm = weight_items(P, model, W, H, ids, units)
    P.weight_items, P.cm = items, cm
    wb = 0.0
    if cm is not None:
        cmx, cmy = cm[0] / W, cm[1] / H
        k = 12.0 if model.get("centered") else 8.0
        wb -= k * abs(cmx - 0.5) + 0.5 * k * abs(cmy - 0.5)
    if P.bg_focal is not None and W > 1.2 * H:
        fx = P.bg_focal[0] / W
        d = min(abs(fx - 1 / 3), abs(fx - 2 / 3), abs(fx - 0.5) * 1.5)
        wb += 1.5 * max(0.0, 1 - d / (1 / 6))
    # faithfulness: a centred design that comes out left-aligned has to earn it
    if model.get("centered"):
        tot = left = 0.0
        for eid in ids:
            el = model["els"][eid]
            if el["type"] != "text":
                continue
            a = _area(P.vis[eid])
            tot += a
            if P.props.get(eid, {}).get("text_align", el.get("text_align", "left")) == "left":
                left += a
        if tot and left / tot > 0.5:
            wb -= 1.5
    br["weight_balance"] = wb

    # 8. figure-ground relations: a unit that sat on the photo's subject in the
    #    source (a frame around the heart, a caption on the product) stays on it
    rel = 0.0
    if P.bg_focal is not None:
        fx, fy = P.bg_focal[0], P.bg_focal[1]
        for u in units:
            if not getattr(u, "on_subject", False):
                continue
            bs = [P.vis[i] for i in u.ids if i in P.vis]
            if not bs:
                continue
            x0, y0 = min(b[0] for b in bs), min(b[1] for b in bs)
            x1, y1 = max(b[2] for b in bs), max(b[3] for b in bs)
            inside = x0 <= fx <= x1 and y0 <= fy <= y1
            rel += 2.5 if inside else -2.5
    br["subject_relation"] = rel

    return sum(br.values()), br
