"""Feature score of a realised layout. Higher is better."""
import math

WEIGHTS = dict(text=0.436, leg=3.0, img=3.088, fid=2.473, fill=0.376, panes=0.0, kdrop=0.1)  # fit by lab/learn.py


def features(L, W, H, sW, sH, s_ref):
    tl = ll = 0.0
    wsum = 0.0
    legthr = min(14.0, max(8.0, 0.02 * max(W, H)))
    for u, f, X, Y in L.placed:
        src_by_id = {e["id"]: e for e in u.members}
        for e in f.els:
            if e["type"] != "text":
                continue
            fs0 = float(src_by_id[e["id"]]["font_size"])
            wt = fs0 * math.sqrt(max(1, len(e["content"])))
            q = e["font_size"] / (fs0 * s_ref)
            lq = math.log(max(q, 1e-3))
            tl += wt * (max(0.0, -lq) ** 2 - 0.1 * min(max(lq, 0), 0.5))
            ll += wt * max(0.0, (legthr - e["font_size"]) / legthr) ** 2
            wsum += wt
    tl /= max(wsum, 1e-9)
    ll /= max(wsum, 1e-9)
    il, asum = 0.0, 0.0
    for u, f, X, Y in L.placed:
        if u.texts and not u.is_image:
            continue
        if not u.is_image:
            continue
        a0 = u.area
        ratio = (f.w * f.h) / (a0 * s_ref ** 2)
        il += math.sqrt(a0) * max(0.0, -0.5 * math.log(max(ratio, 1e-3))) ** 2
        asum += math.sqrt(a0)
    il /= max(asum, 1e-9)
    d, dn = 0.0, 0.0
    boxes = []
    for u, f, X, Y in L.placed:
        c1 = ((X + f.w / 2) / W, (Y + f.h / 2) / H)
        c0 = (u.cx / sW, u.cy / sH)
        w = math.sqrt(u.area)
        d += w * math.hypot(c1[0] - c0[0], c1[1] - c0[1])
        dn += w
        boxes.append((X, Y, X + f.w, Y + f.h))
    d /= max(dn, 1e-9)
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    cov = (x1 - x0) * (y1 - y0) / (W * H)
    fill = max(0.0, 0.5 - cov)
    kd = max(0.0, -math.log(max(L.k / s_ref, 1e-3)))
    L.feat = dict(text=tl, leg=ll, img=il, fid=d, fill=fill, panes=float(len(getattr(L, 'panes', [1])) - 1), kdrop=kd)
    return L.feat


def dot(feat, weights=None):
    w = weights or WEIGHTS
    return -sum(w[k] * v for k, v in feat.items())
