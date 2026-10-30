import math
from typing import Optional

def _norm(s) -> str:
    return " ".join(str(s or "").lower().split())


def similarity(ours: dict, ref: dict) -> Optional[float]:
    """Mean IoU of our element boxes with the designer's, in canvas fractions. Texts are matched
    by content (duplicates in order). Other elements are matched greedily, in z order, to the
    unused designer element of the same type with the closest aspect ratio and canvas share (the
    designer re-orders layers between sizes). None if nothing matches. A crude closeness measure,
    not a quality score: the designer is one good answer among many."""
    import math

    def boxes(d):
        W, H = d["canvas"]["width"], d["canvas"]["height"]
        return [(e, (e["x"] / W, e["y"] / H, (e["x"] + e["width"]) / W, (e["y"] + e["height"]) / H)) for e in d["elements"]]

    def area(b):
        return max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))

    def asp(b):
        return max(1e-6, (b[2] - b[0])) / max(1e-6, (b[3] - b[1]))

    def iou(b, r):
        iw = max(0.0, min(b[2], r[2]) - max(b[0], r[0]))
        ih = max(0.0, min(b[3], r[3]) - max(b[1], r[1]))
        inter = iw * ih
        uni = area(b) + area(r) - inter
        return inter / uni if uni > 0 else 0.0

    txt, oth = {}, {}
    for e, b in boxes(ref):
        if e.get("type") == "text":
            txt.setdefault(_norm(e.get("content")), []).append(b)
        else:
            oth.setdefault(e.get("type"), []).append(b)
    seen, ious = {}, []
    for e, b in sorted(boxes(ours), key=lambda eb: eb[0].get("z_index", 0)):
        if e.get("type") == "text":
            k = _norm(e.get("content"))
            i = seen.get(k, 0)
            seen[k] = i + 1
            if i < len(txt.get(k, [])):
                ious.append(iou(b, txt[k][i]))
        else:
            cand = oth.get(e.get("type"), [])
            if cand:
                r = min(cand, key=lambda c: abs(math.log(asp(b) / asp(c))) + 0.3 * abs(math.log(area(b) / area(c))))
                cand.remove(r)
                ious.append(iou(b, r))
    return sum(ious) / len(ious) if ious else None


