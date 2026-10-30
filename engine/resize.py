"""
THIS IS THE FILE YOU IMPLEMENT.

Everything else in this repo (input loading, output writing, rendering,
the Dockerfile) is plumbing so you can focus on exactly one function:
`resize()`. See RULES.md and the design schema for the full contract.
"""

from typing import Any


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """
    Args:
        source: {"canvas": {"width": int, "height": int}, "elements": [...]}
                 — the design at its original size. See
                 chameleon_design_schema_v0.9.json for the full element spec
                 (text / image / shape, position, size, rotation, opacity,
                 z_index, group_id, and type-specific fields).
        target_canvas: {"width": int, "height": int} — the shape you're
                 resizing into.

    Returns:
        A design dict with the SAME shape as `source`:
        {"canvas": target_canvas, "elements": [...]}

        Hard requirements (see RULES.md for the full list):
        - Every element `id` in `source` must appear exactly once in your
          output. Do not add, remove, or invent elements.
        - Do not regenerate or redraw any asset's pixel content — you are
          repositioning and resizing the elements you were given, nothing
          more. This is a disqualifying rule, not a style preference.
        - Non-background image elements should be kept in proportion
          (matching the source aspect ratio within ~5%) unless they are a
          full-bleed / `fit: "cover"` element, which is expected to crop.
        - `font_size` may change to help text fit. `font_family` and
          `content` should not, for this stage of scoring (see RULES.md,
          "Adaptive typography and copy").
        - Nothing should end up outside `target_canvas`'s bounds.

    This is where your actual approach goes: constraint solving, a learned
    layout model, heuristics, whatever you choose. Latency is not a scoring
    concern (see RULES.md) — spend your effort on quality, not speed.
    """

    try:
        from engine.opt.analyse import analyse
        from engine.opt.emit import emit
        from engine.opt.solve import search
        W = float(target_canvas["width"])
        H = float(target_canvas["height"])
        S = analyse(source)
        best = None
        # the legibility floor is relaxed only when the canvas provably cannot hold the copy
        # without collisions (capacity): full floor, then 60%, then none
        for floor_k in (1.0, 0.6, 0.0):
            S.floor_k = floor_k
            scored = search(S, W, H)
            tried = set()
            for e, _, name, placed in scored:
                if len(tried) >= 6 or name in tried:
                    if len(tried) >= 6:
                        break
                    continue
                tried.add(name)
                cand = emit(S, placed, W, H)
                bad = _collisions(source, cand)   # verify at the destination: drawn boxes, not the plan
                if best is None or bad < best[0]:
                    best = (bad, cand)
                if bad == 0:
                    break
            if best is not None and best[0] == 0:
                break
        out = best[1] if best else emit(S, {}, W, H)
        out["canvas"] = {"width": target_canvas["width"], "height": target_canvas["height"]}
        return out
    except Exception:  # noqa: BLE001
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        return _fallback(source, target_canvas)


# new overlap (share of the smaller box) beyond the source's own layering that counts as a
# collision. Dev set (120 family-disjoint tasks): 0.02 -> sim 0.357, 2 copy-over-copy;
# 0.15 -> 0.370, 3; off -> 0.392, 11.
_GUARD = 0.15


def _collisions(source: dict, out: dict) -> int:
    """Pairs that collide in the output but not in the source, where at least one is text."""
    src = {e["id"]: e for e in source["elements"]}
    els = [e for e in out["elements"] if e["id"] in src]
    W, H = float(out["canvas"]["width"]), float(out["canvas"]["height"])

    def inter(a, b):
        iw = min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"])
        ih = min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"])
        return max(0.0, iw) * max(0.0, ih)

    n = 0
    for i in range(len(els)):
        a = els[i]
        for j in range(i + 1, len(els)):
            b = els[j]
            if a["width"] * a["height"] >= 0.8 * W * H or b["width"] * b["height"] >= 0.8 * W * H:
                continue          # backdrop
            sa, sb = src[a["id"]], src[b["id"]]
            if a.get("type") != "text" and b.get("type") != "text":
                # a picture newly buried under a later one (hidden content)
                lo, hi = (a, b) if a.get("z_index", 0) <= b.get("z_index", 0) else (b, a)
                slo, shi = src[lo["id"]], src[hi["id"]]
                c_out = inter(lo, hi) / max(1e-6, lo["width"] * lo["height"])
                c_src = inter(slo, shi) / max(1e-6, slo["width"] * slo["height"])
                if c_out > 0.5 and c_src < 0.2:
                    n += 1
                continue
            r_src = inter(sa, sb) / max(1e-6, min(sa["width"] * sa["height"], sb["width"] * sb["height"]))
            r_out = inter(a, b) / max(1e-6, min(a["width"] * a["height"], b["width"] * b["height"]))
            if r_out > r_src + _GUARD and not (r_src > 0.5 and r_out > 0.5):   # deeper than the source's own layering
                n += 1
    return n


def _fallback(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """Uniform fit of the whole design, centred; backdrop-sized images cover the canvas."""
    W, H = float(target_canvas["width"]), float(target_canvas["height"])
    W0, H0 = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    s = min(W / W0, H / H0)
    ox, oy = (W - W0 * s) / 2, (H - H0 * s) / 2
    out_elements = []
    for el in source["elements"]:
        el = dict(el)
        w, h = float(el.get("width", 1)), float(el.get("height", 1))
        if el.get("type") != "text" and w * h >= 0.85 * W0 * H0:
            el.update(x=0.0, y=0.0, width=W, height=H)
            if el.get("type") == "image":
                el["fit"] = "cover"
        else:
            nw, nh = max(1.0, w * s), max(1.0, h * s)
            el["x"] = min(max(0.0, ox + float(el.get("x", 0)) * s), max(0.0, W - nw))
            el["y"] = min(max(0.0, oy + float(el.get("y", 0)) * s), max(0.0, H - nh))
            el["width"], el["height"] = min(nw, W), min(nh, H)
            if el.get("type") == "text":
                el["font_size"] = max(1, int(float(el.get("font_size", 12)) * s))
        out_elements.append(el)
    return {"canvas": {"width": target_canvas["width"], "height": target_canvas["height"]}, "elements": out_elements}
