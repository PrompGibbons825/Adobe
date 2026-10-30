"""Hard-rule checker (own, until L0's fundamentals.check lands)."""
def check(src, out, tol=0.5):
    errs = []
    S = {e["id"]: e for e in src["elements"]}
    ids = [e["id"] for e in out["elements"]]
    if sorted(ids) != sorted(S): errs.append("ids")
    W, H = out["canvas"]["width"], out["canvas"]["height"]
    sbg = src["canvas"]
    for e in out["elements"]:
        s = S.get(e["id"])
        if s is None: continue
        if e["x"] < -tol or e["y"] < -tol or e["x"] + e["width"] > W + tol or e["y"] + e["height"] > H + tol:
            if not e.get("rotation"): errs.append(f"off:{e['id']}")
        if e["type"] == "image" and e.get("fit") != "cover":
            a0 = s["width"] / max(s["height"], 1e-9); a1 = e["width"] / max(e["height"], 1e-9)
            if abs(a1 / a0 - 1) > .05: errs.append(f"aspect:{e['id']}")
        if e["type"] == "text":
            if e["content"] != s["content"] or e["font_family"] != s["font_family"]: errs.append(f"copy:{e['id']}")
            if e["font_size"] < 1: errs.append(f"fs:{e['id']}")
        for k in ("width", "height"):
            if e[k] <= 0: errs.append(f"size:{e['id']}")
    return errs
