"""Dev tool: measure designer decisions on Crello families and fit the shallow decision trees.

    /data/agents/work/designer/venv-dev/bin/python -m tools.designer_fit dataset   # -> rows.jsonl
    /data/agents/work/designer/venv-dev/bin/python -m tools.designer_fit fit       # CV table + designer_tree.json

Data: /data/agents/work/crello-wide/families.jsonl (geometry only). Tournament families are
excluded from every fit. Rows: one per ordered pair (source member, target member) of a family.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.smart import decisions as D  # noqa: E402

FAMILIES = Path(os.environ.get("FAMILIES", "/data/agents/work/crello-wide/families.jsonl"))
TOURNAMENT = Path("/data/agents/work/designer/tournament_families.txt")
WORK = Path(os.environ.get("DESIGNER_WORK", "/data/agents/work/designer/tree"))
TREE_OUT = ROOT / "engine" / "smart" / "designer_tree.json"


def member_to_design(m: dict) -> dict:
    """families.jsonl member -> Chameleon-schema design. TextElement -> text (content, font_size),
    ImageElement/SvgElement/SvgMaskElement -> image, others -> shape. ``_kind`` keeps the Crello kind
    for matching; ids are unique per member so matching across members goes by content/kind."""
    els = []
    for e in m["els"]:
        k = e["kind"]
        base = {"id": f"{m['id']}_{e['i']}", "x": e["x"], "y": e["y"], "width": e["w"], "height": e["h"],
                "z_index": e["i"], "rotation": e.get("angle", 0.0), "opacity": e.get("opacity", 1.0), "_kind": k}
        if k == "TextElement":
            base.update(type="text", content=e.get("text", ""), font_size=e.get("fs") or 0.0,
                        font_family=e.get("font", ""), text_align=e.get("align") or "left")
        elif k in ("ImageElement", "SvgElement", "SvgMaskElement"):
            base.update(type="image", asset_id="", fit="fill")
        else:
            base.update(type="shape", shape_type="rectangle", fill_color=e.get("color") or "rgba(0,0,0,1)")
        els.append(base)
    return {"canvas": {"width": m["W"], "height": m["H"]}, "elements": els}


def tournament() -> set:
    return {l.strip() for l in TOURNAMENT.read_text().split() if l.strip()}


def fold_of(fam: str, k: int = 5) -> int:
    return int(hashlib.md5(fam.encode()).hexdigest(), 16) % k


def iter_families():
    with FAMILIES.open() as fh:
        for line in fh:
            yield json.loads(line)


def aspect_class(x: float) -> str:
    return "small" if x < 0.3 else ("mid" if x <= 0.9 else "large")


def build_rows(out_path: Path):
    tour = tournament()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as fo:
        for f in iter_families():
            designs = [member_to_design(m) for m in f["members"]]
            for i, s in enumerate(designs):
                for j, t in enumerate(designs):
                    if i == j:
                        continue
                    feats = D.source_features(s, t)
                    dec = D.decisions_of(s, t)
                    keep = D.keep_source(s)
                    fo.write(json.dumps({"fam": f["fam"], "tour": f["fam"] in tour, "n_sizes": f["n_sizes"],
                                         "src": f["members"][i]["id"], "tgt": f["members"][j]["id"],
                                         "feats": feats, "dec": dec, "keep": keep}) + "\n")
                    n += 1
    return n


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    if cmd == "dataset":
        print(build_rows(WORK / "rows.jsonl"), "rows")


# ---------------------------------------------------------------- fit

CAT = ("bg_kind", "copy_on_photo", "cut", "layout", "src_aspect", "tgt_aspect",
       "src_anchor:headline", "src_anchor:hero", "src_anchor:art", "src_anchor:small", "src_align", "src_photo",
       "src_anchor_row:headline", "src_anchor_row:hero", "src_anchor_row:art", "src_anchor_row:small",
       "src_anchor_col:headline", "src_anchor_col:hero", "src_anchor_col:art", "src_anchor_col:small",
       "src_coverage", "src_scale:headline")
NUM = ("cut_n", "n_images", "text_only", "headline_share", "hero_share", "n_text", "log_src_ar", "log_tgt_ar",
       "log_aspect_change", "abs_aspect_change", "area_ratio")
DECISIONS = ("arrangement", "group_kept", "order_kept",
             "drop:headline", "drop:body", "drop:small", "drop:hero", "drop:art", "drop:shape", "drop:background",
             "anchor:headline", "anchor:hero", "anchor:art", "anchor:small",
             "anchor_row:headline", "anchor_row:hero", "anchor_row:art", "anchor_row:small",
             "anchor_col:headline", "anchor_col:hero", "anchor_col:art", "anchor_col:small",
             "align", "scale:headline", "scale:body", "scale:hero", "scale:art", "emphasis", "photo", "coverage")


def load_rows(path: Path = None):
    path = path or WORK / "rows.jsonl"
    return [json.loads(l) for l in path.open()]


def encoder(rows):
    cols = list(NUM)
    for c in CAT:
        for v in sorted({str(r["feats"].get(c)) for r in rows} - {"None"}):
            cols.append(f"{c}={v}")
    return cols


def vec(feats, cols):
    out = []
    for c in cols:
        if "=" in c:
            f, v = c.split("=", 1)
            out.append(1.0 if str(feats.get(f)) == v else 0.0)
        else:
            x = feats.get(c, 0.0)
            out.append(float(x) if isinstance(x, (int, float)) else 0.0)
    return out


def export(clf, cols, X, y):
    """sklearn tree -> nested dict; leaf value/purity/count from the *unweighted* training rows."""
    import numpy as np
    t = clf.tree_
    leaves = clf.apply(X)
    counts = collections.defaultdict(collections.Counter)
    for lf, lab in zip(leaves, y):
        counts[lf][lab] += 1
    classes = list(clf.classes_)

    def node(i):
        if t.children_left[i] == -1:
            c = counts.get(i)
            if c:
                v, k = c.most_common(1)[0]
                n = sum(c.values())
            else:  # leaf with no unweighted rows (cannot happen with min_samples_leaf) -> tree's argmax
                v, k, n = classes[int(np.argmax(t.value[i][0]))], 0, 0
            # value predicted by sklearn (weighted argmax) is what predict() must reproduce
            v_sk = classes[int(np.argmax(t.value[i][0]))]
            return {"leaf": {"v": v_sk, "p": round((c[v_sk] / n) if n else 0.0, 3), "n": int(n),
                             "dist": {kk: int(vv) for kk, vv in c.most_common()}}}
        name = cols[t.feature[i]]
        d = {"l": node(t.children_left[i]), "r": node(t.children_right[i])}
        if "=" in name:
            f, v = name.split("=", 1)
            d.update(f=f, eq=v)
        else:
            d.update(f=name, t=float(t.threshold[i]))
        return d
    return node(0)


def macro_f1(y, p):
    labs = sorted(set(y) | set(p))
    fs = []
    for l in labs:
        if l not in set(y):
            continue
        tp = sum(1 for a, b in zip(y, p) if a == l and b == l)
        fp = sum(1 for a, b in zip(y, p) if a != l and b == l)
        fn = sum(1 for a, b in zip(y, p) if a == l and b != l)
        fs.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(fs) / max(1, len(fs))


def acc(y, p):
    return sum(1 for a, b in zip(y, p) if a == b) / max(1, len(y))


def fit(depth=4, leaf=30, regime=None, write=True):
    """regime: None = all rows, else only rows of that aspect-change class (small/mid/large)."""
    from sklearn.tree import DecisionTreeClassifier
    rows = [r for r in load_rows() if not r["tour"]]
    if regime:
        rows = [r for r in rows if aspect_class(r["feats"]["abs_aspect_change"]) == regime]
    cols = encoder(rows)
    X_all = [vec(r["feats"], cols) for r in rows]
    folds = [fold_of(r["fam"]) for r in rows]
    report = {"n_rows": len(rows), "n_families": len({r["fam"] for r in rows}), "decisions": {}}
    trees = {}
    for dec in DECISIONS:
        idx = [i for i, r in enumerate(rows) if dec in r["dec"]]
        if len(idx) < 150:
            continue
        y = [rows[i]["dec"][dec] for i in idx]
        keep = [rows[i]["keep"].get(dec, "?") for i in idx]
        # relative target: "=" when the designer kept the source's own value (a depth-4 tree cannot
        # express the identity map over 9 zones); predict() maps "=" back to the source value
        rel = ("src_" + dec) in rows[idx[0]]["feats"]
        y_abs = y
        if rel:
            y = ["=" if v == rows[i]["feats"].get("src_" + dec) else v for v, i in zip(y, idx)]
        X = [X_all[i] for i in idx]
        fo = [folds[i] for i in idx]
        ac = [aspect_class(rows[i]["feats"]["abs_aspect_change"]) for i in idx]
        best = None
        for cw in (None, "balanced"):
            pred = [None] * len(idx)
            maj = [None] * len(idx)
            for k in range(5):
                tr = [j for j in range(len(idx)) if fo[j] != k]
                te = [j for j in range(len(idx)) if fo[j] == k]
                clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, class_weight=cw, random_state=0)
                clf.fit([X[j] for j in tr], [y[j] for j in tr])
                pp = clf.predict([X[j] for j in te])
                mj = collections.Counter(y[j] for j in tr).most_common(1)[0][0]
                mj = collections.Counter(y_abs[j] for j in tr).most_common(1)[0][0]
                for j, v in zip(te, pp):
                    pred[j] = keep[j] if v == "=" else v
                    maj[j] = mj
            f1 = macro_f1(y_abs, pred)
            a = acc(y_abs, pred)
            # balanced only if it clearly helps macro-F1 without costing accuracy
            if best is None or (f1 > best[0] + 0.02 and a > best[4] - 0.03):
                best = (f1, cw, pred, maj, a)
        f1, cw, pred, maj, _ = best
        res = {"n": len(idx), "class_weight": cw, "relative": rel, "dist": dict(collections.Counter(y_abs).most_common())}
        for sub in ("all", "small", "mid", "large"):
            js = [j for j in range(len(idx)) if sub == "all" or ac[j] == sub]
            if not js:
                continue
            yy = [y_abs[j] for j in js]
            res[sub] = {"n": len(js),
                        "tree_acc": acc(yy, [pred[j] for j in js]), "tree_f1": macro_f1(yy, [pred[j] for j in js]),
                        "maj_acc": acc(yy, [maj[j] for j in js]), "maj_f1": macro_f1(yy, [maj[j] for j in js]),
                        "keep_acc": acc(yy, [keep[j] for j in js]), "keep_f1": macro_f1(yy, [keep[j] for j in js]),
                        "dist": dict(collections.Counter(yy).most_common())}
            # per-fold: does the tree beat the better baseline (accuracy) in that fold?
            w = 0
            for k in range(5):
                jk = [j for j in js if fo[j] == k]
                if not jk:
                    continue
                yk = [y_abs[j] for j in jk]
                bt = max(acc(yk, [maj[j] for j in jk]), acc(yk, [keep[j] for j in jk]))
                w += acc(yk, [pred[j] for j in jk]) > bt
            res[sub]["fold_wins"] = w
        report["decisions"][dec] = res
        clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, class_weight=cw, random_state=0)
        clf.fit(X, y)
        trees[dec] = export(clf, cols, X, y)
        # pure-Python predict must reproduce sklearn on held-out-style rows
        D._TREES = {dec: trees[dec]}
        sk = [keep[j] if v == "=" else v for j, v in enumerate(clf.predict(X))]
        py = [D.predict(rows[i]["feats"])[dec][0] for i in idx]
        res["py_matches_sklearn"] = sum(1 for a, b in zip(sk, py) if a == b) / len(idx)
    D._TREES = None
    report["trees"] = trees
    if not write:
        return report
    TREE_OUT.write_text(json.dumps({"version": 1, "depth": depth, "min_leaf": leaf, "features": cols,
                                    "n_rows": len(rows), "trees": trees}, indent=1))
    (WORK / "cv_report.json").write_text(json.dumps(report, indent=1))
    return report


def table(report):
    lines = ["| decision | n | tree acc | maj acc | keep acc | tree F1 | maj F1 | keep F1 | folds won | >0.9: n | tree acc | maj acc | keep acc | tree F1 | maj F1 | keep F1 | folds won |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for dec, r in report["decisions"].items():
        a, L = r["all"], r.get("large", {"n": 0})
        g = lambda d, k: f"{d[k]:.3f}" if k in d else "-"
        lines.append(f"| {dec}{' (bal)' if r['class_weight'] else ''} | {a['n']} | {g(a,'tree_acc')} | {g(a,'maj_acc')} | {g(a,'keep_acc')} | "
                     f"{g(a,'tree_f1')} | {g(a,'maj_f1')} | {g(a,'keep_f1')} | {a.get('fold_wins','-')}/5 | {L['n']} | {g(L,'tree_acc')} | {g(L,'maj_acc')} | "
                     f"{g(L,'keep_acc')} | {g(L,'tree_f1')} | {g(L,'maj_f1')} | {g(L,'keep_f1')} | {L.get('fold_wins','-')}/5 |")
    return "\n".join(lines)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("fit", "regime"):
    rep = fit(regime=sys.argv[2], write=False) if sys.argv[1] == "regime" else fit()
    print(rep["n_rows"], "rows", rep["n_families"], "families")
    print(table(rep))
    (WORK / ("cv_table.md" if sys.argv[1] == "fit" else f"cv_table_{sys.argv[2]}.md")).write_text(table(rep) + "\n")
    print({d: r["py_matches_sklearn"] for d, r in rep["decisions"].items()})
