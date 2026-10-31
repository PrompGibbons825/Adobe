"""Add the families LICA uses but fonts.pickle lacks, from google/fonts, as
regular/bold/italic faces in fonts.pickle's format. Variable fonts are pinned
to wght 400 and 700 with their other axes at default.

usage: python tools/lica_fonts.py --data lica-data --fonts IN.pickle --out OUT.pickle
"""

import argparse
import glob
import io
import json
import pickle
import re
import subprocess
from collections import Counter
from urllib.parse import quote

from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

RAW = "https://raw.githubusercontent.com/google/fonts/main"


def run(*cmd):
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def listing(family):
    slug = re.sub(r"[^a-z0-9]", "", family.lower())
    for lic in ("ofl", "apache", "ufl"):
        try:
            names = json.loads(run("gh", "api", f"repos/google/fonts/contents/{lic}/{slug}", "--jq", "[.[].name]"))
            return f"{lic}/{slug}", [n for n in names if n.endswith(".ttf")]
        except subprocess.CalledProcessError:
            continue
    return None, []


def get(path, name):
    return run("curl", "-fsSL", "-m", "60", f"{RAW}/{path}/{quote(name)}")


def pin(data, weight):
    f = TTFont(io.BytesIO(data))
    axes = {a.axisTag: a for a in f["fvar"].axes}
    pins = {t: a.defaultValue for t, a in axes.items()}
    if "wght" in axes:
        pins["wght"] = min(max(weight, axes["wght"].minValue), axes["wght"].maxValue)
    buf = io.BytesIO()
    instancer.instantiateVariableFont(f, pins).save(buf)
    return buf.getvalue()


def faces(path, names):
    """(weight, style) -> ttf bytes"""
    out = {}
    variable = [n for n in names if "[" in n]
    for n in variable:
        style = "italic" if "Italic" in n else "regular"
        data = get(path, n)
        for label, w in (("regular", 400), ("bold", 700)):
            out[label, style] = pin(data, w)
    if variable:
        return out
    want = {"Regular": ("regular", "regular"), "Bold": ("bold", "regular"),
            "Italic": ("regular", "italic"), "BoldItalic": ("bold", "italic")}
    for n in names:
        stem = n[:-4].split("-")[-1]
        if stem in want:
            out[want[stem]] = get(path, n)
    if not out and names:          # single-face families name no style
        out["regular", "regular"] = get(path, names[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.fonts, "rb") as f:
        fonts = pickle.load(f)
    known = {k.lower() for k in fonts}
    missing = Counter()
    for p in glob.glob(f"{a.data}/layouts/*/*.json"):
        for c in json.load(open(p))["components"]:
            if c["type"].startswith("TEXT"):
                fam = str(c.get("fontFamily", "")).split("--")[0].strip()
                if fam and fam.lower() not in known:
                    missing[fam] += 1
    failed = []
    for fam in missing:
        path, names = listing(fam)
        try:
            got = faces(path, names) if path else {}
        except Exception as e:
            print(fam, repr(e)[:120])
            got = {}
        if not got:
            failed.append(fam)
            continue
        key = fam.title()
        fonts[key] = [{"fontFamily": key, "fontWeight": w, "fontStyle": s, "bytes": b} for (w, s), b in got.items()]
        print(fam, sorted(got), flush=True)
    with open(a.out, "wb") as f:
        pickle.dump(fonts, f)
    print(f"added {len(missing) - len(failed)}/{len(missing)}; not found: {failed}")


if __name__ == "__main__":
    main()
