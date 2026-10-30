"""Dev harness for lane L3 (not in the image): run inputs x formats, check hard rules, render a sheet.

    python -m tools.lane formats [--sheet out.png] [files...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FORMATS = {
    "ig-post": (1080, 1080), "ig-portrait": (1080, 1350), "ig-story": (1080, 1920), "fb-post": (1200, 630),
    "linkedin": (1200, 627), "x-post": (1600, 900), "yt-thumb": (1280, 720), "pinterest": (1000, 1500),
    "mrec": (300, 250), "leaderboard": (728, 90), "skyscraper": (160, 600), "mobile": (320, 50),
    "billboard": (970, 250), "hd": (1920, 1080), "og": (1200, 628), "a4": (2480, 3508),
    "half-page": (300, 600), "fb-cover": (1640, 624), "yt-banner": (2560, 1440), "pin-long": (1000, 2100),
}


def check(src: dict, out: dict, target: dict) -> list:
    errs = []
    W, H = target["width"], target["height"]
    sid = [e["id"] for e in src["elements"]]
    oid = [e["id"] for e in out["elements"]]
    if sorted(sid) != sorted(oid):
        errs.append("ids")
    if out["canvas"]["width"] != W or out["canvas"]["height"] != H:
        errs.append("canvas")
    byid = {e["id"]: e for e in src["elements"]}
    for e in out["elements"]:
        s = byid.get(e["id"])
        if s is None:
            continue
        for k in ("content", "font_family", "type", "asset_id"):
            if s.get(k) != e.get(k):
                errs.append(f"{e['id']}:{k}")
        sx, sy, sw, sh = s["x"], s["y"], s["width"], s["height"]
        src_bleeds = sx < -0.5 or sy < -0.5 or sx + sw > src["canvas"]["width"] + .5 or sy + sh > src["canvas"]["height"] + .5
        x, y, w, h = e["x"], e["y"], e["width"], e["height"]
        if not src_bleeds and (x < -0.5 or y < -0.5 or x + w > W + 0.5 or y + h > H + 0.5) and not e.get("rotation"):
            errs.append(f"{e['id']}:off")
        if e["type"] == "image" and e.get("fit") != "cover":
            r0, r1 = sw / sh, w / h
            if abs(r1 / r0 - 1) > 0.05:
                errs.append(f"{e['id']}:aspect")
        if e["type"] == "text":
            if int(e["font_size"]) != e["font_size"] or e["font_size"] < 1:
                errs.append(f"{e['id']}:fs")
            from engine.opt.measure import text_block
            n, mw, th = text_block(e, int(e["font_size"]), w)
            if mw > w + 1:
                errs.append(f"{e['id']}:textw")
            if th > h + 1:
                errs.append(f"{e['id']}:texth")
    return errs


def run(files, sheet=None, formats=None):
    from engine.resize import resize
    tiles = []
    times = []
    nerr = 0
    for f in files:
        d = json.loads(Path(f).read_text())
        src = d["source"] if "source" in d else d
        for name, (W, H) in (formats or FORMATS).items():
            t0 = time.time()
            out = resize(json.loads(json.dumps(src)), {"width": W, "height": H})
            times.append(time.time() - t0)
            errs = check(src, out, {"width": W, "height": H})
            if errs:
                nerr += 1
                print(Path(f).stem, name, errs)
            tiles.append((Path(f).stem, name, out))
    times.sort()
    print(f"tiles {len(tiles)} faulty {nerr} p50 {times[len(times)//2]:.2f}s p95 {times[int(len(times)*.95)]:.2f}s max {times[-1]:.2f}s")
    if sheet:
        render_sheet(tiles, sheet)


def render_sheet(tiles, path, cell=300, kit=None):
    import os
    import tempfile
    from PIL import Image, ImageDraw
    from renderer.converter.convert import render_design_to_png
    kit = Path(kit or os.environ.get("CHAMELEON_BRAND_KIT", ROOT / "brand_kit"))
    fonts = ROOT / "brand_kit" / "fonts"
    cols = 6 if any(t[1] == 'src' for t in tiles) else 5
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * (cell + 10), rows * (cell + 26)), (90, 90, 90))
    dr = ImageDraw.Draw(sheet)
    with tempfile.TemporaryDirectory() as td:
        for k, (stem, name, out) in enumerate(tiles):
            p = Path(td) / f"{k}.png"
            try:
                render_design_to_png(out, kit / "assets", p, fonts_dir=fonts, max_size=600)
                im = Image.open(p).convert("RGB")
            except Exception as ex:  # noqa: BLE001
                im = Image.new("RGB", (100, 100), (255, 0, 0))
                print("render fail", stem, name, ex)
            im.thumbnail((cell, cell))
            cx, cy = (k % cols) * (cell + 10), (k // cols) * (cell + 26)
            sheet.paste(im, (cx + 5, cy + 22))
            dr.text((cx + 5, cy + 4), f"{stem[:22]} {name}", fill=(255, 255, 255))
    sheet.save(path)
    print("sheet", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--sheet")
    ap.add_argument("--only")
    a = ap.parse_args()
    files = a.files or sorted(str(p) for p in (ROOT / "input").glob("*.json"))
    fm = None
    if a.cmd == "crello":
        import random
        os_env = __import__("os").environ
        os_env.setdefault("CHAMELEON_BRAND_KIT", "/data/agents/work/crello-wide/cases")
        from engine.resize import resize
        cases = sorted(Path("/data/agents/work/crello-wide/cases").glob("*.json"))
        random.Random(int(a.only or 0)).shuffle(cases)
        tiles = []
        for c in cases[:10]:
            f = json.loads(c.read_text())
            t = f["targets"][0]
            out = resize(json.loads(json.dumps(f["source"])), dict(t["canvas"]))
            errs = check(f["source"], out, t["canvas"])
            if errs:
                print(c.stem, errs)
            tiles += [(c.stem, "src", f["source"]), (c.stem, f"ours {t['canvas']['width']}x{t['canvas']['height']}", out),
                      (c.stem, "designer", t)]
        render_sheet(tiles, a.sheet, kit="/data/agents/work/crello-wide/cases")
        sys.exit(0)
    if a.only:
        fm = {k: FORMATS[k] for k in a.only.split(",")}
    run(files, a.sheet, fm)
