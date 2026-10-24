"""Report: every finicky case on each of its targets, and what went wrong.

    .venv/bin/python -m tests.run                       # all cases
    .venv/bin/python -m tests.run rotated no_background # cases whose name contains a word
    .venv/bin/python -m tests.run --sheet /tmp/f.png    # also a contact sheet: source, then each target

Unlike the unit tests this never stops at the first failure, so one run shows
the state of the whole suite.
"""

from __future__ import annotations

import argparse
import sys
import time

from tests.finicky import cases, malformed
from tests.harness import evaluate, run
from tools.rules import hard_rule_errors, render_image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*")
    ap.add_argument("--sheet", help="write a contact sheet of every output here")
    ap.add_argument("--no-render", action="store_true", help="skip the ghost-text render check")
    a = ap.parse_args()

    from tools.sheet import tile  # dev-only imports
    from PIL import Image

    tiles = []
    bad = total = 0
    t0 = time.time()
    for case in cases():
        if a.only and not any(w in case.name for w in a.only):
            continue
        if a.sheet:
            tiles.append(tile(_thumb(render_image(case.source)), f"SOURCE {case.name}", 300))
        for target in case.targets:
            out, errs = evaluate(case, target, render_check=not a.no_render)
            total += 1
            if errs:
                bad += 1
                print(f"FAIL {case.name} @ {target}: " + "; ".join(errs[:5]) + (" ..." if len(errs) > 5 else ""))
            if a.sheet:
                tiles.append(tile(_thumb(render_image(out)), f"{target} {'FAIL' if errs else 'ok'}", 300))
    for name, src, tgt in malformed():
        if a.only and not any(w in name for w in a.only):
            continue
        total += 1
        try:
            out, _ = run(src, tgt)
            errs = hard_rule_errors(src, out, tgt)
            render_image(out)
        except Exception as ex:  # noqa: BLE001
            errs = [f"CRASH {type(ex).__name__}: {ex}"]
        if errs:
            bad += 1
            print(f"FAIL malformed/{name}: " + "; ".join(errs[:5]))
    print(f"\n{total - bad}/{total} passed in {time.time() - t0:.0f}s")
    if a.sheet and tiles:
        _save_sheet(tiles, a.sheet, Image)
    return 1 if bad else 0


def _thumb(im, side: int = 300):
    f = min(side / im.width, side * 1.3 / im.height)
    return im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))))


def _save_sheet(tiles, path, Image, cols: int = 5, cell: int = 300):
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    heights = [max(t.height for t in r) + 4 for r in rows]
    sheet = Image.new("RGB", (cols * (cell + 4), sum(heights)), (200, 200, 200))
    y = 0
    for r, h in zip(rows, heights):
        for i, t in enumerate(r):
            sheet.paste(t, (i * (cell + 4), y))
        y += h
    sheet.save(path)
    print("sheet:", path, sheet.size)


if __name__ == "__main__":
    sys.exit(main())
