"""What a judge sees: the official render, downsampled, under a name that gives nothing away.

    from tools.judge.views import views, blind_ask
    v = views(design, "fam03@story#e0", cache_dir)        # {512: Path, 1024: Path}
    ask = blind_ask(ref_png, a_png, b_png, ask_dir, rng)  # {"id", "ref", "A", "B"} with random names

`views` renders the design exactly as engine/main.py does (white canvas, long
side at most 2048) and resamples it with Lanczos to each view size on the long
side; the native render is not kept. `blind_ask` hard-links (or copies, across
devices) three views into an ask folder as `<random>_ref.png`, `<random>_A.png`
and `<random>_B.png`, so nothing in the folder names the candidate, the engine
version or the order. Ported from ws/judge-calibration (tools/judge/corpus.py
`_views` and the ask linking in `cmd_asks`, 10-06).
"""

from __future__ import annotations

import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional

from PIL import Image

VIEWS = (512, 1024)


def downsample(im: Image.Image, long_side: int) -> Image.Image:
    """Lanczos to `long_side` on the long side; never upsamples."""
    im = im.convert("RGB")
    f = min(1.0, long_side / max(im.size))
    if f >= 1.0:
        return im
    return im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)


def views(design: dict, name: str, out_dir: Path, sizes: Iterable[int] = VIEWS) -> Dict[int, Path]:
    """`<out_dir>/<name>@<size>.png` for every size (cached: an existing set is reused)."""
    from tools.judge.terms import render_png

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {v: out_dir / f"{name}@{v}.png" for v in sizes}
    if all(p.exists() for p in out.values()):
        return out
    fd, tmp = tempfile.mkstemp(suffix=".png", dir=out_dir)
    os.close(fd)
    try:
        render_png(design, Path(tmp))
        with Image.open(tmp) as im:
            native = im.convert("RGB")
        for v, p in out.items():
            downsample(native, v).save(p, optimize=True)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return out


def blind_name(rng: random.Random, n: int = 8) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def _link(src: Path, dest: Path) -> None:
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)  # no extra disk
    except OSError:
        shutil.copyfile(src, dest)


def blind_ask(ref: Path, a: Path, b: Path, ask_dir: Path, rng: random.Random,
              taken: Optional[set] = None) -> Dict[str, object]:
    """Place one ask's three images under a fresh random id. The caller keeps the
    id -> candidates mapping somewhere the judge cannot read."""
    ask_dir = Path(ask_dir)
    ask_dir.mkdir(parents=True, exist_ok=True)
    taken = taken if taken is not None else set()
    while True:
        aid = blind_name(rng)
        if aid not in taken and not (ask_dir / f"{aid}_ref.png").exists():
            break
    taken.add(aid)
    res: Dict[str, object] = {"id": aid}
    for tag, src in (("ref", ref), ("A", a), ("B", b)):
        dest = ask_dir / f"{aid}_{tag}.png"
        _link(Path(src), dest)
        res[tag] = dest
    return res
