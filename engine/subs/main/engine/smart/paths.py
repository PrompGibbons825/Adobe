"""Where the brand kit lives, both locally and inside the scoring container."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

_REPO = Path(__file__).resolve().parents[2]


def _candidates(sub: str) -> List[Path]:
    env = os.environ.get("CHAMELEON_BRAND_KIT")
    out = []
    if env:
        out.append(Path(env) / sub)
    out += [
        Path("brand_kit") / sub,  # cwd = /app in Docker, repo root locally
        _REPO / "brand_kit" / sub,
        Path("/app/brand_kit") / sub,
    ]
    return out


@lru_cache(maxsize=None)
def assets_dirs() -> tuple:
    seen, dirs = set(), []
    for p in _candidates("assets") + [Path("input") / "assets", Path("assets")]:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp.is_dir() and rp not in seen:
            seen.add(rp)
            dirs.append(rp)
    return tuple(dirs)


@lru_cache(maxsize=None)
def fonts_dir() -> Optional[Path]:
    for p in _candidates("fonts"):
        if p.is_dir() and any(p.glob("*.ttf")) or (p.is_dir() and any(p.glob("*.otf"))):
            return p.resolve()
    return None


def find_asset(asset_id: Optional[str]) -> Optional[Path]:
    if not asset_id:
        return None
    for d in assets_dirs():
        cand = d / asset_id
        if cand.is_file():
            return cand
    return None
