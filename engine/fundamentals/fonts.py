"""Brand-kit font loading and face lookup, exactly as the official renderer does it.

Mirrors ``renderer/converter/brand_fonts.py`` (file name -> family/weight/style) and
``cr_renderer.fonts.FontManager.lookup`` / ``normalize_family`` (fallback chain: the family, then
"Montserrat", then the first family loaded; a missing weight/style falls back to the family's first
file). Pure Python + skia; imports nothing from ``engine/``, ``renderer/`` or ``cr_renderer``.
``tests/conformance`` asserts it returns the same bytes as the renderer's own FontManager.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FONT_WEIGHTS = {"thin", "light", "extralight", "regular", "medium", "semibold", "bold", "extrabold", "black"}
FONT_STYLES = {"regular", "bold", "italic", "bolditalic"}
BOLD_WEIGHTS = {"bold", "semibold", "extrabold", "black"}  # converter._BOLD_WEIGHTS

_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")
_ITALIC = re.compile("italic", re.IGNORECASE)
_FONT_MAP = {
    "Arkana Script": "Arkana", "Blogger": "Blogger Sans", "Delius Swash": "Delius Swash Caps",
    "Elsie Swash": "Elsie Swash Caps", "Gluk Glametrix": "Gluk Foglihtenno06",
    "Gluk Znikomitno25": "Gluk Foglihtenno06", "Im Fell": "Im Fell Dw Pica Sc",
    "Medieval Sharp": "Medievalsharp", "Playlist Caps": "Playlist", "Rissa Typeface": "Rissatypeface",
    "Selima": "Selima Script", "Six": "Six Caps", "V T323": "Vt323", "Different Summer": "Montserrat",
    "Dukomdesign Constantine": "Montserrat", "Sunday": "Montserrat",
}


def normalize_family(name: str) -> str:
    name = name.replace("_", " ").title()
    for suf in (" Bold", " Regular", " Light", " Italic", " Medium"):
        name = name.replace(suf, "")
    return _FONT_MAP.get(name, name)


def _weight_from_name(name: str, default: str) -> str:
    key = name.replace("_", " ").lower().split(" ")[-1]
    return key if key in FONT_WEIGHTS else default


def _style_from_name(name: str, default: str) -> str:
    key = name.replace("_", " ").lower().split(" ")[-1]
    return key if key in FONT_STYLES else default


def find_fonts_dir(hint: Optional[str] = None) -> Path:
    """brand_kit/fonts: ``hint``, $CHAMELEON_FONTS, $CHAMELEON_BRAND_KIT/fonts, or the first
    ``brand_kit/fonts`` found walking up from the cwd and from this file (works in /app too)."""
    cands = [hint, os.environ.get("CHAMELEON_FONTS")]
    kit = os.environ.get("CHAMELEON_BRAND_KIT")
    if kit:
        cands.append(str(Path(kit) / "fonts"))
    for c in cands:
        if c and Path(c).is_dir():
            return Path(c)
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for d in (start, *start.parents):
            p = d / "brand_kit" / "fonts"
            if p.is_dir():
                return p
    raise FileNotFoundError("brand_kit/fonts not found (set CHAMELEON_FONTS)")


class FontBook:
    """The renderer's font table for one fonts folder."""

    def __init__(self, fonts_dir: Optional[str] = None):
        self.dir = find_fonts_dir(fonts_dir)
        self.families: Dict[str, List[dict]] = {}
        for path in sorted(self.dir.glob("*")):  # sorted: decides family[0] and the first family
            if path.suffix.lower() not in (".ttf", ".otf"):
                continue
            prefix, _, suffix = path.stem.partition("-")
            family = _CAMEL.sub(" ", prefix).strip()
            suffix = suffix or "Regular"
            italic = bool(_ITALIC.search(suffix))
            wt = _ITALIC.sub("", suffix).strip().lower() or "regular"
            weight = wt if wt in FONT_WEIGHTS else "regular"
            style = ("bolditalic" if weight == "bold" else "italic") if italic else ("bold" if weight == "bold" else "regular")
            self.families.setdefault(normalize_family(family), []).append(
                {"fontFamily": family, "fontWeight": weight, "fontStyle": style, "path": str(path)})
        if not self.families:
            raise FileNotFoundError(f"no .ttf/.otf in {self.dir}")

    def resolve(self, family: str, weight: str = "regular", style: str = "regular") -> Tuple[str, str, bool]:
        """(font file path, resolved family name, exact) for a lookup, following FontManager.lookup.
        ``exact`` is False when the family or the weight/style fell back."""
        assert weight in FONT_WEIGHTS and style in FONT_STYLES
        fam = self.families.get(normalize_family(family))
        exact = fam is not None
        if fam is None:
            fam = self.families.get(normalize_family("Montserrat")) or next(iter(self.families.values()))
        weight = _weight_from_name(family, weight)
        style = _style_from_name(family, style)
        for f in fam:
            if f["fontWeight"] == weight and f["fontStyle"] == style:
                return f["path"], f["fontFamily"], exact
        return fam[0]["path"], fam[0]["fontFamily"], False

    def typeface(self, family: str, weight: str = "regular", style: str = "regular"):
        return _typeface(self.resolve(family, weight, style)[0])

    def font(self, family: str, weight: str, style: str, size: float):
        return _font(self.resolve(family, weight, style)[0], float(size))


@lru_cache(maxsize=64)
def _typeface(path: str):
    import skia  # type: ignore

    return skia.Typeface.MakeFromData(skia.Data.MakeFromFileName(path))


@lru_cache(maxsize=4096)
def _font(path: str, size: float):
    import skia  # type: ignore

    return skia.Font(_typeface(path), size)


@lru_cache(maxsize=8)
def fontbook(fonts_dir: Optional[str] = None) -> FontBook:
    return FontBook(fonts_dir)


def normalize_weight(weight) -> str:
    """converter._normalize_weight."""
    if not weight:
        return "regular"
    weight = str(weight).lower()
    return weight if weight in FONT_WEIGHTS else "regular"


def is_italic(value) -> bool:
    """converter._is_italic."""
    if isinstance(value, bool):
        return value
    return value == "italic"


def face_style(bold: bool, italic: bool) -> str:
    if bold and italic:
        return "bolditalic"
    if bold:
        return "bold"
    if italic:
        return "italic"
    return "regular"
