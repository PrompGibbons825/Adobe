"""Format profiles: what a target size is, and where its platform covers it.

The engine only receives ``{width, height}``. Most targets are real platform
formats, though, and several of those have UI drawn over them or get cropped:

* a 9:16 story or reel has the account bar on top (~14%) and the reply box /
  caption at the bottom (~20%);
* YouTube channel art (2560x1440) is shown in full only on TVs: every device
  shows the centre 1546x423 strip;
* a YouTube thumbnail carries the duration badge in its bottom-right corner;
* X and LinkedIn headers have the profile picture over the bottom-left.

This module maps a size to a profile: an exact match in a table of sizes
(Adobe Express's own presets, Crello/VistaCreate sizes, IAB and Google ad
sizes, current social specs; ~150 sizes), else a near match within 2% on both sides, else
the aspect-ratio class. A profile has a safe rect and avoid rects (fractions of
the canvas). They are soft constraints: ``zone_penalty`` charges a plan for
headline/CTA/logo (and, less, other copy) painted inside an avoid rect, and
``safe_plans`` adds candidates laid out inside the safe rect only. General
platform knowledge, nothing tuned to a particular design.

Sources: context/research/05-formats-catalogue-and-design-rules.md (A2-A5, B4,
C1) and scripts/formats/express_canvas.json (Embed SDK CanvasTypeInfo).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .geom import Box, el_visual_box
from .items import Ctx

# Feature flags (module level so a merge can switch each part off on its own).
PROFILES_ON = os.environ.get("CHAMELEON_FORMAT_PROFILES", "1") != "0"  # avoid-zone scoring
SAFE_PLANS_ON = os.environ.get("CHAMELEON_SAFE_PLANS", "1") != "0"  # extra candidates inside the safe rect

Rect = Tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of the canvas

# ----------------------------------------------------------------------------
# the size table: (width, height, name, zones)

_EXPRESS_PX = [  # Adobe Express pixel presets (81 presets, 48 sizes)
    (1600, 213, "etsy-shop-banner"), (1920, 256, "reddit-cover"), (300, 60, "youtube-display-ads"),
    (2480, 520, "soundcloud-banner"), (1600, 400, "etsy-shop-cover"), (1000, 250, "google-classroom-banner"),
    (1536, 396, "linkedin-profile-cover"), (1500, 500, "twitter-header"), (600, 200, "email-header"),
    (2500, 843, "line-rich-menu-small"), (3000, 1055, "tumblr-banner"), (851, 315, "facebook-profile-cover"),
    (820, 312, "facebook-page-cover"), (1200, 480, "twitch-banner"), (1000, 500, "pinterest-post"),
    (1640, 856, "facebook-group-cover"), (1200, 628, "linkedin-blog-post"), (1280, 670, "note-header-image"),
    (1200, 630, "facebook-event-cover"), (1080, 602, "instagram-landscape-post"), (1200, 675, "twitter-post"),
    (2560, 1440, "youtube-banner"), (1280, 720, "youtube-thumbnail"), (2240, 1260, "blog-header"),
    (1920, 1080, "presentation"), (600, 400, "line-ad-small"), (2500, 1686, "line-rich-menu-large"),
    (1024, 768, "concept-map"), (1600, 1200, "facebook-shop-cover"), (1200, 1000, "banner-web"),
    (1050, 1020, "place-card"), (1080, 1080, "instagram-square-post"), (1040, 1040, "line-rich-message"),
    (750, 750, "album-cover"), (1920, 1920, "linkedin-post"), (500, 500, "logo"), (1200, 1200, "meme"),
    (600, 600, "pinterest-square"), (3000, 3000, "podcast-cover"), (800, 800, "youtube-profile-photo"),
    (1080, 1350, "instagram-portrait-post"), (800, 1200, "blog-graphic"), (600, 900, "pinterest-vertical"),
    (1410, 2250, "book-cover"), (1600, 2560, "ebook-cover"), (1080, 1920, "instagram-story"),
    (300, 600, "display-ad"), (800, 2000, "infographic"),
]

_EXPRESS_PHYSICAL = [  # inches (one in mm); Express stores print sizes physically
    (5.5, 2.0, "ticket"), (8.0, 3.75, "gift-certificate"), (72.0, 36.0, "banner-print"),
    (3.5, 2.0, "business-card"), (5.0, 3.0, "flash-card"), (3.375, 2.125, "id-card"), (6.0, 4.0, "postcard"),
    (4.0, 3.0, "name-tag"), (11.0, 8.5, "brochure"), (3.3, 3.3, "mug"), (14.0, 18.0, "tshirt"),
    (8.5, 11.0, "letter"), (5.0, 7.0, "invitation"), (210 / 25.4, 297 / 25.4, "leaflet-trifold-a4"),
    (4.0, 6.0, "label"), (11.0, 17.0, "poster"), (2.0, 6.0, "bookmark"),
]

_OTHER = [  # Crello/VistaCreate, IAB, Google Ads, platform specs (research 05, A3-A4)
    (728, 90, "leaderboard"), (1128, 191, "linkedin-company-cover"), (1590, 400, "vk-community-cover"),
    (576, 216, "coupon"), (537, 240, "vk-post-button"), (1200, 600, "crello-image"), (1024, 512, "twitter-post-2:1"),
    (560, 315, "blog-title"), (241, 142, "business-card-72dpi"), (432, 288, "gift-certificate-72dpi"),
    (1000, 700, "vk-post"), (419, 298, "postcard-a6-72dpi"), (792, 612, "certificate-72dpi"),
    (300, 250, "medium-rectangle"), (336, 280, "large-rectangle"), (940, 788, "facebook-post"),
    (1600, 1600, "album-cover-large"), (1296, 1728, "poster-us-72dpi"), (1190, 1683, "poster-a2-72dpi"),
    (595, 841, "a4-72dpi"), (360, 504, "flyer-5x7-72dpi"), (735, 1102, "pinterest-pin"), (540, 810, "tumblr-post"),
    (420, 654, "igtv-cover"), (600, 1200, "crello-graphic"), (1080, 2340, "snapchat-filter"),
    (160, 600, "wide-skyscraper"), (970, 250, "billboard"), (320, 50, "mobile-banner"), (300, 50, "smartphone-banner"),
    (970, 90, "super-leaderboard"), (300, 1050, "portrait-ad"), (120, 60, "button-ad"), (320, 480, "interstitial"),
    (480, 320, "interstitial-landscape"), (640, 1136, "mobile-interstitial"), (750, 1334, "mobile-interstitial-2"),
    (120, 20, "feature-phone-banner-s"), (168, 28, "feature-phone-banner-m"), (216, 36, "feature-phone-banner-l"),
    (200, 200, "small-square"), (240, 400, "vertical-rectangle"), (250, 250, "square-ad"),
    (250, 360, "triple-widescreen"), (580, 400, "netboard"), (120, 600, "skyscraper"), (468, 60, "banner"),
    (930, 180, "top-banner"), (980, 120, "panorama"), (320, 100, "large-mobile-banner"),
    (1200, 300, "google-logo-4:1"), (1080, 1440, "instagram-post-3:4"), (1080, 566, "instagram-landscape"),
    (1920, 1005, "facebook-event-cover-2"), (1600, 900, "x-post"), (800, 418, "x-card"), (1200, 627, "linkedin-link"),
    (1584, 396, "linkedin-personal-cover"), (1400, 425, "linkedin-banner-canva"), (720, 900, "linkedin-vertical-ad"),
    (1000, 1500, "pinterest-standard-pin"), (1000, 2100, "pinterest-long-pin"), (1920, 480, "twitch-channel-banner"),
    (320, 160, "twitch-panel"), (960, 540, "discord-banner"), (3360, 840, "etsy-full-banner"),
    (1200, 160, "etsy-mini-banner"), (720, 1280, "story-720p"), (1440, 2560, "story-1440p"),
    (1170, 2532, "phone-wallpaper"), (1290, 2796, "phone-wallpaper-max"), (1080, 2400, "android-wallpaper"),
]

# Platform overlays, as fractions of the canvas. Weight scales the penalty
# (channel art outside the strip is cropped on every phone and laptop: heavy).
_STORY = dict(safe=(0.04, 0.14, 0.96, 0.80), in_safe=True,
              avoid=(("story-top-ui", (0.0, 0.0, 1.0, 0.14), 1.0), ("story-bottom-ui", (0.0, 0.80, 1.0, 1.0), 1.0)))
# YouTube channel art: TVs show all of 2560x1440, desktops a 2560x423 band, tablets
# 1855x423 and phones 1546x423, all centred. Copy belongs in the band (outside it
# is cropped on every phone and laptop: heavy), ideally inside the tablet width.
_YA_Y0, _YA_Y1 = (1440 - 423) / 2 / 1440, 1 - (1440 - 423) / 2 / 1440
_YA_TAB, _YA_PHONE = (2560 - 1855) / 2 / 2560, (2560 - 1546) / 2 / 2560
_YT_ART = dict(safe=(_YA_TAB, _YA_Y0, 1 - _YA_TAB, _YA_Y1), in_safe=True, cls="banner",
               avoid=(("channel-art-top", (0.0, 0.0, 1.0, _YA_Y0), 3.0),
                      ("channel-art-bottom", (0.0, _YA_Y1, 1.0, 1.0), 3.0),
                      ("channel-art-phone-left", (0.0, _YA_Y0, _YA_PHONE, _YA_Y1), 0.5),
                      ("channel-art-phone-right", (1 - _YA_PHONE, _YA_Y0, 1.0, _YA_Y1), 0.5)))
_YT_THUMB = dict(avoid=(("yt-duration-badge", (0.80, 0.84, 1.0, 1.0), 1.0),))
_X_HEADER = dict(safe=(0.0, 0.12, 1.0, 0.88),
                 avoid=(("x-crop-top", (0.0, 0.0, 1.0, 0.12), 0.5), ("x-crop-bottom", (0.0, 0.88, 1.0, 1.0), 0.5),
                        ("x-avatar", (0.0, 0.60, 0.25, 1.0), 1.0)))
_LI_COVER = dict(avoid=(("li-avatar", (0.0, 0.55, 0.25, 1.0), 1.0),))
_LI_COMPANY = dict(avoid=(("li-company-logo", (0.0, 0.50, 0.20, 1.0), 1.0),))
_PHONE_WALL = dict(safe=(0.0, 0.25, 1.0, 0.85), in_safe=True,
                   avoid=(("lockscreen-clock", (0.0, 0.0, 1.0, 0.25), 1.0), ("home-dock", (0.0, 0.85, 1.0, 1.0), 1.0)))

_ZONES: Dict[Tuple[int, int], dict] = {
    (1080, 1920): _STORY, (720, 1280): _STORY, (1440, 2560): _STORY, (1080, 2340): _STORY,
    (2560, 1440): _YT_ART, (1280, 720): _YT_THUMB, (1500, 500): _X_HEADER,
    (1584, 396): _LI_COVER, (1536, 396): _LI_COVER, (1400, 425): _LI_COVER, (1128, 191): _LI_COMPANY,
    (1170, 2532): _PHONE_WALL, (1290, 2796): _PHONE_WALL, (1080, 2400): _PHONE_WALL,
}


def _table() -> Dict[Tuple[int, int], str]:
    out: Dict[Tuple[int, int], str] = {}
    for w, h, n in _EXPRESS_PX + _OTHER:
        out.setdefault((int(w), int(h)), n)
    for w, h, n in _EXPRESS_PHYSICAL:  # Crello's 72 dpi and print's 300 dpi
        for dpi in (72, 300):
            out.setdefault((round(w * dpi), round(h * dpi)), f"{n}@{dpi}dpi")
    return out


SIZES: Dict[Tuple[int, int], str] = _table()

# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    name: str
    match: str  # exact | near | class
    cls: str  # aspect class used for layout (channel art lays out as a banner)
    tiny: bool  # short side < 150 px: absolute floors dominate
    micro: bool  # height <= 60 px: one line of content at most
    safe: Rect = (0.0, 0.0, 1.0, 1.0)
    avoid: Tuple[Tuple[str, Rect, float], ...] = ()
    in_safe: bool = False  # worth generating candidates inside the safe rect

    def safe_box(self, W: float, H: float) -> Box:
        x0, y0, x1, y1 = self.safe
        return Box(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H)

    def avoid_boxes(self, W: float, H: float) -> List[Tuple[str, Box, float]]:
        return [(n, Box(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H), wt) for n, (x0, y0, x1, y1), wt in self.avoid]


def aspect_class(ar: float) -> str:
    """Research 05, A5 (AR = W/H)."""
    if ar >= 5.0:
        return "strip"
    if ar >= 2.4:
        return "banner"
    if ar >= 1.4:
        return "landscape"
    if ar >= 1.1:
        return "near-square"
    if ar > 0.9:
        return "square"
    if ar >= 0.7:
        return "portrait"
    if ar >= 0.5:
        return "tall"
    if ar >= 0.3:
        return "very-tall"
    return "skyscraper"


WIDE_CLASSES = ("strip", "banner", "landscape")
TALL_CLASSES = ("tall", "very-tall", "skyscraper")


@lru_cache(maxsize=512)
def profile_for(W: float, H: float) -> Profile:
    W, H = float(W), float(H)
    tiny, micro = min(W, H) < 150, H <= 60
    key = (int(round(W)), int(round(H)))
    name, match = SIZES.get(key), "exact"
    if name is None:
        best = None
        for (w, h), n in SIZES.items():
            dw, dh = abs(W / w - 1.0), abs(H / h - 1.0)
            if dw <= 0.02 and dh <= 0.02 and (best is None or dw + dh < best[0]):
                best, key = (dw + dh, n), (w, h)
        if best is not None:
            name, match = best[1], "near"
    if name is None:
        ar = W / max(H, 1e-6)
        if abs(ar / (9.0 / 16.0) - 1.0) <= 0.02 and H >= 640:  # any vertical-video size gets story UI
            name, match, key = "class:9:16", "class", (1080, 1920)
        else:
            return Profile(name="class:" + aspect_class(ar), match="class", cls=aspect_class(ar), tiny=tiny,
                           micro=micro)
    z = _ZONES.get(key, {})
    return Profile(name=name, match=match, cls=z.get("cls", aspect_class(W / max(H, 1e-6))), tiny=tiny,
                   micro=micro, safe=z.get("safe", (0.0, 0.0, 1.0, 1.0)), avoid=tuple(z.get("avoid", ())),
                   in_safe=bool(z.get("in_safe", False)))


# ----------------------------------------------------------------------------
# soft constraints on candidate plans

# How much each role minds sitting under platform UI (1.0 = headline/CTA/logo).
ROLE_WEIGHT = {"headline": 1.0, "cta": 1.0, "logo": 1.0, "subhead": 0.6, "body": 0.5, "text": 0.5,
               "badge": 0.5, "ribbon": 0.5, "group": 0.5, "legal": 0.25}
ZONE_K = 40.0  # score points for a critical element entirely inside a weight-1 avoid rect


def _element_weight(item, e: dict) -> float:
    if item.role == "panel" or item.kind == "composite":
        if e.get("type") != "text":
            return 0.0  # the panel or photo itself may run under UI; its copy may not
        return 1.0 if item.core else 0.6
    if item.role in ("logo", "cta"):
        return 1.0
    if e.get("type") == "text":
        return ROLE_WEIGHT.get(item.role, 0.5)
    if item.kind == "group":
        return ROLE_WEIGHT.get(item.role, 0.5) * 0.5  # a chip's shape counts, a little
    return 0.0  # hero photos and art may sit under UI


def zone_hits(plan, W: float, H: float, prof: Profile):
    """(penalty, [(element id, zone name, fraction inside, weight)]) for one plan."""
    zones = prof.avoid_boxes(W, H)
    if not zones:
        return 0.0, []
    pen, hits = 0.0, []
    for r, x, y in plan.placed + plan.floaters:
        for e in r.emit(x, y):
            wt = _element_weight(r.item, e)
            if wt <= 0:
                continue
            b = el_visual_box(e)
            if b.area <= 0:
                continue
            frac = 0.0
            for name, z, zw in zones:
                f = b.inter(z) / b.area
                if f > 0.02:
                    frac += f * zw
                    hits.append((e["id"], name, round(f, 3), wt))
            pen += ZONE_K * wt * min(3.0, frac)
    return pen, hits


def score_zones(plans, d, ctx) -> Optional[Profile]:
    """Charge every plan for critical copy under platform UI. Safe-rect plans
    are also re-scored for coverage and format prior against the safe rect,
    the canvas they were actually laid out in."""
    if not PROFILES_ON or not plans:
        return None
    from .score import _prior

    prof = profile_for(ctx.W, ctx.H)
    sb = prof.safe_box(ctx.W, ctx.H)
    for p in plans:
        pen, _ = zone_hits(p, ctx.W, ctx.H, prof)
        if getattr(p, "safe_zone", False):
            old = p.breakdown.get("prior", 0.0) + p.breakdown.get("coverage", 0.0)
            p.breakdown["prior"] = _prior(p.name, sb.w / max(sb.h, 1e-6), d.W / d.H)
            area = sum(r.w * r.h for r, _, _ in p.placed + p.floaters)
            src_cov = sum(it.src_box.area for it in d.items if it.role != "missing") / max(1.0, d.W * d.H)
            want = min(0.55, src_cov)
            cov = area / max(1.0, sb.area)
            p.breakdown["coverage"] = -40.0 * max(0.0, want * 0.7 - cov) / max(1e-6, want)
            p.score += p.breakdown["prior"] + p.breakdown["coverage"] - old
        if pen:
            p.breakdown["zones"] = -pen
            p.score -= pen
    return prof


# ----------------------------------------------------------------------------
# candidates laid out inside the safe rect


@dataclass
class ZoneCtx(Ctx):
    """A layout context the size of the safe rect whose legibility floors stay
    those of the full canvas (the judge still sees the whole canvas)."""

    FW: float = 0.0
    FH: float = 0.0

    @property
    def base_fs(self) -> float:
        return max(10.0, 0.0115 * max(self.FW, self.FH), 0.016 * min(self.FW, self.FH))

    @property
    def judge_floor(self) -> int:
        return max(10, math.ceil(10.0 * max(self.FW, self.FH) / 1024.0))


def zone_ctx(ctx, box: Box) -> ZoneCtx:
    return ZoneCtx(W=box.w, H=box.h, W0=ctx.W0, H0=ctx.H0, FW=ctx.W, FH=ctx.H)


def safe_plans(d, ctx, generate) -> list:
    """For formats whose platform covers part of the canvas (stories, channel
    art, phone wallpapers), also lay the design out as if the safe rect were
    the canvas, then move it into place. The full-canvas candidates stay;
    scoring decides."""
    if not (PROFILES_ON and SAFE_PLANS_ON):
        return []
    prof = profile_for(ctx.W, ctx.H)
    if not prof.in_safe:
        return []
    sb = prof.safe_box(ctx.W, ctx.H)
    if sb.w < 40 or sb.h < 40:
        return []
    zc = zone_ctx(ctx, sb)
    out = []
    for p in generate(d, zc):
        if p.name == "empty":
            continue
        p.placed = [(r, x + sb.x, y + sb.y) for r, x, y in p.placed]
        p.floaters = [(r, x + sb.x, y + sb.y) for r, x, y in p.floaters]
        p.zone = p.zone.moved(sb.x, sb.y)
        p.name = p.name + "@safe"
        p.safe_zone = True  # type: ignore[attr-defined]
        out.append(p)
    return out
