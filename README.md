# CHAMELEON: a design re-layout engine

`resize(source, target_canvas)` takes a finished design (text, image and shape
elements with no semantic labels) and rebuilds it for a new canvas. It moves the
furniture instead of stretching the picture. The output has the same element ids,
is valid against the v0.9 schema, keeps everything on the canvas, keeps image
proportions, leaves copy and font families alone, and never regenerates or alters
asset pixels.

```
input/*.json  ->  engine/resize.py  ->  output/<task>_output.json + .png
```

## How to run

```bash
pip install -r requirements.txt          # Python 3.10+ (the image uses 3.11)
python -m engine.main                    # every input/*.json -> output/
```

As scored (offline, in the container):

```bash
docker build --platform linux/amd64 -t my-chameleon-engine .
docker run --rm --network none \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output my-chameleon-engine
```

No network at run time. Besides what the starter installs (skia and numpy through
cr-renderer, and Pillow), the engine uses OpenCV (`opencv-python-headless`) and one
face-detection and photo-analysis models committed to the repo (YuNet, 230 KB, MIT;
U2-Netp saliency, ~4.4 MB, Apache-2.0; PP-OCRv4 text detector only, ~4.6 MB,
Apache-2.0; all in `engine/smart/models/`); nothing is downloaded at run time. A task takes
about 0.3 s at the median and under 2 s at the 95th percentile; the first task
in a run also pays a few seconds of one-time font and library loading.

## Approach

The engine works the way a designer resizing by hand would. It works out what each
piece is and what belongs together, considers several ways the design could be
rebuilt for the new shape, sizes each one as large as it will legibly go, and keeps
the one that reads best. When the shape does not change, it changes nothing.

**1. Understand the design** (`engine/smart/analyze.py`, `structure.py`)

The input carries only ids, types and geometry, so roles are inferred:

| Inferred role | Signal |
|---|---|
| background | a full-canvas image or shape under all copy; or, for a canvas split into flat panels, the panel under the headline |
| decoration | free shapes with no copy: translucent, elliptical, or bleeding off the canvas |
| headline / subhead / body / legal | prominence (size × weight), position, opacity, length; a small line is legal only if very small, faint or set at the very bottom |
| CTA / badge / ribbon | a filled shape carrying a short label (explicit `group_id` or detected); the largest is the CTA, an alike button beside it is a CTA too, a rotated one is a ribbon |
| logo / hero / watermark | image geometry and stacking (the asset name is only a weak hint); the hero is the largest solid picture, or a design's only picture however small (the product it features); in a column it may grow into room no copy needed until it covers a fifth of the canvas; a flat block carrying copy is a panel, not a logo. With no logo artwork, a short styled text (capitals, tracking or bold) that leads the reading order and stands apart from the headline is the brand set in type |
| panel | a shape or flat image that frames copy (a dark band, a card); its padding is reserved in the layout (its members are laid out as one padded block) and it is redrawn around them; a layout that would split its members apart is not offered |
| composite | a photo with captions or stickers on it, or a row of product shots: one piece. A sticker is a round button-like group, or one stuck on a picture; a real button outranks it for CTA. Stickers and copy on the art keep their place on it but keep to the design's type scale when the picture grows, so a caption never outgrows the copy around it |
| lockup | texts set as one piece of type: a drop shadow, stacked display type ("UP TO / 50% / OFF"), a price and its cents. They overlap on purpose, or sit closer than a word space (stacked lines only in one typeface: a subhead right under its headline is copy that follows it), and keep their arrangement |
| corner items | small badges, ribbons and watermarks pinned to a canvas corner, and copy turned on its side |
| missing asset | a small image whose file is not in the brand kit (never drawn as a grey box) |

A recursive XY-cut gives the reading structure: what sits beside what, and the
order a single column should present it in. It cuts along the widest
whitespace first, so a sliver of space under a headline does not split it from
its subhead when gutters set logo, copy and button apart.

**2. Measure exactly like the renderer** (`textfit.py`, `assets.py`)

Text is wrapped with the renderer's own greedy rule at real skia glyph advances
(a test checks the fast path against the renderer's wrap), at integer sizes,
top-aligned. Footprints are where the glyphs are painted: letter spacing makes a
drawn line wider than its wrap, and a rotated text turns about its box centre, as
the renderer turns it. Letter spacing is kept in ems, so it scales with the type.
For images, the alpha channel gives where the visible pixels sit.

**3. Generate candidate layouts** (`compose.py`, `layout.py`)

| Archetype | When | Shape |
|---|---|---|
| scaled | the target has the source's aspect ratio (within 3%) | the design itself, uniformly scaled, with its own line breaks |
| adapt | nearly the same aspect | the source arrangement, edges kept as edges |
| stack | portrait, square, mild landscape | one column in reading order, rows near their source-relative height |
| split | square to wide | hero on one side, copy stacked on the other |
| regions | a photo filling one side edge to edge beside a flat panel carrying the copy (`regions.py`) | the two still tile the canvas, side by side on a wide one, stacked (photo on top) on a tall one; the panel fills its region, the copy is one padded column in it, the photo fills its region by cropping (its own proportions, the overflow past the canvas edges or under the opaque panel, the window kept on its subject and faces) |
| row | banners and strips | logo, hero, copy, CTA on one line; a row of three or more parts, or one that ends in its CTA, is set like a banner (the brand holds one end, the CTA the other); any other pair stays a centred group; the hero may fill the strip's height |

Each is solved by a small flexbox-like engine for the largest scale that fits
(within a range: type grows by about √area, a column moved to a taller canvas a
little further, and no design grows past the point where its headline reaches the
canvas's display size, so nothing outgrows the headline; a design of copy alone on
a canvas taller than wide may go 1.4× past it, as a typographic poster does),
and **nothing is left out** to make it fit. When everything does not fit, the
layout climbs a demotion ladder instead, stopping at the first rung where every
element is placed:
1. type relaxes further toward the legibility floor (stickers and captions stay on
   their art at the floor rather than leaving it);
2. small secondary items (badges, ribbon, small print, watermark) move out of the
   column into the margins and corners, a set of badges queued in one line along
   the edge; a corner item with no room in its corner joins the column instead;
3. on strips the copy re-stacks into two or three columns side by side, and small
   print runs as one line along the foot of the strip;
4. last, secondary copy (small print, badges, a subhead) may go under the usual
   floor down to an explicit minimum (70% of it, never under 7 px); headline and
   CTA stay at the usual floor.
Only content that cannot fit even then (a crowded poster on a 320×50) still has
low-importance items left out, least important first, as a repair of last resort.

When a design reflows, a short headline is also offered as **display type**, the
way Adobe's reference output sets it: 1.35× and 1.7× larger than the rest of the
design, up to the canvas's display size (15% of the short side on posts and
stories, up to 42% of a strip's height), at about two words a line and at most
three lines. A longer headline is offered the first step only, and only into
the lines it already takes. A headline or subhead never ends on a lone word after a
fuller line if setting it up to 15% smaller avoids that (the renderer's wrap is
greedy, so only a change of size moves the break). Two-line headlines break after punctuation or where
a style run changes colour ("Chase / the Light") when that is nearly as even as
the most balanced break. On a canvas much wider than the source, copy is also
offered a **longer line** (1.8× its source measure, never past a 70-character
reading line), so a column of copy can trade lines for size. Columns follow a
typographic **rhythm**:
- a gap is at most the smaller of the two rows it separates;
- gaps give way before a hero shrinks, but never below half the smaller row
  (unless the source set them closer), and a logo keeps clear space of half
  its height;
- leftover height is shared between the margins and gaps in proportion to the
  room the source left there (its top and bottom margins moved half way to
  even, so a design that left its bottom third empty no longer does so on a
  story); rows set tight (a subhead tucked under its headline) are bonded and
  take none of it;
- across, the column is one block centred where the source's content was
  centred, with each row's text alignment kept inside it.
Margins follow the source's breathing room but are never thinner than a
banner's (about 5% of the canvas's length, at most 0.3 of its short side).
On a story or reel (1.6 or more times taller than wide, not an ad unit) copy,
logo, button and corner items also keep clear of the very edges the platform UI
covers (the top 5% and the bottom 6%; a deeper band made stories look small in a
blind comparison). Backgrounds and decoration still run under it. On a roomy
canvas (short side 400 px or more) a hero is held at no less than about half of
its source share of the canvas and pays below 70%, and any other picture that
covered 8% of the source pays below 60% (a cut-out, which cannot be cropped, is
asked for no more than 70% of the largest share it could show whole): the copy
gives way first, so a product never shrinks to a thumbnail. Texts that shared a font
size, family and role in the source share one size in every layout, the
smallest any of them was set at (Adobe's copyfit rule: each old size maps to
one new size). The headline is exempt, since display type is sized on its own.
Headline and CTA are never dropped. Copy never goes below a legibility floor
(cap height of about 7 px when the canvas is viewed at 1024 px), except secondary
copy on the ladder's last rung. When a label
is held up at the floor, its button gets a tighter pill rather than a bigger one.
On a strip (3:1 or wider) the call to action is the banner's second focal point,
in a column of its own, so its label is set at no less than 45% of the strip's
display size (about 17 px on a 728×90, as Adobe's reference leaderboard sets its
button); on a tall ad unit (half-page, skyscraper) at no less than 30% of it. That
is done only when it is free: on one line (a wrapped pill reads as a blob), no
larger than copy that outranked it, no wider than the headline, and at no more
than 5% of the rest of the layout's scale. The bolder button is scored at the size
it would have had, so it never decides what else stays;
a round button or sticker stays round, and a sticker (a button-like group on a
picture, not a text lockup) whose label would have to be blown up past twice its
art's scale to stay legible is set at the floor on the art instead (when it would
otherwise be left out). An ellipse that bleeds off the
canvas is a corner ornament, never a panel.

**4. Score like the rubric** (`score.py`, `pixels.py`)

Visible importance, copy size against the best any candidate achieved, small or
crowded type, collisions, coverage, balance, hierarchy (a headline that stands out
up to 1.6× more than in the source earns credit; flattening or overshooting costs
it; copy set level with copy it was 1.5× smaller than, as the legibility floor does
on tiny formats, costs more than leaving it out), copy density (copy covering
more than the larger of a quarter of the canvas and 1.5× the source's share, off
strips, reads as a wall of text), headline line breaks (more lines than display
type takes, about two words a line, counts as short lines), a prior for the archetype that suits the format, and a charge for secondary
items that take far more of the canvas than they had in the source. On a roomy canvas a layout also earns
credit for filling it, up to 1.1× the source's coverage (at most 55%), and pays when its headline or hero
falls below 70% of its source share of the canvas, or a large picture below 60%. A layout that is only
the source scaled uniformly into the middle of a canvas of another shape (aspect changed 1.5× or more,
empty bands either side) pays heavily, as VLM rubrics mark "trivial scaling" down. Corner items keep
a clear gap of at least 2% of the short side from the content, since judges misread near-touching
shapes as collisions. The best eight are then re-ranked by the pixels under
their copy (contrast and texture, rendered through the real renderer), relative to
the source. Those eight are shared out first: identical layouts count once and no
archetype takes more than two places, so variants of one plan cannot crowd out a
different one.

**Photos** (`vision.py`). A background photo is looked at once, small: an
importance map in the manner of smartcrop.js (edges, skin tone, saturation) plus
spectral-residual saliency, and faces from YuNet (OpenCV's Haar cascade if it
cannot load), padded to the head (never cut) and to head and shoulders. Every
candidate pays for copy, logo or button set over a face or on the busy part of the
picture (an item left out pays as much, so copy moves off the subject but is never
dropped for it), and over a photo with a subject the column is also offered at
either side, or at the top or bottom of a tall canvas. A canvas-covering photo is
then cropped by bleed: drawn at its own aspect, scaled to cover the canvas, and slid
so no head is cut, what the source showed stays in view, and the subject sits on
the third line opposite the copy. A flat or diffuse picture (a gradient, a
texture) keeps the plain centre crop, so designs on one do not change.

**5. Check, emit and enforce the rules** (`__init__.py`)

The winning plan must pass two hard checks on its finished output: no copy drawn
over copy that the source did not draw that way, and no glyph past the canvas
edge. Then:

- backgrounds are redrawn full-bleed (a photo with a subject by its focal crop),
  panels around their members, decoration in its corner;
- `finalize()` enforces every hard rule on whatever was produced: same ids, same
  order, immutable fields (copy, fonts, colours, assets) from the source, image
  ratios, every painted box (rotation included) on the canvas except a photo that
  covers the whole canvas (its crop) or fills a photo region (an opaque picture
  with at least 60% of its box on the canvas), integer font sizes,
  and only values the renderer can draw.

If lockups make every layout fail, the design is planned again with their parts
as separate copy. If the layout engine still finds nothing (impossible content,
such as 500 characters in 320×50), `resize()` falls back to a distortion-free
fit-and-centre layout, so a run never crashes.

### Notes on specific decisions

- **Nothing is hidden** (a panel is redrawn hugging its members when there is no
  room for its padding). In the last-resort repair, a left-out element stays in
  the JSON with its id and `opacity: 0`, stacked under everything. The reference renderer ignores `opacity` on text, so dropped
  copy is also set at 1 px and tucked under the largest opaque element; a rendered
  check confirms it paints nothing.
- **Missing assets:** a small image whose file is not in the brand kit is dropped
  instead of rendering as a grey placeholder. A large one is laid out normally,
  in case the scorer's renderer has the file.
- **Bleed:** decoration that bled off the source canvas is shrunk to its visible
  part and kept inside, because "nothing off-canvas" is a structural check. A
  rotated element counts both its box and its turned box.
- Nothing is tuned to the three samples. Development used them, eight more source
  designs (`tools/make_cases.py`), a rebuild of the brief's reference post
  (`tools/cases/reference_post.json`, to lay the engine's story, poster and
  leaderboard beside Adobe's own), 24 synthetic finicky designs (`tests/`), 64 real
  Crello templates that designers re-made at other sizes, designs generated from a
  small design grammar (`tools/designs.py`), and about 20 real platform formats
  from 320×50 to A4.

## Testing (dev only, not part of the image)

```bash
python -m unittest discover -s tests -t .        # everything (a few minutes)
python -m tests.run [names] --sheet out.png      # the finicky suite as one report + contact sheet
python -m tools.stress --n 300 --seed 1          # random and degenerate designs, every hard rule
python -m tools.sheet input/*.json --catalogue -o sheet.png   # designs across ~20 formats
python -m tools.audit --n 40 --seed 1 --sheet audit.png        # layout quality on unseen designs
```

- `tests/finicky.py`: designs that each probe one finicky part: unbreakable words,
  newlines, letter spacing, non-ASCII copy, empty texts, drop shadows, stacked
  type, price lockups, rotation, bleed, no background, a scrim, a split background,
  two CTAs, a badge grid, product cards, a wall of copy, tiny and wide sources,
  missing assets, hairlines and the Level 3 load. Each run is checked for the hard
  rules, the engine's promises, what must stay visible, what must stay together,
  ghost text and lost contrast. Six malformed inputs check robustness.
- `tests/test_engine.py`: the same under unittest, plus renderer parity of text
  wrapping, same-shape and uniform-scale resizes (the design must be unchanged),
  determinism and the samples.
- `tools/rules.py` is the one definition of a valid output, shared by the tests
  and the stress tool.
- `tools/designs.py` writes plausible source designs the engine was never tuned
  on: six arrangements (centred column, left column beside a product, product
  beside copy, banner row, card, band), copy of any length from four domains,
  light and dark themes, optional kicker, subhead, legal, sticker and decoration.
  `tools/audit.py` resizes them to every format and flags what a reviewer would
  notice, relative to the source: empty bands, lopsided or sparse content,
  cramped edges, lost core items, flattened hierarchy, a product reduced to a
  speck. It prints the counts and a contact sheet of the worst runs.
  `tests/test_engine.py` runs a dozen of them (on a seed the audit does not use)
  through the hard rules.

- `tools/unseen.py` is the unseen suite: eight families of medium-to-hard designs
  built on real Crello photos, cut-outs and the Solstice kit (a person on a
  full-bleed photo, half photo / half colour panel, products with badges and a
  price, a photo collage, an event poster, a sale promo with sticker and ribbon,
  a glass card over a photo, a Level-3-dense poster with a missing asset), plus
  real Crello pairs. Every tile must pass six named checks: hard rules, nothing
  hidden, legible type, copy contrast on photos, no cut or covered face, no
  overlap or flattened hierarchy. `--split dev` is for development; the holdout
  seeds and pairs are run only to check.

## Prior art this engine builds on

- No-drop demotion instead of hiding, and growth into blank space: ideas from
  Jeffrey Yu's independent engine for this challenge.
- Photo importance map in the manner of smartcrop.js (MIT), spectral-residual
  saliency (Hou and Zhang, 2007), YuNet face detection (OpenCV Zoo, MIT).
- One new size per old size: Adobe's copyfit patent (US11436404).
- Penalties for trivial scaling and near-touching shapes: published VLM layout
  rubrics (One Figure, Every Canvas; PosterOmni; UI-Lens; "VLMs are blind").

## Repo layout

```
engine/
  resize.py          entry point (resize + safe fallback)
  main.py            orchestrator from the starter kit (unchanged)
  smart/             the layout engine described above
renderer/            vendored converter + cr-renderer wiring (unchanged)
brand_kit/           assets and fonts referenced by the designs
tests/               finicky cases, harness, unit tests (dev only)
tools/               dev only: contact sheets, stress test, rule checks, Crello evaluation
RULES.md, FAQ.md, chameleon_design_schema_v0.9.json   from the starter kit
```
