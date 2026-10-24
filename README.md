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

No model, no network, no extra dependencies: the engine uses only what the starter
already installs (skia and numpy through cr-renderer, and Pillow). A task takes
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
| headline / subhead / body / legal | prominence (size × weight), position, opacity, length |
| CTA / badge / ribbon | a filled shape carrying a short label (explicit `group_id` or detected); the largest is the CTA, an alike button beside it is a CTA too, a rotated one is a ribbon |
| logo / hero / watermark | image geometry and stacking (the asset name is only a weak hint) |
| panel | a shape or flat image that frames copy (a dark band, a card); it is redrawn around wherever its content lands |
| composite | a photo with captions or stickers on it, or a row of product shots: one piece. A sticker is a round button-like group, or one stuck on a picture; a real button outranks it for CTA, and it keeps to the design's scale when its picture grows |
| lockup | texts set as one piece of type: a drop shadow, stacked display type ("UP TO / 50% / OFF"), a price and its cents. They overlap on purpose or sit closer than a word space, and keep their arrangement |
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
| row | banners and strips | logo, hero, copy, CTA on one line; the hero may fill the strip's height |

Each is solved by a small flexbox-like engine for the largest scale that fits,
then retried with low-importance items dropped: legal, watermark, ribbon, a set
of badges, the subhead; a row of product cards gives way one card at a time.

When a design reflows, a short headline is also offered as **display type**, the
way Adobe's reference output sets it: 1.35× and 1.7× larger than the rest of the
design, up to the canvas's display size (15% of the short side on posts and
stories, up to 42% of a strip's height), at about two words a line and at most
three lines. A longer headline is offered the first step only, and only into
the lines it already takes. Two-line headlines break after punctuation or where
a style run changes colour ("Chase / the Light") when that is nearly as even as
the most balanced break. On a canvas much wider than the source, copy is also
offered a **longer line** (1.8× its source measure, never past a 70-character
reading line), so a column of copy can trade lines for size. Columns follow a
typographic **rhythm**:
- a gap is at most the smaller of the two rows it separates;
- gaps give way before a hero shrinks;
- leftover height is shared between the margins and gaps in proportion to the
  room the source left there; rows set tight (a subhead tucked under its
  headline) are bonded and take none of it;
- across, the column is one block centred where the source's content was
  centred, with each row's text alignment kept inside it.
Headline and CTA are never dropped. Copy never goes below a legibility floor
(cap height of about 7 px when the canvas is viewed at 1024 px). When a label
is held up at the floor, its button gets a tighter pill rather than a bigger one.

**4. Score like the rubric** (`score.py`, `pixels.py`)

Visible importance, copy size against the best any candidate achieved, small or
crowded type, collisions, coverage, balance, hierarchy (a headline that stands out
up to 1.6× more than in the source earns credit; flattening or overshooting costs
it), a prior for the archetype that suits the format, and a charge for secondary
items that take far more of the canvas than they had in the source. The best eight are then re-ranked by the pixels under
their copy (contrast and texture, rendered through the real renderer), relative to
the source. Those eight are shared out first: identical layouts count once and no
archetype takes more than two places, so variants of one plan cannot crowd out a
different one.

**5. Check, emit and enforce the rules** (`__init__.py`)

The winning plan must pass two hard checks on its finished output: no copy drawn
over copy that the source did not draw that way, and no glyph past the canvas
edge. Then:

- backgrounds are redrawn full-bleed, panels around their members, decoration in
  its corner;
- `finalize()` enforces every hard rule on whatever was produced: same ids, same
  order, immutable fields (copy, fonts, colours, assets) from the source, image
  ratios, every painted box (rotation included) on the canvas, integer font sizes,
  and only values the renderer can draw.

If lockups make every layout fail, the design is planned again with their parts
as separate copy. If the layout engine still finds nothing (impossible content,
such as 500 characters in 320×50), `resize()` falls back to a distortion-free
fit-and-centre layout, so a run never crashes.

### Notes on specific decisions

- **Dropped elements stay in the JSON** with their ids and `opacity: 0`, stacked
  under everything. The reference renderer ignores `opacity` on text, so dropped
  copy is also set at 1 px and tucked under the largest opaque element; a rendered
  check confirms it paints nothing.
- **Missing assets:** a small image whose file is not in the brand kit is dropped
  instead of rendering as a grey placeholder. A large one is laid out normally,
  in case the scorer's renderer has the file.
- **Bleed:** decoration that bled off the source canvas is shrunk to its visible
  part and kept inside, because "nothing off-canvas" is a structural check. A
  rotated element counts both its box and its turned box.
- Nothing is tuned to the three samples. Development used them, eight more source
  designs (`tools/make_cases.py`), 24 synthetic finicky designs (`tests/`), 64 real
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
