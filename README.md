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

No model, no network, no extra dependencies. The engine uses only what the starter
already installs: skia (via cr-renderer) and Pillow. A task takes a few
hundredths of a second to about a second. The first task in a run also pays about
3 s of one-time font and library loading.

## Approach

The engine works the way a designer resizing by hand would. It works out what each
piece is, considers several ways the design could be rebuilt for the new shape,
sizes each one as large as it will legibly go, and keeps the one that reads best.

**1. Understand the design** (`engine/smart/analyze.py`, `structure.py`)

The input carries only ids, types and geometry, so roles are inferred:

| Inferred role | Signal |
|---|---|
| background | full-canvas image or shape at the bottom of the stack |
| decoration | free shapes with no text: translucent, elliptical, or bleeding off the canvas |
| headline / subhead / body / legal | text prominence (size × weight), position, opacity, length |
| CTA / badge / ribbon | a filled shape carrying a short label, explicit `group_id` or detected; the largest is the CTA, a rotated one is a ribbon |
| logo / hero / watermark | image geometry, stacking, and (as a weak hint) the asset name |
| panel (backdrop) | a shape that frames several items, such as a dark band behind the copy |
| composite | stickers sitting on the product, or a row of product shots, travel as one piece |
| corner items | small badges, ribbons and watermarks pinned to a canvas corner |
| missing asset | an image whose file is not in the brand kit |

A recursive XY-cut gives the reading structure: which things sit beside which, and
what order a single column should present them in. Two-column sources are read
column by column, and a set of alike items (badges, product shots) stays a row.

**2. Measure exactly like the renderer** (`textfit.py`, `assets.py`)

Text is wrapped with the renderer's own greedy rule at real skia glyph advances,
with integer font sizes because the renderer truncates them. Lines are top-aligned,
as the renderer draws them, and wraps are balanced so there are no orphans. Every
emitted text box is re-checked with the renderer's exact character-level algorithm.
Button labels are optically centred on the cap height. For images, the alpha channel
gives where the visible pixels sit, so spacing follows what the viewer sees. The
transparent margins are still kept on the canvas.

**3. Generate candidate layouts** (`compose.py`, `layout.py`)

A small flexbox-like engine (column, row and wrapping flow) is solved for the
largest global scale that fits. Copy shares width with a narrowest-fit search, and
the hero image takes whatever space is left.

| Archetype | When | Shape |
|---|---|---|
| adapt | nearly the same aspect ratio | the designer's own arrangement, scaled |
| stack | portrait, square, mild landscape | one column in reading order; rows stay near their source-relative height (weighted isotonic regression) |
| split | landscape (16:9 … 3:1) | hero on one side, copy stacked on the other; several sides and widths |
| row | banners and leaderboards | logo, hero, copy, CTA on one line (a variant stacks the logo over the copy) |

Each archetype is also tried with low-importance items dropped, one unit at a time:
legal line, watermark, ribbon, the set of badges, then the subhead. On a 728×90 a
designer drops those rather than shrinking everything into illegibility. Headline
and CTA are never dropped (the CTA only when nothing else fits at all). Type
has per-role legibility floors scaled to the canvas, and goes below them only when
no layout fits at full legibility.

**4. Score like the rubric** (`score.py`)

Each plan is scored on:

- visible importance, minus anything hidden
- copy size against the best any candidate achieved
- a penalty for type below a comfortable size
- line-count and narrow-column penalties
- any collision (layouts are built not to have one)
- coverage against the source
- visual balance
- a small prior for the archetype that suits the format

**5. Emit and enforce the rules** (`__init__.py`)

- Backgrounds are redrawn full-bleed (`fit: cover`).
- Panels are redrawn around wherever their members landed.
- Decoration keeps its corner, scaled to the short side.
- `finalize()` then enforces every hard rule on whatever was produced:
  - same ids, same order
  - immutable fields (content, fonts, colours, assets) restored from the source
  - image ratios restored
  - every painted box, rotation included, moved fully onto the canvas
  - integer font sizes
- If the layout engine ever raises, `resize()` falls back to a distortion-free
  fit-and-centre layout, so a run never crashes.

### Notes on specific decisions

- **Dropped elements stay in the JSON** with their own ids, `opacity: 0`, and a
  `z_index` just under the opaque background. The reference renderer ignores
  `opacity` on text, so stacking order is what keeps a dropped line from showing.
- **Missing assets** (the Level 3 `holiday_watermark`) are dropped the same way
  instead of rendering as a grey "missing asset" box. This only happens when the
  brand kit is present and other assets in the design resolve.
- **Decoration that bled off the source canvas** is shrunk to the part that was
  visible and kept inside the canvas, because "nothing off-canvas" is a structural
  check.
- **Transparent image margins** may overlap a neighbour's box. The visible pixels
  never overlap.
- Nothing is tuned to the three samples. The engine was developed against them,
  eight additional source designs (`tools/make_cases.py`) and about 20 real
  platform formats, from 320×50 to A4.

## Developer tools (not part of the image)

```bash
python -m tools.sheet input/x.json --catalogue -o sheet.png       # one design across ~20 formats
python -m tools.sheet input/x.json --targets 1200x628 --top 4      # compare the best candidates
python -m tools.make_cases                                          # write extra source designs to tools/cases/
python -m tools.stress --n 300 --seed 1                             # random + degenerate designs, every hard rule checked
```

## Repo layout

```
engine/
  resize.py          entry point (resize + safe fallback)
  main.py            orchestrator from the starter kit (unchanged)
  smart/             the layout engine described above
renderer/            vendored converter + cr-renderer wiring (unchanged)
brand_kit/           assets and fonts referenced by the designs
tools/               dev-only: contact sheets, extra cases, stress test
RULES.md, FAQ.md, chameleon_design_schema_v0.9.json   from the starter kit
```
