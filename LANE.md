# Lane L3 `optimize` — lab notebook

Paradigm: resize = minimise a layout energy (O'Donovan, Agarwala, Hertzmann, TVCG 2014) over
every element's box, under hard constraints. Weights learned from Crello designer resizes.

## 1. Fundamentals (2026-10-30 06:15Z)

**Input.** `source.canvas` (W,H) + elements {id, type text|image|shape, x,y,width,height,
rotation, opacity, z_index, group_id?, type fields}. No roles. Unknowns: which element is the
backdrop, which are decor, which belong together (badge = shape + text), reading order, what
the importance hierarchy is. Crello dev data: fonts mostly not in the brand kit (renderer falls
back to its first/Montserrat family), images often `fit: fill`.

**Output.** Same ids exactly once, `canvas = target`, every box inside the canvas (bleeding
decor excepted only where the source bleeds), image aspect within 5% unless full-bleed `cover`,
`font_size` may change, `font_family`/`content` may not.

**Renderer (read from `renderer/` + cr_renderer):**
- wrap: greedy at the box width, measured with `float(font_size)` and *no* letter spacing;
  glyphs drawn at `int(font_size)` *with* letter spacing. -> use integer font sizes, and check
  rendered line widths (with spacing) against the box myself.
- text is never clipped: lines i are drawn at y = i*fs*lh - ascent. Height of n lines =
  (n-1)*fs*lh + (descent - ascent). A box shorter than that overflows visibly.
- one word wider than the box overflows (never hard-broken) -> font must shrink to fit longest word.
- images: `contain` letterboxes into the box (transparent), `cover` center-crops, `fill` stretches.
- shapes are rasterised; rotation about box centre; z order = z_index sort.

**Score.** Structural JSON checks + AI judge of the PNG: collisions (text on busy image),
readability (size, contrast, nothing cut), brand (logo/colours intact, in proportion),
alignment/spacing/hierarchy, consistency across formats.

**Good resize, first principles.** (1) every content block visible, legible and not colliding;
(2) the canvas is used (content fills a reasonable share, margins even); (3) the source's
relations survive: reading order, what sits on what (badge text on its pill), relative sizes
(hierarchy), alignment family (centred stays centred); (4) backdrop still full-bleed;
(5) decor stays decor (corners/edges), never covers content.
**Failure modes.** Stretch; tiny text in big empty canvas; text overflowing / word cut;
overlap of unrelated blocks; logo blown up or shrunk to nothing; content crowded into one band
of a tall canvas; text over a busy photo region.

## 2. Process (stages, one owner per decision)

1. **Perceive** (`analyse.py`): backdrop (full-cover, lowest z), edge decor (bleeding or faint
   shapes), blocks (union of group_id + containment + strong overlap). Decides: roles of
   structure only (backdrop/decor/block). Evidence: geometry, z, opacity.
2. **Measure** (`measure.py`): the renderer's own glyph advances; text box (w,h) for (fs,width).
   Owns: text metrics only.
3. **Generate** (`solve.py`): structural candidates = the source's reading-order rows packed
   into k bands along either axis (k = 1..4) plus "keep relative positions". Each candidate is
   a set of block boxes with a global scale found by bisection to fill the canvas. Owns: topology.
4. **Energy** (`energy.py`): feature vector of a placed layout (alignment, balance, whitespace /
   fill, overlap, hierarchy kept, reading order kept, relative size kept, legibility, aspect
   distortion, position drift). E = w . f. Owns: preference. Weights fit from Crello.
5. **Refine** (`solve.py`): coordinate descent on (scale, offset) of the winner under E with
   projection onto the hard constraints. Owns: continuous geometry.
6. **Emit + finalize** (`resize.py`): write boxes, clamp to canvas, integer fonts, rule check;
   on any exception fall back to a safe uniform-fit layout. Owns: hard-rule guarantee.

## Log
- 06:40Z walking skeleton: analyse -> topologies (stack, column, line, k-cols, hero split, pairs,
  keep, keep-spread) x valign -> energy (hand prior) -> emit/finalize. 3 inputs x 20 formats:
  0 rule faults (own checker `tools/lane.py`; L0 `fundamentals.check` not landed yet). Search is
  fast-path (size-100 advances scaled), exact measure only at emit. Box load avg ~59.
