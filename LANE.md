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

## 3. Research for this process

- O'Donovan, Agarwala, Hertzmann, "Learning Layouts for Single-Page Graphic Designs", TVCG 2014
  (Adobe Research). Energy = weighted sum of design-principle terms (alignment, balance, white
  space, overlap, flow/reading order, text size/hierarchy, emphasis, boundary), weights by
  Nonlinear Inverse Optimization from example layouts; retargeting = minimise E at the new size
  by annealing. Copied: the term list, "designer layout must have lower energy than alternatives"
  as the learning signal (here: convex surrogate = pairwise logistic, weights >= 0, L2 to a prior).
  No code/licence published; nothing vendored. (Notes: context/research/2026-10-30-buildable-tonight.md.)
- Not copied: CP-SAT. The decision that matters is topology (stack / columns / hero split), a
  small discrete set; enumerating it beats a MIP under a 3 s cap. Continuous knobs (margin, gap,
  text widening) are refined by coordinate descent on E.
- Data: Crello families (B0 index, CDLA-Permissive-2.0), geometry only.

## Log
- 06:40Z walking skeleton: analyse -> topologies (stack, column, line, k-cols, hero split, pairs,
  keep, keep-spread) x valign -> energy (hand prior) -> emit/finalize. 3 inputs x 20 formats:
  0 rule faults (own checker `tools/lane.py`; L0 `fundamentals.check` not landed yet). Search is
  fast-path (size-100 advances scaled), exact measure only at emit. Box load avg ~59.
- 07:05Z learned weights (`tools/fit.py`): 500 Crello pairs (families.jsonl minus all 302 dev
  `cases/` families, so the tournament set is unseen), family-disjoint 80/20 split. Pairs: designer
  beats each of our ~40 candidates + 4 perturbations of the designer; plus listwise (candidate
  closer to designer, IoU, beats farther). Held-out: pairwise acc 0.64 -> 0.93; IoU of the
  selected candidate with the designer 0.313 -> 0.458 (oracle over our candidates 0.480).
  Learned: drift (keep normalised positions) dominates, then balance, span/fill, shrink, lines,
  hierarchy, align; grow, order, relsize, margin_uneven -> 0. Overlap fitted ~0.26 (Crello
  designers do overlap); floored to 3 as a constraint term (held-out IoU 0.430). busy/contrast
  stay at prior (no pixels in the fit data).
- Fixes from looking at Crello renders: letter spacing scales with font size; large photo below
  all text = backdrop (cover); large non-backdrop images = aspect-kept panels; text on its plate
  is one block; text boxes hug rendered lines.
- 07:20Z two-axis alignment term + refit (774 pairs; held-out sel IoU 0.445). Tournament-set
  similarity (B0's `similarity`, reporting only, `tools/sim_eval.py`): uniform fit 0.351 (160/160
  tiles with text faults), v3 0.360, 0 faults; strip 0.20 -> 0.34, extreme 0.15 -> 0.29,
  near-square 0.469 -> 0.413 (worse).
- 07:35Z `keep-canvas` candidate (uniform canvas fit, re-fit text): oracle IoU 0.472 -> 0.501;
  refit sel IoU 0.458. Tournament sim 0.3645. Stress 300 designs x 20 formats: 0 faults,
  deterministic and order-independent (fixed a time-budgeted polish and rounding-keyed caches).
- 07:45Z vendored L0 `fundamentals/` (exp/night-foundations); tools now also run
  `fundamentals.check` (hard + promise). Legibility floor = L0 small-print floor (7 px cap at a
  1024 px long-side view, >= 10 px), relaxed proportionally for copy the designer set below the
  source's own floor. Images never below 12 px (rounding distortion).
- 07:55Z near-square diagnosis on non-tournament dev tasks: stack was chosen because the
  overlap constraint term punished keep-spread's small collisions. Fix: polish (collision
  coordinate descent, small steps first) the 2 best and the position-keeping candidates before
  the final pick. Chosen IoU near-square 0.408 -> 0.437 (uniform 0.455, oracle 0.50); tall 0.484
  (uniform 0.448); wide 0.416 (uniform 0.423).
- 08:25Z regressions seen in the sheet (copy over copy on 300x250, blown-up pills) after the
  floor change. Fixes: group legibility boost capped at 1.8x its proportional scale; a hard
  "no new text collision" check *at the destination* (drawn boxes after emit, collisions deeper
  than the source's own layering): emit the best candidates in energy order and keep the first
  clean one; if none is clean, relax the legibility floor (100% -> 60% -> 0) and search again
  (capacity: the canvas cannot hold the copy at the floor). Inputs x 20 formats: 0 hard faults,
  copy_over_copy promises 13 -> 4.
- 08:40Z `fundamentals.check` on stress found decor bleeding further than in the source (132
  off_canvas) and aspect faults on cover panels: decor now keeps at least its source on-canvas
  share per axis; large non-backdrop images always keep aspect; group images >= 20 px.
