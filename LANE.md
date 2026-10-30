# Lane L2 `learn`: lab notebook

Paradigm: learn the resize from Crello designers. Branch `lane/learn` from `upstream/main`.

## 1. Fundamentals (06:05Z)

- **Input**: canvas + elements (text/image/shape) with id, type, box, z, rotation, opacity, style,
  optional group_id. No roles. Unknown: what each element *is* and which relations matter.
- **Output**: same ids exactly once, inside the target canvas, images keep their aspect (except a
  full-bleed `fit: cover`), content and font family unchanged, font size free.
- **Renderer** (read `renderer/`, cr-renderer, L0's QUIRKS.md): text wraps greedily at the float
  size and draws at int(size); letter spacing/capitalize are not in the wrap; text height is
  ignored (never clipped); `fill` images stretch, `cover` centre-crops; white canvas.
- **Score**: hard rules + an AI judge of the PNG: collisions, legibility, brand intact, alignment/
  spacing/hierarchy, consistency across formats.
- **Good resize**: the same message hierarchy, every element legible and whole, layers that covered
  the canvas still cover it, groups intact, no new collisions, the canvas used (not a letterbox).
  **Fails**: stretch, letterbox with dead space, overlaps that were not in the source, tiny type,
  clipped/overflowing text, broken groups (label off its button).

## 2. Process (one owner per decision)

1. `units`: which elements move together (group_id; text inside a small box). Structure.
2. `transfer`: each element's target centre, size and (text) box aspect = distance-weighted mean
   of the k nearest Crello training elements' designer moves (residual over the proportional
   anchor). Feature weights and k are fitted on held-out families.
3. `bleed`: layers covering the source canvas cover the target (cover-fit images).
4. `type`: designers' font-size change, then the largest integer size that fits the box, measured
   with the renderer's own wrap and fonts; a legibility floor.
5. `repair`: units inside the canvas with a margin; collisions absent in the source are pushed
   apart (smaller unit moves, minimal translation), shrinking when there is no room.

## 3. Data

- B0's wide family index (`/data/agents/work/crello-wide/families.jsonl`, CDLA-Permissive-2.0).
  All 293 dev-case families (the bench and tournament pool) are held out of training.
  Training: 895 families, 9120 ordered resize pairs, 105,140 element rows.
- Element correspondence: texts by content, others by kind in z order.

## Log

- 06:20Z offline (held-out families, 4000 element rows, mean box IoU with the designer):
  proportional anchor 0.442, letterbox 0.402, kNN on absolute positions 0.259 (averaging
  positions across designs is worse than not moving). Switched to residuals over the anchor.
- 06:25Z walking skeleton end to end; 60 sample tasks (3 inputs x 20 formats): 1 fault
  (headline at 5 px on 320x50), p95 1.5 s.
- 06:30Z whole-layout retrieval (match the nearest Crello composition one-to-one, copy its
  designer's layout): dev IoU 0.217 vs proportional anchor 0.407 (150 dev tasks, not the
  tournament set). Composition matching across different families transfers badly.
- 06:33Z kNN weight fit (coordinate descent, 1500 held-out rows): 0.392 -> 0.409, still below the
  anchor 0.442. Ridge on 74 aspect-interaction features: 0.388. Oracle: true positions give 0.79,
  true sizes 0.47, so the signal is in positions, and it is not predictable from geometry features.
  **Finding: the designers' median move is the proportional anchor** (median residual 0 for every
  target), with font and box scale = sqrt(area ratio) x a learned per-aspect-change factor
  (+0.40 log for > 7x wider, -0.40 for > 7x taller; `engine/learn/policy.json`, fitted on 105k rows).
- 06:35Z engine = learned median policy + units + text fit + repair; reflow (pack units in source
  reading order into rows/columns, largest scale) when the aspect changes > 2.5x (|log| > 0.9).
  L2 sample 1600x400 now reads as visual | product | copy+CTA. Dev IoU 0.357 (reflow costs
  closeness to the designer but removes the scatter/collisions).
- 06:37Z lane_check: 125 tasks (60 sample x formats + 65 Crello), 4 with faults, all legibility on
  extreme strips (type < 6 px on 320x50, 300x250, 728x90 for the 17-element L3) plus one Crello
  text overflow. p95 0.1 s (warm).
- 06:45Z resumed (the stop at 06:40 was a misread of the meter: wrap-up keys on the weekly figure).
- 06:50-07:15Z repair hardening, all general mechanisms:
  - **Legibility floor** = L0's small-print definition (7 px cap height at a 1024 px long-side
    view: `ceil(10 * max(W,H) / 1024)` px). Copy is raised to it when the designer policy would go
    below. The reflow packer now searches text re-wraps (keep, 1-4 lines) x rows/columns x line
    count x {uniform scale, text held at the floor while pictures shrink}, scored by
    mean scale x (smallest type / floor)^3 x (smallest picture side / 0.3 short side)^2.
  - Letter spacing scales with the font size (em-relative) during fitting.
  - Collisions are judged on drawn extents: text ink (not its padded box), rotated boxes.
  - Reading order: top to bottom, items sharing a line left to right.
  - Final guard (`engine/resize.py`): text measured as the renderer draws it stays inside the
    canvas with a 1.5 px ink margin (shrink, widen, narrower wrap box for letter-spaced copy,
    last resort tighten letter spacing); rotated elements contained by their drawn extent and
    raw box; `fill` images whose box is off the asset's aspect switch to `contain`; images below
    12 px drawn size grow (pixel rounding alone distorts > 5%); non-schema font weights mapped to
    what the renderer draws (light -> regular, semibold -> bold).
- Vendored L0's `fundamentals/` (exp/night-foundations f595ca4) into `engine/fundamentals/`;
  `tools/lane_fcheck.py` runs `fundamentals.check` over input x 20 formats, random designs and
  Crello dev tasks.

## Contamination note

Before my worktree existed (06:03Z) a failed `git worktree add` left my shell in the main
checkout and I printed main's `engine/resize.py`: its docstring names the incumbent's modules
(analyze/textfit/layout/compose/score, archetypes stack/split/row). I read nothing else of the
incumbent, its notes, README or any other lane.
