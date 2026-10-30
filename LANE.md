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
