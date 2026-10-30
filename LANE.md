# L4 lane/compose lab notebook (Liquid Layout, object-based)

## 06:20Z fundamentals (from renderer/ reading)
- Text is drawn top-anchored in its box, unclipped, line pitch = int(font)*line_height; renderer truncates font_size to int, wraps with the base weight only (style_runs bold ignored in wrap), so I emit integer sizes and replicate `wrap_text` exactly (engine/measure.py).
- Shapes are pre-rasterised images; image `contain`/`cover` is emulated in assets.py inside the box, so image box aspect = visible aspect.
- Input has no roles: only id/type/geometry/z/group_id. Resize good = hierarchy, reading order, brand elements proportionate, no collisions, text legible; fails = stretch, tiny text, text on busy image, off-canvas, overflow.

## Process (one owner per decision)
1. units: group_id + implicit merge (text >=60% inside a non-text element), bg (>=80% canvas), decor (shape-only overlapping others/bleeding).
2. anchors per unit (h: l/c/r, v: edge-pinned or proportional).
3. candidates: `pin` (uniform s=min(sx,sy), slack distributed by anchors) and `panes` (reading-order partitions into 1..4 columns, each stacked, common scale k bisected on a grid).
4. text fit: largest int font that fits width/height via renderer wrap.
5. choose by feature score (text shrink, legibility, image retention, fidelity of centres, fill).

## 06:45Z learned weights
- Fit on 200 Crello dev tasks (families outside the tournament set; CDLA-Permissive-2.0 geometry only): choose-by-score candidate vs designer resize, mean IoU (sim). Default hand weights 0.370 -> fit 0.42 (2-fold held-out 0.405/0.424 vs default 0.375/0.364; oracle best candidate 0.439). One weight set for all formats. Weights in engine/scoring.py; tool lab/learn.py.
- Candidate generator: area-balanced contiguous pane partitions (pane count up to 1+1.5*aspect) + visual|text split.

## 07:00Z refit with log-fill feature (226 train tasks, 9308 candidates)
- Pure sim fit gives leg 0.8/fill 0.11 (sim 0.419) but designer-IoU cannot see readability, so leg=3, fill=1 are floors set by hand: sim 0.413, oracle 0.434. Disclosed hand-set: leg, fill, text(0.5).
- Visual review (lab/viz.py): near-aspect Crello tiles are close to the designers; 728x90 with 10+ units is inherently tiny (pin or 6 panes), nothing is dropped.
