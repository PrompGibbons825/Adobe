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

## 08:00Z bench (tournament set 160 tasks, run via tools.bench, uncapped niced; box load ~45 so secs are wall, CPU p95 ~1s)
| head | hard tiles | promise tiles | audit tiles | hidden tiles | sim_mean (wide/mid/tall) |
|---|---|---|---|---|---|
| 579f93a first | 8 | 50 | 53 | 66 | 0.356 (.311/.421/.237) |
| +ls scaling, rotated footprint | 0 | 44 | 47 | 67 | 0.363 |
| +drawn block height | 0 | 34 | 37 | 68 | 0.360 |
| +text floor variants | 0 | 27 | 31 | 66 | 0.361 |
| +bg-ish layers (final) | 0 | 25 | 29 | 70 | 0.358 |
- Faults found by the bench and fixed: letter_spacing was not scaled with font in fit (1px text), rotated text footprint, glyph descent past last line pitch.
- Tried and reverted: partial-bleed bg layers clipped with cover transform (aspect rule fires: 13 hard tiles).
- Not tuned on the tournament set: weights fit on 226 non-tournament tasks; the tournament set is used only for evaluation.
