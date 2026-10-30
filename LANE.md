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
